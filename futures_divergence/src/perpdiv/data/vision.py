"""바이낸스 공개 아카이브(data.binance.vision) — USDⓈ-M 선물.

- 경로: ``data/futures/um/{monthly|daily}/{kind}/{SYMBOL}/{interval}/{SYMBOL}-{interval}-{period}.zip``
  (kind: ``klines``, ``markPriceKlines``; 펀딩비는 ``fundingRate/{SYMBOL}/{SYMBOL}-fundingRate-{YYYY-MM}.zip``)
- 지난달까지는 월 파일, 월 파일이 아직 없거나 이번 달이면 일 파일(어제까지)을 쓴다.
- 각 zip 의 ``.CHECKSUM``(SHA-256)을 검증한다.
- 심볼 목록은 S3 목록 API 로 받는다. 상장폐지 심볼도 남아 있다(폐지 후에는 체결 0 고정가 봉이 계속 게시됨).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import re
import threading
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeVar

import pandas as pd
import requests

from perpdiv.core.timeutil import UTC, ensure_utc, next_month, utc_now
from perpdiv.data.base import DataSourceError, empty_ohlcv, normalize_ohlcv

T = TypeVar("T")
Kind = Literal["klines", "markPriceKlines", "fundingRate"]
_MICROSECONDS = 10**14  # epoch ms ≈ 1.7e12, epoch µs ≈ 1.7e15


class _Retryable(Exception):
    """5xx·429 응답."""


class RetryPolicy:
    """일시적 오류를 ``max_retries`` 회까지 ``backoff × 2**n`` 초 간격으로 재시도.

    :class:`DataSourceError` (지역 차단·검증 실패 등 영구 오류)는 ``retry_on`` 과 겹쳐도 재시도하지 않는다.
    """

    def __init__(self, max_retries: int, backoff_sec: float,
                 retry_on: tuple[type[BaseException], ...] = (requests.ConnectionError, requests.Timeout, _Retryable),
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.max_retries = max_retries
        self.backoff_sec = backoff_sec
        self.retry_on = retry_on
        self._sleep = sleep

    def call(self, what: str, fn: Callable[[], T]) -> T:
        for attempt in range(self.max_retries + 1):
            try:
                return fn()
            except DataSourceError:
                raise
            except self.retry_on as exc:
                if attempt == self.max_retries:
                    raise DataSourceError(f"{what}: {attempt + 1}회 시도 실패 ({type(exc).__name__}: {exc})") from exc
                self._sleep(self.backoff_sec * 2**attempt)
        raise AssertionError("unreachable")


@dataclass(frozen=True, slots=True)
class Dataset:
    kind: Kind
    symbol: str
    interval: str | None = None  # 펀딩비는 None

    @property
    def key(self) -> str:
        return f"{self.kind}/{self.interval or 'none'}/{self.symbol}"


def parse_kline_csv(content: bytes) -> pd.DataFrame:
    """kline CSV → 규격 OHLCV. 헤더 유무와 ms/µs 시각을 자동 판별한다."""
    raw = pd.read_csv(io.BytesIO(content), header=None, dtype=str)
    if raw.empty:
        return empty_ohlcv()
    if not str(raw.iloc[0, 0]).strip().isdigit():
        raw = raw.iloc[1:]
    if raw.empty:
        return empty_ohlcv()
    times = raw.iloc[:, 0].astype("int64")
    unit: Literal["us", "ms"] = "us" if int(times.max()) >= _MICROSECONDS else "ms"
    frame = pd.DataFrame(
        {
            "open": raw.iloc[:, 1].astype("float64").to_numpy(),
            "high": raw.iloc[:, 2].astype("float64").to_numpy(),
            "low": raw.iloc[:, 3].astype("float64").to_numpy(),
            "close": raw.iloc[:, 4].astype("float64").to_numpy(),
            "volume": raw.iloc[:, 5].astype("float64").to_numpy(),
            "quote_volume": raw.iloc[:, 7].astype("float64").to_numpy() if raw.shape[1] > 7 else float("nan"),
            "trades": raw.iloc[:, 8].astype("int64").to_numpy() if raw.shape[1] > 8 else -1,
        },
        index=pd.DatetimeIndex(pd.to_datetime(times, unit=unit, utc=True)),
    )
    return normalize_ohlcv(frame)


def parse_funding_csv(content: bytes) -> pd.DataFrame:
    """fundingRate CSV (calc_time, funding_interval_hours, last_funding_rate) → 시각 인덱스 프레임."""
    raw = pd.read_csv(io.BytesIO(content), header=None, dtype=str)
    if not raw.empty and not str(raw.iloc[0, 0]).strip().isdigit():
        raw = raw.iloc[1:]
    if raw.empty:
        return empty_funding()
    frame = pd.DataFrame(
        {"funding_rate": raw.iloc[:, -1].astype("float64").to_numpy(),
         "interval_hours": raw.iloc[:, 1].astype("float64").to_numpy() if raw.shape[1] >= 3 else float("nan")},
        index=pd.DatetimeIndex(pd.to_datetime(raw.iloc[:, 0].astype("int64"), unit="ms", utc=True)),
    )
    frame.index = pd.DatetimeIndex(frame.index).as_unit("ns").rename("time")
    frame = frame.sort_index()
    return frame[~frame.index.duplicated(keep="last")]


def empty_funding() -> pd.DataFrame:
    frame = pd.DataFrame({"funding_rate": pd.Series(dtype="float64"), "interval_hours": pd.Series(dtype="float64")})
    frame.index = pd.DatetimeIndex([], tz="UTC", name="time").as_unit("ns")
    return frame


def _single_csv(payload: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [n for n in archive.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise DataSourceError(f"zip 안의 CSV 가 1개가 아닙니다: {names}")
        return archive.read(names[0])


class VisionArchive:
    """아카이브 접근 (스레드 안전: 스레드마다 세션을 따로 쓴다)."""

    def __init__(self, *, base_url: str, listing_url: str, retry: RetryPolicy, verify_checksum: bool = True,
                 timeout_sec: float = 30.0, clock: Callable[[], dt.datetime] = utc_now) -> None:
        self._base = base_url.rstrip("/")
        self._listing = listing_url.rstrip("/")
        self._retry = retry
        self._verify = verify_checksum
        self._timeout = timeout_sec
        self._clock = clock
        self._local = threading.local()
        self.bytes_downloaded = 0
        self._lock = threading.Lock()

    @property
    def source_name(self) -> str:
        return "binance-vision"

    def _session(self) -> requests.Session:
        if not hasattr(self._local, "session"):
            self._local.session = requests.Session()
        session: requests.Session = self._local.session
        return session

    # --- HTTP ---------------------------------------------------------------

    def _get(self, url: str, params: dict[str, str] | None = None) -> bytes | None:
        def attempt() -> bytes | None:
            response = self._session().get(url, params=params, timeout=self._timeout)
            if response.status_code == 404:
                return None
            if response.status_code == 429 or response.status_code >= 500:
                raise _Retryable(f"HTTP {response.status_code}")
            if response.status_code != 200:
                raise DataSourceError(f"{url}: HTTP {response.status_code}")
            with self._lock:
                self.bytes_downloaded += len(response.content)
            return response.content

        return self._retry.call(f"GET {url}", attempt)

    def _download_csv(self, path: str) -> bytes | None:
        url = f"{self._base}/{path}"
        payload = self._get(url)
        if payload is None:
            return None
        if self._verify:
            checksum = self._get(url + ".CHECKSUM")
            if checksum is None:
                raise DataSourceError(f"{url}: CHECKSUM 없음")
            if hashlib.sha256(payload).hexdigest() != checksum.decode().split()[0].lower():
                raise DataSourceError(f"{url}: SHA-256 불일치")
        return _single_csv(payload)

    # --- 목록 ---------------------------------------------------------------

    def _list(self, prefix: str, *, delimiter: bool) -> list[tuple[str, int]]:
        """S3 목록: (키 또는 하위 접두어, 크기) 목록."""
        out: list[tuple[str, int]] = []
        marker = ""
        while True:
            params = {"prefix": prefix, "marker": marker}
            if delimiter:
                params["delimiter"] = "/"
            body = self._get(self._listing, params)
            if body is None:
                raise DataSourceError(f"목록 조회 실패: {prefix}")
            text = body.decode()
            if delimiter:
                items = [(p, 0) for p in re.findall(r"<Prefix>([^<]+)</Prefix>", text) if p != prefix]
            else:
                items = [(k, int(sz)) for k, sz in re.findall(r"<Key>([^<]+)</Key>.*?<Size>(\d+)</Size>", text)]
            out += items
            if "<IsTruncated>true</IsTruncated>" not in text or not items:
                return out
            next_marker = re.search(r"<NextMarker>([^<]+)</NextMarker>", text)
            marker = next_marker.group(1) if next_marker else items[-1][0]

    def list_symbols(self) -> list[str]:
        """아카이브에 월별 kline 이 있는 USDⓈ-M 심볼 전체 (상장폐지·분기물 포함)."""
        return [p.rstrip("/").split("/")[-1] for p, _ in self._list("data/futures/um/monthly/klines/", delimiter=True)]

    def list_monthly(self, dataset: Dataset) -> dict[str, int]:
        """월 파일 목록: 'YYYY-MM' → zip 크기(바이트)."""
        prefix = self._prefix("monthly", dataset)
        out = {}
        for key, size in self._list(prefix, delimiter=False):
            match = re.search(r"-(\d{4}-\d{2})\.zip$", key)
            if match:
                out[match.group(1)] = size
        return out

    # --- 파일 ---------------------------------------------------------------

    def _prefix(self, freq: str, dataset: Dataset) -> str:
        if dataset.kind == "fundingRate":
            return f"data/futures/um/{freq}/fundingRate/{dataset.symbol}/"
        return f"data/futures/um/{freq}/{dataset.kind}/{dataset.symbol}/{dataset.interval}/"

    def _path(self, freq: str, dataset: Dataset, period: str) -> str:
        if dataset.kind == "fundingRate":
            return f"{self._prefix(freq, dataset)}{dataset.symbol}-fundingRate-{period}.zip"
        return f"{self._prefix(freq, dataset)}{dataset.symbol}-{dataset.interval}-{period}.zip"

    def _parse(self, dataset: Dataset, content: bytes) -> pd.DataFrame:
        return parse_funding_csv(content) if dataset.kind == "fundingRate" else parse_kline_csv(content)

    def empty_frame(self, dataset: Dataset) -> pd.DataFrame:
        return empty_funding() if dataset.kind == "fundingRate" else empty_ohlcv()

    def month_frame(self, dataset: Dataset, month: dt.datetime, *, from_day: dt.datetime | None = None) -> pd.DataFrame:
        """한 달치 데이터. 월 파일이 있으면 그것, 없으면 일 파일(어제까지)을 모은다. 펀딩비는 월 파일만.

        ``from_day`` 를 주면 진행 중인 달에서 그날부터의 일 파일만 받는다 (캐시 이어받기용)."""
        month = ensure_utc(month)
        today = self._clock().astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        month_end = next_month(month)
        if month_end <= today:
            content = self._download_csv(self._path("monthly", dataset, month.strftime("%Y-%m")))
            if content is not None:
                return self._parse(dataset, content)
        if dataset.kind == "fundingRate":
            return self.empty_frame(dataset)
        frames = []
        day = max(month, ensure_utc(from_day)) if from_day is not None and month_end > today else month
        while day < min(month_end, today):
            content = self._download_csv(self._path("daily", dataset, day.strftime("%Y-%m-%d")))
            if content is not None:
                frames.append(self._parse(dataset, content))
            day += dt.timedelta(days=1)
        if not frames:
            return self.empty_frame(dataset)
        return normalize_ohlcv(pd.concat(frames))
