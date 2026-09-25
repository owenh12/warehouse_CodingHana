"""바이낸스 공개 아카이브(data.binance.vision) 공급자.

REST API 와 같은 봉을 월·일 단위 zip(CSV)으로 제공한다. 대량 과거 구간을 빠르게 받을 수 있고,
REST 결과와 교차검증하는 용도로 쓴다.

- 지난달까지는 월별 파일을 쓴다. 월별 파일이 아직 게시되지 않았으면 일별 파일로 대체한다.
- 이번 달은 어제까지의 일별 파일을 쓴다. 오늘 진행 중인 봉은 아카이브에 없다.
- 현물 CSV 는 2025-01-01 부터 시각이 마이크로초 단위다. 값의 크기로 ms/us 를 판별한다.
- 선물 CSV 에는 헤더 행이 있을 수 있다.
- 각 zip 의 ``.CHECKSUM``(SHA-256)을 검증한다.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import zipfile
from collections.abc import Callable
from typing import Literal

import pandas as pd
import requests

from rsidiv.core.timeutil import UTC, ensure_utc, month_starts, next_month, utc_now
from rsidiv.data.base import (
    OHLCV_COLUMNS,
    DataProvider,
    DataSourceError,
    FundingProvider,
    empty_ohlcv,
    normalize_ohlcv,
    slice_range,
)
from rsidiv.data.binance import MarketType, RetryPolicy, exchange_id, funding_series

#: 바이낸스 현물 거래 시작 월 (아카이브 탐색 하한)
ARCHIVE_FIRST_MONTH = dt.datetime(2017, 7, 1, tzinfo=UTC)
_MICROSECOND_THRESHOLD = 10**14  # epoch ms 는 ~1.7e12, epoch us 는 ~1.7e15


class _RetryableHTTPError(Exception):
    """5xx·429 등 재시도할 HTTP 응답."""


def parse_kline_csv(content: bytes) -> pd.DataFrame:
    """아카이브 kline CSV → 규격 OHLCV 프레임 (헤더 유무, ms/us 단위 자동 판별)."""
    raw = pd.read_csv(io.BytesIO(content), header=None, dtype=str)
    if raw.empty:
        return empty_ohlcv()
    if not str(raw.iloc[0, 0]).strip().lstrip("-").isdigit():  # 헤더 행 (open_time, ...)
        raw = raw.iloc[1:]
    times = raw.iloc[:, 0].astype("int64")
    unit: Literal["us", "ms"] = "us" if int(times.max()) >= _MICROSECOND_THRESHOLD else "ms"
    frame = pd.DataFrame(
        {col: raw.iloc[:, i + 1].astype("float64").to_numpy() for i, col in enumerate(OHLCV_COLUMNS)},
        index=pd.DatetimeIndex(pd.to_datetime(times, unit=unit, utc=True)),
    )
    return normalize_ohlcv(frame)


def parse_funding_csv(content: bytes) -> dict[int, float]:
    """아카이브 fundingRate CSV (calc_time, funding_interval_hours, last_funding_rate) → {ms: rate}."""
    raw = pd.read_csv(io.BytesIO(content), header=None, dtype=str)
    if raw.empty:
        return {}
    if not str(raw.iloc[0, 0]).strip().isdigit():
        raw = raw.iloc[1:]
    return {int(t): float(r) for t, r in zip(raw.iloc[:, 0], raw.iloc[:, -1], strict=True)}


def _read_single_csv(zip_bytes: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        names = [n for n in archive.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise DataSourceError(f"zip 안의 CSV 개수가 1이 아닙니다: {names}")
        return archive.read(names[0])


class BinanceVisionProvider(DataProvider, FundingProvider):
    """data.binance.vision 월·일별 zip 아카이브에서 봉과 펀딩비를 읽는다."""

    def __init__(
        self,
        market_type: MarketType,
        *,
        base_url: str,
        retry: RetryPolicy,
        verify_checksum: bool = True,
        timeout_sec: float = 30.0,
        session: requests.Session | None = None,
        clock: Callable[[], dt.datetime] = utc_now,
    ) -> None:
        self.market_type = market_type
        self._base = base_url.rstrip("/")
        self._retry = retry
        self._verify = verify_checksum
        self._timeout = timeout_sec
        self._session = session or requests.Session()
        self._clock = clock

    @property
    def source_name(self) -> str:
        return "binance-vision"

    @staticmethod
    def retryable_errors() -> tuple[type[BaseException], ...]:
        return (requests.ConnectionError, requests.Timeout, _RetryableHTTPError)

    # --- URL ---------------------------------------------------------------

    def _market_path(self) -> str:
        return "spot" if self.market_type == "spot" else "futures/um"

    def kline_url(self, symbol: str, timeframe: str, period: str) -> str:
        """period: 'YYYY-MM'(월별) 또는 'YYYY-MM-DD'(일별)."""
        sym = exchange_id(symbol)
        freq = "monthly" if len(period) == 7 else "daily"
        return (
            f"{self._base}/data/{self._market_path()}/{freq}/klines/{sym}/{timeframe}/"
            f"{sym}-{timeframe}-{period}.zip"
        )

    def funding_url(self, symbol: str, month: str) -> str:
        sym = exchange_id(symbol)
        return f"{self._base}/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{month}.zip"

    # --- HTTP --------------------------------------------------------------

    def _get(self, url: str) -> bytes | None:
        def attempt() -> bytes | None:
            response = self._session.get(url, timeout=self._timeout)
            if response.status_code == 404:
                return None
            if response.status_code == 429 or response.status_code >= 500:
                raise _RetryableHTTPError(f"HTTP {response.status_code}")
            if response.status_code != 200:
                raise DataSourceError(f"{url}: HTTP {response.status_code}")
            return response.content

        return self._retry.call(f"GET {url}", attempt)

    def _exists(self, url: str) -> bool:
        def attempt() -> bool:
            response = self._session.head(url, timeout=self._timeout)
            if response.status_code == 429 or response.status_code >= 500:
                raise _RetryableHTTPError(f"HTTP {response.status_code}")
            return response.status_code == 200

        return self._retry.call(f"HEAD {url}", attempt)

    def _download_csv(self, url: str) -> bytes | None:
        payload = self._get(url)
        if payload is None:
            return None
        if self._verify:
            checksum = self._get(url + ".CHECKSUM")
            if checksum is None:
                raise DataSourceError(f"{url}: CHECKSUM 파일 없음")
            expected = checksum.decode().split()[0].lower()
            actual = hashlib.sha256(payload).hexdigest()
            if actual != expected:
                raise DataSourceError(f"{url}: SHA-256 불일치")
        return _read_single_csv(payload)

    # --- DataProvider ------------------------------------------------------

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, start: dt.datetime, end: dt.datetime
    ) -> pd.DataFrame:
        start, end = ensure_utc(start), ensure_utc(end)
        today = self._clock().astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        end = min(end, today)  # 아카이브는 어제까지 (완결된 일자)
        frames: list[pd.DataFrame] = []
        for month in month_starts(start, end):
            month_end = next_month(month)
            monthly = None
            if month_end <= today:
                monthly = self._download_csv(self.kline_url(symbol, timeframe, month.strftime("%Y-%m")))
            if monthly is not None:
                frames.append(parse_kline_csv(monthly))
                continue
            day = max(month, start.replace(hour=0, minute=0, second=0, microsecond=0))
            while day < min(month_end, end):
                daily = self._download_csv(self.kline_url(symbol, timeframe, day.strftime("%Y-%m-%d")))
                if daily is not None:
                    frames.append(parse_kline_csv(daily))
                day += dt.timedelta(days=1)
        if not frames:
            return empty_ohlcv()
        return slice_range(normalize_ohlcv(pd.concat(frames)), start, end)

    def earliest_available(self, symbol: str, timeframe: str) -> dt.datetime | None:
        """월별 파일 존재 여부를 이분 탐색해 첫 게시 월의 첫 봉 시각을 찾는다."""
        this_month = self._clock().astimezone(UTC).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        months = month_starts(ARCHIVE_FIRST_MONTH, this_month)  # 완결된 달만

        def exists(month: dt.datetime) -> bool:
            return self._exists(self.kline_url(symbol, timeframe, month.strftime("%Y-%m")))

        # 지난달 파일은 월초 며칠간 아직 게시되지 않았을 수 있어 그 전달까지 확인한다
        latest = next((m for m in reversed(months[-2:]) if exists(m)), None)
        if latest is None:
            return None
        months = months[: months.index(latest) + 1]
        lo, hi = 0, len(months) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if exists(months[mid]):
                hi = mid
            else:
                lo = mid + 1
        first = self.fetch_ohlcv(symbol, timeframe, months[lo], next_month(months[lo]))
        return first.index[0].to_pydatetime() if not first.empty else None

    def funding_rates(self, symbol: str, start: dt.datetime, end: dt.datetime) -> pd.Series:
        """월별 펀딩비 파일만 게시되므로 이번 달 데이터는 포함되지 않는다."""
        if self.market_type != "usdm_futures":
            raise ValueError("펀딩비는 usdm_futures 시장에서만 조회할 수 있습니다")
        start, end = ensure_utc(start), ensure_utc(end)
        records: dict[int, float] = {}
        for month in month_starts(start, end):
            content = self._download_csv(self.funding_url(symbol, month.strftime("%Y-%m")))
            if content is not None:
                records.update(parse_funding_csv(content))
        return funding_series(records, start, end)
