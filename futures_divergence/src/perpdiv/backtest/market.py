"""백테스트 체결 판정용 시장 데이터: 15분 실행 봉, 거래 중단·상장폐지, 펀딩비·mark price, 1분봉(정밀 모드).

모든 조회는 아카이브 캐시를 거친다. 월 목록(listing)에 없는 달은 요청하지 않는다(상장 전 구간의 404 연쇄 방지).
"""

from __future__ import annotations

import datetime as dt
import json
import threading
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from perpdiv.core.config import Settings
from perpdiv.data.base import empty_ohlcv
from perpdiv.data.cache import ArchiveCache
from perpdiv.data.quality import check_ohlcv, halt_mask
from perpdiv.data.resample import resample_ohlcv
from perpdiv.data.vision import Dataset, VisionArchive


class Listing:
    """심볼·데이터셋별 아카이브 월 파일 목록 (JSON 캐시)."""

    def __init__(self, archive: VisionArchive | None, path: Path) -> None:
        self._archive = archive
        self._path = path
        self._lock = threading.Lock()
        self._data: dict[str, list[str]] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    def months(self, dataset: Dataset) -> set[str]:
        with self._lock:
            if dataset.key in self._data:
                return set(self._data[dataset.key])
        if self._archive is None:
            return set()
        found = sorted(self._archive.list_monthly(dataset))
        with self._lock:
            self._data[dataset.key] = found
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._data), encoding="utf-8")
        return set(found)

    def prime(self, datasets: list[Dataset], workers: int = 32) -> None:
        import concurrent.futures as cf

        missing = [d for d in datasets if d.key not in self._data]
        with cf.ThreadPoolExecutor(workers) as pool:
            list(pool.map(self.months, missing))


def read(cache: ArchiveCache, listing: Listing, dataset: Dataset, start: dt.datetime, end: dt.datetime
         ) -> pd.DataFrame:
    return cache.get_many([dataset], start, end, {dataset: listing.months(dataset)})[dataset]


@dataclass(frozen=True, slots=True)
class ExecBars:
    symbol: str
    bars: pd.DataFrame  # 15분봉 (거래 중단 봉 제거 후), 인덱스 = 봉 시작
    delisted_at: pd.Timestamp | None  # 구간 끝까지 이어지는 거래 중단의 시작 (상장폐지)
    last_trade_close: float | None  # 상장폐지 직전 마지막 체결 봉 종가


class MarketStore:
    def __init__(self, settings: Settings, cache: ArchiveCache, listing: Listing,
                 minute_cache_dir: Path | None = None) -> None:
        self._settings = settings
        self._cache = cache
        self._listing = listing
        self._collect = settings.data.timeframes.collect
        self._exec = settings.data.timeframes.execution
        self._halt = settings.data.quality.inactive_bar.halt_min_bars
        self._minute_dir = minute_cache_dir

    def source(self, symbol: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        return read(self._cache, self._listing, Dataset("klines", symbol, self._collect), start, end)

    def exec_bars(self, symbol: str, start: dt.datetime, end: dt.datetime) -> ExecBars:
        raw = self.source(symbol, start, end)
        if raw.empty:
            return ExecBars(symbol, empty_ohlcv(), None, None)
        report = check_ohlcv(raw, self._collect, start, end, halt_min_bars=self._halt)
        clean = raw[~halt_mask(raw, self._halt)]
        bars = resample_ohlcv(clean, self._exec, source_timeframe=self._collect, as_of=end)
        last_close = float(clean["close"].iloc[-1]) if report.delisted_at is not None and len(clean) else None
        return ExecBars(symbol, bars, report.delisted_at, last_close)

    def funding(self, symbol: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        """(start, end] 의 펀딩 정산: 열 rate, mark. mark 는 1시간 mark price 봉 시가, 없으면 5분봉 시가."""
        data = read(self._cache, self._listing, Dataset("fundingRate", symbol), start, end + dt.timedelta(minutes=1))
        data = data[(data.index > pd.Timestamp(start)) & (data.index <= pd.Timestamp(end))]
        if data.empty:
            return pd.DataFrame({"rate": pd.Series(dtype=float), "mark": pd.Series(dtype=float)})
        times = pd.DatetimeIndex(data.index).floor("min")
        marks = read(self._cache, self._listing, Dataset("markPriceKlines", symbol, "1h"),
                     times.min().floor("h").to_pydatetime(), (times.max() + pd.Timedelta(hours=1)).to_pydatetime())
        mark = pd.Series(marks["open"].reindex(times.floor("h")).to_numpy(), index=times) if not marks.empty else \
            pd.Series(float("nan"), index=times)
        if mark.isna().any():
            src = self.source(symbol, times.min().to_pydatetime(), (times.max() + pd.Timedelta(minutes=5)).to_pydatetime())
            fallback = src["open"].reindex(times.floor("5min")).to_numpy()
            mark = mark.fillna(pd.Series(fallback, index=times))
        return pd.DataFrame({"rate": data["funding_rate"].to_numpy(), "mark": mark.to_numpy()}, index=times)

    def minute_bars(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """[start, end) 의 1분봉 (정밀 모드). 날짜별 일 파일을 받아 캐시한다."""
        if self._minute_dir is None:
            return empty_ohlcv()
        frames = []
        for day in pd.date_range(start.floor("D"), (end - pd.Timedelta(seconds=1)).floor("D"), freq="D"):
            path = self._minute_dir / symbol / f"{day:%Y-%m-%d}.parquet"
            if path.is_file():
                frame = pd.read_parquet(path)
            else:
                frame = self._cache.archive.day_frame(Dataset("klines", symbol, "1m"), day.to_pydatetime())
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(path)
            frames.append(frame)
        if not frames:
            return empty_ohlcv()
        out = pd.concat(frames).sort_index()
        return out[(out.index >= start) & (out.index < end)]
