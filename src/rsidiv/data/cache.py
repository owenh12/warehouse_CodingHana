"""Parquet 로컬 캐시.

``{root}/{dataset}/{YYYY-MM}.parquet`` 로 월 단위 파티션을 저장하고, 같은 폴더의
``_manifest.json`` 에 파티션별 수집 시각·출처·행 수를 기록한다.

파티션 확정(final) 조건: 수집 시각 ≥ 월말 + ``refresh_recent_days``.
확정된 파티션은 다시 받지 않는다. 확정 전 파티션(이번 달·최근 월)은 요청할 때마다 다시
받아 덮어쓴다. 거래소가 확정 직후 데이터를 정정하는 경우에 대비한 것이다.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from rsidiv.core.timeutil import ensure_utc, month_starts, next_month, utc_now
from rsidiv.data.base import (
    DataProvider,
    FundingProvider,
    empty_funding,
    empty_ohlcv,
    normalize_ohlcv,
    slice_range,
)

_MANIFEST = "_manifest.json"
ParquetCompression = Literal["zstd", "snappy", "gzip"]
CompressionName = Literal["zstd", "snappy", "gzip", "none"]


@dataclass(frozen=True, slots=True)
class PartitionInfo:
    fetched_at: str  # ISO 8601 UTC
    source: str
    rows: int
    final: bool


class ParquetStore:
    """데이터셋(문자열 키)별 월 파티션 Parquet 저장소."""

    def __init__(self, root: str | Path, compression: CompressionName = "zstd") -> None:
        self.root = Path(root)
        self.compression: ParquetCompression | None = None if compression == "none" else compression

    def _dir(self, dataset: str) -> Path:
        return self.root / dataset

    def partition_path(self, dataset: str, month: dt.datetime) -> Path:
        return self._dir(dataset) / f"{month:%Y-%m}.parquet"

    def manifest(self, dataset: str) -> dict[str, PartitionInfo]:
        path = self._dir(dataset) / _MANIFEST
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return {month: PartitionInfo(**info) for month, info in data.items()}

    def write_month(
        self, dataset: str, month: dt.datetime, frame: pd.DataFrame, *, source: str,
        fetched_at: dt.datetime, final: bool,
    ) -> None:
        directory = self._dir(dataset)
        directory.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(self.partition_path(dataset, month), compression=self.compression)
        manifest = self.manifest(dataset)
        manifest[f"{month:%Y-%m}"] = PartitionInfo(
            fetched_at=ensure_utc(fetched_at).isoformat(), source=source, rows=len(frame), final=final
        )
        payload = {key: asdict(info) for key, info in sorted(manifest.items())}
        tmp = directory / (_MANIFEST + ".tmp")
        tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(directory / _MANIFEST)

    def read(self, dataset: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame | None:
        """[start, end) 구간 파티션을 이어 붙인다. 파티션이 하나도 없으면 None."""
        frames = [
            pd.read_parquet(path)
            for month in month_starts(start, end)
            if (path := self.partition_path(dataset, month)).is_file()
        ]
        if not frames:
            return None
        combined = pd.concat(frames).sort_index()
        return slice_range(combined, start, end)


def _is_final(month: dt.datetime, fetched_at: dt.datetime, refresh_recent_days: int) -> bool:
    return fetched_at >= next_month(month) + dt.timedelta(days=refresh_recent_days)


class _MonthlyCache:
    """월 파티션 단위로 '없거나 확정 전이면 원천에서 받아 저장'하는 공통 로직."""

    def __init__(
        self, store: ParquetStore, refresh_recent_days: int, clock: Callable[[], dt.datetime]
    ) -> None:
        self.store = store
        self.refresh_recent_days = refresh_recent_days
        self.clock = clock

    def ensure(
        self, dataset: str, start: dt.datetime, end: dt.datetime, source: str,
        fetch_month: Callable[[dt.datetime, dt.datetime], pd.DataFrame],
    ) -> None:
        now = ensure_utc(self.clock())
        manifest = self.store.manifest(dataset)
        for month in month_starts(start, min(ensure_utc(end), now)):
            info = manifest.get(f"{month:%Y-%m}")
            if info is not None and info.final:
                continue
            frame = fetch_month(month, min(next_month(month), now))
            self.store.write_month(
                dataset, month, frame, source=source, fetched_at=now,
                final=_is_final(month, now, self.refresh_recent_days),
            )


class CachedDataProvider(DataProvider, FundingProvider):
    """다른 공급자를 감싸 월 단위 Parquet 캐시를 적용한다.

    ``namespace`` 는 캐시 폴더 접두어이며 거래소·시장을 구분한다 (예: ``binance/spot``).
    같은 거래소의 REST·아카이브 공급자는 같은 데이터를 주므로 namespace 를 공유한다.
    """

    def __init__(
        self,
        inner: DataProvider,
        store: ParquetStore,
        namespace: str,
        *,
        refresh_recent_days: int,
        source: str,
        clock: Callable[[], dt.datetime] = utc_now,
    ) -> None:
        self.inner = inner
        self.namespace = namespace
        self.source = source
        self._cache = _MonthlyCache(store, refresh_recent_days, clock)

    def dataset(self, symbol: str, kind: str) -> str:
        return f"{self.namespace}/{symbol.replace('/', '-').replace(':', '-')}/{kind}"

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, start: dt.datetime, end: dt.datetime
    ) -> pd.DataFrame:
        dataset = self.dataset(symbol, timeframe)
        self._cache.ensure(
            dataset, start, end, self.source,
            lambda s, e: self.inner.fetch_ohlcv(symbol, timeframe, s, e),
        )
        frame = self._cache.store.read(dataset, start, end)
        return normalize_ohlcv(frame) if frame is not None else empty_ohlcv()

    def earliest_available(self, symbol: str, timeframe: str) -> dt.datetime | None:
        return self.inner.earliest_available(symbol, timeframe)

    def funding_rates(self, symbol: str, start: dt.datetime, end: dt.datetime) -> pd.Series:
        if not isinstance(self.inner, FundingProvider):
            raise TypeError(f"{type(self.inner).__name__} 는 펀딩비를 제공하지 않습니다")
        inner = self.inner
        dataset = self.dataset(symbol, "funding")
        self._cache.ensure(
            dataset, start, end, self.source,
            lambda s, e: inner.funding_rates(symbol, s, e).to_frame(),
        )
        frame = self._cache.store.read(dataset, start, end)
        return frame["funding_rate"] if frame is not None else empty_funding()
