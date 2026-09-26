"""월 파티션 Parquet 캐시.

``{root}/{kind}/{interval}/{SYMBOL}/{YYYY-MM}.parquet`` 와 같은 폴더의 ``_manifest.json``(수집 시각·출처·행 수·확정 여부).
확정(final) 조건: 수집 시각 ≥ 월말 + ``refresh_recent_days``. 확정 파티션은 다시 받지 않는다.
진행 중인 달은 지난 수집일 2일 전부터의 일 파일만 이어 받는다. 끝난 달의 미확정 파티션은 월 파일로 통째로 다시 받는다.
여러 데이터셋·여러 달을 스레드 풀로 병렬 수집한다.
"""

from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import json
import threading
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from perpdiv.core.timeutil import ensure_utc, month_starts, next_month, utc_now
from perpdiv.data.base import slice_range
from perpdiv.data.vision import Dataset, VisionArchive

_MANIFEST = "_manifest.json"
RESUME_OVERLAP = dt.timedelta(days=2)
Compression = Literal["zstd", "snappy", "gzip", "none"]


@dataclass(frozen=True, slots=True)
class PartitionInfo:
    fetched_at: str
    source: str
    rows: int
    final: bool


class ArchiveCache:
    def __init__(self, root: str | Path, archive: VisionArchive, *, compression: Compression = "zstd",
                 refresh_recent_days: int = 3, workers: int = 8,
                 clock: Callable[[], dt.datetime] = utc_now) -> None:
        self.root = Path(root)
        self.archive = archive
        self._compression = None if compression == "none" else compression
        self._refresh = dt.timedelta(days=refresh_recent_days)
        self._workers = workers
        self._clock = clock
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _dir(self, dataset: Dataset) -> Path:
        return self.root / dataset.key

    def _lock(self, dataset: Dataset) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(dataset.key, threading.Lock())

    def manifest(self, dataset: Dataset) -> dict[str, PartitionInfo]:
        path = self._dir(dataset) / _MANIFEST
        if not path.is_file():
            return {}
        return {k: PartitionInfo(**v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}

    def _write(self, dataset: Dataset, month: dt.datetime, frame: pd.DataFrame, fetched_at: dt.datetime) -> None:
        directory = self._dir(dataset)
        directory.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(directory / f"{month:%Y-%m}.parquet", compression=self._compression)
        final = fetched_at >= next_month(month) + self._refresh
        with self._lock(dataset):
            manifest = self.manifest(dataset)
            manifest[f"{month:%Y-%m}"] = PartitionInfo(fetched_at.isoformat(), self.archive.source_name,
                                                       len(frame), final)
            tmp = directory / (_MANIFEST + ".tmp")
            tmp.write_text(json.dumps({k: asdict(v) for k, v in sorted(manifest.items())}, indent=1),
                           encoding="utf-8")
            tmp.replace(directory / _MANIFEST)

    def _month(self, dataset: Dataset, month: dt.datetime) -> pd.DataFrame:
        info = self.manifest(dataset).get(f"{month:%Y-%m}")
        path = self._dir(dataset) / f"{month:%Y-%m}.parquet"
        if info is not None and info.final and path.is_file():
            return pd.read_parquet(path)
        fetched_at = self._clock()
        if info is not None and path.is_file() and next_month(month) > fetched_at:
            # 진행 중인 달: 지난 수집일 − RESUME_OVERLAP 부터만 다시 받는다 (일 파일 게시 지연 대비 겹쳐 받음)
            last = dt.datetime.fromisoformat(info.fetched_at).replace(hour=0, minute=0, second=0, microsecond=0)
            resume = last - RESUME_OVERLAP
            cached = pd.read_parquet(path)
            fresh = self.archive.month_frame(dataset, month, from_day=resume)
            parts = [f for f in (cached[cached.index < pd.Timestamp(resume)], fresh) if not f.empty]
            if len(parts) == 2:
                merged = pd.concat(parts)
                frame = merged[~merged.index.duplicated(keep="last")].sort_index()
            else:
                frame = parts[0] if parts else fresh
        else:
            frame = self.archive.month_frame(dataset, month)
        self._write(dataset, month, frame, fetched_at)
        return frame

    def get(self, dataset: Dataset, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        return self.get_many([dataset], start, end)[dataset]

    def get_many(self, datasets: Sequence[Dataset], start: dt.datetime, end: dt.datetime,
                 months_available: dict[Dataset, set[str]] | None = None,
                 progress: Callable[[int, int], None] | None = None) -> dict[Dataset, pd.DataFrame]:
        """여러 데이터셋의 [start, end) 를 병렬로 모은다. ``months_available`` 을 주면 목록에 없는 월 파일은
        요청하지 않는다(이번 달 일 파일은 항상 시도)."""
        start, end = ensure_utc(start), ensure_utc(end)
        now = self._clock()
        jobs: list[tuple[Dataset, dt.datetime]] = []
        for ds in datasets:
            for month in month_starts(start, end):
                complete = next_month(month) <= now
                if complete and months_available is not None and f"{month:%Y-%m}" not in months_available.get(ds, set()):
                    continue
                jobs.append((ds, month))
        parts: dict[Dataset, list[pd.DataFrame]] = {ds: [] for ds in datasets}
        with cf.ThreadPoolExecutor(self._workers) as pool:
            futures = {pool.submit(self._month, ds, m): ds for ds, m in jobs}
            for done, future in enumerate(cf.as_completed(futures), 1):
                parts[futures[future]].append(future.result())
                if progress is not None:
                    progress(done, len(jobs))
        out = {}
        for ds, frames in parts.items():
            frames = [f for f in frames if not f.empty]
            if not frames:
                out[ds] = self.archive.empty_frame(ds)
                continue
            merged = pd.concat(frames).sort_index()
            out[ds] = slice_range(merged[~merged.index.duplicated(keep="last")], start, end)
        return out
