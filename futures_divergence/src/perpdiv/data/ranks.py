"""정밀 순위: 후보 코인의 5분봉 거래대금으로 15분 경계 T 마다 롤링 24h(`[T−24h, T)` 마감 봉) 코인 순위.

신호 시각(= t3 봉 마감)은 항상 15분 경계이므로 이 표에서 바로 찾는다. 상위 ``keep`` 위까지만 저장한다(긴 형식:
time, coin, rank, volume_24h). 표에 없는 코인은 순위 > keep (= 순위 밖).

코인마다 5분봉 사용 구간은 후보군의 데이터 구간으로 제한한다(그 밖은 받지 않았고, 1시간 순위상 상위권이 아니다).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from perpdiv.core.config import Settings
from perpdiv.core.timeutil import month_starts, next_month
from perpdiv.data.cache import ArchiveCache
from perpdiv.data.candidates import CandidateSet, holding_limit, universe_dir
from perpdiv.data.symbols import parse_contract
from perpdiv.data.universe import coin_volume, ranks, rolling_volume
from perpdiv.data.vision import Dataset

RANK_FILE = "ranks_15m.parquet"


def build_rank_table(settings: Settings, cache: ArchiveCache, candidates: CandidateSet, *, keep: int = 20,
                     log: Callable[[str], None] = print, path: Path | None = None,
                     months_available: Callable[[Dataset], set[str]] | None = None) -> Path:
    """``months_available``: 데이터셋의 아카이브 월 목록 (상장 전 달을 요청하지 않도록). 없으면 캐시가 전부 시도한다."""
    uni = settings.universe
    tf = settings.data.timeframes.collect
    start, end = candidates.bounds
    warmup, hold = dt.timedelta(days=settings.data.warmup_days), holding_limit(settings)
    ranges = {coin: w.data_range(warmup, hold, start, end) for coin, w in candidates.windows.items()}
    parts: list[pd.DataFrame] = []
    for month in month_starts(start, end):
        lo_m, hi_m = month - dt.timedelta(days=1), min(next_month(month), end)
        frames = {}
        for coin, w in candidates.windows.items():
            lo, hi = ranges[coin]
            a, b = max(lo, lo_m), min(hi, hi_m)
            if a >= b:
                continue
            for symbol in w.rank_symbols:
                contract = parse_contract(symbol)
                assert contract is not None
                ds = Dataset("klines", symbol, tf)
                available = {ds: months_available(ds)} if months_available is not None else None
                frame = cache.get_many([ds], a, b, available)[ds]
                if not frame.empty:
                    frames[contract] = frame[["quote_volume"]]
        if not frames:
            continue
        volume = coin_volume(frames, uni.ranking.usd_per_quote, tf)
        rolled = rolling_volume(volume, tf, uni.ranking.window_hours)
        idx = pd.DatetimeIndex(rolled.index)
        rolled = rolled[(idx >= pd.Timestamp(month)) & (idx < pd.Timestamp(hi_m)) & (idx.minute % 15 == 0)]
        table = ranks(rolled)
        values = table.to_numpy(dtype=float)
        rows, cols = np.nonzero(values <= keep)  # NaN 은 비교에서 빠짐
        long = pd.DataFrame({
            "time": pd.DatetimeIndex(table.index)[rows],
            "coin": np.asarray(table.columns)[cols],
            "rank": values[rows, cols].astype(np.int16),
            "volume_24h": rolled.to_numpy(dtype=float)[rows, cols],
        })
        parts.append(long)
        log(f"  순위 {month:%Y-%m}: 코인 {volume.shape[1]}개, 경계 {len(table)}곳")
    out = pd.concat(parts, ignore_index=True).sort_values(["time", "rank"], kind="stable")
    target = path or universe_dir(settings) / RANK_FILE
    out.to_parquet(target, compression="zstd", index=False)
    return target


class RankBook:
    """신호 시각의 코인 순위 조회. 표에 없으면 None (= 저장 범위 밖, 즉 순위 밖)."""

    def __init__(self, table: pd.DataFrame) -> None:
        self.table = table
        self._index = pd.MultiIndex.from_frame(table[["time", "coin"]])
        self._ranks = table["rank"].to_numpy()
        self._lookup = pd.Series(self._ranks, index=self._index)
        self._times = pd.DatetimeIndex(table["time"].unique())

    @classmethod
    def load(cls, path: Path) -> RankBook:
        return cls(pd.read_parquet(path))

    @property
    def covered(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self._times.min(), self._times.max()

    def rank(self, coin: str, time: pd.Timestamp) -> int | None:
        value = self._lookup.get((time, coin))
        return None if value is None else int(value)

    def ranks_for(self, coins: pd.Series, times: pd.Series) -> pd.Series:
        """벡터 조회: 없으면 NaN."""
        keys = pd.MultiIndex.from_arrays([pd.DatetimeIndex(times), coins.to_numpy()])
        return pd.Series(self._lookup.reindex(keys).to_numpy(dtype=float), index=coins.index)

    def top(self, time: pd.Timestamp, n: int) -> list[str]:
        rows = self.table[(self.table["time"] == time) & (self.table["rank"] <= n)]
        return list(rows.sort_values("rank")["coin"])
