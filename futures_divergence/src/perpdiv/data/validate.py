"""리샘플 검증: 5분봉을 합친 결과를 거래소가 직접 만든 봉(아카이브 원본 15m/1h/4h/1d)과 비교한다.

비교 기준:
- 봉 시각 집합이 같은가 (원본에만 있는 봉 / 리샘플에만 있는 봉)
- OHLC 가 정확히 같은가 (아카이브 값은 호가 단위 10진 문자열이므로 상대오차 1e−12 이내면 같음으로 본다)
- 거래량·거래대금 상대오차 최대값 (부동소수 합산 오차 수준이면 정상), 체결 수가 같은가
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from perpdiv.core.config import timeframe_minutes
from perpdiv.core.timeutil import next_month
from perpdiv.data.cache import ArchiveCache
from perpdiv.data.resample import resample_ohlcv
from perpdiv.data.vision import Dataset

PRICE_RTOL = 1e-12


@dataclass(frozen=True, slots=True)
class ResampleCheck:
    symbol: str
    timeframe: str
    month: str
    native_bars: int
    resampled_bars: int
    only_native: int
    only_resampled: int
    partial_buckets: int  # 원천 5분봉 일부가 빠진 버킷
    price_mismatches: int
    trades_mismatches: int
    max_volume_rel_err: float
    max_quote_volume_rel_err: float

    @property
    def ok(self) -> bool:
        return (self.only_native == 0 and self.only_resampled == 0 and self.price_mismatches == 0
                and self.trades_mismatches == 0 and self.max_volume_rel_err < 1e-9
                and self.max_quote_volume_rel_err < 1e-9)


def _rel_err(a: pd.Series, b: pd.Series) -> float:
    x, y = a.to_numpy(np.float64), b.to_numpy(np.float64)
    scale = np.maximum(np.abs(y), 1e-12)
    return float(np.max(np.abs(x - y) / scale)) if len(x) else 0.0


def compare(resampled: pd.DataFrame, native: pd.DataFrame, *, symbol: str, timeframe: str, month: str,
            source_timeframe: str) -> ResampleCheck:
    common = resampled.index.intersection(native.index)
    r, n = resampled.loc[common], native.loc[common]
    price_bad = np.zeros(len(common), dtype=bool)
    for col in ("open", "high", "low", "close"):
        a, b = r[col].to_numpy(np.float64), n[col].to_numpy(np.float64)
        price_bad |= ~np.isclose(a, b, rtol=PRICE_RTOL, atol=0.0)
    per_bucket = timeframe_minutes(timeframe) // timeframe_minutes(source_timeframe)
    return ResampleCheck(
        symbol=symbol, timeframe=timeframe, month=month, native_bars=len(native), resampled_bars=len(resampled),
        only_native=len(native.index.difference(resampled.index)),
        only_resampled=len(resampled.index.difference(native.index)),
        partial_buckets=int((resampled["bars"] < per_bucket).sum()),
        price_mismatches=int(price_bad.sum()),
        trades_mismatches=int((r["trades"].to_numpy() != n["trades"].to_numpy()).sum()),
        max_volume_rel_err=_rel_err(r["volume"], n["volume"]),
        max_quote_volume_rel_err=_rel_err(r["quote_volume"], n["quote_volume"]),
    )


def check_resample(cache: ArchiveCache, symbols: Sequence[str], months: Sequence[dt.datetime],
                   timeframes: Sequence[str], source_timeframe: str = "5m") -> list[ResampleCheck]:
    """심볼·월마다 5분봉과 원본 상위 봉을 받아 비교한다 (월 경계를 넘는 버킷이 없도록 월 단위로)."""
    datasets = [Dataset("klines", s, tf) for s in symbols for tf in (source_timeframe, *timeframes)]
    out: list[ResampleCheck] = []
    for month in months:
        end = next_month(month)
        frames = cache.get_many(datasets, month, end)
        for symbol in symbols:
            source = frames[Dataset("klines", symbol, source_timeframe)]
            if source.empty:
                continue
            for tf in timeframes:
                resampled = resample_ohlcv(source, tf, source_timeframe=source_timeframe, as_of=end)
                out.append(compare(resampled, frames[Dataset("klines", symbol, tf)], symbol=symbol, timeframe=tf,
                                   month=f"{month:%Y-%m}", source_timeframe=source_timeframe))
    return out


def to_frame(checks: Sequence[ResampleCheck]) -> pd.DataFrame:
    rows = [{f: getattr(c, f) for f in ResampleCheck.__slots__} | {"ok": c.ok} for c in checks]
    return pd.DataFrame(rows)
