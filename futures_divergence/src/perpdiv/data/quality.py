"""데이터 품질: 결측·OHLC 모순·거래 없음(거래 중단·상장폐지) 판정.

거래 없음 봉 = 체결 수 0 이고 OHLC 가 모두 같은 봉. 이런 봉이 ``halt_min_bars`` 개 이상 연속이면 거래 중단 구간이다.
아카이브는 상장폐지 뒤에도 이런 봉을 계속 게시하므로, 데이터 끝까지 이어지는 거래 중단 구간의 시작을 상장폐지 시각으로 본다.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from perpdiv.core.config import timeframe_minutes


@dataclass(frozen=True, slots=True)
class Segment:
    start: pd.Timestamp  # 첫 봉 시작
    end: pd.Timestamp  # 마지막 봉 시작
    bars: int


@dataclass(slots=True)
class QualityReport:
    rows: int
    expected: int
    first: pd.Timestamp | None
    last: pd.Timestamp | None
    missing_bars: int
    gaps: list[Segment] = field(default_factory=list)
    ohlc_violations: int = 0
    inactive_bars: int = 0
    halts: list[Segment] = field(default_factory=list)
    delisted_at: pd.Timestamp | None = None  # 데이터 끝까지 이어지는 거래 중단의 시작


def inactive_bars(frame: pd.DataFrame) -> pd.Series:
    """체결 0 + 고정가 봉 여부."""
    flat = (frame["open"] == frame["high"]) & (frame["high"] == frame["low"]) & (frame["low"] == frame["close"])
    return (frame["trades"] == 0) & flat


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """True 연속 구간 [(시작, 끝 포함)]."""
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(s), int(e) - 1) for s, e in zip(edges[::2], edges[1::2], strict=True)]


def halt_mask(frame: pd.DataFrame, halt_min_bars: int) -> pd.Series:
    """거래 중단 구간(거래 없음 봉이 halt_min_bars 개 이상 연속)에 속한 봉."""
    inactive = inactive_bars(frame).to_numpy()
    out = np.zeros(len(frame), dtype=bool)
    for s, e in _runs(inactive):
        if e - s + 1 >= halt_min_bars:
            out[s : e + 1] = True
    return pd.Series(out, index=frame.index)


def check_ohlcv(frame: pd.DataFrame, timeframe: str, start: dt.datetime, end: dt.datetime, *,
                halt_min_bars: int) -> QualityReport:
    step = pd.Timedelta(minutes=timeframe_minutes(timeframe))
    grid = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=step, inclusive="left", tz="UTC")
    present = grid.isin(frame.index)
    gaps = [Segment(grid[s], grid[e], e - s + 1) for s, e in _runs(~present)]
    body_hi = frame[["open", "close"]].max(axis=1)
    body_lo = frame[["open", "close"]].min(axis=1)
    violations = int(((frame["high"] < body_hi) | (frame["low"] > body_lo) | (frame["low"] > frame["high"])).sum())
    inactive = inactive_bars(frame)
    halts = [Segment(frame.index[s], frame.index[e], e - s + 1) for s, e in _runs(inactive.to_numpy())
             if e - s + 1 >= halt_min_bars]
    delisted = halts[-1].start if halts and halts[-1].end == frame.index[-1] else None
    return QualityReport(
        rows=len(frame), expected=len(grid),
        first=frame.index[0] if len(frame) else None, last=frame.index[-1] if len(frame) else None,
        missing_bars=int((~present).sum()), gaps=gaps, ohlc_violations=violations,
        inactive_bars=int(inactive.sum()), halts=halts, delisted_at=delisted,
    )
