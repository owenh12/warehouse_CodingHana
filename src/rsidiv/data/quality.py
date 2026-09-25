"""OHLCV 품질 점검과 결측 봉 처리.

점검 항목: 중복·역순 시각, OHLC 모순(고가 < 시가/종가 등), 0 이하 가격, 음수 거래량,
거래량 0 봉, 기대 시각 대비 결측 봉(구간 목록 포함).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from rsidiv.core.config import timeframe_minutes


@dataclass(frozen=True, slots=True)
class Gap:
    """연속 결측 구간 [start, end) 과 결측 봉 수."""

    start: dt.datetime
    end: dt.datetime
    bars: int


@dataclass(frozen=True, slots=True)
class QualityReport:
    rows: int
    first: dt.datetime | None
    last: dt.datetime | None
    duplicated_times: int
    unsorted: bool
    ohlc_violations: int
    nonpositive_prices: int
    negative_volume: int
    zero_volume_bars: int
    expected_bars: int
    missing_bars: int
    gaps: list[Gap] = field(default_factory=list)

    @property
    def longest_gap_bars(self) -> int:
        return max((gap.bars for gap in self.gaps), default=0)

    @property
    def is_clean(self) -> bool:
        return not (
            self.duplicated_times or self.unsorted or self.ohlc_violations
            or self.nonpositive_prices or self.negative_volume
        )


def continuous_index(start: dt.datetime, end: dt.datetime, timeframe: str) -> pd.DatetimeIndex:
    """24시간 시장(가상화폐)의 기대 봉 시각 [start, end)."""
    minutes = timeframe_minutes(timeframe)
    first = pd.Timestamp(start).ceil(f"{minutes}min")
    index = pd.date_range(first, pd.Timestamp(end), freq=f"{minutes}min", inclusive="left")
    return index.as_unit("ns").rename("time")


def check_ohlcv(frame: pd.DataFrame, expected: pd.DatetimeIndex) -> QualityReport:
    """``expected`` (기대 봉 시각) 기준으로 품질을 점검한다. 입력 프레임은 정렬 전이어도 된다."""
    index = frame.index
    o, h, lo, c = (frame[col].to_numpy() for col in ("open", "high", "low", "close"))
    violations = (h < np.maximum(o, c)) | (lo > np.minimum(o, c)) | (h < lo)
    present = expected.isin(index)
    return QualityReport(
        rows=len(frame),
        first=index.min().to_pydatetime() if len(index) else None,
        last=index.max().to_pydatetime() if len(index) else None,
        duplicated_times=int(index.duplicated().sum()),
        unsorted=not index.is_monotonic_increasing,
        ohlc_violations=int(violations.sum()),
        nonpositive_prices=int((frame[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()),
        negative_volume=int((frame["volume"] < 0).sum()),
        zero_volume_bars=int((frame["volume"] == 0).sum()),
        expected_bars=len(expected),
        missing_bars=int((~present).sum()),
        gaps=_gaps(expected, present),
    )


def _gaps(expected: pd.DatetimeIndex, present: np.ndarray) -> list[Gap]:
    gaps: list[Gap] = []
    if len(expected) == 0:
        return gaps
    step = expected[1] - expected[0] if len(expected) > 1 else pd.Timedelta(0)
    run_start: int | None = None
    for i, ok in enumerate([*present.tolist(), True]):
        if not ok and run_start is None:
            run_start = i
        elif ok and run_start is not None:
            start = expected[run_start]
            end = expected[i] if i < len(expected) else expected[-1] + step
            gaps.append(Gap(start.to_pydatetime(), end.to_pydatetime(), i - run_start))
            run_start = None
    return gaps


def fill_missing_bars(
    frame: pd.DataFrame, expected: pd.DatetimeIndex, policy: Literal["ffill_flat", "keep_gap"]
) -> pd.DataFrame:
    """결측 봉 처리. 결과에는 채운 봉 여부를 나타내는 ``filled`` 컬럼이 추가된다.

    - ``ffill_flat``: 직전 종가로 OHLC 를 채우고 거래량 0. 첫 실제 봉 이전 구간은 채우지 않는다.
    - ``keep_gap``: 결측 봉을 만들지 않는다 (피벗의 좌우 봉 수는 실제 봉 기준으로 센다).
    """
    if policy == "keep_gap" or frame.empty:
        return frame.assign(filled=False)
    first = frame.index[0]
    target = expected[expected >= first].union(frame.index)
    out = frame.reindex(target)
    filled = out["close"].isna().to_numpy()
    prev_close = out["close"].ffill()
    for col in ("open", "high", "low", "close"):
        out[col] = out[col].fillna(prev_close)
    out["volume"] = out["volume"].fillna(0.0)
    out["filled"] = filled
    return out
