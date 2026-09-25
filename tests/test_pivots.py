"""피벗 판별과 미래참조 방지.

핵심 성질:
1. 피벗 i 는 i+R 봉까지의 데이터가 있어야 처음 나타난다 (그 전에는 절대 보이지 않음).
2. prefix 불변성: 데이터를 k 번째 봉까지만 넣은 결과 = 전체 결과 중 confirm_index ≤ k 인 것.
3. 미래 봉을 어떻게 바꿔도 이미 확정된 피벗은 바뀌지 않는다.
4. 증분 판별기(실시간)와 일괄 판별 결과가 같고, 확정되는 호출 순번이 confirm_index 와 같다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rsidiv.core.config import PivotCfg
from rsidiv.indicators.pivots import Pivot, PivotTracker, detect_pivots, pivot_masks, pivots_frame


def cfg(left: int = 5, right: int = 3, strict: bool = True, source: str = "low") -> PivotCfg:
    return PivotCfg(left=left, right=right, price_source=source, strict=strict)  # type: ignore[arg-type]


def frame_from_lows(lows: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=len(lows), freq="15min", tz="UTC").as_unit("ns").rename("time")
    low = np.asarray(lows, dtype=float)
    return pd.DataFrame({"open": low + 0.5, "high": low + 1, "low": low, "close": low + 0.5,
                         "volume": 1.0}, index=idx)


def random_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = np.round(100 + np.cumsum(rng.normal(0, 1, n)), 0)  # 정수 반올림 → 동일값 다수
    high = close + rng.integers(0, 3, n)
    low = close - rng.integers(0, 3, n)
    idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC").as_unit("ns").rename("time")
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": 1.0}, index=idx)


def test_hand_example_strict_and_ties() -> None:
    #          0  1  2  3  4  5  6  7  8
    lows = [5, 4, 3, 4, 5, 3, 3, 5, 6]
    frame = frame_from_lows(lows)
    strict = detect_pivots(frame, cfg(2, 2, strict=True))
    assert strict == [Pivot("low", 2, 4, 3.0), Pivot("high", 4, 6, 6.0)]
    # 동일 저점(5, 6번 봉): strict 는 피벗 없음, non-strict 는 첫 봉(5)만 피벗
    loose = detect_pivots(frame, cfg(2, 2, strict=False))
    assert Pivot("low", 5, 7, 3.0) in loose and Pivot("low", 6, 8, 3.0) not in loose


def test_edges_need_full_left_and_right_windows() -> None:
    lows = [1, 5, 6, 7, 2, 8, 9, 10, 0]
    is_low, _ = pivot_masks(np.array(lows, float), np.array(lows, float) + 1, left=3, right=3)
    assert np.flatnonzero(is_low).tolist() == [4]  # 0번(왼쪽 부족), 8번(오른쪽 부족)은 제외


def test_price_source_close_uses_close_for_both() -> None:
    frame = frame_from_lows([5, 4, 3, 4, 5, 3, 3, 5, 6])
    frame["close"] = [9, 8, 9, 7, 9, 9, 9, 9, 9]  # 종가 기준 저점은 3번 봉
    lows = [p for p in detect_pivots(frame, cfg(2, 2, source="close")) if p.kind == "low"]
    assert lows == [Pivot("low", 3, 5, 7.0)]


def test_pivots_frame_times() -> None:
    frame = frame_from_lows([5, 4, 3, 4, 5, 3, 3, 5, 6])
    table = pivots_frame(frame, cfg(2, 2), timeframe_minutes=15)
    row = table[table["kind"] == "low"].iloc[0]
    assert row["time"] == frame.index[2]
    # 확정 시각 = 확정 봉(4번) 종가 시각 = 4번 봉 시작 + 15분
    assert row["confirm_time"] == frame.index[4] + pd.Timedelta("15min")


PARAMS = [(5, 3, True), (5, 3, False), (3, 2, True), (2, 5, True), (1, 1, False)]


@pytest.mark.parametrize(("left", "right", "strict"), PARAMS)
def test_pivot_invisible_until_right_bars_closed(left: int, right: int, strict: bool) -> None:
    frame = random_frame(300, seed=left * 10 + right)
    config = cfg(left, right, strict)
    pivots = detect_pivots(frame, config)
    assert any(p.kind == "low" for p in pivots)
    for p in pivots:
        before = detect_pivots(frame.iloc[: p.index + right], config)  # t3+R-1 봉까지
        after = detect_pivots(frame.iloc[: p.index + right + 1], config)  # t3+R 봉 확정
        assert p not in before
        assert p in after
        assert p.confirm_index == p.index + right


@pytest.mark.parametrize(("left", "right", "strict"), PARAMS)
def test_prefix_invariance(left: int, right: int, strict: bool) -> None:
    frame = random_frame(250, seed=100 + left + right)
    config = cfg(left, right, strict)
    full = detect_pivots(frame, config)
    for k in range(len(frame)):
        partial = detect_pivots(frame.iloc[: k + 1], config)
        assert partial == [p for p in full if p.confirm_index <= k], f"k={k}"


def test_future_changes_do_not_alter_confirmed_pivots() -> None:
    frame = random_frame(400, seed=42)
    config = cfg()
    rng = np.random.default_rng(0)
    for k in (50, 150, 300):
        mutated = frame.copy()
        noise = rng.normal(0, 5, len(frame) - k - 1)
        for col in ("open", "high", "low", "close"):
            mutated.iloc[k + 1 :, mutated.columns.get_loc(col)] += noise
        confirmed = [p for p in detect_pivots(frame, config) if p.confirm_index <= k]
        assert [p for p in detect_pivots(mutated, config) if p.confirm_index <= k] == confirmed


@pytest.mark.parametrize(("left", "right", "strict"), PARAMS)
@pytest.mark.parametrize("source", ["low", "close"])
def test_tracker_matches_batch_and_emits_at_confirmation(
    left: int, right: int, strict: bool, source: str
) -> None:
    frame = random_frame(600, seed=7)
    config = cfg(left, right, strict, source)
    tracker = PivotTracker(config)
    streamed: list[Pivot] = []
    for n, bar in enumerate(frame.itertuples()):
        emitted = tracker.update(bar.high, bar.low, bar.close)
        assert all(p.confirm_index == n for p in emitted)
        streamed.extend(emitted)
    assert streamed == detect_pivots(frame, config)
    assert tracker.bars_seen == len(frame)
