"""피벗: 손으로 만든 예(동률 포함), 가장자리, 확정 시점, 미래참조(prefix·미래 변경 불변), 증분 = 일괄."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from perpdiv.indicators.pivots import Pivot, PivotTracker, detect_pivots, pivot_masks


def frame(lows: Sequence[float], highs: Sequence[float] | None = None) -> pd.DataFrame:
    lo = np.asarray(lows, dtype=float)
    hi = np.asarray(highs, dtype=float) if highs is not None else lo + 1
    idx = pd.date_range("2025-01-01", periods=len(lo), freq="15min", tz="UTC")
    return pd.DataFrame({"open": lo + 0.5, "high": hi, "low": lo, "close": lo + 0.5}, index=idx)


def random_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = np.round(100 + np.cumsum(rng.normal(0, 1, n)))  # 정수 반올림 → 동률 다수
    return pd.DataFrame({"open": close, "high": close + rng.integers(0, 3, n), "low": close - rng.integers(0, 3, n),
                         "close": close}, index=pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC"))


def test_l1_r1_basic() -> None:
    #           0  1  2  3  4  5
    lows = [5, 3, 4, 2, 6, 7]
    pivots = [p for p in detect_pivots(frame(lows), 1, 1) if p.kind == "low"]
    assert pivots == [Pivot("low", 1, 2, 3.0), Pivot("low", 3, 4, 2.0)]


def test_tie_rules() -> None:
    #           0  1  2  3  4
    lows = [5, 3, 3, 4, 6]  # 1·2 번 봉 동률
    strict = [p.index for p in detect_pivots(frame(lows), 1, 1, "strict") if p.kind == "low"]
    first = [p.index for p in detect_pivots(frame(lows), 1, 1, "first") if p.kind == "low"]
    assert strict == []  # 이웃과 같으면 피벗 아님
    assert first == [1]  # 같은 값이 이어지면 첫 봉만
    highs = [1, 4, 4, 2, 0]
    fr = frame([0, 0, 0, 0, 0], highs)
    assert [p.index for p in detect_pivots(fr, 1, 1, "strict") if p.kind == "high"] == []
    assert [p.index for p in detect_pivots(fr, 1, 1, "first") if p.kind == "high"] == [1]


def test_edges_need_full_windows() -> None:
    lows = [1, 5, 6, 2, 8, 9, 0]
    is_low, _ = pivot_masks(np.array(lows) + 1.0, np.array(lows, float), left=2, right=2)
    assert np.flatnonzero(is_low).tolist() == [3]  # 0번(왼쪽 부족)·6번(오른쪽 부족)은 가장 낮아도 제외
    tracker = PivotTracker(2, 2)
    assert [p for x in lows for p in tracker.update(x + 1.0, float(x))] == [Pivot("low", 3, 5, 2.0)]


def test_invisible_until_confirmed() -> None:
    fr = random_frame(400, 1)
    for left, right in ((1, 1), (2, 3)):
        full = detect_pivots(fr, left, right)
        for p in full[:40]:
            before = detect_pivots(fr.iloc[: p.confirm_index], left, right)  # 확정 봉 직전까지
            after = detect_pivots(fr.iloc[: p.confirm_index + 1], left, right)
            assert p not in before and p in after


@pytest.mark.parametrize("tie_rule", ["strict", "first"])
@pytest.mark.parametrize(("left", "right"), [(1, 1), (2, 1), (3, 4)])
def test_prefix_invariance_and_incremental(tie_rule: str, left: int, right: int) -> None:
    fr = random_frame(600, 2)
    full = detect_pivots(fr, left, right, tie_rule)  # type: ignore[arg-type]
    for k in range(1, len(fr) + 1, 7):
        assert detect_pivots(fr.iloc[:k], left, right, tie_rule) == [p for p in full if p.confirm_index < k]  # type: ignore[arg-type]
    tracker = PivotTracker(left, right, tie_rule)  # type: ignore[arg-type]
    streamed: list[Pivot] = []
    for n, (h, lo) in enumerate(zip(fr["high"], fr["low"], strict=True)):
        found = tracker.update(float(h), float(lo))
        assert all(p.confirm_index == n for p in found)
        streamed += found
    assert sorted(streamed, key=lambda p: (p.index, p.kind != "low")) == full


def test_future_changes_do_not_alter_confirmed() -> None:
    fr = random_frame(500, 3)
    base = detect_pivots(fr, 1, 1)
    rng = np.random.default_rng(3)
    for k in (100, 250, 400):
        changed = fr.copy()
        noise = np.concatenate([np.zeros(k), rng.normal(0, 5, len(fr) - k)])
        for col in ("open", "high", "low", "close"):
            changed[col] = changed[col].to_numpy() + noise
        again = detect_pivots(changed, 1, 1)
        assert [p for p in again if p.confirm_index < k] == [p for p in base if p.confirm_index < k]
