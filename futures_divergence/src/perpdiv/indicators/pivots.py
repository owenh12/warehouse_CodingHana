"""피벗(스윙 저점·고점)과 확정 시점. t1·p2(강세), p1·t2(약세)에만 쓴다.

피벗 저점 i:  Low[i] < Low[j]  (i−L ≤ j < i)   그리고   Low[i] <op> Low[j]  (i < j ≤ i+R)
피벗 고점 i:  High[i] > High[j] (왼쪽)          그리고   High[i] <op'> High[j] (오른쪽)

동률 규칙(``tie_rule``):
- ``strict``: 오른쪽도 엄격(<, >). 이웃에 같은 값이 있으면 피벗이 아니다.
- ``first`` : 오른쪽은 같음 허용(≤, ≥). 같은 값이 이어지면 첫 봉만 피벗(왼쪽이 엄격하므로 둘째 봉부터는 탈락).
왼쪽 L개·오른쪽 R개가 모두 있어야 한다(양 끝 제외).

미래참조 방지: 피벗 i 는 ``confirm_index = i + R`` 봉 종가가 확정된 뒤에만 쓸 수 있다.
:class:`PivotTracker` 는 i+R 번째 봉을 받는 순간 피벗 i 를 내보낸다.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

PivotKind = Literal["low", "high"]
TieRule = Literal["strict", "first"]


@dataclass(frozen=True, slots=True)
class Pivot:
    kind: PivotKind
    index: int
    confirm_index: int
    price: float  # 저점은 Low, 고점은 High


def pivot_masks(high: npt.ArrayLike, low: npt.ArrayLike, left: int, right: int,
                tie_rule: TieRule = "strict") -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """(피벗 저점 여부, 피벗 고점 여부) — 피벗 봉 위치 기준 bool 배열."""
    lo = pd.Series(np.asarray(low, dtype=np.float64))
    hi = pd.Series(np.asarray(high, dtype=np.float64))

    def left_ext(s: pd.Series, fn: str) -> npt.NDArray[np.float64]:
        return np.asarray(getattr(s.shift(1).rolling(left, min_periods=left), fn)().to_numpy(), dtype=np.float64)

    def right_ext(s: pd.Series, fn: str) -> npt.NDArray[np.float64]:
        rev = s.iloc[::-1].reset_index(drop=True).shift(1).rolling(right, min_periods=right)
        return np.asarray(getattr(rev, fn)().to_numpy()[::-1], dtype=np.float64)

    lv, hv = lo.to_numpy(), hi.to_numpy()
    with np.errstate(invalid="ignore"):
        if tie_rule == "strict":
            is_low = (lv < left_ext(lo, "min")) & (lv < right_ext(lo, "min"))
            is_high = (hv > left_ext(hi, "max")) & (hv > right_ext(hi, "max"))
        else:
            is_low = (lv < left_ext(lo, "min")) & (lv <= right_ext(lo, "min"))
            is_high = (hv > left_ext(hi, "max")) & (hv >= right_ext(hi, "max"))
    return is_low, is_high


def detect_pivots(frame: pd.DataFrame, left: int, right: int, tie_rule: TieRule = "strict") -> list[Pivot]:
    """프레임 전체의 피벗 (피벗 위치 순, 같은 위치면 low 먼저)."""
    low, high = frame["low"].to_numpy(np.float64), frame["high"].to_numpy(np.float64)
    is_low, is_high = pivot_masks(high, low, left, right, tie_rule)
    out = [Pivot("low", int(i), int(i) + right, float(low[i])) for i in np.flatnonzero(is_low)]
    out += [Pivot("high", int(i), int(i) + right, float(high[i])) for i in np.flatnonzero(is_high)]
    out.sort(key=lambda p: (p.index, p.kind != "low"))
    return out


class PivotTracker:
    """증분 피벗 판별기. n 번째(0부터) ``update`` 가 돌려주는 피벗의 ``confirm_index`` 는 항상 n 이다."""

    def __init__(self, left: int, right: int, tie_rule: TieRule = "strict") -> None:
        self.left, self.right, self.tie_rule = left, right, tie_rule
        self._lows: deque[float] = deque(maxlen=left + right + 1)
        self._highs: deque[float] = deque(maxlen=left + right + 1)
        self._count = 0

    @property
    def bars_seen(self) -> int:
        return self._count

    def update(self, high: float, low: float) -> list[Pivot]:
        self._lows.append(low)
        self._highs.append(high)
        n = self._count
        self._count += 1
        if len(self._lows) < self.left + self.right + 1:
            return []
        lows, highs = list(self._lows), list(self._highs)
        c_lo, c_hi = lows[self.left], highs[self.left]
        strict = self.tie_rule == "strict"
        found: list[Pivot] = []
        if all(c_lo < v for v in lows[: self.left]) and all(
                (c_lo < v) if strict else (c_lo <= v) for v in lows[self.left + 1:]):
            found.append(Pivot("low", n - self.right, n, c_lo))
        if all(c_hi > v for v in highs[: self.left]) and all(
                (c_hi > v) if strict else (c_hi >= v) for v in highs[self.left + 1:]):
            found.append(Pivot("high", n - self.right, n, c_hi))
        return found
