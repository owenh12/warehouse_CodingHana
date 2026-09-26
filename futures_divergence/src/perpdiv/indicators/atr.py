"""ATR (Wilder, 기간 14). ``TR[i] = max(H−L, |H−C[i−1]|, |L−C[i−1]|)`` (i ≥ 1), ATR = RMA(TR).

첫 유효값은 인덱스 ``period`` (TA-Lib·TradingView ``ta.atr`` 과 같은 방식). 손절폭에는 신호 확정 봉(t3·p3)의 값을 쓴다.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from perpdiv.indicators.wilder import FloatArray, WilderSmoother, wilder_smooth


def true_range(high: npt.ArrayLike, low: npt.ArrayLike, close: npt.ArrayLike) -> FloatArray:
    h, lo, c = (np.asarray(x, dtype=np.float64) for x in (high, low, close))
    prev = np.concatenate([[np.nan], c[:-1]])
    tr = np.asarray(np.maximum.reduce([h - lo, np.abs(h - prev), np.abs(lo - prev)]), dtype=np.float64)
    tr[0] = np.nan
    return tr


def atr_wilder(high: npt.ArrayLike, low: npt.ArrayLike, close: npt.ArrayLike, period: int = 14) -> FloatArray:
    return wilder_smooth(true_range(high, low, close), period, first=1)


class AtrState:
    """증분 ATR (일괄과 비트 단위로 같음)."""

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self._prev: float | None = None
        self._smoother = WilderSmoother(period)
        self.value: float | None = None

    def update(self, high: float, low: float, close: float) -> float | None:
        prev, self._prev = self._prev, close
        if prev is None:
            return None
        self.value = self._smoother.update(max(high - low, abs(high - prev), abs(low - prev)))
        return self.value
