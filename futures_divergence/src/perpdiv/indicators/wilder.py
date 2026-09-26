"""Wilder 평활(RMA) — RSI·ATR 공용.

    seed      avg[first + n − 1] = (x[first] + … + x[first + n − 1]) / n
    이후      avg[t] = (avg[t−1] × (n − 1) + x[t]) / n

TradingView ``ta.rma`` 와 같다(첫 값은 단순평균으로 시작). 바이낸스 차트의 RSI 도 TradingView 기준이다.
일괄 함수와 증분 클래스가 같은 연산을 같은 순서로 하므로 결과가 **비트 단위로** 같다. 백테스트 신호와
실거래 신호가 부동소수점 차이로 갈라지는 일을 막기 위해서다.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def wilder_smooth(values: FloatArray, period: int, first: int) -> FloatArray:
    """``values[first:]`` 에 RMA 적용. seed 이전 위치는 NaN."""
    if period < 1:
        raise ValueError("period 는 1 이상이어야 합니다")
    n = len(values)
    out = np.full(n, np.nan)
    seed_at = first + period - 1
    if n <= seed_at:
        return out
    data = values.tolist()
    if any(math.isnan(v) for v in data[first:]):
        raise ValueError("입력에 NaN 이 있습니다")
    total = 0.0
    for v in data[first : seed_at + 1]:
        total += v
    avg = total / period
    out[seed_at] = avg
    for i in range(seed_at + 1, n):
        avg = (avg * (period - 1) + data[i]) / period
        out[i] = avg
    return out


class WilderSmoother:
    """:func:`wilder_smooth` 의 증분 버전."""

    def __init__(self, period: int) -> None:
        if period < 1:
            raise ValueError("period 는 1 이상이어야 합니다")
        self.period = period
        self._count = 0
        self._total = 0.0
        self.value: float | None = None

    def update(self, x: float) -> float | None:
        if self.value is None:
            self._count += 1
            self._total += x
            if self._count == self.period:
                self.value = self._total / self.period
            return self.value
        self.value = (self.value * (self.period - 1) + x) / self.period
        return self.value
