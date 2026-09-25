"""이동평균 (추세 필터용). EMA 는 첫 ``period`` 개 단순평균을 seed 로 쓴다 (TA-Lib 과 같음).

일괄 함수(:func:`sma`, :func:`ema`)와 증분 클래스(:class:`SmaState`, :class:`EmaState`)는
같은 연산을 같은 순서로 수행해 결과가 비트 단위로 같다. SMA 는 합산 순서에 영향받지 않도록
``math.fsum``(정확한 합)을 쓴다.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import numpy.typing as npt

from rsidiv.indicators.wilder import FloatArray


def sma(values: npt.ArrayLike, period: int) -> FloatArray:
    """단순이동평균. 앞 ``period-1`` 개는 NaN."""
    data = np.asarray(values, dtype=np.float64).tolist()
    out = np.full(len(data), np.nan)
    for i in range(period - 1, len(data)):
        out[i] = math.fsum(data[i - period + 1 : i + 1]) / period
    return out


def ema(values: npt.ArrayLike, period: int) -> FloatArray:
    """지수이동평균 ``k = 2/(period+1)``. 앞 ``period-1`` 개는 NaN."""
    v = np.asarray(values, dtype=np.float64)
    out = np.full(len(v), np.nan)
    if len(v) < period:
        return out
    k = 2.0 / (period + 1)
    data = v.tolist()
    total = 0.0
    for x in data[:period]:
        total += x
    prev = total / period
    out[period - 1] = prev
    for i in range(period, len(data)):
        prev = (data[i] - prev) * k + prev
        out[i] = prev
    return out


def moving_average(values: npt.ArrayLike, period: int, kind: str) -> FloatArray:
    if kind == "sma":
        return sma(values, period)
    if kind == "ema":
        return ema(values, period)
    raise ValueError(f"알 수 없는 이동평균 종류: {kind!r}")


class SmaState:
    """:func:`sma` 의 증분 버전 (봉 1개씩)."""

    def __init__(self, period: int) -> None:
        self.period = period
        self._window: deque[float] = deque(maxlen=period)
        self.value: float | None = None

    def update(self, x: float) -> float | None:
        self._window.append(x)
        if len(self._window) == self.period:
            self.value = math.fsum(self._window) / self.period
        return self.value


class EmaState:
    """:func:`ema` 의 증분 버전 (봉 1개씩)."""

    def __init__(self, period: int) -> None:
        self.period = period
        self._k = 2.0 / (period + 1)
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
        self.value = (x - self.value) * self._k + self.value
        return self.value


def moving_average_state(period: int, kind: str) -> SmaState | EmaState:
    if kind == "sma":
        return SmaState(period)
    if kind == "ema":
        return EmaState(period)
    raise ValueError(f"알 수 없는 이동평균 종류: {kind!r}")
