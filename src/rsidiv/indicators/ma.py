"""이동평균 (추세 필터용). EMA 는 첫 ``period`` 개 단순평균을 seed 로 쓴다 (TA-Lib 과 같음)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from rsidiv.indicators.wilder import FloatArray


def sma(values: npt.ArrayLike, period: int) -> FloatArray:
    """단순이동평균. 앞 ``period-1`` 개는 NaN."""
    v = np.asarray(values, dtype=np.float64)
    out = np.full(len(v), np.nan)
    if len(v) >= period:
        window = np.lib.stride_tricks.sliding_window_view(v, period)
        out[period - 1 :] = window.mean(axis=1)
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
