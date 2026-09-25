"""Wilder ATR (손절폭·트레일링 스탑용).

``TR[i] = max(high-low, |high-close[i-1]|, |low-close[i-1]|)`` (i ≥ 1),
ATR 은 TR 의 Wilder 평활이며 첫 유효값은 인덱스 ``period`` 이다 (TA-Lib 과 같음).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from rsidiv.indicators.wilder import FloatArray, wilder_smooth


def true_range(high: npt.ArrayLike, low: npt.ArrayLike, close: npt.ArrayLike) -> FloatArray:
    h, lo, c = (np.asarray(x, dtype=np.float64) for x in (high, low, close))
    prev_close = np.concatenate([[np.nan], c[:-1]])
    tr = np.asarray(
        np.maximum.reduce([h - lo, np.abs(h - prev_close), np.abs(lo - prev_close)]), dtype=np.float64
    )
    tr[0] = np.nan  # 전일 종가가 없는 첫 봉은 사용하지 않음
    return tr


def atr_wilder(
    high: npt.ArrayLike, low: npt.ArrayLike, close: npt.ArrayLike, period: int = 14
) -> FloatArray:
    """ATR 배열 (앞 ``period`` 개는 NaN)."""
    return wilder_smooth(true_range(high, low, close), period, first=1)
