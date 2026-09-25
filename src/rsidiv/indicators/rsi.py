"""Wilder RSI.

변화량 ``d[i] = close[i] - close[i-1]`` (i ≥ 1) 에서 상승분·하락분을 나누고,
각각 Wilder 평활한 평균으로 ``RSI = 100 * avg_gain / (avg_gain + avg_loss)`` 를 계산한다.
첫 유효값은 인덱스 ``period`` 이다 (TA-Lib 과 같음).

상승·하락이 모두 0 인 구간(완전 횡보)은 50 으로 정의한다. TA-Lib 은 이 경우 0 을 반환한다.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from rsidiv.indicators.wilder import FloatArray, WilderSmoother, wilder_smooth


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    total = avg_gain + avg_loss
    return 50.0 if total == 0.0 else 100.0 * avg_gain / total


def rsi_wilder(close: npt.ArrayLike, period: int = 14) -> FloatArray:
    """종가 배열 → RSI 배열 (앞 ``period`` 개는 NaN)."""
    c = np.asarray(close, dtype=np.float64)
    if np.isnan(c).any():
        raise ValueError("종가에 NaN 이 있습니다")
    delta = np.diff(c, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = wilder_smooth(gain, period, first=1)
    avg_loss = wilder_smooth(loss, period, first=1)
    out = np.full(len(c), np.nan)
    for i in np.flatnonzero(~np.isnan(avg_gain)):
        out[i] = _rsi_value(float(avg_gain[i]), float(avg_loss[i]))
    return out


class RsiState:
    """증분 RSI (실시간). ``update`` 결과는 :func:`rsi_wilder` 의 같은 위치 값과 비트 단위로 같다."""

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self._prev_close: float | None = None
        self._gain = WilderSmoother(period)
        self._loss = WilderSmoother(period)

    def update(self, close: float) -> float | None:
        if self._prev_close is None:
            self._prev_close = close
            return None
        delta = close - self._prev_close
        self._prev_close = close
        avg_gain = self._gain.update(delta if delta > 0 else 0.0)
        avg_loss = self._loss.update(-delta if delta < 0 else 0.0)
        if avg_gain is None or avg_loss is None:
            return None
        return _rsi_value(avg_gain, avg_loss)
