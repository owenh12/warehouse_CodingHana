"""RSI (바이낸스 차트 = TradingView ``ta.rsi`` 기준): 종가, 기간 14, Wilder(RMA) 평활.

    d[i] = close[i] − close[i−1] (i ≥ 1),  gain = max(d, 0),  loss = max(−d, 0)
    avg_gain, avg_loss = RMA(gain), RMA(loss)   (첫 평균 = 첫 period 개 변화량의 단순평균)
    RSI = 100 × avg_gain / (avg_gain + avg_loss)

워밍업: 첫 유효값은 인덱스 ``period`` (변화량 period 개 필요). 그 앞은 NaN.
RMA 는 과거 값의 가중치가 (1 − 1/period)^k 로 줄어든다 (period=14: k=186 에서 1e−6). 시작점이 다른 두 RSI 의
값 차이는 실측(일간 변동 1% 랜덤워크 150경로) 최악 기준 첫 유효값 뒤 약 120봉에 0.01, 250봉에 1e−6 아래가 된다.
백테스트는 ``data.warmup_days``(200일 → 1d 도 200봉) 앞선 데이터부터 지표를 계산하고 그 구간 신호는 버린다.
상장 직후 코인은 상장 첫 봉부터 계산한다 (바이낸스 차트도 상장 첫 봉부터 그리므로 같은 값).

상승·하락이 모두 0 인 완전 횡보 구간은 50 으로 정의한다 (TradingView 는 이 경우 100 − 100/(1+0/0) = NaN 처리 후 이전 값).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from perpdiv.indicators.wilder import FloatArray, WilderSmoother, wilder_smooth


def _value(avg_gain: float, avg_loss: float) -> float:
    total = avg_gain + avg_loss
    return 50.0 if total == 0.0 else 100.0 * avg_gain / total


def rsi_wilder(close: npt.ArrayLike, period: int = 14) -> FloatArray:
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
        out[i] = _value(float(avg_gain[i]), float(avg_loss[i]))
    return out


class RsiState:
    """증분 RSI. ``update`` 결과는 :func:`rsi_wilder` 의 같은 위치 값과 비트 단위로 같다."""

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self._prev: float | None = None
        self._gain = WilderSmoother(period)
        self._loss = WilderSmoother(period)
        self.value: float | None = None

    def update(self, close: float) -> float | None:
        if self._prev is None:
            self._prev = close
            return None
        delta = close - self._prev
        self._prev = close
        g = self._gain.update(delta if delta > 0 else 0.0)
        lo = self._loss.update(-delta if delta < 0 else 0.0)
        self.value = None if g is None or lo is None else _value(g, lo)
        return self.value
