"""스윙 피벗(고점·저점) 판별과 확정 시점.

정의 (strict=True, 기본):
    피벗 저점 i  ⇔  v[i] < v[j]  (i-L ≤ j ≤ i-1, i+1 ≤ j ≤ i+R 인 모든 j)
    피벗 고점 i  ⇔  v[i] > v[j]  (같은 범위)
strict=False: 왼쪽은 엄격 비교, 오른쪽은 같은 값을 허용한다. 같은 저점(고점)이 연속되면
첫 봉 하나만 피벗이 된다.

미래참조 방지: 피벗 i 는 오른쪽 R 개 봉이 모두 마감된 뒤, 즉 ``confirm_index = i + R`` 봉의
종가가 확정된 시각에야 알 수 있다. 신호·전략 계층은 ``confirm_index`` 이후에만 피벗을
사용해야 한다. :class:`PivotTracker` 는 실시간으로 확정되는 순간에 피벗을 내보낸다.

가격 기준(price_source): ``low`` 는 저점 판별에 저가, 고점 판별에 고가를 쓴다.
``close`` 는 둘 다 종가를 쓴다.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from rsidiv.core.config import PivotCfg

PivotKind = Literal["low", "high"]


@dataclass(frozen=True, slots=True)
class Pivot:
    kind: PivotKind
    index: int  # 피벗 봉 위치
    confirm_index: int  # index + R. 이 봉 종가 확정 후부터 사용 가능
    price: float  # 판별 기준 가격 (price_source 에 따른 저가/고가/종가)


def _sources(frame: pd.DataFrame, price_source: str) -> tuple[npt.NDArray[np.float64], ...]:
    if price_source == "low":
        return frame["low"].to_numpy(np.float64), frame["high"].to_numpy(np.float64)
    if price_source == "close":
        close = frame["close"].to_numpy(np.float64)
        return close, close
    raise ValueError(f"알 수 없는 price_source: {price_source!r}")


def pivot_masks(
    low_src: npt.ArrayLike, high_src: npt.ArrayLike, left: int, right: int, strict: bool = True
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """피벗 저점·고점 여부 bool 배열 (피벗 봉 위치 기준). 좌우 봉이 부족한 양끝은 False."""
    lows = pd.Series(np.asarray(low_src, dtype=np.float64))
    highs = pd.Series(np.asarray(high_src, dtype=np.float64))

    def left_ext(s: pd.Series, fn: str) -> npt.NDArray[np.float64]:
        rolled = s.shift(1).rolling(left, min_periods=left)
        return np.asarray(getattr(rolled, fn)().to_numpy(), dtype=np.float64)

    def right_ext(s: pd.Series, fn: str) -> npt.NDArray[np.float64]:
        rev = s.iloc[::-1].reset_index(drop=True).shift(1).rolling(right, min_periods=right)
        return np.asarray(getattr(rev, fn)().to_numpy()[::-1], dtype=np.float64)

    lo, hi = lows.to_numpy(), highs.to_numpy()
    lo_left, lo_right = left_ext(lows, "min"), right_ext(lows, "min")
    hi_left, hi_right = left_ext(highs, "max"), right_ext(highs, "max")
    with np.errstate(invalid="ignore"):
        if strict:
            is_low = (lo < lo_left) & (lo < lo_right)
            is_high = (hi > hi_left) & (hi > hi_right)
        else:
            is_low = (lo < lo_left) & (lo <= lo_right)
            is_high = (hi > hi_left) & (hi >= hi_right)
    return is_low, is_high


def detect_pivots(frame: pd.DataFrame, cfg: PivotCfg) -> list[Pivot]:
    """프레임 전체의 피벗 목록 (확정 순서 = 피벗 위치 순)."""
    low_src, high_src = _sources(frame, cfg.price_source)
    is_low, is_high = pivot_masks(low_src, high_src, cfg.left, cfg.right, cfg.strict)
    pivots = [Pivot("low", int(i), int(i) + cfg.right, float(low_src[i])) for i in np.flatnonzero(is_low)]
    pivots += [Pivot("high", int(i), int(i) + cfg.right, float(high_src[i])) for i in np.flatnonzero(is_high)]
    pivots.sort(key=lambda p: (p.index, p.kind))
    return pivots


def pivots_frame(frame: pd.DataFrame, cfg: PivotCfg, timeframe_minutes: int) -> pd.DataFrame:
    """리포트·차트용: 피벗 시각과 확정 시각(확정 봉의 종가 시각, UTC)을 붙인 표."""
    pivots = detect_pivots(frame, cfg)
    bar_len = pd.Timedelta(minutes=timeframe_minutes)
    return pd.DataFrame(
        {
            "kind": [p.kind for p in pivots],
            "index": [p.index for p in pivots],
            "confirm_index": [p.confirm_index for p in pivots],
            "time": [frame.index[p.index] for p in pivots],
            "confirm_time": [frame.index[p.confirm_index] + bar_len for p in pivots],
            "price": [p.price for p in pivots],
        }
    )


class PivotTracker:
    """증분 피벗 판별기 (실시간). 봉을 한 개씩 넣으면 그 봉에서 확정된 피벗을 반환한다.

    n 번째 ``update`` 호출(0부터)에서 반환되는 피벗의 ``confirm_index`` 는 항상 n 이며,
    결과는 :func:`detect_pivots` 와 같다.
    """

    def __init__(self, cfg: PivotCfg) -> None:
        self.cfg = cfg
        size = cfg.left + cfg.right + 1
        self._lows: deque[float] = deque(maxlen=size)
        self._highs: deque[float] = deque(maxlen=size)
        self._count = 0

    @property
    def bars_seen(self) -> int:
        return self._count

    def update(self, high: float, low: float, close: float) -> list[Pivot]:
        source_low, source_high = (low, high) if self.cfg.price_source == "low" else (close, close)
        self._lows.append(source_low)
        self._highs.append(source_high)
        index = self._count
        self._count += 1
        if len(self._lows) < self._lows.maxlen:  # type: ignore[operator]
            return []
        left, right, strict = self.cfg.left, self.cfg.right, self.cfg.strict
        pivot_index = index - right
        found: list[Pivot] = []
        lows, highs = list(self._lows), list(self._highs)
        c_low, c_high = lows[left], highs[left]
        if all(c_low < v for v in lows[:left]) and all(
            (c_low < v) if strict else (c_low <= v) for v in lows[left + 1 :]
        ):
            found.append(Pivot("low", pivot_index, index, c_low))
        if all(c_high > v for v in highs[:left]) and all(
            (c_high > v) if strict else (c_high >= v) for v in highs[left + 1 :]
        ):
            found.append(Pivot("high", pivot_index, index, c_high))
        found.sort(key=lambda p: p.kind)
        return found
