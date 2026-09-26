"""리샘플링: 5분봉 → 15m/1h/4h/1d (바이낸스와 같은 UTC epoch 경계).

- 버킷 레이블 = 버킷 시작 시각. 15m 은 :00/:15/:30/:45, 4h 는 00/04/…/20 UTC, 1d 는 00:00 UTC(= 09:00 KST).
- open=첫 봉 시가, high=최고, low=최저, close=마지막 봉 종가, volume·quote_volume·trades=합.
- 원천 봉이 하나도 없는 버킷은 만들지 않는다(결측). 원천 봉 일부만 있는 버킷은 있는 봉으로 계산하고 ``bars`` 에 개수를 남긴다.
- 미래참조 방지: ``as_of`` 까지 끝나지 않은 버킷(진행 중인 상위 봉)은 버린다.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from perpdiv.core.config import timeframe_minutes
from perpdiv.data.base import OHLCV_COLUMNS, normalize_ohlcv

_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "quote_volume": "sum",
        "trades": "sum"}


def resample_ohlcv(frame: pd.DataFrame, timeframe: str, *, source_timeframe: str,
                   as_of: dt.datetime | None = None) -> pd.DataFrame:
    """규격 OHLCV + ``bars``(버킷에 들어간 원천 봉 수) 열."""
    target, source = timeframe_minutes(timeframe), timeframe_minutes(source_timeframe)
    if target % source != 0:
        raise ValueError(f"{timeframe} 은 {source_timeframe} 의 정수배여야 합니다")
    if frame.empty:
        out = normalize_ohlcv(frame)
        out["bars"] = pd.Series(dtype="int64")
        return out
    grouped = frame.resample(f"{target}min", label="left", closed="left", origin="epoch")
    out = grouped.agg(_AGG)
    out["bars"] = grouped["close"].count()
    out = out[out["bars"] > 0]
    cutoff = pd.Timestamp(as_of) if as_of is not None else frame.index[-1] + pd.Timedelta(minutes=source)
    out = out[out.index + pd.Timedelta(minutes=target) <= cutoff]
    bars = out["bars"].astype("int64")
    result = normalize_ohlcv(out[list(OHLCV_COLUMNS)])
    result["bars"] = bars
    return result


@dataclass(frozen=True, slots=True)
class AggregatedBar:
    start: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trades: int
    bars: int


class BarAggregator:
    """:func:`resample_ohlcv` 의 증분 버전 (실시간). 원천 봉을 시간순으로 넣으면 완성된 상위 봉을 돌려준다.

    완성 시점: 버킷의 마지막 원천 봉이 들어왔을 때(그 봉 종가 시각 = 버킷 종료) 또는 다음 버킷의 봉이 들어왔을 때.
    """

    def __init__(self, timeframe: str, *, source_timeframe: str) -> None:
        target, source = timeframe_minutes(timeframe), timeframe_minutes(source_timeframe)
        if target % source != 0:
            raise ValueError(f"{timeframe} 은 {source_timeframe} 의 정수배여야 합니다")
        self._target = pd.Timedelta(minutes=target)
        self._source = pd.Timedelta(minutes=source)
        self._start: pd.Timestamp | None = None
        self._last: pd.Timestamp | None = None
        self._acc: list[float] = []
        self._trades = 0
        self._count = 0

    def _finish(self) -> AggregatedBar:
        assert self._start is not None
        o, h, lo, c, v, qv = self._acc
        bar = AggregatedBar(self._start, o, h, lo, c, v, qv, self._trades, self._count)
        self._start, self._acc, self._trades, self._count = None, [], 0, 0
        return bar

    def update(self, time: dt.datetime | pd.Timestamp, open_: float, high: float, low: float, close: float,
               volume: float, quote_volume: float, trades: int) -> list[AggregatedBar]:
        ts = pd.Timestamp(time).tz_convert("UTC")
        if self._last is not None and ts <= self._last:
            raise ValueError(f"봉 시각이 역행·중복했습니다: {ts} (직전 {self._last})")
        self._last = ts
        step = self._target.value
        start = pd.Timestamp(ts.value // step * step, tz="UTC")
        done: list[AggregatedBar] = []
        if self._start is not None and start != self._start:
            done.append(self._finish())
        if self._start is None:
            self._start, self._acc = start, [open_, high, low, close, volume, quote_volume]
        else:
            o, h, lo, _, v, qv = self._acc
            self._acc = [o, max(h, high), min(lo, low), close, v + volume, qv + quote_volume]
        self._trades += trades
        self._count += 1
        if ts + self._source == start + self._target:
            done.append(self._finish())
        return done
