"""OHLCV 리샘플링 (예: 1분봉 → 15분봉, 15분봉 → 4시간봉).

버킷은 epoch(1970-01-01 00:00 UTC) 기준으로 정렬한다. 레이블은 버킷 시작 시각이다.
그래서 15분봉은 매시 :00/:15/:30/:45, 4시간봉은 00/04/08/… UTC에 시작하고,
바이낸스 네이티브 봉과 경계가 같다. KRX 09:00 KST(= 00:00 UTC)도 15분 경계와 일치한다.

미래참조 방지: ``as_of`` 시각까지 끝나지 않은 버킷(진행 중인 봉)은 결과에서 뺀다.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from rsidiv.core.config import timeframe_minutes
from rsidiv.data.base import normalize_ohlcv

_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def resample_ohlcv(
    frame: pd.DataFrame,
    timeframe: str,
    *,
    source_timeframe: str,
    as_of: dt.datetime | None = None,
) -> pd.DataFrame:
    """``source_timeframe`` 봉을 ``timeframe`` 봉으로 합친다.

    Args:
        frame: 규격 OHLCV 프레임 (봉 시작 시각 인덱스).
        timeframe: 목표 타임프레임. ``source_timeframe`` 의 정수배여야 한다.
        source_timeframe: 입력 봉 길이.
        as_of: 이 시각까지 끝난 버킷만 남긴다. None 이면 입력 마지막 봉의 종료 시각을 쓴다.
            원천 데이터가 있어도 목표 버킷이 끝나지 않았으면 제외한다.

    원천 봉이 하나도 없는 버킷(거래 없음·거래정지)은 만들지 않는다. 결측 처리는
    :mod:`rsidiv.data.quality` 가 정책에 따라 맡는다.
    """
    target, source = timeframe_minutes(timeframe), timeframe_minutes(source_timeframe)
    if target % source != 0 or target < source:
        raise ValueError(f"{timeframe} 은 {source_timeframe} 의 정수배여야 합니다")
    if frame.empty:
        return normalize_ohlcv(frame)
    grouped = frame.resample(f"{target}min", label="left", closed="left", origin="epoch").agg(_AGG)
    grouped = grouped.dropna(subset=["open"])
    cutoff = (
        pd.Timestamp(as_of)
        if as_of is not None
        else frame.index[-1] + pd.Timedelta(minutes=source)
    )
    complete = grouped.index + pd.Timedelta(minutes=target) <= cutoff
    return normalize_ohlcv(grouped[complete])


@dataclass(frozen=True, slots=True)
class AggregatedBar:
    start: pd.Timestamp  # 버킷 시작 시각 (UTC)
    open: float
    high: float
    low: float
    close: float
    volume: float


class BarAggregator:
    """:func:`resample_ohlcv` 의 증분 버전 (실시간·신호 탐지기용).

    원천 봉을 시간순으로 한 개씩 넣으면, 그 시점에 완성된 상위 봉을 반환한다. 버킷이 완성되는 때는
    (1) 버킷의 마지막 원천 봉이 들어왔을 때 (그 봉의 종가 시각 = 버킷 종료 시각), 또는
    (2) 다음 버킷의 봉이 들어왔을 때(마지막 원천 봉이 없던 버킷, 예: 장 마감 후)다.
    두 경우 모두 반환 시점의 시각이 버킷 종료 시각 이후이므로 미래참조가 없다.
    """

    def __init__(self, timeframe: str, *, source_timeframe: str) -> None:
        target, source = timeframe_minutes(timeframe), timeframe_minutes(source_timeframe)
        if target % source != 0 or target < source:
            raise ValueError(f"{timeframe} 은 {source_timeframe} 의 정수배여야 합니다")
        self._target = pd.Timedelta(minutes=target)
        self._source = pd.Timedelta(minutes=source)
        self._start: pd.Timestamp | None = None
        self._ohlcv: list[float] = []

    def _bucket_start(self, ts: pd.Timestamp) -> pd.Timestamp:
        step = self._target.value
        return pd.Timestamp(ts.value // step * step, tz="UTC")

    def _finish(self) -> AggregatedBar:
        assert self._start is not None
        bar = AggregatedBar(self._start, *self._ohlcv)
        self._start, self._ohlcv = None, []
        return bar

    def update(
        self, time: dt.datetime | pd.Timestamp, open_: float, high: float, low: float,
        close: float, volume: float,
    ) -> list[AggregatedBar]:
        ts = pd.Timestamp(time).tz_convert("UTC")
        start = self._bucket_start(ts)
        done: list[AggregatedBar] = []
        if self._start is not None and start != self._start:
            if start < self._start:
                raise ValueError(f"봉 시각이 역행했습니다: {ts}")
            done.append(self._finish())
        if self._start is None:
            self._start, self._ohlcv = start, [open_, high, low, close, volume]
        else:
            o, h, lo, _, v = self._ohlcv
            self._ohlcv = [o, max(h, high), min(lo, low), close, v + volume]
        if ts + self._source == start + self._target:
            done.append(self._finish())
        return done
