"""OHLCV 리샘플링 (예: 1분봉 → 15분봉, 15분봉 → 4시간봉).

버킷은 epoch(1970-01-01 00:00 UTC) 기준으로 정렬한다. 레이블은 버킷 시작 시각이다.
그래서 15분봉은 매시 :00/:15/:30/:45, 4시간봉은 00/04/08/… UTC에 시작하고,
바이낸스 네이티브 봉과 경계가 같다. KRX 09:00 KST(= 00:00 UTC)도 15분 경계와 일치한다.

미래참조 방지: ``as_of`` 시각까지 끝나지 않은 버킷(진행 중인 봉)은 결과에서 뺀다.
"""

from __future__ import annotations

import datetime as dt

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
