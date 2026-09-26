"""거래대금 순위 (시점별, 코인 단위).

- 코인 거래대금 = 그 코인의 순위 대상 계약(USDT·USDC·USD1 무기한) 거래대금 × USD 환산 비율의 합.
- 시각 T 의 24시간 거래대금 = 봉 시작 시각이 ``[T − 24h, T)`` 인 **마감된** 봉의 합. T 는 봉 경계(신호 시각)다.
- 순위 1 = 거래대금 최대. 거래대금이 0 인 코인(상장 전·폐지 후)은 순위가 없다(NaN).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from perpdiv.core.config import timeframe_minutes
from perpdiv.data.symbols import Contract


def coin_volume(frames: Mapping[Contract, pd.DataFrame], usd_per_quote: Mapping[str, float],
                timeframe: str) -> pd.DataFrame:
    """코인별 봉 거래대금(USD) 표. 인덱스 = 모든 봉 시작 시각의 연속 격자, 없는 봉은 0."""
    series: dict[str, pd.Series] = {}
    for contract, frame in frames.items():
        if frame.empty:
            continue
        qv = frame["quote_volume"].fillna(0.0) * usd_per_quote[contract.quote]
        series[contract.base] = series[contract.base].add(qv, fill_value=0.0) if contract.base in series else qv
    if not series:
        return pd.DataFrame()
    table = pd.DataFrame(series)
    step = pd.Timedelta(minutes=timeframe_minutes(timeframe))
    grid = pd.date_range(table.index.min(), table.index.max(), freq=step, tz="UTC")
    return table.reindex(grid, fill_value=0.0).fillna(0.0).sort_index(axis=1)


def rolling_volume(volume: pd.DataFrame, timeframe: str, window_hours: int) -> pd.DataFrame:
    """행 T = 봉 시작이 [T − window, T) 인 봉들의 합. 행 레이블은 경계 시각 T (마지막 봉 다음 경계까지 포함)."""
    step = pd.Timedelta(minutes=timeframe_minutes(timeframe))
    n = int(pd.Timedelta(hours=window_hours) / step)
    if n * step != pd.Timedelta(hours=window_hours):
        raise ValueError("window_hours 는 봉 길이의 정수배여야 합니다")
    summed = volume.rolling(n, min_periods=n).sum()
    summed.index = summed.index + step  # 봉 [t, t+step) 까지 합한 값은 경계 t+step 에서 알 수 있다
    return summed


def ranks(rolling: pd.DataFrame) -> pd.DataFrame:
    """행마다 거래대금 내림차순 순위 (1 = 최대). 0 이하·NaN 은 순위 없음."""
    positive = rolling.where(rolling > 0)
    return positive.rank(axis=1, ascending=False, method="first")


def rank_at(rank_table: pd.DataFrame, time: pd.Timestamp, coin: str) -> float:
    """경계 시각 time 에서의 코인 순위 (없으면 NaN)."""
    if time not in rank_table.index or coin not in rank_table.columns:
        return float("nan")
    position = int(pd.DatetimeIndex(rank_table.index).searchsorted(time))
    return float(rank_table[coin].to_numpy(dtype=float)[position])


def ever_ranked(rank_table: pd.DataFrame, limit: int) -> list[str]:
    """한 번이라도 순위 ≤ limit 이었던 코인."""
    hit = (rank_table <= limit).any(axis=0)
    return sorted(hit[hit].index)


def rank_windows(rank_table: pd.DataFrame, limit: int) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """코인별로 순위 ≤ limit 이었던 첫 시각과 마지막 시각."""
    out = {}
    for coin in rank_table.columns:
        mask = rank_table[coin] <= limit
        if mask.any():
            times = rank_table.index[mask.to_numpy()]
            out[coin] = (times[0], times[-1])
    return out


def weekend_ratio(daily_close: pd.DataFrame) -> pd.Series:
    """코인별 (주말 일간 |수익률| 중앙값) / (평일 중앙값). TradFi(주식·원자재 연동) 계약은 기초자산 시장이
    쉬는 주말에 가격이 거의 움직이지 않아 이 값이 작다. 거래 없는 날(가격 변화 0 연속)은 제외하지 않는다."""
    returns = daily_close.pct_change().abs()
    weekend = np.asarray(pd.DatetimeIndex(returns.index).weekday >= 5)
    return returns[weekend].median() / returns[~weekend].median()


US_SESSION_HOURS = tuple(range(13, 21))  # 13:00–20:59 UTC: 미국 정규장(13:30/14:30–20:00/21:00 UTC)
ASIA_SESSION_HOURS = tuple(range(0, 8))  # 00:00–07:59 UTC: 한국·일본(00:00–06:30), 홍콩·중국(01:30–08:00)


def session_share(hourly_close: pd.DataFrame, hours: tuple[int, ...]) -> pd.Series:
    """코인별: 평일 시간대(UTC)별 평균 |로그수익률| 중 ``hours`` 가 차지하는 비중. 24시간 균등하면 len(hours)/24.
    주식·ETF 연동 계약은 기초자산 거래소가 열린 시간에 변동이 몰려 이 값이 크다."""
    log_close = pd.DataFrame(np.log(hourly_close.to_numpy(dtype=float)), index=hourly_close.index,
                             columns=hourly_close.columns)
    moves = log_close.diff().abs()
    weekday = moves[np.asarray(pd.DatetimeIndex(moves.index).weekday < 5)]
    by_hour = weekday.groupby(pd.DatetimeIndex(weekday.index).hour).mean()
    inside = np.asarray(pd.Index(by_hour.index).isin(hours))
    return pd.Series(by_hour[inside].sum() / by_hour.sum())
