"""성과 지표.

- 수익률 지표(CAGR, 샤프, 소르티노, 변동성)는 **일별** 평가금액(표시 시간대 KST 기준 날짜의 마지막 값)으로 계산하고
  연율화는 365일(가상화폐는 매일 거래)을 쓴다. 무위험 수익률은 backtest.yaml 의 ``risk_free_rate``.
- MDD 는 봉 단위(15분) 평가금액으로 계산한다.
- 거래 지표는 청산된 거래 기준. R = 순손익 / (수량 × (진입가 − 최초 손절가)). 기대값(R) = 평균 R.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rsidiv.backtest.engine import BacktestResult
from rsidiv.broker.costs import CostModel
from rsidiv.broker.instruments import CryptoInstrument
from rsidiv.core.models import Liquidity, Side
from rsidiv.strategy.controller import TradeRecord

PERIODS_PER_YEAR = 365


@dataclass(slots=True)
class ReturnMetrics:
    start: pd.Timestamp
    end: pd.Timestamp
    initial: float
    final: float
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    max_drawdown_days: float  # 최고점 회복까지 가장 오래 걸린(또는 아직 회복 못 한) 기간
    calmar: float


@dataclass(slots=True)
class TradeMetrics:
    trades: int
    win_rate: float
    expectancy_r: float
    median_r: float
    avg_win_r: float
    avg_loss_r: float
    profit_factor: float
    gross_expectancy_r: float
    avg_bars_held: float
    max_consecutive_losses: int
    avg_mae_r: float
    avg_mfe_r: float
    net_pnl: float
    gross_pnl: float
    fees: float
    tax: float
    funding: float
    exits: dict[str, int] = field(default_factory=dict)


def daily_equity(equity: pd.Series, initial: float, tz: str) -> pd.Series:
    """일별 마지막 평가금액 (표시 시간대 날짜). 첫 값 앞에 시작 자본을 둔다."""
    local = pd.DatetimeIndex(equity.index).tz_convert(tz)
    daily = equity.groupby(local.normalize()).last()
    start = daily.index[0] - pd.Timedelta(days=1)
    out: pd.Series = pd.concat([pd.Series([initial], index=[start]), daily])
    return out


def return_metrics(equity: pd.Series, initial: float, tz: str, *, risk_free_rate: float = 0.0) -> ReturnMetrics:
    if equity.empty:
        raise ValueError("평가금액 시계열이 비었습니다")
    final = float(equity.iloc[-1])
    start, end = pd.Timestamp(equity.index[0]), pd.Timestamp(equity.index[-1])
    days = max((end - start).total_seconds() / 86_400, 1e-9)
    total = final / initial - 1
    cagr = (final / initial) ** (365.25 / days) - 1 if final > 0 else -1.0
    daily = daily_equity(equity, initial, tz).pct_change().dropna()
    excess = daily - risk_free_rate / PERIODS_PER_YEAR
    std = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    downside = float(np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2))) if len(daily) else 0.0
    ann = math.sqrt(PERIODS_PER_YEAR)
    sharpe = float(excess.mean()) / std * ann if std > 0 else math.nan
    sortino = float(excess.mean()) / downside * ann if downside > 0 else math.nan

    values = np.concatenate([[initial], equity.to_numpy(dtype=float)])
    times = pd.DatetimeIndex([start - pd.Timedelta(minutes=15), *equity.index])
    peak = np.maximum.accumulate(values)
    drawdown = 1 - values / peak
    mdd = float(drawdown.max())
    longest, run_start = 0.0, None
    for t, dd in zip(times, drawdown, strict=True):
        if dd > 0 and run_start is None:
            run_start = t
        elif dd == 0 and run_start is not None:
            longest = max(longest, (t - run_start).total_seconds() / 86_400)
            run_start = None
    if run_start is not None:
        longest = max(longest, (times[-1] - run_start).total_seconds() / 86_400)
    calmar = cagr / mdd if mdd > 0 else math.nan
    return ReturnMetrics(start, end, initial, final, total, cagr, std * ann, sharpe, sortino, mdd, longest, calmar)


def trade_metrics(trades: Sequence[TradeRecord]) -> TradeMetrics:
    closed = [t for t in trades if not t.is_open]
    r = np.array([t.r_multiple for t in closed], dtype=float)
    r = r[~np.isnan(r)]
    net = np.array([t.net_pnl for t in closed], dtype=float)
    wins, losses = net[net > 0], net[net <= 0]
    streak = longest = 0
    for value in net:
        streak = streak + 1 if value <= 0 else 0
        longest = max(longest, streak)

    def mean(values: Sequence[float] | np.ndarray) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        return float(arr.mean()) if len(arr) else math.nan

    return TradeMetrics(
        trades=len(closed),
        win_rate=float(len(wins) / len(closed)) if closed else math.nan,
        expectancy_r=mean(r),
        median_r=float(np.median(r)) if len(r) else math.nan,
        avg_win_r=mean(r[r > 0]),
        avg_loss_r=mean(r[r <= 0]),
        profit_factor=float(wins.sum() / -losses.sum()) if losses.sum() < 0 else math.inf if len(wins) else math.nan,
        gross_expectancy_r=mean([t.gross_r for t in closed]),
        avg_bars_held=mean([t.bars_held for t in closed]),
        max_consecutive_losses=longest,
        avg_mae_r=mean([t.mae_r for t in closed]),
        avg_mfe_r=mean([t.mfe_r for t in closed]),
        net_pnl=float(net.sum()),
        gross_pnl=float(sum(t.gross_pnl for t in closed)),
        fees=float(sum(t.entry_fee + t.exit_fee for t in closed)),
        tax=float(sum(t.tax for t in closed)),
        funding=float(sum(t.funding for t in closed)),
        exits=dict(Counter(t.exit_reason or "-" for t in closed)),
    )


def exposure(result: BacktestResult) -> float:
    """보유 포지션이 하나라도 있었던 봉의 비율."""
    return float((result.equity["positions"] > 0).mean()) if len(result.equity) else math.nan


def buy_and_hold(
    data: Mapping[str, pd.DataFrame], initial: float, *, costs: CostModel,
    instruments: Mapping[str, CryptoInstrument],
) -> pd.Series:
    """종목 균등 배분 매수 후 보유. 첫 봉 시가에 테이커 비용으로 매수하고 매도 비용은 넣지 않는다.

    반환: 봉 종가 확정 시각 인덱스의 평가금액.
    """
    symbols = list(data)
    budget = initial / len(symbols)
    closes = pd.concat({s: data[s]["close"] for s in symbols}, axis=1).ffill()
    first = data[symbols[0]].index[0]
    units: dict[str, float] = {}
    cash = initial
    for s in symbols:
        price = costs.fill_price(Side.BUY, float(data[s]["open"].iloc[0]), Liquidity.TAKER, instruments[s])
        fee_rate, _ = costs.fee_rates(Side.BUY, Liquidity.TAKER, first.to_pydatetime())
        units[s] = budget / (price * (1 + fee_rate))
        cash -= budget
    value = pd.Series(cash, index=closes.index, dtype=float)
    for s in symbols:
        value = value + closes[s] * units[s]
    value.index = value.index + pd.Timedelta(minutes=15)
    return value.rename("buy_and_hold")


def per_symbol(trades: Sequence[TradeRecord]) -> pd.DataFrame:
    rows = []
    for symbol in sorted({t.symbol for t in trades}):
        m = trade_metrics([t for t in trades if t.symbol == symbol])
        rows.append({"symbol": symbol, "trades": m.trades, "win_rate": m.win_rate, "expectancy_r": m.expectancy_r,
                     "gross_expectancy_r": m.gross_expectancy_r, "profit_factor": m.profit_factor,
                     "net_pnl": m.net_pnl})
    return pd.DataFrame(rows)


def period_returns(equity: pd.Series, initial: float, tz: str, freq: str = "YE") -> pd.Series:
    """기간별 수익률 (기본: 연도별). 첫 기간은 시작 자본 대비."""
    daily = daily_equity(equity, initial, tz).iloc[1:]
    ends = daily.resample(freq).last()
    starts = pd.Series([initial, *ends.iloc[:-1].tolist()], index=ends.index)
    return (ends / starts - 1).rename("return")
