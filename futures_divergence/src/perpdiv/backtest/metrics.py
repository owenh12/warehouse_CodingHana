"""성과 지표.

- 수익률 지표는 평가금액 곡선에서: 총수익률, CAGR(365일), MDD(기록된 모든 평가 시점 기준), 샤프·소르티노(일별 수익률,
  하루 = 표시 시간대 00:00 경계, 연 365일, 무위험 수익률 설정값), 칼마(CAGR / MDD), 연도별 수익률.
- 거래 지표: 거래 수, 승률(순손익 > 0), 손익비(평균 이익 / 평균 손실 절대값), 기대값(R), 평균 보유 시간, 이익 계수.
"""

from __future__ import annotations

import math
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

TRADE_COLUMNS = ("trades", "win_rate", "payoff", "expectancy_r", "profit_factor", "avg_hold_h", "net_pnl")


def daily_equity(equity: pd.Series, start: pd.Timestamp, end: pd.Timestamp, tz: ZoneInfo) -> pd.Series:
    """표시 시간대 하루 경계마다의 평가금액 (직전 기록값)."""
    first = start.tz_convert(tz).normalize()
    grid = pd.date_range(first, end.tz_convert(tz), freq="D").tz_convert("UTC")
    grid = grid[(grid > start) & (grid <= end)]
    stamps = pd.DatetimeIndex([start, *grid, end]).unique()
    return equity.sort_index().reindex(equity.index.union(stamps)).ffill().reindex(stamps)


def max_drawdown(equity: pd.Series) -> float:
    values = equity.to_numpy(dtype=float)
    if len(values) == 0:
        return 0.0
    peak = np.maximum.accumulate(values)
    return float(np.max(1 - values / peak))


def performance(equity: pd.Series, start: pd.Timestamp, end: pd.Timestamp, tz: ZoneInfo,
                risk_free: float = 0.0) -> dict[str, float]:
    daily = daily_equity(equity, start, end, tz)
    first, last = float(daily.iloc[0]), float(daily.iloc[-1])
    days = (end - start).total_seconds() / 86400
    total = last / first - 1
    cagr = (last / first) ** (365 / days) - 1 if last > 0 and days > 0 else -1.0
    rets = daily.pct_change().dropna()
    excess = rets - risk_free / 365
    std = float(excess.std(ddof=1)) if len(excess) > 1 else float("nan")
    sharpe = float(excess.mean()) / std * math.sqrt(365) if std and std > 0 else float("nan")
    downside = float(np.sqrt(np.mean(np.minimum(excess.to_numpy(), 0.0) ** 2))) if len(excess) else float("nan")
    sortino = float(excess.mean()) / downside * math.sqrt(365) if downside and downside > 0 else float("nan")
    mdd = max_drawdown(equity)
    return {"total_return": total, "cagr": cagr, "mdd": mdd, "sharpe": sharpe, "sortino": sortino,
            "calmar": cagr / mdd if mdd > 0 else float("nan"), "final_equity": last, "days": days}


def yearly_returns(equity: pd.Series, start: pd.Timestamp, end: pd.Timestamp, tz: ZoneInfo) -> dict[int, float]:
    """표시 시간대 달력 연도별 수익률 (첫해·마지막 해는 구간 안의 부분)."""
    daily = daily_equity(equity, start, end, tz)
    values = pd.Series(daily.to_numpy(dtype=float), index=pd.DatetimeIndex(daily.index))
    out = {}
    local_start, local_end = start.tz_convert(tz), end.tz_convert(tz)
    for year in range(local_start.year, local_end.year + 1):
        a = max(pd.Timestamp(f"{year}-01-01", tz=tz).tz_convert("UTC"), start)
        b = min(pd.Timestamp(f"{year + 1}-01-01", tz=tz).tz_convert("UTC"), end)
        if a < b:
            out[year] = float(values[values.index <= b].iloc[-1]) / float(values[values.index <= a].iloc[-1]) - 1
    return out


def trade_stats(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {"trades": 0, "win_rate": float("nan"), "payoff": float("nan"), "expectancy_r": float("nan"),
                "profit_factor": float("nan"), "avg_hold_h": float("nan"), "net_pnl": 0.0}
    pnl = trades["net_pnl"]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    avg_loss = float(-losses.mean()) if len(losses) else float("nan")
    return {
        "trades": float(len(trades)),
        "win_rate": float(len(wins) / len(trades)),
        "payoff": float(wins.mean() / avg_loss) if len(wins) and avg_loss and avg_loss > 0 else float("nan"),
        "expectancy_r": float(trades["r_multiple"].mean()),
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else float("nan"),
        "avg_hold_h": float(trades["holding_hours"].mean()),
        "net_pnl": float(pnl.sum()),
    }


def split_stats(trades: pd.DataFrame, by: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=list(TRADE_COLUMNS))
    rows = {key: trade_stats(group) for key, group in trades.groupby(by, sort=True)}
    return pd.DataFrame.from_dict(rows, orient="index")
