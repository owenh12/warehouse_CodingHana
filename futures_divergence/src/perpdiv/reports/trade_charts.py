"""거래 표본 차트: 신호 TF 봉 위에 t1(p1)·p2(t2)·t3(p3), 진입·청산, 손절·익절선."""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from perpdiv.reports.signal_charts import DOWN, UP, _candles, _labels


def _pos(index: pd.DatetimeIndex, t: pd.Timestamp) -> float:
    """시각 → 봉 위치 (봉 사이면 비례 보간)."""
    i = int(index.searchsorted(t, side="right")) - 1
    if i < 0:
        return 0.0
    if i >= len(index) - 1:
        return float(len(index) - 1)
    span = (index[i + 1] - index[i]).total_seconds()
    return i + (t - index[i]).total_seconds() / span if span > 0 else float(i)


def plot_trade(bars: pd.DataFrame, trade: pd.Series, signal: pd.Series, *, tz: ZoneInfo, path: Path,
               number: int) -> None:
    long = trade["side"] == "long"
    idx = pd.DatetimeIndex(bars.index)
    x = np.arange(len(bars))
    fig, ax = plt.subplots(figsize=(11, 5.2))
    _candles(ax, bars, x)
    names = ("t1", "p2", "t3") if long else ("p1", "t2", "p3")
    color = UP if long else DOWN
    points = [(signal["anchor_time"], signal["anchor_price"], names[0]), (signal["swing_time"], signal["swing_price"], names[1]),
              (signal["trigger_time"], signal["trigger_close"], names[2])]
    for t, price, name in points:
        xi = _pos(idx, pd.Timestamp(t))
        ax.scatter([xi], [price], s=60, color="#ffd92f" if name in ("p2", "t2") else color, edgecolors="black", zorder=6)
        ax.annotate(name, (xi, price), (0, 8), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    xe, xx = _pos(idx, pd.Timestamp(trade["entry_time"])), _pos(idx, pd.Timestamp(trade["exit_time"]))
    ax.scatter([xe], [trade["entry_price"]], marker=">", s=80, color="black", zorder=7, label=f"entry {trade['entry_price']:,.6g}")
    ax.scatter([xx], [trade["exit_price"]], marker="X", s=80, color="#6a3d9a", zorder=7,
               label=f"exit {trade['exit_price']:,.6g} ({trade['exit_reason']})")
    ax.hlines(trade["stop"], xe, max(xx, xe + 1), colors="#7f0000", linewidth=0.9, label=f"stop {trade['stop']:,.6g}")
    if trade["target"] == trade["target"]:
        ax.hlines(trade["target"], xe, max(xx, xe + 1), colors="#00441b", linewidth=0.9,
                  label=f"target {trade['target']:,.6g}")
    ax.legend(loc="best", fontsize=7, framealpha=0.85)
    ax.grid(alpha=0.2)
    _labels(ax, idx, x, tz, "%m-%d %H:%M" if trade["timeframe"] != "1d" else "%y-%m-%d")
    entry_local = pd.Timestamp(trade["entry_time"]).tz_convert(tz)
    ax.set_title(f"#{number} {trade['coin']} {trade['timeframe']} {trade['side'].upper()} (rank {trade['rank']:.0f})  "
                 f"entry {entry_local:%Y-%m-%d %H:%M} {tz.key}  |  {trade['r_multiple']:+.2f}R  "
                 f"net {trade['net_pnl']:+,.1f} USDT  |  held {trade['holding_hours']:.1f}h", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def plot_equity(curves: dict[str, pd.Series], *, tz: ZoneInfo, path: Path, title: str) -> None:
    fig, (ax, ax_dd) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    for name, curve in curves.items():
        local = curve.copy()
        local.index = pd.DatetimeIndex(local.index).tz_convert(tz).tz_localize(None)
        style = {"linewidth": 1.6} if name.startswith("strategy") else {"linewidth": 1.0, "alpha": 0.8}
        ax.plot(local.index, local.to_numpy(), label=name, **style)
        dd = 1 - local / local.cummax()
        ax_dd.plot(local.index, -dd.to_numpy() * 100, label=name, **style)
    ax.set_yscale("log")
    ax.set_ylabel("equity (USDT, log)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")
    ax_dd.set_ylabel("drawdown %")
    ax_dd.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
