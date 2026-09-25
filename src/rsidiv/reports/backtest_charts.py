"""백테스트 차트 (PNG): 수익률 곡선, 낙폭, R 분포, 거래 표본.

색: 기준 전략 파랑, 비용 반영 전 주황, 매수 후 보유 청록 (검증된 범주형 팔레트의 앞 3칸).
수익률 곡선은 모두 시작 대비 %로 같은 축에 그린다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter

from rsidiv.reports.signal_charts import (
    AXIS,
    CONFIRM_BAND,
    DIVERGENCE,
    DOWN,
    INK,
    INK_2,
    MUTED,
    SURFACE,
    UP,
    _marker,
    _price_fmt,
    _time_ticks,
)
from rsidiv.strategy.controller import TradeRecord

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
_REASON_KO = {"take_profit": "익절", "stop_loss": "손절", "trailing_stop": "트레일링 손절", "time_exit": "시간 청산",
              "session_flatten": "장 마감 청산", "kill_switch": "킬 스위치", "end_of_test": "데이터 끝"}


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_returns(curves: Mapping[str, pd.Series], initials: Mapping[str, float], *, title: str, display_tz: str,
                 path: Path) -> Path:
    """시작 대비 누적 수익률 (%). ``curves``: 이름 → 평가금액 시계열(순서대로 색 배정)."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for (name, series), color in zip(curves.items(), SERIES, strict=False):
        pct = series / initials[name] - 1
        x = pd.DatetimeIndex(pct.index).tz_convert(display_tz)
        ax.plot(x, pct.to_numpy(), color=color, linewidth=2 if len(pct) < 5000 else 1.2, label=name)
        ax.annotate(f"{pct.iloc[-1]:+.1%}", (x[-1], pct.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title(title, loc="left", fontsize=13, color=INK, fontweight="bold", pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(curves), frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.93, top=0.85, bottom=0.1)
    return _save(fig, path)


def plot_drawdown(equity: pd.Series, initial: float, *, title: str, display_tz: str, path: Path) -> Path:
    values = np.concatenate([[initial], equity.to_numpy(dtype=float)])
    peak = np.maximum.accumulate(values)
    dd = -(1 - values / peak)[1:]
    x = pd.DatetimeIndex(equity.index).tz_convert(display_tz)
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.fill_between(x, dd, 0, color=SERIES[0], alpha=0.12, linewidth=0)
    ax.plot(x, dd, color=SERIES[0], linewidth=1.2)
    worst = int(np.argmin(dd))
    ax.annotate(f"MDD {dd[worst]:.1%}", (x[worst], dd[worst]), xytext=(6, -4), textcoords="offset points",
                fontsize=9, color=INK)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title(title, loc="left", fontsize=13, color=INK, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.14)
    return _save(fig, path)


def plot_r_distribution(trades: Sequence[TradeRecord], *, title: str, path: Path) -> Path:
    r = np.array([t.r_multiple for t in trades], dtype=float)
    r = r[~np.isnan(r)]
    fig, ax = plt.subplots(figsize=(9, 4))
    if len(r):
        lo, hi = np.floor(min(r.min(), -1.5) * 4) / 4, np.ceil(max(r.max(), 2.5) * 4) / 4
        bins = np.arange(lo, hi + 0.25, 0.25)
        ax.hist(r, bins=bins.tolist(), color=SERIES[0], edgecolor=SURFACE, linewidth=2)
        ax.axvline(float(r.mean()), color=INK, linewidth=1)
        ax.annotate(f"평균 {r.mean():+.2f}R", (float(r.mean()), 1.0), xycoords=("data", "axes fraction"),
                    xytext=(4, -12), textcoords="offset points", fontsize=9, color=INK)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_xlabel("거래당 R (순손익 ÷ 최초 위험)")
    ax.set_ylabel("거래 수")
    ax.set_title(title, loc="left", fontsize=13, color=INK, fontweight="bold")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.14)
    return _save(fig, path)


def plot_trade(frame: pd.DataFrame, rsi: np.ndarray, trade: TradeRecord, *, title: str, display_tz: str,
               path: Path) -> Path:
    """거래 하나: t1·p2·t3, 신호 시각, 진입(▲)·청산(▼), 최초 손절·최종 손절·목표가."""
    sig = trade.signal
    idx = frame.index

    def pos(ts: pd.Timestamp | None) -> int | None:
        return int(idx.searchsorted(ts)) if ts is not None and ts in idx else None

    t1, p2, t3 = pos(sig.t1_time), pos(sig.p2_time), pos(sig.t3_time)
    entry, exit_ = pos(trade.entry_time), pos(trade.exit_time)
    assert t3 is not None and entry is not None
    exit_i = exit_ if exit_ is not None else len(frame) - 1
    lo, hi = max(0, (t1 if t1 is not None else t3) - 8), min(len(frame), exit_i + 9)
    view = frame.iloc[lo:hi]
    x = np.arange(lo, hi)
    times = pd.DatetimeIndex(view.index).tz_convert(display_tz)

    fig, (ax, ax_rsi) = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1.1], "hspace": 0.08})
    for a in (ax, ax_rsi):
        a.axvspan(entry - 0.5, exit_i + 0.5, color=CONFIRM_BAND, zorder=0, linewidth=0)
    o, h, low, cl = (view[k].to_numpy() for k in ("open", "high", "low", "close"))
    colors = np.where(cl >= o, UP, DOWN)
    ax.vlines(x, low, h, colors=colors, linewidth=1, zorder=2)
    span = h.max() - low.min()
    ax.bar(x, np.maximum(np.abs(cl - o), span * 0.002), bottom=np.minimum(o, cl), width=0.62, color=colors,
           linewidth=0, zorder=3)

    if t1 is not None:
        ax.plot([t1, t3], [sig.price_t1, sig.price_t3], color=DIVERGENCE, linewidth=2, zorder=4)
        ax_rsi.plot([t1, t3], [sig.rsi_t1, sig.rsi_t3], color=DIVERGENCE, linewidth=2, zorder=4)
        for i, price, value in ((t1, sig.price_t1, sig.rsi_t1), (t3, sig.price_t3, sig.rsi_t3)):
            _marker(ax, i, price, "o", DIVERGENCE, size=7)
            _marker(ax_rsi, i, value, "o", DIVERGENCE, size=6)
    if p2 is not None:
        _marker(ax, p2, sig.price_p2, "v", INK, size=7)

    ax.hlines(trade.initial_stop, entry - 0.4, exit_i + 0.4, colors=DOWN, linewidth=1.2, zorder=4)
    if trade.final_stop > trade.initial_stop:
        ax.hlines(trade.final_stop, entry - 0.4, exit_i + 0.4, colors=DOWN, linewidth=1.2, linestyles=(0, (4, 3)),
                  zorder=4)
    if trade.target is not None:
        ax.hlines(trade.target, entry - 0.4, exit_i + 0.4, colors=UP, linewidth=1.2, zorder=4)
    ax.axvline(entry - 0.5, color=INK, linewidth=1, zorder=4)
    _marker(ax, entry, trade.entry_price, "^", INK, size=10)
    if trade.exit_price is not None:
        _marker(ax, exit_i, trade.exit_price, "v", INK, size=10)

    ax_rsi.plot(x, rsi[lo:hi], color=INK_2, linewidth=1.4)
    ax_rsi.axhline(30, color=AXIS, linewidth=1)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI(14)")
    ax.yaxis.set_major_formatter(FuncFormatter(_price_fmt))
    ax.set_ylim(min(low.min(), trade.initial_stop) - span * 0.06,
                max(h.max(), trade.target or h.max()) + span * 0.08)
    ax.set_xlim(lo - 0.8, hi - 0.2)
    _time_ticks(ax_rsi, times, 8, "%m-%d\n%H:%M", offset=lo)

    reason = _REASON_KO.get(trade.exit_reason or "", trade.exit_reason or "-")
    fig.suptitle(title, x=0.06, ha="left", fontsize=13, color=INK, fontweight="bold")
    ax.set_title(
        f"진입 {trade.entry_price:,.2f} → 청산 {trade.exit_price or float('nan'):,.2f} ({reason}) · "
        f"{trade.r_multiple:+.2f}R · 보유 {trade.bars_held}봉 · 수량 {trade.qty:g}",
        loc="left", fontsize=9.5, color=INK_2, pad=24)
    handles = [
        Line2D([], [], color=DIVERGENCE, marker="o", linewidth=2, label="t1 → t3"),
        Line2D([], [], color=INK, marker="v", markersize=5, linestyle="none", label="p2"),
        Line2D([], [], color=INK, marker="^", linestyle="none", label="진입 (신호 다음 봉 시가)"),
        Line2D([], [], color=INK, marker="v", linestyle="none", label="청산"),
        Line2D([], [], color=DOWN, linewidth=1.2, label="손절 (점선: 트레일링)"),
        Line2D([], [], color=UP, linewidth=1.2, label="목표가"),
        Line2D([], [], color=MUTED, linewidth=6, alpha=0.3, label="보유 구간"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=7, frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.1)
    return _save(fig, path)
