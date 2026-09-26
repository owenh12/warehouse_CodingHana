"""신호 시각 검증 차트 (matplotlib, 화면 표시 시각 = 설정의 표시 시간대).

- 개요: 기간 전체 가격 + 신호 위치, RSI 패널 (30/70 선)
- 개별 신호: t1(p1)·p2(t2)·t3(p3), Low(t1)(High(p1)) 선, 이탈 기준선 min(Low[p2..t3−1]), p2 확정 봉, 창 안의 모든 피벗,
  가격·RSI 다이버전스 연결선, 참고용 진입(다음 봉 시가)·손절(ATR×mult)·익절(R 배수) 선

차트 글꼴에 한글이 없을 수 있어 라벨은 영어로 쓴다.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from perpdiv.core.config import StrategyCfg
from perpdiv.indicators.pivots import Pivot
from perpdiv.signals.divergence import Signal

UP, DOWN = "#1a9850", "#d73027"
NEUTRAL = "#4d4d4d"


def _labels(ax: Axes, times: pd.DatetimeIndex, positions: np.ndarray, tz: ZoneInfo, fmt: str, count: int = 8) -> None:
    ticks = np.unique(np.linspace(positions[0], positions[-1], min(count, len(positions))).round().astype(int))
    ax.set_xticks(ticks)
    ax.set_xticklabels([times[i - positions[0]].tz_convert(tz).strftime(fmt) for i in ticks], fontsize=8)


def _candles(ax: Axes, frame: pd.DataFrame, x: np.ndarray) -> None:
    o, h, lo, c = (frame[col].to_numpy(float) for col in ("open", "high", "low", "close"))
    colors = [UP if up else DOWN for up in c >= o]
    ax.vlines(x, lo, h, colors=colors, linewidth=0.8)
    ax.bar(x, np.maximum(np.abs(c - o), (h - lo) * 0.02 + 1e-12), bottom=np.minimum(o, c), width=0.6,
           color=colors, edgecolor=colors, linewidth=0.5)


def plot_overview(bars: pd.DataFrame, rsi: np.ndarray, signals: Sequence[Signal], *, symbol: str, timeframe: str,
                  cfg: StrategyCfg, tz: ZoneInfo, path: Path, start: pd.Timestamp) -> None:
    """기간 전체 (start 이후만 표시, 지표는 워밍업 포함 전체로 계산된 값)."""
    shown = np.asarray(bars.index >= start)
    offset = int(np.argmax(shown))
    frame = bars[shown]
    x = np.arange(offset, offset + len(frame))
    fig, (ax, ax_rsi) = plt.subplots(2, 1, figsize=(16, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    if len(frame) <= 400:
        _candles(ax, frame, x)
    else:
        ax.plot(x, frame["close"].to_numpy(float), color=NEUTRAL, linewidth=0.6)
    for s in signals:
        long = s.side == "long"
        y = s.trigger_low if long else s.trigger_high
        ax.scatter([s.trigger_index], [y], marker="^" if long else "v", color=UP if long else DOWN, s=40, zorder=5,
                   edgecolors="black", linewidths=0.4)
        ax_rsi.scatter([s.trigger_index], [s.trigger_rsi], marker="^" if long else "v", color=UP if long else DOWN,
                       s=18, zorder=5)
    ax_rsi.plot(x, rsi[shown], color="#6a3d9a", linewidth=0.6)
    for level in (cfg.bullish.oversold, cfg.bearish.overbought):
        ax_rsi.axhline(level, color="grey", linestyle="--", linewidth=0.7)
    ax_rsi.set_ylim(0, 100)
    n_long = sum(s.side == "long" for s in signals)
    ax.set_title(f"{symbol} {timeframe}  {frame.index[0].tz_convert(tz):%Y-%m-%d} ~ "
                 f"{frame.index[-1].tz_convert(tz):%Y-%m-%d} ({tz.key})   long {n_long} / short {len(signals) - n_long}",
                 fontsize=11)
    ax.set_ylabel("price")
    ax_rsi.set_ylabel(f"RSI({cfg.rsi.period})")
    _labels(ax_rsi, pd.DatetimeIndex(frame.index), x, tz, "%y-%m-%d", 12)
    ax.grid(alpha=0.2)
    ax_rsi.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_signal(bars: pd.DataFrame, rsi: np.ndarray, pivots: Sequence[Pivot], signal: Signal, *, number: int,
                cfg: StrategyCfg, tz: ZoneInfo, path: Path, before: int = 20, after: int = 20) -> None:
    """``pivots`` 는 전체 봉 기준 피벗 목록 (창 가장자리에서 피벗이 사라지지 않도록)."""
    long = signal.side == "long"
    a, s, t = signal.anchor_index, signal.swing_index, signal.trigger_index
    lo_i, hi_i = max(0, a - before), min(len(bars), t + after + 1)
    frame = bars.iloc[lo_i:hi_i]
    x = np.arange(lo_i, hi_i)
    fig, (ax, ax_rsi) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    _candles(ax, frame, x)

    # 창 안의 피벗 (t3 마감까지 확정된 것만 진하게)
    for p in pivots:
        if not lo_i <= p.index < hi_i:
            continue
        idx = p.index
        known = p.confirm_index <= t
        ax.scatter([idx], [p.price], marker="o", s=14, facecolors="none",
                   edgecolors=("#08519c" if p.kind == "high" else "#a50f15") if known else "#bdbdbd", linewidths=0.8,
                   zorder=4)

    names = ("t1", "p2", "t3") if long else ("p1", "t2", "p3")
    color = UP if long else DOWN
    ax.scatter([a], [signal.anchor_price], marker="o", s=70, color=color, zorder=6, edgecolors="black")
    ax.scatter([s], [signal.swing_price], marker="o", s=70, color="#ffd92f", zorder=6, edgecolors="black")
    ax.scatter([t], [signal.trigger_close], marker="*", s=160, color=color, zorder=7, edgecolors="black")
    span = float(frame["high"].max() - frame["low"].min())
    pad = span * 0.04
    ax.annotate(names[0], (a, signal.anchor_price), (0, -14 if long else 8), textcoords="offset points",
                ha="center", fontsize=9, fontweight="bold")
    ax.annotate(names[1], (s, signal.swing_price), (0, 8 if long else -14), textcoords="offset points",
                ha="center", fontsize=9, fontweight="bold")
    ax.annotate(names[2], (t, signal.trigger_close), (8, 0), textcoords="offset points", fontsize=9,
                fontweight="bold")
    ax.hlines(signal.anchor_price, a, t, colors=color, linestyles="--", linewidth=0.9,
              label=f"{'Low(t1)' if long else 'High(p1)'} {signal.anchor_price:,.2f}")
    ax.hlines(signal.breakout_level, s, t, colors="#ff7f00", linestyles=":", linewidth=1.4,
              label=f"{'min Low[p2..t3-1]' if long else 'max High[t2..p3-1]'} {signal.breakout_level:,.2f}")
    ax.plot([a, t], [signal.anchor_price, signal.trigger_close], color=color, linewidth=1.0, alpha=0.7)
    confirm = s + cfg.pivot.right
    ax.axvline(confirm, color="#ffd92f", linewidth=0.8, alpha=0.8, linestyle="-.")
    ax.text(confirm, ax.get_ylim()[1], f"{names[1]} confirmed", fontsize=7, va="top", ha="left", color="#b8860b")

    # 참고: 진입(다음 봉 시가)·손절·익절 (4단계 규칙, 시각화용)
    if t + 1 < len(bars):
        entry = float(bars["open"].iloc[t + 1])
        stop = (signal.trigger_low - cfg.exit.stop.atr_mult * signal.trigger_atr if long
                else signal.trigger_high + cfg.exit.stop.atr_mult * signal.trigger_atr)
        target = entry + cfg.exit.take_profit.r_multiple * (entry - stop)
        end_x = min(hi_i - 1, t + after)
        ax.scatter([t + 1], [entry], marker=">", s=50, color="black", zorder=7)
        ax.hlines(stop, t + 1, end_x, colors="#7f0000", linewidth=0.8, alpha=0.6,
                  label=f"stop {stop:,.2f} (ATR {signal.trigger_atr:,.2f} x{cfg.exit.stop.atr_mult:g})")
        ax.hlines(target, t + 1, end_x, colors="#00441b", linewidth=0.8, alpha=0.6,
                  label=f"TP {cfg.exit.take_profit.r_multiple:g}R {target:,.2f} (entry {entry:,.2f})")
    ax.set_ylim(min(ax.get_ylim()[0], float(frame["low"].min()) - pad), ax.get_ylim()[1])
    ax.legend(loc="best", fontsize=7, framealpha=0.85)
    ax.grid(alpha=0.2)

    ax_rsi.plot(x, rsi[lo_i:hi_i], color="#6a3d9a", linewidth=1.0)
    threshold = cfg.bullish.oversold if long else cfg.bearish.overbought
    ax_rsi.axhline(threshold, color="grey", linestyle="--", linewidth=0.8)
    ax_rsi.plot([a, t], [signal.anchor_rsi, signal.trigger_rsi], color=color, linewidth=1.0)
    ax_rsi.scatter([a, t], [signal.anchor_rsi, signal.trigger_rsi], color=color, s=25, zorder=5)
    ax_rsi.annotate(f"{signal.anchor_rsi:.1f}", (a, signal.anchor_rsi), (4, -10 if long else 4),
                    textcoords="offset points", fontsize=8)
    ax_rsi.annotate(f"{signal.trigger_rsi:.1f}", (t, signal.trigger_rsi), (4, 4 if long else -10),
                    textcoords="offset points", fontsize=8)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel(f"RSI({cfg.rsi.period})")
    ax_rsi.grid(alpha=0.2)
    _labels(ax_rsi, pd.DatetimeIndex(frame.index), x, tz, "%m-%d %H:%M" if signal.timeframe != "1d" else "%y-%m-%d")

    sig_time = signal.signal_time.tz_convert(tz)
    ax.set_title(
        f"{signal.symbol} {signal.timeframe} {signal.side.upper()} #{number}  signal {sig_time:%Y-%m-%d %H:%M} {tz.key}"
        f"  |  {names[0]}->{names[2]} {signal.gap_bars} bars  |  RSI {signal.anchor_rsi:.1f} -> {signal.trigger_rsi:.1f}"
        + (f"  |  concurrent {signal.concurrent}" if signal.concurrent > 1 else ""), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
