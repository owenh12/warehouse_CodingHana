"""신호 육안 검증 차트 (matplotlib, PNG).

- :func:`plot_overview`: 기간 전체 종가 + RSI, 신호(채운 원)·필터 탈락 후보(빈 원) 위치.
- :func:`plot_candidate`: 후보 하나의 확대 캔들 차트. t1·p2·t3, 확정된 피벗, R봉 확인 구간,
  신호 시각(= t3+R 봉 종가 확정 = 다음 봉 시가 진입)을 표시한다.

색: 상승봉 빨강·하락봉 파랑(국내 관례), 다이버전스 선·표식 초록. 초록과 빨강은 색각이상에서
구분이 약하므로 표식 모양과 글자(t1·t3)를 함께 쓴다. 시각은 표시 시간대(KST)로 그린다.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from rsidiv.indicators.pivots import Pivot
from rsidiv.signals.divergence import REASON_LABELS, DivergenceCandidate

# 한글 글꼴 후보 중 없는 것을 건너뛸 때 나오는 경고는 끈다
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
UP = "#e34948"
DOWN = "#2a78d6"
DIVERGENCE = "#008300"
CONFIRM_BAND = "#f0efec"

FONTS = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR",
         "WenQuanYi Zen Hei", "DejaVu Sans"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": FONTS,
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 130,
    "savefig.facecolor": SURFACE,
})


def _price_fmt(value: float, _pos: int | None = None) -> str:
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:,.2f}"


def _time_ticks(ax: Axes, times: pd.DatetimeIndex, n: int, fmt: str, offset: int = 0) -> None:
    """x 좌표가 봉 위치(offset + i)인 축에 표시 시간대 시각 눈금을 단다."""
    step = max(1, len(times) // n)
    ticks = list(range(0, len(times), step))
    ax.set_xticks([offset + i for i in ticks], [times[i].strftime(fmt) for i in ticks])


def _marker(ax: Axes, x: float, y: float, marker: str, color: str, size: float = 9, filled: bool = True) -> None:
    ax.plot([x], [y], marker=marker, markersize=size, color=color,
            markerfacecolor=color if filled else SURFACE, markeredgecolor=color if not filled else SURFACE,
            markeredgewidth=1.5 if not filled else 2, linestyle="none", zorder=5)


def describe(c: DivergenceCandidate) -> str:
    parts = []
    if c.t1 is not None:
        parts.append(f"간격 {c.t3 - c.t1}봉")
        parts.append(f"RSI {c.rsi_t1:.1f} → {c.rsi_t3:.1f} ({c.rsi_t3 - c.rsi_t1:+.1f})")
        parts.append(f"저가 {_price_fmt(c.price_t1)} → {_price_fmt(c.price_t3)} "
                     f"({(c.price_t3 / c.price_t1 - 1) * 100:+.2f}%)")
    if c.failed:
        parts.append("탈락: " + ", ".join(REASON_LABELS[r] for r in c.failed))
    return " · ".join(parts)


def plot_candidate(
    frame: pd.DataFrame,
    rsi: np.ndarray,
    candidate: DivergenceCandidate,
    pivots: Sequence[Pivot],
    *,
    title: str,
    display_tz: str,
    right: int,
    rsi_levels: Sequence[float] = (30.0,),
    path: Path,
) -> Path:
    """후보 하나의 확대 차트를 저장한다. 표시 구간은 t1 앞 12봉 ~ 신호 봉 뒤 12봉."""
    c = candidate
    first = (c.t1 if c.t1 is not None else c.t3) - 12
    lo, hi = max(0, first), min(len(frame), c.signal_index + 13)
    view = frame.iloc[lo:hi]
    x = np.arange(lo, hi)
    times = pd.DatetimeIndex(view.index).tz_convert(display_tz)

    fig, (ax, ax_rsi) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.08})
    # R봉 확인 구간 (t3 다음 봉 ~ 신호 봉)
    for a in (ax, ax_rsi):
        a.axvspan(c.t3 + 0.5, c.signal_index + 0.5, color=CONFIRM_BAND, zorder=0, linewidth=0)

    # 캔들
    o, h, low, cl = (view[k].to_numpy() for k in ("open", "high", "low", "close"))
    up = cl >= o
    colors = np.where(up, UP, DOWN)
    ax.vlines(x, low, h, colors=colors, linewidth=1, zorder=2)
    body_lo = np.minimum(o, cl)
    body_h = np.maximum(np.abs(cl - o), (h.max() - low.min()) * 0.002)
    ax.bar(x, body_h, bottom=body_lo, width=0.62, color=colors, linewidth=0, zorder=3)

    # 확정된 피벗 (신호 시각까지 확정된 것만)
    span = h.max() - low.min()
    marked = {c.t1, c.p2, c.t3}  # 이미 t1·p2·t3 로 표시한 봉은 건너뛴다
    for p in pivots:
        if lo <= p.index < hi and p.confirm_index <= c.signal_index and p.index not in marked:
            if p.kind == "low":
                _marker(ax, p.index, p.price - span * 0.02, "o", MUTED, size=5, filled=False)
            else:
                _marker(ax, p.index, p.price + span * 0.02, "o", MUTED, size=5, filled=False)

    # t1·p2·t3
    if c.t1 is not None:
        ax.plot([c.t1, c.t3], [c.price_t1, c.price_t3], color=DIVERGENCE, linewidth=2, zorder=4,
                solid_capstyle="round")
        ax_rsi.plot([c.t1, c.t3], [c.rsi_t1, c.rsi_t3], color=DIVERGENCE, linewidth=2, zorder=4,
                    solid_capstyle="round")
        for idx, price, value, name in ((c.t1, c.price_t1, c.rsi_t1, "t1"), (c.t3, c.price_t3, c.rsi_t3, "t3")):
            _marker(ax, idx, price, "o", DIVERGENCE)
            _marker(ax_rsi, idx, value, "o", DIVERGENCE)
            ax.annotate(name, (idx, price), xytext=(0, -16), textcoords="offset points", ha="center",
                        color=INK, fontsize=10, fontweight="bold")
    else:
        _marker(ax, c.t3, c.price_t3, "o", DIVERGENCE)
    if c.p2 is not None:
        _marker(ax, c.p2, c.price_p2, "v", INK, size=8)
        ax.annotate("p2", (c.p2, c.price_p2), xytext=(0, 9), textcoords="offset points", ha="center",
                    color=INK, fontsize=10, fontweight="bold")

    # 신호 시각 = 신호 봉 종가 확정 = 다음 봉 시가
    for a in (ax, ax_rsi):
        a.axvline(c.signal_index + 0.5, color=INK, linewidth=1, zorder=4)
    ax.annotate(f"신호 {c.signal_time.tz_convert(display_tz):%m-%d %H:%M}\n→ 다음 봉 시가 진입",
                (c.signal_index + 0.5, 1.0), xycoords=("data", "axes fraction"), xytext=(4, -4),
                textcoords="offset points", va="top", fontsize=9, color=INK)

    # RSI
    ax_rsi.plot(x, rsi[lo:hi], color=INK_2, linewidth=1.5, zorder=3)
    for level in rsi_levels:
        ax_rsi.axhline(level, color=AXIS, linewidth=1, zorder=1)
        ax_rsi.annotate(f"{level:g}", (1.0, level), xycoords=("axes fraction", "data"), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=8, color=MUTED)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI(14)")

    ax.yaxis.set_major_formatter(FuncFormatter(_price_fmt))
    ax.set_ylim(low.min() - span * 0.08, h.max() + span * 0.10)
    ax.set_xlim(lo - 0.8, hi - 0.2)
    _time_ticks(ax_rsi, times, 8, "%m-%d\n%H:%M", offset=lo)

    fig.suptitle(title, x=0.06, ha="left", fontsize=13, color=INK, fontweight="bold")
    ax.set_title(describe(c), loc="left", fontsize=9.5, color=INK_2, pad=26)
    handles = [
        Patch(color=UP, label="상승봉"), Patch(color=DOWN, label="하락봉"),
        Line2D([], [], color=DIVERGENCE, linewidth=2, marker="o", label="t1 → t3"),
        Line2D([], [], color=INK, marker="v", linestyle="none", label="p2"),
        Line2D([], [], color=MUTED, marker="o", markerfacecolor=SURFACE, linestyle="none",
               label="확정된 피벗"),
        Patch(color=CONFIRM_BAND, label=f"피벗 확인 {right}봉"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=6, frameon=False,
              fontsize=8.5, handlelength=1.6, columnspacing=1.2, borderaxespad=0.2)
    fig.subplots_adjust(left=0.07, right=0.95, top=0.86, bottom=0.1)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_overview(
    frame: pd.DataFrame,
    rsi: np.ndarray,
    signals: Sequence[DivergenceCandidate],
    near_misses: Sequence[DivergenceCandidate],
    *,
    offset: int,
    title: str,
    display_tz: str,
    path: Path,
) -> Path:
    """표시 기간 전체: 종가 선 + RSI, 신호 번호와 필터 탈락 후보 위치.

    ``frame``·``rsi`` 는 표시 구간만, 후보의 봉 위치는 원본 기준이다 (``offset`` = 표시 구간 시작 위치).
    """
    x = np.arange(len(frame))
    times = pd.DatetimeIndex(frame.index).tz_convert(display_tz)
    close = frame["close"].to_numpy()

    fig, (ax, ax_rsi) = plt.subplots(2, 1, figsize=(13, 6.4), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.08})
    ax.plot(x, close, color=INK_2, linewidth=0.8, zorder=2)
    ax_rsi.plot(x, rsi, color=INK_2, linewidth=0.5, zorder=2)
    ax_rsi.axhline(30, color=AXIS, linewidth=1)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI(14)")

    for c in near_misses:
        _marker(ax, c.t3 - offset, c.price_t3, "o", MUTED, size=7, filled=False)
    for n, c in enumerate(signals, 1):
        _marker(ax, c.t3 - offset, c.price_t3, "o", DIVERGENCE, size=9)
        _marker(ax_rsi, c.t3 - offset, c.rsi_t3, "o", DIVERGENCE, size=7)
        ax.annotate(f"#{n}", (c.t3 - offset, c.price_t3), xytext=(0, -15), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold", color=INK)

    ax.yaxis.set_major_formatter(FuncFormatter(_price_fmt))
    _time_ticks(ax_rsi, times, 10, "%m-%d")
    fig.suptitle(title, x=0.06, ha="left", fontsize=13, color=INK, fontweight="bold")
    handles = [
        Line2D([], [], color=INK_2, linewidth=1, label="종가 (15분봉)"),
        Line2D([], [], color=DIVERGENCE, marker="o", linestyle="none", label=f"신호 {len(signals)}건 (t3 위치)"),
        Line2D([], [], color=MUTED, marker="o", markerfacecolor=SURFACE, linestyle="none",
               label=f"패턴은 맞고 필터에서 탈락 {len(near_misses)}건"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.1)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
