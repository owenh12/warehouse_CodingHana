"""신호 육안 검증 보고서 (3단계): 차트 PNG + 후보 CSV + 요약 Markdown.

신호는 백테스트와 똑같이 데이터 시작(backtest_period.start)부터 탐지기를 돌려 얻고, 표시 기간
(기본: 마지막 3개월)에 신호 시각이 들어오는 것만 보여 준다. 그래서 차트의 신호는 백테스트가 보게 될
신호와 같다(워밍업 차이 없음).
"""

from __future__ import annotations

import datetime as dt
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rsidiv.core.config import AssetClassName, StrategyParams
from rsidiv.indicators.pivots import detect_pivots
from rsidiv.indicators.rsi import rsi_wilder
from rsidiv.reports.signal_charts import describe, plot_candidate, plot_overview
from rsidiv.signals.divergence import (
    PATTERN_REASONS,
    REASON_LABELS,
    REASONS,
    DivergenceCandidate,
    candidates_frame,
    detect_divergences,
)


def pattern_ok(c: DivergenceCandidate) -> bool:
    """정의(가격 LL, RSI HL, 구조)는 만족하는지. 필터 탈락만 있으면 참."""
    return not PATTERN_REASONS.intersection(c.failed)


def first_reason_counts(candidates: Sequence[DivergenceCandidate]) -> Counter[str]:
    """각 후보를 첫 번째 실패 사유(REASONS 순서)로 집계. 신호는 'accepted'."""
    return Counter(next((r for r in REASONS if r in c.failed), "accepted") for c in candidates)


@dataclass(slots=True)
class SignalReport:
    out_dir: Path
    summary_path: Path
    signals: list[DivergenceCandidate]
    near_misses: list[DivergenceCandidate]
    window_candidates: list[DivergenceCandidate]
    all_candidates: list[DivergenceCandidate]
    charts: list[Path]


def params_lines(p: StrategyParams) -> list[str]:
    f = p.filters
    on = []
    if f.gap_bars.enabled:
        on.append(f"간격 {f.gap_bars.min_bars}~{f.gap_bars.max_bars}봉")
    if f.rsi_t1_oversold.enabled:
        on.append(f"RSI(t1) < {f.rsi_t1_oversold.threshold:g}")
    if f.rsi_diff_min.enabled:
        on.append(f"RSI(t3)-RSI(t1) ≥ {f.rsi_diff_min.min_diff:g}")
    if f.price_drop_min.enabled:
        on.append(f"가격 하락폭 ≥ {f.price_drop_min.min_pct:.2%}")
    if f.no_lower_low_between.enabled:
        on.append("t1~t3 사이 더 낮은 저가 없음")
    if f.trend.enabled:
        on.append(f"추세 {f.trend.mode} {f.trend.ma_type}{f.trend.ma_period} {f.trend.condition}")
    if f.volume.enabled:
        on.append(f"거래량 {f.volume.mode} ≥ {f.volume.min_ratio:g}배")
    return [
        f"- 피벗 L={p.pivot.left}, R={p.pivot.right}, 가격 기준 {p.pivot.price_source}, "
        f"strict={str(p.pivot.strict).lower()}; RSI({p.rsi.period}, Wilder)",
        f"- t1 선택 {p.divergence.t1_selection}, p2 선택 {p.divergence.p2_selection}",
        "- 켜진 필터: " + (", ".join(on) if on else "없음"),
    ]


def _funnel_table(window: Sequence[DivergenceCandidate], full: Sequence[DivergenceCandidate]) -> list[str]:
    w, a = first_reason_counts(window), first_reason_counts(full)
    rows = ["| 첫 번째 탈락 사유 | 표시 기간 | 전체 기간 |", "|---|---:|---:|"]
    for reason in (*REASONS, "accepted"):
        if w[reason] or a[reason]:
            label = "✅ 신호" if reason == "accepted" else REASON_LABELS[reason]
            rows.append(f"| {label} | {w[reason]:,} | {a[reason]:,} |")
    rows.append(f"| **합계 (확정된 피벗 저점 = t3 후보)** | **{len(window):,}** | **{len(full):,}** |")
    return rows


def _candidate_rows(
    frame: pd.DataFrame, cands: Sequence[DivergenceCandidate], tz: str, charts: dict[int, str]
) -> list[str]:
    rows = ["| # | 신호 시각 (KST) | t1 | p2 | t3 | 간격 | RSI t1 → t3 | 저가 t1 → t3 | 탈락 사유 | 차트 |",
            "|---|---|---|---|---|---:|---|---|---|---|"]

    def at(i: int | None) -> str:
        return frame.index[i].tz_convert(tz).strftime("%m-%d %H:%M") if i is not None else "-"

    for n, c in enumerate(cands, 1):
        gap = c.t3 - c.t1 if c.t1 is not None else "-"
        reasons = ", ".join(REASON_LABELS[r] for r in c.failed) or "-"
        drop = f"{(c.price_t3 / c.price_t1 - 1) * 100:+.2f}%" if c.t1 is not None else "-"
        rows.append(
            f"| {n} | {c.signal_time.tz_convert(tz):%Y-%m-%d %H:%M} | {at(c.t1)} | {at(c.p2)} | {at(c.t3)} | "
            f"{gap} | {c.rsi_t1:.1f} → {c.rsi_t3:.1f} | {c.price_t1:,.2f} → {c.price_t3:,.2f} ({drop}) | "
            f"{reasons} | [{charts[c.t3]}]({charts[c.t3]}) |"
        )
    return rows


def build_signal_report(
    frame: pd.DataFrame,
    params: StrategyParams,
    *,
    symbol: str,
    market: str,
    timeframe: str,
    asset_class: AssetClassName,
    window_start: dt.datetime,
    window_end: dt.datetime,
    display_tz: str,
    out_dir: Path,
    max_near_miss_charts: int = 40,
) -> SignalReport:
    """``frame`` 전체로 신호를 탐지하고, 신호 시각이 [window_start, window_end) 인 후보를 보고한다."""
    candidates = detect_divergences(frame, params, timeframe=timeframe, asset_class=asset_class)
    rsi = rsi_wilder(frame["close"].to_numpy(), params.rsi.period)
    pivots = detect_pivots(frame, params.pivot)
    ws, we = pd.Timestamp(window_start), pd.Timestamp(window_end)
    window = [c for c in candidates if ws <= c.signal_time < we]
    signals = [c for c in window if c.accepted]
    near = [c for c in window if not c.accepted and pattern_ok(c)]

    out_dir.mkdir(parents=True, exist_ok=True)
    charts: list[Path] = []
    names: dict[int, str] = {}
    label = f"{symbol} {market} {timeframe}"
    for kind, items in (("signal", signals), ("near", near[:max_near_miss_charts])):
        for n, c in enumerate(items, 1):
            name = f"{kind}_{n:02d}.png"
            title = (f"신호 #{n} · {label}" if kind == "signal"
                     else f"필터 탈락 #{n} · {label}")
            charts.append(plot_candidate(frame, rsi, c, pivots, title=title, display_tz=display_tz,
                                         right=params.pivot.right, path=out_dir / name))
            names[c.t3] = name

    in_window = (frame.index >= ws) & (frame.index < we)
    offset = int(np.flatnonzero(in_window)[0]) if in_window.any() else 0
    overview = plot_overview(
        frame[in_window], rsi[in_window], signals, near, offset=offset,
        title=f"{label} · {ws.tz_convert(display_tz):%Y-%m-%d} ~ {we.tz_convert(display_tz):%Y-%m-%d} (KST)",
        display_tz=display_tz, path=out_dir / "overview.png",
    )
    charts.insert(0, overview)

    table = candidates_frame(frame, window)
    table.insert(1, "signal_time_kst", table["signal_time"].dt.tz_convert(display_tz))
    table.to_csv(out_dir / "candidates.csv", index=False)

    lines = [
        f"# 신호 육안 검증: {label}",
        "",
        f"- 표시 기간 (신호 시각 기준): {ws.tz_convert(display_tz):%Y-%m-%d %H:%M} ~ "
        f"{we.tz_convert(display_tz):%Y-%m-%d %H:%M} KST",
        f"- 탐지 구간: {frame.index[0].tz_convert(display_tz):%Y-%m-%d} ~ "
        f"{(frame.index[-1] + pd.Timedelta(minutes=15)).tz_convert(display_tz):%Y-%m-%d %H:%M} KST "
        f"({len(frame):,}봉). 백테스트와 같은 시작점에서 탐지해 워밍업 차이가 없다.",
        *params_lines(params),
        "",
        "![overview](overview.png)",
        "",
        "## 판정 퍼널",
        "",
        "확정된 피벗 저점마다 한 번 판정한다. 첫 번째로 걸린 사유로 집계한다(사유 순서는 정의 → 필터).",
        "",
        *_funnel_table(window, candidates),
        "",
        f"## 신호 ({len(signals)}건)",
        "",
        *(_candidate_rows(frame, signals, display_tz, names) if signals else ["없음"]),
        "",
        f"## 패턴은 맞고 필터에서 탈락한 후보 ({len(near)}건)",
        "",
        "가격 LL·RSI HL·p2 구조는 만족하지만 선택 필터(간격, RSI 과매도, 상승폭, 하락폭, 사이 저가 등)에 걸린 후보.",
        "필터가 의도대로 동작하는지 확인하는 용도다.",
        "",
        *(_candidate_rows(frame, near[:max_near_miss_charts], display_tz, names) if near else ["없음"]),
        "",
        "## 차트 읽는 법",
        "",
        "- 초록 선·원: t1 → t3 (위: 가격 저점, 아래: RSI). 검은 역삼각형: p2.",
        "- 회색 빈 원: 신호 시각까지 확정된 피벗 (저점은 아래, 고점은 위).",
        f"- 회색 띠: t3 다음 {params.pivot.right}봉 (피벗 확인 구간). 띠가 끝나는 세로선이 신호 시각이며, 진입(모드 A)은 그다음 봉 시가다.",
        "- 상승봉 빨강, 하락봉 파랑. 시각은 KST.",
        "",
        "파일: `candidates.csv` (표시 기간 전체 후보, 시각은 UTC 봉 시작 시각, `signal_time` 은 종가 확정 시각).",
    ]
    summary = out_dir / "summary.md"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return SignalReport(out_dir, summary, signals, near, window, candidates, charts)


__all__ = ["SignalReport", "build_signal_report", "describe", "first_reason_counts", "pattern_ok"]
