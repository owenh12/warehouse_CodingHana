"""3단계 시각 검증 실행: 한 심볼의 최근 N개월 신호를 타임프레임별로 찾아 CSV·차트·요약을 만든다.

- 구간: 백테스트 끝(``data.backtest_period.end`` 규칙, 기본 지난달 말)에서 N개월 전까지. 지표는 ``warmup_days`` 앞부터 계산.
- 표시 구간 밖(워밍업)에서 난 신호는 버린다. 판정 집계도 표시 구간만 센다.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from perpdiv.core.config import Settings, resolve_project_path
from perpdiv.data.store import MarketData
from perpdiv.indicators.pivots import detect_pivots
from perpdiv.indicators.rsi import rsi_wilder
from perpdiv.reports.signal_charts import plot_overview, plot_signal
from perpdiv.signals.divergence import DivergenceDetector, Side, Signal, signals_frame

FUNNEL = ("anchor_pivots", "structures", "swing_set", "swing_replaced", "breakouts", "fail_swing_not_highest",
          "discarded_anchor_broken", "fail_price", "fail_rsi", "fail_gap_min", "passed", "signals", "expired")


@dataclass(slots=True)
class TimeframeResult:
    timeframe: str
    bars: int  # 표시 구간 봉 수
    missing_source_bars: int
    halted_source_bars: int
    signals: list[Signal]
    stats: dict[Side, Counter[str]]  # 방향 → 판정 집계 (표시 구간)


def _run(frame: pd.DataFrame, settings: Settings, symbol: str, timeframe: str, start: pd.Timestamp
         ) -> tuple[list[Signal], dict[Side, Counter[str]]]:
    detector = DivergenceDetector(settings.strategy, symbol=symbol, timeframe=timeframe)
    signals: list[Signal] = []
    before: dict[Side, Counter[str]] | None = None
    cols = [frame[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close")]
    for time, o, h, lo, c in zip(pd.DatetimeIndex(frame.index), *cols, strict=True):
        if before is None and time >= start:
            before = {k: Counter(v) for k, v in detector.stats().items()}
        found = detector.update(time, float(o), float(h), float(lo), float(c))
        if time >= start:
            signals += found
    after = detector.stats()
    stats = {side: after[side] - (before or {}).get(side, Counter()) for side in after}
    return signals, stats


def run_signal_check(settings: Settings, market: MarketData, *, symbol: str, months: int, timeframes: list[str],
                     now: dt.datetime, out_dir: Path | None = None, charts: bool = True
                     ) -> tuple[list[TimeframeResult], Path]:
    _, end = settings.data.backtest_period.bounds(now)
    start = pd.Timestamp(end) - pd.DateOffset(months=months)
    load_from = (start - pd.Timedelta(days=settings.data.warmup_days)).to_pydatetime()
    tz = ZoneInfo(settings.base.project.display_timezone)
    out = out_dir or resolve_project_path(settings.base.paths.report_dir) / "signals" / symbol
    out.mkdir(parents=True, exist_ok=True)
    cfg = settings.strategy
    results: list[TimeframeResult] = []
    rows: list[pd.DataFrame] = []
    for tf in timeframes:
        loaded = market.bars(symbol, tf, load_from, end)
        bars = loaded.bars
        signals, stats = _run(bars, settings, symbol, tf, start)
        shown = int((bars.index >= start).sum())
        results.append(TimeframeResult(tf, shown, loaded.quality.missing_bars, loaded.halted_source_bars, signals, stats))
        frame = signals_frame(signals)
        if not frame.empty:
            frame.insert(0, "number", range(1, len(frame) + 1))
            frame["anchor_is_extreme"] = [anchor_is_extreme(bars, sig) for sig in signals]
            for col in ("anchor_time", "swing_time", "trigger_time", "signal_time"):
                frame[col + "_kst"] = pd.DatetimeIndex(frame[col]).tz_convert(tz).strftime("%Y-%m-%d %H:%M")
            frame["chart"] = [f"{tf}/{n:03d}_{s.side}.png" for n, s in zip(frame["number"], signals, strict=True)]
            rows.append(frame)
        if charts:
            rsi = rsi_wilder(bars["close"].to_numpy(float), cfg.rsi.period)
            plot_overview(bars, rsi, signals, symbol=symbol, timeframe=tf, cfg=cfg, tz=tz,
                          path=out / f"overview_{tf}.png", start=start)
            pivots = detect_pivots(bars, cfg.pivot.left, cfg.pivot.right, cfg.pivot.tie_rule)
            (out / tf).mkdir(exist_ok=True)
            for old in (out / tf).glob("*.png"):
                old.unlink()
            for number, signal in enumerate(signals, 1):
                plot_signal(bars, rsi, pivots, signal, number=number, cfg=cfg, tz=tz,
                            path=out / tf / f"{number:03d}_{signal.side}.png")
    table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    table.to_csv(out / f"signals_{symbol}.csv", index=False)
    (out / "summary.md").write_text(render_summary(results, symbol=symbol, start=start, end=pd.Timestamp(end),
                                                   tz=tz, settings=settings), encoding="utf-8")
    return results, out


def anchor_is_extreme(bars: pd.DataFrame, signal: Signal) -> bool:
    """t1 = min Low[t1..p2] (약세 p1 = max High[p1..t2]) 인가 — ``discard_on_anchor_break`` 규칙과 같은 기준."""
    a, s = signal.anchor_index, signal.swing_index
    if signal.side == "long":
        return bool(bars["low"].iloc[a:s + 1].min() >= signal.anchor_price)
    return bool(bars["high"].iloc[a:s + 1].max() <= signal.anchor_price)


def render_summary(results: list[TimeframeResult], *, symbol: str, start: pd.Timestamp, end: pd.Timestamp,
                   tz: ZoneInfo, settings: Settings) -> str:
    cfg = settings.strategy
    lines = [
        f"# {symbol} 신호 요약",
        "",
        f"- 구간: {start.tz_convert(tz):%Y-%m-%d %H:%M} ~ {end.tz_convert(tz):%Y-%m-%d %H:%M} {tz.key} "
        f"(UTC {start:%Y-%m-%d} ~ {end - pd.Timedelta(seconds=1):%Y-%m-%d})",
        f"- 피벗 L={cfg.pivot.left} R={cfg.pivot.right} ({cfg.pivot.tie_rule}), RSI {cfg.rsi.period}, "
        f"강세 <{cfg.bullish.oversold:g}, 약세 >{cfg.bearish.overbought:g}, 간격 ≤ "
        f"{cfg.structure.gap_bars.max if cfg.structure.gap_bars.max_enabled else '-'}봉",
        "",
        "| TF | 봉 | 롱 | 숏 | 동시 성립>1 | 간격 중앙값 | 간격 최대 | 원천 결측(5m) | 거래중단 제외(5m) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        gaps = [s.gap_bars for s in r.signals]
        lines.append(
            f"| {r.timeframe} | {r.bars:,} | {sum(s.side == 'long' for s in r.signals)} | "
            f"{sum(s.side == 'short' for s in r.signals)} | {sum(s.concurrent > 1 for s in r.signals)} | "
            f"{np.median(gaps) if gaps else float('nan'):.0f} | {max(gaps) if gaps else 0} | "
            f"{r.missing_source_bars} | {r.halted_source_bars} |")
    lines += ["", "## 판정 단계별 집계 (표시 구간)", "",
              "anchor_pivots: t1(p1) 후보 피벗 · structures: RSI 조건 통과 · swing_set/replaced: p2(t2) 지정·교체 · "
              "breakouts: 이탈 봉×구조 · fail_*: 이탈했지만 처음 실패한 조건 · passed: 성립 구조 · signals: 신호(봉당 1건) · "
              "expired: 간격 초과 삭제", "",
              "| TF | 방향 | " + " | ".join(FUNNEL) + " |", "|---|---|" + "---:|" * len(FUNNEL)]
    sides: tuple[Side, Side] = ("long", "short")
    for r in results:
        for side in sides:
            st = r.stats.get(side, Counter())
            lines.append(f"| {r.timeframe} | {side} | " + " | ".join(str(st.get(k, 0)) for k in FUNNEL) + " |")
    for r in results:
        lines += ["", f"## {r.timeframe} 신호 ({tz.key})", "",
                  "| # | 방향 | t1/p1 | p2/t2 | t3/p3 (신호 확정) | 간격 | RSI | 가격 | 이탈 기준 |", "|---:|---|---|---|---|---:|---|---|---:|"]
        for n, s in enumerate(r.signals, 1):
            lines.append(
                f"| {n} | {s.side} | {s.anchor_time.tz_convert(tz):%m-%d %H:%M} | {s.swing_time.tz_convert(tz):%m-%d %H:%M} | "
                f"{s.trigger_time.tz_convert(tz):%y-%m-%d %H:%M} ({s.signal_time.tz_convert(tz):%H:%M}) | {s.gap_bars} | "
                f"{s.anchor_rsi:.1f}→{s.trigger_rsi:.1f} | {s.anchor_price:,.1f}→{s.trigger_close:,.1f} | "
                f"{s.breakout_level:,.1f} |")
    return "\n".join(lines) + "\n"
