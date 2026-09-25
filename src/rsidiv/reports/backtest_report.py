"""백테스트 보고서 (4단계): 시나리오 비교, 거래·신호·평가금액 CSV, 차트, 요약 Markdown."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from rsidiv.backtest.engine import BacktestResult, Scenario, run_backtest
from rsidiv.backtest.metrics import (
    ReturnMetrics,
    TradeMetrics,
    buy_and_hold,
    exposure,
    per_symbol,
    period_returns,
    return_metrics,
    trade_metrics,
)
from rsidiv.broker.costs import CostModel
from rsidiv.broker.instruments import CryptoInstrument
from rsidiv.core.config import CryptoMarketType, Settings
from rsidiv.core.models import AssetClass
from rsidiv.indicators.rsi import rsi_wilder
from rsidiv.reports.backtest_charts import (
    plot_drawdown,
    plot_r_distribution,
    plot_returns,
    plot_trade,
)
from rsidiv.reports.signal_report import params_lines
from rsidiv.strategy.controller import OUTCOME_LABELS, TradeRecord

_EXIT_LABELS = {"take_profit": "익절", "stop_loss": "손절", "trailing_stop": "트레일링 손절",
                "time_exit": "시간 청산", "session_flatten": "장 마감 청산", "kill_switch": "킬 스위치",
                "end_of_test": "데이터 끝 정리"}


@dataclass(slots=True)
class ScenarioSummary:
    result: BacktestResult
    returns: ReturnMetrics
    trades: TradeMetrics
    exposure: float


@dataclass(slots=True)
class BacktestReport:
    out_dir: Path
    summary_path: Path
    scenarios: list[ScenarioSummary]
    benchmark: ReturnMetrics
    charts: list[Path]


def trades_frame(results: Sequence[BacktestResult], tz: str) -> pd.DataFrame:
    rows = []
    for r in results:
        for t in r.trades:
            s = t.signal
            rows.append({
                "scenario": r.scenario.name, "trade_id": t.trade_id, "symbol": t.symbol,
                "signal_time": s.signal_time, "t1_time": s.t1_time, "p2_time": s.p2_time, "t3_time": s.t3_time,
                "price_t1": s.price_t1, "price_p2": s.price_p2, "price_t3": s.price_t3,
                "rsi_t1": s.rsi_t1, "rsi_t3": s.rsi_t3,
                "decision_time": t.decision_time, "reference_price": t.reference_price,
                "entry_time": t.entry_time, "entry_time_kst": t.entry_time.tz_convert(tz),
                "entry_price": t.entry_price, "qty": t.qty, "initial_stop": t.initial_stop, "target": t.target,
                "final_stop": t.final_stop, "exit_time": t.exit_time,
                "exit_time_kst": t.exit_time.tz_convert(tz) if t.exit_time is not None else None,
                "exit_price": t.exit_price, "exit_reason": t.exit_reason, "bars_held": t.bars_held,
                "gross_pnl": t.gross_pnl, "fees": t.entry_fee + t.exit_fee, "tax": t.tax, "funding": t.funding,
                "net_pnl": t.net_pnl, "r_multiple": t.r_multiple, "gross_r": t.gross_r, "mae_r": t.mae_r,
                "mfe_r": t.mfe_r, "risk_amount": t.risk_amount, "capped": t.capped,
            })
    return pd.DataFrame(rows)


def signals_frame(result: BacktestResult, tz: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "symbol": s.symbol, "signal_time": s.signal_time, "signal_time_kst": s.signal_time.tz_convert(tz),
        "t1_time": s.t1_time, "p2_time": s.p2_time, "t3_time": s.t3_time,
        "price_t1": s.price_t1, "price_p2": s.price_p2, "price_t3": s.price_t3, "rsi_t1": s.rsi_t1,
        "rsi_t3": s.rsi_t3, "outcome": s.outcome, "outcome_label": OUTCOME_LABELS.get(s.outcome, s.outcome),
        "decision_time": s.decision_time, "trade_id": s.trade_id,
    } for s in result.signals])


def _pct(v: float) -> str:
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:+.1%}"


def _num(v: float, fmt: str = ".2f") -> str:
    return "-" if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else format(v, fmt)


def _scenario_table(summaries: Sequence[ScenarioSummary], bench: ReturnMetrics) -> list[str]:
    rows = ["| 시나리오 | 총수익률 | CAGR | MDD | 샤프 | 소르티노 | 칼마 | 거래 | 승률 | 기대값(R) | PF | 비용 합계 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        r, t = s.returns, s.trades
        rows.append(
            f"| {s.result.scenario.label} | {_pct(r.total_return)} | {_pct(r.cagr)} | {r.max_drawdown:.1%} | "
            f"{_num(r.sharpe)} | {_num(r.sortino)} | {_num(r.calmar)} | {t.trades} | {_num(t.win_rate, '.0%')} | "
            f"{_num(t.expectancy_r, '+.3f')} | {_num(t.profit_factor)} | {t.fees + t.tax + t.funding:,.2f} |")
    rows.append(f"| 매수 후 보유 (BTC·ETH 균등) | {_pct(bench.total_return)} | {_pct(bench.cagr)} | "
                f"{bench.max_drawdown:.1%} | {_num(bench.sharpe)} | {_num(bench.sortino)} | {_num(bench.calmar)} | "
                "- | - | - | - | - |")
    return rows


def _pick_samples(trades: Sequence[TradeRecord], count: int) -> list[TradeRecord]:
    if count <= 0 or not trades:
        return []
    if len(trades) <= count:
        return list(trades)
    return [trades[int(i)] for i in np.linspace(0, len(trades) - 1, count).round().astype(int)]


def build_backtest_report(
    settings: Settings,
    data: Mapping[str, pd.DataFrame],
    *,
    scenarios: Sequence[Scenario],
    market_type: CryptoMarketType,
    out_dir: Path,
    funding: Mapping[str, pd.Series] | None = None,
    charts: bool = True,
) -> BacktestReport:
    tz = settings.base.project.display_timezone
    rf = settings.backtest.metrics.risk_free_rate
    results = [run_backtest(settings, data, scenario=s, market_type=market_type, funding=funding) for s in scenarios]
    summaries = [
        ScenarioSummary(r, return_metrics(r.equity["equity"], r.initial_equity, tz, risk_free_rate=rf),
                        trade_metrics(r.trades), exposure(r))
        for r in results
    ]
    base = summaries[0]
    filters = settings.markets.binance.fallback_filters[market_type]
    instruments = {s: CryptoInstrument.from_filter(s, filters[s]) for s in data}
    bench_curve = buy_and_hold(data, base.result.initial_equity, instruments=instruments,
                               costs=CostModel(settings.costs, AssetClass.CRYPTO, market_type=market_type))
    bench = return_metrics(bench_curve, base.result.initial_equity, tz, risk_free_rate=rf)

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = settings.backtest.outputs
    if outputs.trades_csv:
        trades_frame(results, tz).to_csv(out_dir / "trades.csv", index=False)
    if outputs.signals_csv:
        signals_frame(base.result, tz).to_csv(out_dir / "signals.csv", index=False)
    if outputs.equity_csv:
        equity = pd.concat({s.result.scenario.name: s.result.equity["equity"] for s in summaries}, axis=1)
        equity["buy_and_hold"] = bench_curve.reindex(equity.index)
        equity.to_csv(out_dir / "equity.csv")
    pd.DataFrame([vars(e) for r in results for e in r.risk_events]).to_csv(out_dir / "risk_events.csv", index=False)

    chart_paths: list[Path] = []
    label = f"{', '.join(base.result.symbols)} {market_type} 15m"
    if charts and "equity" in outputs.charts:
        gross = next((s for s in summaries if s.result.scenario.name == "gross"), None)
        curves = {"전략 (비용 반영)": base.result.equity["equity"]}
        initials = {"전략 (비용 반영)": base.result.initial_equity}
        if gross is not None:
            curves["전략 (비용 반영 전)"] = gross.result.equity["equity"]
            initials["전략 (비용 반영 전)"] = gross.result.initial_equity
        curves["매수 후 보유"] = bench_curve
        initials["매수 후 보유"] = base.result.initial_equity
        chart_paths.append(plot_returns(curves, initials, title=f"누적 수익률 · {label}", display_tz=tz,
                                        path=out_dir / "returns.png"))
    if charts and "drawdown" in outputs.charts:
        chart_paths.append(plot_drawdown(base.result.equity["equity"], base.result.initial_equity,
                                         title="낙폭 (기준 시나리오)", display_tz=tz, path=out_dir / "drawdown.png"))
    if charts:
        chart_paths.append(plot_r_distribution(base.result.trades, title="거래별 R 분포 (기준 시나리오)",
                                               path=out_dir / "r_distribution.png"))
    samples = _pick_samples(base.result.trades, outputs.signal_sample_count) if charts and \
        "signal_samples" in outputs.charts else []
    rsi = {s: rsi_wilder(f["close"].to_numpy(), base.result.params.rsi.period) for s, f in data.items()}
    sample_names: dict[int, str] = {}
    for n, trade in enumerate(samples, 1):
        name = f"trades/trade_{n:02d}.png"
        chart_paths.append(plot_trade(data[trade.symbol], rsi[trade.symbol], trade, display_tz=tz,
                                      title=f"거래 #{trade.trade_id} · {trade.symbol}", path=out_dir / name))
        sample_names[trade.trade_id] = name

    summary = out_dir / "summary.md"
    summary.write_text(_render(settings, summaries, bench, market_type, sample_names, tz), encoding="utf-8")
    return BacktestReport(out_dir, summary, summaries, bench, chart_paths)


def _render(settings: Settings, summaries: Sequence[ScenarioSummary], bench: ReturnMetrics, market: str,
            samples: Mapping[int, str], tz: str) -> str:
    base = summaries[0]
    r, t, res = base.returns, base.trades, base.result
    costs = settings.costs.crypto
    tier = costs.spot if market == "spot" else costs.usdm_futures
    lines = [
        f"# 백테스트: {', '.join(res.symbols)} ({market}, 15분봉)",
        "",
        f"- 기간: {r.start.tz_convert(tz):%Y-%m-%d %H:%M} ~ {r.end.tz_convert(tz):%Y-%m-%d %H:%M} KST",
        f"- 계좌: 가상화폐 슬리브 {res.initial_equity:,.2f} {res.currency} "
        f"(초기자본 {settings.base.capital.initial:,.0f} USD × {settings.base.capital.allocation['crypto']:.0%}). "
        "국내주식 슬리브는 데이터 보류로 제외",
        f"- 비용: 수수료 메이커 {tier.maker:.3%} / 테이커 {tier.taker:.3%}, 슬리피지 테이커 {costs.slippage.taker_pct:.3%}",
        f"- 사이징: 거래당 위험 {settings.risk.sizing.risk_per_trade:.0%}, 비중 상한 "
        f"{settings.risk.limits.crypto.max_weight:.0%}, 동시 보유 {settings.risk.limits.crypto.max_positions}종목",
        *params_lines(res.params),
        f"- 진입 {res.params.entry.mode}, 손절 {res.params.exit.stop.mode} ×{res.params.exit.stop.atr_mult:g}, "
        f"익절 {res.params.exit.take_profit.mode} {res.params.exit.take_profit.r_multiple:g}R, "
        f"시간 청산 {res.params.exit.time_exit.max_bars}봉",
        "",
        "## 시나리오 비교",
        "",
        *_scenario_table(summaries, bench),
        "",
        "샤프·소르티노는 일별(KST) 수익률, 연 365일 기준. 비용 합계 = 수수료 + 세금 + 펀딩비 (슬리피지는 체결가에 포함).",
        "",
        "![returns](returns.png)",
        "",
        "![drawdown](drawdown.png)",
        "",
        "## 기준 시나리오 상세",
        "",
        "### 거래",
        "",
        "| 항목 | 값 |",
        "|---|---:|",
        f"| 거래 수 | {t.trades} |",
        f"| 승률 | {_num(t.win_rate, '.1%')} |",
        f"| 기대값 (순손익 R) | {_num(t.expectancy_r, '+.3f')} |",
        f"| 기대값 (수수료 전 R, 체결가는 슬리피지 포함) | {_num(t.gross_expectancy_r, '+.3f')} |",
        f"| R 중앙값 | {_num(t.median_r, '+.3f')} |",
        f"| 평균 이익 / 손실 (R) | {_num(t.avg_win_r, '+.2f')} / {_num(t.avg_loss_r, '+.2f')} |",
        f"| Profit factor | {_num(t.profit_factor)} |",
        f"| 평균 보유 | {_num(t.avg_bars_held, '.1f')}봉 |",
        f"| 최대 연속 손실 | {t.max_consecutive_losses}회 |",
        f"| 평균 MAE / MFE (R) | {_num(t.avg_mae_r)} / {_num(t.avg_mfe_r)} |",
        f"| 포지션 보유 시간 비율 | {base.exposure:.1%} |",
        f"| 비중 상한으로 수량이 줄어든 진입 | {res.stats.capped_entries}건 ({res.stats.capped_entries / max(t.trades, 1):.0%}) |",
        f"| 비용 전 손익 / 수수료 / 펀딩비 / 순손익 | {t.gross_pnl:,.2f} / {t.fees:,.2f} / {t.funding:,.2f} / {t.net_pnl:,.2f} {res.currency} |",
        "",
        "![r](r_distribution.png)",
        "",
        "### 청산 사유",
        "",
        "| 사유 | 건수 | 평균 R |",
        "|---|---:|---:|",
    ]
    for reason, count in sorted(t.exits.items(), key=lambda kv: -kv[1]):
        rs = [x.r_multiple for x in res.trades if x.exit_reason == reason]
        lines.append(f"| {_EXIT_LABELS.get(reason, reason)} | {count} | {np.nanmean(rs):+.3f} |")
    lines += ["", "### 종목별", "", "| 종목 | 거래 | 승률 | 기대값(R) | 수수료 전 R | PF | 순손익 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for _, row in per_symbol(res.trades).iterrows():
        lines.append(f"| {row['symbol']} | {row['trades']} | {row['win_rate']:.0%} | {row['expectancy_r']:+.3f} | "
                     f"{row['gross_expectancy_r']:+.3f} | {_num(row['profit_factor'])} | {row['net_pnl']:,.2f} |")
    yearly = period_returns(res.equity["equity"], res.initial_equity, tz)
    bench_years = {s.result.scenario.name: period_returns(s.result.equity["equity"], s.result.initial_equity, tz)
                   for s in summaries}
    lines += ["", "### 연도별 수익률", "", "| 연도 | " + " | ".join(s.result.scenario.label for s in summaries) + " |",
              "|---|" + "---:|" * len(summaries)]
    for ts in yearly.index:
        cells = [_pct(float(bench_years[s.result.scenario.name].get(ts, math.nan))) for s in summaries]
        lines.append(f"| {ts:%Y} | " + " | ".join(cells) + " |")
    outcomes = Counter(s.outcome for s in res.signals)
    lines += ["", "### 신호 처리 결과", "", f"신호 {len(res.signals)}건 중:", "", "| 결과 | 건수 |", "|---|---:|"]
    for outcome, count in outcomes.most_common():
        lines.append(f"| {OUTCOME_LABELS.get(outcome, outcome)} | {count} |")
    lines += ["", "### 리스크 이벤트", ""]
    if res.risk_events:
        lines += ["| 시각 (KST) | 종류 | 실측 | 기준 |", "|---|---|---:|---:|"]
        for e in res.risk_events:
            lines.append(f"| {e.time.astimezone(ZoneInfo(tz)):%Y-%m-%d %H:%M} | {e.kind} | "
                         f"{e.value:.2%} | {e.threshold:.0%} |")
    else:
        lines.append("없음 (일일 손실 3%·누적 MDD 30%에 한 번도 닿지 않음)")
    if samples:
        lines += ["", "## 거래 표본 차트", "", "거래 목록에서 고르게 뽑았다 (전체는 `trades.csv`).", ""]
        lines += [f"- 거래 #{tid}: [{name}]({name})" for tid, name in samples.items()]
    lines += ["", "파일: `trades.csv`(모든 시나리오), `signals.csv`, `equity.csv`, `risk_events.csv`. 시각은 UTC, `_kst` 열은 KST.",
              *[f"- 참고: {note}" for s in summaries for note in s.result.notes]]
    return "\n".join(lines) + "\n"
