"""4단계 백테스트 실행: 유니버스 준비물 확인 → 신호 → 순위 → 시나리오 × 리스크 규칙 → BTC 보유 비교 → 보고서.

시나리오 (costs.yaml·backtest.yaml):
- 비용: gross(수수료·슬리피지·펀딩 없음) + net × 슬리피지 배수(0/1/2)
- 리스크 규칙: with_live_rules(일일 손실·킬 스위치) / without_live_rules
- (선택) 정밀 모드: 기본 시나리오(net ×1, 규칙 적용)를 1분봉 판정으로 한 번 더
기본 시나리오 = net ×1 + 규칙 적용.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from perpdiv.backtest.engine import BacktestEngine, BacktestResult, Scenario
from perpdiv.backtest.instruments import InstrumentBook
from perpdiv.backtest.market import ExecBars, Listing, MarketStore
from perpdiv.backtest.metrics import performance, split_stats, trade_stats, yearly_returns
from perpdiv.backtest.universe_signals import attach_ranks, generate_universe_signals
from perpdiv.core.config import Settings, resolve_project_path
from perpdiv.core.timeutil import utc_now
from perpdiv.data.candidates import CandidateSet, universe_dir
from perpdiv.data.check import build_archive, build_cache
from perpdiv.data.ranks import RANK_FILE, build_rank_table
from perpdiv.data.resample import resample_ohlcv
from perpdiv.reports.trade_charts import plot_equity, plot_trade

PRIMARY = "net_x1_rules"
STATUS_ORDER = ("entered", "skipped_holding", "skipped_simultaneous", "skipped_daily_loss", "skipped_kill_switch",
                "skipped_min_notional", "skipped_no_data", "skipped_stop_too_tight", "no_confluence", "out_of_rank")


class MemoMarket:
    """시나리오마다 같은 봉·펀딩 조회가 반복되므로 메모리에 기억한다."""

    def __init__(self, market: MarketStore) -> None:
        self.market = market
        self._bars = lru_cache(maxsize=8192)(market.exec_bars)
        self._funding = lru_cache(maxsize=8192)(market.funding)

    def exec_bars(self, symbol: str, start: dt.datetime, end: dt.datetime) -> ExecBars:
        return self._bars(symbol, start, end)

    def funding(self, symbol: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        return self._funding(symbol, start, end)

    def minute_bars(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        return self.market.minute_bars(symbol, start, end)


def scenario_matrix(settings: Settings, precise: bool) -> list[Scenario]:
    costs = []
    if settings.costs.scenarios.include_gross:
        costs.append(("gross", {"fees": False, "slippage_mult": 0.0, "funding": False}))
    costs += [(f"net_x{m:g}", {"fees": True, "slippage_mult": m, "funding": settings.costs.funding.enabled})
              for m in settings.costs.scenarios.slippage_multipliers]
    out = []
    for variant in settings.backtest.risk_rules_variants:
        rules = variant == "with_live_rules"
        out += [Scenario(f"{name}_{'rules' if rules else 'norules'}", live_rules=rules, **kw)  # type: ignore[arg-type]
                for name, kw in costs]
    if precise:
        out += [Scenario(f"net_x1_{rules}_precise", live_rules=rules == "rules", precise=True,
                         funding=settings.costs.funding.enabled) for rules in ("rules", "norules")]
    return out


def buy_and_hold(settings: Settings, market: MemoMarket, start: dt.datetime, end: dt.datetime) -> pd.Series:
    """BTCUSDT 무기한 1배 롱: 첫 봉 시가에 전액 매수(테이커 수수료·슬리피지), 펀딩비 반영, 15분봉 종가로 평가."""
    symbol = settings.backtest.benchmark.symbol
    bars = market.market.exec_bars(symbol, start, end).bars
    capital = float(settings.base.account.initial_capital)
    taker = settings.costs.fees.rate("taker")
    price = float(bars["open"].iloc[0]) * (1 + settings.costs.slippage.taker_pct)
    qty = capital / (price * (1 + taker))
    cash = capital - qty * price * taker
    funding = market.market.funding(symbol, start, end) if settings.costs.funding.enabled else pd.DataFrame()
    paid = pd.Series(0.0, index=bars.index)
    if not funding.empty:
        amounts = qty * funding["mark"] * funding["rate"]
        cum = amounts.cumsum()
        paid = cum.reindex(bars.index + pd.Timedelta(minutes=15), method="ffill").fillna(0.0)
        paid.index = bars.index
    equity = cash + qty * (bars["close"] - price) - paid
    equity.index = bars.index + pd.Timedelta(minutes=15)
    out: pd.Series = pd.concat([pd.Series([capital], index=[pd.Timestamp(start)]), equity])
    return out


def run_backtest(settings: Settings, *, precise: bool = False, sample_charts: int | None = None,
                 log: Callable[[str], None] = print, out_dir: Path | None = None) -> Path:
    archive = build_archive(settings)
    cache = build_cache(settings, archive, workers=16)
    udir = universe_dir(settings)
    candidates = CandidateSet.load(udir / "candidates.json")
    start, end = candidates.bounds
    expected = settings.data.backtest_period.bounds(utc_now())
    if (start, end) != expected:
        log(f"주의: 후보군 구간 {start}~{end} 이 설정 구간 {expected[0]}~{expected[1]} 과 다릅니다 (universe 재실행 필요)")
    rank_path = udir / RANK_FILE
    listing = Listing(archive, udir / "listing.json")
    if not rank_path.is_file():
        log("정밀 순위 표 생성")
        build_rank_table(settings, cache, candidates, log=log, months_available=listing.months)
    log("신호 생성 (캐시 있으면 재사용)")
    signals = attach_ranks(generate_universe_signals(settings, candidates, log=log), rank_path)
    signals.index = pd.RangeIndex(len(signals))
    log(f"신호 {len(signals):,}건 (순위 ≤ {settings.universe.top_n}: {(signals['rank'] <= settings.universe.top_n).sum():,})")
    minute_dir = resolve_project_path(settings.base.paths.cache_dir) / "vision_um_daily" / "klines_1m"
    market = MemoMarket(MarketStore(settings, cache, listing, minute_dir))
    instruments = InstrumentBook(settings.backtest, archive, udir / "instruments.json")
    results: dict[str, BacktestResult] = {}
    for scenario in scenario_matrix(settings, precise):
        engine = BacktestEngine(settings, market, instruments, scenario, start=start, end=end)
        results[scenario.name] = engine.run(signals)
        r = results[scenario.name]
        log(f"  {scenario.name}: 거래 {len(r.trades)}, 최종 {r.equity.iloc[-1]:,.1f} USDT")
    bench = buy_and_hold(settings, market, start, end)
    out = out_dir or resolve_project_path(settings.base.paths.report_dir) / "backtest" / f"{utc_now():%Y%m%dT%H%M%SZ}"
    out.mkdir(parents=True, exist_ok=True)
    write_report(settings, results, bench, signals, market, out, sample_charts, log)
    return out


def _fmt(x: float, pct: bool = False, digits: int = 2) -> str:
    if x != x:
        return "–"
    return f"{x * 100:,.{digits - 1}f}%" if pct else f"{x:,.{digits}f}"


def write_report(settings: Settings, results: dict[str, BacktestResult], bench: pd.Series, signals: pd.DataFrame,
                 market: MemoMarket, out: Path, sample_charts: int | None, log: Callable[[str], None]) -> None:
    tz = ZoneInfo(settings.base.project.display_timezone)
    first = next(iter(results.values()))
    start, end = first.start, first.end
    assert start is not None and end is not None
    rows = []
    for name, r in results.items():
        perf = performance(r.equity, start, end, tz, settings.backtest.metrics.risk_free_rate)
        rows.append({"scenario": name} | perf | trade_stats(r.trades) |
                    {"fees": float(r.trades["fees"].sum()) if len(r.trades) else 0.0,
                     "funding": float(r.trades["funding"].sum()) if len(r.trades) else 0.0,
                     "risk_events": len(r.risk_events)})
    bench_perf = performance(bench, start, end, tz)
    rows.append({"scenario": "btc_buy_hold"} | bench_perf)
    table = pd.DataFrame(rows).set_index("scenario")
    table.to_csv(out / "scenarios.csv")
    for name, r in results.items():
        r.trades.to_csv(out / f"trades_{name}.csv", index=False)
        r.equity.rename("equity").to_csv(out / f"equity_{name}.csv")
    bench.rename("equity").to_csv(out / "equity_btc_buy_hold.csv")
    primary = results.get(PRIMARY) or first
    primary.signals.to_csv(out / f"signals_{primary.scenario.name}.csv", index=False)
    (out / f"risk_events_{primary.scenario.name}.json").write_text(
        json.dumps([asdict(e) for e in primary.risk_events], default=str, ensure_ascii=False, indent=1), encoding="utf-8")
    # 차트
    curves = {f"strategy {n}": results[n].equity for n in (PRIMARY, "net_x1_norules", "gross_norules") if n in results}
    curves["BTC buy & hold"] = bench
    plot_equity(curves, tz=tz, path=out / "equity.png",
                title=f"Equity (log) {start.tz_convert(tz):%Y-%m-%d} ~ {end.tz_convert(tz):%Y-%m-%d} {tz.key}")
    count = settings.backtest.outputs.signal_sample_count if sample_charts is None else sample_charts
    sampled = results.get("net_x1_norules") or primary  # 규칙 적용 버전은 킬 스위치로 거래가 적어 미적용 버전에서 뽑는다
    if count and len(sampled.trades):
        (out / "trades").mkdir(exist_ok=True)
        picks = np.unique(np.linspace(0, len(sampled.trades) - 1, min(count, len(sampled.trades))).round().astype(int))
        for n in picks:
            trade = sampled.trades.iloc[n]
            sig: pd.Series = signals.iloc[int(trade["signal_id"])]
            tf_min = int(sig["tf_minutes"])
            a = pd.Timestamp(sig["anchor_time"]) - pd.Timedelta(minutes=tf_min * 15)
            b = max(pd.Timestamp(trade["exit_time"]), pd.Timestamp(sig["signal_time"])) + pd.Timedelta(minutes=tf_min * 10)
            raw = market.market.source(trade["symbol"], a.to_pydatetime(), b.to_pydatetime())
            bars = resample_ohlcv(raw, sig["timeframe"], source_timeframe=settings.data.timeframes.collect)
            if len(bars) > 5:
                plot_trade(bars, trade, sig, tz=tz, path=out / "trades" / f"{n + 1:04d}.png", number=int(n) + 1)
    (out / "summary.md").write_text(render_summary(settings, results, table, bench, signals, tz), encoding="utf-8")
    log(f"보고서: {out}")


def _split_table(frame: pd.DataFrame, title: str) -> list[str]:
    lines = ["| " + title + " | 거래 | 승률 | 손익비 | 기대값(R) | 이익계수 | 평균 보유(h) | 순손익 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, row in frame.iterrows():
        lines.append(f"| {key} | {row['trades']:.0f} | {_fmt(row['win_rate'], True)} | {_fmt(row['payoff'])} | "
                     f"{_fmt(row['expectancy_r'], digits=3)} | {_fmt(row['profit_factor'])} | "
                     f"{_fmt(row['avg_hold_h'], digits=1)} | {_fmt(row['net_pnl'], digits=1)} |")
    return [*lines, ""]


def _detail_sections(settings: Settings, result: BacktestResult, tz: ZoneInfo) -> list[str]:
    name = result.scenario.name
    trades = result.trades.copy()
    lines = ["", f"## 거래 통계 ({name})", ""]
    if trades.empty:
        return [*lines, "거래 없음", ""]
    trades["year"] = pd.DatetimeIndex(trades["entry_time"]).tz_convert(tz).year
    trades["stop_pct"] = (trades["entry_price"] - trades["stop"]).abs() / trades["entry_price"]
    trades["equity_return"] = trades["net_pnl"] / trades["equity_before"]
    splits = [("side", "방향"), ("timeframe", "타임프레임")]
    if settings.strategy.confluence.enabled:
        splits += [("confluence_tfs", "동시 성립 TF 조합")]
    for by, title in (*splits, ("exit_reason", "청산 사유"), ("year", "진입 연도")):
        lines += [f"### {title}별", "", *_split_table(split_stats(trades, by), title)]
    stops = trades.groupby("timeframe")["stop_pct"].describe(percentiles=[0.5, 0.9])
    lines += ["### 손절폭 (진입가 대비, 100% 진입이므로 = 손절 시 평가금액 손실률)", "",
              "| TF | 거래 | 중앙값 | 90% | 최대 |", "|---|---:|---:|---:|---:|"]
    for tf, row in stops.iterrows():
        lines.append(f"| {tf} | {row['count']:.0f} | {row['50%']:.1%} | {row['90%']:.1%} | {row['max']:.1%} |")
    worst = trades.nsmallest(5, "equity_return")
    lines += ["", "### 평가금액 대비 손실이 가장 큰 거래", "",
              "| 진입 (KST) | 코인 | TF | 방향 | 청산 | 손절폭 | 평가금액 대비 |", "|---|---|---|---|---|---:|---:|"]
    for _, t in worst.iterrows():
        lines.append(f"| {pd.Timestamp(t['entry_time']).tz_convert(tz):%Y-%m-%d %H:%M} | {t['coin']} | {t['timeframe']} | "
                     f"{t['side']} | {t['exit_reason']} | {t['stop_pct']:.1%} | {t['equity_return']:+.1%} |")
    coins = split_stats(trades, "coin").sort_values("trades", ascending=False)
    lines += ["", "### 코인별 (거래 많은 순 상위 15, 전체는 CSV)", "",
              "| 코인 | 거래 | 승률 | 기대값(R) | 순손익 |", "|---|---:|---:|---:|---:|"]
    for key, row in coins.head(15).iterrows():
        lines.append(f"| {key} | {row['trades']:.0f} | {_fmt(row['win_rate'], True)} | "
                     f"{_fmt(row['expectancy_r'], digits=3)} | {_fmt(row['net_pnl'], digits=1)} |")
    lines.append(f"\n거래한 코인 {coins.shape[0]}개. 순손익 상위: " + ", ".join(
        f"{k} {v:+,.0f}" for k, v in coins["net_pnl"].sort_values(ascending=False).head(5).items()) +
        " / 하위: " + ", ".join(f"{k} {v:+,.0f}" for k, v in coins["net_pnl"].sort_values().head(5).items()))
    sig = result.signals
    in_period = sig[sig["status"] != "outside_period"]
    tfs = settings.data.timeframes.signal
    lines += ["", f"### 신호 집계 ({name})", "", "| 구분 | " + " | ".join(tfs) + " | 합계 |",
              "|---|" + "---:|" * (len(tfs) + 1),
              "| 전체 신호 | " + " | ".join(str(int((in_period["timeframe"] == tf).sum())) for tf in tfs) +
              f" | {len(in_period)} |"]
    for status in STATUS_ORDER:
        sub = in_period[in_period["status"] == status]
        lines.append(f"| {status} | " + " | ".join(str(int((sub["timeframe"] == tf).sum())) for tf in tfs) +
                     f" | {len(sub)} |")
    amb = int((trades["ambiguous_bars"] > 0).sum())
    lines += ["", f"같은 15분봉에서 손절·익절이 모두 닿은 거래: {amb}건", ""]
    return lines


def render_summary(settings: Settings, results: dict[str, BacktestResult], table: pd.DataFrame, bench: pd.Series,
                   signals: pd.DataFrame, tz: ZoneInfo) -> str:
    first = next(iter(results.values()))
    start, end = first.start, first.end
    assert start is not None and end is not None
    primary = results.get(PRIMARY) or first
    st = settings.strategy
    lines = [
        "# 백테스트 요약",
        "",
        f"- 구간 {start.tz_convert(tz):%Y-%m-%d} ~ {end.tz_convert(tz):%Y-%m-%d} ({tz.key}), 초기 자본 "
        f"{settings.base.account.initial_capital:,.0f} USDT, 레버리지 {settings.base.exchange.leverage}배, 진입 100%, 동시 1개",
        f"- 유니버스: 신호 시각 직전 24h 거래대금 상위 {settings.universe.top_n} 코인 (TradFi·스테이블 제외, 상장폐지 포함)",
        f"- 신호: 피벗 L{st.pivot.left}/R{st.pivot.right} {st.pivot.tie_rule}, RSI {st.rsi.period} "
        f"{st.bullish.oversold:g}/{st.bearish.overbought:g}, 간격 ≤ {st.structure.gap_bars.max}, 구조 폐기 "
        f"{'on' if st.structure.discard_on_anchor_break else 'off'} · TF {', '.join(settings.data.timeframes.signal)}",
        f"- 진입: 동시 성립 {'서로 다른 TF ' + str(st.confluence.min_timeframes) + '개 이상 (신호 유효 ' + str(st.confluence.validity_bars) + '봉)' if st.confluence.enabled else '끔(TF 독립)'}"
        f", 손절폭 > 진입가 × {st.exit.stop.min_distance_pct:.2%}",
        f"- 청산: 손절 {'진입가' if st.exit.stop.basis == 'entry_price' else '신호 봉'} ∓ ATR×{st.exit.stop.atr_mult:g}, "
        f"익절 {st.exit.take_profit.mode} "
        f"{st.exit.take_profit.r_multiple:g}R, 시간 {st.exit.time_exit.bars}봉, 판정 {st.exit.evaluation_timeframe}",
        f"- 비용: 테이커 {settings.costs.fees.taker:.3%}·메이커 {settings.costs.fees.maker:.3%}, 슬리피지 "
        f"{settings.costs.slippage.taker_pct:.3%} × 배수, 펀딩비 반영",
        f"- 기본 시나리오: `{primary.scenario.name}`",
        "",
        "## 시나리오별 성과",
        "",
        "| 시나리오 | 총수익 | CAGR | MDD | 샤프 | 소르티노 | 칼마 | 거래 | 승률 | 손익비 | 기대값(R) | 평균 보유(h) | 수수료 | 펀딩 | 리스크 이벤트 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in table.iterrows():
        def g(key: str, row: pd.Series = r) -> float:
            value = row.get(key, float("nan"))
            return float(value) if value == value else float("nan")
        lines.append(
            f"| {name} | {_fmt(g('total_return'), True)} | {_fmt(g('cagr'), True)} | {_fmt(g('mdd'), True)} | "
            f"{_fmt(g('sharpe'))} | {_fmt(g('sortino'))} | {_fmt(g('calmar'))} | {_fmt(g('trades'), digits=0)} | "
            f"{_fmt(g('win_rate'), True)} | {_fmt(g('payoff'))} | {_fmt(g('expectancy_r'), digits=3)} | "
            f"{_fmt(g('avg_hold_h'), digits=1)} | {_fmt(g('fees'), digits=0)} | {_fmt(g('funding'), digits=0)} | "
            f"{_fmt(g('risk_events'), digits=0)} |")
    lines += ["", "## 연도별 수익률", "", "| 연도 | " + " | ".join([PRIMARY, "net_x1_norules", "BTC B&H"]) + " |",
              "|---|---:|---:|---:|"]
    curves = [results[n].equity if n in results else None for n in (PRIMARY, "net_x1_norules")] + [bench]
    yearly = [yearly_returns(c, start, end, tz) if c is not None else {} for c in curves]
    for year in sorted(set().union(*[set(y) for y in yearly])):
        lines.append(f"| {year} | " + " | ".join(_fmt(y.get(year, float('nan')), True) for y in yearly) + " |")
    details = [results[n] for n in (PRIMARY, "net_x1_norules") if n in results] or [primary]
    for detail in details:
        lines += _detail_sections(settings, detail, tz)
    events = primary.risk_events
    lines += ["", f"## 리스크 이벤트 ({primary.scenario.name})", ""]
    if events:
        kinds = pd.Series([e.kind for e in events]).value_counts()
        lines.append(", ".join(f"{k} {v}회" for k, v in kinds.items()))
        lines += ["", "| 시각 | 종류 | 평가금액 | 기준 | 값 | 내용 |", "|---|---|---:|---:|---:|---|"]
        for e in events[:40]:
            lines.append(f"| {e.time.tz_convert(tz):%Y-%m-%d %H:%M} | {e.kind} | {e.equity:,.1f} | {e.reference:,.1f} | "
                         f"{e.value:.2%} | {e.detail} |")
    else:
        lines.append("없음")
    for base in (PRIMARY, "net_x1_norules"):
        precise, plain = results.get(base + "_precise"), results.get(base)
        if precise is None or plain is None or plain.trades.empty:
            continue
        amb = int((plain.trades["ambiguous_bars"] > 0).sum())
        lines += ["", f"## 정밀 모드 (1분봉) — {base}", "",
                  f"같은 15분봉에서 손절·익절이 모두 닿은 거래 {amb}건. 최종 평가금액 "
                  f"{plain.equity.iloc[-1]:,.1f} → {precise.equity.iloc[-1]:,.1f} USDT (1분봉 판정)."]
    return "\n".join(lines) + "\n"
