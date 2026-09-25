"""명령행 진입점: ``python -m rsidiv <command>``.

- ``config``  : 설정 파일 전체를 검증하고 핵심 값을 요약 출력
- ``secrets`` : .env / 환경변수에 어떤 키가 설정되어 있는지 이름만 출력 (값은 출력하지 않음)
- ``data-check`` : 바이낸스 데이터 확보 가능 여부 점검 → Markdown 보고서
- ``fetch``   : 백테스트 기간의 바이낸스 15분봉(선물은 펀딩비 포함)을 받아 Parquet 캐시에 저장
- ``signals`` : 다이버전스 신호 탐지 + 육안 검증 차트·CSV·요약 (기본: 마지막 3개월)
- ``backtest``: 백테스트 (시나리오 비교, 거래·신호·평가금액 CSV, 차트, 요약)
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections.abc import Sequence
from typing import get_args

from pydantic import ValidationError

from rsidiv.core.config import (
    ASSET_CLASSES,
    Settings,
    apply_dotted,
    load_settings,
    resolve_config_dir,
    timeframe_minutes,
)
from rsidiv.core.secrets import ENV_VARS, load_secrets
from rsidiv.core.timeutil import UTC, utc_now


def _summarize(settings: Settings) -> str:
    base, risk = settings.base, settings.risk
    lines = [
        f"기준통화 {base.project.base_currency}, 초기자본 {base.capital.initial:,.2f}, "
        f"배분 {dict(base.capital.allocation)}",
        f"타임프레임 {settings.data.timeframe}, 백테스트 {settings.data.backtest_period.start} ~ "
        f"{settings.data.backtest_period.end or '현재'}",
        f"코인 시장 {settings.universe.crypto.market_type}, 심볼 {settings.universe.crypto.symbols}",
    ]
    for asset_class in ASSET_CLASSES:
        params = settings.strategy.for_asset_class(asset_class)
        limits = getattr(risk.limits, asset_class)
        lines.append(
            f"[{asset_class}] pivot L={params.pivot.left} R={params.pivot.right} "
            f"({params.pivot.price_source}), RSI {params.rsi.period}, 진입 {params.entry.mode}, "
            f"손절 {params.exit.stop.mode}, 익절 {params.exit.take_profit.mode}, "
            f"최대 {limits.max_positions}종목/비중 {limits.max_weight:.0%}"
        )
    lines.append(
        f"리스크: 거래당 {risk.sizing.risk_per_trade:.1%}, 일일손실 {risk.daily_loss.threshold:.0%}, "
        f"킬스위치 MDD {risk.kill_switch.max_drawdown:.0%} ({risk.kill_switch.on_trip}), "
        f"API 연속오류 {risk.api_errors.max_consecutive}회"
    )
    lines.append(f"실행 모드: {settings.live.mode}")
    return "\n".join(lines)


def _period(settings: Settings, start: str | None, end: str | None) -> tuple[dt.datetime, dt.datetime]:
    period = settings.data.backtest_period
    start_date = dt.date.fromisoformat(start) if start else period.start
    end_date = dt.date.fromisoformat(end) if end else period.end
    start_ts = dt.datetime.combine(start_date, dt.time(), tzinfo=UTC)
    end_ts = dt.datetime.combine(end_date, dt.time(), tzinfo=UTC) if end_date else utc_now()
    return start_ts, end_ts


def _crypto_settings(args: argparse.Namespace) -> Settings:
    settings = load_settings(args.config_dir)
    if args.retries is not None:
        settings = apply_dotted(settings, {"data.providers.binance.max_retries": args.retries})
    return settings


def _cmd_data_check(args: argparse.Namespace) -> int:
    from rsidiv.data.base import DataProvider
    from rsidiv.data.binance import MarketType
    from rsidiv.data.check import render_markdown, run_crypto_check
    from rsidiv.data.factory import binance_provider, binance_rest, resolve_project_path

    settings = _crypto_settings(args)
    source = args.source or settings.data.providers.binance.history_source
    markets = args.markets or [settings.universe.crypto.market_type]
    start, end = _period(settings, args.start, args.end)
    providers: dict[MarketType, DataProvider] = {
        m: binance_provider(settings, m, source=source, cached=not args.no_cache) for m in markets
    }
    filters = {m: binance_rest(settings, m) for m in markets} if not args.skip_filters else None
    report = run_crypto_check(
        settings, providers, source=source, start=start, end=end, filters_providers=filters
    )
    text = render_markdown(report)
    out_dir = resolve_project_path(settings.base.paths.report_dir) / "data_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"data_check_{report.generated_at:%Y%m%dT%H%M%SZ}.md"
    out_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"보고서 저장: {out_path}")
    return 0 if not any(r.errors for r in report.results) else 2


def _cmd_fetch(args: argparse.Namespace) -> int:
    from rsidiv.data.base import DataSourceError
    from rsidiv.data.factory import binance_provider

    settings = _crypto_settings(args)
    markets = args.markets or [settings.universe.crypto.market_type]
    start, end = _period(settings, args.start, args.end)
    failed = False
    for market in markets:
        provider = binance_provider(settings, market, source=args.source)
        for symbol in settings.universe.crypto.symbols:
            try:
                frame = provider.fetch_ohlcv(symbol, settings.data.timeframe, start, end)
                span = f"{frame.index[0]} ~ {frame.index[-1]}" if len(frame) else "데이터 없음"
                print(f"[{market}] {symbol} {settings.data.timeframe}: {len(frame)}봉 ({span})")
                if market == "usdm_futures":
                    funding = provider.funding_rates(symbol, start, end)
                    print(f"[{market}] {symbol} funding: {len(funding)}건")
            except DataSourceError as exc:
                failed = True
                print(f"[{market}] {symbol} 실패: {exc}", file=sys.stderr)
    return 2 if failed else 0


def _cmd_signals(args: argparse.Namespace) -> int:
    import pandas as pd

    from rsidiv.data.base import DataSourceError
    from rsidiv.data.factory import binance_provider, resolve_project_path
    from rsidiv.reports.signal_report import build_signal_report

    settings = load_settings(args.config_dir)
    crypto = settings.universe.crypto
    symbol = args.symbol or crypto.symbols[0]
    market = args.market or crypto.market_type
    timeframe = settings.data.timeframe
    start, end = _period(settings, None, args.end)
    try:
        frame = binance_provider(settings, market, source=args.source).fetch_ohlcv(symbol, timeframe, start, end)
    except DataSourceError as exc:
        print(f"데이터 조회 실패: {exc}", file=sys.stderr)
        return 2
    if frame.empty:
        print("데이터가 없습니다", file=sys.stderr)
        return 2
    window_end = frame.index[-1] + pd.Timedelta(minutes=timeframe_minutes(timeframe))
    window_start = window_end - pd.DateOffset(months=args.months)
    out_dir = resolve_project_path(settings.base.paths.report_dir) / "signals" / (
        f"{symbol.replace('/', '-')}_{market}_{window_start:%Y%m%d}_{window_end:%Y%m%d}"
    )
    report = build_signal_report(
        frame, settings.strategy.for_asset_class("crypto"),
        symbol=symbol, market=market, timeframe=timeframe, asset_class="crypto",
        window_start=window_start.to_pydatetime(), window_end=window_end.to_pydatetime(),
        display_tz=settings.base.project.display_timezone, out_dir=out_dir,
    )
    total = sum(c.accepted for c in report.all_candidates)
    print(f"{symbol} {market}: 표시 기간 신호 {len(report.signals)}건, 필터 탈락 후보 {len(report.near_misses)}건 "
          f"(전체 기간 신호 {total}건, t3 후보 {len(report.all_candidates)}건)")
    print(f"보고서: {report.summary_path}")
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    from rsidiv.backtest.engine import default_scenarios
    from rsidiv.data.base import DataSourceError
    from rsidiv.data.factory import binance_provider, resolve_project_path
    from rsidiv.reports.backtest_report import build_backtest_report

    try:
        settings = load_settings(args.config_dir, args.override)
    except (ValidationError, ValueError) as exc:
        print(f"설정 검증 실패: {exc}", file=sys.stderr)
        return 1
    market = args.market or settings.universe.crypto.market_type
    start, end = _period(settings, args.start, args.end)
    provider = binance_provider(settings, market, source=args.source)
    try:
        data = {s: provider.fetch_ohlcv(s, settings.data.timeframe, start, end)
                for s in settings.universe.crypto.symbols}
        funding = ({s: provider.funding_rates(s, start, end) for s in data}
                   if market == "usdm_futures" else None)
    except DataSourceError as exc:
        print(f"데이터 조회 실패: {exc}", file=sys.stderr)
        return 2
    first = min(f.index[0] for f in data.values())
    last = max(f.index[-1] for f in data.values())
    scenarios = default_scenarios(settings)
    if args.scenarios == "base":
        scenarios = scenarios[:1]
    run_id = args.name or f"{market}_{first:%Y%m%d}_{last:%Y%m%d}"
    out_dir = resolve_project_path(settings.base.paths.report_dir) / "backtest" / run_id
    report = build_backtest_report(settings, data, scenarios=scenarios, market_type=market, out_dir=out_dir,
                                   funding=funding, charts=not args.no_charts)
    for s in report.scenarios:
        print(f"[{s.result.scenario.name}] 수익률 {s.returns.total_return:+.2%}, MDD {s.returns.max_drawdown:.2%}, "
              f"거래 {s.trades.trades}건, 기대값 {s.trades.expectancy_r:+.3f}R")
    print(f"[매수 후 보유] 수익률 {report.benchmark.total_return:+.2%}, MDD {report.benchmark.max_drawdown:.2%}")
    print(f"보고서: {report.summary_path}")
    return 0


def _add_crypto_args(cmd: argparse.ArgumentParser) -> None:
    from rsidiv.data.binance import MarketType

    cmd.add_argument("--config-dir", default=None)
    cmd.add_argument("--markets", nargs="+", choices=get_args(MarketType), default=None,
                     help="점검할 시장 (기본: universe.yaml 의 market_type)")
    cmd.add_argument("--source", choices=["rest", "vision"], default=None,
                     help="데이터 출처 (기본: data.yaml 의 history_source)")
    cmd.add_argument("--start", default=None, help="YYYY-MM-DD (기본: backtest_period.start)")
    cmd.add_argument("--end", default=None, help="YYYY-MM-DD, 미포함 (기본: 현재)")
    cmd.add_argument("--retries", type=int, default=None, help="재시도 횟수 덮어쓰기")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rsidiv")
    sub = parser.add_subparsers(dest="command", required=True)

    config_cmd = sub.add_parser("config", help="설정 검증 및 요약")
    config_cmd.add_argument("--config-dir", default=None)
    config_cmd.add_argument("--override", action="append", default=[], help="덮어쓰기 YAML")

    secrets_cmd = sub.add_parser("secrets", help="비밀정보 설정 여부 (이름만)")
    secrets_cmd.add_argument("--env-file", default=".env")

    check_cmd = sub.add_parser("data-check", help="바이낸스 데이터 확보 가능 여부 점검")
    _add_crypto_args(check_cmd)
    check_cmd.add_argument("--no-cache", action="store_true", help="캐시를 쓰지 않고 원천에서 직접 조회")
    check_cmd.add_argument("--skip-filters", action="store_true", help="거래소 주문 필터 조회 생략")

    fetch_cmd = sub.add_parser("fetch", help="바이낸스 데이터 다운로드 → Parquet 캐시")
    _add_crypto_args(fetch_cmd)

    signals_cmd = sub.add_parser("signals", help="다이버전스 신호 + 육안 검증 차트")
    signals_cmd.add_argument("--config-dir", default=None)
    signals_cmd.add_argument("--symbol", default=None, help="기본: universe.yaml 의 첫 코인 심볼")
    signals_cmd.add_argument("--market", choices=["spot", "usdm_futures"], default=None)
    signals_cmd.add_argument("--source", choices=["rest", "vision"], default=None)
    signals_cmd.add_argument("--months", type=int, default=3, help="표시 기간 (마지막 N개월)")
    signals_cmd.add_argument("--end", default=None, help="YYYY-MM-DD, 미포함 (기본: 현재)")

    bt_cmd = sub.add_parser("backtest", help="백테스트 + 성과 보고서")
    bt_cmd.add_argument("--config-dir", default=None)
    bt_cmd.add_argument("--override", action="append", default=[], help="실험용 덮어쓰기 YAML")
    bt_cmd.add_argument("--market", choices=["spot", "usdm_futures"], default=None)
    bt_cmd.add_argument("--source", choices=["rest", "vision"], default=None)
    bt_cmd.add_argument("--start", default=None, help="YYYY-MM-DD (기본: backtest_period.start)")
    bt_cmd.add_argument("--end", default=None, help="YYYY-MM-DD, 미포함 (기본: 현재)")
    bt_cmd.add_argument("--scenarios", choices=["all", "base"], default="all")
    bt_cmd.add_argument("--name", default=None, help="보고서 폴더 이름 (기본: 시장_시작_끝)")
    bt_cmd.add_argument("--no-charts", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "data-check":
        return _cmd_data_check(args)
    if args.command == "fetch":
        return _cmd_fetch(args)
    if args.command == "signals":
        return _cmd_signals(args)
    if args.command == "backtest":
        return _cmd_backtest(args)

    if args.command == "config":
        try:
            settings = load_settings(args.config_dir, args.override)
        except (ValidationError, ValueError) as exc:
            print(f"설정 검증 실패 ({resolve_config_dir(args.config_dir)}):\n{exc}", file=sys.stderr)
            return 1
        print(f"설정 검증 통과: {resolve_config_dir(args.config_dir)}")
        print(_summarize(settings))
        return 0

    secrets = load_secrets(args.env_file)
    for field, env_name in ENV_VARS.items():
        status = "설정됨" if getattr(secrets, field) is not None else "-"
        print(f"{env_name:28s} {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
