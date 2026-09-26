"""명령행 진입점: ``python -m perpdiv <command>``.

- ``config``  : 설정 9개 파일 검증 + 요약
- ``secrets`` : .env 키 설정 여부 (이름만, 값은 출력하지 않음)
- ``data-check``: 데이터 확보 점검 (심볼·순위·후보군·TradFi·용량·펀딩비·상장폐지) → Markdown 보고서
- ``resample-check``: 5분봉 리샘플 결과를 아카이브 원본 15m/1h/4h/1d 와 비교 → CSV
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from pydantic import ValidationError

from perpdiv.core.config import Settings, load_settings, resolve_config_dir
from perpdiv.core.secrets import ENV_VARS, load_secrets


def summarize(s: Settings) -> str:
    st, ex, tfs = s.strategy, s.base.exchange, s.data.timeframes
    tp = st.exit.take_profit
    lines = [
        f"계좌 {s.base.account.initial_capital:,.0f} {s.base.account.quote_asset}, {ex.venue} "
        f"{ex.position_mode}/{ex.margin_mode}, 레버리지 {ex.leverage}배",
        f"유니버스: 거래대금 상위 {s.universe.top_n} 코인 (직전 {s.universe.ranking.window_hours}h, "
        f"{'+'.join(s.universe.ranking.rank_quote_assets)} 합산, 주문 {s.universe.trading.quote_asset})",
        f"데이터: 수집 {tfs.collect}, 신호 {tfs.signal}, 실행 {tfs.execution}, 기간 {s.data.backtest_period.start} ~ "
        f"{s.data.backtest_period.end or '현재'}",
        f"피벗 L={st.pivot.left} R={st.pivot.right} ({st.pivot.tie_rule}), RSI {st.rsi.period}, "
        f"강세 <{st.bullish.oversold:g}·약세 >{st.bearish.overbought:g}, 최대 간격 "
        f"{st.structure.gap_bars.max if st.structure.gap_bars.max_enabled else '-'}봉",
        f"청산: 손절 ATR×{st.exit.stop.atr_mult:g}, 익절 {tp.mode}"
        + (f" {tp.r_multiple:g}R" if tp.mode == "r_multiple" else "")
        + f", 시간 {st.exit.time_exit.bars}봉(신호 TF), 판정 {st.exit.evaluation_timeframe}",
        f"리스크: 진입 {s.risk.sizing.equity_fraction:.0%}, 동시 {s.risk.positions.max_concurrent}개, 일일손실 "
        f"{s.risk.daily_loss.threshold:.0%}, MDD 킬스위치 {s.risk.kill_switch.max_drawdown:.0%} "
        f"({s.risk.kill_switch.on_trip}), API 오류 {s.risk.api_errors.max_consecutive}회",
        f"비용: 메이커 {s.costs.fees.maker:.3%}·테이커 {s.costs.fees.taker:.3%}, 슬리피지 "
        f"{s.costs.slippage.taker_pct:.3%}, 펀딩비 {'반영' if s.costs.funding.enabled else '미반영'}",
        f"실행 모드: {s.live.mode}",
    ]
    return "\n".join(lines)


def _data_check(args: argparse.Namespace) -> int:
    import datetime as dt

    from perpdiv.core.config import resolve_project_path
    from perpdiv.core.timeutil import utc_now
    from perpdiv.data.check import render, run_data_check

    settings = load_settings(args.config_dir)
    now = utc_now()
    samples = (args.sample_month or f"{(now.replace(day=1) - dt.timedelta(days=1)):%Y-%m}").split(",")
    check = run_data_check(settings, sample_months=samples, workers=args.workers,
                           log=lambda m: print(m, flush=True))
    text = render(check, settings)
    out_dir = resolve_project_path(settings.base.paths.report_dir) / "data_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"data_check_{now:%Y%m%dT%H%M%SZ}.md"
    path.write_text(text, encoding="utf-8")
    check.tradfi_detected.to_csv(out_dir / "tradfi_detected.csv")
    (out_dir / "candidates.txt").write_text("\n".join(check.candidates) + "\n", encoding="utf-8")
    print(text)
    print(f"보고서: {path}")
    return 0


def _resample_check(args: argparse.Namespace) -> int:
    import datetime as dt

    from perpdiv.core.config import resolve_project_path
    from perpdiv.core.timeutil import UTC
    from perpdiv.data.check import build_archive, build_cache
    from perpdiv.data.validate import check_resample, to_frame

    settings = load_settings(args.config_dir)
    tfs = settings.data.timeframes
    months = [dt.datetime.strptime(m, "%Y-%m").replace(tzinfo=UTC) for m in args.months.split(",")]
    cache = build_cache(settings, build_archive(settings))
    table = to_frame(check_resample(cache, args.symbols.split(","), months, tfs.signal, tfs.collect))
    out_dir = resolve_project_path(settings.base.paths.report_dir) / "data_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "resample_check.csv", index=False)
    print(table.to_string(index=False))
    print(f"일치 {int(table['ok'].sum())}/{len(table)} → {out_dir / 'resample_check.csv'}")
    return 0 if bool(table["ok"].all()) else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="perpdiv")
    sub = parser.add_subparsers(dest="command", required=True)
    cfg = sub.add_parser("config", help="설정 검증 및 요약")
    cfg.add_argument("--config-dir", default=None)
    cfg.add_argument("--override", action="append", default=[])
    sec = sub.add_parser("secrets", help=".env 키 설정 여부 (이름만)")
    sec.add_argument("--env-file", default=".env")
    chk = sub.add_parser("data-check", help="데이터 확보 점검")
    chk.add_argument("--config-dir", default=None)
    chk.add_argument("--sample-month", default=None,
                     help="1시간 순위 근사 검증용 월 YYYY-MM[,YYYY-MM...] (기본: 지난달)")
    chk.add_argument("--workers", type=int, default=32)
    rs = sub.add_parser("resample-check", help="리샘플 결과를 원본 봉과 비교")
    rs.add_argument("--config-dir", default=None)
    rs.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT,1000PEPEUSDT")
    rs.add_argument("--months", default="2024-03,2025-06", help="YYYY-MM,YYYY-MM,...")
    args = parser.parse_args(argv)

    if args.command == "data-check":
        return _data_check(args)
    if args.command == "resample-check":
        return _resample_check(args)

    if args.command == "config":
        try:
            settings = load_settings(args.config_dir, args.override)
        except (ValidationError, ValueError) as exc:
            print(f"설정 검증 실패 ({resolve_config_dir(args.config_dir)}):\n{exc}", file=sys.stderr)
            return 1
        print(f"설정 검증 통과: {resolve_config_dir(args.config_dir)}")
        print(summarize(settings))
        return 0

    secrets = load_secrets(args.env_file)
    for field, env_name in ENV_VARS.items():
        print(f"{env_name:22s} {'설정됨' if getattr(secrets, field) is not None else '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
