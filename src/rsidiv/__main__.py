"""명령행 진입점: ``python -m rsidiv <command>``.

- ``config``  : 설정 파일 전체를 검증하고 핵심 값을 요약 출력
- ``secrets`` : .env / 환경변수에 어떤 키가 설정되어 있는지 이름만 출력 (값은 출력하지 않음)
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from pydantic import ValidationError

from rsidiv.core.config import ASSET_CLASSES, Settings, load_settings, resolve_config_dir
from rsidiv.core.secrets import ENV_VARS, load_secrets


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rsidiv")
    sub = parser.add_subparsers(dest="command", required=True)

    config_cmd = sub.add_parser("config", help="설정 검증 및 요약")
    config_cmd.add_argument("--config-dir", default=None)
    config_cmd.add_argument("--override", action="append", default=[], help="덮어쓰기 YAML")

    secrets_cmd = sub.add_parser("secrets", help="비밀정보 설정 여부 (이름만)")
    secrets_cmd.add_argument("--env-file", default=".env")

    args = parser.parse_args(argv)

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
