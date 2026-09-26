"""설정 스키마: 기본값 로드, 오타·잘못된 값·파일 간 불일치 거부, 덮어쓰기."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from perpdiv.core.config import DEFAULT_CONFIG_DIR, deep_merge, load_settings, timeframe_minutes
from perpdiv.core.secrets import load_secrets


def config_copy(tmp_path: Path, patch: dict[str, Any] | None = None) -> Path:
    target = tmp_path / "config"
    shutil.copytree(DEFAULT_CONFIG_DIR, target)
    for name, update in (patch or {}).items():
        path = target / f"{name}.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        path.write_text(yaml.safe_dump(deep_merge(data, update), allow_unicode=True), encoding="utf-8")
    return target


def test_defaults_load() -> None:
    s = load_settings()
    assert s.universe.top_n == 10 and s.base.exchange.leverage == 1
    assert s.data.timeframes.signal == ["15m", "1h", "4h", "1d"]
    assert s.strategy.pivot.left == 1 and s.strategy.pivot.right == 1
    assert s.strategy.exit.time_exit.bars == 60 and s.strategy.exit.stop.atr_mult == 2.5
    assert s.costs.fees.rate("maker") == 0.0002 and s.costs.fees.rate("taker") == 0.0005


def test_timeframe_minutes() -> None:
    assert [timeframe_minutes(t) for t in ("1m", "5m", "15m", "1h", "4h", "1d")] == [1, 5, 15, 60, 240, 1440]
    with pytest.raises(ValueError):
        timeframe_minutes("15")


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"strategy": {"pivot": {"lefft": 2}}}, "lefft"),
        ({"risk": {"daily_loss": {"reset_time": 0}}}, "따옴표"),
        ({"data": {"timeframes": {"signal": ["15m", "3m"]}}}, "정수배"),
        ({"data": {"timeframes": {"signal": ["15m", "7m"]}}}, "나누어떨어지게"),
        ({"data": {"timeframes": {"execution": "1h"}}}, "실행 봉"),
        ({"strategy": {"bullish": {"oversold": 75.0}}}, "oversold"),
        ({"base": {"exchange": {"leverage": 3}}, "risk": {"liquidation": {"enabled": False}}}, "레버리지"),
        ({"risk": {"kill_switch": {"auto_resume": True}}}, "auto_resume"),
        ({"live": {"mode": "live"}}, "live_confirm"),
        ({"optimize": {"search_space": {"pivot.middle": {"type": "int", "low": 1, "high": 2}}}}, "pivot.middle"),
        ({"universe": {"ranking": {"bar_timeframe": "15m"}}}, "bar_timeframe"),
        ({"universe": {"candidates": {"hourly_rank_margin": -1}}}, "hourly_rank_margin"),
        ({"universe": {"exclude": {"reviewed_not_tradfi": ["XAU"]}}}, "reviewed_not_tradfi"),
        ({"strategy": {"structure": {"gap_bars": {"min_enabled": True, "min": 80}}}}, "min < max"),
        ({"risk": {"positions": {"max_concurrent": 2}}}, "max_concurrent"),
    ],
)
def test_invalid_configs_rejected(tmp_path: Path, patch: dict[str, Any], message: str) -> None:
    with pytest.raises((ValidationError, ValueError), match=message):
        load_settings(config_copy(tmp_path, patch))


def test_override_and_strategy_with(tmp_path: Path) -> None:
    override = tmp_path / "exp.yaml"
    override.write_text("strategy:\n  pivot:\n    left: 3\n", encoding="utf-8")
    s = load_settings(overrides=[override])
    assert s.strategy.pivot.left == 3
    changed = s.strategy_with({"exit.stop.atr_mult": 1.5, "pivot.tie_rule": "first"})
    assert changed.exit.stop.atr_mult == 1.5 and changed.pivot.tie_rule == "first"
    with pytest.raises(KeyError):
        s.strategy_with({"pivot.middle": 1})
    bad = tmp_path / "bad.yaml"
    bad.write_text("strategyy: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="strategyy"):
        load_settings(overrides=[bad])


def test_secrets_are_masked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text("BINANCE_API_KEY=abc123\nBINANCE_API_SECRET=shh\n", encoding="utf-8")
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    secrets = load_secrets(env)
    assert secrets.binance_api_key is not None and secrets.binance_api_key.get_secret_value() == "abc123"
    assert "abc123" not in repr(secrets) and "shh" not in str(secrets)
    assert secrets.telegram_chat_id is not None and secrets.telegram_bot_token is None
