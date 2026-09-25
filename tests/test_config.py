"""설정 스키마·로더 테스트: 저장소의 config/*.yaml 이 유효하고, 잘못된 값은 거부되는지."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from rsidiv.core.config import (
    CONFIG_FILES,
    DEFAULT_CONFIG_DIR,
    Settings,
    apply_dotted,
    deep_merge,
    load_settings,
    timeframe_minutes,
)


@pytest.fixture(scope="module")
def raw_config() -> dict[str, Any]:
    return {
        key: yaml.safe_load((DEFAULT_CONFIG_DIR / name).read_text(encoding="utf-8"))
        for key, name in CONFIG_FILES.items()
    }


def _with(raw: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    return deep_merge(raw, patch)


def test_repository_config_is_valid() -> None:
    settings = load_settings()
    assert settings.data.timeframe == "15m"
    assert settings.base.capital.initial == 1000.0
    assert settings.risk.daily_loss.threshold == 0.03
    assert settings.risk.kill_switch.max_drawdown == 0.30
    assert settings.risk.api_errors.max_consecutive == 5


def test_user_specified_defaults(raw_config: dict[str, Any]) -> None:
    """요구사항에 명시된 기본값이 그대로 들어 있는지."""
    s = Settings.model_validate(raw_config)
    p = s.strategy.default
    assert (p.pivot.left, p.pivot.right, p.pivot.price_source) == (5, 3, "low")
    assert p.rsi.period == 14 and p.rsi.method == "wilder"
    assert (p.filters.gap_bars.min_bars, p.filters.gap_bars.max_bars) == (5, 60)
    assert p.filters.rsi_t1_oversold.threshold == 30.0
    assert p.filters.rsi_diff_min.min_diff == 2.0
    assert p.filters.price_drop_min.min_pct == 0.001
    assert p.filters.session_kr.exclude_first_bars == 2
    assert p.exit.overnight_kr.flatten_minutes_before_close == 30
    assert s.risk.sizing.risk_per_trade == 0.01
    assert s.base.capital.allocation == {"stock_kr": 0.5, "crypto": 0.5}
    assert s.costs.stock_kr.commission.rate == pytest.approx(0.000140527)
    assert s.costs.stock_kr.sell_tax.rate_on(dt.date(2026, 1, 2)) == 0.0020
    assert s.costs.crypto.spot.rate("taker") == 0.0010
    assert s.costs.crypto.usdm_futures.rate("maker") == 0.0002
    assert s.costs.crypto.slippage.taker_pct == 0.0002
    assert s.costs.scenarios.slippage_multipliers == [0, 1, 2]
    assert s.optimize.monte_carlo.n_sims == 1000
    assert (s.optimize.walk_forward.train, s.optimize.walk_forward.test) == ("12M", "6M")


def test_unknown_key_is_rejected(raw_config: dict[str, Any]) -> None:
    bad = _with(raw_config, {"strategy": {"default": {"pivot": {"lefft": 4}}}})
    with pytest.raises(ValidationError, match="lefft"):
        Settings.model_validate(bad)


def test_allocation_must_sum_to_one(raw_config: dict[str, Any]) -> None:
    bad = _with(raw_config, {"base": {"capital": {"allocation": {"stock_kr": 0.6, "crypto": 0.5}}}})
    with pytest.raises(ValidationError, match="합계"):
        Settings.model_validate(bad)


def test_volume_window_cannot_look_ahead(raw_config: dict[str, Any]) -> None:
    bad = _with(
        raw_config, {"strategy": {"default": {"filters": {"volume": {"bars_after": 4}}}}}
    )
    with pytest.raises(ValidationError, match="미래참조"):
        Settings.model_validate(bad)


def test_gap_bars_order(raw_config: dict[str, Any]) -> None:
    bad = _with(
        raw_config,
        {"strategy": {"default": {"filters": {"gap_bars": {"min_bars": 60, "max_bars": 5}}}}},
    )
    with pytest.raises(ValidationError, match="min_bars"):
        Settings.model_validate(bad)


def test_kill_switch_cannot_auto_resume(raw_config: dict[str, Any]) -> None:
    bad = _with(raw_config, {"risk": {"kill_switch": {"auto_resume": True}}})
    with pytest.raises(ValidationError, match="수동 해제"):
        Settings.model_validate(bad)


def test_live_mode_requires_confirmation(raw_config: dict[str, Any]) -> None:
    bad = _with(raw_config, {"live": {"mode": "live"}})
    with pytest.raises(ValidationError, match="live_confirm"):
        Settings.model_validate(bad)
    ok = _with(raw_config, {"live": {"mode": "live", "live_confirm": True}})
    assert Settings.model_validate(ok).live.mode == "live"


def test_unquoted_clock_time_is_rejected(raw_config: dict[str, Any]) -> None:
    # YAML 1.1 에서 따옴표 없는 15:30 은 60진수 정수 930 으로 읽힌다
    unquoted = yaml.safe_load("regular_close: 15:30")["regular_close"]
    assert unquoted == 930
    bad = _with(raw_config, {"markets": {"krx": {"regular_close": unquoted}}})
    with pytest.raises(ValidationError, match="HH:MM"):
        Settings.model_validate(bad)


def test_daily_loss_reset_rules(raw_config: dict[str, Any]) -> None:
    s = Settings.model_validate(raw_config)
    crypto_reset = s.risk.daily_loss.reset["crypto"]
    assert crypto_reset.type == "clock"
    assert crypto_reset.time == dt.time(0, 0) and crypto_reset.timezone == "Asia/Seoul"
    assert s.risk.daily_loss.reset["stock_kr"].type == "market_open"

    no_clock = _with(raw_config, {"risk": {"daily_loss": {"reset": {"stock_kr": {"type": "clock"}}}}})
    with pytest.raises(ValidationError, match="time 과 timezone"):
        Settings.model_validate(no_clock)
    stray_clock = _with(
        raw_config, {"risk": {"daily_loss": {"reset": {"crypto": {"type": "market_open"}}}}}
    )
    with pytest.raises(ValidationError, match="market_open"):
        Settings.model_validate(stray_clock)


def test_htf_must_be_multiple_of_base_timeframe(raw_config: dict[str, Any]) -> None:
    bad = _with(raw_config, {"strategy": {"default": {"filters": {"trend": {"htf": "10m"}}}}})
    with pytest.raises(ValidationError, match="htf"):
        Settings.model_validate(bad)


def test_search_space_keys_must_exist(raw_config: dict[str, Any]) -> None:
    bad = _with(
        raw_config,
        {"optimize": {"search_space": {"pivot.lft": {"type": "int", "low": 1, "high": 3}}}},
    )
    with pytest.raises(ValidationError, match="pivot.lft"):
        Settings.model_validate(bad)


def test_search_space_values_must_be_valid(raw_config: dict[str, Any]) -> None:
    # threshold 는 (0, 100) 범위여야 하므로 high=120 은 적용 불가
    bad = _with(
        raw_config,
        {
            "optimize": {
                "search_space": {
                    "filters.rsi_t1_oversold.threshold": {"type": "float", "low": 20, "high": 120}
                }
            }
        },
    )
    with pytest.raises(ValidationError, match="적용 불가"):
        Settings.model_validate(bad)


def test_crypto_symbols_need_exchange_filters(raw_config: dict[str, Any]) -> None:
    bad = _with(raw_config, {"universe": {"crypto": {"symbols": ["BTC/USDT", "SOL/USDT"]}}})
    with pytest.raises(ValidationError, match="SOL/USDT"):
        Settings.model_validate(bad)


def test_strategy_override_merges_per_asset_class(raw_config: dict[str, Any]) -> None:
    patched = _with(
        raw_config, {"strategy": {"overrides": {"crypto": {"pivot": {"left": 7}}}}}
    )
    s = Settings.model_validate(patched)
    assert s.strategy.for_asset_class("crypto").pivot.left == 7
    assert s.strategy.for_asset_class("stock_kr").pivot.left == 5
    assert s.strategy.for_asset_class("crypto").pivot.right == 3


def test_invalid_override_is_rejected(raw_config: dict[str, Any]) -> None:
    bad = _with(raw_config, {"strategy": {"overrides": {"crypto": {"pivot": {"left": 0}}}}})
    with pytest.raises(ValidationError):
        Settings.model_validate(bad)


def test_apply_dotted_returns_validated_copy(raw_config: dict[str, Any]) -> None:
    params = Settings.model_validate(raw_config).strategy.default
    changed = apply_dotted(params, {"pivot.left": 4, "exit.take_profit.r_multiple": 3.0})
    assert changed.pivot.left == 4 and changed.exit.take_profit.r_multiple == 3.0
    assert params.pivot.left == 5  # 원본 불변
    with pytest.raises(KeyError):
        apply_dotted(params, {"pivot.nope": 1})


def test_override_file(tmp_path: Path) -> None:
    override = tmp_path / "exp.yaml"
    override.write_text("strategy:\n  default:\n    entry:\n      mode: B\n", encoding="utf-8")
    assert load_settings(overrides=[override]).strategy.default.entry.mode == "B"

    typo = tmp_path / "typo.yaml"
    typo.write_text("strategi: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="strategi"):
        load_settings(overrides=[typo])


@pytest.mark.parametrize(
    ("price", "tick"),
    [(1999, 1), (2000, 5), (4999, 5), (19990, 10), (49950, 50), (50000, 100),
     (199900, 100), (200000, 500), (499500, 500), (500000, 1000), (1_200_000, 1000)],
)
def test_krx_tick_table(price: float, tick: float) -> None:
    assert load_settings().markets.krx.tick_size(price) == tick


def test_sell_tax_schedule_by_date(raw_config: dict[str, Any]) -> None:
    patched = _with(
        raw_config,
        {
            "costs": {
                "stock_kr": {
                    "sell_tax": {
                        "schedule": [
                            {"effective_from": dt.date(2025, 1, 1), "rate": 0.0015},
                            {"effective_from": dt.date(2026, 1, 1), "rate": 0.0020},
                        ]
                    }
                }
            }
        },
    )
    tax = Settings.model_validate(patched).costs.stock_kr.sell_tax
    assert tax.rate_on(dt.date(2025, 6, 30)) == 0.0015
    assert tax.rate_on(dt.date(2026, 1, 1)) == 0.0020
    with pytest.raises(ValueError):
        tax.rate_on(dt.date(2024, 12, 31))


def test_bnb_discount(raw_config: dict[str, Any]) -> None:
    patched = _with(
        raw_config,
        {"costs": {"crypto": {"spot": {"bnb_discount": {"enabled": True}},
                              "usdm_futures": {"bnb_discount": {"enabled": True}}}}},
    )
    costs = Settings.model_validate(patched).costs.crypto
    assert costs.spot.rate("taker") == pytest.approx(0.00075)
    assert costs.usdm_futures.rate("taker") == pytest.approx(0.00045)


@pytest.mark.parametrize(("tf", "minutes"), [("1m", 1), ("15m", 15), ("4h", 240), ("1d", 1440)])
def test_timeframe_minutes(tf: str, minutes: int) -> None:
    assert timeframe_minutes(tf) == minutes


@pytest.mark.parametrize("tf", ["0m", "15", "15min", "h4"])
def test_timeframe_minutes_rejects_bad_format(tf: str) -> None:
    with pytest.raises(ValueError):
        timeframe_minutes(tf)
