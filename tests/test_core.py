"""비밀정보 마스킹, UTC 시간 규칙, 도메인 모델 기본 검증."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from rsidiv.__main__ import main
from rsidiv.core.models import AssetClass, OrderPurpose, OrderRequest, OrderType, Side
from rsidiv.core.secrets import MissingSecretError, load_secrets
from rsidiv.core.timeutil import UTC, bar_close_time, ensure_utc, to_display

FAKE_SECRET = "sk-test-DO-NOT-LEAK-1234567890"


def test_secrets_are_masked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
    env = tmp_path / ".env"
    env.write_text(f"BINANCE_API_SECRET={FAKE_SECRET}\nKIS_APP_KEY=\n", encoding="utf-8")
    secrets = load_secrets(env)
    assert secrets.binance_api_secret is not None
    assert secrets.binance_api_secret.get_secret_value() == FAKE_SECRET
    assert FAKE_SECRET not in repr(secrets)
    assert FAKE_SECRET not in str(secrets)
    assert FAKE_SECRET not in secrets.model_dump_json()
    assert secrets.kis_app_key is None  # 빈 문자열은 미설정


def test_environment_overrides_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text("TELEGRAM_CHAT_ID=from-file\n", encoding="utf-8")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "from-env")
    secrets = load_secrets(env)
    assert secrets.telegram_chat_id is not None
    assert secrets.telegram_chat_id.get_secret_value() == "from-env"


def test_require_reports_names_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KIS_APP_KEY", raising=False)
    monkeypatch.delenv("KIS_APP_SECRET", raising=False)
    monkeypatch.setenv("KIS_APP_SECRET", FAKE_SECRET)
    secrets = load_secrets(tmp_path / "missing.env")
    with pytest.raises(MissingSecretError) as info:
        secrets.require("kis_app_key", "kis_app_secret")
    assert "KIS_APP_KEY" in str(info.value)
    assert FAKE_SECRET not in str(info.value)


def test_secrets_cli_prints_names_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("BINANCE_API_KEY", FAKE_SECRET)
    assert main(["secrets", "--env-file", str(tmp_path / "none.env")]) == 0
    out = capsys.readouterr().out
    assert "BINANCE_API_KEY" in out and FAKE_SECRET not in out


def test_config_cli(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["config"]) == 0
    assert "설정 검증 통과" in capsys.readouterr().out


def test_naive_datetime_rejected() -> None:
    with pytest.raises(ValueError, match="naive"):
        ensure_utc(dt.datetime(2025, 1, 1, 9, 0))


def test_kst_display_and_bar_close() -> None:
    bar_open = dt.datetime(2025, 1, 2, 0, 0, tzinfo=UTC)  # 09:00 KST 봉
    assert to_display(bar_open, "Asia/Seoul").hour == 9
    assert bar_close_time(bar_open, 15) == dt.datetime(2025, 1, 2, 0, 15, tzinfo=UTC)


def test_order_request_validation() -> None:
    now = dt.datetime(2025, 1, 2, tzinfo=UTC)
    common = {"client_id": "c1", "symbol": "BTC/USDT", "asset_class": AssetClass.CRYPTO,
              "side": Side.SELL, "qty": 0.01, "purpose": OrderPurpose.STOP_LOSS, "created_at": now}
    with pytest.raises(ValueError, match="stop_price"):
        OrderRequest(order_type=OrderType.STOP_MARKET, **common)  # type: ignore[arg-type]
    req = OrderRequest(order_type=OrderType.STOP_MARKET, stop_price=90000.0, **common)  # type: ignore[arg-type]
    assert req.created_at.tzinfo is UTC
