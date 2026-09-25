"""API 키 등 비밀정보 로더.

값은 환경변수 또는 ``.env`` 에서만 읽고(환경변수가 우선), :class:`pydantic.SecretStr` 로
감싸 ``repr``·로그·예외 메시지에 원문이 노출되지 않게 한다. 누락 점검 결과에도
환경변수 이름만 표시하고 값은 표시하지 않는다.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, SecretStr

#: Secrets 필드명 → 환경변수 이름
ENV_VARS: dict[str, str] = {
    "kis_app_key": "KIS_APP_KEY",
    "kis_app_secret": "KIS_APP_SECRET",
    "kis_account_no": "KIS_ACCOUNT_NO",
    "kis_account_product_code": "KIS_ACCOUNT_PRODUCT_CODE",
    "kis_hts_id": "KIS_HTS_ID",
    "binance_api_key": "BINANCE_API_KEY",
    "binance_api_secret": "BINANCE_API_SECRET",
    "telegram_bot_token": "TELEGRAM_BOT_TOKEN",
    "telegram_chat_id": "TELEGRAM_CHAT_ID",
    "slack_webhook_url": "SLACK_WEBHOOK_URL",
    "fred_api_key": "FRED_API_KEY",
    "ecos_api_key": "ECOS_API_KEY",
}


class MissingSecretError(RuntimeError):
    """필요한 비밀정보가 설정되지 않았을 때 발생. 메시지에는 변수 이름만 담는다."""


class Secrets(BaseModel):
    """비밀정보 묶음. 모든 값은 SecretStr 이며 ``get_secret_value()`` 로만 꺼낼 수 있다."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kis_app_key: SecretStr | None = None
    kis_app_secret: SecretStr | None = None
    kis_account_no: SecretStr | None = None
    kis_account_product_code: SecretStr | None = None
    kis_hts_id: SecretStr | None = None
    binance_api_key: SecretStr | None = None
    binance_api_secret: SecretStr | None = None
    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: SecretStr | None = None
    slack_webhook_url: SecretStr | None = None
    fred_api_key: SecretStr | None = None
    ecos_api_key: SecretStr | None = None

    def missing(self, *fields: str) -> list[str]:
        """지정한 필드 중 비어 있는 것의 환경변수 이름 목록."""
        return [ENV_VARS[name] for name in fields if getattr(self, name) is None]

    def require(self, *fields: str) -> None:
        """지정한 필드가 모두 설정되어 있지 않으면 :class:`MissingSecretError`."""
        missing = self.missing(*fields)
        if missing:
            raise MissingSecretError(f".env 또는 환경변수에 다음 값이 필요합니다: {missing}")


def load_secrets(env_file: str | Path | None = ".env") -> Secrets:
    """환경변수와 ``.env`` 파일에서 비밀정보를 읽는다. 빈 문자열은 미설정으로 본다."""
    file_values: dict[str, str | None] = {}
    if env_file is not None and Path(env_file).is_file():
        file_values = dotenv_values(env_file)
    values: dict[str, str] = {}
    for field, env_name in ENV_VARS.items():
        value = os.environ.get(env_name) or file_values.get(env_name)
        if value:
            values[field] = value
    return Secrets.model_validate(values)
