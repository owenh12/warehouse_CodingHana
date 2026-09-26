"""비밀정보(.env). 값은 ``SecretStr`` 로 감싸 로그·repr 에 드러나지 않게 한다.

API 키는 **출금 권한이 없는** 키를 쓰고, 가능하면 IP 제한을 건다(README 참고).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, SecretStr

ENV_VARS = {
    "binance_api_key": "BINANCE_API_KEY",
    "binance_api_secret": "BINANCE_API_SECRET",
    "telegram_bot_token": "TELEGRAM_BOT_TOKEN",
    "telegram_chat_id": "TELEGRAM_CHAT_ID",
}


class Secrets(BaseModel):
    model_config = ConfigDict(frozen=True)

    binance_api_key: SecretStr | None = None
    binance_api_secret: SecretStr | None = None
    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: SecretStr | None = None


def load_secrets(env_file: str | Path | None = ".env") -> Secrets:
    """환경변수가 .env 파일보다 우선한다. 파일이 없어도 오류가 아니다."""
    file_values = dotenv_values(env_file) if env_file and Path(env_file).is_file() else {}
    values: dict[str, str] = {}
    for field, env_name in ENV_VARS.items():
        value = os.environ.get(env_name) or file_values.get(env_name)
        if value:
            values[field] = value
    return Secrets.model_validate(values)
