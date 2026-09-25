"""데이터 계층 공통 인터페이스.

모든 시세 구현체는 다음 형식의 DataFrame 을 반환한다.

- index: UTC tz-aware ``DatetimeIndex`` (봉 시작 시각, left label), 오름차순, 중복 없음
- columns: :data:`OHLCV_COLUMNS`
- 진행 중(미확정)인 봉은 포함하지 않는다.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod

import pandas as pd

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


class DataProvider(ABC):
    """시세 공급자 (KIS, ccxt/바이낸스)."""

    @abstractmethod
    def fetch_ohlcv(
        self, symbol: str, timeframe: str, start: dt.datetime, end: dt.datetime
    ) -> pd.DataFrame:
        """[start, end) 구간의 확정 봉을 반환한다. 시각 인자는 UTC tz-aware."""

    @abstractmethod
    def earliest_available(self, symbol: str, timeframe: str) -> dt.datetime | None:
        """조회 가능한 가장 이른 봉 시각 (데이터 확보 가능 여부 점검용). 없으면 None."""


class UniverseProvider(ABC):
    """시점별 유니버스 (KOSPI200 구성종목 변경 이력)."""

    @abstractmethod
    def members(self, as_of: dt.date) -> list[str]:
        """해당 일자에 실제 편입되어 있던 종목코드."""

    @abstractmethod
    def membership_history(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """편입 기간 이력. columns: symbol, name, start_date, end_date(NaT=현재 편입), delisted."""


class FxProvider(ABC):
    """환율 공급자."""

    @abstractmethod
    def daily_rates(self, start: dt.date, end: dt.date) -> pd.Series:
        """1 USD 당 KRW. index 는 날짜, 결측일은 설정(fx.fill_method)에 따라 채운 값."""


class FundingProvider(ABC):
    """무기한 선물 펀딩비 이력."""

    @abstractmethod
    def funding_rates(self, symbol: str, start: dt.datetime, end: dt.datetime) -> pd.Series:
        """정산 시각(UTC) → 펀딩비율. 양수면 롱이 지불."""
