"""데이터 계층 공통 인터페이스.

모든 시세 구현체는 다음 형식의 DataFrame 을 반환한다 (:func:`normalize_ohlcv` 로 강제).

- index: UTC tz-aware ``DatetimeIndex`` (이름 ``time``, 봉 시작 시각 = left label), 오름차순, 중복 없음
- columns: :data:`OHLCV_COLUMNS` (float64)
- 진행 중(미확정)인 봉은 포함하지 않는다.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from collections.abc import Sequence

import pandas as pd

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


class DataSourceError(RuntimeError):
    """데이터 소스 접근 실패 (네트워크 차단, 재시도 소진, 응답 형식 오류 등)."""


def empty_ohlcv() -> pd.DataFrame:
    """규격에 맞는 빈 OHLCV 프레임."""
    index = pd.DatetimeIndex([], tz="UTC", name="time").as_unit("ns")
    return pd.DataFrame({col: pd.Series(dtype="float64") for col in OHLCV_COLUMNS}, index=index)


def empty_funding() -> pd.Series:
    """규격에 맞는 빈 펀딩비 Series."""
    index = pd.DatetimeIndex([], tz="UTC", name="time").as_unit("ns")
    return pd.Series([], index=index, name="funding_rate", dtype="float64")


def ohlcv_from_rows(rows: Sequence[Sequence[float]]) -> pd.DataFrame:
    """``[epoch_ms, open, high, low, close, volume]`` 행 목록(ccxt 형식) → 규격 프레임."""
    if not rows:
        return empty_ohlcv()
    frame = pd.DataFrame([list(row[:6]) for row in rows], columns=["time", *OHLCV_COLUMNS])
    frame["time"] = pd.to_datetime(frame["time"].astype("int64"), unit="ms", utc=True)
    return normalize_ohlcv(frame.set_index("time"))


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """인덱스를 UTC·ns·오름차순으로 맞추고 중복 시각은 마지막 값을 남긴다."""
    missing = [col for col in OHLCV_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"OHLCV 컬럼 누락: {missing}")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("OHLCV 인덱스는 tz-aware DatetimeIndex 여야 합니다")
    out = frame.loc[:, list(OHLCV_COLUMNS)].astype("float64")
    out.index = pd.DatetimeIndex(out.index).tz_convert("UTC").as_unit("ns").rename("time")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def slice_range(frame: pd.DataFrame, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    """[start, end) 구간 (봉 시작 시각 기준)."""
    index = frame.index
    return frame[(index >= pd.Timestamp(start)) & (index < pd.Timestamp(end))]


class DataProvider(ABC):
    """시세 공급자 (바이낸스 REST, 바이낸스 아카이브, KIS 등)."""

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
        """정산 시각(UTC, 이름 ``time``) → 펀딩비율(이름 ``funding_rate``). 양수면 롱이 지불.

        [start, end) 구간의 정산 시각만 포함한다.
        """
