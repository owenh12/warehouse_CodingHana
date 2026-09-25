"""바이낸스 공통 규칙과 ccxt 기반 REST 공급자.

- 심볼 표기: 설정·내부는 ``BTC/USDT``. 바이낸스 원시 ID 는 ``BTCUSDT``,
  ccxt 무기한 선물 심볼은 ``BTC/USDT:USDT``.
- 봉 시각은 봉 시작 시각(UTC). 응답의 마지막 봉이 아직 마감 전이면 제외한다.
- 일시적 네트워크 오류는 지수 백오프로 재시도하고, 소진되면 :class:`DataSourceError`.
"""

from __future__ import annotations

import datetime as dt
import time
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any, Literal, TypeVar

import pandas as pd

from rsidiv.core.config import SymbolFilter, timeframe_minutes
from rsidiv.core.timeutil import ensure_utc, from_epoch_ms, to_epoch_ms, utc_now
from rsidiv.data.base import (
    DataProvider,
    DataSourceError,
    FundingProvider,
    ohlcv_from_rows,
    slice_range,
)

MarketType = Literal["spot", "usdm_futures"]
T = TypeVar("T")

#: 바이낸스 REST 한 번 요청의 최대 행 수 (현물 klines·선물 fundingRate 모두 1000)
BINANCE_PAGE_MAX = 1000


def exchange_id(symbol: str) -> str:
    """``BTC/USDT`` → ``BTCUSDT``."""
    base, quote = split_symbol(symbol)
    return f"{base}{quote}"


def ccxt_symbol(symbol: str, market_type: MarketType) -> str:
    """설정 심볼 → ccxt 심볼. 무기한 선물은 ``BTC/USDT:USDT``."""
    base, quote = split_symbol(symbol)
    return f"{base}/{quote}" if market_type == "spot" else f"{base}/{quote}:{quote}"


def split_symbol(symbol: str) -> tuple[str, str]:
    parts = symbol.split(":")[0].split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"심볼 형식은 BASE/QUOTE 여야 합니다: {symbol!r}")
    return parts[0], parts[1]


def parse_symbol_filters(raw_filters: Sequence[Mapping[str, Any]]) -> SymbolFilter:
    """바이낸스 exchangeInfo 의 ``filters`` 목록 → :class:`SymbolFilter`.

    현물은 ``NOTIONAL.minNotional`` (구 ``MIN_NOTIONAL.minNotional``),
    선물은 ``MIN_NOTIONAL.notional`` 을 최소 주문금액으로 쓴다.
    """
    by_type = {f.get("filterType"): f for f in raw_filters}
    try:
        lot = by_type["LOT_SIZE"]
        price = by_type["PRICE_FILTER"]
    except KeyError as exc:
        raise DataSourceError(f"거래소 필터 누락: {exc}") from exc
    notional_filter = by_type.get("NOTIONAL") or by_type.get("MIN_NOTIONAL") or {}
    min_notional = notional_filter.get("minNotional", notional_filter.get("notional", 0.0))
    return SymbolFilter(
        min_qty=float(lot["minQty"]),
        qty_step=float(lot["stepSize"]),
        min_notional=float(min_notional),
        price_tick=float(price["tickSize"]),
    )


class RetryPolicy:
    """일시적 오류 재시도: ``max_retries`` 회까지 ``backoff_sec * 2**n`` 초 대기."""

    def __init__(
        self,
        max_retries: int,
        backoff_sec: float,
        retry_on: tuple[type[BaseException], ...],
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.max_retries = max_retries
        self.backoff_sec = backoff_sec
        self.retry_on = retry_on
        self._sleep = sleep

    def call(self, what: str, fn: Callable[[], T]) -> T:
        for attempt in range(self.max_retries + 1):
            try:
                return fn()
            except self.retry_on as exc:
                if attempt == self.max_retries:
                    raise DataSourceError(
                        f"{what}: {self.max_retries + 1}회 시도 실패 ({type(exc).__name__}: {exc})"
                    ) from exc
                self._sleep(self.backoff_sec * (2**attempt))
        raise AssertionError("unreachable")


def create_ccxt_exchange(
    market_type: MarketType, *, enable_rate_limit: bool, timeout_sec: float,
    spot_public_api: str | None = None,
) -> Any:
    """공개 시세용 ccxt 거래소 객체 (API 키 없음). 현물은 현물 시장 정보만 로드한다."""
    import ccxt

    options: dict[str, Any] = {"enableRateLimit": enable_rate_limit, "timeout": int(timeout_sec * 1000)}
    if market_type == "spot":
        exchange = ccxt.binance({**options, "options": {"fetchMarkets": {"types": ["spot"]}}})
        if spot_public_api:
            exchange.urls["api"]["public"] = spot_public_api
        return exchange
    return ccxt.binanceusdm(options)


class BinanceRestProvider(DataProvider, FundingProvider):
    """ccxt 로 바이낸스 REST API 에서 봉·펀딩비·거래소 필터를 조회한다."""

    def __init__(
        self,
        market_type: MarketType,
        exchange: Any,
        *,
        retry: RetryPolicy,
        page_limit: int = BINANCE_PAGE_MAX,
        clock: Callable[[], dt.datetime] = utc_now,
    ) -> None:
        self.market_type = market_type
        self._exchange = exchange
        self._retry = retry
        self._page_limit = min(page_limit, BINANCE_PAGE_MAX)
        self._clock = clock
        self._markets_loaded = False

    @property
    def source_name(self) -> str:
        return "binance-rest"

    def _load_markets(self) -> None:
        if not self._markets_loaded:
            self._retry.call("load_markets", self._exchange.load_markets)
            self._markets_loaded = True

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, start: dt.datetime, end: dt.datetime
    ) -> pd.DataFrame:
        start, end = ensure_utc(start), ensure_utc(end)
        self._load_markets()
        tf_ms = timeframe_minutes(timeframe) * 60_000
        now_ms = to_epoch_ms(self._clock())
        end_ms = min(to_epoch_ms(end), now_ms)
        market_symbol = ccxt_symbol(symbol, self.market_type)
        rows: list[list[float]] = []
        since = to_epoch_ms(start)
        while since < end_ms:
            batch = self._retry.call(
                f"fetch_ohlcv {symbol} {timeframe}",
                partial(
                    self._exchange.fetch_ohlcv, market_symbol, timeframe,
                    since=since, limit=self._page_limit,
                ),
            )
            if not batch:
                break
            rows.extend(batch)
            last_open = int(batch[-1][0])
            if last_open < since:
                raise DataSourceError(f"fetch_ohlcv {symbol}: 응답 시각이 요청보다 과거 ({last_open})")
            since = last_open + tf_ms
        closed = [row for row in rows if int(row[0]) + tf_ms <= now_ms]
        return slice_range(ohlcv_from_rows(closed), start, end)

    def earliest_available(self, symbol: str, timeframe: str) -> dt.datetime | None:
        self._load_markets()
        batch = self._retry.call(
            f"earliest {symbol} {timeframe}",
            lambda: self._exchange.fetch_ohlcv(
                ccxt_symbol(symbol, self.market_type), timeframe, since=0, limit=1
            ),
        )
        return from_epoch_ms(int(batch[0][0])) if batch else None

    def funding_rates(self, symbol: str, start: dt.datetime, end: dt.datetime) -> pd.Series:
        if self.market_type != "usdm_futures":
            raise ValueError("펀딩비는 usdm_futures 시장에서만 조회할 수 있습니다")
        start, end = ensure_utc(start), ensure_utc(end)
        self._load_markets()
        market_symbol = ccxt_symbol(symbol, self.market_type)
        end_ms = min(to_epoch_ms(end), to_epoch_ms(self._clock()))
        records: dict[int, float] = {}
        since = to_epoch_ms(start)
        while since < end_ms:
            batch = self._retry.call(
                f"funding {symbol}",
                partial(
                    self._exchange.fetch_funding_rate_history, market_symbol,
                    since=since, limit=self._page_limit,
                ),
            )
            if not batch:
                break
            for item in batch:
                records[int(item["timestamp"])] = float(item["fundingRate"])
            last = int(batch[-1]["timestamp"])
            if last < since:
                raise DataSourceError(f"funding {symbol}: 응답 시각이 요청보다 과거 ({last})")
            since = last + 1
        return funding_series(records, start, end)

    def symbol_filters(self, symbol: str) -> SymbolFilter:
        """거래소 exchangeInfo 기준 최소 주문수량·수량단위·최소 주문금액·호가단위."""
        self._load_markets()
        market = self._exchange.market(ccxt_symbol(symbol, self.market_type))
        return parse_symbol_filters(market["info"]["filters"])


def funding_series(records: Mapping[int, float], start: dt.datetime, end: dt.datetime) -> pd.Series:
    """{epoch_ms: rate} → [start, end) 구간의 규격 펀딩비 Series."""
    index = pd.to_datetime(pd.Series(sorted(records), dtype="int64"), unit="ms", utc=True)
    series = pd.Series(
        [records[k] for k in sorted(records)],
        index=pd.DatetimeIndex(index).as_unit("ns").rename("time"),
        name="funding_rate",
        dtype="float64",
    )
    mask = (series.index >= pd.Timestamp(start)) & (series.index < pd.Timestamp(end))
    return series[mask]
