"""ccxt 기반 바이낸스 USDⓈ-M REST 공급자 (실시간·실거래, 국내 PC 에서의 과거 데이터).

- 봉: ``fapiPublicGetKlines`` 원본 응답(거래대금·체결 수 포함)을 페이지 단위로 받고, 진행 중인 마지막 봉은 버린다.
- 펀딩비: ``fapiPublicGetFundingRate``.
- 거래소 정보: ``fapiPublicGetExchangeInfo`` → 심볼 상태·계약 유형·기초자산 유형(COIN/TRADFI 등)·주문 필터.
- 지역 차단(HTTP 451)은 재시도하지 않고 바로 실패한다. ccxt 가 프록시·CA 환경변수를 따르도록 설정한다.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

import pandas as pd

from perpdiv.core.config import timeframe_minutes
from perpdiv.core.timeutil import ensure_utc, to_epoch_ms, utc_now
from perpdiv.data.base import DataSourceError, empty_ohlcv, ohlcv_from_rows, slice_range
from perpdiv.data.vision import RetryPolicy, empty_funding

KLINE_PAGE_MAX = 1500
FUNDING_PAGE_MAX = 1000


def is_region_blocked(exc: BaseException) -> bool:
    text = str(exc)
    return " 451 " in text or "restricted location" in text


def create_exchange(*, enable_rate_limit: bool, timeout_sec: float, api_key: str | None = None,
                    secret: str | None = None) -> Any:
    import ccxt

    options: dict[str, Any] = {"enableRateLimit": enable_rate_limit, "timeout": int(timeout_sec * 1000),
                               "requests_trust_env": True, "options": {"defaultType": "future"}}
    if api_key and secret:
        options.update(apiKey=api_key, secret=secret)
    return ccxt.binanceusdm(options)


def ccxt_retry_policy(max_retries: int, backoff_sec: float) -> RetryPolicy:
    """ccxt 네트워크 계열 오류(시간 초과·요청 한도·일시 장애)만 재시도한다. 451 은 :class:`RestProvider` 가 먼저 영구
    오류로 바꾼다."""
    import ccxt

    return RetryPolicy(max_retries, backoff_sec, retry_on=(ccxt.NetworkError,))


@dataclass(frozen=True, slots=True)
class SymbolInfo:
    symbol: str
    status: str
    contract_type: str
    underlying_type: str
    base: str
    quote: str
    min_qty: float
    qty_step: float
    min_notional: float
    price_tick: float


class RestProvider:
    def __init__(self, exchange: Any, *, retry: RetryPolicy, page_limit: int = KLINE_PAGE_MAX,
                 clock: Callable[[], dt.datetime] = utc_now) -> None:
        self._ex = exchange
        self._retry = retry
        self._page = min(page_limit, KLINE_PAGE_MAX)
        self._clock = clock

    @property
    def source_name(self) -> str:
        return "binance-rest"

    def _call(self, what: str, fn: Callable[[], Any]) -> Any:
        def guarded() -> Any:
            try:
                return fn()
            except Exception as exc:
                if is_region_blocked(exc):
                    raise DataSourceError(f"{what}: 지역 차단 (HTTP 451): {exc}") from exc
                raise

        return self._retry.call(what, guarded)

    def klines(self, symbol: str, interval: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        """[start, end) 의 마감된 봉 (거래대금·체결 수 포함)."""
        start, end = ensure_utc(start), ensure_utc(end)
        step_ms = timeframe_minutes(interval) * 60_000
        now_ms = to_epoch_ms(self._clock())
        end_ms = min(to_epoch_ms(end), now_ms)
        rows: list[list[float]] = []
        since = to_epoch_ms(start)
        while since < end_ms:
            params = {"symbol": symbol, "interval": interval, "startTime": since, "limit": self._page}
            batch = self._call(f"klines {symbol} {interval}", partial(self._ex.fapiPublicGetKlines, params))
            if not batch:
                break
            rows += [[float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[7]),
                      float(r[8])] for r in batch]
            last = int(batch[-1][0])
            if last < since:
                raise DataSourceError(f"klines {symbol}: 응답 시각이 요청보다 과거 ({last})")
            since = last + step_ms
        closed = [r for r in rows if int(r[0]) + step_ms <= now_ms]
        if not closed:
            return empty_ohlcv()
        return slice_range(ohlcv_from_rows(closed), start, end)

    def funding_rates(self, symbol: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        start, end = ensure_utc(start), ensure_utc(end)
        end_ms = min(to_epoch_ms(end), to_epoch_ms(self._clock()))
        records: dict[int, float] = {}
        since = to_epoch_ms(start)
        while since < end_ms:
            params = {"symbol": symbol, "startTime": since, "limit": FUNDING_PAGE_MAX}
            batch = self._call(f"funding {symbol}", partial(self._ex.fapiPublicGetFundingRate, params))
            if not batch:
                break
            for item in batch:
                records[int(item["fundingTime"])] = float(item["fundingRate"])
            last = int(batch[-1]["fundingTime"])
            if last < since:
                raise DataSourceError(f"funding {symbol}: 응답 시각이 요청보다 과거")
            since = last + 1
        if not records:
            return empty_funding()
        frame = pd.DataFrame({"funding_rate": list(records.values()), "interval_hours": float("nan")},
                             index=pd.to_datetime(list(records), unit="ms", utc=True).rename("time"))
        frame = frame.sort_index()
        upper = pd.Timestamp(end_ms, unit="ms", tz="UTC")  # 미래(아직 정산 전) 값은 받지 않는다
        return frame[(frame.index >= pd.Timestamp(start)) & (frame.index < upper)]

    def exchange_info(self) -> dict[str, SymbolInfo]:
        info = self._call("exchangeInfo", self._ex.fapiPublicGetExchangeInfo)
        return {s["symbol"]: parse_symbol_info(s) for s in info["symbols"]}


def parse_symbol_info(raw: Mapping[str, Any]) -> SymbolInfo:
    filters = {f["filterType"]: f for f in raw.get("filters", [])}
    lot = filters.get("LOT_SIZE", {})
    price = filters.get("PRICE_FILTER", {})
    notional = filters.get("MIN_NOTIONAL", {})
    return SymbolInfo(
        symbol=raw["symbol"], status=raw.get("status", ""), contract_type=raw.get("contractType", ""),
        underlying_type=raw.get("underlyingType", ""), base=raw.get("baseAsset", ""),
        quote=raw.get("marginAsset", raw.get("quoteAsset", "")),
        min_qty=float(lot.get("minQty", 0)), qty_step=float(lot.get("stepSize", 0)),
        min_notional=float(notional.get("notional", notional.get("minNotional", 0))),
        price_tick=float(price.get("tickSize", 0)),
    )
