"""바이낸스 REST(ccxt) 공급자와 아카이브(data.binance.vision) 공급자."""

from __future__ import annotations

import datetime as dt

import ccxt
import pandas as pd
import pytest
from fakes import (
    FUTURES_FILTERS,
    TF_MS,
    FakeExchange,
    FakeVisionSession,
    ms,
    synthetic_bar,
    utc,
)

from rsidiv.core.config import SymbolFilter
from rsidiv.data.base import OHLCV_COLUMNS, DataSourceError
from rsidiv.data.binance import (
    BinanceRestProvider,
    RetryPolicy,
    ccxt_symbol,
    exchange_id,
    is_region_blocked,
    parse_symbol_filters,
)
from rsidiv.data.binance_vision import BinanceVisionProvider, parse_kline_csv


def rest(exchange: FakeExchange, now: dt.datetime, market: str = "spot", page: int = 100,
         retries: int = 3, sleeps: list[float] | None = None) -> BinanceRestProvider:
    retry = RetryPolicy(retries, 1.0, (ccxt.NetworkError,), give_up=is_region_blocked,
                        sleep=(sleeps.append if sleeps is not None else lambda s: None))
    return BinanceRestProvider(market, exchange, retry=retry, page_limit=page, clock=lambda: now)  # type: ignore[arg-type]


def test_symbol_mapping() -> None:
    assert exchange_id("BTC/USDT") == "BTCUSDT"
    assert ccxt_symbol("ETH/USDT", "spot") == "ETH/USDT"
    assert ccxt_symbol("ETH/USDT", "usdm_futures") == "ETH/USDT:USDT"
    with pytest.raises(ValueError):
        exchange_id("BTCUSDT")


def test_fetch_paginates_and_drops_unclosed_bar() -> None:
    now = utc(2025, 1, 15, 10, 7)  # 10:00 봉은 진행 중
    ex = FakeExchange(utc(2024, 12, 1), now)
    frame = rest(ex, now).fetch_ohlcv("BTC/USDT", "15m", utc(2025, 1, 1), utc(2025, 2, 1))
    assert list(frame.columns) == list(OHLCV_COLUMNS)
    assert str(frame.index.tz) == "UTC" and frame.index.name == "time"
    assert frame.index[0] == pd.Timestamp("2025-01-01 00:00", tz="UTC")
    assert frame.index[-1] == pd.Timestamp("2025-01-15 09:45", tz="UTC")
    assert len(frame) == 14 * 96 + 40
    assert frame.index.is_monotonic_increasing and not frame.index.duplicated().any()
    assert len(ex.ohlcv_calls) > 10 and all(limit == 100 for _, _, limit in ex.ohlcv_calls)
    row = frame.loc[pd.Timestamp("2025-01-03 12:15", tz="UTC")]
    expected = synthetic_bar(ms(utc(2025, 1, 3, 12, 15)))
    assert row.tolist() == expected[1:]


def test_fetch_respects_half_open_range_and_gaps() -> None:
    now = utc(2025, 3, 1)
    gap = {utc(2025, 1, 2, 3, 0) + dt.timedelta(minutes=15 * i) for i in range(8)}
    ex = FakeExchange(utc(2025, 1, 1), now, missing=gap)
    frame = rest(ex, now).fetch_ohlcv("BTC/USDT", "15m", utc(2025, 1, 2), utc(2025, 1, 3))
    assert frame.index[0] == pd.Timestamp("2025-01-02", tz="UTC")
    assert frame.index[-1] == pd.Timestamp("2025-01-02 23:45", tz="UTC")
    assert len(frame) == 96 - 8
    assert not frame.index.isin([pd.Timestamp(t) for t in gap]).any()


def test_futures_uses_ccxt_futures_symbol() -> None:
    now = utc(2025, 1, 2)
    ex = FakeExchange(utc(2025, 1, 1), now, filters=FUTURES_FILTERS)
    provider = rest(ex, now, market="usdm_futures")
    provider.fetch_ohlcv("BTC/USDT", "15m", utc(2025, 1, 1), utc(2025, 1, 1, 1))
    assert ex.ohlcv_calls[0][0] == "BTC/USDT:USDT"
    assert provider.symbol_filters("BTC/USDT") == SymbolFilter(
        min_qty=0.001, qty_step=0.001, min_notional=100.0, price_tick=0.1
    )
    assert ex.market_symbols == ["BTC/USDT:USDT"]


def test_retry_then_success_with_backoff() -> None:
    now = utc(2025, 1, 2)
    sleeps: list[float] = []
    ex = FakeExchange(utc(2025, 1, 1), now, fail_times=2)
    frame = rest(ex, now, sleeps=sleeps).fetch_ohlcv("BTC/USDT", "15m", utc(2025, 1, 1), utc(2025, 1, 1, 1))
    assert len(frame) == 4
    assert sleeps == [1.0, 2.0]


def test_retry_exhausted_raises_data_source_error() -> None:
    now = utc(2025, 1, 2)
    ex = FakeExchange(utc(2025, 1, 1), now, fail_times=10)
    with pytest.raises(DataSourceError, match="4회 시도 실패"):
        rest(ex, now).fetch_ohlcv("BTC/USDT", "15m", utc(2025, 1, 1), utc(2025, 1, 2))


def test_region_block_is_not_retried() -> None:
    # ccxt 가 HTTP 451 을 던지는 형식: "<id> <method> <url> 451 <reason> <body>"
    blocked = ccxt.ExchangeNotAvailable(
        'binance GET https://api.binance.com/api/v3/exchangeInfo 451  '
        '{"code": 0, "msg": "Service unavailable from a restricted location ..."}'
    )
    assert isinstance(blocked, ccxt.NetworkError) and is_region_blocked(blocked)
    assert not is_region_blocked(ccxt.NetworkError("binance GET https://api.binance.com/api/v3/ping"))
    now = utc(2025, 1, 2)
    sleeps: list[float] = []
    ex = FakeExchange(utc(2025, 1, 1), now, fail_times=10, fail_with=blocked)
    with pytest.raises(DataSourceError, match=r"재시도하지 않음.*451"):
        rest(ex, now, sleeps=sleeps).symbol_filters("BTC/USDT")
    assert sleeps == [] and ex.fail_times == 9


def test_earliest_available() -> None:
    now = utc(2025, 1, 2)
    ex = FakeExchange(utc(2017, 8, 17, 4), now)
    assert rest(ex, now).earliest_available("BTC/USDT", "15m") == utc(2017, 8, 17, 4)


def test_funding_rates_paginated_and_sliced() -> None:
    now = utc(2025, 3, 1)
    times = [ms(utc(2025, 1, 1)) + 8 * 3600_000 * i for i in range(150)]
    funding = {t: (0.0001 if i % 3 else -0.00005) for i, t in enumerate(times)}
    ex = FakeExchange(utc(2025, 1, 1), now, funding=funding, filters=FUTURES_FILTERS)
    series = rest(ex, now, market="usdm_futures", page=40).funding_rates(
        "BTC/USDT", utc(2025, 1, 5), utc(2025, 2, 1)
    )
    assert series.name == "funding_rate" and series.index.name == "time"
    assert series.index[0] == pd.Timestamp("2025-01-05", tz="UTC")
    assert series.index[-1] == pd.Timestamp("2025-01-31 16:00", tz="UTC")
    assert len(series) == 27 * 3
    assert len(ex.funding_calls) >= 3
    assert (series.index.to_series().diff().dropna() == pd.Timedelta(hours=8)).all()


def test_funding_rejected_for_spot() -> None:
    now = utc(2025, 1, 2)
    with pytest.raises(ValueError):
        rest(FakeExchange(utc(2025, 1, 1), now), now).funding_rates("BTC/USDT", utc(2025, 1, 1), now)


@pytest.mark.parametrize(
    ("filters", "expected_notional"),
    [
        ([{"filterType": "NOTIONAL", "minNotional": "5"}], 5.0),
        ([{"filterType": "MIN_NOTIONAL", "minNotional": "10"}], 10.0),  # 구 현물 형식
        ([{"filterType": "MIN_NOTIONAL", "notional": "20"}], 20.0),  # 선물 형식
    ],
)
def test_parse_symbol_filters(filters: list[dict[str, str]], expected_notional: float) -> None:
    base = [
        {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
        {"filterType": "LOT_SIZE", "minQty": "0.0001", "stepSize": "0.0001"},
    ]
    parsed = parse_symbol_filters(base + filters)
    assert parsed.min_notional == expected_notional
    assert (parsed.min_qty, parsed.qty_step, parsed.price_tick) == (0.0001, 0.0001, 0.01)


def test_parse_symbol_filters_missing_lot_size() -> None:
    with pytest.raises(DataSourceError):
        parse_symbol_filters([{"filterType": "PRICE_FILTER", "tickSize": "0.01"}])


# ---------------------------------------------------------------------------
# data.binance.vision
# ---------------------------------------------------------------------------


def vision(session: FakeVisionSession, now: dt.datetime, market: str = "spot",
           verify: bool = True) -> BinanceVisionProvider:
    retry = RetryPolicy(0, 0.0, BinanceVisionProvider.retryable_errors())
    return BinanceVisionProvider(market, base_url="https://data.binance.vision", retry=retry,  # type: ignore[arg-type]
                                 verify_checksum=verify, session=session, clock=lambda: now)  # type: ignore[arg-type]


def test_parse_kline_csv_detects_microseconds_and_header() -> None:
    open_ms = ms(utc(2025, 2, 1))
    micro = f"{open_ms * 1000},1,2,0.5,1.5,10,{(open_ms + TF_MS - 1) * 1000},0,0,0,0,0\n".encode()
    milli_with_header = (
        "open_time,open,high,low,close,volume,close_time,quote_volume,count,tbv,tbqv,ignore\n"
        f"{open_ms},1,2,0.5,1.5,10,{open_ms + TF_MS - 1},0,0,0,0,0\n"
    ).encode()
    for content in (micro, milli_with_header):
        frame = parse_kline_csv(content)
        assert frame.index[0] == pd.Timestamp("2025-02-01", tz="UTC")
        assert frame.iloc[0].tolist() == [1.0, 2.0, 0.5, 1.5, 10.0]


def test_vision_monthly_then_daily_and_matches_rest() -> None:
    now = utc(2025, 3, 3, 5)  # 3월은 1·2일 일별 파일까지, 3일은 아직 없음
    session = FakeVisionSession(unpublished={"2025-03-03"})
    frame = vision(session, now).fetch_ohlcv("BTC/USDT", "15m", utc(2025, 1, 1), utc(2025, 4, 1))
    assert frame.index[0] == pd.Timestamp("2025-01-01", tz="UTC")
    assert frame.index[-1] == pd.Timestamp("2025-03-02 23:45", tz="UTC")
    assert len(frame) == (31 + 28 + 2) * 96
    urls = [u for _, u in session.requests if not u.endswith(".CHECKSUM")]
    assert any("/monthly/" in u and "2025-02" in u for u in urls)
    assert any("/daily/" in u and "2025-03-02" in u for u in urls)
    # 같은 봉을 REST 로 받은 결과와 동일해야 한다 (교차검증)
    ex = FakeExchange(utc(2017, 8, 17), now)
    rest_frame = rest(ex, now, page=1000).fetch_ohlcv("BTC/USDT", "15m", utc(2025, 1, 1), utc(2025, 3, 3))
    pd.testing.assert_frame_equal(frame, rest_frame)


def test_vision_falls_back_to_daily_when_monthly_unpublished() -> None:
    now = utc(2025, 3, 2, 12)
    session = FakeVisionSession(unpublished={"2025-02", "2025-03-02"})
    frame = vision(session, now).fetch_ohlcv("ETH/USDT", "15m", utc(2025, 2, 27), utc(2025, 3, 5))
    assert frame.index[0] == pd.Timestamp("2025-02-27", tz="UTC")
    assert frame.index[-1] == pd.Timestamp("2025-03-01 23:45", tz="UTC")
    assert len(frame) == 3 * 96


def test_vision_futures_csv_with_header() -> None:
    now = utc(2025, 6, 1)
    frame = vision(FakeVisionSession(), now, market="usdm_futures").fetch_ohlcv(
        "BTC/USDT", "15m", utc(2025, 1, 1), utc(2025, 2, 1)
    )
    assert len(frame) == 31 * 96


def test_vision_checksum_mismatch_raises() -> None:
    now = utc(2025, 6, 1)
    session = FakeVisionSession(corrupt={"2025-01"})
    with pytest.raises(DataSourceError, match="SHA-256"):
        vision(session, now).fetch_ohlcv("BTC/USDT", "15m", utc(2025, 1, 1), utc(2025, 2, 1))


def test_vision_earliest_available_binary_search() -> None:
    now = utc(2025, 9, 3)
    session = FakeVisionSession(first_month="2019-09", unpublished={"2025-08"})
    assert vision(session, now).earliest_available("BTC/USDT", "15m") == utc(2019, 9, 1)
    heads = [u for method, u in session.requests if method == "HEAD"]
    assert len(heads) < 12  # 이분 탐색


def test_vision_funding() -> None:
    now = utc(2025, 6, 1)
    times = [ms(utc(2025, 1, 1)) + 8 * 3600_000 * i for i in range(200)]
    session = FakeVisionSession(funding=dict.fromkeys(times, 0.0001))
    series = vision(session, now, market="usdm_futures").funding_rates(
        "BTC/USDT", utc(2025, 1, 1), utc(2025, 2, 1)
    )
    assert len(series) == 31 * 3
    assert series.name == "funding_rate"
