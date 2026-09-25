"""거래비용 모델과 종목 규칙 (수량·호가 반올림)."""

from __future__ import annotations

import datetime as dt

import pytest

from rsidiv.broker.costs import CostModel
from rsidiv.broker.instruments import CryptoInstrument, KrxInstrument, ceil_to_step, floor_to_step
from rsidiv.core.config import deep_merge, load_settings
from rsidiv.core.models import AssetClass, Liquidity, OrderType, Side

SETTINGS = load_settings()
COSTS = SETTINGS.costs
KRX = SETTINGS.markets.krx
BTC = CryptoInstrument.from_filter("BTC/USDT", SETTINGS.markets.binance.fallback_filters["spot"]["BTC/USDT"])
SAMSUNG = KrxInstrument("005930", KRX)
T = dt.datetime(2026, 3, 2, 1, 0, tzinfo=dt.UTC)  # 10:00 KST


def test_step_rounding_avoids_float_error() -> None:
    assert floor_to_step(0.3, 0.1) == 0.3  # 0.3 / 0.1 = 2.9999999999999996 이지만 3단위
    assert floor_to_step(0.00012999, 0.00001) == 0.00012
    assert ceil_to_step(61632.231, 0.01) == 61632.24
    assert BTC.round_qty(0.0123456) == 0.01234 and BTC.round_qty(-1) == 0.0
    assert BTC.floor_price(61632.239) == 61632.23 and BTC.ceil_price(61632.231) == 61632.24


def test_krx_tick_rounding_across_price_bands() -> None:
    assert SAMSUNG.tick_size(1999) == 1 and SAMSUNG.tick_size(2000) == 5
    assert SAMSUNG.floor_price(60_070) == 60_000 and SAMSUNG.ceil_price(60_010) == 60_100
    assert SAMSUNG.ceil_price(1999.5) == 2000  # 가격대 경계
    assert SAMSUNG.round_qty(7.9) == 7.0 and SAMSUNG.min_qty == 1.0


def test_liquidity_by_order_type() -> None:
    model = CostModel(COSTS, AssetClass.CRYPTO)
    assert model.liquidity(OrderType.MARKET) is Liquidity.TAKER
    assert model.liquidity(OrderType.STOP_MARKET) is Liquidity.TAKER
    assert model.liquidity(OrderType.LIMIT, resting=True) is Liquidity.MAKER
    assert model.liquidity(OrderType.LIMIT, resting=False) is Liquidity.TAKER


def test_crypto_spot_costs() -> None:
    model = CostModel(COSTS, AssetClass.CRYPTO)
    assert model.fill_price(Side.BUY, 100_000.0, Liquidity.TAKER, BTC) == pytest.approx(100_020.0)
    assert model.fill_price(Side.SELL, 100_000.0, Liquidity.TAKER, BTC) == pytest.approx(99_980.0)
    assert model.fill_price(Side.SELL, 100_000.0, Liquidity.MAKER, BTC) == 100_000.0
    assert model.fees(Side.BUY, 0.01, 100_000.0, Liquidity.TAKER, T) == pytest.approx((1.0, 0.0))  # 0.1%
    double = CostModel(COSTS, AssetClass.CRYPTO, slippage_multiplier=2)
    assert double.fill_price(Side.BUY, 100_000.0, Liquidity.TAKER, BTC) == pytest.approx(100_040.0)
    gross = CostModel(COSTS, AssetClass.CRYPTO, slippage_multiplier=0, include_fees=False)
    assert gross.fill_price(Side.BUY, 100_000.0, Liquidity.TAKER, BTC) == 100_000.0
    assert gross.fees(Side.SELL, 1, 100_000.0, Liquidity.TAKER, T) == (0.0, 0.0)
    assert model.funding_payment(1, 100_000.0, 0.0001) == 0.0  # 현물은 펀딩비 없음


def test_crypto_futures_fees_bnb_and_funding() -> None:
    futures = CostModel(COSTS, AssetClass.CRYPTO, market_type="usdm_futures")
    assert futures.fees(Side.BUY, 1, 10_000.0, Liquidity.MAKER, T)[0] == pytest.approx(2.0)   # 0.02%
    assert futures.fees(Side.BUY, 1, 10_000.0, Liquidity.TAKER, T)[0] == pytest.approx(5.0)   # 0.05%
    assert futures.funding_payment(2, 10_000.0, 0.0001) == pytest.approx(2.0)
    assert futures.funding_payment(2, 10_000.0, -0.0001) == pytest.approx(-2.0)
    cfg = type(COSTS).model_validate(deep_merge(COSTS.model_dump(), {"crypto": {"spot": {"bnb_discount": {"enabled": True}}}}))
    bnb = CostModel(cfg, AssetClass.CRYPTO)
    assert bnb.fees(Side.BUY, 1, 10_000.0, Liquidity.TAKER, T)[0] == pytest.approx(7.5)  # 0.1% × 0.75


def test_stock_costs_floor_krw_and_tick_slippage() -> None:
    model = CostModel(COSTS, AssetClass.STOCK_KR, krx=KRX)
    assert model.fill_price(Side.BUY, 60_000, Liquidity.TAKER, SAMSUNG) == 60_100   # +1호가
    assert model.fill_price(Side.SELL, 60_000, Liquidity.TAKER, SAMSUNG) == 59_900
    assert model.fill_price(Side.SELL, 2_000, Liquidity.TAKER, SAMSUNG) == 1_999     # 경계 아래 호가단위
    assert model.fill_price(Side.SELL, 60_000, Liquidity.MAKER, SAMSUNG) == 60_000   # 지정가 0호가
    fee, tax = model.fees(Side.SELL, 7, 60_100, Liquidity.TAKER, T)
    assert fee == 59.0  # 420,700 × 0.0140527% = 59.12 → 59원
    assert tax == 841.0  # 420,700 × 0.20% = 841.4 → 841원
    assert model.fees(Side.BUY, 7, 60_100, Liquidity.TAKER, T)[1] == 0.0
    with pytest.raises(ValueError, match="krx"):
        CostModel(COSTS, AssetClass.STOCK_KR)


def test_sell_tax_schedule_uses_krx_trade_date() -> None:
    cfg = type(COSTS).model_validate(deep_merge(COSTS.model_dump(), {"stock_kr": {"sell_tax": {"schedule": [
        {"effective_from": "2025-01-01", "rate": 0.0015}, {"effective_from": "2026-01-01", "rate": 0.0020}]}}}))
    model = CostModel(cfg, AssetClass.STOCK_KR, krx=KRX)
    # 2025-12-31 15:30 UTC = 2026-01-01 00:30 KST → 2026년 세율
    assert model.fee_rates(Side.SELL, Liquidity.TAKER, dt.datetime(2025, 12, 31, 15, 30, tzinfo=dt.UTC))[1] == 0.0020
    assert model.fee_rates(Side.SELL, Liquidity.TAKER, dt.datetime(2025, 12, 31, 6, 0, tzinfo=dt.UTC))[1] == 0.0015
