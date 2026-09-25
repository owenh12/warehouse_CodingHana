"""SimBroker 체결 규칙: 시가 시장가, 손절 우선, 갭 체결, 지정가 touch/through, OCO, 봉 중간 주문, 펀딩비."""

from __future__ import annotations

import datetime as dt

import pytest

from rsidiv.broker.costs import CostModel
from rsidiv.broker.instruments import CryptoInstrument
from rsidiv.broker.sim import SimBroker
from rsidiv.core.config import FillCfg, load_settings
from rsidiv.core.models import (
    AssetClass,
    Bar,
    Liquidity,
    OrderPurpose,
    OrderRequest,
    OrderStatus,
    OrderType,
    Side,
)

SETTINGS = load_settings()
INST = CryptoInstrument("BTC/USDT", 0.001, 0.001, 5.0, 0.1)
T0 = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
BAR = dt.timedelta(minutes=15)


def broker(cash: float = 10_000.0, *, gross: bool = True, market: str = "spot", through: bool = False,
           funding: dict[dt.datetime, float] | None = None) -> SimBroker:
    costs = CostModel(SETTINGS.costs, AssetClass.CRYPTO, market_type=market,  # type: ignore[arg-type]
                      slippage_multiplier=0 if gross else 1, include_fees=not gross)
    fill = FillCfg.model_validate({**SETTINGS.backtest.fill.model_dump(),
                                   "limit_fill_rule": "through" if through else "touch"})
    return SimBroker(AssetClass.CRYPTO, cash=cash, instruments={"BTC/USDT": INST}, costs=costs, fill=fill,
                     funding_rates={"BTC/USDT": funding or {}})


def req(side: Side, qty: float, kind: OrderType, *, limit: float | None = None, stop: float | None = None,
        oco: str | None = None, purpose: OrderPurpose = OrderPurpose.ENTRY) -> OrderRequest:
    return OrderRequest(f"c-{side}-{kind}-{limit}-{stop}", "BTC/USDT", AssetClass.CRYPTO, side, qty, kind,
                        purpose, T0, limit_price=limit, stop_price=stop, oco_group=oco)


def step(b: SimBroker, i: int, o: float, h: float, lo: float, c: float, on_open=None, on_intrabar=None):  # type: ignore[no-untyped-def]
    bar = Bar(T0 + i * BAR, o, h, lo, c, 1.0)
    b.begin_bar(bar.time)
    b.apply_funding(bar.time, {"BTC/USDT": o})
    fills = b.execute_open({"BTC/USDT": bar})
    if on_open:
        on_open(fills)
    intra = b.execute_intrabar({"BTC/USDT": bar})
    if on_intrabar:
        on_intrabar(intra)
    late = b.execute_late({"BTC/USDT": bar})
    b.mark({"BTC/USDT": c})
    b.end_bar()
    return fills + intra + late


def test_market_order_fills_at_next_open_with_costs() -> None:
    b = broker(gross=False)
    order = b.submit(req(Side.BUY, 0.1, OrderType.MARKET))
    fills = step(b, 1, 100.0, 101.0, 99.0, 100.5)
    assert len(fills) == 1 and fills[0].price == pytest.approx(100.02)  # +0.02% 슬리피지
    assert fills[0].fee == pytest.approx(0.1 * 100.02 * 0.001) and fills[0].liquidity is Liquidity.TAKER
    assert order.status is OrderStatus.FILLED
    assert b.cash() == pytest.approx(10_000 - 10.002 - 0.010002)
    assert b.equity() == pytest.approx(b.cash() + 0.1 * 100.5)
    assert b.positions()["BTC/USDT"].avg_price == pytest.approx(100.02)


def test_market_buy_rejected_without_cash_and_sell_without_position() -> None:
    b = broker(cash=5.0)
    buy = b.submit(req(Side.BUY, 1.0, OrderType.MARKET))
    sell = b.submit(req(Side.SELL, 1.0, OrderType.STOP_MARKET, stop=200.0))
    step(b, 1, 100, 101, 99, 100)
    assert buy.status is OrderStatus.REJECTED and sell.status is OrderStatus.REJECTED
    assert b.cash() == 5.0 and not b.positions()


def entered(b: SimBroker, *, stop: float, target: float, i: int = 1, o: float = 100.0, h: float = 100.5,
            lo: float = 99.5, c: float = 100.0):  # type: ignore[no-untyped-def]
    """i 봉 시가 진입 직후 손절·익절(OCO)을 내는 흐름."""
    b.submit(req(Side.BUY, 1.0, OrderType.MARKET))
    placed = {}

    def protect(fills):  # type: ignore[no-untyped-def]
        if fills:
            placed["stop"] = b.submit(req(Side.SELL, 1.0, OrderType.STOP_MARKET, stop=stop, oco="g",
                                          purpose=OrderPurpose.STOP_LOSS))
            placed["target"] = b.submit(req(Side.SELL, 1.0, OrderType.LIMIT, limit=target, oco="g",
                                            purpose=OrderPurpose.TAKE_PROFIT))

    fills = step(b, i, o, h, lo, c, on_open=protect)
    return placed, fills


def test_stop_and_target_same_bar_stop_first_and_oco() -> None:
    b = broker()
    placed, _ = entered(b, stop=95.0, target=110.0)
    fills = step(b, 2, 100, 111, 94, 105)  # 둘 다 닿음
    assert [f.price for f in fills] == [95.0]
    assert placed["stop"].status is OrderStatus.FILLED and placed["target"].status is OrderStatus.CANCELED
    assert not b.positions()


def test_stop_gap_fills_at_open() -> None:
    b = broker()
    entered(b, stop=95.0, target=110.0)
    fills = step(b, 2, 90, 92, 88, 91)  # 시가가 손절가 아래
    assert fills[0].price == 90.0


def test_stop_applies_on_entry_bar() -> None:
    b = broker()
    placed, fills = entered(b, stop=99.0, target=110.0, lo=98.0)
    assert [f.price for f in fills] == [100.0, 99.0]
    assert placed["stop"].status is OrderStatus.FILLED


def test_resting_target_touch_vs_through() -> None:
    for through, high, filled in ((False, 110.0, True), (True, 110.0, False), (True, 110.1, True)):
        b = broker(through=through)
        placed, _ = entered(b, stop=95.0, target=110.0)
        fills = step(b, 2, 105, high, 104, 106)
        assert (placed["target"].status is OrderStatus.FILLED) is filled
        if filled:
            assert fills[0].price == 110.0 and fills[0].liquidity is Liquidity.MAKER


def test_resting_target_gap_up_fills_at_limit_not_open() -> None:
    b = broker()
    _placed, _ = entered(b, stop=95.0, target=110.0)
    fills = step(b, 2, 115, 116, 114, 115)
    assert fills[0].price == 110.0  # 보수적: 갭 이득을 주지 않음


def test_target_marketable_at_entry_open_fills_at_open_as_taker() -> None:
    b = broker(gross=False)
    _placed, fills = entered(b, stop=95.0, target=99.0)  # 시가 100 ≥ 목표 99
    target_fill = next(f for f in fills if f.side is Side.SELL)
    assert target_fill.liquidity is Liquidity.TAKER
    assert target_fill.price == pytest.approx(100.0 * (1 - 0.0002))


def test_orders_placed_mid_bar_only_check_stop_conservatively() -> None:
    b = broker()
    b.submit(req(Side.BUY, 1.0, OrderType.LIMIT, limit=99.0))  # 지정가 진입 (종가 이후 제출)
    placed = {}

    def protect(fills):  # type: ignore[no-untyped-def]
        if any(f.side is Side.BUY for f in fills):
            placed["stop"] = b.submit(req(Side.SELL, 1.0, OrderType.STOP_MARKET, stop=98.0, oco="g"))
            placed["target"] = b.submit(req(Side.SELL, 1.0, OrderType.LIMIT, limit=100.5, oco="g"))

    fills = step(b, 1, 100, 101, 97.5, 100, on_intrabar=protect)
    assert [f.price for f in fills] == [99.0, 98.0]  # 봉 안 순서를 모르므로 손절로 본다 (목표가는 판정 안 함)
    b2 = broker()
    b2.submit(req(Side.BUY, 1.0, OrderType.LIMIT, limit=99.0))
    placed.clear()

    def protect2(fills):  # type: ignore[no-untyped-def]
        if any(f.side is Side.BUY for f in fills):
            placed["target"] = b2.submit(req(Side.SELL, 1.0, OrderType.LIMIT, limit=100.5, oco="g"))

    step(b2, 1, 100, 101, 98.5, 100, on_intrabar=protect2)
    assert placed["target"].is_open  # 같은 봉에서 목표가 체결 없음
    fills = step(b2, 2, 100, 100.6, 99.8, 100.2)
    assert fills[0].price == 100.5


def test_funding_charged_only_for_positions_held_before_settlement() -> None:
    settle = T0 + 2 * BAR
    b = broker(gross=False, market="usdm_futures", funding={settle: 0.0001, T0 + 1 * BAR: 0.0001})
    b.submit(req(Side.BUY, 1.0, OrderType.MARKET))
    step(b, 1, 100, 101, 99, 100)  # T0+1 봉 시가 진입: 같은 시각 정산은 제외
    assert b.funding_events == []
    cash = b.cash()
    step(b, 2, 102, 103, 101, 102)  # 보유 중 정산
    assert len(b.funding_events) == 1 and b.funding_events[0].payment == pytest.approx(102 * 0.0001)
    assert b.cash() == pytest.approx(cash - 0.0102)


def test_cancel_and_open_orders() -> None:
    b = broker()
    order = b.submit(req(Side.BUY, 1.0, OrderType.LIMIT, limit=50.0))
    assert b.open_orders("BTC/USDT") == [order]
    b.cancel(order.order_id)
    assert order.status is OrderStatus.CANCELED and b.open_orders() == []
    step(b, 1, 100, 101, 40, 100)
    assert not b.positions()
