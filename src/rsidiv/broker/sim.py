"""백테스트용 브로커 (SimBroker): 과거 봉으로 주문을 체결한다.

전략·리스크 계층은 :class:`rsidiv.broker.base.Broker` 인터페이스로만 주문한다. SimBroker 는 그 인터페이스에
봉 단위 체결 단계(엔진이 호출)를 더한 구현체다. 한 봉(시작 시각 T)의 처리 순서:

1. :meth:`apply_funding` — 선물 펀딩 정산 시각이 T 이면, T 이전부터 보유한 포지션에 펀딩비를 반영한다.
2. :meth:`execute_open` — T 이전에 낸 시장가 주문을 T 시가로 체결한다 (매도 먼저, 그다음 매수).
3. :meth:`execute_intrabar` — 대기 중인 손절(스탑)·지정가 주문을 봉의 고가·저가로 판정한다.
   같은 종목에서 손절을 먼저 본다(같은 봉에서 둘 다 닿으면 손절). 시가가 손절가보다 불리하면 시가 체결.
   T 시가 체결 직후 낸 주문(진입 직후의 손절·익절)도 이 단계에서 판정한다.
4. :meth:`execute_late` — 3단계 체결에 반응해 봉 중간에 낸 주문은 봉 안의 순서를 알 수 없으므로,
   손절만 보수적으로 판정한다(저가가 손절가 이하면 손절가 체결). 지정가는 다음 봉부터 본다.
5. :meth:`mark` — 종가로 평가금액을 갱신한다.

지정가 체결 규칙:
- 봉 시작 전부터 대기(resting): ``touch`` 면 고가(매도)·저가(매수)가 지정가에 닿으면, ``through`` 면
  1호가 이상 넘어서면 지정가로 체결(메이커). 갭으로 더 유리한 가격에 열려도 지정가로 체결한다(보수적).
- 시가 체결 직후 낸 지정가가 이미 시가 기준으로 체결 가능하면 시가로 즉시 체결(테이커).

선물은 레버리지 1배 롱(명목가 전액을 현금으로 담보)으로 본다. 그래서 현금·평가금액 계산은 현물과 같다.
"""

from __future__ import annotations

import datetime as dt
import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field

from rsidiv.broker.base import Broker
from rsidiv.broker.costs import CostModel
from rsidiv.core.config import FillCfg
from rsidiv.core.models import (
    AssetClass,
    Bar,
    Fill,
    InstrumentRules,
    Liquidity,
    Order,
    OrderRequest,
    OrderStatus,
    OrderType,
    Position,
    Side,
)

_OPEN, _INTRABAR, _LATE = "open", "intrabar", "late"


@dataclass(frozen=True, slots=True)
class FundingEvent:
    symbol: str
    time: dt.datetime
    qty: float
    mark_price: float
    rate: float
    payment: float  # 양수 = 비용


@dataclass(slots=True)
class _Meta:
    placed_bar: dt.datetime | None  # 주문을 낸 시점에 진행 중이던 봉 (None = 봉 사이, 즉 직전 봉 종가 이후)
    placed_phase: str | None = None  # 진행 중이던 봉의 어느 단계에서 냈는지


@dataclass(slots=True)
class _State:
    orders: dict[str, Order] = field(default_factory=dict)
    open: dict[str, Order] = field(default_factory=dict)  # 미체결 주문만 (봉마다 전체 주문을 훑지 않도록)
    meta: dict[str, _Meta] = field(default_factory=dict)
    positions: dict[str, Position] = field(default_factory=dict)
    last_price: dict[str, float] = field(default_factory=dict)


class SimBroker(Broker):
    def __init__(
        self,
        asset_class: AssetClass,
        *,
        cash: float,
        instruments: Mapping[str, InstrumentRules],
        costs: CostModel,
        fill: FillCfg,
        funding_rates: Mapping[str, Mapping[dt.datetime, float]] | None = None,
    ) -> None:
        self.asset_class = asset_class
        self._cash = cash
        self._instruments = dict(instruments)
        self._costs = costs
        self._fill = fill
        self._funding = {s: dict(r) for s, r in (funding_rates or {}).items()}
        self._s = _State()
        self._ids = itertools.count(1)
        self._bar_time: dt.datetime | None = None
        self._phase: str | None = None
        self.funding_events: list[FundingEvent] = []

    # --- Broker 인터페이스 --------------------------------------------------

    def submit(self, request: OrderRequest) -> Order:
        if request.symbol not in self._instruments:
            raise KeyError(f"알 수 없는 심볼: {request.symbol}")
        order_id = f"S{next(self._ids)}"
        now = self._bar_time or request.created_at
        order = Order(order_id, request, OrderStatus.NEW, now)
        self._s.orders[order_id] = order
        self._s.open[order_id] = order
        self._s.meta[order_id] = _Meta(self._bar_time if self._phase else None, self._phase)
        return order

    def cancel(self, order_id: str) -> Order:
        order = self._s.orders[order_id]
        if order.is_open:
            order.status = OrderStatus.CANCELED
            order.updated_at = self._bar_time or order.updated_at
            self._s.open.pop(order_id, None)
        return order

    def get_order(self, order_id: str) -> Order:
        return self._s.orders[order_id]

    def open_orders(self, symbol: str | None = None) -> list[Order]:
        return [o for o in self._s.open.values() if symbol is None or o.request.symbol == symbol]

    def positions(self) -> dict[str, Position]:
        return dict(self._s.positions)

    def cash(self) -> float:
        return self._cash

    def equity(self) -> float:
        return self._cash + sum(p.qty * self._s.last_price.get(s, p.avg_price) for s, p in self._s.positions.items())

    def instrument_rules(self, symbol: str) -> InstrumentRules:
        return self._instruments[symbol]

    # --- 봉 단위 체결 (엔진 전용) --------------------------------------------

    def begin_bar(self, time: dt.datetime) -> None:
        self._bar_time = time
        self._phase = _OPEN

    def end_bar(self) -> None:
        self._phase = None

    def apply_funding(self, time: dt.datetime, opens: Mapping[str, float]) -> list[FundingEvent]:
        events = []
        for symbol, pos in self._s.positions.items():
            rate = self._funding.get(symbol, {}).get(time)
            if rate is None or symbol not in opens:
                continue
            payment = self._costs.funding_payment(pos.qty, opens[symbol], rate)
            self._cash -= payment
            events.append(FundingEvent(symbol, time, pos.qty, opens[symbol], rate, payment))
        self.funding_events += events
        return events

    def execute_open(self, bars: Mapping[str, Bar]) -> list[Fill]:
        """직전 봉 종가 이후에 낸 시장가 주문을 이번 봉 시가로 체결한다."""
        self._phase = _OPEN
        pending = [o for o in self._pending(bars) if o.request.order_type is OrderType.MARKET]
        pending.sort(key=lambda o: o.request.side is Side.BUY)  # 매도 먼저 → 현금 확보
        fills = []
        for order in pending:
            bar = bars[order.request.symbol]
            liquidity = self._costs.liquidity(OrderType.MARKET)
            fill = self._fill_order(order, bar.open, liquidity, bar.time)
            if fill is not None:
                fills.append(fill)
        return fills  # 이 체결에 반응해 내는 주문은 '시가 직후' 주문으로 기록된다 (phase 유지)

    def execute_intrabar(self, bars: Mapping[str, Bar]) -> list[Fill]:
        self._phase = _INTRABAR
        return self._intrabar(bars, late=False)

    def execute_late(self, bars: Mapping[str, Bar]) -> list[Fill]:
        self._phase = _LATE
        return self._intrabar(bars, late=True)

    def mark(self, closes: Mapping[str, float]) -> None:
        self._s.last_price.update(closes)

    def execute_close_out(self, closes: Mapping[str, float], time: dt.datetime) -> list[Fill]:
        """백테스트 종료: 대기 중인 시장가 주문을 마지막 종가로 체결한다 (테이커 비용 적용)."""
        pending = [o for o in self._s.open.values()
                   if o.request.order_type is OrderType.MARKET and o.request.symbol in closes]
        pending.sort(key=lambda o: o.request.side is Side.BUY)
        fills = []
        for order in pending:
            fill = self._fill_order(order, closes[order.request.symbol],
                                    self._costs.liquidity(OrderType.MARKET), time)
            if fill is not None:
                fills.append(fill)
        return fills

    # --- 내부 ---------------------------------------------------------------

    def _pending(self, bars: Mapping[str, Bar]) -> list[Order]:
        return [o for o in self._s.open.values() if o.request.symbol in bars]

    def _intrabar(self, bars: Mapping[str, Bar], *, late: bool) -> list[Fill]:
        fills: list[Fill] = []
        for symbol, bar in bars.items():
            orders = [o for o in self._pending({symbol: bar}) if o.request.order_type is not OrderType.MARKET]
            if late:
                orders = [o for o in orders if self._s.meta[o.order_id].placed_phase == _INTRABAR
                          and self._s.meta[o.order_id].placed_bar == bar.time]
            else:
                orders = [o for o in orders if not (self._s.meta[o.order_id].placed_bar == bar.time
                                                   and self._s.meta[o.order_id].placed_phase != _OPEN)]
            orders.sort(key=lambda o: o.request.order_type is not OrderType.STOP_MARKET)  # 손절 먼저
            for order in orders:
                if not order.is_open:  # 같은 OCO 그룹의 다른 주문이 먼저 체결돼 취소됨
                    continue
                fill = self._try_resting(order, bar, late=late)
                if fill is not None:
                    fills.append(fill)
        return fills

    def _try_resting(self, order: Order, bar: Bar, *, late: bool) -> Fill | None:
        req = order.request
        inst = self._instruments[req.symbol]
        meta = self._s.meta[order.order_id]
        placed_at_open = meta.placed_bar == bar.time and meta.placed_phase == _OPEN
        if req.order_type is OrderType.STOP_MARKET:
            assert req.stop_price is not None
            stop = req.stop_price
            if req.side is Side.SELL and bar.low <= stop:
                base = stop if late else min(bar.open, stop)  # 시가가 손절가 아래로 갭 → 시가
            elif req.side is Side.BUY and bar.high >= stop:
                base = stop if late else max(bar.open, stop)
            else:
                return None
            return self._fill_order(order, base, self._costs.liquidity(OrderType.STOP_MARKET), bar.time)
        if late:
            return None
        assert req.limit_price is not None
        limit = req.limit_price
        margin = inst.tick_size(limit) if self._fill.limit_fill_rule == "through" else 0.0
        if req.side is Side.SELL:
            if placed_at_open and bar.open >= limit:
                return self._fill_order(order, bar.open, self._costs.liquidity(OrderType.LIMIT), bar.time)
            touched = bar.high >= limit + margin
        else:
            if placed_at_open and bar.open <= limit:
                return self._fill_order(order, bar.open, self._costs.liquidity(OrderType.LIMIT), bar.time)
            touched = bar.low <= limit - margin
        if not touched:
            return None
        return self._fill_order(order, limit, self._costs.liquidity(OrderType.LIMIT, resting=True), bar.time)

    def _fill_order(self, order: Order, base_price: float, liquidity: Liquidity, time: dt.datetime) -> Fill | None:
        req = order.request
        inst = self._instruments[req.symbol]
        price = self._costs.fill_price(req.side, base_price, liquidity, inst)
        fee, tax = self._costs.fees(req.side, req.qty, price, liquidity, time)
        pos = self._s.positions.get(req.symbol)
        if req.side is Side.BUY:
            cost = req.qty * price + fee + tax
            if cost > self._cash + 1e-9:
                self._reject(order, time)
                return None
            self._cash -= cost
            if pos is None:
                self._s.positions[req.symbol] = Position(req.symbol, self.asset_class, req.qty, price, time)
            else:
                total = pos.qty + req.qty
                pos.avg_price = (pos.avg_price * pos.qty + price * req.qty) / total
                pos.qty = total
        else:
            if pos is None or req.qty > pos.qty + 1e-12:
                self._reject(order, time)
                return None
            self._cash += req.qty * price - fee - tax
            pos.qty -= req.qty
            if pos.qty <= 1e-12:
                del self._s.positions[req.symbol]
        fill = Fill(order.order_id, req.symbol, req.side, req.qty, price, fee, tax, liquidity, time)
        order.fills.append(fill)
        order.filled_qty = req.qty
        order.avg_fill_price = price
        order.status = OrderStatus.FILLED
        order.updated_at = time
        self._s.open.pop(order.order_id, None)
        self._s.last_price[req.symbol] = price
        if req.oco_group is not None:
            for other in list(self._s.open.values()):
                if other.request.oco_group == req.oco_group:
                    self.cancel(other.order_id)
        return fill

    def _reject(self, order: Order, time: dt.datetime) -> None:
        order.status = OrderStatus.REJECTED
        order.updated_at = time
        self._s.open.pop(order.order_id, None)
