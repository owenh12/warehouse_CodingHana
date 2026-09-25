"""계좌 하나의 매매 관리. 백테스트·모의투자·실거래가 같은 코드를 쓰고, 주문은 Broker 인터페이스로만 낸다.

봉 종가 확정 후 :meth:`AccountController.on_bar_close` 가 종목마다 다음을 순서대로 처리한다.

1. 지표·신호 갱신 (다이버전스 탐지기, 손절용·트레일링용 ATR)
2. 직전에 낸 진입 주문 확인 (거부·지정가 미체결 취소)
3. 보유 포지션 관리: 보유 봉 수, 장 마감 청산(국내주식), 시간 청산, 트레일링 손절 갱신
4. 진입 확인 대기(B·C) 판정
5. 새 신호: 종목 중복 → 손절가 → 진입 모드 → 리스크 게이트(킬 스위치·일일 손실·세션·보유 한도·p2) → 사이징 → 주문

체결은 :meth:`on_fills` 로 전달받는다. 진입 체결 직후 손절(스탑)·익절(지정가)을 OCO 로 낸다.
모든 신호는 결과(진입·스킵 사유)와 함께 :attr:`signals` 에, 청산된 거래는 :attr:`trades` 에 쌓인다.
"""

from __future__ import annotations

import datetime as dt
import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

import pandas as pd

from rsidiv.broker.base import Broker
from rsidiv.broker.costs import CostModel
from rsidiv.core.config import AssetClassName, KrxCfg, RiskCfg, StrategyParams, timeframe_minutes
from rsidiv.core.models import (
    AssetClass,
    Bar,
    Fill,
    OrderPurpose,
    OrderRequest,
    OrderStatus,
    OrderType,
    Side,
)
from rsidiv.indicators.atr import AtrState
from rsidiv.risk.manager import DailyLossGuard, RiskEvent
from rsidiv.risk.sizing import SizingResult, size_position
from rsidiv.signals.divergence import DivergenceCandidate, DivergenceDetector
from rsidiv.strategy.rules import (
    EntryConfirmer,
    initial_stop,
    p2_blocks_entry,
    target_price,
    trailing_stop,
)

_NAN = float("nan")

#: 신호 결과 코드 → 보고서 표기
OUTCOME_LABELS: dict[str, str] = {
    "entered": "진입",
    "ordered": "주문 대기",
    "awaiting_confirmation": "진입 확인 대기",
    "already_in_position": "같은 종목 보유 중",
    "entry_pending": "같은 종목 진입 주문 대기 중",
    "awaiting_other_signal": "같은 종목 다른 신호 확인 대기 중",
    "atr_warmup": "ATR 워밍업",
    "not_confirmed": "진입 확인 실패 (기간 만료)",
    "canceled_below_stop": "확인 대기 중 손절가 이탈",
    "kill_switch": "킬 스위치 발동 중",
    "daily_loss": "일일 손실 한도",
    "session_closed": "장 마감 전 진입 금지 구간",
    "max_positions": "동시 보유 한도",
    "p2_below_entry": "p2 목표가 ≤ 진입가",
    "stop_above_entry": "손절가 ≥ 진입가",
    "cap_exceeded": "비중 상한 초과 (skip 설정)",
    "insufficient_capital": "자본 부족 (최소 주문 미만)",
    "order_rejected": "주문 거부 (현금 부족)",
    "limit_not_filled": "지정가 미체결",
    "data_end": "데이터 끝",
}


@dataclass(slots=True)
class SignalRecord:
    account: str
    symbol: str
    signal_time: pd.Timestamp
    t1_time: pd.Timestamp | None
    p2_time: pd.Timestamp | None
    t3_time: pd.Timestamp
    price_t1: float
    price_p2: float
    price_t3: float
    rsi_t1: float
    rsi_t3: float
    outcome: str = "pending"
    decision_time: pd.Timestamp | None = None
    trade_id: int | None = None


@dataclass(slots=True)
class TradeRecord:
    trade_id: int
    account: str
    symbol: str
    signal: SignalRecord
    decision_time: pd.Timestamp
    reference_price: float
    qty: float
    initial_stop: float
    target: float | None
    risk_amount: float
    capped: bool
    entry_time: pd.Timestamp
    entry_price: float
    entry_fee: float
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    exit_fee: float = 0.0
    tax: float = 0.0
    funding: float = 0.0
    bars_held: int = 0
    lowest: float = math.inf
    highest: float = -math.inf
    final_stop: float = _NAN

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    @property
    def gross_pnl(self) -> float:
        assert self.exit_price is not None
        return self.qty * (self.exit_price - self.entry_price)

    @property
    def costs(self) -> float:
        return self.entry_fee + self.exit_fee + self.tax + self.funding

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.costs

    @property
    def initial_risk(self) -> float:
        """거래의 1R (금액) = 수량 × (진입 체결가 − 최초 손절가)."""
        return self.qty * (self.entry_price - self.initial_stop)

    @property
    def r_multiple(self) -> float:
        risk = self.initial_risk
        return self.net_pnl / risk if risk > 0 else _NAN

    @property
    def gross_r(self) -> float:
        risk = self.initial_risk
        return self.gross_pnl / risk if risk > 0 else _NAN

    @property
    def mae_r(self) -> float:
        unit = self.entry_price - self.initial_stop
        return (self.entry_price - self.lowest) / unit if unit > 0 else _NAN

    @property
    def mfe_r(self) -> float:
        unit = self.entry_price - self.initial_stop
        return (self.highest - self.entry_price) / unit if unit > 0 else _NAN


@dataclass(slots=True)
class _Entry:
    record: SignalRecord
    candidate: DivergenceCandidate
    stop: float
    sizing: SizingResult
    reference_price: float
    decision_time: pd.Timestamp
    order_id: str
    submitted_index: int


@dataclass(slots=True)
class _Open:
    trade: TradeRecord
    candidate: DivergenceCandidate
    stop: float
    stop_order: str | None = None
    target_order: str | None = None
    exit_order: str | None = None
    frozen: bool = False  # 킬 스위치 keep_stops: 손절 주문만 유지, 트레일링·시간 청산 중지


@dataclass(slots=True)
class _Symbol:
    detector: DivergenceDetector
    atr_stop: AtrState
    atr_trail: AtrState
    confirmer: EntryConfirmer
    index: int = -1
    prev_bar: Bar | None = None
    awaiting: SignalRecord | None = None


@dataclass(slots=True)
class ControllerStats:
    capped_entries: int = 0
    skipped: dict[str, int] = field(default_factory=dict)


class AccountController:
    def __init__(
        self,
        name: str,
        broker: Broker,
        *,
        symbols: Sequence[str],
        params: StrategyParams,
        risk: RiskCfg,
        costs: CostModel,
        asset_class: AssetClassName,
        timeframe: str,
        initial_equity: float,
        daily_loss: DailyLossGuard,
        krx: KrxCfg | None = None,
    ) -> None:
        if not risk.limits.one_position_per_symbol:
            raise NotImplementedError("종목당 여러 포지션은 지원하지 않습니다 (one_position_per_symbol: true)")
        self.name = name
        self.broker = broker
        self.symbols = list(symbols)
        self.params = params
        self.risk = risk
        self.costs = costs
        self.asset_class = asset_class
        self.limits = getattr(risk.limits, asset_class)
        self.initial_equity = initial_equity
        self.daily_loss = daily_loss
        self.krx = krx
        self._bar = pd.Timedelta(minutes=timeframe_minutes(timeframe))
        self._sym = {
            s: _Symbol(
                DivergenceDetector(params, timeframe=timeframe, asset_class=asset_class, krx=krx),
                AtrState(params.exit.stop.atr_period),
                AtrState(params.exit.take_profit.trailing_atr_period),
                EntryConfirmer(params.entry),
            )
            for s in self.symbols
        }
        self._entries: dict[str, _Entry] = {}
        self._open: dict[str, _Open] = {}
        self._roles: dict[str, tuple[str, str]] = {}  # order_id → (종류, 심볼)
        self._ids = itertools.count(1)
        self._client_seq = itertools.count(1)
        self._entry_block: str | None = None
        self.signals: list[SignalRecord] = []
        self.trades: list[TradeRecord] = []
        self.stats = ControllerStats()

    # --- 조회 ---------------------------------------------------------------

    @property
    def open_trades(self) -> list[TradeRecord]:
        return [o.trade for o in self._open.values()]

    def reserved_cash(self) -> float:
        return sum(e.sizing.qty * e.sizing.unit_cost for e in self._entries.values())

    # --- 봉 마감 -------------------------------------------------------------

    def on_bar_close(self, bars: Mapping[str, Bar], *, entry_block: str | None = None) -> None:
        """``entry_block``: 계좌 밖(포트폴리오 킬 스위치 등)에서 신규 진입을 막는 사유."""
        self._entry_block = entry_block
        for symbol in self.symbols:
            bar = bars.get(symbol)
            if bar is not None:
                self._on_symbol_bar(symbol, bar)

    def _on_symbol_bar(self, symbol: str, bar: Bar) -> None:
        st = self._sym[symbol]
        st.index += 1
        rsi_prev = st.detector.rsi_at(-1) if st.index > 0 else _NAN
        candidate = st.detector.update(bar.time, bar.open, bar.high, bar.low, bar.close, bar.volume)
        rsi_now = st.detector.rsi_at(-1)
        atr_stop = st.atr_stop.update(bar.high, bar.low, bar.close)
        atr_trail = st.atr_trail.update(bar.high, bar.low, bar.close)
        close_time = pd.Timestamp(bar.time) + self._bar

        self._check_entry_order(symbol, st.index)
        self._manage_open(symbol, bar, close_time, atr_trail)

        if st.confirmer.pending is not None and st.prev_bar is not None:
            outcome = st.confirmer.on_bar(st.index, bar, st.prev_bar, rsi_prev, rsi_now)
            record = st.awaiting
            assert record is not None
            if outcome == "enter":
                pending = st.confirmer.take()
                st.awaiting = None
                self._try_enter(symbol, record, pending.candidate, pending.stop, bar, close_time, st.index)
            elif outcome is not None:
                st.awaiting = None
                self._set_outcome(record, outcome)

        if candidate is not None and candidate.accepted:
            record = self._new_record(symbol, candidate, st)
            busy = self._symbol_busy(symbol)
            stop = initial_stop(candidate, atr_stop, self.params.exit.stop)
            if busy is not None:
                self._set_outcome(record, busy)
            elif stop is None:
                self._set_outcome(record, "atr_warmup")
            elif st.confirmer.immediate:
                self._try_enter(symbol, record, candidate, stop, bar, close_time, st.index)
            else:
                st.confirmer.start(candidate, stop, st.index)
                st.awaiting = record
                record.outcome = "awaiting_confirmation"
        st.prev_bar = bar

    def _new_record(self, symbol: str, c: DivergenceCandidate, st: _Symbol) -> SignalRecord:
        d = st.detector
        record = SignalRecord(
            self.name, symbol, c.signal_time,
            d.time_at(c.t1) if c.t1 is not None else None,
            d.time_at(c.p2) if c.p2 is not None else None,
            d.time_at(c.t3), c.price_t1, c.price_p2, c.price_t3, c.rsi_t1, c.rsi_t3,
        )
        self.signals.append(record)
        return record

    def _set_outcome(self, record: SignalRecord, outcome: str) -> None:
        record.outcome = outcome
        if outcome not in ("entered", "ordered", "awaiting_confirmation"):
            self.stats.skipped[outcome] = self.stats.skipped.get(outcome, 0) + 1

    def _symbol_busy(self, symbol: str) -> str | None:
        if symbol in self._open:
            return "already_in_position"
        if symbol in self._entries:
            return "entry_pending"
        if self._sym[symbol].confirmer.pending is not None:
            return "awaiting_other_signal"
        return None

    # --- 진입 ---------------------------------------------------------------

    def _session_closed(self, close_time: pd.Timestamp) -> bool:
        """국내주식 오버나잇 미보유: 다음 봉 시가가 '장 마감 − N분' 이후면 진입 금지·청산 대상."""
        overnight = self.params.exit.overnight_kr
        if self.asset_class != "stock_kr" or overnight.hold_overnight:
            return False
        assert self.krx is not None
        local = close_time.astimezone(ZoneInfo(self.krx.timezone))
        special = next((s for s in self.krx.special_sessions if s.date == local.date()), None)
        close_t = special.close if special else self.krx.regular_close
        session_close = dt.datetime.combine(local.date(), close_t, tzinfo=local.tzinfo)
        cutoff = session_close - dt.timedelta(minutes=overnight.flatten_minutes_before_close)
        return local >= cutoff

    def _try_enter(
        self, symbol: str, record: SignalRecord, candidate: DivergenceCandidate, stop: float, bar: Bar,
        close_time: pd.Timestamp, index: int,
    ) -> None:
        record.decision_time = close_time
        reference = bar.close
        if self._entry_block is not None:
            return self._set_outcome(record, self._entry_block)
        if self.daily_loss.blocked:
            return self._set_outcome(record, "daily_loss")
        if self._session_closed(close_time):
            return self._set_outcome(record, "session_closed")
        if len(self._open) + len(self._entries) >= self.limits.max_positions:
            return self._set_outcome(record, "max_positions")
        if p2_blocks_entry(reference, candidate, self.params.exit.take_profit):
            return self._set_outcome(record, "p2_below_entry")
        inst = self.broker.instrument_rules(symbol)
        stop = inst.floor_price(stop)
        sizing = size_position(
            cfg=self.risk.sizing, limits=self.limits, equity=self.broker.equity(),
            initial_equity=self.initial_equity, available_cash=self.broker.cash() - self.reserved_cash(),
            reference_price=reference, stop=stop, instrument=inst, costs=self.costs, when=close_time,
        )
        if sizing.skip_reason is not None:
            return self._set_outcome(record, sizing.skip_reason)
        limit = self.params.entry.order_type == "limit"
        order = self.broker.submit(OrderRequest(
            f"{self.name}-{next(self._client_seq)}", symbol, AssetClass(self.asset_class), Side.BUY, sizing.qty,
            OrderType.LIMIT if limit else OrderType.MARKET, OrderPurpose.ENTRY, close_time,
            limit_price=reference if limit else None,
        ))
        self._roles[order.order_id] = ("entry", symbol)
        self._entries[symbol] = _Entry(record, candidate, stop, sizing, reference, close_time, order.order_id, index)
        record.outcome = "ordered"
        return None

    def _check_entry_order(self, symbol: str, index: int) -> None:
        entry = self._entries.get(symbol)
        if entry is None:
            return
        order = self.broker.get_order(entry.order_id)
        if order.status is OrderStatus.REJECTED:
            del self._entries[symbol]
            self._set_outcome(entry.record, "order_rejected")
        elif order.is_open and index > entry.submitted_index:  # 다음 봉에서 체결되지 않은 지정가
            self.broker.cancel(order.order_id)
            del self._entries[symbol]
            self._set_outcome(entry.record, "limit_not_filled")

    # --- 체결 ---------------------------------------------------------------

    def on_fills(self, fills: Sequence[Fill]) -> None:
        for fill in fills:
            role = self._roles.get(fill.order_id)
            if role is None:
                continue
            kind, symbol = role
            if kind == "entry":
                self._open_trade(symbol, fill)
            else:
                self._close_trade(symbol, fill)

    def on_funding(self, symbol: str, payment: float) -> None:
        position = self._open.get(symbol)
        if position is not None:
            position.trade.funding += payment

    def _open_trade(self, symbol: str, fill: Fill) -> None:
        entry = self._entries.pop(symbol)
        inst = self.broker.instrument_rules(symbol)
        tp = self.params.exit.take_profit
        target = target_price(fill.price, entry.stop, entry.candidate, tp)
        target = inst.ceil_price(target) if target is not None else None
        trade = TradeRecord(
            next(self._ids), self.name, symbol, entry.record, entry.decision_time, entry.reference_price,
            fill.qty, entry.stop, target, entry.sizing.risk_amount, entry.sizing.capped,
            pd.Timestamp(fill.time), fill.price, fill.fee + fill.tax, final_stop=entry.stop,
            lowest=fill.price, highest=fill.price,
        )
        entry.record.outcome = "entered"
        entry.record.trade_id = trade.trade_id
        if entry.sizing.capped:
            self.stats.capped_entries += 1
        position = _Open(trade, entry.candidate, entry.stop)
        self._open[symbol] = position
        group = f"{self.name}-T{trade.trade_id}"
        position.stop_order = self._submit_exit(symbol, OrderType.STOP_MARKET, OrderPurpose.STOP_LOSS, fill.time,
                                                stop=entry.stop, oco=group)
        if target is not None:
            position.target_order = self._submit_exit(symbol, OrderType.LIMIT, OrderPurpose.TAKE_PROFIT, fill.time,
                                                      limit=target, oco=group)

    def _close_trade(self, symbol: str, fill: Fill) -> None:
        position = self._open.pop(symbol, None)
        if position is None:
            return
        for order_id in (position.stop_order, position.target_order, position.exit_order):
            if order_id is not None and order_id != fill.order_id:
                self.broker.cancel(order_id)
        trade = position.trade
        order = self.broker.get_order(fill.order_id)
        trade.exit_time = pd.Timestamp(fill.time)
        trade.exit_price = fill.price
        trade.exit_fee = fill.fee
        trade.tax = fill.tax
        trade.exit_reason = order.request.purpose.value
        trade.lowest = min(trade.lowest, fill.price)
        trade.highest = max(trade.highest, fill.price)
        self.trades.append(trade)

    def _submit_exit(self, symbol: str, order_type: OrderType, purpose: OrderPurpose, when: dt.datetime, *,
                     stop: float | None = None, limit: float | None = None, oco: str | None = None) -> str:
        position = self._open[symbol]
        order = self.broker.submit(OrderRequest(
            f"{self.name}-{next(self._client_seq)}", symbol, AssetClass(self.asset_class), Side.SELL,
            position.trade.qty, order_type, purpose, when, limit_price=limit, stop_price=stop, oco_group=oco,
        ))
        self._roles[order.order_id] = ("exit", symbol)
        return order.order_id

    # --- 보유 관리 -----------------------------------------------------------

    def _manage_open(self, symbol: str, bar: Bar, close_time: pd.Timestamp, atr_trail: float | None) -> None:
        position = self._open.get(symbol)
        if position is None:
            return
        trade = position.trade
        trade.bars_held += 1  # 진입 봉을 1봉으로 센다
        trade.lowest = min(trade.lowest, bar.low)
        trade.highest = max(trade.highest, bar.high)
        if position.exit_order is not None:
            return
        if self._session_closed(close_time):
            return self._exit_market(symbol, OrderPurpose.SESSION_FLATTEN, close_time)
        if position.frozen:
            return None
        time_exit = self.params.exit.time_exit
        if time_exit.enabled and trade.bars_held >= time_exit.max_bars:
            return self._exit_market(symbol, OrderPurpose.TIME_EXIT, close_time)
        new_stop = trailing_stop(trade.highest, atr_trail, self.params.exit.take_profit)
        if new_stop is not None:
            new_stop = self.broker.instrument_rules(symbol).floor_price(new_stop)
            if new_stop > position.stop:
                if position.stop_order is not None:
                    self.broker.cancel(position.stop_order)
                position.stop = new_stop
                trade.final_stop = new_stop
                position.stop_order = self._submit_exit(symbol, OrderType.STOP_MARKET, OrderPurpose.TRAILING_STOP,
                                                        close_time, stop=new_stop, oco=f"{self.name}-T{trade.trade_id}")
        return None

    def _exit_market(self, symbol: str, purpose: OrderPurpose, when: dt.datetime) -> None:
        position = self._open[symbol]
        for order_id in (position.stop_order, position.target_order):
            if order_id is not None:
                self.broker.cancel(order_id)
        position.stop_order = position.target_order = None
        position.exit_order = self._submit_exit(symbol, OrderType.MARKET, purpose, when)

    # --- 리스크 이벤트 대응 ---------------------------------------------------

    def evaluate_daily_loss(self, close_time: dt.datetime) -> list[RiskEvent]:
        return self.daily_loss.evaluate(close_time, self.broker.equity())

    def _cancel_pending(self, outcome: str) -> None:
        for symbol, entry in list(self._entries.items()):
            self.broker.cancel(entry.order_id)
            self._set_outcome(entry.record, outcome)
            del self._entries[symbol]
        for st in self._sym.values():
            if st.awaiting is not None:
                self._set_outcome(st.awaiting, outcome)
                st.awaiting = None
                st.confirmer.pending = None

    def on_kill_switch(self, on_trip: str, when: dt.datetime) -> None:
        """flatten_all: 전량 시장가 청산. keep_stops: 익절 주문 취소, 손절 주문만 유지."""
        self._cancel_pending("kill_switch")
        for symbol, position in self._open.items():
            if position.exit_order is not None:
                continue
            if on_trip == "flatten_all":
                self._exit_market(symbol, OrderPurpose.KILL_SWITCH, when)
            else:
                if position.target_order is not None:
                    self.broker.cancel(position.target_order)
                    position.target_order = None
                position.frozen = True

    def close_out(self, when: dt.datetime) -> None:
        """데이터 끝: 대기 주문·확인 대기를 정리하고 보유 포지션에 시장가 청산 주문을 낸다."""
        self._cancel_pending("data_end")
        for symbol, position in self._open.items():
            if position.exit_order is None:
                self._exit_market(symbol, OrderPurpose.END_OF_TEST, when)
