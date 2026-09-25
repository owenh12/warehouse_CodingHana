"""백테스트·모의투자·실거래가 공유하는 도메인 모델.

가격·수량은 float 로 다루고, 호가단위·수량단위 반올림은 :class:`InstrumentRules` 구현체가
정수 배수 연산으로 처리한다. 모든 시각은 UTC tz-aware 이다.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from rsidiv.core.timeutil import ensure_utc


class AssetClass(StrEnum):
    STOCK_KR = "stock_kr"
    CRYPTO = "crypto"


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"


class OrderStatus(StrEnum):
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class Liquidity(StrEnum):
    MAKER = "maker"
    TAKER = "taker"


class OrderPurpose(StrEnum):
    """주문 목적. 거래 내역의 청산 사유와 리스크 이벤트 기록에 사용한다."""

    ENTRY = "entry"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    TIME_EXIT = "time_exit"
    SESSION_FLATTEN = "session_flatten"
    KILL_SWITCH = "kill_switch"


class InstrumentRules(Protocol):
    """종목별 거래 규칙 (KRX 호가단위·1주 단위, 바이낸스 LOT_SIZE·MIN_NOTIONAL 등)."""

    symbol: str
    asset_class: AssetClass
    min_qty: float
    min_notional: float

    def round_qty(self, qty: float) -> float:
        """주문 가능한 수량으로 내림한다 (0 이 될 수 있음)."""
        ...

    def round_price(self, price: float, side: Side) -> float:
        """호가단위에 맞춘다. 매수는 올림·매도는 내림 등 불리한 방향으로 맞춘다."""
        ...

    def tick_size(self, price: float) -> float:
        """해당 가격의 호가단위."""
        ...


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """전략·리스크 계층이 브로커에 넘기는 주문 요청."""

    client_id: str
    symbol: str
    asset_class: AssetClass
    side: Side
    qty: float
    order_type: OrderType
    purpose: OrderPurpose
    created_at: dt.datetime
    limit_price: float | None = None
    stop_price: float | None = None

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError("qty 는 양수여야 합니다")
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("지정가 주문에는 limit_price 가 필요합니다")
        if self.order_type is OrderType.STOP_MARKET and self.stop_price is None:
            raise ValueError("스탑 주문에는 stop_price 가 필요합니다")
        object.__setattr__(self, "created_at", ensure_utc(self.created_at))


@dataclass(frozen=True, slots=True)
class Fill:
    """체결 1건. 수수료·세금은 계좌 통화 기준이며 반올림(원 단위 절사 등) 적용 후 값이다."""

    order_id: str
    symbol: str
    side: Side
    qty: float
    price: float
    fee: float
    tax: float
    liquidity: Liquidity
    time: dt.datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", ensure_utc(self.time))


@dataclass(slots=True)
class Order:
    """브로커가 관리하는 주문 상태."""

    order_id: str
    request: OrderRequest
    status: OrderStatus
    updated_at: dt.datetime
    filled_qty: float = 0.0
    avg_fill_price: float | None = None
    fills: list[Fill] = field(default_factory=list)

    @property
    def remaining_qty(self) -> float:
        return self.request.qty - self.filled_qty

    @property
    def is_open(self) -> bool:
        return self.status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED)


@dataclass(slots=True)
class Position:
    """보유 포지션(롱 전용). 보호 주문 가격은 전략 계층이 관리한다."""

    symbol: str
    asset_class: AssetClass
    qty: float
    avg_price: float
    opened_at: dt.datetime
    stop_price: float
    target_price: float | None = None
    bars_held: int = 0
    highest_close: float | None = None
