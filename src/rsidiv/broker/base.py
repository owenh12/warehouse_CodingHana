"""주문 실행 인터페이스.

전략·리스크 계층은 이 인터페이스에만 의존한다. 백테스트(SimBroker), 모의투자(PaperBroker),
실거래(KISBroker, CcxtBroker)는 구현체만 교체하므로 신호 → 사이징 → 주문 경로가 동일하다.
금액은 모두 해당 계좌 통화(KRW 또는 USDT) 기준이다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rsidiv.core.models import AssetClass, InstrumentRules, Order, OrderRequest, Position


class BrokerError(RuntimeError):
    """브로커 API 호출 실패. 리스크 계층의 API 연속 오류 카운터가 이 예외를 센다."""


class Broker(ABC):
    """한 계좌(자산군 1개)에 대한 주문·잔고 조회."""

    asset_class: AssetClass

    @abstractmethod
    def submit(self, request: OrderRequest) -> Order:
        """주문 제출. 거부 시 status=REJECTED 인 Order 를 반환하고, 통신 실패는 BrokerError."""

    @abstractmethod
    def cancel(self, order_id: str) -> Order:
        """주문 취소 후 최신 상태 반환 (이미 체결된 수량은 유지)."""

    @abstractmethod
    def get_order(self, order_id: str) -> Order:
        """주문 상태 조회 (부분체결 포함)."""

    @abstractmethod
    def open_orders(self, symbol: str | None = None) -> list[Order]:
        """미체결 주문 목록."""

    @abstractmethod
    def positions(self) -> dict[str, Position]:
        """심볼 → 보유 포지션."""

    @abstractmethod
    def cash(self) -> float:
        """주문 가능 현금."""

    @abstractmethod
    def equity(self) -> float:
        """계좌 평가금액 = 현금 + 보유 포지션 평가액."""

    @abstractmethod
    def instrument_rules(self, symbol: str) -> InstrumentRules:
        """호가단위·최소 주문단위·최소 주문금액."""
