"""종목별 거래 규칙 (:class:`rsidiv.core.models.InstrumentRules` 구현).

수량·가격 반올림은 ``Decimal`` 로 계산해 ``0.1 + 0.2`` 같은 부동소수점 오차로 한 단위가 깎이거나
최소 단위 아래로 떨어지는 일을 막는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from rsidiv.core.config import KrxCfg, SymbolFilter
from rsidiv.core.models import AssetClass


def _to_step(value: float, step: float, rounding: str) -> float:
    if step <= 0:
        raise ValueError("step 은 양수여야 합니다")
    d_step = Decimal(repr(step))
    units = (Decimal(repr(value)) / d_step).to_integral_value(rounding=rounding)
    return float(units * d_step)


def floor_to_step(value: float, step: float) -> float:
    return _to_step(value, step, ROUND_FLOOR)


def ceil_to_step(value: float, step: float) -> float:
    return _to_step(value, step, ROUND_CEILING)


@dataclass(frozen=True, slots=True)
class CryptoInstrument:
    """바이낸스 현물·선물 심볼 (LOT_SIZE·PRICE_FILTER·NOTIONAL)."""

    symbol: str
    min_qty: float
    qty_step: float
    min_notional: float
    price_tick: float
    asset_class: AssetClass = AssetClass.CRYPTO

    @classmethod
    def from_filter(cls, symbol: str, f: SymbolFilter) -> CryptoInstrument:
        return cls(symbol, f.min_qty, f.qty_step, f.min_notional, f.price_tick)

    def round_qty(self, qty: float) -> float:
        return floor_to_step(max(qty, 0.0), self.qty_step)

    def floor_price(self, price: float) -> float:
        return floor_to_step(price, self.price_tick)

    def ceil_price(self, price: float) -> float:
        return ceil_to_step(price, self.price_tick)

    def tick_size(self, price: float) -> float:
        return self.price_tick


@dataclass(frozen=True, slots=True)
class KrxInstrument:
    """KRX 주식: 1주 단위, 가격대별 호가단위."""

    symbol: str
    krx: KrxCfg
    min_notional: float = 0.0
    asset_class: AssetClass = AssetClass.STOCK_KR

    @property
    def min_qty(self) -> float:
        return float(self.krx.lot_size)

    def round_qty(self, qty: float) -> float:
        return floor_to_step(max(qty, 0.0), float(self.krx.lot_size))

    def tick_size(self, price: float) -> float:
        return self.krx.tick_size(price)

    def floor_price(self, price: float) -> float:
        return floor_to_step(price, self.krx.tick_size(price))

    def ceil_price(self, price: float) -> float:
        # 올림 결과가 다음 가격대로 넘어가면 그 가격대의 호가단위에 다시 맞춘다 (예: 1,999.5 → 2,000)
        up = ceil_to_step(price, self.krx.tick_size(price))
        return ceil_to_step(up, self.krx.tick_size(up))
