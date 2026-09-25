"""포지션 사이징 (설계 §7).

    위험금액  = risk_per_trade × 기준 자본 (슬리브 평가금액 또는 최초 배분액)
    1R(단위당) = 예상 진입 체결가×(1+매수수수료율) − 예상 손절 체결가×(1−매도수수료율−세율)   ← include_costs_in_risk
               = 기준가 − 손절가                                                              ← 비용 제외 시
    수량      = floor_to_lot(위험금액 / 1R)
    상한      = min(max_weight × 슬리브 평가금액, 가용현금 × (1 − cash_buffer)) / 단위당 매수 비용
    최소 주문단위·최소 주문금액 미만이면 '자본 부족 스킵'

기준가는 신호(또는 진입 확인) 봉 종가다. 다음 봉 시가는 주문 시점에 알 수 없기 때문이다.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from rsidiv.broker.costs import CostModel
from rsidiv.core.config import AssetLimitsCfg, SizingCfg
from rsidiv.core.models import InstrumentRules, Liquidity, Side


@dataclass(frozen=True, slots=True)
class SizingResult:
    qty: float
    skip_reason: str | None  # None 이면 진입
    capped: bool  # 비중 상한·가용현금 때문에 위험 기준 수량보다 줄었는지
    risk_amount: float  # 목표 위험금액
    unit_risk: float  # 단위당 예상 손실 (1R)
    unit_cost: float  # 단위당 예상 매수 비용 (현금 예약용)

    @property
    def accepted(self) -> bool:
        return self.skip_reason is None


def size_position(
    *,
    cfg: SizingCfg,
    limits: AssetLimitsCfg,
    equity: float,
    initial_equity: float,
    available_cash: float,
    reference_price: float,
    stop: float,
    instrument: InstrumentRules,
    costs: CostModel,
    when: dt.datetime,
) -> SizingResult:
    def skip(reason: str, *, risk: float = 0.0, unit_risk: float = 0.0, unit_cost: float = 0.0,
             capped: bool = False) -> SizingResult:
        return SizingResult(0.0, reason, capped, risk, unit_risk, unit_cost)

    if stop >= reference_price:
        return skip("stop_above_entry")
    base = equity if cfg.capital_base == "sleeve_equity" else initial_equity
    risk_amount = cfg.risk_per_trade * base

    entry_est = costs.fill_price(Side.BUY, reference_price, Liquidity.TAKER, instrument)
    buy_fee, _ = costs.fee_rates(Side.BUY, Liquidity.TAKER, when)
    unit_cost = entry_est * (1 + buy_fee)
    if cfg.include_costs_in_risk:
        exit_est = costs.fill_price(Side.SELL, stop, Liquidity.TAKER, instrument)
        sell_fee, sell_tax = costs.fee_rates(Side.SELL, Liquidity.TAKER, when)
        unit_risk = unit_cost - exit_est * (1 - sell_fee - sell_tax)
    else:
        unit_risk = reference_price - stop
    if unit_risk <= 0:
        return skip("stop_above_entry", risk=risk_amount)

    qty_risk = risk_amount / unit_risk
    cap_value = min(limits.max_weight * equity, max(available_cash, 0.0) * (1 - cfg.cash_buffer))
    qty_cap = cap_value / unit_cost
    capped = qty_risk > qty_cap
    if capped and cfg.on_cap_exceeded == "skip":
        return skip("cap_exceeded", risk=risk_amount, unit_risk=unit_risk, unit_cost=unit_cost, capped=True)
    qty = instrument.round_qty(min(qty_risk, qty_cap))
    if qty < instrument.min_qty or qty <= 0 or qty * reference_price < instrument.min_notional:
        return skip("insufficient_capital", risk=risk_amount, unit_risk=unit_risk, unit_cost=unit_cost,
                    capped=capped)
    return SizingResult(qty, None, capped, risk_amount, unit_risk, unit_cost)
