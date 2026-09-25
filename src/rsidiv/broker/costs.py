"""거래비용: 체결가(슬리피지), 수수료·세금, 펀딩비.

``config/costs.yaml`` 하나를 백테스트(SimBroker)·모의투자(PaperBroker)·사이징(1R 비용 추정)이 같이 쓴다.

- 가상화폐: 수수료 = 명목가 × 메이커/테이커 요율(BNB 할인 옵션). 슬리피지 = 가격 × 비율(불리한 방향).
- 국내주식: 수수료 = floor(명목가 × 요율) 원, 매도세 = floor(명목가 × 매도일 세율) 원. 슬리피지 = N호가.
- 선물 펀딩비: 롱 보유 중 정산 시 ``수량 × 표시가격 × 펀딩비율`` 을 낸다(음수면 받는다).

``include_fees=False`` 와 ``slippage_multiplier=0`` 을 함께 쓰면 비용 반영 전(gross) 성과가 된다.
"""

from __future__ import annotations

import datetime as dt
import math
from zoneinfo import ZoneInfo

from rsidiv.core.config import CostsCfg, CryptoMarketType, FeeTierCfg, KrxCfg
from rsidiv.core.models import AssetClass, InstrumentRules, Liquidity, OrderType, Side


class CostModel:
    def __init__(
        self,
        costs: CostsCfg,
        asset_class: AssetClass,
        *,
        market_type: CryptoMarketType = "spot",
        krx: KrxCfg | None = None,
        slippage_multiplier: float = 1.0,
        include_fees: bool = True,
    ) -> None:
        if asset_class is AssetClass.STOCK_KR and krx is None:
            raise ValueError("국내주식 비용 계산에는 krx 설정(호가단위·시간대)이 필요합니다")
        if slippage_multiplier < 0:
            raise ValueError("slippage_multiplier 는 0 이상이어야 합니다")
        self.costs = costs
        self.asset_class = asset_class
        self.market_type = market_type
        self.krx = krx
        self.slippage_multiplier = slippage_multiplier
        self.include_fees = include_fees

    @property
    def _tier(self) -> FeeTierCfg:
        crypto = self.costs.crypto
        return crypto.spot if self.market_type == "spot" else crypto.usdm_futures

    @property
    def funding_enabled(self) -> bool:
        return (
            self.include_fees
            and self.asset_class is AssetClass.CRYPTO
            and self.market_type == "usdm_futures"
            and self.costs.crypto.usdm_futures.funding.enabled
        )

    # --- 체결 구분 ---------------------------------------------------------

    def liquidity(self, order_type: OrderType, *, resting: bool = False) -> Liquidity:
        """주문 유형 → 메이커/테이커. 지정가는 호가창에 대기했다가 체결되면(resting) 메이커."""
        cfg = self.costs.order_liquidity
        if order_type is OrderType.MARKET:
            name = cfg.market
        elif order_type is OrderType.STOP_MARKET:
            name = cfg.stop_market
        else:
            name = cfg.limit_resting if resting else cfg.limit_marketable
        return Liquidity(name)

    # --- 슬리피지 -----------------------------------------------------------

    def fill_price(self, side: Side, price: float, liquidity: Liquidity, instrument: InstrumentRules) -> float:
        """기준 가격에 슬리피지를 불리한 방향으로 적용한 체결가."""
        if self.asset_class is AssetClass.STOCK_KR:
            stock = self.costs.stock_kr.slippage
            ticks = stock.market_ticks if liquidity is Liquidity.TAKER else stock.limit_ticks
            n = round(ticks * self.slippage_multiplier)
            for _ in range(n):
                if side is Side.BUY:
                    price += instrument.tick_size(price)
                else:
                    price -= instrument.tick_size(price - 1e-9)  # 가격대 경계 아래쪽 호가단위
            return price
        crypto = self.costs.crypto.slippage
        pct = (crypto.taker_pct if liquidity is Liquidity.TAKER else crypto.maker_pct) * self.slippage_multiplier
        return price * (1 + pct) if side is Side.BUY else price * (1 - pct)

    # --- 수수료·세금 -------------------------------------------------------

    def fee_rates(self, side: Side, liquidity: Liquidity, when: dt.datetime) -> tuple[float, float]:
        """(수수료율, 세율). 사이징 추정용이며 원 단위 절사 전 값이다."""
        if not self.include_fees:
            return 0.0, 0.0
        if self.asset_class is AssetClass.STOCK_KR:
            stock = self.costs.stock_kr
            tax = stock.sell_tax.rate_on(self._krx_day(when)) if side is Side.SELL else 0.0
            return stock.commission.rate, tax
        return self._tier.rate(liquidity.value), 0.0

    def fees(
        self, side: Side, qty: float, price: float, liquidity: Liquidity, when: dt.datetime
    ) -> tuple[float, float]:
        """체결 1건의 (수수료, 세금). 국내주식은 각각 원 미만 절사."""
        fee_rate, tax_rate = self.fee_rates(side, liquidity, when)
        notional = qty * price
        if self.asset_class is AssetClass.STOCK_KR:
            return float(math.floor(notional * fee_rate)), float(math.floor(notional * tax_rate))
        return notional * fee_rate, notional * tax_rate

    def _krx_day(self, when: dt.datetime) -> dt.date:
        assert self.krx is not None
        return when.astimezone(ZoneInfo(self.krx.timezone)).date()

    # --- 펀딩비 -------------------------------------------------------------

    def funding_payment(self, qty: float, mark_price: float, rate: float) -> float:
        """롱 포지션이 내는 펀딩비 (양수 = 비용, 음수 = 수익). 현물·비용 제외 시 0."""
        if not self.funding_enabled:
            return 0.0
        return qty * mark_price * rate
