"""심볼 해석과 유니버스 분류.

아카이브·거래소의 USDⓈ-M 심볼 이름 → (코인, 증거금 자산, 정산 여부).
- ``BTCUSDT`` → (BTC, USDT). ``1000PEPEUSDC`` → (1000PEPE, USDC). ``ETHUSD1`` → (ETH, USD1).
- ``AERGOUSDTSETTLED`` → (AERGO, USDT, settled): 상장폐지·재상장 때 옛 시리즈가 이 이름으로 옮겨진다.
- 분기물(``BTCUSDT_240329``)과 코인 증거금·코인 호가(``ETHBTC``)는 무기한 스테이블 계약이 아니므로 None.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from perpdiv.core.config import UniverseCfg

KNOWN_QUOTES = ("USDT", "USDC", "USD1", "BUSD")


@dataclass(frozen=True, slots=True)
class Contract:
    symbol: str  # 아카이브·거래소 원래 이름
    base: str  # 코인 (1000PEPE 등 접두 포함)
    quote: str  # 증거금 자산
    settled: bool  # ...SETTLED 시리즈

    @property
    def ccxt_symbol(self) -> str:
        return f"{self.base}/{self.quote}:{self.quote}"


def parse_contract(symbol: str) -> Contract | None:
    if "_" in symbol:  # 분기물
        return None
    stripped = re.sub(r"(SETTLED)+$", "", symbol)
    settled = stripped != symbol
    for quote in KNOWN_QUOTES:
        if stripped.endswith(quote) and len(stripped) > len(quote):
            return Contract(symbol, stripped[: -len(quote)], quote, settled)
    return None


def excluded_reason(base: str, cfg: UniverseCfg) -> str | None:
    """코인이 순위·거래 대상에서 빠지는 이유 (없으면 None)."""
    if base in cfg.exclude.stablecoin_bases:
        return "stablecoin"
    if base in cfg.exclude.tradfi_bases:
        return "tradfi"
    if base in cfg.exclude.symbols:
        return "manual"
    return None


def rank_contracts(symbols: list[str], cfg: UniverseCfg) -> dict[str, list[Contract]]:
    """순위 합산 대상 계약을 코인별로 묶는다 (제외 코인·다른 증거금 자산은 뺀다)."""
    out: dict[str, list[Contract]] = {}
    for symbol in symbols:
        contract = parse_contract(symbol)
        if contract is None or contract.quote not in cfg.ranking.rank_quote_assets:
            continue
        if excluded_reason(contract.base, cfg) is not None:
            continue
        out.setdefault(contract.base, []).append(contract)
    return out


def trading_contract(contracts: list[Contract], cfg: UniverseCfg) -> Contract | None:
    """코인의 주문용 계약 (USDT 무기한, 정산 시리즈가 아닌 것 우선)."""
    usdt = [c for c in contracts if c.quote == cfg.trading.quote_asset]
    return next((c for c in usdt if not c.settled), usdt[0] if usdt else None)
