"""진입 확인(A/B/C)과 손절·익절·트레일링 가격 규칙 (strategy.yaml 의 entry·exit).

- 손절: ``atr`` = Low(t3) − ATR(신호 봉) × atr_mult, ``pct`` = Low(t3) × (1 − pct).
  Low(t3) 는 신호의 가격 기준(``pivot.price_source``)으로 잰 t3 가격이다.
- 익절: ``r_multiple`` = 진입가 + r × (진입가 − 손절가). 진입가는 실제 체결가.
  ``p2`` = High(p2). p2 가 진입 기준가 이하이면 ``skip``(진입 안 함) 또는 ``fallback_r_multiple``.
  ``trailing`` = 고정 목표 없이 진입 후 최고가 − ATR × mult 로 손절을 올린다 (봉 마감 후 갱신, 다음 봉부터 적용).
  ``none`` = 목표 없음 (손절·시간 청산만).
- 진입 확인: A = 신호 봉 종가 후 바로. B = 신호 뒤 N봉 안에 종가 > 직전 봉 고가. C = 신호 뒤 N봉 안에
  RSI 가 기준값을 아래에서 위로 돌파 (직전 RSI < 기준 ≤ 현재 RSI). B·C 대기 중 저가가 손절가 이하면 취소.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rsidiv.core.config import EntryCfg, StopCfg, TakeProfitCfg
from rsidiv.core.models import Bar
from rsidiv.signals.divergence import DivergenceCandidate


def initial_stop(candidate: DivergenceCandidate, atr: float | None, cfg: StopCfg) -> float | None:
    """손절가. ATR 이 아직 없으면(워밍업) None."""
    if cfg.mode == "atr":
        return None if atr is None else candidate.price_t3 - atr * cfg.atr_mult
    return candidate.price_t3 * (1 - cfg.pct)


def p2_blocks_entry(reference_price: float, candidate: DivergenceCandidate, cfg: TakeProfitCfg) -> bool:
    return (cfg.mode == "p2" and cfg.p2_below_entry == "skip"
            and not candidate.price_p2 > reference_price)


def target_price(entry_price: float, stop: float, candidate: DivergenceCandidate,
                 cfg: TakeProfitCfg) -> float | None:
    """고정 익절 목표가. 트레일링·없음이면 None."""
    r_target = entry_price + cfg.r_multiple * (entry_price - stop)
    if cfg.mode == "r_multiple":
        return r_target
    if cfg.mode == "p2":
        if candidate.price_p2 > entry_price or cfg.p2_below_entry == "skip":
            return candidate.price_p2  # skip 인데 체결가가 p2 이상으로 열린 경우: 목표가 즉시 체결 가능
        return r_target
    return None


def trailing_stop(highest: float, atr: float | None, cfg: TakeProfitCfg) -> float | None:
    if cfg.mode != "trailing" or atr is None:
        return None
    return highest - atr * cfg.trailing_atr_mult


@dataclass(slots=True)
class PendingEntry:
    candidate: DivergenceCandidate
    stop: float
    signal_index: int
    expires_index: int  # 이 봉까지 확인되지 않으면 만료


class EntryConfirmer:
    """종목 하나의 진입 확인 대기 상태 (모드 B·C). 모드 A 는 대기하지 않는다."""

    def __init__(self, cfg: EntryCfg) -> None:
        self.cfg = cfg
        self.pending: PendingEntry | None = None

    @property
    def immediate(self) -> bool:
        return self.cfg.mode == "A"

    def start(self, candidate: DivergenceCandidate, stop: float, signal_index: int) -> None:
        self.pending = PendingEntry(candidate, stop, signal_index, signal_index + self.cfg.confirm_window_bars)

    def on_bar(self, index: int, bar: Bar, prev_bar: Bar, rsi_prev: float, rsi_now: float) -> str | None:
        """대기 중 신호의 이번 봉 결과: 'enter' | 'canceled_below_stop' | 'not_confirmed' | None(계속 대기)."""
        pending = self.pending
        if pending is None or index <= pending.signal_index:
            return None
        outcome: str | None = None
        if self.cfg.cancel_if_below_stop and bar.low <= pending.stop:
            outcome = "canceled_below_stop"
        elif self.cfg.mode == "B" and bar.close > prev_bar.high:
            outcome = "enter"
        elif (self.cfg.mode == "C" and not math.isnan(rsi_prev)
              and rsi_prev < self.cfg.rsi_cross_level <= rsi_now):
            outcome = "enter"
        elif index >= pending.expires_index:
            outcome = "not_confirmed"
        if outcome is not None and outcome != "enter":
            self.pending = None
        return outcome

    def take(self) -> PendingEntry:
        assert self.pending is not None
        pending, self.pending = self.pending, None
        return pending
