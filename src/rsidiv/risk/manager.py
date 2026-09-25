"""실거래 리스크 규칙 (설계 §8): 일일 손실 한도(계좌별)와 누적 MDD 킬 스위치.

판정은 봉 종가로 평가금액을 갱신한 뒤에 하고, 결과는 다음 봉의 신규 진입부터 적용된다.
API 연속 오류 중단은 실거래 전용이라 백테스트에는 없다(7단계).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from rsidiv.core.config import AssetClassName, DailyLossCfg, KillSwitchCfg, KrxCfg


@dataclass(frozen=True, slots=True)
class RiskEvent:
    time: dt.datetime  # 판정 시각 (봉 종가 확정 시각, UTC)
    kind: str  # daily_loss_trip | daily_loss_reset | kill_switch_trip
    scope: str  # 계좌 이름 또는 portfolio
    threshold: float
    value: float  # 실측 손실률·낙폭
    equity: float
    reference: float  # 당일 시작 평가금액 또는 최고점
    trip_id: str | None = None


class DailyLossGuard:
    """계좌 하나의 일일 손실 한도. 리셋 시각(코인 00:00 KST, 주식 장 시작)마다 자동 해제된다."""

    def __init__(self, cfg: DailyLossCfg, asset_class: AssetClassName, *, scope: str, enabled: bool,
                 krx: KrxCfg | None = None) -> None:
        self.cfg = cfg
        self.scope = scope
        self.enabled = enabled and cfg.enabled
        self._reset = cfg.reset[asset_class]
        if self._reset.type == "market_open" and krx is None:
            raise ValueError("장 시작 리셋에는 krx 설정이 필요합니다")
        self._krx = krx
        self._key: dt.date | None = None
        self._boundary: dt.datetime | None = None  # 다음 날로 넘어가는 시각 (매 봉 시간대 변환을 피하려고 캐시)
        self._start: float | None = None
        self._last: float | None = None
        self.blocked = False

    def _day_key(self, close_time: dt.datetime) -> dt.date:
        if self._reset.type == "clock":
            assert self._reset.time is not None and self._reset.timezone is not None
            local = close_time.astimezone(ZoneInfo(self._reset.timezone))
            shift = dt.timedelta(hours=self._reset.time.hour, minutes=self._reset.time.minute)
            return (local - shift).date()
        assert self._krx is not None
        bar_inside = close_time - dt.timedelta(microseconds=1)  # 봉 종료 시각이 아닌 봉 안의 날짜
        return bar_inside.astimezone(ZoneInfo(self._krx.timezone)).date()

    def _next_boundary(self, key: dt.date) -> dt.datetime:
        nxt = key + dt.timedelta(days=1)
        if self._reset.type == "clock":
            assert self._reset.time is not None and self._reset.timezone is not None
            return dt.datetime.combine(nxt, self._reset.time, tzinfo=ZoneInfo(self._reset.timezone))
        assert self._krx is not None
        midnight = dt.datetime.combine(nxt, dt.time(), tzinfo=ZoneInfo(self._krx.timezone))
        return midnight + dt.timedelta(microseconds=1)

    @property
    def day_start_equity(self) -> float | None:
        return self._start

    def evaluate(self, close_time: dt.datetime, equity: float) -> list[RiskEvent]:
        events: list[RiskEvent] = []
        if self._boundary is not None and close_time < self._boundary:
            key = self._key
        else:
            key = self._day_key(close_time)
            self._boundary = self._next_boundary(key)
        if key != self._key:
            if self.blocked:
                events.append(RiskEvent(close_time, "daily_loss_reset", self.scope, self.cfg.threshold, 0.0,
                                        equity, self._start or equity))
            self._key = key
            # 시각 리셋은 리셋 시각의 평가금액, 장 시작 리셋은 전날 마지막 평가금액이 기준
            self._start = equity if self._reset.type == "clock" or self._last is None else self._last
            self.blocked = False
        self._last = equity
        assert self._start is not None
        if self.enabled and not self.blocked and self._start > 0:
            loss = (self._start - equity) / self._start
            if loss >= self.cfg.threshold:
                self.blocked = True
                events.append(RiskEvent(close_time, "daily_loss_trip", self.scope, self.cfg.threshold, loss,
                                        equity, self._start))
        return events


class KillSwitch:
    """운용 시작 이후 최고점 대비 낙폭이 한도에 닿으면 발동. 자동 해제 없음 (release_ack 로만 해제)."""

    def __init__(self, cfg: KillSwitchCfg, *, scope: str, enabled: bool) -> None:
        self.cfg = cfg
        self.scope = scope
        self.enabled = enabled and cfg.enabled
        self.peak: float | None = None
        self.trip: RiskEvent | None = None

    @property
    def tripped(self) -> bool:
        return self.trip is not None and self.cfg.release_ack != self.trip.trip_id

    def evaluate(self, close_time: dt.datetime, equity: float) -> list[RiskEvent]:
        self.peak = equity if self.peak is None else max(self.peak, equity)
        if not self.enabled or self.trip is not None or self.peak <= 0:
            return []
        drawdown = 1 - equity / self.peak
        if drawdown < self.cfg.max_drawdown:
            return []
        trip_id = f"KS-{close_time.astimezone(dt.UTC):%Y%m%dT%H%M}Z"
        self.trip = RiskEvent(close_time, "kill_switch_trip", self.scope, self.cfg.max_drawdown, drawdown,
                              equity, self.peak, trip_id)
        return [self.trip]
