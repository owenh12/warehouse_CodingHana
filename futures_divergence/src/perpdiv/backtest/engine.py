"""이벤트 기반 백테스트 엔진 (DESIGN §5·§6·§8·§10).

신호 이벤트를 시각 순서로 처리한다. 같은 시각의 신호는 상위 TF → 순위 상위 순. 포지션은 최대 1개이며, 보유 중에는
그 심볼의 15분 실행 봉만 진행해 청산을 판정한다.

체결 규칙
- 진입: 신호 시각(= t3 마감)에 시작하는 15분봉 시가, 시장가(테이커) + 불리한 슬리피지.
  수량 = 평가금액 × equity_fraction × 레버리지 × (1 − fee_buffer) ÷ 신호 봉 종가, 수량 단위로 내림.
- 진입 조건(전략 수정 후): 신호 시각 순위 ≤ top_n, 같은 코인·방향으로 유효한 서로 다른 TF 신호 ≥ min_timeframes
  (signals/confluence.py), 손절폭 > 진입가 × min_distance_pct (신호 봉 종가로 판정).
- 손절: 진입가 ∓ ATR(확정 신호 TF)×mult (basis=entry_price) 또는 Low(t3)/High(p3) 기준(signal_bar).
  스탑마켓(테이커 + 슬리피지). 시가가 이미 손절가보다 불리하면 시가 체결(갭).
- 익절(r_multiple·structure): 대기 지정가(메이커, 슬리피지 없음), 가격이 닿으면 지정가 체결.
- 트레일링: 손절선이 15분봉 마감마다 (진입 후 최고가 − ATR×배수) 로 올라가고(숏 대칭) 다음 봉부터 적용. 익절 주문 없음.
- 같은 봉에서 손절·익절이 모두 닿으면 손절 우선. 정밀 모드면 그 15분 구간의 1분봉으로 순서를 판정.
- 시간 청산: 진입 + 신호 TF × bars 시각에 시작하는 15분봉 시가, 시장가.
- 펀딩비: 정산 시각 F 에 진입 < F ≤ 청산 이면 수량 × mark(F) × rate (롱은 양수면 지불).
- 청산가(레버리지 > 1, 격리): 손절보다 먼저 닿으면 청산가에 강제 청산.
- 상장폐지: 봉이 끝나면 마지막 체결가로 청산. 백테스트 끝까지 보유 중이면 마지막 종가로 정리.

리스크 규칙(시나리오 live_rules): 일일 손실(00:00 KST 기준 평가금액 대비) → 그날 신규 진입 중단,
MDD 킬 스위치 → 이후 신규 진입 영구 중단 (flatten_all 이면 다음 봉 시가 청산, keep_stops 면 기존 청산 규칙 유지).
평가금액은 15분봉 마감(보유 중)과 이벤트 시각에 갱신한다 (봉 내부 평가는 하지 않음).
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import asdict, dataclass, field
from typing import Literal, Protocol
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from perpdiv.backtest.instruments import Instrument
from perpdiv.backtest.market import ExecBars
from perpdiv.core.config import Settings, timeframe_minutes
from perpdiv.signals.confluence import annotate_confluence

ExitReason = Literal["stop", "stop_gap", "target", "trailing_stop", "time", "liquidation", "delisted",
                     "end_of_period", "kill_switch", "switch", "target_passed"]


class MarketLike(Protocol):
    def exec_bars(self, symbol: str, start: dt.datetime, end: dt.datetime) -> ExecBars: ...

    def funding(self, symbol: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame: ...

    def minute_bars(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame: ...


class InstrumentsLike(Protocol):
    def get(self, symbol: str, month: str) -> Instrument: ...


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    fees: bool = True
    slippage_mult: float = 1.0
    funding: bool = True
    live_rules: bool = True
    precise: bool = False


@dataclass(slots=True)
class _Position:
    signal_id: int
    coin: str
    symbol: str
    timeframe: str
    side: int  # +1 롱, −1 숏
    rank: float
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float
    confluence_tfs: str
    qty: float
    stop: float
    initial_stop: float
    target: float | None
    trail_distance: float | None
    extreme: float
    time_exit_at: pd.Timestamp
    liq_price: float | None
    bars: ExecBars
    funding: pd.DataFrame
    equity_before: float
    entry_fee: float
    bar_pos: int = 0
    funding_pos: int = 0
    funding_paid: float = 0.0
    slippage_cost: float = 0.0
    flatten_pending: bool = False
    ambiguous_bars: int = 0
    precise_used: bool = False


@dataclass(frozen=True, slots=True)
class Trade:
    trade_id: int
    signal_id: int
    coin: str
    symbol: str
    timeframe: str
    side: str
    rank: float
    confluence_tfs: str
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    notional: float
    stop: float
    target: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str
    gross_pnl: float
    fees: float
    funding: float
    slippage_cost: float
    net_pnl: float
    risk_amount: float
    r_multiple: float
    equity_before: float
    equity_after: float
    holding_hours: float
    ambiguous_bars: int
    precise_used: bool


@dataclass(frozen=True, slots=True)
class RiskEvent:
    time: pd.Timestamp
    kind: str  # daily_loss | kill_switch
    equity: float
    reference: float  # 당일 시작 평가금액 / 최고 평가금액
    value: float  # 손실률 / 낙폭
    detail: str


@dataclass(slots=True)
class BacktestResult:
    scenario: Scenario
    trades: pd.DataFrame
    signals: pd.DataFrame  # 입력 신호 + status
    equity: pd.Series  # 시각 → 평가금액 (이벤트·보유 중 15분봉 마감)
    risk_events: list[RiskEvent] = field(default_factory=list)
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    initial_capital: float = 0.0


class BacktestEngine:
    def __init__(self, settings: Settings, market: MarketLike, instruments: InstrumentsLike, scenario: Scenario,
                 *, start: dt.datetime, end: dt.datetime) -> None:
        self.s = settings
        self.market = market
        self.instruments = instruments
        self.sc = scenario
        self.start, self.end = pd.Timestamp(start), pd.Timestamp(end)
        self.exec_step = pd.Timedelta(minutes=timeframe_minutes(settings.data.timeframes.execution))
        costs = settings.costs
        self.taker = costs.fees.rate("taker") if scenario.fees else 0.0
        self.maker = costs.fees.rate("maker") if scenario.fees else 0.0
        self.slip_taker = costs.slippage.taker_pct * scenario.slippage_mult
        self.slip_maker = costs.slippage.maker_pct * scenario.slippage_mult
        self.leverage = float(settings.base.exchange.leverage)
        risk = settings.risk
        self.liq_on = risk.liquidation.enabled and self.leverage > 1
        self.tz = ZoneInfo(risk.daily_loss.reset_timezone)
        reset = risk.daily_loss.reset_time
        self.reset_offset = pd.Timedelta(hours=reset.hour, minutes=reset.minute)
        # 상태
        self.cash = float(settings.base.account.initial_capital)
        self.pos: _Position | None = None
        self.trades: list[Trade] = []
        self.equity_points: list[tuple[pd.Timestamp, float]] = [(self.start, self.cash)]
        self.events: list[RiskEvent] = []
        self.peak = self.cash
        self.kill_tripped = False
        self.day_key: pd.Timestamp | None = None
        self.day_start_equity = self.cash
        self.blocked_until: pd.Timestamp | None = None

    # --- 가격·비용 ---------------------------------------------------------------------------------------

    def _fill(self, price: float, side: int, taker: bool) -> float:
        """side: 체결 방향(+1 매수, −1 매도). 불리한 슬리피지."""
        slip = self.slip_taker if taker else self.slip_maker
        return price * (1 + slip * side)

    def _fee(self, qty: float, price: float, taker: bool) -> float:
        return qty * price * (self.taker if taker else self.maker)

    # --- 평가금액·리스크 ----------------------------------------------------------------------------------

    def _equity(self, mark: float | None = None) -> float:
        if self.pos is None or mark is None:
            return self.cash
        p = self.pos
        return self.cash + p.side * p.qty * (mark - p.entry_price)

    def _day_of(self, t: pd.Timestamp) -> pd.Timestamp:
        local = t.tz_convert(self.tz)
        return (local - self.reset_offset).normalize()

    def _next_reset(self, t: pd.Timestamp) -> pd.Timestamp:
        day = self._day_of(t)
        return (day + pd.Timedelta(days=1) + self.reset_offset).tz_convert("UTC")

    def _update(self, t: pd.Timestamp, equity: float) -> None:
        """평가금액 기록 + 일일 손실·킬 스위치 판정."""
        self.equity_points.append((t, equity))
        key = self._day_of(t)
        if key != self.day_key:
            self.day_key = key
            self.day_start_equity = equity
        self.peak = max(self.peak, equity)
        if not self.sc.live_rules:
            return
        risk = self.s.risk
        if risk.daily_loss.enabled and self.day_start_equity > 0:
            loss = (self.day_start_equity - equity) / self.day_start_equity
            if loss >= risk.daily_loss.threshold and (self.blocked_until is None or t >= self.blocked_until):
                self.blocked_until = self._next_reset(t)
                self.events.append(RiskEvent(t, "daily_loss", equity, self.day_start_equity, loss,
                                             f"신규 진입 중단 ~ {self.blocked_until}"))
        if risk.kill_switch.enabled and not self.kill_tripped and self.peak > 0:
            dd = 1 - equity / self.peak
            if dd >= risk.kill_switch.max_drawdown:
                self.kill_tripped = True
                self.events.append(RiskEvent(t, "kill_switch", equity, self.peak, dd, risk.kill_switch.on_trip))
                if risk.kill_switch.on_trip == "flatten_all" and self.pos is not None:
                    self.pos.flatten_pending = True

    # --- 진입 --------------------------------------------------------------------------------------------

    def _stop_price(self, sig: pd.Series, entry: float) -> float:
        """손절가: basis=entry_price 면 진입가 ∓ ATR×mult, signal_bar 면 Low(t3) − ATR×mult (숏 High(p3) + …)."""
        stop_cfg = self.s.strategy.exit.stop
        side = 1 if sig["side"] == "long" else -1
        distance = stop_cfg.atr_mult * float(sig["trigger_atr"])
        if stop_cfg.basis == "entry_price":
            return entry - side * distance
        return float(sig["trigger_low"]) - distance if side > 0 else float(sig["trigger_high"]) + distance

    def _stop_too_tight(self, sig: pd.Series) -> bool:
        """진입 조건: 손절폭 > 진입가 × min_distance_pct. 주문 전에 판정하므로 신호 봉 종가를 진입가로 본다."""
        ref = float(sig["trigger_close"])
        distance = abs(ref - self._stop_price(sig, ref))
        return not distance > self.s.strategy.exit.stop.min_distance_pct * ref

    def _enter(self, sid: int, sig: pd.Series, t: pd.Timestamp) -> str:
        cfg = self.s
        side = 1 if sig["side"] == "long" else -1
        hold = pd.Timedelta(minutes=int(sig["tf_minutes"]) * cfg.strategy.exit.time_exit.bars)
        horizon = min(t + hold + pd.Timedelta(days=1), self.end)
        bars = self.market.exec_bars(sig["symbol"], t.to_pydatetime(), horizon.to_pydatetime())
        if bars.bars.empty or bars.bars.index[0] != t:
            return "skipped_no_data"
        inst = self.instruments.get(sig["symbol"], f"{t:%Y-%m}")
        equity = self.cash
        notional_target = equity * cfg.risk.sizing.equity_fraction * self.leverage * (1 - cfg.risk.sizing.fee_buffer)
        qty = inst.round_qty(notional_target / float(sig["trigger_close"]))
        if qty <= 0 or qty * float(sig["trigger_close"]) < inst.min_notional:
            return "skipped_min_notional"
        open_ = float(bars.bars["open"].iloc[0])
        price = self._fill(open_, side, taker=True)
        fee = self._fee(qty, price, taker=True)
        atr = float(sig["trigger_atr"])
        stop = self._stop_price(sig, price)
        tp = cfg.strategy.exit.take_profit
        target: float | None
        trail: float | None = None
        if tp.mode == "r_multiple":
            target = price + side * tp.r_multiple * abs(price - stop)
        elif tp.mode == "structure":
            target = float(sig["swing_price"])
        else:
            target = None
            trail = tp.trailing_atr_mult * atr
        liq = None
        if self.liq_on:
            mmr = cfg.risk.liquidation.maintenance_margin_rate
            liq = price * (1 - 1 / self.leverage) / (1 - mmr) if side > 0 else price * (1 + 1 / self.leverage) / (1 + mmr)
        funding = (self.market.funding(sig["symbol"], t.to_pydatetime(), horizon.to_pydatetime())
                   if self.sc.funding else pd.DataFrame({"rate": [], "mark": []}))
        self.cash -= fee
        self.pos = _Position(
            signal_id=sid, coin=sig["coin"], symbol=sig["symbol"], timeframe=sig["timeframe"], side=side,
            rank=float(sig["rank"]), signal_time=pd.Timestamp(sig["signal_time"]), entry_time=t, entry_price=price,
            confluence_tfs=str(sig.get("confluence_tfs", sig["timeframe"])),
            qty=qty, stop=stop, initial_stop=stop, target=target, trail_distance=trail, extreme=open_,
            time_exit_at=t + hold, liq_price=liq, bars=bars, funding=funding, equity_before=equity, entry_fee=fee,
            slippage_cost=qty * abs(price - open_),
        )
        self._update(t, self._equity(open_))
        if target is not None and side * (target - price) <= 0:  # 구조 목표가 이미 지나 있음 → 즉시 청산
            self._close(t, open_, "target_passed", taker=True)
        return "entered"

    # --- 청산 --------------------------------------------------------------------------------------------

    def _apply_funding(self, until: pd.Timestamp) -> None:
        p = self.pos
        assert p is not None
        f = p.funding
        while p.funding_pos < len(f) and f.index[p.funding_pos] <= until:
            rate, mark = float(f["rate"].iloc[p.funding_pos]), float(f["mark"].iloc[p.funding_pos])
            if f.index[p.funding_pos] > p.entry_time and not math.isnan(mark):
                amount = p.qty * mark * rate * p.side  # 롱: 양수 펀딩 = 지불
                self.cash -= amount
                p.funding_paid += amount
            p.funding_pos += 1

    def _close(self, t: pd.Timestamp, price: float, reason: ExitReason, *, taker: bool, raw: bool = False) -> None:
        """price: 기준가(슬리피지 전). raw=True 면 그대로 체결 (청산가·상장폐지)."""
        p = self.pos
        assert p is not None
        self._apply_funding(t)
        fill = price if raw else self._fill(price, -p.side, taker)
        fee = self._fee(p.qty, fill, taker)
        gross = p.side * p.qty * (fill - p.entry_price)
        self.cash += gross - fee
        p.slippage_cost += p.qty * abs(fill - price)
        risk_amount = p.qty * abs(p.entry_price - p.initial_stop)
        net = gross - fee - p.entry_fee - p.funding_paid
        self.trades.append(Trade(
            trade_id=len(self.trades) + 1, signal_id=p.signal_id, coin=p.coin, symbol=p.symbol, timeframe=p.timeframe,
            side="long" if p.side > 0 else "short", rank=p.rank, confluence_tfs=p.confluence_tfs,
            signal_time=p.signal_time, entry_time=p.entry_time,
            entry_price=p.entry_price, qty=p.qty, notional=p.qty * p.entry_price, stop=p.initial_stop,
            target=p.target if p.target is not None else float("nan"), exit_time=t, exit_price=fill,
            exit_reason=reason, gross_pnl=p.side * p.qty * (fill - p.entry_price), fees=p.entry_fee + fee,
            funding=p.funding_paid, slippage_cost=p.slippage_cost, net_pnl=net, risk_amount=risk_amount,
            r_multiple=net / risk_amount if risk_amount > 0 else float("nan"), equity_before=p.equity_before,
            equity_after=self.cash, holding_hours=(t - p.entry_time).total_seconds() / 3600,
            ambiguous_bars=p.ambiguous_bars, precise_used=p.precise_used,
        ))
        self.pos = None
        self._update(t, self.cash)

    def _resolve_minutes(self, p: _Position, start: pd.Timestamp) -> tuple[str, float] | None:
        """15분 구간의 1분봉으로 손절·익절 순서 판정. 1분봉이 없으면 None."""
        minutes = self.market.minute_bars(p.symbol, start, start + self.exec_step)
        if minutes.empty:
            return None
        p.precise_used = True
        assert p.target is not None
        for o, h, lo in zip(minutes["open"].to_numpy(float), minutes["high"].to_numpy(float),
                            minutes["low"].to_numpy(float), strict=True):
            adverse, favorable = (lo, h) if p.side > 0 else (h, lo)
            if p.side * (o - p.stop) <= 0:
                return "stop_gap", o
            if p.side * (o - p.target) >= 0:
                return "target", p.target
            if p.side * (adverse - p.stop) <= 0:
                return "stop", p.stop  # 같은 1분봉에서 둘 다 닿으면 여전히 손절 우선
            if p.side * (favorable - p.target) >= 0:
                return "target", p.target
        return None

    def _process_bar(self, i: int) -> bool:
        """보유 포지션의 i 번째 실행 봉. 청산되면 True."""
        p = self.pos
        assert p is not None
        frame = p.bars.bars
        s = frame.index[i]
        o, h, lo, c = (float(frame[col].iloc[i]) for col in ("open", "high", "low", "close"))
        self._apply_funding(s)
        if p.flatten_pending:
            self._close(s, o, "kill_switch", taker=True)
            return True
        if s >= p.time_exit_at:
            self._close(s, o, "time", taker=True)
            return True
        stop_reason: ExitReason = "trailing_stop" if p.trail_distance is not None else "stop"
        # 청산가가 손절보다 가까우면 먼저 닿는다
        liq_first = p.liq_price is not None and p.side * (p.liq_price - p.stop) > 0
        if liq_first and p.liq_price is not None and p.side * (o - p.liq_price) <= 0:
            self._close(s, p.liq_price, "liquidation", taker=True, raw=True)
            return True
        # 시가 갭
        if p.side * (o - p.stop) <= 0:
            self._close(s, o, "stop_gap", taker=True)
            return True
        if p.target is not None and p.side * (o - p.target) >= 0:
            self._close(s, p.target, "target", taker=False)
            return True
        adverse, favorable = (lo, h) if p.side > 0 else (h, lo)
        if liq_first and p.liq_price is not None and p.side * (adverse - p.liq_price) <= 0:
            self._close(s, p.liq_price, "liquidation", taker=True, raw=True)
            return True
        stop_hit = p.side * (adverse - p.stop) <= 0
        target_hit = p.target is not None and p.side * (favorable - p.target) >= 0
        if stop_hit and target_hit:
            p.ambiguous_bars += 1
            resolved = self._resolve_minutes(p, s) if self.sc.precise else None
            if resolved is not None and resolved[0] == "target":
                self._close(s, resolved[1], "target", taker=False)
            elif resolved is not None and resolved[0] == "stop_gap":
                self._close(s, resolved[1], "stop_gap", taker=True)
            else:
                self._close(s, p.stop, stop_reason, taker=True)
            return True
        if stop_hit:
            self._close(s, p.stop, stop_reason, taker=True)
            return True
        if target_hit and p.target is not None:
            self._close(s, p.target, "target", taker=False)
            return True
        # 봉 마감: 트레일링 갱신(다음 봉부터), 평가금액
        if p.trail_distance is not None:
            p.extreme = max(p.extreme, h) if p.side > 0 else min(p.extreme, lo)
            candidate = p.extreme - p.side * p.trail_distance
            p.stop = max(p.stop, candidate) if p.side > 0 else min(p.stop, candidate)
        self._update(s + self.exec_step, self._equity(c))
        return False

    def _advance(self, until: pd.Timestamp) -> None:
        """시작 시각이 until 보다 이른 실행 봉을 모두 처리한다."""
        p = self.pos
        if p is None:
            return
        frame = p.bars.bars
        while p.bar_pos < len(frame) and frame.index[p.bar_pos] < until:
            if self._process_bar(p.bar_pos):
                return
            p.bar_pos += 1
        if p.bar_pos >= len(frame):
            last_end = frame.index[-1] + self.exec_step
            if p.bars.delisted_at is not None and p.bars.delisted_at <= until:
                self._close(max(last_end, p.bars.delisted_at), p.bars.last_trade_close or float(frame["close"].iloc[-1]),
                            "delisted", taker=True, raw=True)
            elif until >= self.end:
                self._close(min(last_end, self.end), float(frame["close"].iloc[-1]), "end_of_period", taker=True)

    # --- 실행 --------------------------------------------------------------------------------------------

    def _open_at(self, t: pd.Timestamp) -> float | None:
        """보유 심볼의 다음 미처리 실행 봉이 t 에 시작하면 그 시가."""
        p = self.pos
        if p is None:
            return None
        frame = p.bars.bars
        if p.bar_pos < len(frame) and frame.index[p.bar_pos] == t:
            return float(frame["open"].iloc[p.bar_pos])
        return None

    def _scheduled_exit_at(self, t: pd.Timestamp) -> None:
        """t 시가에 예정된 청산(시간 청산·킬 스위치 전량 청산)은 같은 시각의 신규 진입보다 먼저 처리한다."""
        p = self.pos
        open_ = self._open_at(t)
        if p is None or open_ is None:
            return
        if p.flatten_pending:
            self._close(t, open_, "kill_switch", taker=True)
        elif p.time_exit_at <= t:
            self._close(t, open_, "time", taker=True)

    def run(self, signals: pd.DataFrame) -> BacktestResult:
        """signals: coin, symbol, timeframe, tf_minutes, side, signal_time, trigger_close/low/high, trigger_atr,
        swing_price, rank (NaN = 순위 표 밖). 인덱스는 신호 ID 로 쓴다."""
        top_n = self.s.universe.top_n
        opposite = self.s.strategy.positioning.opposite_signal
        confluence = self.s.strategy.confluence
        if confluence.enabled and "confluence" not in signals.columns:
            signals = annotate_confluence(signals, self.s.strategy)
        order = signals.assign(_rank=signals["rank"].fillna(np.inf)).sort_values(
            ["signal_time", "tf_minutes", "_rank"], ascending=[True, False, True], kind="stable")
        status: dict[int, str] = {}
        times = pd.DatetimeIndex(order["signal_time"])
        ids = [int(i) for i in order.index]
        records = order.to_dict("records")
        k = 0
        while k < len(ids):
            t = times[k]
            j = k
            while j < len(ids) and times[j] == t:
                j += 1
            group = list(zip(ids[k:j], records[k:j], strict=True))
            k = j
            if t < self.start or t >= self.end:
                status.update((sid, "outside_period") for sid, _ in group)
                continue
            self._advance(t)
            self._scheduled_exit_at(t)
            if self.pos is None:
                self._update(t, self.cash)
            entered = False
            for sid, sig in group:
                rank = float(sig["rank"])
                if not rank <= top_n:  # NaN 이거나 top_n 밖
                    status[sid] = "out_of_rank"
                    continue
                if confluence.enabled and int(sig["confluence"]) < confluence.min_timeframes:
                    status[sid] = "no_confluence"
                    continue
                if self._stop_too_tight(pd.Series(sig)):
                    status[sid] = "skipped_stop_too_tight"
                    continue
                if entered:  # 같은 시각에 더 높은 우선순위 신호로 이미 진입
                    status[sid] = "skipped_simultaneous"
                    continue
                if self.pos is not None:
                    opposite_side = (sig["side"] == "long") != (self.pos.side > 0)
                    open_ = self._open_at(t)
                    if not (opposite == "switch" and opposite_side and open_ is not None):
                        status[sid] = "skipped_holding"
                        continue
                    self._close(t, open_, "switch", taker=True)
                if self.kill_tripped:
                    status[sid] = "skipped_kill_switch"
                    continue
                if self.blocked_until is not None and t < self.blocked_until:
                    status[sid] = "skipped_daily_loss"
                    continue
                outcome = self._enter(sid, pd.Series(sig), t)
                status[sid] = outcome
                entered = outcome == "entered"
        if self.pos is not None:
            self._advance(self.end + pd.Timedelta(days=400))
            if self.pos is not None:  # 봉이 남아 있으면(이론상 없음) 마지막 종가로 정리
                frame = self.pos.bars.bars
                self._close(self.end, float(frame["close"].iloc[-1]), "end_of_period", taker=True)
        self._update(self.end, self.cash)
        trades = pd.DataFrame([asdict(t) for t in self.trades])
        equity = pd.Series([e for _, e in self.equity_points], index=pd.DatetimeIndex([t for t, _ in self.equity_points]))
        equity = equity[~equity.index.duplicated(keep="last")].sort_index()
        out = signals.copy()
        out["status"] = pd.Series(status, dtype=object).reindex(out.index).fillna("not_processed")
        return BacktestResult(self.sc, trades, out, equity, self.events, self.start, self.end,
                              float(self.s.base.account.initial_capital))
