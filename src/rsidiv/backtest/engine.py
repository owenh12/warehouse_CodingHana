"""이벤트 기반 bar-by-bar 백테스트 엔진 (설계 §5).

봉 하나(시작 시각 T)의 처리 순서:

1. 펀딩 정산 (선물, 정산 시각 = T 이면 T 이전부터 보유한 포지션)
2. T 시가: 대기 중인 시장가 주문 체결 → 진입 체결이면 손절·익절(OCO) 주문
3. 봉 내부: 손절 → 익절 순 판정 (진입 봉 포함), 봉 중간 주문은 손절만 보수적으로 판정
4. T 종가: 평가금액 갱신 → 지표·신호·진입 확인·리스크 게이트·사이징 → 다음 봉 시가 주문, 보유 관리
5. 리스크 판정: 일일 손실(계좌별)·킬 스위치(합산 USD). 결과는 다음 봉 결정부터 적용

전략·사이징·리스크·비용 코드는 모의투자·실거래와 같은 :class:`AccountController` 를 쓰고,
체결만 :class:`SimBroker` 가 맡는다. 현재는 가상화폐 계좌(USDT)만 시뮬레이션한다.
국내주식 계좌는 KIS 데이터와 환율이 준비되면 같은 루프에 계좌를 추가한다.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rsidiv.broker.costs import CostModel
from rsidiv.broker.instruments import CryptoInstrument
from rsidiv.broker.sim import FundingEvent, SimBroker
from rsidiv.core.config import CryptoMarketType, Settings, StrategyParams, timeframe_minutes
from rsidiv.core.models import AssetClass, Bar
from rsidiv.risk.manager import DailyLossGuard, KillSwitch, RiskEvent
from rsidiv.strategy.controller import AccountController, ControllerStats, SignalRecord, TradeRecord


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    label: str
    slippage_multiplier: float = 1.0
    include_fees: bool = True
    live_rules: bool = True


def default_scenarios(settings: Settings) -> list[Scenario]:
    """costs.yaml 의 민감도 시나리오와 backtest.yaml 의 리스크 규칙 on/off 조합.

    기준 = 비용 반영, 슬리피지 ×1, 리스크 규칙 적용. 나머지는 한 가지씩만 바꾼다.
    """
    out = [Scenario("base", "기준 (비용·슬리피지×1·리스크 규칙)")]
    if settings.costs.scenarios.include_gross:
        out.append(Scenario("gross", "비용 반영 전 (수수료·슬리피지·펀딩비 0)", 0.0, False))
    for mult in settings.costs.scenarios.slippage_multipliers:
        if mult != 1:
            out.append(Scenario(f"slip_x{mult:g}", f"슬리피지 ×{mult:g}", float(mult)))
    if "without_live_rules" in settings.backtest.risk_rules_variants:
        out.append(Scenario("no_live_rules", "리스크 규칙 미적용 (일일 손실·킬 스위치 끔)", live_rules=False))
    return out


@dataclass(slots=True)
class BacktestResult:
    scenario: Scenario
    account: str
    currency: str
    symbols: list[str]
    initial_equity: float
    trades: list[TradeRecord]
    signals: list[SignalRecord]
    equity: pd.DataFrame  # index = 봉 종가 확정 시각(UTC). equity·cash·exposure·positions·equity_usd
    risk_events: list[RiskEvent]
    funding_events: list[FundingEvent]
    stats: ControllerStats
    params: StrategyParams
    kill_switch: RiskEvent | None = None
    notes: list[str] = field(default_factory=list)


def _bars_by_time(data: Mapping[str, pd.DataFrame]) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray], dict[str, dict[int, int]]]:
    timeline = pd.DatetimeIndex(sorted(set().union(*(frame.index for frame in data.values()))))
    arrays = {s: f[["open", "high", "low", "close", "volume"]].to_numpy(dtype="float64") for s, f in data.items()}
    positions = {s: {int(t.value): i for i, t in enumerate(f.index)} for s, f in data.items()}
    return timeline, arrays, positions


def run_backtest(
    settings: Settings,
    data: Mapping[str, pd.DataFrame],
    *,
    scenario: Scenario,
    market_type: CryptoMarketType = "spot",
    funding: Mapping[str, pd.Series] | None = None,
    params: StrategyParams | None = None,
) -> BacktestResult:
    """가상화폐 계좌 하나를 시뮬레이션한다. ``data``: 심볼 → 15분봉 프레임 (UTC 봉 시작 시각 인덱스)."""
    params = params or settings.strategy.for_asset_class("crypto")
    symbols = [s for s in settings.universe.crypto.symbols if s in data] + [
        s for s in data if s not in settings.universe.crypto.symbols
    ]
    usdt_per_usd = settings.data.fx.usdt_per_usd
    initial = settings.base.capital.initial * settings.base.capital.allocation["crypto"] * usdt_per_usd
    filters = settings.markets.binance.fallback_filters[market_type]
    instruments = {s: CryptoInstrument.from_filter(s, filters[s]) for s in symbols}
    costs = CostModel(settings.costs, AssetClass.CRYPTO, market_type=market_type,
                      slippage_multiplier=scenario.slippage_multiplier, include_fees=scenario.include_fees)
    funding_rates = {
        s: dict(zip(pd.DatetimeIndex(series.index).to_pydatetime(), series.to_numpy(dtype=float).tolist(),
                    strict=True))
        for s, series in (funding or {}).items()
    } if market_type == "usdm_futures" else {}
    broker = SimBroker(AssetClass.CRYPTO, cash=initial, instruments=instruments, costs=costs,
                       fill=settings.backtest.fill, funding_rates=funding_rates)
    risk = settings.risk
    daily = DailyLossGuard(risk.daily_loss, "crypto", scope="crypto", enabled=scenario.live_rules)
    kill = KillSwitch(risk.kill_switch, scope=risk.kill_switch.scope, enabled=scenario.live_rules)
    controller = AccountController(
        "crypto", broker, symbols=symbols, params=params, risk=risk, costs=costs, asset_class="crypto",
        timeframe=settings.data.timeframe, initial_equity=initial, daily_loss=daily,
    )

    bar_len = pd.Timedelta(minutes=timeframe_minutes(settings.data.timeframe))
    timeline, arrays, index_of = _bars_by_time(data)
    events: list[RiskEvent] = []
    rows: list[tuple[pd.Timestamp, float, float, float, int]] = []
    last_close: dict[str, float] = {}
    for ts in timeline:
        key = int(ts.value)
        bars: dict[str, Bar] = {}
        for s in symbols:
            i = index_of[s].get(key)
            if i is not None:
                o, h, lo, c, v = arrays[s][i]
                bars[s] = Bar(ts, float(o), float(h), float(lo), float(c), float(v))
        broker.begin_bar(ts)
        for event in broker.apply_funding(ts, {s: b.open for s, b in bars.items()}):
            controller.on_funding(event.symbol, event.payment)
        controller.on_fills(broker.execute_open(bars))
        controller.on_fills(broker.execute_intrabar(bars))
        controller.on_fills(broker.execute_late(bars))
        closes = {s: b.close for s, b in bars.items()}
        broker.mark(closes)
        last_close.update(closes)
        broker.end_bar()

        close_time = ts + bar_len
        controller.on_bar_close(bars, entry_block="kill_switch" if kill.tripped else None)
        events += controller.evaluate_daily_loss(close_time)
        tripped = kill.evaluate(close_time, broker.equity() / usdt_per_usd)
        if tripped:
            events += tripped
            controller.on_kill_switch(risk.kill_switch.on_trip, close_time)
        equity = broker.equity()
        held = sum(p.qty * last_close[s] for s, p in broker.positions().items())
        rows.append((close_time, equity, broker.cash(), held, len(broker.positions())))

    notes = []
    if len(timeline):
        end_time = timeline[-1] + bar_len
        controller.close_out(end_time)
        controller.on_fills(broker.execute_close_out(last_close, end_time))
        held = sum(p.qty * last_close[s] for s, p in broker.positions().items())
        rows[-1] = (end_time, broker.equity(), broker.cash(), held, len(broker.positions()))
        if controller.open_trades:
            notes.append(f"종료 시 정리되지 않은 포지션 {len(controller.open_trades)}건")

    frame = pd.DataFrame(rows, columns=["time", "equity", "cash", "positions_value", "positions"]).set_index("time")
    frame["exposure"] = np.where(frame["equity"] > 0, frame["positions_value"] / frame["equity"], 0.0)
    frame["equity_usd"] = frame["equity"] / usdt_per_usd
    return BacktestResult(
        scenario, "crypto", "USDT", symbols, initial, sorted(controller.trades, key=lambda t: t.entry_time),
        controller.signals, frame, events, broker.funding_events, controller.stats, params, kill.trip, notes,
    )


def run_scenarios(
    settings: Settings,
    data: Mapping[str, pd.DataFrame],
    scenarios: Sequence[Scenario],
    *,
    market_type: CryptoMarketType = "spot",
    funding: Mapping[str, pd.Series] | None = None,
) -> list[BacktestResult]:
    return [run_backtest(settings, data, scenario=s, market_type=market_type, funding=funding) for s in scenarios]


def slice_data(data: Mapping[str, pd.DataFrame], start: dt.datetime, end: dt.datetime) -> dict[str, pd.DataFrame]:
    return {s: f[(f.index >= pd.Timestamp(start)) & (f.index < pd.Timestamp(end))] for s, f in data.items()}
