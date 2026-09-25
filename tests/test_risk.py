"""사이징, 손절·익절 규칙, 진입 확인, 일일 손실 한도, 킬 스위치."""

from __future__ import annotations

import datetime as dt
import math

import pandas as pd
import pytest

from rsidiv.broker.costs import CostModel
from rsidiv.broker.instruments import CryptoInstrument, KrxInstrument
from rsidiv.core.config import AssetLimitsCfg, SizingCfg, load_settings
from rsidiv.core.models import AssetClass, Bar
from rsidiv.risk.manager import DailyLossGuard, KillSwitch
from rsidiv.risk.sizing import size_position
from rsidiv.signals.divergence import DivergenceCandidate
from rsidiv.strategy.rules import (
    EntryConfirmer,
    initial_stop,
    p2_blocks_entry,
    target_price,
    trailing_stop,
)

SETTINGS = load_settings()
RISK = SETTINGS.risk
T = dt.datetime(2025, 3, 3, 1, 0, tzinfo=dt.UTC)
BTC = CryptoInstrument("BTC/USDT", 0.00001, 0.00001, 5.0, 0.01)
GROSS = CostModel(SETTINGS.costs, AssetClass.CRYPTO, slippage_multiplier=0, include_fees=False)
NET = CostModel(SETTINGS.costs, AssetClass.CRYPTO)


def sizing(cfg: dict[str, object] | None = None, limits: dict[str, object] | None = None, **kw: object):  # type: ignore[no-untyped-def]
    args: dict[str, object] = {
        "cfg": SizingCfg.model_validate({**RISK.sizing.model_dump(), **(cfg or {})}),
        "limits": AssetLimitsCfg.model_validate({**RISK.limits.crypto.model_dump(), **(limits or {})}),
        "equity": 500.0, "initial_equity": 500.0, "available_cash": 500.0, "reference_price": 100_000.0, "stop": 96_000.0,
        "instrument": BTC, "costs": GROSS, "when": T,
    }
    args.update(kw)
    return size_position(**args)  # type: ignore[arg-type]


def test_risk_based_quantity() -> None:
    r = sizing()
    assert r.accepted and not r.capped
    assert r.risk_amount == pytest.approx(5.0) and r.unit_risk == pytest.approx(4000.0)
    assert r.qty == pytest.approx(0.00125)  # 5 / 4000 → 명목가 125 USDT < 비중 상한 250


def test_costs_increase_one_r() -> None:
    r = sizing(costs=NET)
    entry = 100_000 * 1.0002 * 1.001
    exit_ = 96_000 * 0.9998 * (1 - 0.001)
    assert r.unit_risk == pytest.approx(entry - exit_)
    assert r.qty == pytest.approx(math.floor(5.0 / (entry - exit_) * 1e5) / 1e5)
    assert r.qty < 0.00125


def test_weight_and_cash_caps() -> None:
    tight = sizing(stop=99_900.0)  # 1R = 100 → 위험 기준 0.05 BTC = 5,000 USDT > 상한 250
    assert tight.capped and tight.qty == pytest.approx(0.0025)  # min(비중 250, 현금 497.5) / 100,000
    cash_bound = sizing(stop=99_900.0, available_cash=100.0)
    assert cash_bound.qty == pytest.approx(math.floor(100 * (1 - 0.005) / 100_000 * 1e5) / 1e5)
    skip = sizing({"on_cap_exceeded": "skip"}, stop=99_900.0)
    assert skip.skip_reason == "cap_exceeded"


def test_skip_reasons() -> None:
    assert sizing(stop=100_000.0).skip_reason == "stop_above_entry"
    assert sizing(equity=1.0, available_cash=1.0).skip_reason == "insufficient_capital"  # 명목가 < 5 USDT
    assert sizing({"capital_base": "sleeve_initial"}, equity=1000.0).risk_amount == pytest.approx(5.0)
    krx = KrxInstrument("005930", SETTINGS.markets.krx)
    stock = CostModel(SETTINGS.costs, AssetClass.STOCK_KR, krx=SETTINGS.markets.krx)
    r = sizing(limits={"max_weight": 0.34}, equity=730_000.0, available_cash=730_000.0,
               reference_price=260_000.0, stop=255_000.0, instrument=krx, costs=stock)
    assert r.skip_reason == "insufficient_capital"  # 34% 상한 = 248,200원 < 1주 (설계 §7 소액 자본 경고)


def candidate(price_t3: float = 100.0, price_p2: float = 110.0) -> DivergenceCandidate:
    return DivergenceCandidate(60, 63, pd.Timestamp("2025-01-01", tz="UTC"), 41, 49, 101.0, price_p2, price_t3,
                               25.0, 32.0)


def test_stop_and_target_rules() -> None:
    exit_cfg = SETTINGS.strategy.default.exit
    assert initial_stop(candidate(), 2.0, exit_cfg.stop) == pytest.approx(98.0)  # Low(t3) − ATR × 1
    assert initial_stop(candidate(), None, exit_cfg.stop) is None
    pct = exit_cfg.stop.model_copy(update={"mode": "pct"})
    assert initial_stop(candidate(), None, pct) == pytest.approx(99.5)
    tp = exit_cfg.take_profit
    assert target_price(102.0, 98.0, candidate(), tp) == pytest.approx(110.0)  # 2R
    p2 = tp.model_copy(update={"mode": "p2"})
    assert target_price(102.0, 98.0, candidate(), p2) == 110.0
    assert p2_blocks_entry(111.0, candidate(), p2) and not p2_blocks_entry(105.0, candidate(), p2)
    fallback = p2.model_copy(update={"p2_below_entry": "fallback_r_multiple"})
    assert target_price(111.0, 98.0, candidate(), fallback) == pytest.approx(137.0)
    assert not p2_blocks_entry(111.0, candidate(), fallback)
    trail = tp.model_copy(update={"mode": "trailing"})
    assert target_price(102.0, 98.0, candidate(), trail) is None
    assert trailing_stop(120.0, 2.0, trail) == pytest.approx(114.0)
    assert trailing_stop(120.0, 2.0, tp) is None


def bar(close: float, high: float | None = None, low: float | None = None) -> Bar:
    return Bar(T, close, high if high is not None else close + 0.5, low if low is not None else close - 0.5, close, 1.0)


def test_entry_confirmer_modes() -> None:
    entry = SETTINGS.strategy.default.entry
    b = EntryConfirmer(entry.model_copy(update={"mode": "B"}))
    b.start(candidate(), 98.0, 10)
    assert b.on_bar(10, bar(100), bar(100), 30, 30) is None  # 신호 봉 자체는 보지 않음
    assert b.on_bar(11, bar(100.4), bar(100), 30, 30) is None  # 100.4 ≤ 직전 고가 100.5
    assert b.on_bar(12, bar(101), bar(100.4), 30, 30) == "enter"
    b.take()
    b.start(candidate(), 98.0, 20)
    assert b.on_bar(21, bar(99, low=97.9), bar(100), 30, 30) == "canceled_below_stop" and b.pending is None
    b.start(candidate(), 98.0, 30)
    assert [b.on_bar(k, bar(99), bar(99), 30, 30) for k in (31, 32, 33)] == [None, None, "not_confirmed"]
    c = EntryConfirmer(entry.model_copy(update={"mode": "C", "rsi_cross_level": 30.0}))
    c.start(candidate(), 98.0, 10)
    assert c.on_bar(11, bar(100), bar(100), 25.0, 29.9) is None
    assert c.on_bar(12, bar(100), bar(100), 29.9, 30.0) == "enter"  # 직전 < 30 ≤ 현재
    c.take()
    c.start(candidate(), 98.0, 10)
    assert c.on_bar(11, bar(100), bar(100), 31.0, 35.0) is None  # 이미 기준 위: 돌파 아님
    assert EntryConfirmer(entry).immediate


def test_daily_loss_clock_reset_kst() -> None:
    guard = DailyLossGuard(RISK.daily_loss, "crypto", scope="crypto", enabled=True)
    # 00:00 KST = 15:00 UTC. 전날 14:45 UTC 봉 종가(=00:00 KST)가 새 날의 시작 평가금액
    start = dt.datetime(2025, 3, 2, 15, 0, tzinfo=dt.UTC)
    assert guard.evaluate(start - dt.timedelta(minutes=15), 1000.0) == []
    assert guard.evaluate(start, 1000.0) == [] and guard.day_start_equity == 1000.0
    assert guard.evaluate(start + dt.timedelta(hours=3), 971.0) == [] and not guard.blocked  # 2.9%
    events = guard.evaluate(start + dt.timedelta(hours=4), 969.0)  # 3.1%
    assert guard.blocked and events[0].kind == "daily_loss_trip" and events[0].value == pytest.approx(0.031)
    assert guard.evaluate(start + dt.timedelta(hours=5), 1000.0) == [] and guard.blocked  # 당일은 유지
    events = guard.evaluate(start + dt.timedelta(days=1), 990.0)
    assert not guard.blocked and events[0].kind == "daily_loss_reset" and guard.day_start_equity == 990.0
    off = DailyLossGuard(RISK.daily_loss, "crypto", scope="crypto", enabled=False)
    off.evaluate(start, 1000.0)
    assert off.evaluate(start + dt.timedelta(hours=1), 500.0) == [] and not off.blocked


def test_daily_loss_market_open_reset_uses_previous_close() -> None:
    guard = DailyLossGuard(RISK.daily_loss, "stock_kr", scope="stock_kr", enabled=True, krx=SETTINGS.markets.krx)
    day1_last = dt.datetime(2025, 3, 3, 6, 30, tzinfo=dt.UTC)  # 15:30 KST 마지막 봉 종가
    guard.evaluate(day1_last - dt.timedelta(minutes=15), 1_000_000.0)
    guard.evaluate(day1_last, 1_010_000.0)
    day2_first = dt.datetime(2025, 3, 4, 0, 15, tzinfo=dt.UTC)  # 09:15 KST (09:00 봉 종가)
    guard.evaluate(day2_first, 980_000.0)  # 전날 마지막 평가금액 대비 2.97% 하락
    assert guard.day_start_equity == 1_010_000.0 and not guard.blocked
    guard.evaluate(day2_first + dt.timedelta(minutes=15), 979_000.0)  # 3.07%
    assert guard.blocked
    with pytest.raises(ValueError, match="krx"):
        DailyLossGuard(RISK.daily_loss, "stock_kr", scope="stock_kr", enabled=True)


def test_kill_switch_peak_drawdown_and_release() -> None:
    ks = KillSwitch(RISK.kill_switch, scope="portfolio_usd", enabled=True)
    t0 = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
    assert ks.evaluate(t0, 1000.0) == [] and ks.evaluate(t0, 1200.0) == []
    assert ks.evaluate(t0, 841.0) == [] and not ks.tripped  # 29.9%
    events = ks.evaluate(t0 + dt.timedelta(hours=1), 840.0)  # 30%
    assert ks.tripped and events[0].trip_id == "KS-20250101T0100Z" and events[0].reference == 1200.0
    assert ks.evaluate(t0, 2000.0) == [] and ks.tripped  # 자동 재개 없음
    released = KillSwitch(RISK.kill_switch.model_copy(update={"release_ack": "KS-20250101T0100Z"}),
                          scope="portfolio_usd", enabled=True)
    released.evaluate(t0, 1200.0)
    released.evaluate(t0 + dt.timedelta(hours=1), 840.0)
    assert not released.tripped  # 발동 ID 가 일치하면 해제
