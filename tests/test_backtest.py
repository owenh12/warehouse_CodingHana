"""백테스트 엔진 + 매매 관리(AccountController) 통합 테스트 (합성 데이터).

기본 경로: 워밍업 → 급락(t1=41) → 반등(p2=49) → 완만한 하락(t3=60, 더 낮은 저점) → 신호 봉 63 → 64 시가 진입.
그 뒤 경로(tail)를 바꿔 익절·손절·시간 청산·트레일링·진입 확인·리스크 규칙을 확인한다.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import pytest
from test_divergence import RELAXED, crypto_index, params, random_frame

from rsidiv.backtest.engine import BacktestResult, Scenario, run_backtest
from rsidiv.core.config import Settings, apply_dotted, load_settings
from rsidiv.indicators.atr import atr_wilder

SETTINGS = load_settings()
BAR = pd.Timedelta("15min")
SIGNAL = 63  # 신호 봉 (t3=60 + R=3)
ENTRY = 64  # 진입 봉 (신호 봉 다음 봉 시가)
T3_LOW = 86.6


def build(tail: Sequence[float], *, lead: int = 0, start: str = "2025-01-01") -> pd.DataFrame:
    """합성 15분봉. ``tail`` 은 61번 봉(t3 다음)부터의 종가. 시가 = 직전 종가, 고가·저가 = ±0.1."""
    closes = [100.0] * lead
    closes += [100 + 0.2 * (i % 2) for i in range(30)]
    closes += [100 - 1.0 * (i + 1) for i in range(12)]
    closes += [88 + 1.2 * (i + 1) for i in range(8)]
    closes += [97.6 - 1.0 * (i + 1) + (0.6 if i % 2 else 0) for i in range(10)]
    closes += [87.0]
    closes += list(tail)
    c = np.asarray(closes, dtype=float)
    o = np.concatenate([[c[0]], c[:-1]])
    h, lo = np.maximum(o, c) + 0.1, np.minimum(o, c) - 0.1
    lo[lead + 41] -= 0.3  # 피벗 저점 t1·t3 와 고점 p2 를 분명하게
    lo[lead + 60] -= 0.3
    h[lead + 49] += 0.3
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": 1.0},
                        index=crypto_index(len(c), start))


def settings(dotted: dict[str, Any] | None = None) -> Settings:
    return apply_dotted(SETTINGS, dotted or {}) if dotted else SETTINGS


def run(data: dict[str, pd.DataFrame] | pd.DataFrame, *, gross: bool = True, dotted: dict[str, Any] | None = None,
        strategy: dict[str, Any] | None = None, market: str = "spot",
        funding: dict[str, pd.Series] | None = None, live_rules: bool = True) -> BacktestResult:
    frames = data if isinstance(data, dict) else {"BTC/USDT": data}
    scenario = Scenario("t", "t", 0.0 if gross else 1.0, not gross, live_rules)
    return run_backtest(settings(dotted), frames, scenario=scenario, params=params(strategy or {}),
                        market_type=market, funding=funding)  # type: ignore[arg-type]


def stop_price(frame: pd.DataFrame, signal: int = SIGNAL, low: float = T3_LOW) -> float:
    atr = atr_wilder(frame["high"], frame["low"], frame["close"], 14)[signal]
    return float(np.floor((low - atr) * 100) / 100)


RISE_TAIL = [88, 89, 90] + [90 + 0.5 * i for i in range(1, 60)]


def test_take_profit_trade() -> None:
    frame = build(RISE_TAIL)
    result = run(frame)
    assert len(result.trades) == 1
    t = result.trades[0]
    stop = stop_price(frame)
    assert t.signal.signal_time == frame.index[SIGNAL] + BAR == t.decision_time
    assert t.entry_time == frame.index[ENTRY] and t.entry_price == frame["open"].iloc[ENTRY] == 90.0
    assert t.initial_stop == pytest.approx(stop)
    assert t.qty == pytest.approx(np.floor(5.0 / (90.0 - stop) * 1e5) / 1e5)  # 위험 1% = 5 USDT
    target = float(np.ceil((90 + 2 * (90 - stop)) * 100) / 100)
    assert t.target == pytest.approx(target)
    first_touch = int(np.flatnonzero(frame["high"].to_numpy()[ENTRY:] >= target)[0]) + ENTRY
    assert t.exit_reason == "take_profit" and t.exit_price == pytest.approx(target)
    assert t.exit_time == frame.index[first_touch]
    assert t.r_multiple == pytest.approx(2.0, abs=0.01) and t.costs == 0.0
    assert result.signals[0].outcome == "entered" and result.signals[0].trade_id == t.trade_id


def test_stop_loss_trade() -> None:
    frame = build([88, 89, 90, 89, 87, 85, 83, 81, 80] + [80] * 20)
    t = run(frame).trades[0]
    stop = stop_price(frame)
    assert t.exit_reason == "stop_loss"
    first = int(np.flatnonzero(frame["low"].to_numpy()[ENTRY:] <= stop)[0]) + ENTRY
    expected = min(frame["open"].iloc[first], stop)
    assert t.exit_price == pytest.approx(expected) and t.exit_time == frame.index[first]
    assert t.r_multiple <= -0.99


def test_time_exit_after_max_bars() -> None:
    frame = build([88, 89, 90] + [90.5, 90.3] * 30)
    t = run(frame).trades[0]
    assert t.exit_reason == "time_exit" and t.bars_held == 32
    assert t.exit_time == frame.index[ENTRY + 32]  # 32봉째 종가 후 다음 봉 시가
    assert t.exit_price == frame["open"].iloc[ENTRY + 32]


def test_trailing_stop_ratchets_up() -> None:
    tail = [88, 89, 90] + [90 + 0.8 * i for i in range(1, 16)] + [101 - 1.5 * i for i in range(1, 15)]
    frame = build(tail)
    result = run(frame, strategy={"exit": {"take_profit": {"mode": "trailing", "trailing_atr_mult": 2.0},
                                           "time_exit": {"enabled": False}}})
    t = result.trades[0]
    assert t.target is None and t.exit_reason == "trailing_stop"
    assert t.final_stop > t.initial_stop and t.exit_price > t.entry_price


def test_entry_mode_b_waits_for_close_above_previous_high() -> None:
    # 신호 뒤 64: 89.9 (직전 고가 90.1 못 넘음), 65: 90.5 (> 64 고가 90.1) → 66 시가 진입
    frame = build([88, 89, 90, 89.9, 90.5, 91] + [91.5] * 40)
    t = run(frame, strategy={"entry": {"mode": "B"}}).trades[0]
    assert t.decision_time == frame.index[SIGNAL + 2] + BAR and t.entry_time == frame.index[SIGNAL + 3]


def test_entry_mode_b_cancel_and_expiry() -> None:
    stop = stop_price(build(RISE_TAIL))
    below = build([88, 89, 90, stop - 0.5, 91] + [91.5] * 40)  # 대기 중 손절가 이탈
    r = run(below, strategy={"entry": {"mode": "B"}})
    assert not r.trades and r.signals[0].outcome == "canceled_below_stop"
    flat = build([88, 89, 90, 89.9, 89.8, 89.7] + [89.6] * 40)  # 3봉 안에 돌파 없음
    r = run(flat, strategy={"entry": {"mode": "B"}})
    assert not r.trades and r.signals[0].outcome == "not_confirmed"


def test_entry_mode_c_waits_for_rsi_cross() -> None:
    frame = build(RISE_TAIL)
    result = run(frame, strategy={"entry": {"mode": "C", "rsi_cross_level": 60.0, "confirm_window_bars": 10}})
    from rsidiv.indicators.rsi import rsi_wilder

    rsi = rsi_wilder(frame["close"].to_numpy(), 14)
    cross = next(k for k in range(SIGNAL + 1, SIGNAL + 11) if rsi[k - 1] < 60.0 <= rsi[k])
    t = result.trades[0]
    assert t.decision_time == frame.index[cross] + BAR and t.entry_time == frame.index[cross + 1]


DRIFT_DOWN_TAIL = [88, 89, 90] + [90 - 0.05 * i for i in range(1, 60)]  # 진입 후 손절가 위에서 조금씩 하락


def two_symbols(lead_eth: int = 10, btc_tail: Sequence[float] = RISE_TAIL) -> dict[str, pd.DataFrame]:
    btc = build(list(btc_tail) + [btc_tail[-1]] * lead_eth)
    eth = build(RISE_TAIL, lead=lead_eth)
    return {"BTC/USDT": btc, "ETH/USDT": eth}


def test_max_positions_and_insufficient_capital() -> None:
    r = run(two_symbols(0), dotted={"risk.limits.crypto.max_positions": 1})
    outcomes = {s.symbol: s.outcome for s in r.signals}
    assert outcomes == {"BTC/USDT": "entered", "ETH/USDT": "max_positions"}
    tiny = run(build(RISE_TAIL), dotted={"base.capital.initial": 2.0})  # 코인 슬리브 1 USDT < 최소 주문 5 USDT
    assert not tiny.trades and tiny.signals[0].outcome == "insufficient_capital"


def test_daily_loss_blocks_later_entries_same_day() -> None:
    # 비용 반영 → BTC 진입 수수료만으로 당일 손실 > 0.001% → ETH 신호(10봉 뒤)는 차단
    data = two_symbols(10, DRIFT_DOWN_TAIL)
    r = run(data, gross=False, dotted={"risk.daily_loss.threshold": 0.00001})
    outcomes = {s.symbol: s.outcome for s in r.signals}
    assert outcomes == {"BTC/USDT": "entered", "ETH/USDT": "daily_loss"}
    assert r.risk_events[0].kind == "daily_loss_trip"
    off = run(data, gross=False, dotted={"risk.daily_loss.threshold": 0.00001}, live_rules=False)
    assert {s.outcome for s in off.signals} == {"entered"}


def test_kill_switch_flatten_all() -> None:
    r = run(two_symbols(10, DRIFT_DOWN_TAIL), gross=False,
            dotted={"risk.kill_switch.max_drawdown": 0.00001, "risk.kill_switch.on_trip": "flatten_all"})
    assert r.kill_switch is not None and r.kill_switch.trip_id is not None
    btc = r.trades[0]
    assert btc.exit_reason == "kill_switch" and btc.exit_time == r.kill_switch.time  # 다음 봉 시가 = 발동 시각
    assert {s.symbol: s.outcome for s in r.signals}["ETH/USDT"] == "kill_switch"


def test_kill_switch_keep_stops_cancels_target() -> None:
    r = run(two_symbols(10, DRIFT_DOWN_TAIL), gross=False, dotted={"risk.kill_switch.max_drawdown": 0.00001})
    assert r.kill_switch is not None and r.kill_switch.time < r.trades[0].exit_time
    # 익절 주문은 취소되고 시간 청산도 멈춘다 → 손절 또는 데이터 끝에서만 청산
    assert r.trades[0].exit_reason in ("stop_loss", "end_of_test") and r.trades[0].bars_held > 32


def test_order_rejected_and_limit_not_filled() -> None:
    gap = build(RISE_TAIL)
    gap.iloc[ENTRY, gap.columns.get_loc("open")] = 150.0
    gap.iloc[ENTRY, gap.columns.get_loc("high")] = 150.1
    r = run(gap, dotted={"risk.sizing.risk_per_trade": 0.5, "risk.limits.crypto.max_weight": 1.0,
                         "risk.sizing.cash_buffer": 0.0})
    assert not r.trades and r.signals[0].outcome == "order_rejected"
    up = build([88, 89, 90, 92] + [92.5] * 40)
    up.iloc[ENTRY, up.columns.get_loc("open")] = 91.5  # 갭 상승: 진입 봉 저가 91.4 > 지정가 90
    up.iloc[ENTRY, up.columns.get_loc("low")] = 91.4
    r = run(up, strategy={"entry": {"order_type": "limit"}})
    assert not r.trades and r.signals[0].outcome == "limit_not_filled"


def test_end_of_test_closes_open_position() -> None:
    frame = build([88, 89, 90, 90.5, 90.6])
    r = run(frame)
    t = r.trades[0]
    assert t.exit_reason == "end_of_test" and t.exit_price == frame["close"].iloc[-1]
    assert r.equity["equity"].iloc[-1] == pytest.approx(r.initial_equity + t.net_pnl)


def test_futures_funding_is_charged_to_trade() -> None:
    frame = build(RISE_TAIL)
    times = frame.index[frame.index.minute == 0]  # 매시 정산 (진입 시각과 같은 정산은 제외되어야 함)
    rates = pd.Series(0.001, index=times)
    r = run(frame, gross=False, market="usdm_futures", funding={"BTC/USDT": rates},
            dotted={"risk.sizing.risk_per_trade": 0.03})  # 선물 최소 주문금액 100 USDT
    t = r.trades[0]
    held = [ts for ts in times if t.entry_time < ts <= t.exit_time]
    expected = sum(t.qty * frame.loc[ts, "open"] * 0.001 for ts in held)
    assert t.entry_time in set(times) and len(held) >= 3 and t.funding == pytest.approx(expected)
    assert r.equity["equity"].iloc[-1] == pytest.approx(r.initial_equity + t.net_pnl)


# ---------------------------------------------------------------------------
# 무작위 데이터: 회계 항등식, 미래참조
# ---------------------------------------------------------------------------


def random_pair(n: int, seed: int) -> dict[str, pd.DataFrame]:
    return {"BTC/USDT": random_frame(n, seed), "ETH/USDT": random_frame(n, seed + 100)}


@pytest.mark.parametrize("gross", [True, False])
def test_equity_equals_initial_plus_trade_pnl(gross: bool) -> None:
    data = random_pair(4000, 1)
    r = run(data, gross=gross, strategy=RELAXED)
    assert len(r.trades) >= 20
    assert r.equity["equity"].iloc[-1] == pytest.approx(r.initial_equity + sum(t.net_pnl for t in r.trades),
                                                        abs=1e-9)
    if gross:
        assert all(t.costs == 0 for t in r.trades)
    reasons = {t.exit_reason for t in r.trades}
    assert {"stop_loss", "time_exit"} <= reasons


def test_no_look_ahead_trades_before_cutoff_unchanged() -> None:
    """데이터를 k 봉에서 잘라도, 그 전에 끝난 거래와 그 전에 내린 진입 결정은 같다."""
    data = random_pair(3000, 2)
    full = run(data, gross=False, strategy=RELAXED)
    for k in (900, 1700, 2500):
        cut = {s: f.iloc[:k] for s, f in data.items()}
        part = run(cut, gross=False, strategy=RELAXED)
        end = data["BTC/USDT"].index[k - 1] + BAR
        closed = [t for t in part.trades if t.exit_reason != "end_of_test"]
        expected = [t for t in full.trades if t.exit_time is not None and t.exit_time < end]
        key = lambda t: (t.symbol, t.entry_time, t.qty, t.entry_price, t.exit_time, t.exit_price, t.exit_reason)  # noqa: E731
        assert [key(t) for t in closed] == [key(t) for t in expected if t.exit_time <= end - BAR]
        decided = [(s.symbol, s.signal_time, s.outcome) for s in part.signals
                   if s.outcome not in ("ordered", "awaiting_confirmation", "data_end")]
        assert decided == [(s.symbol, s.signal_time, s.outcome) for s in full.signals
                           if (s.symbol, s.signal_time) in {(d[0], d[1]) for d in decided}]
