"""백테스트 엔진: 체결 가격·수수료·슬리피지, 손절 우선·갭, 정밀 모드, 시간 청산, 펀딩비 부호, 사이징·수량 단위·최소 명목가,
동시 신호 우선순위·스킵 사유, 일일 손실·킬 스위치, 청산가, 상장폐지, 트레일링·구조 익절, 반대 신호 전환, 평가금액 정합성."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from perpdiv.backtest.engine import BacktestEngine, BacktestResult, Scenario
from perpdiv.backtest.instruments import Instrument
from perpdiv.backtest.market import ExecBars
from perpdiv.core.config import Settings, deep_merge, load_settings
from perpdiv.data.base import empty_ohlcv

UTC = dt.UTC
T0 = pd.Timestamp("2025-01-06 00:00", tz="UTC")  # 월요일 09:00 KST
Q = pd.Timedelta(minutes=15)


# 기존 규칙(단일 TF 진입, 신호 봉 기준 손절, 최소 손절폭 없음)으로 체결 로직을 시험한다. 새 규칙은 아래 별도 테스트.
LEGACY: dict[str, Any] = {"strategy": {"confluence": {"enabled": False},
                                       "exit": {"stop": {"basis": "signal_bar", "min_distance_pct": 0.0}}}}


def settings_with(tmp_path: Path, patch: dict[str, Any] | None = None, *, legacy: bool = True) -> Settings:
    merged = deep_merge(LEGACY if legacy else {}, patch or {})
    if not merged:
        return load_settings()
    path = tmp_path / f"patch_{abs(hash(repr(merged)))}.yaml"
    path.write_text(yaml.safe_dump(merged), encoding="utf-8")
    return load_settings(overrides=[path])


def bars_from(rows: Sequence[Sequence[float]], start: pd.Timestamp = T0) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                         index=pd.date_range(start, periods=len(rows), freq="15min", tz="UTC"))
    frame["volume"] = 1.0
    frame["quote_volume"] = 1.0
    frame["trades"] = 1
    return frame


class FakeMarket:
    def __init__(self) -> None:
        self.frames: dict[str, pd.DataFrame] = {}
        self.delisted: dict[str, pd.Timestamp] = {}
        self.funding_rows: dict[str, pd.DataFrame] = {}
        self.minutes: dict[str, pd.DataFrame] = {}

    def exec_bars(self, symbol: str, start: dt.datetime, end: dt.datetime) -> ExecBars:
        frame = self.frames.get(symbol, empty_ohlcv())
        part = frame[(frame.index >= pd.Timestamp(start)) & (frame.index < pd.Timestamp(end))]
        delisted = self.delisted.get(symbol)
        last = float(frame["close"].iloc[-1]) if delisted is not None else None
        return ExecBars(symbol, part, delisted, last)

    def funding(self, symbol: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        f = self.funding_rows.get(symbol, pd.DataFrame({"rate": [], "mark": []}, index=pd.DatetimeIndex([], tz="UTC")))
        return f[(f.index > pd.Timestamp(start)) & (f.index <= pd.Timestamp(end))]

    def minute_bars(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        m = self.minutes.get(symbol, empty_ohlcv())
        return m[(m.index >= start) & (m.index < end)]


class FakeInstruments:
    def __init__(self, step: float = 0.001, min_notional: float = 5.0) -> None:
        self.step, self.min_notional = step, min_notional

    def get(self, symbol: str, month: str) -> Instrument:
        return Instrument(symbol, self.step, self.min_notional, "test")


def signal(time: pd.Timestamp, *, side: str = "long", symbol: str = "AAAUSDT", tf: str = "15m", close: float = 100.0,
           low: float = 99.0, high: float = 101.0, atr: float = 1.0, swing: float = 110.0, rank: float = 1.0,
           ) -> dict[str, Any]:
    minutes = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}[tf]
    return {"coin": symbol.removesuffix("USDT"), "symbol": symbol, "timeframe": tf, "tf_minutes": minutes, "side": side,
            "signal_time": time, "trigger_close": close, "trigger_low": low, "trigger_high": high, "trigger_atr": atr,
            "swing_price": swing, "rank": rank}


def run(settings: Settings, market: FakeMarket, sigs: list[dict[str, Any]], scenario: Scenario | None = None,
        instruments: FakeInstruments | None = None, end: pd.Timestamp | None = None) -> BacktestResult:
    engine = BacktestEngine(settings, market, instruments or FakeInstruments(), scenario or Scenario("net"),
                            start=T0.to_pydatetime(), end=(end or T0 + pd.Timedelta(days=30)).to_pydatetime())
    return engine.run(pd.DataFrame(sigs))


@pytest.fixture()
def cfg(tmp_path: Path) -> Settings:
    return settings_with(tmp_path)


def test_long_target_fill_fees_and_slippage(cfg: Settings) -> None:
    market = FakeMarket()
    market.frames["AAAUSDT"] = bars_from([(100, 101, 99.5, 100.5), (100.5, 107.5, 100, 107), (107, 108, 106, 107)])
    result = run(cfg, market, [signal(T0)])
    tr = result.trades.iloc[0]
    qty = np.floor(1000 * 0.998 / 100 / 0.001) * 0.001
    entry = 100 * 1.0002
    stop = 99 - 2.5
    target = entry + 2 * (entry - stop)
    assert tr.qty == pytest.approx(qty) and tr.entry_price == pytest.approx(entry)
    assert tr.exit_reason == "target" and tr.exit_price == pytest.approx(target)
    fees = qty * entry * 0.0005 + qty * target * 0.0002
    assert tr.fees == pytest.approx(fees) and tr.net_pnl == pytest.approx(qty * (target - entry) - fees)
    assert tr.r_multiple == pytest.approx(tr.net_pnl / (qty * (entry - stop)))
    assert tr.exit_time == T0 + Q  # 청산이 일어난 실행 봉 시작 시각
    assert result.equity.iloc[-1] == pytest.approx(1000 + tr.net_pnl)
    assert result.signals["status"].tolist() == ["entered"]


def test_stop_first_gap_and_precise_mode(cfg: Settings) -> None:
    market = FakeMarket()
    # 1번 봉에서 손절(96.5)·익절(~107.06) 모두 닿음 → 손절 우선
    market.frames["AAAUSDT"] = bars_from([(100, 101, 99.5, 100.5), (100.5, 108, 96, 100)])
    tr = run(cfg, market, [signal(T0)]).trades.iloc[0]
    assert tr.exit_reason == "stop" and tr.exit_price == pytest.approx(96.5 * (1 - 0.0002)) and tr.ambiguous_bars == 1
    # 정밀 모드: 1분봉에서 익절이 먼저 → 익절
    minutes = pd.DataFrame({"open": [100.5, 104, 107.5], "high": [104, 108, 107.5], "low": [100, 103, 96],
                            "close": [104, 107.5, 97]}, index=pd.date_range(T0 + Q, periods=3, freq="1min", tz="UTC"))
    market.minutes["AAAUSDT"] = minutes
    precise = run(cfg, market, [signal(T0)], Scenario("precise", precise=True)).trades.iloc[0]
    assert precise.exit_reason == "target" and precise.precise_used
    # 시가가 손절가 아래로 갭 → 시가 체결
    market.frames["AAAUSDT"] = bars_from([(100, 101, 99.5, 100.5), (95, 96, 94, 95)])
    gap = run(cfg, market, [signal(T0)]).trades.iloc[0]
    assert gap.exit_reason == "stop_gap" and gap.exit_price == pytest.approx(95 * (1 - 0.0002))


def test_short_mirror(cfg: Settings) -> None:
    market = FakeMarket()
    market.frames["AAAUSDT"] = bars_from([(100, 100.5, 99, 99.5), (99.5, 100, 92, 93)])
    tr = run(cfg, market, [signal(T0, side="short", high=101)]).trades.iloc[0]
    entry = 100 * (1 - 0.0002)
    stop = 101 + 2.5
    target = entry - 2 * (stop - entry)
    assert tr.side == "short" and tr.exit_reason == "target" and tr.exit_price == pytest.approx(target)
    assert tr.gross_pnl == pytest.approx(tr.qty * (entry - target))


def test_time_exit_at_open_after_n_signal_bars(tmp_path: Path) -> None:
    s = settings_with(tmp_path, {"strategy": {"exit": {"time_exit": {"bars": 3}}}})
    market = FakeMarket()
    market.frames["AAAUSDT"] = bars_from([(100, 100.5, 99.5, 100)] * 20)
    tr = run(s, market, [signal(T0, tf="1h")]).trades.iloc[0]
    assert tr.exit_reason == "time" and tr.exit_time == T0 + pd.Timedelta(hours=3)
    assert tr.exit_price == pytest.approx(100 * (1 - 0.0002))


def test_funding_sign_and_window(cfg: Settings) -> None:
    market = FakeMarket()
    market.frames["AAAUSDT"] = bars_from([(100, 100.5, 99.5, 100)] * 8 + [(100, 120, 100, 110)])
    idx = pd.DatetimeIndex([T0, T0 + pd.Timedelta(hours=1), T0 + pd.Timedelta(hours=2), T0 + pd.Timedelta(hours=5)])
    market.funding_rows["AAAUSDT"] = pd.DataFrame({"rate": [0.01, 0.001, -0.002, 0.5], "mark": [100.0] * 4}, index=idx)
    long = run(cfg, market, [signal(T0)]).trades.iloc[0]
    qty = long.qty
    # 진입(T0) 시각의 정산은 제외(진입 < F), 청산 봉(2시간째, 8번 봉 T0+2h) 까지: +0.001, −0.002 → 롱 지불 합 = −0.001×명목
    assert long.exit_time == T0 + pd.Timedelta(hours=2)
    assert long.funding == pytest.approx(qty * 100 * (0.001 - 0.002))
    short = run(cfg, market, [signal(T0, side="short", high=100.5)]).trades.iloc[0]
    assert short.exit_time == T0 + pd.Timedelta(hours=2) and short.exit_reason == "stop"
    assert short.funding == pytest.approx(-short.qty * 100 * (0.001 - 0.002))  # 숏은 음수 펀딩에 지불
    gross = run(cfg, market, [signal(T0)], Scenario("gross", fees=False, slippage_mult=0, funding=False)).trades.iloc[0]
    assert gross.funding == 0 and gross.fees == 0 and gross.net_pnl == pytest.approx(gross.gross_pnl)


def test_sizing_step_and_min_notional(cfg: Settings) -> None:
    market = FakeMarket()
    market.frames["AAAUSDT"] = bars_from([(100, 101, 99.5, 100.5), (100.5, 110, 100, 107)])
    tr = run(cfg, market, [signal(T0)], instruments=FakeInstruments(step=1.0)).trades.iloc[0]
    assert tr.qty == 9.0  # 9.98 → 9
    res = run(cfg, market, [signal(T0)], instruments=FakeInstruments(step=0.001, min_notional=2000))
    assert res.trades.empty and res.signals["status"].tolist() == ["skipped_min_notional"]


def test_priority_and_skip_reasons(cfg: Settings) -> None:
    market = FakeMarket()
    for sym in ("AAAUSDT", "BBBUSDT", "CCCUSDT"):
        market.frames[sym] = bars_from([(100, 100.5, 99.5, 100)] * 30)
    sigs = [
        signal(T0, symbol="AAAUSDT", tf="15m", rank=1),
        signal(T0, symbol="BBBUSDT", tf="1h", rank=5),  # 상위 TF → 먼저
        signal(T0, symbol="CCCUSDT", tf="1h", rank=2),  # 같은 TF 에서는 순위 상위(2) 가 5 보다 먼저
        signal(T0, symbol="AAAUSDT", tf="4h", rank=11),  # 순위 밖
        signal(T0 + Q * 2, symbol="AAAUSDT", tf="15m", rank=1),  # 보유 중
    ]
    res = run(cfg, market, sigs)
    assert res.signals["status"].tolist() == ["skipped_simultaneous", "skipped_simultaneous", "entered",
                                              "out_of_rank", "skipped_holding"]
    assert res.trades.iloc[0].symbol == "CCCUSDT"


def test_daily_loss_blocks_until_next_kst_day(tmp_path: Path) -> None:
    s = settings_with(tmp_path, {"strategy": {"exit": {"stop": {"atr_mult": 5.0}}}})
    market = FakeMarket()
    # 진입 후 급락해 손절(99−5=94) → 약 −6% 손실 → 일일 손실 3% 발동
    rows = [(100, 100.5, 99.5, 100), (99, 99, 93, 93.5)] + [(93.5, 94, 93, 93.5)] * 200
    market.frames["AAAUSDT"] = bars_from(rows)
    market.frames["BBBUSDT"] = bars_from([(100, 100.5, 99.5, 100)] * 202)
    later_same_day = T0 + pd.Timedelta(hours=3)
    next_day = T0 + pd.Timedelta(hours=15)  # 00:00 KST 다음 날 = 15:00 UTC
    sigs = [signal(T0), signal(later_same_day, symbol="BBBUSDT"), signal(next_day, symbol="BBBUSDT")]
    res = run(s, market, sigs)
    assert res.signals["status"].tolist() == ["entered", "skipped_daily_loss", "entered"]
    assert [e.kind for e in res.risk_events] == ["daily_loss"]
    free = run(s, market, sigs, Scenario("no_rules", live_rules=False))
    assert free.signals["status"].tolist()[1] == "entered"


def test_kill_switch_blocks_and_flattens(tmp_path: Path) -> None:
    s = settings_with(tmp_path, {"strategy": {"exit": {"stop": {"atr_mult": 50.0}}},
                                 "risk": {"daily_loss": {"enabled": False},
                                          "kill_switch": {"enabled": True, "on_trip": "flatten_all"}}})
    market = FakeMarket()
    rows = [(100, 100.5, 99.5, 100), (90, 90, 60, 65), (66, 70, 64, 68), (68, 69, 67, 68)] + [(68, 69, 67, 68)] * 20
    market.frames["AAAUSDT"] = bars_from(rows)
    market.frames["BBBUSDT"] = bars_from([(100, 100.5, 99.5, 100)] * 24)
    res = run(s, market, [signal(T0), signal(T0 + Q * 6, symbol="BBBUSDT")])
    tr = res.trades.iloc[0]
    assert tr.exit_reason == "kill_switch" and tr.exit_time == T0 + Q * 2 and tr.exit_price == pytest.approx(66 * 0.9998)
    assert res.signals["status"].tolist() == ["entered", "skipped_kill_switch"]
    assert res.risk_events[0].kind == "kill_switch" and res.risk_events[0].value >= 0.30


def test_liquidation_before_stop_with_leverage(tmp_path: Path) -> None:
    s = settings_with(tmp_path, {"base": {"exchange": {"leverage": 5}},
                                 "strategy": {"exit": {"stop": {"atr_mult": 20.0}}}})
    market = FakeMarket()
    market.frames["AAAUSDT"] = bars_from([(100, 100.5, 99.5, 100), (99, 99, 70, 75)])
    tr = run(s, market, [signal(T0)]).trades.iloc[0]
    entry = 100 * 1.0002
    liq = entry * (1 - 1 / 5) / (1 - 0.004)
    assert tr.exit_reason == "liquidation" and tr.exit_price == pytest.approx(liq)
    assert tr.notional == pytest.approx(tr.qty * entry) and tr.qty == pytest.approx(np.floor(5000 * 0.998 / 100 / 0.001) * 0.001)


def test_delisting_closes_at_last_trade(cfg: Settings) -> None:
    market = FakeMarket()
    market.frames["AAAUSDT"] = bars_from([(100, 100.5, 99.5, 100), (100, 100.5, 99.5, 98)])
    market.delisted["AAAUSDT"] = T0 + Q * 2
    tr = run(cfg, market, [signal(T0)]).trades.iloc[0]
    assert tr.exit_reason == "delisted" and tr.exit_price == 98 and tr.exit_time == T0 + Q * 2


def test_trailing_and_structure_targets(tmp_path: Path) -> None:
    trailing = settings_with(tmp_path, {"strategy": {"exit": {"take_profit": {"mode": "trailing", "trailing_atr_mult": 2.0}}}})
    market = FakeMarket()
    rows = [(100, 101, 99.5, 101), (101, 105, 100.5, 104), (104, 106, 103.5, 105), (105, 105, 103.9, 104)]
    market.frames["AAAUSDT"] = bars_from(rows)
    tr = run(trailing, market, [signal(T0)]).trades.iloc[0]
    # 봉 마감마다 최고가 − 2: 101 → 99, 105 → 103, 106 → 104. 3번 봉 저가 103.9 ≤ 104 → 트레일링 손절
    assert tr.exit_reason == "trailing_stop" and tr.exit_price == pytest.approx(104 * 0.9998)
    structure = tmp_path / "s"
    structure.mkdir()
    st = settings_with(structure, {"strategy": {"exit": {"take_profit": {"mode": "structure"}}}})
    market.frames["AAAUSDT"] = bars_from([(100, 101, 99.5, 100.5), (100.5, 104, 100, 103)])
    tr2 = run(st, market, [signal(T0, swing=103.5)]).trades.iloc[0]
    assert tr2.exit_reason == "target" and tr2.exit_price == 103.5
    passed = run(st, market, [signal(T0, swing=99.0)]).trades.iloc[0]  # 목표가 이미 진입가 아래
    assert passed.exit_reason == "target_passed"


def test_switch_on_opposite_signal(tmp_path: Path) -> None:
    s = settings_with(tmp_path, {"strategy": {"positioning": {"opposite_signal": "switch"}}})
    market = FakeMarket()
    market.frames["AAAUSDT"] = bars_from([(100, 100.5, 99.5, 100)] * 10)
    res = run(s, market, [signal(T0), signal(T0 + Q * 3, side="short", high=100.5)])
    assert res.trades.iloc[0].exit_reason == "switch" and res.trades.iloc[0].exit_time == T0 + Q * 3
    assert res.signals["status"].tolist() == ["entered", "entered"]
    ignore = run(settings_with(tmp_path), market, [signal(T0), signal(T0 + Q * 3, side="short", high=100.5)])
    assert ignore.signals["status"].tolist() == ["entered", "skipped_holding"]


def test_equity_consistency_many_trades(cfg: Settings) -> None:
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, 3000)))
    opens = np.concatenate([[100], close[:-1]])
    high = np.maximum(opens, close) * (1 + rng.uniform(0, 0.003, 3000))
    low = np.minimum(opens, close) * (1 - rng.uniform(0, 0.003, 3000))
    market = FakeMarket()
    market.frames["AAAUSDT"] = bars_from(list(zip(opens, high, low, close, strict=True)))
    idx = market.frames["AAAUSDT"].index
    sigs = [signal(idx[i], side="long" if i % 2 else "short", close=close[i - 1], low=low[i - 1], high=high[i - 1],
                   atr=close[i - 1] * 0.005) for i in range(10, 2900, 37)]
    res = run(cfg, market, sigs, end=idx[-1] + Q)
    assert len(res.trades) > 20
    assert res.equity.iloc[-1] == pytest.approx(1000 + res.trades["net_pnl"].sum())
    assert (res.trades["entry_time"] >= res.trades["signal_time"]).all()
    assert (res.trades["exit_time"].iloc[:-1].to_numpy() <= res.trades["entry_time"].iloc[1:].to_numpy()).all()


def test_metrics_basics() -> None:
    from zoneinfo import ZoneInfo

    from perpdiv.backtest.metrics import max_drawdown, performance, trade_stats, yearly_returns

    tz = ZoneInfo("Asia/Seoul")
    start, end = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")
    days = pd.date_range(start, end, freq="D", tz="UTC")
    equity = pd.Series(1000 * 1.001 ** np.arange(len(days)), index=days)
    perf = performance(equity, start, end, tz)
    assert perf["total_return"] == pytest.approx(1.001 ** (len(days) - 1) - 1, rel=1e-9)
    assert perf["mdd"] == 0 and perf["sharpe"] > 10  # 매일 같은 수익 → 변동성 ~0
    dip = pd.Series([100.0, 120.0, 90.0, 130.0, 104.0])
    assert max_drawdown(dip) == pytest.approx(0.25)
    years = yearly_returns(equity, start, end, tz)
    assert set(years) == {2024, 2025, 2026} and years[2025] == pytest.approx(1.001 ** 365 - 1, rel=1e-3)
    trades = pd.DataFrame({"net_pnl": [10.0, -5.0, 20.0, -5.0], "r_multiple": [1.0, -0.5, 2.0, -0.5],
                           "holding_hours": [1.0, 2.0, 3.0, 4.0]})
    st = trade_stats(trades)
    assert st["win_rate"] == 0.5 and st["payoff"] == pytest.approx(3.0) and st["expectancy_r"] == pytest.approx(0.5)
    assert st["profit_factor"] == pytest.approx(3.0)



# --- 전략 수정 후 규칙: 다중 TF 동시 성립, 진입가 기준 손절, 최소 손절폭 ---------------------------------------------


def test_confluence_annotation() -> None:
    from perpdiv.signals.confluence import annotate_confluence

    s = load_settings().strategy  # validity_bars 3
    sigs = pd.DataFrame([
        signal(T0, tf="1h"),                                   # 0: 혼자 → 1
        signal(T0 + Q * 2, tf="15m"),                          # 1: 1h(T0~T0+3h) 유효 → 2 (1h+15m)
        signal(T0 + Q * 3, tf="15m"),                          # 2: 같은 TF 15m 여러 개는 1개 → 2
        signal(T0 + pd.Timedelta(hours=3), tf="15m"),          # 3: 1h·15m 만료(끝 미포함), 6번 4h(~T0+13h30m) 유효 → 2
        signal(T0 + Q * 4, tf="15m", side="short"),            # 4: 반대 방향은 안 셈 → 1
        signal(T0 + Q * 5, tf="4h", symbol="BBBUSDT"),         # 5: 다른 코인 → 1
        signal(T0 + Q * 6, tf="4h"),                           # 6: T0+1h30m
    ])
    out = annotate_confluence(sigs, s)
    assert out["confluence"].tolist()[:6] == [1, 2, 2, 2, 1, 1]
    assert out["confluence_tfs"].iloc[3] == "4h+15m"
    # 6번 시각 T0+1h30m: 1h(유효), 15m 2번(T0+45m ~ T0+1h30m, 끝 미포함 → 만료), 1번(T0+30m~T0+1h15m 만료) → 4h+1h
    assert out["confluence"].iloc[6] == 2 and out["confluence_tfs"].iloc[6] == "4h+1h"
    assert out["confluence_tfs"].iloc[1] == "1h+15m"
    same_time = annotate_confluence(pd.DataFrame([signal(T0, tf="1d"), signal(T0, tf="4h")]), s)
    assert same_time["confluence"].tolist() == [2, 2]  # 같은 시각 확정도 서로 센다


CONFLUENCE_ON: dict[str, Any] = {"strategy": {"confluence": {"enabled": True}}}


def test_defaults_after_user_decisions() -> None:
    s = load_settings()
    assert not s.strategy.confluence.enabled  # 다중 TF 조건 원복 (TF 독립 진입)
    assert s.strategy.exit.stop.basis == "entry_price" and s.strategy.exit.stop.min_distance_pct == 0.005
    assert not s.risk.kill_switch.enabled and s.risk.daily_loss.enabled  # MDD 킬 스위치 제거, 일일 손실 유지


def test_default_rules_single_tf_entry_stop_and_no_kill_switch(tmp_path: Path) -> None:
    s = settings_with(tmp_path, {"strategy": {"exit": {"stop": {"atr_mult": 50.0}}}}, legacy=False)
    market = FakeMarket()
    rows = [(100, 100.5, 99.5, 100), (90, 90, 60, 65), (66, 70, 64, 68)] + [(68, 69, 67, 68)] * 60
    market.frames["AAAUSDT"] = bars_from(rows)
    market.frames["BBBUSDT"] = bars_from([(100, 100.5, 99.5, 100)] * 63)
    # 단일 TF 신호로 바로 진입, 손절 = 진입가 − 50 × ATR(1.0), 30%+ 낙폭에도 킬 스위치 없음 → 다음 날(00:00 KST 이후) 신호는 진입
    res = run(s, market, [signal(T0, tf="15m"), signal(T0 + pd.Timedelta(hours=15), symbol="BBBUSDT")])
    tr = res.trades.iloc[0]
    assert tr.stop == pytest.approx(100 * 1.0002 - 50.0)
    assert res.signals["status"].tolist()[0] == "entered" and "kill_switch" not in [e.kind for e in res.risk_events]


def test_confluence_gate_and_entry_stop(tmp_path: Path) -> None:
    s = settings_with(tmp_path, CONFLUENCE_ON, legacy=False)
    assert s.strategy.confluence.enabled and s.strategy.exit.stop.basis == "entry_price"
    market = FakeMarket()
    market.frames["AAAUSDT"] = bars_from([(100, 100.5, 99.5, 100)] * 12 + [(100, 102, 90, 91)] * 4)
    sigs = [signal(T0, tf="1h", atr=0.8), signal(T0 + Q * 2, tf="15m", atr=0.4)]
    res = run(s, market, sigs)
    assert res.signals["status"].tolist() == ["no_confluence", "entered"]
    tr = res.trades.iloc[0]
    entry = 100 * 1.0002
    assert tr.timeframe == "15m" and tr.confluence_tfs == "1h+15m"
    assert tr.stop == pytest.approx(entry - 2.5 * 0.4)  # 진입가 − 2.5 × ATR(확정 신호 15m)
    assert tr.target == pytest.approx(entry + 2 * 2.5 * 0.4)
    assert tr.exit_reason == "stop" and tr.exit_time == T0 + Q * 12


def test_min_stop_distance_filter(tmp_path: Path) -> None:
    s = settings_with(tmp_path, CONFLUENCE_ON, legacy=False)
    market = FakeMarket()
    market.frames["AAAUSDT"] = bars_from([(100, 100.5, 99.5, 100)] * 30)
    # 손절폭 2.5 × ATR: ATR 0.2 → 0.5 = 진입가(100)의 0.5% → "보다 커야" 하므로 불성립. ATR 0.21 → 0.525 → 진입
    tight = run(s, market, [signal(T0, tf="1h", atr=0.2), signal(T0 + Q, tf="15m", atr=0.2)])
    assert tight.signals["status"].tolist() == ["no_confluence", "skipped_stop_too_tight"]
    ok = run(s, market, [signal(T0, tf="1h", atr=0.2), signal(T0 + Q, tf="15m", atr=0.21)])
    assert ok.signals["status"].tolist() == ["no_confluence", "entered"]


def test_time_exit_uses_trigger_timeframe(tmp_path: Path) -> None:
    s = settings_with(tmp_path, {"strategy": {"confluence": {"enabled": True}, "exit": {"time_exit": {"bars": 2}}}},
                      legacy=False)
    market = FakeMarket()
    market.frames["AAAUSDT"] = bars_from([(100, 100.5, 99.5, 100)] * 40)
    # 15m 이 먼저, 1h 가 45분 안에 확정 → 확정 신호 = 1h → 시간 청산 2 × 1h
    res = run(s, market, [signal(T0, tf="15m", atr=1.0), signal(T0 + Q * 2, tf="1h", atr=1.0)])
    tr = res.trades.iloc[0]
    assert tr.timeframe == "1h" and tr.exit_reason == "time" and tr.exit_time == T0 + Q * 2 + pd.Timedelta(hours=2)



def test_min_stop_distance_filter_single_tf(tmp_path: Path) -> None:
    s = settings_with(tmp_path, legacy=False)  # 기본(TF 독립)에서도 0.5% 조건은 진입 조건
    market = FakeMarket()
    market.frames["AAAUSDT"] = bars_from([(100, 100.5, 99.5, 100)] * 30)
    res = run(s, market, [signal(T0, atr=0.2), signal(T0 + Q * 20, atr=0.21)])
    assert res.signals["status"].tolist() == ["skipped_stop_too_tight", "entered"]
