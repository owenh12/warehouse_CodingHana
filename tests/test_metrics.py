"""성과 지표 (손 계산 값과 비교)와 백테스트 보고서 생성."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_divergence import random_frame

from rsidiv.backtest.engine import Scenario, default_scenarios
from rsidiv.backtest.metrics import buy_and_hold, period_returns, return_metrics, trade_metrics
from rsidiv.broker.costs import CostModel
from rsidiv.broker.instruments import CryptoInstrument
from rsidiv.core.config import apply_dotted, load_settings
from rsidiv.core.models import AssetClass
from rsidiv.strategy.controller import SignalRecord, TradeRecord

SETTINGS = load_settings()
TZ = "Asia/Seoul"


def equity_series(values: list[float], start: str = "2025-01-01 15:00") -> pd.Series:
    """KST 자정(= 15:00 UTC) 직전 봉 종가마다 한 값 → 일별 값과 같다."""
    idx = pd.date_range(start, periods=len(values), freq="1D", tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


def test_return_metrics_hand_calculated() -> None:
    eq = equity_series([110.0, 99.0, 108.9, 120.0])
    m = return_metrics(eq, 100.0, TZ)
    assert m.total_return == pytest.approx(0.20)
    daily = np.array([0.10, -0.10, 0.10, 120 / 108.9 - 1])
    assert m.sharpe == pytest.approx(daily.mean() / daily.std(ddof=1) * math.sqrt(365))
    downside = math.sqrt(np.mean(np.minimum(daily, 0) ** 2))
    assert m.sortino == pytest.approx(daily.mean() / downside * math.sqrt(365))
    assert m.max_drawdown == pytest.approx(0.10)  # 110 → 99
    assert m.max_drawdown_days == pytest.approx(2.0)  # 99(1일) → 120(3일)에 회복
    days = (eq.index[-1] - eq.index[0]).total_seconds() / 86_400
    assert m.cagr == pytest.approx(1.2 ** (365.25 / days) - 1)
    assert m.calmar == pytest.approx(m.cagr / 0.10)


def test_period_returns_by_year() -> None:
    eq = equity_series([100.0, 110.0], start="2025-12-30 15:00")  # KST 2025-12-31, 2026-01-01
    years = period_returns(eq, 100.0, TZ)
    assert years.iloc[0] == pytest.approx(0.0) and years.iloc[1] == pytest.approx(0.10)


def trade(net: float, *, risk: float = 10.0, exit_reason: str = "stop_loss") -> TradeRecord:
    ts = pd.Timestamp("2025-01-01", tz="UTC")
    sig = SignalRecord("crypto", "BTC/USDT", ts, ts, ts, ts, 1, 1, 1, 20, 30)
    # 수량 1, 진입 100, 손절 100 − risk → 1R = risk. 청산가로 순손익을 맞춘다 (비용 0)
    t = TradeRecord(1, "crypto", "BTC/USDT", sig, ts, 100.0, 1.0, 100.0 - risk, None, risk, False, ts, 100.0, 0.0,
                    lowest=95.0, highest=110.0)
    t.exit_time, t.exit_price, t.exit_reason, t.bars_held = ts, 100.0 + net, exit_reason, 4
    return t


def test_trade_metrics_hand_calculated() -> None:
    trades = [trade(20, exit_reason="take_profit"), trade(-10), trade(-10), trade(5, exit_reason="time_exit")]
    m = trade_metrics(trades)
    assert m.trades == 4 and m.win_rate == pytest.approx(0.5)
    assert m.expectancy_r == pytest.approx((2 - 1 - 1 + 0.5) / 4)
    assert m.profit_factor == pytest.approx(25 / 20)
    assert m.max_consecutive_losses == 2 and m.avg_bars_held == 4
    assert m.exits == {"take_profit": 1, "stop_loss": 2, "time_exit": 1}
    assert m.avg_mae_r == pytest.approx(0.5) and m.avg_mfe_r == pytest.approx(1.0)


def test_buy_and_hold_equal_weight() -> None:
    idx = pd.date_range("2025-01-01", periods=3, freq="15min", tz="UTC")
    a = pd.DataFrame({"open": [10.0, 11, 12], "high": 13.0, "low": 9.0, "close": [11.0, 12, 20]}, index=idx)
    b = pd.DataFrame({"open": [100.0, 100, 100], "high": 110.0, "low": 90.0, "close": [100.0, 90, 50]}, index=idx)
    inst = {s: CryptoInstrument(s, 0.001, 0.001, 0.0, 0.01) for s in ("A", "B")}
    gross = CostModel(SETTINGS.costs, AssetClass.CRYPTO, slippage_multiplier=0, include_fees=False)
    value = buy_and_hold({"A": a, "B": b}, 1000.0, costs=gross, instruments=inst)
    assert value.index[0] == idx[0] + pd.Timedelta("15min")
    assert value.iloc[-1] == pytest.approx(50 * 20 + 5 * 50)


def test_default_scenarios_follow_config() -> None:
    names = [s.name for s in default_scenarios(SETTINGS)]
    assert names == ["base", "gross", "slip_x0", "slip_x2", "no_live_rules"]
    gross = default_scenarios(SETTINGS)[1]
    assert gross.slippage_multiplier == 0 and not gross.include_fees


def test_backtest_report_files(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from rsidiv.reports.backtest_report import build_backtest_report

    settings = apply_dotted(SETTINGS, {
        "strategy.default.filters.rsi_t1_oversold.threshold": 45.0,
        "strategy.default.filters.rsi_diff_min.min_diff": 0.5,
        "backtest.outputs.signal_sample_count": 2,
    })
    data = {"BTC/USDT": random_frame(3000, 1), "ETH/USDT": random_frame(3000, 101)}
    report = build_backtest_report(settings, data, scenarios=[Scenario("base", "기준"), Scenario("gross", "전", 0, False)],
                                   market_type="spot", out_dir=tmp_path)
    base = report.scenarios[0]
    assert base.trades.trades > 0 and report.scenarios[1].trades.fees == 0
    for name in ("summary.md", "trades.csv", "signals.csv", "equity.csv", "returns.png", "drawdown.png",
                 "r_distribution.png", "trades/trade_01.png", "trades/trade_02.png"):
        assert (tmp_path / name).is_file(), name
    trades = pd.read_csv(tmp_path / "trades.csv")
    assert set(trades["scenario"]) == {"base", "gross"}
    assert trades.loc[trades["scenario"] == "base", "net_pnl"].sum() == pytest.approx(
        base.returns.final - base.returns.initial)
    summary = report.summary_path.read_text(encoding="utf-8")
    assert "## 시나리오 비교" in summary and "매수 후 보유" in summary
