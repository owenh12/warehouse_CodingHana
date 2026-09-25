"""RSI·ATR·이동평균: 외부 라이브러리(TA-Lib) 값과 직접 구현 값 비교, 증분 계산 일치."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
import talib

from rsidiv.indicators.atr import atr_wilder
from rsidiv.indicators.ma import ema, moving_average, sma
from rsidiv.indicators.rsi import RsiState, rsi_wilder

TOL = 1e-10


def price_path(n: int = 3000, seed: int = 7, tick: float = 0.1) -> npt.NDArray[np.float64]:
    """호가단위로 반올림한 랜덤워크 (변화량 0 인 봉 포함)."""
    rng = np.random.default_rng(seed)
    path = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    return np.round(path / tick) * tick


def ohlc(n: int = 3000, seed: int = 11) -> tuple[npt.NDArray[np.float64], ...]:
    rng = np.random.default_rng(seed)
    close = price_path(n, seed)
    high = close + np.round(rng.uniform(0, 0.8, n), 1)
    low = close - np.round(rng.uniform(0, 0.8, n), 1)
    return high, low, close


def assert_same_as_talib(ours: npt.NDArray[np.float64], ref: npt.NDArray[np.float64]) -> None:
    assert np.array_equal(np.isnan(ours), np.isnan(ref)), "NaN(워밍업) 구간이 다름"
    np.testing.assert_allclose(ours[~np.isnan(ours)], ref[~np.isnan(ref)], rtol=0, atol=TOL)


@pytest.mark.parametrize("period", [2, 5, 9, 14, 21, 50])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_rsi_matches_talib(period: int, seed: int) -> None:
    close = price_path(seed=seed)
    ours = rsi_wilder(close, period)
    assert_same_as_talib(ours, talib.RSI(close, timeperiod=period))
    assert np.isnan(ours[:period]).all() and not np.isnan(ours[period:]).any()


@pytest.mark.parametrize("period", [2, 14, 21])
def test_incremental_rsi_is_bit_identical(period: int) -> None:
    close = price_path(seed=5)
    state = RsiState(period)
    streamed = np.array([np.nan if (v := state.update(float(c))) is None else v for c in close])
    assert np.array_equal(streamed, rsi_wilder(close, period), equal_nan=True)


def test_rsi_edge_cases() -> None:
    assert rsi_wilder(np.full(40, 5.0), 14)[-1] == 50.0  # 완전 횡보 (TA-Lib 은 0)
    assert rsi_wilder(np.arange(40, dtype=float), 14)[-1] == 100.0
    assert rsi_wilder(np.arange(40, 0, -1, dtype=float), 14)[-1] == 0.0
    assert np.isnan(rsi_wilder(np.arange(14, dtype=float), 14)).all()  # 변화량 13개 < 14
    assert not np.isnan(rsi_wilder(np.arange(15, dtype=float), 14)[-1])
    with pytest.raises(ValueError):
        rsi_wilder([1.0, np.nan, 2.0], 2)


def test_rsi_hand_calculated() -> None:
    # period=2: 변화량 +1, -2 → seed 평균상승 0.5, 평균하락 1.0 → RSI 33.33
    # 다음 변화 +3 → 상승 (0.5*1+3)/2=1.75, 하락 (1.0*1+0)/2=0.5 → 77.78
    out = rsi_wilder([10.0, 11.0, 9.0, 12.0], 2)
    assert np.isnan(out[:2]).all()
    assert out[2] == pytest.approx(100 * 0.5 / 1.5)
    assert out[3] == pytest.approx(100 * 1.75 / 2.25)


@pytest.mark.parametrize("period", [2, 14, 21])
def test_atr_matches_talib(period: int) -> None:
    high, low, close = ohlc()
    assert_same_as_talib(atr_wilder(high, low, close, period), talib.ATR(high, low, close, timeperiod=period))


@pytest.mark.parametrize("period", [2, 20, 50, 200])
def test_moving_averages_match_talib(period: int) -> None:
    close = price_path(seed=9)
    assert_same_as_talib(ema(close, period), talib.EMA(close, timeperiod=period))
    np.testing.assert_allclose(sma(close, period), talib.SMA(close, timeperiod=period), atol=1e-9)
    assert np.array_equal(moving_average(close, period, "ema"), ema(close, period), equal_nan=True)
    with pytest.raises(ValueError):
        moving_average(close, period, "wma")
