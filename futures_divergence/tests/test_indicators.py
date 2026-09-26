"""RSI·ATR: 외부 라이브러리(pandas_ta_classic, TA-Lib)와 비교, 손 계산, 증분 = 일괄, 워밍업."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import pandas_ta_classic as pta
import pytest
import talib

from perpdiv.indicators.atr import AtrState, atr_wilder
from perpdiv.indicators.rsi import RsiState, rsi_wilder


def prices(n: int = 3000, seed: int = 7, tick: float = 0.1) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    path = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    return np.round(path / tick) * tick  # 호가 반올림 → 변화량 0 인 봉 포함


def ohlc(n: int = 3000, seed: int = 11) -> tuple[npt.NDArray[np.float64], ...]:
    rng = np.random.default_rng(seed)
    close = prices(n, seed)
    return close + np.round(rng.uniform(0, 0.8, n), 1), close - np.round(rng.uniform(0, 0.8, n), 1), close


@pytest.mark.parametrize("period", [2, 9, 14, 21])
@pytest.mark.parametrize("seed", [1, 2])
def test_rsi_matches_pandas_ta_and_talib(period: int, seed: int) -> None:
    close = prices(seed=seed)
    ours = rsi_wilder(close, period)
    ref_pta = pta.rsi(pd.Series(close), length=period, talib=False).to_numpy()
    ref_talib = talib.RSI(close, timeperiod=period)
    assert np.array_equal(np.isnan(ours), np.isnan(ref_talib))  # 워밍업: 앞 period 개 NaN
    valid = ~np.isnan(ours)
    np.testing.assert_allclose(ours[valid], ref_talib[valid], rtol=0, atol=1e-10)
    np.testing.assert_allclose(ours[valid], ref_pta[valid], rtol=0, atol=1e-10)


def test_rsi_hand_calculated_and_edges() -> None:
    # period 2: 변화 +1, −2 → 평균 상승 0.5, 하락 1.0 → 33.33 / 다음 +3 → (0.5+3)/2=1.75, (1+0)/2=0.5 → 77.78
    out = rsi_wilder([10.0, 11.0, 9.0, 12.0], 2)
    assert np.isnan(out[:2]).all()
    assert out[2] == pytest.approx(100 / 3) and out[3] == pytest.approx(100 * 1.75 / 2.25)
    assert rsi_wilder(np.full(30, 5.0), 14)[-1] == 50.0  # 완전 횡보
    assert rsi_wilder(np.arange(30, dtype=float), 14)[-1] == 100.0
    assert np.isnan(rsi_wilder(np.arange(14, dtype=float), 14)).all()  # 변화량 13개 < 14
    with pytest.raises(ValueError):
        rsi_wilder([1.0, np.nan, 2.0], 2)


@pytest.mark.parametrize("seed", range(10))
def test_rsi_warmup_convergence(seed: int) -> None:
    """시작점이 다른 두 계산(앞 700봉 유무)의 차이: 첫 유효값 뒤 130봉에 0.01, 260봉에 1e-6 아래 (워밍업 근거)."""
    rng = np.random.default_rng(seed)
    close = np.round(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 1500))), 1)
    diff = np.abs(rsi_wilder(close, 14)[700:] - rsi_wilder(close[700:], 14))
    assert diff[14] > 1e-3  # 처음에는 차이가 크다
    assert np.nanmax(diff[14 + 130:]) < 1e-2
    assert np.nanmax(diff[14 + 260:]) < 1e-6


@pytest.mark.parametrize("period", [2, 14])
def test_incremental_rsi_and_atr_bit_identical(period: int) -> None:
    high, low, close = ohlc()
    rsi_state, atr_state = RsiState(period), AtrState(period)
    rsi_stream = np.array([np.nan if (v := rsi_state.update(float(c))) is None else v for c in close])
    atr_stream = np.array([np.nan if (v := atr_state.update(float(h), float(lo), float(c))) is None else v
                           for h, lo, c in zip(high, low, close, strict=True)])
    assert np.array_equal(rsi_stream, rsi_wilder(close, period), equal_nan=True)
    assert np.array_equal(atr_stream, atr_wilder(high, low, close, period), equal_nan=True)


@pytest.mark.parametrize("period", [2, 14, 21])
def test_atr_matches_talib_and_pandas_ta(period: int) -> None:
    high, low, close = ohlc()
    ours = atr_wilder(high, low, close, period)
    ref = talib.ATR(high, low, close, timeperiod=period)
    valid = ~np.isnan(ref)
    assert np.array_equal(np.isnan(ours), np.isnan(ref))
    np.testing.assert_allclose(ours[valid], ref[valid], rtol=0, atol=1e-10)
    ref_pta = pta.atr(pd.Series(high), pd.Series(low), pd.Series(close), length=period, talib=False).to_numpy()
    both = valid & ~np.isnan(ref_pta)
    np.testing.assert_allclose(ours[both], ref_pta[both], rtol=0, atol=1e-10)
