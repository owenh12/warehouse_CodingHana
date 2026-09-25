"""실데이터 검증 (바이낸스 아카이브). 기본 실행에서는 건너뛴다.

    RSIDIV_REAL_DATA=1 python -m pytest tests/test_real_data.py

처음 실행하면 data.binance.vision 에서 받아 ``storage/cache`` 에 저장하고, 이후에는 캐시를 쓴다.
- 결측·OHLC 모순 없음
- RSI·ATR 이 TA-Lib 과 일치, 증분 RSI 가 일괄 계산과 비트 단위로 같음
- 피벗: 일괄 = 증분, prefix 불변성
- 1분봉 → 15분봉 리샘플이 바이낸스 네이티브 15분봉과 같음
"""

from __future__ import annotations

import datetime as dt
import os
from functools import cache

import numpy as np
import pandas as pd
import pytest
import talib

from rsidiv.core.config import Settings, load_settings
from rsidiv.core.timeutil import UTC
from rsidiv.data.factory import binance_provider
from rsidiv.data.quality import check_ohlcv, continuous_index
from rsidiv.data.resample import resample_ohlcv
from rsidiv.indicators.atr import atr_wilder
from rsidiv.indicators.pivots import PivotTracker, detect_pivots
from rsidiv.indicators.rsi import RsiState, rsi_wilder

pytestmark = pytest.mark.skipif(
    os.environ.get("RSIDIV_REAL_DATA") != "1", reason="실데이터 검증은 RSIDIV_REAL_DATA=1 일 때만"
)

START = dt.datetime(2025, 1, 1, tzinfo=UTC)
END = dt.datetime(2026, 9, 1, tzinfo=UTC)  # 월별 아카이브가 게시된 마지막 달의 끝
CASES = [(m, s) for m in ("spot", "usdm_futures") for s in ("BTC/USDT", "ETH/USDT")]


@cache
def settings() -> Settings:
    return load_settings()


@cache
def bars(market: str, symbol: str, timeframe: str = "15m",
         start: dt.datetime = START, end: dt.datetime = END) -> pd.DataFrame:
    provider = binance_provider(settings(), market, source="vision")  # type: ignore[arg-type]
    return provider.fetch_ohlcv(symbol, timeframe, start, end)


@pytest.mark.parametrize(("market", "symbol"), CASES)
def test_archive_is_complete(market: str, symbol: str) -> None:
    frame = bars(market, symbol)
    report = check_ohlcv(frame, continuous_index(START, END, "15m"))
    assert report.missing_bars == 0 and report.ohlc_violations == 0
    assert frame.index[0] == pd.Timestamp(START) and frame.index[-1] == pd.Timestamp(END) - pd.Timedelta("15min")


@pytest.mark.parametrize(("market", "symbol"), CASES)
def test_indicators_match_talib(market: str, symbol: str) -> None:
    frame = bars(market, symbol)
    high, low, close = (frame[c].to_numpy() for c in ("high", "low", "close"))
    rsi = rsi_wilder(close, 14)
    np.testing.assert_allclose(rsi, talib.RSI(close, 14), rtol=0, atol=1e-9)
    np.testing.assert_allclose(atr_wilder(high, low, close, 14), talib.ATR(high, low, close, 14), rtol=1e-12)
    state = RsiState(14)
    streamed = np.array([np.nan if (v := state.update(float(c))) is None else v for c in close])
    assert np.array_equal(streamed, rsi, equal_nan=True)


@pytest.mark.parametrize("symbol", ["BTC/USDT", "ETH/USDT"])
def test_pivots_incremental_and_prefix_invariant(symbol: str) -> None:
    frame = bars("spot", symbol)
    cfg = settings().strategy.for_asset_class("crypto").pivot
    pivots = detect_pivots(frame, cfg)
    tracker = PivotTracker(cfg)
    streamed = []
    for i, (h, lo, c) in enumerate(zip(frame["high"], frame["low"], frame["close"], strict=True)):
        found = tracker.update(float(h), float(lo), float(c))
        assert all(p.confirm_index == i for p in found)
        streamed += found
    assert streamed == pivots
    for k in np.random.default_rng(0).integers(cfg.left + cfg.right + 1, len(frame), 25):
        assert detect_pivots(frame.iloc[:k], cfg) == [p for p in pivots if p.confirm_index < k]


def test_resampled_1m_matches_native_15m() -> None:
    month, month_end = dt.datetime(2026, 8, 1, tzinfo=UTC), END
    native = bars("spot", "BTC/USDT", "15m", month, month_end)
    resampled = resample_ohlcv(bars("spot", "BTC/USDT", "1m", month, month_end), "15m", source_timeframe="1m")
    assert resampled.index.equals(native.index)
    pd.testing.assert_frame_equal(resampled[["open", "high", "low", "close"]],
                                  native[["open", "high", "low", "close"]], check_freq=False)
    np.testing.assert_allclose(resampled["volume"], native["volume"], rtol=1e-12)
