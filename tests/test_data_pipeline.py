"""Parquet 캐시, 리샘플링, 품질 점검·결측 처리, 데이터 확보 점검 보고서."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fakes import synthetic_bar, utc

from rsidiv.core.config import load_settings
from rsidiv.data.base import (
    DataProvider,
    DataSourceError,
    FundingProvider,
    ohlcv_from_rows,
    slice_range,
)
from rsidiv.data.binance import funding_series
from rsidiv.data.cache import CachedDataProvider, ParquetStore
from rsidiv.data.check import render_markdown, run_crypto_check
from rsidiv.data.quality import check_ohlcv, continuous_index, fill_missing_bars
from rsidiv.data.resample import resample_ohlcv


class CountingProvider(DataProvider, FundingProvider):
    """합성 15분봉을 주고 호출 구간을 기록하는 공급자."""

    def __init__(self, now: dt.datetime, first: dt.datetime | None = None,
                 fail: bool = False) -> None:
        self.now, self.first, self.fail = now, first or utc(2024, 1, 1), fail
        self.calls: list[tuple[dt.datetime, dt.datetime]] = []
        self.funding_calls: list[tuple[dt.datetime, dt.datetime]] = []

    def fetch_ohlcv(self, symbol: str, timeframe: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        if self.fail:
            raise DataSourceError("blocked")
        self.calls.append((start, end))
        lo = max(start, self.first)
        hi = min(end, self.now - dt.timedelta(minutes=15) + dt.timedelta(microseconds=1))
        times = pd.date_range(pd.Timestamp(lo).ceil("15min"), hi, freq="15min", inclusive="left")
        rows = [synthetic_bar(int(t.timestamp() * 1000)) for t in times]
        return slice_range(ohlcv_from_rows(rows), start, end)

    def earliest_available(self, symbol: str, timeframe: str) -> dt.datetime | None:
        if self.fail:
            raise DataSourceError("blocked")
        return self.first

    def funding_rates(self, symbol: str, start: dt.datetime, end: dt.datetime) -> pd.Series:
        self.funding_calls.append((start, end))
        t0 = int(utc(2024, 1, 1).timestamp() * 1000)
        records = {t0 + 8 * 3600_000 * i: 0.0001 for i in range(3000)}
        return funding_series({k: v for k, v in records.items() if k <= self.now.timestamp() * 1000}, start, end)


# ---------------------------------------------------------------------------
# 캐시
# ---------------------------------------------------------------------------


def cached(tmp_path: Path, inner: CountingProvider, clock: list[dt.datetime], refresh_days: int = 3) -> CachedDataProvider:
    return CachedDataProvider(inner, ParquetStore(tmp_path), "binance/spot",
                              refresh_recent_days=refresh_days, source="fake", clock=lambda: clock[0])


def test_cache_final_months_are_not_refetched(tmp_path: Path) -> None:
    now = [utc(2025, 6, 1)]
    inner = CountingProvider(now[0])
    provider = cached(tmp_path, inner, now)
    first = provider.fetch_ohlcv("BTC/USDT", "15m", utc(2025, 1, 10), utc(2025, 3, 20))
    assert len(inner.calls) == 3  # 1·2·3월 파티션
    second = provider.fetch_ohlcv("BTC/USDT", "15m", utc(2025, 1, 10), utc(2025, 3, 20))
    assert len(inner.calls) == 3
    pd.testing.assert_frame_equal(first, second)
    direct = inner.fetch_ohlcv("BTC/USDT", "15m", utc(2025, 1, 10), utc(2025, 3, 20))
    pd.testing.assert_frame_equal(first, direct)
    manifest = ParquetStore(tmp_path).manifest("binance/spot/BTC-USDT/15m")
    assert set(manifest) == {"2025-01", "2025-02", "2025-03"}
    assert all(info.final and info.source == "fake" for info in manifest.values())


def test_cache_refreshes_recent_partitions(tmp_path: Path) -> None:
    now = [utc(2025, 3, 2, 12)]
    inner = CountingProvider(now[0])
    provider = cached(tmp_path, inner, now, refresh_days=3)
    provider.fetch_ohlcv("BTC/USDT", "15m", utc(2025, 1, 1), utc(2025, 4, 1))
    manifest = ParquetStore(tmp_path).manifest("binance/spot/BTC-USDT/15m")
    assert manifest["2025-01"].final
    assert not manifest["2025-02"].final  # 월말 + 3일이 지나지 않음
    assert not manifest["2025-03"].final
    calls_before = len(inner.calls)

    now[0] = inner.now = utc(2025, 3, 10)
    frame = provider.fetch_ohlcv("BTC/USDT", "15m", utc(2025, 1, 1), utc(2025, 4, 1))
    refetched = [start for start, _ in inner.calls[calls_before:]]
    assert refetched == [utc(2025, 2, 1), utc(2025, 3, 1)]  # 1월은 확정이라 제외
    assert frame.index[-1] == pd.Timestamp("2025-03-09 23:45", tz="UTC")
    assert ParquetStore(tmp_path).manifest("binance/spot/BTC-USDT/15m")["2025-02"].final


def test_cache_funding(tmp_path: Path) -> None:
    now = [utc(2025, 6, 1)]
    inner = CountingProvider(now[0])
    provider = cached(tmp_path, inner, now)
    a = provider.funding_rates("BTC/USDT", utc(2025, 1, 1), utc(2025, 2, 15))
    b = provider.funding_rates("BTC/USDT", utc(2025, 1, 1), utc(2025, 2, 15))
    assert len(inner.funding_calls) == 2 and len(a) == (31 + 14) * 3
    pd.testing.assert_series_equal(a, b)


# ---------------------------------------------------------------------------
# 리샘플링
# ---------------------------------------------------------------------------


def minute_bars(start: dt.datetime, count: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=count, freq="1min", tz="UTC").as_unit("ns").rename("time")
    base = np.arange(count, dtype=float)
    return pd.DataFrame({"open": 100 + base, "high": 101 + base, "low": 99 + base,
                         "close": 100.5 + base, "volume": np.ones(count)}, index=idx)


def test_resample_1m_to_15m_aggregation() -> None:
    frame = minute_bars(utc(2025, 1, 2, 0, 0), 45)
    out = resample_ohlcv(frame, "15m", source_timeframe="1m")
    assert list(out.index.strftime("%H:%M")) == ["00:00", "00:15", "00:30"]
    first = out.iloc[0]
    assert (first.open, first.high, first.low, first.close, first.volume) == (100, 115, 99, 114.5, 15)


def test_resample_drops_incomplete_tail() -> None:
    frame = minute_bars(utc(2025, 1, 2, 0, 0), 38)  # 00:37 까지 → 00:30 버킷 미완성
    out = resample_ohlcv(frame, "15m", source_timeframe="1m")
    assert list(out.index.strftime("%H:%M")) == ["00:00", "00:15"]
    # 마지막 1분이 무체결이어도 as_of 가 지났으면 완성 버킷
    frame = minute_bars(utc(2025, 1, 2, 0, 0), 44)
    kept = resample_ohlcv(frame, "15m", source_timeframe="1m", as_of=utc(2025, 1, 2, 0, 46))
    assert len(kept) == 3


def test_resample_15m_to_4h_aligned_to_utc_midnight() -> None:
    idx = pd.date_range("2025-01-01 02:00", "2025-01-02 00:00", freq="15min", tz="UTC", inclusive="left")
    rows = [synthetic_bar(int(t.timestamp() * 1000)) for t in idx]
    out = resample_ohlcv(ohlcv_from_rows(rows), "4h", source_timeframe="15m")
    assert list(out.index.strftime("%H:%M")) == ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"]
    block = ohlcv_from_rows(rows).loc["2025-01-01 04:00":"2025-01-01 07:45"]
    assert out.loc["2025-01-01 04:00", "high"] == block["high"].max()
    assert out.loc["2025-01-01 04:00", "open"] == block["open"].iloc[0]


def test_resample_rejects_non_multiple() -> None:
    with pytest.raises(ValueError):
        resample_ohlcv(minute_bars(utc(2025, 1, 1), 10), "10m", source_timeframe="15m")


# ---------------------------------------------------------------------------
# 품질 점검·결측
# ---------------------------------------------------------------------------


def test_quality_detects_gaps_and_violations() -> None:
    idx = pd.date_range("2025-01-01", periods=20, freq="15min", tz="UTC")
    rows = [synthetic_bar(int(t.timestamp() * 1000)) for t in idx]
    frame = ohlcv_from_rows(rows).drop(idx[[5, 6, 7, 12]])
    frame.iloc[0, frame.columns.get_loc("high")] = 1.0  # 고가 < 시가
    frame.iloc[1, frame.columns.get_loc("volume")] = 0.0
    report = check_ohlcv(frame, continuous_index(idx[0], idx[-1] + pd.Timedelta("15min"), "15m"))
    assert report.expected_bars == 20 and report.missing_bars == 4
    assert [(g.bars) for g in report.gaps] == [3, 1]
    assert report.gaps[0].start == idx[5].to_pydatetime() and report.gaps[0].end == idx[8].to_pydatetime()
    assert report.longest_gap_bars == 3
    assert report.ohlc_violations == 1 and report.zero_volume_bars == 1 and not report.is_clean


def test_fill_missing_bars_flat() -> None:
    idx = pd.date_range("2025-01-01", periods=6, freq="15min", tz="UTC")
    rows = [synthetic_bar(int(t.timestamp() * 1000)) for t in idx]
    frame = ohlcv_from_rows(rows).drop(idx[[2, 3]])
    expected = continuous_index(idx[0], idx[-1] + pd.Timedelta("15min"), "15m")
    filled = fill_missing_bars(frame, expected, "ffill_flat")
    assert len(filled) == 6 and filled["filled"].tolist() == [False, False, True, True, False, False]
    prev_close = frame["close"].iloc[1]
    assert (filled.loc[idx[2], ["open", "high", "low", "close"]] == prev_close).all()
    assert filled.loc[idx[3], "volume"] == 0.0
    kept = fill_missing_bars(frame, expected, "keep_gap")
    assert len(kept) == 4 and not kept["filled"].any()


# ---------------------------------------------------------------------------
# 데이터 확보 점검 보고서
# ---------------------------------------------------------------------------


def test_data_check_report() -> None:
    settings = load_settings()
    now = utc(2025, 3, 1)
    ok = CountingProvider(now, first=utc(2019, 9, 8, 17, 45))
    blocked = CountingProvider(now, fail=True)
    report = run_crypto_check(
        settings, {"usdm_futures": ok, "spot": blocked}, source="fake",
        start=utc(2025, 1, 1), end=utc(2025, 2, 1), now=now,
    )
    futures = [r for r in report.results if r.market_type == "usdm_futures"]
    assert [r.symbol for r in futures] == ["BTC/USDT", "ETH/USDT"]
    assert all(r.quality and r.quality.missing_bars == 0 and r.quality.rows == 31 * 96 for r in futures)
    assert all(r.funding and r.funding.count == 93 and r.funding.interval_hours == {8.0: 92} for r in futures)
    assert report.usable_period("usdm_futures") == (utc(2025, 1, 1), utc(2025, 2, 1))
    assert report.usable_period("spot") is None
    spot = [r for r in report.results if r.market_type == "spot"]
    assert all(len(r.errors) == 2 for r in spot)  # earliest, ohlcv

    text = render_markdown(report)
    assert "2019-09-08 17:45" in text
    assert "usdm_futures: 2025-01-01 00:00 ~ 2025-02-01 00:00 UTC" in text
    assert "spot: 확정 불가" in text
    assert "DataSourceError: blocked" in text
    assert "min_notional=100" in text  # 설정 대체값 표시
