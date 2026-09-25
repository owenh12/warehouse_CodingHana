"""강세 다이버전스 탐지기.

1. 독립 참조 구현(일괄 계산 + 단순 반복)과 모든 t3 후보의 판정 결과가 같다 (여러 파라미터 조합).
2. 미래참조 방지: 신호 봉 = t3 + R, prefix 불변성, 미래 봉 변경 불변성.
3. 필터별 동작 (손으로 만든 예), 국내주식 세션 필터, 상위 TF 추세 필터가 마감된 봉만 쓰는지.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from rsidiv.core.config import KrxCfg, StrategyParams, deep_merge, load_settings
from rsidiv.data.resample import BarAggregator, resample_ohlcv
from rsidiv.indicators.ma import EmaState, SmaState, ema, moving_average, sma
from rsidiv.indicators.pivots import detect_pivots
from rsidiv.indicators.rsi import rsi_wilder
from rsidiv.signals.divergence import (
    REASONS,
    DivergenceCandidate,
    DivergenceDetector,
    candidates_frame,
    detect_divergences,
)

SETTINGS = load_settings()
DEFAULT = SETTINGS.strategy.default
KRX = SETTINGS.markets.krx
BAR = pd.Timedelta("15min")

# 랜덤 데이터에서도 신호가 충분히 나오도록 완화한 필터
RELAXED: dict[str, Any] = {
    "filters": {
        "rsi_t1_oversold": {"threshold": 45.0},
        "rsi_diff_min": {"min_diff": 0.5},
        "price_drop_min": {"min_pct": 0.0005},
    }
}


def params(*updates: dict[str, Any]) -> StrategyParams:
    merged = DEFAULT.model_dump()
    for update in updates:
        merged = deep_merge(merged, update)
    return StrategyParams.model_validate(merged)


def crypto_index(n: int, start: str = "2025-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="15min", tz="UTC").as_unit("ns").rename("time")


def krx_index(days: int) -> pd.DatetimeIndex:
    """KRX 15분봉 격자 09:00~15:15 KST (= 00:00~06:15 UTC), 평일만."""
    stamps: list[pd.Timestamp] = []
    day = pd.Timestamp("2025-03-03", tz="UTC")
    while len(stamps) < days * 26:
        if day.weekday() < 5:
            stamps += [day + i * BAR for i in range(26)]
        day += pd.Timedelta("1D")
    return pd.DatetimeIndex(stamps).as_unit("ns").rename("time")


def random_frame(n: int, seed: int, tick: float = 0.0, index: pd.DatetimeIndex | None = None,
                 drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.006, n)))
    high = close * (1 + rng.uniform(0, 0.004, n))
    low = close * (1 - rng.uniform(0, 0.004, n))
    if tick:
        close, high, low = (np.round(x / tick) * tick for x in (close, high, low))
    return pd.DataFrame(
        {"open": close, "high": np.maximum(high, close), "low": np.minimum(low, close), "close": close,
         "volume": rng.lognormal(3, 1, n)},
        index=index if index is not None else crypto_index(n),
    )


# ---------------------------------------------------------------------------
# 독립 참조 구현
# ---------------------------------------------------------------------------


def _krx_excluded(ts: pd.Timestamp, p: StrategyParams, krx: KrxCfg) -> bool:
    local = ts.tz_convert(krx.timezone)
    minutes = local.hour * 60 + local.minute
    special = next((s for s in krx.special_sessions if s.date == local.date()), None)
    open_t, close_t = (special.open, special.close) if special else (krx.regular_open, krx.regular_close)
    o, c = open_t.hour * 60 + open_t.minute, close_t.hour * 60 + close_t.minute
    cfg = p.filters.session_kr
    auction = (krx.regular_close.hour * 60 + krx.regular_close.minute) - (
        krx.closing_auction.start.hour * 60 + krx.closing_auction.start.minute)
    first = o <= minutes < o + 15 * cfg.exclude_first_bars
    last = c - 15 * cfg.exclude_last_bars <= minutes < c
    in_auction = cfg.exclude_auction_bars and minutes < c and minutes + 15 > c - auction
    return first or last or in_auction


def reference(frame: pd.DataFrame, p: StrategyParams, asset_class: str = "crypto",
              krx: KrxCfg | None = None) -> list[tuple[Any, ...]]:
    """(t3, 신호 봉, t1, p2, 실패 사유 집합) 목록."""
    f, R = p.filters, p.pivot.right
    close = frame["close"].to_numpy()
    low = (frame["low"] if p.pivot.price_source == "low" else frame["close"]).to_numpy()
    high = (frame["high"] if p.pivot.price_source == "low" else frame["close"]).to_numpy()
    vol = frame["volume"].to_numpy()
    rsi = rsi_wilder(close, p.rsi.period)
    pivots = detect_pivots(frame, p.pivot)
    plows = [q.index for q in pivots if q.kind == "low"]
    phighs = [q.index for q in pivots if q.kind == "high"]
    ltf_ma = moving_average(close, f.trend.ma_period, f.trend.ma_type)

    def trend_ok(n: int) -> bool:
        if f.trend.mode == "ltf_ma":
            now, prev = ltf_ma[n], (ltf_ma[n - 1] if n > 0 else np.nan)
        else:
            htf = resample_ohlcv(frame.iloc[: n + 1], f.trend.htf, source_timeframe="15m",
                                 as_of=frame.index[n] + BAR)
            ma = moving_average(htf["close"].to_numpy(), f.trend.ma_period, f.trend.ma_type)
            now = ma[-1] if len(ma) else np.nan
            prev = ma[-2] if len(ma) > 1 else np.nan
        if np.isnan(now):
            return False
        above, rising = close[n] > now, (not np.isnan(prev)) and now > prev
        return {"close_above": above, "slope_up": rising, "close_above_and_slope_up": above and rising}[
            f.trend.condition]

    def mean(a: int, b: int) -> float | None:  # [a, b]
        return None if a < 0 else float(np.mean(vol[a : b + 1]))

    def volume_ok(t1: int, t3: int) -> bool:
        v = f.volume
        cur = mean(t3 - v.bars_before + 1, t3 + v.bars_after)
        if v.mode == "t3_vs_t1":
            base = mean(t1 - v.bars_before + 1, t1 + v.bars_after)
        else:
            s = t3 - v.bars_before + 1
            base = mean(s - v.lookback, s - 1)
        if cur is None or base is None:
            return False
        return cur > 0 if base == 0 else cur / base >= v.min_ratio - 1e-12

    def evaluate(t1: int, t3: int) -> tuple[Any, ...]:
        n = t3 + R
        between = [i for i in phighs if t1 < i < t3] if p.divergence.p2_selection == "highest_pivot" \
            else list(range(t1 + 1, t3))
        p2 = max(between, key=lambda i: (high[i], -i)) if between else None
        warm = not (np.isnan(rsi[t1]) or np.isnan(rsi[t3]))
        failed = set()
        if not warm:
            failed.add("rsi_warmup")
        if p2 is None:
            failed.add("no_p2")
        if not low[t3] < low[t1]:
            failed.add("price_not_lower_low")
        if warm and not rsi[t3] > rsi[t1]:
            failed.add("rsi_not_higher_low")
        if p2 is not None and not (high[p2] > low[t1] and high[p2] > low[t3]):
            failed.add("p2_not_above")
        if f.gap_bars.enabled and not f.gap_bars.min_bars <= t3 - t1 <= f.gap_bars.max_bars:
            failed.add("gap_bars")
        if warm and f.rsi_t1_oversold.enabled and not rsi[t1] < f.rsi_t1_oversold.threshold:
            failed.add("rsi_t1_oversold")
        if warm and f.rsi_diff_min.enabled and not rsi[t3] - rsi[t1] >= f.rsi_diff_min.min_diff:
            failed.add("rsi_diff_min")
        if f.price_drop_min.enabled and not (low[t1] - low[t3]) / low[t1] >= f.price_drop_min.min_pct:
            failed.add("price_drop_min")
        if f.no_lower_low_between.enabled and any(low[j] < low[t3] for j in range(t1 + 1, t3)):
            failed.add("no_lower_low_between")
        if f.trend.enabled and not trend_ok(n):
            failed.add("trend")
        if f.volume.enabled and not volume_ok(t1, t3):
            failed.add("volume")
        if asset_class == "stock_kr" and f.session_kr.enabled:
            assert krx is not None
            bars = {"signal_bar": [n], "t3_bar": [t3], "both": [t3, n]}[f.session_kr.applies_to]
            if any(_krx_excluded(frame.index[i], p, krx) for i in bars):
                failed.add("session_kr")
        return (t3, n, t1, p2, frozenset(failed))

    out = []
    for k, t3 in enumerate(plows):
        earlier = plows[:k]
        if p.divergence.t1_selection == "previous_pivot":
            options = earlier[-1:]
        else:
            options = [i for i in reversed(earlier)
                       if f.gap_bars.min_bars <= t3 - i <= f.gap_bars.max_bars]
        if not options:
            out.append((t3, t3 + R, None, None, frozenset({"no_t1"})))
            continue
        results = [evaluate(t1, t3) for t1 in options]
        out.append(next((r for r in results if not r[4]), results[0]))
    return out


def as_tuples(candidates: list[DivergenceCandidate]) -> list[tuple[Any, ...]]:
    return [(c.t3, c.signal_index, c.t1, c.p2, frozenset(c.failed)) for c in candidates]


VARIANTS: list[dict[str, Any]] = [
    {},
    RELAXED,
    deep_merge(RELAXED, {"divergence": {"t1_selection": "scan_window"}}),
    deep_merge(RELAXED, {"divergence": {"p2_selection": "max_high"}}),
    deep_merge(RELAXED, {"pivot": {"price_source": "close", "left": 3, "right": 2}}),
    deep_merge(RELAXED, {"pivot": {"strict": False}, "filters": {"no_lower_low_between": {"enabled": False}}}),
    deep_merge(RELAXED, {"filters": {"trend": {"enabled": True, "mode": "ltf_ma", "ma_period": 50,
                                                "condition": "close_above_and_slope_up"}}}),
    deep_merge(RELAXED, {"filters": {"trend": {"enabled": True, "mode": "htf_ma", "htf": "1h",
                                                "ma_type": "sma", "ma_period": 10, "condition": "slope_up"}}}),
    deep_merge(RELAXED, {"filters": {"volume": {"enabled": True, "mode": "t3_vs_average", "bars_before": 2,
                                                 "bars_after": 1, "lookback": 20, "min_ratio": 1.1}}}),
    deep_merge(RELAXED, {"filters": {"volume": {"enabled": True, "mode": "t3_vs_t1", "bars_before": 1,
                                                 "bars_after": 0, "min_ratio": 0.8}}}),
]


FILTER_OF_VARIANT = {6: "trend", 7: "trend", 8: "volume", 9: "volume"}


@pytest.mark.parametrize("variant", range(len(VARIANTS)))
@pytest.mark.parametrize("seed", [1, 2])
def test_matches_reference(variant: int, seed: int) -> None:
    p = params(VARIANTS[variant])
    frame = random_frame(3000, seed, tick=0.05 if seed == 2 else 0.0)
    got = detect_divergences(frame, p, timeframe="15m", asset_class="crypto")
    assert as_tuples(got) == reference(frame, p)
    evaluated = [c for c in got if c.t1 is not None and "rsi_warmup" not in c.failed]
    if variant in FILTER_OF_VARIANT:  # 필터가 통과·실패 두 경우 모두 검증되었는지
        name = FILTER_OF_VARIANT[variant]
        assert any(name in c.failed for c in evaluated) and any(name not in c.failed for c in evaluated)
    elif variant:
        assert sum(c.accepted for c in got) >= 5, "검증할 신호가 너무 적음"
    for c in got:
        assert list(c.failed) == [r for r in REASONS if r in c.failed]  # 사유는 정해진 순서


@pytest.mark.parametrize("applies_to", ["signal_bar", "t3_bar", "both"])
def test_stock_session_filter_matches_reference(applies_to: str) -> None:
    p = params(RELAXED, {"filters": {"session_kr": {"applies_to": applies_to}}})
    frame = random_frame(26 * 120, 3, index=krx_index(120))
    got = detect_divergences(frame, p, timeframe="15m", asset_class="stock_kr", krx=KRX)
    assert as_tuples(got) == reference(frame, p, "stock_kr", KRX)
    assert any("session_kr" in c.failed for c in got) and any(c.accepted for c in got)
    # 가상화폐에는 세션 필터가 적용되지 않는다
    crypto = detect_divergences(frame, p, timeframe="15m", asset_class="crypto")
    assert not any("session_kr" in c.failed for c in crypto)


# ---------------------------------------------------------------------------
# 미래참조 방지
# ---------------------------------------------------------------------------


def test_signal_time_is_close_of_t3_plus_r() -> None:
    p = params(RELAXED)
    frame = random_frame(2000, 4)
    for c in detect_divergences(frame, p, timeframe="15m", asset_class="crypto"):
        assert c.signal_index == c.t3 + p.pivot.right
        assert c.signal_time == frame.index[c.signal_index] + BAR
        if c.t1 is not None:
            assert c.t1 < c.t3 and (c.p2 is None or c.t1 < c.p2 < c.t3)


@pytest.mark.parametrize("variant", [1, 2, 7])
def test_prefix_invariance(variant: int) -> None:
    """앞 k 봉만 넣은 결과 = 전체 결과 중 신호 봉 < k 인 것 (모든 k)."""
    p = params(VARIANTS[variant])
    frame = random_frame(700, 5)
    full = as_tuples(detect_divergences(frame, p, timeframe="15m", asset_class="crypto"))
    for k in range(1, len(frame) + 1, 3):
        part = as_tuples(detect_divergences(frame.iloc[:k], p, timeframe="15m", asset_class="crypto"))
        assert part == [c for c in full if c[1] < k], k


@pytest.mark.parametrize("seed", [6, 7])
def test_future_bars_do_not_change_past_decisions(seed: int) -> None:
    p = params(VARIANTS[6])
    frame = random_frame(1500, seed)
    base = as_tuples(detect_divergences(frame, p, timeframe="15m", asset_class="crypto"))
    rng = np.random.default_rng(seed)
    for k in rng.integers(100, 1400, 5):
        changed = frame.copy()
        tail = slice(int(k), None)
        noise = rng.uniform(0.9, 1.1, len(frame) - int(k))
        for col in ("open", "high", "low", "close"):
            changed.iloc[tail, changed.columns.get_loc(col)] *= noise
        changed.iloc[tail, changed.columns.get_loc("volume")] *= rng.uniform(0, 3, len(frame) - int(k))
        again = as_tuples(detect_divergences(changed, p, timeframe="15m", asset_class="crypto"))
        assert [c for c in again if c[1] < k] == [c for c in base if c[1] < k]


def test_htf_trend_uses_only_closed_bars() -> None:
    """4h 이평은 신호 봉 종가 시각까지 마감된 4h 봉만 쓴다: 마지막 4h 봉이 막 마감된 경우와 아닌 경우."""
    p = params(RELAXED, {"filters": {"trend": {"enabled": True, "mode": "htf_ma", "htf": "4h",
                                                "ma_type": "ema", "ma_period": 3, "condition": "close_above"}}})
    frame = random_frame(96 * 10, 8)
    det = DivergenceDetector(p, timeframe="15m", asset_class="crypto")
    for i, (ts, row) in enumerate(frame.iterrows()):
        det.update(ts, *row.tolist())
        htf = resample_ohlcv(frame.iloc[: i + 1], "4h", source_timeframe="15m", as_of=ts + BAR)
        expected = ema(htf["close"].to_numpy(), 3)
        now = expected[-1] if len(expected) else np.nan
        assert (det._ma_now is None and np.isnan(now)) or det._ma_now == now


# ---------------------------------------------------------------------------
# 손으로 만든 예
# ---------------------------------------------------------------------------


def frame_from(lows: list[float], highs: list[float] | None = None, closes: list[float] | None = None,
               volume: list[float] | None = None) -> pd.DataFrame:
    low = np.asarray(lows, dtype=float)
    high = np.asarray(highs, dtype=float) if highs is not None else low + 1.0
    close = np.asarray(closes, dtype=float) if closes is not None else (low + high) / 2
    vol = np.asarray(volume, dtype=float) if volume is not None else np.ones(len(low))
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": vol},
                        index=crypto_index(len(low)))


def textbook() -> tuple[pd.DataFrame, dict[str, int]]:
    """급락 → t1 → 반등 p2 → 완만한 하락으로 더 낮은 t3 → 반등. RSI(t1) 매우 낮고 RSI(t3) 는 더 높다."""
    closes: list[float] = []
    closes += [100 + 0.2 * (i % 2) for i in range(30)]           # 0-29 횡보 (RSI 워밍업)
    closes += [100 - 1.0 * (i + 1) for i in range(12)]            # 30-41 급락 → t1 = 41 (88)
    closes += [88 + 1.2 * (i + 1) for i in range(8)]              # 42-49 반등 → p2 = 49 (97.6)
    closes += [97.6 - 1.0 * (i + 1) + (0.6 if i % 2 else 0) for i in range(10)]  # 50-59 완만한 하락
    closes += [87.0]                                               # 60 = t3 (87.0 < 88)
    closes += [88 + i for i in range(10)]                          # 61-70 반등
    close = np.asarray(closes)
    frame = frame_from((close - 0.1).tolist(), (close + 0.1).tolist(), close.tolist())
    return frame, {"t1": 41, "p2": 49, "t3": 60}


def test_textbook_divergence() -> None:
    frame, at = textbook()
    p = params()
    rsi = rsi_wilder(frame["close"].to_numpy(), 14)
    assert rsi[at["t1"]] < 30 and rsi[at["t3"]] - rsi[at["t1"]] >= 2
    candidates = detect_divergences(frame, p, timeframe="15m", asset_class="crypto")
    signals = [c for c in candidates if c.accepted]
    assert len(signals) == 1
    s = signals[0]
    assert (s.t1, s.p2, s.t3, s.signal_index) == (at["t1"], at["p2"], at["t3"], at["t3"] + 3)
    assert s.signal_time == frame.index[at["t3"] + 3] + BAR
    assert s.price_t3 == pytest.approx(86.9) and s.price_t1 == pytest.approx(87.9)
    table = candidates_frame(frame, candidates)
    assert table.loc[table["accepted"], "t3_time"].iloc[0] == frame.index[at["t3"]]


def test_textbook_signal_disappears_when_rsi_or_price_condition_breaks() -> None:
    frame, at = textbook()
    p = params()
    # t1 이 t3 보다 낮으면 가격 조건 실패 (t1 은 여전히 피벗 저점)
    lower_t1 = frame.copy()
    lower_t1.iloc[at["t1"], lower_t1.columns.get_loc("low")] = 86.0
    c = next(c for c in detect_divergences(lower_t1, p, timeframe="15m", asset_class="crypto")
             if c.t3 == at["t3"])
    assert c.t1 == at["t1"] and "price_not_lower_low" in c.failed
    # RSI(t1) 과매도 기준을 낮추면 필터 실패
    strict = params({"filters": {"rsi_t1_oversold": {"threshold": 1.0}}})
    c = next(c for c in detect_divergences(frame, strict, timeframe="15m", asset_class="crypto")
             if c.t3 == at["t3"])
    assert c.failed == ("rsi_t1_oversold",)


def test_no_lower_low_between() -> None:
    frame, at = textbook()
    dipped = frame.copy()
    # t1~t3 사이에 t3(86.9)보다 낮은 같은 저가 두 봉: strict 피벗이 아니므로 t1 은 그대로 41
    dipped.iloc[[52, 53], dipped.columns.get_loc("low")] = 86.0
    c = next(c for c in detect_divergences(dipped, params(), timeframe="15m", asset_class="crypto")
             if c.t3 == at["t3"])
    assert c.t1 == at["t1"] and c.failed == ("no_lower_low_between",)
    off = params({"filters": {"no_lower_low_between": {"enabled": False}}})
    c = next(c for c in detect_divergences(dipped, off, timeframe="15m", asset_class="crypto")
             if c.t3 == at["t3"])
    assert "no_lower_low_between" not in c.failed


def test_scan_window_falls_back_to_older_t1() -> None:
    """가장 가까운 피벗 저점이 실패하면 더 오래된 피벗 저점을 t1 으로 쓴다."""
    frame = random_frame(3000, 1)
    scan = params(VARIANTS[2])
    nearest = params(RELAXED)
    pivot_lows = [q.index for q in detect_pivots(frame, scan.pivot) if q.kind == "low"]
    signals = [c for c in detect_divergences(frame, scan, timeframe="15m", asset_class="crypto") if c.accepted]
    fallback = [c for c in signals if c.t1 != max(i for i in pivot_lows if i < c.t3)]
    assert fallback, "더 오래된 t1 을 고른 신호가 없음"
    nearest_by_t3 = {c.t3: c for c in detect_divergences(frame, nearest, timeframe="15m", asset_class="crypto")}
    for c in fallback:
        assert not nearest_by_t3[c.t3].accepted  # 가장 가까운 후보로는 실패하는 경우에만 거슬러 올라간다
        assert scan.filters.gap_bars.min_bars <= c.t3 - c.t1 <= scan.filters.gap_bars.max_bars


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="bars_before"):
        DivergenceDetector(params({"filters": {"volume": {"enabled": True, "bars_before": 0, "bars_after": 0}}}),
                           timeframe="15m", asset_class="crypto")
    with pytest.raises(ValueError, match="krx"):
        DivergenceDetector(params(), timeframe="15m", asset_class="stock_kr")
    det = DivergenceDetector(params(), timeframe="15m", asset_class="crypto")
    det.update(pd.Timestamp("2025-01-01 00:15", tz="UTC"), 1, 1, 1, 1, 1)
    with pytest.raises(ValueError, match="증가"):
        det.update(pd.Timestamp("2025-01-01 00:00", tz="UTC"), 1, 1, 1, 1, 1)


# ---------------------------------------------------------------------------
# 증분 이평·리샘플
# ---------------------------------------------------------------------------


def stream(state: SmaState | EmaState, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.array([np.nan if (v := state.update(float(x))) is None else v for x in values])


@pytest.mark.parametrize("period", [2, 20, 200])
def test_incremental_moving_averages_are_bit_identical(period: int) -> None:
    values = random_frame(1500, 9)["close"].to_numpy()
    assert np.array_equal(stream(SmaState(period), values), sma(values, period), equal_nan=True)
    assert np.array_equal(stream(EmaState(period), values), ema(values, period), equal_nan=True)


@pytest.mark.parametrize("index_kind", ["crypto", "krx"])
def test_bar_aggregator_matches_batch_resample(index_kind: str) -> None:
    index = crypto_index(96 * 5 + 7) if index_kind == "crypto" else krx_index(6)
    frame = random_frame(len(index), 10, index=index)
    frame = frame.drop(frame.index[[5, 6, 40]])  # 결측 봉 포함
    agg = BarAggregator("4h", source_timeframe="15m")
    done = [bar for ts, row in frame.iterrows() for bar in agg.update(ts, *row.tolist())]
    batch = resample_ohlcv(frame, "4h", source_timeframe="15m", as_of=frame.index[-1] + pd.Timedelta("1D"))
    streamed = pd.DataFrame([[b.open, b.high, b.low, b.close, b.volume] for b in done],
                            columns=["open", "high", "low", "close", "volume"],
                            index=pd.DatetimeIndex([b.start for b in done]).as_unit("ns").rename("time"))
    # 마지막 버킷은 다음 봉이 오지 않아 아직 완성되지 않았을 수 있다
    batch = batch.iloc[: len(streamed)]
    pd.testing.assert_frame_equal(streamed[["open", "high", "low", "close"]],
                                  batch[["open", "high", "low", "close"]], check_freq=False)
    np.testing.assert_allclose(streamed["volume"], batch["volume"], rtol=1e-12)


def test_bar_aggregator_emits_on_last_bar_of_bucket() -> None:
    agg = BarAggregator("1h", source_timeframe="15m")
    base = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
    assert agg.update(base + dt.timedelta(minutes=30), 1, 2, 0.5, 1.5, 1) == []
    done = agg.update(base + dt.timedelta(minutes=45), 1.5, 3, 1, 2.5, 2)  # 01:00 에 마감
    assert len(done) == 1 and done[0].start == pd.Timestamp(base) and done[0].close == 2.5
    assert done[0].open == 1 and done[0].high == 3 and done[0].low == 0.5 and done[0].volume == 3
