"""다이버전스 구조 추적기: 손으로 만든 시나리오(경계·재판정·조합당 1회·교체·유효성·간격·동시 성립·워밍업),
정의를 그대로 옮긴 단순 구현과의 일치(무작위 데이터), 약세 = 강세의 가격 반전, 미래참조 없음."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import pytest

from perpdiv.core.config import StrategyCfg, load_settings
from perpdiv.indicators.pivots import detect_pivots
from perpdiv.indicators.rsi import rsi_wilder
from perpdiv.signals.divergence import DivergenceDetector, Hit, StructureTracker, detect_signals

NAN = math.nan


@pytest.fixture(scope="module")
def base_cfg() -> StrategyCfg:
    return load_settings().strategy


def cfg_with(base: StrategyCfg, **updates: Any) -> StrategyCfg:
    """기본 설정에서 점 표기 경로(``a__b`` = ``a.b``)만 바꾼 전략 설정."""
    settings = load_settings()
    assert settings.strategy == base
    return settings.strategy_with({k.replace("__", "."): v for k, v in updates.items()})


Bar = tuple[float, float, float]  # (high, low, close)


def run(cfg: StrategyCfg, bars: Sequence[Bar], rsi: Sequence[float]) -> list[Hit]:
    tracker = StructureTracker(cfg)
    hits: list[Hit] = []
    for (h, lo, c), r in zip(bars, rsi, strict=True):
        hits += [x for x in tracker.update(h, lo, c, None if math.isnan(r) else r) if x is not None]
    return hits


# 기본 시나리오 (L=R=1): t1=1 (Low 10, RSI 25) → p2=3 (High 16, 4번 봉에서 확정) → 6번 봉 종가 9.5 가 min(Low[3..5])=12 이탈
BASE: list[Bar] = [(15, 13, 14), (12, 10, 11), (14, 11, 13), (16, 12, 15), (15, 13, 14), (14, 12.5, 13), (13, 9, 9.5)]
BASE_RSI = [NAN, 25, 40, 50, 50, 45, 35]


def with_bar(bars: list[Bar], i: int, bar: Bar) -> list[Bar]:
    out = list(bars)
    out[i] = bar
    return out


def with_rsi(rsi: list[float], i: int, value: float) -> list[float]:
    out = list(rsi)
    out[i] = value
    return out


def test_basic_bullish(base_cfg: StrategyCfg) -> None:
    assert run(base_cfg, BASE, BASE_RSI) == [Hit("long", 1, 3, 6, 12.0, 1)]


def test_price_and_breakout_boundaries(base_cfg: StrategyCfg) -> None:
    assert run(base_cfg, with_bar(BASE, 6, (13, 9, 10.0)), BASE_RSI) == []  # Close(t3) = Low(t1) → 불성립 (엄격)
    assert len(run(base_cfg, with_bar(BASE, 6, (13, 9, 9.99)), BASE_RSI)) == 1
    # 5번 봉 저가 9.5 (t1 아래로 꼬리만) → 이탈 기준 9.5. 종가 9.5 는 이탈 아님, 9.49 는 이탈
    dipped = with_bar(BASE, 5, (14, 9.5, 13))
    assert run(base_cfg, with_bar(dipped, 6, (13, 9, 9.5)), BASE_RSI) == []
    assert run(base_cfg, with_bar(dipped, 6, (13, 9, 9.49)), BASE_RSI) == [Hit("long", 1, 3, 6, 9.5, 1)]


def test_rsi_boundaries_and_warmup(base_cfg: StrategyCfg) -> None:
    assert run(base_cfg, BASE, with_rsi(BASE_RSI, 6, 30.0)) == []  # RSI(t3) = 30 → 불성립
    assert run(base_cfg, BASE, with_rsi(BASE_RSI, 1, 30.0)) == []  # RSI(t1) = 30 → 구조 없음
    assert len(run(base_cfg, BASE, with_rsi(BASE_RSI, 1, 29.99))) == 1
    assert run(base_cfg, BASE, with_rsi(BASE_RSI, 1, NAN)) == []  # 워밍업 중 t1 → 구조 없음
    assert run(base_cfg, BASE, with_rsi(BASE_RSI, 6, NAN)) == []


def test_breakout_bar_fails_then_later_bar_signals(base_cfg: StrategyCfg) -> None:
    bars = [*BASE, (10, 8.5, 9.2), (9.5, 8, 8.4)]
    rsi = [*with_rsi(BASE_RSI, 6, 28.0), 36, 33]  # 6번: 이탈했지만 RSI 28 → 실패
    # 7번: 이탈 기준 = min(Low[3..6]) = 9 → 종가 9.2 는 이탈 아님. 8번: 기준 8.5 → 종가 8.4 이탈
    assert run(base_cfg, bars, rsi) == [Hit("long", 1, 3, 8, 8.5, 1)]


def test_one_signal_per_pair(base_cfg: StrategyCfg) -> None:
    bars = [*BASE, (10, 8.5, 8.7), (9, 7, 7.5)]
    assert run(base_cfg, bars, [*BASE_RSI, 35, 34]) == [Hit("long", 1, 3, 6, 12.0, 1)]


def test_higher_pivot_high_replaces_p2(base_cfg: StrategyCfg) -> None:
    # 5번 봉 High 17 이 새 피벗 고점 → (t1=1, p2=3) 폐기, (1, 5) 로 교체. 6번 종가 11.8 은 가격 조건 불성립, 7번에서 신호
    bars: list[Bar] = [*BASE[:5], (17, 14, 16), (16, 11.5, 11.8), (12, 9, 9.5)]
    rsi = [NAN, 25, 40, 50, 50, 55, 40, 35]
    assert run(base_cfg, bars, rsi) == [Hit("long", 1, 5, 7, 11.5, 1)]
    # 같은 모양에서 5번이 더 낮으면(15.5) p2 는 3 그대로
    lower = with_bar(bars, 5, (15.5, 14, 15))
    assert [h.swing for h in run(base_cfg, lower, rsi)] == [3]


def test_non_pivot_bar_above_p2_blocks_pair(base_cfg: StrategyCfg) -> None:
    # 4·5번 고가가 계속 올라 피벗이 아님(5번의 오른쪽 6번이 더 높음) → max(High[1..5]) = 17 > High(p2)=16 → 6번 신호 없음
    bars: list[Bar] = [*BASE[:4], (16.5, 13, 16), (17, 12.5, 16.5), (18, 9, 9.5)]
    assert run(base_cfg, bars, BASE_RSI) == []
    # t1 봉 자체의 고가가 p2 보다 높아도 무효 ([t1, t3−1] 에 t1 포함)
    assert run(base_cfg, with_bar(BASE, 1, (17, 10, 11)), BASE_RSI) == []


def test_tie_rule_changes_p2(base_cfg: StrategyCfg) -> None:
    tied = with_bar(BASE, 4, (16, 13, 14))  # 4번 고가 = p2 고가 16
    assert run(cfg_with(base_cfg, pivot__tie_rule="first"), tied, BASE_RSI) == [Hit("long", 1, 3, 6, 12.0, 1)]
    assert run(cfg_with(base_cfg, pivot__tie_rule="strict"), tied, BASE_RSI) == []  # 3번이 피벗 아님 → p2 없음


def test_gap_limits(base_cfg: StrategyCfg) -> None:
    assert run(cfg_with(base_cfg, structure__gap_bars__max=4), BASE, BASE_RSI) == []  # 간격 5 > 4
    assert len(run(cfg_with(base_cfg, structure__gap_bars__max=5), BASE, BASE_RSI)) == 1
    assert len(run(cfg_with(base_cfg, structure__gap_bars__max=4, structure__gap_bars__max_enabled=False), BASE,
                   BASE_RSI)) == 1
    on_min = {"structure__gap_bars__min_enabled": True, "structure__gap_bars__max": 60}
    assert run(cfg_with(base_cfg, structure__gap_bars__min=6, **on_min), BASE, BASE_RSI) == []
    assert len(run(cfg_with(base_cfg, structure__gap_bars__min=5, **on_min), BASE, BASE_RSI)) == 1


def test_same_bar_multiple_structures_one_signal(base_cfg: StrategyCfg) -> None:
    bars: list[Bar] = [(15, 13, 14), (12, 10, 11), (13, 11, 12), (12, 10.5, 11), (14, 11, 13), (16, 12, 15),
                       (15, 13, 14), (13, 9, 9.5), (10, 8, 8.5)]
    rsi = [NAN, 25, 35, 28, 40, 50, 50, 35, 34]
    # t1=1 과 t1=3 두 구조가 p2=5 를 공유, 7번에서 동시 성립 → 신호 1건(가장 최근 t1=3), 두 조합 모두 발신 완료
    assert run(base_cfg, bars, rsi) == [Hit("long", 3, 5, 7, 12.0, 2)]


def test_disabled_side(base_cfg: StrategyCfg) -> None:
    assert run(cfg_with(base_cfg, bullish__enabled=False), BASE, BASE_RSI) == []


def mirror(bars: Sequence[Bar], rsi: Sequence[float], c: float = 1000.0) -> tuple[list[Bar], list[float]]:
    """가격 반전(C − p): 고가 ↔ 저가. RSI → 100 − RSI. 강세 구조가 그대로 약세 구조가 된다."""
    return [(c - lo, c - h, c - cl) for h, lo, cl in bars], [100.0 - r for r in rsi]


def test_bearish_is_mirror_of_bullish(base_cfg: StrategyCfg) -> None:
    bars, rsi = mirror(BASE, BASE_RSI)
    assert run(base_cfg, bars, rsi) == [Hit("short", 1, 3, 6, 1000.0 - 12.0, 1)]  # 이탈 기준 = max(High[3..5]) = 988


# --- 정의 그대로의 단순 구현 (O(n²)) 과 비교 -------------------------------------------------------------------


def naive_long(high: np.ndarray, low: np.ndarray, close: np.ndarray, rsi: np.ndarray, cfg: StrategyCfg
               ) -> list[tuple[int, int, int, float, int]]:
    frame = pd.DataFrame({"high": high, "low": low})
    pivots = detect_pivots(frame, cfg.pivot.left, cfg.pivot.right, cfg.pivot.tie_rule)
    lows = [p for p in pivots if p.kind == "low"]
    highs = [p for p in pivots if p.kind == "high"]
    gap = cfg.structure.gap_bars
    oversold = cfg.bullish.oversold
    fired: set[tuple[int, int]] = set()
    out = []
    for n in range(len(close)):
        passed = []
        for pl in lows:
            j = pl.index
            if pl.confirm_index > n or not rsi[j] < oversold:
                continue
            if (gap.max_enabled and n - j > gap.max) or (gap.min_enabled and n - j < gap.min):
                continue
            ks = [ph.index for ph in highs if ph.index > j and ph.confirm_index <= n]
            if not ks:
                continue
            p2 = min(ks, key=lambda k: (-high[k], k))  # 최고가, 같으면 먼저 확정된 것
            if high[p2] < high[j:n].max():
                continue
            if cfg.structure.anchor_must_be_extreme and low[j:p2 + 1].min() < low[j]:
                continue
            level = float(low[p2:n].min())
            if close[n] < level and close[n] < low[j] and rsi[n] > oversold and (j, p2) not in fired:
                passed.append((j, p2, level))
        if passed:
            fired.update((j, p2) for j, p2, _ in passed)
            j, p2, level = max(passed)
            out.append((j, p2, n, level, len(passed)))
    return out


def random_ohlc(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """두 주기의 사인파 + 랜덤워크 (다이버전스가 자주 생기도록 진동), 정수 반올림 → 동률 다수."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    wave = 20 * np.sin(2 * np.pi * t / rng.uniform(40, 90)) + 12 * np.sin(2 * np.pi * t / rng.uniform(9, 17))
    close = np.round(300 + wave + np.cumsum(rng.normal(0, 2.0, n)), 0)
    high = close + rng.integers(0, 3, n)
    low = close - rng.integers(0, 3, n)
    return high.astype(float), low.astype(float), close.astype(float)


CASES = [
    {},
    {"pivot__tie_rule": "strict"},
    {"pivot__left": 2, "pivot__right": 2},
    {"pivot__left": 3, "pivot__right": 1, "structure__gap_bars__max": 25},
    {"structure__gap_bars__max_enabled": False},
    {"structure__gap_bars__min_enabled": True, "structure__gap_bars__min": 8},
    {"bullish__oversold": 40.0},
    {"structure__anchor_must_be_extreme": True},
]


@pytest.mark.parametrize("updates", CASES)
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_matches_naive_definition(base_cfg: StrategyCfg, updates: dict[str, Any], seed: int) -> None:
    cfg = cfg_with(base_cfg, **updates)
    high, low, close = random_ohlc(1500, seed)
    rsi = rsi_wilder(close, 14)
    got = [h for h in run(cfg, list(zip(high, low, close, strict=True)), list(rsi)) if h.side == "long"]
    expected = naive_long(high, low, close, rsi, cfg)
    assert [(h.anchor, h.swing, h.trigger, h.breakout_level, h.concurrent) for h in got] == expected
    assert len(expected) >= 2  # 시험이 비어 있지 않은지


@pytest.mark.parametrize("seed", [4, 5])
def test_bearish_mirror_on_random_data(base_cfg: StrategyCfg, seed: int) -> None:
    high, low, close = random_ohlc(1500, seed)
    rsi = list(rsi_wilder(close, 14))
    bars = list(zip(high, low, close, strict=True))
    longs = [h for h in run(base_cfg, bars, rsi) if h.side == "long"]
    m_bars, m_rsi = mirror(bars, rsi)
    shorts = [h for h in run(base_cfg, m_bars, m_rsi) if h.side == "short"]
    assert [(h.anchor, h.swing, h.trigger, 1000.0 - h.breakout_level, h.concurrent) for h in shorts] == \
        [(h.anchor, h.swing, h.trigger, h.breakout_level, h.concurrent) for h in longs]
    assert len(longs) >= 3


# --- 신호 레코드·미래참조 -------------------------------------------------------------------------------------


def ohlc_frame(n: int, seed: int) -> pd.DataFrame:
    high, low, close = random_ohlc(n, seed)
    opens = np.concatenate([[close[0]], close[:-1]])
    high, low = np.maximum(high, opens), np.minimum(low, opens)
    return pd.DataFrame({"open": opens, "high": high, "low": low, "close": close},
                        index=pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC"))


def test_signal_record_fields(base_cfg: StrategyCfg) -> None:
    frame = ohlc_frame(2000, 7)
    signals = detect_signals(frame, base_cfg, symbol="TEST", timeframe="1h")
    assert {s.side for s in signals} == {"long", "short"}
    rsi = rsi_wilder(frame["close"].to_numpy(), 14)
    for s in signals:
        assert s.signal_time == s.trigger_time + pd.Timedelta(hours=1)
        assert s.trigger_time == frame.index[s.trigger_index]
        assert s.anchor_rsi == rsi[s.anchor_index] and s.trigger_rsi == rsi[s.trigger_index]
        row = frame.iloc[s.trigger_index]
        assert (s.trigger_high, s.trigger_low, s.trigger_close) == (row.high, row.low, row.close)
        if s.side == "long":
            assert s.anchor_price == frame["low"].iloc[s.anchor_index] and s.swing_price == frame["high"].iloc[s.swing_index]
            assert s.trigger_close < s.breakout_level == frame["low"].iloc[s.swing_index:s.trigger_index].min()
            assert s.trigger_close < s.anchor_price and s.anchor_rsi < 30 < s.trigger_rsi
        else:
            assert s.anchor_price == frame["high"].iloc[s.anchor_index] and s.swing_price == frame["low"].iloc[s.swing_index]
            assert s.trigger_close > s.breakout_level == frame["high"].iloc[s.swing_index:s.trigger_index].max()
            assert s.trigger_close > s.anchor_price and s.anchor_rsi > 70 > s.trigger_rsi
        assert s.anchor_index < s.swing_index < s.trigger_index and s.gap_bars <= 60
        assert s.swing_index + base_cfg.pivot.right <= s.trigger_index  # p2 는 t3 마감까지 확정
        assert not math.isnan(s.trigger_atr) and s.trigger_atr > 0


def test_no_lookahead_prefix_and_future_changes(base_cfg: StrategyCfg) -> None:
    frame = ohlc_frame(1500, 8)
    full = detect_signals(frame, base_cfg, symbol="T", timeframe="1h")
    for k in (200, 555, 900, 1499):
        assert detect_signals(frame.iloc[:k], base_cfg, symbol="T", timeframe="1h") == \
            [s for s in full if s.trigger_index < k]
        changed = frame.copy()
        rng = np.random.default_rng(k)
        shock = np.concatenate([np.zeros(k), rng.normal(0, 8, len(frame) - k)])
        for col in ("open", "high", "low", "close"):
            changed[col] = changed[col].to_numpy() + shock
        again = detect_signals(changed, base_cfg, symbol="T", timeframe="1h")
        assert [s for s in again if s.trigger_index < k] == [s for s in full if s.trigger_index < k]


def test_detector_rejects_out_of_order(base_cfg: StrategyCfg) -> None:
    detector = DivergenceDetector(base_cfg, symbol="T", timeframe="15m")
    t = pd.Timestamp("2025-01-01", tz="UTC")
    detector.update(t, 1, 2, 0.5, 1.5)
    with pytest.raises(ValueError):
        detector.update(t, 1, 2, 0.5, 1.5)


def test_anchor_must_be_extreme_option(base_cfg: StrategyCfg) -> None:
    # 3번 봉 저가 9.8 < Low(t1)=10 (t1 과 p2=4 사이에 더 낮은 저점, RSI 40 이라 자체 구조는 없음)
    # → 기본(요구사항 그대로)은 (t1=1, p2=4) 신호, 옵션을 켜면 t1 이 [t1, p2] 최저가가 아니므로 신호 없음
    deeper: list[Bar] = [(15, 13, 14), (12, 10, 11), (14, 11, 13), (13, 9.8, 12), (16, 12, 15), (15, 13, 14),
                         (14, 12.5, 13), (13, 9, 9.5)]
    rsi = [NAN, 25, 40, 40, 50, 50, 45, 35]
    assert run(base_cfg, deeper, rsi) == [Hit("long", 1, 4, 7, 12.0, 1)]
    assert run(cfg_with(base_cfg, structure__anchor_must_be_extreme=True), deeper, rsi) == []
    assert len(run(cfg_with(base_cfg, structure__anchor_must_be_extreme=True), BASE, BASE_RSI)) == 1
