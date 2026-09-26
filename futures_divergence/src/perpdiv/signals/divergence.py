"""RSI 다이버전스 구조 추적기 — 강세 t1→p2→t3, 약세 p1→t2→p3 (각 심볼×타임프레임 독립).

봉 n 마감 직후 처리 순서 (docs/DESIGN.md §4.2):

1. RSI(n)·ATR(n) 갱신
2. 이번에 확정된 피벗(인덱스 n−R) 반영
   - 강세: 새 피벗 저점 j 이고 RSI(j) < oversold → 구조 S(t1=j) 생성 (RSI 워밍업 중이면 만들지 않음)
   - 강세: 새 피벗 고점 k → t1 < k 인 구조마다 p2 가 없거나 High(k) > High(p2) 이면 p2 ← k (옛 조합 폐기, 새 조합)
3. 만료: n − t1 > gap_max 인 구조 삭제 (간격 상한이 켜진 경우)
4. 판정 (p2 가 있는 구조):
   이탈 Close(n) < min(Low[p2..n−1]) · p2 유효 High(p2) ≥ max(High[t1..n−1]) · 가격 Close(n) < Low(t1) ·
   RSI(n) > oversold · 간격 하한(켜진 경우) · (t1, p2) 조합 미발신
   (``discard_on_anchor_break``: p2 지정·교체 때 t1~p2 사이에 Low(t1) 보다 낮은 저가가 있으면 구조 폐기 — 2단계 처리)
5. 성립한 구조가 여럿이면 신호 1건(가장 최근 t1), 성립한 조합은 모두 발신 완료로 표시
6. 다음 봉을 위해 구조별 누적값 갱신: max(High[t1..n]), min(Low[p2..n])

약세는 가격·RSI 부호를 뒤집은 대칭이다. 미래 데이터는 구조적으로 쓸 수 없다 (봉을 하나씩 받는 증분 방식이 유일한 구현).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Literal

import pandas as pd

from perpdiv.core.config import StrategyCfg, timeframe_minutes
from perpdiv.indicators.atr import AtrState
from perpdiv.indicators.pivots import PivotTracker
from perpdiv.indicators.rsi import RsiState

Side = Literal["long", "short"]


@dataclass(frozen=True, slots=True)
class Hit:
    """구조 추적기 판정 결과 (인덱스·가격만)."""

    side: Side
    anchor: int  # t1 (long) / p1 (short)
    swing: int  # p2 / t2
    trigger: int  # t3 / p3
    breakout_level: float  # min(Low[p2..t3−1]) / max(High[t2..p3−1]) (원래 가격)
    concurrent: int  # 이 봉에서 동시에 성립한 구조 수


@dataclass(slots=True)
class _Structure:
    anchor: int
    swing: int | None = None
    extreme: float = math.nan  # long: max(High[anchor..n−1]) / short: min(Low[anchor..n−1])
    level: float = math.nan  # long: min(Low[swing..n−1]) / short: max(High[swing..n−1])
    fired: set[int] = field(default_factory=set)  # 이미 신호를 낸 swing 인덱스


class _SideTracker:
    """한 방향의 구조 목록. 약세는 가격을 부호 반전해 같은 코드로 처리한다 (high ↔ −low)."""

    def __init__(self, side: Side, threshold: float, gap_max: int | None, gap_min: int | None,
                 discard_on_anchor_break: bool = False) -> None:
        self.side = side
        self.discard_on_anchor_break = discard_on_anchor_break
        self.threshold = threshold  # long: oversold / short: −overbought (부호 반전 공간)
        self.gap_max, self.gap_min = gap_max, gap_min
        self.structures: list[_Structure] = []
        self.stats: Counter[str] = Counter()  # 판정 단계별 집계 (보고용)

    # on_anchor·on_swing 은 봉 n 이 목록 끝에 추가된 뒤 호출된다 → 누적값은 [index..n−1] (끝 원소 제외).
    # 피벗 인덱스는 n−R (R ≥ 1) 이므로 구간은 비지 않는다.

    def on_anchor(self, index: int, rsi: float, highs: list[float]) -> None:
        self.stats["anchor_pivots"] += 1
        if not math.isnan(rsi) and rsi < self.threshold:
            self.stats["structures"] += 1
            self.structures.append(_Structure(anchor=index, extreme=max(highs[index:-1])))

    def on_swing(self, index: int, highs: list[float], lows: list[float]) -> None:
        kept: list[_Structure] = []
        for s in self.structures:
            if s.anchor < index and (s.swing is None or highs[index] > highs[s.swing]):
                # 선택 규칙: t1~p2 사이에 Low(t1) 보다 낮은 저가가 있으면 구조 폐기. p2 는 뒤로만 바뀌므로
                # 한 번 깨진 구조는 이후 어떤 p2 로도 조건을 되찾지 못한다 → 여기서 지운다.
                if self.discard_on_anchor_break and min(lows[s.anchor:index + 1]) < lows[s.anchor]:
                    self.stats["discarded_anchor_broken"] += 1
                    continue
                self.stats["swing_set" if s.swing is None else "swing_replaced"] += 1
                s.swing = index
                s.level = min(lows[index:-1])
            kept.append(s)
        self.structures = kept

    def evaluate(self, n: int, highs: list[float], lows: list[float], close: float, rsi: float) -> Hit | None:
        if self.gap_max is not None:
            alive = [s for s in self.structures if n - s.anchor <= self.gap_max]
            self.stats["expired"] += len(self.structures) - len(alive)
            self.structures = alive
        passed: list[_Structure] = []
        for s in self.structures:
            if s.swing is None or s.swing in s.fired or not close < s.level:
                continue
            # 이탈한 봉: 나머지 조건을 차례로 확인 (보고용 집계는 첫 번째로 실패한 조건)
            self.stats["breakouts"] += 1
            if highs[s.swing] < s.extreme:
                self.stats["fail_swing_not_highest"] += 1
            elif not close < lows[s.anchor]:
                self.stats["fail_price"] += 1
            elif math.isnan(rsi) or not rsi > self.threshold:
                self.stats["fail_rsi"] += 1
            elif self.gap_min is not None and n - s.anchor < self.gap_min:
                self.stats["fail_gap_min"] += 1
            else:
                passed.append(s)
        hit = None
        if passed:
            latest = max(passed, key=lambda s: s.anchor)
            assert latest.swing is not None
            hit = Hit(self.side, latest.anchor, latest.swing, n, latest.level, len(passed))
            self.stats["passed"] += len(passed)
            self.stats["signals"] += 1
            for s in passed:
                assert s.swing is not None
                s.fired.add(s.swing)
        for s in self.structures:  # 다음 봉 판정용 누적값: [anchor..n], [swing..n]
            s.extreme = max(s.extreme, highs[n])
            if s.swing is not None:
                s.level = min(s.level, lows[n])
        return hit


class StructureTracker:
    """지표값을 받아 구조만 추적한다 (RSI 를 직접 넣어 경계 조건을 시험할 수 있게 분리).

    ``update`` 는 봉 n 마감 때 한 번 호출하고, 이 봉에서 성립한 (강세, 약세) 판정을 돌려준다.
    """

    def __init__(self, cfg: StrategyCfg) -> None:
        gap = cfg.structure.gap_bars
        gap_max = gap.max if gap.max_enabled else None
        gap_min = gap.min if gap.min_enabled else None
        self._pivots = PivotTracker(cfg.pivot.left, cfg.pivot.right, cfg.pivot.tie_rule)
        extreme = cfg.structure.discard_on_anchor_break
        self._long = (_SideTracker("long", cfg.bullish.oversold, gap_max, gap_min, extreme)
                      if cfg.bullish.enabled else None)
        self._short = (_SideTracker("short", -cfg.bearish.overbought, gap_max, gap_min, extreme)
                       if cfg.bearish.enabled else None)
        self._high: list[float] = []
        self._low: list[float] = []
        self._neg_high: list[float] = []  # 약세용 부호 반전 가격: high' = −low, low' = −high
        self._neg_low: list[float] = []
        self._rsi: list[float] = []

    @property
    def bars_seen(self) -> int:
        return len(self._high)

    def update(self, high: float, low: float, close: float, rsi: float | None) -> tuple[Hit | None, Hit | None]:
        n = len(self._high)
        r = math.nan if rsi is None else float(rsi)
        self._high.append(high)
        self._low.append(low)
        self._neg_high.append(-low)
        self._neg_low.append(-high)
        self._rsi.append(r)
        confirmed = self._pivots.update(high, low)
        # 2. 확정 피벗 반영 (생성 → 교체 순서; 같은 봉에서 확정된 저점·고점은 인덱스가 같아 서로 영향 없음)
        for p in confirmed:
            if p.kind == "low":
                if self._long is not None:
                    self._long.on_anchor(p.index, self._rsi[p.index], self._high)
                if self._short is not None:
                    self._short.on_swing(p.index, self._neg_high, self._neg_low)
            else:
                if self._short is not None:
                    self._short.on_anchor(p.index, -self._rsi[p.index], self._neg_high)
                if self._long is not None:
                    self._long.on_swing(p.index, self._high, self._low)
        # 3~6. 만료·판정·누적값 갱신
        long_hit = self._long.evaluate(n, self._high, self._low, close, r) if self._long is not None else None
        short_hit = (self._short.evaluate(n, self._neg_high, self._neg_low, -close, -r)
                     if self._short is not None else None)
        if short_hit is not None:  # 부호 반전 공간 → 원래 가격
            short_hit = replace(short_hit, breakout_level=-short_hit.breakout_level)
        return long_hit, short_hit

    def open_structures(self) -> dict[Side, int]:
        return {"long": len(self._long.structures) if self._long else 0,
                "short": len(self._short.structures) if self._short else 0}

    def stats(self) -> dict[Side, Counter[str]]:
        """방향별 판정 집계: anchor_pivots(t1·p1 후보 피벗) → structures(RSI 조건 통과) → swing_set/replaced →
        breakouts(이탈 봉 × 구조) → fail_*(첫 실패 조건) / passed → signals(봉당 1건), expired(간격 초과 삭제)."""
        return {"long": Counter(self._long.stats) if self._long else Counter(),
                "short": Counter(self._short.stats) if self._short else Counter()}


@dataclass(frozen=True, slots=True)
class Signal:
    """신호 레코드. 가격은 원래 부호. 강세: anchor=t1(Low), swing=p2(High), trigger=t3 / 약세: p1(High), t2(Low), p3."""

    symbol: str
    timeframe: str
    side: Side
    anchor_index: int
    anchor_time: pd.Timestamp
    anchor_price: float
    anchor_rsi: float
    swing_index: int
    swing_time: pd.Timestamp
    swing_price: float
    trigger_index: int
    trigger_time: pd.Timestamp  # t3(p3) 봉 시작
    signal_time: pd.Timestamp  # t3(p3) 봉 마감 = 신호 확정 = 진입 봉 시가 시각
    trigger_open: float
    trigger_high: float
    trigger_low: float
    trigger_close: float
    trigger_rsi: float
    trigger_atr: float
    breakout_level: float  # 강세: min(Low[p2..t3−1]) / 약세: max(High[t2..p3−1])
    concurrent: int

    @property
    def gap_bars(self) -> int:
        return self.trigger_index - self.anchor_index


class DivergenceDetector:
    """봉을 하나씩 받아 신호를 낸다 (백테스트·페이퍼·실거래 공용). 입력 봉은 마감된 봉이어야 한다."""

    def __init__(self, cfg: StrategyCfg, *, symbol: str, timeframe: str) -> None:
        self.symbol, self.timeframe = symbol, timeframe
        self._step = pd.Timedelta(minutes=timeframe_minutes(timeframe))
        self._tracker = StructureTracker(cfg)
        self._rsi = RsiState(cfg.rsi.period)
        self._atr = AtrState(cfg.atr.period)
        self._times: list[pd.Timestamp] = []
        self._ohlc: list[tuple[float, float, float, float]] = []
        self._rsi_values: list[float] = []
        self._atr_values: list[float] = []

    def update(self, time: pd.Timestamp, open_: float, high: float, low: float, close: float) -> list[Signal]:
        if self._times and time <= self._times[-1]:
            raise ValueError(f"봉 시각이 역행·중복했습니다: {time}")
        rsi = self._rsi.update(close)
        atr = self._atr.update(high, low, close)
        self._times.append(time)
        self._ohlc.append((open_, high, low, close))
        self._rsi_values.append(math.nan if rsi is None else rsi)
        self._atr_values.append(math.nan if atr is None else atr)
        return [self._signal(hit) for hit in self._tracker.update(high, low, close, rsi) if hit is not None]

    def stats(self) -> dict[Side, Counter[str]]:
        return self._tracker.stats()

    def _signal(self, hit: Hit) -> Signal:
        long = hit.side == "long"
        a, s, t = hit.anchor, hit.swing, hit.trigger
        o, h, lo, c = self._ohlc[t]
        return Signal(
            symbol=self.symbol, timeframe=self.timeframe, side=hit.side,
            anchor_index=a, anchor_time=self._times[a], anchor_price=self._ohlc[a][2 if long else 1],
            anchor_rsi=self._rsi_values[a],
            swing_index=s, swing_time=self._times[s], swing_price=self._ohlc[s][1 if long else 2],
            trigger_index=t, trigger_time=self._times[t], signal_time=self._times[t] + self._step,
            trigger_open=o, trigger_high=h, trigger_low=lo, trigger_close=c,
            trigger_rsi=self._rsi_values[t], trigger_atr=self._atr_values[t],
            breakout_level=hit.breakout_level, concurrent=hit.concurrent,
        )


def run_detector(frame: pd.DataFrame, cfg: StrategyCfg, *, symbol: str, timeframe: str
                 ) -> tuple[list[Signal], DivergenceDetector]:
    """프레임 전체를 봉 순서대로 흘린다 (증분 경로 그대로 → 미래참조 없음). 판정 집계를 보려면 검출기도 받는다."""
    detector = DivergenceDetector(cfg, symbol=symbol, timeframe=timeframe)
    out: list[Signal] = []
    cols = [frame[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close")]
    for time, o, h, lo, c in zip(pd.DatetimeIndex(frame.index), *cols, strict=True):
        out += detector.update(time, float(o), float(h), float(lo), float(c))
    return out, detector


def detect_signals(frame: pd.DataFrame, cfg: StrategyCfg, *, symbol: str, timeframe: str) -> list[Signal]:
    return run_detector(frame, cfg, symbol=symbol, timeframe=timeframe)[0]


def signals_frame(signals: Iterable[Signal]) -> pd.DataFrame:
    rows = [{f: getattr(s, f) for f in Signal.__slots__} | {"gap_bars": s.gap_bars} for s in signals]
    return pd.DataFrame(rows)
