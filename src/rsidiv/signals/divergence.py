"""RSI 정규 강세 다이버전스 판정 (t1 → p2 → t3) 과 필터.

정의 (strategy.yaml):
    가격 조건  Low(t3) < Low(t1)
    RSI 조건   RSI(t3) > RSI(t1)
    구조 조건  t1 < p2 < t3, High(p2) > Low(t1), High(p2) > Low(t3)
Low/High 는 ``pivot.price_source`` 가 ``low`` 면 저가/고가, ``close`` 면 둘 다 종가다.

판정 시점: 피벗 저점 t3 가 확정되는 봉(n = t3 + R)의 종가 확정 직후. 이 시점에 t3 이전의 피벗은
모두 확정되어 있으므로(피벗 i 의 확정 봉 i + R < t3 + R) t1·p2 선택에 미래 데이터가 쓰이지 않는다.
필터도 n 봉까지의 데이터만 쓴다 (거래량 창의 우측 길이 ≤ R 은 설정 검증이 강제).

해석 (docs/DESIGN.md §4, §14):
- t1 ``previous_pivot``: t3 직전 피벗 저점 하나. ``scan_window``: t3 - t1 이 gap_bars 범위
  (enabled 와 무관하게 탐색 창으로 사용) 안인 피벗 저점을 가까운 것부터 보며 모든 조건·필터를
  통과하는 첫 번째. 없으면 가장 가까운 후보의 실패 사유를 기록한다.
- p2 ``highest_pivot``: t1 < i < t3 인 피벗 고점 중 최고가(같으면 앞선 것). ``max_high``: 같은 구간의
  최고가 봉.
- ``no_lower_low_between``: t1 < j < t3 인 모든 봉의 저가(가격 기준)가 Low(t3) 이상이어야 한다.
- 추세 필터의 ``close`` 는 신호 봉(n)의 종가다. ``htf_ma`` 는 n 봉 종가 시각까지 마감된 상위 봉만으로
  이평을 계산한다.

탐지기는 증분형 하나뿐이며, 백테스트(:func:`detect_divergences`)도 봉을 한 개씩 넣어 같은 코드로 판정한다.
"""

from __future__ import annotations

import bisect
import datetime as dt
import math
from dataclasses import dataclass, field

import pandas as pd

from rsidiv.core.config import AssetClassName, KrxCfg, StrategyParams, timeframe_minutes
from rsidiv.data.base import OHLCV_COLUMNS
from rsidiv.data.resample import BarAggregator
from rsidiv.indicators.ma import EmaState, SmaState, moving_average_state
from rsidiv.indicators.pivots import PivotTracker
from rsidiv.indicators.rsi import RsiState

#: 실패 사유 코드. 순서는 보고서 퍼널(첫 번째 실패 사유 기준 집계) 순서다.
REASONS: tuple[str, ...] = (
    "no_t1",
    "rsi_warmup",
    "no_p2",
    "price_not_lower_low",
    "rsi_not_higher_low",
    "p2_not_above",
    "gap_bars",
    "rsi_t1_oversold",
    "rsi_diff_min",
    "price_drop_min",
    "no_lower_low_between",
    "trend",
    "volume",
    "session_kr",
)

REASON_LABELS: dict[str, str] = {
    "no_t1": "t1 후보 없음",
    "rsi_warmup": "RSI 워밍업 구간",
    "no_p2": "t1~t3 사이 p2 없음",
    "price_not_lower_low": "가격 저점이 낮아지지 않음",
    "rsi_not_higher_low": "RSI 저점이 높아지지 않음",
    "p2_not_above": "High(p2)가 두 저점보다 높지 않음",
    "gap_bars": "t1~t3 간격 범위 밖",
    "rsi_t1_oversold": "RSI(t1) ≥ 과매도 기준",
    "rsi_diff_min": "RSI 상승폭 부족",
    "price_drop_min": "가격 하락폭 부족",
    "no_lower_low_between": "t1~t3 사이에 더 낮은 저가",
    "trend": "추세 필터",
    "volume": "거래량 필터",
    "session_kr": "세션 필터(국내주식)",
}

#: 정의 자체(패턴)에 해당하는 사유. 나머지는 선택 필터다.
PATTERN_REASONS = frozenset(REASONS[:6])

_NAN = float("nan")


@dataclass(frozen=True, slots=True)
class DivergenceCandidate:
    """확정된 피벗 저점 t3 하나에 대한 판정 결과. ``failed`` 가 비어 있으면 신호다."""

    t3: int
    signal_index: int  # t3 + R. 이 봉의 종가 확정 시각이 신호 시각
    signal_time: pd.Timestamp  # 신호 봉의 종가 확정 시각 (UTC)
    t1: int | None
    p2: int | None
    price_t1: float
    price_p2: float
    price_t3: float
    rsi_t1: float
    rsi_t3: float
    failed: tuple[str, ...] = field(default=())

    @property
    def accepted(self) -> bool:
        return not self.failed


class DivergenceDetector:
    """봉을 한 개씩 넣으면, 그 봉에서 확정된 t3 후보의 판정 결과를 반환한다 (없으면 None).

    백테스트·모의투자·실거래가 모두 이 클래스를 쓴다. 입력 봉은 마감된 봉이어야 하며 시간순이어야 한다.
    """

    def __init__(
        self,
        params: StrategyParams,
        *,
        timeframe: str,
        asset_class: AssetClassName,
        krx: KrxCfg | None = None,
    ) -> None:
        self.params = params
        self._bar = pd.Timedelta(minutes=timeframe_minutes(timeframe))
        self._low_source = params.pivot.price_source == "low"
        self._rsi_state = RsiState(params.rsi.period)
        self._pivots = PivotTracker(params.pivot)

        self._times: list[pd.Timestamp] = []
        self._low: list[float] = []  # 가격 기준 저가 (low 또는 close)
        self._high: list[float] = []  # 가격 기준 고가 (high 또는 close)
        self._close: list[float] = []
        self._volume: list[float] = []
        self._rsi: list[float] = []
        self._pivot_lows: list[int] = []
        self._pivot_highs: list[int] = []

        trend = params.filters.trend
        self._trend_ma: SmaState | EmaState | None = None
        self._htf: BarAggregator | None = None
        self._ma_now: float | None = None
        self._ma_prev: float | None = None
        if trend.enabled:
            self._trend_ma = moving_average_state(trend.ma_period, trend.ma_type)
            if trend.mode == "htf_ma":
                self._htf = BarAggregator(trend.htf, source_timeframe=timeframe)

        volume = params.filters.volume
        if volume.enabled and volume.bars_before + volume.bars_after == 0:
            raise ValueError("filters.volume: bars_before + bars_after 가 0 이면 비교할 구간이 없습니다")

        session = params.filters.session_kr
        self._session_on = session.enabled and asset_class == "stock_kr"
        if self._session_on and krx is None:
            raise ValueError("국내주식 세션 필터에는 markets.yaml 의 krx 설정이 필요합니다")
        self._krx = krx

    @property
    def bars_seen(self) -> int:
        return len(self._times)

    def time_at(self, index: int) -> pd.Timestamp:
        """index 번째 봉의 시작 시각 (UTC)."""
        return self._times[index]

    def rsi_at(self, index: int) -> float:
        """index 번째 봉의 RSI (워밍업 구간은 NaN). 음수 index 는 뒤에서부터."""
        return self._rsi[index]

    # --- 입력 --------------------------------------------------------------

    def update(
        self, time: dt.datetime | pd.Timestamp, open_: float, high: float, low: float,
        close: float, volume: float,
    ) -> DivergenceCandidate | None:
        ts = pd.Timestamp(time).tz_convert("UTC")
        if self._times and ts <= self._times[-1]:
            raise ValueError(f"봉 시각이 증가하지 않습니다: {ts}")
        n = len(self._times)
        self._times.append(ts)
        self._low.append(low if self._low_source else close)
        self._high.append(high if self._low_source else close)
        self._close.append(close)
        self._volume.append(volume)
        value = self._rsi_state.update(close)
        self._rsi.append(_NAN if value is None else value)
        self._update_trend(ts, open_, high, low, close, volume)

        # 같은 봉이 피벗 고점이면서 저점일 수 있다. 고점을 먼저 등록해도 p2 는 t3 보다 앞선 봉에서만
        # 고르므로 결과에 영향이 없다. t3 는 판정 후에 등록해 자기 자신이 t1 이 되지 않게 한다.
        found = self._pivots.update(high, low, close)
        self._pivot_highs += [p.index for p in found if p.kind == "high"]
        candidate = None
        for pivot in found:
            if pivot.kind == "low":
                candidate = self._evaluate(pivot.index, n)
                self._pivot_lows.append(pivot.index)
        return candidate

    def _update_trend(
        self, ts: pd.Timestamp, open_: float, high: float, low: float, close: float, volume: float
    ) -> None:
        if self._trend_ma is None:
            return
        if self._htf is None:
            closes = [close]
        else:
            closes = [bar.close for bar in self._htf.update(ts, open_, high, low, close, volume)]
        for value in closes:
            self._ma_prev, self._ma_now = self._ma_now, self._trend_ma.update(value)

    # --- 판정 --------------------------------------------------------------

    def _evaluate(self, t3: int, n: int) -> DivergenceCandidate:
        assert n == t3 + self.params.pivot.right
        gap = self.params.filters.gap_bars
        lows = self._pivot_lows
        if self.params.divergence.t1_selection == "previous_pivot":
            options = lows[-1:]
        else:
            lo = bisect.bisect_left(lows, t3 - gap.max_bars)
            hi = bisect.bisect_right(lows, t3 - gap.min_bars)
            options = lows[lo:hi][::-1]  # 가까운 것부터
        if not options:
            return DivergenceCandidate(
                t3, n, self._signal_time(n), None, None, _NAN, _NAN, self._low[t3], _NAN,
                self._rsi[t3], failed=("no_t1",),
            )
        nearest: DivergenceCandidate | None = None
        for t1 in options:
            candidate = self._check(t1, t3, n)
            if candidate.accepted:
                return candidate
            if nearest is None:
                nearest = candidate
        assert nearest is not None
        return nearest

    def _signal_time(self, n: int) -> pd.Timestamp:
        return self._times[n] + self._bar

    def _select_p2(self, t1: int, t3: int) -> int | None:
        high = self._high
        if self.params.divergence.p2_selection == "highest_pivot":
            lo = bisect.bisect_right(self._pivot_highs, t1)
            hi = bisect.bisect_left(self._pivot_highs, t3)
            between = self._pivot_highs[lo:hi]
        else:
            between = list(range(t1 + 1, t3))
        best: int | None = None
        for i in between:
            if best is None or high[i] > high[best]:
                best = i
        return best

    def _check(self, t1: int, t3: int, n: int) -> DivergenceCandidate:
        f = self.params.filters
        low1, low3 = self._low[t1], self._low[t3]
        rsi1, rsi3 = self._rsi[t1], self._rsi[t3]
        p2 = self._select_p2(t1, t3)
        high2 = self._high[p2] if p2 is not None else _NAN
        warm = not (math.isnan(rsi1) or math.isnan(rsi3))

        failed: list[str] = []
        if not warm:
            failed.append("rsi_warmup")
        if p2 is None:
            failed.append("no_p2")
        if not low3 < low1:
            failed.append("price_not_lower_low")
        if warm and not rsi3 > rsi1:
            failed.append("rsi_not_higher_low")
        if p2 is not None and not (high2 > low1 and high2 > low3):
            failed.append("p2_not_above")

        if f.gap_bars.enabled and not f.gap_bars.min_bars <= t3 - t1 <= f.gap_bars.max_bars:
            failed.append("gap_bars")
        if warm and f.rsi_t1_oversold.enabled and not rsi1 < f.rsi_t1_oversold.threshold:
            failed.append("rsi_t1_oversold")
        if warm and f.rsi_diff_min.enabled and not rsi3 - rsi1 >= f.rsi_diff_min.min_diff:
            failed.append("rsi_diff_min")
        if f.price_drop_min.enabled and not (low1 - low3) / low1 >= f.price_drop_min.min_pct:
            failed.append("price_drop_min")
        if f.no_lower_low_between.enabled and t3 - t1 > 1 and min(self._low[t1 + 1 : t3]) < low3:
            failed.append("no_lower_low_between")
        if f.trend.enabled and not self._trend_ok(n):
            failed.append("trend")
        if f.volume.enabled and not self._volume_ok(t1, t3):
            failed.append("volume")
        if self._session_on and self._session_excluded(t3, n):
            failed.append("session_kr")

        return DivergenceCandidate(
            t3, n, self._signal_time(n), t1, p2, low1, high2, low3, rsi1, rsi3, failed=tuple(failed)
        )

    def _trend_ok(self, n: int) -> bool:
        condition = self.params.filters.trend.condition
        now, prev = self._ma_now, self._ma_prev
        if now is None:
            return False
        above = self._close[n] > now
        rising = prev is not None and now > prev
        if condition == "close_above":
            return above
        if condition == "slope_up":
            return rising
        return above and rising

    def _window_mean(self, center: int) -> float | None:
        cfg = self.params.filters.volume
        start, end = center - cfg.bars_before + 1, center + cfg.bars_after  # [start, end]
        if start < 0:
            return None
        return math.fsum(self._volume[start : end + 1]) / (end - start + 1)

    def _volume_ok(self, t1: int, t3: int) -> bool:
        cfg = self.params.filters.volume
        current = self._window_mean(t3)
        if cfg.mode == "t3_vs_t1":
            base = self._window_mean(t1)
        else:
            start = t3 - cfg.bars_before + 1
            base = (
                math.fsum(self._volume[start - cfg.lookback : start]) / cfg.lookback
                if start - cfg.lookback >= 0
                else None
            )
        if current is None or base is None:
            return False
        if base == 0.0:
            return current > 0.0
        return current / base >= cfg.min_ratio

    def _session_excluded(self, t3: int, n: int) -> bool:
        applies = self.params.filters.session_kr.applies_to
        bars = {"signal_bar": (n,), "t3_bar": (t3,), "both": (t3, n)}[applies]
        return any(self._krx_bar_excluded(self._times[i]) for i in bars)

    def _krx_bar_excluded(self, bar_open: pd.Timestamp) -> bool:
        krx, cfg = self._krx, self.params.filters.session_kr
        assert krx is not None
        local = bar_open.tz_convert(krx.timezone)
        day = local.date()
        special = next((s for s in krx.special_sessions if s.date == day), None)
        open_t, close_t = (special.open, special.close) if special else (krx.regular_open, krx.regular_close)
        tz = local.tzinfo
        session_open = pd.Timestamp(dt.datetime.combine(day, open_t), tz=tz)
        session_close = pd.Timestamp(dt.datetime.combine(day, close_t), tz=tz)
        if session_open <= local < session_open + cfg.exclude_first_bars * self._bar:
            return True
        if session_close - cfg.exclude_last_bars * self._bar <= local < session_close:
            return True
        if cfg.exclude_auction_bars:
            regular_close = dt.datetime.combine(day, krx.regular_close)
            auction_len = regular_close - dt.datetime.combine(day, krx.closing_auction.start)
            auction_start = session_close - auction_len
            if local < session_close and local + self._bar > auction_start:
                return True
        return False


def detect_divergences(
    frame: pd.DataFrame,
    params: StrategyParams,
    *,
    timeframe: str,
    asset_class: AssetClassName,
    krx: KrxCfg | None = None,
) -> list[DivergenceCandidate]:
    """프레임 전체를 증분 탐지기에 넣어 모든 t3 후보의 판정 결과를 반환한다 (신호 = ``accepted``)."""
    detector = DivergenceDetector(params, timeframe=timeframe, asset_class=asset_class, krx=krx)
    columns = [frame[c].to_numpy(dtype="float64").tolist() for c in OHLCV_COLUMNS]
    out: list[DivergenceCandidate] = []
    for i, ts in enumerate(frame.index):
        candidate = detector.update(ts, *(col[i] for col in columns))
        if candidate is not None:
            out.append(candidate)
    return out


def candidates_frame(frame: pd.DataFrame, candidates: list[DivergenceCandidate]) -> pd.DataFrame:
    """리포트·CSV 용 표. 시각은 UTC 봉 시작 시각(신호 시각만 종가 확정 시각)."""

    def at(i: int | None) -> pd.Timestamp | None:
        return frame.index[i] if i is not None else None

    return pd.DataFrame(
        {
            "signal_time": [c.signal_time for c in candidates],
            "accepted": [c.accepted for c in candidates],
            "t1_time": [at(c.t1) for c in candidates],
            "p2_time": [at(c.p2) for c in candidates],
            "t3_time": [at(c.t3) for c in candidates],
            "gap_bars": [c.t3 - c.t1 if c.t1 is not None else None for c in candidates],
            "price_t1": [c.price_t1 for c in candidates],
            "price_p2": [c.price_p2 for c in candidates],
            "price_t3": [c.price_t3 for c in candidates],
            "rsi_t1": [c.rsi_t1 for c in candidates],
            "rsi_t3": [c.rsi_t3 for c in candidates],
            "failed": [",".join(c.failed) for c in candidates],
            "t1": [c.t1 for c in candidates],
            "p2": [c.p2 for c in candidates],
            "t3": [c.t3 for c in candidates],
            "signal_index": [c.signal_index for c in candidates],
        }
    )
