"""데이터 확보 가능 여부 점검 (2단계 시작 전 보고용).

심볼·시장(현물/선물)마다 다음을 점검한다.
- 거래소가 제공하는 가장 이른 봉 시각
- 요청 기간 안에서 실제로 받은 첫·마지막 봉
- 결측 봉과 OHLC 모순
- 펀딩비 이력 (선물)
- 거래소 주문 필터를 설정의 대체값과 비교

접근 실패는 예외로 중단하지 않고 항목별 오류로 기록한다. 그래서 보고서가 항상 나온다.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from rsidiv.core.config import Settings, SymbolFilter, timeframe_minutes
from rsidiv.core.timeutil import UTC, ensure_utc
from rsidiv.data.base import DataProvider, FundingProvider
from rsidiv.data.binance import BinanceRestProvider, MarketType
from rsidiv.data.quality import QualityReport, check_ohlcv, continuous_index


@dataclass(slots=True)
class FundingCheck:
    count: int
    first: dt.datetime | None
    last: dt.datetime | None
    interval_hours: dict[float, int]  # 정산 간격(시간) → 횟수


@dataclass(slots=True)
class SymbolCheck:
    market_type: MarketType
    symbol: str
    source: str
    earliest: dt.datetime | None = None
    quality: QualityReport | None = None
    funding: FundingCheck | None = None
    exchange_filters: SymbolFilter | None = None
    fallback_filters: SymbolFilter | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def data_first(self) -> dt.datetime | None:
        return self.quality.first if self.quality else None

    @property
    def data_last(self) -> dt.datetime | None:
        return self.quality.last if self.quality else None


@dataclass(slots=True)
class CheckReport:
    timeframe: str
    period_start: dt.datetime
    period_end: dt.datetime
    generated_at: dt.datetime
    results: list[SymbolCheck]

    def usable_period(self, market_type: MarketType) -> tuple[dt.datetime, dt.datetime] | None:
        """해당 시장의 모든 심볼에 공통으로 데이터가 있는 구간 [첫 봉, 마지막 봉 종료)."""
        rows = [r for r in self.results if r.market_type == market_type]
        if not rows or any(r.data_first is None or r.data_last is None for r in rows):
            return None
        bar = dt.timedelta(minutes=timeframe_minutes(self.timeframe))
        start = max(r.data_first for r in rows if r.data_first is not None)
        end = min(r.data_last for r in rows if r.data_last is not None) + bar
        return start, end


def _check_funding(provider: FundingProvider, symbol: str, start: dt.datetime, end: dt.datetime) -> FundingCheck:
    series = provider.funding_rates(symbol, start, end)
    diffs = series.index.to_series().diff().dropna().dt.total_seconds() / 3600
    return FundingCheck(
        count=len(series),
        first=series.index[0].to_pydatetime() if len(series) else None,
        last=series.index[-1].to_pydatetime() if len(series) else None,
        interval_hours=dict(Counter(round(h, 2) for h in diffs)),
    )


def check_symbol(
    provider: DataProvider,
    market_type: MarketType,
    symbol: str,
    timeframe: str,
    start: dt.datetime,
    end: dt.datetime,
    *,
    source: str,
    filters_provider: BinanceRestProvider | None,
    fallback_filters: SymbolFilter | None,
) -> SymbolCheck:
    result = SymbolCheck(market_type, symbol, source, fallback_filters=fallback_filters)
    try:
        result.earliest = provider.earliest_available(symbol, timeframe)
    except Exception as exc:
        result.errors.append(f"earliest: {type(exc).__name__}: {exc}")
    try:
        frame = provider.fetch_ohlcv(symbol, timeframe, start, end)
        horizon = frame.index[-1] + pd.Timedelta(minutes=timeframe_minutes(timeframe)) if len(frame) else start
        result.quality = check_ohlcv(frame, continuous_index(start, horizon, timeframe))
    except Exception as exc:
        result.errors.append(f"ohlcv: {type(exc).__name__}: {exc}")
    if market_type == "usdm_futures" and isinstance(provider, FundingProvider):
        try:
            result.funding = _check_funding(provider, symbol, start, end)
        except Exception as exc:
            result.errors.append(f"funding: {type(exc).__name__}: {exc}")
    if filters_provider is not None:
        try:
            result.exchange_filters = filters_provider.symbol_filters(symbol)
        except Exception as exc:
            result.errors.append(f"filters: {type(exc).__name__}: {exc}")
    return result


def run_crypto_check(
    settings: Settings,
    providers: dict[MarketType, DataProvider],
    *,
    source: str,
    start: dt.datetime,
    end: dt.datetime,
    filters_providers: dict[MarketType, BinanceRestProvider] | None = None,
    now: dt.datetime | None = None,
) -> CheckReport:
    """설정의 코인 심볼 전체를 ``providers`` 의 각 시장에 대해 점검한다."""
    timeframe = settings.data.timeframe
    results = []
    for market_type, provider in providers.items():
        fallback = settings.markets.binance.fallback_filters.get(market_type, {})
        for symbol in settings.universe.crypto.symbols:
            results.append(
                check_symbol(
                    provider, market_type, symbol, timeframe, start, end,
                    source=source,
                    filters_provider=(filters_providers or {}).get(market_type),
                    fallback_filters=fallback.get(symbol),
                )
            )
    return CheckReport(timeframe, ensure_utc(start), ensure_utc(end), now or dt.datetime.now(UTC), results)


# ---------------------------------------------------------------------------
# 보고서 (Markdown)
# ---------------------------------------------------------------------------


def _fmt(ts: dt.datetime | None) -> str:
    return ts.astimezone(UTC).strftime("%Y-%m-%d %H:%M") if ts else "-"


def _fmt_filter(f: SymbolFilter | None) -> str:
    if f is None:
        return "-"
    return f"min_qty={f.min_qty:g}, step={f.qty_step:g}, min_notional={f.min_notional:g}, tick={f.price_tick:g}"


def render_markdown(report: CheckReport) -> str:
    lines = [
        "# 데이터 확보 가능 여부 점검 (바이낸스)",
        "",
        f"- 생성: {_fmt(report.generated_at)} UTC",
        f"- 타임프레임: {report.timeframe}",
        f"- 요청 기간: {_fmt(report.period_start)} ~ {_fmt(report.period_end)} UTC",
        "",
        "| 시장 | 심볼 | 출처 | 거래소 최초 봉 | 기간 내 첫 봉 | 기간 내 마지막 봉 | 봉 수 | 결측 봉 | 최장 결측 | OHLC 모순 | 거래량 0 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report.results:
        q = r.quality
        lines.append(
            f"| {r.market_type} | {r.symbol} | {r.source} | {_fmt(r.earliest)} | {_fmt(r.data_first)} | "
            f"{_fmt(r.data_last)} | {q.rows if q else '-'} | {q.missing_bars if q else '-'} | "
            f"{q.longest_gap_bars if q else '-'} | {q.ohlc_violations if q else '-'} | "
            f"{q.zero_volume_bars if q else '-'} |"
        )

    funding_rows = [r for r in report.results if r.funding is not None]
    if funding_rows:
        lines += ["", "## 펀딩비 이력 (USDⓈ-M 선물)", "",
                  "| 심볼 | 건수 | 첫 정산 | 마지막 정산 | 정산 간격(시간: 횟수) |", "|---|---|---|---|---|"]
        for r in funding_rows:
            assert r.funding is not None
            intervals = ", ".join(f"{h:g}h: {n}" for h, n in sorted(r.funding.interval_hours.items()))
            lines.append(f"| {r.symbol} | {r.funding.count} | {_fmt(r.funding.first)} | "
                         f"{_fmt(r.funding.last)} | {intervals or '-'} |")

    filter_rows = [r for r in report.results if r.exchange_filters is not None or r.fallback_filters is not None]
    if filter_rows:
        lines += ["", "## 주문 필터 (거래소 실측 vs markets.yaml 대체값)", "",
                  "| 시장 | 심볼 | 거래소 | 설정 대체값 | 일치 |", "|---|---|---|---|---|"]
        for r in filter_rows:
            match = "-" if r.exchange_filters is None else ("✅" if r.exchange_filters == r.fallback_filters else "❌ 갱신 필요")
            lines.append(f"| {r.market_type} | {r.symbol} | {_fmt_filter(r.exchange_filters)} | "
                         f"{_fmt_filter(r.fallback_filters)} | {match} |")

    gap_rows = [r for r in report.results if r.quality and r.quality.gaps]
    if gap_rows:
        lines += ["", "## 결측 구간 (상위 10개)", ""]
        for r in gap_rows:
            assert r.quality is not None
            top = sorted(r.quality.gaps, key=lambda g: g.bars, reverse=True)[:10]
            lines.append(f"- {r.market_type} {r.symbol}: " + "; ".join(
                f"{_fmt(g.start)}~{_fmt(g.end)} ({g.bars}봉)" for g in top))

    lines += ["", "## 확정 가능한 백테스트 기간", ""]
    for market_type in dict.fromkeys(r.market_type for r in report.results):
        period = report.usable_period(market_type)
        lines.append(f"- {market_type}: " + (f"{_fmt(period[0])} ~ {_fmt(period[1])} UTC" if period
                                             else "확정 불가 (데이터 없음 또는 오류)"))

    error_rows = [r for r in report.results if r.errors]
    if error_rows:
        lines += ["", "## 오류", ""]
        for r in error_rows:
            lines += [f"- {r.market_type} {r.symbol}: {err}" for err in r.errors]
    return "\n".join(lines) + "\n"
