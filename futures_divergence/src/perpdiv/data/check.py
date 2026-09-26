"""데이터 확보 가능 여부 점검 (2단계).

1. 아카이브 심볼 목록 (상장폐지 포함) → 코인별 순위 대상 계약
2. 전 계약 1시간봉 → 코인 거래대금 → 매시 정각 롤링 24시간 순위
3. TradFi 판별 (주말/평일 변동폭 비율, 주식시장 시간대 집중도) → 설정의 제외 목록과 비교
4. 후보군 크기 (순위 ≤ top_n + 여유)
5. 1시간 순위 근사의 타당성: 표본 월들의 후보 코인 5분봉으로 15분 경계마다 정확한 순위를 내고,
   5분봉 기준 상위 10 코인의 직전 정시 1시간 기준 순위가 최대 몇 위였는지 확인 → 필요한 여유 폭.
   정시 순위가 없는 경우(상장 직후)는 따로 세고, 5분봉 상위 10 이 후보군 안에 모두 있는지 직접 확인한다
6. 용량 추정: 후보 계약의 5분봉·1분봉 zip 크기 (S3 목록), 전체 기간 vs 순위 구간만
7. 펀딩비·mark price 파일 확보 범위, 상장폐지 시점, REST(fapi) 접근 가능 여부
"""

from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TypeVar

import numpy as np
import pandas as pd

from perpdiv.core.config import Settings, resolve_project_path
from perpdiv.core.timeutil import utc_now
from perpdiv.data.cache import ArchiveCache
from perpdiv.data.quality import check_ohlcv
from perpdiv.data.resample import resample_ohlcv
from perpdiv.data.symbols import Contract, excluded_reason, parse_contract, trading_contract
from perpdiv.data.universe import (
    ASIA_SESSION_HOURS,
    US_SESSION_HOURS,
    coin_volume,
    ever_ranked,
    rank_windows,
    ranks,
    rolling_volume,
    session_share,
    weekend_ratio,
)
from perpdiv.data.vision import Dataset, RetryPolicy, VisionArchive

R = TypeVar("R")
TRADFI_RATIO = 0.45  # 주말/평일 일간 변동폭 비율이 이보다 작으면 TradFi
TRADFI_SESSION_SHARE = 0.45  # 평일 변동의 이 비율 이상이 미국장 또는 아시아장 8시간에 몰리면 TradFi (균등 = 0.33)
TRADFI_MIN_DAYS = 20


def build_archive(settings: Settings, clock: Callable[[], dt.datetime] = utc_now) -> VisionArchive:
    cfg = settings.data.providers.binance
    return VisionArchive(base_url=cfg.vision_base_url, listing_url=cfg.vision_listing_url,
                         retry=RetryPolicy(cfg.max_retries, cfg.retry_backoff_sec),
                         verify_checksum=cfg.verify_checksum, timeout_sec=cfg.timeout_sec, clock=clock)


def build_cache(settings: Settings, archive: VisionArchive, workers: int | None = None) -> ArchiveCache:
    return ArchiveCache(resolve_project_path(settings.base.paths.cache_dir) / "vision_um", archive,
                        compression=settings.data.cache.compression,
                        refresh_recent_days=settings.data.cache.refresh_recent_days,
                        workers=workers or settings.data.providers.binance.download_workers)


@dataclass(slots=True)
class DataCheck:
    generated_at: dt.datetime
    period: tuple[dt.datetime, dt.datetime]
    n_symbols: int
    n_perp_stable: int
    n_rank_contracts: int
    quote_counts: dict[str, int]
    tradfi_detected: pd.DataFrame  # 코인별 비율·시간대 비중·일수 (판정된 것만)
    tradfi_missing_in_config: list[str]  # 판정됐지만 설정 제외 목록·검토 목록에 없음
    tradfi_not_detected: list[str]  # 설정 제외 목록에 있지만 이번 판정에서 빠짐
    candidates_by_margin: dict[int, int]
    margin_needed: int | None
    sample_stats: pd.DataFrame  # 표본 월별 통계
    candidates: list[str]
    uncovered_top: list[str]  # 표본 월 5분봉 상위 10 중 후보군 밖 코인 (비어 있어야 함)
    sample_download_mb: float
    sample_download_sec: float
    size_5m_full: float
    size_1m_full: float
    size_5m_windowed: float
    size_1m_windowed: float
    months_5m_windowed: int
    download_mb: float
    download_sec: float
    funding_coverage: dict[str, tuple[str | None, str | None]]
    mark_price_intervals: list[str]
    delisted: dict[str, str]
    settled_series: list[str]
    rest_status: str
    notes: list[str] = field(default_factory=list)


def _parallel(fn: Callable[[Contract], R], items: Sequence[Contract], workers: int) -> dict[Contract, R]:
    with cf.ThreadPoolExecutor(workers) as pool:
        return dict(zip(items, pool.map(fn, items), strict=True))


def run_data_check(settings: Settings, *, sample_months: Sequence[str], workers: int = 32,
                   log: Callable[[str], None] = print, clock: Callable[[], dt.datetime] = utc_now) -> DataCheck:
    uni = settings.universe
    now = clock()
    start, end = settings.data.backtest_period.bounds(now)
    archive = build_archive(settings, clock)
    cache = build_cache(settings, archive, workers)

    # 1. 심볼 목록
    symbols = archive.list_symbols()
    contracts = [c for c in (parse_contract(s) for s in symbols) if c is not None]
    quote_counts = {str(k): int(v) for k, v in pd.Series([c.quote for c in contracts]).value_counts().items()}
    ranked = [c for c in contracts if c.quote in uni.ranking.rank_quote_assets
              and excluded_reason(c.base, uni) != "stablecoin"]
    log(f"심볼 {len(symbols)}개, 무기한 스테이블 계약 {len(contracts)}개, 순위 대상 후보 {len(ranked)}개")

    # 2. 1시간봉 (월 목록 → 필요한 달만)
    months_1h = _parallel(lambda c: archive.list_monthly(Dataset("klines", c.symbol, "1h")), ranked, workers)
    first_month, last_month = f"{start - dt.timedelta(days=1):%Y-%m}", f"{end:%Y-%m}"
    active = [c for c in ranked if any(first_month <= m <= last_month for m in months_1h[c])
              or (c.quote in ("USDT",) and not months_1h[c])]
    datasets = {c: Dataset("klines", c.symbol, "1h") for c in active}
    available = {datasets[c]: set(months_1h[c]) for c in active}
    log(f"기간 내 1시간봉이 있는 계약 {len(active)}개 → 다운로드")
    t0, b0 = time.time(), archive.bytes_downloaded
    last_report = [0.0]

    def progress(done: int, total: int) -> None:
        if time.time() - last_report[0] > 30:
            last_report[0] = time.time()
            log(f"  1시간봉 {done}/{total} 파일")

    hourly = cache.get_many(list(datasets.values()), start - dt.timedelta(days=1), end, available, progress)
    download_sec, download_mb = time.time() - t0, (archive.bytes_downloaded - b0) / 1e6
    frames = {c: hourly[datasets[c]] for c in active}

    # 3. TradFi 판별 (USDT 계약, 체결 있는 봉만)
    hourly_close, daily_close = {}, {}
    for c, frame in frames.items():
        if c.quote == "USDT" and not c.settled and len(frame) > 24 * TRADFI_MIN_DAYS:
            traded = frame[frame["trades"] > 0]
            hourly_close[c.base] = traded["close"]
            daily = resample_ohlcv(frame, "1d", source_timeframe="1h")
            daily_close[c.base] = daily["close"][daily["trades"] > 0]
    daily_table, hourly_table = pd.DataFrame(daily_close), pd.DataFrame(hourly_close)
    stats = pd.DataFrame({
        "ratio": weekend_ratio(daily_table),
        "us_share": session_share(hourly_table, US_SESSION_HOURS),
        "asia_share": session_share(hourly_table, ASIA_SESSION_HOURS),
        "days": daily_table.notna().sum(),
        "first": daily_table.apply(lambda col: col.first_valid_index()),
    })
    session = stats[["us_share", "asia_share"]].max(axis=1)
    flagged = (stats["days"] >= TRADFI_MIN_DAYS) & ((stats["ratio"] < TRADFI_RATIO) | (session >= TRADFI_SESSION_SHARE))
    detected = stats[flagged].sort_values("ratio")
    known = set(uni.exclude.tradfi_bases) | set(uni.exclude.reviewed_not_tradfi)
    missing = sorted(set(detected.index) - known)
    not_detected = sorted(set(uni.exclude.tradfi_bases) - set(detected.index))
    log(f"TradFi 판정 {len(detected)}개 (설정·검토 목록에 없는 것 {len(missing)}개, 설정에만 있는 것 {len(not_detected)}개)")

    # 4. 순위 (설정의 제외 목록 기준)
    usable = {c: f for c, f in frames.items() if excluded_reason(c.base, uni) is None}
    volume = coin_volume(usable, uni.ranking.usd_per_quote, "1h")
    rank_table = ranks(rolling_volume(volume, "1h", uni.ranking.window_hours))
    rank_table = rank_table[(rank_table.index >= pd.Timestamp(start)) & (rank_table.index <= pd.Timestamp(end))]
    by_margin = {m: len(ever_ranked(rank_table, uni.top_n + m)) for m in (0, 2, 5, 10)}
    log(f"순위 ≤ top_n+여유 코인 수: {by_margin}")

    # 5. 표본 월 5분봉으로 1시간 근사 검증
    sample_rows, top_sets = [], []
    t5, b5 = time.time(), archive.bytes_downloaded
    for month in sample_months:
        ms = pd.Timestamp(f"{month}-01", tz="UTC")
        me = ms + pd.offsets.MonthBegin(1)
        sample_ranks = rank_table[(rank_table.index >= ms) & (rank_table.index < me)]
        subset = ever_ranked(sample_ranks, uni.top_n + 10)
        sub_contracts = [c for c in usable if c.base in subset]
        five = cache.get_many([Dataset("klines", c.symbol, "5m") for c in sub_contracts],
                              (ms - pd.Timedelta(days=1)).to_pydatetime(), me.to_pydatetime())
        five_frames = {c: five[Dataset("klines", c.symbol, "5m")] for c in sub_contracts}
        vol5 = coin_volume(five_frames, uni.ranking.usd_per_quote, "5m")
        rank5 = ranks(rolling_volume(vol5, "5m", uni.ranking.window_hours))
        idx5 = pd.DatetimeIndex(rank5.index)
        rank5 = rank5[(idx5 >= ms) & (idx5 < me) & (idx5.minute % 15 == 0)]
        worst: list[float] = []
        unranked = 0
        for boundary, row in zip(pd.DatetimeIndex(rank5.index), rank5.to_numpy(dtype=float), strict=True):
            top = [coin for coin, r in zip(rank5.columns, row, strict=True) if r <= uni.top_n]
            top_sets.append(set(top))
            hour = boundary.floor("h")
            if hour not in sample_ranks.index or not top:
                continue
            at_hour = sample_ranks.loc[hour, top].to_numpy(dtype=float)
            unranked += int(np.isnan(at_hour).sum())
            if (~np.isnan(at_hour)).any():
                worst.append(float(np.nanmax(at_hour)))
        worst_s = pd.Series(worst)
        sample_rows.append({"month": month, "boundaries": len(worst_s), "max": worst_s.max(),
                            "p99": worst_s.quantile(0.99), "within_top_n": (worst_s <= uni.top_n).mean(),
                            "unranked": unranked})
        log(f"표본 {month}: 5분봉 상위 {uni.top_n} 의 1시간 순위 최대 {worst_s.max():.0f}, 정시 순위 없음 {unranked}건")
    sample_download_sec, sample_download_mb = time.time() - t5, (archive.bytes_downloaded - b5) / 1e6
    sample_stats = pd.DataFrame(sample_rows).set_index("month")
    margin_needed = int(sample_stats["max"].max() - uni.top_n) if len(sample_stats) else None

    # 6. 후보군과 용량
    margin = max(margin_needed or 0, uni.candidates.hourly_rank_margin)
    windows = rank_windows(rank_table, uni.top_n + margin)
    cands = sorted(windows)
    uncovered = sorted(set().union(*top_sets) - set(cands)) if top_sets else []
    cand_contracts = [c for c in usable if c.base in cands]
    sizes5 = _parallel(lambda c: archive.list_monthly(Dataset("klines", c.symbol, "5m")), cand_contracts, workers)
    sizes1 = _parallel(lambda c: archive.list_monthly(Dataset("klines", c.symbol, "1m")), cand_contracts, workers)
    warm = dt.timedelta(days=settings.data.warmup_days)
    hold = dt.timedelta(days=60)  # 1d 신호의 시간 청산 60일
    full_lo, full_hi = f"{start - warm:%Y-%m}", last_month
    s5 = s1 = w5 = w1 = 0.0
    months_windowed = 0
    for c in cand_contracts:
        lo_t, hi_t = windows[c.base]
        w_lo, w_hi = f"{lo_t - warm:%Y-%m}", f"{min(hi_t + hold, pd.Timestamp(end)):%Y-%m}"
        for month, size in sizes5[c].items():
            if full_lo <= month <= full_hi:
                s5 += size
            if w_lo <= month <= w_hi:
                w5 += size
                months_windowed += 1
        for month, size in sizes1[c].items():
            if full_lo <= month <= full_hi:
                s1 += size
            if w_lo <= month <= w_hi:
                w1 += size

    # 7. 펀딩비·mark price·상장폐지·REST
    trade_contracts = {}
    for coin in cands:
        tc = trading_contract([c for c in usable if c.base == coin], uni)
        if tc is not None:
            trade_contracts[coin] = tc
    funding_months = _parallel(lambda c: archive.list_monthly(Dataset("fundingRate", c.symbol)),
                               list(trade_contracts.values()), workers)
    funding: dict[str, tuple[str | None, str | None]] = {}
    for coin, c in trade_contracts.items():
        months = sorted(funding_months[c])
        funding[coin] = (months[0], months[-1]) if months else (None, None)
    btc = next((c for c in trade_contracts.values() if c.base == "BTC"), None)
    mark_intervals = []
    if btc is not None:
        for interval in ("1m", "5m", "15m", "1h"):
            if archive.list_monthly(Dataset("markPriceKlines", btc.symbol, interval)):
                mark_intervals.append(interval)
    delisted = {}
    for coin, c in trade_contracts.items():
        found = frames.get(c)
        if found is not None and len(found):
            frame = found
            rep = check_ohlcv(frame, "1h", frame.index[0].to_pydatetime(),
                              (frame.index[-1] + pd.Timedelta(hours=1)).to_pydatetime(), halt_min_bars=1)
            if rep.delisted_at is not None and rep.delisted_at < pd.Timestamp(end) - pd.Timedelta(days=1):
                delisted[coin] = f"{rep.delisted_at:%Y-%m-%d %H:%M}"
    settled = sorted(c.symbol for c in usable if c.settled and c.base in cands)
    rest_status = _rest_status(settings)
    return DataCheck(
        generated_at=now, period=(start, end), n_symbols=len(symbols), n_perp_stable=len(contracts),
        n_rank_contracts=len(usable), quote_counts=quote_counts, tradfi_detected=detected,
        tradfi_missing_in_config=missing, tradfi_not_detected=not_detected, candidates_by_margin=by_margin,
        margin_needed=margin_needed, sample_stats=sample_stats, candidates=cands, uncovered_top=uncovered,
        sample_download_mb=sample_download_mb, sample_download_sec=sample_download_sec,
        size_5m_full=s5, size_1m_full=s1, size_5m_windowed=w5, size_1m_windowed=w1,
        months_5m_windowed=months_windowed, download_mb=download_mb, download_sec=download_sec,
        funding_coverage=funding, mark_price_intervals=mark_intervals, delisted=delisted,
        settled_series=settled, rest_status=rest_status,
        notes=[f"후보군 여유 폭 = max(측정값, 설정 hourly_rank_margin {uni.candidates.hourly_rank_margin}) = {margin}"],
    )


def _rest_status(settings: Settings) -> str:
    try:
        from perpdiv.data.rest import RestProvider, create_exchange

        cfg = settings.data.providers.binance
        provider = RestProvider(create_exchange(enable_rate_limit=True, timeout_sec=cfg.timeout_sec),
                                retry=RetryPolicy(0, 0.0))
        info = provider.exchange_info()
        return f"접근 가능 (심볼 {len(info)}개)"
    except Exception as exc:
        return f"접근 불가: {type(exc).__name__}: {str(exc)[:160]}"


def render(check: DataCheck, settings: Settings) -> str:
    uni = settings.universe
    gb = 1e9
    lines = [
        "# 데이터 확보 점검 (바이낸스 USDⓈ-M)",
        "",
        f"- 생성: {check.generated_at:%Y-%m-%d %H:%M} UTC",
        f"- 기간: {check.period[0]:%Y-%m-%d} ~ {check.period[1]:%Y-%m-%d %H:%M} UTC",
        f"- 아카이브 심볼 {check.n_symbols}개 (무기한 스테이블 계약 {check.n_perp_stable}개: "
        + ", ".join(f"{q} {n}" for q, n in sorted(check.quote_counts.items())) + ")",
        f"- 순위 대상 계약 {check.n_rank_contracts}개 (스테이블·TradFi 제외 후)",
        f"- REST(fapi.binance.com): {check.rest_status}",
        "",
        f"## TradFi 판정 (주말/평일 일간 변동폭 비율 < {TRADFI_RATIO} 또는 평일 변동의 미국장·아시아장 8시간 비중 "
        f"≥ {TRADFI_SESSION_SHARE}, 체결일 {TRADFI_MIN_DAYS}일 이상)",
        "",
        f"판정 {len(check.tradfi_detected)}개. 설정(`tradfi_bases`·`reviewed_not_tradfi`)에 없는 것 "
        f"{len(check.tradfi_missing_in_config)}개: {', '.join(check.tradfi_missing_in_config) or '없음'}",
        "",
        f"설정 `tradfi_bases` 에 있지만 이번에 판정되지 않은 것 {len(check.tradfi_not_detected)}개: "
        f"{', '.join(check.tradfi_not_detected) or '없음'}",
        "",
        "| 코인 | 주말/평일 | 미국장 비중 | 아시아장 비중 | 체결일 | 첫 날 |",
        "|---|---:|---:|---:|---:|---|",
        *[f"| {coin} | {r.ratio:.2f} | {r.us_share:.2f} | {r.asia_share:.2f} | {int(r.days)} | {r.first:%Y-%m-%d} |"
          for coin, r in check.tradfi_detected.iterrows()],
        "",
        "## 순위와 후보군",
        "",
        "| 여유 폭 | 후보 코인 수 (기간 중 1시간 기준 순위 ≤ 10 + 여유) |",
        "|---:|---:|",
        *[f"| {m} | {n} |" for m, n in check.candidates_by_margin.items()],
        "",
        f"표본 월별: 15분 경계마다 5분봉 기준 상위 {uni.top_n} 코인의 직전 정시 1시간 기준 순위",
        "",
        "| 월 | 경계 수 | 최대 | 99% 분위 | 1시간 기준도 상위 10 | 정시 순위 없음(상장 직후) |",
        "|---|---:|---:|---:|---:|---:|",
        *[f"| {m} | {int(r.boundaries)} | {r['max']:.0f} | {r.p99:.0f} | {r.within_top_n:.1%} | {int(r.unranked)} |"
          for m, r in check.sample_stats.iterrows()],
        "",
        f"→ 필요한 여유 폭 {check.margin_needed}. 후보 코인 {len(check.candidates)}개. 표본 월 5분봉 상위 {uni.top_n} 중 "
        f"후보군 밖: {', '.join(check.uncovered_top) or '없음'}.",
        "",
        "## 용량 추정 (후보 코인의 순위 대상 계약, zip 기준)",
        "",
        "| 범위 | 5분봉 | 1분봉 |",
        "|---|---:|---:|",
        f"| 전체 기간 (워밍업 {settings.data.warmup_days}일 포함) | {check.size_5m_full / gb:.2f} GB | {check.size_1m_full / gb:.2f} GB |",
        f"| 순위 구간만 (첫 진입 − 워밍업 ~ 마지막 + 60일) | {check.size_5m_windowed / gb:.2f} GB | {check.size_1m_windowed / gb:.2f} GB |",
        "",
        f"측정 다운로드 속도 (이 환경, 체크섬 포함, 병렬): 1시간봉 {check.download_mb:,.0f} MB / "
        f"{check.download_sec:,.0f}초 = {check.download_mb / max(check.download_sec, 1e-9):.2f} MB/s, 표본 5분봉 "
        f"{check.sample_download_mb:,.0f} MB / {check.sample_download_sec:,.0f}초 = "
        f"{check.sample_download_mb / max(check.sample_download_sec, 1e-9):.2f} MB/s",
        "",
        "## 펀딩비·mark price",
        "",
        f"- mark price kline (BTCUSDT 기준 제공 간격): {', '.join(check.mark_price_intervals) or '없음'}",
        f"- 펀딩비 월 파일이 없는 후보: {', '.join(c for c, (lo, _) in check.funding_coverage.items() if lo is None) or '없음'}",
        "",
        "## 상장폐지 (데이터 끝까지 체결 0 고정가)",
        "",
        ", ".join(f"{c} ({t})" for c, t in sorted(check.delisted.items())) or "없음",
        "",
        f"`...SETTLED` 시리즈가 있는 후보: {', '.join(check.settled_series) or '없음'}",
        "",
        *[f"- {n}" for n in check.notes],
    ]
    return "\n".join(lines) + "\n"
