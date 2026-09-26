"""백테스트 유니버스 준비 (DESIGN §9.3).

1. 후보군: 전 계약 1시간봉으로 매시 롤링 24h 코인 순위 → 기간 중 한 번이라도 순위 ≤ top_n + hourly_rank_margin 인 코인.
   코인별로 그 순위 구간(처음·마지막 시각)을 기록한다.
2. 5분봉 수집: 후보 코인의 순위 대상 계약(USDT·USDC·USD1)을 [처음 − 워밍업, 마지막 + 보유 한도] 구간만 받는다.
   구간 밖에서는 1시간 순위가 top_n + margin 밖이므로 5분 기준으로도 상위 top_n 이 될 수 없다(2단계 측정).
3. 정밀 순위(ranks.py): 후보 안에서 5분봉 거래대금으로 15분 경계마다 순위.
"""

from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from perpdiv.core.config import Settings, resolve_project_path, timeframe_minutes
from perpdiv.data.cache import ArchiveCache
from perpdiv.data.symbols import Contract, excluded_reason, parse_contract, trading_contract
from perpdiv.data.universe import coin_volume, rank_windows, ranks, rolling_volume
from perpdiv.data.vision import Dataset, VisionArchive


@dataclass(frozen=True, slots=True)
class CoinWindow:
    coin: str
    first_ranked: str  # ISO UTC — 1시간 순위 ≤ top_n + margin 인 첫 시각
    last_ranked: str
    rank_symbols: list[str]  # 거래대금 합산 계약
    trading_symbol: str | None  # 주문 계약 (USDT 무기한). 없으면 순위에만 쓰임

    def data_range(self, warmup: dt.timedelta, hold: dt.timedelta, start: dt.datetime, end: dt.datetime
                   ) -> tuple[dt.datetime, dt.datetime]:
        """이 코인의 5분봉이 필요한 구간 [시작, 끝)."""
        lo = pd.Timestamp(self.first_ranked).to_pydatetime() - warmup
        hi = min(pd.Timestamp(self.last_ranked).to_pydatetime() + hold, end)
        return max(lo, start - warmup), hi


@dataclass(slots=True)
class CandidateSet:
    created_at: str
    start: str
    end: str
    top_n: int
    margin: int
    windows: dict[str, CoinWindow]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CandidateSet:
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["windows"] = {k: CoinWindow(**v) for k, v in raw["windows"].items()}
        return cls(**raw)

    @property
    def bounds(self) -> tuple[dt.datetime, dt.datetime]:
        return pd.Timestamp(self.start).to_pydatetime(), pd.Timestamp(self.end).to_pydatetime()


def universe_dir(settings: Settings) -> Path:
    return resolve_project_path(settings.base.paths.cache_dir) / "universe"


def holding_limit(settings: Settings) -> dt.timedelta:
    """보유 가능한 최대 기간 = 시간 청산 봉 수 × 가장 긴 신호 TF (+ 여유 하루)."""
    longest = max(timeframe_minutes(tf) for tf in settings.data.timeframes.signal)
    return dt.timedelta(minutes=settings.strategy.exit.time_exit.bars * longest) + dt.timedelta(days=1)


def _parallel_map(fn: Callable[[Contract], dict[str, int]], items: Sequence[Contract], workers: int
                  ) -> dict[Contract, dict[str, int]]:
    with cf.ThreadPoolExecutor(workers) as pool:
        return dict(zip(items, pool.map(fn, items), strict=True))


def rank_contracts_all(settings: Settings, archive: VisionArchive) -> list[Contract]:
    uni = settings.universe
    contracts = [c for c in (parse_contract(s) for s in archive.list_symbols()) if c is not None]
    return [c for c in contracts if c.quote in uni.ranking.rank_quote_assets and excluded_reason(c.base, uni) is None]


def build_candidates(settings: Settings, archive: VisionArchive, cache: ArchiveCache, *, now: dt.datetime,
                     workers: int = 32, log: Callable[[str], None] = print) -> CandidateSet:
    uni = settings.universe
    start, end = settings.data.backtest_period.bounds(now)
    contracts = rank_contracts_all(settings, archive)
    months = _parallel_map(lambda c: archive.list_monthly(Dataset("klines", c.symbol, "1h")), contracts, workers)
    first_month, last_month = f"{start - dt.timedelta(days=1):%Y-%m}", f"{end - dt.timedelta(seconds=1):%Y-%m}"
    active = [c for c in contracts if any(first_month <= m <= last_month for m in months[c])]
    log(f"순위 대상 계약 {len(contracts)}개 중 기간 내 1시간봉이 있는 {len(active)}개")
    datasets = {c: Dataset("klines", c.symbol, "1h") for c in active}
    hourly = cache.get_many(list(datasets.values()), start - dt.timedelta(days=1), end,
                            {datasets[c]: set(months[c]) for c in active})
    volume = coin_volume({c: hourly[datasets[c]] for c in active}, uni.ranking.usd_per_quote, "1h")
    table = ranks(rolling_volume(volume, "1h", uni.ranking.window_hours))
    table = table[(table.index >= pd.Timestamp(start)) & (table.index < pd.Timestamp(end))]
    margin = uni.candidates.hourly_rank_margin
    windows = rank_windows(table, uni.top_n + margin)
    by_coin: dict[str, list[Contract]] = {}
    for c in active:
        by_coin.setdefault(c.base, []).append(c)
    out = {}
    for coin, (lo, hi) in sorted(windows.items()):
        members = by_coin[coin]
        trade = trading_contract(members, uni)
        out[coin] = CoinWindow(coin, lo.isoformat(), hi.isoformat(), sorted(c.symbol for c in members),
                               trade.symbol if trade else None)
    log(f"후보 코인 {len(out)}개 (여유 {margin})")
    return CandidateSet(now.isoformat(), start.isoformat(), end.isoformat(), uni.top_n, margin, out)


def fetch_candidate_klines(settings: Settings, archive: VisionArchive, cache: ArchiveCache, candidates: CandidateSet,
                           *, interval: str, workers: int = 32, log: Callable[[str], None] = print) -> dict[str, float]:
    """후보 코인 계약의 ``interval`` 봉을 코인별 필요 구간만 캐시에 채운다 (프레임은 메모리에 쌓지 않음)."""
    start, end = candidates.bounds
    warmup, hold = dt.timedelta(days=settings.data.warmup_days), holding_limit(settings)
    ranges: dict[str, tuple[dt.datetime, dt.datetime]] = {}
    for w in candidates.windows.values():
        lo, hi = w.data_range(warmup, hold, start, end)
        for symbol in w.rank_symbols:
            ranges[symbol] = lo, hi
    contracts = [c for c in (parse_contract(s) for s in ranges) if c is not None]
    listed = _parallel_map(lambda c: archive.list_monthly(Dataset("klines", c.symbol, interval)), contracts, workers)
    jobs: list[tuple[Dataset, dt.datetime, dt.datetime]] = []
    for c in contracts:
        lo, hi = ranges[c.symbol]
        jobs.append((Dataset("klines", c.symbol, interval), lo, hi))
    available = {Dataset("klines", c.symbol, interval): set(listed[c]) for c in contracts}
    t0, b0 = time.time(), archive.bytes_downloaded
    last = [0.0]

    def progress(done: int, total: int) -> None:
        if time.time() - last[0] > 30 or done == total:
            last[0] = time.time()
            mb = (archive.bytes_downloaded - b0) / 1e6
            log(f"  {interval} {done}/{total} 파티션, {mb:,.0f} MB, {mb / max(time.time() - t0, 1e-9):.2f} MB/s")

    cache.prefetch(jobs, available, progress)
    return {"seconds": time.time() - t0, "mb": (archive.bytes_downloaded - b0) / 1e6, "contracts": float(len(jobs))}
