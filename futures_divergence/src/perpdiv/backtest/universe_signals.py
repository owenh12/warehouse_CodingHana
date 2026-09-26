"""후보 코인 전체의 신호 생성 (코인 × 신호 TF, 주문 계약 = USDT 무기한).

- 코인마다 데이터 구간 [처음 순위 − 워밍업, 마지막 순위 + 보유 한도] 의 5분봉 → 거래 중단 봉 제거 → TF 리샘플 →
  증분 검출기. 신호는 포지션과 무관하므로 한 번 만들어 캐시하고, 순위·엔진 단계에서 다시 쓴다.
- 백테스트 구간 [start, end) 안에서 확정된 신호만 남긴다 (t3 봉 마감 시각 기준).
- 캐시 키: 전략 설정 + 데이터 설정 + 후보군 생성 시각의 해시.
"""

from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from perpdiv.core.config import Settings, timeframe_minutes
from perpdiv.data.candidates import CandidateSet, CoinWindow, holding_limit, universe_dir
from perpdiv.data.quality import halt_mask
from perpdiv.data.resample import resample_ohlcv
from perpdiv.data.vision import Dataset
from perpdiv.signals.divergence import run_detector, signals_frame


def signal_cache_key(settings: Settings, candidates: CandidateSet) -> str:
    signal_part = settings.strategy.model_dump(mode="json", include={"rsi", "atr", "pivot", "bullish", "bearish",
                                                                     "structure"})
    payload = {"strategy": signal_part, "data": settings.data.model_dump(mode="json"),
               "candidates": candidates.created_at, "version": 2}  # 신호 생성에 쓰는 설정만 (청산·진입 조건 제외)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _coin_signals(settings: Settings, window: CoinWindow, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    from perpdiv.backtest.market import Listing, read
    from perpdiv.data.check import build_archive, build_cache

    assert window.trading_symbol is not None
    archive = build_archive(settings)
    cache = build_cache(settings, archive, workers=4)
    listing = Listing(None, universe_dir(settings) / "listing.json")
    lo, hi = window.data_range(dt.timedelta(days=settings.data.warmup_days), holding_limit(settings), start, end)
    collect = settings.data.timeframes.collect
    raw = read(cache, listing, Dataset("klines", window.trading_symbol, collect), lo, hi)
    if raw.empty:
        return pd.DataFrame()
    clean = raw[~halt_mask(raw, settings.data.quality.inactive_bar.halt_min_bars)]
    rows = []
    for tf in settings.data.timeframes.signal:
        bars = resample_ohlcv(clean, tf, source_timeframe=collect, as_of=hi)
        signals, _ = run_detector(bars, settings.strategy, symbol=window.trading_symbol, timeframe=tf)
        frame = signals_frame(signals)
        if frame.empty:
            continue
        frame = frame[(frame["signal_time"] >= pd.Timestamp(start)) & (frame["signal_time"] <= pd.Timestamp(end))]
        frame.insert(0, "coin", window.coin)
        frame["tf_minutes"] = timeframe_minutes(tf)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def generate_universe_signals(settings: Settings, candidates: CandidateSet, *, workers: int = 4,
                              log: Callable[[str], None] = print, refresh: bool = False) -> pd.DataFrame:
    key = signal_cache_key(settings, candidates)
    path = universe_dir(settings) / f"signals_{key}.parquet"
    if path.is_file() and not refresh:
        log(f"신호 캐시 사용: {path.name}")
        return pd.read_parquet(path)
    start, end = candidates.bounds
    windows = [w for w in candidates.windows.values() if w.trading_symbol is not None]
    prime_listing(settings, [Dataset("klines", w.trading_symbol, settings.data.timeframes.collect)
                             for w in windows if w.trading_symbol is not None])
    parts: list[pd.DataFrame] = []
    with cf.ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(_coin_signals, settings, w, start, end): w.coin for w in windows}
        for done, future in enumerate(cf.as_completed(futures), 1):
            frame = future.result()
            if not frame.empty:
                parts.append(frame)
            if done % 50 == 0 or done == len(futures):
                log(f"  신호 {done}/{len(futures)} 코인")
    out = pd.concat(parts, ignore_index=True).sort_values(["signal_time", "tf_minutes", "coin"],
                                                          ascending=[True, False, True], kind="stable")
    out = out.reset_index(drop=True)
    out.to_parquet(path, compression="zstd", index=False)
    return out


def prime_listing(settings: Settings, datasets: list[Dataset]) -> None:
    """작업 프로세스들이 읽기 전용으로 쓸 월 목록을 미리 채운다."""
    from perpdiv.backtest.market import Listing
    from perpdiv.data.check import build_archive

    Listing(build_archive(settings), universe_dir(settings) / "listing.json").prime(datasets)


def attach_ranks(signals: pd.DataFrame, rank_path: Path) -> pd.DataFrame:
    """신호 시각의 코인 순위 (정밀 순위 표에 없으면 NaN = 저장 범위 밖)."""
    from perpdiv.data.ranks import RankBook

    book = RankBook.load(rank_path)
    out = signals.copy()
    out["rank"] = book.ranks_for(out["coin"], out["signal_time"])
    return out
