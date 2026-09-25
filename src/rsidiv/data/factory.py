"""설정(Settings)으로부터 데이터 공급자를 조립한다."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from rsidiv.core.config import DEFAULT_CONFIG_DIR, Settings
from rsidiv.core.timeutil import utc_now
from rsidiv.data.binance import (
    BinanceRestProvider,
    MarketType,
    RetryPolicy,
    create_ccxt_exchange,
    is_region_blocked,
)
from rsidiv.data.binance_vision import BinanceVisionProvider
from rsidiv.data.cache import CachedDataProvider, ParquetStore

HistorySource = Literal["rest", "vision"]


def resolve_project_path(path: Path) -> Path:
    """설정의 상대 경로는 프로젝트 루트(config/ 의 상위) 기준으로 해석한다."""
    return path if path.is_absolute() else DEFAULT_CONFIG_DIR.parent / path


def binance_rest(
    settings: Settings, market_type: MarketType, clock: Callable[[], dt.datetime] = utc_now
) -> BinanceRestProvider:
    cfg = settings.data.providers.binance
    exchange = create_ccxt_exchange(
        market_type,
        enable_rate_limit=cfg.enable_rate_limit,
        timeout_sec=cfg.timeout_sec,
        spot_public_api=cfg.spot_public_api,
    )
    import ccxt

    retry = RetryPolicy(
        cfg.max_retries, cfg.retry_backoff_sec, (ccxt.NetworkError,), give_up=is_region_blocked
    )
    return BinanceRestProvider(
        market_type, exchange, retry=retry, page_limit=cfg.page_limit, clock=clock
    )


def binance_vision(
    settings: Settings, market_type: MarketType, clock: Callable[[], dt.datetime] = utc_now
) -> BinanceVisionProvider:
    cfg = settings.data.providers.binance
    retry = RetryPolicy(
        cfg.max_retries, cfg.retry_backoff_sec, BinanceVisionProvider.retryable_errors()
    )
    return BinanceVisionProvider(
        market_type,
        base_url=cfg.vision_base_url,
        retry=retry,
        verify_checksum=cfg.vision_verify_checksum,
        timeout_sec=cfg.timeout_sec,
        clock=clock,
    )


def binance_provider(
    settings: Settings,
    market_type: MarketType,
    *,
    source: HistorySource | None = None,
    cached: bool = True,
    clock: Callable[[], dt.datetime] = utc_now,
) -> BinanceRestProvider | BinanceVisionProvider | CachedDataProvider:
    """바이낸스 공급자. 기본은 설정의 history_source 에 Parquet 캐시를 씌운 것."""
    chosen = source or settings.data.providers.binance.history_source
    inner = (binance_rest if chosen == "rest" else binance_vision)(settings, market_type, clock)
    if not cached:
        return inner
    store = ParquetStore(
        resolve_project_path(settings.base.paths.cache_dir), settings.data.cache.compression
    )
    return CachedDataProvider(
        inner,
        store,
        namespace=f"binance/{market_type}",
        refresh_recent_days=settings.data.cache.refresh_recent_days,
        source=inner.source_name,
        clock=clock,
    )
