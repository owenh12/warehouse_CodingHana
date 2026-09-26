"""백테스트·신호 단계가 쓰는 시장 데이터 창구.

- 원천: 수집 TF(5m) 아카이브 → 월 파티션 캐시. 상위 TF 는 모두 여기서 리샘플한다 (원본 봉과 일치 검증: validate.py).
- 거래 중단 봉(체결 0·고정가가 ``halt_min_bars`` 개 이상 연속, 상장폐지 뒤 봉 포함)은 리샘플 전에 뺀다 → 결측으로 취급.
- 펀딩비: 아카이브 월 파일 (이번 달 분은 아카이브에 없음 → 실시간·페이퍼는 REST 로 보충).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from perpdiv.core.config import Settings
from perpdiv.data.cache import ArchiveCache
from perpdiv.data.quality import QualityReport, check_ohlcv, halt_mask
from perpdiv.data.resample import resample_ohlcv
from perpdiv.data.vision import Dataset


@dataclass(frozen=True, slots=True)
class SymbolBars:
    symbol: str
    timeframe: str
    bars: pd.DataFrame  # 규격 OHLCV + bars (버킷의 원천 봉 수)
    quality: QualityReport  # 원천(수집 TF) 품질
    halted_source_bars: int  # 거래 중단으로 뺀 원천 봉 수


class MarketData:
    def __init__(self, settings: Settings, cache: ArchiveCache) -> None:
        self._settings = settings
        self._cache = cache
        self._collect = settings.data.timeframes.collect
        self._halt_min = settings.data.quality.inactive_bar.halt_min_bars

    def source(self, symbol: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        """수집 TF 원천 봉 [start, end)."""
        return self._cache.get(Dataset("klines", symbol, self._collect), start, end)

    def bars(self, symbol: str, timeframe: str, start: dt.datetime, end: dt.datetime, *,
             drop_halts: bool = True) -> SymbolBars:
        """[start, end) 의 ``timeframe`` 봉. end 에 걸친 미완성 버킷은 버린다."""
        raw = self.source(symbol, start, end)
        quality = check_ohlcv(raw, self._collect, start, end, halt_min_bars=self._halt_min)
        halted = halt_mask(raw, self._halt_min) if drop_halts and not raw.empty else None
        clean = raw[~halted] if halted is not None else raw
        bars = resample_ohlcv(clean, timeframe, source_timeframe=self._collect, as_of=end)
        return SymbolBars(symbol, timeframe, bars, quality, int(halted.sum()) if halted is not None else 0)

    def funding(self, symbol: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        return self._cache.get(Dataset("fundingRate", symbol), start, end)
