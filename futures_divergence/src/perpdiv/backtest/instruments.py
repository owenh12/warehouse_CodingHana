"""과거 거래 규칙(수량 단위·최소 명목가) 대체값.

이 환경에서는 exchangeInfo 를 받을 수 없어, 아카이브 kline CSV 의 거래량 문자열 소수 자릿수로 수량 단위를 추정한다
(거래소가 거래량을 수량 정밀도 그대로 기록한다: BTCUSDT "167.437" → 0.001, DOGEUSDT "8760221" → 1).
실거래(7단계)에서는 exchangeInfo 의 LOT_SIZE·MIN_NOTIONAL 을 쓴다.
"""

from __future__ import annotations

import csv
import io
import json
import math
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from perpdiv.core.config import BacktestCfg
from perpdiv.data.vision import Dataset, VisionArchive


@dataclass(frozen=True, slots=True)
class Instrument:
    symbol: str
    qty_step: float
    min_notional: float
    source: str  # 추정 근거 (예: "klines 1d 2025-03")

    def round_qty(self, qty: float) -> float:
        """수량 단위로 내림."""
        if self.qty_step <= 0:
            return qty
        steps = math.floor(qty / self.qty_step + 1e-9)
        return round(steps * self.qty_step, 12)


def decimals_of(values: list[str]) -> int:
    return max((len(v.split(".", 1)[1]) if "." in v else 0) for v in values) if values else 0


def infer_qty_decimals(content: bytes) -> int:
    rows = list(csv.reader(io.StringIO(content.decode())))
    if rows and not rows[0][0].strip().isdigit():
        rows = rows[1:]
    return decimals_of([r[5].strip() for r in rows if len(r) > 5])


class InstrumentBook:
    """심볼별 거래 규칙. 추정 결과는 JSON 으로 캐시한다."""

    def __init__(self, cfg: BacktestCfg, archive: VisionArchive | None, cache_file: Path) -> None:
        self._cfg = cfg.instruments
        self._archive = archive
        self._file = cache_file
        self._lock = threading.Lock()
        self._known: dict[str, Instrument] = {}
        if cache_file.is_file():
            self._known = {k: Instrument(**v) for k, v in json.loads(cache_file.read_text(encoding="utf-8")).items()}

    def _min_notional(self, symbol: str) -> float:
        return self._cfg.min_notional_overrides.get(symbol, self._cfg.min_notional)

    def get(self, symbol: str, month: str) -> Instrument:
        """``month``('YYYY-MM') 의 1d kline 으로 추정 (없으면 그 이후 몇 달을 더 찾아본다)."""
        if self._cfg.qty_step == "none":
            return Instrument(symbol, 0.0, self._min_notional(symbol), "none")
        with self._lock:
            if symbol in self._known:
                return self._known[symbol]
        if self._archive is None:
            raise KeyError(f"{symbol}: 수량 단위를 모릅니다 (아카이브 없음)")
        year, mon = map(int, month.split("-"))
        content = None
        tried = []
        for k in range(6):
            y, m = year + (mon - 1 + k) // 12, (mon - 1 + k) % 12 + 1
            period = f"{y:04d}-{m:02d}"
            tried.append(period)
            content = self._archive.raw_csv(Dataset("klines", symbol, "1d"), period)
            if content:
                break
        if not content:
            raise KeyError(f"{symbol}: 1d kline 월 파일이 없습니다 ({tried})")
        inst = Instrument(symbol, 10.0 ** -infer_qty_decimals(content), self._min_notional(symbol),
                          f"klines 1d {tried[-1]}")
        with self._lock:
            self._known[symbol] = inst
            self._file.parent.mkdir(parents=True, exist_ok=True)
            self._file.write_text(json.dumps({k: asdict(v) for k, v in sorted(self._known.items())}, indent=1),
                                  encoding="utf-8")
        return inst
