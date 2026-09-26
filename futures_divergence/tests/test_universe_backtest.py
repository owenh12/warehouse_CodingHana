"""백테스트 유니버스: 수량 단위 추정, 정밀 순위 표(5분봉 롤링 24h, 코인 합산, 15분 경계), 순위 조회."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import ClassVar, cast

import numpy as np
import pandas as pd
import pytest

from perpdiv.backtest.instruments import Instrument, infer_qty_decimals
from perpdiv.core.config import load_settings
from perpdiv.data.base import normalize_ohlcv
from perpdiv.data.cache import ArchiveCache
from perpdiv.data.candidates import CandidateSet, CoinWindow
from perpdiv.data.ranks import RankBook, build_rank_table
from perpdiv.data.vision import Dataset, VisionArchive

UTC = dt.UTC


def test_infer_qty_decimals_and_rounding() -> None:
    header = "open_time,open,high,low,close,volume,close_time,quote_volume,count,tbv,tbqv,ignore\n"
    btc = header + "1,63014.90,63039.00,62998.80,62998.90,167.437,2,1.0,3,0,0,0\n1,1,1,1,1,48.600,2,1,1,0,0,0\n"
    doge = "1,0.07005,0.07008,0.07,0.07006,8760221,2,1.0,3,0,0,0\n"
    xrp = "1,0.9989,0.9994,0.9983,0.9989,2164546.0,2,1.0,3,0,0,0\n"
    assert infer_qty_decimals(btc.encode()) == 3
    assert infer_qty_decimals(doge.encode()) == 0
    assert infer_qty_decimals(xrp.encode()) == 1
    inst = Instrument("BTCUSDT", 0.001, 100.0, "t")
    assert inst.round_qty(0.0099) == 0.009 and inst.round_qty(0.012) == 0.012  # 부동소수 경계에서 한 단위 잃지 않음
    assert Instrument("X", 1.0, 5.0, "t").round_qty(9.98) == 9.0


class VolumeArchive:
    """코인별 일정한 5분 거래대금 (A 는 USDT 10 + USDC 5 = 15, B 12, C 1; B 는 둘째 날 12:00 부터 30 으로 급증)."""

    source_name = "stub"
    volumes: ClassVar[dict[str, float]] = {"AUSDT": 10.0, "AUSDC": 5.0, "BUSDT": 12.0, "CUSDT": 1.0}

    def month_frame(self, dataset: Dataset, month: dt.datetime, *, from_day: dt.datetime | None = None) -> pd.DataFrame:
        idx = pd.date_range("2025-03-01", "2025-03-05", freq="5min", tz="UTC", inclusive="left")
        qv = np.full(len(idx), self.volumes[dataset.symbol])
        if dataset.symbol == "BUSDT":
            qv[idx >= pd.Timestamp("2025-03-02 12:00", tz="UTC")] = 30.0
        frame = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0, "quote_volume": qv,
                              "trades": 1}, index=idx)
        return normalize_ohlcv(frame)

    def empty_frame(self, dataset: Dataset) -> pd.DataFrame:
        return normalize_ohlcv(pd.DataFrame())


def test_build_rank_table_and_lookup(tmp_path: Path) -> None:
    settings = load_settings()
    cache = ArchiveCache(tmp_path / "cache", cast(VisionArchive, VolumeArchive()),
                         clock=lambda: dt.datetime(2025, 6, 1, tzinfo=UTC))
    first, last = "2025-03-01T00:00:00+00:00", "2025-03-04T00:00:00+00:00"
    windows = {
        "A": CoinWindow("A", first, last, ["AUSDC", "AUSDT"], "AUSDT"),
        "B": CoinWindow("B", first, last, ["BUSDT"], "BUSDT"),
        "C": CoinWindow("C", first, last, ["CUSDT"], "CUSDT"),
    }
    candidates = CandidateSet("x", "2025-03-02T00:00:00+00:00", "2025-03-04T00:00:00+00:00", 10, 6, windows)
    path = build_rank_table(settings, cache, candidates, path=tmp_path / "ranks.parquet", log=lambda _: None)
    book = RankBook.load(path)
    t = pd.Timestamp("2025-03-02 12:00", tz="UTC")
    assert book.top(t, 3) == ["A", "B", "C"]  # A 15×288 > B 12×288
    row = book.table[(book.table["time"] == t) & (book.table["coin"] == "A")]
    assert row["volume_24h"].iloc[0] == pytest.approx(15 * 288)  # USDT + USDC 합산, 직전 24시간 마감 봉
    # B 는 12:00 부터 30: 24h 합이 A 를 넘는 시점 = 30k + 12(288−k) > 15·288 → k > 48 → 12:00 + 49봉 = 16:05 → 15분 경계 16:15
    assert book.rank("B", pd.Timestamp("2025-03-02 16:00", tz="UTC")) == 2
    assert book.rank("B", pd.Timestamp("2025-03-02 16:15", tz="UTC")) == 1
    assert set(pd.DatetimeIndex(book.table["time"]).minute % 15) == {0}
    assert book.rank("C", pd.Timestamp("2025-03-02 00:07", tz="UTC")) is None  # 15분 경계가 아님
    ranks = book.ranks_for(pd.Series(["A", "Z"]), pd.Series([t, t]))
    assert ranks.iloc[0] == 1 and np.isnan(ranks.iloc[1])
