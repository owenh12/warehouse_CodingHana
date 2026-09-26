"""데이터 계층: 아카이브 파싱·체크섬·월/일 대체, 캐시 확정, 리샘플(일괄·증분), 품질, 심볼, 순위, REST."""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import os
import zipfile
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from perpdiv.core.config import UniverseCfg, load_settings
from perpdiv.data.base import DataSourceError, empty_ohlcv, normalize_ohlcv, ohlcv_from_rows
from perpdiv.data.cache import ArchiveCache
from perpdiv.data.quality import check_ohlcv, halt_mask
from perpdiv.data.resample import BarAggregator, resample_ohlcv
from perpdiv.data.rest import RestProvider, parse_symbol_info
from perpdiv.data.symbols import Contract, parse_contract, rank_contracts, trading_contract
from perpdiv.data.universe import (
    coin_volume,
    ever_ranked,
    rank_at,
    rank_windows,
    ranks,
    rolling_volume,
    weekend_ratio,
)
from perpdiv.data.vision import Dataset, RetryPolicy, VisionArchive, parse_funding_csv, parse_kline_csv

UTC = dt.UTC
NOW = dt.datetime(2025, 3, 10, 12, 0, tzinfo=UTC)
KLINE_HEADER = "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume," \
               "taker_buy_quote_volume,ignore"


def no_sleep(_: float) -> None:
    pass


def synthetic(start: str, n: int, minutes: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = np.round(100 + np.cumsum(rng.normal(0, 0.3, n)), 2)
    opens = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(opens, close) + np.round(rng.uniform(0, 0.5, n), 2)
    low = np.minimum(opens, close) - np.round(rng.uniform(0, 0.5, n), 2)
    volume = np.round(rng.uniform(1, 10, n), 3)
    frame = pd.DataFrame({"open": opens, "high": high, "low": low, "close": close, "volume": volume,
                          "quote_volume": np.round(volume * close, 4), "trades": rng.integers(1, 50, n)},
                         index=pd.date_range(start, periods=n, freq=f"{minutes}min", tz="UTC"))
    out = normalize_ohlcv(frame)
    out.index = pd.DatetimeIndex(list(out.index), name="time", tz="UTC")  # freq 없음 (파싱·리샘플 결과와 같게)
    return out


Record = tuple[pd.Timestamp, float, float, float, float, float, float, int]


def records(frame: pd.DataFrame) -> list[Record]:
    """(시각, open, high, low, close, volume, quote_volume, trades) 행."""
    cols = [frame[c].to_numpy() for c in ("open", "high", "low", "close", "volume", "quote_volume")]
    return [(ts, *map(float, vals), int(tr))
            for ts, *vals, tr in zip(pd.DatetimeIndex(frame.index), *cols, frame["trades"].to_numpy(), strict=True)]


def kline_csv(frame: pd.DataFrame, *, header: bool = True, unit: str = "ms") -> bytes:
    scale = 1000 if unit == "us" else 1
    lines = [KLINE_HEADER] if header else []
    for ts, o, h, lo, c, v, qv, trades in records(frame):
        ms = ts.value // 1_000_000 * scale
        lines.append(f"{ms},{o},{h},{lo},{c},{v},{ms + 1},{qv},{trades},0,0,0")
    return ("\n".join(lines) + "\n").encode()


def zipped(name: str, content: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, content)
    return buffer.getvalue()


def checksum(payload: bytes, name: str) -> bytes:
    return f"{hashlib.sha256(payload).hexdigest()}  {name}\n".encode()


# --- base -----------------------------------------------------------------------------------------------


def test_normalize_dedup_and_conflict() -> None:
    frame = synthetic("2025-01-01", 5)
    doubled = pd.concat([frame, frame.iloc[[2]]])
    assert normalize_ohlcv(doubled).equals(frame)
    conflict = frame.iloc[[2]].copy()
    conflict["close"] += 1
    with pytest.raises(DataSourceError):
        normalize_ohlcv(pd.concat([frame, conflict]))
    naive = frame.tz_localize(None)
    assert str(pd.DatetimeIndex(normalize_ohlcv(naive).index).tz) == "UTC"


def test_ohlcv_from_rows() -> None:
    rows = [[1_735_689_600_000.0, 1, 2, 0.5, 1.5, 10, 15, 3], [1_735_689_900_000.0, 1.5, 2, 1, 1, 5, 6, 2]]
    frame = ohlcv_from_rows(rows)
    assert frame.index[0] == pd.Timestamp("2025-01-01", tz="UTC")
    assert frame["trades"].dtype == np.int64 and frame["quote_volume"].tolist() == [15.0, 6.0]


# --- vision parsing --------------------------------------------------------------------------------------


@pytest.mark.parametrize("header", [True, False])
@pytest.mark.parametrize("unit", ["ms", "us"])
def test_parse_kline_csv(header: bool, unit: str) -> None:
    frame = synthetic("2025-01-01", 12)
    parsed = parse_kline_csv(kline_csv(frame, header=header, unit=unit))
    pd.testing.assert_frame_equal(parsed, frame)


def test_parse_funding_csv() -> None:
    body = b"calc_time,funding_interval_hours,last_funding_rate\n1735689600000,8,0.0001\n1735718400000,8,-0.00005\n"
    frame = parse_funding_csv(body)
    assert frame.index[1] == pd.Timestamp("2025-01-01 08:00", tz="UTC")
    assert frame["funding_rate"].tolist() == [0.0001, -0.00005] and frame["interval_hours"].iloc[0] == 8


# --- vision HTTP (가짜 세션) ------------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code: int, content: bytes = b"") -> None:
        self.status_code, self.content = status_code, content


class FakeSession:
    """url(목록이면 url|prefix|marker) → 응답 또는 응답 목록(차례로 소비)."""

    def __init__(self, routes: dict[str, Any]) -> None:
        self.routes = routes
        self.calls: list[str] = []

    def get(self, url: str, params: dict[str, str] | None = None, timeout: float | None = None) -> FakeResponse:
        key = f"{url}|{params['prefix']}|{params['marker']}" if params else url
        self.calls.append(key)
        entry = self.routes.get(key, (404, b""))
        if isinstance(entry, list):
            entry = entry.pop(0) if len(entry) > 1 else entry[0]
        return FakeResponse(*entry)


BASE = "https://data.binance.vision"
LISTING = "https://listing.example"


def fake_archive(routes: dict[str, Any], retries: int = 2) -> tuple[VisionArchive, FakeSession]:
    archive = VisionArchive(base_url=BASE, listing_url=LISTING, retry=RetryPolicy(retries, 0.0, sleep=no_sleep),
                            clock=lambda: NOW)
    session = FakeSession(routes)
    archive._session = lambda: session  # type: ignore[assignment,method-assign,return-value]
    return archive, session


def serve(routes: dict[str, Any], path: str, csv: bytes, *, bad_checksum: bool = False) -> None:
    name = path.rsplit("/", 1)[-1]
    payload = zipped(name.replace(".zip", ".csv"), csv)
    routes[f"{BASE}/{path}"] = (200, payload)
    routes[f"{BASE}/{path}.CHECKSUM"] = (200, checksum(b"x" if bad_checksum else payload, name))


DS = Dataset("klines", "BTCUSDT", "1h")
MONTHLY = "data/futures/um/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-{}.zip"
DAILY = "data/futures/um/daily/klines/BTCUSDT/1h/BTCUSDT-1h-{}.zip"


def test_month_prefers_monthly_file() -> None:
    frame = synthetic("2025-02-01", 28 * 24, 60)
    routes: dict[str, Any] = {}
    serve(routes, MONTHLY.format("2025-02"), kline_csv(frame))
    archive, session = fake_archive(routes)
    pd.testing.assert_frame_equal(archive.month_frame(DS, dt.datetime(2025, 2, 1, tzinfo=UTC)), frame)
    assert len(session.calls) == 2 and archive.bytes_downloaded > 0  # zip + CHECKSUM


def test_month_falls_back_to_daily_until_yesterday() -> None:
    days = [synthetic(f"2025-03-0{d}", 24, 60, seed=d) for d in (1, 2, 9)]
    routes: dict[str, Any] = {}
    for d, frame in zip((1, 2, 9), days, strict=True):
        serve(routes, DAILY.format(f"2025-03-0{d}"), kline_csv(frame))
    serve(routes, DAILY.format("2025-03-10"), kline_csv(synthetic("2025-03-10", 12, 60)))  # 오늘: 요청하면 안 됨
    archive, session = fake_archive(routes)
    out = archive.month_frame(DS, dt.datetime(2025, 3, 1, tzinfo=UTC))
    pd.testing.assert_frame_equal(out, pd.concat(days))
    requested = [c for c in session.calls if not c.endswith(".CHECKSUM")]
    assert len(requested) == 9 and not any("2025-03-10" in c for c in requested)  # 이번 달: 월 파일 요청 없음


def test_checksum_mismatch_and_missing() -> None:
    routes: dict[str, Any] = {}
    serve(routes, MONTHLY.format("2025-01"), kline_csv(synthetic("2025-01-01", 5, 60)), bad_checksum=True)
    archive, _ = fake_archive(routes)
    with pytest.raises(DataSourceError, match="SHA-256"):
        archive.month_frame(DS, dt.datetime(2025, 1, 1, tzinfo=UTC))
    del routes[f"{BASE}/{MONTHLY.format('2025-01')}.CHECKSUM"]
    with pytest.raises(DataSourceError, match="CHECKSUM"):
        archive.month_frame(DS, dt.datetime(2025, 1, 1, tzinfo=UTC))


def test_retry_on_5xx_then_give_up() -> None:
    routes: dict[str, Any] = {}
    frame = synthetic("2025-01-01", 5, 60)
    serve(routes, MONTHLY.format("2025-01"), kline_csv(frame))
    url = f"{BASE}/{MONTHLY.format('2025-01')}"
    routes[url] = [(503, b""), (429, b""), routes[url]]
    archive, session = fake_archive(routes, retries=2)
    pd.testing.assert_frame_equal(archive.month_frame(DS, dt.datetime(2025, 1, 1, tzinfo=UTC)), frame)
    assert session.calls.count(url) == 3
    routes[url] = [(500, b"")]
    archive, session = fake_archive(routes, retries=2)
    with pytest.raises(DataSourceError, match="3회"):
        archive.month_frame(DS, dt.datetime(2025, 1, 1, tzinfo=UTC))
    routes[url] = [(403, b"")]  # 4xx(404 제외)는 재시도하지 않는다
    archive, session = fake_archive(routes, retries=2)
    with pytest.raises(DataSourceError, match="403"):
        archive.month_frame(DS, dt.datetime(2025, 1, 1, tzinfo=UTC))
    assert session.calls.count(url) == 1


def test_listing_pagination_and_monthly_sizes() -> None:
    prefix = "data/futures/um/monthly/klines/"
    page1 = (f"<ListBucketResult><Prefix>{prefix}</Prefix><IsTruncated>true</IsTruncated>"
             f"<NextMarker>{prefix}ETHUSDT/</NextMarker><CommonPrefixes><Prefix>{prefix}BTCUSDT/</Prefix>"
             f"</CommonPrefixes><CommonPrefixes><Prefix>{prefix}ETHUSDT/</Prefix></CommonPrefixes></ListBucketResult>")
    page2 = (f"<ListBucketResult><Prefix>{prefix}</Prefix><IsTruncated>false</IsTruncated>"
             f"<CommonPrefixes><Prefix>{prefix}XRPUSDT/</Prefix></CommonPrefixes></ListBucketResult>")
    key = "data/futures/um/monthly/klines/BTCUSDT/1h/"
    files = "".join(f"<Contents><Key>{key}BTCUSDT-1h-{m}.zip{suffix}</Key><Size>{size}</Size></Contents>"
                    for m, size in (("2024-01", 100), ("2024-02", 120)) for suffix in ("", ".CHECKSUM"))
    routes = {f"{LISTING}|{prefix}|": (200, page1.encode()),
              f"{LISTING}|{prefix}|{prefix}ETHUSDT/": (200, page2.encode()),
              f"{LISTING}|{key}|": (200, f"<ListBucketResult><IsTruncated>false</IsTruncated>{files}"
                                         f"</ListBucketResult>".encode())}
    archive, _ = fake_archive(routes)
    assert archive.list_symbols() == ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
    assert archive.list_monthly(DS) == {"2024-01": 100, "2024-02": 120}


def test_retry_policy_never_retries_permanent_errors() -> None:
    calls: list[int] = []

    def fail() -> None:
        calls.append(1)
        raise DataSourceError("451")

    with pytest.raises(DataSourceError):
        RetryPolicy(5, 0.0, retry_on=(Exception,), sleep=no_sleep).call("x", fail)
    assert len(calls) == 1


# --- cache -----------------------------------------------------------------------------------------------


class StubArchive:
    source_name = "stub"

    def __init__(self, clock: Any = None) -> None:
        self.requests: list[str] = []
        self.clock = clock or (lambda: NOW)

    def month_frame(self, dataset: Dataset, month: dt.datetime, *, from_day: dt.datetime | None = None) -> pd.DataFrame:
        self.requests.append(f"{month:%Y-%m}" + (f"@{from_day:%d}" if from_day is not None else ""))
        month_end = pd.Timestamp(month) + pd.offsets.MonthBegin(1)
        whole = synthetic(f"{month:%Y-%m-%d}", int((month_end - pd.Timestamp(month)) / pd.Timedelta(hours=1)), 60,
                          seed=month.month)  # 시각별 값이 수집 시점과 무관하게 같도록
        out = whole[whole.index < min(month_end, pd.Timestamp(self.clock()).floor("D"))]
        return out[out.index >= pd.Timestamp(from_day)] if from_day is not None else out

    def empty_frame(self, dataset: Dataset) -> pd.DataFrame:
        return empty_ohlcv()


def test_cache_final_partitions_and_slicing(tmp_path: Any) -> None:
    stub = StubArchive()
    clock = [NOW]
    cache = ArchiveCache(tmp_path, cast(VisionArchive, stub), refresh_recent_days=3, workers=2,
                         clock=lambda: clock[0])
    start, end = dt.datetime(2025, 1, 15, 5, tzinfo=UTC), dt.datetime(2025, 3, 5, tzinfo=UTC)
    first = cache.get(DS, start, end)
    assert sorted(stub.requests) == ["2025-01", "2025-02", "2025-03"]
    assert first.index[0] == pd.Timestamp(start) and first.index[-1] == pd.Timestamp(end) - pd.Timedelta(hours=1)
    manifest = cache.manifest(DS)
    assert manifest["2025-01"].final and manifest["2025-02"].final and not manifest["2025-03"].final
    stub.requests.clear()
    again = cache.get(DS, start, end)
    assert stub.requests == ["2025-03@08"]  # 확정 파티션은 디스크에서, 진행 중인 달은 지난 수집일 2일 전부터
    pd.testing.assert_frame_equal(again, first, check_freq=False)
    # 월말 직후(여유 3일 이내)에 받은 파티션은 확정이 아니다
    cache2 = ArchiveCache(tmp_path / "b", cast(VisionArchive, stub), refresh_recent_days=3,
                          clock=lambda: dt.datetime(2025, 3, 2, tzinfo=UTC))
    cache2.get(DS, dt.datetime(2025, 2, 1, tzinfo=UTC), dt.datetime(2025, 3, 1, tzinfo=UTC))
    assert not cache2.manifest(DS)["2025-02"].final


def test_cache_skips_months_missing_from_listing(tmp_path: Any) -> None:
    stub = StubArchive()
    cache = ArchiveCache(tmp_path, cast(VisionArchive, stub), clock=lambda: NOW)
    cache.get_many([DS], dt.datetime(2025, 1, 1, tzinfo=UTC), dt.datetime(2025, 3, 5, tzinfo=UTC),
                   months_available={DS: {"2025-02"}})
    assert sorted(stub.requests) == ["2025-02", "2025-03"]  # 이번 달은 목록과 무관하게 시도


# --- resample --------------------------------------------------------------------------------------------


def manual(frame: pd.DataFrame, per: int) -> pd.DataFrame:
    arr = {c: frame[c].to_numpy().reshape(-1, per) for c in frame.columns}
    return pd.DataFrame({"open": arr["open"][:, 0], "high": arr["high"].max(1), "low": arr["low"].min(1),
                         "close": arr["close"][:, -1], "volume": arr["volume"].sum(1),
                         "quote_volume": arr["quote_volume"].sum(1), "trades": arr["trades"].sum(1)},
                        index=frame.index[::per])


@pytest.mark.parametrize(("tf", "per"), [("15m", 3), ("1h", 12), ("4h", 48), ("1d", 288)])
def test_resample_matches_manual_on_utc_boundaries(tf: str, per: int) -> None:
    frame = synthetic("2024-12-30", 288 * 3)
    out = resample_ohlcv(frame, tf, source_timeframe="5m")
    expected = normalize_ohlcv(manual(frame, per))
    pd.testing.assert_frame_equal(out.drop(columns="bars"), expected, check_freq=False)
    assert (out["bars"] == per).all()
    if tf == "1d":
        assert [t.hour for t in out.index] == [0, 0, 0]  # 00:00 UTC = 09:00 KST
    if tf == "4h":
        assert sorted({t.hour for t in out.index}) == [0, 4, 8, 12, 16, 20]


def test_resample_partial_missing_and_incomplete() -> None:
    frame = synthetic("2025-01-01", 12 * 5)  # 5시간
    gapped = frame.drop(frame.index[[3, 4]]).drop(frame.index[24:36])  # 00:15·00:20 결측, 02시 전체 결측
    out = resample_ohlcv(gapped, "1h", source_timeframe="5m")
    assert out["bars"].tolist() == [10, 12, 12, 12]
    assert pd.Timestamp("2025-01-01 02:00", tz="UTC") not in out.index
    # as_of 이전에 끝나지 않은 버킷(진행 중)은 버린다
    partial = resample_ohlcv(frame, "4h", source_timeframe="5m", as_of=dt.datetime(2025, 1, 1, 5, tzinfo=UTC))
    assert len(partial) == 1
    head = resample_ohlcv(frame.iloc[:50], "1h", source_timeframe="5m")  # 04:05 까지 → 04시 버킷 미완성
    assert head.index[-1] == pd.Timestamp("2025-01-01 03:00", tz="UTC")
    with pytest.raises(ValueError):
        resample_ohlcv(frame, "1h", source_timeframe="7m")


@pytest.mark.parametrize("tf", ["15m", "1h", "4h", "1d"])
def test_bar_aggregator_equals_batch(tf: str) -> None:
    frame = synthetic("2024-12-31", 288 * 3, seed=5)
    frame = frame.drop(frame.index[[7, 8, 100, 101, 102, 500]])  # 결측 포함
    agg = BarAggregator(tf, source_timeframe="5m")
    bars = [b for rec in records(frame) for b in agg.update(*rec)]
    streamed = pd.DataFrame([{k: getattr(b, k) for k in ("open", "high", "low", "close", "volume", "quote_volume",
                                                         "trades", "bars")} for b in bars],
                            index=pd.DatetimeIndex([b.start for b in bars], name="time"))
    batch = resample_ohlcv(frame, tf, source_timeframe="5m")
    pd.testing.assert_frame_equal(streamed, batch, check_freq=False, check_index_type=False)
    with pytest.raises(ValueError):
        agg.update(frame.index[0], 1, 1, 1, 1, 0, 0, 0)


# --- quality ---------------------------------------------------------------------------------------------


def test_quality_gaps_violations_halts_delisting() -> None:
    frame = synthetic("2025-01-01", 100)
    frame = frame.drop(frame.index[10:13])
    frame.loc[frame.index[5], "high"] = frame["low"].iloc[5] - 1  # high < low
    for rows in (slice(30, 45), slice(80, 97)):  # 15봉 거래 중단, 끝까지 17봉(상장폐지)
        idx = frame.index[rows]
        frame.loc[idx, ["open", "high", "low", "close"]] = 50.0
        frame.loc[idx, "trades"] = 0
    frame.loc[frame.index[60], ["open", "high", "low", "close", "trades"]] = [50, 50, 50, 50, 0]  # 1봉은 중단 아님
    report = check_ohlcv(frame, "5m", dt.datetime(2025, 1, 1, tzinfo=UTC),
                         dt.datetime(2025, 1, 1, 8, 20, tzinfo=UTC), halt_min_bars=12)
    assert report.expected == 100 and report.missing_bars == 3 and report.gaps[0].bars == 3
    assert report.ohlc_violations == 1 and report.inactive_bars == 33
    assert [h.bars for h in report.halts] == [15, 17]
    assert report.delisted_at == frame.index[80]
    assert int(halt_mask(frame, 12).sum()) == 32


# --- symbols ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def universe() -> UniverseCfg:
    return load_settings().universe


def test_parse_contract() -> None:
    assert parse_contract("BTCUSDT") == Contract("BTCUSDT", "BTC", "USDT", False)
    assert parse_contract("1000PEPEUSDC") == Contract("1000PEPEUSDC", "1000PEPE", "USDC", False)
    assert parse_contract("ETHUSD1") == Contract("ETHUSD1", "ETH", "USD1", False)
    assert parse_contract("AERGOUSDTSETTLED") == Contract("AERGOUSDTSETTLED", "AERGO", "USDT", True)
    for symbol in ("BTCUSDT_240329", "ETHBTC", "USDT", "BTCUSD"):
        assert parse_contract(symbol) is None
    assert parse_contract("BTCUSDT").ccxt_symbol == "BTC/USDT:USDT"  # type: ignore[union-attr]


def test_rank_contracts_and_trading_contract(universe: UniverseCfg) -> None:
    grouped = rank_contracts(["BTCUSDT", "BTCUSDC", "BTCUSDT_250328", "USDCUSDT", "XAUUSDT", "ETHUSD1", "ETHBUSD",
                              "AERGOUSDTSETTLED", "AERGOUSDT"], universe)
    assert sorted(grouped) == ["AERGO", "BTC", "ETH"]  # 스테이블·TradFi 제외, BUSD 는 순위 대상 아님
    assert [c.quote for c in grouped["BTC"]] == ["USDT", "USDC"]
    assert trading_contract(grouped["AERGO"], universe) == Contract("AERGOUSDT", "AERGO", "USDT", False)
    assert trading_contract(grouped["ETH"], universe) is None  # USDT 계약이 없으면 거래 불가


# --- universe --------------------------------------------------------------------------------------------


def vol_frame(values: list[float], start: str = "2025-01-01", freq: str = "1h") -> pd.DataFrame:
    frame = pd.DataFrame({"quote_volume": values}, index=pd.date_range(start, periods=len(values), freq=freq, tz="UTC"))
    return frame


def test_coin_volume_sums_quotes_and_fills_grid() -> None:
    btc_t, btc_c = Contract("BTCUSDT", "BTC", "USDT", False), Contract("BTCUSDC", "BTC", "USDC", False)
    eth = Contract("ETHUSDT", "ETH", "USDT", False)
    frames = {btc_t: vol_frame([1, 2, 3]), btc_c: vol_frame([10, 10], "2025-01-01 01:00"), eth: vol_frame([5], "2025-01-01 04:00")}
    table = coin_volume(frames, {"USDT": 1.0, "USDC": 1.0}, "1h")
    assert table["BTC"].tolist() == [1, 12, 13, 0, 0] and table["ETH"].tolist() == [0, 0, 0, 0, 5]


def test_rolling_window_is_closed_bars_before_boundary() -> None:
    volume = pd.DataFrame({"A": np.arange(1.0, 49.0)}, index=pd.date_range("2025-01-01", periods=48, freq="1h", tz="UTC"))
    rolled = rolling_volume(volume, "1h", 24)
    t = pd.Timestamp("2025-01-02 00:00", tz="UTC")
    assert rolled.loc[t, "A"] == sum(range(1, 25))  # 1월 1일 00~23시 봉
    assert np.isnan(rolled["A"].to_numpy()[22]) and rolled.index[22] == t - pd.Timedelta(hours=1)  # 23개뿐
    assert rolled.index[-1] == volume.index[-1] + pd.Timedelta(hours=1)  # 마지막 봉 마감 경계
    with pytest.raises(ValueError):
        rolling_volume(volume, "2h", 1)  # 창이 봉 길이의 정수배가 아님


def test_ranks_and_lookups() -> None:
    idx = pd.date_range("2025-01-01", periods=3, freq="1h", tz="UTC")
    rolled = pd.DataFrame({"A": [5.0, 1.0, 0.0], "B": [3.0, 2.0, 4.0], "C": [np.nan, 2.0, 1.0]}, index=idx)
    table = ranks(rolled)
    assert table["A"].tolist()[:2] == [1.0, 3.0] and np.isnan(table["A"].iloc[2])  # 0 → 순위 없음
    assert table.to_numpy()[1].tolist() == [3.0, 1.0, 2.0]  # 동률(B·C)은 열 순서
    assert rank_at(table, idx[2], "B") == 1.0 and np.isnan(rank_at(table, idx[0], "C"))
    assert np.isnan(rank_at(table, idx[0] - pd.Timedelta(hours=1), "A")) and np.isnan(rank_at(table, idx[0], "Z"))
    assert ever_ranked(table, 1) == ["A", "B"]
    assert rank_windows(table, 2) == {"A": (idx[0], idx[0]), "B": (idx[0], idx[2]), "C": (idx[1], idx[2])}


def test_weekend_ratio_flags_tradfi_like() -> None:
    days = pd.date_range("2025-01-06", periods=56, freq="D", tz="UTC")  # 월요일부터 8주
    rng = np.random.default_rng(1)
    crypto = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, len(days))))
    moves = np.where(days.weekday >= 5, 0.0, rng.normal(0, 0.01, len(days)))
    tradfi = 100 * np.exp(np.cumsum(moves))
    ratio = weekend_ratio(pd.DataFrame({"BTC": crypto, "XAU": tradfi}, index=days))
    assert ratio["XAU"] < 0.45 < ratio["BTC"]


# --- REST (가짜 거래소) -----------------------------------------------------------------------------------


class FakeExchange:
    def __init__(self, bars: pd.DataFrame, fail_451: bool = False) -> None:
        self.bars, self.fail_451, self.calls = bars, fail_451, 0

    def fapiPublicGetKlines(self, params: dict[str, Any]) -> list[list[Any]]:
        self.calls += 1
        if self.fail_451:
            raise RuntimeError('binanceusdm GET https://fapi.binance.com/fapi/v1/klines 451 {"msg":"restricted location"}')
        start = pd.Timestamp(params["startTime"], unit="ms", tz="UTC")
        chunk = self.bars[self.bars.index >= start].iloc[: params["limit"]]
        return [[str(t.value // 1_000_000), str(o), str(h), str(lo), str(c), str(v), 0, str(qv), n, "0", "0", "0"]
                for t, o, h, lo, c, v, qv, n in records(chunk)]

    def fapiPublicGetFundingRate(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        times = pd.date_range("2025-01-01", periods=9, freq="8h", tz="UTC")
        out = [{"fundingTime": int(t.value // 1_000_000), "fundingRate": f"{i * 1e-5}"} for i, t in enumerate(times)]
        return [o for o in out if o["fundingTime"] >= params["startTime"]][: 4]

    def fapiPublicGetExchangeInfo(self) -> dict[str, Any]:
        return {"symbols": [{"symbol": "BTCUSDT", "status": "TRADING", "contractType": "PERPETUAL",
                             "underlyingType": "COIN", "baseAsset": "BTC", "marginAsset": "USDT",
                             "filters": [{"filterType": "LOT_SIZE", "minQty": "0.001", "stepSize": "0.001"},
                                         {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                                         {"filterType": "MIN_NOTIONAL", "notional": "100"}]}]}


def test_rest_klines_paginates_and_drops_open_bar() -> None:
    bars = synthetic("2025-01-01", 40, 15)
    now = dt.datetime(2025, 1, 1, 9, 50, tzinfo=UTC)  # 09:45 봉 진행 중
    provider = RestProvider(FakeExchange(bars), retry=RetryPolicy(0, 0.0), page_limit=7, clock=lambda: now)
    out = provider.klines("BTCUSDT", "15m", dt.datetime(2025, 1, 1, tzinfo=UTC), dt.datetime(2025, 1, 2, tzinfo=UTC))
    pd.testing.assert_frame_equal(out, bars.iloc[:39], check_freq=False)
    funding = provider.funding_rates("BTCUSDT", dt.datetime(2025, 1, 1, tzinfo=UTC), dt.datetime(2025, 1, 3, tzinfo=UTC))
    assert len(funding) == 2  # 09:50 기준 이미 정산된 것만 (00:00, 08:00)
    provider = RestProvider(FakeExchange(bars), retry=RetryPolicy(0, 0.0), clock=lambda: NOW)
    funding = provider.funding_rates("BTCUSDT", dt.datetime(2025, 1, 1, tzinfo=UTC), dt.datetime(2025, 1, 3, tzinfo=UTC))
    assert len(funding) == 6 and funding["funding_rate"].iloc[-1] == pytest.approx(5e-5)
    info = provider.exchange_info()["BTCUSDT"]
    assert (info.underlying_type, info.qty_step, info.price_tick, info.min_notional) == ("COIN", 0.001, 0.1, 100.0)


def test_rest_region_block_is_not_retried() -> None:
    exchange = FakeExchange(synthetic("2025-01-01", 5, 15), fail_451=True)
    provider = RestProvider(exchange, retry=RetryPolicy(3, 0.0, retry_on=(Exception,), sleep=no_sleep),
                            clock=lambda: NOW)
    with pytest.raises(DataSourceError, match="451"):
        provider.klines("BTCUSDT", "15m", dt.datetime(2025, 1, 1, tzinfo=UTC), dt.datetime(2025, 1, 2, tzinfo=UTC))
    assert exchange.calls == 1


def test_parse_symbol_info_defaults() -> None:
    info = parse_symbol_info({"symbol": "X", "filters": []})
    assert info.qty_step == 0.0 and info.quote == ""


# --- 원본 봉 대조 ----------------------------------------------------------------------------------------


def test_compare_detects_mismatches() -> None:
    from perpdiv.data.validate import compare

    source = synthetic("2025-01-01", 288)
    native = resample_ohlcv(source, "1h", source_timeframe="5m").drop(columns="bars")
    ok = compare(resample_ohlcv(source, "1h", source_timeframe="5m"), native, symbol="X", timeframe="1h",
                 month="2025-01", source_timeframe="5m")
    assert ok.ok and ok.native_bars == 24
    broken = source.drop(source.index[[0, 1]])
    broken.loc[broken.index[100], "high"] += 1.0
    bad = compare(resample_ohlcv(broken, "1h", source_timeframe="5m"), native.iloc[1:], symbol="X", timeframe="1h",
                  month="2025-01", source_timeframe="5m")
    assert (bad.partial_buckets, bad.only_resampled, bad.price_mismatches, bad.ok) == (1, 1, 1, False)


@pytest.mark.skipif(not os.environ.get("PERPDIV_NETWORK_TESTS"), reason="PERPDIV_NETWORK_TESTS=1 일 때만 (아카이브 접속)")
def test_resample_matches_native_archive_candles(tmp_path: Any) -> None:
    from perpdiv.data.check import build_archive
    from perpdiv.data.validate import check_resample

    settings = load_settings()
    cache = ArchiveCache(tmp_path, build_archive(settings), workers=8)
    checks = check_resample(cache, ["BTCUSDT", "DOGEUSDT"], [dt.datetime(2024, 3, 1, tzinfo=UTC)],
                            ["15m", "1h", "4h", "1d"])
    assert len(checks) == 8 and all(c.ok for c in checks)


def test_market_data_drops_halts_and_incomplete(tmp_path: Any) -> None:
    from perpdiv.data.store import MarketData

    class OneMonth(StubArchive):
        def month_frame(self, dataset: Dataset, month: dt.datetime, *,
                        from_day: dt.datetime | None = None) -> pd.DataFrame:
            frame = synthetic("2025-01-01", 288 * 2)  # 1월 1~2일
            idx = frame.index[288 + 12 * 3: 288 + 12 * 5]  # 2일 03~05시 거래 중단 24봉
            frame.loc[idx, ["open", "high", "low", "close"]] = 99.0
            frame.loc[idx, "trades"] = 0
            return frame

    market = MarketData(load_settings(), ArchiveCache(tmp_path, cast(VisionArchive, OneMonth()), clock=lambda: NOW))
    out = market.bars("BTCUSDT", "1h", dt.datetime(2025, 1, 1, tzinfo=UTC), dt.datetime(2025, 1, 2, 23, 30, tzinfo=UTC))
    assert out.halted_source_bars == 24 and out.quality.halts[0].bars == 24
    hours = [t.hour for t in out.bars.index if t.day == 2]
    assert 3 not in hours and 4 not in hours and hours[-1] == 22  # 23시 버킷은 end 에 걸려 미완성


def test_cache_resumes_current_month(tmp_path: Any) -> None:
    clock = [dt.datetime(2025, 3, 5, 12, tzinfo=UTC)]
    stub = StubArchive(clock=lambda: clock[0])
    cache = ArchiveCache(tmp_path, cast(VisionArchive, stub), clock=lambda: clock[0])
    march = (dt.datetime(2025, 3, 1, tzinfo=UTC), dt.datetime(2025, 4, 1, tzinfo=UTC))
    assert len(cache.get(DS, *march)) == 4 * 24  # 3월 1~4일
    clock[0] = dt.datetime(2025, 3, 9, 1, tzinfo=UTC)
    out = cache.get(DS, *march)
    assert stub.requests[-1] == "2025-03@03" and len(out) == 8 * 24
    assert out.index.is_unique and out.index.is_monotonic_increasing
    pd.testing.assert_frame_equal(out, StubArchive(clock=lambda: clock[0]).month_frame(DS, march[0]), check_freq=False)


def test_cache_resume_with_no_data(tmp_path: Any) -> None:
    class Empty(StubArchive):
        def month_frame(self, dataset: Dataset, month: dt.datetime, *,
                        from_day: dt.datetime | None = None) -> pd.DataFrame:
            self.requests.append(f"{month:%Y-%m}")
            return self.empty_frame(dataset)  # 상장폐지 등으로 이번 달 파일이 없음

    clock = [dt.datetime(2025, 3, 5, 12, tzinfo=UTC)]
    stub = Empty()
    cache = ArchiveCache(tmp_path, cast(VisionArchive, stub), clock=lambda: clock[0])
    march = (dt.datetime(2025, 3, 1, tzinfo=UTC), dt.datetime(2025, 4, 1, tzinfo=UTC))
    assert cache.get(DS, *march).empty
    clock[0] = dt.datetime(2025, 3, 9, tzinfo=UTC)
    assert cache.get(DS, *march).empty and stub.requests == ["2025-03", "2025-03"]
