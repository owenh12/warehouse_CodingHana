"""네트워크 없이 데이터 계층을 검증하기 위한 가짜 거래소·HTTP 세션."""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import re
import zipfile
from dataclasses import dataclass
from typing import Any

import ccxt

TF_MS = 15 * 60_000
UTC = dt.UTC


def ms(ts: dt.datetime) -> int:
    return int(ts.timestamp() * 1000)


def utc(*args: int) -> dt.datetime:
    return dt.datetime(*args, tzinfo=UTC)


def synthetic_bar(open_ms: int) -> list[float]:
    """시각에서 결정되는 OHLCV (재현 가능)."""
    k = open_ms // TF_MS
    o = 100.0 + (k % 97) * 0.5
    c = o + ((k % 5) - 2) * 0.25
    return [float(open_ms), o, max(o, c) + 0.3, min(o, c) - 0.3, c, float(1 + k % 7)]


SPOT_FILTERS = [
    {"filterType": "PRICE_FILTER", "minPrice": "0.01", "maxPrice": "1000000", "tickSize": "0.01"},
    {"filterType": "LOT_SIZE", "minQty": "0.00001", "maxQty": "9000", "stepSize": "0.00001"},
    {"filterType": "NOTIONAL", "minNotional": "5.00000000", "applyMinToMarket": True},
]
FUTURES_FILTERS = [
    {"filterType": "PRICE_FILTER", "minPrice": "556.80", "maxPrice": "4529764", "tickSize": "0.10"},
    {"filterType": "LOT_SIZE", "minQty": "0.001", "maxQty": "1000", "stepSize": "0.001"},
    {"filterType": "MARKET_LOT_SIZE", "minQty": "0.001", "maxQty": "120", "stepSize": "0.001"},
    {"filterType": "MIN_NOTIONAL", "notional": "100"},
]


class FakeExchange:
    """ccxt 거래소 흉내. 바이낸스처럼 진행 중인 봉까지 응답에 포함한다."""

    def __init__(
        self,
        first: dt.datetime,
        now: dt.datetime,
        *,
        missing: set[dt.datetime] | None = None,
        fail_times: int = 0,
        funding: dict[int, float] | None = None,
        filters: list[dict[str, Any]] | None = None,
        fail_with: Exception | None = None,
    ) -> None:
        self.first_ms, self.now_ms = ms(first), ms(now)
        self.missing = {ms(t) for t in (missing or set())}
        self.fail_times = fail_times
        self.fail_with = fail_with or ccxt.NetworkError("simulated network error")
        self.funding = funding or {}
        self.filters = filters or SPOT_FILTERS
        self.ohlcv_calls: list[tuple[str, int | None, int | None]] = []
        self.funding_calls: list[int | None] = []
        self.load_calls = 0
        self.market_symbols: list[str] = []

    def _maybe_fail(self) -> None:
        if self.fail_times > 0:
            self.fail_times -= 1
            raise self.fail_with

    def load_markets(self) -> dict[str, Any]:
        self._maybe_fail()
        self.load_calls += 1
        return {}

    def market(self, symbol: str) -> dict[str, Any]:
        self.market_symbols.append(symbol)
        return {"info": {"filters": self.filters}}

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, since: int | None = None, limit: int | None = None
    ) -> list[list[float]]:
        self._maybe_fail()
        self.ohlcv_calls.append((symbol, since, limit))
        start = max(since or 0, self.first_ms)
        t = -(-start // TF_MS) * TF_MS
        rows: list[list[float]] = []
        while len(rows) < (limit or 500) and t <= self.now_ms:
            if t not in self.missing:
                rows.append(synthetic_bar(t))
            t += TF_MS
        return rows

    def fetch_funding_rate_history(
        self, symbol: str, since: int | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        self._maybe_fail()
        self.funding_calls.append(since)
        times = [t for t in sorted(self.funding) if t >= (since or 0) and t <= self.now_ms]
        return [{"timestamp": t, "fundingRate": self.funding[t]} for t in times[: limit or 100]]


# ---------------------------------------------------------------------------
# data.binance.vision 흉내
# ---------------------------------------------------------------------------


def make_zip(name: str, text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, text)
    return buffer.getvalue()


def kline_csv(rows: list[list[float]], *, microseconds: bool, header: bool) -> str:
    lines = []
    if header:
        lines.append("open_time,open,high,low,close,volume,close_time,quote_volume,count,"
                     "taker_buy_volume,taker_buy_quote_volume,ignore")
    scale = 1000 if microseconds else 1
    for r in rows:
        open_t = int(r[0]) * scale
        close_t = (int(r[0]) + TF_MS - 1) * scale
        lines.append(f"{open_t},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{close_t},0,0,0,0,0")
    return "\n".join(lines) + "\n"


@dataclass
class FakeResponse:
    status_code: int
    content: bytes = b""


_KLINE_RE = re.compile(
    r"/data/(?P<market>spot|futures/um)/(?P<freq>monthly|daily)/klines/(?P<sym>\w+)/15m/"
    r"(?P=sym)-15m-(?P<period>[\d-]+)\.zip(?P<checksum>\.CHECKSUM)?$"
)
_FUNDING_RE = re.compile(
    r"/data/futures/um/monthly/fundingRate/(?P<sym>\w+)/(?P=sym)-fundingRate-(?P<period>[\d-]+)"
    r"\.zip(?P<checksum>\.CHECKSUM)?$"
)


class FakeVisionSession:
    """URL 을 해석해 합성 zip·CHECKSUM 을 돌려주는 requests.Session 흉내.

    ``first_month`` 이전 월과 ``unpublished`` 에 든 기간(YYYY-MM 또는 YYYY-MM-DD)은 404.
    """

    def __init__(
        self,
        first_month: str = "2017-08",
        unpublished: set[str] | None = None,
        funding: dict[int, float] | None = None,
        corrupt: set[str] | None = None,
    ) -> None:
        self.first_month = first_month
        self.unpublished = unpublished or set()
        self.funding = funding or {}
        self.corrupt = corrupt or set()
        self.requests: list[tuple[str, str]] = []

    def _period_range(self, period: str) -> tuple[dt.datetime, dt.datetime]:
        if len(period) == 7:
            start = dt.datetime.strptime(period, "%Y-%m").replace(tzinfo=UTC)
            end = (start.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        else:
            start = dt.datetime.strptime(period, "%Y-%m-%d").replace(tzinfo=UTC)
            end = start + dt.timedelta(days=1)
        return start, end

    def _payload(self, url: str) -> tuple[str, bytes] | None:
        if m := _KLINE_RE.search(url):
            period = m["period"]
            if period[:7] < self.first_month or period in self.unpublished:
                return None
            start, end = self._period_range(period)
            rows = [synthetic_bar(t) for t in range(ms(start), ms(end), TF_MS)]
            spot = m["market"] == "spot"
            text = kline_csv(rows, microseconds=spot and start >= utc(2025, 1, 1), header=not spot)
            name = f"{m['sym']}-15m-{period}"
        elif m := _FUNDING_RE.search(url):
            period = m["period"]
            if period in self.unpublished:
                return None
            start, end = self._period_range(period)
            rows_f = [(t, r) for t, r in sorted(self.funding.items()) if ms(start) <= t < ms(end)]
            text = "calc_time,funding_interval_hours,last_funding_rate\n" + "".join(
                f"{t},8,{r}\n" for t, r in rows_f
            )
            name = f"{m['sym']}-fundingRate-{period}"
        else:
            return None
        blob = make_zip(name + ".csv", text)
        if m["checksum"]:
            digest = hashlib.sha256(blob).hexdigest()
            if period in self.corrupt:
                digest = "0" * 64
            return "checksum", f"{digest}  {name}.zip".encode()
        return "zip", blob

    def get(self, url: str, timeout: float | None = None) -> FakeResponse:
        self.requests.append(("GET", url))
        payload = self._payload(url)
        return FakeResponse(404) if payload is None else FakeResponse(200, payload[1])

    def head(self, url: str, timeout: float | None = None) -> FakeResponse:
        self.requests.append(("HEAD", url))
        return FakeResponse(404 if self._payload(url) is None else 200)
