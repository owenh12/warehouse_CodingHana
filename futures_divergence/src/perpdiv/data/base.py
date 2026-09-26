"""OHLCV 프레임 규격과 공통 예외.

규격 프레임:
- 인덱스: 봉 시작 시각 ``time`` (UTC tz-aware ``DatetimeIndex``), 오름차순, 중복 없음
- 열: ``open high low close volume quote_volume trades`` (``trades`` 는 int64, 나머지 float64)

``quote_volume`` 은 순위(거래대금) 계산에, ``trades`` 는 거래 없음(상장폐지 후 고정가 봉) 판정에 쓴다.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume", "quote_volume", "trades")
PRICE_COLUMNS = ("open", "high", "low", "close")


class DataSourceError(RuntimeError):
    """원천(거래소·아카이브) 접근 실패, 검증 실패, 형식 오류."""


def empty_ohlcv() -> pd.DataFrame:
    frame = pd.DataFrame({c: pd.Series(dtype="float64") for c in OHLCV_COLUMNS})
    frame["trades"] = frame["trades"].astype("int64")
    frame.index = pd.DatetimeIndex([], tz="UTC", name="time").as_unit("ns")
    return frame


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """열 순서·dtype·인덱스를 규격에 맞춘다. 중복 시각은 값이 같으면 하나로 줄이고, 다르면 오류."""
    if frame.empty:
        return empty_ohlcv()
    out = frame.loc[:, list(OHLCV_COLUMNS)].copy()
    for col in OHLCV_COLUMNS[:-1]:
        out[col] = out[col].astype("float64")
    out["trades"] = out["trades"].astype("int64")
    index = pd.DatetimeIndex(out.index)
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    out.index = index.as_unit("ns").rename("time")
    out = out.sort_index(kind="stable")
    dup = out.index.duplicated(keep=False)
    if dup.any():
        groups = out[dup].groupby(level=0)
        conflicting = [ts for ts, g in groups if len(g.drop_duplicates()) > 1]
        if conflicting:
            raise DataSourceError(f"같은 시각에 값이 다른 봉이 있습니다: {conflicting[:3]}")
        out = out[~out.index.duplicated(keep="last")]
    return out


def slice_range(frame: pd.DataFrame, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    """봉 시작 시각 기준 [start, end)."""
    mask = (frame.index >= pd.Timestamp(start)) & (frame.index < pd.Timestamp(end))
    return frame.loc[mask]


def ohlcv_from_rows(rows: list[list[float]]) -> pd.DataFrame:
    """ccxt/바이낸스 응답 행 [open_ms, o, h, l, c, v, (quote_v, trades)] → 규격 프레임."""
    if not rows:
        return empty_ohlcv()
    arr = np.asarray(rows, dtype="float64")
    n = arr.shape[1]
    frame = pd.DataFrame(
        {
            "open": arr[:, 1], "high": arr[:, 2], "low": arr[:, 3], "close": arr[:, 4], "volume": arr[:, 5],
            "quote_volume": arr[:, 6] if n > 6 else np.nan,
            "trades": arr[:, 7].astype("int64") if n > 7 else -1,
        },
        index=pd.to_datetime(arr[:, 0].astype("int64"), unit="ms", utc=True),
    )
    return normalize_ohlcv(frame)
