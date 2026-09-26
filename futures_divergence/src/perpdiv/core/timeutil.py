"""시간 규칙: 저장·계산은 UTC(tz-aware), 표시만 KST.

봉 타임스탬프는 봉 **시작** 시각이다. 15분봉 ``00:00`` 봉은 [00:00, 00:15) 구간이고,
종가는 00:15 에 확정된다. 신호는 종가 확정 시각 이후에만 발생할 수 있다.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

UTC = dt.UTC


def ensure_utc(ts: dt.datetime) -> dt.datetime:
    """tz-aware 시각을 UTC로 바꾼다. naive 시각은 해석이 모호하므로 거부한다."""
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError(f"naive datetime 은 허용하지 않습니다 (UTC tz-aware 필요): {ts!r}")
    return ts.astimezone(UTC)


def to_display(ts: dt.datetime, tz_name: str) -> dt.datetime:
    return ensure_utc(ts).astimezone(ZoneInfo(tz_name))


def utc_now() -> dt.datetime:
    return dt.datetime.now(UTC)


def to_epoch_ms(ts: dt.datetime) -> int:
    return int(ensure_utc(ts).timestamp() * 1000)


def from_epoch_ms(ms: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(ms / 1000, tz=UTC)


def floor_time(ts: dt.datetime, minutes: int) -> dt.datetime:
    """epoch(1970-01-01 00:00 UTC) 기준으로 minutes 단위 내림. 바이낸스 봉 경계와 같다."""
    epoch_s = int(ensure_utc(ts).timestamp())
    step = minutes * 60
    return dt.datetime.fromtimestamp(epoch_s - epoch_s % step, tz=UTC)


def month_starts(start: dt.datetime, end: dt.datetime) -> list[dt.datetime]:
    """[start, end) 와 겹치는 각 달의 1일 00:00 UTC."""
    start, end = ensure_utc(start), ensure_utc(end)
    current = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    out: list[dt.datetime] = []
    while current < end:
        out.append(current)
        current = next_month(current)
    return out


def next_month(month_start: dt.datetime) -> dt.datetime:
    if month_start.month == 12:
        return month_start.replace(year=month_start.year + 1, month=1)
    return month_start.replace(month=month_start.month + 1)
