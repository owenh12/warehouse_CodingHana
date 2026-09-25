"""시간 규칙: 모든 타임스탬프는 UTC(tz-aware)로 저장·계산하고, 표시할 때만 변환한다.

봉 타임스탬프는 봉 시작 시각(left label)이다. 15분봉 ``09:00 KST`` 봉은
[09:00, 09:15) 구간이며, 그 봉의 종가는 09:15 에 확정된다.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

UTC = dt.UTC


def ensure_utc(ts: dt.datetime) -> dt.datetime:
    """tz-aware 시각을 UTC로 변환한다. naive 시각은 해석이 모호하므로 거부한다."""
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError(f"naive datetime 은 허용되지 않습니다 (UTC tz-aware 필요): {ts!r}")
    return ts.astimezone(UTC)


def to_display(ts: dt.datetime, tz_name: str) -> dt.datetime:
    """표시용 변환 (예: Asia/Seoul). 저장·계산에는 사용하지 않는다."""
    return ensure_utc(ts).astimezone(ZoneInfo(tz_name))


def bar_close_time(bar_open: dt.datetime, timeframe_minutes: int) -> dt.datetime:
    """봉 시작 시각 → 종가 확정 시각. 신호는 이 시각 이후에만 발생할 수 있다."""
    return ensure_utc(bar_open) + dt.timedelta(minutes=timeframe_minutes)


def utc_now() -> dt.datetime:
    """현재 UTC 시각. 테스트에서는 clock 인자로 대체한다."""
    return dt.datetime.now(UTC)


def to_epoch_ms(ts: dt.datetime) -> int:
    """UTC 시각 → epoch 밀리초 (거래소 API 형식)."""
    return int(ensure_utc(ts).timestamp() * 1000)


def from_epoch_ms(ms: int) -> dt.datetime:
    """epoch 밀리초 → UTC 시각."""
    return dt.datetime.fromtimestamp(ms / 1000, tz=UTC)


def month_starts(start: dt.datetime, end: dt.datetime) -> list[dt.datetime]:
    """[start, end) 구간과 겹치는 각 달의 1일 00:00 UTC 목록 (캐시 파티션 단위)."""
    start, end = ensure_utc(start), ensure_utc(end)
    current = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    months: list[dt.datetime] = []
    while current < end:
        months.append(current)
        current = next_month(current)
    return months


def next_month(month_start: dt.datetime) -> dt.datetime:
    """해당 달 1일 → 다음 달 1일."""
    if month_start.month == 12:
        return month_start.replace(year=month_start.year + 1, month=1)
    return month_start.replace(month=month_start.month + 1)
