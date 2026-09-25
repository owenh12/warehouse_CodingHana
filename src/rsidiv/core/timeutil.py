"""시간 규칙: 모든 타임스탬프는 UTC(tz-aware)로 저장·계산하고, 표시할 때만 변환한다.

봉 타임스탬프는 봉 시작 시각(left label)이다. 15분봉 ``09:00 KST`` 봉은
[09:00, 09:15) 구간이며, 그 봉의 종가는 09:15 에 확정된다.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

UTC = dt.timezone.utc


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
