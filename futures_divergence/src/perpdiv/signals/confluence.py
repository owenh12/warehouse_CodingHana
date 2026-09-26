"""다중 TF 동시 성립 (전략 수정 후 진입 조건).

신호 s 는 확정 시각부터 자기 TF 로 ``validity_bars`` 봉 동안 유효하다: ``[t_s, t_s + validity_bars × TF_s)``.
새 신호(확정 신호) i 가 시각 T 에 확정되면, 같은 코인·같은 방향으로 T 에 유효한 신호들의 **서로 다른 TF 수**를 센다
(i 자신 포함, 같은 TF 여러 개는 1개). 이 수가 ``min_timeframes`` 이상이면 i 는 진입 후보다.
이미 다른 진입에 쓰인 신호도 유효 기간 안이면 다시 센다(사용자 결정: 판정은 새 신호가 올 때만).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from perpdiv.core.config import StrategyCfg


def annotate_confluence(signals: pd.DataFrame, cfg: StrategyCfg) -> pd.DataFrame:
    """열 추가: ``confluence``(유효한 서로 다른 TF 수, 자신 포함), ``confluence_tfs``(예: "4h+15m", 큰 TF 먼저)."""
    out = signals.copy()
    count = np.ones(len(out), dtype=np.int64)
    names = out["timeframe"].astype(str).to_numpy(dtype=object).copy()
    validity = cfg.confluence.validity_bars
    positions = pd.RangeIndex(len(out))
    for _, group in out.assign(_pos=positions).groupby(["coin", "side"], sort=False):
        g = group.sort_values("signal_time", kind="stable")
        index = pd.DatetimeIndex(g["signal_time"])
        naive = index.tz_convert("UTC").tz_localize(None) if index.tz is not None else index
        times = naive.as_unit("ns").to_numpy().view(np.int64)
        minutes = g["tf_minutes"].to_numpy(dtype=np.int64)
        ends = times + minutes * validity * 60_000_000_000
        labels = g["timeframe"].astype(str).to_numpy()
        pos = g["_pos"].to_numpy()
        longest = int(minutes.max()) * validity * 60_000_000_000
        for i in range(len(g)):
            lo = int(np.searchsorted(times, times[i] - longest, side="left"))
            hi = int(np.searchsorted(times, times[i], side="right"))
            active = {int(minutes[j]): labels[j] for j in range(lo, hi) if ends[j] > times[i]}
            active[int(minutes[i])] = labels[i]
            count[pos[i]] = len(active)
            names[pos[i]] = "+".join(active[m] for m in sorted(active, reverse=True))
    out["confluence"] = count
    out["confluence_tfs"] = names
    return out
