"""신호 육안 검증 보고서: 파일 생성, 표시 기간 선택, 퍼널 합계."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from test_divergence import RELAXED, params, random_frame

from rsidiv.reports.signal_report import (
    build_signal_report,
    first_reason_counts,
    pattern_ok,
)


def test_build_signal_report(tmp_path: Path) -> None:
    frame = random_frame(3000, 1)
    p = params(RELAXED)
    start, end = frame.index[1500], frame.index[-1] + pd.Timedelta("15min")
    report = build_signal_report(
        frame, p, symbol="TEST/USDT", market="spot", timeframe="15m", asset_class="crypto",
        window_start=start.to_pydatetime(), window_end=end.to_pydatetime(), display_tz="Asia/Seoul",
        out_dir=tmp_path, max_near_miss_charts=3,
    )
    assert report.signals and all(start <= c.signal_time < end for c in report.window_candidates)
    assert all(c.accepted for c in report.signals)
    assert all(pattern_ok(c) and not c.accepted for c in report.near_misses)
    assert sum(first_reason_counts(report.all_candidates).values()) == len(report.all_candidates)
    names = {path.name for path in report.charts}
    assert "overview.png" in names and "signal_01.png" in names
    assert len([n for n in names if n.startswith("near_")]) == min(3, len(report.near_misses))
    assert all(path.stat().st_size > 10_000 for path in report.charts)
    table = pd.read_csv(tmp_path / "candidates.csv")
    assert len(table) == len(report.window_candidates) and table["accepted"].sum() == len(report.signals)
    summary = report.summary_path.read_text(encoding="utf-8")
    assert f"## 신호 ({len(report.signals)}건)" in summary and "signal_01.png" in summary
