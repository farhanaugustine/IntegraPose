"""Validate workbook contents and failure handling for EDA bout exports."""
from pathlib import Path

import pandas as pd

from integra_pose.plugins.plugin_eda.core import bout_analysis_utils as bouts


def write_bouts(path):
    rows = []
    for bout_id, start, end, label in [(1, 0, 9, 'investigation'), (2, 15, 19, 'investigation'), (3, 20, 29, 'rest')]:
        for frame in range(start, end + 1):
            rows.append({
                'Bout ID (Global)': bout_id, 'Frame Number': frame,
                'Class Label': label, 'Track ID': 7,
                'Bout Start Frame': start, 'Bout End Frame': end,
                'Bout Duration (Frames)': end - start + 1,
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_workbook_preserves_bout_counts_and_time_units(tmp_path):
    source = tmp_path/'bouts.csv'
    write_bouts(source)
    result = bouts.save_analysis_to_excel_per_track(str(source), str(tmp_path), {0: 'investigation', 1: 'rest'}, 10., 'trial')
    assert result is not None
    summary = pd.read_excel(result).set_index('Class Label')
    assert len(summary) == 2
    assert summary.loc['investigation', 'Number of Bouts'] == 2
    assert summary.loc['investigation', 'Avg Bout Duration (frames)'] == 7.5
    assert summary.loc['investigation', 'Avg Bout Duration (seconds)'] == .75
    assert summary.loc['investigation', 'Avg Time Between Bouts (seconds) for this Track/Class'] == .5
    assert summary.loc['rest', 'Number of Bouts'] == 1
    assert summary.loc['rest', 'Avg Bout Duration (seconds)'] == 1.


def test_export_failure_does_not_substitute_an_empty_workbook(tmp_path, monkeypatch):
    source = tmp_path/'bouts.csv'
    write_bouts(source)
    def unavailable(*args, **kwargs):
        raise PermissionError('Workbook destination unavailable')
    monkeypatch.setattr(bouts.pd, 'ExcelWriter', unavailable)
    result = bouts.save_analysis_to_excel_per_track(str(source), str(tmp_path), {0: 'investigation'}, 10., 'trial')
    assert result is None
    assert not (tmp_path/'trial_bout_summary_per_track.xlsx').exists()
