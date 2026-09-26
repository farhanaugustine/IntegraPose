"""Protect analytics artifacts when changing their visual presentation."""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from PIL import Image

from integra_pose.logic.batch_figures import export_batch_figure_bundle
from integra_pose.logic.supervision_runner import GridMetricsRecorder, SupervisionInferenceRunner
from integra_pose.utils.bout_analyzer import save_analysis_outputs


def test_bout_visual_options_preserve_tables_and_produce_all_requested_plots(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("names:\n  0: Explore\n  1: Groom\nrois: {}\n", encoding="utf-8")
    source = pd.DataFrame([
        {"track_id": track, "frame": frame, "class_id": (frame // 12 + track) % 2,
         "ROI Name": "Center" if frame % 24 < 12 else "Edge",
         "ROI Memberships": ("Center",) if frame % 24 < 12 else ("Edge",)}
        for track in (1, 2) for frame in range(120)
    ])
    original = source.copy(deep=True)
    previous_figures = plt.get_fignums()
    previous_style = matplotlib.rcParams.copy()
    for mode in ("off", "on"):
        result = save_analysis_outputs(
            source, str(config), str(tmp_path / mode), 0, 2, 10,
            video_name="sample", roi_column="ROI Name", class_names={0: "Explore", 1: "Groom"},
            run_id="plot-contract",
            enabled_modules=["temporal_trends", "activity_budgets", "bout_timeline_export"],
            visual_prefs={
                "temporal_trends": "line bar" if mode == "on" else "none",
                "activity_budgets": "stacked violin" if mode == "on" else "none",
                "bout_timeline": "gantt" if mode == "on" else "none",
            },
        )
        assert len(result[0]) == 20
    before, after = tmp_path / "off", tmp_path / "on"
    csv_names = {p.relative_to(before) for p in before.rglob("*.csv")}
    assert len(csv_names) >= 10
    assert csv_names == {p.relative_to(after) for p in after.rglob("*.csv")}
    for relative in csv_names:
        assert (before / relative).read_bytes() == (after / relative).read_bytes(), relative
    workbook_names = {p.relative_to(before) for p in before.rglob("*.xlsx")}
    assert workbook_names
    assert workbook_names == {p.relative_to(after) for p in after.rglob("*.xlsx")}
    for relative in workbook_names:
        with pd.ExcelFile(before / relative) as a, pd.ExcelFile(after / relative) as b:
            assert a.sheet_names == b.sheet_names
            for sheet in a.sheet_names:
                pd.testing.assert_frame_equal(a.parse(sheet), b.parse(sheet))
    expected = {
        "sample_analytics_dashboard.png",
        "temporal_trends/sample_behavior_time_cumulative.png",
        "temporal_trends/sample_behavior_time_stacked.png",
        "activity_budgets/sample_activity_budget_stacked.png",
        "activity_budgets/sample_activity_budget_violin.png",
        "bout_timeline_export/sample_bout_timeline_gantt.png",
    }
    assert {p.relative_to(after).as_posix() for p in after.rglob("*.png")} == expected
    assert {p.name for p in before.rglob("*.png")} == {"sample_analytics_dashboard.png"}
    for relative in expected:
        with Image.open(after / relative) as im:
            im.verify()
    pd.testing.assert_frame_equal(source, original)
    assert plt.get_fignums() == previous_figures
    assert matplotlib.rcParams == previous_style


@pytest.mark.parametrize("tracks", [1, 3])
def test_metrics_dashboard_keeps_source_measurements_and_exports_summary(tmp_path, tracks):
    source = pd.DataFrame([
        dict(frame=frame, object_id=str(track), movement_speed_px_per_frame=track + 2 + np.sin(frame),
             acceleration_px_per_frame2=np.cos(frame), orientation_deg=(frame * 7) % 360,
             signed_angular_velocity_deg_per_frame=7, body_length_px=20, body_aspect_ratio=2,
             total_path_length_px=frame * (track + 2), turn_count=frame // 10,
             nearest_distance_px=20 + frame, objects_in_frame=tracks)
        for track in range(tracks) for frame in range(60)
    ])
    path = tmp_path / "tracking_metrics.csv"
    source.to_csv(path, index=False)
    original = path.read_bytes()
    previous_style = matplotlib.rcParams.copy()
    previous_figures = plt.get_fignums()
    runner = SupervisionInferenceRunner.__new__(SupervisionInferenceRunner)
    messages = []
    runner.log = lambda message, level: messages.append((message, level))
    runner._render_metrics_dashboard(path)
    assert path.read_bytes() == original
    assert not [message for message, level in messages if level in {"ERROR", "WARNING"}]
    with Image.open(tmp_path / "metrics_dashboard.png") as im:
        im.verify()
    assert len(list(tmp_path.rglob("*.csv"))) > 1
    assert matplotlib.rcParams == previous_style
    assert plt.get_fignums() == previous_figures


def test_grid_heatmap_render_preserves_counts_and_produces_three_images(tmp_path):
    grid = GridMetricsRecorder(tmp_path, 20, True, lambda *args: None)
    grid._rows, grid._cols = 2, 3
    grid._dwell_counts = np.arange(6, dtype=float).reshape(2, 3)
    grid._occupancy_counts = grid._dwell_counts.copy()
    grid._behavior_counts = {"Explore": grid._dwell_counts.copy()}
    original = grid._dwell_counts.copy()
    previous_style = matplotlib.rcParams.copy()
    grid._save_heatmap_images(10)
    np.testing.assert_array_equal(grid._dwell_counts, original)
    assert {p.name for p in tmp_path.glob("*.png")} == {
        "grid_dwell_heatmap.png", "grid_occupancy_heatmap.png", "grid_dominant_behavior_heatmap.png"}
    assert matplotlib.rcParams == previous_style


def test_batch_export_does_not_change_the_callers_plot_style(tmp_path):
    with plt.rc_context({"axes.facecolor": "#123456", "grid.linestyle": "--"}):
        previous_style = matplotlib.rcParams.copy()
        export_batch_figure_bundle(
            video_summary_df=pd.DataFrame(), omnibus_df=pd.DataFrame(), pairwise_df=pd.DataFrame(),
            output_dir=tmp_path, export_individual_profiles=False,
        )
        assert matplotlib.rcParams == previous_style
