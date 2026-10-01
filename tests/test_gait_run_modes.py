import json
import tkinter as tk
from unittest.mock import Mock

import pandas as pd
import pytest

from integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.gui_launcher import AnalysisGUI
from integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.profiles import preset_config, PRESET_NAMES
from integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.compare_gait import main as compare_groups


@pytest.fixture(scope="module")
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def dashboard(tk_root):
    root = tk_root
    app = AnalysisGUI(root)
    app.withdraw()
    yield app
    app.destroy()


def test_single_video_needs_no_groups_and_keeps_advanced_controls(dashboard, tmp_path, monkeypatch):
    app = dashboard
    app.results_dir_var.set(str(tmp_path))
    app.detected_pairs = {"demo": {"video_path": "demo.avi", "yolo_dir": "labels"}}
    app.video_listbox.insert(tk.END, "demo")
    app.video_listbox.selection_set(0)
    error = Mock()
    thread = Mock()
    monkeypatch.setattr("integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.gui_launcher.messagebox.showerror", error)
    monkeypatch.setattr("integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.gui_launcher.threading.Thread", thread)
    app.start_analysis_thread()
    error.assert_not_called()
    thread.return_value.start.assert_called_once()
    assert app.run_videos == ["demo"]
    assert not app.group_a_videos and not app.group_b_videos
    assert "Run Convergent Cross-Mapping (CCM)" in app.analysis_vars
    assert "Run Advanced Behavioral Analysis (UMAP)" in app.analysis_vars
    assert "Run Decision Dynamics Analysis" in app.analysis_vars


def test_comparison_requires_groups_then_accepts_them(dashboard, tmp_path, monkeypatch):
    app = dashboard
    app.results_dir_var.set(str(tmp_path))
    app.detected_pairs = {"a": {}, "b": {}}
    app.analysis_vars["Compare Gait Metrics"].set(True)
    error, thread = Mock(), Mock()
    monkeypatch.setattr("integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.gui_launcher.messagebox.showerror", error)
    monkeypatch.setattr("integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.gui_launcher.threading.Thread", thread)
    app.start_analysis_thread()
    thread.assert_not_called()
    assert error.call_args.args[0] == "Groups Required"
    app.run_mode_var.set("Groups")
    app.group_a_listbox.insert(tk.END, "a")
    app.group_b_listbox.insert(tk.END, "b")
    error.reset_mock()
    app.start_analysis_thread()
    error.assert_not_called()
    thread.return_value.start.assert_called_once()
    assert app.run_videos == ["a", "b"]


def test_gui_preserves_custom_settings_and_human_mapping(dashboard):
    config = preset_config(PRESET_NAMES[1])
    config["REVIEW"]["PADDING_SECONDS"] = 1.25
    config["GENERAL_PARAMS"]["TARGET_TRACK_ID"] = 0
    config["REVIEW"]["EXPORT_COORDINATION"] = True
    dashboard.apply_config(json.loads(json.dumps(config)))
    saved = dashboard.gather_current_config()
    assert saved["GAIT_ANALYSIS"]["OPPOSING_PAW"] == "Right Ankle"
    assert saved["GENERAL_PARAMS"]["TARGET_TRACK_ID"] == 0
    assert saved["REVIEW"]["PADDING_SECONDS"] == 1.25
    assert saved["POSE_METRICS"]["BODY_ANGLE_CONNECTION"] == []
    assert saved['SUBJECT_TYPE'] == 'human'
    assert saved['REVIEW']['EXPORT_COORDINATION'] is True
    assert str(dashboard.analysis_buttons['Run Advanced Behavioral Analysis (UMAP)'].cget('state')) == 'disabled'
    assert str(dashboard.analysis_buttons['Compare Gait Metrics'].cget('state')) == 'normal'
    dashboard.apply_config(preset_config(PRESET_NAMES[0]))
    animal = dashboard.gather_current_config()
    assert animal['SUBJECT_TYPE'] == 'animal'
    assert animal['GAIT_ANALYSIS']['PAW_ORDER_HILDEBRAND'] == preset_config()['GAIT_ANALYSIS']['PAW_ORDER_HILDEBRAND']
    assert animal['REVIEW']['EXPORT_COORDINATION'] is False
    assert str(dashboard.analysis_buttons['Run Advanced Behavioral Analysis (UMAP)'].cget('state')) == 'normal'


def test_import_reordered_human_schema_maps_anatomy_by_name(dashboard, monkeypatch):
    monkeypatch.setattr(dashboard, 'show_mapping', lambda: None)
    names = list(reversed(preset_config(PRESET_NAMES[1])['DATASET']['KEYPOINT_ORDER']))
    dashboard.install_schema({'keypoint_names': names, 'keypoint_count': 17, 'source': 'test model', 'model_path': 'best.pt', 'edges': []})
    result = dashboard.gather_current_config()
    assert result['DATASET']['KEYPOINT_ORDER'] == names
    assert result['GAIT_ANALYSIS']['STRIDE_REFERENCE_PAW'] == 'Left Ankle'
    assert result['GAIT_ANALYSIS']['OPPOSING_PAW'] == 'Right Ankle'
    assert names.index('Left Ankle') == 1
    assert result['DATASET']['SKELETON_EDGES']


def test_aggregation_uses_only_selected_videos(dashboard, tmp_path):
    dashboard.base_results_dir = str(tmp_path)
    dashboard.run_videos = ["chosen"]
    for name, value in [("chosen", 1), ("unrelated_mouse_run", 999)]:
        folder = tmp_path / name
        folder.mkdir()
        pd.DataFrame({"stride_speed": [value]}).to_csv(folder / "gait_analysis_summary.csv", index=False)
    dashboard.aggregate_gait_data_dynamically()
    result = pd.read_csv(tmp_path / "aggregated_gait_analysis.csv")
    assert result.video_source.tolist() == ["chosen"]


def test_group_html_exports_per_video_units_without_pooling_strides(tmp_path):
    for name in ('a', 'b'):
        (tmp_path/name).mkdir()
        (tmp_path/name/'analysis_config.json').write_text(json.dumps(preset_config()))
    # Three strides in A and one in B still yield two observations.
    data = pd.DataFrame({"video_source": ["a", "a", "a", "b"], "stride_speed": [1, 2, 3, 20],
                         "stride_duration_s": [.2, .3, .4, .5]})
    data.to_csv(tmp_path / "aggregated_gait_analysis.csv", index=False)
    report = compare_groups(str(tmp_path), [{"name": "A", "videos": ["a"]}, {"name": "B", "videos": ["b"]}], preset_config())
    assert report.endswith("gait_group_report.html")
    table = pd.read_csv(next(tmp_path.glob("comparison_plots_*/per_video_means.csv")))
    speeds = table[table.metric == "stride_speed"].set_index("video_source").mean_value.to_dict()
    assert speeds == {"a": 2., "b": 20.}
