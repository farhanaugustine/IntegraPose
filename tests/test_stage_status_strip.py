from types import MethodType, SimpleNamespace

import pandas as pd
import pytest
import tkinter as tk

from integra_pose.main_gui_app import YoloApp
from integra_pose.utils.config_manager import ConfigManager


def _make_status_app():
    try:
        root = tk.Tk(useTk=False)
        root.withdraw()
    except tk.TclError:
        pytest.skip("Tk is unavailable in this environment")

    host_app = SimpleNamespace(root=root)
    config = ConfigManager(host_app)

    fake = SimpleNamespace(
        config=config,
        root=root,
        stage_status_labels={"setup": object(), "webcam": object(), "pose_clustering": object()},
        roi_manager=SimpleNamespace(rois={}),
        detailed_bouts_df=pd.DataFrame(),
        summary_bouts_df=pd.DataFrame(),
        _last_cluster_report_path=None,
        _log_level_counts={"DEBUG": 0, "INFO": 0, "WARNING": 0, "ERROR": 0, "CRITICAL": 0},
        _embedded_behavior_analysis_app=None,
    )

    statuses = {}

    def capture(self, key, state, detail=None):
        statuses[key] = (state, detail)

    fake._set_stage_status = MethodType(capture, fake)
    fake._schedule_stage_status_refresh = MethodType(lambda self, delay_ms=250: None, fake)
    fake._stage_path_state = lambda raw_path, *, kind: YoloApp._stage_path_state(raw_path, kind=kind)
    fake._stage_has_nondefault_text = lambda value, *, default="": YoloApp._stage_has_nondefault_text(value, default=default)
    fake._stage_has_tab7_groups = MethodType(YoloApp._stage_has_tab7_groups, fake)

    return fake, statuses, root


def test_fresh_launch_keeps_setup_webcam_and_tab7_neutral():
    fake, statuses, root = _make_status_app()
    try:
        YoloApp._refresh_stage_status_strip(fake)

        assert statuses["setup"][0] == "not_started"
        assert statuses["webcam"][0] == "not_started"
        assert statuses["pose_clustering"][0] == "not_started"
    finally:
        root.destroy()


def test_tab7_warning_requires_real_imported_groups_not_default_keypoints():
    fake, statuses, root = _make_status_app()
    try:
        fake._embedded_behavior_analysis_app = SimpleNamespace(groups={"Imported": {"sources": []}})

        YoloApp._refresh_stage_status_strip(fake)

        assert statuses["pose_clustering"][0] == "warning"
    finally:
        root.destroy()
