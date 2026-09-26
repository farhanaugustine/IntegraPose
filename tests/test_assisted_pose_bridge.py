"""Controller integration with synthetic frames and model calls prohibited."""
from __future__ import annotations

import os
from pathlib import Path
import tkinter as tk

import cv2
import numpy as np
import pytest

from integra_pose.plugins.plugin_assisted_pose_curation import qt_bridge


@pytest.fixture(scope="module")
def tk_root():
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def bridge(tk_root, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    def no_model(*args, **kwargs):
        raise AssertionError("Model execution is prohibited in these tests")
    for name in ("_get_pose_model", "_get_embed_model", "_schedule_assist_for_current_image"):
        monkeypatch.setattr(qt_bridge._Controller, name, no_model)
    monkeypatch.setattr(qt_bridge._Controller, "_load_prefill_from_main_app", lambda *a, **k: None)
    controller = qt_bridge._Controller(None, parent=tk_root)
    controller._assist_enabled_var.set(False)
    controller._memory_assist_enabled_var.set(False)
    images = tmp_path / "images_all"
    images.mkdir(exist_ok=True)
    for index in range(2):
        assert cv2.imwrite(str(images / f"frame_{index}.png"), np.zeros((80, 120, 3), dtype=np.uint8))
    controller.apply_prefill(project_root=str(tmp_path), image_dir=str(images), label_dir=str(tmp_path / "labels_all"), keypoint_names=["nose", "tail"], skeleton_edges=[(0, 1)], auto_load_images=True)
    window = object.__new__(qt_bridge.AssistedPoseCurationWindow)
    window._controller, window._closing = controller, False
    yield window
    controller.destroy()


def test_controller_edit_save_navigate_and_reload_without_model(bridge):
    c = bridge._controller
    first = c._current_path
    bridge._dispatch(dict(command="edit", data=dict(path=str(first), instance=0, point=0, x=40, y=30, visibility=2)))
    assert c._dirty
    bridge._dispatch(dict(command="save"))
    assert not c._dirty
    assert (Path(c._label_dir_var.get()) / (first.stem + ".txt")).is_file()
    bridge._dispatch(dict(command="next"))
    assert c._current_path != first
    bridge._dispatch(dict(command="previous"))
    assert c._current_path == first
    assert c._current_pose[0].x == pytest.approx(40, abs=0.01)
    assert c._current_pose[0].y == pytest.approx(30, abs=0.01)
    snapshot = bridge._snapshot()
    assert snapshot["instances"][0]["pose"][0]["v"] == 2
    assert "reviewed" in snapshot["review"]


def test_stale_frame_edits_and_nonfinite_coordinates_are_rejected(bridge):
    c = bridge._controller
    with pytest.raises(ValueError, match="frame changed"):
        bridge._dispatch(dict(command="edit", data=dict(path="old.png", instance=0, point=0, x=20, y=30)))
    with pytest.raises(ValueError, match="finite"):
        bridge._dispatch(dict(command="edit", data=dict(path=str(c._current_path), instance=0, point=0, x=float("nan"), y=30)))
    assert not c._dirty
    assert c._current_pose[0].x is None


def test_settings_followup_and_snapshot(bridge):
    c = bridge._controller
    bridge._dispatch(dict(command="settings", data=dict(values={"assist_enabled": False, "assist_conf": "0.45"}, then="load")))
    assert len(c._image_paths) == 2
    assert bridge._snapshot()["settings"]["assist_conf"] == "0.45"
