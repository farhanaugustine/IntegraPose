from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from integra_pose.logic.analytics import Analytics
from integra_pose.utils.frame_identity import write_frame_label_manifest


class _Config:
    def __init__(self, setup_names):
        self._setup_names = setup_names

    def get_setting(self, setting_path):
        if setting_path == "setup.behaviors_list":
            return self._setup_names
        return None


def test_core_analytics_prefers_saved_label_metadata(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    write_frame_label_manifest(
        labels_dir,
        source="video.mp4",
        max_det=1,
        class_names={0: "Sniffing", 1: "Wall-Rearing", 2: "Ambulatory"},
        class_names_source="model.names",
        model_task="detect",
    )
    app = SimpleNamespace(
        config=_Config([{"id": 0, "name": "WrongSetupName"}])
    )
    params = {
        "yolo_folder": str(labels_dir),
        "behavior_names_override": ["WrongSelectedModelName"],
        "behavior_names_source": "model class names",
    }

    resolved = Analytics(app)._resolve_behavior_name_map(params)

    assert resolved == {
        0: "Sniffing",
        1: "Wall-Rearing",
        2: "Ambulatory",
    }
    assert params["behavior_names_source"] == "inference label metadata"


def test_core_analytics_keeps_legacy_setup_fallback(tmp_path: Path) -> None:
    labels_dir = tmp_path / "legacy_labels"
    write_frame_label_manifest(labels_dir, source="video.mp4", max_det=1)
    app = SimpleNamespace(
        config=_Config(
            [
                {"id": 0, "name": "Sniffing"},
                {"id": 1, "name": "Wall-Rearing"},
            ]
        )
    )

    resolved = Analytics(app)._resolve_behavior_name_map(
        {"yolo_folder": str(labels_dir)}
    )

    assert resolved == {0: "Sniffing", 1: "Wall-Rearing"}
