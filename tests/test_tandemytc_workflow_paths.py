import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from integra_pose.plugins.plugin_tandem_yolo_toolkit.yolo_temporal_classifier.annotation_app.workflow_panels import (
    TrainPanel,
    _local_path_text,
)


def test_local_path_text_converts_windows_file_uri() -> None:
    assert _local_path_text("file:///C:/data/mars/source_manifest.csv") == "C:\\data\\mars\\source_manifest.csv"


def test_local_path_text_keeps_non_file_sources() -> None:
    assert _local_path_text("0") == "0"
    assert _local_path_text("rtsp://camera.local/live") == "rtsp://camera.local/live"
    assert _local_path_text("E:\\models\\best.pt") == "E:\\models\\best.pt"


def _qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_train_preflight_flags_source_manifest_csv(tmp_path: Path) -> None:
    _qapp()
    csv_path = tmp_path / "source_manifest.csv"
    csv_path.write_text("split,video_path,annotation_path\ntrain,video.mp4,video.annot\n", encoding="utf-8")

    panel = TrainPanel()
    panel.manifest_path.setText(str(csv_path))

    summary = panel._temporal_preflight_summary()

    assert summary["valid"] is False
    assert summary["status"] == "Invalid training manifest"
    assert "Do not use source_manifest.csv" in summary["message"]


def test_train_preflight_accepts_prepare_json(tmp_path: Path) -> None:
    _qapp()
    manifest_path = tmp_path / "sequence_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "meta": {"schema_version": "yolo-temporal-full-video-v1", "window_size": 32, "window_stride": 16},
                "class_to_idx": {"attack": 0, "other": 1},
                "splits": {"train": [{}, {}], "val": [{}]},
            }
        ),
        encoding="utf-8",
    )

    panel = TrainPanel()
    panel.manifest_path.setText(str(manifest_path))

    summary = panel._temporal_preflight_summary()

    assert summary["valid"] is True
    assert "window 32" in summary["status"]
    assert "Samples: 3" in summary["message"]
