from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from integra_pose.gui.services.project_io_service import ProjectIOService
from integra_pose.logic.yolo_parser import YoloParser
from integra_pose.utils.frame_identity import FrameIdentityError
from integra_pose.utils.roi_manager import ROIManager
from integra_pose.utils.yolo_pose_labels import (
    YoloPoseLabelSchema,
    format_yolo_pose_label,
    write_pose_label_schema,
)


class _Var:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _ParserApp:
    def __init__(self, labels_dir, source_video=""):
        self.logs = []
        settings = {
            "analytics.yolo_output_path_var": str(labels_dir),
            "analytics.source_video_path_var": str(source_video),
        }
        self.config = SimpleNamespace(get_setting=settings.get)

    def log_message(self, message, level):
        self.logs.append((level, message))


def _pose_row(*, track_id=None, bbox_confidence=None):
    return format_yolo_pose_label(
        class_id=0,
        bbox=(0.5, 0.5, 0.2, 0.2),
        keypoints=((0.1, 0.2, 2.0), (0.3, 0.4, 1.0)),
        bbox_confidence=bbox_confidence,
        track_id=track_id,
    )


def test_clustering_frame_lookup_handles_numeric_legacy_source_and_ignores_auxiliary_txt(tmp_path):
    (tmp_path / "20260710.txt").write_text(_pose_row(), encoding="utf-8")
    (tmp_path / "20260710_000001.txt").write_text(_pose_row(), encoding="utf-8")
    (tmp_path / "classes.txt").write_text("walking\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("review later\n", encoding="utf-8")

    lookup = YoloParser.resolve_frame_files(tmp_path, source="20260710.mp4")

    assert lookup == {0: "20260710.txt", 1: "20260710_000001.txt"}


def test_clustering_frame_lookup_rejects_duplicate_frame_aliases(tmp_path):
    (tmp_path / "mouse.txt").write_text(_pose_row(), encoding="utf-8")
    (tmp_path / "mouse_frame_000000.txt").write_text(_pose_row(), encoding="utf-8")

    with pytest.raises(FrameIdentityError, match="same frame"):
        YoloParser.resolve_frame_files(tmp_path, source="mouse.mp4")


def test_pose_parser_uses_schema_for_confidence_track_and_keypoint_visibility(tmp_path):
    schema = YoloPoseLabelSchema(
        keypoint_count=2,
        keypoint_dimensions=3,
        include_bbox_confidence=True,
        include_track_id=True,
    )
    write_pose_label_schema(tmp_path, schema)
    (tmp_path / "trial_frame_000000.txt").write_text(
        _pose_row(track_id=7, bbox_confidence=0.83),
        encoding="utf-8",
    )
    app = _ParserApp(tmp_path, source_video="trial.mp4")
    parser = YoloParser(app)

    is_valid, keypoint_count, error = parser.validate_yolo_files(
        tmp_path,
        source="trial.mp4",
        expected_keypoints=2,
    )
    pose, detected_count = parser.get_pose_from_frame(
        0,
        track_id=7,
        expected_keypoints=2,
    )

    assert (is_valid, keypoint_count, error) == (True, 2, "")
    assert detected_count == 2
    np.testing.assert_allclose(pose, [0.1, 0.2, 0.3, 0.4])


def test_pose_parser_does_not_treat_final_visibility_as_track_id(tmp_path):
    schema = YoloPoseLabelSchema(
        keypoint_count=2,
        keypoint_dimensions=3,
        include_bbox_confidence=False,
        include_track_id=True,
    )
    write_pose_label_schema(tmp_path, schema)
    (tmp_path / "trial_frame_000000.txt").write_text(_pose_row(), encoding="utf-8")
    parser = YoloParser(_ParserApp(tmp_path, source_video="trial.mp4"))

    pose, detected_count = parser.get_pose_from_frame(0, expected_keypoints=2)
    tracked_pose, _ = parser.get_pose_from_frame(0, track_id=1, expected_keypoints=2)

    assert detected_count == 2
    assert pose is not None
    assert tracked_pose is None


def test_roi_reader_preserves_zero_based_frames_for_numeric_legacy_source(tmp_path):
    label = "0 0.500000 0.500000 0.100000 0.100000\n"
    (tmp_path / "20260710.txt").write_text(label, encoding="utf-8")
    (tmp_path / "20260710_000001.txt").write_text(label, encoding="utf-8")
    (tmp_path / "classes.txt").write_text("walking\n", encoding="utf-8")

    manager = ROIManager()
    manager.add_roi("arena", [(0, 0), (99, 0), (99, 99), (0, 99)])
    manager.process_yolo_path(
        str(tmp_path),
        {0: "walking"},
        100,
        100,
        max_frame_gap=2,
        min_bout_duration_frames=1,
    )

    assert manager.get_analytics() == [
        {
            "ROI Name": "arena",
            "Animal ID": 0,
            "Behavior": "walking",
            "Bout Start Frame": 0,
            "Bout End Frame": 2,
        }
    ]


def test_roi_reader_uses_pose_schema_instead_of_visibility_as_track_id(tmp_path):
    schema = YoloPoseLabelSchema(
        keypoint_count=2,
        keypoint_dimensions=3,
        include_bbox_confidence=False,
        include_track_id=True,
    )
    write_pose_label_schema(tmp_path, schema)
    (tmp_path / "trial_frame_000000.txt").write_text(_pose_row(), encoding="utf-8")

    manager = ROIManager()
    manager.add_roi("arena", [(0, 0), (99, 0), (99, 99), (0, 99)])
    manager.process_yolo_path(
        str(tmp_path),
        {0: "walking"},
        100,
        100,
        max_frame_gap=2,
        min_bout_duration_frames=1,
        source_video="trial.mp4",
    )

    assert manager.get_analytics()[0]["Animal ID"] == 0


def test_roi_reader_rejects_duplicate_untracked_detections(tmp_path):
    schema = YoloPoseLabelSchema(
        keypoint_count=2,
        keypoint_dimensions=3,
        include_bbox_confidence=False,
        include_track_id=True,
    )
    write_pose_label_schema(tmp_path, schema)
    row = _pose_row()
    (tmp_path / "trial_frame_000000.txt").write_text(
        f"{row}\n{row}\n",
        encoding="utf-8",
    )

    manager = ROIManager()
    manager.add_roi("arena", [(0, 0), (99, 0), (99, 99), (0, 99)])
    with pytest.raises(ValueError, match="refusing to overwrite"):
        manager.process_yolo_path(
            str(tmp_path),
            {0: "walking"},
            100,
            100,
            max_frame_gap=2,
            min_bout_duration_frames=1,
            source_video="trial.mp4",
        )


def test_yolo_folder_picker_rejects_auxiliary_txt_only(tmp_path):
    (tmp_path / "classes.txt").write_text("walking\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not detections\n", encoding="utf-8")
    output_var = _Var()
    app = SimpleNamespace(
        root=object(),
        logs=[],
        config=SimpleNamespace(
            analytics=SimpleNamespace(
                source_video_path_var=_Var("trial.mp4"),
                yolo_output_path_var=output_var,
            )
        ),
    )
    app.log_message = lambda message, level: app.logs.append((level, message))

    with (
        patch(
            "integra_pose.gui.services.project_io_service.filedialog.askdirectory",
            return_value=str(tmp_path),
        ),
        patch(
            "integra_pose.gui.services.project_io_service.messagebox.showwarning"
        ) as warning,
    ):
        ProjectIOService(app).select_yolo_output_directory()

    assert output_var.get() == ""
    warning.assert_called_once()
    assert "no frame-indexed detection labels" in warning.call_args.args[1].lower()
