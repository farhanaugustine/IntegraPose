from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from integra_pose.gui.controllers.webcam_controller import (
    _create_unique_run_dir,
    _write_webcam_label_file,
)
from integra_pose.utils.detection_contract import enforce_ultralytics_max_det
from integra_pose.utils.frame_identity import (
    frame_label_filename,
    load_frame_label_manifest,
    write_frame_label_manifest,
)
from integra_pose.utils.yolo_pose_labels import (
    load_pose_label_schema,
    parse_yolo_pose_label,
)


class _Boxes:
    def __init__(self, *, xywhn, xyxy, confidence, classes, track_ids) -> None:
        self.xywhn = np.asarray(xywhn, dtype=float)
        self.xyxy = np.asarray(xyxy, dtype=float)
        self.conf = np.asarray(confidence, dtype=float)
        self.cls = np.asarray(classes, dtype=float)
        self.id = None if track_ids is None else np.asarray(track_ids, dtype=float)

    def __len__(self) -> int:
        return int(self.xywhn.shape[0])

    def subset(self, indices) -> "_Boxes":
        return _Boxes(
            xywhn=self.xywhn[indices],
            xyxy=self.xyxy[indices],
            confidence=self.conf[indices],
            classes=self.cls[indices],
            track_ids=None if self.id is None else self.id[indices],
        )


class _Keypoints:
    def __init__(self, *, xyn, confidence) -> None:
        self.xyn = np.asarray(xyn, dtype=float)
        self.xy = self.xyn * np.asarray([640.0, 480.0])
        self.conf = None if confidence is None else np.asarray(confidence, dtype=float)
        if self.conf is None:
            self.data = self.xy
        else:
            self.data = np.concatenate([self.xy, self.conf[..., None]], axis=2)

    def subset(self, indices) -> "_Keypoints":
        return _Keypoints(
            xyn=self.xyn[indices],
            confidence=None if self.conf is None else self.conf[indices],
        )


class _Result:
    def __init__(self, boxes: _Boxes, keypoints: _Keypoints | None = None) -> None:
        self.boxes = boxes
        self.keypoints = keypoints

    def __getitem__(self, indices) -> "_Result":
        selected = np.asarray(indices, dtype=int)
        return _Result(
            self.boxes.subset(selected),
            None if self.keypoints is None else self.keypoints.subset(selected),
        )


def _pose_result() -> _Result:
    return _Result(
        _Boxes(
            xywhn=[
                [0.2, 0.2, 0.1, 0.1],
                [0.5, 0.5, 0.2, 0.3],
                [0.8, 0.8, 0.1, 0.2],
            ],
            xyxy=[
                [96, 72, 160, 120],
                [256, 168, 384, 312],
                [480, 336, 544, 432],
            ],
            confidence=[0.2, 0.95, 0.8],
            classes=[0, 1, 2],
            track_ids=[10, 11, 12],
        ),
        _Keypoints(
            xyn=[
                [[0.1, 0.1], [0.2, 0.2]],
                [[0.4, 0.4], [0.6, 0.6]],
                [[0.7, 0.7], [0.9, 0.9]],
            ],
            confidence=[
                [0.5, 0.6],
                [0.91, 0.72],
                [0.8, 0.7],
            ],
        ),
    )


def test_webcam_run_directory_is_incremented_instead_of_reused(tmp_path: Path) -> None:
    first = _create_unique_run_dir(tmp_path, "camera_trial")
    (first / "labels").mkdir()
    (first / "labels" / "stale.txt").write_text("old", encoding="utf-8")

    second = _create_unique_run_dir(tmp_path, "camera_trial")

    assert first == tmp_path / "camera_trial"
    assert second == tmp_path / "camera_trial_1"
    assert not (second / "labels" / "stale.txt").exists()


def test_capped_webcam_pose_label_preserves_selected_track_and_keypoints(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    write_frame_label_manifest(labels_dir, source="webcam_1", max_det=1)

    outcome = enforce_ultralytics_max_det(_pose_result(), 1)
    label_path = _write_webcam_label_file(
        outcome.result,
        labels_dir=labels_dir,
        source_name="webcam_1",
        frame_index=0,
        frame_width=640,
        frame_height=480,
    )

    assert outcome.original_count == 3
    assert outcome.retained_count == 1
    assert label_path.name == frame_label_filename("webcam_1", 0)
    assert len(label_path.read_text(encoding="utf-8").splitlines()) == 1

    schema = load_pose_label_schema(labels_dir, expected_keypoint_count=2)
    assert schema is not None
    parsed = parse_yolo_pose_label(
        label_path.read_text(encoding="utf-8"),
        keypoint_count=2,
        schema=schema,
    )
    assert parsed.class_id == 1
    assert parsed.track_id == 11
    assert parsed.bbox_confidence == pytest.approx(0.95)
    assert np.asarray(parsed.keypoints) == pytest.approx(
        np.asarray(((0.4, 0.4, 0.91), (0.6, 0.6, 0.72)))
    )
    assert load_frame_label_manifest(labels_dir)["max_det"] == 1


def test_webcam_detect_label_keeps_confidence_and_track_id(tmp_path: Path) -> None:
    result = _Result(
        _Boxes(
            xywhn=[[0.5, 0.5, 0.25, 0.4]],
            xyxy=[[240, 144, 400, 336]],
            confidence=[0.87],
            classes=[3],
            track_ids=[42],
        )
    )

    label_path = _write_webcam_label_file(
        result,
        labels_dir=tmp_path,
        source_name="webcam",
        frame_index=7,
        frame_width=640,
        frame_height=480,
    )

    assert label_path.name == "webcam_frame_000007.txt"
    assert label_path.read_text(encoding="utf-8").split() == [
        "3",
        "0.500000",
        "0.500000",
        "0.250000",
        "0.400000",
        "0.870000",
        "42",
    ]


def test_webcam_writes_canonical_empty_label_for_no_detection_frame(tmp_path: Path) -> None:
    result = _Result(
        _Boxes(
            xywhn=np.empty((0, 4)),
            xyxy=np.empty((0, 4)),
            confidence=np.empty((0,)),
            classes=np.empty((0,)),
            track_ids=None,
        )
    )

    label_path = _write_webcam_label_file(
        result,
        labels_dir=tmp_path,
        source_name="camera_2",
        frame_index=0,
        frame_width=640,
        frame_height=480,
    )

    assert label_path.name == "camera_2_frame_000000.txt"
    assert label_path.read_text(encoding="utf-8") == ""
