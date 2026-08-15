from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from integra_pose.utils.yolo_pose_labels import (
    YoloPoseLabelSchema,
    load_pose_label_schema,
    parse_yolo_pose_label,
)


class _ArrayValue:
    def __init__(self, values) -> None:
        self._values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _Boxes:
    def __init__(self, *, xywhn, classes, confidence, track_ids) -> None:
        self.xywhn = _ArrayValue(xywhn)
        self.cls = _ArrayValue(classes)
        self.conf = _ArrayValue(confidence)
        self.id = _ArrayValue(track_ids) if track_ids is not None else None

    def __len__(self) -> int:
        return len(self.xywhn._values)


class _Keypoints:
    def __init__(self, *, xyn, confidence) -> None:
        self.xyn = _ArrayValue(xyn)
        self.conf = _ArrayValue(confidence) if confidence is not None else None


def _emit_core_label(
    labels_dir: Path,
    *,
    keypoints,
    keypoint_confidence,
    bbox_confidence: float,
    track_id: int,
    save_confidence: bool,
) -> Path:
    from integra_pose.logic.supervision_runner import SupervisionInferenceRunner

    runner = object.__new__(SupervisionInferenceRunner)
    runner._labels_dir = labels_dir
    runner._labels_csv_recorder = None
    runner._labels_error_logged = False
    runner._tracker_label_warning_emitted = False
    runner._pose_label_schema = None
    runner._pose_label_schema_warning_emitted = False
    runner.settings = SimpleNamespace(
        single_animal_mode=False,
        source_path=labels_dir / "trial.mp4",
        use_tracker=True,
        save_conf=save_confidence,
    )
    runner.log = lambda *_args, **_kwargs: None

    boxes = _Boxes(
        xywhn=[[0.5, 0.4, 0.2, 0.3]],
        classes=[2],
        confidence=[bbox_confidence],
        track_ids=[track_id],
    )
    result = SimpleNamespace(
        boxes=boxes,
        keypoints=_Keypoints(xyn=[keypoints], confidence=[keypoint_confidence] if keypoint_confidence is not None else None),
        path=str(labels_dir / "trial_frame_000004.jpg"),
    )
    detections = SimpleNamespace(tracker_id=np.asarray([track_id]))

    runner._write_labels_file(result, detections, frame_index=4)
    return labels_dir / "trial_frame_000004.txt"


def test_core_writer_never_uses_an_unindexed_first_frame_for_numeric_video_stem(tmp_path: Path) -> None:
    from integra_pose.logic.supervision_runner import SupervisionInferenceRunner

    runner = object.__new__(SupervisionInferenceRunner)
    runner._labels_dir = tmp_path
    runner._labels_csv_recorder = None
    runner._labels_error_logged = False
    runner._tracker_label_warning_emitted = False
    runner._pose_label_schema = None
    runner._pose_label_schema_warning_emitted = False
    runner.settings = SimpleNamespace(
        single_animal_mode=False,
        source_path=tmp_path / "mouse_1.mp4",
        use_tracker=True,
        save_conf=False,
        max_det=1,
    )
    runner.log = lambda *_args, **_kwargs: None

    boxes = _Boxes(
        xywhn=[[0.5, 0.4, 0.2, 0.3]],
        classes=[0],
        confidence=[0.9],
        track_ids=[4],
    )
    result = SimpleNamespace(
        boxes=boxes,
        keypoints=None,
        path=str(tmp_path / "mouse_1.mp4"),
    )
    detections = SimpleNamespace(tracker_id=np.asarray([4]))

    runner._write_labels_file(result, detections, frame_index=0)
    runner._write_labels_file(result, detections, frame_index=1)

    assert sorted(path.name for path in tmp_path.glob("*.txt")) == [
        "mouse_1_frame_000000.txt",
        "mouse_1_frame_000001.txt",
    ]


def test_core_3d_output_round_trips_through_tab7_and_gait(tmp_path: Path) -> None:
    label_path = _emit_core_label(
        tmp_path,
        keypoints=[[0.1, 0.2], [0.3, 0.4]],
        keypoint_confidence=[0.91, 0.72],
        bbox_confidence=0.87,
        track_id=7,
        save_confidence=True,
    )

    schema = load_pose_label_schema(tmp_path, expected_keypoint_count=2)
    assert schema == YoloPoseLabelSchema(
        keypoint_count=2,
        keypoint_dimensions=3,
        include_bbox=True,
        include_bbox_confidence=True,
        include_track_id=True,
    )
    parsed = parse_yolo_pose_label(label_path.read_text(encoding="utf-8"), keypoint_count=2, schema=schema)
    assert np.asarray(parsed.keypoints) == pytest.approx(
        np.asarray(((0.1, 0.2, 0.91), (0.3, 0.4, 0.72)))
    )
    assert parsed.bbox_confidence == pytest.approx(0.87)
    assert parsed.track_id == 7

    from integra_pose.hmm_vae_toolkit.data_processing_hdbscan import read_detections

    tab7_df, keypoint_names, _ = read_detections(
        {"control": [str(tmp_path)]},
        "nose,tail",
        "idle,walk,run",
        use_bbox_hint=True,
    )
    assert keypoint_names == ["nose", "tail"]
    assert len(tab7_df) == 1
    tab7_row = tab7_df.iloc[0]
    assert tab7_row["track_id"] == 7
    assert tab7_row["bbox_conf"] == pytest.approx(0.87)
    assert tab7_row["keypoints"]["nose"] == pytest.approx((0.1, 0.2, 0.91))
    assert tab7_row["keypoints"]["tail"] == pytest.approx((0.3, 0.4, 0.72))

    from integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.data_loader import load_yolo_data

    gait_df = load_yolo_data(
        txt_dir=str(tmp_path),
        video_width=100,
        video_height=50,
        keypoint_order=["nose", "tail"],
        conf_threshold=0.5,
        total_frames=10,
        video_base_name="trial_frame",
    )
    assert len(gait_df) == 1
    gait_row = gait_df.iloc[0]
    assert gait_row["track_id"] == 7
    assert gait_row["confidence"] == pytest.approx(0.87)
    assert gait_row["nose_x"] == pytest.approx(10.0)
    assert gait_row["nose_y"] == pytest.approx(10.0)
    assert gait_row["nose_conf"] == pytest.approx(0.91)
    assert gait_row["tail_x"] == pytest.approx(30.0)
    assert gait_row["tail_y"] == pytest.approx(20.0)
    assert gait_row["tail_conf"] == pytest.approx(0.72)


def test_core_2d_output_keeps_track_suffix_out_of_keypoints(tmp_path: Path) -> None:
    label_path = _emit_core_label(
        tmp_path,
        keypoints=[[0.1, 0.2], [0.0, 0.0]],
        keypoint_confidence=None,
        bbox_confidence=0.94,
        track_id=0,
        save_confidence=False,
    )

    schema = load_pose_label_schema(tmp_path, expected_keypoint_count=2)
    assert schema is not None
    assert schema.keypoint_dimensions == 2
    assert not schema.include_bbox_confidence
    parsed = parse_yolo_pose_label(label_path.read_text(encoding="utf-8"), keypoint_count=2, schema=schema)
    assert parsed.keypoints == ((0.1, 0.2), (0.0, 0.0))
    assert parsed.keypoints_xyc() == ((0.1, 0.2, 1.0), (0.0, 0.0, 0.0))
    assert parsed.bbox_confidence is None
    assert parsed.track_id == 0

    from integra_pose.hmm_vae_toolkit.data_processing_hdbscan import read_detections

    tab7_df, _, _ = read_detections(
        {"control": [str(tmp_path)]},
        "nose,tail",
        "idle,walk,run",
        use_bbox_hint=True,
    )
    tab7_row = tab7_df.iloc[0]
    assert tab7_row["track_id"] == 0
    assert tab7_row["keypoints"]["nose"] == (0.1, 0.2, 1.0)
    assert tab7_row["keypoints"]["tail"] == (0.0, 0.0, 0.0)

    from integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.data_loader import load_yolo_data

    gait_df = load_yolo_data(
        txt_dir=str(tmp_path),
        video_width=100,
        video_height=50,
        keypoint_order=["nose", "tail"],
        conf_threshold=0.5,
        total_frames=10,
        video_base_name="trial_frame",
    )
    gait_row = gait_df.iloc[0]
    assert gait_row["track_id"] == 0
    assert gait_row["nose_x"] == pytest.approx(10.0)
    assert gait_row["nose_conf"] == pytest.approx(1.0)
    assert np.isnan(gait_row["tail_x"])
    assert gait_row["tail_conf"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("line", "keypoint_count", "expected_dimensions", "expected_bbox_confidence", "expected_track_id"),
    [
        (
            "0 0.5 0.5 0.2 0.2 0.1 0.2 0.9 0.3 0.4 0.8",
            2,
            3,
            None,
            None,
        ),
        (
            "0 0.5 0.5 0.2 0.2 0.1 0.2 0.9 0.3 0.4 0.8 0.75 12",
            2,
            3,
            0.75,
            12,
        ),
        (
            "0 0.5 0.5 0.2 0.2 0.1 0.2 0.9 0.3 0.4 0.8 0.75",
            2,
            3,
            0.75,
            None,
        ),
        (
            "0 0.5 0.5 0.2 0.2 0.1 0.2 0.9 0.3 0.4 0.8 12",
            2,
            3,
            None,
            12,
        ),
        (
            "0 0.5 0.5 0.2 0.2 0.1 0.2 0.3 0.4",
            2,
            2,
            None,
            None,
        ),
        (
            "0 0.5 0.5 0.2 0.2 0.1 0.2 0.9",
            1,
            3,
            None,
            None,
        ),
    ],
)
def test_standard_ultralytics_rows_remain_supported(
    line: str,
    keypoint_count: int,
    expected_dimensions: int,
    expected_bbox_confidence: float | None,
    expected_track_id: int | None,
) -> None:
    parsed = parse_yolo_pose_label(line, keypoint_count=keypoint_count)
    assert parsed.keypoint_dimensions == expected_dimensions
    assert parsed.bbox_confidence == expected_bbox_confidence
    assert parsed.track_id == expected_track_id


def test_explicit_kpt_shape_disambiguates_standard_2d_track_zero() -> None:
    line = "0 0.5 0.5 0.2 0.2 0.1 0.2 0"
    parsed = parse_yolo_pose_label(
        line,
        keypoint_count=1,
        keypoint_dimensions=2,
        include_bbox=True,
        include_bbox_confidence=False,
        include_track_id=True,
    )
    assert parsed.keypoints == ((0.1, 0.2),)
    assert parsed.track_id == 0


def test_schema_disambiguates_integer_confidence_from_track_id() -> None:
    line = "0 0.5 0.5 0.2 0.2 0.1 0.2 0.9 1"
    confidence_schema = YoloPoseLabelSchema(
        keypoint_count=1,
        keypoint_dimensions=3,
        include_bbox_confidence=True,
    )
    tracked_schema = YoloPoseLabelSchema(
        keypoint_count=1,
        keypoint_dimensions=3,
        include_bbox_confidence=False,
    )

    confidence_label = parse_yolo_pose_label(line, keypoint_count=1, schema=confidence_schema)
    tracked_label = parse_yolo_pose_label(line, keypoint_count=1, schema=tracked_schema)

    assert confidence_label.bbox_confidence == 1.0
    assert confidence_label.track_id is None
    assert tracked_label.bbox_confidence is None
    assert tracked_label.track_id == 1
