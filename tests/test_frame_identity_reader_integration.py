from __future__ import annotations

from pathlib import Path

import pytest

from integra_pose.hmm_vae_toolkit.data_processing_hdbscan import read_detections
from integra_pose.plugins.plugin_eda.core.bout_analysis_utils import (
    analyze_detections_per_track,
)
from integra_pose.plugins.plugin_eda.core.data_handler import DataHandler
from integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.data_loader import (
    load_yolo_data,
)
from integra_pose.utils.frame_identity import (
    FrameIdentityError,
    write_frame_label_manifest,
)
from integra_pose.utils.video_creator import (
    _build_detection_schedule,
    _load_labels_csv_track_map,
)
from integra_pose.utils.yolo_pose_labels import (
    YoloPoseLabelSchema,
    write_pose_label_schema,
)

_POSE_ROW = "0 0.5 0.5 0.2 0.2 0.4 0.6 0.9\n"


def _write_pose_run(labels_dir: Path, source_stem: str, filenames: list[str]) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    write_frame_label_manifest(labels_dir, source=f"{source_stem}.mp4", max_det=1)
    write_pose_label_schema(
        labels_dir,
        YoloPoseLabelSchema(
            keypoint_count=1,
            keypoint_dimensions=3,
            include_track_id=False,
        ),
    )
    for filename in filenames:
        (labels_dir / filename).write_text(_POSE_ROW, encoding="utf-8")


def _reader_frame_sets(labels_dir: Path, source_stem: str) -> dict[str, list[int]]:
    video_path = labels_dir.parent / f"{source_stem}.mp4"
    schedule = _build_detection_schedule(str(labels_dir), frame_count=20)

    hmm_df, _keypoints, _behaviors = read_detections(
        {"control": [str(labels_dir)]},
        "nose",
        "rest",
        video_path_map={str(labels_dir): str(video_path)},
    )
    gait_df = load_yolo_data(
        str(labels_dir),
        video_width=100,
        video_height=80,
        keypoint_order=["nose"],
        conf_threshold=0.0,
        total_frames=20,
        video_base_name=source_stem,
    )
    eda = DataHandler()
    assert eda.load_inference_data(str(labels_dir), "nose", True)

    return {
        "video": [frame for frame, _path in schedule],
        "hmm": sorted(hmm_df["frame"].astype(int).tolist()),
        "gait": sorted(gait_df["frame"].astype(int).tolist()),
        "eda": sorted(eda.df_raw["frame_id"].astype(int).tolist()),
    }


def test_canonical_frame_names_are_identical_across_readers(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    source_stem = "trial_2026_07"
    _write_pose_run(
        labels_dir,
        source_stem,
        [
            f"{source_stem}_frame_000000.txt",
            f"{source_stem}_frame_000001.txt",
        ],
    )
    (labels_dir / "notes.txt").write_text("auxiliary file\n", encoding="utf-8")
    (labels_dir / f"{source_stem}0_frame_000000.txt").write_text(
        _POSE_ROW,
        encoding="utf-8",
    )

    assert _reader_frame_sets(labels_dir, source_stem) == {
        "video": [0, 1],
        "hmm": [0, 1],
        "gait": [0, 1],
        "eda": [0, 1],
    }


def test_legacy_numeric_source_base_maps_to_frame_zero_across_readers(tmp_path: Path) -> None:
    labels_dir = tmp_path / "legacy_labels"
    source_stem = "mouse_1"
    _write_pose_run(
        labels_dir,
        source_stem,
        [f"{source_stem}.txt", f"{source_stem}_000001.txt"],
    )

    assert _reader_frame_sets(labels_dir, source_stem) == {
        "video": [0, 1],
        "hmm": [0, 1],
        "gait": [0, 1],
        "eda": [0, 1],
    }


def test_manifest_prevents_one_based_guess_for_sparse_labels_csv(tmp_path: Path) -> None:
    write_frame_label_manifest(tmp_path, source="trial.mp4", max_det=1)
    (tmp_path / "labels.csv").write_text(
        "frame,track_id\n1,10\n2,11\n",
        encoding="utf-8",
    )

    assert _load_labels_csv_track_map(str(tmp_path), frame_count=2) == {
        1: [10],
        2: [11],
    }


def test_duplicate_frame_aliases_fail_in_every_directory_reader(tmp_path: Path) -> None:
    labels_dir = tmp_path / "duplicate_labels"
    source_stem = "trial"
    _write_pose_run(
        labels_dir,
        source_stem,
        ["trial_frame_000001.txt", "trial_img_000001.txt"],
    )

    with pytest.raises(FrameIdentityError, match="same frame"):
        _build_detection_schedule(str(labels_dir), frame_count=10)

    with pytest.raises(FrameIdentityError, match="same frame"):
        read_detections(
            {"control": [str(labels_dir)]},
            "nose",
            "rest",
            video_path_map={str(labels_dir): str(tmp_path / "trial.mp4")},
        )

    with pytest.raises(FrameIdentityError, match="same frame"):
        load_yolo_data(
            str(labels_dir),
            100,
            80,
            ["nose"],
            0.0,
            10,
            source_stem,
        )

    with pytest.raises(FrameIdentityError, match="same frame"):
        analyze_detections_per_track(
            str(labels_dir),
            {0: "rest"},
            str(tmp_path / "eda_output"),
            source_stem,
        )

    eda = DataHandler()
    assert not eda.load_inference_data(str(labels_dir), "nose", True)
    assert "same frame" in str(eda.last_error).lower()
