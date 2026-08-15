from __future__ import annotations

import numpy as np
import pandas as pd
from types import SimpleNamespace

from integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.analysis import (
    calculate_original_gait_metrics,
    calculate_pose_metrics,
    smooth_behavior,
)
from integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.decision_dynamics_analysis import (
    extract_transition_windows,
)
from integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.compare_ccm import find_longest_bout
from integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.compare_gait import summarize_metric_per_video
from integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.main import run


POSE_CONFIG = {
    "POSE_METRICS": {
        "ELONGATION_CONNECTION": ["Nose", "Tail"],
        "BODY_ANGLE_CONNECTION": ["Neck", "Nose"],
    }
}


def _pose_rows(frames: list[int]) -> pd.DataFrame:
    values = np.asarray([0.0, 1.0, 101.0, 102.0])[: len(frames)]
    return pd.DataFrame(
        {
            "track_id": 1,
            "frame": frames,
            "center_x": values,
            "center_y": 0.0,
            "Nose_x": values + 1.0,
            "Nose_y": 0.0,
            "Tail_x": values - 1.0,
            "Tail_y": 0.0,
            "Neck_x": values,
            "Neck_y": 0.0,
        }
    )


def test_pose_derivatives_do_not_bridge_dropped_frames_and_use_fps() -> None:
    result = calculate_pose_metrics(_pose_rows([0, 1, 3, 4]), POSE_CONFIG, fps=30.0)

    assert np.isnan(result.loc[0, "speed_px_per_frame"])
    assert result.loc[1, "speed_px_per_frame"] == 1.0
    assert np.isnan(result.loc[2, "speed_px_per_frame"])
    assert result.loc[3, "speed_px_per_s"] == 30.0
    assert np.isnan(result.loc[2, "turning_speed_deg_per_s"])


def test_behavior_bouts_are_split_at_unobserved_frames() -> None:
    data = pd.DataFrame(
        {
            "track_id": [1, 1, 1],
            "frame": [0, 1, 3],
            "behavior_name": ["Walking", "Walking", "Walking"],
        }
    )

    smoothed = smooth_behavior(data, min_bout_duration=3)

    assert smoothed.tolist() == ["Unknown", "Unknown", "Unknown"]


def test_smoothing_never_bridges_a_short_bout_across_a_frame_gap() -> None:
    data = pd.DataFrame(
        {
            "track_id": [1] * 7,
            "frame": [0, 1, 2, 5, 6, 7, 8],
            "behavior_name": ["Walking", "Walking", "Walking", "Grooming", "Walking", "Walking", "Walking"],
        }
    )

    smoothed = smooth_behavior(data, min_bout_duration=2)

    assert smoothed.loc[3] == "Unknown"


def test_stride_duration_is_frame_interval_count_and_speed_is_per_second() -> None:
    events = pd.DataFrame(
        [
            {"track_id": 1, "frame": 10, "paw": "Reference Paw", "event": "foot_strike", "x": 0.0, "y": 0.0},
            {"track_id": 1, "frame": 16, "paw": "Reference Paw", "event": "foot_strike", "x": 12.0, "y": 0.0},
        ]
    )
    full = pd.DataFrame(
        {
            "track_id": 1,
            "frame": np.arange(10, 17),
            "speed_px_per_frame": 2.0,
            "speed_px_per_s": 60.0,
            "Reference Paw_x": np.arange(7, dtype=float) * 2.0,
            "Reference Paw_y": 0.0,
            "Reference Paw_speed_px_per_frame": 2.0,
        }
    )
    config = {"GAIT_ANALYSIS": {"STRIDE_REFERENCE_PAW": "Reference Paw", "GAIT_PAWS": ["Reference Paw"]}}

    result = calculate_original_gait_metrics(events, full, config, fps=30.0)

    assert len(result) == 1
    assert result.loc[0, "stride_duration_frames"] == 6
    assert result.loc[0, "stride_duration_s"] == 0.2
    assert result.loc[0, "stride_speed"] == 60.0
    assert result.loc[0, "stride_speed_px_per_frame"] == 2.0


def test_stride_with_missing_internal_frame_is_rejected() -> None:
    events = pd.DataFrame(
        [
            {"track_id": 1, "frame": 10, "paw": "Reference Paw", "event": "foot_strike", "x": 0.0, "y": 0.0},
            {"track_id": 1, "frame": 16, "paw": "Reference Paw", "event": "foot_strike", "x": 12.0, "y": 0.0},
        ]
    )
    full = pd.DataFrame(
        {
            "track_id": 1,
            "frame": [10, 11, 12, 14, 15, 16],
            "speed_px_per_frame": 2.0,
            "speed_px_per_s": 60.0,
            "Reference Paw_x": [0.0, 2.0, 4.0, 8.0, 10.0, 12.0],
            "Reference Paw_y": 0.0,
            "Reference Paw_speed_px_per_frame": 2.0,
        }
    )
    config = {"GAIT_ANALYSIS": {"STRIDE_REFERENCE_PAW": "Reference Paw", "GAIT_PAWS": ["Reference Paw"]}}

    result = calculate_original_gait_metrics(events, full, config, fps=30.0)

    assert result.empty


def test_transition_window_stays_within_track_and_requires_every_frame() -> None:
    data = pd.DataFrame(
        {
            "video_source": ["v1"] * 7 + ["v2"] * 5,
            "track_id": [1] * 12,
            "frame": [0, 1, 2, 3, 4, 5, 6, 0, 1, 3, 4, 5],
            "smoothed_behavior": ["Walking", "Walking", "Walking", "Grooming", "Grooming", "Grooming", "Grooming"]
            + ["Walking", "Walking", "Grooming", "Grooming", "Grooming"],
            "previous_behavior": [None, "Walking", "Walking", "Walking", "Grooming", "Grooming", "Grooming"]
            + [None, "Walking", "Walking", "Grooming", "Grooming"],
        }
    )

    windows = extract_transition_windows(data, "Walking", "Grooming", 2, 2)

    assert windows["video_source"].unique().tolist() == ["v1"]
    assert windows["frame"].tolist() == [1, 2, 3, 4, 5]
    assert windows["time_to_transition"].tolist() == [-2, -1, 0, 1, 2]


def test_ccm_bout_cannot_span_tracks_or_missing_frames() -> None:
    data = pd.DataFrame(
        {
            "track_id": [1, 1, 1, 1, 2, 2, 2],
            "frame": [0, 1, 4, 5, 0, 1, 2],
            "smoothed_behavior": ["Walking"] * 7,
        }
    )

    bout = find_longest_bout(data, "Walking", min_duration=2)

    assert bout is not None
    assert bout["track_id"].unique().tolist() == [2]
    assert bout["frame"].tolist() == [0, 1, 2]


def test_group_comparison_uses_one_mean_per_video_not_one_row_per_stride() -> None:
    strides = pd.DataFrame(
        {
            "group": ["A", "A", "A", "A"],
            "video_source": ["many", "many", "many", "few"],
            "stride_speed": [0.0, 0.0, 0.0, 10.0],
        }
    )

    per_video = summarize_metric_per_video(strides, "stride_speed")

    assert per_video.set_index("video_source")["stride_speed"].to_dict() == {
        "few": 10.0,
        "many": 0.0,
    }
    assert per_video["stride_speed"].mean() == 5.0


def test_single_video_run_reports_missing_labels_as_failure(tmp_path) -> None:
    args = SimpleNamespace(
        output_dir=str(tmp_path / "output"),
        yolo_dir=str(tmp_path / "missing-labels"),
        video_path=str(tmp_path / "video.mp4"),
    )

    result = run(args, {})

    assert result.failed
    assert "labels directory not found" in result.message.lower()
