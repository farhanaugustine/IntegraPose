import json

import numpy as np
import pandas as pd
import pytest

from integra_pose.logic.canonical_metrics import compile_metrics_from_labels


def _pose_labels() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "frame": 0,
                "track_id": 1,
                "class_id": 0,
                "bbox_conf": 0.9,
                "x_center_n": 0.9,
                "y_center_n": 0.9,
                "width_n": 0.2,
                "height_n": 0.2,
                "kp_Nose_x_n": 0.3,
                "kp_Nose_y_n": 0.5,
                "kp_Thorax_x_n": 0.2,
                "kp_Thorax_y_n": 0.5,
                "kp_TailBase_x_n": 0.1,
                "kp_TailBase_y_n": 0.5,
            },
            {
                "frame": 2,
                "track_id": 1,
                "class_id": 0,
                "bbox_conf": 0.9,
                "x_center_n": 0.8,
                "y_center_n": 0.8,
                "width_n": 0.2,
                "height_n": 0.2,
                "kp_Nose_x_n": 0.5,
                "kp_Nose_y_n": 0.5,
                "kp_Thorax_x_n": 0.4,
                "kp_Thorax_y_n": 0.5,
                "kp_TailBase_x_n": 0.3,
                "kp_TailBase_y_n": 0.5,
            },
        ]
    )


def test_canonical_metrics_use_thorax_anchor_and_frame_delta(tmp_path):
    labels_path = tmp_path / "labels.csv"
    _pose_labels().to_csv(labels_path, index=False)

    result = compile_metrics_from_labels(
        labels_csv_path=labels_path,
        run_dir=tmp_path,
        frame_width=100,
        frame_height=100,
        fps=30.0,
        heading_indices=(2, 0),
        anchor_keypoint_index=1,
        keypoint_names=["Nose", "Thorax", "TailBase"],
    )

    metrics = pd.read_csv(result.metrics_csv)
    assert metrics["anchor_x_px"].tolist() == pytest.approx([20.0, 40.0])
    assert metrics["anchor_y_px"].tolist() == pytest.approx([50.0, 50.0])
    assert set(metrics["anchor_source"]) == {"keypoint_1"}
    assert metrics["orientation_deg"].tolist() == pytest.approx([90.0, 90.0])
    assert set(metrics["orientation_source"]) == {"keypoints_2_to_0"}
    assert metrics.loc[1, "movement_speed_px_per_frame"] == pytest.approx(10.0)
    assert metrics.loc[1, "acceleration_px_per_frame2"] == pytest.approx(5.0)
    assert metrics.loc[1, "total_path_length_px"] == pytest.approx(20.0)
    assert metrics["body_aspect_ratio"].isna().all()

    metadata = json.loads(result.metadata_json.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 2
    assert metadata["speed_semantics"] == "pixel_displacement_divided_by_frame_delta"
    assert metadata["orientation_fallback"] == "nan_when_heading_unavailable"


def test_canonical_metrics_do_not_invent_radial_orientation(tmp_path):
    labels_path = tmp_path / "labels.csv"
    labels = _pose_labels().iloc[[0]].copy()
    labels.to_csv(labels_path, index=False)

    result = compile_metrics_from_labels(
        labels_csv_path=labels_path,
        run_dir=tmp_path,
        frame_width=100,
        frame_height=100,
        keypoint_names=["Nose", "Thorax", "TailBase"],
    )

    metrics = pd.read_csv(result.metrics_csv)
    assert np.isnan(metrics.loc[0, "orientation_deg"])
    assert metrics.loc[0, "orientation_source"] == "unavailable"
    assert metrics.loc[0, "anchor_source"] == "bbox_center"


def test_canonical_metrics_reject_invalid_heading_indices(tmp_path):
    labels_path = tmp_path / "labels.csv"
    _pose_labels().to_csv(labels_path, index=False)

    with pytest.raises(ValueError, match="Heading indices"):
        compile_metrics_from_labels(
            labels_csv_path=labels_path,
            run_dir=tmp_path,
            frame_width=100,
            frame_height=100,
            heading_indices=(2, 9),
            keypoint_names=["Nose", "Thorax", "TailBase"],
        )
