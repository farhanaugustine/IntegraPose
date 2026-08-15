from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from integra_pose.hmm_vae_toolkit.main import (
    BehaviorAnalysisApp,
    compute_feature_vectors,
)


KEYPOINT_NAMES = ["left", "right"]


def _row(
    *,
    directory: str,
    frame: int,
    track_id: int,
    left_xy: tuple[float, float],
    right_xy: tuple[float, float],
    confidence: float = 0.95,
    video_size: tuple[int, int] | None = (1000, 1000),
) -> dict:
    width, height = video_size if video_size is not None else (None, None)
    return {
        "group": "control",
        "directory": directory,
        "frame": frame,
        "track_id": track_id,
        "keypoints": {
            "left": (*left_xy, confidence),
            "right": (*right_xy, confidence),
        },
        "bbox": None,
        "video_width": width,
        "video_height": height,
    }


def _compute(rows: list[dict], **kwargs):
    return compute_feature_vectors(
        pd.DataFrame(rows),
        KEYPOINT_NAMES,
        kwargs.pop("conf_threshold", 0.8),
        normalization_ref_points=("left", "right"),
        **kwargs,
    )


def test_social_distances_use_finite_off_diagonal_neighbors() -> None:
    rows = [
        _row(
            directory="video_a",
            frame=0,
            track_id=0,
            left_xy=(0.0, 0.0),
            right_xy=(10.0, 0.0),
        ),
        _row(
            directory="video_a",
            frame=0,
            track_id=1,
            left_xy=(30.0, 0.0),
            right_xy=(40.0, 0.0),
        ),
    ]

    result, diagnostics = _compute(rows, social_mode=True)

    # With two animals, the off-diagonal mean and nearest distance are both
    # 30 px. The source body reference is 10 px, so both features equal 3.
    for vector in result["feature_vector"]:
        assert np.all(np.isfinite(vector))
        assert vector[5] == pytest.approx(3.0)
        assert vector[6] == pytest.approx(3.0)
    assert diagnostics["nonfinite_repaired"] == 0
    assert diagnostics["social_missing_centroids"] == 0


def test_configured_confidence_threshold_is_applied_without_lowering() -> None:
    rows = [
        _row(
            directory="video_a",
            frame=0,
            track_id=0,
            left_xy=(0.0, 0.0),
            right_xy=(10.0, 0.0),
            confidence=0.95,
        ),
        _row(
            directory="video_a",
            frame=1,
            track_id=0,
            left_xy=(100.0, 0.0),
            right_xy=(110.0, 0.0),
            confidence=0.50,
        ),
    ]

    result, diagnostics = _compute(rows, conf_threshold=0.8)
    low_conf_vector = result.loc[result["frame"] == 1, "feature_vector"].iloc[0]

    assert diagnostics["requested_confidence_threshold"] == pytest.approx(0.8)
    assert diagnostics["applied_confidence_threshold"] == pytest.approx(0.8)
    assert diagnostics["confidence_threshold_policy"] == "configured"
    assert diagnostics["threshold_lowered"] is False
    assert diagnostics["low_confidence_frames"] == 1
    # Pair distance uses the high-confidence source median; position and
    # temporal features are explicit missing-keypoint defaults.
    assert low_conf_vector[0] == pytest.approx(1.0)
    assert low_conf_vector[1:] == pytest.approx([0.0] * 8)


def test_temporal_features_use_frame_deltas_for_missing_frames() -> None:
    rows = [
        _row(
            directory="video_a",
            frame=0,
            track_id=0,
            left_xy=(0.0, 0.0),
            right_xy=(10.0, 0.0),
        ),
        _row(
            directory="video_a",
            frame=2,
            track_id=0,
            left_xy=(4.0, 0.0),
            right_xy=(14.0, 0.0),
        ),
        _row(
            directory="video_a",
            frame=3,
            track_id=0,
            left_xy=(6.0, 0.0),
            right_xy=(16.0, 0.0),
        ),
    ]

    result, diagnostics = _compute(rows, temporal_max_frame_gap=3)
    frame_two = result.loc[result["frame"] == 2, "feature_vector"].iloc[0]
    frame_three = result.loc[result["frame"] == 3, "feature_vector"].iloc[0]

    # Four pixels over two source frames is 2 px/frame, or 0.2 body
    # lengths/frame. Constant speed across the unequal intervals has zero
    # acceleration.
    assert frame_two[5:9] == pytest.approx([0.2, 0.0, 0.2, 0.0])
    assert frame_three[5:9] == pytest.approx([0.2, 0.0, 0.2, 0.0])
    assert diagnostics["temporal_irregular_intervals"] == 1
    assert diagnostics["temporal_gap_resets"] == 0


def test_temporal_features_reset_when_gap_exceeds_configured_limit() -> None:
    rows = [
        _row(
            directory="video_a",
            frame=0,
            track_id=0,
            left_xy=(0.0, 0.0),
            right_xy=(10.0, 0.0),
        ),
        _row(
            directory="video_a",
            frame=10,
            track_id=0,
            left_xy=(100.0, 0.0),
            right_xy=(110.0, 0.0),
        ),
    ]

    result, diagnostics = _compute(rows, temporal_max_frame_gap=3)
    after_gap = result.loc[result["frame"] == 10, "feature_vector"].iloc[0]

    assert after_gap[5:9] == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert diagnostics["temporal_gap_resets"] == 1
    assert any("Temporal history reset" in warning for warning in diagnostics["warnings"])


def test_pair_distances_are_source_normalized_across_coordinate_spaces() -> None:
    rows = [
        _row(
            directory="pixel_video",
            frame=0,
            track_id=0,
            left_xy=(0.0, 0.0),
            right_xy=(10.0, 0.0),
            video_size=(1000, 1000),
        ),
        _row(
            directory="normalized_video",
            frame=0,
            track_id=0,
            left_xy=(0.0, 0.0),
            right_xy=(0.1, 0.0),
            video_size=None,
        ),
    ]

    result, diagnostics = _compute(rows)

    assert [vector[0] for vector in result["feature_vector"]] == pytest.approx([1.0, 1.0])
    assert diagnostics["coordinate_spaces"] == {"pixel": 1, "normalized": 1}


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_invalid_confidence_threshold_is_rejected(threshold: float) -> None:
    row = _row(
        directory="video_a",
        frame=0,
        track_id=0,
        left_xy=(0.0, 0.0),
        right_xy=(10.0, 0.0),
    )

    with pytest.raises(ValueError, match="between 0 and 1"):
        _compute([row], conf_threshold=threshold)


def test_feature_diagnostics_are_saved_with_run_outputs(tmp_path) -> None:
    diagnostics = {
        "applied_confidence_threshold": 0.8,
        "distance_normalization": "source_body_reference",
        "temporal_max_frame_gap": 3,
    }
    clustered = pd.DataFrame(
        [{"group": "control", "frame": 0, "track_id": 0, "cluster_label": 1}]
    )

    saved = BehaviorAnalysisApp._save_sub_behavior_outputs(
        object(),
        output_folder=str(tmp_path),
        clustered_df=clustered,
        bouts=[],
        multi_result=SimpleNamespace(classes=[]),
        class_names={},
        feature_diagnostics=diagnostics,
        run_id="run-123",
    )

    diagnostics_path = tmp_path / "sub_behavior_feature_diagnostics.json"
    assert saved["feature_diagnostics_json"] == str(diagnostics_path)
    payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert payload == {"run_id": "run-123", **diagnostics}
