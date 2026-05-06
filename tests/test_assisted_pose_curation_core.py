from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from integra_pose.plugins.plugin_assisted_pose_curation.core import (
    ACTIVE_LEARNING_CONTEXT_SCALE,
    ActiveLearningCandidate,
    PosePoint,
    PROVENANCE_ASSIST_ACCEPTED,
    PROVENANCE_ASSIST_CORRECTED,
    PROVENANCE_COPIED_FORWARD,
    PROVENANCE_MANUAL,
    PROVENANCE_PENDING_REVIEW,
    VISIBILITY_OCCLUDED,
    VISIBILITY_VISIBLE,
    blend_memory_pose_into_assist_pose,
    decode_pose_label_line,
    derive_review_provenance,
    encode_pose_label_line,
    normalize_pose_to_bbox,
    parse_keypoint_names,
    pose_from_normalized_bbox_points,
    pose_bbox_xyxy,
    scale_bbox_xyxy,
    select_active_learning_candidates,
    serialize_pose_points,
    sanitize_skeleton_edges,
    summarize_curation_manifest_records,
    write_active_learning_candidates_csv,
    write_dataset_yaml,
)


def test_parse_keypoint_names_supports_lines_and_commas():
    raw = "Nose, Left Ear\nRight Ear\r\nBase of Tail"
    assert parse_keypoint_names(raw) == ["Nose", "Left Ear", "Right Ear", "Base of Tail"]


def test_sanitize_skeleton_edges_filters_invalid_edges():
    names = ["Nose", "Tail", "Paw"]
    raw = [(0, 1), ("Tail", "Paw"), ("Tail", "Tail"), (4, 0), ("Missing", "Paw")]
    assert sanitize_skeleton_edges(raw, names) == [(0, 1), (1, 2)]


def test_pose_encode_decode_roundtrip():
    keypoint_names = ["Nose", "Tail"]
    points = [
        PosePoint(name="Nose", x=10.0, y=20.0, v=VISIBILITY_VISIBLE, conf=0.8),
        PosePoint(name="Tail", x=30.0, y=40.0, v=VISIBILITY_OCCLUDED, conf=0.4),
    ]
    line = encode_pose_label_line(0, points, 100, 100, keypoint_names)
    class_id, decoded, bbox = decode_pose_label_line(line, keypoint_names, 100, 100)

    assert class_id == 0
    assert bbox[2] > bbox[0]
    assert bbox[3] > bbox[1]
    assert round(decoded[0].x or 0.0, 2) == 10.0
    assert round(decoded[0].y or 0.0, 2) == 20.0
    assert decoded[0].v == VISIBILITY_VISIBLE
    assert round(decoded[1].x or 0.0, 2) == 30.0
    assert round(decoded[1].y or 0.0, 2) == 40.0
    assert decoded[1].v == VISIBILITY_OCCLUDED


def test_pose_bbox_xyxy_returns_box_for_visible_points():
    points = [
        PosePoint(name="A", x=50.0, y=60.0, v=VISIBILITY_VISIBLE),
        PosePoint(name="B", x=90.0, y=120.0, v=VISIBILITY_VISIBLE),
    ]
    bbox = pose_bbox_xyxy(points, 200, 200, padding_ratio=0.1)
    assert bbox is not None
    x1, y1, x2, y2 = bbox
    assert x1 < 50
    assert y1 < 60
    assert x2 > 90
    assert y2 > 120


def test_write_dataset_yaml_includes_keypoint_metadata(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    train_dir = dataset_root / "images" / "train"
    val_dir = dataset_root / "images" / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    out_path = tmp_path / "dataset.yaml"

    result = write_dataset_yaml(
        out_path,
        dataset_root=dataset_root,
        train_images_path=train_dir,
        val_images_path=val_dir,
        class_name="mouse",
        keypoint_names=["Nose", "Tail"],
        skeleton_edges=[(0, 1)],
    )

    text = result.read_text(encoding="utf-8")
    assert "# - Nose" in text
    assert "# - [0, 1]" in text
    assert "kpt_shape:" in text


def test_derive_review_provenance_maps_assist_and_copy_states():
    assert derive_review_provenance(source_origin="assist", manual_edits=False, has_labels=True) == PROVENANCE_ASSIST_ACCEPTED
    assert derive_review_provenance(source_origin="assist", manual_edits=True, has_labels=True) == PROVENANCE_ASSIST_CORRECTED
    assert derive_review_provenance(source_origin="copied_forward", manual_edits=False, has_labels=True) == PROVENANCE_COPIED_FORWARD
    assert derive_review_provenance(source_origin="manual", manual_edits=True, has_labels=True) == PROVENANCE_MANUAL
    assert derive_review_provenance(source_origin="assist", manual_edits=False, has_labels=False) is None


def test_summarize_curation_manifest_records_counts_review_states():
    records = {
        "frame_001.jpg": {"provenance": PROVENANCE_ASSIST_ACCEPTED, "has_saved_labels": True},
        "frame_002.jpg": {"provenance": PROVENANCE_ASSIST_CORRECTED, "has_saved_labels": True},
        "frame_003.jpg": {"provenance": PROVENANCE_COPIED_FORWARD, "has_saved_labels": True},
        "frame_004.jpg": {"provenance": PROVENANCE_MANUAL, "has_saved_labels": True},
        "frame_005.jpg": {"provenance": PROVENANCE_PENDING_REVIEW, "has_saved_labels": False},
    }

    summary = summarize_curation_manifest_records(
        records,
        expected_frame_keys=[
            "frame_001.jpg",
            "frame_002.jpg",
            "frame_003.jpg",
            "frame_004.jpg",
            "frame_005.jpg",
            "frame_006.jpg",
        ],
    )

    assert summary["total_frames"] == 6
    assert summary["reviewed_frames"] == 4
    assert summary["pending_review_frames"] == 1
    assert summary["labeled_frames"] == 4
    assert summary[PROVENANCE_ASSIST_ACCEPTED] == 1
    assert summary[PROVENANCE_ASSIST_CORRECTED] == 1
    assert summary[PROVENANCE_COPIED_FORWARD] == 1
    assert summary[PROVENANCE_MANUAL] == 1


def test_scale_bbox_xyxy_doubles_bbox_size_for_active_learning_context():
    bbox = (40, 50, 80, 90)
    scaled = scale_bbox_xyxy(bbox, 200, 200, scale=ACTIVE_LEARNING_CONTEXT_SCALE)
    assert scaled is not None
    x1, y1, x2, y2 = scaled
    assert (x2 - x1) >= 80
    assert (y2 - y1) >= 80


def test_normalize_pose_to_bbox_roundtrip():
    keypoint_names = ["Nose", "Tail"]
    bbox = (10, 20, 110, 220)
    points = [
        PosePoint(name="Nose", x=20.0, y=40.0, v=VISIBILITY_VISIBLE, conf=0.8),
        PosePoint(name="Tail", x=60.0, y=180.0, v=VISIBILITY_OCCLUDED, conf=0.4),
    ]
    normalized = normalize_pose_to_bbox(points, bbox)
    recovered = pose_from_normalized_bbox_points(normalized, bbox, keypoint_names)
    assert round(recovered[0].x or 0.0, 3) == 20.0
    assert round(recovered[0].y or 0.0, 3) == 40.0
    assert recovered[1].v == VISIBILITY_OCCLUDED
    assert round(recovered[1].x or 0.0, 3) == 60.0


def test_serialize_pose_points_preserves_visible_and_missing_points():
    points = [
        PosePoint(name="A", x=10.5, y=12.5, v=VISIBILITY_VISIBLE, conf=0.7),
        PosePoint(name="B", x=None, y=None, v=0, conf=0.0),
    ]
    payload = serialize_pose_points(points)
    assert payload[0]["name"] == "A"
    assert payload[0]["x"] == 10.5
    assert payload[1]["x"] is None


def test_select_active_learning_candidates_respects_temporal_gap_when_possible():
    candidates = [
        ActiveLearningCandidate(frame_index=0, uncertainty_score=0.9, novelty_score=0.8, transition_score=0.4),
        ActiveLearningCandidate(frame_index=5, uncertainty_score=0.95, novelty_score=0.85, transition_score=0.5),
        ActiveLearningCandidate(frame_index=30, uncertainty_score=0.7, novelty_score=0.7, transition_score=0.4),
        ActiveLearningCandidate(frame_index=60, uncertainty_score=0.6, novelty_score=0.9, transition_score=0.4),
    ]
    selected = select_active_learning_candidates(candidates, target_count=3, min_frame_gap=20, reference_embeddings=[])
    assert len(selected) == 3
    frame_indices = [candidate.frame_index for candidate in selected]
    assert 5 in frame_indices
    assert 30 in frame_indices
    assert 60 in frame_indices


def test_select_active_learning_candidates_applies_gap_per_video_not_globally():
    candidates = [
        ActiveLearningCandidate(frame_index=5, source_video_path="video_a.mp4", uncertainty_score=0.95, novelty_score=0.85, transition_score=0.5),
        ActiveLearningCandidate(frame_index=6, source_video_path="video_b.mp4", uncertainty_score=0.94, novelty_score=0.84, transition_score=0.5),
        ActiveLearningCandidate(frame_index=30, source_video_path="video_a.mp4", uncertainty_score=0.70, novelty_score=0.70, transition_score=0.4),
    ]
    selected = select_active_learning_candidates(candidates, target_count=2, min_frame_gap=20, reference_embeddings=[])
    assert len(selected) == 2
    assert {candidate.source_video_path for candidate in selected} == {"video_a.mp4", "video_b.mp4"}


def test_select_active_learning_candidates_records_nearest_selected_metadata():
    candidates = [
        ActiveLearningCandidate(
            frame_index=0,
            image_name="video_a__frame_000000.jpg",
            source_video_path="video_a.mp4",
            uncertainty_score=0.95,
            novelty_score=0.80,
            transition_score=0.50,
            embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        ),
        ActiveLearningCandidate(
            frame_index=1,
            image_name="video_a__frame_000001.jpg",
            source_video_path="video_a.mp4",
            uncertainty_score=0.30,
            novelty_score=0.10,
            transition_score=0.10,
            embedding=np.asarray([0.95, 0.05], dtype=np.float32),
        ),
        ActiveLearningCandidate(
            frame_index=30,
            image_name="video_b__frame_000030.jpg",
            source_video_path="video_b.mp4",
            uncertainty_score=0.92,
            novelty_score=0.78,
            transition_score=0.45,
            embedding=np.asarray([0.0, 1.0], dtype=np.float32),
        ),
    ]
    selected = select_active_learning_candidates(candidates, target_count=2, min_frame_gap=0, reference_embeddings=[])
    assert len(selected) == 2
    middle = candidates[1]
    assert middle.nearest_selected_image != ""
    assert middle.nearest_selected_source_video != ""
    assert middle.nearest_selected_distance is not None
    assert middle.nearest_selected_distance < 0.2


def test_write_active_learning_candidates_csv_includes_source_video_columns(tmp_path: Path):
    out_path = tmp_path / "al_candidates.csv"
    write_active_learning_candidates_csv(
        out_path,
        [
            ActiveLearningCandidate(
                frame_index=12,
                source_video_path="mouse_trial_A.mp4",
                image_name="mouse_trial_A__frame_000012.jpg",
                nearest_reference_distance=0.123,
                nearest_reference_image="reviewed_mouse_trial_A__frame_000001.jpg",
                nearest_reference_source_video="reviewed_session.mp4",
                nearest_selected_distance=0.234,
                nearest_selected_image="mouse_trial_B__frame_000020.jpg",
                nearest_selected_source_video="mouse_trial_B.mp4",
                selected=True,
            )
        ],
        run_metadata={
            "run_id": "al_20260325T120000Z_deadbeef",
            "run_timestamp": "2026-03-25T12:00:00+00:00",
            "app_version": "1.2.3",
            "assist_model_path": "models/assist.pt",
            "assist_model_sha256": "abc123",
            "score_weight_uncertainty": 0.45,
            "score_weight_transition": 0.25,
            "score_weight_diversity": 0.30,
            "stride": 10,
            "target_count": 50,
            "min_frame_gap": 15,
            "context_crop_scale": 2.0,
            "assist_conf_threshold": 0.25,
            "reviewed_reference_count": 12,
        },
    )
    text = out_path.read_text(encoding="utf-8")
    assert "run_id" in text
    assert "assist_model_sha256" in text
    assert "score_weight_uncertainty" in text
    assert "source_video_path" in text
    assert "source_video_name" in text
    assert "nearest_reference_distance" in text
    assert "nearest_selected_distance" in text
    assert "al_20260325T120000Z_deadbeef" in text
    assert "models/assist.pt" in text
    assert "mouse_trial_A.mp4" in text
    assert "reviewed_mouse_trial_A__frame_000001.jpg" in text
    assert "mouse_trial_B__frame_000020.jpg" in text


def test_write_active_learning_candidates_csv_appends_runs_without_rewriting_header(tmp_path: Path):
    out_path = tmp_path / "al_candidates_log.csv"
    write_active_learning_candidates_csv(
        out_path,
        [
            ActiveLearningCandidate(
                frame_index=1,
                source_video_path="video_a.mp4",
                image_name="video_a__frame_000001.jpg",
                selected=True,
            )
        ],
        run_metadata={"run_id": "run_one"},
    )
    write_active_learning_candidates_csv(
        out_path,
        [
            ActiveLearningCandidate(
                frame_index=2,
                source_video_path="video_b.mp4",
                image_name="video_b__frame_000002.jpg",
                selected=True,
            )
        ],
        run_metadata={"run_id": "run_two"},
    )

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("run_id,")
    assert sum(1 for line in lines if line.startswith("run_id,")) == 1
    assert any("run_one" in line for line in lines[1:])
    assert any("run_two" in line for line in lines[1:])


def test_memory_blend_preserves_high_confidence_anchor_points():
    keypoint_names = ["Nose", "Left Front Paw", "Base of Tail"]
    assist_pose = [
        PosePoint(name="Nose", x=20.0, y=20.0, v=VISIBILITY_VISIBLE, conf=0.95),
        PosePoint(name="Left Front Paw", x=50.0, y=50.0, v=VISIBILITY_VISIBLE, conf=0.70),
        PosePoint(name="Base of Tail", x=80.0, y=80.0, v=VISIBILITY_VISIBLE, conf=0.90),
    ]
    memory_pose = [
        {"name": "Nose", "x": 0.40, "y": 0.40, "v": VISIBILITY_VISIBLE, "conf": 1.0},
        {"name": "Left Front Paw", "x": 0.62, "y": 0.62, "v": VISIBILITY_VISIBLE, "conf": 1.0},
        {"name": "Base of Tail", "x": 0.60, "y": 0.60, "v": VISIBILITY_VISIBLE, "conf": 1.0},
    ]
    blended, meta = blend_memory_pose_into_assist_pose(
        assist_pose,
        (0, 0, 100, 100),
        memory_normalized_pose=memory_pose,
        keypoint_names=keypoint_names,
        similarity=0.98,
    )

    assert meta["memory_refined"] is True
    assert blended[0].x == 20.0
    assert blended[0].y == 20.0
    assert blended[2].x == 80.0
    assert blended[2].y == 80.0
    assert blended[1].x is not None and blended[1].x > 50.0
    assert blended[1].y is not None and blended[1].y > 50.0


def test_memory_blend_only_fills_missing_points_at_very_high_similarity():
    keypoint_names = ["Nose", "Tail"]
    assist_pose = [
        PosePoint(name="Nose", x=10.0, y=10.0, v=VISIBILITY_VISIBLE, conf=0.8),
        PosePoint(name="Tail"),
    ]
    memory_pose = [
        {"name": "Nose", "x": 0.10, "y": 0.10, "v": VISIBILITY_VISIBLE, "conf": 1.0},
        {"name": "Tail", "x": 0.80, "y": 0.80, "v": VISIBILITY_VISIBLE, "conf": 1.0},
    ]
    low_sim_blend, low_meta = blend_memory_pose_into_assist_pose(
        assist_pose,
        (0, 0, 100, 100),
        memory_normalized_pose=memory_pose,
        keypoint_names=keypoint_names,
        similarity=0.91,
    )
    high_sim_blend, high_meta = blend_memory_pose_into_assist_pose(
        assist_pose,
        (0, 0, 100, 100),
        memory_normalized_pose=memory_pose,
        keypoint_names=keypoint_names,
        similarity=0.96,
    )

    assert low_meta["memory_filled_points"] == 0
    assert low_sim_blend[1].x is None
    assert high_meta["memory_filled_points"] == 1
    assert round(high_sim_blend[1].x or 0.0, 1) == 80.0
    assert round(high_sim_blend[1].y or 0.0, 1) == 80.0
