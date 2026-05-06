import numpy as np
import pytest

sv = pytest.importorskip("supervision")

from integra_pose.logic.supervision_runner import (
    LightweightHeadingArrowAnnotator,
    LightweightSkeletonAnnotator,
    LightweightVertexAnnotator,
    LightweightTraceAnnotator,
    SupervisionInferenceRunner,
)


def test_lightweight_trace_draws_after_multiple_frames() -> None:
    annotator = LightweightTraceAnnotator(thickness=2, trace_length=20)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    detections_a = sv.Detections(
        xyxy=np.array([[20.0, 20.0, 40.0, 40.0]], dtype=np.float32),
        tracker_id=np.array([7], dtype=int),
    )
    detections_b = sv.Detections(
        xyxy=np.array([[60.0, 60.0, 80.0, 80.0]], dtype=np.float32),
        tracker_id=np.array([7], dtype=int),
    )

    first = annotator.annotate(frame.copy(), detections=detections_a)
    second = annotator.annotate(frame.copy(), detections=detections_b)

    assert np.array_equal(first, frame)
    assert not np.array_equal(second, frame)


def test_lightweight_trace_accepts_hex_color() -> None:
    annotator = LightweightTraceAnnotator(thickness=2, trace_length=20, color_name="#38A3FF")
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    detections_a = sv.Detections(
        xyxy=np.array([[10.0, 10.0, 20.0, 20.0]], dtype=np.float32),
        tracker_id=np.array([1], dtype=int),
    )
    detections_b = sv.Detections(
        xyxy=np.array([[40.0, 40.0, 50.0, 50.0]], dtype=np.float32),
        tracker_id=np.array([1], dtype=int),
    )

    annotator.annotate(frame.copy(), detections=detections_a)
    rendered = annotator.annotate(frame.copy(), detections=detections_b)

    assert not np.array_equal(rendered, frame)


def test_lightweight_trace_can_persist_until_reset() -> None:
    annotator = LightweightTraceAnnotator(thickness=2, trace_length=2, persistent=True)
    frame = np.zeros((160, 160, 3), dtype=np.uint8)

    for idx, coord in enumerate((20.0, 50.0, 80.0, 110.0), start=1):
        detections = sv.Detections(
            xyxy=np.array([[coord, coord, coord + 10.0, coord + 10.0]], dtype=np.float32),
            tracker_id=np.array([3], dtype=int),
        )
        annotator.annotate(frame.copy(), detections=detections)

    history = annotator._history.get(3)
    assert history is not None
    assert len(history) == 4


def test_lightweight_trace_survives_tracker_id_churn() -> None:
    annotator = LightweightTraceAnnotator(thickness=2, trace_length=20)
    frame = np.zeros((160, 160, 3), dtype=np.uint8)

    detections_a = sv.Detections(
        xyxy=np.array([[20.0, 20.0, 30.0, 30.0]], dtype=np.float32),
        tracker_id=np.array([11], dtype=int),
    )
    detections_b = sv.Detections(
        xyxy=np.array([[28.0, 28.0, 38.0, 38.0]], dtype=np.float32),
        tracker_id=np.array([72], dtype=int),
    )

    annotator.annotate(frame.copy(), detections=detections_a)
    rendered = annotator.annotate(frame.copy(), detections=detections_b)

    assert not np.array_equal(rendered, frame)


def test_lightweight_trace_opacity_zero_leaves_frame_unchanged() -> None:
    annotator = LightweightTraceAnnotator(thickness=2, trace_length=20, opacity=0.0)
    frame = np.zeros((160, 160, 3), dtype=np.uint8)
    detections_a = sv.Detections(
        xyxy=np.array([[20.0, 20.0, 30.0, 30.0]], dtype=np.float32),
        tracker_id=np.array([1], dtype=int),
    )
    detections_b = sv.Detections(
        xyxy=np.array([[40.0, 40.0, 50.0, 50.0]], dtype=np.float32),
        tracker_id=np.array([1], dtype=int),
    )

    annotator.annotate(frame.copy(), detections=detections_a)
    rendered = annotator.annotate(frame.copy(), detections=detections_b)

    assert np.array_equal(rendered, frame)


def test_lightweight_heading_arrow_draws_for_selected_keypoints() -> None:
    annotator = LightweightHeadingArrowAnnotator(thickness=2)
    frame = np.zeros((180, 180, 3), dtype=np.uint8)
    key_points = sv.KeyPoints(
        xy=np.array(
            [[[30.0, 40.0], [90.0, 40.0]]],
            dtype=np.float32,
        )
    )

    rendered = annotator.annotate(frame.copy(), key_points=key_points, heading_indices=(0, 1))

    assert not np.array_equal(rendered, frame)


def test_lightweight_skeleton_annotator_draws_configured_edges() -> None:
    annotator = LightweightSkeletonAnnotator(color=(0, 255, 0), thickness=2)
    frame = np.zeros((180, 180, 3), dtype=np.uint8)
    key_points = sv.KeyPoints(
        xy=np.array(
            [[[30.0, 40.0], [90.0, 40.0], [90.0, 100.0]]],
            dtype=np.float32,
        )
    )

    rendered = annotator.annotate(frame.copy(), key_points=key_points, edges=[(0, 1), (1, 2)])

    assert not np.array_equal(rendered, frame)


def test_format_tracker_label_handles_partial_tracker_arrays() -> None:
    labels = np.array([5, -1, 12], dtype=int)

    assert SupervisionInferenceRunner._format_tracker_label(labels, 0) == "ID 5"
    assert SupervisionInferenceRunner._format_tracker_label(labels, 1) == "ID ?"
    assert SupervisionInferenceRunner._format_tracker_label(labels, 2) == "ID 12"


def test_lightweight_skeleton_annotator_draws_normalized_edges() -> None:
    frame = np.zeros((180, 180, 3), dtype=np.uint8)
    key_points = sv.KeyPoints(
        xy=np.array(
            [[[20.0, 20.0], [80.0, 20.0], [80.0, 80.0]]],
            dtype=np.float32,
        )
    )
    annotator = LightweightSkeletonAnnotator(color=(0, 255, 0), thickness=2)

    rendered = annotator.annotate(frame.copy(), key_points=key_points, edges=[(0, 1), (1, 2)])

    assert not np.array_equal(rendered, frame)


def test_lightweight_vertex_annotator_draws_palette_markers() -> None:
    frame = np.zeros((160, 160, 3), dtype=np.uint8)
    key_points = sv.KeyPoints(
        xy=np.array(
            [[[20.0, 20.0], [60.0, 40.0], [90.0, 90.0]]],
            dtype=np.float32,
        )
    )
    annotator = LightweightVertexAnnotator(radius=4, palette_name="summer", fallback_color=(0, 0, 255))

    rendered = annotator.annotate(frame.copy(), key_points=key_points)

    assert not np.array_equal(rendered, frame)


def test_preview_resize_downscales_large_frame_only() -> None:
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    rendered = SupervisionInferenceRunner._resize_preview_frame(frame, 1280)

    assert rendered.shape[:2] == (720, 1280)


def test_preview_resize_keeps_small_frame() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    rendered = SupervisionInferenceRunner._resize_preview_frame(frame, 1280)

    assert rendered is frame


def test_should_render_preview_uses_stride_but_keeps_first_frame() -> None:
    assert SupervisionInferenceRunner._should_render_preview(0, 3) is True
    assert SupervisionInferenceRunner._should_render_preview(1, 3) is False
    assert SupervisionInferenceRunner._should_render_preview(3, 3) is True
