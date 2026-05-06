import io
from pathlib import Path

import numpy as np
import supervision as sv

from integra_pose.logic.supervision_runner import (
    AdvancedOverlaySettings,
    AnnotationOptions,
    InferenceSettings,
    LabelsAggregateRecorder,
    SupervisionInferenceRunner,
    TrackingMetricsRecorder,
)


class _TrackedFile(io.StringIO):
    def __init__(self):
        super().__init__()
        self.flush_count = 0

    def flush(self):
        self.flush_count += 1
        return super().flush()


def _make_settings(tmp_path: Path) -> InferenceSettings:
    advanced = AdvancedOverlaySettings(
        halo_kernel=1,
        halo_opacity=0.0,
        blur_kernel=1,
        pixelate_size=1,
        heatmap_radius=1,
        heatmap_kernel=1,
        heatmap_opacity=0.0,
        heatmap_decay=1.0,
        heatmap_source="bbox",
        heatmap_anchor="CENTER",
        heatmap_keypoint_index=0,
        background_opacity=0.0,
        background_force_box=False,
        vertex_radius=1,
        edge_thickness=1,
        keypoint_palette="jet",
        edge_palette="jet",
        trace_length=1,
        trace_thickness=1,
        trace_opacity=0.0,
        trace_persistent=False,
        trace_source="bbox",
        trace_keypoint_index=None,
        trace_color="",
    )
    annotation = AnnotationOptions(
        use_boxes=False,
        use_labels=False,
        use_trace=False,
        use_tracker_ids=False,
        use_edges=False,
        use_vertices=False,
        use_heading_arrows=False,
        use_halo=False,
        use_blur=False,
        use_pixelate=False,
        use_background_overlay=False,
        use_heatmap=False,
        skeleton_source="config",
        skeleton_color="red",
        skeleton_file=None,
        config_skeleton=[],
        hide_labels=False,
        hide_conf=False,
        tracker_enabled=False,
        line_width=None,
        overlay_order=[],
        advanced=advanced,
        performance_safe=True,
        trace_source="bbox",
        trace_keypoint_index=None,
        trace_color="",
    )
    return InferenceSettings(
        model_path=tmp_path / "best.pt",
        source_path=tmp_path / "video.mp4",
        tracker_config=None,
        use_tracker=False,
        conf=0.25,
        iou=0.45,
        imgsz=640,
        device="cpu",
        max_det=10,
        augment=False,
        show=False,
        preview_max_side=1280,
        preview_frame_stride=2,
        save=True,
        async_video_save=True,
        async_video_queue_size=4,
        save_video_stride=2,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        project=tmp_path / "runs",
        run_name="guardrail_test",
        capture_metrics=False,
        grid_metrics_enabled=False,
        grid_size_px=50,
        grid_save_heatmaps=False,
        grid_heatmap_use_frame=False,
        heading_indices=None,
        heading_names=None,
        model_keypoint_names=None,
        keypoint_names=None,
        annotation=annotation,
        motion_direction_threshold_deg=15.0,
        motion_velocity_threshold_px=0.0,
        metrics_flush_interval_frames=60,
        labels_csv_flush_interval_frames=60,
        resource_guardrails_enabled=True,
        min_free_disk_gb=0.0,
        min_free_memory_mb=1024,
    )


def test_tracking_metrics_recorder_flushes_on_interval(monkeypatch, tmp_path: Path) -> None:
    tracked_file = _TrackedFile()

    def _fake_open(self, *args, **kwargs):
        tracked_file.seek(0)
        tracked_file.truncate(0)
        tracked_file.flush_count = 0
        return tracked_file

    monkeypatch.setattr(Path, "open", _fake_open)

    recorder = TrackingMetricsRecorder(tmp_path / "metrics.csv", lambda *_: None, flush_interval_frames=2)
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0], dtype=np.int32),
    )
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    recorder.record(0, frame, detections, tracker_expected=False)
    assert tracked_file.flush_count == 1

    recorder.record(1, frame, detections, tracker_expected=False)
    assert tracked_file.flush_count == 2

    recorder.close()
    assert tracked_file.flush_count == 3


def test_labels_aggregate_recorder_flushes_on_interval(monkeypatch, tmp_path: Path) -> None:
    tracked_file = _TrackedFile()

    def _fake_open(self, *args, **kwargs):
        tracked_file.seek(0)
        tracked_file.truncate(0)
        tracked_file.flush_count = 0
        return tracked_file

    monkeypatch.setattr(Path, "open", _fake_open)

    recorder = LabelsAggregateRecorder(
        tmp_path / "labels.csv",
        lambda *_: None,
        include_confidence=True,
        flush_interval_frames=2,
    )
    xywhn = np.array([[0.5, 0.5, 0.4, 0.4]], dtype=np.float32)
    cls = np.array([0], dtype=np.int32)
    conf = np.array([0.95], dtype=np.float32)

    recorder.record(0, xywhn, cls, conf, None, None, ["1"])
    assert tracked_file.flush_count == 1

    recorder.record(1, xywhn, cls, conf, None, None, ["1"])
    assert tracked_file.flush_count == 2

    recorder.close()
    assert tracked_file.flush_count == 3


def test_resource_guardrails_account_for_async_queue(monkeypatch, tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)

    monkeypatch.setattr(
        SupervisionInferenceRunner,
        "_available_system_memory_bytes",
        staticmethod(lambda: 9 * 1024 * 1024),
    )
    monkeypatch.setattr(
        SupervisionInferenceRunner,
        "_estimate_source_frame_bytes",
        classmethod(lambda cls, configured: 2 * 1024 * 1024),
    )

    try:
        SupervisionInferenceRunner.validate_resource_guardrails_for_settings(settings)
    except ValueError as exc:
        assert "RAM" in str(exc)
    else:
        raise AssertionError("Expected RAM guardrail validation to fail.")
