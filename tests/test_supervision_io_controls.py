import io
from pathlib import Path

import numpy as np
import pytest
import supervision as sv

from integra_pose.logic.supervision_runner import (
    AdvancedOverlaySettings,
    AnnotationOptions,
    InferenceSettings,
    LabelsAggregateRecorder,
    SupervisionInferenceRunner,
    TrackingMetricsRecorder,
)
from integra_pose.utils.frame_identity import load_frame_label_class_metadata


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


def test_labels_recorder_rejects_schema_drift_without_truncating(tmp_path: Path) -> None:
    target = tmp_path / "labels.csv"
    recorder = LabelsAggregateRecorder(
        target,
        lambda *_: None,
        include_confidence=True,
        flush_interval_frames=1,
    )
    xywhn = np.array([[0.5, 0.5, 0.4, 0.4]], dtype=np.float32)
    cls = np.array([0], dtype=np.int32)
    conf = np.array([0.95], dtype=np.float32)
    recorder.record(
        0,
        xywhn,
        cls,
        conf,
        np.array([[[0.1, 0.2]]], dtype=np.float32),
        np.array([[0.8]], dtype=np.float32),
        ["1"],
    )
    before = target.read_text(encoding="utf-8")

    with pytest.raises(RuntimeError, match="refusing to truncate"):
        recorder.record(
            1,
            xywhn,
            cls,
            conf,
            np.array([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32),
            np.array([[0.8, 0.7]], dtype=np.float32),
            ["1"],
        )

    recorder.close()
    assert target.read_text(encoding="utf-8") == before


@pytest.mark.parametrize("model_task", ["detect", "pose"])
def test_supervision_output_manifest_persists_model_class_names(
    tmp_path: Path,
    model_task: str,
) -> None:
    settings = _make_settings(tmp_path)
    settings.save = False
    settings.save_txt = True
    runner = SupervisionInferenceRunner(settings, None, lambda *_: None)
    runner._model_class_names = ["Sniffing", "Wall-Rearing", "Ambulatory"]
    runner._model_task = model_task

    runner._prepare_output_directories()

    metadata = load_frame_label_class_metadata(runner._labels_dir)
    assert metadata["class_names"] == ["Sniffing", "Wall-Rearing", "Ambulatory"]
    assert metadata["class_names_source"] == "model.names"
    assert metadata["model_task"] == model_task


def test_video_crops_include_frame_identity_and_do_not_overwrite(tmp_path: Path) -> None:
    runner = object.__new__(SupervisionInferenceRunner)
    runner._crops_dir = tmp_path / "crops"
    runner.settings = type("Settings", (), {"source_path": tmp_path / "mouse_1.mp4"})()
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 4, 4]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0], dtype=np.int32),
    )
    detections.data["class_name"] = np.array(["mouse"])
    result = type("Result", (), {"path": str(tmp_path / "mouse_1.mp4")})()
    frame = np.full((8, 8, 3), 255, dtype=np.uint8)

    runner._write_crops(frame, detections, result, frame_index=0)
    runner._write_crops(frame, detections, result, frame_index=1)

    assert sorted(path.name for path in (tmp_path / "crops" / "mouse").glob("*.jpg")) == [
        "mouse_1_frame_000000_det_000.jpg",
        "mouse_1_frame_000001_det_000.jpg",
    ]


def test_label_writer_records_empty_frames_and_rejects_duplicate_identity(tmp_path: Path) -> None:
    runner = object.__new__(SupervisionInferenceRunner)
    runner._labels_dir = tmp_path / "labels"
    runner._labels_dir.mkdir()
    runner._labels_csv_recorder = None
    runner.settings = type(
        "Settings",
        (),
        {
            "source_path": tmp_path / "mouse_1.mp4",
            "max_det": 1,
            "single_animal_mode": False,
            "use_tracker": False,
            "save_conf": False,
        },
    )()
    result = type("Result", (), {"path": str(tmp_path / "mouse_1.mp4"), "boxes": None})()

    runner._write_labels_file(result, None, frame_index=0)

    label_path = runner._labels_dir / "mouse_1_frame_000000.txt"
    assert label_path.is_file()
    assert label_path.read_text(encoding="utf-8") == ""
    with pytest.raises(RuntimeError, match="Duplicate label output"):
        runner._write_labels_file(result, None, frame_index=0)


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
