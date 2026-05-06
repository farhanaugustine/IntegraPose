from pathlib import Path
from types import SimpleNamespace
import threading

from integra_pose.logic.batch_pipeline import BatchPipeline
from integra_pose.utils.bout_analyzer import BoutAnalysisCancelledError


class _Var:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _FakeAnalytics:
    def run_analysis(self, params):
        raise BoutAnalysisCancelledError("cancelled")


class _FakeApp:
    def __init__(self):
        self.analytics = _FakeAnalytics()
        self.config = SimpleNamespace(
            analytics=SimpleNamespace(
                min_bout_duration_var=_Var("5"),
                max_frame_gap_var=_Var("5"),
            )
        )
        self.logs = []

    def log_message(self, message, level="INFO"):
        self.logs.append((level, message))


def test_run_video_analytics_marks_item_cancelled_on_cooperative_stop(tmp_path):
    pipeline = BatchPipeline(_FakeApp())
    session = SimpleNamespace(
        roi_strategy="per_video",
        shared_rois={},
        shared_object_rois={},
        save_annotated_video=False,
        roi_event_mode="bbox_only",
        roi_entry_threshold=0.75,
        roi_exit_threshold=0.25,
        keypoint_entry_index=0,
        keypoint_entry_indices=[],
        keypoint_entry_ratio_threshold=0.5,
        object_interaction_enabled=False,
        object_count=0,
        object_roi_size_px=20,
        object_roi_shape="circle",
        object_interaction_keypoint_index=0,
        object_interaction_distance_px=0.0,
        single_animal_mode=False,
        analytics_assay_preset="custom",
        analytics_enabled_metrics=[],
        analytics_enabled_modules=[],
        review_policy="after_all",
        model_capabilities=SimpleNamespace(class_names=[]),
    )
    item = SimpleNamespace(
        video_id="video-1",
        video_name="video.mp4",
        video_path=str(tmp_path / "video.mp4"),
        rois={},
        object_rois={},
        group="",
        subject_id="",
        time_point="",
        analytics_status="pending",
        status_message="",
    )

    result = pipeline._run_video_analytics(
        session=session,
        item=item,
        run_root=Path(tmp_path),
        run_dir=Path(tmp_path),
        labels_dir=Path(tmp_path),
        labels_csv=Path(tmp_path / "labels.csv"),
        metrics_csv=Path(tmp_path / "metrics.csv"),
        metrics_track_csv=Path(tmp_path / "metrics_track.csv"),
        metrics_frame_csv=Path(tmp_path / "metrics_frame.csv"),
        yaml_path=Path(tmp_path / "dataset.yaml"),
        labels_root=None,
        use_existing_labels=False,
        stop_event=threading.Event(),
    )

    assert result is None
    assert item.analytics_status == "cancelled"
    assert item.status_message == "Analytics cancelled."


def test_resolve_tracker_config_preserves_aliases_and_resolves_paths(tmp_path):
    alias = BatchPipeline._resolve_tracker_config("bytetrack.yaml")
    custom_yaml = tmp_path / "custom_tracker.yaml"
    custom_yaml.write_text("tracker_type: bytetrack\n", encoding="utf-8")
    resolved = BatchPipeline._resolve_tracker_config(str(custom_yaml))

    assert str(alias) == "bytetrack.yaml"
    assert resolved == custom_yaml.resolve()
