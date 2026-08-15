from pathlib import Path
from types import SimpleNamespace
import json
import threading

from integra_pose.logic.batch_pipeline import BatchPipeline
from integra_pose.utils.bout_analyzer import BoutAnalysisCancelledError
from integra_pose.utils.frame_identity import write_frame_label_manifest


class _Var:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _FakeAnalytics:
    def __init__(self):
        self.params = None

    def run_analysis(self, params):
        self.params = dict(params)
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
    app = _FakeApp()
    del app.config.analytics.min_bout_duration_var
    del app.config.analytics.max_frame_gap_var
    pipeline = BatchPipeline(app)
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
        model_capabilities=SimpleNamespace(class_names=["WrongCurrentModelName"]),
        min_bout_frames=9,
        max_gap_frames=9,
        roi_min_dwell_frames=9,
        roi_max_gap_frames=9,
        temporal_threshold_unit="frames",
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

    write_frame_label_manifest(
        tmp_path,
        source="video.mp4",
        max_det=1,
        class_names={0: "Sniffing", 1: "Wall-Rearing", 2: "Ambulatory"},
        class_names_source="model.names",
        model_task="detect",
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
    assert app.analytics.params["min_bout_frames"] == 9
    assert app.analytics.params["max_gap_frames"] == 9
    assert app.analytics.params["roi_min_dwell_frames"] == 9
    assert app.analytics.params["roi_max_gap_frames"] == 9
    assert app.analytics.params["object_min_dwell_frames"] == 9
    assert app.analytics.params["object_max_gap_frames"] == 9
    assert app.analytics.params["behavior_names_override"] == [
        "Sniffing",
        "Wall-Rearing",
        "Ambulatory",
    ]
    assert app.analytics.params["behavior_names_source"] == "inference label metadata"


def test_resolve_tracker_config_preserves_aliases_and_resolves_paths(tmp_path):
    alias = BatchPipeline._resolve_tracker_config("bytetrack.yaml")
    custom_yaml = tmp_path / "custom_tracker.yaml"
    custom_yaml.write_text("tracker_type: bytetrack\n", encoding="utf-8")
    resolved = BatchPipeline._resolve_tracker_config(str(custom_yaml))

    assert str(alias) == "bytetrack.yaml"
    assert resolved == custom_yaml.resolve()


def test_second_based_thresholds_resolve_separately_for_each_video_fps(tmp_path):
    pipeline = BatchPipeline(_FakeApp())
    session = SimpleNamespace(
        video_fps=0.0,
        min_bout_seconds=0.10,
        max_gap_seconds=0.17,
        roi_min_dwell_seconds=0.20,
        roi_max_gap_seconds=0.05,
    )
    item = SimpleNamespace(video_name="video.mp4", video_path=str(tmp_path / "video.mp4"))
    run_30 = tmp_path / "run_30"
    run_60 = tmp_path / "run_60"
    run_30.mkdir()
    run_60.mkdir()
    (run_30 / "inference_metadata.json").write_text(
        json.dumps({"fps": 30.0}), encoding="utf-8"
    )
    (run_60 / "inference_metadata.json").write_text(
        json.dumps({"fps": 60.0}), encoding="utf-8"
    )

    assert pipeline._time_threshold_frames(
        session=session, item=item, run_dir=run_30
    ) == (3, 5, 6, 1, 30.0)
    assert pipeline._time_threshold_frames(
        session=session, item=item, run_dir=run_60
    ) == (6, 10, 12, 3, 60.0)
