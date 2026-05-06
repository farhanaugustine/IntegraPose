import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd


def _load_module(rel_path: str, module_name: str):
    repo_root = Path(__file__).resolve().parent.parent
    module_path = repo_root / rel_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_analytics_service_mod = _load_module(
    "integra_pose/gui/services/analytics_service.py",
    "test_analytics_service_module",
)
AnalyticsService = _analytics_service_mod.AnalyticsService


class _FakeROIManager:
    def __init__(self, payload=None):
        self._payload = payload or {}

    def to_serializable(self):
        return self._payload


class _FakeConfig:
    def __init__(self, values):
        self._values = dict(values)

    def get_setting(self, setting_path):
        return self._values.get(setting_path)


class _FakeVar:
    def __init__(self, value=""):
        self.value = value

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


class _FakePipeline:
    @staticmethod
    def _index_existing_label_dirs(_root):
        return []

    @staticmethod
    def _find_existing_labels_dir(**_kwargs):
        return None


class _FakeBatchProcessingService:
    def __init__(self, capabilities=None):
        self._capabilities = capabilities
        self.pipeline = _FakePipeline()

    def preflight_model_capabilities(self, _model_path):
        if isinstance(self._capabilities, Exception):
            raise self._capabilities
        return self._capabilities


class _FakeApp:
    def __init__(self, values, *, roi_payload=None):
        self.config = _FakeConfig(values)
        self.roi_manager = _FakeROIManager(roi_payload)


def test_preview_preflight_prefers_model_class_names_for_detection():
    values = {
        "inference.trained_model_path_infer": "model.pt",
        "analytics.roi_analytics_yaml_path_var": "",
        "analytics.source_video_path_var": "video.mp4",
        "analytics.yolo_output_path_var": "labels",
        "analytics.object_interaction_enabled_var": False,
        "analytics.object_count_var": "0",
        "analytics.max_frame_gap_var": "50",
        "analytics.min_bout_duration_var": "20",
        "analytics.roi_max_gap_frames_var": "8",
        "analytics.roi_min_dwell_frames_var": "3",
        "analytics.roi_use_keypoint_var": False,
        "analytics.single_animal_analysis_var": True,
        "setup.behaviors_list": [{"id": 0, "name": "Standing"}, {"id": 1, "name": "Walking"}],
    }
    fake_app = _FakeApp(values, roi_payload={"Center": {"polygons": [[(1, 1), (5, 1), (5, 5)]], "enabled": True}})
    fake_app.root = _ImmediateRoot()
    fake_app.config.analytics = SimpleNamespace()
    fake_app.object_roi_manager = _FakeROIManager({})
    fake_app.analytics_behavior_source_var = _FakeVar()
    fake_app.analytics_preflight_summary_var = _FakeVar()
    fake_app.analytics_metric_checkbox_by_key = {}
    fake_app.batch_processing_service = _FakeBatchProcessingService(
        SimpleNamespace(
            model_path="model.pt",
            task="detect",
            has_keypoints=False,
            keypoint_count=0,
            class_count=1,
            class_names=["mouse"],
            supports_multiclass_stats=False,
            warnings=[],
        )
    )

    service = AnalyticsService(fake_app)
    preflight = service.preview_preflight(show_dialog=False)

    assert preflight["behavior_names_override"] == ["mouse"]
    assert preflight["behavior_source"] == "model class names"
    assert "Behavior labels: model class names" in fake_app.analytics_behavior_source_var.get()


def test_preview_preflight_uses_class_id_fallback_for_detection_without_names():
    values = {
        "inference.trained_model_path_infer": "model.pt",
        "analytics.roi_analytics_yaml_path_var": "",
        "analytics.source_video_path_var": "video.mp4",
        "analytics.yolo_output_path_var": "labels",
        "analytics.object_interaction_enabled_var": False,
        "analytics.object_count_var": "0",
        "analytics.max_frame_gap_var": "50",
        "analytics.min_bout_duration_var": "20",
        "analytics.roi_max_gap_frames_var": "8",
        "analytics.roi_min_dwell_frames_var": "3",
        "analytics.roi_use_keypoint_var": False,
        "analytics.single_animal_analysis_var": True,
        "setup.behaviors_list": [{"id": 0, "name": "Standing"}, {"id": 1, "name": "Walking"}],
    }
    fake_app = _FakeApp(values)
    fake_app.root = _ImmediateRoot()
    fake_app.config.analytics = SimpleNamespace()
    fake_app.object_roi_manager = _FakeROIManager({})
    fake_app.analytics_behavior_source_var = _FakeVar()
    fake_app.analytics_preflight_summary_var = _FakeVar()
    fake_app.analytics_metric_checkbox_by_key = {}
    fake_app.batch_processing_service = _FakeBatchProcessingService(
        SimpleNamespace(
            model_path="model.pt",
            task="detect",
            has_keypoints=False,
            keypoint_count=0,
            class_count=1,
            class_names=[],
            supports_multiclass_stats=False,
            warnings=[],
        )
    )

    service = AnalyticsService(fake_app)
    preflight = service.preview_preflight(show_dialog=False)

    assert preflight["behavior_names_override"] == ["class_0"]
    assert preflight["behavior_source"] == "generated detection class ids"



class _ImmediateRoot:
    def after(self, _delay, callback):
        callback()


class _FakeButton:
    def __init__(self):
        self.calls = []

    def config(self, **kwargs):
        self.calls.append(kwargs)


class _FakeRuntimeAnalytics:
    def __init__(self, result):
        self._result = result
        self.progress_updates = []

    def run_analysis(self, _params):
        return self._result

    def _update_progress(self, **kwargs):
        self.progress_updates.append(kwargs)


class _FakeStartThread:
    def __init__(self, *, target=None, args=(), daemon=None):
        self.target = target
        self.args = args
        self.daemon = daemon

    def is_alive(self):
        return False

    def start(self):
        return None


def test_build_roi_entry_settings_defaults_to_bbox_mode():
    service = AnalyticsService(_FakeApp({"analytics.roi_use_keypoint_var": False, "analytics.roi_entry_keypoint_index_var": "2"}))

    settings = service._build_roi_entry_settings()

    assert settings["roi_event_mode"] == "bbox_only"
    assert settings["keypoint_entry_index"] == 2
    assert settings["keypoint_entry_indices"] == []


def test_build_roi_entry_settings_uses_keypoint_mode_when_enabled():
    service = AnalyticsService(_FakeApp({"analytics.roi_use_keypoint_var": True, "analytics.roi_entry_keypoint_index_var": "4"}))

    settings = service._build_roi_entry_settings()

    assert settings["roi_event_mode"] == "keypoint_index"
    assert settings["keypoint_entry_index"] == 4
    assert settings["keypoint_entry_ratio_threshold"] == 0.5


def test_snapshot_roi_payload_uses_serialized_manager_state():
    payload = {
        "Center": {
            "polygons": [[(1, 1), (5, 1), (5, 5), (1, 5)]],
            "enabled": True,
        }
    }
    service = AnalyticsService(_FakeApp({}, roi_payload=payload))

    snapshot = service._snapshot_roi_payload()

    assert snapshot == payload


def test_resolve_video_roi_payload_prefers_frozen_param_override():
    frozen_payload = {
        "Frozen": {
            "polygons": [[(10, 10), (20, 10), (20, 20), (10, 20)]],
            "enabled": True,
        }
    }
    live_payload = {
        "Live": {
            "polygons": [[(1, 1), (5, 1), (5, 5), (1, 5)]],
            "enabled": True,
        }
    }
    service = AnalyticsService(_FakeApp({}, roi_payload=live_payload))

    resolved = service._resolve_video_roi_payload(
        {"roi_polygon_map_override": frozen_payload},
        use_rois=True,
    )

    assert resolved == frozen_payload


def test_worker_renders_video_even_when_no_bouts(monkeypatch):
    detailed_df = pd.DataFrame(columns=["Track ID", "Behavior", "Start Frame", "End Frame"])
    summary_df = pd.DataFrame()
    analytics_result = (
        detailed_df,
        summary_df,
        "",
        {},
        "out_dir",
        "video_name",
        False,
        {"entries": [], "exits": []},
        {},
    )

    fake_app = _FakeApp({})
    fake_app.analytics = _FakeRuntimeAnalytics(analytics_result)
    fake_app.root = _ImmediateRoot()
    fake_app.process_btn = _FakeButton()
    fake_app.tk = type("tk", (), {"NORMAL": "normal"})()
    fake_app.log_messages = []
    fake_app.activities = []
    fake_app.statuses = []
    fake_app.toasts = []
    fake_app.displayed = 0
    fake_app.populated = []
    fake_app.updated_filter = 0
    fake_app.updated_modules = 0
    fake_app.roi_metrics = None
    fake_app.additional_module_outputs = {}
    fake_app.object_roi_manager = _FakeROIManager({})
    fake_app._ensure_roi_lookup = lambda force=False: None
    fake_app._update_module_outputs_tree = lambda: setattr(fake_app, "updated_modules", fake_app.updated_modules + 1)
    fake_app._display_analytics_results = lambda: setattr(fake_app, "displayed", fake_app.displayed + 1)
    fake_app._populate_roi_metrics_tabs = lambda metrics: fake_app.populated.append(metrics)
    fake_app._update_roi_filter_combobox = lambda: setattr(fake_app, "updated_filter", fake_app.updated_filter + 1)
    fake_app._set_process_activity = lambda key, value: fake_app.activities.append((key, value))
    fake_app._toast = lambda message, level="info", duration_ms=0: fake_app.toasts.append((message, level, duration_ms))
    fake_app._set_job_status = lambda value: fake_app.statuses.append(("job", value))
    fake_app.update_status = lambda value: fake_app.statuses.append(("status", value))
    fake_app.log_message = lambda message, level="INFO": fake_app.log_messages.append((level, message))

    service = AnalyticsService(fake_app)
    render_calls = []
    monkeypatch.setattr(
        _analytics_service_mod.video_creator,
        "create_annotated_video",
        lambda *args, **kwargs: render_calls.append((args, kwargs)) or True,
    )
    monkeypatch.setattr(
        _analytics_service_mod.video_creator,
        "load_object_overlay_metrics",
        lambda module_outputs: ({}, {"entries": [], "exits": []}),
    )
    monkeypatch.setattr(_analytics_service_mod.messagebox, "showinfo", lambda *args, **kwargs: None)
    monkeypatch.setattr(_analytics_service_mod.messagebox, "showwarning", lambda *args, **kwargs: None)
    monkeypatch.setattr(_analytics_service_mod.messagebox, "showerror", lambda *args, **kwargs: None)

    service._worker(
        {
            "create_video": True,
            "video_file": "video.mp4",
            "yolo_folder": "labels",
            "yaml_file": "",
            "object_roi_polygon_map_override": {},
        }
    )

    assert len(render_calls) == 1
    assert ("analytics", "completed") in fake_app.activities
    assert any("No bouts found in analysis." in message for _level, message in fake_app.log_messages)


def test_start_includes_roi_debounce_settings(monkeypatch):
    values = {
        "inference.trained_model_path_infer": "model.pt",
        "analytics.yolo_output_path_var": "labels",
        "analytics.source_video_path_var": "video.mp4",
        "analytics.roi_analytics_yaml_path_var": "",
        "analytics.assay_preset_var": "custom",
        "analytics.min_bout_duration_var": "50",
        "analytics.max_frame_gap_var": "50",
        "analytics.roi_min_dwell_frames_var": "4",
        "analytics.roi_max_gap_frames_var": "12",
        "analytics.create_video_output_var": True,
        "analytics.object_interaction_enabled_var": False,
        "analytics.object_count_var": "0",
        "analytics.object_roi_size_px_var": "20",
        "analytics.object_roi_shape_var": "circle",
        "analytics.object_interaction_keypoint_index_var": "0",
        "analytics.object_interaction_distance_px_var": "0.0",
        "analytics.roi_use_keypoint_var": False,
        "analytics.roi_entry_keypoint_index_var": "0",
    }
    fake_app = _FakeApp(values, roi_payload={"ROI": {"polygons": [[(1, 1), (2, 1), (2, 2)]], "enabled": True}})
    fake_app.root = _ImmediateRoot()
    fake_app.config.analytics = SimpleNamespace()
    fake_app.process_btn = _FakeButton()
    fake_app.tk = type("tk", (), {"DISABLED": "disabled", "NORMAL": "normal"})()
    fake_app.log_messages = []
    fake_app.statuses = []
    fake_app.activities = []
    fake_app.additional_module_outputs = {}
    fake_app.object_roi_manager = _FakeROIManager({})
    fake_app.analytics_behavior_source_var = _FakeVar()
    fake_app.analytics_preflight_summary_var = _FakeVar()
    fake_app.analytics_metric_checkbox_by_key = {}
    fake_app.batch_processing_service = _FakeBatchProcessingService(
        SimpleNamespace(
            model_path="model.pt",
            task="detect",
            has_keypoints=False,
            keypoint_count=0,
            class_count=1,
            class_names=["mouse"],
            supports_multiclass_stats=False,
            warnings=[],
        )
    )
    fake_app.log_message = lambda message, level="INFO": fake_app.log_messages.append((level, message))
    fake_app.update_status = lambda value: fake_app.statuses.append(("status", value))
    fake_app._set_job_status = lambda value: fake_app.statuses.append(("job", value))
    fake_app._set_process_activity = lambda key, value: fake_app.activities.append((key, value))
    fake_app._update_module_outputs_tree = lambda: None

    captured = {}

    def _thread_factory(*, target=None, args=(), daemon=None):
        captured["target"] = target
        captured["args"] = args
        captured["daemon"] = daemon
        return _FakeStartThread(target=target, args=args, daemon=daemon)

    monkeypatch.setattr(_analytics_service_mod, "collect_enabled_metric_keys", lambda _cfg: [])
    monkeypatch.setattr(_analytics_service_mod, "expand_metrics_to_modules", lambda _keys: [])
    monkeypatch.setattr(_analytics_service_mod.os.path, "isdir", lambda path: path == "labels")
    monkeypatch.setattr(_analytics_service_mod.os, "listdir", lambda path: ["frame_0001.txt"] if path == "labels" else [])
    monkeypatch.setattr(_analytics_service_mod.threading, "Thread", _thread_factory)

    service = AnalyticsService(fake_app)
    service.start()

    params = captured["args"][0]
    assert params["behavior_names_override"] == ["mouse"]
    assert params["min_bout_frames"] == 50
    assert params["max_gap_frames"] == 50
    assert params["roi_min_dwell_frames"] == 4
    assert params["roi_max_gap_frames"] == 12
