import os
import sys
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from integra_pose.gui.controllers.inference_controller import InferenceController
from integra_pose.utils.config_manager import ConfigManager


def _make_tk_root():
    try:
        root = tk.Tk()
        root.withdraw()
        return root
    except tk.TclError:
        try:
            return tk.Tk(useTk=False)
        except tk.TclError as exc:  # pragma: no cover
            raise RuntimeError(f"Tk is unavailable in this environment: {exc}") from exc


def test_build_supervision_settings_keeps_trace_enabled_without_tracker(tmp_path: Path) -> None:
    previous_default_root = getattr(tk, "_default_root", None)
    root = _make_tk_root()
    tk._default_root = root
    try:
        app = MagicMock()
        app.root = root
        app.log_message = MagicMock()
        app.keypoint_names = []
        app.config = ConfigManager(app)
        app.config.pose_clustering.skeleton_connections = []
        app.config.analytics.single_animal_analysis_var.set(False)

        model_path = tmp_path / "best.pt"
        source_path = tmp_path / "video.mp4"
        model_path.write_bytes(b"placeholder")
        source_path.write_bytes(b"placeholder")

        cfg = app.config.inference
        cfg.trained_model_path_infer.set(str(model_path))
        cfg.video_infer_path.set(str(source_path))
        cfg.use_tracker_var.set(False)
        cfg.sv_use_trace_var.set(True)

        controller = InferenceController(app)
        settings = controller.build_supervision_settings()

        assert settings.annotation.use_trace is True
        assert settings.use_tracker is False
    finally:
        try:
            commands = set(root.tk.call("info", "commands"))
        except tk.TclError:
            commands = set()
        if "destroy" in commands:
            root.destroy()
        tk._default_root = previous_default_root


def test_build_supervision_settings_parses_advanced_io_controls(tmp_path: Path) -> None:
    previous_default_root = getattr(tk, "_default_root", None)
    root = _make_tk_root()
    tk._default_root = root
    try:
        app = MagicMock()
        app.root = root
        app.log_message = MagicMock()
        app.keypoint_names = []
        app.config = ConfigManager(app)
        app.config.pose_clustering.skeleton_connections = []
        app.config.analytics.single_animal_analysis_var.set(False)

        model_path = tmp_path / "best.pt"
        source_path = tmp_path / "video.mp4"
        model_path.write_bytes(b"placeholder")
        source_path.write_bytes(b"placeholder")

        cfg = app.config.inference
        cfg.trained_model_path_infer.set(str(model_path))
        cfg.video_infer_path.set(str(source_path))
        cfg.infer_async_video_save_var.set(True)
        cfg.infer_async_video_queue_size_var.set("12")
        cfg.infer_save_video_stride_var.set("3")
        cfg.metrics_flush_interval_frames_var.set("90")
        cfg.labels_csv_flush_interval_frames_var.set("45")
        cfg.infer_resource_guardrails_enabled_var.set(True)
        cfg.infer_min_free_disk_gb_var.set("3.5")
        cfg.infer_min_free_memory_mb_var.set("2048")

        controller = InferenceController(app)
        settings = controller.build_supervision_settings()

        assert settings.async_video_save is True
        assert settings.async_video_queue_size == 12
        assert settings.save_video_stride == 3
        assert settings.metrics_flush_interval_frames == 90
        assert settings.labels_csv_flush_interval_frames == 45
        assert settings.resource_guardrails_enabled is True
        assert settings.min_free_disk_gb == 3.5
        assert settings.min_free_memory_mb == 2048
    finally:
        try:
            commands = set(root.tk.call("info", "commands"))
        except tk.TclError:
            commands = set()
        if "destroy" in commands:
            root.destroy()
        tk._default_root = previous_default_root


def test_run_file_inference_routes_to_runner_for_async_save(tmp_path: Path) -> None:
    previous_default_root = getattr(tk, "_default_root", None)
    root = _make_tk_root()
    tk._default_root = root
    try:
        app = MagicMock()
        app.root = root
        app.log_message = MagicMock()
        app.update_status = MagicMock()
        app._set_job_status = MagicMock()
        app._clear_process_stop_requested = MagicMock()
        app._set_process_activity = MagicMock()
        app._consume_process_stop_requested = MagicMock(return_value=False)
        app._toast = MagicMock()
        app.set_performance_warning = MagicMock()
        app._supervision_annotations_selected = MagicMock(return_value=False)
        app._handle_performance_metrics = MagicMock()
        app.performance_metrics_var = tk.StringVar(master=root, value="Idle")
        app.performance_warning_var = tk.StringVar(master=root, value="")
        app.overlay_controller = SimpleNamespace(
            _capture_overlay_state=MagicMock(return_value={"booleans": {}}),
            restore_overlay_bools=MagicMock(),
        )
        app.process_manager = SimpleNamespace(start_process=MagicMock(), terminate_process=MagicMock())
        app.inference_lifecycle = SimpleNamespace(active_runner=None, start=MagicMock(), stop=MagicMock())
        app.run_inference_button = MagicMock()
        app.stop_inference_button = MagicMock()
        app.snapshot_button = MagicMock()
        app.config = ConfigManager(app)
        app.config.analytics.single_animal_analysis_var.set(False)
        app.config.inference.show_infer_var.set(False)
        app.config.inference.capture_metrics_var.set(False)
        app.config.inference.infer_save_var.set(True)
        app.config.inference.infer_async_video_save_var.set(True)

        controller = InferenceController(app)
        fake_settings = SimpleNamespace(resource_guardrails_enabled=False)
        controller.build_supervision_settings = MagicMock(return_value=fake_settings)
        controller.start_supervision_inference = MagicMock(return_value=True)

        controller.run_file_inference()

        controller.build_supervision_settings.assert_called_once()
        controller.start_supervision_inference.assert_called_once()
        app.process_manager.start_process.assert_not_called()
    finally:
        try:
            commands = set(root.tk.call("info", "commands"))
        except tk.TclError:
            commands = set()
        if "destroy" in commands:
            root.destroy()
        tk._default_root = previous_default_root
