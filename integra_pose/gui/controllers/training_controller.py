from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import messagebox

from integra_pose.utils import command_builder, model_registry


class TrainingController:
    """Training/export orchestration."""

    def __init__(self, app: Any):
        self.app = app

    def start_training(self):
        app = self.app
        app.log_message("Attempting to start training...", "INFO")
        if hasattr(app, 'train_button'):
            app.train_button.config(state=tk.DISABLED)
            app.log_message("Disabled Start Training button.", "INFO")
        if hasattr(app, "stop_training_button"):
            app.stop_training_button.config(state=tk.NORMAL)
            app.log_message("Enabled Stop Training button.", "INFO")
        app._set_job_status("Training in progress")
        app._clear_process_stop_requested("training")
        app._set_process_activity("training", "running", "starting")

        def on_finish():
            stopped = app._consume_process_stop_requested("training")
            if hasattr(app, 'train_button'):
                app.train_button.config(state=tk.NORMAL)
                app.log_message("Enabled Start Training button.", "INFO")
            if hasattr(app, "stop_training_button"):
                app.stop_training_button.config(state=tk.DISABLED)
                app.log_message("Disabled Stop Training button.", "INFO")
            app.update_status("Training finished or stopped.")
            app._set_job_status("No active jobs")
            try:
                project_root = Path(app.config.training.model_save_dir.get() or "runs/pose/train")
                run_name = (app.config.training.run_name_var.get() or "").strip()
                candidate = project_root / run_name / "weights" / "best.pt"
                results_csv_path = project_root / run_name / "results.csv"
                if candidate.exists():
                    app.config.training.export_model_path.set(str(candidate))
                    app.log_message(f"Detected new weights at {candidate}; pre-filling export panel.", "INFO")
                    metrics = model_registry.parse_results_metrics(results_csv_path)
                    summary = model_registry.summarize_core_metrics(metrics)
                    if summary:
                        app.log_message("--- Training Validation Summary ---", "INFO")
                        for label, value in summary:
                            app.log_message(f"{label:14s}: {value:.4f}", "INFO")
                        app.log_message("-----------------------------------", "INFO")
                    record = model_registry.register_training_run(
                        app,
                        model_path=candidate,
                        results_csv_path=results_csv_path,
                    )
                    if record is not None:
                        app.log_message("Registered training run in Model Registry.", "INFO")
                    refresh_registry = getattr(app, "_refresh_model_registry_views", None)
                    if callable(refresh_registry):
                        refresh_registry()
            except Exception:
                pass
            if stopped:
                app._set_process_activity("training", "idle")
            else:
                app._set_process_activity("training", "completed")
                app._toast("Training finished.", level="info", duration_ms=6000)

        app.process_manager.start_process(command_builder.build_training_command, 'training_process', on_finish_callback=on_finish)
        process = getattr(app, 'training_process', None)
        if not process or process.poll() is not None:
            if hasattr(app, 'train_button'):
                app.train_button.config(state=tk.NORMAL)
                app.log_message("Enabled Start Training button.", "INFO")
            if hasattr(app, "stop_training_button"):
                app.stop_training_button.config(state=tk.DISABLED)
                app.log_message("Disabled Stop Training button.", "INFO")
            app._set_job_status("No active jobs")
            app._set_process_activity("training", "idle")

    def start_export(self):
        app = self.app
        app.log_message("Preparing export/quantization...", "INFO")

        cmd, err = command_builder.build_export_command(app)
        if err:
            messagebox.showerror("Export Configuration Error", err, parent=app.root)
            return

        if hasattr(app, 'export_button'):
            app.export_button.config(state=tk.DISABLED)

        if app.config.training.export_int8_var.get():
            fmt = (app.config.training.export_format_var.get() or "").strip()
            app.log_message(
                f"INT8 export requested for format '{fmt}'. Ensure TensorRT or OpenVINO is installed; otherwise the export may fail.",
                "WARNING",
            )

        def on_finish():
            if hasattr(app, 'export_button'):
                app.export_button.config(state=app.tk.NORMAL if hasattr(app, "tk") else app.tk.NORMAL)
            app.log_message("Enabled Export / Quantize button.", "INFO")
            app.update_status("Export finished or stopped.")

        app.process_manager.start_process(lambda _app, c=cmd: (c, None), 'export_process', on_finish_callback=on_finish)

        process = getattr(app, 'export_process', None)
        if not process or process.poll() is not None:
            if hasattr(app, 'export_button'):
                app.export_button.config(state=tk.NORMAL)

    def stop_training(self):
        app = self.app
        app.log_message("Attempting to stop training...", "INFO")
        app._mark_process_stop_requested("training")
        app.process_manager.terminate_process('training_process')
        if hasattr(app, 'train_button'):
            app.train_button.config(state=tk.NORMAL)
            app.log_message("Enabled Start Training button.", "INFO")
        if hasattr(app, "stop_training_button"):
            app.stop_training_button.config(state=tk.DISABLED)
            app.log_message("Disabled Stop Training button.", "INFO")
        app.update_status("Training stopped.")
        app._set_job_status("No active jobs")
        app._set_process_activity("training", "idle")
        try:
            app._toast("Training stopped.", level="info")
        except Exception:
            pass
