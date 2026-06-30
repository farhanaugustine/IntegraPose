from __future__ import annotations

import subprocess
import sys
import importlib.util
from pathlib import Path
from tkinter import messagebox


PLUGIN_DISPLAY_NAME = "TandemYTC - Tandem YOLO + Temporal Classifier"
PLUGIN_SHORT_NAME = "TandemYTC"


class YoloTemporalClassifierPlugin:
    """Tk host adapter for the Qt-based TandemYTC workspace."""

    def __init__(self) -> None:
        self._main_app = None
        self._process: subprocess.Popen | None = None

    def register(self, main_app) -> None:
        self._main_app = main_app
        menu = getattr(main_app, "plugins_menu", None)
        if menu is None:
            main_app.log_message(
                "plugins_menu not available; cannot register TandemYTC plugin.",
                "ERROR",
            )
            return

        menu.add_command(label=PLUGIN_DISPLAY_NAME, command=self.open_workspace)
        main_app.log_message("TandemYTC plugin registered", "INFO")

    def open_workspace(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._notify_info("The TandemYTC workspace is already running.")
            return
        if importlib.util.find_spec("PySide6") is None:
            self._notify_error(
                "Missing optional dependency: PySide6.\n\n"
                "Install plugin dependencies with:\n"
                "pip install -r integra_pose/plugins/requirements.txt"
            )
            return

        workspace_root = Path.home() / ".integrapose" / "yolo_temporal_classifier"
        workspace_root.mkdir(parents=True, exist_ok=True)
        project_db = workspace_root / "workspace.sqlite"

        cmd = [
            sys.executable,
            "-m",
            "integra_pose.plugins.plugin_tandem_yolo_toolkit.yolo_temporal_classifier.qt_app",
            "--project_db",
            str(project_db),
        ]
        try:
            self._process = subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parents[3]))
        except ModuleNotFoundError as exc:
            self._notify_error(f"Missing optional Qt dependency: {exc}")
        except Exception as exc:  # pragma: no cover - surfaced to UI
            self._notify_error(f"Failed to launch TandemYTC: {exc}")
        else:
            self._log(f"Opened TandemYTC workspace: {project_db}", "INFO")

    def _log(self, message: str, level: str = "INFO") -> None:
        if self._main_app is not None:
            self._main_app.log_message(message, level)

    def _notify_error(self, message: str) -> None:
        self._log(message, "ERROR")
        parent = getattr(self._main_app, "root", None)
        messagebox.showerror(PLUGIN_DISPLAY_NAME, message, parent=parent)

    def _notify_info(self, message: str) -> None:
        self._log(message, "INFO")
        parent = getattr(self._main_app, "root", None)
        messagebox.showinfo(PLUGIN_DISPLAY_NAME, message, parent=parent)


def register_plugin(main_app) -> None:
    existing = getattr(main_app, "_yolo_temporal_classifier_plugin", None)
    if existing is not None:
        return
    plugin = YoloTemporalClassifierPlugin()
    plugin.register(main_app)
    setattr(main_app, "_yolo_temporal_classifier_plugin", plugin)


__all__ = ["register_plugin", "YoloTemporalClassifierPlugin"]
