from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Optional

import numpy as np
try:
    import supervision as sv
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional dependency
    sv = None  # type: ignore[assignment]
    _SV_IMPORT_ERROR = exc
else:  # pragma: no cover - import success exercised in environments with dependency
    _SV_IMPORT_ERROR = None

from integra_pose.utils.roi_drawing_tool import draw_roi


def _missing_supervision_message() -> str:
    missing = _SV_IMPORT_ERROR.name if _SV_IMPORT_ERROR and _SV_IMPORT_ERROR.name else "supervision"
    return (
        f"Optional dependency '{missing}' is required for the Zone Counter plugin. "
        "Install it to enable this feature."
    )


class ZoneCounterWindow(tk.Toplevel):
    def __init__(self, parent: tk.Misc, on_draw_zone, on_reset_count, get_display_text) -> None:
        super().__init__(parent)
        self.title("Zone Counter")
        self.geometry("260x150")
        self.resizable(False, False)

        self._get_display_text = get_display_text

        self.count_var = tk.StringVar(value=get_display_text())
        count_label = ttk.Label(self, textvariable=self.count_var, font=("TkDefaultFont", 12, "bold"))
        count_label.pack(pady=(12, 6))

        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=12)

        draw_button = ttk.Button(button_frame, text="Draw Zone", command=on_draw_zone)
        draw_button.pack(side=tk.LEFT, expand=True, fill=tk.X)

        if on_reset_count is not None:
            reset_button = ttk.Button(button_frame, text="Reset Count", command=on_reset_count)
            reset_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(8, 0))

        instructions = ttk.Label(
            self,
            text="Open ROI tool, press Enter to confirm polygon or Esc to cancel.",
            wraplength=220,
            justify=tk.CENTER,
        )
        instructions.pack(padx=12, pady=(10, 0))

    def update_display(self, text: str) -> None:
        self.count_var.set(text)


class IntegraPosePlugin:
    POLL_INTERVAL_MS = 100
    OVERLAY_KEY = "zone_counter"

    def __init__(self) -> None:
        self.main_app: Any = None
        self._window: Optional[ZoneCounterWindow] = None
        self._poll_job: Optional[str] = None
        self._normalized_polygon = np.array(
            [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]], dtype=np.float32
        )
        self._frame_resolution_wh: Optional[tuple[int, int]] = None
        self._zone: Optional[sv.PolygonZone] = None
        self._zone_annotator: Optional[sv.PolygonZoneAnnotator] = None
        self._overlay_registered_runner: Any = None
        self._overlay_support_warning_emitted = False
        self._current_count_display = "Count: 0"

    def register(self, main_app: Any) -> None:
        self.main_app = main_app
        self._log("Registering Zone Counter plugin", "INFO")

        menu = getattr(self.main_app, "plugins_menu", None)
        if menu is None:
            self._log("plugins_menu not available; cannot register Zone Counter plugin", "ERROR")
            return

        menu.add_command(label="Zone Counter", command=self._open_window)
        self._log("Zone Counter plugin registered", "INFO")

    def _open_window(self) -> None:
        if not self._ensure_supervision():
            return

        if self._window and self._window.winfo_exists():
            self._window.lift()
            self._window.focus_force()
            return

        root = getattr(self.main_app, "root", None)
        if root is None:
            self._log("Application root not available; cannot open Zone Counter window", "ERROR")
            return

        self._window = ZoneCounterWindow(root, self._draw_zone, self._reset_count, self._get_display_text)
        self._window.bind("<Destroy>", self._on_window_destroyed, add="+")
        self._window.update_display(self._current_count_display)
        self._schedule_poll()

    def _on_window_destroyed(self, event) -> None:
        if event.widget is self._window:
            self._cancel_poll()
            self._window = None
            self._set_display_text("Count: 0", None)
            self._update_overlay_hook(None)

    def _schedule_poll(self) -> None:
        if self._poll_job is not None:
            return
        window = self._window
        if window is None or not window.winfo_exists():
            return
        self._poll_job = window.after(self.POLL_INTERVAL_MS, self._update_count)

    def _cancel_poll(self) -> None:
        if self._poll_job is None:
            return
        window = self._window
        try:
            if window and window.winfo_exists():
                window.after_cancel(self._poll_job)
        except Exception:
            pass
        self._poll_job = None

    def _update_count(self) -> None:
        self._poll_job = None
        window = self._window
        if window is None or not window.winfo_exists():
            self._window = None
            self._update_overlay_hook(None)
            return

        runner = getattr(self.main_app, "_active_supervision_runner", None)
        if runner is None:
            self._set_display_text("Count: 0", window)
            self._update_overlay_hook(None)
            self._schedule_poll()
            return

        frame = self._safe_runner_call(runner, "get_current_frame")
        if frame is not None and hasattr(frame, "shape") and len(frame.shape) >= 2:
            resolution = (int(frame.shape[1]), int(frame.shape[0]))
            self._ensure_zone(resolution)

        detections = self._safe_runner_call(runner, "get_current_detections")
        display_text = self._update_zone_count(detections)
        self._set_display_text(display_text, window)
        self._update_overlay_hook(runner)
        self._schedule_poll()

    def _safe_runner_call(self, runner: Any, method_name: str) -> Any:
        method = getattr(runner, method_name, None)
        if not callable(method):
            return None
        try:
            return method()
        except Exception as exc:
            self._log(f"Zone Counter could not execute {method_name}: {exc}", "ERROR")
            return None

    def _set_display_text(self, text: str, window: Optional[ZoneCounterWindow]) -> None:
        self._current_count_display = text
        if window:
            window.update_display(text)

    def _get_display_text(self) -> str:
        return self._current_count_display

    def _update_zone_count(self, detections: Any) -> str:
        if sv is None:
            return "Count: 0"
        if self._zone is None:
            return "Count: 0"

        have_detections = False
        if detections is not None:
            try:
                have_detections = len(detections) > 0
            except Exception:
                have_detections = False
        if have_detections:
            try:
                self._zone.trigger(detections=detections)
            except Exception as exc:
                self._log(f"Zone Counter failed to process detections: {exc}", "ERROR")

        return f"Count: {self._zone.current_count if self._zone is not None else 0}"

    def _draw_zone(self) -> None:
        if not self._ensure_supervision():
            return

        runner = getattr(self.main_app, "_active_supervision_runner", None)
        if runner is None:
            self._inform_user("Start an inference run before drawing a zone.", level="warning")
            return

        get_frame = getattr(runner, "get_current_frame", None)
        if not callable(get_frame):
            self._inform_user("Active runner does not expose a frame capture API.", level="error")
            return

        try:
            frame = get_frame()
        except Exception as exc:
            self._log(f"Zone Counter could not read current frame: {exc}", "ERROR")
            self._inform_user("Could not capture a frame for zone drawing.", level="error")
            return

        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            self._inform_user("No frame is available yet. Try again once inference has produced output.", level="warning")
            return

        resolution = (int(frame.shape[1]), int(frame.shape[0]))
        points = None
        try:
            points = draw_roi(
                frame.copy(),
                master=getattr(self.main_app, "root", None),
                title="Draw Zone",
            )
        except Exception as exc:
            self._log(f"Zone Counter ROI tool failed: {exc}", "ERROR")
            self._inform_user("Zone drawing failed to launch. Check logs for details.", level="error")
            return

        if not points:
            self._log("Zone drawing cancelled or no points captured.", "INFO")
            return

        polygon = np.array(points, dtype=np.float32)
        if len(polygon) > 1 and np.allclose(polygon[0], polygon[-1]):
            polygon = polygon[:-1]

        if len(polygon) < 3:
            self._inform_user("Define at least three vertices to create a zone.", level="warning")
            return

        width, height = resolution
        scale = np.array([width, height], dtype=np.float32)
        self._normalized_polygon = np.clip(polygon, [0.0, 0.0], scale) / scale
        self._frame_resolution_wh = None
        self._ensure_zone(resolution)
        self._set_display_text("Count: 0", self._window if self._window and self._window.winfo_exists() else None)
        self._update_overlay_hook(runner)
        self._schedule_poll()
        self._log(f"Zone updated with {len(polygon)} vertices.", "INFO")

    def _reset_count(self) -> None:
        if sv is None:
            return
        if self._zone is None:
            self._inform_user("Draw a zone before resetting the count.", level="warning")
            return

        resolution = self._frame_resolution_wh
        self._zone = None
        self._zone_annotator = None
        self._frame_resolution_wh = None
        if resolution is not None:
            self._ensure_zone(resolution)
        self._set_display_text("Count: 0", self._window if self._window and self._window.winfo_exists() else None)
        runner = getattr(self.main_app, "_active_supervision_runner", None)
        self._update_overlay_hook(runner if runner is not None else None)
        self._log("Zone counter reset.", "INFO")

    def _ensure_zone(self, resolution: tuple[int, int]) -> None:
        if sv is None:
            return
        if self._frame_resolution_wh == resolution and self._zone is not None:
            return

        width, height = resolution
        scale = np.array([width, height], dtype=np.float32)
        max_bounds = np.array([max(width - 1, 0), max(height - 1, 0)], dtype=np.float32)
        polygon = np.clip(self._normalized_polygon * scale, [0.0, 0.0], max_bounds)
        polygon = np.round(polygon).astype(np.int32)

        if np.unique(polygon, axis=0).shape[0] < 3:
            self._zone = None
            self._zone_annotator = None
            self._frame_resolution_wh = resolution
            return

        self._zone = sv.PolygonZone(polygon=polygon)
        line_thickness = max(sv.calculate_optimal_line_thickness(resolution_wh=resolution), 1)
        text_scale = max(sv.calculate_optimal_text_scale(resolution_wh=resolution), 0.1)
        self._zone_annotator = sv.PolygonZoneAnnotator(
            zone=self._zone,
            thickness=line_thickness,
            text_thickness=max(line_thickness * 2, 1),
            text_scale=text_scale,
            opacity=0.25,
        )
        self._frame_resolution_wh = resolution
        self._set_display_text("Count: 0", self._window if self._window and self._window.winfo_exists() else None)

    def _overlay_frame(self, frame: np.ndarray) -> np.ndarray:
        if sv is None:
            return frame
        if self._zone is None or self._zone_annotator is None:
            return frame
        try:
            return self._zone_annotator.annotate(scene=frame)
        except Exception as exc:
            self._log(f"Zone overlay failed: {exc}", "ERROR")
            return frame

    def _update_overlay_hook(self, runner: Any) -> None:
        if sv is None:
            return
        if not (self._window and self._window.winfo_exists()):
            if self._overlay_registered_runner is not None:
                register = getattr(self._overlay_registered_runner, "register_frame_overlay", None)
                if callable(register):
                    try:
                        register(self.OVERLAY_KEY, None)
                    except Exception:
                        pass
            self._overlay_registered_runner = None
            return

        if runner is None:
            if self._overlay_registered_runner is not None:
                register = getattr(self._overlay_registered_runner, "register_frame_overlay", None)
                if callable(register):
                    try:
                        register(self.OVERLAY_KEY, None)
                    except Exception:
                        pass
            self._overlay_registered_runner = None
            return

        register = getattr(runner, "register_frame_overlay", None)
        if not callable(register):
            if not self._overlay_support_warning_emitted:
                self._log("Active runner does not support zone overlay visuals; update IntegraPose to a newer build.", "WARNING")
                self._overlay_support_warning_emitted = True
            return

        self._overlay_support_warning_emitted = False
        if self._zone is None or self._zone_annotator is None:
            register(self.OVERLAY_KEY, None)
        else:
            register(self.OVERLAY_KEY, self._overlay_frame)
        self._overlay_registered_runner = runner

    def _inform_user(self, message: str, *, level: str = "info") -> None:
        parent = None
        if self._window and self._window.winfo_exists():
            parent = self._window
        elif hasattr(self.main_app, "root"):
            parent = getattr(self.main_app, "root")

        dialog_map = {
            "info": messagebox.showinfo,
            "warning": messagebox.showwarning,
            "error": messagebox.showerror,
        }
        dialog = dialog_map.get(level, messagebox.showinfo)
        try:
            dialog("Zone Counter", message, parent=parent)
        except Exception:
            pass

    def _log(self, message: str, level: str = "INFO") -> None:
        log_fn = getattr(self.main_app, "log_message", None)
        if callable(log_fn):
            log_fn(message, level)

    def _ensure_supervision(self) -> bool:
        if sv is not None:
            return True
        message = _missing_supervision_message()
        self._log(message, "ERROR")
        self._inform_user(message, level="error")
        return False


def register_plugin(main_app) -> None:
    plugin = IntegraPosePlugin()
    plugin.register(main_app)
