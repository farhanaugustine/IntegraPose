"""Supervision runner lifecycle management (thread + stop signals)."""

from __future__ import annotations

import threading
import traceback
from tkinter import messagebox

from integra_pose.utils.operation_result import OperationResult


class InferenceLifecycleService:
    """Owns the Supervision runner thread/stop event and UI cleanup hooks."""

    def __init__(self, app) -> None:
        self.app = app
        self._thread = None
        self._stop_event = None
        self._active_runner = None

    @property
    def active_runner(self):
        return self._active_runner

    def start(self, runner, on_finish):
        app = self.app
        if self._thread and self._thread.is_alive():
            app.log_message("Supervision inference is already running.", "WARNING")
            return False

        self._stop_event = runner.stop_event if hasattr(runner, "stop_event") else None
        self._active_runner = runner
        app._active_supervision_runner = runner
        app._supervision_stop_event = self._stop_event

        def _worker():
            result = OperationResult.success("Inference completed successfully.")
            error_for_dialog = None
            error_traceback = ""
            try:
                runner.run()
            except Exception as exc:
                result = OperationResult.failure("Inference failed.", error=str(exc))
                error_for_dialog = exc
                error_traceback = traceback.format_exc()
            finally:
                self._active_runner = None
                app._active_supervision_runner = None
                self._thread = None
                self._stop_event = None
                app._supervision_thread = None
                app._supervision_stop_event = None
                if on_finish:
                    try:
                        app.root.after(0, lambda result=result: on_finish(result))
                    except Exception as exc:
                        app.log_message(f"Failed to schedule inference completion callback: {exc}", "ERROR")
                if error_for_dialog is not None:
                    app.log_message(
                        f"Supervision inference failed: {error_for_dialog}\n{error_traceback}",
                        "ERROR",
                    )
                    app._set_process_activity("inference", "error")
                    try:
                        app.root.after(
                            0,
                            lambda exc=error_for_dialog: messagebox.showerror(
                                "Inference Error",
                                str(exc),
                                parent=app.root,
                            ),
                        )
                    except Exception as exc:
                        app.log_message(f"Failed to schedule inference error dialog: {exc}", "ERROR")

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()
        app._supervision_thread = self._thread
        return True

    def stop(self):
        if self._thread and self._thread.is_alive():
            if self._stop_event:
                self._stop_event.set()
            self.app.log_message("Signaled Supervision inference runner to stop.", "INFO")

    def clear(self):
        self._active_runner = None
        self._thread = None
        self._stop_event = None
        self.app._active_supervision_runner = None
        self.app._supervision_thread = None
        self.app._supervision_stop_event = None
