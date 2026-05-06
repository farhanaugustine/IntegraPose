"""Supervision runner lifecycle management (thread + stop signals)."""

from __future__ import annotations

import threading
import traceback
from tkinter import messagebox


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
            try:
                runner.run()
            except Exception as exc:
                app.log_message(
                    f"Supervision inference failed: {exc}\n{traceback.format_exc()}",
                    "ERROR",
                )
                app._set_process_activity("inference", "error")
                app.root.after(0, lambda exc=exc: messagebox.showerror("Inference Error", str(exc), parent=app.root))
            finally:
                self._active_runner = None
                app._active_supervision_runner = None
                self._thread = None
                self._stop_event = None
                app._supervision_thread = None
                app._supervision_stop_event = None
                if on_finish:
                    app.root.after(0, on_finish)

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
