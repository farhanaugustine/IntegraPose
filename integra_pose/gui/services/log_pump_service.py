"""Background log queue pump to update the UI safely."""

from __future__ import annotations

import tkinter as tk


class LogPumpService:
    def __init__(self, app, interval_ms: int = 300) -> None:
        self.app = app
        self.interval_ms = interval_ms
        self._job = None

    def start(self):
        self._pump()

    def stop(self):
        if self._job is not None:
            try:
                self.app.root.after_cancel(self._job)
            except Exception:
                pass
            self._job = None

    def _pump(self):
        app = self.app
        while not app.log_queue.empty():
            line = app.log_queue.get_nowait()
            if getattr(app, "TERMINAL_LOG_OUTPUT", False):
                print(line)
            ingest = getattr(app, "_ingest_log_line", None)
            if callable(ingest):
                try:
                    ingest(line)
                except Exception:
                    pass
            if hasattr(app, "log_text_widget"):
                try:
                    app.log_text_widget.config(state=tk.NORMAL)
                    app.log_text_widget.insert(tk.END, line + "\n")
                    app.log_text_widget.see(tk.END)
                    app.log_text_widget.config(state=tk.DISABLED)
                except Exception:
                    pass
        self._job = app.root.after(self.interval_ms, self._pump)
