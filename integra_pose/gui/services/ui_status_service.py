"""UI helpers for status/toasts and safe status updates."""

from __future__ import annotations

import tkinter as tk


class UIStatusService:
    """Encapsulates status updates and transient toasts."""

    def __init__(self, app) -> None:
        self.app = app

    def set_preprocessing_status(self, text: str) -> None:
        try:
            self.app.root.after(0, lambda: self.app.data_preprocessing_status.set(text))
            if hasattr(self.app, "status_var"):
                self.app.root.after(0, lambda: self.app.status_var.set(text))
        except Exception:
            pass

    def set_job_status(self, text: str) -> None:
        try:
            self.app.root.after(0, lambda: self.app.job_status_var.set(text))
        except Exception:
            pass

    def toast(self, message: str, *, level: str = "info", duration_ms: int = 2500) -> None:
        try:
            toast = tk.Toplevel(self.app.root)
            toast.overrideredirect(True)
            toast.attributes("-topmost", True)
            bg = "#0f172a" if level == "info" else "#a40000" if level == "error" else "#92400e"
            fg = "#fafafa"
            frame = tk.Frame(toast, bg=bg, padx=12, pady=8)
            tk.Label(frame, text=message, bg=bg, fg=fg, font=("Segoe UI", 9)).pack()
            frame.pack()
            toast.update_idletasks()
            x = self.app.root.winfo_rootx() + self.app.root.winfo_width() - toast.winfo_width() - 20
            y = self.app.root.winfo_rooty() + 20
            toast.geometry(f"+{x}+{y}")
            toast.after(duration_ms, toast.destroy)
        except Exception:
            pass
