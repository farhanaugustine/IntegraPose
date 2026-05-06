"""Application lifecycle helpers for title/status/log and clean shutdown."""

from __future__ import annotations

from tkinter import messagebox


class AppLifecycleService:
    """Handles status/log/title updates and clean shutdown."""

    def __init__(self, app) -> None:
        self.app = app

    def stop_all_processes(self):
        self.app.update_status("Stopping all background processes...")
        self.app.process_manager.terminate_process("training_process")
        self.app.process_manager.terminate_process("file_inference_process")
        self.app.process_manager.terminate_process("webcam_inference_process")
        self.app.log_message("All processes stopped.", "INFO")

    def on_close(self):
        if messagebox.askokcancel("Quit", "Are you sure you want to quit?", parent=self.app.root):
            self.stop_all_processes()
            self.app.root.destroy()

    def update_title(self):
        try:
            title = self.app._compose_window_title()
        except Exception:
            title = self.app.base_title
        self.app.root.title(title)

    def update_status(self, message: str):
        self.app.status_var.set(message)
        self.app.log_message(f"Status updated: {message}", "INFO")

    def update_log(self, message: str, level: str = "INFO"):
        self.app.log_queue.put(f"[{level}] {message}")
