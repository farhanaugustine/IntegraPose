"""Sanity-check dialog — Toplevel that runs the runner live and reports.

Public API:

    open_sanity_check_dialog(parent)

The dialog fires the runner on a background thread so the UI stays
responsive (the runner's ~2 s might still feel sticky on slow disks).
Per-stage status updates land on the main thread via ``after()`` so
Tk's single-thread invariant is respected.
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional

from integra_pose.sanity_check.runner import (
    SanityReport,
    StageResult,
    run_sanity_check,
)


_STATUS_BADGE = {
    "running": ("⏳", "#888888"),
    "ok": ("✔", "#0a8a0a"),
    "warn": ("⚠", "#b07a00"),
    "fail": ("✘", "#c01515"),
    "queued": ("○", "#888888"),
}


# Stage-name list mirrors runner._DEFAULT_STAGES so the dialog can render
# rows before the runner has produced any results.
_STAGE_LABELS = (
    "Runtime inventory",
    "GUI smoke test",
    "YOLO label parsing",
    "Bout analyzer end-to-end",
    "Atomic file IO",
)


class _SanityCheckDialog(tk.Toplevel):
    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.title("IntegraPose — Sanity Check")
        self.transient(parent)
        self.minsize(560, 420)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._row_labels: dict[str, ttk.Label] = {}
        self._row_summaries: dict[str, tk.StringVar] = {}
        self._final_report: Optional[SanityReport] = None
        self._worker: Optional[threading.Thread] = None
        self._closing = False

        self._build_ui()
        self._centre_on_parent(parent)
        # Kick the runner immediately — the dialog's job is to report,
        # not to demand another click.
        self.after(50, self._start_worker)

    # ------------------------------------------------------------------
    # UI build
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)

        intro = ttk.Label(
            outer,
            text=(
                "Verifying that this IntegraPose install can run a complete "
                "non-ML pipeline on a synthetic batch. Each stage is "
                "independent — one failure does not block the rest."
            ),
            wraplength=520,
            justify=tk.LEFT,
        )
        intro.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        # Per-stage status strip.
        strip = ttk.Frame(outer)
        strip.grid(row=1, column=0, sticky="ew")
        strip.columnconfigure(2, weight=1)
        for idx, label_text in enumerate(_STAGE_LABELS):
            badge, colour = _STATUS_BADGE["queued"]
            row_badge = ttk.Label(strip, text=badge, foreground=colour, width=2)
            row_badge.grid(row=idx, column=0, sticky="w", padx=(0, 6), pady=2)
            row_label = ttk.Label(strip, text=label_text, anchor="w")
            row_label.grid(row=idx, column=1, sticky="w", padx=(0, 12), pady=2)
            summary_var = tk.StringVar(value="(queued)")
            row_summary = ttk.Label(strip, textvariable=summary_var, foreground="#555")
            row_summary.grid(row=idx, column=2, sticky="ew", pady=2)
            self._row_labels[label_text] = row_badge
            self._row_summaries[label_text] = summary_var

        # Detailed report text box (readonly).
        details_frame = ttk.LabelFrame(outer, text="Detailed report", padding=6)
        details_frame.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        details_frame.columnconfigure(0, weight=1)
        details_frame.rowconfigure(0, weight=1)

        self._details_text = tk.Text(
            details_frame,
            wrap=tk.NONE,
            font=("Consolas" if tk.Tcl().eval("info exists tcl_platform(os)") else "Courier", 9),
            background="#fafafa",
            relief="flat",
            state="disabled",
        )
        self._details_text.grid(row=0, column=0, sticky="nsew")
        details_scroll = ttk.Scrollbar(details_frame, orient=tk.VERTICAL, command=self._details_text.yview)
        details_scroll.grid(row=0, column=1, sticky="ns")
        self._details_text.configure(yscrollcommand=details_scroll.set)

        # Footer buttons.
        footer = ttk.Frame(outer)
        footer.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        footer.columnconfigure(0, weight=1)
        self._copy_button = ttk.Button(
            footer, text="Copy report to clipboard", command=self._on_copy_clicked, state=tk.DISABLED
        )
        self._copy_button.grid(row=0, column=1, sticky="e", padx=(0, 6))
        self._close_button = ttk.Button(footer, text="Close", command=self._on_close)
        self._close_button.grid(row=0, column=2, sticky="e")

    def _centre_on_parent(self, parent: tk.Misc) -> None:
        try:
            self.update_idletasks()
            top = parent.winfo_toplevel() if parent is not None else None
            if top is not None:
                px = top.winfo_rootx()
                py = top.winfo_rooty()
                pw = top.winfo_width()
                ph = top.winfo_height()
            else:
                px = py = 0
                pw = self.winfo_screenwidth()
                ph = self.winfo_screenheight()
            x = px + max(0, (pw - self.winfo_width()) // 2)
            y = py + max(0, (ph - self.winfo_height()) // 2)
            self.geometry(f"+{int(x)}+{int(y)}")
        except Exception:  # pragma: no cover - centering best-effort
            pass

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _start_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._run_in_thread, name="sanity-check-runner", daemon=True
        )
        self._worker.start()

    def _run_in_thread(self) -> None:
        # Marshal each progress callback into the main thread via after(0, ...).
        def _on_progress(stage_name: str, status: str) -> None:
            if self._closing:
                return
            self.after(0, lambda n=stage_name, s=status: self._update_row(n, s))

        try:
            report = run_sanity_check(progress_callback=_on_progress)
        except Exception as exc:  # pragma: no cover — runner already wraps
            self.after(0, lambda e=exc: self._on_unexpected_error(e))
            return
        if not self._closing:
            self.after(0, lambda r=report: self._on_finished(r))

    def _update_row(self, stage_name: str, status: str) -> None:
        badge_widget = self._row_labels.get(stage_name)
        summary_var = self._row_summaries.get(stage_name)
        if badge_widget is None or summary_var is None:
            return
        badge_text, colour = _STATUS_BADGE.get(status, _STATUS_BADGE["queued"])
        try:
            badge_widget.configure(text=badge_text, foreground=colour)
        except Exception:
            pass
        if status == "running":
            summary_var.set("running…")

    def _on_finished(self, report: SanityReport) -> None:
        self._final_report = report
        # Patch each row's summary with the resolved one-liner.
        for stage in report.stages:
            badge_widget = self._row_labels.get(stage.name)
            summary_var = self._row_summaries.get(stage.name)
            if badge_widget is not None:
                badge_text, colour = _STATUS_BADGE.get(stage.status, _STATUS_BADGE["queued"])
                try:
                    badge_widget.configure(text=badge_text, foreground=colour)
                except Exception:
                    pass
            if summary_var is not None:
                summary_var.set(stage.summary)

        # Render full text into the details box.
        try:
            self._details_text.configure(state="normal")
            self._details_text.delete("1.0", tk.END)
            self._details_text.insert("1.0", report.as_text())
            self._details_text.configure(state="disabled")
        except Exception:
            pass
        try:
            self._copy_button.configure(state=tk.NORMAL)
        except Exception:
            pass

    def _on_unexpected_error(self, exc: Exception) -> None:  # pragma: no cover
        try:
            self._details_text.configure(state="normal")
            self._details_text.delete("1.0", tk.END)
            self._details_text.insert(
                "1.0",
                f"Unexpected error inside the sanity-check runner: {exc!r}\n",
            )
            self._details_text.configure(state="disabled")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Footer button handlers
    # ------------------------------------------------------------------

    def _on_copy_clicked(self) -> None:
        if self._final_report is None:
            return
        try:
            self.clipboard_clear()
            self.clipboard_append(self._final_report.as_text())
            self.update()  # ensures clipboard is committed
            messagebox.showinfo(
                "Sanity Check",
                "Report copied to clipboard. Paste it into an issue or "
                "email when reporting an install problem.",
                parent=self,
            )
        except Exception as exc:
            messagebox.showerror("Sanity Check", f"Could not copy: {exc}", parent=self)

    def _on_close(self) -> None:
        self._closing = True
        try:
            self.destroy()
        except Exception:
            pass


def open_sanity_check_dialog(parent: tk.Misc) -> _SanityCheckDialog:
    """Open the sanity-check dialog. Returns the Toplevel for testing hooks."""
    return _SanityCheckDialog(parent)


__all__ = ["open_sanity_check_dialog"]
