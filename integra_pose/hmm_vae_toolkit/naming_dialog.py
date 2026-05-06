"""ADP-4 Commit G — Cluster naming dialog (per-class sub-cluster naming).

Per the user's locked design (2026-04-26):

  * 3-frame strips per bout (start / middle / end) — disambiguates motion
    without needing video playback.
  * 9 longest bouts per sub-cluster, laid out as a 3×3 grid of triptychs.
  * Names persist to ``state_names.json`` next to the run output.
  * No video playback in v1 (Commit H is the optional follow-up).

The dialog walks the user through one sub-cluster at a time. They type a
name (or skip), click "Save & Next", and on "Save & Close" the entire
``state_names.json`` is written. Existing names from a prior run are
loaded so re-opening the dialog continues where they left off.

The dialog is intentionally lightweight: synchronous frame extraction
(takes a few seconds per cluster), single-pane layout, no fancy
animations. Sufficient for the user's stated mission ("ease of use") and
does not require us to manage a thread pool / Tk-safe image cache.
"""

from __future__ import annotations

import json
import logging
import os
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional

from integra_pose.gui.theme import BOLD_FONT, MUTED_FG, ERROR_FG

logger = logging.getLogger(__name__)


# Filename for the names sidecar. Sibling of run output. Re-opening
# the dialog reads existing names; saving overwrites.
STATE_NAMES_FILENAME = "state_names.json"

# Layout constants — match select_top_n_bouts(n=9) and
# extract_bout_thumbnails(n_per_bout=3).
GRID_ROWS = 3
GRID_COLS = 3


def load_state_names(output_folder: str, expected_run_id: Optional[str] = None) -> dict:
    """Read ``state_names.json`` from ``output_folder``, honouring run_id.

    When ``expected_run_id`` is provided AND the file's stored ``run_id``
    is non-empty AND they don't match, return ``{}`` and log a warning —
    cluster ids likely mean different things between runs.

    A file without a ``run_id`` field (older schema v1 saves) is loaded
    unconditionally — the user's previous workflow shouldn't break just
    because we added the field.
    """
    if not output_folder:
        return {}
    path = os.path.join(output_folder, STATE_NAMES_FILENAME)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return {}
    data = payload.get("state_names")
    if not isinstance(data, dict):
        return {}
    file_run_id = str(payload.get("run_id") or "").strip()
    expected = str(expected_run_id or "").strip()
    # Stale-state guard. The previous check only treated the file as stale
    # when BOTH ids were present and differed. That left a hole: an older
    # state_names.json without a run_id field would be loaded for a
    # brand-new run and silently apply names to clusters that may have
    # shifted (different parameters, different data, different cluster IDs).
    # Now we require an exact match whenever the caller supplies an
    # expected_run_id; missing or mismatched file ids both count as stale.
    if expected:
        if not file_run_id or file_run_id != expected:
            logger.warning(
                "Skipping stale %s (file run_id=%r, expected=%r).",
                STATE_NAMES_FILENAME, file_run_id, expected,
            )
            return {}
    return {str(k): str(v) for k, v in data.items()}


class ClusterNamingDialog(tk.Toplevel):
    """Modal naming dialog walking the user through each sub-cluster.

    Attributes:
        state_names: ``{namespaced_label: name}`` — populated from disk
            on open and written back on save.
        candidates: Ordered list of namespaced labels to walk. Default:
            ranked by signal score (when a signal_report is provided);
            otherwise alphabetical for stability.

    The dialog uses the cached ``last_sub_behavior_result`` from
    BehaviorAnalysisApp, plus the optional signal_report (Commit F)
    so users can name strongest candidates first.
    """

    def __init__(
        self,
        parent: tk.Misc,
        *,
        bouts: list,
        video_path_map: dict,
        class_names: dict,
        output_folder: str,
        signal_report=None,
        on_close=None,
        run_id: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self.title("Name Sub-Behaviors")
        self.transient(parent)
        self.minsize(900, 650)

        self._bouts = bouts or []
        self._video_path_map = video_path_map or {}
        self._class_names = class_names or {}
        self._output_folder = output_folder or "."
        self._on_close_callback = on_close
        # ADP-4: per-run UUID. When set, ``state_names.json`` is stamped
        # with it; the loader skips a names file whose run_id doesn't
        # match (different params → different clusters → stale names).
        self._run_id = str(run_id) if run_id else ""

        # Build the candidate ordering — strongest signal score first when
        # available. Without scores, fall back to whatever order the
        # bouts produced.
        self._candidates = self._build_candidate_list(signal_report)
        self._cursor = 0

        # Existing names from prior run (or empty if first time).
        self._state_names = self._load_existing_names()

        # Image references must outlive the cell they're rendered in,
        # otherwise Tk garbage-collects them and the cells go blank.
        self._image_cache: list = []

        # PIL/cv2 will be imported lazily when needed; bail early if
        # the user has no candidates.
        self._build_ui()

        if not self._candidates:
            messagebox.showinfo(
                "Name Sub-Behaviors",
                "No sub-clusters to name. Run Sub-Behavior Discovery first.",
                parent=self,
            )
            self.destroy()
            return

        self._render_current_cluster()
        self.protocol("WM_DELETE_WINDOW", self._on_window_close)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_candidate_list(self, signal_report) -> list:
        """Return namespaced labels in the order to walk through."""
        if signal_report is not None and signal_report.candidates:
            return [c.namespaced_label for c in signal_report.candidates]
        # Fallback: collect from bouts, sort alphabetically for stability.
        labels = set()
        for b in self._bouts:
            try:
                cid = int(b.get("class_id", -1))
                sid_raw = b.get("state")
                sid = int(sid_raw)
            except Exception:
                continue
            if sid < 0:
                continue  # noise
            labels.add(f"{cid}:{sid}")
        return sorted(labels)

    def _load_existing_names(self) -> dict:
        # Shared helper handles run_id check and schema v1 fallback.
        return load_state_names(self._output_folder, self._run_id)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        # Header — class name + progress + name entry.
        header = ttk.Frame(outer)
        header.pack(fill=tk.X)
        self._title_var = tk.StringVar(value="")
        ttk.Label(header, textvariable=self._title_var, font=BOLD_FONT).pack(
            side=tk.LEFT, anchor=tk.W,
        )
        self._progress_var = tk.StringVar(value="")
        ttk.Label(header, textvariable=self._progress_var, foreground=MUTED_FG).pack(
            side=tk.RIGHT, anchor=tk.E,
        )

        # Name entry row.
        name_row = ttk.Frame(outer)
        name_row.pack(fill=tk.X, pady=(8, 6))
        ttk.Label(name_row, text="Name (e.g., 'forward-locomotion'):").pack(side=tk.LEFT)
        self._name_var = tk.StringVar(value="")
        self._name_entry = ttk.Entry(name_row, textvariable=self._name_var, width=40)
        self._name_entry.pack(side=tk.LEFT, padx=(8, 0), fill=tk.X, expand=True)
        self._name_entry.bind("<Return>", lambda _e: self._on_save_and_next())

        # 3×3 grid of triptychs.
        self._grid_frame = ttk.LabelFrame(
            outer,
            text="9 longest bouts (start / middle / end frame each)",
            padding=8,
        )
        self._grid_frame.pack(fill=tk.BOTH, expand=True, pady=(6, 6))
        self._loading_label = ttk.Label(
            self._grid_frame,
            text="Loading frames...",
            foreground=MUTED_FG,
        )
        self._loading_label.pack(anchor=tk.CENTER, pady=20)

        # Footer buttons.
        footer = ttk.Frame(outer)
        footer.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(footer, text="Skip", command=self._on_skip).pack(side=tk.LEFT)
        ttk.Button(
            footer, text="Save & Close", command=self._on_save_and_close,
        ).pack(side=tk.RIGHT)
        ttk.Button(
            footer,
            text="Save & Next",
            command=self._on_save_and_next,
            style="Accent.TButton",
        ).pack(side=tk.RIGHT, padx=(0, 6))
        ttk.Button(
            footer, text="Previous", command=self._on_previous,
        ).pack(side=tk.RIGHT, padx=(0, 6))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_cluster(self) -> None:
        # Clear grid frame (image cache + child widgets).
        for child in self._grid_frame.winfo_children():
            child.destroy()
        self._image_cache.clear()
        loading = ttk.Label(self._grid_frame, text="Loading frames...", foreground=MUTED_FG)
        loading.pack(anchor=tk.CENTER, pady=20)
        self.update_idletasks()  # show "Loading..." before the heavy work

        ns_label = self._candidates[self._cursor]
        # Update header.
        try:
            class_id, sub_id = ns_label.split(":", 1)
            class_id_int = int(class_id)
            class_name = self._class_names.get(class_id_int, "")
        except Exception:
            class_id_int = -1
            class_name = ""
        if class_name:
            self._title_var.set(
                f"Class: {class_name}   ·   Sub-cluster: {ns_label}"
            )
        else:
            self._title_var.set(f"Sub-cluster: {ns_label}")
        self._progress_var.set(
            f"{self._cursor + 1} of {len(self._candidates)}"
        )
        # Pre-fill the entry with any previously-saved name.
        self._name_var.set(self._state_names.get(ns_label, ""))
        self._name_entry.focus_set()

        # Pull the longest 9 bouts and extract triptychs.
        try:
            from .frame_extraction import (  # noqa: PLC0415
                extract_bout_thumbnails,
                select_top_n_bouts,
            )
        except Exception as exc:
            loading.destroy()
            ttk.Label(
                self._grid_frame,
                text=f"Frame extraction unavailable: {exc}",
                foreground=ERROR_FG,
            ).pack(anchor=tk.CENTER, pady=20)
            return

        top_bouts = select_top_n_bouts(
            self._bouts, n=GRID_ROWS * GRID_COLS, target_state=ns_label,
        )
        thumbs = extract_bout_thumbnails(
            top_bouts, video_path_map=self._video_path_map, n_per_bout=3,
        )

        loading.destroy()

        if not top_bouts:
            ttk.Label(
                self._grid_frame,
                text="(No bouts in this sub-cluster — likely all noise.)",
                foreground=MUTED_FG,
            ).pack(anchor=tk.CENTER, pady=20)
            return

        # Lay out the grid. Each cell is a triptych: 3 thumbnails left
        # to right + a tiny caption (frame range, source video).
        try:
            from PIL import ImageTk  # noqa: PLC0415
        except Exception as exc:
            ttk.Label(
                self._grid_frame,
                text=f"Pillow ImageTk unavailable: {exc}",
                foreground=ERROR_FG,
            ).pack(anchor=tk.CENTER, pady=20)
            return

        for idx, bout in enumerate(top_bouts):
            row = idx // GRID_COLS
            col = idx % GRID_COLS
            cell = ttk.Frame(self._grid_frame, padding=4)
            cell.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")

            # Triptych row.
            trip_row = ttk.Frame(cell)
            trip_row.pack(fill=tk.X)
            entry = thumbs[idx] if idx < len(thumbs) else {"frames": []}
            for img in entry.get("frames", []) or []:
                try:
                    photo = ImageTk.PhotoImage(img)
                    self._image_cache.append(photo)
                    ttk.Label(trip_row, image=photo).pack(side=tk.LEFT, padx=2)
                except Exception:
                    continue
            if not entry.get("frames"):
                ttk.Label(trip_row, text="(no frames)", foreground=ERROR_FG).pack()

            # Caption.
            caption = (
                f"frames {bout.get('start_frame', '?')}–{bout.get('end_frame', '?')}  ·  "
                f"{bout.get('duration_frames', '?')}f  ·  "
                f"{os.path.basename(str(bout.get('directory', '')))}"
            )
            ttk.Label(cell, text=caption, foreground=MUTED_FG).pack(anchor=tk.W)

        for r in range(GRID_ROWS):
            self._grid_frame.rowconfigure(r, weight=1)
        for c in range(GRID_COLS):
            self._grid_frame.columnconfigure(c, weight=1)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _commit_current_name(self) -> None:
        """Stash the current name into ``_state_names`` (in memory)."""
        ns_label = self._candidates[self._cursor]
        text = self._name_var.get().strip()
        if text:
            self._state_names[ns_label] = text
        else:
            # Empty string clears any prior name.
            self._state_names.pop(ns_label, None)

    def _on_save_and_next(self) -> None:
        self._commit_current_name()
        if self._cursor + 1 >= len(self._candidates):
            self._on_save_and_close()
            return
        self._cursor += 1
        self._render_current_cluster()

    def _on_skip(self) -> None:
        # Skip = leave current name unchanged, advance.
        if self._cursor + 1 >= len(self._candidates):
            self._on_save_and_close()
            return
        self._cursor += 1
        self._render_current_cluster()

    def _on_previous(self) -> None:
        self._commit_current_name()
        self._cursor = max(0, self._cursor - 1)
        self._render_current_cluster()

    def _on_save_and_close(self) -> None:
        self._commit_current_name()
        self._save_to_disk()
        self._invoke_close_callback()
        self.destroy()

    def _on_window_close(self) -> None:
        # User clicked the X — give them a chance to cancel.
        if not self._state_names:
            self.destroy()
            return
        if messagebox.askyesno(
            "Save Names?",
            "Save the names you've entered before closing?",
            parent=self,
        ):
            self._save_to_disk()
        self._invoke_close_callback()
        self.destroy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_to_disk(self) -> None:
        from datetime import datetime, timezone  # noqa: PLC0415

        try:
            os.makedirs(self._output_folder, exist_ok=True)
        except Exception:
            pass
        path = os.path.join(self._output_folder, STATE_NAMES_FILENAME)
        payload = {
            "schema_version": 2,
            "named_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": self._run_id,
            "state_names": dict(self._state_names),
        }
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            logger.info("Saved %d sub-cluster name(s) to %s", len(self._state_names), path)
        except Exception as exc:
            logger.error("Could not write %s: %s", path, exc)
            messagebox.showerror(
                "Save Names",
                f"Could not write names file: {exc}",
                parent=self,
            )

    def _invoke_close_callback(self) -> None:
        cb = self._on_close_callback
        if cb is None:
            return
        try:
            cb(dict(self._state_names))
        except Exception as exc:
            logger.warning("on_close callback raised: %s", exc)


def open_naming_dialog(parent, **kwargs) -> ClusterNamingDialog:
    """Convenience constructor — same args as ``ClusterNamingDialog``."""
    return ClusterNamingDialog(parent, **kwargs)


__all__ = [
    "ClusterNamingDialog",
    "open_naming_dialog",
    "load_state_names",
    "STATE_NAMES_FILENAME",
]
