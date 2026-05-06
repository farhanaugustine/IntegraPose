from __future__ import annotations

import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional

import cv2
import pandas as pd
from PIL import Image, ImageTk

from integra_pose.gui.windowing import apply_adaptive_window_geometry

ROI_EVENT_COLUMNS = [
    "Event ID",
    "Source",
    "Event Type",
    "Target Name",
    "Track ID",
    "Frame",
    "Review Status",
    "Corrected Event Type",
    "Corrected Target Name",
    "Corrected Track ID",
    "Corrected Frame",
    "Reviewer Notes",
    "Corrected Manually",
    "Original Event Type",
    "Original Target Name",
    "Original Track ID",
    "Original Frame",
]


def _coerce_int(value, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def normalize_roi_event_dataframe(events) -> pd.DataFrame:
    if isinstance(events, pd.DataFrame):
        work = events.copy()
    elif isinstance(events, dict):
        rows: list[dict[str, object]] = []
        for event_type, review_type in (("entries", "entry"), ("exits", "exit")):
            for event in list(events.get(event_type, []) or []):
                if not isinstance(event, dict):
                    continue
                target_name = str(event.get("roi_name", "") or "").strip()
                frame = pd.to_numeric(event.get("frame"), errors="coerce")
                if not target_name or pd.isna(frame):
                    continue
                rows.append(
                    {
                        "Source": "zone",
                        "Event Type": review_type,
                        "Target Name": target_name,
                        "Track ID": _coerce_int(event.get("track_id"), 0),
                        "Frame": int(frame),
                    }
                )
        work = pd.DataFrame(rows)
    else:
        work = pd.DataFrame()

    if work.empty:
        return pd.DataFrame(columns=ROI_EVENT_COLUMNS)

    if "Source" not in work.columns:
        work["Source"] = "zone"
    if "Event Type" not in work.columns and "event_type" in work.columns:
        work["Event Type"] = work["event_type"]
    if "Target Name" not in work.columns and "roi_name" in work.columns:
        work["Target Name"] = work["roi_name"]
    if "Track ID" not in work.columns and "track_id" in work.columns:
        work["Track ID"] = work["track_id"]
    if "Frame" not in work.columns and "frame" in work.columns:
        work["Frame"] = work["frame"]

    normalized_rows: list[dict[str, object]] = []
    for idx, (_, row) in enumerate(work.iterrows(), start=1):
        event_type = str(row.get("Event Type", "") or "").strip().lower()
        target_name = str(row.get("Target Name", "") or "").strip()
        frame = _coerce_int(row.get("Frame"), -1)
        if event_type not in {"entry", "exit", "null"}:
            continue
        if not target_name and event_type != "null":
            continue
        if frame < 0:
            continue
        review_status = str(row.get("Review Status", "") or "").strip().lower() or "detected"
        corrected_event_type = str(row.get("Corrected Event Type", "") or "").strip().lower() or event_type
        corrected_target_name = str(row.get("Corrected Target Name", "") or "").strip() or target_name
        corrected_track_id = _coerce_int(row.get("Corrected Track ID"), _coerce_int(row.get("Track ID"), 0))
        corrected_frame = _coerce_int(row.get("Corrected Frame"), frame)
        if corrected_event_type == "null":
            corrected_target_name = ""
        normalized_rows.append(
            {
                "Event ID": _coerce_int(row.get("Event ID"), idx),
                "Source": str(row.get("Source", "zone") or "zone").strip() or "zone",
                "Event Type": event_type,
                "Target Name": target_name,
                "Track ID": _coerce_int(row.get("Track ID"), 0),
                "Frame": frame,
                "Review Status": review_status,
                "Corrected Event Type": corrected_event_type,
                "Corrected Target Name": corrected_target_name,
                "Corrected Track ID": corrected_track_id,
                "Corrected Frame": corrected_frame,
                "Reviewer Notes": str(row.get("Reviewer Notes", "") or "").strip(),
                "Corrected Manually": bool(row.get("Corrected Manually", False)),
                "Original Event Type": str(row.get("Original Event Type", event_type) or event_type).strip().lower(),
                "Original Target Name": str(row.get("Original Target Name", target_name) or target_name).strip(),
                "Original Track ID": _coerce_int(row.get("Original Track ID"), _coerce_int(row.get("Track ID"), 0)),
                "Original Frame": _coerce_int(row.get("Original Frame"), frame),
            }
        )

    normalized = pd.DataFrame(normalized_rows, columns=ROI_EVENT_COLUMNS)
    if normalized.empty:
        return normalized
    normalized.sort_values(["Frame", "Track ID", "Target Name", "Event Type"], inplace=True, kind="stable")
    normalized.reset_index(drop=True, inplace=True)
    normalized["Event ID"] = list(range(1, len(normalized) + 1))
    return normalized


class ROIEventReviewTool(tk.Toplevel):
    DISPLAY_COLUMNS = ("Track ID", "Event Type", "Target Name", "Frame", "Review Status", "Corrected")

    def __init__(
        self,
        parent,
        *,
        video_path: str,
        events_df: pd.DataFrame | None,
        autosave_path: str | None = None,
        on_review_saved: Optional[Callable[[pd.DataFrame], None]] = None,
    ) -> None:
        super().__init__(parent)
        self.title("ROI Entry/Exit Reviewer")
        apply_adaptive_window_geometry(
            self,
            preferred_size=(1320, 860),
            min_size=(1120, 760),
        )
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.video_path = str(video_path or "").strip()
        self.autosave_path = str(autosave_path or "").strip()
        self.on_review_saved = on_review_saved

        self.cap = None
        self.total_frames = 0
        self.video_fps = 30.0
        self.current_frame_index = 0
        self.is_playing = False
        self._scale_callback_suppressed = False
        self._current_photo = None
        self._selected_event_id: int | None = None

        self.working_events_df = normalize_roi_event_dataframe(events_df)
        self.target_names = sorted(
            {
                str(value).strip()
                for value in self.working_events_df.get("Target Name", pd.Series(dtype=object)).dropna().tolist()
                if str(value).strip()
            }
        )

        self.filter_target_var = tk.StringVar(value="All")
        self.filter_event_type_var = tk.StringVar(value="All")
        self.filter_status_var = tk.StringVar(value="All")
        self.event_type_var = tk.StringVar(value="entry")
        self.target_name_var = tk.StringVar(value=self.target_names[0] if self.target_names else "")
        self.track_id_var = tk.StringVar(value="0")
        self.frame_var = tk.StringVar(value="0")
        self.status_var = tk.StringVar(value="Ready.")
        self.review_status_var = tk.StringVar(value="detected")
        self.time_label_var = tk.StringVar(value="F: 0 / 0 | T: 00.00s / 00.00s")

        self._create_widgets()
        self._load_video()
        self._refresh_results_tree()
        if self.results_tree.get_children():
            self._select_event_by_id(1, seek=True)

    def _create_widgets(self) -> None:
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=1)

        video_frame = ttk.Frame(main_frame)
        video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        video_frame.rowconfigure(0, weight=1)
        video_frame.columnconfigure(0, weight=1)

        self.video_label = ttk.Label(video_frame, background="black", anchor=tk.CENTER)
        self.video_label.grid(row=0, column=0, sticky="nsew")

        controls_frame = ttk.Frame(video_frame)
        controls_frame.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        controls_frame.columnconfigure(3, weight=1)
        ttk.Button(controls_frame, text="Prev Frame", command=lambda: self._seek_relative(-1)).grid(row=0, column=0, padx=(0, 4))
        self.play_pause_btn = ttk.Button(controls_frame, text="Play", command=self._toggle_play_pause, state=tk.DISABLED)
        self.play_pause_btn.grid(row=0, column=1, padx=4)
        ttk.Button(controls_frame, text="Next Frame", command=lambda: self._seek_relative(1)).grid(row=0, column=2, padx=4)
        self.video_scale = ttk.Scale(controls_frame, from_=0, to=1, orient=tk.HORIZONTAL, command=self._on_scale_drag, state=tk.DISABLED)
        self.video_scale.grid(row=0, column=3, sticky="ew", padx=6)
        ttk.Label(controls_frame, textvariable=self.time_label_var).grid(row=0, column=4, padx=(6, 0))

        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky="nsew")
        right_panel.rowconfigure(2, weight=1)
        right_panel.columnconfigure(0, weight=1)

        filter_frame = ttk.LabelFrame(right_panel, text="Filters", padding=10)
        filter_frame.grid(row=0, column=0, sticky="ew")
        ttk.Label(filter_frame, text="ROI:").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=2)
        target_combo = ttk.Combobox(filter_frame, textvariable=self.filter_target_var, state="readonly")
        target_combo.grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Label(filter_frame, text="Event:").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=2)
        event_combo = ttk.Combobox(filter_frame, textvariable=self.filter_event_type_var, state="readonly", values=("All", "entry", "exit", "null"))
        event_combo.grid(row=1, column=1, sticky="ew", pady=2)
        ttk.Label(filter_frame, text="Review:").grid(row=2, column=0, sticky="w", padx=(0, 6), pady=2)
        status_combo = ttk.Combobox(filter_frame, textvariable=self.filter_status_var, state="readonly", values=("All", "detected", "confirmed", "corrected"))
        status_combo.grid(row=2, column=1, sticky="ew", pady=2)
        filter_frame.columnconfigure(1, weight=1)
        target_combo["values"] = ["All", *self.target_names]
        target_combo.bind("<<ComboboxSelected>>", self._refresh_results_tree)
        event_combo.bind("<<ComboboxSelected>>", self._refresh_results_tree)
        status_combo.bind("<<ComboboxSelected>>", self._refresh_results_tree)

        editor_frame = ttk.LabelFrame(right_panel, text="Correction Editor", padding=10)
        editor_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        editor_frame.columnconfigure(1, weight=1)

        ttk.Label(editor_frame, text="Corrected event:").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=3)
        event_type_combo = ttk.Combobox(editor_frame, textvariable=self.event_type_var, state="readonly", values=("entry", "exit", "null"))
        event_type_combo.grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Label(editor_frame, text="Corrected ROI:").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=3)
        self.target_name_combo = ttk.Combobox(editor_frame, textvariable=self.target_name_var, state="normal", values=self.target_names)
        self.target_name_combo.grid(row=1, column=1, sticky="ew", pady=3)
        ttk.Label(editor_frame, text="Track ID:").grid(row=2, column=0, sticky="w", padx=(0, 6), pady=3)
        ttk.Entry(editor_frame, textvariable=self.track_id_var, width=12).grid(row=2, column=1, sticky="w", pady=3)
        ttk.Label(editor_frame, text="Frame:").grid(row=3, column=0, sticky="w", padx=(0, 6), pady=3)
        frame_row = ttk.Frame(editor_frame)
        frame_row.grid(row=3, column=1, sticky="ew", pady=3)
        ttk.Entry(frame_row, textvariable=self.frame_var, width=12).pack(side=tk.LEFT)
        ttk.Button(frame_row, text="Set Current", command=self._set_current_frame).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(frame_row, text="Jump", command=self._jump_to_frame).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(editor_frame, text="Notes:").grid(row=4, column=0, sticky="nw", padx=(0, 6), pady=(6, 3))
        self.notes_text = tk.Text(editor_frame, height=4, wrap="word")
        self.notes_text.grid(row=4, column=1, sticky="ew", pady=(6, 3))

        action_row = ttk.Frame(editor_frame)
        action_row.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(action_row, text="Confirm Event", command=self._confirm_selected).pack(side=tk.LEFT)
        ttk.Button(action_row, text="Mark Null", command=self._mark_null).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(action_row, text="Apply Correction", command=self._apply_correction).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(action_row, text="Save Progress", command=self._save_review).pack(side=tk.RIGHT)

        results_frame = ttk.LabelFrame(right_panel, text="ROI Events", padding=10)
        results_frame.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        results_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)

        self.results_tree = ttk.Treeview(results_frame, columns=self.DISPLAY_COLUMNS, show="headings", selectmode="browse")
        for column in self.DISPLAY_COLUMNS:
            self.results_tree.heading(column, text=column)
            self.results_tree.column(column, width=130, anchor="w")
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        self.results_tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.results_tree.bind("<Double-1>", lambda _event: self._jump_to_frame())
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.results_tree.configure(yscrollcommand=scrollbar.set)

        footer = ttk.Frame(right_panel)
        footer.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(footer, text="Export ROI Event Review CSV", command=self._export_csv).pack(side=tk.LEFT)
        ttk.Label(footer, textvariable=self.status_var, justify=tk.LEFT, wraplength=420).pack(side=tk.LEFT, padx=(10, 0))

    def _load_video(self) -> None:
        if not self.video_path:
            self.status_var.set("No source video selected.")
            return
        self.cap = cv2.VideoCapture(self.video_path)
        if self.cap is None or not self.cap.isOpened():
            self.cap = None
            self.status_var.set("Could not open the selected video.")
            return
        self.total_frames = max(0, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        self.video_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if self.video_fps <= 0:
            self.video_fps = 30.0
        self.video_scale.configure(to=max(1, self.total_frames - 1), state=tk.NORMAL)
        self.play_pause_btn.configure(state=tk.NORMAL)
        self._display_frame_at(0)
        self.status_var.set(f"Loaded {len(self.working_events_df)} ROI event(s).")

    def _display_frame_at(self, frame_index: int) -> bool:
        if self.cap is None or self.total_frames <= 0:
            return False
        clamped = max(0, min(int(frame_index), self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, clamped)
        ok, frame = self.cap.read()
        if not ok:
            return False
        self.current_frame_index = clamped
        label_width = max(1, self.video_label.winfo_width())
        label_height = max(1, self.video_label.winfo_height())
        frame_height, frame_width = frame.shape[:2]
        scale = min(label_width / frame_width, label_height / frame_height)
        if scale <= 0:
            scale = 1.0
        resized = cv2.resize(frame, (max(1, int(frame_width * scale)), max(1, int(frame_height * scale))))
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self._current_photo = ImageTk.PhotoImage(image=Image.fromarray(rgb_frame))
        self.video_label.configure(image=self._current_photo)
        self._scale_callback_suppressed = True
        self.video_scale.set(clamped)
        self._scale_callback_suppressed = False
        current_seconds = clamped / self.video_fps if self.video_fps > 0 else 0.0
        total_seconds = self.total_frames / self.video_fps if self.video_fps > 0 else 0.0
        self.time_label_var.set(f"F: {clamped} / {self.total_frames} | T: {current_seconds:06.2f}s / {total_seconds:06.2f}s")
        return True

    def _toggle_play_pause(self) -> None:
        if self.cap is None:
            return
        self.is_playing = not self.is_playing
        self.play_pause_btn.configure(text="Pause" if self.is_playing else "Play")
        if self.is_playing:
            self._play_next_frame()

    def _play_next_frame(self) -> None:
        if not self.is_playing:
            return
        if self.current_frame_index >= max(0, self.total_frames - 1):
            self.is_playing = False
            self.play_pause_btn.configure(text="Play")
            return
        self._display_frame_at(self.current_frame_index + 1)
        delay_ms = max(1, int(1000 / self.video_fps)) if self.video_fps > 0 else 33
        self.after(delay_ms, self._play_next_frame)

    def _seek_relative(self, delta: int) -> None:
        if self.is_playing:
            self.is_playing = False
            self.play_pause_btn.configure(text="Play")
        self._display_frame_at(self.current_frame_index + int(delta))

    def _on_scale_drag(self, value_str: str) -> None:
        if self._scale_callback_suppressed:
            return
        if self.is_playing:
            self.is_playing = False
            self.play_pause_btn.configure(text="Play")
        try:
            self._display_frame_at(int(float(value_str)))
        except Exception:
            return

    def _filtered_events_df(self) -> pd.DataFrame:
        filtered = self.working_events_df.copy()
        target_name = str(self.filter_target_var.get() or "All").strip()
        if target_name and target_name != "All":
            filtered = filtered[filtered["Target Name"].astype(str) == target_name]
        event_type = str(self.filter_event_type_var.get() or "All").strip()
        if event_type and event_type != "All":
            filtered = filtered[filtered["Event Type"].astype(str) == event_type]
        review_status = str(self.filter_status_var.get() or "All").strip()
        if review_status and review_status != "All":
            filtered = filtered[filtered["Review Status"].astype(str) == review_status]
        return filtered

    def _refresh_results_tree(self, _event=None) -> None:
        self.results_tree.delete(*self.results_tree.get_children())
        filtered = self._filtered_events_df()
        for _, row in filtered.iterrows():
            corrected = "Yes" if bool(row.get("Corrected Manually", False)) else "No"
            self.results_tree.insert(
                "",
                "end",
                iid=str(int(row["Event ID"])),
                values=(
                    int(row["Track ID"]),
                    str(row["Event Type"]),
                    str(row["Target Name"]),
                    int(row["Frame"]),
                    str(row["Review Status"]),
                    corrected,
                ),
            )

    def _selected_row(self) -> pd.Series | None:
        if self._selected_event_id is None:
            return None
        match = self.working_events_df[self.working_events_df["Event ID"] == int(self._selected_event_id)]
        if match.empty:
            return None
        return match.iloc[0]

    def _select_event_by_id(self, event_id: int | None, *, seek: bool = False) -> None:
        if event_id is None:
            return
        iid = str(int(event_id))
        if iid not in self.results_tree.get_children():
            return
        self.results_tree.selection_set(iid)
        self.results_tree.focus(iid)
        self.results_tree.see(iid)
        self._selected_event_id = int(event_id)
        self._on_tree_select()
        if seek:
            self._jump_to_frame()

    def _on_tree_select(self, _event=None) -> None:
        selection = self.results_tree.selection()
        if not selection:
            self._selected_event_id = None
            return
        self._selected_event_id = _coerce_int(selection[0], -1)
        row = self._selected_row()
        if row is None:
            return
        self.event_type_var.set(str(row.get("Corrected Event Type", row.get("Event Type", "entry"))))
        self.target_name_var.set(str(row.get("Corrected Target Name", row.get("Target Name", ""))))
        self.track_id_var.set(str(_coerce_int(row.get("Corrected Track ID"), _coerce_int(row.get("Track ID"), 0))))
        self.frame_var.set(str(_coerce_int(row.get("Corrected Frame"), _coerce_int(row.get("Frame"), 0))))
        self.notes_text.delete("1.0", tk.END)
        notes = str(row.get("Reviewer Notes", "") or "").strip()
        if notes:
            self.notes_text.insert("1.0", notes)
        self.review_status_var.set(str(row.get("Review Status", "detected")))

    def _set_current_frame(self) -> None:
        self.frame_var.set(str(int(self.current_frame_index)))

    def _jump_to_frame(self) -> None:
        frame = _coerce_int(self.frame_var.get(), 0)
        self._display_frame_at(frame)

    def _collect_payload(self, *, selected_row: pd.Series) -> dict[str, object]:
        corrected_event_type = str(self.event_type_var.get() or "").strip().lower()
        if corrected_event_type not in {"entry", "exit", "null"}:
            raise ValueError("Corrected event type must be entry, exit, or null.")
        corrected_target_name = str(self.target_name_var.get() or "").strip()
        if corrected_event_type != "null" and not corrected_target_name:
            raise ValueError("Corrected ROI name is required unless the event is marked null.")
        corrected_track_id = _coerce_int(self.track_id_var.get(), _coerce_int(selected_row.get("Track ID"), 0))
        corrected_frame = _coerce_int(self.frame_var.get(), -1)
        if corrected_frame < 0:
            raise ValueError("Corrected frame must be a non-negative integer.")
        notes = self.notes_text.get("1.0", tk.END).strip()
        changed = (
            corrected_event_type != str(selected_row.get("Original Event Type", selected_row.get("Event Type", ""))).strip().lower()
            or corrected_target_name != str(selected_row.get("Original Target Name", selected_row.get("Target Name", ""))).strip()
            or corrected_track_id != _coerce_int(selected_row.get("Original Track ID"), _coerce_int(selected_row.get("Track ID"), 0))
            or corrected_frame != _coerce_int(selected_row.get("Original Frame"), _coerce_int(selected_row.get("Frame"), 0))
            or notes != str(selected_row.get("Reviewer Notes", "") or "").strip()
        )
        review_status = "corrected" if changed else "confirmed"
        if corrected_event_type == "null":
            corrected_target_name = ""
        return {
            "Review Status": review_status,
            "Corrected Event Type": corrected_event_type,
            "Corrected Target Name": corrected_target_name,
            "Corrected Track ID": corrected_track_id,
            "Corrected Frame": corrected_frame,
            "Reviewer Notes": notes,
            "Corrected Manually": bool(changed),
        }

    def _confirm_selected(self) -> None:
        row = self._selected_row()
        if row is None:
            messagebox.showinfo("ROI Event Review", "Select an ROI event first.", parent=self)
            return
        mask = self.working_events_df["Event ID"] == int(row["Event ID"])
        self.working_events_df.loc[mask, "Review Status"] = "confirmed"
        self.working_events_df.loc[mask, "Corrected Event Type"] = str(row.get("Original Event Type", row.get("Event Type", "")))
        self.working_events_df.loc[mask, "Corrected Target Name"] = str(row.get("Original Target Name", row.get("Target Name", "")))
        self.working_events_df.loc[mask, "Corrected Track ID"] = _coerce_int(row.get("Original Track ID"), _coerce_int(row.get("Track ID"), 0))
        self.working_events_df.loc[mask, "Corrected Frame"] = _coerce_int(row.get("Original Frame"), _coerce_int(row.get("Frame"), 0))
        self.working_events_df.loc[mask, "Corrected Manually"] = False
        self._save_review(silent=True)
        self._refresh_results_tree()
        self._select_event_by_id(int(row["Event ID"]), seek=False)
        self.status_var.set(f"Confirmed ROI event {int(row['Event ID'])}.")

    def _mark_null(self) -> None:
        row = self._selected_row()
        if row is None:
            messagebox.showinfo("ROI Event Review", "Select an ROI event first.", parent=self)
            return
        self.event_type_var.set("null")
        self.target_name_var.set("")
        self.track_id_var.set(str(_coerce_int(row.get("Track ID"), 0)))
        self.frame_var.set(str(_coerce_int(row.get("Frame"), 0)))
        self._apply_correction()

    def _apply_correction(self) -> None:
        row = self._selected_row()
        if row is None:
            messagebox.showinfo("ROI Event Review", "Select an ROI event first.", parent=self)
            return
        try:
            payload = self._collect_payload(selected_row=row)
            mask = self.working_events_df["Event ID"] == int(row["Event ID"])
            for column, value in payload.items():
                self.working_events_df.loc[mask, column] = value
            self._save_review(silent=True)
            self._refresh_results_tree()
            self._select_event_by_id(int(row["Event ID"]), seek=False)
            self.status_var.set(f"Updated ROI event {int(row['Event ID'])}.")
        except Exception as exc:
            messagebox.showerror("ROI Event Review", str(exc), parent=self)

    def _export_dataframe(self) -> pd.DataFrame:
        if self.working_events_df.empty:
            return pd.DataFrame(columns=ROI_EVENT_COLUMNS)
        exported = self.working_events_df.copy()
        exported.sort_values(["Frame", "Track ID", "Target Name", "Event Type"], inplace=True, kind="stable")
        exported.reset_index(drop=True, inplace=True)
        return exported.loc[:, ROI_EVENT_COLUMNS]

    def _save_review(self, silent: bool = False) -> None:
        try:
            export_df = self._export_dataframe()
            if self.autosave_path:
                save_path = Path(self.autosave_path).expanduser().resolve()
                save_path.parent.mkdir(parents=True, exist_ok=True)
                export_df.to_csv(save_path, index=False, encoding="utf-8")
                if not silent:
                    messagebox.showinfo("ROI Event Review", f"Saved progress to:\n{save_path}", parent=self)
            if callable(self.on_review_saved):
                self.on_review_saved(export_df.copy())
        except Exception as exc:
            if not silent:
                messagebox.showerror("Save Error", f"Failed to save ROI event review:\n{exc}", parent=self)

    def _export_csv(self) -> None:
        export_df = self._export_dataframe()
        if export_df.empty:
            messagebox.showinfo("ROI Event Review", "No ROI events are available to export.", parent=self)
            return
        initial_name = f"roi_event_review_{Path(self.video_path).stem or 'video'}.csv"
        if self.autosave_path:
            initial_name = Path(self.autosave_path).name
        target = filedialog.asksaveasfilename(
            title="Export ROI Event Review",
            defaultextension=".csv",
            filetypes=[("CSV File", "*.csv")],
            initialfile=initial_name,
            parent=self,
        )
        if not target:
            return
        try:
            export_df.to_csv(target, index=False, encoding="utf-8")
            self.status_var.set(f"Exported ROI event review to {target}")
            messagebox.showinfo("ROI Event Review", f"Exported ROI event review to:\n{target}", parent=self)
        except Exception as exc:
            messagebox.showerror("Export Error", f"Failed to export ROI event review:\n{exc}", parent=self)

    def _on_closing(self) -> None:
        self.is_playing = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self._save_review(silent=True)
        self.destroy()
