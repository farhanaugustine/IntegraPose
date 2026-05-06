from __future__ import annotations

import os
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional

import cv2
import pandas as pd
from PIL import Image, ImageTk

from integra_pose.gui.windowing import apply_adaptive_window_geometry

LogFn = Callable[[str, str], None]

SCORER_COLUMNS = [
    "Bout ID",
    "Track ID",
    "Behavior",
    "Behavior ID",
    "Start Frame",
    "End Frame",
    "Duration (s)",
    "Reviewer Notes",
    "Edited Manually",
    "Source Video",
    "Original Behavior",
    "Original Start Frame",
    "Original End Frame",
    "Review Status",
]


def _coerce_int(value, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _coerce_behavior_id(value) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
    except Exception:
        pass
    return str(value).strip()


def _format_track_id(value) -> str:
    if value is None or pd.isna(value):
        return "0"
    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
    except Exception:
        pass
    text = str(value).strip()
    return text or "0"


def _duration_seconds(start_frame: int, end_frame: int, fps: float) -> float:
    safe_fps = float(fps) if fps and fps > 0 else 30.0
    return max(0.0, float(end_frame - start_frame) / safe_fps)


def _first_present_column(df: pd.DataFrame, *candidates: str) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _sorted_track_ids(values) -> list[str]:
    tokens = {_format_track_id(value) for value in values if str(value).strip()}

    def _sort_key(token: str):
        try:
            return (0, int(token))
        except Exception:
            return (1, token.lower())

    ordered = sorted(tokens, key=_sort_key)
    return ordered or ["0"]


def normalize_advanced_bout_dataframe(
    raw_df: pd.DataFrame | None,
    *,
    video_path: str,
    fps: float,
    behavior_map: dict[str, object] | None = None,
) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=SCORER_COLUMNS)

    behavior_map = dict(behavior_map or {})
    work = raw_df.copy()

    bout_id_col = _first_present_column(work, "Bout ID")
    track_col = _first_present_column(work, "Track ID", "Animal ID")
    behavior_col = _first_present_column(work, "Behavior", "Behavior Name")
    behavior_id_col = _first_present_column(work, "Behavior ID", "Class ID", "class_id")
    start_col = _first_present_column(work, "Start Frame", "Bout Start Frame")
    end_col = _first_present_column(work, "End Frame", "Bout End Frame")
    notes_col = _first_present_column(work, "Reviewer Notes", "Notes", "Comment", "Comments")
    edited_col = _first_present_column(work, "Edited Manually")
    source_video_col = _first_present_column(work, "Source Video")
    original_behavior_col = _first_present_column(work, "Original Behavior")
    original_start_col = _first_present_column(work, "Original Start Frame")
    original_end_col = _first_present_column(work, "Original End Frame")

    rows: list[dict[str, object]] = []
    for offset, (_, row) in enumerate(work.iterrows(), start=1):
        track_id = _format_track_id(row.get(track_col) if track_col else 0)
        behavior = str(row.get(behavior_col, "") if behavior_col else "").strip()
        start_frame = _coerce_int(row.get(start_col) if start_col else 0, 0)
        end_frame = _coerce_int(row.get(end_col) if end_col else start_frame + 1, start_frame + 1)
        if end_frame <= start_frame:
            end_frame = start_frame + 1
        rows.append(
            {
                "Bout ID": _coerce_int(row.get(bout_id_col) if bout_id_col else offset, offset),
                "Track ID": track_id,
                "Behavior": behavior,
                "Behavior ID": _coerce_behavior_id(
                    row.get(behavior_id_col) if behavior_id_col else behavior_map.get(behavior, "")
                ),
                "Start Frame": start_frame,
                "End Frame": end_frame,
                "Duration (s)": round(_duration_seconds(start_frame, end_frame, fps), 6),
                "Reviewer Notes": str(row.get(notes_col, "") if notes_col else "").strip(),
                "Edited Manually": bool(row.get(edited_col, False)) if edited_col else False,
                "Source Video": str(row.get(source_video_col, video_path) if source_video_col else video_path).strip()
                or str(video_path),
                "Original Behavior": str(
                    row.get(original_behavior_col, behavior) if original_behavior_col else behavior
                ).strip(),
                "Original Start Frame": _coerce_int(
                    row.get(original_start_col) if original_start_col else start_frame, start_frame
                ),
                "Original End Frame": _coerce_int(
                    row.get(original_end_col) if original_end_col else end_frame, end_frame
                ),
                "Review Status": str(row.get("Review Status", "") or "").strip() or "detected",
            }
        )

    normalized = pd.DataFrame(rows, columns=SCORER_COLUMNS)
    if normalized.empty:
        return normalized
    normalized.sort_values(["Start Frame", "End Frame", "Track ID", "Behavior"], inplace=True, kind="stable")
    normalized.reset_index(drop=True, inplace=True)
    if normalized["Bout ID"].duplicated().any():
        normalized["Bout ID"] = list(range(1, len(normalized) + 1))
    return normalized


def resolve_initial_bout_id(df: pd.DataFrame | None, details) -> int | None:
    if df is None or df.empty or details is None:
        return None
    try:
        payload = dict(details)
    except Exception:
        return None
    if not payload:
        return None

    if "Bout ID" in payload:
        bout_id = _coerce_int(payload.get("Bout ID"), -1)
        if bout_id >= 0 and not df[df["Bout ID"] == bout_id].empty:
            return int(bout_id)

    track_value = None
    for key in ("Track ID", "Animal ID"):
        if key in payload and str(payload.get(key, "")).strip():
            track_value = _format_track_id(payload.get(key))
            break

    behavior_value = None
    for key in ("Behavior", "Behavior Name"):
        if key in payload and str(payload.get(key, "")).strip():
            behavior_value = str(payload.get(key)).strip()
            break

    start_value = None
    for key in ("Start Frame", "Bout Start Frame"):
        if key in payload and str(payload.get(key, "")).strip():
            start_value = _coerce_int(payload.get(key), -1)
            break

    end_value = None
    for key in ("End Frame", "Bout End Frame"):
        if key in payload and str(payload.get(key, "")).strip():
            end_value = _coerce_int(payload.get(key), -1)
            break

    candidates = df.copy()
    if track_value is not None:
        filtered = candidates[candidates["Track ID"].astype(str) == str(track_value)]
        if not filtered.empty:
            candidates = filtered
    if behavior_value is not None:
        filtered = candidates[candidates["Behavior"].astype(str) == str(behavior_value)]
        if not filtered.empty:
            candidates = filtered
    if start_value is not None and end_value is not None:
        filtered = candidates[
            (pd.to_numeric(candidates["Start Frame"], errors="coerce") == start_value)
            & (pd.to_numeric(candidates["End Frame"], errors="coerce") == end_value)
        ]
        if not filtered.empty:
            return int(filtered.iloc[0]["Bout ID"])
    if start_value is not None:
        filtered = candidates[pd.to_numeric(candidates["Start Frame"], errors="coerce") == start_value]
        if not filtered.empty:
            return int(filtered.iloc[0]["Bout ID"])
    if not candidates.empty:
        return int(candidates.iloc[0]["Bout ID"])
    return None


class AdvancedBoutAnalyzer(tk.Toplevel):
    DISPLAY_COLUMNS = ("Track ID", "Behavior", "Start Frame", "End Frame", "Duration (s)", "Review", "Edited", "Notes")

    def __init__(
        self,
        parent,
        video_path,
        initial_bout_details,
        behavior_map,
        all_bouts_df,
        *,
        autosave_path: str | None = None,
        on_scores_saved: Optional[Callable[[pd.DataFrame], None]] = None,
        log_fn: Optional[LogFn] = None,
    ):
        super().__init__(parent)
        self.title("Advanced Bout Scorer")
        apply_adaptive_window_geometry(
            self,
            preferred_size=(1320, 860),
            min_size=(1120, 760),
        )
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.video_path = str(video_path or "").strip()
        self.behavior_map = dict(behavior_map or {})
        self.all_bouts_df = all_bouts_df.copy() if isinstance(all_bouts_df, pd.DataFrame) else pd.DataFrame()
        self.autosave_path = str(autosave_path or "").strip()
        self.on_scores_saved = on_scores_saved
        self.log_fn = log_fn

        self.cap = None
        self.total_frames = 0
        self.video_fps = 30.0
        self.current_frame_index = 0
        self.is_playing = False
        self._scale_callback_suppressed = False
        self._current_photo = None
        self._selected_bout_id: int | None = None

        behaviors = {str(name).strip() for name in self.behavior_map.keys() if str(name).strip()}
        if not self.all_bouts_df.empty and "Behavior" in self.all_bouts_df.columns:
            behaviors.update(
                str(name).strip() for name in self.all_bouts_df["Behavior"].dropna().tolist() if str(name).strip()
            )
        self.behavior_names_list = sorted(behaviors)

        self.track_id_var = tk.StringVar()
        self.behavior_var = tk.StringVar()
        self.start_frame_var = tk.StringVar()
        self.end_frame_var = tk.StringVar()
        self.duration_var = tk.StringVar(value="Duration: 0.00 s")
        self.status_var = tk.StringVar(value="Ready.")
        self.time_label_var = tk.StringVar(value="F: 0 / 0 | T: 00.00s / 00.00s")

        self._create_widgets()
        self._bind_duration_updates()
        self._load_video(self.video_path)
        self.working_bouts_df = self._load_initial_dataframe()
        self._refresh_track_ids()
        self._refresh_results_tree()
        self.after(100, lambda: self._load_initial_selection(initial_bout_details))

    def _log(self, message: str, level: str = "INFO") -> None:
        if callable(self.log_fn):
            try:
                self.log_fn(message, level)
            except Exception:
                pass

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

        self.prev_frame_btn = ttk.Button(
            controls_frame,
            text="Prev Frame",
            command=lambda: self._seek_relative(-1),
            state=tk.DISABLED,
        )
        self.prev_frame_btn.grid(row=0, column=0, padx=(0, 4))
        self.play_pause_btn = ttk.Button(
            controls_frame,
            text="Play",
            command=self._toggle_play_pause,
            state=tk.DISABLED,
        )
        self.play_pause_btn.grid(row=0, column=1, padx=4)
        self.next_frame_btn = ttk.Button(
            controls_frame,
            text="Next Frame",
            command=lambda: self._seek_relative(1),
            state=tk.DISABLED,
        )
        self.next_frame_btn.grid(row=0, column=2, padx=4)
        self.video_scale = ttk.Scale(
            controls_frame,
            from_=0,
            to=1,
            orient=tk.HORIZONTAL,
            command=self._on_scale_drag,
            state=tk.DISABLED,
        )
        self.video_scale.grid(row=0, column=3, sticky="ew", padx=6)
        ttk.Label(controls_frame, textvariable=self.time_label_var).grid(row=0, column=4, padx=(6, 0))

        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky="nsew")
        right_panel.rowconfigure(1, weight=1)
        right_panel.columnconfigure(0, weight=1)

        scoring_frame = ttk.LabelFrame(right_panel, text="Scoring", padding=10)
        scoring_frame.grid(row=0, column=0, sticky="ew")
        scoring_frame.columnconfigure(1, weight=1)

        ttk.Label(scoring_frame, text="Animal ID:").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=3)
        self.track_id_combo = ttk.Combobox(scoring_frame, textvariable=self.track_id_var, state="normal")
        self.track_id_combo.grid(row=0, column=1, sticky="ew", pady=3)

        ttk.Label(scoring_frame, text="Behavior:").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=3)
        self.behavior_combo = ttk.Combobox(
            scoring_frame,
            textvariable=self.behavior_var,
            values=self.behavior_names_list,
            state="readonly",
        )
        self.behavior_combo.grid(row=1, column=1, sticky="ew", pady=3)

        ttk.Label(scoring_frame, text="Start Frame:").grid(row=2, column=0, sticky="w", padx=(0, 6), pady=3)
        start_row = ttk.Frame(scoring_frame)
        start_row.grid(row=2, column=1, sticky="ew", pady=3)
        ttk.Entry(start_row, textvariable=self.start_frame_var, width=12).pack(side=tk.LEFT)
        ttk.Button(start_row, text="Set Current", command=self._set_start_frame).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(start_row, text="Jump", command=self._jump_to_start_frame).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(scoring_frame, text="End Frame:").grid(row=3, column=0, sticky="w", padx=(0, 6), pady=3)
        end_row = ttk.Frame(scoring_frame)
        end_row.grid(row=3, column=1, sticky="ew", pady=3)
        ttk.Entry(end_row, textvariable=self.end_frame_var, width=12).pack(side=tk.LEFT)
        ttk.Button(end_row, text="Set Current", command=self._set_end_frame).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(end_row, text="Jump", command=self._jump_to_end_frame).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(scoring_frame, textvariable=self.duration_var).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(4, 2)
        )

        ttk.Label(scoring_frame, text="Reviewer Notes:").grid(
            row=5, column=0, sticky="nw", padx=(0, 6), pady=(6, 3)
        )
        self.notes_text = tk.Text(scoring_frame, height=4, wrap="word")
        self.notes_text.grid(row=5, column=1, sticky="ew", pady=(6, 3))

        action_row = ttk.Frame(scoring_frame)
        action_row.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(action_row, text="Update Selected", command=self._update_selected_bout).pack(side=tk.LEFT)
        ttk.Button(action_row, text="Add New Bout", command=self._add_new_bout).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(action_row, text="Clear Form", command=self._clear_form).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(action_row, text="Save Progress", command=self._save_scores).pack(side=tk.RIGHT)

        results_frame = ttk.LabelFrame(right_panel, text="Editable Bouts", padding=10)
        results_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        results_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)

        self.results_tree = ttk.Treeview(
            results_frame,
            columns=self.DISPLAY_COLUMNS,
            show="headings",
            selectmode="browse",
        )
        for column in self.DISPLAY_COLUMNS:
            self.results_tree.heading(column, text=column)
            self.results_tree.column(column, width=120 if column != "Notes" else 220, anchor="w")
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        self.results_tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.results_tree.bind("<Double-1>", lambda _event: self._jump_to_selected_bout())

        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.results_tree.configure(yscrollcommand=scrollbar.set)

        nav_row = ttk.Frame(results_frame)
        nav_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(nav_row, text="Previous Bout", command=lambda: self._move_selection(-1)).pack(side=tk.LEFT)
        ttk.Button(nav_row, text="Next Bout", command=lambda: self._move_selection(1)).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(nav_row, text="Delete Selected", command=self._delete_selected_bout).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(nav_row, text="Export CSV...", command=self._export_csv).pack(side=tk.RIGHT)

        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(fill=tk.X, padx=10, pady=(0, 10))

    def _bind_duration_updates(self) -> None:
        self.start_frame_var.trace_add("write", lambda *_args: self._update_duration_preview())
        self.end_frame_var.trace_add("write", lambda *_args: self._update_duration_preview())

    def _set_transport_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        for widget in (self.prev_frame_btn, self.play_pause_btn, self.next_frame_btn):
            widget.config(state=state)
        self.video_scale.config(state=state)

    def _load_video(self, filepath: str) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.total_frames = 0
        self.video_fps = 30.0
        self.current_frame_index = 0
        self._set_transport_enabled(False)
        if not filepath:
            self.status_var.set("No video selected.")
            return
        self.cap = cv2.VideoCapture(filepath)
        if not self.cap.isOpened():
            messagebox.showerror("Advanced Bout Scorer", "Could not open the selected video file.", parent=self)
            self._log(f"Advanced Bout Scorer could not open video: {filepath}", "ERROR")
            self.cap = None
            self.status_var.set("Failed to open video.")
            return
        self.total_frames = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
        fps_value = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.video_fps = fps_value if fps_value > 0 else 30.0
        self.video_scale.configure(to=max(0, self.total_frames - 1))
        self._set_transport_enabled(True)
        self._display_frame_at(0)
        self.status_var.set(f"Loaded video: {os.path.basename(filepath)}")

    def _load_initial_dataframe(self) -> pd.DataFrame:
        source_df = self.all_bouts_df
        if self.autosave_path:
            autosave_target = Path(self.autosave_path)
            if autosave_target.is_file():
                try:
                    source_df = pd.read_csv(autosave_target)
                    self._log(f"Loaded autosaved advanced bout scores from {autosave_target}", "INFO")
                except Exception as exc:
                    self._log(f"Failed to load autosaved advanced bout scores: {exc}", "WARNING")
        normalized = normalize_advanced_bout_dataframe(
            source_df,
            video_path=self.video_path,
            fps=self.video_fps,
            behavior_map=self.behavior_map,
        )
        if normalized.empty and self.behavior_names_list:
            self.behavior_var.set(self.behavior_names_list[0])
        return normalized

    def _load_initial_selection(self, initial_bout_details) -> None:
        if not self.working_bouts_df.empty:
            bout_id = resolve_initial_bout_id(self.working_bouts_df, initial_bout_details)
            if bout_id is None:
                bout_id = int(self.working_bouts_df.iloc[0]["Bout ID"])
            self._select_bout_by_id(bout_id, seek=True)
            return
        if initial_bout_details:
            self._populate_form_from_details(initial_bout_details)
            self._jump_to_start_frame()
        else:
            if not self.track_id_var.get():
                self.track_id_var.set("0")
            if self.behavior_names_list and not self.behavior_var.get():
                self.behavior_var.set(self.behavior_names_list[0])

    def _refresh_track_ids(self) -> None:
        values = []
        if not self.working_bouts_df.empty:
            values.extend(self.working_bouts_df["Track ID"].tolist())
        if not self.all_bouts_df.empty:
            if "Track ID" in self.all_bouts_df.columns:
                values.extend(self.all_bouts_df["Track ID"].tolist())
            elif "Animal ID" in self.all_bouts_df.columns:
                values.extend(self.all_bouts_df["Animal ID"].tolist())
        ordered = _sorted_track_ids(values)
        self.track_id_combo["values"] = ordered
        if not self.track_id_var.get():
            self.track_id_var.set(ordered[0])

    def _sort_working_df(self) -> None:
        if self.working_bouts_df.empty:
            return
        self.working_bouts_df.sort_values(["Start Frame", "End Frame", "Track ID", "Behavior"], inplace=True, kind="stable")
        self.working_bouts_df.reset_index(drop=True, inplace=True)

    def _refresh_results_tree(self) -> None:
        children = self.results_tree.get_children()
        if children:
            self.results_tree.delete(*children)
        self._sort_working_df()
        for _, row in self.working_bouts_df.iterrows():
            note_text = str(row.get("Reviewer Notes", "") or "").strip()
            if len(note_text) > 48:
                note_text = f"{note_text[:45]}..."
            self.results_tree.insert(
                "",
                "end",
                iid=str(int(row["Bout ID"])),
                values=(
                    str(row["Track ID"]),
                    str(row["Behavior"]),
                    int(row["Start Frame"]),
                    int(row["End Frame"]),
                    f"{float(row['Duration (s)']):.2f}",
                    str(row.get("Review Status", "detected")),
                    "Yes" if bool(row.get("Edited Manually", False)) else "No",
                    note_text,
                ),
            )
        self.status_var.set(f"{len(self.working_bouts_df)} bout(s) loaded for manual curation.")

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
        if scale <= 0 or label_width <= 1 or label_height <= 1:
            scale = 1.0
        resized = cv2.resize(frame, (max(1, int(frame_width * scale)), max(1, int(frame_height * scale))))
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self._current_photo = ImageTk.PhotoImage(image=Image.fromarray(rgb_frame))
        self.video_label.configure(image=self._current_photo)
        self._scale_callback_suppressed = True
        self.video_scale.set(clamped)
        self._scale_callback_suppressed = False
        self._update_time_label(clamped)
        return True

    def _update_time_label(self, frame_index: int) -> None:
        total_seconds = self.total_frames / self.video_fps if self.video_fps > 0 else 0.0
        current_seconds = frame_index / self.video_fps if self.video_fps > 0 else 0.0
        self.time_label_var.set(
            f"F: {frame_index} / {self.total_frames} | T: {current_seconds:06.2f}s / {total_seconds:06.2f}s"
        )

    def _toggle_play_pause(self) -> None:
        if self.cap is None:
            return
        if self.current_frame_index >= max(0, self.total_frames - 1):
            self._display_frame_at(0)
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

    def _set_start_frame(self) -> None:
        self.start_frame_var.set(str(int(self.current_frame_index)))

    def _set_end_frame(self) -> None:
        self.end_frame_var.set(str(int(self.current_frame_index)))

    def _jump_to_start_frame(self) -> None:
        self._display_frame_at(_coerce_int(self.start_frame_var.get(), 0))

    def _jump_to_end_frame(self) -> None:
        self._display_frame_at(_coerce_int(self.end_frame_var.get(), 0))

    def _update_duration_preview(self) -> None:
        start_frame = _coerce_int(self.start_frame_var.get(), 0)
        end_frame = _coerce_int(self.end_frame_var.get(), start_frame)
        self.duration_var.set(
            f"Duration: {_duration_seconds(start_frame, max(end_frame, start_frame), self.video_fps):.2f} s"
        )

    def _populate_form_from_details(self, details) -> None:
        try:
            payload = dict(details)
        except Exception:
            return
        track_value = ""
        for key in ("Track ID", "Animal ID"):
            if key in payload and str(payload.get(key, "")).strip():
                track_value = _format_track_id(payload.get(key))
                break
        behavior_value = ""
        for key in ("Behavior", "Behavior Name"):
            if key in payload and str(payload.get(key, "")).strip():
                behavior_value = str(payload.get(key)).strip()
                break
        start_value = ""
        for key in ("Start Frame", "Bout Start Frame"):
            if key in payload and str(payload.get(key, "")).strip():
                start_value = str(_coerce_int(payload.get(key), 0))
                break
        end_value = ""
        for key in ("End Frame", "Bout End Frame"):
            if key in payload and str(payload.get(key, "")).strip():
                end_value = str(_coerce_int(payload.get(key), 0))
                break
        notes_value = ""
        for key in ("Reviewer Notes", "Notes", "Comment", "Comments"):
            if key in payload and str(payload.get(key, "")).strip():
                notes_value = str(payload.get(key)).strip()
                break
        self.track_id_var.set(track_value or self.track_id_var.get() or "0")
        if behavior_value:
            self.behavior_var.set(behavior_value)
        self.start_frame_var.set(start_value)
        self.end_frame_var.set(end_value)
        self.notes_text.delete("1.0", tk.END)
        if notes_value:
            self.notes_text.insert("1.0", notes_value)

    def _populate_form_from_row(self, row: pd.Series) -> None:
        self.track_id_var.set(str(row.get("Track ID", "0")))
        self.behavior_var.set(str(row.get("Behavior", "")))
        self.start_frame_var.set(str(int(row.get("Start Frame", 0))))
        self.end_frame_var.set(str(int(row.get("End Frame", 0))))
        self.notes_text.delete("1.0", tk.END)
        notes = str(row.get("Reviewer Notes", "") or "").strip()
        if notes:
            self.notes_text.insert("1.0", notes)
        self._update_duration_preview()

    def _select_bout_by_id(self, bout_id: int | None, *, seek: bool = False) -> None:
        if bout_id is None:
            return
        iid = str(int(bout_id))
        if iid not in self.results_tree.get_children():
            return
        self.results_tree.selection_set(iid)
        self.results_tree.focus(iid)
        self.results_tree.see(iid)
        self._selected_bout_id = int(bout_id)
        self._on_tree_select()
        if seek:
            self._jump_to_start_frame()

    def _on_tree_select(self, _event=None) -> None:
        selection = self.results_tree.selection()
        if not selection:
            self._selected_bout_id = None
            return
        bout_id = _coerce_int(selection[0], -1)
        row_match = self.working_bouts_df[self.working_bouts_df["Bout ID"] == bout_id]
        if row_match.empty:
            return
        self._selected_bout_id = bout_id
        self._populate_form_from_row(row_match.iloc[0])

    def _selected_bout_row(self) -> pd.Series | None:
        if self._selected_bout_id is None:
            return None
        row_match = self.working_bouts_df[self.working_bouts_df["Bout ID"] == int(self._selected_bout_id)]
        if row_match.empty:
            return None
        return row_match.iloc[0]

    def _move_selection(self, delta: int) -> None:
        children = list(self.results_tree.get_children())
        if not children:
            return
        selection = self.results_tree.selection()
        if not selection:
            self._select_bout_by_id(_coerce_int(children[0], 1), seek=True)
            return
        try:
            current_index = children.index(selection[0])
        except ValueError:
            current_index = 0
        next_index = max(0, min(len(children) - 1, current_index + int(delta)))
        self._select_bout_by_id(_coerce_int(children[next_index], 1), seek=True)

    def _jump_to_selected_bout(self) -> None:
        row = self._selected_bout_row()
        if row is not None:
            self._display_frame_at(int(row.get("Start Frame", 0)))

    def _clear_form(self) -> None:
        self._selected_bout_id = None
        self.results_tree.selection_remove(self.results_tree.selection())
        self.start_frame_var.set("")
        self.end_frame_var.set("")
        self.notes_text.delete("1.0", tk.END)
        if not self.track_id_var.get():
            self.track_id_var.set("0")
        if self.behavior_names_list and not self.behavior_var.get():
            self.behavior_var.set(self.behavior_names_list[0])
        self._update_duration_preview()
        self.status_var.set("Cleared current bout form.")

    def _collect_form_payload(self, *, selected_row: pd.Series | None) -> dict[str, object]:
        track_id = _format_track_id(self.track_id_var.get())
        behavior = str(self.behavior_var.get() or "").strip()
        if not behavior:
            raise ValueError("Select a behavior label before saving.")
        start_frame = _coerce_int(self.start_frame_var.get(), -1)
        end_frame = _coerce_int(self.end_frame_var.get(), -1)
        if start_frame < 0 or end_frame < 0:
            raise ValueError("Start and end frames are required.")
        if end_frame <= start_frame:
            raise ValueError("End frame must be greater than start frame.")
        if self.total_frames > 0 and end_frame >= self.total_frames:
            raise ValueError(f"End frame must be less than {self.total_frames}.")
        notes = self.notes_text.get("1.0", tk.END).strip()
        original_behavior = behavior
        original_start = start_frame
        original_end = end_frame
        edited_manually = True
        review_status = "added_manual"
        bout_id = None
        if selected_row is not None:
            bout_id = int(selected_row["Bout ID"])
            original_behavior = str(
                selected_row.get("Original Behavior", selected_row.get("Behavior", behavior)) or behavior
            ).strip()
            original_start = _coerce_int(
                selected_row.get("Original Start Frame", selected_row.get("Start Frame", start_frame)),
                start_frame,
            )
            original_end = _coerce_int(
                selected_row.get("Original End Frame", selected_row.get("End Frame", end_frame)),
                end_frame,
            )
            changed = (
                str(selected_row.get("Track ID", "")) != track_id
                or str(selected_row.get("Behavior", "")) != behavior
                or _coerce_int(selected_row.get("Start Frame", start_frame), start_frame) != start_frame
                or _coerce_int(selected_row.get("End Frame", end_frame), end_frame) != end_frame
                or str(selected_row.get("Reviewer Notes", "") or "").strip() != notes
            )
            edited_manually = bool(selected_row.get("Edited Manually", False)) or changed
            review_status = "corrected" if changed else str(selected_row.get("Review Status", "detected") or "detected")
        return {
            "Bout ID": bout_id,
            "Track ID": track_id,
            "Behavior": behavior,
            "Behavior ID": _coerce_behavior_id(self.behavior_map.get(behavior, "")),
            "Start Frame": int(start_frame),
            "End Frame": int(end_frame),
            "Duration (s)": round(_duration_seconds(start_frame, end_frame, self.video_fps), 6),
            "Reviewer Notes": notes,
            "Edited Manually": bool(edited_manually),
            "Source Video": self.video_path,
            "Original Behavior": original_behavior,
            "Original Start Frame": int(original_start),
            "Original End Frame": int(original_end),
            "Review Status": review_status,
        }

    def _update_selected_bout(self) -> None:
        row = self._selected_bout_row()
        if row is None:
            messagebox.showinfo("Advanced Bout Scorer", "Select a bout from the table to update it.", parent=self)
            return
        try:
            payload = self._collect_form_payload(selected_row=row)
            mask = self.working_bouts_df["Bout ID"] == int(row["Bout ID"])
            for column, value in payload.items():
                if column == "Bout ID":
                    continue
                self.working_bouts_df.loc[mask, column] = value
            self._refresh_track_ids()
            self._refresh_results_tree()
            self._select_bout_by_id(int(row["Bout ID"]), seek=False)
            self._save_scores(silent=True)
            self.status_var.set(f"Updated bout {int(row['Bout ID'])}.")
        except Exception as exc:
            messagebox.showerror("Advanced Bout Scorer", str(exc), parent=self)

    def _next_bout_id(self) -> int:
        if self.working_bouts_df.empty:
            return 1
        return int(pd.to_numeric(self.working_bouts_df["Bout ID"], errors="coerce").fillna(0).max()) + 1

    def _add_new_bout(self) -> None:
        try:
            payload = self._collect_form_payload(selected_row=None)
            payload["Bout ID"] = self._next_bout_id()
            payload["Edited Manually"] = True
            payload["Review Status"] = "added_manual"
            new_row = pd.DataFrame([payload], columns=SCORER_COLUMNS)
            self.working_bouts_df = pd.concat([self.working_bouts_df, new_row], ignore_index=True)
            self._refresh_track_ids()
            self._refresh_results_tree()
            self._select_bout_by_id(int(payload["Bout ID"]), seek=False)
            self._save_scores(silent=True)
            self.status_var.set(f"Added new bout {int(payload['Bout ID'])}.")
        except Exception as exc:
            messagebox.showerror("Advanced Bout Scorer", str(exc), parent=self)

    def _delete_selected_bout(self) -> None:
        row = self._selected_bout_row()
        if row is None:
            messagebox.showinfo("Advanced Bout Scorer", "Select a bout to delete.", parent=self)
            return
        bout_id = int(row["Bout ID"])
        if not messagebox.askyesno("Delete Bout", f"Delete bout {bout_id}?", parent=self):
            return
        self.working_bouts_df = self.working_bouts_df[self.working_bouts_df["Bout ID"] != bout_id].copy()
        self._selected_bout_id = None
        self._refresh_track_ids()
        self._refresh_results_tree()
        self._clear_form()
        self._save_scores(silent=True)
        self.status_var.set(f"Deleted bout {bout_id}.")

    def _export_dataframe(self) -> pd.DataFrame:
        if self.working_bouts_df.empty:
            return pd.DataFrame(columns=SCORER_COLUMNS)
        exported = self.working_bouts_df.copy()
        exported.sort_values(["Start Frame", "End Frame", "Track ID", "Behavior"], inplace=True, kind="stable")
        exported.reset_index(drop=True, inplace=True)
        return exported.loc[:, SCORER_COLUMNS]

    def _save_scores(self, silent: bool = False) -> None:
        try:
            export_df = self._export_dataframe()
            if self.autosave_path:
                save_path = Path(self.autosave_path).expanduser().resolve()
                save_path.parent.mkdir(parents=True, exist_ok=True)
                export_df.to_csv(save_path, index=False, encoding="utf-8")
                if not silent:
                    messagebox.showinfo("Advanced Bout Scorer", f"Saved progress to:\n{save_path}", parent=self)
            if callable(self.on_scores_saved):
                self.on_scores_saved(export_df.copy())
        except Exception as exc:
            self._log(f"Failed to save advanced bout scores: {exc}\n{traceback.format_exc()}", "ERROR")
            if not silent:
                messagebox.showerror("Save Error", f"Failed to save bout scores:\n{exc}", parent=self)

    def _export_csv(self) -> None:
        export_df = self._export_dataframe()
        if export_df.empty:
            messagebox.showinfo("Advanced Bout Scorer", "No bouts are available to export.", parent=self)
            return
        initial_name = f"advanced_bout_scores_{Path(self.video_path).stem or 'video'}.csv"
        if self.autosave_path:
            initial_name = Path(self.autosave_path).name
        target = filedialog.asksaveasfilename(
            title="Export Advanced Bout Scores",
            defaultextension=".csv",
            filetypes=[("CSV File", "*.csv")],
            initialfile=initial_name,
            parent=self,
        )
        if not target:
            return
        try:
            export_df.to_csv(target, index=False, encoding="utf-8")
            self.status_var.set(f"Exported advanced bout scores to {target}")
            messagebox.showinfo("Advanced Bout Scorer", f"Exported bout scores to:\n{target}", parent=self)
        except Exception as exc:
            messagebox.showerror("Export Error", f"Failed to export bout scores:\n{exc}", parent=self)

    def _on_closing(self) -> None:
        self.is_playing = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self._save_scores(silent=True)
        self.destroy()
