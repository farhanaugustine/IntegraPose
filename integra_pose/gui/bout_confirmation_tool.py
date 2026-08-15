import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import pandas as pd
from PIL import Image, ImageTk
import os
import logging
from typing import Callable, Optional

from integra_pose.gui.windowing import apply_adaptive_window_geometry
from integra_pose.utils.bout_review import (
    BoutReviewPaths,
    append_review_decision,
    build_review_workspace,
    decisions_from_workspace,
    ethogram_window,
    load_review_decisions,
    migrate_legacy_review_workspace,
    normalize_detected_bouts,
    save_review_bundle,
)
from integra_pose.utils.operation_result import OperationResult, OperationStatus
from integra_pose.utils.review_keybinds import DEFAULT_KEYBINDS, validate_keybinds

REVIEW_EXPORT_COLUMNS = [
    "status",
    "Corrected Manually",
    "Review Status",
    "Original Behavior",
    "Corrected Behavior",
]


def normalize_bout_confirmation_dataframe(raw_df: pd.DataFrame | None) -> pd.DataFrame:
    if raw_df is None:
        return pd.DataFrame()
    if raw_df.empty:
        return raw_df.copy()
    raw = normalize_detected_bouts(raw_df)
    legacy_workspace = raw_df.copy(deep=True).reset_index(drop=True)
    legacy_workspace["Bout ID"] = raw["Bout ID"].tolist()
    decisions = decisions_from_workspace(legacy_workspace)
    return build_review_workspace(raw, decisions)


class BoutConfirmationTool(tk.Toplevel):
    def __init__(
        self,
        master,
        video_path,
        detected_bouts_df,
        behavior_map,
        *,
        keybinds: Optional[dict[str, str]] = None,
        autosave_path: str | None = None,
        on_review_saved: Optional[Callable[[pd.DataFrame], None]] = None,
        on_save_result: Optional[Callable[[OperationResult], None]] = None,
        context_label: str = "",
    ):
        super().__init__(master)
        self.title("Bout Confirmation Tool")
        apply_adaptive_window_geometry(
            self,
            preferred_size=(1300, 850),
            min_size=(980, 680),
            width_ratio=0.95,
            height_ratio=0.93,
        )
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing BoutConfirmationTool")

        self._detected_bouts_input = detected_bouts_df.copy(deep=True)
        self.original_bouts_df = normalize_bout_confirmation_dataframe(detected_bouts_df)
        self.behavior_map = behavior_map
        self.behavior_names = sorted(str(name).strip() for name in dict(behavior_map or {}).keys() if str(name).strip())
        self.video_path = video_path
        self.is_playing = False
        self.after_id = None
        self.playback_speed = 1.0
        self.current_frame = 0
        self.photo = None
        self.autosave_path = str(autosave_path or "").strip()
        self.on_review_saved = on_review_saved
        self.on_save_result = on_save_result
        self.last_save_result = OperationResult.cancel("Review has not been saved yet.")
        self.context_label = str(context_label or "").strip()
        self.keybinds, keybind_warnings = validate_keybinds(keybinds or DEFAULT_KEYBINDS)

        self.id_column_name = 'Track ID' if 'Track ID' in self.original_bouts_df.columns else 'Animal ID'
        required_columns = [self.id_column_name, 'Behavior', 'Start Frame', 'End Frame']
        missing_cols = [col for col in required_columns if col not in self.original_bouts_df.columns]
        if missing_cols:
            messagebox.showerror("Data Error", f"Missing required columns: {', '.join(missing_cols)}", parent=self)
            self.destroy()
            return
            
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Video Error", f"Could not open video file: {video_path}", parent=self)
            self.logger.error(f"Failed to open video: {video_path}")
            self.destroy()
            return
        
        probed_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        analysis_fps = pd.to_numeric(
            self._detected_bouts_input.get("Analysis FPS", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        analysis_fps_values = sorted({float(value) for value in analysis_fps if float(value) > 0})
        if len(analysis_fps_values) > 1:
            messagebox.showerror(
                "FPS Error",
                "Detected bouts contain multiple Analysis FPS values. Review cannot continue with ambiguous timing provenance.",
                parent=self,
            )
            self.cap.release()
            self.destroy()
            return
        self.fps = analysis_fps_values[0] if analysis_fps_values else probed_fps
        if self.fps <= 0:
            messagebox.showerror(
                "FPS Error",
                "Could not resolve a positive FPS from the analysis output or source video. Bout review cannot safely calculate durations.",
                parent=self,
            )
            self.cap.release()
            self.destroy()
            return
        if probed_fps > 0 and analysis_fps_values and abs(probed_fps - self.fps) > 1e-3:
            self.logger.warning(
                "Analysis FPS (%s) differs from video metadata FPS (%s); preserving the analysis calibration.",
                self.fps,
                probed_fps,
            )
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames <= 0:
            self.logger.warning("Invalid frame count, estimating frames")
            self.total_frames = self._estimate_frame_count()
            if self.total_frames <= 0:
                messagebox.showwarning("Video Warning", "Could not determine frame count. Some features may not work.", parent=self)
                self.total_frames = 10000  # Arbitrary fallback
        self.logger.info(f"Video loaded: FPS={self.fps}, Total Frames={self.total_frames}")

        self.raw_detected_bouts_df = normalize_detected_bouts(
            self._detected_bouts_input,
            source_video=self.video_path,
            fps=self.fps,
        )
        legacy_workspace = normalize_bout_confirmation_dataframe(self._detected_bouts_input)
        if len(legacy_workspace) == len(self.raw_detected_bouts_df):
            legacy_workspace["Bout ID"] = self.raw_detected_bouts_df["Bout ID"].tolist()
        if self.autosave_path:
            review_paths = BoutReviewPaths.from_authoritative(self.autosave_path)
            if (
                not review_paths.decisions.is_file()
                and not review_paths.workspace.is_file()
                and review_paths.authoritative.is_file()
            ):
                try:
                    legacy_workspace = migrate_legacy_review_workspace(
                        self.raw_detected_bouts_df,
                        pd.read_csv(review_paths.authoritative),
                        source_video=self.video_path,
                        fps=self.fps,
                    )
                except Exception as exc:
                    self.logger.warning("Legacy bout review was not migrated: %s", exc)
        self.review_decisions_df = (
            load_review_decisions(self.autosave_path, legacy_workspace=legacy_workspace)
            if self.autosave_path
            else decisions_from_workspace(legacy_workspace)
        )
        self.original_bouts_df = build_review_workspace(
            self.raw_detected_bouts_df,
            self.review_decisions_df,
            fps=self.fps,
        )

        self._create_widgets()
        self._populate_filter_options()
        self._update_bout_list_display()
        self._update_stats_display()
        self._bind_keyboard_shortcuts()
        if keybind_warnings:
            self.logger.warning("Keybind warnings: %s", "; ".join(keybind_warnings))
        if self.context_label:
            self.current_frame_var.set(f"{self.context_label} | Frame: 0")
        
        children = self.tree.get_children()
        if children:
            self.tree.selection_set(children[0])
            self._on_bout_select()

    def _estimate_frame_count(self):
        """Estimate frame count by reading until the end."""
        self.logger.debug("Estimating frame count")
        self.cap.release()
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.logger.error("Failed to re-open video for frame count estimation")
            return 0
        count = 0
        while count < 10000:  # Safety limit
            ret, _ = self.cap.read()
            if not ret:
                break
            count += 1
        self.cap.release()
        self.cap = cv2.VideoCapture(self.video_path)
        self.logger.debug(f"Estimated frame count: {count}")
        return count

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        left_panel = ttk.Frame(main_frame, width=450)
        left_panel.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        left_panel.rowconfigure(1, weight=1) 
        
        filter_frame = ttk.LabelFrame(left_panel, text="Filters")
        filter_frame.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        self._setup_filter_controls(filter_frame)

        tree_frame = ttk.LabelFrame(left_panel, text="Detected Bouts")
        tree_frame.grid(row=1, column=0, sticky='nsew')
        self._setup_treeview(tree_frame)
        
        bout_nav_frame = ttk.Frame(left_panel)
        bout_nav_frame.grid(row=2, column=0, sticky='ew', pady=5)
        self._setup_bout_navigation(bout_nav_frame)
        
        stats_frame = ttk.LabelFrame(left_panel, text="Verification Stats")
        stats_frame.grid(row=3, column=0, sticky='ew', pady=(0, 10))
        self._setup_stats_display(stats_frame)
        
        save_btn = ttk.Button(left_panel, text="Save Review Progress", command=self._save_review_state)
        save_btn.grid(row=4, column=0, sticky='ew')
        export_btn = ttk.Button(left_panel, text="Export Decision Report...", command=self._export_to_csv)
        export_btn.grid(row=5, column=0, sticky='ew', pady=(5, 0))

        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky='nsew')
        right_panel.rowconfigure(0, weight=1)
        right_panel.columnconfigure(0, weight=1)

        self.video_label = ttk.Label(right_panel, background="black")
        self.video_label.grid(row=0, column=0, sticky='nsew')

        timeline_frame = ttk.LabelFrame(right_panel, text="Ethogram Context")
        timeline_frame.grid(row=1, column=0, sticky="ew", pady=(8, 2))
        timeline_frame.columnconfigure(0, weight=1)
        self.ethogram_canvas = tk.Canvas(
            timeline_frame,
            height=170,
            background="#ffffff",
            highlightthickness=0,
        )
        self.ethogram_canvas.grid(row=0, column=0, sticky="ew")
        timeline_scroll = ttk.Scrollbar(timeline_frame, orient="vertical", command=self.ethogram_canvas.yview)
        timeline_scroll.grid(row=0, column=1, sticky="ns")
        self.ethogram_canvas.configure(yscrollcommand=timeline_scroll.set)
        self.ethogram_canvas.bind("<Configure>", lambda _event: self._draw_ethogram())

        video_controls = self._create_video_controls(right_panel)
        video_controls.grid(row=2, column=0, sticky='ew', pady=5)
        
        confirmation_controls = self._create_confirmation_controls_v2(right_panel)
        confirmation_controls.grid(row=3, column=0, sticky='ew', pady=10)

        self.keybind_hint_var = tk.StringVar(value=self._format_keybind_hint())
        ttk.Label(
            right_panel,
            textvariable=self.keybind_hint_var,
            justify=tk.LEFT,
            wraplength=760,
        ).grid(row=4, column=0, sticky='w', pady=(2, 0))

    def _setup_filter_controls(self, parent):
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text="Track ID:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.track_id_filter_var = tk.StringVar(value="All")
        self.track_id_filter_combo = ttk.Combobox(parent, textvariable=self.track_id_filter_var, state="readonly")
        self.track_id_filter_combo.grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        self.track_id_filter_combo.bind("<<ComboboxSelected>>", self._update_bout_list_display)
        
        ttk.Label(parent, text="Behavior:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.behavior_filter_var = tk.StringVar(value="All")
        self.behavior_filter_combo = ttk.Combobox(parent, textvariable=self.behavior_filter_var, state="readonly")
        self.behavior_filter_combo.grid(row=1, column=1, sticky='ew', padx=5, pady=2)
        self.behavior_filter_combo.bind("<<ComboboxSelected>>", self._update_bout_list_display)
        
        ttk.Label(parent, text="Status:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.status_filter_var = tk.StringVar(value="All")
        self.status_filter_combo = ttk.Combobox(parent, textvariable=self.status_filter_var, state="readonly", values=["All", "unreviewed", "confirmed", "rejected", "corrected"])
        self.status_filter_combo.grid(row=2, column=1, sticky='ew', padx=5, pady=2)
        self.status_filter_combo.bind("<<ComboboxSelected>>", self._update_bout_list_display)

    def _setup_treeview(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        cols = (self.id_column_name, 'Behavior', 'Start Frame', 'End Frame', 'status')
        self.tree = ttk.Treeview(parent, columns=cols, show='headings')
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor=tk.W, stretch=True)
        self.tree.column('Behavior', width=150)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.tree.bind('<<TreeviewSelect>>', self._on_bout_select)
        self.tree.tag_configure('confirmed', background='#d9ead3')
        self.tree.tag_configure('rejected', background='#f4cccc')
        self.tree.tag_configure('corrected', background='#fff2cc')
        
    def _setup_bout_navigation(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        self.prev_bout_btn = ttk.Button(parent, text="Previous Bout", command=self._prev_bout)
        self.prev_bout_btn.grid(row=0, column=0, sticky='ew', padx=(0,2))
        self.next_bout_btn = ttk.Button(parent, text="Next Bout", command=self._next_bout)
        self.next_bout_btn.grid(row=0, column=1, sticky='ew', padx=(2,0))

    def _setup_stats_display(self, parent):
        parent.columnconfigure(1, weight=1)
        self.total_bouts_var = tk.StringVar(value="Total: 0")
        self.reviewed_bouts_var = tk.StringVar(value="Reviewed: 0")
        self.confirmed_bouts_var = tk.StringVar(value="Confirmed: 0")
        self.rejected_bouts_var = tk.StringVar(value="Rejected: 0")
        self.corrected_bouts_var = tk.StringVar(value="Corrected: 0")
        ttk.Label(parent, textvariable=self.total_bouts_var).grid(row=0, column=0, sticky='w', padx=5)
        ttk.Label(parent, textvariable=self.reviewed_bouts_var).grid(row=0, column=1, sticky='w', padx=5)
        ttk.Label(parent, textvariable=self.confirmed_bouts_var).grid(row=1, column=0, sticky='w', padx=5)
        ttk.Label(parent, textvariable=self.rejected_bouts_var).grid(row=1, column=1, sticky='w', padx=5)
        ttk.Label(parent, textvariable=self.corrected_bouts_var).grid(row=2, column=0, sticky='w', padx=5)

    def _create_video_controls(self, parent):
        frame = ttk.Frame(parent)
        frame.columnconfigure(0, weight=1)
        
        self.current_frame_var = tk.StringVar(value="Frame: 0")
        frame_display_label = ttk.Label(frame, textvariable=self.current_frame_var, anchor=tk.CENTER)
        frame_display_label.grid(row=0, column=0, sticky='ew', pady=(0, 5))
        
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=1, column=0)
        
        self.step_back_btn = ttk.Button(button_frame, text="◀ Frame", command=lambda: self._step_frame(-1))
        self.step_back_btn.pack(side=tk.LEFT, padx=2)
        
        self.play_pause_btn = ttk.Button(button_frame, text="▶ Play", command=self._toggle_play_pause)
        self.play_pause_btn.pack(side=tk.LEFT, padx=5)

        self.step_fwd_btn = ttk.Button(button_frame, text="Frame ▶", command=lambda: self._step_frame(1))
        self.step_fwd_btn.pack(side=tk.LEFT, padx=2)
        
        self.reset_bout_btn = ttk.Button(button_frame, text="↩ Reset Bout", command=self._reset_bout_playback)
        self.reset_bout_btn.pack(side=tk.LEFT, padx=(20, 5))

        self.speed_var = tk.StringVar(value="1.0x")
        speed_label = ttk.Label(button_frame, text="Speed:")
        speed_label.pack(side=tk.LEFT, padx=(20,2))
        speed_combo = ttk.Combobox(button_frame, textvariable=self.speed_var, values=["0.5x", "1.0x", "2.0x"], state="readonly", width=5)
        speed_combo.pack(side=tk.LEFT, padx=2)
        speed_combo.bind("<<ComboboxSelected>>", self._set_playback_speed)
        
        return frame

    def _create_confirmation_controls(self, parent):
        frame = ttk.Frame(parent)
        confirm_btn = ttk.Button(frame, text="✔ Confirm Bout", command=self._confirm_bout, style="Confirm.TButton")
        reject_btn = ttk.Button(frame, text="✖ Reject Bout", command=self._reject_bout, style="Danger.TButton")
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        confirm_btn.grid(row=0, column=0, sticky='ew', padx=5, ipady=5)
        reject_btn.grid(row=0, column=1, sticky='ew', padx=5, ipady=5)
        return frame

    def _create_confirmation_controls_v2(self, parent):
        frame = ttk.Frame(parent)
        confirm_btn = ttk.Button(frame, text="Confirm Bout", command=self._confirm_bout, style="Confirm.TButton")
        reject_btn = ttk.Button(frame, text="Reject Bout", command=self._reject_bout, style="Danger.TButton")
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)
        confirm_btn.grid(row=0, column=0, sticky='ew', padx=5, ipady=5)
        reject_btn.grid(row=0, column=1, sticky='ew', padx=5, ipady=5)
        correction_row = ttk.Frame(frame)
        correction_row.grid(row=1, column=0, columnspan=3, sticky='ew', padx=5, pady=(8, 0))
        ttk.Label(correction_row, text="Correct Label:").pack(side=tk.LEFT)
        self.corrected_behavior_var = tk.StringVar(value="")
        self.corrected_behavior_combo = ttk.Combobox(
            correction_row,
            textvariable=self.corrected_behavior_var,
            values=self.behavior_names,
            state="readonly" if self.behavior_names else "normal",
            width=22,
        )
        self.corrected_behavior_combo.pack(side=tk.LEFT, padx=(6, 6))
        ttk.Button(correction_row, text="Apply Correction", command=self._correct_bout_label).pack(side=tk.LEFT)
        metadata_row = ttk.Frame(frame)
        metadata_row.grid(row=2, column=0, columnspan=3, sticky='ew', padx=5, pady=(8, 0))
        metadata_row.columnconfigure(3, weight=1)
        ttk.Label(metadata_row, text="Reviewer:").grid(row=0, column=0, sticky="w")
        self.reviewer_var = tk.StringVar(value="")
        ttk.Entry(metadata_row, textvariable=self.reviewer_var, width=16).grid(row=0, column=1, sticky="w", padx=(6, 12))
        ttk.Label(metadata_row, text="Notes:").grid(row=0, column=2, sticky="w")
        self.review_notes_var = tk.StringVar(value="")
        ttk.Entry(metadata_row, textvariable=self.review_notes_var).grid(row=0, column=3, sticky="ew", padx=(6, 0))
        return frame

    def _populate_filter_options(self):
        if self.original_bouts_df.empty:
            return
        track_ids = sorted(self.original_bouts_df[self.id_column_name].unique().tolist())
        self.track_id_filter_combo['values'] = ["All"] + [str(tid) for tid in track_ids]
        behaviors = sorted(self.original_bouts_df['Behavior'].unique().tolist())
        self.behavior_filter_combo['values'] = ["All"] + behaviors

    def _update_bout_list_display(self, event=None):
        self._stop_playback()
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        filtered_df = self.original_bouts_df.copy()
        selected_id = self.track_id_filter_var.get()
        if selected_id != "All":
            filtered_df = filtered_df[filtered_df[self.id_column_name].astype(str) == selected_id]
        
        selected_behavior = self.behavior_filter_var.get()
        if selected_behavior != "All":
            filtered_df = filtered_df[filtered_df['Behavior'] == selected_behavior]
            
        selected_status = self.status_filter_var.get()
        if selected_status != "All":
            filtered_df = filtered_df[filtered_df['status'] == selected_status]
            
        for index, row in filtered_df.iterrows():
            values = (row[self.id_column_name], row['Behavior'], row['Start Frame'], row['End Frame'], row['status'])
            self.tree.insert('', 'end', iid=str(row['Bout ID']), values=values, tags=(row['status'],))
        
        children = self.tree.get_children()
        if children:
            self.tree.selection_set(children[0])
            self._on_bout_select()

    def _get_current_bout_info(self):
        selected_items = self.tree.selection()
        if not selected_items:
            return None
        item_id = str(selected_items[0])
        matches = self.original_bouts_df[self.original_bouts_df['Bout ID'].astype(str) == item_id]
        if matches.empty:
            return None
        return matches.iloc[0]

    def _on_bout_select(self, event=None):
        self._stop_playback()
        bout_info = self._get_current_bout_info()
        if bout_info is None:
            return
        try:
            corrected_value = str(bout_info.get('Corrected Behavior', bout_info.get('Behavior', '')) or '').strip()
            if corrected_value:
                self.corrected_behavior_var.set(corrected_value)
            reviewer_value = str(bout_info.get('Reviewer', '') or '').strip()
            if reviewer_value:
                self.reviewer_var.set(reviewer_value)
            self.review_notes_var.set(str(bout_info.get('Reviewer Notes', '') or '').strip())
        except Exception:
            pass
        start_frame = int(bout_info['Start Frame'])
        self._seek_to_frame(start_frame)
        self._draw_ethogram()

    def _select_bout_id(self, bout_id: str) -> None:
        iid = str(bout_id)
        if iid not in self.tree.get_children():
            self.track_id_filter_var.set("All")
            self.behavior_filter_var.set("All")
            self.status_filter_var.set("All")
            self._update_bout_list_display()
        if iid in self.tree.get_children():
            self.tree.selection_set(iid)
            self.tree.focus(iid)
            self.tree.see(iid)
            self._on_bout_select()

    def _draw_ethogram(self) -> None:
        canvas = getattr(self, "ethogram_canvas", None)
        if canvas is None:
            return
        canvas.delete("all")
        bout_info = self._get_current_bout_info()
        if bout_info is None or self.original_bouts_df.empty:
            canvas.create_text(12, 12, text="Select a bout to view ethogram context.", anchor="nw", fill="#444444")
            return
        focus_id = str(bout_info["Bout ID"])
        duration = int(bout_info["End Frame"]) - int(bout_info["Start Frame"]) + 1
        context_frames = max(int(round(self.fps * 5.0)), duration * 2)
        try:
            window_start, window_end, visible = ethogram_window(
                self.original_bouts_df,
                focus_bout_id=focus_id,
                context_frames=context_frames,
                total_frames=self.total_frames,
            )
        except Exception as exc:
            self.logger.warning("Failed to render ethogram context: %s", exc)
            return

        label_width = 138
        plot_left = label_width + 8
        plot_right = max(plot_left + 100, canvas.winfo_width() - 12)
        top = 28
        lane_height = 25
        lane_keys = list(
            dict.fromkeys(
                (str(row["Track ID"]), str(row["Behavior"]))
                for _, row in visible.iterrows()
            )
        )
        canvas_height = max(170, top + lane_height * max(1, len(lane_keys)) + 22)
        canvas.configure(scrollregion=(0, 0, plot_right, canvas_height))
        span = max(1, window_end - window_start + 1)

        def _x(frame: int) -> float:
            return plot_left + ((int(frame) - window_start) / span) * (plot_right - plot_left)

        canvas.create_line(plot_left, top - 7, plot_right, top - 7, fill="#555555")
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            frame = min(window_end, window_start + int(round((span - 1) * fraction)))
            x = _x(frame)
            canvas.create_line(x, top - 11, x, top - 3, fill="#555555")
            canvas.create_text(x, 4, text=f"F {frame} | {frame / self.fps:.2f}s", anchor="n", fill="#333333")

        colors = {
            "unreviewed": "#78A6C8",
            "confirmed": "#66A061",
            "rejected": "#9A9A9A",
            "corrected": "#D49B45",
        }
        lane_lookup = {key: index for index, key in enumerate(lane_keys)}
        for lane_key, lane_index in lane_lookup.items():
            y0 = top + lane_index * lane_height
            canvas.create_text(
                label_width,
                y0 + lane_height / 2,
                text=f"T{lane_key[0]}  {lane_key[1]}",
                anchor="e",
                fill="#222222",
            )
            canvas.create_line(plot_left, y0 + lane_height, plot_right, y0 + lane_height, fill="#E1E1E1")
        for _, row in visible.iterrows():
            lane_index = lane_lookup[(str(row["Track ID"]), str(row["Behavior"]))]
            y0 = top + lane_index * lane_height + 4
            y1 = y0 + lane_height - 8
            x0 = max(plot_left, _x(int(row["Start Frame"])))
            x1 = min(plot_right, _x(int(row["End Frame"]) + 1))
            bout_id = str(row["Bout ID"])
            status = str(row.get("status", "unreviewed"))
            tag = f"bout::{bout_id}"
            canvas.create_rectangle(
                x0,
                y0,
                max(x0 + 2, x1),
                y1,
                fill=colors.get(status, colors["unreviewed"]),
                outline="#B22222" if bout_id == focus_id else "#333333",
                width=3 if bout_id == focus_id else 1,
                tags=(tag,),
            )
            canvas.tag_bind(tag, "<Button-1>", lambda _event, selected=bout_id: self._select_bout_id(selected))

        playhead_x = _x(max(window_start, min(self.current_frame, window_end)))
        canvas.create_line(playhead_x, top - 3, playhead_x, canvas_height - 8, fill="#B22222", width=2, tags=("playhead",))

    def _format_keybind_hint(self) -> str:
        kb = self.keybinds
        return (
            "Keybinds: "
            f"confirm={kb.get('confirm_bout', 'a')}, "
            f"reject={kb.get('reject_bout', 'r')}, "
            f"next={kb.get('next_bout', 'k')}, "
            f"prev={kb.get('prev_bout', 'j')}, "
            f"next-unreviewed={kb.get('next_unreviewed', ']')}, "
            f"play/pause={kb.get('toggle_play_pause', 'space')}, "
            f"frame+={kb.get('step_frame_forward', '.')}, "
            f"frame-={kb.get('step_frame_back', ',')}, "
            f"save={kb.get('save_review_state', 'Control-s')}"
        )

    def _bind_keyboard_shortcuts(self) -> None:
        action_map = {
            "confirm_bout": self._confirm_bout,
            "reject_bout": self._reject_bout,
            "next_bout": self._next_bout,
            "prev_bout": self._prev_bout,
            "next_unreviewed": self._jump_to_next_unreviewed,
            "prev_unreviewed": self._jump_to_prev_unreviewed,
            "toggle_play_pause": self._toggle_play_pause,
            "step_frame_forward": lambda: self._step_frame(1),
            "step_frame_back": lambda: self._step_frame(-1),
            "save_review_state": self._save_review_state,
        }
        for action, callback in action_map.items():
            token = self.keybinds.get(action)
            if not token:
                continue
            try:
                self.bind(f"<{token}>", lambda _evt, cb=callback: cb())
            except Exception:
                self.logger.warning("Failed to bind key token '%s' for action '%s'.", token, action)

    def _toggle_play_pause(self):
        if not self.tree.selection():
            messagebox.showwarning("No Selection", "Please select a bout to play.", parent=self)
            return
        if self.is_playing:
            self._stop_playback()
        else:
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.video_path)
                if not self.cap.isOpened():
                    messagebox.showerror("Video Error", f"Failed to re-open video file: {self.video_path}", parent=self)
                    self._stop_playback()
                    return
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            self.is_playing = True
            self.play_pause_btn.config(text="❚❚ Pause")
            self._playback_loop()

    def _stop_playback(self):
        if self.after_id:
            self.after_cancel(self.after_id)
            self.after_id = None
        self.is_playing = False
        self.play_pause_btn.config(text="▶ Play")

    def _step_frame(self, direction):
        self._stop_playback()
        target_frame = max(0, min(self.current_frame + direction, self.total_frames - 1))
        self._seek_to_frame(target_frame)

    def _reset_bout_playback(self):
        self._stop_playback()
        bout_info = self._get_current_bout_info()
        if bout_info is not None:
            start_frame = int(bout_info['Start Frame'])
            self._seek_to_frame(start_frame)

    def _playback_loop(self):
        if not self.is_playing:
            return

        bout_info = self._get_current_bout_info()
        if bout_info is None:
            self._stop_playback()
            return
        end_frame = int(bout_info['End Frame'])

        if self.current_frame > end_frame:
            self._stop_playback()
            return

        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                messagebox.showerror("Video Error", f"Failed to re-open video file: {self.video_path}", parent=self)
                self._stop_playback()
                return
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

        ret, frame = self.cap.read()
        if not ret:
            self._stop_playback()
            self.logger.error(f"Failed to read frame {self.current_frame} in playback")
            return
            
        displayed_frame = self.current_frame
        self._update_video_display(frame)
        self.current_frame_var.set(f"Frame: {displayed_frame}")
        self._draw_ethogram()
        if displayed_frame >= end_frame:
            self._stop_playback()
            return
        self.current_frame = displayed_frame + 1
        
        delay = int(1000 / (self.fps * self.playback_speed))
        self.after_id = self.after(delay, self._playback_loop)

    def _set_playback_speed(self, event=None):
        speed_str = self.speed_var.get().replace('x', '')
        self.playback_speed = float(speed_str)
        if self.is_playing:
            self._stop_playback()
            self._toggle_play_pause()

    def _seek_to_frame(self, frame_num):
        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                messagebox.showerror("Video Error", f"Failed to re-open video file: {self.video_path}", parent=self)
                self._stop_playback()
                self.logger.error(f"Failed to re-open video: {self.video_path}")
                return

        bout_info = self._get_current_bout_info()
        if bout_info is not None:
            start_frame = int(bout_info['Start Frame'])
            end_frame = int(bout_info['End Frame'])
            frame_num = max(start_frame, min(frame_num, end_frame))
        else:
            frame_num = max(0, min(frame_num, self.total_frames - 1))

        self.logger.debug(f"Seeking to frame {frame_num}")
        max_retries = 3
        for attempt in range(max_retries):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()
            if ret:
                actual_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                if actual_frame != frame_num:
                    self.logger.warning(f"Attempt {attempt+1}: Sought frame {frame_num}, got {actual_frame}, using fallback")
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.video_path)
                    if not self.cap.isOpened():
                        messagebox.showerror("Video Error", f"Failed to re-open video file: {self.video_path}", parent=self)
                        self._stop_playback()
                        self.logger.error(f"Failed to re-open video: {self.video_path}")
                        return
                    for i in range(frame_num):
                        ret, _ = self.cap.read()
                        if not ret:
                            self.logger.error(f"Failed to reach frame {frame_num} during fallback at frame {i}")
                            break
                    ret, frame = self.cap.read()
                    if ret:
                        actual_frame = frame_num
                        break
                else:
                    break
            else:
                self.logger.warning(f"Attempt {attempt+1}: Failed to read frame {frame_num}, retrying")
                self.cap.release()
                self.cap = cv2.VideoCapture(self.video_path)
                if not self.cap.isOpened():
                    messagebox.showerror("Video Error", f"Failed to re-open video file: {self.video_path}", parent=self)
                    self._stop_playback()
                    self.logger.error(f"Failed to re-open video: {self.video_path}")
                    return

        if not ret:
            messagebox.showerror("Error", f"Failed to read frame {frame_num} after {max_retries} attempts", parent=self)
            self._stop_playback()
            self.logger.error(f"Failed to read frame {frame_num} after {max_retries} attempts")
            return

        self.current_frame = frame_num
        self._update_video_display(frame)
        self.current_frame_var.set(f"Frame: {self.current_frame}")
        self._draw_ethogram()
        self.logger.debug(f"Successfully sought and displayed frame {frame_num}")

    def _update_video_display(self, frame):
        try:
            self.update_idletasks()
            l_w, l_h = self.video_label.winfo_width(), self.video_label.winfo_height()
            if l_w <= 1 or l_h <= 1:
                return

            f_h, f_w, _ = frame.shape
            scale = min(l_w / f_w, l_h / f_h)
            new_w, new_h = int(f_w * scale), int(f_h * scale)
            if new_w <= 0 or new_h <= 0:
                return
            resized_frame = cv2.resize(frame, (new_w, new_h))
            img = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(img))
            self.video_label.config(image=self.photo)
            self.video_label.image = self.photo
            self.logger.debug(f"Displayed frame {self.current_frame}")
        except Exception as e:
            messagebox.showerror("Display Error", f"Failed to update video display: {e}", parent=self)
            self._stop_playback()
            self.logger.error(f"Display error: {e}")

    def _update_bout_status(self, status):
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showwarning("No Selection", "Please select a bout from the list.", parent=self)
            return
        for bout_id in selected_items:
            raw_match = self.raw_detected_bouts_df[
                self.raw_detected_bouts_df['Bout ID'].astype(str) == str(bout_id)
            ]
            if raw_match.empty:
                continue
            self.review_decisions_df = append_review_decision(
                self.review_decisions_df,
                raw_match.iloc[0],
                status,
                reviewer_notes=self.review_notes_var.get(),
                reviewer=self.reviewer_var.get(),
            )
        self.original_bouts_df = build_review_workspace(
            self.raw_detected_bouts_df,
            self.review_decisions_df,
            fps=self.fps,
        )
        self._update_bout_list_display()
        self._update_stats_display()
        self._save_review_state(silent=True)
        self._select_next_unreviewed(selected_items[-1])

    def _update_stats_display(self):
        total = len(self.original_bouts_df)
        status_counts = self.original_bouts_df['status'].value_counts()
        reviewed = status_counts.get('confirmed', 0) + status_counts.get('rejected', 0) + status_counts.get('corrected', 0)
        confirmed = status_counts.get('confirmed', 0)
        rejected = status_counts.get('rejected', 0)
        corrected = status_counts.get('corrected', 0)
        self.total_bouts_var.set(f"Total Bouts: {total}")
        self.reviewed_bouts_var.set(f"Reviewed: {reviewed}")
        self.confirmed_bouts_var.set(f"Confirmed: {confirmed}")
        self.rejected_bouts_var.set(f"Rejected: {rejected}")
        self.corrected_bouts_var.set(f"Corrected: {corrected}")

    def _next_bout(self):
        selection = self.tree.selection()
        if not selection:
            return
        next_item = self.tree.next(selection[0])
        if next_item:
            self.tree.selection_set(next_item)
            self.tree.see(next_item)

    def _prev_bout(self):
        selection = self.tree.selection()
        if not selection:
            return
        prev_item = self.tree.prev(selection[0])
        if prev_item:
            self.tree.selection_set(prev_item)
            self.tree.see(prev_item)
            
    def _confirm_bout(self):
        self._update_bout_status('confirmed')

    def _reject_bout(self):
        self._update_bout_status('rejected')

    def _correct_bout_label(self):
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showwarning("No Selection", "Please select a bout from the list.", parent=self)
            return
        corrected_behavior = str(self.corrected_behavior_var.get() or '').strip()
        if not corrected_behavior:
            messagebox.showwarning("Missing Label", "Select a corrected behavior label first.", parent=self)
            return
        for item_id_str in selected_items:
            raw_match = self.raw_detected_bouts_df[
                self.raw_detected_bouts_df['Bout ID'].astype(str) == str(item_id_str)
            ]
            if raw_match.empty:
                continue
            self.review_decisions_df = append_review_decision(
                self.review_decisions_df,
                raw_match.iloc[0],
                'corrected',
                corrected_behavior=corrected_behavior,
                reviewer_notes=self.review_notes_var.get(),
                reviewer=self.reviewer_var.get(),
            )
        self.original_bouts_df = build_review_workspace(
            self.raw_detected_bouts_df,
            self.review_decisions_df,
            fps=self.fps,
        )
        self._update_bout_list_display()
        self._update_stats_display()
        self._save_review_state(silent=True)
        self._select_next_unreviewed(selected_items[-1])

    def _select_next_unreviewed(self, current_item_id_str):
        items = list(self.tree.get_children())
        if not items:
            return
        try:
            start_index = items.index(str(current_item_id_str)) + 1
        except ValueError:
            start_index = 0
        ordered = items[start_index:] + items[:start_index]
        for item in ordered:
            match = self.original_bouts_df[self.original_bouts_df['Bout ID'].astype(str) == str(item)]
            if not match.empty and match.iloc[0]['status'] == 'unreviewed':
                self.tree.selection_set(item)
                self.tree.see(item)
                self._on_bout_select()
                break

    def _jump_to_next_unreviewed(self):
        selection = self.tree.selection()
        start = selection[0] if selection else None
        if start:
            self._select_next_unreviewed(start)
            return
        for item in self.tree.get_children():
            match = self.original_bouts_df[self.original_bouts_df['Bout ID'].astype(str) == str(item)]
            if not match.empty and match.iloc[0]['status'] == 'unreviewed':
                self.tree.selection_set(item)
                self.tree.see(item)
                break

    def _jump_to_prev_unreviewed(self):
        items = list(self.tree.get_children())
        if not items:
            return
        selection = self.tree.selection()
        if selection:
            try:
                idx = items.index(selection[0])
            except ValueError:
                idx = len(items)
        else:
            idx = len(items)
        for item in reversed(items[:idx]):
            match = self.original_bouts_df[self.original_bouts_df['Bout ID'].astype(str) == str(item)]
            if not match.empty and match.iloc[0]['status'] == 'unreviewed':
                self.tree.selection_set(item)
                self.tree.see(item)
                break

    def _export_to_csv(self):
        reviewed_df = self.original_bouts_df[self.original_bouts_df['status'] != 'unreviewed'].copy()
        if reviewed_df.empty:
            messagebox.showinfo("No Data", "No bouts have been reviewed yet.", parent=self)
            return OperationResult.cancel("No reviewed bouts are available to export.")
        total_reviewed = len(reviewed_df)
        total_confirmed = len(reviewed_df[reviewed_df['status'] == 'confirmed'])
        total_rejected = len(reviewed_df[reviewed_df['status'] == 'rejected'])
        total_corrected = len(reviewed_df[reviewed_df['status'] == 'corrected'])
        summary_message = (
            f"Export Summary:\n\n"
            f"Total Reviewed Bouts: {total_reviewed}\n"
            f"  - Confirmed (Correct): {total_confirmed}\n"
            f"  - Rejected (Incorrect): {total_rejected}\n\n"
            f"  - Corrected (Relabeled): {total_corrected}\n\n"
            f"Do you want to save this report?"
        )
        if not messagebox.askyesno("Export Reviewed Bouts", summary_message, parent=self):
            return OperationResult.cancel("Decision report export cancelled.")
        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            title="Save Reviewed Bouts Report",
            initialfile=f"reviewed_bouts_report_{os.path.basename(self.video_path)}.csv"
        )
        if not save_path:
            return OperationResult.cancel("Decision report export cancelled.")
        try:
            reviewed_df.to_csv(save_path, index=False)
            messagebox.showinfo("Success", f"Reviewed bouts report saved to:\n{save_path}", parent=self)
            return OperationResult.success("Decision report exported.", report_path=save_path)
        except Exception as e:
            messagebox.showerror("Export Error", f"An error occurred while saving the file:\n{e}", parent=self)
            return OperationResult.failure("Decision report export failed.", error=str(e))

    def _save_review_state(self, silent: bool = False):
        if not self.autosave_path:
            result = OperationResult.cancel("No review output path was configured.")
            self.last_save_result = result
            if callable(self.on_save_result):
                try:
                    self.on_save_result(result)
                except Exception as exc:
                    self.logger.warning("Bout review outcome callback failed: %s", exc)
            return result
        result = save_review_bundle(
            self.raw_detected_bouts_df,
            self.review_decisions_df,
            authoritative_path=self.autosave_path,
            fps=self.fps,
        )
        self.last_save_result = result
        if result.status in {OperationStatus.SUCCESS, OperationStatus.PARTIAL}:
            if callable(self.on_review_saved):
                try:
                    self.on_review_saved(self.original_bouts_df.copy())
                except Exception as exc:
                    self.logger.warning("Bout review saved, but the UI refresh callback failed: %s", exc)
            if not silent:
                paths = BoutReviewPaths.from_authoritative(self.autosave_path)
                if result.succeeded:
                    messagebox.showinfo(
                        "Review Saved",
                        f"Review complete. Authoritative bouts saved to:\n{paths.authoritative}",
                        parent=self,
                    )
                else:
                    messagebox.showinfo(
                        "Progress Saved",
                        f"Review progress saved to:\n{paths.workspace}",
                        parent=self,
                    )
        else:
            self.logger.error("Failed to save bout review: %s", result.error or result.message)
            messagebox.showerror(
                "Save Error",
                f"{result.message}\n{result.error}".strip(),
                parent=self,
            )
        if callable(self.on_save_result):
            try:
                self.on_save_result(result)
            except Exception as exc:
                self.logger.warning("Bout review outcome callback failed: %s", exc)
        return result

    def _on_closing(self):
        self._stop_playback()
        result = self._save_review_state(silent=True)
        if result.failed:
            return
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.destroy()
        self.logger.info("BoutConfirmationTool closed")
