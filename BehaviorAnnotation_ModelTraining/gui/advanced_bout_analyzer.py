import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import pandas as pd
import os
import traceback
from PIL import Image, ImageTk

class AdvancedBoutAnalyzer(tk.Toplevel):
    """
    A Toplevel window for manually scoring and verifying behavioral bouts from a video.
    This version is integrated into the main application and receives data directly.
    """

    def __init__(self, parent, video_path, initial_bout_details, behavior_map, all_bouts_df):
        """
        Initializes the Advanced Bout Scorer.

        Args:
            parent: The parent window (main app).
            video_path (str): Path to the video file.
            initial_bout_details (pd.Series or dict): Details of the bout to preload.
            behavior_map (dict): Mapping from behavior names to their class IDs.
            all_bouts_df (pd.DataFrame): The complete DataFrame of all detected bouts.
        """
        super().__init__(parent)
        self.title("Advanced Bout Scorer")
        self.geometry("1200x800")
        self.minsize(1000, 700)
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        # --- Initialize State & Data ---
        self.cap = None
        self.video_path = video_path
        self.all_bouts_df = all_bouts_df
        self.behavior_map = behavior_map # e.g., {'Walking': 1, 'Standing': 0}
        self.behavior_names_list = sorted(self.behavior_map.keys())

        self.total_frames = 0
        self.video_fps = 30
        self.is_playing = False
        self.is_dragging_scale = False
        self.scored_bouts = []

        # --- Create UI ---
        self._create_widgets()
        
        # --- Auto-load data ---
        if self.video_path:
            self._load_video(self.video_path)
            self._populate_track_ids()
            # Use after() to ensure main window is ready before preloading
            self.after(100, lambda: self._preload_bout_data(initial_bout_details))

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.rowconfigure(1, weight=1)
        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=1)

        # --- Video Display and Controls ---
        video_frame = ttk.Frame(main_frame)
        video_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))
        video_frame.rowconfigure(0, weight=1)
        video_frame.columnconfigure(0, weight=1)

        self.video_label = ttk.Label(video_frame, background="black", anchor=tk.CENTER)
        self.video_label.grid(row=0, column=0, sticky="nsew")
        
        controls_frame = self._create_video_controls(video_frame)
        controls_frame.grid(row=1, column=0, sticky="ew", pady=5)

        # --- Scoring and Results ---
        scoring_frame = ttk.LabelFrame(main_frame, text="Scoring", padding=10)
        scoring_frame.grid(row=0, column=1, sticky="new")
        self._populate_scoring_frame(scoring_frame)

        results_frame = ttk.LabelFrame(main_frame, text="Scored Bouts", padding=10)
        results_frame.grid(row=1, column=1, sticky="nsew", pady=(10, 0))
        self._populate_results_frame(results_frame)
        
    def _create_video_controls(self, parent):
        controls_frame = ttk.Frame(parent)
        controls_frame.columnconfigure(1, weight=1)

        self.play_pause_btn = ttk.Button(controls_frame, text="Play", command=self._toggle_play_pause, state=tk.DISABLED)
        self.play_pause_btn.grid(row=0, column=0, padx=5)
        
        self.video_scale = ttk.Scale(controls_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self._on_scale_drag, state=tk.DISABLED)
        self.video_scale.grid(row=0, column=1, sticky="ew", padx=5)

        self.time_label_var = tk.StringVar(value="F: 0 / 0 | T: 00:00.0 / 00:00.0")
        time_label = ttk.Label(controls_frame, textvariable=self.time_label_var)
        time_label.grid(row=0, column=2, padx=5)
        return controls_frame

    def _populate_scoring_frame(self, frame):
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Animal ID:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.track_id_var = tk.StringVar()
        self.track_id_combo = ttk.Combobox(frame, textvariable=self.track_id_var, state="readonly")
        self.track_id_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=2)

        ttk.Label(frame, text="Behavior to Score:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.behavior_var = tk.StringVar()
        self.behavior_combo = ttk.Combobox(frame, textvariable=self.behavior_var, values=self.behavior_names_list, state="readonly")
        self.behavior_combo.grid(row=1, column=1, sticky="ew", padx=5, pady=2)

        self.start_frame_var = tk.StringVar()
        self.end_frame_var = tk.StringVar()
        
        timing_frame = ttk.Frame(frame)
        timing_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        set_start_btn = ttk.Button(timing_frame, text="Set Start Frame", command=self._set_start_frame)
        set_start_btn.pack(side=tk.LEFT, padx=5)
        ttk.Entry(timing_frame, textvariable=self.start_frame_var, width=10, state="readonly").pack(side=tk.LEFT)
        
        set_end_btn = ttk.Button(timing_frame, text="Set End Frame", command=self._set_end_frame)
        set_end_btn.pack(side=tk.LEFT, padx=10)
        ttk.Entry(timing_frame, textvariable=self.end_frame_var, width=10, state="readonly").pack(side=tk.LEFT)

        action_frame = ttk.Frame(frame)
        action_frame.grid(row=3, column=0, columnspan=2, pady=10)
        log_bout_btn = ttk.Button(action_frame, text="Log Bout", command=self._log_bout) # Style removed for compatibility
        log_bout_btn.pack(side=tk.LEFT, padx=5)
        clear_btn = ttk.Button(action_frame, text="Clear Timings", command=self._clear_timings)
        clear_btn.pack(side=tk.LEFT, padx=5)

    def _populate_results_frame(self, frame):
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        
        cols = ("Animal ID", "Behavior", "Start Frame", "End Frame", "Duration (s)")
        self.results_tree = ttk.Treeview(frame, columns=cols, show="headings")
        for col in cols:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=100)
        self.results_tree.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.results_tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.results_tree.configure(yscrollcommand=scrollbar.set)

        export_btn = ttk.Button(frame, text="Export to CSV...", command=self._export_csv)
        export_btn.grid(row=1, column=0, columnspan=2, pady=(5, 0))

    def _populate_track_ids(self):
        """Populates the track ID dropdown from the main analysis results."""
        if self.all_bouts_df is not None and not self.all_bouts_df.empty:
            id_col = 'Track ID' if 'Track ID' in self.all_bouts_df.columns else 'Animal ID'
            if id_col in self.all_bouts_df.columns:
                all_track_ids = sorted(self.all_bouts_df[id_col].unique().tolist())
                self.track_id_combo['values'] = all_track_ids
            else:
                 messagebox.showwarning("Warning", "Could not find a 'Track ID' or 'Animal ID' column in the provided data.", parent=self)

    def _load_video(self, filepath):
        if self.is_playing: self._toggle_play_pause()
        
        if self.cap: self.cap.release()
        self.cap = cv2.VideoCapture(filepath)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open video file.", parent=self)
            return

        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.play_pause_btn.config(state=tk.NORMAL)
        self.video_scale.config(state=tk.NORMAL, to=self.total_frames - 1)
        self._update_frame(is_seek=True)

    def _preload_bout_data(self, details):
        """Pre-populates the scoring fields with details from the selected bout."""
        try:
            # Handle both Series and dict for details
            get_detail = details.get if isinstance(details, dict) else lambda key: details[key]

            animal_id = get_detail('Track ID')
            if animal_id is not None:
                self.track_id_var.set(animal_id)

            behavior = get_detail('Behavior')
            if behavior: self.behavior_var.set(behavior)

            start_frame = get_detail('Start Frame')
            if start_frame is not None:
                seek_frame = int(start_frame)
            else: return # Cannot proceed without a start frame

            self.start_frame_var.set(str(seek_frame))
            self.video_scale.set(seek_frame)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, seek_frame)
            self._update_frame(is_seek=True)

        except Exception as e:
            messagebox.showerror("Preload Error", f"Could not load bout details: {e}", parent=self)
            traceback.print_exc()

    def _toggle_play_pause(self):
        self.is_playing = not self.is_playing
        self.play_pause_btn.config(text="Pause" if self.is_playing else "Play")
        if self.is_playing:
            self._update_frame()

    def _update_frame(self, is_seek=False):
        if not self.cap or not (self.is_playing or is_seek):
            return

        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_frame >= self.total_frames:
            self.is_playing = False
            self.play_pause_btn.config(text="Play")
            return

        ret, frame = self.cap.read()
        if ret:
            l_w, l_h = self.video_label.winfo_width(), self.video_label.winfo_height()
            if l_w > 1 and l_h > 1:
                f_h, f_w, _ = frame.shape
                scale = min(l_w / f_w, l_h / f_h)
                new_w, new_h = int(f_w * scale), int(f_h * scale)
                
                resized_frame = cv2.resize(frame, (new_w, new_h))
                img = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                photo_image = ImageTk.PhotoImage(image=Image.fromarray(img))
                
                self.video_label.config(image=photo_image)
                self.video_label.image = photo_image

            if not self.is_dragging_scale:
                self.video_scale.set(current_frame)
            self._update_time_label(current_frame)
        else:
            self.is_playing = False
            self.play_pause_btn.config(text="Play")

        if self.is_playing:
            self.after(int(1000 / self.video_fps), self._update_frame)

    def _update_time_label(self, current_frame):
        total_s = self.total_frames / self.video_fps if self.video_fps > 0 else 0
        current_s = min(current_frame / self.video_fps, total_s) if self.video_fps > 0 else 0
        self.time_label_var.set(f"F: {current_frame} / {self.total_frames} | T: {current_s:06.2f}s / {total_s:06.2f}s")
    
    def _on_scale_drag(self, value_str):
        if self.is_playing: self._toggle_play_pause()
        
        self.is_dragging_scale = True
        if self.cap:
            frame_num = int(float(value_str))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            self._update_frame(is_seek=True)
        
        self.after_idle(lambda: setattr(self, 'is_dragging_scale', False))

    def _set_start_frame(self):
        if self.cap: self.start_frame_var.set(str(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))))

    def _set_end_frame(self):
        if self.cap: self.end_frame_var.set(str(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))))

    def _clear_timings(self):
        self.start_frame_var.set("")
        self.end_frame_var.set("")

    def _log_bout(self):
        track_id, behavior = self.track_id_var.get(), self.behavior_var.get()
        start_frame_str, end_frame_str = self.start_frame_var.get(), self.end_frame_var.get()

        if not all([track_id, behavior, start_frame_str, end_frame_str]):
            messagebox.showwarning("Incomplete Data", "All fields are required.", parent=self)
            return
        
        try:
            start_f, end_f = int(start_frame_str), int(end_frame_str)
            if start_f >= end_f:
                messagebox.showerror("Validation Error", "Start frame must be before end frame.", parent=self)
                return
            duration_s = (end_f - start_f) / self.video_fps if self.video_fps > 0 else 0
            
            bout_data = (track_id, behavior, start_f, end_f, f"{duration_s:.2f}")
            self.scored_bouts.append(bout_data)
            self.results_tree.insert("", "end", values=bout_data)
            self._clear_timings()
        except ValueError:
            messagebox.showerror("Invalid Frame", "Frame numbers must be valid integers.", parent=self)

    def _export_csv(self):
        if not self.scored_bouts:
            messagebox.showinfo("No Data", "No scored bouts to export.", parent=self)
            return
        
        initial_filename = f"manually_scored_{os.path.basename(self.video_path)}.csv"
        filepath = filedialog.asksaveasfilename(title="Save Scored Bouts", defaultextension=".csv", filetypes=[("CSV File", "*.csv")], initialfile=initial_filename)
        if filepath:
            try:
                df = pd.DataFrame(self.scored_bouts, columns=["Animal ID", "Behavior", "Start Frame", "End Frame", "Duration (s)"])
                df.to_csv(filepath, index=False)
                messagebox.showinfo("Success", f"Scored bouts exported to:\n{filepath}", parent=self)
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export CSV: {e}", parent=self)

    def _on_closing(self):
        self.is_playing = False 
        if self.cap:
            self.cap.release()
        self.destroy()
