import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import pandas as pd
from PIL import Image, ImageTk
import os
import logging

class BoutConfirmationTool(tk.Toplevel):
    def __init__(self, master, video_path, detected_bouts_df, behavior_map):
        super().__init__(master)
        self.title("Bout Confirmation Tool")
        self.geometry("1300x850")
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Setup logging
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", filename="bout_tool.log")
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing BoutConfirmationTool")

        # Initialize Data & State
        self.original_bouts_df = detected_bouts_df.copy()
        self.behavior_map = behavior_map
        self.video_path = video_path
        self.is_playing = False
        self.after_id = None
        self.playback_speed = 1.0
        self.current_frame = 0
        self.photo = None

        # Validate required columns
        self.id_column_name = 'Track ID' if 'Track ID' in self.original_bouts_df.columns else 'Animal ID'
        required_columns = [self.id_column_name, 'Behavior', 'Start Frame', 'End Frame']
        missing_cols = [col for col in required_columns if col not in self.original_bouts_df.columns]
        if missing_cols:
            messagebox.showerror("Data Error", f"Missing required columns: {', '.join(missing_cols)}", parent=self)
            self.destroy()
            return
            
        if 'status' not in self.original_bouts_df.columns:
            self.original_bouts_df['status'] = 'unreviewed'

        # Initialize Video Capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Video Error", f"Could not open video file: {video_path}", parent=self)
            self.logger.error(f"Failed to open video: {video_path}")
            self.destroy()
            return
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30
            messagebox.showwarning("FPS Warning", "Could not determine video FPS. Using default 30 FPS.", parent=self)
            self.logger.warning("FPS not detected, defaulting to 30")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames <= 0:
            self.logger.warning("Invalid frame count, estimating frames")
            self.total_frames = self._estimate_frame_count()
            if self.total_frames <= 0:
                messagebox.showwarning("Video Warning", "Could not determine frame count. Some features may not work.", parent=self)
                self.total_frames = 10000  # Arbitrary fallback
        self.logger.info(f"Video loaded: FPS={self.fps}, Total Frames={self.total_frames}")

        # UI Setup
        self._create_widgets()
        self._populate_filter_options()
        self._update_bout_list_display()
        self._update_stats_display()
        
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
        
        export_btn = ttk.Button(left_panel, text="Export Reviewed Bouts Report...", command=self._export_to_csv)
        export_btn.grid(row=4, column=0, sticky='ew')

        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky='nsew')
        right_panel.rowconfigure(0, weight=1)
        right_panel.columnconfigure(0, weight=1)

        self.video_label = ttk.Label(right_panel, background="black")
        self.video_label.grid(row=0, column=0, sticky='nsew')

        video_controls = self._create_video_controls(right_panel)
        video_controls.grid(row=1, column=0, sticky='ew', pady=5)
        
        confirmation_controls = self._create_confirmation_controls(right_panel)
        confirmation_controls.grid(row=2, column=0, sticky='ew', pady=10)

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
        self.status_filter_combo = ttk.Combobox(parent, textvariable=self.status_filter_var, state="readonly", values=["All", "unreviewed", "confirmed", "rejected"])
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
        ttk.Label(parent, textvariable=self.total_bouts_var).grid(row=0, column=0, sticky='w', padx=5)
        ttk.Label(parent, textvariable=self.reviewed_bouts_var).grid(row=0, column=1, sticky='w', padx=5)
        ttk.Label(parent, textvariable=self.confirmed_bouts_var).grid(row=1, column=0, sticky='w', padx=5)
        ttk.Label(parent, textvariable=self.rejected_bouts_var).grid(row=1, column=1, sticky='w', padx=5)

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
        style = ttk.Style()
        style.configure("Confirm.TButton", font=('Helvetica', 12, 'bold'), foreground="green")
        style.configure("Reject.TButton", font=('Helvetica', 12, 'bold'), foreground="red")
        confirm_btn = ttk.Button(frame, text="✔ Confirm Bout", command=self._confirm_bout, style="Confirm.TButton")
        reject_btn = ttk.Button(frame, text="✖ Reject Bout", command=self._reject_bout, style="Reject.TButton")
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        confirm_btn.grid(row=0, column=0, sticky='ew', padx=5, ipady=5)
        reject_btn.grid(row=0, column=1, sticky='ew', padx=5, ipady=5)
        return frame

    def _populate_filter_options(self):
        if self.original_bouts_df.empty: return
        track_ids = sorted(self.original_bouts_df[self.id_column_name].unique().tolist())
        self.track_id_filter_combo['values'] = ["All"] + [str(tid) for tid in track_ids]
        behaviors = sorted(self.original_bouts_df['Behavior'].unique().tolist())
        self.behavior_filter_combo['values'] = ["All"] + behaviors

    def _update_bout_list_display(self, event=None):
        self._stop_playback()
        for item in self.tree.get_children(): self.tree.delete(item)
        
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
            self.tree.insert('', 'end', iid=index, values=values, tags=(row['status'],))
        
        children = self.tree.get_children()
        if children:
            self.tree.selection_set(children[0])
            self._on_bout_select()

    def _get_current_bout_info(self):
        selected_items = self.tree.selection()
        if not selected_items: return None
        item_id = int(selected_items[0])
        return self.original_bouts_df.loc[item_id]

    def _on_bout_select(self, event=None):
        self._stop_playback()
        bout_info = self._get_current_bout_info()
        if bout_info is None: return
        start_frame = int(bout_info['Start Frame'])
        self._seek_to_frame(start_frame)

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
        if not self.is_playing: return

        bout_info = self._get_current_bout_info()
        if bout_info is None: self._stop_playback(); return
        end_frame = int(bout_info['End Frame'])

        if self.current_frame >= end_frame:
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
            
        self.current_frame += 1
        self._update_video_display(frame)
        self.current_frame_var.set(f"Frame: {self.current_frame}")
        
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
        self.logger.debug(f"Successfully sought and displayed frame {frame_num}")

    def _update_video_display(self, frame):
        try:
            self.update_idletasks()
            l_w, l_h = self.video_label.winfo_width(), self.video_label.winfo_height()
            if l_w <= 1 or l_h <= 1: return

            f_h, f_w, _ = frame.shape
            scale = min(l_w / f_w, l_h / f_h)
            new_w, new_h = int(f_w * scale), int(f_h * scale)
            if new_w <= 0 or new_h <= 0: return
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
        for item_id_str in selected_items:
            item_id = int(item_id_str)
            self.original_bouts_df.loc[item_id, 'status'] = status
            current_values = list(self.tree.item(item_id, 'values'))
            current_values[-1] = status
            self.tree.item(item_id, values=tuple(current_values), tags=(status,))
        self._update_stats_display()
        self._select_next_unreviewed(selected_items[-1])

    def _update_stats_display(self):
        total = len(self.original_bouts_df)
        status_counts = self.original_bouts_df['status'].value_counts()
        reviewed = status_counts.get('confirmed', 0) + status_counts.get('rejected', 0)
        confirmed = status_counts.get('confirmed', 0)
        rejected = status_counts.get('rejected', 0)
        self.total_bouts_var.set(f"Total Bouts: {total}")
        self.reviewed_bouts_var.set(f"Reviewed: {reviewed}")
        self.confirmed_bouts_var.set(f"Confirmed: {confirmed}")
        self.rejected_bouts_var.set(f"Rejected: {rejected}")

    def _next_bout(self):
        selection = self.tree.selection()
        if not selection: return
        next_item = self.tree.next(selection[0])
        if next_item:
            self.tree.selection_set(next_item)
            self.tree.see(next_item)

    def _prev_bout(self):
        selection = self.tree.selection()
        if not selection: return
        prev_item = self.tree.prev(selection[0])
        if prev_item:
            self.tree.selection_set(prev_item)
            self.tree.see(prev_item)
            
    def _confirm_bout(self):
        self._update_bout_status('confirmed')

    def _reject_bout(self):
        self._update_bout_status('rejected')

    def _select_next_unreviewed(self, current_item_id_str):
        next_item_id = self.tree.next(current_item_id_str)
        if next_item_id:
            self.tree.selection_set(next_item_id)
            self.tree.see(next_item_id)
        else:
            for item in self.tree.get_children():
                if self.original_bouts_df.loc[int(item), 'status'] == 'unreviewed':
                    self.tree.selection_set(item)
                    self.tree.see(item)
                    break

    def _export_to_csv(self):
        reviewed_df = self.original_bouts_df[self.original_bouts_df['status'] != 'unreviewed'].copy()
        if reviewed_df.empty:
            messagebox.showinfo("No Data", "No bouts have been reviewed yet.", parent=self)
            return
        total_reviewed = len(reviewed_df)
        total_confirmed = len(reviewed_df[reviewed_df['status'] == 'confirmed'])
        total_rejected = len(reviewed_df[reviewed_df['status'] == 'rejected'])
        summary_message = (
            f"Export Summary:\n\n"
            f"Total Reviewed Bouts: {total_reviewed}\n"
            f"  - Confirmed (Correct): {total_confirmed}\n"
            f"  - Rejected (Incorrect): {total_rejected}\n\n"
            f"Do you want to save this report?"
        )
        if not messagebox.askyesno("Export Reviewed Bouts", summary_message, parent=self):
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            title="Save Reviewed Bouts Report",
            initialfile=f"reviewed_bouts_report_{os.path.basename(self.video_path)}.csv"
        )
        if save_path:
            try:
                reviewed_df.to_csv(save_path, index=False)
                messagebox.showinfo("Success", f"Reviewed bouts report saved to:\n{save_path}", parent=self)
            except Exception as e:
                messagebox.showerror("Export Error", f"An error occurred while saving the file:\n{e}", parent=self)

    def _on_closing(self):
        self._stop_playback()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.destroy()
        self.logger.info("BoutConfirmationTool closed")