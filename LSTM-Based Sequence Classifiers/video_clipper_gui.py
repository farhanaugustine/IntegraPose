# video_clipper_gui.py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import os
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VideoClipperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("IntegraPose - Video Clipper and Labeler")

        # --- Member Variables ---
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.fps = 30
        self.current_frame_num = 0
        self.start_frame = None
        self.end_frame = None
        self.selected_behavior = None
        self.key_mapping = {}
        self.clip_save_dir = "clips_for_labeling"
        self.is_playing = False
        self.after_id = None

        # Slider state flags
        self.programmatic_update = False   # guard for code-driven slider.set()
        self.slider_dragging = False       # true while user is holding mouse on slider
        self.was_playing_before_drag = False

        self.clips_to_save = []
        self.photo = None                  # keep reference to avoid GC
        self.canvas_image_id = None        # reuse one Canvas image

        # --- Load Config ---
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
            prep_config = config.get('dataset_preparation', {})
            self.key_mapping = prep_config.get('key_mapping', {})
            self.clip_save_dir = prep_config.get('source_directory', self.clip_save_dir)
            if not self.key_mapping:
                messagebox.showwarning("Config Warning", "No 'key_mapping' found. Labeling shortcuts disabled.")
        except FileNotFoundError:
            messagebox.showerror("Config Error", "config.json not found! Please create it.")
            self.root.destroy()
            return
        except json.JSONDecodeError:
            messagebox.showerror("Config Error", "config.json is not a valid JSON file.")
            self.root.destroy()
            return

        # --- GUI Layout ---
        top_frame = tk.Frame(root, padx=10, pady=5)
        top_frame.pack(fill=tk.X)
        tk.Button(top_frame, text="Load Video", command=self.load_video).pack(side=tk.LEFT)
        self.lbl_video_path = tk.Label(top_frame, text="No video loaded.", fg="gray")
        self.lbl_video_path.pack(side=tk.LEFT, padx=10)

        self.canvas = tk.Canvas(root, bg="black")
        self.canvas.pack()

        slider_frame = tk.Frame(root, padx=10, pady=5)
        slider_frame.pack(fill=tk.X)
        self.lbl_frame_info = tk.Label(slider_frame, text="Frame: 0 / 0", width=20)
        self.lbl_frame_info.pack(side=tk.RIGHT)

        # IMPORTANT: no command=... here. We’ll manage interactions via mouse bindings.
        self.slider = tk.Scale(slider_frame, from_=0, to=100, orient=tk.HORIZONTAL, showvalue=0)
        self.slider.pack(fill=tk.X, expand=True)

        # Mouse bindings to cleanly distinguish user drags from programmatic updates
        self.slider.bind("<ButtonPress-1>", self._on_slider_press)
        self.slider.bind("<B1-Motion>", self._on_slider_drag)
        self.slider.bind("<ButtonRelease-1>", self._on_slider_release)
        # Also handle clicks on the trough (jump without drag): on Windows/Linux,
        # ButtonRelease will still fire, so _on_slider_release covers it.

        playback_frame = tk.Frame(root)
        playback_frame.pack()
        tk.Button(playback_frame, text="<<", command=lambda: self.step_frame(-1)).pack(side=tk.LEFT, padx=5)
        self.btn_play_pause = tk.Button(playback_frame, text="Play", command=self.toggle_play_pause, width=10)
        self.btn_play_pause.pack(side=tk.LEFT, padx=5)
        tk.Button(playback_frame, text=">>", command=lambda: self.step_frame(1)).pack(side=tk.LEFT, padx=5)

        controls_frame = tk.Frame(root, padx=10, pady=5)
        controls_frame.pack(fill=tk.X)
        tk.Button(controls_frame, text="Set Start Frame", command=self.set_start_frame).pack(side=tk.LEFT, padx=5)
        self.lbl_start_frame = tk.Label(controls_frame, text="Start: None", fg="blue")
        self.lbl_start_frame.pack(side=tk.LEFT)
        tk.Button(controls_frame, text="Set End Frame", command=self.set_end_frame, padx=5).pack(side=tk.LEFT, padx=(20, 5))
        self.lbl_end_frame = tk.Label(controls_frame, text="End: None", fg="red")
        self.lbl_end_frame.pack(side=tk.LEFT)

        label_frame = tk.Frame(root, padx=10, pady=10)
        label_frame.pack(fill=tk.X)
        shortcuts = ", ".join([f"'{k}' for {v}" for k, v in self.key_mapping.items()]) if self.key_mapping else "None"
        tk.Label(label_frame, text=f"Labeling Keys: {shortcuts}").pack(side=tk.LEFT)
        self.lbl_selected_behavior = tk.Label(label_frame, text="Behavior: None", font=("Helvetica", 10, "bold"), fg="green")
        self.lbl_selected_behavior.pack(side=tk.RIGHT, padx=10)

        queue_frame = tk.LabelFrame(root, text="Clip Queue", padx=10, pady=10)
        queue_frame.pack(fill=tk.X, padx=10, pady=5)

        listbox_container = tk.Frame(queue_frame)
        listbox_container.pack(fill=tk.X, expand=True)
        self.clip_listbox = tk.Listbox(listbox_container, height=5)
        self.clip_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scrollbar = tk.Scrollbar(listbox_container, orient="vertical", command=self.clip_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill="y")
        self.clip_listbox.config(yscrollcommand=scrollbar.set)

        queue_button_frame = tk.Frame(queue_frame)
        queue_button_frame.pack(fill=tk.X, pady=(5, 0))
        tk.Button(queue_button_frame, text="Add Clip to Queue", command=self.add_clip_to_queue).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        tk.Button(queue_button_frame, text="Remove Selected", command=self.remove_selected_clip).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        action_frame = tk.Frame(root, padx=10, pady=10)
        action_frame.pack(fill=tk.X)
        self.btn_extract = tk.Button(action_frame, text="Extract All Queued Clips", command=self.extract_all_clips, bg="#22c55e", fg="white", font=("Helvetica", 10, "bold"))
        self.btn_extract.pack(fill=tk.X)

        self.lbl_status = tk.Label(root, text="Ready.", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.lbl_status.pack(side=tk.BOTTOM, fill=tk.X)

        self.root.bind("<Key>", self.on_key_press)
        self.root.bind("<space>", lambda event: self.toggle_play_pause())
        self.root.bind("<Delete>", lambda event: self.remove_selected_clip())
        self.root.bind("<BackSpace>", lambda event: self.remove_selected_clip())

    # ---------- Slider mouse handlers (no command= on Scale) ----------
    def _on_slider_press(self, _event):
        self.slider_dragging = True
        self.was_playing_before_drag = self.is_playing
        if self.is_playing:
            self.pause_video()  # pause while dragging for responsiveness

    def _on_slider_drag(self, _event):
        # Don't seek on every pixel move; just update the thumb visually.
        # We’ll do one seek on release.
        pass

    def _on_slider_release(self, _event):
        self.slider_dragging = False
        # Perform a single seek at the final position of the thumb.
        self.seek_to_frame(int(self.slider.get()))
        # If the user had been playing, resume automatically.
        if self.was_playing_before_drag:
            self.play_video()

    # ------------------------------------------------------------------

    def load_video(self):
        self.pause_video()
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if not path:
            return

        self.video_path = Path(path)
        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open video file.")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Robust FPS handling (covers 0, None, NaN)
        fps_val = self.cap.get(cv2.CAP_PROP_FPS)
        try:
            if not fps_val or fps_val != fps_val or fps_val <= 0:
                fps_val = 30
        except Exception:
            fps_val = 30
        self.fps = fps_val

        self.slider.config(to=self.total_frames - 1)
        self.lbl_video_path.config(text=self.video_path.name)
        self.reset_clip_selection()
        self.seek_to_frame(0)
        self.lbl_status.config(text=f"Loaded {self.video_path.name}")

    def _display_frame(self, frame):
        if frame is None:
            return

        h, w, _ = frame.shape
        max_h, max_w = 540, 960
        if h > max_h or w > max_w:
            scale = min(max_h / h, max_w / w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        self.photo = ImageTk.PhotoImage(image=img)

        self.canvas.config(width=self.photo.width(), height=self.photo.height())
        if self.canvas_image_id is None:
            self.canvas_image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        else:
            self.canvas.itemconfig(self.canvas_image_id, image=self.photo)

        self.lbl_frame_info.config(text=f"Frame: {self.current_frame_num} / {self.total_frames - 1}")

    def seek_to_frame(self, frame_num):
        if not self.cap:
            return
        frame_num = max(0, min(int(frame_num), self.total_frames - 1))
        self.current_frame_num = frame_num
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_num)
        ret, frame = self.cap.read()
        if ret:
            self._display_frame(frame)

        # Safe programmatic slider update (will not trigger any callbacks)
        self.programmatic_update = True
        try:
            self.slider.set(self.current_frame_num)
        finally:
            self.programmatic_update = False

    def toggle_play_pause(self):
        if self.is_playing:
            self.pause_video()
        else:
            self.play_video()

    def play_video(self):
        if not self.cap:
            return
        if self.current_frame_num >= self.total_frames - 1:
            self.seek_to_frame(0)  # Restart if at the end

        self.is_playing = True
        self.btn_play_pause.config(text="Pause")
        self.playback_loop()

    def pause_video(self):
        if self.after_id:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None
        self.is_playing = False
        self.btn_play_pause.config(text="Play")

    def playback_loop(self):
        if not self.is_playing:
            return

        ret, frame = self.cap.read()
        if ret:
            self.current_frame_num += 1
            self._display_frame(frame)

            # Programmatic slider update (no command bound, so safe)
            self.programmatic_update = True
            try:
                self.slider.set(self.current_frame_num)
            finally:
                self.programmatic_update = False

            delay_ms = max(1, int(1000 / self.fps))
            self.after_id = self.root.after(delay_ms, self.playback_loop)
        else:
            # End of video (or read hiccup)
            self.pause_video()

    def step_frame(self, delta):
        if not self.cap:
            return
        self.pause_video()
        new_frame = self.current_frame_num + delta
        if 0 <= new_frame < self.total_frames:
            self.seek_to_frame(new_frame)

    def on_key_press(self, event):
        # Only map keys when focus is on the root or canvas (prevents typing into other widgets)
        if not isinstance(event.widget, (tk.Canvas, tk.Tk)):
            return
        key = event.char.lower()
        if key in self.key_mapping:
            self.selected_behavior = self.key_mapping[key]
            self.lbl_selected_behavior.config(text=f"Behavior: {self.selected_behavior}")
            self.lbl_status.config(text=f"Behavior '{self.selected_behavior}' selected.")

    def set_start_frame(self):
        self.pause_video()
        self.start_frame = self.current_frame_num
        self.lbl_start_frame.config(text=f"Start: {self.start_frame}")
        self.lbl_status.config(text=f"Start frame set to {self.start_frame}.")

    def set_end_frame(self):
        self.pause_video()
        self.end_frame = self.current_frame_num
        self.lbl_end_frame.config(text=f"End: {self.end_frame}")
        self.lbl_status.config(text=f"End frame set to {self.end_frame}.")

    def reset_clip_selection(self):
        self.start_frame, self.end_frame, self.selected_behavior = None, None, None
        self.lbl_start_frame.config(text="Start: None")
        self.lbl_end_frame.config(text="End: None")
        self.lbl_selected_behavior.config(text="Behavior: None")

    def add_clip_to_queue(self):
        if not all([self.start_frame is not None, self.end_frame is not None, self.selected_behavior]):
            messagebox.showwarning("Incomplete Selection", "Please set a start frame, end frame, and select a behavior first.")
            return
        if self.start_frame >= self.end_frame:
            messagebox.showerror("Invalid Range", "Start frame must be before end frame.")
            return

        clip_data = {
            "behavior": self.selected_behavior,
            "start": self.start_frame,
            "end": self.end_frame
        }
        self.clips_to_save.append(clip_data)

        listbox_text = f"'{clip_data['behavior']}' | Frames: {clip_data['start']} -> {clip_data['end']}"
        self.clip_listbox.insert(tk.END, listbox_text)
        self.clip_listbox.see(tk.END)

        self.lbl_status.config(text=f"Added clip for '{clip_data['behavior']}' to the queue.")
        self.reset_clip_selection()

    def remove_selected_clip(self):
        selected_indices = self.clip_listbox.curselection()
        if not selected_indices:
            self.lbl_status.config(text="No clip selected to remove.")
            return

        for index in reversed(selected_indices):
            self.clip_listbox.delete(index)
            removed = self.clips_to_save.pop(index)
            self.lbl_status.config(text=f"Removed clip for '{removed['behavior']}' from queue.")

    def extract_all_clips(self):
        self.pause_video()
        if not self.video_path:
            messagebox.showerror("Error", "Please load a video first.")
            return
        if not self.clips_to_save:
            messagebox.showinfo("Queue Empty", "There are no clips in the queue to extract.")
            return

        self.lbl_status.config(text="Starting batch extraction...")
        self.root.update_idletasks()

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to read frame dimensions for saving.")
            return
        height, width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        saved_count = 0
        failed_count = 0

        for i, clip_data in enumerate(self.clips_to_save):
            behavior = clip_data['behavior']
            start = clip_data['start']
            end = clip_data['end']

            self.lbl_status.config(text=f"Extracting clip {i+1}/{len(self.clips_to_save)} ('{behavior}')...")
            self.root.update_idletasks()

            behavior_dir = Path(self.clip_save_dir) / behavior
            os.makedirs(behavior_dir, exist_ok=True)

            base_filename = f"{self.video_path.stem}_{behavior}_frames_{start}_{end}"
            save_path = behavior_dir / f"{base_filename}.mp4"
            counter = 1
            while save_path.exists():
                save_path = behavior_dir / f"{base_filename}_{counter}.mp4"
                counter += 1

            try:
                writer = cv2.VideoWriter(str(save_path), fourcc, self.fps, (width, height))

                # Seek to the start of the clip before writing frames
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                for frame_idx in range(start, end + 1):
                    ret_read, frame_to_write = self.cap.read()
                    if ret_read:
                        writer.write(frame_to_write)
                    else:
                        logging.warning(f"Could not read frame {frame_idx} while writing clip {save_path}")
                        break

                writer.release()
                saved_count += 1
            except Exception as e:
                logging.error(f"Failed to save clip {save_path}: {e}")
                failed_count += 1

        self.clips_to_save.clear()
        self.clip_listbox.delete(0, tk.END)

        summary_message = f"Extraction Complete!\n\nSuccessfully saved: {saved_count} clips.\nFailed: {failed_count} clips."
        messagebox.showinfo("Success", summary_message)
        self.lbl_status.config(text="Batch extraction finished. Ready for next video.")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoClipperApp(root)
    root.mainloop()
