import json
import logging
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any, Dict, List, Optional

import numpy as np
import cv2
from PIL import Image, ImageTk

from integra_pose.gui.theme import apply_global_ttk_theme

logger = logging.getLogger(__name__)

class VideoClipperApp:
    """A GUI application for clipping and labeling video segments for dataset creation."""

    def __init__(self, root: tk.Tk, config: Dict[str, Any]):
        """
        Initializes the Video Clipper application.

        Args:
            root (tk.Tk): The root Tkinter window.
            config (Dict[str, Any]): The loaded configuration dictionary.
        """
        self.root = root
        self.root.title("IntegraPose - Video Clipper and Labeler")

        try:
            prep_config = config['dataset_preparation']
            self.key_mapping: Dict[str, str] = prep_config['key_mapping']
            self.clip_save_dir: str = prep_config['source_directory']
        except KeyError as e:
            messagebox.showerror("Config Error", f"Missing required key in config.json: {e}")
            self.root.destroy()
            return

        self.video_path: Optional[Path] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames: int = 0
        self.fps: float = 30.0
        self.current_frame_num: int = 0
        self.start_frame: Optional[int] = None
        self.end_frame: Optional[int] = None
        self.selected_behavior: Optional[str] = None
        self.is_playing: bool = False
        self.after_id: Optional[str] = None
        self.slider_dragging: bool = False
        self.was_playing_before_drag: bool = False
        self.clips_to_save: List[Dict] = []
        self.photo: Optional[ImageTk.PhotoImage] = None
        self.canvas_image_id: Optional[int] = None

        self.style = None
        if isinstance(self.root, tk.Tk):
            try:
                self.style = apply_global_ttk_theme(self.root)
            except Exception:
                self.style = None

        self._setup_gui()
        self._bind_events()

    def _setup_gui(self):
        """Creates and arranges all the GUI widgets."""
        container = tk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True)

        # Scrollable viewport so controls remain accessible on small or maximized windows
        self._scroll_canvas = tk.Canvas(container, highlightthickness=0)
        v_scrollbar = tk.Scrollbar(container, orient="vertical", command=self._scroll_canvas.yview)
        self._scroll_canvas.configure(yscrollcommand=v_scrollbar.set)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.content_frame = tk.Frame(self._scroll_canvas)
        self._canvas_window = self._scroll_canvas.create_window((0, 0), window=self.content_frame, anchor="nw")

        self.content_frame.bind(
            "<Configure>",
            lambda event: self._scroll_canvas.configure(scrollregion=self._scroll_canvas.bbox("all"))
        )
        self._scroll_canvas.bind(
            "<Configure>",
            lambda event: self._scroll_canvas.itemconfig(self._canvas_window, width=event.width)
        )

        self.content_frame.bind("<Enter>", lambda event: self._bind_mousewheel())
        self.content_frame.bind("<Leave>", lambda event: self._unbind_mousewheel())

        top_frame = tk.Frame(self.content_frame, padx=10, pady=5)
        top_frame.pack(fill=tk.X)
        tk.Button(top_frame, text="Load Video", command=self.load_video).pack(side=tk.LEFT)
        self.lbl_video_path = tk.Label(top_frame, text="No video loaded.", fg="gray")
        self.lbl_video_path.pack(side=tk.LEFT, padx=10)

        self.canvas = tk.Canvas(self.content_frame, bg="black")
        self.canvas.pack()

        slider_frame = tk.Frame(self.content_frame, padx=10, pady=5)
        slider_frame.pack(fill=tk.X)
        self.lbl_frame_info = tk.Label(slider_frame, text="Frame: 0 / 0", width=20, anchor='e')
        self.lbl_frame_info.pack(side=tk.RIGHT)
        self.slider = tk.Scale(slider_frame, from_=0, to=100, orient=tk.HORIZONTAL, showvalue=0)
        self.slider.pack(fill=tk.X, expand=True)

        playback_frame = tk.Frame(self.content_frame)
        playback_frame.pack()
        tk.Button(playback_frame, text="<<", command=lambda: self.step_frame(-1)).pack(side=tk.LEFT, padx=5)
        self.btn_play_pause = tk.Button(playback_frame, text="Play", command=self.toggle_play_pause, width=10)
        self.btn_play_pause.pack(side=tk.LEFT, padx=5)
        tk.Button(playback_frame, text=">>", command=lambda: self.step_frame(1)).pack(side=tk.LEFT, padx=5)

        controls_frame = tk.Frame(self.content_frame, padx=10, pady=5)
        controls_frame.pack(fill=tk.X)
        tk.Button(controls_frame, text="Set Start Frame", command=self.set_start_frame).pack(side=tk.LEFT, padx=5)
        self.lbl_start_frame = tk.Label(controls_frame, text="Start: None", fg="blue")
        self.lbl_start_frame.pack(side=tk.LEFT)
        tk.Button(controls_frame, text="Set End Frame", command=self.set_end_frame).pack(side=tk.LEFT, padx=(20, 5))
        self.lbl_end_frame = tk.Label(controls_frame, text="End: None", fg="red")
        self.lbl_end_frame.pack(side=tk.LEFT)

        label_frame = tk.Frame(self.content_frame, padx=10, pady=10)
        label_frame.pack(fill=tk.X)
        shortcuts = ", ".join([f"'{k}' for {v}" for k, v in self.key_mapping.items()])
        tk.Label(label_frame, text=f"Labeling Keys: {shortcuts}").pack(side=tk.LEFT)
        self.lbl_selected_behavior = tk.Label(label_frame, text="Behavior: None", font=("Helvetica", 10, "bold"), fg="green")
        self.lbl_selected_behavior.pack(side=tk.RIGHT, padx=10)

        queue_frame = tk.LabelFrame(self.content_frame, text="Clip Queue", padx=10, pady=10)
        queue_frame.pack(fill=tk.X, padx=10, pady=5)
        listbox_container = tk.Frame(queue_frame)
        listbox_container.pack(fill=tk.X, expand=True)
        self.clip_listbox = tk.Listbox(listbox_container, height=5)
        self.clip_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scrollbar = tk.Scrollbar(listbox_container, orient="vertical", command=self.clip_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill="y")
        self.clip_listbox.config(yscrollcommand=scrollbar.set)
        
        queue_buttons = tk.Frame(queue_frame)
        queue_buttons.pack(fill=tk.X, pady=(5, 0))
        tk.Button(queue_buttons, text="Add Clip to Queue", command=self.add_clip_to_queue).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        tk.Button(queue_buttons, text="Remove Selected", command=self.remove_selected_clip).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        action_frame = tk.Frame(self.content_frame, padx=10, pady=10)
        action_frame.pack(fill=tk.X)
        self.btn_extract = tk.Button(action_frame, text="Extract All Queued Clips", command=self.extract_all_clips, bg="#22c55e", fg="white", font=("Helvetica", 10, "bold"))
        self.btn_extract.pack(fill=tk.X)

        self.lbl_status = tk.Label(self.root, text="Ready. Load a video to begin.", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.lbl_status.pack(side=tk.BOTTOM, fill=tk.X)

    def _bind_events(self):
        """Binds all keyboard and mouse events."""
        self.slider.bind("<ButtonPress-1>", self._on_slider_press)
        self.slider.bind("<B1-Motion>", self._on_slider_drag)
        self.slider.bind("<ButtonRelease-1>", self._on_slider_release)
        self.root.bind("<Key>", self.on_key_press)
        self.root.bind("<space>", lambda event: self.toggle_play_pause())
        self.root.bind("<Delete>", lambda event: self.remove_selected_clip())
        self.root.bind("<BackSpace>", lambda event: self.remove_selected_clip())

    def _bind_mousewheel(self):
        """Enable mouse wheel scrolling when the pointer is over the scrollable area."""
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)
        self.root.bind_all("<Button-4>", self._on_mousewheel)  # Linux scroll up
        self.root.bind_all("<Button-5>", self._on_mousewheel)  # Linux scroll down

    def _unbind_mousewheel(self):
        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Button-4>")
        self.root.unbind_all("<Button-5>")

    def _on_mousewheel(self, event: tk.Event):
        direction = -1 if getattr(event, "delta", 0) > 0 or getattr(event, "num", None) == 4 else 1
        self._scroll_canvas.yview_scroll(direction, "units")


    def _on_slider_press(self, event: Any):
        self.slider_dragging = True
        self.was_playing_before_drag = self.is_playing
        if self.is_playing:
            self.pause_video()

    def _on_slider_drag(self, event: Any):
        pass

    def _on_slider_release(self, event: Any):
        self.slider_dragging = False
        self.seek_to_frame(self.slider.get())
        if self.was_playing_before_drag:
            self.play_video()

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
        fps_val = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps_val) if fps_val and fps_val > 0 else 30.0
        self.slider.config(to=self.total_frames - 1)
        self.lbl_video_path.config(text=self.video_path.name)
        self.reset_clip_selection()
        self.seek_to_frame(0)
        self.lbl_status.config(text=f"Loaded {self.video_path.name}")

    def _display_frame(self, frame: np.ndarray):
        if frame is None: return
        h, w, _ = frame.shape
        max_h, max_w = 640, 1060 #540 , 960
        scale = min(max_h / h, max_w / w) if h > max_h or w > max_w else 1
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.photo = ImageTk.PhotoImage(image=img)
        self.canvas.config(width=self.photo.width(), height=self.photo.height())
        if self.canvas_image_id is None:
            self.canvas_image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        else:
            self.canvas.itemconfig(self.canvas_image_id, image=self.photo)
        self.lbl_frame_info.config(text=f"Frame: {self.current_frame_num} / {self.total_frames - 1}")

    def seek_to_frame(self, frame_num: int):
        if not self.cap: return
        self.current_frame_num = max(0, min(int(frame_num), self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_num)
        ret, frame = self.cap.read()
        if ret:
            self._display_frame(frame)
        if not self.slider_dragging:
            self.slider.set(self.current_frame_num)

    def toggle_play_pause(self):
        if self.is_playing: self.pause_video()
        else: self.play_video()

    def play_video(self):
        if not self.cap: return
        if self.current_frame_num >= self.total_frames - 1: self.seek_to_frame(0)
        self.is_playing = True
        self.btn_play_pause.config(text="Pause")
        self.playback_loop()

    def pause_video(self):
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.is_playing = False
        self.btn_play_pause.config(text="Play")

    def playback_loop(self):
        if not self.is_playing or not self.cap: return
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_num += 1
            self._display_frame(frame)
            self.slider.set(self.current_frame_num)
            delay_ms = max(1, int(1000 / self.fps))
            self.after_id = self.root.after(delay_ms, self.playback_loop)
        else:
            self.pause_video()

    def step_frame(self, delta: int):
        if not self.cap: return
        self.pause_video()
        new_frame = self.current_frame_num + delta
        if 0 <= new_frame < self.total_frames: self.seek_to_frame(new_frame)

    def on_key_press(self, event: tk.Event):
        widget = event.widget
        if not isinstance(widget, (tk.Tk, tk.Toplevel, tk.Canvas, tk.Scale)):
            return
        keysym = getattr(event, "keysym", "")
        if keysym == "Left":
            self.step_frame(-1)
            return
        if keysym == "Right":
            self.step_frame(1)
            return
        key = (event.char or "").lower()
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
            messagebox.showwarning("Incomplete Selection", "Please set start, end, and behavior.")
            return
        if self.start_frame >= self.end_frame:
            messagebox.showerror("Invalid Range", "Start frame must be before end frame.")
            return
        clip_data = {"behavior": self.selected_behavior, "start": self.start_frame, "end": self.end_frame}
        self.clips_to_save.append(clip_data)
        listbox_text = f"'{clip_data['behavior']}' | Frames: {clip_data['start']} -> {clip_data['end']}"
        self.clip_listbox.insert(tk.END, listbox_text)
        self.clip_listbox.see(tk.END)
        self.lbl_status.config(text=f"Added clip for '{clip_data['behavior']}' to queue.")
        self.reset_clip_selection()

    def remove_selected_clip(self):
        indices = self.clip_listbox.curselection()
        if not indices: return
        for index in reversed(indices):
            self.clip_listbox.delete(index)
            removed = self.clips_to_save.pop(index)
            self.lbl_status.config(text=f"Removed '{removed['behavior']}' clip from queue.")

    def extract_all_clips(self):
        self.pause_video()
        if not self.video_path or not self.cap:
            messagebox.showerror("Error", "Please load a video first.")
            return
        if not self.clips_to_save:
            messagebox.showinfo("Queue Empty", "No clips in the queue to extract.")
            return

        self.lbl_status.config(text="Starting batch extraction..."); self.root.update_idletasks()
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret or frame is None:
                messagebox.showerror("Error", "Failed to determine video dimensions for saving.")
                return
            height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        saved_count, failed_count = 0, 0

        for i, clip_data in enumerate(self.clips_to_save):
            behavior, start, end = clip_data['behavior'], clip_data['start'], clip_data['end']
            self.lbl_status.config(text=f"Extracting clip {i+1}/{len(self.clips_to_save)} ('{behavior}')...")
            self.root.update_idletasks()
            behavior_dir = Path(self.clip_save_dir) / behavior
            os.makedirs(behavior_dir, exist_ok=True)
            base_filename = f"{self.video_path.stem}_{behavior}_frames_{start}_{end}"
            save_path, counter = behavior_dir / f"{base_filename}.mp4", 1
            while save_path.exists():
                save_path = behavior_dir / f"{base_filename}_{counter}.mp4"; counter += 1
            try:
                writer = cv2.VideoWriter(str(save_path), fourcc, self.fps, (width, height))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                for _ in range(start, end + 1):
                    ret_read, frame_to_write = self.cap.read()
                    if ret_read: writer.write(frame_to_write)
                    else: break
                writer.release()
                saved_count += 1
            except Exception as e:
                logger.error(f"Failed to save clip {save_path}: {e}"); failed_count += 1

        self.clips_to_save.clear(); self.clip_listbox.delete(0, tk.END)
        summary = f"Extraction Complete!\n\nSaved: {saved_count}\nFailed: {failed_count}"
        messagebox.showinfo("Success", summary)
        self.lbl_status.config(text="Batch extraction finished. Ready for next video.")

def launch_clipper_gui(config: Dict[str, Any], parent: Optional[tk.Misc] = None):
    """Launch the clipper UI, optionally attached to a parent window."""
    owns_parent = False
    if parent is None:
        root = tk.Tk()
        owns_parent = True
    else:
        root = tk.Toplevel(parent)
    app = VideoClipperApp(root, config)
    if owns_parent:
        root.mainloop()
    else:
        root.focus_set()
        root.lift()
    return app

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        with open('../config.json', 'r') as f: # Assumes config is in the parent directory
            config_data = json.load(f)
        launch_clipper_gui(config_data)
    except FileNotFoundError:
        logger.error("config.json not found in the parent directory. Cannot launch GUI.")
    except json.JSONDecodeError:
        logger.error("Error decoding config.json. Please check its format.")
