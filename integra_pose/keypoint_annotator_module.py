import cv2
import os
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
import numpy as np
import sys
import yaml
from collections import OrderedDict
import time
from integra_pose.gui.scrollable import create_scrollable_section
from integra_pose.gui.windowing import apply_adaptive_window_geometry

KP_INVISIBLE_COLOR = (0, 0, 255)
BONE_COLOR = (0, 165, 255)
BBOX_CURRENT_TEMP_COLOR = (255, 255, 0)
BBOX_COMPLETED_COLOR = (255, 0, 255)
BBOX_HIGHLIGHT_COLOR = (0, 215, 255)
TEXT_COLOR = (200, 0, 0)
STATUS_TEXT_COLOR = (0, 0, 0)
HELP_TEXT_COLOR = (50, 50, 50)
ZOOM_PAN_TEXT_COLOR = (0, 100, 0)
AUTOSAVE_MSG_COLOR = (0, 0, 150)
BEHAVIOR_TEXT_ON_BOX_COLOR = (0,0,0)
BEHAVIOR_BOX_BACKGROUND_COLOR = (255, 255, 255)
HELPER_WINDOW_NAME = "Annotator Helper"

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_MAIN_INSTRUCTION = 0.6
FONT_SCALE_MODE_STATUS = 0.5
FONT_SCALE_GENERAL_STATUS = 0.45
FONT_SCALE_HELP_PROMPT = 0.4
FONT_SCALE_FULL_HELP = 0.45
FONT_SCALE_TOOLTIP = 0.45
FONT_SCALE_ANNOTATION_INFO = 0.4
FONT_SCALE_KP_NUMBER = 0.35

FONT_THICKNESS = 1
COMPLETED_LINE_THICKNESS = 1
CURRENT_LINE_THICKNESS = 2
BONE_LINE_THICKNESS = 1

MODE_SELECT_BEHAVIOR = "select_behavior"
MODE_KP = "keypoints"
MODE_BBOX_START = "bbox_start"
MODE_BBOX_END = "bbox_end"

VISIBILITY_NOT_LABELED = 0
VISIBILITY_LABELED_NOT_VISIBLE = 1
VISIBILITY_LABELED_VISIBLE = 2

AUTOSAVE_INTERVAL_SECONDS = 300
ENABLE_AUTOSAVE = True
MIN_WINDOW_DIM_THRESHOLD = 100
MIN_BBOX_DIMENSION = 4
DEFAULT_MARKER_SIZE_SCALE = 1.0
MIN_MARKER_SIZE_SCALE = 0.7
MAX_MARKER_SIZE_SCALE = 2.8
GRAB_RADIUS_BASE_PX = 12

KEYPOINT_PALETTE = [
    (255, 99, 71),
    (60, 179, 113),
    (30, 144, 255),
    (255, 215, 0),
    (186, 85, 211),
    (0, 206, 209),
    (255, 140, 0),
    (220, 20, 60),
    (46, 139, 87),
    (70, 130, 180),
    (199, 21, 133),
    (154, 205, 50),
]

SHORTCUT_LINES = [
    "1-9/0 : select behavior  |  N / P : next / prev image  |  G : go to image",
    "S : save annotations  |  R : reset this frame  |  Q : quit",
    "Drag keypoints to adjust  |  drag bbox to draw  |  U : undo",
    "+ / - : zoom  |  Alt+drag : pan  |  0 : reset view",
    "[ / ] : marker size  |  L : toggle labels  |  V : next KP visibility",
    "H/? : toggle helper  |  MMB : toggle last placed keypoint visibility",
]

class KeypointBehaviorAnnotator:
    def __init__(self, master_tk_window, keypoint_names, available_behaviors, image_dir, output_dir, show_keypoint_labels=False, skeleton_edges=None, use_info_panel=True, helper_popup_enabled=True):
        self.master_tk_window = master_tk_window

        if not keypoint_names:
            messagebox.showerror("Error", "Keypoint names list cannot be empty.", parent=self.master_tk_window)
            raise ValueError("Keypoint names list cannot be empty.")
        if not available_behaviors:
            messagebox.showerror("Error", "Available behaviors dictionary cannot be empty.", parent=self.master_tk_window)
            raise ValueError("Available behaviors dictionary cannot be empty.")

        self.keypoint_names = keypoint_names
        self.num_keypoints = len(keypoint_names)
        self.kp_name_to_index = {name: idx for idx, name in enumerate(self.keypoint_names)}
        self.available_behaviors = OrderedDict(sorted(available_behaviors.items()))
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.skeleton_edges = self._sanitize_skeleton_edges(skeleton_edges)
        self.use_info_panel = bool(use_info_panel)
        self.helper_popup_enabled = bool(helper_popup_enabled)
        self.info_panel = None
        self.helper_window = None
        self._last_info_payload = None
        self._annotated_stems = set()
        self._drag_state = None
        self._hover_target = None
        self._hovered_annotation_index = None
        self.marker_size_scale = float(DEFAULT_MARKER_SIZE_SCALE)
        self.panel_behavior_listbox = None
        self.panel_behavior_id_var = None
        self.panel_behavior_name_var = None
        self.panel_progress_var = None
        self.panel_marker_size_var = None

        self.image_files = self._get_image_files()
        if not self.image_files:
            messagebox.showerror("Error", f"No valid image files found in annotation image directory: {self.image_dir}", parent=self.master_tk_window)
            raise FileNotFoundError(f"No valid image files for annotation in {self.image_dir}")

        print(f"Annotator initialized for image directory: {self.image_dir}")
        print(f"Found {len(self.image_files)} images for annotation.")
        print(f"Output directory for annotations (labels): {self.output_dir}")
        print(f"Keypoint sequence: {self.keypoint_names}")
        print("Available Behaviors (ID: Name):")
        for bid, bname in self.available_behaviors.items():
            print(f"  {bid}: {bname}")

        self.current_image_index = 0
        self.current_image_path = None
        self.img_original_unmodified = None
        self.img_display = None
        self.img_height_orig, self.img_width_orig = 0, 0

        self.mode = MODE_SELECT_BEHAVIOR
        self.current_behavior_id = None
        self.current_behavior_name = None
        self.current_keypoint_index = 0
        self.current_keypoints = []
        self.current_bbox_start_point = None
        self.current_bbox_end_point = None
        self.annotations_in_image = []
        self.current_mouse_pos_screen = None
        self.current_mouse_pos_image = (0,0)

        self.current_kp_visibility_mode = VISIBILITY_LABELED_VISIBLE

        self.zoom_level = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        self.zoom_step_factor = 0.1
        self.pan_offset = np.array([0.0, 0.0])
        self.is_panning_with_alt = False
        self.pan_start_mouse_screen_pos = None
        self.pan_start_offset = None

        self.last_autosave_time = time.time()
        self.annotations_changed_since_last_save = False
        self.autosave_message = ""
        self.autosave_message_display_time = 0

        self.show_help_text = False
        self.show_keypoint_labels = bool(show_keypoint_labels)
        self.window_name = 'YOLO Keypoint Annotator - CV'
        self.default_window_width = 1280
        self.default_window_height = 720
        self.helper_popup_enabled = bool(helper_popup_enabled)
        self.helper_window_open = False

        if self.use_info_panel:
            self._init_info_panel()

        # Keep OpenCV single-threaded to avoid GUI/thread surprises with Tk on Windows.
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

    def _poll_key(self, delay_ms=30):
        """
        Retrieve the next key press without blocking Tk; prefer pollKey when available
        to avoid prolonged GIL release from waitKey on some Windows builds.
        """
        try:
            if hasattr(cv2, "pollKey"):
                key = cv2.pollKey()
                if delay_ms > 0:
                    time.sleep(delay_ms / 1000.0)
                return key & 0xFF
        except Exception:
            pass
        return cv2.waitKey(max(1, delay_ms)) & 0xFF

    def _sanitize_skeleton_edges(self, edges):
        if not edges:
            return []
        cleaned = []
        for edge in edges:
            try:
                a, b = int(edge[0]), int(edge[1])
            except Exception:
                continue
            if a == b:
                continue
            if 0 <= a < self.num_keypoints and 0 <= b < self.num_keypoints:
                cleaned.append((a, b))
        return cleaned

    def _init_info_panel(self):
        try:
            self.info_panel = tk.Toplevel(self.master_tk_window)
        except Exception:
            self.info_panel = None
            self.use_info_panel = False
            return

        self.info_panel.title("Annotator Controls")
        apply_adaptive_window_geometry(
            self.info_panel,
            preferred_size=(460, 720),
            min_size=(380, 420),
            width_ratio=0.5,
            height_ratio=0.88,
        )
        self.info_panel.protocol("WM_DELETE_WINDOW", self._destroy_info_panel)
        try:
            self.info_panel.attributes("-topmost", True)
        except Exception:
            pass

        self.panel_status_var = tk.StringVar(master=self.info_panel, value="")
        self.panel_instruction_var = tk.StringVar(master=self.info_panel, value="")
        self.panel_progress_var = tk.StringVar(master=self.info_panel, value="")
        self.panel_behavior_id_var = tk.StringVar(master=self.info_panel, value="")
        self.panel_behavior_name_var = tk.StringVar(master=self.info_panel, value="")
        self.panel_marker_size_var = tk.DoubleVar(master=self.info_panel, value=self.marker_size_scale)

        # Wrap the controls panel in a scrollable section so the panel
        # stays usable on small displays where the listboxes plus buttons
        # would otherwise be clipped below the visible area.
        scroll_parent = tk.Frame(self.info_panel)
        scroll_parent.pack(fill="both", expand=True)
        _annot_canvas, scroll_content = create_scrollable_section(self.master_tk_window, scroll_parent)
        root = tk.Frame(scroll_content, padx=10, pady=10)
        root.pack(fill="both", expand=True)

        header = tk.Label(root, text="Keypoint Annotation Controls", font=("Arial", 11, "bold"), anchor="w", justify="left")
        header.pack(fill="x", pady=(0, 6))

        status_label = tk.Label(root, textvariable=self.panel_status_var, justify="left", anchor="w", wraplength=410)
        status_label.pack(fill="x")
        progress_label = tk.Label(root, textvariable=self.panel_progress_var, justify="left", anchor="w", fg="#1f4f82", wraplength=410)
        progress_label.pack(fill="x", pady=(4, 0))

        separator = tk.Frame(root, height=1, bg="#cccccc")
        separator.pack(fill="x", pady=8)

        instruction_label = tk.Label(root, textvariable=self.panel_instruction_var, justify="left", anchor="w", wraplength=410)
        instruction_label.pack(fill="x")

        behavior_box = tk.LabelFrame(root, text="Behaviors", padx=8, pady=8)
        behavior_box.pack(fill="both", expand=False, pady=(10, 0))
        self.panel_behavior_listbox = tk.Listbox(behavior_box, height=7, exportselection=False)
        self.panel_behavior_listbox.pack(fill="x")
        self.panel_behavior_listbox.bind("<<ListboxSelect>>", self._on_panel_behavior_select)

        behavior_form = tk.Frame(behavior_box)
        behavior_form.pack(fill="x", pady=(8, 0))
        tk.Label(behavior_form, text="ID").grid(row=0, column=0, sticky="w")
        tk.Entry(behavior_form, textvariable=self.panel_behavior_id_var, width=8).grid(row=0, column=1, sticky="w", padx=(4, 10))
        tk.Label(behavior_form, text="Name").grid(row=0, column=2, sticky="w")
        tk.Entry(behavior_form, textvariable=self.panel_behavior_name_var).grid(row=0, column=3, sticky="ew", padx=(4, 0))
        behavior_form.grid_columnconfigure(3, weight=1)
        tk.Button(behavior_box, text="Add / Update Behavior", command=self._add_or_update_behavior_from_panel).pack(anchor="w", pady=(6, 0))

        actions_box = tk.LabelFrame(root, text="Navigation & Actions", padx=8, pady=8)
        actions_box.pack(fill="x", pady=(10, 0))
        button_grid = tk.Frame(actions_box)
        button_grid.pack(fill="x")
        for col in range(3):
            button_grid.grid_columnconfigure(col, weight=1)
        tk.Button(button_grid, text="Prev", command=self._prev_image).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        tk.Button(button_grid, text="Next", command=self._next_image).grid(row=0, column=1, sticky="ew", padx=4)
        tk.Button(button_grid, text="Jump...", command=self._jump_to_image_prompt).grid(row=0, column=2, sticky="ew", padx=(4, 0))
        tk.Button(button_grid, text="Save", command=self._save_annotations_for_image).grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=(6, 0))
        tk.Button(button_grid, text="Reset Frame", command=self._reset_frame_annotations).grid(row=1, column=1, sticky="ew", padx=4, pady=(6, 0))
        tk.Button(button_grid, text="Toggle Labels", command=lambda: self._toggle_keypoint_labels()).grid(row=1, column=2, sticky="ew", padx=(4, 0), pady=(6, 0))

        marker_box = tk.LabelFrame(root, text="Marker Size", padx=8, pady=8)
        marker_box.pack(fill="x", pady=(10, 0))
        tk.Scale(
            marker_box,
            from_=MIN_MARKER_SIZE_SCALE,
            to=MAX_MARKER_SIZE_SCALE,
            resolution=0.1,
            orient="horizontal",
            variable=self.panel_marker_size_var,
            command=self._set_marker_size_scale,
            length=320,
        ).pack(fill="x")

        static_help_lbl = tk.Label(root, text="\n".join(["Shortcuts:"] + [f"  {line}" for line in SHORTCUT_LINES]), justify="left", anchor="w", fg="#333333", wraplength=410)
        static_help_lbl.pack(fill="x", pady=(10, 0))

        if self.skeleton_edges:
            skeleton_lines = ["Skeleton edges:"]
            for a, b in self.skeleton_edges:
                name_a = self.keypoint_names[a] if a < len(self.keypoint_names) else f"kp{a+1}"
                name_b = self.keypoint_names[b] if b < len(self.keypoint_names) else f"kp{b+1}"
                skeleton_lines.append(f"  {a + 1}-{b + 1}: {name_a} <-> {name_b}")
            skeleton_lbl = tk.Label(root, text="\n".join(skeleton_lines), justify="left", anchor="w", fg="#444444", wraplength=410)
            skeleton_lbl.pack(fill="x", pady=(8, 0))

        self._refresh_behavior_listbox()

    def _helper_text_lines(self):
        body = ["Shortcuts:"]
        body.extend([f"  {line}" for line in SHORTCUT_LINES])
        body.append("")
        body.append("Workflow:")
        body.append("  Select behavior -> place/adjust keypoints -> drag bounding box -> save.")
        body.append("")
        if self.skeleton_edges:
            body.append("Skeleton edges (1-based indices):")
            for a, b in self.skeleton_edges:
                name_a = self.keypoint_names[a] if a < len(self.keypoint_names) else f"kp{a+1}"
                name_b = self.keypoint_names[b] if b < len(self.keypoint_names) else f"kp{b+1}"
                body.append(f"  {a + 1}-{b + 1}: {name_a} <-> {name_b}")
        return body

    def _show_helper_panel(self):
        lines = self._helper_text_lines()
        width, height = 760, max(240, 28 * len(lines) + 30)
        panel = np.full((height, width, 3), 245, dtype=np.uint8)
        y = 28
        for line in lines:
            cv2.putText(panel, line, (18, y), FONT, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
            y += 26
        try:
            cv2.imshow(HELPER_WINDOW_NAME, panel)
            cv2.resizeWindow(HELPER_WINDOW_NAME, width, height)
            cv2.waitKey(1)
            self.helper_window_open = True
        except Exception:
            self.helper_window_open = False

    def _close_helper_panel(self):
        try:
            cv2.destroyWindow(HELPER_WINDOW_NAME)
        except Exception:
            pass
        self.helper_window_open = False

    def _update_info_panel(self, payload):
        if not self.info_panel or not self.info_panel.winfo_exists():
            return
        if payload == self._last_info_payload:
            return
        self._last_info_payload = payload

        status_lines = [
            f"Image: {payload.get('image_name', 'N/A')} ({payload.get('index', 0)}/{payload.get('total', 0)})",
            f"Mode: {payload.get('mode_text', '')}",
        ]
        behavior_name = payload.get('behavior_name')
        if behavior_name:
            status_lines.append(f"Behavior: {behavior_name} (ID {payload.get('behavior_id', '?')})")
        next_kp = payload.get('next_kp')
        if next_kp:
            status_lines.append(f"Next keypoint: {next_kp}")
        status_lines.append(f"Zoom: {payload.get('zoom', 1.0):.2f}x  Pan: {payload.get('pan', (0,0))}")
        self.panel_status_var.set("\n".join(status_lines))
        if self.panel_progress_var is not None:
            self.panel_progress_var.set(payload.get("progress_text", ""))

        instruction_parts = []
        if payload.get('instruction_text'):
            instruction_parts.append(payload['instruction_text'])
        if payload.get('tooltip_text'):
            instruction_parts.append(payload['tooltip_text'])
        instruction_text = "\n".join(instruction_parts)
        self.panel_instruction_var.set(instruction_text)
        self._sync_behavior_panel_selection()

        try:
            self.info_panel.update_idletasks()
            self.info_panel.update()
        except Exception:
            pass

    def _destroy_info_panel(self):
        try:
            if self.info_panel and self.info_panel.winfo_exists():
                self.info_panel.destroy()
        except Exception:
            pass
        finally:
            self.info_panel = None
            self.use_info_panel = False

    def _scan_annotated_stems(self):
        if not os.path.isdir(self.output_dir):
            return set()
        try:
            return {
                os.path.splitext(name)[0]
                for name in os.listdir(self.output_dir)
                if name.lower().endswith(".txt") and not name.startswith(".")
            }
        except Exception:
            return set()

    def _current_image_stem(self):
        if not self.current_image_path:
            return ""
        return os.path.splitext(os.path.basename(self.current_image_path))[0]

    def _sync_current_stem_saved_state(self, *, has_saved_label: bool):
        stem = self._current_image_stem()
        if not stem:
            return
        if has_saved_label:
            self._annotated_stems.add(stem)
        else:
            self._annotated_stems.discard(stem)

    def _behavior_keys_map(self):
        return {
            str((i + 1) % 10): b_id
            for i, b_id in enumerate(list(self.available_behaviors.keys())[:10])
        }

    def _set_marker_size_scale(self, value):
        try:
            numeric = float(value)
        except Exception:
            numeric = DEFAULT_MARKER_SIZE_SCALE
        self.marker_size_scale = float(np.clip(numeric, MIN_MARKER_SIZE_SCALE, MAX_MARKER_SIZE_SCALE))
        if self.panel_marker_size_var is not None:
            try:
                self.panel_marker_size_var.set(round(self.marker_size_scale, 2))
            except Exception:
                pass
        self._update_display()

    def _adjust_marker_size(self, delta):
        self._set_marker_size_scale(self.marker_size_scale + float(delta))

    def _keypoint_color(self, index, visibility=VISIBILITY_LABELED_VISIBLE):
        base = KEYPOINT_PALETTE[int(index) % len(KEYPOINT_PALETTE)]
        if visibility == VISIBILITY_LABELED_VISIBLE:
            return base
        if visibility == VISIBILITY_LABELED_NOT_VISIBLE:
            return tuple(int((channel * 0.35) + 90) for channel in base)
        return (120, 120, 120)

    def _keypoint_radius(self, *, current=False, active=False):
        base = 4 if current else 3
        if active:
            base += 1
        scaled = base * max(1.0, self.zoom_level) * self.marker_size_scale
        return int(max(3 if current else 2, scaled))

    def _find_keypoint_hit(self, screen_x, screen_y):
        grab_radius = max(8, int(GRAB_RADIUS_BASE_PX * self.marker_size_scale))
        best = None
        best_dist_sq = float(grab_radius * grab_radius)

        for idx, kp in enumerate(self.current_keypoints):
            kp_x_s, kp_y_s = self._image_to_screen_coords(kp['x'], kp['y'])
            dist_sq = float((kp_x_s - screen_x) ** 2 + (kp_y_s - screen_y) ** 2)
            if dist_sq <= best_dist_sq:
                best = ("current", idx)
                best_dist_sq = dist_sq

        for ann_idx, ann in enumerate(self.annotations_in_image):
            for kp_idx, kp in enumerate(ann['keypoints']):
                kp_x_s, kp_y_s = self._image_to_screen_coords(kp['x'], kp['y'])
                dist_sq = float((kp_x_s - screen_x) ** 2 + (kp_y_s - screen_y) ** 2)
                if dist_sq <= best_dist_sq:
                    best = (ann_idx, kp_idx)
                    best_dist_sq = dist_sq
        return best

    def _find_annotation_at(self, screen_x, screen_y):
        for ann_idx, ann in enumerate(self.annotations_in_image):
            x1, y1, x2, y2 = ann['bbox']
            x1_s, y1_s = self._image_to_screen_coords(x1, y1)
            x2_s, y2_s = self._image_to_screen_coords(x2, y2)
            left, right = sorted((x1_s, x2_s))
            top, bottom = sorted((y1_s, y2_s))
            if left <= screen_x <= right and top <= screen_y <= bottom:
                return ann_idx
        return None

    def _refresh_hover_target(self, screen_x, screen_y):
        hit = self._find_keypoint_hit(screen_x, screen_y)
        if hit is not None:
            self._hover_target = hit
            self._hovered_annotation_index = hit[0] if hit[0] != "current" else None
            return
        ann_idx = self._find_annotation_at(screen_x, screen_y)
        self._hover_target = None
        self._hovered_annotation_index = ann_idx

    def _set_current_behavior(self, behavior_id):
        if behavior_id not in self.available_behaviors:
            return
        self._reset_current_instance_state()
        self.current_behavior_id = behavior_id
        self.current_behavior_name = self.available_behaviors[behavior_id]
        self.mode = MODE_KP
        print(f"Selected Behavior: '{self.current_behavior_name}'. Ready for keypoints.")
        self._sync_behavior_panel_selection()
        self._update_display()

    def _sync_behavior_panel_selection(self):
        if not self.panel_behavior_listbox:
            return
        self.panel_behavior_listbox.selection_clear(0, tk.END)
        keys = list(self.available_behaviors.keys())
        if self.current_behavior_id in keys:
            idx = keys.index(self.current_behavior_id)
            self.panel_behavior_listbox.selection_set(idx)
            self.panel_behavior_listbox.see(idx)
        if self.panel_behavior_id_var is not None:
            self.panel_behavior_id_var.set("" if self.current_behavior_id is None else str(self.current_behavior_id))
        if self.panel_behavior_name_var is not None:
            self.panel_behavior_name_var.set("" if self.current_behavior_name is None else str(self.current_behavior_name))

    def _refresh_behavior_listbox(self):
        if not self.panel_behavior_listbox:
            return
        self.panel_behavior_listbox.delete(0, tk.END)
        for idx, (behavior_id, behavior_name) in enumerate(self.available_behaviors.items()):
            key_map = self._behavior_keys_map()
            shortcut = ""
            for key_char, mapped_id in key_map.items():
                if mapped_id == behavior_id:
                    shortcut = f"[{key_char}] "
                    break
            self.panel_behavior_listbox.insert(tk.END, f"{shortcut}{behavior_id}: {behavior_name}")
        self._sync_behavior_panel_selection()

    def _on_panel_behavior_select(self, _event=None):
        if not self.panel_behavior_listbox:
            return
        selection = self.panel_behavior_listbox.curselection()
        if not selection:
            return
        keys = list(self.available_behaviors.keys())
        idx = selection[0]
        if 0 <= idx < len(keys):
            self._set_current_behavior(keys[idx])

    def _add_or_update_behavior_from_panel(self):
        if self.panel_behavior_id_var is None or self.panel_behavior_name_var is None:
            return
        try:
            behavior_id = int(str(self.panel_behavior_id_var.get()).strip())
        except Exception:
            messagebox.showerror("Behavior Error", "Behavior ID must be an integer.", parent=self.master_tk_window)
            return
        behavior_name = str(self.panel_behavior_name_var.get()).strip()
        if not behavior_name:
            messagebox.showerror("Behavior Error", "Behavior name cannot be empty.", parent=self.master_tk_window)
            return
        self.available_behaviors[behavior_id] = behavior_name
        self.available_behaviors = OrderedDict(sorted(self.available_behaviors.items()))
        self._refresh_behavior_listbox()
        self._update_display()

    def _go_to_image_index(self, index):
        if not (0 <= index < len(self.image_files)):
            return False
        if not self._save_annotations_for_image():
            return False
        if self._load_image(index):
            self._reset_zoom_pan()
            self._update_display()
            return True
        return False

    def _prev_image(self):
        new_index = self.current_image_index - 1
        if new_index >= 0:
            return self._go_to_image_index(new_index)
        else:
            messagebox.showinfo("End of Set", "You are at the first image.", parent=self.master_tk_window)
            return False

    def _next_image(self):
        new_index = self.current_image_index + 1
        if new_index < len(self.image_files):
            return self._go_to_image_index(new_index)
        else:
            messagebox.showinfo("End of Set", "You are at the last image.", parent=self.master_tk_window)
            return False

    def _jump_to_image_prompt(self):
        if not self.image_files:
            return False
        target = simpledialog.askinteger(
            "Go to image",
            f"Image number (1-{len(self.image_files)}):",
            minvalue=1,
            maxvalue=len(self.image_files),
            parent=self.master_tk_window,
        )
        if target is not None:
            return self._go_to_image_index(target - 1)
        return False

    def _toggle_keypoint_labels(self):
        self.show_keypoint_labels = not self.show_keypoint_labels
        state = "ON" if self.show_keypoint_labels else "OFF"
        print(f"   Keypoint labels toggled {state}.")
        self._update_display()
        return True

    def _reset_frame_annotations(self):
        message = "Clear all annotations for this image and delete any saved label file?"
        if not messagebox.askyesno("Reset Frame", message, parent=self.master_tk_window):
            return False
        self._clear_all_annotations_for_image()
        output_filename = self._current_image_stem() + ".txt"
        output_path = os.path.join(self.output_dir, output_filename) if output_filename else ""
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
                self._sync_current_stem_saved_state(has_saved_label=False)
            except Exception as exc:
                messagebox.showerror("Reset Error", f"Could not remove saved label file:\n{exc}", parent=self.master_tk_window)
                return False
        else:
            self._sync_current_stem_saved_state(has_saved_label=False)
        self.annotations_changed_since_last_save = False
        self.last_autosave_time = time.time()
        self.autosave_message = f"Reset: {self._current_image_stem()}.txt removed"
        self.autosave_message_display_time = time.time() + 3
        self._update_display()
        return True

    def _get_image_files(self):
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        try:
            if not os.path.isdir(self.image_dir):
                messagebox.showerror("Error", f"Annotation image directory not found: {self.image_dir}", parent=self.master_tk_window)
                return []
            files = [f for f in os.listdir(self.image_dir) if f.lower().endswith(valid_extensions)]
            files = [f for f in files if not f.startswith('.')]
            self._annotated_stems = self._scan_annotated_stems()
            return sorted([os.path.join(self.image_dir, f) for f in files])
        except Exception as e:
            messagebox.showerror("Error", f"Error reading image directory: {e}", parent=self.master_tk_window)
            return []

    def _reset_current_instance_state(self):
        self.mode = MODE_SELECT_BEHAVIOR
        self.current_behavior_id = None
        self.current_behavior_name = None
        self.current_keypoint_index = 0
        self.current_keypoints = []
        self.current_bbox_start_point = None
        self.current_bbox_end_point = None
        self.current_kp_visibility_mode = VISIBILITY_LABELED_VISIBLE
        print(" -> Ready to annotate next instance. Select Behavior.")

    def _clear_all_annotations_for_image(self):
        self.annotations_in_image = []
        self._drag_state = None
        self._hover_target = None
        self._hovered_annotation_index = None
        self._reset_current_instance_state()
        self.annotations_changed_since_last_save = True
        print("Cleared all annotations for the current image.")

    def _load_annotations_for_image(self):
        if not self.current_image_path:
            return

        label_filename = os.path.splitext(os.path.basename(self.current_image_path))[0] + ".txt"
        label_path = os.path.join(self.output_dir, label_filename)

        if not os.path.exists(label_path):
            return

        loaded_annotations = []
        img_w = float(self.img_width_orig)
        img_h = float(self.img_height_orig)

        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = [float(p) for p in line.strip().split()]
                    if len(parts) < 5 + self.num_keypoints * 3:
                        print(f"Warning: Skipping malformed line in {label_path}")
                        continue

                    b_id = int(parts[0])
                    norm_cx, norm_cy, norm_w, norm_h = parts[1:5]
                    box_w = norm_w * img_w
                    box_h = norm_h * img_h
                    x1 = int((norm_cx * img_w) - (box_w / 2))
                    y1 = int((norm_cy * img_h) - (box_h / 2))
                    x2 = int(x1 + box_w)
                    y2 = int(y1 + box_h)

                    loaded_kps = []
                    kp_data_flat = parts[5:]
                    for i in range(self.num_keypoints):
                        norm_x = kp_data_flat[i * 3]
                        norm_y = kp_data_flat[i * 3 + 1]
                        visibility = int(kp_data_flat[i * 3 + 2])
                        
                        if visibility != VISIBILITY_NOT_LABELED:
                            kp = {
                                'name': self.keypoint_names[i],
                                'x': int(norm_x * img_w),
                                'y': int(norm_y * img_h),
                                'v': visibility
                            }
                            loaded_kps.append(kp)

                    loaded_annotations.append({
                        'behavior_id': b_id,
                        'behavior_name': self.available_behaviors.get(b_id, "Unknown"),
                        'keypoints': loaded_kps,
                        'bbox': (x1, y1, x2, y2)
                    })
            
            if loaded_annotations:
                self.annotations_in_image = loaded_annotations
                self.annotations_changed_since_last_save = False
                print(f"Loaded {len(loaded_annotations)} annotations from {label_path}")

        except Exception as e:
            print(f"Error loading annotation file {label_path}: {e}")

    def _load_image(self, index):
        if not (0 <= index < len(self.image_files)):
            print("Invalid image index.")
            return False
        self.current_image_index = index
        self.current_image_path = self.image_files[self.current_image_index]
        self._drag_state = None
        self._hover_target = None
        self._hovered_annotation_index = None
        self._clear_all_annotations_for_image()
        
        raw_img = cv2.imread(self.current_image_path, cv2.IMREAD_UNCHANGED)
        if raw_img is None:
            messagebox.showerror("Error", f"Failed to load image: {self.current_image_path}", parent=self.master_tk_window)
            return False
        
        display_img = None
        if raw_img.dtype != np.uint8:
            if len(raw_img.shape) == 2 or (len(raw_img.shape) == 3 and raw_img.shape[2] == 1):
                temp_gray = cv2.normalize(raw_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                display_img = cv2.cvtColor(temp_gray, cv2.COLOR_GRAY2BGR)
            elif len(raw_img.shape) == 3 and raw_img.shape[2] == 3:
                display_img = cv2.normalize(raw_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            elif len(raw_img.shape) == 3 and raw_img.shape[2] == 4:
                color_conversion_code = cv2.COLOR_BGRA2BGR
                try:
                    temp_bgr = cv2.cvtColor(raw_img, color_conversion_code)
                except cv2.error:
                    temp_bgr = cv2.cvtColor(raw_img, cv2.COLOR_RGBA2BGR)
                display_img = cv2.normalize(temp_bgr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            else:
                messagebox.showerror("Image Error", f"Unsupported image type/channels for normalization: Shape {raw_img.shape}, Dtype {raw_img.dtype}", parent=self.master_tk_window)
                return False
        else:
            if len(raw_img.shape) == 2 or (len(raw_img.shape) == 3 and raw_img.shape[2] == 1):
                display_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
            elif len(raw_img.shape) == 3 and raw_img.shape[2] == 3:
                display_img = raw_img
            elif len(raw_img.shape) == 3 and raw_img.shape[2] == 4:
                try:
                    display_img = cv2.cvtColor(raw_img, cv2.COLOR_BGRA2BGR)
                except cv2.error:
                    display_img = cv2.cvtColor(raw_img, cv2.COLOR_RGBA2BGR)
            else:
                messagebox.showerror("Image Error", f"Unsupported uint8 image channels: {raw_img.shape}", parent=self.master_tk_window)
                return False
        
        if display_img is None or not (display_img.ndim == 3 and display_img.shape[2] == 3 and display_img.dtype == np.uint8):
            messagebox.showerror("Image Processing Error", f"Failed to convert image to displayable BGR uint8 format.", parent=self.master_tk_window)
            return False
        
        self.img_original_unmodified = display_img.copy()
        self.img_height_orig, self.img_width_orig = self.img_original_unmodified.shape[:2]

        self._load_annotations_for_image()
        
        self.annotations_changed_since_last_save = False
        self.last_autosave_time = time.time()
        self._sync_behavior_panel_selection()

        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1:
            self._reset_zoom_pan()
        else:
            self.zoom_level = 1.0
            self.pan_offset = np.array([0.0,0.0])
        print(f"\nLoaded image: {os.path.basename(self.current_image_path)} ({self.current_image_index + 1}/{len(self.image_files)}) Dim: {self.img_width_orig}x{self.img_height_orig}")
        return True

    def _get_current_window_dimensions(self):
        win_w, win_h = self.default_window_width, self.default_window_height
        try:
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1:
                rect = cv2.getWindowImageRect(self.window_name)
                if rect[2] > MIN_WINDOW_DIM_THRESHOLD and rect[3] > MIN_WINDOW_DIM_THRESHOLD:
                    win_w, win_h = rect[2], rect[3]
        except cv2.error:
            pass
        return win_w, win_h

    def _reset_zoom_pan(self):
        self.zoom_level = 1.0
        win_w, win_h = self._get_current_window_dimensions()

        self.pan_offset = np.array([
            max(0.0, (win_w - self.img_width_orig * self.zoom_level) / 2.0),
            max(0.0, (win_h - self.img_height_orig * self.zoom_level) / 2.0)
        ])
        if self.img_width_orig * self.zoom_level > win_w:
            self.pan_offset[0] = 0.0
        if self.img_height_orig * self.zoom_level > win_h:
            self.pan_offset[1] = 0.0
        self._constrain_pan()

    def _screen_to_image_coords(self, screen_x, screen_y):
        if self.zoom_level == 0: return 0,0
        img_x = (screen_x - self.pan_offset[0]) / self.zoom_level
        img_y = (screen_y - self.pan_offset[1]) / self.zoom_level
        return int(round(img_x)), int(round(img_y))

    def _image_to_screen_coords(self, img_x, img_y):
        screen_x = int(round(img_x * self.zoom_level + self.pan_offset[0]))
        screen_y = int(round(img_y * self.zoom_level + self.pan_offset[1]))
        return screen_x, screen_y

    def _draw_keypoint_label(self, label_text, screen_x, screen_y, color, radius):
        if not self.show_keypoint_labels or not label_text:
            return
        if self.img_display is None:
            return

        label = str(label_text)
        font_scale = max(0.35, FONT_SCALE_KP_NUMBER * max(1.0, self.zoom_level))
        thickness = max(1, FONT_THICKNESS)
        (text_w, text_h), baseline = cv2.getTextSize(label, FONT, font_scale, thickness)
        img_h, img_w = self.img_display.shape[:2]

        text_x = screen_x + radius + 4
        if text_x + text_w > img_w - 2:
            text_x = max(0, screen_x - radius - text_w - 4)
        text_x = max(0, min(text_x, img_w - text_w - 2))

        text_y = screen_y - radius - 6
        if text_y - text_h < 2:
            text_y = screen_y + radius + text_h + 6
        text_y = max(text_h + 2, min(text_y, img_h - baseline - 2))

        cv2.putText(self.img_display, label, (text_x, text_y), FONT, font_scale, color, thickness, cv2.LINE_AA)

    def _constrain_pan(self):
        win_w, win_h = self._get_current_window_dimensions()
        scaled_w = self.img_width_orig * self.zoom_level
        scaled_h = self.img_height_orig * self.zoom_level

        if scaled_w < win_w:
            min_offset_x = max_offset_x = (win_w - scaled_w) / 2.0
        else:
            min_offset_x = win_w - scaled_w
            max_offset_x = 0.0
        if scaled_h < win_h:
            min_offset_y = max_offset_y = (win_h - scaled_h) / 2.0
        else:
            min_offset_y = win_h - scaled_h
            max_offset_y = 0.0

        self.pan_offset[0] = np.clip(self.pan_offset[0], min_offset_x, max_offset_x)
        self.pan_offset[1] = np.clip(self.pan_offset[1], min_offset_y, max_offset_y)

    def _handle_mouse_events(self, event, x, y, flags, param):
        self.current_mouse_pos_screen = (x, y)
        alt_key_pressed = (flags & cv2.EVENT_FLAG_ALTKEY) != 0
        needs_update = False

        if event == cv2.EVENT_LBUTTONDOWN and alt_key_pressed:
            self.is_panning_with_alt = True
            self.pan_start_mouse_screen_pos = np.array([x,y],dtype=float)
            self.pan_start_offset = self.pan_offset.copy()
        elif self.is_panning_with_alt and event == cv2.EVENT_MOUSEMOVE:
            if self.pan_start_mouse_screen_pos is not None:
                delta = np.array([x,y],dtype=float) - self.pan_start_mouse_screen_pos
                self.pan_offset = self.pan_start_offset + delta
                self._constrain_pan()
                needs_update = True
        elif event == cv2.EVENT_LBUTTONUP and self.is_panning_with_alt:
            self.is_panning_with_alt = False
            self.pan_start_mouse_screen_pos = None
            needs_update = True
        elif event == cv2.EVENT_MOUSEWHEEL:
            img_x_before_zoom, img_y_before_zoom = self._screen_to_image_coords(x,y)
            zoom_factor = (1 + self.zoom_step_factor)
            if flags > 0: self.zoom_level *= zoom_factor
            else: self.zoom_level /= zoom_factor
            self.zoom_level = np.clip(self.zoom_level, self.min_zoom, self.max_zoom)

            self.pan_offset[0] = x - img_x_before_zoom * self.zoom_level
            self.pan_offset[1] = y - img_y_before_zoom * self.zoom_level
            self._constrain_pan()
            needs_update = True
        
        if self.zoom_level != 0:
            self.current_mouse_pos_image = self._screen_to_image_coords(x,y)
        else:
            self.current_mouse_pos_image = (0,0)

        if self.is_panning_with_alt:
            if needs_update: self._update_display()
            return

        img_x_clamped = max(0, min(self.current_mouse_pos_image[0], self.img_width_orig - 1))
        img_y_clamped = max(0, min(self.current_mouse_pos_image[1], self.img_height_orig - 1))

        if event == cv2.EVENT_MOUSEMOVE:
            if self._drag_state is not None:
                source = self._drag_state.get('source')
                kp_index = int(self._drag_state.get('kp_index', -1))
                if source == "current" and 0 <= kp_index < len(self.current_keypoints):
                    self.current_keypoints[kp_index]['x'] = img_x_clamped
                    self.current_keypoints[kp_index]['y'] = img_y_clamped
                    self.annotations_changed_since_last_save = True
                    needs_update = True
                elif isinstance(source, int) and 0 <= source < len(self.annotations_in_image):
                    keypoints = self.annotations_in_image[source].get('keypoints', [])
                    if 0 <= kp_index < len(keypoints):
                        keypoints[kp_index]['x'] = img_x_clamped
                        keypoints[kp_index]['y'] = img_y_clamped
                        self.annotations_changed_since_last_save = True
                        needs_update = True
                self._refresh_hover_target(x, y)
            else:
                self._refresh_hover_target(x, y)
                if self.mode == MODE_BBOX_END and self.current_bbox_start_point:
                    needs_update = True
        elif event == cv2.EVENT_LBUTTONDOWN:
            needs_update = True
            hit = self._find_keypoint_hit(x, y) if not alt_key_pressed else None
            if hit is not None:
                self._drag_state = {'source': hit[0], 'kp_index': hit[1]}
                self._refresh_hover_target(x, y)
            elif self.mode == MODE_KP:
                if self.current_keypoint_index < self.num_keypoints:
                    kp_name = self.keypoint_names[self.current_keypoint_index]
                    visibility_to_set = self.current_kp_visibility_mode
                    self.current_keypoints.append({
                        'name': kp_name,
                        'x': img_x_clamped, 'y': img_y_clamped,
                        'v': visibility_to_set
                    })
                    vis_str = "Visible" if visibility_to_set == VISIBILITY_LABELED_VISIBLE else "Not Visible"
                    print(f"   Placed KP {self.current_keypoint_index + 1}/{self.num_keypoints}: {kp_name} at ({img_x_clamped},{img_y_clamped}) [{vis_str}]")
                    self.current_keypoint_index += 1
                    self.annotations_changed_since_last_save = True
                    if self.current_keypoint_index == self.num_keypoints:
                        self.mode = MODE_BBOX_START
                        print("   All keypoints placed. Drag bounding box over the animal.")
            elif self.mode == MODE_BBOX_START:
                self.current_bbox_start_point = (img_x_clamped, img_y_clamped)
                self.mode = MODE_BBOX_END
                self.annotations_changed_since_last_save = True
                self.current_bbox_end_point = None
                print(f"   BBox Drag Start: ({img_x_clamped},{img_y_clamped})")
            elif self.mode == MODE_SELECT_BEHAVIOR:
                messagebox.showinfo("Info", "Please select a behavior first using number keys (0-9).", parent=self.master_tk_window)
                needs_update=False
        elif event == cv2.EVENT_LBUTTONUP:
            if self._drag_state is not None:
                self._drag_state = None
                self._refresh_hover_target(x, y)
                needs_update = True
            elif self.mode == MODE_BBOX_END:
                if self.current_bbox_start_point is None:
                    self.mode = MODE_BBOX_START
                    print("Error: BBox drag ended without a start point. Resetting.")
                    return
                start_x, start_y = self.current_bbox_start_point
                x1, x2 = sorted((start_x, img_x_clamped))
                y1, y2 = sorted((start_y, img_y_clamped))
                if (x2 - x1) < MIN_BBOX_DIMENSION or (y2 - y1) < MIN_BBOX_DIMENSION:
                    messagebox.showwarning(
                        "BBox Error",
                        f"Bounding box is too small.\nBoth width and height must be at least {MIN_BBOX_DIMENSION} pixels.",
                        parent=self.master_tk_window,
                    )
                    needs_update = False
                    self.current_bbox_start_point = None
                    self.current_bbox_end_point = None
                    self.mode = MODE_BBOX_START
                    return

                self.current_bbox_end_point = (x2, y2)
                print(f"   BBox Completed: ({x1},{y1}) -> ({x2},{y2})")
                completed_annotation = {
                    'behavior_id': self.current_behavior_id,
                    'behavior_name': self.current_behavior_name,
                    'keypoints': self.current_keypoints.copy(),
                    'bbox': (x1, y1, x2, y2)
                }
                self.annotations_in_image.append(completed_annotation)
                self.annotations_changed_since_last_save = True
                print(f"   Completed instance {len(self.annotations_in_image)} ({self.current_behavior_name}).")
                self._reset_current_instance_state()
                self._refresh_hover_target(x, y)
                needs_update = True
        elif event == cv2.EVENT_MBUTTONDOWN:
            if self.mode == MODE_KP and self.current_keypoints:
                last_kp_idx = len(self.current_keypoints) - 1
                if 0 <= last_kp_idx < len(self.current_keypoints):
                    kp_to_toggle = self.current_keypoints[last_kp_idx]
                    current_vis = kp_to_toggle.get('v', VISIBILITY_LABELED_VISIBLE)
                    kp_to_toggle['v'] = VISIBILITY_LABELED_NOT_VISIBLE if current_vis == VISIBILITY_LABELED_VISIBLE else VISIBILITY_LABELED_VISIBLE
                    vis_str = "Visible" if kp_to_toggle['v'] == VISIBILITY_LABELED_VISIBLE else "Not Visible"
                    print(f"   MMB Toggled visibility of last placed KP ({kp_to_toggle['name']}) to: {vis_str}")
                    self.annotations_changed_since_last_save = True
                    needs_update = True

        if needs_update:
            self._update_display()

    def _update_display(self):
        if self.img_original_unmodified is None:
            temp_win_w, temp_win_h = self._get_current_window_dimensions()
            self.img_display = np.full((temp_win_h, temp_win_w, 3), 100, dtype=np.uint8)
            cv2.putText(self.img_display, "No image loaded or image error.", (50, temp_win_h // 2), FONT, 1, (255,255,255), 1, cv2.LINE_AA)
            try:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1 :
                    cv2.imshow(self.window_name, self.img_display)
            except cv2.error:
                pass
            return

        win_w, win_h = self._get_current_window_dimensions()
        if win_w <= 0 or win_h <=0:
            return

        self.img_display = np.full((win_h, win_w, 3), 200, dtype=np.uint8)
        M = np.float32([[self.zoom_level, 0, self.pan_offset[0]], [0, self.zoom_level, self.pan_offset[1]]])

        try:
            cv2.warpAffine(self.img_original_unmodified, M, (win_w, win_h), dst=self.img_display, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        except cv2.error as e:
            cv2.putText(self.img_display, "Error during image warp.", (50, win_h // 2), FONT, 0.7, (0,0,255), 1, cv2.LINE_AA)

        for ann_idx, ann in enumerate(self.annotations_in_image):
            is_hovered_ann = ann_idx == self._hovered_annotation_index
            box_color = BBOX_HIGHLIGHT_COLOR if is_hovered_ann else BBOX_COMPLETED_COLOR
            box_thickness = CURRENT_LINE_THICKNESS if is_hovered_ann else COMPLETED_LINE_THICKNESS
            x1_orig, y1_orig, x2_orig, y2_orig = ann['bbox']
            x1_s, y1_s = self._image_to_screen_coords(x1_orig, y1_orig)
            x2_s, y2_s = self._image_to_screen_coords(x2_orig, y2_orig)
            cv2.rectangle(self.img_display, (x1_s, y1_s), (x2_s, y2_s), box_color, box_thickness, lineType=cv2.LINE_AA)

            label = f"{ann['behavior_id']}:{ann['behavior_name']}"
            label_scale = FONT_SCALE_ANNOTATION_INFO + (0.05 if is_hovered_ann else 0.0)
            (lw, lh), baseline = cv2.getTextSize(label, FONT, label_scale, FONT_THICKNESS)
            cv2.rectangle(
                self.img_display,
                (x1_s, y1_s - lh - baseline - 2),
                (x1_s + lw, y1_s),
                BEHAVIOR_BOX_BACKGROUND_COLOR,
                -1,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                self.img_display,
                label,
                (x1_s, y1_s - baseline // 2 - 1),
                FONT,
                label_scale,
                box_color if is_hovered_ann else BEHAVIOR_TEXT_ON_BOX_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

            kp_by_index = {}
            for kp in ann['keypoints']:
                idx = self.kp_name_to_index.get(kp.get('name'))
                if idx is not None:
                    kp_by_index[idx] = kp

            for idx_a, idx_b in self.skeleton_edges:
                kp_a = kp_by_index.get(idx_a)
                kp_b = kp_by_index.get(idx_b)
                if not kp_a or not kp_b:
                    continue
                vis_a = kp_a.get('v', VISIBILITY_NOT_LABELED)
                vis_b = kp_b.get('v', VISIBILITY_NOT_LABELED)
                if vis_a == VISIBILITY_NOT_LABELED and vis_b == VISIBILITY_NOT_LABELED:
                    continue
                pt_a = self._image_to_screen_coords(kp_a['x'], kp_a['y'])
                pt_b = self._image_to_screen_coords(kp_b['x'], kp_b['y'])
                bone_color = box_color if is_hovered_ann else BONE_COLOR
                cv2.line(
                    self.img_display,
                    pt_a,
                    pt_b,
                    bone_color,
                    max(BONE_LINE_THICKNESS, int(1 * self.zoom_level)),
                    lineType=cv2.LINE_AA,
                )

            active_kp_index = None
            if self._hover_target is not None and self._hover_target[0] == ann_idx:
                active_kp_index = int(self._hover_target[1])

            for kp_idx, kp in enumerate(ann['keypoints']):
                kp_name = kp.get('name') or f"KP {kp_idx + 1}"
                palette_idx = self.kp_name_to_index.get(kp_name, kp_idx)
                visibility = kp.get('v', VISIBILITY_NOT_LABELED)
                color = self._keypoint_color(palette_idx, visibility)
                kp_x_s, kp_y_s = self._image_to_screen_coords(kp['x'], kp['y'])
                is_active = kp_idx == active_kp_index
                radius = self._keypoint_radius(current=False, active=is_active or is_hovered_ann)
                cv2.circle(self.img_display, (kp_x_s, kp_y_s), radius + 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                cv2.circle(self.img_display, (kp_x_s, kp_y_s), radius, color, -1, lineType=cv2.LINE_AA)
                if is_active:
                    cv2.circle(self.img_display, (kp_x_s, kp_y_s), radius + 4, box_color, 1, lineType=cv2.LINE_AA)
                num_font_scale = max(0.32, FONT_SCALE_KP_NUMBER * max(1.0, self.zoom_level))
                cv2.putText(
                    self.img_display,
                    str(palette_idx + 1),
                    (kp_x_s + radius, kp_y_s + radius),
                    FONT,
                    num_font_scale,
                    color,
                    FONT_THICKNESS,
                    cv2.LINE_AA,
                )
                self._draw_keypoint_label(kp_name, kp_x_s, kp_y_s, color, radius)

        for idx_a, idx_b in self.skeleton_edges:
            if idx_a >= len(self.current_keypoints) or idx_b >= len(self.current_keypoints):
                continue
            kp_a = self.current_keypoints[idx_a]
            kp_b = self.current_keypoints[idx_b]
            vis_a = kp_a.get('v', VISIBILITY_LABELED_VISIBLE)
            vis_b = kp_b.get('v', VISIBILITY_LABELED_VISIBLE)
            if vis_a == VISIBILITY_NOT_LABELED and vis_b == VISIBILITY_NOT_LABELED:
                continue
            pt_a = self._image_to_screen_coords(kp_a['x'], kp_a['y'])
            pt_b = self._image_to_screen_coords(kp_b['x'], kp_b['y'])
            cv2.line(
                self.img_display,
                pt_a,
                pt_b,
                BONE_COLOR,
                max(BONE_LINE_THICKNESS, int(1 * self.zoom_level)),
                lineType=cv2.LINE_AA,
            )

        for i, kp in enumerate(self.current_keypoints):
            color = self._keypoint_color(i, kp.get('v', VISIBILITY_LABELED_VISIBLE))
            kp_x_s, kp_y_s = self._image_to_screen_coords(kp['x'], kp['y'])
            is_active = self._hover_target == ("current", i) or i == self.current_keypoint_index - 1
            radius = self._keypoint_radius(current=True, active=is_active)
            cv2.circle(self.img_display, (kp_x_s, kp_y_s), radius + 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(self.img_display, (kp_x_s, kp_y_s), radius, color, -1, lineType=cv2.LINE_AA)
            if is_active:
                cv2.circle(self.img_display, (kp_x_s, kp_y_s), radius + 4, BBOX_HIGHLIGHT_COLOR, 1, lineType=cv2.LINE_AA)
            num_font_scale = max(0.32, FONT_SCALE_KP_NUMBER * max(1.0, self.zoom_level))
            cv2.putText(
                self.img_display,
                str(i + 1),
                (kp_x_s + radius, kp_y_s + radius),
                FONT,
                num_font_scale,
                color,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )
            self._draw_keypoint_label(kp.get('name') or f"KP {i + 1}", kp_x_s, kp_y_s, color, radius)

        if self.current_bbox_start_point and self.mode == MODE_BBOX_END:
            st_x_s,st_y_s = self._image_to_screen_coords(self.current_bbox_start_point[0],self.current_bbox_start_point[1])
            cv2.circle(
                self.img_display,
                (st_x_s, st_y_s),
                self._keypoint_radius(current=True, active=True),
                BBOX_CURRENT_TEMP_COLOR,
                -1,
                lineType=cv2.LINE_AA,
            )

        if self.mode == MODE_BBOX_END and self.current_bbox_start_point and self.current_mouse_pos_image:
            start_x, start_y = self.current_bbox_start_point
            cur_m_x_img_clamped = max(0, min(self.current_mouse_pos_image[0], self.img_width_orig -1))
            cur_m_y_img_clamped = max(0, min(self.current_mouse_pos_image[1], self.img_height_orig -1))
            x1, x2 = sorted((start_x, cur_m_x_img_clamped))
            y1, y2 = sorted((start_y, cur_m_y_img_clamped))
            x1_s, y1_s = self._image_to_screen_coords(x1, y1)
            x2_s, y2_s = self._image_to_screen_coords(x2, y2)
            cv2.rectangle(
                self.img_display,
                (x1_s, y1_s),
                (x2_s, y2_s),
                BBOX_CURRENT_TEMP_COLOR,
                CURRENT_LINE_THICKNESS,
                lineType=cv2.LINE_AA,
            )

        y_off = 20; line_height_small = 18; line_height_normal = 22
        def draw_text_line(text_content, color=STATUS_TEXT_COLOR, scale=FONT_SCALE_GENERAL_STATUS, is_main_instr=False):
            nonlocal y_off; lh = line_height_normal if is_main_instr else line_height_small
            if y_off < win_h - lh :
                cv2.putText(self.img_display, text_content, (10, y_off), FONT, scale, color, FONT_THICKNESS, cv2.LINE_AA)
                y_off += int(lh * (scale / FONT_SCALE_GENERAL_STATUS) + 5)

        img_file_name = os.path.basename(self.current_image_path) if self.current_image_path else "N/A"
        current_stem = self._current_image_stem()
        if self.annotations_changed_since_last_save:
            saved_marker = "[Unsaved]"
        elif current_stem in self._annotated_stems:
            saved_marker = "[Saved]"
        else:
            saved_marker = "[Not Saved]"
        mouse_x_str = str(self.current_mouse_pos_image[0]) if self.current_mouse_pos_image else 'N/A'
        mouse_y_str = str(self.current_mouse_pos_image[1]) if self.current_mouse_pos_image else 'N/A'

        draw_text_line(f"{img_file_name} {saved_marker} [{self.current_image_index + 1}/{len(self.image_files)}] Anns: {len(self.annotations_in_image)}")
        draw_text_line(f"Zoom: {self.zoom_level:.2f}x Pan:({int(self.pan_offset[0])},{int(self.pan_offset[1])}) XY_img:({mouse_x_str},{mouse_y_str})", color=ZOOM_PAN_TEXT_COLOR)
        draw_text_line(f"Marker size: {self.marker_size_scale:.1f}x ([ / ])", color=HELP_TEXT_COLOR, scale=FONT_SCALE_TOOLTIP)
        y_off += 2

        mode_text = ""; instruction_text = ""; tooltip_text = ""
        if self.mode == MODE_SELECT_BEHAVIOR:
            mode_text = "MODE: Select Behavior"
            behavior_opts = []
            b_items = list(self.available_behaviors.items())
            for i, (bid, bname) in enumerate(b_items):
                if i < 9: key_char = str(i+1)
                elif i == 9: key_char = "0"
                else: break
                behavior_opts.append(f"({key_char}) {bname}")
                if len(behavior_opts) >=5 and len(b_items) > 5 :
                    if i < len(b_items) -1 : behavior_opts.append("...")
                    break
            instruction_text = " | ".join(behavior_opts)
            tooltip_text = "Use number keys or select a behavior in the control window."
        elif self.mode == MODE_KP:
            mode_text = f"MODE: Keypoints for '{self.current_behavior_name}' (ID: {self.current_behavior_id})"
            if self.current_keypoint_index < self.num_keypoints:
                instruction_text = f"Click KP {self.current_keypoint_index + 1}/{self.num_keypoints}: {self.keypoint_names[self.current_keypoint_index]}"
            else:
                instruction_text = "All keypoints placed for this instance!"
            tooltip_text = "LMB: place point. Drag existing points to correct them. MMB toggles last point visibility."
        elif self.mode == MODE_BBOX_START:
            mode_text = f"MODE: Draw BBox for '{self.current_behavior_name}'"
            instruction_text = "Click and drag a bounding box around the animal."
            tooltip_text = "Press and hold LMB to start the box, then release to finish."
        elif self.mode == MODE_BBOX_END:
            mode_text = f"MODE: Finish BBox for '{self.current_behavior_name}'"
            instruction_text = "Release the mouse button to finish the bounding box."
            tooltip_text = "Box direction is normalized automatically, so draw in any direction."

        if mode_text: draw_text_line(mode_text, color=TEXT_COLOR, scale=FONT_SCALE_MODE_STATUS)
        if instruction_text: draw_text_line(instruction_text, color=TEXT_COLOR, scale=FONT_SCALE_MAIN_INSTRUCTION, is_main_instr=True)

        if self.mode == MODE_KP:
            vis_mode_str_display = "Next KP: VISIBLE" if self.current_kp_visibility_mode == VISIBILITY_LABELED_VISIBLE else "Next KP: NOT VISIBLE"
            draw_text_line(f"{vis_mode_str_display} (Press 'V' to toggle)", color=TEXT_COLOR, scale=FONT_SCALE_TOOLTIP)

        if tooltip_text: draw_text_line(f"Tip: {tooltip_text}", color=HELP_TEXT_COLOR, scale=FONT_SCALE_TOOLTIP)
        label_status = "ON" if self.show_keypoint_labels else "OFF"
        draw_text_line(f"Keypoint labels: {label_status} (Press 'L' to toggle)", color=HELP_TEXT_COLOR, scale=FONT_SCALE_TOOLTIP)
        draw_text_line("Press 'R' to clear this frame and remove its saved label file.", color=HELP_TEXT_COLOR, scale=FONT_SCALE_TOOLTIP)

        if self.autosave_message and time.time() < self.autosave_message_display_time:
            (tw,th),_ = cv2.getTextSize(self.autosave_message,FONT,FONT_SCALE_HELP_PROMPT,FONT_THICKNESS)
            if win_h > th + 20 :
                cv2.putText(self.img_display,self.autosave_message,(10,win_h-th-10),FONT,FONT_SCALE_HELP_PROMPT,AUTOSAVE_MSG_COLOR,FONT_THICKNESS,cv2.LINE_AA)
        elif self.autosave_message:
            self.autosave_message=""

        help_y_start=win_h-10
        helper_prompt = "Helper: Press H or ? to toggle"
        if self.helper_popup_enabled and self.helper_window_open:
            helper_prompt = "Helper open (H or ? to hide)"
        (tw,th),_ = cv2.getTextSize(helper_prompt,FONT,FONT_SCALE_HELP_PROMPT,FONT_THICKNESS)
        if win_h > th + 20 and help_y_start-th > y_off + th :
            cv2.putText(self.img_display,helper_prompt,(10,win_h-th-10),FONT,FONT_SCALE_HELP_PROMPT,HELP_TEXT_COLOR,FONT_THICKNESS,cv2.LINE_AA)

        if self.use_info_panel:
            next_kp_name = None
            if self.mode == MODE_KP and self.current_keypoint_index < self.num_keypoints:
                next_kp_name = f"{self.current_keypoint_index + 1}/{self.num_keypoints}: {self.keypoint_names[self.current_keypoint_index]}"
            progress_text = f"Frame status: {saved_marker}   Instances in frame: {len(self.annotations_in_image)}"
            info_payload = {
                "image_name": img_file_name,
                "index": self.current_image_index + 1,
                "total": len(self.image_files),
                "mode_text": mode_text or "",
                "behavior_name": self.current_behavior_name,
                "behavior_id": self.current_behavior_id,
                "instruction_text": instruction_text,
                "tooltip_text": tooltip_text,
                "next_kp": next_kp_name,
                "zoom": self.zoom_level,
                "pan": (int(self.pan_offset[0]), int(self.pan_offset[1])),
                "progress_text": progress_text,
            }
            self._update_info_panel(info_payload)

        try:
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1 :
                cv2.imshow(self.window_name, self.img_display)
        except cv2.error:
            pass

    def _undo_last_keypoint(self):
        if self.mode == MODE_KP and self.current_keypoints:
            removed_kp = self.current_keypoints.pop()
            self.current_keypoint_index -= 1
            self.annotations_changed_since_last_save = True
            print(f"   Undo: Removed keypoint {removed_kp['name']}.")
        elif self.mode == MODE_BBOX_START:
            self.mode = MODE_KP
            print(f"   Undo: Canceled BBox selection, returned to Keypoint mode.")
        elif self.mode == MODE_BBOX_END:
            self.mode = MODE_BBOX_START
            print(f"   Undo: Canceled BBox end point. Re-click BBox end point.")
        elif self.mode == MODE_SELECT_BEHAVIOR and self.annotations_in_image:
            removed_ann = self.annotations_in_image.pop()
            self.annotations_changed_since_last_save = True
            self.current_behavior_id = removed_ann['behavior_id']
            self.current_behavior_name = removed_ann['behavior_name']
            self.current_keypoints = removed_ann['keypoints']
            self.current_keypoint_index = len(self.current_keypoints)
            self.current_bbox_start_point = (removed_ann['bbox'][0], removed_ann['bbox'][1])
            self.current_bbox_end_point = (removed_ann['bbox'][2], removed_ann['bbox'][3])
            self.mode = MODE_BBOX_END
            print(f"   Undo: Removed completed instance for behavior '{removed_ann['behavior_name']}'.")
        else:
            print("Nothing to undo.")
            return

        self._update_display()

    def _save_annotations_for_image(self, is_autosave=False):
        if not self.current_image_path:
            return False
        if not self.annotations_in_image and is_autosave and not self.annotations_changed_since_last_save:
            return True

        output_filename = os.path.splitext(os.path.basename(self.current_image_path))[0] + ".txt"
        output_path = os.path.join(self.output_dir, output_filename)
        lines_to_write = []

        if not self.annotations_in_image:
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
                    print(f"Cleared annotations: Removed existing file {output_path}")
                else:
                    print(f"No annotations to save for {os.path.basename(self.current_image_path)}.")
                self._sync_current_stem_saved_state(has_saved_label=False)
                self.annotations_changed_since_last_save = False
                self.last_autosave_time = time.time()
                self.autosave_message = f"Saved: {os.path.basename(output_path)} (cleared)"
                self.autosave_message_display_time = time.time() + 3
                return True
            except Exception as e:
                messagebox.showerror("Save Error",f"Could not remove existing annotation file: {e}", parent=self.master_tk_window)
                return False

        for ann in self.annotations_in_image:
            b_id = ann['behavior_id']
            x1, y1, x2, y2 = ann['bbox']
            kps_data = ann['keypoints']
            img_w_f = float(self.img_width_orig)
            img_h_f = float(self.img_height_orig)
            x1 = float(max(0, min(x1, x2)))
            y1 = float(max(0, min(y1, y2)))
            x2 = float(min(self.img_width_orig - 1, max(x1, x2)))
            y2 = float(min(self.img_height_orig - 1, max(y1, y2)))

            x1_norm = np.clip(x1 / img_w_f, 0.0, 1.0)
            y1_norm = np.clip(y1 / img_h_f, 0.0, 1.0)
            x2_norm = np.clip(x2 / img_w_f, 0.0, 1.0)
            y2_norm = np.clip(y2 / img_h_f, 0.0, 1.0)

            norm_cx = np.clip((x1_norm + x2_norm) / 2.0, 0.0, 1.0)
            norm_cy = np.clip((y1_norm + y2_norm) / 2.0, 0.0, 1.0)
            norm_w = np.clip(x2_norm - x1_norm, 0.0, 1.0)
            norm_h = np.clip(y2_norm - y1_norm, 0.0, 1.0)

            yolo_kpts_flat = []
            placed_kpts_map = {kp['name']:kp for kp in kps_data}
            for kp_name_ordered in self.keypoint_names:
                if kp_name_ordered in placed_kpts_map:
                    kp = placed_kpts_map[kp_name_ordered]
                    norm_x = np.clip(kp['x']/img_w_f,0.0,1.0)
                    norm_y = np.clip(kp['y']/img_h_f,0.0,1.0)
                    visibility = int(kp.get('v', VISIBILITY_NOT_LABELED))
                    yolo_kpts_flat.extend([norm_x, norm_y, float(visibility)])
                else:
                    yolo_kpts_flat.extend([0.0,0.0,float(VISIBILITY_NOT_LABELED)])

            bbox_str = f"{norm_cx:.8f} {norm_cy:.8f} {norm_w:.8f} {norm_h:.8f}"
            kpts_str_parts = [f"{v:.8f}" for v in yolo_kpts_flat]
            kpts_str = " ".join(kpts_str_parts)

            lines_to_write.append(f"{b_id} {bbox_str} {kpts_str}")

        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(output_path,'w') as f:
                for line in lines_to_write:
                    f.write(line + "\n")

            save_type = "Autosaved" if is_autosave else "Saved"
            print(f"{save_type} {len(lines_to_write)} instance(s) to: {output_path}")
            self._sync_current_stem_saved_state(has_saved_label=True)
            self.annotations_changed_since_last_save = False
            self.last_autosave_time = time.time()
            self.autosave_message = f"{save_type}: {os.path.basename(output_path)}"
            self.autosave_message_display_time = time.time() + 3
            return True
        except Exception as e:
            messagebox.showerror("Save Error",f"Could not write annotation file: {e}", parent=self.master_tk_window)
            return False

    @staticmethod
    def generate_config_yaml(master_tk_window, num_keypoints, keypoint_names, available_behaviors, dataset_root_path, train_images_path, val_images_path, dataset_name_suggestion="dataset_config"):
        if not all([dataset_root_path, train_images_path, val_images_path]):
            messagebox.showerror("Error", "Dataset root, train images folder, and validation images folder are ALL required for YAML generation.", parent=master_tk_window)
            return None
        try:
            abs_dataset_root = os.path.abspath(dataset_root_path)
            abs_train_images = os.path.abspath(train_images_path)
            abs_val_images = os.path.abspath(val_images_path)
            rel_train_path = os.path.relpath(abs_train_images, abs_dataset_root)
            rel_val_path = os.path.relpath(abs_val_images, abs_dataset_root)
            if rel_train_path.startswith('..') or rel_val_path.startswith('..'):
                messagebox.showwarning("Path Warning", "Train/Validation image paths seem to be outside the specified Dataset Root Path.", parent=master_tk_window)
        except ValueError as e:
            messagebox.showerror("Path Error", f"Could not determine relative paths. Error: {e}", parent=master_tk_window)
            return None

        behavior_names_map = {int(bid): name for bid, name in available_behaviors.items()}
        
        yaml_data = OrderedDict([
            ('path', abs_dataset_root.replace(os.sep,'/')),
            ('train', rel_train_path.replace(os.sep,'/')),
            ('val', rel_val_path.replace(os.sep,'/')),
            ('kpt_shape', [num_keypoints, 3]),
            ('nc', len(available_behaviors)),
            ('names', behavior_names_map)
        ])
        
        header_lines = ["# YOLO Dataset Configuration File"]
        if keypoint_names:
            header_lines.extend(
                [
                    "#",
                    "# Keypoint names (metadata, not used by Ultralytics):",
                    *[f"# - {name}" for name in keypoint_names],
                ]
            )
        header_lines.append("")
        header = "\n".join(header_lines) + "\n"

        save_path_initial_file = f"{dataset_name_suggestion}.yaml"
        save_path = filedialog.asksaveasfilename(
            title="Save Dataset Config YAML",
            initialdir=abs_dataset_root,
            initialfile=save_path_initial_file,
            defaultextension=".yaml",
            filetypes=[("YAML files","*.yaml *.yml")],
            parent=master_tk_window
        )
        if not save_path:
            print("YAML saving cancelled by user.")
            return None
        try:
            with open(save_path,'w',encoding='utf-8') as f:
                f.write(header)
                yaml.dump(dict(yaml_data),f,default_flow_style=None,sort_keys=False,indent=4)
            print(f"Dataset config YAML saved to: {save_path}")
            messagebox.showinfo("YAML Saved",f"Dataset configuration YAML saved to:\n{save_path}", parent=master_tk_window)
            return save_path
        except Exception as e:
            messagebox.showerror("YAML Save Error",f"Could not save YAML configuration: {e}", parent=master_tk_window)
            return None

    def run(self):
        if not self.image_files:
            messagebox.showerror("Error", "Cannot run annotator: No image files found.", parent=self.master_tk_window)
            if self.master_tk_window and self.master_tk_window.winfo_exists(): self.master_tk_window.destroy()
            self._destroy_info_panel()
            return

        try:
            # HighGUI can fight with tkinter on Windows. Spin up its own window thread
            # to avoid GIL/order issues when Tk dialogs fire during annotation.
            cv2.startWindowThread()
        except Exception:
            pass

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.default_window_width, self.default_window_height)
        cv2.setMouseCallback(self.window_name, self._handle_mouse_events)

        if not self._load_image(self.current_image_index):
            messagebox.showerror("Error", "Failed to load the first image. Exiting.", parent=self.master_tk_window)
            cv2.destroyAllWindows()
            if self.master_tk_window and self.master_tk_window.winfo_exists(): self.master_tk_window.destroy()
            self._destroy_info_panel()
            return
        
        self._reset_zoom_pan()
        self._update_display()

        try:
            running = True
            while running:
                if self.img_original_unmodified is None:
                    running = False; break

                try:
                    if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                        running = False; break
                except cv2.error:
                    running = False; break

                if ENABLE_AUTOSAVE and self.annotations_changed_since_last_save and \
                    (time.time() - self.last_autosave_time > AUTOSAVE_INTERVAL_SECONDS):
                    self._save_annotations_for_image(is_autosave=True)
                    self._update_display()

                key = self._poll_key(30)
                action_taken = False
                behavior_keys_map = self._behavior_keys_map()

                if key == ord('q'):
                    if self.annotations_changed_since_last_save:
                        save_choice = messagebox.askyesnocancel("Quit", "Save changes before quitting?", parent=self.master_tk_window)
                        if save_choice is True: self._save_annotations_for_image(); running = False
                        elif save_choice is False: running = False
                    else:
                        if messagebox.askyesno("Quit", "Are you sure you want to quit?", parent=self.master_tk_window):
                            running = False
                elif key == ord('n'):
                    action_taken = self._next_image()
                elif key == ord('p'):
                    action_taken = self._prev_image()
                elif key == ord('g'):
                    action_taken = self._jump_to_image_prompt()
                elif key == ord('s'):
                    self._save_annotations_for_image()
                    action_taken = True
                elif key in (ord('h'), ord('?')):
                    if self.helper_window_open:
                        self._close_helper_panel()
                    elif self.helper_popup_enabled:
                        self._show_helper_panel()
                    action_taken = True
                elif key == ord('u'): self._undo_last_keypoint()
                elif key == ord('r'):
                    action_taken = self._reset_frame_annotations()
                elif key == ord('v') and self.mode == MODE_KP:
                    self.current_kp_visibility_mode = VISIBILITY_LABELED_NOT_VISIBLE if self.current_kp_visibility_mode == VISIBILITY_LABELED_VISIBLE else VISIBILITY_LABELED_VISIBLE
                    action_taken = True
                elif key == ord('l'):
                    action_taken = self._toggle_keypoint_labels()
                elif key in [ord('+'), ord('=')]:
                    center_x, center_y = self._get_current_window_dimensions()
                    self._handle_mouse_events(cv2.EVENT_MOUSEWHEEL, center_x//2, center_y//2, 1, None)
                elif key == ord('-'):
                    center_x, center_y = self._get_current_window_dimensions()
                    self._handle_mouse_events(cv2.EVENT_MOUSEWHEEL, center_x//2, center_y//2, -1, None)
                elif key == ord('['):
                    self._adjust_marker_size(-0.1)
                    action_taken = True
                elif key == ord(']'):
                    self._adjust_marker_size(0.1)
                    action_taken = True
                elif key == ord('0'): self._reset_zoom_pan(); action_taken = True
                elif self.mode == MODE_SELECT_BEHAVIOR and chr(key) in behavior_keys_map:
                    selected_bid = behavior_keys_map[chr(key)]
                    self._set_current_behavior(selected_bid)
                    action_taken = True

                if action_taken or (self.autosave_message and time.time() >= self.autosave_message_display_time):
                    if self.autosave_message and time.time() >= self.autosave_message_display_time:
                        self.autosave_message = ""
                    self._update_display()
                elif key != 255:
                    self._update_display()

            if self.annotations_changed_since_last_save and not running:
                if messagebox.askyesno("Unsaved Changes", "Save changes for the current image?", parent=self.master_tk_window):
                    self._save_annotations_for_image()
        finally:
            try:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            except Exception:
                pass
            self._destroy_info_panel()
            self._close_helper_panel()
            if self.master_tk_window and self.master_tk_window.winfo_exists():
                try:
                    self.master_tk_window.focus_force()
                except Exception:
                    pass
            print("Annotator run loop finished.")
