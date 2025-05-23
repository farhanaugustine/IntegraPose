# keypoint_annotator_module.py

# -*- coding: utf-8 -*-
import cv2
import os
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
import numpy as np
import sys
import yaml
from collections import OrderedDict
import time

# --- Style Configuration ---
KP_VISIBLE_COLOR = (0, 255, 0)      # Green
KP_INVISIBLE_COLOR = (0, 0, 255)    # Red
BBOX_CURRENT_TEMP_COLOR = (255, 255, 0) # Cyan
BBOX_CURRENT_FINAL_COLOR = (0, 255, 255) # Yellow
BBOX_COMPLETED_COLOR = (255, 0, 255)      # Magenta
TEXT_COLOR = (200, 0, 0) # Main instruction color (BGR: Blue)
STATUS_TEXT_COLOR = (0, 0, 0)       # Black
HELP_TEXT_COLOR = (50, 50, 50)      # Dark Gray
ZOOM_PAN_TEXT_COLOR = (0, 100, 0)   # Dark Green
AUTOSAVE_MSG_COLOR = (0, 0, 150)    # Dark Red
BEHAVIOR_TEXT_ON_BOX_COLOR = (0,0,0) # Black
BEHAVIOR_BOX_BACKGROUND_COLOR = (255, 255, 255) # White

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_MAIN_INSTRUCTION = 0.6
FONT_SCALE_MODE_STATUS = 0.5
FONT_SCALE_GENERAL_STATUS = 0.45
FONT_SCALE_HELP_PROMPT = 0.4
FONT_SCALE_FULL_HELP = 0.45
FONT_SCALE_TOOLTIP = 0.45
FONT_SCALE_ANNOTATION_INFO = 0.4 # For behavior text on completed boxes
FONT_SCALE_KP_NUMBER = 0.35      # For numbers next to current keypoints

FONT_THICKNESS = 1
COMPLETED_LINE_THICKNESS = 1
CURRENT_LINE_THICKNESS = 2

# --- Annotation Modes ---
MODE_SELECT_BEHAVIOR = "select_behavior"
MODE_KP = "keypoints"
MODE_BBOX_START = "bbox_start"
MODE_BBOX_END = "bbox_end"

# --- Visibility Flags (YOLO standard) ---
VISIBILITY_NOT_LABELED = 0
VISIBILITY_LABELED_NOT_VISIBLE = 1 # Often used for occluded keypoints
VISIBILITY_LABELED_VISIBLE = 2

# --- Autosave Configuration ---
AUTOSAVE_INTERVAL_SECONDS = 300 # 5 minutes
ENABLE_AUTOSAVE = True

MIN_WINDOW_DIM_THRESHOLD = 100 # Minimum acceptable dimension from getWindowImageRect


class KeypointBehaviorAnnotator:
    def __init__(self, master_tk_window, keypoint_names, available_behaviors, image_dir, output_dir):
        self.master_tk_window = master_tk_window

        if not keypoint_names:
            messagebox.showerror("Error", "Keypoint names list cannot be empty.", parent=self.master_tk_window)
            raise ValueError("Keypoint names list cannot be empty.")
        if not available_behaviors:
            messagebox.showerror("Error", "Available behaviors dictionary cannot be empty.", parent=self.master_tk_window)
            raise ValueError("Available behaviors dictionary cannot be empty.")

        self.keypoint_names = keypoint_names
        self.num_keypoints = len(keypoint_names)
        self.available_behaviors = OrderedDict(sorted(available_behaviors.items()))
        self.image_dir = image_dir
        self.output_dir = output_dir

        self.image_files = self._get_image_files()
        if not self.image_files:
             messagebox.showerror("Error", f"No valid image files found in annotation image directory: {self.image_dir}", parent=self.master_tk_window)
             raise FileNotFoundError(f"No valid image files for annotation in {self.image_dir}")

        os.makedirs(self.output_dir, exist_ok=True)
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
        self.current_mouse_pos_image = (0,0) # Initialize to avoid None access before first mouse move

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
        self.window_name = 'YOLO Keypoint Annotator - CV'
        self.default_window_width = 1280
        self.default_window_height = 720


    def _get_image_files(self):
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        try:
            if not os.path.isdir(self.image_dir):
                 messagebox.showerror("Error", f"Annotation image directory not found: {self.image_dir}", parent=self.master_tk_window)
                 return []
            files = [f for f in os.listdir(self.image_dir) if f.lower().endswith(valid_extensions)]
            files = [f for f in files if not f.startswith('.')] # Ignore hidden files
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
        self._reset_current_instance_state()
        self.annotations_changed_since_last_save = True
        print("Cleared all annotations for the current image.")

    def _load_image(self, index):
        if not (0 <= index < len(self.image_files)):
            print("Invalid image index.")
            return False
        self.current_image_index = index
        self.current_image_path = self.image_files[self.current_image_index]
        self._clear_all_annotations_for_image()
        
        # Load image using IMREAD_UNCHANGED to get original dtype and channels
        raw_img = cv2.imread(self.current_image_path, cv2.IMREAD_UNCHANGED)
        if raw_img is None:
            messagebox.showerror("Error", f"Failed to load image: {self.current_image_path}", parent=self.master_tk_window)
            return False

        print(f"DEBUG: Loaded raw image: {os.path.basename(self.current_image_path)}, Shape: {raw_img.shape}, Dtype: {raw_img.dtype}")

        # --- Convert to BGR uint8 for display ---
        display_img = None
        if raw_img.dtype != np.uint8:
            print(f"DEBUG: Image {os.path.basename(self.current_image_path)} is {raw_img.dtype}, normalizing to uint8 for display.")
            if len(raw_img.shape) == 2 or (len(raw_img.shape) == 3 and raw_img.shape[2] == 1): # Grayscale
                temp_gray = cv2.normalize(raw_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                display_img = cv2.cvtColor(temp_gray, cv2.COLOR_GRAY2BGR)
            elif len(raw_img.shape) == 3 and raw_img.shape[2] == 3: # Color (e.g. BGR, RGB)
                display_img = cv2.normalize(raw_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            elif len(raw_img.shape) == 3 and raw_img.shape[2] == 4: # Color with Alpha (e.g. BGRA, RGBA)
                # Assuming a BGR(A) like order, convert to BGR
                color_conversion_code = cv2.COLOR_BGRA2BGR # Common, check if RGBA if needed
                try:
                    temp_bgr = cv2.cvtColor(raw_img, color_conversion_code)
                except cv2.error: # Fallback if assuming wrong order
                    temp_bgr = cv2.cvtColor(raw_img, cv2.COLOR_RGBA2BGR)
                display_img = cv2.normalize(temp_bgr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            else:
                messagebox.showerror("Image Error", f"Unsupported image type/channels for normalization: Shape {raw_img.shape}, Dtype {raw_img.dtype}", parent=self.master_tk_window)
                return False
        else: # Already uint8
            if len(raw_img.shape) == 2 or (len(raw_img.shape) == 3 and raw_img.shape[2] == 1): # Grayscale uint8
                display_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
            elif len(raw_img.shape) == 3 and raw_img.shape[2] == 3: # BGR uint8
                display_img = raw_img
            elif len(raw_img.shape) == 3 and raw_img.shape[2] == 4: # BGRA/RGBA uint8
                try:
                    display_img = cv2.cvtColor(raw_img, cv2.COLOR_BGRA2BGR)
                except cv2.error:
                    display_img = cv2.cvtColor(raw_img, cv2.COLOR_RGBA2BGR)
            else:
                messagebox.showerror("Image Error", f"Unsupported uint8 image channels: {raw_img.shape}", parent=self.master_tk_window)
                return False
        
        if display_img is None or not (display_img.ndim == 3 and display_img.shape[2] == 3 and display_img.dtype == np.uint8):
            messagebox.showerror("Image Processing Error", f"Failed to convert image to displayable BGR uint8 format. Processed shape {display_img.shape if display_img is not None else 'None'}, dtype {display_img.dtype if display_img is not None else 'None'}", parent=self.master_tk_window)
            return False
        
        print(f"DEBUG: Processed display image - Shape: {display_img.shape}, Dtype: {display_img.dtype}")
        self.img_original_unmodified = display_img.copy()
        self.img_height_orig, self.img_width_orig = self.img_original_unmodified.shape[:2]
        self.annotations_changed_since_last_save = False
        self.last_autosave_time = time.time()

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
                # rect is (x, y, width, height)
                if rect[2] > MIN_WINDOW_DIM_THRESHOLD and rect[3] > MIN_WINDOW_DIM_THRESHOLD:
                    win_w, win_h = rect[2], rect[3]
                else:
                    print(f"DEBUG: getWindowImageRect returned small/invalid dimensions {rect[2]}x{rect[3]}. Using defaults.")
            # else: window not visible or property not available, use defaults
        except cv2.error as e:
            print(f"DEBUG: cv2.error getting window rect: {e}. Using default dimensions.")
        return win_w, win_h

    def _reset_zoom_pan(self):
        self.zoom_level = 1.0
        win_w, win_h = self._get_current_window_dimensions()
        print(f"DEBUG: _reset_zoom_pan using window dimensions: {win_w}x{win_h}")


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
        if self.zoom_level == 0: return 0,0 # Avoid division by zero
        img_x = (screen_x - self.pan_offset[0]) / self.zoom_level
        img_y = (screen_y - self.pan_offset[1]) / self.zoom_level
        return int(round(img_x)), int(round(img_y))

    def _image_to_screen_coords(self, img_x, img_y):
        screen_x = int(round(img_x * self.zoom_level + self.pan_offset[0]))
        screen_y = int(round(img_y * self.zoom_level + self.pan_offset[1]))
        return screen_x, screen_y

    def _constrain_pan(self):
        win_w, win_h = self._get_current_window_dimensions()
        # print(f"DEBUG: _constrain_pan using window dimensions: {win_w}x{win_h}")


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
        # Defer calculating self.current_mouse_pos_image until after zoom_level is checked for 0
        # self.current_mouse_pos_image updated inside conditions or before use
        
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
            if self.zoom_level == 0 : # Should not happen with clip, but as safeguard
                img_x_before_zoom, img_y_before_zoom = 0,0
            else:
                 img_x_before_zoom, img_y_before_zoom = self._screen_to_image_coords(x,y)
            
            zoom_factor = (1 + self.zoom_step_factor)
            if flags > 0: 
                self.zoom_level *= zoom_factor
            else: 
                self.zoom_level /= zoom_factor
            self.zoom_level = np.clip(self.zoom_level, self.min_zoom, self.max_zoom)
            if self.zoom_level == 0: self.zoom_level = self.min_zoom # Ensure not zero

            self.pan_offset[0] = x - img_x_before_zoom * self.zoom_level
            self.pan_offset[1] = y - img_y_before_zoom * self.zoom_level

            self._constrain_pan()
            needs_update = True
        
        # Update image mouse coordinates here after potential zoom changes
        if self.zoom_level != 0:
            self.current_mouse_pos_image = self._screen_to_image_coords(x,y)
        else: # Should be caught by clip, but as a fallback
             self.current_mouse_pos_image = (0,0)


        if self.is_panning_with_alt: 
            if needs_update: self._update_display()
            return

        img_x_clamped = max(0, min(self.current_mouse_pos_image[0], self.img_width_orig - 1))
        img_y_clamped = max(0, min(self.current_mouse_pos_image[1], self.img_height_orig - 1))

        if event == cv2.EVENT_MOUSEMOVE:
            if self.mode == MODE_BBOX_END: needs_update = True 
        elif event == cv2.EVENT_LBUTTONDOWN:
            needs_update = True
            if self.mode == MODE_KP:
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
                        print("   All keypoints placed. Draw bounding box.")
            elif self.mode == MODE_BBOX_START:
                self.current_bbox_start_point = (img_x_clamped, img_y_clamped)
                self.mode = MODE_BBOX_END
                self.annotations_changed_since_last_save = True
                print(f"   BBox Start Point: ({img_x_clamped},{img_y_clamped})")
            elif self.mode == MODE_BBOX_END:
                if self.current_bbox_start_point is None:
                    self.mode = MODE_BBOX_START 
                    print("Error: BBox end clicked without start point. Resetting to BBox start.")
                    return
                x1,y1 = self.current_bbox_start_point
                x2,y2 = img_x_clamped, img_y_clamped
                if x2 <= x1 or y2 <= y1:
                    messagebox.showwarning("BBox Error", "Bottom-right must be below and to the right of top-left.", parent=self.master_tk_window)
                    needs_update=False 
                    return
                self.current_bbox_end_point = (x2,y2)
                print(f"   BBox End Point: ({x2},{y2})")
                completed_annotation = {
                    'behavior_id':self.current_behavior_id,
                    'behavior_name':self.current_behavior_name,
                    'keypoints':self.current_keypoints.copy(),
                    'bbox':(x1,y1,x2,y2) 
                }
                self.annotations_in_image.append(completed_annotation)
                self.annotations_changed_since_last_save = True
                print(f"   Completed instance {len(self.annotations_in_image)} ({self.current_behavior_name}).")
                self._reset_current_instance_state()
            elif self.mode == MODE_SELECT_BEHAVIOR:
                messagebox.showinfo("Info", "Please select a behavior first using number keys (0-9).", parent=self.master_tk_window)
                needs_update=False 
        elif event == cv2.EVENT_MBUTTONDOWN: 
            if self.mode == MODE_KP and self.current_keypoints: 
                last_kp_idx = len(self.current_keypoints) - 1
                if 0 <= last_kp_idx < len(self.current_keypoints):
                    kp_to_toggle = self.current_keypoints[last_kp_idx]
                    current_vis = kp_to_toggle.get('v', VISIBILITY_LABELED_VISIBLE) 
                    if current_vis == VISIBILITY_LABELED_VISIBLE:
                        kp_to_toggle['v'] = VISIBILITY_LABELED_NOT_VISIBLE
                    else: 
                        kp_to_toggle['v'] = VISIBILITY_LABELED_VISIBLE
                    
                    vis_str = "Visible" if kp_to_toggle['v'] == VISIBILITY_LABELED_VISIBLE else "Not Visible"
                    print(f"   MMB Toggled visibility of last placed KP ({kp_to_toggle['name']}) to: {vis_str}")
                    self.annotations_changed_since_last_save = True
                    needs_update = True

        if needs_update:
            self._update_display()

    def _update_display(self):
        if self.img_original_unmodified is None: 
            print("DEBUG: _update_display called but img_original_unmodified is None.")
            # Display a blank screen or a message if no image is loaded
            temp_win_w, temp_win_h = self._get_current_window_dimensions()
            self.img_display = np.full((temp_win_h, temp_win_w, 3), 100, dtype=np.uint8) # Darker gray
            cv2.putText(self.img_display, "No image loaded or image error.", (50, temp_win_h // 2), FONT, 1, (255,255,255), 1, cv2.LINE_AA)
            try:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1 :
                    cv2.imshow(self.window_name, self.img_display)
            except cv2.error as e:
                 print(f"DEBUG: cv2.error in _update_display (no image) imshow: {e}")
            return

        win_w, win_h = self._get_current_window_dimensions()
        print(f"DEBUG: _update_display using window dimensions for canvas: {win_w}x{win_h}")
        if win_w <= 0 or win_h <=0: # Should be caught by MIN_WINDOW_DIM_THRESHOLD in _get_current_window_dimensions
            print(f"ERROR: _update_display detected invalid window dimensions {win_w}x{win_h} even after checks. Cannot draw.")
            return


        self.img_display = np.full((win_h, win_w, 3), 200, dtype=np.uint8) 

        M = np.float32([[self.zoom_level, 0, self.pan_offset[0]],
                        [0, self.zoom_level, self.pan_offset[1]]])

        try:
            cv2.warpAffine(self.img_original_unmodified, M, (win_w, win_h), dst=self.img_display,
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        except cv2.error as e:
            print(f"ERROR: cv2.warpAffine failed: {e}")
            cv2.putText(self.img_display, "Error during image warp.", (50, win_h // 2), FONT, 0.7, (0,0,255), 1, cv2.LINE_AA)


        for ann in self.annotations_in_image:
            x1_orig,y1_orig,x2_orig,y2_orig = ann['bbox']
            x1_s,y1_s = self._image_to_screen_coords(x1_orig,y1_orig)
            x2_s,y2_s = self._image_to_screen_coords(x2_orig,y2_orig)
            cv2.rectangle(self.img_display,(x1_s,y1_s),(x2_s,y2_s),BBOX_COMPLETED_COLOR,COMPLETED_LINE_THICKNESS, lineType=cv2.LINE_AA)

            label=f"{ann['behavior_id']}:{ann['behavior_name']}"
            (lw,lh),baseline = cv2.getTextSize(label,FONT,FONT_SCALE_ANNOTATION_INFO,FONT_THICKNESS)
            cv2.rectangle(self.img_display,(x1_s,y1_s-lh-baseline-2),(x1_s+lw,y1_s),BEHAVIOR_BOX_BACKGROUND_COLOR,-1, lineType=cv2.LINE_AA)
            cv2.putText(self.img_display,label,(x1_s,y1_s-baseline//2 -1),FONT,FONT_SCALE_ANNOTATION_INFO,BEHAVIOR_TEXT_ON_BOX_COLOR,FONT_THICKNESS,cv2.LINE_AA)

            for kp in ann['keypoints']:
                color = KP_VISIBLE_COLOR if kp.get('v', VISIBILITY_NOT_LABELED) == VISIBILITY_LABELED_VISIBLE else KP_INVISIBLE_COLOR
                kp_x_s,kp_y_s = self._image_to_screen_coords(kp['x'],kp['y'])
                radius = int(max(2, 3 * self.zoom_level)) 
                cv2.circle(self.img_display,(kp_x_s,kp_y_s),radius,color,-1, lineType=cv2.LINE_AA)

        for i, kp in enumerate(self.current_keypoints):
            color = KP_VISIBLE_COLOR if kp.get('v', VISIBILITY_LABELED_VISIBLE) == VISIBILITY_LABELED_VISIBLE else KP_INVISIBLE_COLOR
            kp_x_s, kp_y_s = self._image_to_screen_coords(kp['x'], kp['y'])
            radius = int(max(3, 4 * self.zoom_level)) 
            cv2.circle(self.img_display, (kp_x_s, kp_y_s), radius, color, -1, lineType=cv2.LINE_AA)
            num_font_scale = max(0.3, FONT_SCALE_KP_NUMBER * self.zoom_level)
            cv2.putText(self.img_display, str(i + 1), (kp_x_s + radius, kp_y_s + radius), FONT, num_font_scale , color, FONT_THICKNESS, cv2.LINE_AA)

        if self.current_bbox_start_point and self.mode == MODE_BBOX_END: 
            st_x_s,st_y_s = self._image_to_screen_coords(self.current_bbox_start_point[0],self.current_bbox_start_point[1])
            cv2.circle(self.img_display,(st_x_s,st_y_s),int(max(3,4*self.zoom_level)),BBOX_CURRENT_TEMP_COLOR,-1, lineType=cv2.LINE_AA)

        if self.mode == MODE_BBOX_END and self.current_bbox_start_point and self.current_mouse_pos_image:
            st_x_s,st_y_s = self._image_to_screen_coords(self.current_bbox_start_point[0],self.current_bbox_start_point[1])
            cur_m_x_img_clamped = max(0, min(self.current_mouse_pos_image[0], self.img_width_orig -1))
            cur_m_y_img_clamped = max(0, min(self.current_mouse_pos_image[1], self.img_height_orig -1))
            cur_m_x_s,cur_m_y_s = self._image_to_screen_coords(cur_m_x_img_clamped, cur_m_y_img_clamped)
            cv2.rectangle(self.img_display,(st_x_s,st_y_s),(cur_m_x_s,cur_m_y_s),BBOX_CURRENT_TEMP_COLOR,CURRENT_LINE_THICKNESS, lineType=cv2.LINE_AA)

        y_off = 20; line_height_small = 18; line_height_normal = 22
        def draw_text_line(text_content, color=STATUS_TEXT_COLOR, scale=FONT_SCALE_GENERAL_STATUS, is_main_instr=False):
            nonlocal y_off; lh = line_height_normal if is_main_instr else line_height_small
            # Ensure y_off doesn't go beyond display height
            if y_off < win_h - lh :
                 cv2.putText(self.img_display, text_content, (10, y_off), FONT, scale, color, FONT_THICKNESS, cv2.LINE_AA)
                 y_off += int(lh * (scale / FONT_SCALE_GENERAL_STATUS) + 5)

        img_file_name = os.path.basename(self.current_image_path) if self.current_image_path else "N/A"
        mouse_x_str = str(self.current_mouse_pos_image[0]) if self.current_mouse_pos_image else 'N/A'
        mouse_y_str = str(self.current_mouse_pos_image[1]) if self.current_mouse_pos_image else 'N/A'

        draw_text_line(f"{img_file_name} [{self.current_image_index + 1}/{len(self.image_files)}] Anns: {len(self.annotations_in_image)}")
        draw_text_line(f"Zoom: {self.zoom_level:.2f}x Pan:({int(self.pan_offset[0])},{int(self.pan_offset[1])}) XY_img:({mouse_x_str},{mouse_y_str})", color=ZOOM_PAN_TEXT_COLOR)
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
            tooltip_text = "Use number keys for behaviors (see console for full list if many)."
        elif self.mode == MODE_KP:
            mode_text = f"MODE: Keypoints for '{self.current_behavior_name}' (ID: {self.current_behavior_id})"
            if self.current_keypoint_index < self.num_keypoints:
                kp_name_disp = self.keypoint_names[self.current_keypoint_index]
                instruction_text = f"Click KP {self.current_keypoint_index + 1}/{self.num_keypoints}: {kp_name_disp}"
            else:
                instruction_text = "All keypoints placed for this instance!" 
            tooltip_text = "LMB: Place KP. MMB: Toggle last KP vis. 'V': Toggle NEXT KP vis."
        elif self.mode == MODE_BBOX_START:
            mode_text = f"MODE: BBox Start for '{self.current_behavior_name}'"
            instruction_text = "Click TOP-LEFT of bounding box."
        elif self.mode == MODE_BBOX_END:
            mode_text = f"MODE: BBox End for '{self.current_behavior_name}'"
            instruction_text = "Click BOTTOM-RIGHT of bounding box."

        if mode_text: draw_text_line(mode_text, color=TEXT_COLOR, scale=FONT_SCALE_MODE_STATUS)
        if instruction_text: draw_text_line(instruction_text, color=TEXT_COLOR, scale=FONT_SCALE_MAIN_INSTRUCTION, is_main_instr=True)

        if self.mode == MODE_KP: 
            vis_mode_str_display = "Next KP: VISIBLE" if self.current_kp_visibility_mode == VISIBILITY_LABELED_VISIBLE else "Next KP: NOT VISIBLE"
            draw_text_line(f"{vis_mode_str_display} (Press 'V' to toggle)", color=TEXT_COLOR, scale=FONT_SCALE_TOOLTIP)

        if tooltip_text: draw_text_line(f"Tip: {tooltip_text}", color=HELP_TEXT_COLOR, scale=FONT_SCALE_TOOLTIP)

        if self.autosave_message and time.time() < self.autosave_message_display_time:
            (tw,th),_ = cv2.getTextSize(self.autosave_message,FONT,FONT_SCALE_HELP_PROMPT,FONT_THICKNESS)
            if win_h > th + 20 : # Ensure space to draw
                 cv2.putText(self.img_display,self.autosave_message,(10,win_h-th-10),FONT,FONT_SCALE_HELP_PROMPT,AUTOSAVE_MSG_COLOR,FONT_THICKNESS,cv2.LINE_AA)
        elif self.autosave_message: 
            self.autosave_message=""

        help_y_start=win_h-10; help_line_h=18
        if self.show_help_text:
            help_lines = [
                "Keys: (N)ext Img, (P)rev Img, (S)ave, (Q)uit",
                "(H)ToggleHelp, (U)ndo Last KP, (R)eset/Clear All Anns",
                "(V)Toggle Next KP Vis (KP mode), (MMB)Toggle Last KP Vis (KP mode)",
                "(+/-/=)Zoom In, (-)Zoom Out, (0)Reset Zoom/Pan",
                "(Alt+LMB Drag) Pan Image"
            ]
            for i,line in enumerate(reversed(help_lines)): 
                (tw,th),_ = cv2.getTextSize(line,FONT,FONT_SCALE_FULL_HELP,FONT_THICKNESS)
                if help_y_start-i*help_line_h-th > y_off + th: # Ensure help text doesn't overlap HUD
                    cv2.putText(self.img_display,line,(10,help_y_start-i*help_line_h-th),FONT,FONT_SCALE_FULL_HELP,HELP_TEXT_COLOR,FONT_THICKNESS,cv2.LINE_AA)
        else:
            help_prompt="Press 'H' for Keybindings";
            (tw,th),_ = cv2.getTextSize(help_prompt,FONT,FONT_SCALE_HELP_PROMPT,FONT_THICKNESS)
            if win_h > th + 20 and help_y_start-th > y_off + th : # Ensure space and no overlap
                cv2.putText(self.img_display,help_prompt,(10,win_h-th-10),FONT,FONT_SCALE_HELP_PROMPT,HELP_TEXT_COLOR,FONT_THICKNESS,cv2.LINE_AA)
        
        try:
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1 :
                 cv2.imshow(self.window_name, self.img_display)
        except cv2.error as e:
            print(f"DEBUG: cv2.error in _update_display during imshow: {e}. Window might be closed.")


    def _undo_last_keypoint(self): # Renamed for clarity, as it can undo more than just KPs
        current_mode_before_undo = self.mode
        if self.mode == MODE_KP and self.current_keypoints:
            removed_kp = self.current_keypoints.pop()
            self.current_keypoint_index -= 1
            self.annotations_changed_since_last_save = True
            print(f"   Undo: Removed keypoint {removed_kp['name']}. Current KP index: {self.current_keypoint_index +1}")
        elif self.mode == MODE_BBOX_START: 
            self.mode = MODE_KP
            # self.current_keypoint_index should be num_keypoints, ready to re-place/undo last KP if desired by another 'u'
            print(f"   Undo: Canceled BBox selection, returned to Keypoint mode (after KP {self.current_keypoint_index}).")
        elif self.mode == MODE_BBOX_END: 
            self.mode = MODE_BBOX_START
            # self.current_bbox_start_point is kept, user can re-click end point or undo again
            print(f"   Undo: Canceled BBox end point. Re-click BBox end point or Undo to return to Keypoints.")
        elif self.mode == MODE_SELECT_BEHAVIOR and self.annotations_in_image:
             # Undoing a completed instance
             removed_ann = self.annotations_in_image.pop()
             self.annotations_changed_since_last_save = True
             # Restore state to re-annotate this instance or select new behavior
             self.current_behavior_id = removed_ann['behavior_id']
             self.current_behavior_name = removed_ann['behavior_name']
             self.current_keypoints = removed_ann['keypoints']
             self.current_keypoint_index = len(self.current_keypoints)
             self.current_bbox_start_point = (removed_ann['bbox'][0], removed_ann['bbox'][1])
             self.current_bbox_end_point = (removed_ann['bbox'][2], removed_ann['bbox'][3])
             self.mode = MODE_BBOX_END # Or MODE_KP if you want to allow editing KPs of last instance

             print(f"   Undo: Removed completed instance for behavior '{removed_ann['behavior_name']}'. Restored its state for potential re-annotation.")
        else:
            print("Nothing to undo or invalid state for undo.")
            return # No display update if nothing changed

        self._update_display()


    def _save_annotations_for_image(self, is_autosave=False):
        # ... (rest of the function remains the same as your provided version)
        if not self.annotations_in_image and not is_autosave: # For manual save, don't save if empty
            print(f"No annotations to save for {os.path.basename(self.current_image_path)}.")
            self.annotations_changed_since_last_save = False # No changes to save
            return True
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
                    print(f"No annotations to save for {os.path.basename(self.current_image_path)}, and no existing file.")
                self.annotations_changed_since_last_save=False
                self.last_autosave_time=time.time()
                if is_autosave: 
                    self.autosave_message=f"Autosaved: {os.path.basename(output_path)} (cleared)"
                    self.autosave_message_display_time=time.time()+3
                return True
            except Exception as e:
                messagebox.showerror("Save Error",f"Could not remove existing annotation file: {e}", parent=self.master_tk_window)
                return False


        for ann in self.annotations_in_image:
            b_id = ann['behavior_id']; x1,y1,x2,y2 = ann['bbox']; kps_data = ann['keypoints']
            img_w_f = float(self.img_width_orig)
            img_h_f = float(self.img_height_orig)

            box_w_abs, box_h_abs = (x2-x1), (y2-y1)
            cx_abs, cy_abs = x1 + box_w_abs/2.0, y1 + box_h_abs/2.0

            norm_cx = np.clip(cx_abs/img_w_f,0.0,1.0)
            norm_cy = np.clip(cy_abs/img_h_f,0.0,1.0)
            norm_w = np.clip(box_w_abs/img_w_f,0.0,1.0)
            norm_h = np.clip(box_h_abs/img_h_f,0.0,1.0)

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
                    print(f"Warning: Keypoint '{kp_name_ordered}' not found in current annotation instance. Saving as (0,0,0).")
                    yolo_kpts_flat.extend([0.0,0.0,float(VISIBILITY_NOT_LABELED)])

            bbox_str = f"{norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}"
            kpts_str_parts = []
            for i in range(0, len(yolo_kpts_flat), 3):
                kpts_str_parts.append(f"{yolo_kpts_flat[i]:.6f}") 
                kpts_str_parts.append(f"{yolo_kpts_flat[i+1]:.6f}") 
                kpts_str_parts.append(f"{int(yolo_kpts_flat[i+2])}") 
            kpts_str = " ".join(kpts_str_parts)

            lines_to_write.append(f"{b_id} {bbox_str} {kpts_str}")

        try:
            with open(output_path,'w') as f:
                for line in lines_to_write: f.write(line + "\n")
            save_type = "Autosaved" if is_autosave else "Saved"
            print(f"{save_type} {len(lines_to_write)} instance(s) to: {output_path}")
            self.annotations_changed_since_last_save=False
            self.last_autosave_time=time.time()
            if is_autosave:
                self.autosave_message=f"Autosaved: {os.path.basename(output_path)}"
                self.autosave_message_display_time=time.time()+3 
            return True
        except Exception as e:
            messagebox.showerror("Save Error",f"Could not write annotation file: {e}", parent=self.master_tk_window)
            return False

    def generate_config_yaml(self, dataset_root_path, train_images_path, val_images_path, dataset_name_suggestion="dataset_config"):
        # ... (rest of the function remains the same as your provided version)
        print("\nGenerating dataset configuration YAML...")
        if not all([dataset_root_path, train_images_path, val_images_path]): 
            messagebox.showerror("Error", "Dataset root, train images folder, and validation images folder are ALL required for YAML generation.", parent=self.master_tk_window)
            return None
        try:
            abs_dataset_root = os.path.abspath(dataset_root_path)
            abs_train_images = os.path.abspath(train_images_path)
            abs_val_images = os.path.abspath(val_images_path)

            rel_train_path = os.path.relpath(abs_train_images, abs_dataset_root)
            rel_val_path = os.path.relpath(abs_val_images, abs_dataset_root)

            if rel_train_path.startswith('..') or rel_val_path.startswith('..'):
                 messagebox.showwarning("Path Warning", "Train/Validation image paths seem to be outside the specified Dataset Root Path. Relative paths in YAML might be incorrect (e.g., contain '..'). Consider restructuring or use absolute paths in the YAML if issues arise.", parent=self.master_tk_window)

        except ValueError as e: 
             messagebox.showerror("Path Error", f"Could not determine relative paths. Ensure Train/Validation paths are subdirectories of the Dataset Root Path, or on the same drive.\nError: {e}\nPlease use absolute paths in YAML or restructure.", parent=self.master_tk_window)
             return None

        behavior_names_map = {int(bid): name for bid, name in self.available_behaviors.items()}

        yaml_data = OrderedDict([
            ('path', abs_dataset_root.replace(os.sep,'/')), 
            ('train', rel_train_path.replace(os.sep,'/')), 
            ('val', rel_val_path.replace(os.sep,'/')),     
            ('test', ''), 
            ('kpt_shape', [self.num_keypoints, 3]), 
            ('nc', len(self.available_behaviors)), 
            ('names', behavior_names_map) 
        ])
        header = f"""# YOLO Dataset Configuration File
# Generated by KeypointBehaviorAnnotator
# Date: {time.strftime("%Y-%m-%d %H:%M:%S")}
#
# Relative paths (train, val) are relative to the 'path' directory below.
# Keypoint order: {self.keypoint_names}
# Keypoint visibility flags:
#   {VISIBILITY_NOT_LABELED}: Not labeled (or not present in this image)
#   {VISIBILITY_LABELED_NOT_VISIBLE}: Labeled but not visible (e.g., occluded)
#   {VISIBILITY_LABELED_VISIBLE}: Labeled and visible
# ---
"""
        save_path_initial_file = f"{dataset_name_suggestion}.yaml"
        save_path = filedialog.asksaveasfilename(
            title="Save Dataset Config YAML",
            initialdir=abs_dataset_root, 
            initialfile=save_path_initial_file,
            defaultextension=".yaml",
            filetypes=[("YAML files","*.yaml *.yml")],
            parent=self.master_tk_window
        )
        if not save_path:
            print("YAML saving cancelled by user.")
            return None
        try:
            with open(save_path,'w',encoding='utf-8') as f:
                f.write(header)
                yaml.dump(dict(yaml_data),f,default_flow_style=None,sort_keys=False,indent=4)
            print(f"Dataset config YAML saved to: {save_path}")
            messagebox.showinfo("YAML Saved",f"Dataset configuration YAML saved to:\n{save_path}\n\nPlease review the YAML file, especially the paths and comments, to ensure it's correctly set up for your YOLO training.", parent=self.master_tk_window)
            return save_path
        except Exception as e:
            messagebox.showerror("YAML Save Error",f"Could not save YAML configuration: {e}", parent=self.master_tk_window)
            return None

    def run(self):
        # ... (rest of the function, including the main loop, remains largely the same but benefits from the _load_image and _update_display fixes)
        if not self.image_files:
            messagebox.showerror("Error", "Cannot run annotator: No image files found or image directory is invalid. Please check the image directory.", parent=self.master_tk_window)
            if self.master_tk_window and self.master_tk_window.winfo_exists(): self.master_tk_window.destroy() 
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL) 
        cv2.resizeWindow(self.window_name, self.default_window_width, self.default_window_height)
        cv2.setMouseCallback(self.window_name, self._handle_mouse_events)

        if not self._load_image(self.current_image_index): 
            messagebox.showerror("Error", "Failed to load the first image for annotation. Exiting.", parent=self.master_tk_window)
            cv2.destroyAllWindows()
            if self.master_tk_window and self.master_tk_window.winfo_exists(): self.master_tk_window.destroy()
            return
        
        self._reset_zoom_pan() 
        self._update_display() 

        behavior_keys_map = {}
        b_ids = list(self.available_behaviors.keys())
        for i in range(min(10, len(b_ids))): 
            key_char = str((i + 1) % 10) 
            behavior_keys_map[key_char] = b_ids[i]

        running = True
        while running:
            if self.img_original_unmodified is None: 
                print("Error: Original image is None. Annotator cannot continue.")
                running = False; break

            try:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("Annotation window was closed by user (detected by WND_PROP_VISIBLE).")
                    running = False; break 
            except cv2.error as e: 
                if "could not find window" in str(e).lower() or "NULL window" in str(e).lower() or "Invalid window name" in str(e).lower() :
                    print(f"Annotation window was closed by user or unavailable (detected by cv2.error: {e}).")
                    running = False; break
                else: 
                    print(f"An unexpected cv2.error occurred: {e}")
                    raise e

            if ENABLE_AUTOSAVE and self.annotations_changed_since_last_save and \
               (time.time() - self.last_autosave_time > AUTOSAVE_INTERVAL_SECONDS):
                print("Autosave triggered by interval...")
                self._save_annotations_for_image(is_autosave=True)
                self._update_display() 

            key = cv2.waitKey(30) & 0xFF 
            action_taken_requires_display_update = False

            if key == ord('q'): 
                if self.annotations_changed_since_last_save:
                    save_choice = messagebox.askyesnocancel("Quit Confirmation", "You have unsaved changes. Save before quitting?", parent=self.master_tk_window)
                    if save_choice is True: # Yes, save
                        self._save_annotations_for_image(is_autosave=False)
                        running = False
                    elif save_choice is False: # No, don't save
                        running = False
                    # Else (Cancel), do nothing
                else:
                    if messagebox.askyesno("Quit Annotator", "Are you sure you want to quit?", parent=self.master_tk_window):
                        running = False
            elif key == ord('h'): 
                self.show_help_text = not self.show_help_text
                action_taken_requires_display_update = True
            elif key == ord('v') and self.mode == MODE_KP: 
                if self.current_kp_visibility_mode == VISIBILITY_LABELED_VISIBLE:
                    self.current_kp_visibility_mode = VISIBILITY_LABELED_NOT_VISIBLE
                else:
                    self.current_kp_visibility_mode = VISIBILITY_LABELED_VISIBLE
                action_taken_requires_display_update = True
                vis_mode_str = "VISIBLE" if self.current_kp_visibility_mode == VISIBILITY_LABELED_VISIBLE else "NOT VISIBLE"
                print(f" -> Next keypoint to be placed will be marked as: {vis_mode_str}")
            elif key == ord('n'): 
                if self._save_annotations_for_image(): 
                    if self.current_image_index + 1 < len(self.image_files):
                        if self._load_image(self.current_image_index + 1):
                             self._reset_zoom_pan() 
                             action_taken_requires_display_update = True
                        else: running = False 
                    else:
                        messagebox.showinfo("End of Set", "You are at the last image.", parent=self.master_tk_window)
            elif key == ord('p'): 
                if self._save_annotations_for_image():
                    if self.current_image_index - 1 >= 0:
                        if self._load_image(self.current_image_index - 1):
                            self._reset_zoom_pan()
                            action_taken_requires_display_update = True
                        else: running = False
                    else:
                        messagebox.showinfo("Start of Set", "You are at the first image.", parent=self.master_tk_window)
            elif key == ord('u'): 
                self._undo_last_keypoint() 
            elif key == ord('r'): 
                if messagebox.askyesno("Reset Image Annotations", "Are you sure you want to clear ALL annotations for the current image? This cannot be undone for this image.", parent=self.master_tk_window):
                    self._clear_all_annotations_for_image()
                    action_taken_requires_display_update = True
            elif key == ord('s'): 
                if self._save_annotations_for_image(is_autosave=False):
                    messagebox.showinfo("Saved", "Annotations for current image have been saved.", parent=self.master_tk_window)
                action_taken_requires_display_update = True 
            elif key == ord('+') or key == ord('='): 
                center_x_screen, center_y_screen = self._get_current_window_dimensions()
                center_x_screen //= 2
                center_y_screen //= 2
                
                img_x_center, img_y_center = self._screen_to_image_coords(center_x_screen, center_y_screen)
                self.zoom_level = min(self.max_zoom, self.zoom_level * (1 + self.zoom_step_factor * 2))
                self.pan_offset[0] = center_x_screen - img_x_center * self.zoom_level
                self.pan_offset[1] = center_y_screen - img_y_center * self.zoom_level
                self._constrain_pan(); action_taken_requires_display_update=True
            elif key == ord('-'): 
                center_x_screen, center_y_screen = self._get_current_window_dimensions()
                center_x_screen //= 2
                center_y_screen //= 2

                img_x_center, img_y_center = self._screen_to_image_coords(center_x_screen, center_y_screen)
                self.zoom_level = max(self.min_zoom, self.zoom_level / (1 + self.zoom_step_factor * 2))
                self.pan_offset[0] = center_x_screen - img_x_center * self.zoom_level
                self.pan_offset[1] = center_y_screen - img_y_center * self.zoom_level
                self._constrain_pan(); action_taken_requires_display_update=True
            elif key == ord('0'): 
                self._reset_zoom_pan(); action_taken_requires_display_update=True
            elif self.mode == MODE_SELECT_BEHAVIOR and chr(key) in behavior_keys_map:
                selected_bid = behavior_keys_map[chr(key)]
                self.current_behavior_id = selected_bid
                self.current_behavior_name = self.available_behaviors[selected_bid]
                self.mode = MODE_KP
                self.current_keypoint_index = 0
                self.current_keypoints = [] 
                self.current_bbox_start_point = None
                self.current_bbox_end_point = None
                self.current_kp_visibility_mode = VISIBILITY_LABELED_VISIBLE 
                print(f"Selected Behavior: '{self.current_behavior_name}' (ID: {self.current_behavior_id}). Ready for keypoints.")
                action_taken_requires_display_update = True

            if action_taken_requires_display_update or \
               (self.autosave_message and time.time() >= self.autosave_message_display_time):
                if self.autosave_message and time.time() >= self.autosave_message_display_time:
                    self.autosave_message = "" 
                self._update_display()
            elif key != 255 and key != -1 : 
                 self._update_display()


        if self.annotations_changed_since_last_save and running is False: 
             if messagebox.askyesno("Unsaved Changes", "The annotator session ended with unsaved changes for the current image. Save them now?", parent=self.master_tk_window):
                self._save_annotations_for_image(is_autosave=False)

        cv2.destroyAllWindows()
        if self.master_tk_window and self.master_tk_window.winfo_exists():
            self.master_tk_window.focus_force()
        print("Annotator run loop finished.")

if __name__ == "__main__":
    # ... (main test block remains the same)
    print("Testing KeypointBehaviorAnnotator module...")
    test_root = tk.Tk()
    test_root.withdraw() 

    try:
        _keypoint_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'tail_base']
        _available_behaviors = OrderedDict([
            (0, 'Idle'), (1, 'Walking'), (2, 'Running'),
            (3, 'Eating'), (4, 'Sleeping'), (5, 'Playing'),
            (6, 'Jumping'),(7, 'Sitting'),(8, 'Grooming'),(9, 'Alert') 
        ])

        _image_dir = filedialog.askdirectory(title="Select Image Directory for Annotation Test", parent=test_root)
        if not _image_dir or not os.path.isdir(_image_dir):
            print("No valid image directory selected, or selection cancelled. Exiting test.")
            if test_root.winfo_exists(): test_root.destroy()
            sys.exit()

        _output_dir = filedialog.askdirectory(title="Select Annotation Output Directory for Test", parent=test_root)
        if not _output_dir: 
            print("No output directory selected, or selection cancelled. Exiting test.")
            if test_root.winfo_exists(): test_root.destroy()
            sys.exit()

        print(f"Test Setup: Image Dir: '{_image_dir}', Output Dir: '{_output_dir}'")

        annotator = KeypointBehaviorAnnotator(
            master_tk_window=test_root, 
            keypoint_names=_keypoint_names,
            available_behaviors=_available_behaviors,
            image_dir=_image_dir,
            output_dir=_output_dir
        )
        annotator.run() 

        active_tk_window = test_root if test_root.winfo_exists() else None
        if active_tk_window:
             if messagebox.askyesno("Generate YAML?", "Annotation session finished. Do you want to generate a sample dataset_config.yaml file?", parent=active_tk_window):
                _dataset_root = filedialog.askdirectory(title="Select Dataset ROOT Directory (for YAML 'path')", parent=active_tk_window)
                if _dataset_root:
                    _train_imgs_path = filedialog.askdirectory(title="Select TRAIN Images Folder (relative to Dataset Root)", initialdir=_dataset_root, parent=active_tk_window)
                    _val_imgs_path = filedialog.askdirectory(title="Select VALIDATION Images Folder (relative to Dataset Root)", initialdir=_dataset_root, parent=active_tk_window)

                    if _train_imgs_path and _val_imgs_path:
                         annotator.generate_config_yaml(_dataset_root, _train_imgs_path, _val_imgs_path, "example_dataset_config")
                    else:
                         messagebox.showwarning("YAML Generation Skipped", "Train or Validation image paths not provided. YAML generation skipped.", parent=active_tk_window)
                else:
                    messagebox.showwarning("YAML Generation Skipped", "Dataset root path not provided. YAML generation skipped.", parent=active_tk_window)

    except FileNotFoundError as fnf_err:
        print(f"File Not Found Error during test: {fnf_err}")
        if 'test_root' in locals() and test_root.winfo_exists():
             messagebox.showerror("Test Error - File Not Found", str(fnf_err), parent=test_root)
    except ValueError as val_err:
        print(f"Value Error during test: {val_err}")
        if 'test_root' in locals() and test_root.winfo_exists():
             messagebox.showerror("Test Error - Value Error", str(val_err), parent=test_root)
    except Exception as e:
        print(f"An unexpected error occurred during the standalone test: {e}")
        import traceback
        traceback.print_exc()
        if 'test_root' in locals() and test_root.winfo_exists():
            messagebox.showerror("Test Error - Unexpected", f"An error occurred: {e}", parent=test_root)
    finally:
        if 'test_root' in locals() and test_root.winfo_exists():
            test_root.destroy()
        print("Standalone test finished.")