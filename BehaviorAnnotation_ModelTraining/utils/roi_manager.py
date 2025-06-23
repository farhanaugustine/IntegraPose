import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
import os
import re

def get_frame_num_from_filename(filename):
    """
    Refactored: Extracts the frame number from a filename using a robust regex.
    This version specifically looks for numbers at the end of the filename
    (e.g., '..._123.txt'), making it safer than the previous implementation.
    """
    # Example matches: video_frame_123.txt -> 123
    match = re.search(r'(\d+)\.txt$', filename)
    if match:
        return int(match.group(1))
    
    # Fallback for filenames that are just numbers (e.g., '123.txt')
    match = re.search(r'^(\d+)\.txt$', filename)
    if match:
        return int(match.group(1))
        
    return None

class ROIManager:
    def __init__(self):
        self.rois = {}
        self.animal_states = {}
        self.behavior_names = {}
        self.detailed_bouts = []

    def add_roi(self, name, polygon):
        """Adds a named ROI polygon."""
        if name and polygon:
            self.rois[name] = np.array(polygon, np.int32)

    def _reset_analytics(self):
        """Resets all stored analytics and animal states."""
        self.animal_states = {}
        self.behavior_names = {}
        self.detailed_bouts = []

    def _get_roi_for_point(self, point):
        """Checks which ROI a given point falls into."""
        for name, polygon in self.rois.items():
            if cv2.pointPolygonTest(polygon, point, False) >= 0:
                return name
        return None

    def _finalize_bout(self, animal_id, end_frame, min_bout_duration_frames):
        """
        Finalizes an animal's current bout if it meets the minimum duration.
        This is called when an animal leaves an ROI, changes behavior, or disappears.
        """
        state = self.animal_states.get(animal_id)
        if not state or not state.get("current_roi") or not state.get("current_bout"):
            return

        bout = state["current_bout"]
        duration = end_frame - bout["start_frame"]

        if duration >= min_bout_duration_frames:
            roi_name = state["current_roi"]
            behavior_name = self.behavior_names.get(bout["class_id"], f"Unknown_{bout['class_id']}")
            
            self.detailed_bouts.append({
                "ROI Name": roi_name,
                "Animal ID": animal_id,
                "Behavior": behavior_name,
                "Bout Start Frame": bout["start_frame"],
                "Bout End Frame": end_frame,
            })
        
        # Clear the bout from the animal's state
        state["current_bout"] = None

    def _update_animal_state(self, animal_id, detection, frame_num, video_dims, min_bout_duration_frames):
        """
        Refactored: Helper function to manage state transitions for a single animal on a given frame.
        This improves readability of the main processing loop.
        """
        state = self.animal_states.get(animal_id, {})
        video_width, video_height = video_dims
        
        # Determine the animal's current ROI based on its detection coordinates
        current_roi = None
        if detection:
            x_abs = detection['cx'] * video_width
            y_abs = detection['cy'] * video_height
            current_roi = self._get_roi_for_point((int(x_abs), int(y_abs)))

        # Initialize state for a newly seen animal
        if not state:
            state = {"last_seen_frame": frame_num, "current_roi": None, "current_bout": None}
            self.animal_states[animal_id] = state

        prev_roi = state.get("current_roi")
        
        # --- Handle State Changes ---
        # 1. If the animal changed ROI, finalize the previous bout
        if current_roi != prev_roi:
            if prev_roi is not None:
                self._finalize_bout(animal_id, frame_num, min_bout_duration_frames)
        
        # 2. If the animal is inside an ROI, manage its behavior bout
        if current_roi:
            current_bout = state.get("current_bout")
            detected_class_id = detection['class_id']

            # If no active bout or the behavior changes, start a new bout
            if not current_bout or current_bout["class_id"] != detected_class_id:
                # Finalize any previous bout before starting a new one
                self._finalize_bout(animal_id, frame_num, min_bout_duration_frames)
                state["current_bout"] = {"class_id": detected_class_id, "start_frame": frame_num}
        
        # Update the state with the latest information
        state["current_roi"] = current_roi
        if detection:
            state["last_seen_frame"] = frame_num

    def process_yolo_path(self, yolo_path, behavior_names, video_width, video_height, max_frame_gap, min_bout_duration_frames):
        """
        Processes a directory of YOLO .txt files to analyze behavior within ROIs.
        """
        self._reset_analytics()
        self.behavior_names = behavior_names
        video_dims = (video_width, video_height)

        if not os.path.exists(yolo_path):
            raise FileNotFoundError(f"YOLO output path does not exist: {yolo_path}")

        # --- Step 1: Load and sort all detection files by frame number ---
        txt_files_with_frames = []
        for f in os.listdir(yolo_path):
            if f.endswith('.txt'):
                frame_num = get_frame_num_from_filename(f)
                if frame_num is not None:
                    txt_files_with_frames.append((frame_num, os.path.join(yolo_path, f)))
        txt_files_with_frames.sort()

        if not txt_files_with_frames:
            raise FileNotFoundError("No .txt files with valid frame numbers found.")

        # --- Step 2: Process frames sequentially ---
        last_processed_frame = -1
        for frame_num, file_path in txt_files_with_frames:
            # Handle skipped frames: check if any animals have disappeared
            if frame_num > last_processed_frame + 1:
                for animal_id in list(self.animal_states.keys()):
                    last_seen = self.animal_states[animal_id].get("last_seen_frame", frame_num)
                    if frame_num - last_seen >= max_frame_gap:
                        self._finalize_bout(animal_id, last_seen, min_bout_duration_frames)
                        del self.animal_states[animal_id]

            # Load all detections for the current frame
            detections_this_frame = {}
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 6: continue
                    detections_this_frame[int(float(parts[5]))] = {
                        'class_id': int(float(parts[0])),
                        'cx': float(parts[1]),
                        'cy': float(parts[2])
                    }

            # Update state for all animals (both currently and previously seen)
            all_known_ids = set(self.animal_states.keys()) | set(detections_this_frame.keys())
            for animal_id in all_known_ids:
                self._update_animal_state(animal_id, detections_this_frame.get(animal_id), frame_num, video_dims, min_bout_duration_frames)
            
            last_processed_frame = frame_num
            
        # --- Step 3: Finalize any remaining bouts after the last frame ---
        end_frame = last_processed_frame + 1
        for animal_id in list(self.animal_states.keys()):
            self._finalize_bout(animal_id, end_frame, min_bout_duration_frames)

    def get_analytics(self):
        """Returns the processed bout data, sorted for consistency."""
        self.detailed_bouts.sort(key=lambda x: (x['ROI Name'], x['Animal ID'], x['Bout Start Frame']))
        return self.detailed_bouts

    def export_csv(self, file_path):
        """Exports the detailed bout data to a CSV file."""
        bouts = self.get_analytics()
        if not bouts: return
        df = pd.DataFrame(bouts)
        df.to_csv(file_path, index=False)