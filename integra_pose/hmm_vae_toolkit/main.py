import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, scrolledtext
import os
import re
import json
import logging
import sys
import threading
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
import umap
import hdbscan
import cv2
from scipy.spatial.distance import pdist, squareform
from jsonschema import ValidationError, Draft7Validator

from integra_pose.gui.windowing import apply_adaptive_window_geometry
from integra_pose.gui.theme import (
    BOLD_FONT,
    CAPTION_FONT,
    INLINE_BOLD_FONT,
    MUTED_FG,
)

from .data_processing_hdbscan import read_detections as helper_read_detections


logger = logging.getLogger(__name__)


def configure_toolkit_logging(*, stream_to_stdout: bool | None = None) -> Path:
    """Configure a dedicated logger for the toolkit without touching the root logger.

    Returns the resolved log file path.
    """
    if getattr(logger, "_integrapose_configured", False):
        cached = str(getattr(logger, "_integrapose_log_path", "") or "").strip()
        if cached:
            return Path(cached)
        return Path(__file__).resolve().parent / "behavior_analysis.log"

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    script_dir = Path(__file__).resolve().parent
    log_path = script_dir / "behavior_analysis.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    except Exception:
        log_path = Path.home() / "behavior_analysis.log"
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")

    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    parent_logger = logging.getLogger("integra_pose")
    if stream_to_stdout is None:
        stream_to_stdout = not bool(parent_logger.handlers)

    if stream_to_stdout:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.propagate = False
    else:
        logger.propagate = True

    logger.setLevel(logging.INFO)
    setattr(logger, "_integrapose_configured", True)
    setattr(logger, "_integrapose_log_path", str(log_path))
    return log_path


class Tooltip:
    """Simple tooltip for Tkinter widgets."""

    def __init__(self, widget, text, delay=500, wraplength=320):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.wraplength = wraplength
        self.after_id = None
        self.tip_window = None

        self.widget.bind("<Enter>", self._schedule)
        self.widget.bind("<Leave>", self._unschedule)
        self.widget.bind("<ButtonPress>", self._unschedule)

    def _schedule(self, _event):
        self._unschedule()
        self.after_id = self.widget.after(self.delay, self._show)

    def _unschedule(self, _event=None):
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None
        self._hide()

    def _show(self):
        if self.tip_window or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert") or (0, 0, 0, 0)
        x_root = self.widget.winfo_rootx() + x + cx + 8
        y_root = self.widget.winfo_rooty() + y + cy + 8

        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x_root}+{y_root}")

        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            relief=tk.SOLID,
            borderwidth=1,
            background="#ffffe0",
            wraplength=self.wraplength,
            padx=6,
            pady=4,
        )
        label.pack(ipadx=1)

    def _hide(self):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None

def parse_comma_separated(names_str):
    """Parses a comma-separated string into a list of stripped names."""
    if not names_str:
        return []
    return [s.strip() for s in names_str.split(',') if s.strip()]


def validate_names(names, expected_count, description):
    """Validates the number of names provided and checks for uniqueness."""
    if not names:
        raise ValueError(f"No valid {description} names provided")
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate {description} names found: {names}")
    if expected_count is not None and len(names) != expected_count:
        raise ValueError(f"The number of {description} names provided ({len(names)}) does not match the expected count ({expected_count}).")
    return names


def build_id_label_map(ids, names, *, fallback_prefix="ID"):
    """Build a best-effort mapping from integer IDs to labels.

    Supported patterns:
      - 0-indexed names list: name = names[id]
      - 1-indexed IDs (common user mistake): ids == 1..len(names) => name = names[id-1]
      - Contiguous shifted IDs: ids == k..k+len(names)-1 => name = names[i] in order
    """
    try:
        ids_sorted = sorted({int(v) for v in ids if v is not None})
    except Exception:
        ids_sorted = []
    if not ids_sorted:
        return {}, "empty"

    if not names:
        return {cid: f"{fallback_prefix} {cid}" for cid in ids_sorted}, "no_names"

    names_list = list(names)

    def _label_from_index(idx):
        if 0 <= idx < len(names_list):
            try:
                label = str(names_list[idx]).strip()
            except Exception:
                label = ""
            if label:
                return label
        return f"{fallback_prefix} {idx}"

    min_id = ids_sorted[0]
    max_id = ids_sorted[-1]

    if min_id >= 0 and max_id < len(names_list):
        return {cid: _label_from_index(cid) for cid in ids_sorted}, "direct_index"

    if ids_sorted == list(range(1, len(names_list) + 1)):
        return {cid: _label_from_index(cid - 1) for cid in ids_sorted}, "one_indexed"

    if len(ids_sorted) == len(names_list) and ids_sorted == list(range(min_id, min_id + len(names_list))):
        mapping = {}
        for offset, cid in enumerate(ids_sorted):
            try:
                label = str(names_list[offset]).strip()
            except Exception:
                label = ""
            mapping[cid] = label or f"{fallback_prefix} {cid}"
        return mapping, "contiguous_shift"

    return {cid: _label_from_index(cid) if 0 <= cid < len(names_list) else f"{fallback_prefix} {cid}" for cid in ids_sorted}, "partial"


def validate_roi_definitions(roi_definitions):
    """Validates ROI definitions using JSON schema."""
    schema = {
        "type": "object",
        "patternProperties": {
            "^[a-zA-Z0-9_]+$": {  # ROI name: alphanumeric + underscore
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4
            }
        },
        "additionalProperties": False
    }
    
    validator = Draft7Validator(schema)
    errors = list(validator.iter_errors(roi_definitions))
    
    if errors:
        error_msgs = []
        for error in errors:
            error_msgs.append(f"ROI validation error: {error.message} at path {list(error.path)}")
        raise ValidationError("\n".join(error_msgs))
    
    for name, rect in roi_definitions.items():
        x, y, w, h = rect
        if w <= 0 or h <= 0:
            raise ValidationError(f"ROI '{name}' has non-positive width ({w}) or height ({h})")
        if x < 0 or y < 0:
            logger.warning(f"ROI '{name}' has negative position (x={x}, y={y}). This may be valid but check coordinates.")


def compute_feature_vectors(detections_df, keypoint_names, conf_threshold,
                            normalization_ref_points=('left_shoulder', 'right_shoulder'),
                            use_bbox=False, location_mode='none', location_grid_size=50, 
                            roi_definitions=None, social_mode=False):
    """
    Computes pose feature vectors with group-aware normalization, advanced location-awareness,
    and time-series augmentation. Now with flexible normalization reference points.
    """
    logger.info(f"Computing feature vectors with location_mode='{location_mode}', social_mode={social_mode}.")
    N_keypoints = len(keypoint_names)
    LOW_CONF_THRESHOLD = 0.01
    EPSILON = 1e-6

    conf_threshold = float(conf_threshold)
    diagnostics = {
        'frames_processed': len(detections_df),
        'low_confidence_frames': 0,
        'bbox_clamped': 0,
        'nonfinite_repaired': 0,
        'warnings': [],
        'warning_overflow': False,
        'requested_confidence_threshold': conf_threshold,
        'applied_confidence_threshold': None,
        'threshold_lowered': False
    }
    MAX_WARNINGS = 30

    def record_warning(message):
        if len(diagnostics['warnings']) < MAX_WARNINGS:
            diagnostics['warnings'].append(message)
        else:
            diagnostics['warning_overflow'] = True

    if len(normalization_ref_points) != 2:
        raise ValueError("Normalization reference points must be a pair of two keypoint names.")
    if not all(kp in keypoint_names for kp in normalization_ref_points):
        raise ValueError(f"Normalization reference points {normalization_ref_points} not found in keypoint names.")

    if location_mode == 'roi' and roi_definitions:
        try:
            validate_roi_definitions(roi_definitions)
        except ValidationError as e:
            logger.error(f"ROI validation failed: {e}")
            raise

    if conf_threshold > LOW_CONF_THRESHOLD:
        logger.info(f"Lowering confidence threshold from {conf_threshold:.2f} to {LOW_CONF_THRESHOLD:.2f} to retain marginal keypoints for VAE stability.")
    confidence_cutoff = LOW_CONF_THRESHOLD
    diagnostics['applied_confidence_threshold'] = confidence_cutoff
    diagnostics['threshold_lowered'] = conf_threshold > confidence_cutoff

    group_ref_distances = {}
    for group_name, group_df in detections_df.groupby('group'):
        logger.info(f"Calculating reference distances for group: {group_name}")
        pair_distances = defaultdict(list)
        for _, det in group_df.iterrows():
            keypoints = det['keypoints']
            for i in range(N_keypoints):
                for j in range(i + 1, N_keypoints):
                    kp1_name, kp2_name = keypoint_names[i], keypoint_names[j]
                    if keypoints[kp1_name][2] >= confidence_cutoff and keypoints[kp2_name][2] >= confidence_cutoff:
                        dist = np.linalg.norm(np.array(keypoints[kp1_name][:2]) - np.array(keypoints[kp2_name][:2]))
                        pair_distances[(kp1_name, kp2_name)].append(dist)

        median_distances = {pair: np.median(dists) for pair, dists in pair_distances.items() if dists}

        ref_pair = tuple(sorted(normalization_ref_points))
        body_ref_dist = median_distances.get(ref_pair)

        if body_ref_dist is None or body_ref_dist <= EPSILON:
            logger.warning(f"Reference key pair {normalization_ref_points} not found or invalid for group '{group_name}'.")
            if median_distances:
                body_ref_dist = np.median(list(median_distances.values()))
                logger.warning(f"Falling back to median of all keypoint pairs for normalization: {body_ref_dist:.2f}")
            else:
                raise ValueError(f"Could not determine any valid reference distance for group '{group_name}'. "
                                 "Check keypoint data and confidence threshold. Cannot proceed with feature computation.")

        body_ref_dist = max(body_ref_dist, EPSILON)
        logger.info(f"Group '{group_name}' body reference distance for normalization: {body_ref_dist:.2f}")
        group_ref_distances[group_name] = {'body_ref': body_ref_dist, 'medians': median_distances}

    if location_mode == 'roi' and roi_definitions:
        roi_names = sorted(roi_definitions.keys())
        num_rois = len(roi_names)
        logger.info(f"Using {num_rois} user-defined ROIs: {', '.join(roi_names)}")
    else:
        roi_names = []
        num_rois = 0

    inter_animal_features = {}
    if social_mode:
        logger.info("Social mode enabled: Computing inter-animal interaction features.")
        grouped = detections_df.groupby(['group', 'directory', 'frame'])
        for (_group, _directory, frame), frame_df in grouped:
            if len(frame_df) > 1:
                centroids = []
                for _, det in frame_df.iterrows():
                    confident_kps = np.array([det['keypoints'][kp][:2] for kp in keypoint_names if det['keypoints'][kp][2] >= confidence_cutoff])
                    centroid = np.mean(confident_kps, axis=0) if len(confident_kps) > 0 else np.array([0.0, 0.0])
                    centroids.append(centroid)

                if len(centroids) > 1:
                    dist_matrix = squareform(pdist(centroids, 'euclidean'))
                else:
                    dist_matrix = np.zeros((len(frame_df), len(frame_df)))

                mean_dists = np.mean(dist_matrix, axis=1)
                min_dists = np.min(dist_matrix + np.eye(len(dist_matrix)) * np.inf, axis=1)

                for idx, row_idx in enumerate(frame_df.index):
                    inter_animal_features[row_idx] = {
                        'mean_inter_dist': mean_dists[idx],
                        'min_inter_dist': min_dists[idx],
                    }
            else:
                for row_idx in frame_df.index:
                    inter_animal_features[row_idx] = {'mean_inter_dist': 0.0, 'min_inter_dist': 0.0}

    detections_df = detections_df.sort_values(['group', 'directory', 'track_id', 'frame'])
    detections_df['prev_keypoints'] = detections_df.groupby(['group', 'directory', 'track_id'])['keypoints'].shift(1)
    detections_df['prev_prev_keypoints'] = detections_df.groupby(['group', 'directory', 'track_id'])['keypoints'].shift(2)

    feature_vectors = []
    for idx, det in detections_df.iterrows():
        keypoints = det['keypoints']
        prev_keypoints = det['prev_keypoints'] if pd.notnull(det['prev_keypoints']) else keypoints
        prev_prev_keypoints = det['prev_prev_keypoints'] if pd.notnull(det['prev_prev_keypoints']) else prev_keypoints

        group = det['group']
        body_ref_dist = group_ref_distances.get(group, {}).get('body_ref', 1.0)
        median_distances = group_ref_distances.get(group, {}).get('medians', {})
        body_ref_safe = max(body_ref_dist, EPSILON)

        vec = []
        confidences = [keypoints[kp][2] for kp in keypoint_names]
        if not any(conf >= confidence_cutoff for conf in confidences):
            diagnostics['low_confidence_frames'] += 1
            warning_msg = f"Frame {det['frame']} (track {det['track_id']}, group {group}) has no keypoints above {confidence_cutoff:.2f}. Using fallback statistics."
            logger.warning(warning_msg)
            record_warning(warning_msg)

        for i in range(N_keypoints):
            for j in range(i + 1, N_keypoints):
                kp1, kp2 = keypoint_names[i], keypoint_names[j]
                if keypoints[kp1][2] >= confidence_cutoff and keypoints[kp2][2] >= confidence_cutoff:
                    dist = np.linalg.norm(np.array(keypoints[kp1][:2]) - np.array(keypoints[kp2][:2]))
                else:
                    dist = median_distances.get(tuple(sorted((kp1, kp2))), 0.0)
                vec.append(dist)

        for i in range(N_keypoints):
            for j in range(i + 1, N_keypoints):
                for k in range(j + 1, N_keypoints):
                    kp1, kp2, kp3 = keypoint_names[i], keypoint_names[j], keypoint_names[k]
                    if (keypoints[kp1][2] >= confidence_cutoff and keypoints[kp2][2] >= confidence_cutoff and keypoints[kp3][2] >= confidence_cutoff):
                        v1 = np.array(keypoints[kp1][:2]) - np.array(keypoints[kp2][:2])
                        v2 = np.array(keypoints[kp3][:2]) - np.array(keypoints[kp2][:2])
                        norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
                        if norm_v1 > 0 and norm_v2 > 0:
                            cos_angle = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
                            angle = np.arccos(cos_angle) / np.pi
                        else:
                            angle = 0.0
                    else:
                        angle = 0.0
                    vec.append(angle)

        confident_kps = np.array([keypoints[kp][:2] for kp in keypoint_names if keypoints[kp][2] >= confidence_cutoff])
        centroid = np.mean(confident_kps, axis=0) if len(confident_kps) > 0 else np.array([0.0, 0.0])

        if det['bbox'] and use_bbox:
            x_c, y_c, w, h = det['bbox']
            clamp_needed = (w is not None and w <= EPSILON) or (h is not None and h <= EPSILON)
            if clamp_needed:
                diagnostics['bbox_clamped'] += 1
                record_warning(f"Frame {det['frame']} (track {det['track_id']}): bounding box dimensions near zero; normalization clamped.")
            width = max(w, EPSILON)
            height = max(h, EPSILON)
            for kp in keypoint_names:
                if keypoints[kp][2] >= confidence_cutoff:
                    norm_x = (keypoints[kp][0] - (x_c - width / 2)) / width
                    norm_y = (keypoints[kp][1] - (y_c - height / 2)) / height
                else:
                    norm_x, norm_y = 0.5, 0.5
                vec.extend([norm_x, norm_y])
        else:
            for kp in keypoint_names:
                if keypoints[kp][2] >= confidence_cutoff:
                    norm_x = (keypoints[kp][0] - centroid[0]) / body_ref_safe
                    norm_y = (keypoints[kp][1] - centroid[1]) / body_ref_safe
                else:
                    norm_x, norm_y = 0.0, 0.0
                vec.extend([norm_x, norm_y])

        bbox_center_x, bbox_center_y = (det['bbox'][0], det['bbox'][1]) if det['bbox'] is not None else (centroid[0], centroid[1])

        if location_mode == 'roi' and roi_names:
            roi_vector = [0] * (num_rois + 1)
            found_in_roi = False
            for i, roi_name in enumerate(roi_names):
                x, y, w, h = roi_definitions[roi_name]
                if (x <= bbox_center_x < x + w) and (y <= bbox_center_y < y + h):
                    roi_vector[i] = 1
                    found_in_roi = True
                    break
            if not found_in_roi:
                roi_vector[num_rois] = 1
            vec.extend(roi_vector)
        elif location_mode == 'unsupervised':
            norm_grid_x = bbox_center_x / det['video_width'] if det['video_width'] else 0.0
            norm_grid_y = bbox_center_y / det['video_height'] if det['video_height'] else 0.0
            vec.extend([norm_grid_x, norm_grid_y])

        if social_mode:
            inter_feats = inter_animal_features.get(idx, {'mean_inter_dist': 0.0, 'min_inter_dist': 0.0})
            vec.append(inter_feats['mean_inter_dist'] / body_ref_safe)
            vec.append(inter_feats['min_inter_dist'] / body_ref_safe)

        for kp in keypoint_names:
            curr_pos = np.array(keypoints[kp][:2]) if keypoints[kp][2] >= confidence_cutoff else np.array([0.0, 0.0])
            prev_valid = isinstance(prev_keypoints, dict) and prev_keypoints.get(kp, [0, 0, 0])[2] >= confidence_cutoff
            prev_prev_valid = isinstance(prev_prev_keypoints, dict) and prev_prev_keypoints.get(kp, [0, 0, 0])[2] >= confidence_cutoff
            prev_pos = np.array(prev_keypoints[kp][:2]) if prev_valid else curr_pos
            prev_prev_pos = np.array(prev_prev_keypoints[kp][:2]) if prev_prev_valid else prev_pos

            velocity = curr_pos - prev_pos
            acceleration = velocity - (prev_pos - prev_prev_pos)

            norm_vel = np.linalg.norm(velocity) / body_ref_safe
            norm_acc = np.linalg.norm(acceleration) / body_ref_safe

            vec.append(norm_vel)
            vec.append(norm_acc)

        vec_array = np.asarray(vec, dtype=np.float32)
        if not np.all(np.isfinite(vec_array)):
            diagnostics['nonfinite_repaired'] += 1
            warning_msg = f"Non-finite values detected in feature vector for frame {det['frame']} (track {det['track_id']}). Replacing with safe defaults."
            logger.warning(warning_msg)
            record_warning(warning_msg)
            vec_array = np.nan_to_num(vec_array, nan=0.0, posinf=0.0, neginf=0.0)

        feature_vectors.append(vec_array.tolist())

    detections_df['feature_vector'] = feature_vectors
    detections_df.drop(columns=['prev_keypoints', 'prev_prev_keypoints'], inplace=True, errors='ignore')
    return detections_df, diagnostics


def export_detections_to_csv(detections_df, keypoint_names, output_folder):
    """Exports detections to a wide-format CSV for each video source."""
    logger.info("Exporting keypoint data to wide-format CSVs...")
    if detections_df.empty:
        logger.warning("Detections DataFrame is empty. Nothing to export.")
        return

    header = ['track_id', 'frame']
    for kp_name in keypoint_names:
        header.extend([f'{kp_name}_x', f'{kp_name}_y', f'{kp_name}_conf'])

    for source_dir, group_df in detections_df.groupby('directory'):
        if 'video_source' in group_df.columns and not group_df.empty:
            video_source_name = str(group_df['video_source'].iloc[0])
        else:
            video_source_name = os.path.basename(source_dir)
        video_source_name = video_source_name or os.path.basename(source_dir) or "source"
        logger.info(f"Processing export for: {video_source_name}")

        output_rows = []
        for _, row in group_df.sort_values(by=['track_id', 'frame']).iterrows():
            csv_row = {'track_id': row['track_id'], 'frame': row['frame']}
            for kp_name in keypoint_names:
                x, y, conf = row['keypoints'].get(kp_name, (np.nan, np.nan, np.nan))
                csv_row[f'{kp_name}_x'] = x
                csv_row[f'{kp_name}_y'] = y
                csv_row[f'{kp_name}_conf'] = conf
            output_rows.append(csv_row)

        if not output_rows:
            logger.warning(f"No data to write for {video_source_name}")
            continue

        output_df = pd.DataFrame(output_rows, columns=header)
        output_path = os.path.join(output_folder, f"{video_source_name}_aggregated_keypoints.csv")
        output_df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved aggregated keypoints to {output_path}")


class BehaviorAnalysisApp:
    def __init__(self, root, *, embed: bool = False, integra_app=None):
        self.root = root
        self.embed = embed
        self.integra_app = integra_app
        if not self.embed:
            self.root.title("Behavior Analysis Toolkit")
            apply_adaptive_window_geometry(
                self.root,
                preferred_size=(1100, 850),
                min_size=(900, 680),
                width_ratio=0.94,
                height_ratio=0.92,
            )
        
        self.groups, self.running = {}, False
        self.status_var, self.progress_var = tk.StringVar(value="Ready"), tk.DoubleVar(value=0)
        self.output_folder = tk.StringVar()
        self.keypoints_entry_var = tk.StringVar(value="nose,left_eye,right_eye,left_ear,right_ear,left_shoulder,right_shoulder")
        self.behaviors_entry_var = tk.StringVar()
        self.use_bbox_var = tk.BooleanVar(value=True)
        self.location_mode = tk.StringVar(value="none")
        self.location_grid_size = tk.StringVar(value="50")
        self.generate_location_videos = tk.BooleanVar(value=False)
        self.max_frame_gap = tk.StringVar(value="15")
        self.min_bout_duration = tk.StringVar(value="3")
        self.conf_threshold = tk.StringVar(value="0.3")
        self.umap_neighbors = tk.StringVar(value="15")
        self.umap_components = tk.StringVar(value="5")
        self.min_cluster_size = tk.StringVar(value="10")
        self.social_mode = tk.BooleanVar(value=False)
        # ADP-4 Commit E — opt-in stability audit. When True, the
        # sub-behavior runner re-clusters with N seeds and reports
        # per-class mean ARI alongside the primary results. Off by
        # default so the first run is fast.
        self.run_stability_audit = tk.BooleanVar(value=False)
        self.stability_n_seeds = tk.StringVar(value="3")
        # Session presets for one-click parameter tuning.
        self.preset_var = tk.StringVar(value="Custom")
        self.session_presets = {
            "Custom": {},
            "Quick Preview": {
                'umap_neighbors': '12',
                'umap_components': '3',
                'min_cluster_size': '8',
                'min_bout_duration': '3',
            },
            "Default": {
                'umap_neighbors': '15',
                'umap_components': '5',
                'min_cluster_size': '10',
                'min_bout_duration': '3',
            },
            "High Fidelity": {
                'umap_neighbors': '30',
                'umap_components': '5',
                'min_cluster_size': '20',
                'min_bout_duration': '5',
            },
        }

        self.skeleton_connections = []  # List of tuples (start_kp, end_kp)
        self.normalization_left = tk.StringVar(value="left_shoulder")
        self.normalization_right = tk.StringVar(value="right_shoulder")

        self.feature_diagnostics = {}
        self.state_summary_payload = []
        self.behavior_names = []

        self.create_widgets()
        self._apply_integra_defaults()
        self.update_keypoint_combos()
        logger.info("Application initialized successfully.")

    def _apply_integra_defaults(self) -> None:
        """When embedded in IntegraPose, auto-fill keypoints/behaviors to avoid manual copy/paste errors."""
        app = getattr(self, "integra_app", None)
        if app is None:
            return

        try:
            config = getattr(app, "config", None)
        except Exception:
            config = None
        if config is None:
            return

        try:
            kp_str = config.get_setting("setup.keypoint_names_str")
            if isinstance(kp_str, str) and kp_str.strip():
                self.keypoints_entry_var.set(kp_str.strip())
        except Exception:
            pass

        try:
            behaviors = config.get_setting("setup.behaviors_list")
        except Exception:
            behaviors = None
        if isinstance(behaviors, list) and behaviors:
            try:
                ordered = sorted(
                    (b for b in behaviors if isinstance(b, dict) and b.get("name")),
                    key=lambda item: int(item.get("id", 0)),
                )
            except Exception:
                ordered = [b for b in behaviors if isinstance(b, dict) and b.get("name")]
            names = [str(b.get("name")).strip() for b in ordered if str(b.get("name")).strip()]
            if names:
                self.behaviors_entry_var.set(",".join(names))

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)
        if self.embed:
            toolbar = ttk.Frame(main_frame)
            toolbar.pack(fill="x", pady=(0, 8))
            ttk.Button(toolbar, text="Save Project", command=self.save_project).pack(side="left", padx=(0, 6))
            ttk.Button(toolbar, text="Load Project", command=self.load_project).pack(side="left", padx=6)
            ttk.Button(toolbar, text="Load Multiple", command=self.load_multiple_projects).pack(side="left", padx=6)
            ttk.Button(toolbar, text="Close", command=self._close_embed).pack(side="left", padx=(12, 0))
        else:
            menubar = tk.Menu(self.root); self.root.config(menu=menubar)
            file_menu = tk.Menu(menubar, tearoff=0); menubar.add_cascade(label="File", menu=file_menu)
            file_menu.add_command(label="Save Project...", command=self.save_project)
            file_menu.add_command(label="Load Project...", command=self.load_project)
            file_menu.add_command(label="Load Multiple Projects...", command=self.load_multiple_projects)
            file_menu.add_separator(); file_menu.add_command(label="Exit", command=self.root.quit)
        notebook = ttk.Notebook(main_frame); notebook.pack(fill=tk.BOTH, expand=True)
        tab1 = ttk.Frame(notebook, padding="10")
        tab2 = ttk.Frame(notebook, padding="10")
        tab3 = ttk.Frame(notebook, padding="10")
        tab4 = ttk.Frame(notebook, padding="10")
        tab5 = ttk.Frame(notebook, padding="10")
        notebook.add(tab1, text="1. Setup & Input")
        notebook.add(tab2, text="2. Analysis Parameters")
        notebook.add(tab3, text="3. Execute & Visualize")
        notebook.add(tab4, text="4. Export Data")
        notebook.add(tab5, text="Help & Guidance")
        self.create_tab1_setup(tab1)
        self.create_tab2_params(tab2)
        self.create_tab3_execute(tab3)
        self.create_tab4_export(tab4)
        self.create_tab5_help(tab5)

    def _ui_call(self, callback, *args, **kwargs):
        if threading.current_thread() is threading.main_thread():
            return callback(*args, **kwargs)
        self.root.after(0, lambda: callback(*args, **kwargs))
        return None

    def _ui_call_sync(self, callback, *args, **kwargs):
        if threading.current_thread() is threading.main_thread():
            return callback(*args, **kwargs)

        done = threading.Event()
        result = {}

        def _runner():
            try:
                result["value"] = callback(*args, **kwargs)
            except Exception as exc:
                result["error"] = exc
            finally:
                done.set()

        self.root.after(0, _runner)
        done.wait()
        if "error" in result:
            raise result["error"]
        return result.get("value")

    def _set_status_progress(self, *, current=None, total=None, message=None, progress=None):
        if progress is None and current is not None and total is not None and total > 0:
            progress = (current / total) * 100
        if progress is not None:
            self.progress_var.set(progress)
        if message is not None:
            self.status_var.set(message)

    def _set_status_progress_async(self, *, current=None, total=None, message=None, progress=None):
        self._ui_call(
            self._set_status_progress,
            current=current,
            total=total,
            message=message,
            progress=progress,
        )

    def _reset_run_state_async(self):
        self._ui_call(self._set_status_progress, message="Ready", progress=0)

    def _show_info_async(self, title, message):
        self._ui_call(messagebox.showinfo, title, message)

    def _show_warning_async(self, title, message):
        self._ui_call(messagebox.showwarning, title, message)

    def _show_error_async(self, title, message):
        self._ui_call(messagebox.showerror, title, message)

    def _refresh_plot_async(self):
        self._ui_call(self.update_plot)

    def _refresh_diagnostics_async(self):
        self._ui_call(self.update_diagnostics_panel)

    def _refresh_state_summary_async(self):
        self._ui_call(self.update_state_summary)

    def _read_runtime_detections(self, group_dirs, keypoint_names_str, behavior_names_str, use_bbox_hint=True, *, video_path_map=None, subject_id_map=None):
        return helper_read_detections(
            group_dirs,
            keypoint_names_str,
            behavior_names_str,
            use_bbox_hint,
            video_path_map=video_path_map,
            subject_id_map=subject_id_map,
        )

    def _capture_analysis_params(self):
        return self.get_all_params()

    def _capture_export_inputs(self):
        return {
            "keypoints_entry_var": self.keypoints_entry_var.get(),
            "behaviors_entry_var": self.behaviors_entry_var.get(),
            "use_bbox_var": self.use_bbox_var.get(),
            "output_folder": self.output_folder.get(),
        }

    def _capture_suggestion_inputs(self):
        return {
            "keypoints_entry_var": self.keypoints_entry_var.get(),
            "behaviors_entry_var": self.behaviors_entry_var.get(),
            "use_bbox_var": self.use_bbox_var.get(),
        }

    def _close_embed(self):
        try:
            self.root.destroy()
        except Exception:
            pass

    def create_tab1_setup(self, parent_tab):
        parent_tab.columnconfigure(0, weight=1); parent_tab.rowconfigure(3, weight=1)
        
        input_frame = ttk.LabelFrame(parent_tab, text="Input Configuration", padding="10"); 
        input_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)
        
        ttk.Label(input_frame, text="Keypoint Names:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(input_frame, textvariable=self.keypoints_entry_var).grid(row=0, column=1, sticky="ew")
        self.keypoints_entry_var.trace_add("write", self.update_keypoint_combos)
        
        ttk.Label(input_frame, text="Behavior Names:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(input_frame, textvariable=self.behaviors_entry_var).grid(row=1, column=1, sticky="ew")
        
        ttk.Checkbutton(input_frame, text="YOLO files include bounding box data", variable=self.use_bbox_var).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        ttk.Label(input_frame, text="Normalization Ref Points:").grid(row=3, column=0, sticky=tk.W, pady=2)
        norm_frame = ttk.Frame(input_frame)
        norm_frame.grid(row=3, column=1, sticky="ew")
        self.norm_left_combo = ttk.Combobox(norm_frame, textvariable=self.normalization_left, state="readonly", width=15)
        self.norm_left_combo.pack(side=tk.LEFT, expand=True, fill='x', padx=(0,5))
        ttk.Label(norm_frame, text="to").pack(side=tk.LEFT, padx=5)
        self.norm_right_combo = ttk.Combobox(norm_frame, textvariable=self.normalization_right, state="readonly", width=15)
        self.norm_right_combo.pack(side=tk.LEFT, expand=True, fill='x', padx=(5,0))

        skeleton_frame = ttk.LabelFrame(parent_tab, text="Skeleton Connections", padding="10")
        skeleton_frame.grid(row=1, column=0, sticky="nsew", pady=10)
        skeleton_frame.columnconfigure(0, weight=1); skeleton_frame.rowconfigure(0, weight=1)
        
        self.skeleton_list = tk.Listbox(skeleton_frame, height=5)
        self.skeleton_list.grid(row=0, column=0, sticky="nsew")
        
        add_frame = ttk.Frame(skeleton_frame)
        add_frame.grid(row=1, column=0, sticky="ew", pady=5)
        add_frame.columnconfigure(0, weight=1); add_frame.columnconfigure(2, weight=1)
        self.start_combo = ttk.Combobox(add_frame, state="readonly")
        self.start_combo.grid(row=0, column=0, sticky='ew')
        ttk.Label(add_frame, text=" to ").grid(row=0, column=1, padx=5)
        self.end_combo = ttk.Combobox(add_frame, state="readonly")
        self.end_combo.grid(row=0, column=2, sticky='ew')
        ttk.Button(add_frame, text="Add", command=self.add_skeleton_connection).grid(row=0, column=3, padx=5)
        
        ttk.Button(skeleton_frame, text="Remove Selected", command=self.remove_skeleton_connection).grid(row=2, column=0, sticky="ew")
        
        group_frame = ttk.LabelFrame(parent_tab, text="Data Groups (Add Control/Baseline Group First)", padding="10")
        group_frame.grid(row=3, column=0, sticky="nsew")
        group_frame.columnconfigure(0, weight=1); group_frame.rowconfigure(0, weight=1)
        
        self.group_tree = ttk.Treeview(group_frame, columns=("Type", "Path"), show="headings", height=8)
        self.group_tree.heading("Type", text="Data Type"); self.group_tree.heading("Path", text="Path")
        self.group_tree.column("Type", width=120, anchor='w'); self.group_tree.column("Path", width=300, anchor='w')
        self.group_tree.grid(row=0, column=0, sticky="nsew")
        
        tree_scrollbar = ttk.Scrollbar(group_frame, orient="vertical", command=self.group_tree.yview)
        self.group_tree.configure(yscrollcommand=tree_scrollbar.set); tree_scrollbar.grid(row=0, column=1, sticky='ns')
        
        btn_frame = ttk.Frame(group_frame)
        btn_frame.grid(row=0, column=2, padx=10, sticky='ns')
        ttk.Button(btn_frame, text="Add Group", command=self.add_group).pack(fill='x', pady=2)
        ttk.Button(btn_frame, text="Add Data Source", command=self.add_data_source).pack(fill='x', pady=2)
        ttk.Button(btn_frame, text="Remove Selected", command=self.remove_selected).pack(fill='x', pady=2)
        ttk.Separator(btn_frame, orient="horizontal").pack(fill='x', pady=(10, 8))
        import_btn = ttk.Button(btn_frame, text="Import Analytics Run(s)...", command=self.import_analytics_runs)
        import_btn.pack(fill='x', pady=2)
        Tooltip(
            import_btn,
            "Load one or more analytics run manifests (run_manifest.json) from Tab 6 or Batch Analytics.\n"
            "This auto-adds YOLO folders + videos into groups for modeling in Tab 7.",
        )
    
    def update_keypoint_combos(self, *args):
        keypoints = parse_comma_separated(self.keypoints_entry_var.get())
        keypoints = sorted(list(set(keypoints))) # Ensure unique and sorted
        
        self.norm_left_combo['values'] = keypoints
        self.norm_right_combo['values'] = keypoints
        
        self.start_combo['values'] = keypoints
        self.end_combo['values'] = keypoints
        
        if keypoints:
            if self.normalization_left.get() not in keypoints:
                self.normalization_left.set(keypoints[0] if keypoints else '')
            if self.normalization_right.get() not in keypoints:
                self.normalization_right.set(keypoints[1] if len(keypoints) > 1 else keypoints[0] if keypoints else '')
            if not self.start_combo.get() and keypoints: self.start_combo.set(keypoints[0])
            if not self.end_combo.get() and len(keypoints) > 1: self.end_combo.set(keypoints[1])
        else: # Clear if no keypoints
            self.normalization_left.set('')
            self.normalization_right.set('')
            self.start_combo.set('')
            self.end_combo.set('')

    def add_skeleton_connection(self):
        start = self.start_combo.get()
        end = self.end_combo.get()
        if start and end and start != end:
            conn = tuple(sorted((start, end)))
            if conn not in self.skeleton_connections:
                self.skeleton_connections.append(conn)
                self.skeleton_list.insert(tk.END, f"{conn[0]} -- {conn[1]}")

    def remove_skeleton_connection(self):
        selections = self.skeleton_list.curselection()
        if not selections: return
        
        for sel_idx in reversed(selections):
            del self.skeleton_connections[sel_idx]
            self.skeleton_list.delete(sel_idx)

    def create_tab2_params(self, parent_tab):
        parent_tab.columnconfigure(0, weight=1); parent_tab.columnconfigure(1, weight=1)
        left_frame, right_frame = ttk.Frame(parent_tab), ttk.Frame(parent_tab)
        left_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        right_frame.grid(row=0, column=1, sticky='nsew', padx=(5, 0))
        right_frame.rowconfigure(1, weight=1)

        gen_frame = ttk.LabelFrame(left_frame, text="General Parameters", padding="10")
        gen_frame.pack(fill='x', expand=False, pady=(0, 10))
        ttk.Label(gen_frame, text="Confidence Threshold:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(gen_frame, textvariable=self.conf_threshold, width=10).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(gen_frame, text="Max Frame Gap:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(gen_frame, textvariable=self.max_frame_gap, width=10).grid(row=1, column=1, sticky=tk.W)
        ttk.Checkbutton(gen_frame, text="Enable Multi-Animal Interactions", variable=self.social_mode).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        ttk.Label(gen_frame, text="Session Preset:").grid(row=3, column=0, sticky=tk.W, pady=2)
        preset_combo = ttk.Combobox(gen_frame, textvariable=self.preset_var, values=list(self.session_presets.keys()), state="readonly")
        preset_combo.grid(row=3, column=1, sticky=tk.W)
        ttk.Button(gen_frame, text="Apply", command=self.apply_session_preset).grid(row=3, column=2, padx=(5, 0), sticky=tk.W)

        # ADP-4 Commit E — opt-in stability audit row.
        # When checked, the Sub-Behavior Discovery run re-clusters with
        # N seeds and reports per-class mean ARI alongside the primary
        # results. Off by default — costs N× the primary runtime.
        stability_cb = ttk.Checkbutton(
            gen_frame,
            text="Run stability audit (re-cluster with multiple seeds; slower)",
            variable=self.run_stability_audit,
        )
        stability_cb.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(8, 2))
        Tooltip(
            stability_cb,
            "Re-runs the per-class clustering with N additional random "
            "seeds and reports the mean Adjusted Rand Index across runs. "
            "This roughly multiplies the discovery runtime by N+1 (default "
            "N=3 → ~4x). Use for a final 'is this real?' check, not for "
            "every iteration. Mean ARI ≥ 0.5 = STABLE; < 0.5 = the partition "
            "is sensitive to seed and the cluster boundaries should be "
            "treated as soft.",
        )
        seeds_label = ttk.Label(gen_frame, text="  Stability seeds (N):")
        seeds_label.grid(row=5, column=0, sticky=tk.W, pady=2)
        seeds_entry = ttk.Entry(gen_frame, textvariable=self.stability_n_seeds, width=6)
        seeds_entry.grid(row=5, column=1, sticky=tk.W)
        Tooltip(
            seeds_entry,
            "Number of additional clusterings used for the audit. 3 is a "
            "reasonable default; 5+ is more confident but linearly more "
            "expensive.",
        )

        loc_frame = ttk.LabelFrame(left_frame, text="Location Analysis & Validation", padding="10")
        loc_frame.pack(fill='both', expand=True)
        ttk.Label(loc_frame, text="Tip: For different arenas, use separate project files.", font=CAPTION_FONT, foreground=MUTED_FG).pack(anchor='w', pady=(0, 5))
        ttk.Label(loc_frame, text="Mode:").pack(anchor='w')
        ttk.Radiobutton(loc_frame, text="None (Pose Only)", variable=self.location_mode, value="none").pack(anchor='w', padx=(10, 0))
        unsupervised_frame = ttk.Frame(loc_frame); unsupervised_frame.pack(fill='x', expand=True, padx=(10, 0))
        ttk.Radiobutton(unsupervised_frame, text="Unsupervised Grid", variable=self.location_mode, value="unsupervised").grid(row=0, column=0, sticky='w')
        ttk.Label(unsupervised_frame, text="Grid Size (px):").grid(row=0, column=1, sticky='w', padx=(10, 2))
        ttk.Entry(unsupervised_frame, textvariable=self.location_grid_size, width=8).grid(row=0, column=2, sticky='w')
        roi_frame = ttk.Frame(loc_frame); roi_frame.pack(fill='x', expand=True, pady=(5, 0), padx=(10, 0))
        ttk.Radiobutton(roi_frame, text="User-Defined ROIs", variable=self.location_mode, value="roi").grid(row=0, column=0, sticky='nw')
        self.roi_text = scrolledtext.ScrolledText(roi_frame, height=5, width=40, wrap=tk.WORD)
        self.roi_text.grid(row=1, column=0, columnspan=3, pady=(5, 0), padx=(20, 0))
        self.roi_text.insert(tk.END, '{\n  "center": [200, 200, 300, 300],\n  "corner_A": [0, 0, 150, 150]\n}')
        ttk.Label(roi_frame, text='JSON format: {"name": [x, y, w, h]}', font=CAPTION_FONT, foreground=MUTED_FG).grid(row=2, column=0, columnspan=3, padx=(20, 0), sticky='w')

        # Right column — clustering parameters used by Sub-Behavior Discovery.
        cluster_frame = ttk.LabelFrame(right_frame, text="Clustering Parameters", padding="10")
        cluster_frame.pack(fill='x', expand=False, pady=(0, 10))
        cluster_frame.columnconfigure(1, weight=1)

        ttk.Label(cluster_frame, text="Min Bout Duration (frames):").grid(row=0, column=0, sticky=tk.W, pady=2)
        bout_entry = ttk.Entry(cluster_frame, textvariable=self.min_bout_duration, width=10)
        bout_entry.grid(row=0, column=1, sticky=tk.W)
        Tooltip(
            bout_entry,
            "Bouts shorter than this many contiguous frames are dropped from "
            "the sub-behavior bout table. 3 is a reasonable default for 30 fps "
            "video; raise it to suppress flicker.",
        )

        ttk.Label(cluster_frame, text="UMAP Neighbors:").grid(row=1, column=0, sticky=tk.W, pady=2)
        umap_n_entry = ttk.Entry(cluster_frame, textvariable=self.umap_neighbors, width=10)
        umap_n_entry.grid(row=1, column=1, sticky=tk.W)
        Tooltip(
            umap_n_entry,
            "UMAP `n_neighbors`. Larger values preserve global structure; "
            "smaller values emphasize local pose differences. 15 is the UMAP "
            "default and works well for most projects.",
        )

        ttk.Label(cluster_frame, text="UMAP Components:").grid(row=2, column=0, sticky=tk.W, pady=2)
        umap_c_entry = ttk.Entry(cluster_frame, textvariable=self.umap_components, width=10)
        umap_c_entry.grid(row=2, column=1, sticky=tk.W)
        Tooltip(
            umap_c_entry,
            "Number of UMAP dimensions HDBSCAN clusters in. 5 keeps enough "
            "signal for sub-behavior discovery without inflating runtime.",
        )

        ttk.Label(cluster_frame, text="HDBSCAN Min Cluster Size:").grid(row=3, column=0, sticky=tk.W, pady=2)
        min_cluster_entry = ttk.Entry(cluster_frame, textvariable=self.min_cluster_size, width=10)
        min_cluster_entry.grid(row=3, column=1, sticky=tk.W)
        Tooltip(
            min_cluster_entry,
            "Smallest sub-cluster HDBSCAN will return. Frames in groups smaller "
            "than this become noise (label -1). Raise to suppress micro-variation; "
            "lower if you want fine-grained sub-behaviors.",
        )

        ttk.Label(
            cluster_frame,
            text="These parameters drive Sub-Behavior Discovery on Tab 3.",
            font=CAPTION_FONT,
            foreground=MUTED_FG,
            wraplength=320,
        ).grid(row=4, column=0, columnspan=3, sticky='w', pady=(6, 0))

    def create_tab3_execute(self, parent_tab):
        parent_tab.columnconfigure(0, weight=1)
        parent_tab.rowconfigure(2, weight=1)

        exec_frame = ttk.LabelFrame(parent_tab, text="Execution Controls", padding="10")
        exec_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        exec_frame.columnconfigure(1, weight=1)

        ttk.Label(exec_frame, text="Output Folder:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(exec_frame, textvariable=self.output_folder).grid(row=0, column=1, sticky="ew", padx=(0, 5))
        ttk.Button(exec_frame, text="Browse...", command=self.select_output_folder).grid(row=0, column=2)

        # Sub-Behavior Discovery actions, stacked in a dedicated column.
        sub_run_button = ttk.Button(
            exec_frame,
            text="Run Sub-Behavior Discovery",
            command=self.start_sub_behavior_discovery,
            style="Accent.TButton",
        )
        sub_run_button.grid(row=0, column=3, padx=20, pady=(0, 4), sticky='ew')
        self.review_candidates_button = ttk.Button(
            exec_frame,
            text="Review Candidate Sub-Clusters",
            command=self.open_review_candidates_dialog,
            state=tk.DISABLED,
        )
        self.review_candidates_button.grid(row=1, column=3, padx=20, pady=(0, 4), sticky='ew')
        self.naming_button = ttk.Button(
            exec_frame,
            text="Name Sub-Behaviors...",
            command=self.open_cluster_naming_dialog,
            state=tk.DISABLED,
        )
        self.naming_button.grid(row=2, column=3, padx=20, pady=(0, 4), sticky='ew')
        self.clip_export_button = ttk.Button(
            exec_frame,
            text="Export Sub-cluster Clips",
            command=self.export_sub_cluster_clips_action,
            state=tk.DISABLED,
        )
        self.clip_export_button.grid(row=3, column=3, padx=20, sticky='ew')

        status_frame = ttk.Frame(exec_frame)
        status_frame.grid(row=4, column=0, columnspan=4, sticky='ew', pady=(10, 0))
        status_frame.columnconfigure(1, weight=1)
        ttk.Label(status_frame, text="Status:").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(status_frame, textvariable=self.status_var).grid(row=0, column=1, sticky=tk.W)
        ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100).grid(row=1, column=0, columnspan=2, sticky='ew', pady=5)

        diag_frame = ttk.LabelFrame(parent_tab, text="Data Health Summary", padding="10")
        diag_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        diag_frame.columnconfigure(0, weight=1)
        self.diagnostics_tree = ttk.Treeview(diag_frame, columns=("Metric", "Value"), show="headings", height=5)
        self.diagnostics_tree.heading("Metric", text="Metric")
        self.diagnostics_tree.heading("Value", text="Value")
        self.diagnostics_tree.column("Metric", width=220, anchor='w')
        self.diagnostics_tree.column("Value", width=120, anchor='w')
        self.diagnostics_tree.grid(row=0, column=0, sticky="ew")
        diag_scroll = ttk.Scrollbar(diag_frame, orient="vertical", command=self.diagnostics_tree.yview)
        self.diagnostics_tree.configure(yscrollcommand=diag_scroll.set)
        diag_scroll.grid(row=0, column=1, sticky='ns')
        self.warning_button = ttk.Button(diag_frame, text="View Warnings (0)", command=self.show_diagnostics_warnings, state='disabled')
        self.warning_button.grid(row=1, column=0, columnspan=2, pady=(5, 0), sticky='e')
        self.update_diagnostics_panel()

        state_frame = ttk.LabelFrame(parent_tab, text="Sub-Behavior Summary", padding="10")
        state_frame.grid(row=2, column=0, sticky="nsew")
        state_frame.columnconfigure(0, weight=1)
        state_frame.rowconfigure(0, weight=1)
        self.state_summary = scrolledtext.ScrolledText(state_frame, height=12, wrap=tk.WORD, state='disabled')
        self.state_summary.grid(row=0, column=0, sticky='nsew')
        self.update_state_summary()

    def update_diagnostics_panel(self):
        if not hasattr(self, 'diagnostics_tree'):
            return
        for item in self.diagnostics_tree.get_children():
            self.diagnostics_tree.delete(item)

        diagnostics = self.feature_diagnostics or {}
        def fmt_value(val):
            if isinstance(val, float):
                return f"{val:.3f}"
            if isinstance(val, bool):
                return "Yes" if val else "No"
            return val if val is not None else "N/A"

        rows = [
            ("Frames processed", diagnostics.get('frames_processed', 0)),
            ("Low-confidence frames", diagnostics.get('low_confidence_frames', 0)),
            ("Bounding box clamps", diagnostics.get('bbox_clamped', 0)),
            ("Non-finite repairs", diagnostics.get('nonfinite_repaired', 0)),
            ("Requested conf. threshold", diagnostics.get('requested_confidence_threshold')),
            ("Applied conf. cutoff", diagnostics.get('applied_confidence_threshold')),
            ("Threshold lowered", diagnostics.get('threshold_lowered', False)),
        ]
        for metric, value in rows:
            self.diagnostics_tree.insert('', 'end', values=(metric, fmt_value(value)))

        warnings_count = len(diagnostics.get('warnings', []))
        overflow = diagnostics.get('warning_overflow', False)
        if hasattr(self, 'warning_button'):
            label = f"View Warnings ({warnings_count}{'+' if overflow and warnings_count else ''})"
            state = 'normal' if warnings_count else 'disabled'
            self.warning_button.config(text=label, state=state)

    def update_state_summary(self):
        if not hasattr(self, 'state_summary'):
            return
        summary_widget = self.state_summary
        summary_widget.configure(state='normal')
        summary_widget.delete('1.0', tk.END)

        entries = getattr(self, 'state_summary_payload', []) or []
        if not entries:
            summary_widget.insert(
                tk.END,
                "Run Sub-Behavior Discovery to see per-class results here.\n",
            )
            summary_widget.configure(state='disabled')
            return

        # Sub-Behavior Discovery summary — one block per YOLO class.
        for entry in entries:
            cls_name = entry.get('class_name') or f"class_{entry.get('class_id')}"
            summary_widget.insert(tk.END, f"Class: {cls_name}\n")
            if entry.get('skipped'):
                summary_widget.insert(
                    tk.END,
                    f"  Skipped — {entry.get('skip_reason', 'too few frames')}\n\n",
                )
                continue
            summary_widget.insert(
                tk.END,
                f"  Frames: {entry.get('n_frames', 0)}  |  "
                f"Sub-behaviors: {entry.get('n_clusters', 0)}  |  "
                f"Bouts: {entry.get('n_bouts', 0)}  |  "
                f"Noise frames: {entry.get('n_noise_frames', 0)}\n",
            )
            sub_labels = entry.get('sub_cluster_labels', [])
            if sub_labels:
                summary_widget.insert(
                    tk.END,
                    f"  Sub-behavior IDs: {', '.join(str(s) for s in sub_labels)}\n",
                )
            # Stability audit line (only when the audit was run).
            if 'stability_mean_ari' in entry:
                mean_ari = entry.get('stability_mean_ari')
                is_stable = entry.get('stability_is_stable', False)
                n_seeds = entry.get('stability_n_seeds', 0)
                label = "Stable" if is_stable else "Unstable"
                if mean_ari is None:
                    summary_widget.insert(
                        tk.END,
                        f"  Stability: not computed ({n_seeds} seeds, ARI unavailable)\n",
                    )
                else:
                    summary_widget.insert(
                        tk.END,
                        f"  Stability: {label} — mean ARI {mean_ari:.2f} across {n_seeds} seeds\n",
                    )
            summary_widget.insert(tk.END, "\n")

        summary_widget.configure(state='disabled')

    def show_diagnostics_warnings(self):
        warnings = (self.feature_diagnostics or {}).get('warnings', [])
        overflow = (self.feature_diagnostics or {}).get('warning_overflow', False)
        if not warnings:
            messagebox.showinfo("Warnings", "No data quality warnings recorded for this run.")
            return
        top = tk.Toplevel(self.root)
        top.title("Data Health Warnings")
        top.geometry("640x320")
        text_widget = scrolledtext.ScrolledText(top, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True)
        for msg in warnings:
            text_widget.insert(tk.END, msg + "\n")
        if overflow:
            text_widget.insert(tk.END, "\nAdditional warnings omitted. Check the log file for full details.\n")
        text_widget.configure(state='disabled')
        ttk.Button(top, text="Close", command=top.destroy).pack(pady=5)

    def create_tab4_export(self, parent_tab):
        parent_tab.columnconfigure(0, weight=1)
        export_frame = ttk.LabelFrame(parent_tab, text="Export Raw Data", padding="20"); export_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=20); export_frame.columnconfigure(0, weight=1)
        info_label = ttk.Label(export_frame, text="This tool aggregates individual detection .txt files into a single CSV per video. The output is a 'wide' table with columns for track ID, frame, and x, y, confidence for each keypoint.", wraplength=500, justify=tk.LEFT); info_label.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 20))
        ttk.Label(export_frame, text="Export to same Output Folder:").grid(row=1, column=0, sticky=tk.E, padx=5); ttk.Entry(export_frame, textvariable=self.output_folder, state='readonly', width=60).grid(row=1, column=1, sticky='ew'); ttk.Button(export_frame, text="Export Keypoints to CSV", command=self.start_csv_export, style="Accent.TButton").grid(row=2, column=0, columnspan=2, pady=20)
        ttk.Button(export_frame, text="Export Latent Embeddings (CSV)", command=self.export_latent_embeddings).grid(row=3, column=0, columnspan=2, pady=5)

    def create_tab5_help(self, parent_tab):
        parent_tab.columnconfigure(0, weight=1)
        parent_tab.rowconfigure(0, weight=1)

        help_text = (
            "HMM Selector Cheat Sheet\n"
            "• categorical: choose when your observations are discrete pose/behavior IDs. "
            "Best match for frame-wise pose clusters or labeled ethograms.\n"
            "• gaussian: use when feeding continuous feature vectors (e.g., VAE latents or motion features) "
            "and you expect unimodal state emissions.\n"
            "• gmm: like Gaussian but allows multiple mixture components per hidden state—helpful for richer "
            "feature spaces or when behaviors show multi-modal variability.\n"
            "• sticky_gaussian: same as Gaussian but biases the model to stay in the current state using the "
            "Sticky Self-Transition weight. Ideal when behaviors linger and you want smoother sequences.\n\n"
            "Practical Hints\n"
            "• Frame-based (Poses) mode pairs well with categorical HMMs and UMAP/HDBSCAN clustering.\n"
            "• Event-based (Bouts) works with continuous variants when you aggregate bout-level statistics.\n"
            "• Increase mix components only if groups remain broad after initial runs—each extra component "
            "raises training cost.\n"
            "• Sticky weights near 0.7–0.9 encourage longer dwell times; values ≤0.2 behave similar to plain Gaussian.\n\n"
            "VAE Workflow\n"
            "• Baseline group: the first listed group seeds VAE training. Ensure it contains representative data.\n"
            "• Device selection: enable GPU only when CUDA is available; the app automatically falls back to CPU "
            "and updates the status bar if no GPU is detected.\n"
            "• Latent inspection: use the Preview Latent Space and Show Loss Curve buttons prior to exporting.\n\n"
            "Data & Export Notes\n"
            "• Each group needs at least one pose/video pair; missing directories are skipped with a warning.\n"
            "• Latent exports include optional group, frame, and UMAP projections when available.\n"
            "• CSV export writes one aggregated file per video source with track-level pose coordinates.\n\n"
            "Need More?\n"
            "• Review the diagnostics panel for preprocessing warnings.\n"
            "• Use the Preset selector on the Analysis Parameters tab to load tuned defaults.\n"
            "• For detailed logs, open behavior_analysis.log in your home directory."
        )

        text_widget = scrolledtext.ScrolledText(parent_tab, wrap=tk.WORD)
        text_widget.grid(row=0, column=0, sticky="nsew")
        text_widget.insert(tk.END, help_text)
        text_widget.configure(state='disabled')

    def add_group(self):
        group_name = simpledialog.askstring("Group Name", "Enter group name (e.g., Control):")
        if not group_name or group_name in self.groups:
            if group_name: messagebox.showwarning("Warning", f"Group '{group_name}' already exists.")
            return
        self.groups[group_name] = {'sources': [], 'video_path_map': {}}
        self.group_tree.insert("", "end", text=group_name, iid=group_name, values=("Group", group_name), open=True)

    def import_analytics_runs(self):
        """Import one or more analytics run manifests into a group (Control/Treatment/etc.)."""
        existing = ", ".join(sorted(self.groups.keys())) if self.groups else "(none)"
        group_name = simpledialog.askstring(
            "Import Analytics Runs",
            "Enter the group name to import into (existing or new).\n"
            f"Existing groups: {existing}\n\n"
            "Tip: Add your baseline/control group first (VAE training uses the first group as baseline).",
        )
        if not group_name:
            return
        group_name = group_name.strip()
        if not group_name:
            return

        manifest_paths = filedialog.askopenfilenames(
            title="Select analytics run_manifest.json file(s)",
            filetypes=[("Analytics run manifest", "run_manifest.json"), ("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not manifest_paths:
            return

        group_id = self._ensure_group(group_name)

        imported = 0
        skipped = 0
        errors = []
        for manifest_path in manifest_paths:
            try:
                payload = self._read_analytics_manifest(manifest_path)
                self._apply_manifest_defaults(payload)
                self._add_analytics_source(group_name, group_id, payload)
                imported += 1
            except Exception as exc:
                skipped += 1
                errors.append(f"{os.path.basename(manifest_path)}: {exc}")
                logger.warning("Failed to import analytics manifest %s: %s", manifest_path, exc, exc_info=True)

        if imported and not skipped:
            messagebox.showinfo("Import Complete", f"Imported {imported} analytics run(s) into '{group_name}'.")
        elif imported:
            preview = "\n".join(errors[:5])
            if len(errors) > 5:
                preview += "\n..."
            messagebox.showwarning(
                "Import Partial",
                f"Imported {imported} analytics run(s) into '{group_name}', skipped {skipped}.\n\nIssues:\n{preview}",
            )
        elif errors:
            preview = "\n".join(errors[:8])
            if len(errors) > 8:
                preview += "\n..."
            messagebox.showerror("Import Failed", f"Could not import selected runs.\n\n{preview}")

    def import_analytics_manifest_paths(self, manifest_paths: list[str], *, group_name: str) -> tuple[int, int, list[str]]:
        """Import analytics run_manifest.json files into a group without showing file dialogs."""
        if not manifest_paths:
            return 0, 0, []
        group_name = (group_name or "").strip()
        if not group_name:
            raise ValueError("Group name is required.")

        group_id = self._ensure_group(group_name)
        imported = 0
        skipped = 0
        errors: list[str] = []
        for manifest_path in manifest_paths:
            try:
                payload = self._read_analytics_manifest(manifest_path)
                self._apply_manifest_defaults(payload)
                self._add_analytics_source(group_name, group_id, payload)
                imported += 1
            except Exception as exc:
                skipped += 1
                errors.append(f"{os.path.basename(manifest_path)}: {exc}")
                logger.warning("Failed to import analytics manifest %s: %s", manifest_path, exc, exc_info=True)
        return imported, skipped, errors

    def import_tab6_runs(self):
        """Backward-compatible alias for older integrations."""
        return self.import_analytics_runs()

    def import_tab6_manifest_paths(self, manifest_paths: list[str], *, group_name: str) -> tuple[int, int, list[str]]:
        """Backward-compatible alias for older integrations."""
        return self.import_analytics_manifest_paths(manifest_paths, group_name=group_name)

    def _ensure_group(self, group_name: str) -> str:
        group_name = group_name.strip()
        if group_name not in self.groups:
            self.groups[group_name] = {'sources': [], 'video_path_map': {}}
            self.group_tree.insert("", "end", text=group_name, iid=group_name, values=("Group", group_name), open=True)
        return group_name

    def _read_analytics_manifest(self, manifest_path: str) -> dict:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        if not isinstance(manifest, dict):
            raise ValueError("Manifest JSON must be an object.")
        # Accept v1 (pre-ADP-4) and v2 (adds optional `provenance` block).
        # v1 has no subject_id; v2 may have an empty provenance block when
        # the run wasn't created via the batch pipeline.
        schema = manifest.get("schema_version")
        if schema not in (1, 2):
            raise ValueError(f"Unsupported or missing schema_version (expected 1 or 2, got {schema!r}).")

        run_id = str(manifest.get("run_id") or "").strip()
        inputs = manifest.get("inputs") if isinstance(manifest.get("inputs"), dict) else {}
        outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
        video_meta = manifest.get("video") if isinstance(manifest.get("video"), dict) else {}
        provenance = manifest.get("provenance") if isinstance(manifest.get("provenance"), dict) else {}

        yolo_folder = inputs.get("yolo_folder")
        video_file = inputs.get("video_file")
        yaml_file = inputs.get("yaml_file")

        output_folder = outputs.get("output_folder") or str(Path(manifest_path).resolve().parent)
        detailed_csv = outputs.get("detailed_bouts_csv")
        base_name = video_meta.get("base_name") or ""
        if not detailed_csv and base_name:
            detailed_csv = str(Path(output_folder) / f"{base_name}_detailed_bouts.csv")
        if not detailed_csv:
            candidates = sorted(Path(output_folder).glob("*_detailed_bouts.csv"))
            detailed_csv = str(candidates[0]) if candidates else ""

        yolo_folder = self._resolve_existing_dir(yolo_folder, "Locate YOLO output folder for this run")
        if not yolo_folder:
            raise ValueError("Missing YOLO output folder.")
        video_file = self._resolve_existing_file(
            video_file,
            "Locate source video for this run",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
        )
        yaml_file = self._resolve_existing_file(
            yaml_file,
            "Locate dataset.yaml for this run",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )

        if not detailed_csv or not os.path.exists(detailed_csv):
            raise ValueError("Detailed bouts CSV not found for this run.")

        keypoint_names = inputs.get("keypoint_names")
        behavior_names = inputs.get("behavior_names")

        return {
            "run_id": run_id,
            "manifest_path": str(Path(manifest_path).resolve()),
            "output_folder": str(Path(output_folder).resolve()),
            "yolo_folder": str(Path(yolo_folder).resolve()),
            "video_file": str(Path(video_file).resolve()) if video_file else "",
            "yaml_file": str(Path(yaml_file).resolve()) if yaml_file else "",
            "detailed_bouts_csv": str(Path(detailed_csv).resolve()),
            "keypoint_names": keypoint_names if isinstance(keypoint_names, list) else [],
            "behavior_names": behavior_names if isinstance(behavior_names, list) else [],
            # ADP-4: pull subject identity through. Empty strings for v1
            # manifests and v2 manifests without batch context.
            "subject_id": str(provenance.get("subject_id") or "").strip(),
            "manifest_group": str(provenance.get("group") or "").strip(),
            "time_point": str(provenance.get("time_point") or "").strip(),
        }

    def _read_tab6_manifest(self, manifest_path: str) -> dict:
        """Backward-compatible alias for older integrations."""
        return self._read_analytics_manifest(manifest_path)

    def _resolve_existing_dir(self, path: str | None, title: str) -> str | None:
        if path and os.path.isdir(path):
            return path
        if path:
            ok = messagebox.askyesno("Missing Folder", f"Folder not found:\n{path}\n\nLocate it now?")
            if not ok:
                return None
        chosen = filedialog.askdirectory(title=title, mustexist=True)
        return chosen or None

    def _resolve_existing_file(self, path: str | None, title: str, *, filetypes) -> str | None:
        if path and os.path.exists(path):
            return path
        if path:
            ok = messagebox.askyesno("Missing File", f"File not found:\n{path}\n\nLocate it now?")
            if not ok:
                return None
        chosen = filedialog.askopenfilename(title=title, filetypes=filetypes)
        return chosen or None

    def _apply_manifest_defaults(self, payload: dict) -> None:
        """Auto-fill keypoints/behaviors from analytics metadata when possible."""
        manifest_kps = payload.get("keypoint_names") if isinstance(payload, dict) else None
        manifest_behaviors = payload.get("behavior_names") if isinstance(payload, dict) else None

        if isinstance(manifest_kps, list) and manifest_kps:
            current = parse_comma_separated(self.keypoints_entry_var.get())
            default = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder"]
            if not current or current == default:
                self.keypoints_entry_var.set(",".join([str(v) for v in manifest_kps if str(v).strip()]))
            elif current != manifest_kps:
                raise ValueError("Keypoint names differ from existing session; use a separate project per skeleton.")

        if isinstance(manifest_behaviors, list) and manifest_behaviors:
            current_b = parse_comma_separated(self.behaviors_entry_var.get())
            if not current_b:
                self.behaviors_entry_var.set(",".join([str(v) for v in manifest_behaviors if str(v).strip()]))
            elif current_b != manifest_behaviors:
                # Behaviors can differ across studies; warn but allow import.
                logger.warning("Behavior name list differs from existing session; keeping current entry.")

    def _add_analytics_source(self, group_name: str, group_id: str, payload: dict) -> None:
        pose_dir_raw = payload.get("yolo_folder")
        video_path_raw = payload.get("video_file") or ""
        run_id = payload.get("run_id") or ""
        run_folder = payload.get("output_folder") or ""
        pose_dir = str(Path(pose_dir_raw).resolve()) if pose_dir_raw else ""
        video_path = str(Path(video_path_raw).resolve()) if video_path_raw else ""
        try:
            current_out = self.output_folder.get()
        except Exception:
            current_out = ""
        if (not current_out) or (not os.path.isdir(current_out)):
            if run_folder:
                candidate = os.path.join(run_folder, "tab7_behavior_clustering")
                try:
                    os.makedirs(candidate, exist_ok=True)
                    self.output_folder.set(candidate)
                except Exception:
                    pass

        if not pose_dir or not os.path.isdir(pose_dir):
            raise ValueError("YOLO output folder is missing or invalid.")
        if group_name not in self.groups:
            self.groups[group_name] = {'sources': [], 'video_path_map': {}}

        existing_sources = self.groups[group_name].get("sources", [])
        for src in existing_sources:
            if not isinstance(src, dict):
                continue
            if src.get("pose_dir") == pose_dir:
                raise ValueError(f"YOLO folder already added to group: {os.path.basename(pose_dir)}")

        data_source = {
            "pose_dir": pose_dir,
            "video_path": video_path,
            "run_id": run_id,
            "tab6_manifest": payload.get("manifest_path"),
            "tab6_run_dir": run_folder,
            "tab6_detailed_bouts_csv": payload.get("detailed_bouts_csv"),
            "tab6_yaml": payload.get("yaml_file"),
            # ADP-4: subject_id from manifest provenance (v2). Empty for v1
            # manifests or batches with no subject id assigned. Tab 7's
            # auto-split helper (Commit C) keys on this.
            "subject_id": str(payload.get("subject_id") or "").strip(),
            "time_point": str(payload.get("time_point") or "").strip(),
        }
        self.groups[group_name]["sources"].append(data_source)
        if video_path:
            self.groups[group_name]["video_path_map"][pose_dir] = video_path

        run_label = os.path.basename(run_folder) if run_folder else "Analytics"
        source_display_text = f"{os.path.basename(pose_dir)} | {os.path.basename(video_path) if video_path else 'video'} (Analytics: {run_label})"
        self.group_tree.insert(group_id, "end", text=pose_dir, values=("Data Source", source_display_text))
        logger.info("Imported analytics run into group %s: %s", group_name, data_source)

    def _add_tab6_source(self, group_name: str, group_id: str, payload: dict) -> None:
        """Backward-compatible alias for older integrations."""
        self._add_analytics_source(group_name, group_id, payload)

    def add_data_source(self):
        selection = self.group_tree.selection()
        if not selection: return messagebox.showwarning("Warning", "Select a group first.")
        group_id = selection[0]
        if self.group_tree.parent(group_id): group_id = self.group_tree.parent(group_id)
        group_name = self.group_tree.item(group_id, "text")
        
        pose_dir = filedialog.askdirectory(title=f"Step 1/2: Select POSE DIRECTORY for {group_name}", mustexist=True)
        if not pose_dir: return
        
        video_path = filedialog.askopenfilename(title=f"Step 2/2: Select VIDEO FILE for {os.path.basename(pose_dir)}", filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
        if not video_path: return

        pose_dir = str(Path(pose_dir).resolve())
        video_path = str(Path(video_path).resolve())
        
        data_source = {'pose_dir': pose_dir, 'video_path': video_path}
        
        if group_name not in self.groups: self.groups[group_name] = {'sources': [], 'video_path_map': {}}
        existing_sources = self.groups[group_name].get("sources", [])
        if any(isinstance(src, dict) and src.get("pose_dir") == pose_dir for src in existing_sources):
            return messagebox.showwarning("Warning", f"That pose directory is already in '{group_name}'.")
        self.groups[group_name]['sources'].append(data_source)
        self.groups[group_name]['video_path_map'][pose_dir] = video_path

        source_display_text = f"{os.path.basename(pose_dir)} | {os.path.basename(video_path)}"
        self.group_tree.insert(group_id, "end", text=pose_dir, values=("Data Source", source_display_text))
        logger.info(f"Added data source to group {group_name}: {data_source}")

    def remove_selected(self):
        def norm_path(p: str) -> str:
            return os.path.normcase(os.path.normpath(os.path.abspath(p)))

        for item_id in reversed(self.group_tree.selection()):
            parent_id = self.group_tree.parent(item_id)
            if not parent_id: # It's a group
                group_name = self.group_tree.item(item_id, "text")
                if group_name in self.groups: del self.groups[group_name]
            else: # It's a data source
                group_name = self.group_tree.item(parent_id, "text")
                pose_dir_to_remove = self.group_tree.item(item_id, "text")
                if group_name in self.groups:
                    target_norm = norm_path(pose_dir_to_remove) if pose_dir_to_remove else ""
                    if target_norm:
                        self.groups[group_name]['sources'] = [
                            src for src in self.groups[group_name].get('sources', [])
                            if not (isinstance(src, dict) and norm_path(str(src.get('pose_dir', ''))) == target_norm)
                        ]
                        vmap = self.groups[group_name].get('video_path_map', {})
                        for key in list(vmap.keys()):
                            if norm_path(str(key)) == target_norm:
                                del vmap[key]
                    else:
                        pose_dir_basename = self.group_tree.item(item_id, "values")[1].split(' | ')[0]
                        self.groups[group_name]['sources'] = [
                            src for src in self.groups[group_name].get('sources', [])
                            if not (isinstance(src, dict) and os.path.basename(str(src.get('pose_dir', ''))) == pose_dir_basename)
                        ]
                        vmap = self.groups[group_name].get('video_path_map', {})
                        for key in list(vmap.keys()):
                            if os.path.basename(str(key)) == pose_dir_basename:
                                del vmap[key]
            self.group_tree.delete(item_id)

    def select_output_folder(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if folder: self.output_folder.set(folder)
    
    def get_all_params(self):
        params = {
            'groups': self.groups,
            'output_folder': self.output_folder.get(),
            'keypoints_entry_var': self.keypoints_entry_var.get(),
            'behaviors_entry_var': self.behaviors_entry_var.get(),
            'use_bbox_var': self.use_bbox_var.get(),
            'location_mode': self.location_mode.get(),
            'location_grid_size': self.location_grid_size.get(),
            'generate_location_videos': self.generate_location_videos.get(),
            'max_frame_gap': self.max_frame_gap.get(),
            'min_bout_duration': self.min_bout_duration.get(),
            'conf_threshold': self.conf_threshold.get(),
            'umap_neighbors': self.umap_neighbors.get(),
            'umap_components': self.umap_components.get(),
            'min_cluster_size': self.min_cluster_size.get(),
            # ADP-4 Commit E — stability audit flags.
            'run_stability_audit': bool(self.run_stability_audit.get()),
            'stability_n_seeds': self.stability_n_seeds.get(),
            'social_mode': self.social_mode.get(),
            'skeleton_connections': self.skeleton_connections,
            'normalization_left': self.normalization_left.get(),
            'normalization_right': self.normalization_right.get(),
            'roi_definitions_text': self.roi_text.get("1.0", tk.END),
        }
        return params

    def _collect_group_sources(self):
        """Walk the configured groups and produce dicts the runtime needs.

        Returns:
            (pose_dirs_by_group, video_path_map, subject_id_map)

        ``subject_id_map`` is keyed by pose_dir → subject identifier. The
        identifier comes from the manifest provenance (v2 manifests, set by
        the batch pipeline). When absent, we fall back to the pose-dir
        basename so each source still has a unique stable id — that's what
        the Tab 7 auto-split helper (Commit C) groups on. Manual / raw
        imports without a manifest land here too and get the basename
        fallback automatically.
        """
        pose_dirs_by_group = {}
        video_path_map = {}
        subject_id_map: dict[str, str] = {}
        missing_dirs = []

        for group_name, payload in (self.groups or {}).items():
            sources = payload.get('sources', [])
            valid_pose_dirs = []
            for src in sources:
                pose_dir = src.get('pose_dir')
                video_path = src.get('video_path')
                if not pose_dir:
                    continue
                if not os.path.isdir(pose_dir):
                    missing_dirs.append((group_name, pose_dir))
                    continue
                valid_pose_dirs.append(pose_dir)
                if video_path:
                    video_path_map[pose_dir] = video_path
                # ADP-4: subject_id from manifest provenance, with basename
                # fallback for v1 manifests / manual imports.
                src_subject_id = str((src or {}).get("subject_id") or "").strip()
                if not src_subject_id:
                    src_subject_id = os.path.basename(pose_dir.rstrip(os.sep)) or pose_dir
                subject_id_map[pose_dir] = src_subject_id
            if valid_pose_dirs:
                pose_dirs_by_group[group_name] = valid_pose_dirs

        if not pose_dirs_by_group:
            if missing_dirs:
                preview = ", ".join(f"{grp}: {os.path.basename(path) or path}" for grp, path in missing_dirs[:3])
                if len(missing_dirs) > 3:
                    preview += ", ..."
                raise ValueError(f"No valid pose directories found. Check that the following paths exist: {preview}")
            raise ValueError("No pose directories configured. Add at least one pose/video pair to a group.")

        if missing_dirs:
            truncated = ", ".join(f"{grp}: {os.path.basename(path) or path}" for grp, path in missing_dirs[:3])
            if len(missing_dirs) > 3:
                truncated += ", ..."
            logger.warning("Skipping missing pose directories: %s", truncated)

        return pose_dirs_by_group, video_path_map, subject_id_map

    def _build_bouts_from_analytics_runs(self, detections_df: pd.DataFrame) -> tuple[list[dict], set[str]]:
        """Create bout observations from imported analytics bouts, preserving bout IDs when available.

        Returns (bouts, covered_pose_dirs). If no analytics runs are configured, returns ([], set()).
        """
        tab6_by_dir: dict[str, dict] = {}
        for group_name, payload in (self.groups or {}).items():
            sources = payload.get("sources", [])
            for src in sources:
                if not isinstance(src, dict):
                    continue
                pose_dir = src.get("pose_dir")
                csv_path = src.get("tab6_detailed_bouts_csv")
                if not pose_dir or not csv_path:
                    continue
                if not os.path.exists(csv_path):
                    logger.warning("Analytics bouts CSV missing for %s: %s", pose_dir, csv_path)
                    continue
                tab6_by_dir[str(pose_dir)] = {
                    "group": group_name,
                    "csv": str(csv_path),
                    "run_id": src.get("run_id") or "",
                }

        if not tab6_by_dir:
            return [], set()

        bouts: list[dict] = []
        covered_dirs: set[str] = set()
        skipped_bouts = 0

        for pose_dir, meta in tab6_by_dir.items():
            pose_dir = str(pose_dir)
            df_dir = detections_df[detections_df["directory"] == pose_dir]
            if df_dir.empty:
                continue
            covered_dirs.add(pose_dir)

            bouts_df = pd.read_csv(meta["csv"])
            if bouts_df.empty:
                continue

            track_col = "Track ID" if "Track ID" in bouts_df.columns else ("Animal ID" if "Animal ID" in bouts_df.columns else None)
            required = {track_col, "Behavior", "Start Frame", "End Frame"} if track_col else {"Behavior", "Start Frame", "End Frame"}
            if not track_col or not required.issubset(bouts_df.columns):
                raise ValueError(f"Analytics bouts CSV missing required columns: {meta['csv']}")

            for _, row in bouts_df.iterrows():
                try:
                    track_id = int(row[track_col])
                    start_frame = int(row["Start Frame"])
                    end_frame = int(row["End Frame"])
                except Exception:
                    skipped_bouts += 1
                    continue

                behavior = str(row.get("Behavior", "")).strip()
                det_slice = df_dir[
                    (df_dir["track_id"] == track_id)
                    & (df_dir["frame"] >= start_frame)
                    & (df_dir["frame"] <= end_frame)
                ]
                if det_slice.empty:
                    skipped_bouts += 1
                    continue

                det_behavior = det_slice
                if behavior and "behavior" in det_slice.columns:
                    det_behavior = det_slice[det_slice["behavior"].astype(str) == behavior]
                    if det_behavior.empty:
                        det_behavior = det_slice

                feature_vectors_seq = det_behavior["feature_vector"].tolist()
                if not feature_vectors_seq:
                    skipped_bouts += 1
                    continue
                try:
                    mean_feature_vector = np.mean(np.asarray(feature_vectors_seq, dtype=np.float32), axis=0)
                except Exception:
                    skipped_bouts += 1
                    continue

                class_id = None
                try:
                    class_mode = det_behavior["class_id"].mode()
                    if not class_mode.empty:
                        class_id = int(class_mode.iloc[0])
                except Exception:
                    class_id = None
                if class_id is None:
                    try:
                        class_id = int(det_behavior["class_id"].iloc[0])
                    except Exception:
                        class_id = 0

                bout_centroids = []
                for _, det in det_behavior.iterrows():
                    bbox = det.get("bbox")
                    if isinstance(bbox, (list, tuple)) and len(bbox) >= 2:
                        bout_centroids.append(bbox[:2])
                        continue
                    keypoints = det.get("keypoints")
                    if isinstance(keypoints, dict) and keypoints:
                        kp_arr = np.array(
                            [kp[:2] for kp in keypoints.values() if isinstance(kp, (list, tuple)) and len(kp) >= 3 and kp[2] > 0.1],
                            dtype=np.float32,
                        )
                        if kp_arr.size:
                            bout_centroids.append(np.mean(kp_arr, axis=0))
                mean_location = np.mean(np.asarray(bout_centroids, dtype=np.float32), axis=0) if bout_centroids else (0.0, 0.0)

                if "video_source" in df_dir.columns and not df_dir.empty:
                    video_source = str(df_dir["video_source"].iloc[0])
                else:
                    video_source = os.path.basename(pose_dir) or "source"

                bouts.append(
                    {
                        "group": meta["group"],
                        "video_source": video_source,
                        "directory": pose_dir,
                        "track_id": track_id,
                        "behavior": behavior or det_behavior["behavior"].iloc[0],
                        "class_id": class_id,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "duration_frames": int(end_frame - start_frame + 1),
                        "detection_count": int(len(det_behavior)),
                        "feature_vector": mean_feature_vector,
                        "mean_location": mean_location,
                        "tab6_run_id": str(row.get("Run ID", meta.get("run_id", ""))),
                        "tab6_bout_id": str(row.get("Bout ID", "")),
                        "roi_name": str(row.get("ROI Name", "")) if "ROI Name" in bouts_df.columns else "",
                    }
                )

        if skipped_bouts:
            logger.warning("Skipped %d analytics bouts that could not be matched to detections.", skipped_bouts)

        return bouts, covered_dirs

    def _build_bouts_from_tab6_runs(self, detections_df: pd.DataFrame) -> tuple[list[dict], set[str]]:
        """Backward-compatible alias for older integrations."""
        return self._build_bouts_from_analytics_runs(detections_df)

    def set_all_params(self, state):
        for key, value in state.items():
            if hasattr(self, key):
                attr = getattr(self, key)
                if isinstance(attr, tk.Variable):
                    attr.set(value)

        self.groups = state.get('groups', {})
        self.skeleton_connections = state.get('skeleton_connections', [])
        
        if 'roi_definitions_text' in state:
            self.roi_text.delete("1.0", tk.END)
            self.roi_text.insert(tk.END, state['roi_definitions_text'])

        self.skeleton_list.delete(0, tk.END)
        for conn in self.skeleton_connections:
            self.skeleton_list.insert(tk.END, f"{conn[0]} -- {conn[1]}")

        self.group_tree.delete(*self.group_tree.get_children())
        for group_name, group_data in self.groups.items():
            self.group_tree.insert("", "end", text=group_name, iid=group_name, values=("Group", group_name), open=True)
            sources = group_data.get('sources', [])
            for source in sources:
                if isinstance(source, dict) and 'pose_dir' in source and 'video_path' in source:
                    pose_dir = str(source.get('pose_dir', ''))
                    video_path = str(source.get('video_path', ''))
                    run_folder = str(source.get("tab6_run_dir", "") or "")
                    if run_folder:
                        run_label = os.path.basename(run_folder) or "Tab6"
                        source_text = f"{os.path.basename(pose_dir)} | {os.path.basename(video_path) if video_path else 'video'} (Tab6: {run_label})"
                    else:
                        source_text = f"{os.path.basename(pose_dir)} | {os.path.basename(video_path)}"
                    self.group_tree.insert(group_name, "end", text=pose_dir, values=("Data Source", source_text))
        
        self.update_keypoint_combos()

    def save_project(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Project Files", "*.json")])
        if file_path:
            params = self.get_all_params()
            try:
                with open(file_path, 'w') as f: json.dump(params, f, indent=4)
                logger.info(f"Project saved to {file_path}")
                messagebox.showinfo("Success", f"Project successfully saved to:\n{file_path}")
            except Exception as e:
                logger.error(f"Failed to save project: {e}", exc_info=True)
                messagebox.showerror("Error", f"Failed to save project file:\n{e}")

    def load_project(self, file_path=None):
        if not file_path:
            file_path = filedialog.askopenfilename(filetypes=[("JSON Project Files", "*.json")])
        if not file_path: return
        try:
            with open(file_path, 'r') as f: state = json.load(f)
            self.set_all_params(state)
            logger.info(f"Project loaded from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load project: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to load project file:\n{e}")

    def load_multiple_projects(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("JSON Project Files", "*.json")])
        if file_paths: threading.Thread(target=self.process_multiple_projects, args=(file_paths,), daemon=True).start()

    def process_multiple_projects(self, file_paths):
        try:
            self.running = True
            self._set_status_progress_async(message="Processing multiple projects...", progress=0)
            for i, file_path in enumerate(file_paths):
                self._set_status_progress_async(message=f"Loading project {i+1}/{len(file_paths)}: {os.path.basename(file_path)}")
                self._ui_call_sync(self.load_project, file_path)
                params = self._ui_call_sync(self._capture_analysis_params)
                self._set_status_progress_async(message=f"Running analysis for project {i+1}/{len(file_paths)}")
                self._run_analysis_worker_from_params(params, notify=False)
                self._set_status_progress_async(progress=((i + 1) / len(file_paths)) * 100)
            self._set_status_progress_async(message="Batch processing complete")
            self._show_info_async("Success", "All projects processed!")
        except Exception as e:
            self._set_status_progress_async(message=f"Error: {e}")
            logger.error(f"Batch processing failed: {e}", exc_info=True)
            self._show_error_async("Error", f"Batch processing failed: {e}")
        finally:
            self.running = False
            self._reset_run_state_async()

    def validate_inputs(self, for_export=False):
        if not self.output_folder.get() or not os.path.isdir(self.output_folder.get()):
            raise ValueError("A valid Output folder must be specified.")
        if not self.groups: raise ValueError("At least one data group must be added.")
        total_sources = sum(len(v.get('sources', [])) for v in self.groups.values())
        if total_sources == 0:
            raise ValueError("Add at least one pose/video pair to a group before continuing.")
        if for_export: return True
        
        try:
            float(self.conf_threshold.get())
            int(self.max_frame_gap.get())
            int(self.min_bout_duration.get())
            int(self.umap_neighbors.get())
            int(self.umap_components.get())
            int(self.min_cluster_size.get())
        except ValueError as e:
            raise ValueError(f"Invalid numerical parameter. Please check your inputs. Error: {e}")

        if not self.normalization_left.get() or not self.normalization_right.get():
            raise ValueError("Both normalization reference keypoints must be selected.")

        if self.location_mode.get() == 'roi':
            try:
                roi_defs = json.loads(self.roi_text.get("1.0", tk.END))
                validate_roi_definitions(roi_defs)
            except json.JSONDecodeError: raise ValueError("ROI Definitions are not valid JSON.")
            except ValidationError as e: raise ValueError(f"ROI Definition Error: {e}")
        
        return True

    def apply_session_preset(self):
        preset_name = self.preset_var.get()
        preset = self.session_presets.get(preset_name)
        if not preset:
            messagebox.showinfo("Presets", "Select a preset to apply.")
            return
        for key, value in preset.items():
            if hasattr(self, key):
                attr = getattr(self, key)
                if isinstance(attr, tk.Variable):
                    attr.set(value)
        if preset_name != "Custom":
            self.status_var.set(f"Applied preset '{preset_name}'. Adjust parameters if needed.")
        else:
            self.status_var.set("Preset reset to custom.")

    def start_csv_export(self):
        if self.running: return messagebox.showwarning("Warning", "Process is already running.")
        try: self.validate_inputs(for_export=True)
        except ValueError as e: return messagebox.showerror("Input Error", str(e))
        export_inputs = self._capture_export_inputs()
        self.running = True
        threading.Thread(target=self.run_csv_export, args=(export_inputs,), daemon=True).start()

    def run_csv_export(self, export_inputs):
        try:
            logger.info("="*20 + " Starting CSV Export " + "="*20)
            self._set_status_progress_async(message="Starting CSV export...", progress=0)
            pose_dirs_by_group, video_path_map, subject_id_map = self._collect_group_sources()

            self._set_status_progress_async(message="Loading detection data...", progress=20)
            detections_df, keypoint_names, _ = self._read_runtime_detections(
                pose_dirs_by_group,
                export_inputs['keypoints_entry_var'],
                export_inputs['behaviors_entry_var'],
                export_inputs['use_bbox_var'],
                video_path_map=video_path_map,
                subject_id_map=subject_id_map,
            )
            if detections_df.empty: raise ValueError("No detections found to export.")
            
            self._set_status_progress_async(message="Writing CSV files...", progress=60)
            export_detections_to_csv(detections_df, keypoint_names, export_inputs['output_folder'])
            
            self._set_status_progress_async(message="Export complete", progress=100)
            self._show_info_async("Success", "CSV export complete!")
        except Exception as e:
            self._set_status_progress_async(message=f"Error: {e}")
            logger.error(f"CSV export failed: {e}", exc_info=True)
            self._show_error_async("Error", f"CSV export failed: {e}")
        finally:
            self.running = False
            self._reset_run_state_async()

    def progress_callback(self, current, total, message):
        self._set_status_progress_async(current=current, total=total, message=message)

    # =========================================================================
    # ADP-4 Commit D-MVP — Sub-Behavior Discovery worker
    #
    # New analysis path that REPLACES the HMM-on-pooled-data approach for the
    # user's mission ("dissect sub-behaviors within YOLO classes"). Per-class
    # HDBSCAN on UMAP-reduced features, results aggregated into bouts and
    # saved next to the user's chosen output folder. This worker is wired
    # behind a separate "Run Sub-Behavior Discovery" button so the legacy
    # HMM/VAE/Hybrid paths stay clickable while the user verifies the new
    # flow on real data.
    # =========================================================================

    def run_sub_behavior_discovery_analysis(self, params=None, *, notify=True):
        """Cluster sub-behaviors WITHIN each YOLO class (ADP-4 Commit D-MVP).

        Pipeline:
            1. Read detections + features (same plumbing as legacy paths).
            2. Per-class HDBSCAN via ``cluster_per_class``.
            3. Aggregate per-frame ``cluster_label`` into sub-cluster bouts
               via ``aggregate_states_into_bouts``.
            4. Persist outputs (CSV) to the user's output folder.
            5. Render a per-class State Summary breakdown.
            6. Cache the latest result so the "Export Sub-cluster Clips"
               button has something to act on.

        The legacy HMM-on-pooled-data flow is left in place but is not the
        recommended path. The State Summary now distinguishes per-class
        results (``kind="per_class"``) from HMM-summary entries.
        """
        try:
            from .per_class_clustering import cluster_per_class  # noqa: PLC0415
            from .bouts import aggregate_states_into_bouts  # noqa: PLC0415

            logger.info("=" * 20 + " Starting Sub-Behavior Discovery Run " + "=" * 20)
            self._set_status_progress_async(message="Initializing...", progress=0)
            params = params or self._ui_call_sync(self._capture_analysis_params)

            # Generate a per-run UUID so state_names.json from a prior
            # run with different parameters can't pollute clip filenames
            # for THIS run. Saved alongside the result; the export
            # helper compares before loading names.
            import uuid  # noqa: PLC0415

            run_id = str(uuid.uuid4())

            # Per advisor: don't preemptively wipe the cached result.
            # If THIS run fails, the user can still review / export the
            # prior successful run. Only `feature_diagnostics` is reset
            # because it's purely a per-run artifact and stale values
            # would be misleading during the in-progress run.
            self.feature_diagnostics = {}
            self._refresh_diagnostics_async()

            normalization_ref_points = (
                self.normalization_left.get(),
                self.normalization_right.get(),
            )
            roi_defs = (
                json.loads(self.roi_text.get("1.0", tk.END))
                if params['location_mode'] == 'roi'
                else None
            )
            grid_size = (
                int(params['location_grid_size'])
                if params['location_mode'] == 'unsupervised'
                else 50
            )

            self.progress_callback(5, 100, "Loading detections...")
            pose_dirs_by_group, video_map, subject_id_map = self._collect_group_sources()
            detections_df, keypoint_names, behavior_names = self._read_runtime_detections(
                pose_dirs_by_group,
                params['keypoints_entry_var'],
                params['behaviors_entry_var'],
                params['use_bbox_var'],
                video_path_map=video_map,
                subject_id_map=subject_id_map,
            )
            self.behavior_names = behavior_names
            if detections_df.empty:
                raise ValueError(
                    "No detections found. Check pose directories under each group."
                )

            self.progress_callback(20, 100, "Computing features...")
            detections_df, feature_diag = compute_feature_vectors(
                detections_df,
                keypoint_names,
                float(params['conf_threshold']),
                normalization_ref_points=normalization_ref_points,
                use_bbox=params['use_bbox_var'],
                location_mode=params['location_mode'],
                location_grid_size=grid_size,
                roi_definitions=roi_defs,
                social_mode=params['social_mode'],
            )
            self.feature_diagnostics = feature_diag
            self._refresh_diagnostics_async()

            # Build a {class_id: name} map from behavior_names so the result
            # carries human labels into the State Summary and clip filenames.
            class_names = {}
            for idx, name in enumerate(behavior_names or []):
                class_names[idx] = str(name)

            self.progress_callback(40, 100, "Clustering per class...")
            min_class_size = max(
                10,
                int(params.get('min_bout_duration', 0)) * 5,
            )
            # Reuse the existing UMAP/HDBSCAN params from Tab 2.
            clustered_df, multi_result = cluster_per_class(
                detections_df,
                class_names=class_names,
                min_class_size=min_class_size,
                min_cluster_size=int(params.get('min_cluster_size', 10)),
                umap_neighbors=int(params.get('umap_neighbors', 15)),
                umap_components=int(params.get('umap_components', 5)),
            )

            self.progress_callback(60, 100, "Aggregating sub-cluster bouts...")
            bouts = aggregate_states_into_bouts(
                clustered_df,
                int(params.get('max_frame_gap', 5)),
                int(params.get('min_bout_duration', 3)),
                state_column='cluster_label',
            )

            # ADP-4 Commit E — optional stability audit. The user can
            # opt in via the "Run stability audit" checkbox. We re-cluster
            # with N seeds and compute per-class mean ARI. Cost is N×
            # the primary clustering, so it's behind a flag.
            stability_report = None
            if bool(params.get('run_stability_audit', False)):
                try:
                    n_seeds = max(2, int(params.get('stability_n_seeds', 3)))
                except Exception:
                    n_seeds = 3
                self.progress_callback(
                    70, 100,
                    f"Running stability audit ({n_seeds} seeds)...",
                )
                try:
                    from .stability import compute_class_stability  # noqa: PLC0415

                    stability_report = compute_class_stability(
                        detections_df,
                        n_seeds=n_seeds,
                        cluster_kwargs={
                            'class_names': class_names,
                            'min_class_size': min_class_size,
                            'min_cluster_size': int(params.get('min_cluster_size', 10)),
                            'umap_neighbors': int(params.get('umap_neighbors', 15)),
                            'umap_components': int(params.get('umap_components', 5)),
                        },
                    )
                except Exception as exc:
                    logger.warning("Stability audit failed: %s", exc, exc_info=True)
                    stability_report = None

            # ADP-4 Commit F: signal scoring — always runs (cheap, no
            # extra clustering). Ranks sub-clusters by candidate strength
            # so the user can review strongest first.
            try:
                from .signal_score import compute_signal_scores  # noqa: PLC0415
                self.progress_callback(80, 100, "Scoring sub-clusters...")
                signal_report = compute_signal_scores(
                    bouts,
                    multi_result,
                    stability_report=stability_report,
                )
            except Exception as exc:
                logger.warning("Signal scoring failed: %s", exc, exc_info=True)
                signal_report = None

            # Persist outputs.
            self.progress_callback(85, 100, "Writing outputs...")
            output_folder = params.get('output_folder') or self.output_folder.get()
            saved_paths = self._save_sub_behavior_outputs(
                output_folder=output_folder,
                clustered_df=clustered_df,
                bouts=bouts,
                multi_result=multi_result,
                class_names=class_names,
                stability_report=stability_report,
                signal_report=signal_report,
                run_id=run_id,
            )

            # Build per-class summary entries and surface them in the
            # existing State Summary widget (with a sub-behavior layout).
            summary_entries = self._build_sub_behavior_summary_entries(
                multi_result, bouts, class_names, stability_report=stability_report,
            )
            self.state_summary_payload = summary_entries
            self._refresh_state_summary_async()

            # Cache for the export button + review-candidates panel.
            #
            # Invariant: the steps between _save_sub_behavior_outputs and
            # this assignment must not raise. _save_sub_behavior_outputs
            # wraps every write in try/except; the summary-entries
            # builder is pure dict construction; the state-summary
            # refresh is a root.after() schedule that never raises.
            # If a future change adds a step here that CAN raise, the
            # cache becomes stale + saved_paths point at this run's
            # files — handle that by adding atomic-rename writes or
            # by moving the cache assignment up.
            self.last_sub_behavior_result = {
                'run_id': run_id,
                'clustered_df': clustered_df,
                'bouts': bouts,
                'multi_result': multi_result,
                'class_names': class_names,
                'video_path_map': video_map,
                'output_folder': output_folder,
                'saved_paths': saved_paths,
                'stability_report': stability_report,
                'signal_report': signal_report,
            }
            self._enable_clip_export_button_async(True)
            self._enable_review_button_async(True)
            self._enable_naming_button_async(True)

            self.progress_callback(100, 100, "Sub-behavior discovery complete.")
            if notify:
                self._show_info_async(
                    "Sub-Behavior Discovery",
                    self._format_sub_behavior_summary(
                        multi_result, len(bouts), saved_paths,
                        stability_report=stability_report,
                    ),
                )

        except Exception as exc:  # pragma: no cover - surfaced to UI
            logger.error("Sub-behavior discovery failed: %s", exc, exc_info=True)
            # Per advisor: be explicit that the prior successful run's
            # outputs (if any) are still available for review/export —
            # the failed run did not wipe them.
            prior = getattr(self, "last_sub_behavior_result", None)
            suffix = (
                "\n\nThe previous successful run is still available — the "
                "Review, Name, and Export buttons act on those results."
                if prior else ""
            )
            self._show_error_async(
                "Sub-Behavior Discovery",
                f"{exc}{suffix}",
            )
        finally:
            self.running = False
            self._set_status_progress_async(progress=0)

    def _save_sub_behavior_outputs(self, *, output_folder, clustered_df, bouts, multi_result, class_names, stability_report=None, signal_report=None, run_id: str = ""):
        """Write per-frame and bout CSVs into the output folder.

        Returns a dict with the absolute paths so the summary message and
        the Export button can quote them. When ``stability_report`` is
        provided (Commit E), an additional ``stability.json`` is written
        with per-class ARI matrices and the headline mean ARI.

        ``run_id`` is a UUID stamped on every artefact so that downstream
        consumers (state_names.json loader, clip export) can detect
        stale companion files from a previous run with different params.
        """
        out_root = os.path.abspath(output_folder or ".")
        try:
            os.makedirs(out_root, exist_ok=True)
        except Exception:
            logger.warning("Could not create output folder %s", out_root)

        per_frame_path = os.path.join(out_root, "sub_behavior_per_frame.csv")
        bouts_path = os.path.join(out_root, "sub_behavior_bouts.csv")
        summary_path = os.path.join(out_root, "sub_behavior_summary.txt")
        run_id_path = os.path.join(out_root, "sub_behavior_run_id.txt")

        # Per-frame CSV — drop the heavy columns the user doesn't need
        # in tabular form (keypoints dict, raw feature_vector). They keep
        # the cluster_label and the basic identity columns.
        try:
            export_cols = [
                c for c in [
                    "group", "subject_id", "video_source", "directory",
                    "frame", "track_id", "class_id", "behavior",
                    "cluster_label",
                ]
                if c in clustered_df.columns
            ]
            clustered_df[export_cols].to_csv(per_frame_path, index=False)
        except Exception as exc:
            logger.warning("Failed to write per-frame CSV: %s", exc)

        # Bouts CSV.
        try:
            import pandas as pd  # noqa: PLC0415

            bout_rows = []
            for b in (bouts or []):
                bout_rows.append({
                    "group": b.get("group"),
                    "subject_id": b.get("subject_id"),
                    "video_source": b.get("video_source"),
                    "directory": b.get("directory"),
                    "track_id": b.get("track_id"),
                    "class_id": b.get("class_id"),
                    "behavior": b.get("behavior"),
                    "cluster_label": b.get("state"),
                    "start_frame": b.get("start_frame"),
                    "end_frame": b.get("end_frame"),
                    "duration_frames": b.get("duration_frames"),
                    "detection_count": b.get("detection_count"),
                })
            pd.DataFrame(bout_rows).to_csv(bouts_path, index=False)
        except Exception as exc:
            logger.warning("Failed to write bouts CSV: %s", exc)

        # Plain-text summary.
        try:
            lines = ["IntegraPose Tab 7 — Sub-Behavior Discovery", ""]
            if run_id:
                lines.append(f"run_id: {run_id}")
                lines.append("")
            for cls in multi_result.classes:
                name = cls.class_name or class_names.get(cls.class_id) or f"class_{cls.class_id}"
                if cls.skipped:
                    lines.append(
                        f"- {name} (class_id={cls.class_id}): "
                        f"SKIPPED — {cls.skip_reason}"
                    )
                else:
                    lines.append(
                        f"- {name} (class_id={cls.class_id}): "
                        f"{cls.n_clusters} sub-cluster(s) from {cls.n_frames} frames "
                        f"({cls.n_noise_frames} noise)"
                    )
            lines.append("")
            lines.append(f"Total sub-cluster bouts: {len(bouts or [])}")
            with open(summary_path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines))
        except Exception as exc:
            logger.warning("Failed to write summary text: %s", exc)

        saved = {
            "per_frame_csv": per_frame_path,
            "bouts_csv": bouts_path,
            "summary_txt": summary_path,
        }

        # Stamp this run's UUID to disk. The naming dialog and clip
        # exporter both compare this against state_names.json's stored
        # run_id; mismatch ⇒ stale, ignored.
        if run_id:
            try:
                with open(run_id_path, "w", encoding="utf-8") as fh:
                    fh.write(str(run_id) + "\n")
                saved["run_id_txt"] = run_id_path
            except Exception as exc:
                logger.warning("Failed to write run_id sidecar: %s", exc)

        # ADP-4 Commit F: signal-score CSV. Always written when scores
        # were computed — the review panel and external readers both
        # consume this. Sorted desc by score.
        if signal_report is not None and signal_report.candidates:
            scores_path = os.path.join(out_root, "sub_behavior_candidate_scores.csv")
            try:
                import pandas as pd  # noqa: PLC0415

                rows = []
                for cand in signal_report.candidates:
                    rows.append({
                        "namespaced_label": cand.namespaced_label,
                        "class_id": cand.class_id,
                        "class_name": cand.class_name,
                        "sub_id": cand.sub_id,
                        "score": round(cand.score, 4),
                        "verdict": cand.verdict,
                        "n_frames": cand.n_frames,
                        "n_subjects": cand.n_subjects,
                        "n_bouts": cand.n_bouts,
                        "mean_bout_frames": round(cand.mean_bout_frames, 2),
                        "stability_ari": (
                            None if cand.stability_ari is None
                            else round(cand.stability_ari, 3)
                        ),
                        "size_signal": round(cand.size_signal, 3),
                        "coverage_signal": round(cand.coverage_signal, 3),
                        "duration_signal": round(cand.duration_signal, 3),
                        "stability_signal": (
                            None if cand.stability_signal is None
                            else round(cand.stability_signal, 3)
                        ),
                        "notes": " | ".join(cand.notes),
                    })
                pd.DataFrame(rows).to_csv(scores_path, index=False)
                saved["candidate_scores_csv"] = scores_path
            except Exception as exc:
                logger.warning("Failed to write candidate scores CSV: %s", exc)

        # ADP-4 Commit E: optional stability JSON.
        if stability_report is not None:
            stability_path = os.path.join(out_root, "sub_behavior_stability.json")
            try:
                payload = {
                    "run_id": run_id,
                    "n_seeds": stability_report.n_seeds,
                    "base_seed": stability_report.base_seed,
                    "overall": stability_report.overall_summary,
                    "n_stable": stability_report.n_stable,
                    "n_total": stability_report.n_total,
                    "per_class": [
                        {
                            "class_id": c.class_id,
                            "class_name": c.class_name,
                            "skipped": c.skipped,
                            "skip_reason": c.skip_reason,
                            "seeds_used": list(c.seeds_used),
                            "ari_matrix": c.ari_matrix,
                            "mean_ari": (None if (c.mean_ari != c.mean_ari) else float(c.mean_ari)),
                            "min_ari": (None if (c.min_ari != c.min_ari) else float(c.min_ari)),
                            "n_clusters_per_seed": list(c.n_clusters_per_seed),
                            "is_stable": c.is_stable,
                        }
                        for c in stability_report.per_class
                    ],
                }
                with open(stability_path, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=2)
                saved["stability_json"] = stability_path
            except Exception as exc:
                logger.warning("Failed to write stability JSON: %s", exc)

        return saved

    def _build_sub_behavior_summary_entries(self, multi_result, bouts, class_names, *, stability_report=None):
        """Convert per-class clustering result into State Summary entries.

        Returns a list of dicts with ``kind="per_class"`` so
        ``update_state_summary`` can render them with a per-class layout
        (different from the HMM-summary layout).
        """
        bouts_per_class: dict[int, int] = {}
        for b in (bouts or []):
            try:
                cid = int(b.get("class_id", -1))
            except Exception:
                continue
            bouts_per_class[cid] = bouts_per_class.get(cid, 0) + 1

        # Build a class_id → ClassStability lookup for quick attachment.
        stab_by_class = {}
        if stability_report is not None:
            for c in stability_report.per_class:
                stab_by_class[int(c.class_id)] = c

        entries = []
        for cls in multi_result.classes:
            name = cls.class_name or class_names.get(cls.class_id) or f"class_{cls.class_id}"
            entry = {
                "kind": "per_class",
                "class_id": cls.class_id,
                "class_name": name,
                "n_frames": cls.n_frames,
                "n_clusters": cls.n_clusters,
                "n_noise_frames": cls.n_noise_frames,
                "n_bouts": bouts_per_class.get(cls.class_id, 0),
                "skipped": cls.skipped,
                "skip_reason": cls.skip_reason,
                "sub_cluster_labels": list(cls.sub_cluster_labels),
            }
            stab = stab_by_class.get(int(cls.class_id))
            if stab is not None and not stab.skipped:
                entry["stability_mean_ari"] = (
                    None if (stab.mean_ari != stab.mean_ari) else float(stab.mean_ari)
                )
                entry["stability_min_ari"] = (
                    None if (stab.min_ari != stab.min_ari) else float(stab.min_ari)
                )
                entry["stability_is_stable"] = bool(stab.is_stable)
                entry["stability_n_seeds"] = int(stab.n_seeds)
            entries.append(entry)
        return entries

    def _format_sub_behavior_summary(self, multi_result, total_bouts, saved_paths, *, stability_report=None):
        lines = ["Sub-behavior discovery complete.", ""]

        # Build a class_id → ClassStability map for inline annotation.
        stab_by_class = {}
        if stability_report is not None:
            for c in stability_report.per_class:
                stab_by_class[int(c.class_id)] = c

        for cls in multi_result.classes:
            name = cls.class_name or f"class_{cls.class_id}"
            if cls.skipped:
                lines.append(f"  - {name}: skipped ({cls.skip_reason})")
            else:
                line = (
                    f"  - {name}: {cls.n_clusters} sub-cluster(s) "
                    f"({cls.n_noise_frames} noise of {cls.n_frames} frames)"
                )
                stab = stab_by_class.get(int(cls.class_id))
                if stab is not None and not stab.skipped and stab.mean_ari == stab.mean_ari:
                    badge = "STABLE" if stab.is_stable else "UNSTABLE"
                    line += f"  [{badge} ARI={stab.mean_ari:.2f}]"
                lines.append(line)
        lines.append("")
        lines.append(f"Total bouts: {total_bouts}")
        if stability_report is not None:
            lines.append(f"Stability: {stability_report.overall_summary}")
        lines.append("")
        lines.append("Saved:")
        for k, v in (saved_paths or {}).items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def start_sub_behavior_discovery(self):
        """Handler for the 'Run Sub-Behavior Discovery' button."""
        if self.running:
            return messagebox.showwarning(
                "Warning", "Analysis is already running."
            )
        # Reuse the legacy validate_inputs for output folder + group checks;
        # skip the HMM-specific validation (n_states / mix_components etc.).
        try:
            self.validate_inputs(for_export=True)
        except ValueError as exc:
            return messagebox.showerror("Input Error", str(exc))
        params = self._capture_analysis_params()
        self.running = True
        threading.Thread(
            target=self.run_sub_behavior_discovery_analysis,
            args=(params,),
            kwargs={"notify": True},
            daemon=True,
        ).start()

    def export_sub_cluster_clips_action(self):
        """Handler for the 'Export Sub-cluster Clips' button (Commit I).

        Writes one .mp4 per **bout**, organized into per-behavior folders
        for direct use as classifier training input (BehaviorScope-style).
        Layout::

            <output_folder>/sub_cluster_clips/
              forward_locomotion/
                video1__t0__b0001__f12-87.mp4
                video2__t0__b0023__f120-178.mp4
              walking__subcluster_2/
                video1__t0__b0011__f600-650.mp4
              clip_manifest.csv

        Folder names come from ``state_names.json`` when the user has
        named the sub-cluster; otherwise fall back to
        ``{class_name}__subcluster_{sub_id}``.

        Always optional — the user clicks this button only when they
        decide they want clips. A lab that stops at the bouts CSV can
        ignore this button entirely.
        """
        # Guard: a clustering run writes ``last_sub_behavior_result`` from
        # a daemon thread (see run_sub_behavior_discovery_analysis). If the
        # user clicks Export Clips while the run is still in progress, the
        # export thread could read a partially-written result dict and
        # produce a corrupt manifest. Block the export until the run
        # finishes (self.running is reset to False at the end of the
        # worker thread).
        if getattr(self, "running", False):
            return messagebox.showinfo(
                "Export Sub-cluster Clips",
                "A Behavior Clustering run is currently in progress. "
                "Wait for it to finish before exporting clips.",
            )
        result = getattr(self, "last_sub_behavior_result", None)
        if not result:
            return messagebox.showinfo(
                "Export Sub-cluster Clips",
                "Run Behavior Clustering first; there are no clusters to export.",
            )
        try:
            from .clip_export import export_bout_clips  # noqa: PLC0415
        except Exception as exc:
            return messagebox.showerror(
                "Export Sub-cluster Clips",
                f"Could not load clip exporter: {exc}",
            )

        clustered_df = result.get("clustered_df")
        bouts = result.get("bouts") or []
        video_map = result.get("video_path_map") or {}
        output_folder = result.get("output_folder") or self.output_folder.get() or "."
        class_names = result.get("class_names") or {}
        run_id = str(result.get("run_id") or "")
        # ADP-4 Commit G: prefer user-supplied names from state_names.json
        # for folder/clip names. Loader compares run_id to skip stale files.
        state_names = self._load_state_names_for_export(output_folder, result)

        if not bouts:
            return messagebox.showinfo(
                "Export Sub-cluster Clips",
                "No sub-cluster bouts to export. Re-run Sub-Behavior "
                "Discovery (every class may have been all-noise / skipped).",
            )
        if not video_map:
            return messagebox.showwarning(
                "Export Sub-cluster Clips",
                "No source videos are configured. Add videos to your groups "
                "before exporting clips.",
            )

        # Confirm before writing — clip export can be slow and disk-heavy.
        n_real_bouts = sum(
            1 for b in bouts if str(b.get("state", "")) != "-1"
        )
        if n_real_bouts == 0:
            return messagebox.showinfo(
                "Export Sub-cluster Clips",
                "All bouts landed in noise (label -1); nothing to export.",
            )
        proceed = messagebox.askyesno(
            "Export Sub-cluster Clips",
            f"Will write up to {n_real_bouts} clip file(s) to:\n\n"
            f"  {os.path.join(output_folder, 'sub_cluster_clips')}\n\n"
            f"Folders are named after sub-behaviors (state_names.json) so "
            f"the layout drops directly into BehaviorScope / classifier "
            f"training. Proceed?",
        )
        if not proceed:
            return

        clips_root = os.path.join(output_folder, "sub_cluster_clips")

        # Run on a daemon thread so the UI stays responsive.
        def _worker():
            try:
                os.makedirs(clips_root, exist_ok=True)

                def _progress(done, total, message):
                    # Forward into Tab 7's status bar so the user sees per-clip progress.
                    pct = int(50 + (50 * done / max(1, total)))
                    try:
                        self.progress_callback(pct, 100, f"Clips: {message}")
                    except Exception:
                        pass

                rows = export_bout_clips(
                    bouts,
                    video_path_map=video_map,
                    output_dir=clips_root,
                    class_names=class_names,
                    state_names=state_names,
                    overlay=True,
                    detections_df=clustered_df,
                    progress_callback=_progress,
                    run_id=run_id,
                )
                written = sum(1 for r in rows if r.get("status") == "written")
                skipped = sum(1 for r in rows if r.get("status") == "skipped")
                failed = sum(1 for r in rows if r.get("status") == "failed")
                folders = sorted({
                    r.get("behavior_folder") for r in rows
                    if r.get("status") == "written" and r.get("behavior_folder")
                })

                lines = [
                    f"Wrote {written} clip(s) to:",
                    f"  {clips_root}",
                    "",
                    f"  Behaviors: {len(folders)}",
                    f"  Skipped: {skipped}   Failed: {failed}",
                ]
                if folders:
                    preview = ", ".join(folders[:6])
                    if len(folders) > 6:
                        preview += f", … ({len(folders) - 6} more)"
                    lines.append("")
                    lines.append(f"Folders: {preview}")
                lines.append("")
                lines.append(f"Manifest: {os.path.join(clips_root, 'clip_manifest.csv')}")
                self._show_info_async("Export Sub-cluster Clips", "\n".join(lines))
            except Exception as exc:
                logger.error("Clip export failed: %s", exc, exc_info=True)
                self._show_error_async(
                    "Export Sub-cluster Clips",
                    f"Clip export failed: {exc}",
                )
            finally:
                try:
                    self.progress_callback(0, 100, "")
                except Exception:
                    pass

        threading.Thread(target=_worker, daemon=True).start()
        messagebox.showinfo(
            "Export Sub-cluster Clips",
            "Clip export is running in the background. The status bar will "
            "show per-clip progress; a summary dialog will pop up when it "
            "completes.",
        )

    def _enable_clip_export_button_async(self, enabled: bool) -> None:
        """Enable / disable the 'Export Sub-cluster Clips' button safely
        from a worker thread."""
        button = getattr(self, "clip_export_button", None)
        if button is None:
            return
        try:
            self.root.after(
                0,
                lambda b=button, e=enabled: b.configure(
                    state=tk.NORMAL if e else tk.DISABLED
                ),
            )
        except Exception:
            pass

    def _enable_review_button_async(self, enabled: bool) -> None:
        """Enable / disable the 'Review Candidate Sub-Clusters' button."""
        button = getattr(self, "review_candidates_button", None)
        if button is None:
            return
        try:
            self.root.after(
                0,
                lambda b=button, e=enabled: b.configure(
                    state=tk.NORMAL if e else tk.DISABLED
                ),
            )
        except Exception:
            pass

    def _load_state_names_for_export(self, output_folder: str, result: dict) -> dict:
        """Return the most-recent state_names dict for clip filename use.

        Order of precedence:
          1. The in-memory ``result['state_names']`` dict (set by the
             naming dialog's on_close callback in this session).
          2. ``state_names.json`` on disk in the output folder, if its
             ``run_id`` matches the current run (when known). A mismatch
             means the file was saved for a different parameter set —
             cluster ids may not mean the same thing — so we ignore it.
          3. Empty dict — clip filenames fall back to the
             ``{class_name}__subcluster_{sub_id}.mp4`` pattern.
        """
        in_memory = result.get("state_names") if isinstance(result, dict) else None
        if isinstance(in_memory, dict) and in_memory:
            return {str(k): str(v) for k, v in in_memory.items()}

        if not output_folder:
            return {}
        try:
            from .naming_dialog import load_state_names  # noqa: PLC0415

            current_run_id = (
                str(result.get("run_id") or "") if isinstance(result, dict) else ""
            )
            return load_state_names(output_folder, current_run_id)
        except Exception as exc:
            logger.warning("Could not read state_names.json: %s", exc)
            return {}

    def _enable_naming_button_async(self, enabled: bool) -> None:
        """Enable / disable the 'Name Sub-Behaviors' button."""
        button = getattr(self, "naming_button", None)
        if button is None:
            return
        try:
            self.root.after(
                0,
                lambda b=button, e=enabled: b.configure(
                    state=tk.NORMAL if e else tk.DISABLED
                ),
            )
        except Exception:
            pass

    def open_cluster_naming_dialog(self):
        """Launch the naming dialog (ADP-4 Commit G)."""
        result = getattr(self, "last_sub_behavior_result", None)
        if not result:
            return messagebox.showinfo(
                "Name Sub-Behaviors",
                "Run Sub-Behavior Discovery first; there are no sub-clusters "
                "to name yet.",
            )
        bouts = result.get("bouts") or []
        video_map = result.get("video_path_map") or {}
        class_names = result.get("class_names") or {}
        output_folder = result.get("output_folder") or self.output_folder.get() or "."
        signal_report = result.get("signal_report")
        run_id = str(result.get("run_id") or "")

        if not video_map:
            return messagebox.showwarning(
                "Name Sub-Behaviors",
                "No source videos are configured. Add videos to your groups "
                "before naming sub-clusters (the dialog needs frames to "
                "show you).",
            )

        # The user can name names; we receive the final dict via the
        # on_close callback so we can also stash it on the result for
        # downstream consumers (clip export uses it for filenames).
        def _on_names_finalized(state_names: dict) -> None:
            if isinstance(state_names, dict):
                self.last_sub_behavior_result["state_names"] = dict(state_names)

        try:
            from .naming_dialog import ClusterNamingDialog  # noqa: PLC0415

            ClusterNamingDialog(
                self.root,
                bouts=bouts,
                video_path_map=video_map,
                class_names=class_names,
                output_folder=output_folder,
                signal_report=signal_report,
                on_close=_on_names_finalized,
                run_id=run_id,
            )
        except Exception as exc:
            logger.error("Naming dialog failed: %s", exc, exc_info=True)
            messagebox.showerror("Name Sub-Behaviors", str(exc))

    def open_review_candidates_dialog(self):
        """Show the ranked sub-cluster review panel (ADP-4 Commit F)."""
        result = getattr(self, "last_sub_behavior_result", None)
        if not result:
            return messagebox.showinfo(
                "Review Candidate Sub-Clusters",
                "Run Sub-Behavior Discovery first; there are no candidates "
                "to review yet.",
            )
        signal_report = result.get("signal_report")
        if signal_report is None or not signal_report.candidates:
            return messagebox.showinfo(
                "Review Candidate Sub-Clusters",
                "No sub-clusters were produced for review (every class was "
                "either skipped or all frames landed in noise).",
            )

        top = tk.Toplevel(self.root)
        top.title("Review Candidate Sub-Clusters")
        top.minsize(820, 420)
        outer = ttk.Frame(top, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        intro = ttk.Label(
            outer,
            text=(
                "Sub-clusters ranked by candidate strength score. The score "
                "combines cluster size, subject coverage, mean bout duration, "
                "and (when computed) stability ARI. Verdicts are advisory; "
                "you decide what's real. Strongest candidates are at the top."
            ),
            wraplength=760,
            justify=tk.LEFT,
        )
        intro.pack(fill=tk.X, pady=(0, 8))

        summary_text = (
            f"{signal_report.n_likely_real} likely real  ·  "
            f"{signal_report.n_review} need review  ·  "
            f"{signal_report.n_likely_noise} likely noise"
        )
        ttk.Label(outer, text=summary_text, font=INLINE_BOLD_FONT).pack(
            anchor=tk.W, pady=(0, 6),
        )

        cols = (
            "rank", "namespaced_label", "class", "score", "verdict",
            "n_bouts", "n_subjects", "n_frames", "mean_bout", "ari", "notes",
        )
        tree = ttk.Treeview(outer, columns=cols, show="headings", height=14)
        tree.pack(fill=tk.BOTH, expand=True)
        col_widths = {
            "rank": 40, "namespaced_label": 70, "class": 110, "score": 60,
            "verdict": 100, "n_bouts": 60, "n_subjects": 70,
            "n_frames": 70, "mean_bout": 70, "ari": 60, "notes": 220,
        }
        for c in cols:
            tree.heading(c, text=c.replace("_", " "))
            tree.column(c, width=col_widths.get(c, 80), anchor=tk.W)

        # Color-code rows by verdict so the user's eye lands on the right places.
        from .signal_score import (  # noqa: PLC0415
            VERDICT_LIKELY_REAL,
            VERDICT_REVIEW,
            VERDICT_LIKELY_NOISE,
        )
        tree.tag_configure(VERDICT_LIKELY_REAL, background="#e7f6e7")
        tree.tag_configure(VERDICT_REVIEW, background="#fff7e0")
        tree.tag_configure(VERDICT_LIKELY_NOISE, background="#fbe7e7")

        for rank, cand in enumerate(signal_report.candidates, start=1):
            ari_text = (
                "—" if cand.stability_ari is None else f"{cand.stability_ari:.2f}"
            )
            tree.insert(
                "",
                "end",
                values=(
                    rank,
                    cand.namespaced_label,
                    cand.class_name,
                    f"{cand.score:.2f}",
                    cand.verdict,
                    cand.n_bouts,
                    cand.n_subjects,
                    cand.n_frames,
                    f"{cand.mean_bout_frames:.1f}",
                    ari_text,
                    " | ".join(cand.notes) if cand.notes else "",
                ),
                tags=(cand.verdict,),
            )

        # Footer — show where the CSV is and a Close button.
        footer = ttk.Frame(outer)
        footer.pack(fill=tk.X, pady=(8, 0))
        scores_path = (result.get("saved_paths") or {}).get("candidate_scores_csv")
        if scores_path:
            ttk.Label(
                footer,
                text=f"Saved: {scores_path}",
                wraplength=560,
                justify=tk.LEFT,
            ).pack(side=tk.LEFT)
        ttk.Button(footer, text="Close", command=top.destroy).pack(side=tk.RIGHT)

    def _show_info_async(self, title: str, message: str) -> None:
        try:
            self.root.after(0, lambda t=title, m=message: messagebox.showinfo(t, m))
        except Exception:
            pass

    def _show_error_async(self, title: str, message: str) -> None:
        try:
            self.root.after(0, lambda t=title, m=message: messagebox.showerror(t, m))
        except Exception:
            pass

    # =========================================================================
    # End ADP-4 Commit D-MVP
    # =========================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = BehaviorAnalysisApp(root)
    root.mainloop()
