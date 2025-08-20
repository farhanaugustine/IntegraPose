# main.py
# This is a complete, self-contained script integrating all project modules.
# It has been updated to include user-defined skeleton connections and normalization points.
# It can be run directly to launch the Behavior Analysis Toolkit.

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import re
import json
import logging
import sys
import pandas as pd
from matplotlib.figure import Figure
import threading
import numpy as np
import umap
import hdbscan
from collections import defaultdict, Counter
import cv2
from scipy.spatial.distance import pdist, squareform, euclidean
from scipy.special import rel_entr
from sklearn.metrics import silhouette_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from jsonschema import validate, ValidationError, Draft7Validator
from hmmlearn import hmm
from statsmodels.tsa.stattools import grangercausalitytests
from skccm import Embed, CCM
import seaborn as sns


# --- Logging Setup ---
try:
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    log_file_path = os.path.join(script_dir, 'behavior_analysis.log')
    # Clear existing handlers to prevent duplicate logs in interactive environments
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Configure logging
    logging.basicConfig(
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler(sys.stdout)
        ],
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )
    logging.info("--- Logging configured successfully ---")
except Exception as e:
    # Fallback logger if initial setup fails
    fallback_log = os.path.join(os.path.expanduser("~"), "behavior_analysis.log")
    logging.basicConfig(handlers=[logging.FileHandler(fallback_log, mode='w')], level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logging.warning(f"Failed to configure logging due to: {e}. Using fallback log location: {fallback_log}")

logger = logging.getLogger(__name__)

#========================================================================================
# MODULE: data_processing_hdbscan.py
#========================================================================================
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


def extract_frame_number(filename):
    """Extracts frame number from the filename."""
    match = re.search(r'_(\d+)\.txt$', filename)
    if match:
        frame_num = int(match.group(1))
        if frame_num <= 0:
            raise ValueError(f"Invalid frame number {frame_num} in {filename}")
        return frame_num
    raise ValueError(f"Cannot extract frame number from {filename}")


def read_detections(group_dirs, keypoint_names_str, behavior_names_str, use_bbox_hint=True, video_path_map=None):
    """Reads YOLO-pose .txt files, un-normalizes coordinates, and captures video dimensions."""
    keypoint_names = parse_comma_separated(keypoint_names_str)
    N_keypoints = len(keypoint_names)
    validate_names(keypoint_names, N_keypoints, "keypoint")

    behavior_names = parse_comma_separated(behavior_names_str)
    if behavior_names:
        validate_names(behavior_names, None, "behavior")
    behavior_map = {i: name for i, name in enumerate(behavior_names)} if behavior_names else {}

    detections = []
    expected_pose_values = 3 * N_keypoints
    expected_no_bbox = 1 + expected_pose_values + 1
    expected_with_bbox = 1 + 4 + expected_pose_values + 1

    file_format_has_bbox = None
    video_dims_cache = {}

    for group, directories in group_dirs.items():
        for directory in directories:
            if not os.path.isdir(directory):
                raise ValueError(f"Directory {directory} in group {group} does not exist")

            video_source = os.path.basename(directory)
            logger.info(f"Processing group {group}, directory: {directory}")

            video_width, video_height = None, None
            if video_path_map and directory in video_path_map:
                video_path = video_path_map.get(directory)
                if video_path in video_dims_cache:
                    video_width, video_height = video_dims_cache[video_path]
                elif video_path and os.path.exists(video_path):
                    try:
                        cap = cv2.VideoCapture(video_path)
                        if cap.isOpened():
                            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            video_dims_cache[video_path] = (video_width, video_height)
                            logger.info(f"Read dimensions for {os.path.basename(video_path)}: {video_width}x{video_height}")
                        cap.release()
                    except Exception as e:
                        logger.error(f"Could not read dimensions from video {video_path}: {e}")

            if not video_width or not video_height:
                logger.warning(f"Could not determine video dimensions for {directory}. Coordinates will not be un-normalized.")

            for filename in sorted(os.listdir(directory)):
                if not filename.endswith(".txt"): continue
                try:
                    frame_num = extract_frame_number(filename)
                    file_path = os.path.join(directory, filename)

                    with open(file_path, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            parts = line.strip().split()
                            if not parts: continue

                            if file_format_has_bbox is None:
                                if len(parts) == expected_with_bbox:
                                    file_format_has_bbox = True
                                elif len(parts) == expected_no_bbox:
                                    file_format_has_bbox = False
                                else:
                                    raise ValueError(f"Unexpected number of values in {filename} line {line_num}. Check keypoint names and YOLO output format.")

                            expected_len = expected_with_bbox if file_format_has_bbox else expected_no_bbox
                            if len(parts) != expected_len:
                                logger.warning(f"Skipping malformed line {line_num} in {filename}: got {len(parts)} values, expected {expected_len}.")
                                continue

                            class_id = int(parts[0])
                            track_id = int(parts[-1])

                            if file_format_has_bbox:
                                bbox_normalized = [float(p) for p in parts[1:5]]
                                pose_data_normalized = [float(p) for p in parts[5:-1]]
                            else:
                                bbox_normalized = None
                                pose_data_normalized = [float(p) for p in parts[1:-1]]

                            bbox = None
                            if bbox_normalized and video_width and video_height:
                                x_c, y_c, w, h = bbox_normalized
                                bbox = [x_c * video_width, y_c * video_height, w * video_width, h * video_height]

                            pose_data = pose_data_normalized.copy()
                            if video_width and video_height:
                                for i in range(N_keypoints):
                                    pose_data[3*i] = pose_data_normalized[3*i] * video_width
                                    pose_data[3*i+1] = pose_data_normalized[3*i+1] * video_height
                            
                            keypoints = {keypoint_names[i]: (pose_data[3*i], pose_data[3*i+1], pose_data[3*i+2]) for i in range(N_keypoints)}
                            behavior = behavior_map.get(class_id, f"Class ID {class_id}")
                            
                            detections.append({
                                'group': group, 'video_source': video_source, 'directory': directory,
                                'frame': frame_num, 'class_id': class_id, 'behavior': behavior,
                                'track_id': track_id, 'keypoints': keypoints, 'bbox': bbox,
                                'video_width': video_width, 'video_height': video_height
                            })
                except Exception as e:
                    logger.error(f"Failed to process file {filename}: {e}", exc_info=True)

    if not detections:
        logger.warning(f"No valid detections found in any directories.")
        return pd.DataFrame(), keypoint_names, behavior_names

    return pd.DataFrame(detections), keypoint_names, behavior_names


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
    
    # Additional checks: non-negative dimensions, etc.
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

    if len(normalization_ref_points) != 2:
        raise ValueError("Normalization reference points must be a pair of two keypoint names.")
    if not all(kp in keypoint_names for kp in normalization_ref_points):
        raise ValueError(f"Normalization reference points {normalization_ref_points} not found in keypoint names.")

    # Validate ROI if applicable
    if location_mode == 'roi' and roi_definitions:
        try:
            validate_roi_definitions(roi_definitions)
        except ValidationError as e:
            logger.error(f"ROI validation failed: {e}")
            raise
    
    # Calculate group-specific normalization factors
    group_ref_distances = {}
    for group_name, group_df in detections_df.groupby('group'):
        logger.info(f"Calculating reference distances for group: {group_name}")
        pair_distances = defaultdict(list)
        for _, det in group_df.iterrows():
            keypoints = det['keypoints']
            for i in range(N_keypoints):
                for j in range(i + 1, N_keypoints):
                    kp1_name, kp2_name = keypoint_names[i], keypoint_names[j]
                    if keypoints[kp1_name][2] >= conf_threshold and keypoints[kp2_name][2] >= conf_threshold:
                        dist = np.linalg.norm(np.array(keypoints[kp1_name][:2]) - np.array(keypoints[kp2_name][:2]))
                        pair_distances[(kp1_name, kp2_name)].append(dist)

        median_distances = {pair: np.median(dists) for pair, dists in pair_distances.items() if dists}
        
        # Use user-defined reference points
        ref_pair = tuple(sorted(normalization_ref_points))
        body_ref_dist = median_distances.get(ref_pair)

        if body_ref_dist is None or body_ref_dist == 0:
            # Fallback to median of all pairs if user-defined fails
            logger.warning(f"Reference key pair {normalization_ref_points} not found or invalid for group '{group_name}'.")
            if median_distances:
                body_ref_dist = np.median(list(median_distances.values()))
                logger.warning(f"Falling back to median of all keypoint pairs for normalization: {body_ref_dist:.2f}")
            else:
                # Critical failure - raise an error
                raise ValueError(f"Could not determine any valid reference distance for group '{group_name}'. "
                                 "Check keypoint data and confidence threshold. Cannot proceed with feature computation.")

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
        grouped = detections_df.groupby(['video_source', 'frame'])
        for (video_source, frame), frame_df in grouped:
            if len(frame_df) > 1:
                centroids = []
                for _, det in frame_df.iterrows():
                    confident_kps = np.array([det['keypoints'][kp][:2] for kp in keypoint_names if det['keypoints'][kp][2] >= conf_threshold])
                    centroid = np.mean(confident_kps, axis=0) if len(confident_kps) > 0 else np.array([0, 0])
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
                    inter_animal_features[row_idx] = { 'mean_inter_dist': 0.0, 'min_inter_dist': 0.0 }

    detections_df = detections_df.sort_values(['group', 'video_source', 'track_id', 'frame'])
    detections_df['prev_keypoints'] = detections_df.groupby(['group', 'video_source', 'track_id'])['keypoints'].shift(1)
    detections_df['prev_prev_keypoints'] = detections_df.groupby(['group', 'video_source', 'track_id'])['keypoints'].shift(2)

    feature_vectors = []
    for idx, det in detections_df.iterrows():
        keypoints = det['keypoints']
        prev_keypoints = det['prev_keypoints'] if pd.notnull(det['prev_keypoints']) else keypoints
        prev_prev_keypoints = det['prev_prev_keypoints'] if pd.notnull(det['prev_prev_keypoints']) else prev_keypoints
        
        group = det['group']
        body_ref_dist = group_ref_distances.get(group, {}).get('body_ref', 1.0)
        median_distances = group_ref_distances.get(group, {}).get('medians', {})
        vec = []

        # Pairwise distances
        for i in range(N_keypoints):
            for j in range(i + 1, N_keypoints):
                kp1, kp2 = keypoint_names[i], keypoint_names[j]
                if (keypoints[kp1][2] >= conf_threshold and keypoints[kp2][2] >= conf_threshold):
                    dist = np.linalg.norm(np.array(keypoints[kp1][:2]) - np.array(keypoints[kp2][:2]))
                else:
                    dist = median_distances.get(tuple(sorted((kp1, kp2))), 0)
                vec.append(dist)

        # Angles
        for i in range(N_keypoints):
            for j in range(i + 1, N_keypoints):
                for k in range(j + 1, N_keypoints):
                    kp1, kp2, kp3 = keypoint_names[i], keypoint_names[j], keypoint_names[k]
                    if (keypoints[kp1][2] >= conf_threshold and keypoints[kp2][2] >= conf_threshold and keypoints[kp3][2] >= conf_threshold):
                        v1 = np.array(keypoints[kp1][:2]) - np.array(keypoints[kp2][:2])
                        v2 = np.array(keypoints[kp3][:2]) - np.array(keypoints[kp2][:2])
                        norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
                        if norm_v1 > 0 and norm_v2 > 0:
                            cos_angle = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
                            angle = np.arccos(cos_angle) * 180 / np.pi
                        else: angle = 0
                    else: angle = 0
                    vec.append(angle)

        confident_kps = np.array([keypoints[kp][:2] for kp in keypoint_names if keypoints[kp][2] >= conf_threshold])
        centroid = np.mean(confident_kps, axis=0) if len(confident_kps) > 0 else np.array([0, 0])

        # Centroid-normalized keypoint positions
        if det['bbox'] and use_bbox:
            x_c, y_c, w, h = det['bbox']
            for kp in keypoint_names:
                if keypoints[kp][2] >= conf_threshold:
                    norm_x = (keypoints[kp][0] - (x_c - w/2)) / w if w > 0 else 0
                    norm_y = (keypoints[kp][1] - (y_c - h/2)) / h if h > 0 else 0
                else: norm_x, norm_y = 0.5, 0.5
                vec.extend([norm_x, norm_y])
        else:
            for kp in keypoint_names:
                if keypoints[kp][2] >= conf_threshold:
                    norm_x = (keypoints[kp][0] - centroid[0]) / body_ref_dist
                    norm_y = (keypoints[kp][1] - centroid[1]) / body_ref_dist
                else: norm_x, norm_y = 0.0, 0.0
                vec.extend([norm_x, norm_y])

        bbox_center_x, bbox_center_y = (det['bbox'][0], det['bbox'][1]) if det['bbox'] is not None else (centroid[0], centroid[1])

        # Location features
        if location_mode == 'roi' and roi_names:
            roi_vector = [0] * (num_rois + 1)
            found_in_roi = False
            for i, roi_name in enumerate(roi_names):
                x, y, w, h = roi_definitions[roi_name]
                if (x <= bbox_center_x < x + w) and (y <= bbox_center_y < y + h):
                    roi_vector[i] = 1; found_in_roi = True; break
            if not found_in_roi: roi_vector[num_rois] = 1
            vec.extend(roi_vector)
        elif location_mode == 'unsupervised':
            grid_size_val = max(1, location_grid_size)
            norm_grid_x = bbox_center_x / det['video_width'] if det['video_width'] else 0
            norm_grid_y = bbox_center_y / det['video_height'] if det['video_height'] else 0
            vec.extend([norm_grid_x, norm_grid_y])

        # Social interaction features
        if social_mode:
            inter_feats = inter_animal_features.get(idx, {'mean_inter_dist': 0.0, 'min_inter_dist': 0.0})
            vec.append(inter_feats['mean_inter_dist'] / body_ref_dist) # Normalize by body size
            vec.append(inter_feats['min_inter_dist'] / body_ref_dist)

        # Temporal features (velocity and acceleration)
        for kp in keypoint_names:
            curr_pos = np.array(keypoints[kp][:2]) if keypoints[kp][2] >= conf_threshold else np.array([0, 0])
            prev_pos = np.array(prev_keypoints[kp][:2]) if isinstance(prev_keypoints, dict) and prev_keypoints.get(kp, [0,0,0])[2] >= conf_threshold else curr_pos
            prev_prev_pos = np.array(prev_prev_keypoints[kp][:2]) if isinstance(prev_prev_keypoints, dict) and prev_prev_keypoints.get(kp, [0,0,0])[2] >= conf_threshold else prev_pos
            
            velocity = curr_pos - prev_pos
            acceleration = velocity - (prev_pos - prev_prev_pos)
            
            norm_vel = np.linalg.norm(velocity) / body_ref_dist if body_ref_dist > 0 else 0
            norm_acc = np.linalg.norm(acceleration) / body_ref_dist if body_ref_dist > 0 else 0
            
            vec.append(norm_vel); vec.append(norm_acc)

        feature_vectors.append(vec)

    detections_df['feature_vector'] = feature_vectors
    detections_df.drop(columns=['prev_keypoints', 'prev_prev_keypoints'], inplace=True, errors='ignore')
    return detections_df


def group_by_track(detections_list, separate_by_group=True, separate_by_source=True):
    """
    Groups detections from a list of dictionaries by track ID.
    This version resolves duplicate frames by keeping the one with the highest mean confidence.
    """
    logger.debug(f"Grouping {len(detections_list)} detections by track ID")
    if not detections_list:
        logger.warning("Empty detections list provided to group_by_track")
        return {}
    
    tracks = defaultdict(list)
    for det in detections_list:
        if 'track_id' not in det or 'frame' not in det:
            raise ValueError(f"Detection missing track_id or frame")
        
        key_parts = []
        if separate_by_group: key_parts.append(det['group'])
        if separate_by_source: key_parts.append(det['video_source'])
        key_parts.append(det['track_id'])
        key = tuple(key_parts)
        
        tracks[key].append(det)
    
    # Resolve duplicates and sort
    final_tracks = {}
    for key, dets in tracks.items():
        sorted_dets = sorted(dets, key=lambda x: x['frame'])
        
        unique_dets = []
        frame_groups = defaultdict(list)
        for det in sorted_dets:
            frame_groups[det['frame']].append(det)
            
        for frame, frame_dets in sorted(frame_groups.items()):
            if len(frame_dets) > 1:
                logger.warning(f"Duplicate frame {frame} found in track {key}. Resolving by max confidence.")
                # Keep the detection with the highest average keypoint confidence
                best_det = max(frame_dets, key=lambda d: np.mean([kp[2] for kp in d['keypoints'].values()]))
                unique_dets.append(best_det)
            else:
                unique_dets.append(frame_dets[0])
        final_tracks[key] = unique_dets
            
    logger.info(f"Grouped into {len(final_tracks)} tracks after resolving duplicates.")
    return final_tracks


def split_sequences(tracks, max_gap):
    """Splits tracks into sequences based on maximum frame gap."""
    logger.debug(f"Splitting tracks with max_gap={max_gap}")
    if max_gap <= 0:
        raise ValueError(f"max_gap must be positive, got {max_gap}")
        
    sequences, discarded = [], 0
    for track_key, dets in tracks.items():
        if not dets: continue
        current_seq = [dets[0]]
        for i in range(1, len(dets)):
            if dets[i]['frame'] - dets[i-1]['frame'] > max_gap:
                if len(current_seq) > 1: sequences.append(current_seq)
                else: discarded += 1
                current_seq = [dets[i]]
            else:
                current_seq.append(dets[i])
        if len(current_seq) > 1: sequences.append(current_seq)
        else: discarded += 1
        
    logger.info(f"Generated {len(sequences)} sequences, discarded {discarded} single-frame sequences")
    return sequences


def cluster_poses(detections_df, umap_neighbors=30, umap_components=5, min_cluster_size=75):
    """Clusters pose feature vectors using UMAP and HDBSCAN and assigns labels."""
    logger.debug(f"Clustering {len(detections_df)} feature vectors with umap_neighbors={umap_neighbors}, "
                 f"umap_components={umap_components}, min_cluster_size={min_cluster_size}")
    
    feature_vectors = detections_df['feature_vector'].tolist()
    if not feature_vectors:
        raise ValueError("No feature vectors provided for clustering")

    if len(feature_vectors) < min_cluster_size:
        logger.warning(f"Number of samples ({len(feature_vectors)}) is less than min_cluster_size ({min_cluster_size}).")
        min_cluster_size = max(2, len(feature_vectors) // 2) 
        logger.warning(f"Adjusting min_cluster_size to {min_cluster_size} to prevent clustering failure.")
    
    if len(set(len(vec) for vec in feature_vectors)) != 1:
        raise ValueError("Inconsistent feature vector dimensions")
    
    try:
        if len(feature_vectors) <= umap_neighbors:
            logger.warning(f"n_neighbors ({umap_neighbors}) is >= the number of samples ({len(feature_vectors)}). "
                           f"Adjusting n_neighbors to {len(feature_vectors) - 1}.")
            umap_neighbors = max(2, len(feature_vectors) - 1)

        reducer = umap.UMAP(
            n_neighbors=umap_neighbors,
            n_components=umap_components,
            random_state=42,
            metric='euclidean'
        )
        reduced_vectors = reducer.fit_transform(feature_vectors)
        logger.debug(f"Reduced feature vectors to {umap_components} dimensions")
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            cluster_selection_method='leaf'
        )
        clusterer.fit(reduced_vectors)
        
        detections_df['pose_category'] = clusterer.labels_
        
        num_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
        logger.info(f"Clustering completed with {num_clusters} clusters")
        
        return detections_df, clusterer
    except Exception as e:
        logger.error(f"Clustering failed: {str(e)}", exc_info=True)
        if "n_neighbors" in str(e):
             raise ValueError("UMAP failed: The dataset may be too small for the specified 'UMAP Neighbors' parameter. "
                              "Try reducing it in the UI.") from e
        raise


def find_pose_representatives(detections_df):
    """Finds the most representative detection (medoid) for each pose cluster."""
    logger.info("Finding representative detections (medoids) for each pose cluster.")
    
    if 'pose_category' not in detections_df.columns or 'feature_vector' not in detections_df.columns:
        raise ValueError("Detections DataFrame must contain 'pose_category' and 'feature_vector' columns.")
        
    representatives = {}
    
    unique_clusters = sorted([c for c in detections_df['pose_category'].unique() if c != -1])
    
    for cluster_id in unique_clusters:
        logger.debug(f"Finding medoid for pose cluster {cluster_id}...")
        
        cluster_df = detections_df[detections_df['pose_category'] == cluster_id]
        
        if len(cluster_df) < 2:
            representatives[cluster_id] = cluster_df.iloc[0].to_dict()
            logger.debug(f"Cluster {cluster_id} has only one member.")
            continue
            
        feature_vectors = np.array(cluster_df['feature_vector'].tolist())
        
        dist_matrix = squareform(pdist(feature_vectors, 'euclidean'))
        
        sum_of_dists = np.sum(dist_matrix, axis=1)
        medoid_index_in_cluster = np.argmin(sum_of_dists)
        
        representative_detection = cluster_df.iloc[medoid_index_in_cluster]
        
        representatives[cluster_id] = representative_detection.to_dict()
        logger.info(f"Found representative for cluster {cluster_id} (Frame: {representative_detection['frame']}, Source: {representative_detection['video_source']})")
        
    if not representatives:
        logger.warning("Could not find any pose representatives. This may happen if no clusters were formed.")
        
    return representatives


def aggregate_into_bouts(detections_df, max_frame_gap, min_bout_duration):
    """Aggregates frame-by-frame detections into behavioral bouts."""
    logger.info(f"Aggregating detections into bouts with max_gap={max_frame_gap} and min_duration={min_bout_duration}...")
    if detections_df.empty:
        logger.warning("Cannot aggregate bouts from an empty DataFrame.")
        return []

    detections_df = detections_df.sort_values(by=['group', 'video_source', 'track_id', 'frame'])

    all_bouts = []
    
    for (group, video_source, track_id), track_df in detections_df.groupby(['group', 'video_source', 'track_id']):
        if track_df.empty:
            continue

        track_df['frame_diff'] = track_df['frame'].diff()
        track_df['class_diff'] = (track_df['class_id'] != track_df['class_id'].shift()).astype(int)
        
        bout_start_condition = (track_df['class_diff'] > 0) | (track_df['frame_diff'] > max_frame_gap)
        bout_ids = bout_start_condition.cumsum()

        for bout_id, bout_df in track_df.groupby(bout_ids):
            if bout_df.empty:
                continue

            duration = len(bout_df)
            if duration >= min_bout_duration:
                # --- NEW: Calculate the mean location of the bout ---
                bout_centroids = []
                for _, det in bout_df.iterrows():
                    if det['bbox'] is not None:
                        bout_centroids.append(det['bbox'][:2]) # Use bbox center if available
                    elif det['keypoints']:
                        confident_kps = np.array([kp[:2] for kp in det['keypoints'].values() if kp[2] > 0.1])
                        if len(confident_kps) > 0:
                            bout_centroids.append(np.mean(confident_kps, axis=0))
                
                mean_bout_location = np.mean(bout_centroids, axis=0) if bout_centroids else (0, 0)
                # --- END NEW ---

                start_frame = bout_df['frame'].iloc[0]
                end_frame = bout_df['frame'].iloc[-1]
                
                feature_vectors_seq = list(bout_df['feature_vector'])
                mean_feature_vector = np.mean(np.array(feature_vectors_seq), axis=0)

                bout = {
                    'group': group,
                    'video_source': video_source,
                    'track_id': track_id,
                    'behavior': bout_df['behavior'].iloc[0],
                    'class_id': bout_df['class_id'].iloc[0],
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'duration_frames': (end_frame - start_frame + 1),
                    'detection_count': duration,
                    'feature_vectors': feature_vectors_seq,
                    'mean_feature_vector': mean_feature_vector,
                    'mean_location': mean_bout_location # Store the new location data
                }
                all_bouts.append(bout)

    logger.info(f"Aggregated {len(detections_df)} detections into {len(all_bouts)} bouts.")
    return all_bouts


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
        video_source_name = os.path.basename(source_dir)
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


#========================================================================================
# MODULE: hmm_utils_revised.py
#========================================================================================
def train_hmm(sequences, n_states=5, model_type='categorical', use_poses=True, progress_callback=None):
    """
    Trains an HMM on sequences of observations.
    
    NOTE: The 'hmmlearn' library's fit method is a blocking call. 
    A per-iteration progress update is not available. The callback will be used 
    before and after the main training step.
    """
    logger.info(f"Training HMM with n_states={n_states}, type={model_type}, use_poses={use_poses}")
    if not sequences:
        logger.warning("No sequences provided for HMM training")
        return None, None

    all_observations = []
    lengths = []
    for seq in sequences:
        if not seq: continue
        # For GaussianHMM, observations are the feature vectors themselves.
        if model_type == 'gaussian':
            obs = [det['feature_vector'] for det in seq if 'feature_vector' in det]
        # For CategoricalHMM, observations are discrete category labels.
        else:
            obs = [det['pose_category'] for det in seq if 'pose_category' in det]
        
        if obs:
            all_observations.extend(obs)
            lengths.append(len(obs))

    if not all_observations:
        logger.warning("No valid observations found for HMM training.")
        return None, None

    # For CategoricalHMM, observations need to be mapped to integers.
    if model_type == 'categorical':
        unique_obs = sorted(list(set(all_observations)))
        id_map = {obs: i for i, obs in enumerate(unique_obs)}
        obs_array = np.array([id_map[obs] for obs in all_observations]).reshape(-1, 1)
        model = hmm.CategoricalHMM(n_components=n_states, n_iter=100, tol=1e-4, random_state=42)
    
    # For GaussianHMM, observations are continuous vectors.
    elif model_type == 'gaussian':
        obs_array = np.vstack(all_observations)
        id_map = {}  # No id_map needed for Gaussian HMM
        model = hmm.GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=100, tol=1e-4, random_state=42)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    if progress_callback:
        progress_callback(0, 1, f"Fitting HMM with {len(obs_array)} data points...")

    model.fit(obs_array, lengths)
    
    if progress_callback:
        progress_callback(1, 1, "HMM fitting complete.")

    logger.info("HMM training complete")
    return model, id_map


def infer_state_sequences(model, sequences, id_map, use_poses=True):
    """Infers the most likely state sequences using the trained HMM."""
    inferred_sequences = []
    is_gaussian = isinstance(model, hmm.GaussianHMM)

    for seq in sequences:
        if not seq: continue
        
        if is_gaussian:
            obs = [det['feature_vector'] for det in seq if 'feature_vector' in det]
            if not obs: continue
            obs_array = np.vstack(obs)
        else: # Categorical
            obs = [id_map.get(det['pose_category'], -1) for det in seq if 'pose_category' in det]
            if not obs or any(o == -1 for o in obs): continue
            obs_array = np.array(obs).reshape(-1, 1)

        try:
            states = model.predict(obs_array)
            inferred_sequences.append(states)
        except Exception as e:
            logger.warning(f"Could not predict states for a sequence: {e}")
            
    return inferred_sequences


def compute_transition_matrix(model):
    """Computes the transition matrix from the HMM model."""
    return model.transmat_


def compute_emission_matrix(model, id_map, behavior_names=None, use_poses=True):
    """Computes the emission matrix as a DataFrame. Returns None for GaussianHMM."""
    if isinstance(model, hmm.GaussianHMM):
        logger.info("Emission matrix is not applicable for GaussianHMM; represents continuous distributions.")
        return None
        
    if use_poses:
        # Ensure columns match the id_map used for training
        num_emissions = len(id_map)
        sorted_labels = sorted(id_map.keys())
        labels = [f"Pose Cluster {i}" for i in sorted_labels]

    else:
        labels = behavior_names or [f"Behavior {i}" for i in range(len(id_map))]
        
    emission_df = pd.DataFrame(model.emissionprob_, columns=labels)
    emission_df.index = [f"State {i}" for i in range(model.n_components)]
    return emission_df


def assign_state_labels(emission_df):
    """Assigns descriptive labels to HMM states based on emission probabilities."""
    if emission_df is None:
        return {i: {'label': f"Latent State {i}", 'prob': 1.0} for i in range(len(emission_df.index))}

    state_labels = {}
    for i, state_name in enumerate(emission_df.index):
        max_prob_col_idx = emission_df.loc[state_name].argmax()
        label = emission_df.columns[max_prob_col_idx]
        prob = emission_df.iloc[i, max_prob_col_idx]
        state_labels[i] = {'label': label, 'prob': prob}
    return state_labels


def compare_groups(group_hmms, method='log_likelihood', sequences_by_group=None):
    """
    Compares HMM models between groups using specified method.
    
    Methods:
    - 'log_likelihood': (Recommended) Computes the log-likelihood of one group's sequences 
                        under another group's model. Lower score is better fit.
    - 'kl_divergence': Compares the transition matrices using KL Divergence.
    """
    groups = list(group_hmms.keys())
    if len(groups) < 2:
        return {}

    comparison_results = pd.DataFrame(index=groups, columns=groups, dtype=float)

    for g1_name in groups:
        for g2_name in groups:
            model_g2 = group_hmms[g2_name]['model']
            
            if method == 'log_likelihood':
                if not sequences_by_group:
                    raise ValueError("Sequence data is required for log_likelihood comparison.")
                
                # Prepare sequences from group 1 for scoring
                seqs_g1 = sequences_by_group[g1_name]
                is_gaussian = isinstance(model_g2, hmm.GaussianHMM)
                
                observations = []
                lengths = []
                for seq in seqs_g1:
                    if is_gaussian:
                        obs_list = [det['feature_vector'] for det in seq if 'feature_vector' in det]
                    else: # Categorical
                        id_map_g2 = group_hmms[g2_name]['id_map']
                        obs_list = [id_map_g2.get(det['pose_category']) for det in seq if 'pose_category' in det]
                        # Handle cases where a pose in G1 doesn't exist in G2's map
                        if any(o is None for o in obs_list): continue

                    if obs_list:
                        observations.extend(obs_list)
                        lengths.append(len(obs_list))
                
                if not observations:
                    score = np.nan
                else:
                    if is_gaussian:
                        obs_array = np.vstack(observations)
                    else:
                        obs_array = np.array(observations).reshape(-1, 1)
                    score = model_g2.score(obs_array, lengths)
                comparison_results.loc[g1_name, g2_name] = score

            elif method == 'kl_divergence':
                # Note: This only compares transition probabilities, not the full model.
                trans_g1 = group_hmms[g1_name]['transmat']
                trans_g2 = group_hmms[g2_name]['transmat']
                
                if trans_g1.shape != trans_g2.shape:
                    comparison_results.loc[g1_name, g2_name] = np.nan
                    continue

                kl_div = np.sum([rel_entr(trans_g1[k], trans_g2[k]).sum() for k in range(trans_g1.shape[0])])
                comparison_results.loc[g1_name, g2_name] = kl_div
                
    return comparison_results

#========================================================================================
# MODULE: analysis.py
#========================================================================================
def draw_skeleton(frame, keypoints, keypoint_names, connections, conf_threshold=0.3):
    """
    Draws keypoint skeletons on a frame.
    Now uses a flexible 'connections' parameter.
    """
    # Draw keypoints first
    for kp_name, (x, y, conf) in keypoints.items():
        if conf > conf_threshold:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1) # Green dots for keypoints

    # Draw connections
    for kp1_name, kp2_name in connections:
        if kp1_name in keypoints and kp2_name in keypoints:
            kp1_x, kp1_y, kp1_conf = keypoints[kp1_name]
            kp2_x, kp2_y, kp2_conf = keypoints[kp2_name]
            if kp1_conf > conf_threshold and kp2_conf > conf_threshold:
                cv2.line(frame, (int(kp1_x), int(kp1_y)), (int(kp2_x), int(kp2_y)), (255, 0, 0), 2) # Blue lines for skeleton
                
    return frame

def save_pose_cluster_images(output_folder, representatives, video_path_map, keypoint_names, skeleton_connections, conf_threshold=0.3):
    """Saves an image for the most representative pose of each cluster."""
    logger.info("Saving representative pose cluster images...")
    
    if not representatives:
        logger.warning("No pose representatives found to generate images.")
        return

    pose_img_dir = os.path.join(output_folder, "pose_cluster_representatives")
    os.makedirs(pose_img_dir, exist_ok=True)

    for cluster_id, rep_data in representatives.items():
        try:
            pose_dir = rep_data['directory']
            video_path = video_path_map.get(pose_dir)
            
            if not video_path or not os.path.exists(video_path):
                logger.warning(f"Video not found for pose directory '{pose_dir}'. Skipping image for cluster {cluster_id}.")
                continue
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.warning(f"Could not open video: {video_path}")
                continue
            
            frame_number = rep_data['frame']
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
            ret, frame = cap.read()
            
            if ret:
                frame_with_skeleton = draw_skeleton(frame.copy(), rep_data['keypoints'], keypoint_names, connections=skeleton_connections, conf_threshold=conf_threshold)
                img_path = os.path.join(pose_img_dir, f"pose_cluster_{cluster_id}_representative.png")
                cv2.imwrite(img_path, frame_with_skeleton)
            
            cap.release()
        except Exception as e:
            logger.error(f"Failed to generate image for cluster {cluster_id}: {e}", exc_info=True)

def generate_location_report(detections_df, location_mode, output_folder, grid_size=50, roi_definitions=None):
    """
    Generates a detailed CSV report linking clusters to locations.
    Relies on video dimensions stored in the detections_df.
    """
    if location_mode == 'none' or 'cluster_label' not in detections_df.columns:
        return None

    logger.info(f"Generating location-based behavior report for mode: {location_mode}")
    report_data = []
    location_counts = defaultdict(lambda: defaultdict(int))
    
    valid_loc_df = detections_df.dropna(subset=['video_width', 'video_height'])

    for _, row in valid_loc_df.iterrows():
        cluster = row['cluster_label']
        if cluster == -1: continue

        centroid = np.mean(np.array([kp[:2] for kp in row['keypoints'].values() if kp[2] > 0.1]), axis=0) if row['keypoints'] else (0,0)
        center_x, center_y = (row['bbox'][0], row['bbox'][1]) if row['bbox'] is not None else (centroid[0], centroid[1])
        
        location_id = 'N/A'
        if location_mode == 'unsupervised':
            video_width, video_height = row['video_width'], row['video_height']
            grid_cols = int(np.ceil(video_width / grid_size))
            col = int(center_x // grid_size)
            row_idx = int(center_y // grid_size)
            location_id = f"cell_{row_idx * grid_cols + col}"
        
        elif location_mode == 'roi' and roi_definitions:
            location_id = 'outside_roi'
            for name, (x, y, w, h) in roi_definitions.items():
                if (x <= center_x < x + w) and (y <= center_y < y + h):
                    location_id = name
                    break
        
        report_data.append({'frame': row['frame'], 'cluster_label': cluster, 'location_id': location_id, 'video_source': row['video_source']})
        location_counts[location_id][cluster] += 1

    if not report_data:
        logger.warning("No valid location data was generated for the report.")
        return None

    report_df = pd.DataFrame(report_data)
    report_path = os.path.join(output_folder, "location_behavior_report.csv")
    report_df.to_csv(report_path, index=False)
    logger.info(f"Location behavior report saved to {report_path}")

    if not report_df.empty:
        # Summary statistics
        dominant_cluster_per_loc = report_df.groupby('location_id')['cluster_label'].agg(lambda x: x.value_counts().index[0]).reset_index()
        dominant_cluster_per_loc.rename(columns={'cluster_label': 'dominant_cluster'}, inplace=True)
        
        preferred_loc_per_cluster = report_df.groupby('cluster_label')['location_id'].agg(lambda x: x.value_counts().index[0]).reset_index()
        preferred_loc_per_cluster.rename(columns={'location_id': 'preferred_location'}, inplace=True)

        spatial_dispersion = report_df.groupby('cluster_label')['location_id'].nunique().reset_index()
        spatial_dispersion.rename(columns={'location_id': 'spatial_dispersion_score'}, inplace=True)
        
        summary_df = pd.merge(preferred_loc_per_cluster, spatial_dispersion, on='cluster_label')
        
        summary_path = os.path.join(output_folder, "location_statistics_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Location statistics summary saved to {summary_path}")

    return location_counts


def plot_dominant_cluster_map(ax, counts, bg_frame, location_mode, behavior_names=None, grid_size=50, roi_definitions=None):
    """
    Plots a map where each location is colored by its most frequent cluster.
    """
    ax.imshow(cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Dominant Behavior Map (Mode: {location_mode})")
    ax.axis('off')

    if not counts: return

    # Create a mapping from class ID to behavior name
    behavior_map = {i: name for i, name in enumerate(behavior_names)} if behavior_names else {}

    all_clusters = sorted(list(set(cluster for loc_data in counts.values() for cluster in loc_data.keys())))
    if not all_clusters: return
    
    cmap_name = 'viridis' if len(all_clusters) > 20 else 'tab20'
    cmap = plt.cm.get_cmap(cmap_name, len(all_clusters))
    color_map = {cluster_id: cmap(i) for i, cluster_id in enumerate(all_clusters)}

    if location_mode == 'unsupervised':
        video_height, video_width, _ = bg_frame.shape
        grid_cols = int(np.ceil(video_width / grid_size))
        
        for loc_id, cluster_counts in counts.items():
            if not cluster_counts: continue
            dominant_cluster = max(cluster_counts, key=cluster_counts.get)
            
            try:
                cell_num = int(loc_id.split('_')[1])
            except (IndexError, ValueError):
                logger.warning(f"Could not parse cell number from location_id: '{loc_id}'. Skipping.")
                continue

            row = cell_num // grid_cols
            col = cell_num % grid_cols
            
            rect = patches.Rectangle((col * grid_size, row * grid_size), grid_size, grid_size, 
                                     linewidth=0, facecolor=color_map[dominant_cluster], alpha=0.6)
            ax.add_patch(rect)

    elif location_mode == 'roi' and roi_definitions:
        for name, (x, y, w, h) in roi_definitions.items():
            if name in counts and counts[name]:
                dominant_cluster = max(counts[name], key=counts[name].get)
                # Use behavior name for the text label inside the ROI
                label_text = behavior_map.get(dominant_cluster, f"C{dominant_cluster}")
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='white', facecolor=color_map[dominant_cluster], alpha=0.6)
                ax.add_patch(rect)
                ax.text(x + w / 2, y + h / 2, label_text, color='white', ha='center', va='center', fontsize=8, weight='bold')

    # Use behavior name for the legend
    legend_patches = [patches.Patch(color=color_map[cid], label=behavior_map.get(cid, f'Cluster {cid}')) for cid in all_clusters]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', title="Behaviors")


def save_results(group_hmms, output_folder, comparison_method, keypoint_names, skeleton_connections,
                 pose_representatives=None, video_path_map=None, comparison_results=None, 
                 heatmap_cmap='YlGnBu', figsize=(10, 8), annotate_heatmaps=True):
    """Saves HMM analysis results and visual/tabular keys to the specified folder."""
    logger.info(f"Saving results to: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)
    
    # This part is fine and does not need changes
    if pose_representatives and video_path_map:
        save_pose_cluster_images(output_folder, pose_representatives, video_path_map, keypoint_names, skeleton_connections)

    for group_key, data in group_hmms.items():
        state_labels_map = data['state_labels']
        state_labels_list = [v['label'] for k, v in sorted(state_labels_map.items())]
        
        # Save CSV data (no changes here)
        transmat_df = pd.DataFrame(data['transmat'], index=state_labels_list, columns=state_labels_list)
        transmat_df.to_csv(os.path.join(output_folder, f'{group_key}_transition_matrix.csv'))

        if data['emission_df'] is not None:
            data['emission_df'].to_csv(os.path.join(output_folder, f'{group_key}_emission_matrix.csv'))
        
        # --- CORRECTED PLOT SAVING ---
        # Create a new Figure object for each plot to avoid thread conflicts
        
        # Save Transition Heatmap
        fig_trans = Figure(figsize=figsize, constrained_layout=True)
        ax_trans = fig_trans.add_subplot(111)
        plot_transition_heatmap(data['transmat'], state_labels_list, ax_trans, f'Transition Matrix - {group_key}', cmap=heatmap_cmap, annotate=annotate_heatmaps)
        fig_trans.savefig(os.path.join(output_folder, f'{group_key}_transition_heatmap.png'))

        # Save Emission Heatmap
        if data['emission_df'] is not None:
            fig_emis = Figure(figsize=figsize, constrained_layout=True)
            ax_emis = fig_emis.add_subplot(111)
            plot_emission_matrix(data['emission_df'], state_labels_list, ax_emis, f'Emission Matrix - {group_key}', cmap=heatmap_cmap, annotate=annotate_heatmaps)
            fig_emis.savefig(os.path.join(output_folder, f'{group_key}_emission_heatmap.png'))

    # Save Comparison Heatmap (if it exists)
    if isinstance(comparison_results, pd.DataFrame) and not comparison_results.empty:
        comparison_results.to_csv(os.path.join(output_folder, f'comparison_{comparison_method}.csv'))
        
        fig_comp = Figure(figsize=figsize, constrained_layout=True)
        ax_comp = fig_comp.add_subplot(111)
        plot_comparison_heatmap(comparison_results, comparison_method, ax_comp, cmap='viridis', annotate=annotate_heatmaps)
        fig_comp.savefig(os.path.join(output_folder, f'comparison_{comparison_method}_heatmap.png'))

    logger.info("Completed saving all results.")

def plot_transition_heatmap(transmat, state_labels, ax, title, cmap='YlGnBu', annotate=True):
    """Plots a heatmap of the transition matrix."""
    sns.heatmap(transmat, xticklabels=state_labels, yticklabels=state_labels, 
                annot=annotate, fmt=".2f", cmap=cmap, ax=ax, linewidths=.5)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel("To State", fontsize=12)
    ax.set_ylabel("From State", fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)

def plot_emission_matrix(emission_df, state_labels, ax, title, cmap='Greens', annotate=True):
    """Plots a heatmap of the emission matrix."""
    emission_df.index = state_labels
    sns.heatmap(emission_df, annot=annotate, fmt=".2f", cmap=cmap, ax=ax, linewidths=.5)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel("Observed Feature" if 'Feature' in str(emission_df.columns[0]) else "Observed State (Pose/Behavior)", fontsize=12)
    ax.set_ylabel("Latent State", fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)

def plot_comparison_heatmap(df, method, ax, cmap='viridis', annotate=True):
    """Plots a heatmap for the comparison results."""
    title = f"Group Comparison - {method}"
    fmt = ".2f"
    if method == 'log_likelihood':
        title = "Log-Likelihood Comparison\n(Rows: Sequence Source, Cols: Model Source)"
        fmt = ".2e"
    elif method == 'kl_divergence':
        title = "KL Divergence of Transition Matrices"
    elif "Latent" in method:
        title = method # Use the title passed from the VAE comparison
        
    sns.heatmap(df, annot=annotate, fmt=fmt, cmap=cmap, ax=ax, linewidths=.5)
    ax.set_title(title, fontsize=14, weight='bold')
    xlabel = "Model Source"
    ylabel = "Sequence Source"
    if "Latent" in method:
        xlabel = "Group"
        ylabel = "Group"
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)

def compare_groups_vae(encoded_sequences_by_group):
    """
    Compares groups in the VAE latent space by calculating the Euclidean distance between their centroids.
    """
    logger.info("Comparing group distributions in VAE latent space.")
    if not encoded_sequences_by_group or len(encoded_sequences_by_group) < 2:
        logger.warning("VAE comparison requires at least two groups with encoded sequences.")
        return pd.DataFrame()

    group_centroids = {}
    for group, sequences in encoded_sequences_by_group.items():
        if not sequences:
            logger.warning(f"No sequences found for group '{group}' in VAE comparison.")
            continue
        all_points = np.concatenate([s for s in sequences if s.ndim == 2 and s.shape[0] > 0], axis=0)
        if all_points.size == 0:
            logger.warning(f"No valid data points for group '{group}' after concatenation.")
            continue
        group_centroids[group] = np.mean(all_points, axis=0)

    group_names = sorted(list(group_centroids.keys()))
    distances = pd.DataFrame(index=group_names, columns=group_names, dtype=float)

    for g1 in group_names:
        for g2 in group_names:
            if g1 == g2:
                dist = 0.0
            else:
                centroid1 = group_centroids.get(g1)
                centroid2 = group_centroids.get(g2)
                if centroid1 is None or centroid2 is None:
                    dist = np.nan
                else:
                    dist = euclidean(centroid1, centroid2)
            distances.loc[g1, g2] = dist
            
    return distances

#========================================================================================
# MODULE: video_utils.py
#========================================================================================
def draw_on_frame(frame, text_overlay, rows):
    """Draws keypoints, bounding boxes, and text overlays on a single frame for multiple detections (animals)."""
    # Draw for each detection (animal) in the frame
    for row in rows:
        # Draw bounding box if it exists
        if 'bbox' in row and row['bbox'] is not None and len(row['bbox']) == 4:
            x_c, y_c, w, h = row['bbox']
            x1, y1 = int(x_c - w / 2), int(y_c - h / 2)
            x2, y2 = int(x_c + w / 2), int(y_c + h / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Optional: Label track ID near bbox
            cv2.putText(frame, f"ID: {row.get('track_id', 'N/A')}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw keypoints
        if 'keypoints' in row and row['keypoints'] is not None:
            for kp_name, (x, y, conf) in row['keypoints'].items():
                if conf > 0.1:
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    # Draw main text overlay (e.g., cluster info) once per frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text_overlay, (15, 40), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def create_cluster_videos(output_folder, all_detections_df, behavioral_labels, video_path_map):
    """
    Generates a video for each behavioral cluster (from VAE/LSTM analysis).
    This is optimized to minimize file I/O by grouping frames per video and supports multi-animal frames.
    """
    logger.info("Starting video generation for VAE/LSTM behavioral clusters.")
    
    if len(all_detections_df) != len(behavioral_labels):
        raise ValueError(f"Mismatch between detections ({len(all_detections_df)}) and labels ({len(behavioral_labels)}).")

    df = all_detections_df.copy()
    df['cluster_label'] = behavioral_labels

    grouped_by_cluster = df.groupby('cluster_label')
    
    for cluster_id, cluster_df in grouped_by_cluster:
        if cluster_id == -1:
            logger.info("Skipping noise points (cluster -1).")
            continue

        logger.info(f"Processing video for cluster {cluster_id}...")
        output_video_path = os.path.join(output_folder, f"behavior_cluster_{cluster_id}.mp4")
        
        # Group by video and frame for multi-animal handling
        frames_by_video_frame = defaultdict(lambda: defaultdict(list))
        for _, row in cluster_df.iterrows():
            pose_dir = row['directory']
            if pose_dir in video_path_map:
                video_file = video_path_map[pose_dir]
                frames_by_video_frame[video_file][row['frame']].append(row.to_dict())
        
        if not frames_by_video_frame:
            logger.warning(f"No valid video frames found for cluster {cluster_id}. Skipping video creation.")
            continue

        video_writer = None
        try:
            for video_path in sorted(frames_by_video_frame.keys()):  # Process videos in order
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    logger.warning(f"Could not open video file: {video_path}. Skipping.")
                    continue

                if video_writer is None:
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30
                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

                # Sort frames for this video
                frame_nums = sorted(frames_by_video_frame[video_path].keys())
                
                for frame_number in frame_nums:
                    rows = frames_by_video_frame[video_path][frame_number]  # All detections for this frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
                    ret, frame = cap.read()

                    if ret:
                        text = f"Behavioral Cluster: {cluster_id} | Frame: {frame_number}"
                        frame_with_overlay = draw_on_frame(frame, text, rows)
                        video_writer.write(frame_with_overlay)
                
                cap.release()
        finally:
            if video_writer:
                video_writer.release()
                logger.info(f"Finished creating video: {output_video_path}")

    logger.info("Finished generating all cluster videos.")


def create_location_specific_videos(output_folder, detections_df, video_path_map, location_mode, 
                                    grid_size=50, roi_definitions=None, min_frames_for_video=5):
    """
    Generates short video montages for specific behavior-location pairs, supporting multi-animal frames.
    """
    if location_mode == 'none' or 'cluster_label' not in detections_df.columns:
        logger.info("Location mode is 'none' or no cluster labels found. Skipping location-specific video generation.")
        return

    logger.info("Starting location-specific video generation...")
    video_clips_dir = os.path.join(output_folder, "location_specific_videos")
    os.makedirs(video_clips_dir, exist_ok=True)
    
    # Augment DataFrame with location_id
    temp_df = detections_df.copy()
    video_width = temp_df['video_width'].max()
    video_height = temp_df['video_height'].max()
    if pd.isna(video_width): video_width = 1280
    if pd.isna(video_height): video_height = 720
    
    def get_loc_id(row):
        centroid = np.mean(np.array([kp[:2] for kp in row['keypoints'].values() if kp[2] > 0.1]), axis=0) if row['keypoints'] else (0,0)
        center_x, center_y = (row['bbox'][0], row['bbox'][1]) if row['bbox'] else (centroid[0], centroid[1])
        
        if location_mode == 'unsupervised':
            grid_cols = int(np.ceil(video_width / grid_size))
            col = int(center_x // grid_size)
            row_idx = int(center_y // grid_size)
            return f"cell_{row_idx * grid_cols + col}"
        elif location_mode == 'roi' and roi_definitions:
            for name, (x, y, w, h) in roi_definitions.items():
                if (x <= center_x < x + w) and (y <= center_y < y + h):
                    return name
            return 'outside_roi'
        return 'N/A'

    temp_df['location_id'] = temp_df.apply(get_loc_id, axis=1)

    # Group by cluster and location to create video for each pair
    for (cluster, loc_id), group_df in temp_df.groupby(['cluster_label', 'location_id']):
        if cluster == -1 or len(group_df) < min_frames_for_video:
            continue

        logger.info(f"Processing video for Cluster {cluster} in Location '{loc_id}'...")
        safe_loc_id = "".join(c for c in loc_id if c.isalnum() or c in ('_','-')).rstrip()
        output_video_path = os.path.join(video_clips_dir, f"cluster_{cluster}_loc_{safe_loc_id}.mp4")

        # Group by video and frame for multi-animal handling
        frames_by_video_frame = defaultdict(lambda: defaultdict(list))
        for _, row in group_df.iterrows():
            video_file = video_path_map.get(row['directory'])
            if video_file:
                frames_by_video_frame[video_file][row['frame']].append(row.to_dict())

        video_writer = None
        try:
            for video_path in sorted(frames_by_video_frame.keys()):  # Process videos in order
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened(): continue

                if video_writer is None:
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30
                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

                # Sort frames for this video
                frame_nums = sorted(frames_by_video_frame[video_path].keys())
                
                for frame_number in frame_nums:
                    rows = frames_by_video_frame[video_path][frame_number]  # All detections for this frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
                    ret, frame = cap.read()
                    if ret:
                        text = f"Cluster: {cluster}, Loc: {loc_id}, Frame: {frame_number}"
                        frame_with_overlay = draw_on_frame(frame, text, rows)
                        video_writer.write(frame_with_overlay)
                cap.release()
        finally:
            if video_writer:
                video_writer.release()
                logger.info(f"Finished creating video: {output_video_path}")

#========================================================================================
# MODULE: vae_lstm_utils.py
#========================================================================================
class VAE(nn.Module):
    """Variational Autoencoder model for pose feature encoding."""
    def __init__(self, input_dim, intermediate_dim, latent_dim, dropout_rate=0.2):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(intermediate_dim, intermediate_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.fc_mu = nn.Linear(intermediate_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(intermediate_dim // 2, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(intermediate_dim // 2, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(intermediate_dim, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    """VAE loss function (reconstruction + KL divergence)."""
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

def train_vae(feature_vectors, latent_dim=8, intermediate_dim=128, epochs=100, dropout_rate=0.2, batch_size=64, device='cpu', progress_callback=None):
    """Trains the VAE model on feature vectors."""
    logger.info(f"Training VAE with latent_dim={latent_dim}, epochs={epochs}, device={device}")
    input_dim = len(feature_vectors[0])
    model = VAE(input_dim, intermediate_dim, latent_dim, dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    dataset = TensorDataset(torch.tensor(feature_vectors, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (batch,) in enumerate(dataloader):
            batch = batch.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            loss = vae_loss(recon_batch, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader.dataset)
        logger.info(f"VAE Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        if progress_callback:
            progress_callback(epoch + 1, epochs, f"VAE training: Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model, avg_loss

def encode_sequences(vae, sequences, device='cpu'):
    """Encodes sequences of detections to latent representations using VAE."""
    vae.eval()
    encoded_sequences = []
    with torch.no_grad():
        for seq in sequences:
            feature_vectors = [det['feature_vector'] for det in seq]
            tensor = torch.tensor(feature_vectors, dtype=torch.float32).to(device)
            mu, _ = vae.encode(tensor)
            encoded_sequences.append(mu.cpu().numpy())
    return encoded_sequences

class LSTM(nn.Module):
    """LSTM model for sequence classification or prediction."""
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout_rate=0.2):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, input_dim)  # Bidirectional, so *2

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # Predict based on the last time step's output

def train_lstm(encoded_sequences, latent_dim, num_layers=2, epochs=50, dropout_rate=0.2, device='cpu', progress_callback=None):
    """
    Trains the LSTM model on encoded sequences.
    This function now returns the trained model.
    """
    logger.info(f"Training LSTM with num_layers={num_layers}, epochs={epochs}, device={device}")
    input_dim = encoded_sequences[0].shape[1] if encoded_sequences else latent_dim
    model = LSTM(input_dim, latent_dim, num_layers, dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Prepare data by padding sequences to the same length
    max_len = max(len(seq) for seq in encoded_sequences)
    # Create input (X) and target (y) tensors. The target is the next frame in the sequence.
    # We will predict the last element of each sequence from the preceding elements.
    X_list = [seq[:-1] for seq in encoded_sequences if len(seq) > 1]
    y_list = [seq[1:] for seq in encoded_sequences if len(seq) > 1]

    if not X_list:
        logger.warning("No sequences long enough for LSTM training (length > 1). Skipping.")
        return None

    padded_X = [np.pad(seq, ((0, max_len - 1 - len(seq)), (0, 0)), mode='constant') for seq in X_list]
    padded_y = [np.pad(seq, ((0, max_len - 1 - len(seq)), (0, 0)), mode='constant') for seq in y_list]
    
    X_tensor = torch.tensor(np.array(padded_X), dtype=torch.float32).to(device)
    y_tensor = torch.tensor(np.array(padded_y), dtype=torch.float32).to(device)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            # We need to reshape the output of the LSTM to match the target shape
            lstm_out, _ = model.lstm(X_batch)
            outputs = model.fc(lstm_out)
            
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"LSTM Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        if progress_callback:
            progress_callback(epoch + 1, epochs, f"LSTM training: Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    logger.info("LSTM training complete.")
    model.eval()
    return model

#========================================================================================
# MODULE: causal_analysis.py
#========================================================================================
def prepare_timeseries(detections_df, track_id, feature_type, **kwargs):
    """
    Extracts and prepares a specific time-series for a given animal track.
    """
    logger.info(f"Preparing time-series for track {track_id}, feature: {feature_type}, details: {kwargs}")
    track_df = detections_df[detections_df['track_id'] == track_id].sort_values('frame').set_index('frame')
    if track_df.empty:
        raise ValueError(f"No data found for track_id {track_id}")

    ts = None
    if feature_type == 'position':
        kp = kwargs['keypoint']
        ax_idx = 0 if kwargs['axis'] == 'x' else 1
        ts = track_df['keypoints'].apply(lambda kps: kps.get(kp, (np.nan, np.nan))[ax_idx])
    
    elif feature_type == 'velocity':
        kp = kwargs['keypoint']
        pos = track_df['keypoints'].apply(lambda kps: np.array(kps.get(kp, (np.nan, np.nan, np.nan))[:2]))
        # Calculate frame-to-frame velocity magnitude
        ts = pos.diff().apply(np.linalg.norm)

    elif feature_type == 'vae_latent':
        dim_idx = kwargs['latent_dim_index']
        col_name = f'latent_{dim_idx}'
        if col_name not in track_df.columns:
            raise ValueError(f"Latent dimension {col_name} not found in data.")
        ts = track_df[col_name]
        
    else:
        raise NotImplementedError(f"Feature extraction for '{feature_type}' is not implemented.")

    if ts is None:
        raise ValueError("Could not generate the requested time-series.")

    # Reindex to create a continuous time-series and interpolate gaps
    full_index = pd.RangeIndex(start=ts.index.min(), stop=ts.index.max() + 1, name='frame')
    ts = ts.reindex(full_index)
    ts_interpolated = ts.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    
    if ts_interpolated.isnull().any():
        logger.warning(f"Time-series for track {track_id} still contains NaNs after cleaning. This may affect results.")

    return ts_interpolated


def run_granger_causality(ts1, ts2, max_lag=10):
    """
    Performs Granger Causality test to see if ts1 "Granger-causes" ts2.
    """
    logger.info(f"Running Granger Causality with max_lag={max_lag}")
    # Combine series and drop NaNs for the test
    data = pd.concat([ts2, ts1], axis=1).dropna()
    data.columns = ['target', 'causal']

    if len(data) < 3 * max_lag:
        logger.warning(f"Not enough data for Granger Causality test with max_lag={max_lag}. Need at least {3*max_lag}, have {len(data)}. Skipping.")
        return None
        
    gc_results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
    # Return the results for the final specified lag
    return gc_results[max_lag][0]


def run_tlcc(ts1, ts2, max_lag=100):
    """
    Performs Time-Lagged Cross-Correlation on two time-series.
    """
    logger.info(f"Running TLCC with max_lag={max_lag}")
    correlations = []
    lags = range(-max_lag, max_lag + 1)
    
    for lag in lags:
        if lag < 0: # ts2 leads ts1
            corr = ts1.corr(ts2.shift(lag))
        else: # ts1 leads ts2
            corr = ts1.shift(-lag).corr(ts2)
        correlations.append(corr)
    
    correlations = np.array(correlations)
    if np.all(np.isnan(correlations)):
        return lags, correlations, 0, 0

    peak_idx = np.nanargmax(np.abs(correlations))
    peak_lag = lags[peak_idx]
    peak_corr = correlations[peak_idx]
    
    logger.info(f"TLCC complete. Peak correlation of {peak_corr:.3f} at lag {peak_lag}.")
    return lags, correlations, peak_lag, peak_corr


def run_ccm(ts1, ts2, embed_dim=3, tau=1, lib_sizes=(10, 200, 10)):
    """
    Performs Convergent Cross Mapping.
    """
    logger.info(f"Running CCM with embed_dim={embed_dim}, tau={tau}")
    # skccm requires numpy arrays
    v1 = ts1.dropna().values
    v2 = ts2.dropna().values

    if len(v1) < lib_sizes[1] or len(v2) < lib_sizes[1]:
        logger.warning("Not enough non-NaN data for CCM analysis. Skipping.")
        return None, [], []

    e1 = Embed(v1); e2 = Embed(v2)
    X1 = e1.embed(dimension=embed_dim, lag=tau)
    X2 = e2.embed(dimension=embed_dim, lag=tau)
    
    lib_lens = np.arange(lib_sizes[0], min(len(X1), lib_sizes[1]), lib_sizes[2])
    
    # CCM object: library, target, embed_dim, tau
    ccm = CCM(X1, X2, embed_dim, tau)
    
    ts1_xmap_ts2 = ccm.cross_map_predict(lib_lengths=lib_lens)
    
    # Swap library and target to test causality in the other direction
    ccm.set_dat(X2, X1)
    ts2_xmap_ts1 = ccm.cross_map_predict(lib_lengths=lib_lens)

    logger.info("CCM analysis complete.")
    return lib_lens, ts1_xmap_ts2, ts2_xmap_ts1

#========================================================================================
# MODULE: GUI Application (main.py core)
#========================================================================================
class BehaviorAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Behavior Analysis Toolkit"); self.root.geometry("1100x850")
        
        # --- State and Parameter Variables ---
        self.groups, self.running, self.group_hmms, self.comparison_results = {}, False, {}, {}
        self.status_var, self.progress_var = tk.StringVar(value="Ready"), tk.DoubleVar(value=0)
        self.plot_type = tk.StringVar(value="transition")
        self.output_folder = tk.StringVar()
        self.keypoints_entry_var = tk.StringVar(value="nose,left_eye,right_eye,left_ear,right_ear,left_shoulder,right_shoulder")
        self.behaviors_entry_var = tk.StringVar()
        self.analysis_type = tk.StringVar(value="HMM"); self.use_bbox_var = tk.BooleanVar(value=True)
        self.location_mode = tk.StringVar(value="none"); self.location_grid_size = tk.StringVar(value="50")
        self.generate_location_videos = tk.BooleanVar(value=False); self.model_type = tk.StringVar(value="categorical")
        self.use_poses = tk.BooleanVar(value=True); self.n_states = tk.StringVar(value="5")
        self.max_frame_gap = tk.StringVar(value="15"); self.min_bout_duration = tk.StringVar(value="15")
        self.conf_threshold = tk.StringVar(value="0.3"); self.umap_neighbors = tk.StringVar(value="30")
        self.umap_components = tk.StringVar(value="5"); self.min_cluster_size = tk.StringVar(value="75")
        self.comparison_method = tk.StringVar(value="log_likelihood")
        self.latent_dim, self.intermediate_dim = tk.StringVar(value="8"), tk.StringVar(value="128")
        self.vae_epochs, self.vae_dropout = tk.StringVar(value="100"), tk.StringVar(value="0.2")
        self.lstm_layers, self.lstm_epochs, self.lstm_dropout = tk.StringVar(value="2"), tk.StringVar(value="50"), tk.StringVar(value="0.2")
        self.run_lstm = tk.BooleanVar(value=False)
        self.vae_num_behavioral_clusters = tk.StringVar(value="8"); self.vae_min_postural_cluster_size = tk.StringVar(value="100")
        self.hybrid_mode = tk.BooleanVar(value=False); self.use_gpu = tk.BooleanVar(value=torch.cuda.is_available())
        self.social_mode = tk.BooleanVar(value=False)
        
        # New variables for user-defined skeleton and normalization
        self.skeleton_connections = []  # List of tuples (start_kp, end_kp)
        self.normalization_left = tk.StringVar(value="left_shoulder")
        self.normalization_right = tk.StringVar(value="right_shoulder")
        
        # Results Storage
        self.all_latent_points, self.postural_labels, self.behavioral_labels = None, None, None
        self.pose_representatives, self.location_counts = None, None
        self.background_frame_info = {}
        self.behavior_names = []

        self.create_widgets()
        self.update_keypoint_combos()
        logger.info("Application initialized successfully.")

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)
        menubar = tk.Menu(self.root); self.root.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0); menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save Project...", command=self.save_project)
        file_menu.add_command(label="Load Project...", command=self.load_project)
        file_menu.add_command(label="Load Multiple Projects...", command=self.load_multiple_projects)
        file_menu.add_separator(); file_menu.add_command(label="Exit", command=self.root.quit)
        notebook = ttk.Notebook(main_frame); notebook.pack(fill=tk.BOTH, expand=True)
        tab1, tab2, tab3, tab4 = ttk.Frame(notebook, padding="10"), ttk.Frame(notebook, padding="10"), ttk.Frame(notebook, padding="10"), ttk.Frame(notebook, padding="10")
        notebook.add(tab1, text="1. Setup & Input"); notebook.add(tab2, text="2. Analysis Parameters")
        notebook.add(tab3, text="3. Execute & Visualize"); notebook.add(tab4, text="4. Export Data")
        self.create_tab1_setup(tab1); self.create_tab2_params(tab2); self.create_tab3_execute(tab3); self.create_tab4_export(tab4)

    def create_tab1_setup(self, parent_tab):
        parent_tab.columnconfigure(0, weight=1); parent_tab.rowconfigure(3, weight=1)
        
        # --- Input Configuration ---
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

        # --- Skeleton Connections ---
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
        
        # --- Data Groups ---
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
    
    def update_keypoint_combos(self, *args):
        keypoints = parse_comma_separated(self.keypoints_entry_var.get())
        keypoints = sorted(list(set(keypoints))) # Ensure unique and sorted
        
        # Update normalization point combos
        self.norm_left_combo['values'] = keypoints
        self.norm_right_combo['values'] = keypoints
        
        # Update skeleton connection combos
        self.start_combo['values'] = keypoints
        self.end_combo['values'] = keypoints
        
        # Set sensible defaults if current values are invalid
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
            # Sort to avoid duplicates like (A,B) and (B,A)
            conn = tuple(sorted((start, end)))
            if conn not in self.skeleton_connections:
                self.skeleton_connections.append(conn)
                self.skeleton_list.insert(tk.END, f"{conn[0]} -- {conn[1]}")

    def remove_skeleton_connection(self):
        selections = self.skeleton_list.curselection()
        if not selections: return
        
        # Iterate backwards to safely delete from list
        for sel_idx in reversed(selections):
            del self.skeleton_connections[sel_idx]
            self.skeleton_list.delete(sel_idx)

    def create_tab2_params(self, parent_tab):
        parent_tab.columnconfigure(0, weight=1); parent_tab.columnconfigure(1, weight=1)
        left_frame, right_frame = ttk.Frame(parent_tab), ttk.Frame(parent_tab); left_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5)); right_frame.grid(row=0, column=1, sticky='nsew', padx=(5, 0)); right_frame.rowconfigure(1, weight=1)

        gen_frame = ttk.LabelFrame(left_frame, text="General Parameters", padding="10"); gen_frame.pack(fill='x', expand=False, pady=(0, 10))
        ttk.Label(gen_frame, text="Analysis Type:").grid(row=0, column=0, sticky=tk.W, pady=2); ttk.Combobox(gen_frame, textvariable=self.analysis_type, values=["HMM", "VAE/LSTM"], state="readonly").grid(row=0, column=1, sticky=tk.W)
        ttk.Label(gen_frame, text="Confidence Threshold:").grid(row=1, column=0, sticky=tk.W, pady=2); ttk.Entry(gen_frame, textvariable=self.conf_threshold, width=10).grid(row=1, column=1, sticky=tk.W)
        ttk.Label(gen_frame, text="Max Frame Gap:").grid(row=2, column=0, sticky=tk.W, pady=2); ttk.Entry(gen_frame, textvariable=self.max_frame_gap, width=10).grid(row=2, column=1, sticky=tk.W)
        ttk.Checkbutton(gen_frame, text="Hybrid HMM + VAE Mode", variable=self.hybrid_mode).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=5)
        ttk.Checkbutton(gen_frame, text="Enable Multi-Animal Interactions", variable=self.social_mode).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=5)
        ttk.Checkbutton(gen_frame, text="Use GPU (if available)", variable=self.use_gpu).grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=5)

        loc_frame = ttk.LabelFrame(left_frame, text="Location Analysis & Validation", padding="10"); loc_frame.pack(fill='both', expand=True); ttk.Label(loc_frame, text="NOTE: For different arenas, use separate project files.", font=('TkDefaultFont', 8, 'italic')).pack(anchor='w', pady=(0,5)); ttk.Label(loc_frame, text="Mode:").pack(anchor='w'); ttk.Radiobutton(loc_frame, text="None (Pose Only)", variable=self.location_mode, value="none").pack(anchor='w', padx=(10,0))
        unsupervised_frame = ttk.Frame(loc_frame); unsupervised_frame.pack(fill='x', expand=True, padx=(10,0)); ttk.Radiobutton(unsupervised_frame, text="Unsupervised Grid", variable=self.location_mode, value="unsupervised").grid(row=0, column=0, sticky='w'); ttk.Label(unsupervised_frame, text="Grid Size (px):").grid(row=0, column=1, sticky='w', padx=(10, 2)); ttk.Entry(unsupervised_frame, textvariable=self.location_grid_size, width=8).grid(row=0, column=2, sticky='w')
        roi_frame = ttk.Frame(loc_frame); roi_frame.pack(fill='x', expand=True, pady=(5,0), padx=(10,0)); ttk.Radiobutton(roi_frame, text="User-Defined ROIs", variable=self.location_mode, value="roi").grid(row=0, column=0, sticky='nw'); self.roi_text = scrolledtext.ScrolledText(roi_frame, height=5, width=40, wrap=tk.WORD); self.roi_text.grid(row=1, column=0, columnspan=3, pady=(5,0), padx=(20,0)); self.roi_text.insert(tk.END, '{\n  "center": [200, 200, 300, 300],\n  "corner_A": [0, 0, 150, 150]\n}'); ttk.Label(roi_frame, text='JSON format: {"name": [x, y, w, h]}', font=('TkDefaultFont', 8)).grid(row=2, column=0, columnspan=3, padx=(20,0), sticky='w'); ttk.Separator(loc_frame, orient='horizontal').pack(fill='x', pady=10); ttk.Checkbutton(loc_frame, text="Generate Location-Specific Video Clips", variable=self.generate_location_videos).pack(anchor='w', padx=(10,0))

        hmm_frame = ttk.LabelFrame(right_frame, text="HMM Parameters", padding="10"); hmm_frame.pack(fill='x', expand=False, pady=(0, 10)); hmm_frame.columnconfigure(1, weight=1); ttk.Label(hmm_frame, text="Mode:").grid(row=0, column=0, sticky=tk.W, pady=2); ttk.Radiobutton(hmm_frame, text="Frame-based (Poses)", variable=self.use_poses, value=True).grid(row=0, column=1, sticky='w', padx=2); ttk.Radiobutton(hmm_frame, text="Event-based (Bouts)", variable=self.use_poses, value=False).grid(row=0, column=2, sticky='w', padx=2); ttk.Label(hmm_frame, text="HMM Type:").grid(row=1, column=0, sticky=tk.W, pady=2); ttk.Combobox(hmm_frame, textvariable=self.model_type, values=["categorical", "gaussian"], state="readonly").grid(row=1, column=1, sticky=tk.W, columnspan=2); ttk.Label(hmm_frame, text="HMM States (N):").grid(row=2, column=0, sticky=tk.W, pady=2); ttk.Entry(hmm_frame, textvariable=self.n_states, width=10).grid(row=2, column=1, sticky=tk.W, columnspan=2); ttk.Label(hmm_frame, text="Min Bout Duration (frames):").grid(row=3, column=0, sticky=tk.W, pady=2); ttk.Entry(hmm_frame, textvariable=self.min_bout_duration, width=10).grid(row=3, column=1, sticky=tk.W, columnspan=2); ttk.Label(hmm_frame, text="UMAP Neighbors:").grid(row=4, column=0, sticky=tk.W, pady=2); ttk.Entry(hmm_frame, textvariable=self.umap_neighbors, width=10).grid(row=4, column=1, sticky=tk.W, columnspan=2); ttk.Label(hmm_frame, text="UMAP Components:").grid(row=5, column=0, sticky=tk.W, pady=2); ttk.Entry(hmm_frame, textvariable=self.umap_components, width=10).grid(row=5, column=1, sticky=tk.W, columnspan=2); ttk.Label(hmm_frame, text="HDBSCAN Min Cluster Size:").grid(row=6, column=0, sticky=tk.W, pady=2); ttk.Entry(hmm_frame, textvariable=self.min_cluster_size, width=10).grid(row=6, column=1, sticky=tk.W, columnspan=2); ttk.Label(hmm_frame, text="Group Comparison Method:").grid(row=7, column=0, sticky=tk.W, pady=2); ttk.Combobox(hmm_frame, textvariable=self.comparison_method, values=["log_likelihood", "kl_divergence"], state="readonly").grid(row=7, column=1, sticky=tk.W, columnspan=2)
        
        vae_frame = ttk.LabelFrame(right_frame, text="VAE/LSTM Parameters", padding="10"); vae_frame.pack(fill='both', expand=True); vae_frame.columnconfigure(1, weight=1); ttk.Label(vae_frame, text="Latent Dimensions:").grid(row=0, column=0, sticky=tk.W, pady=2); ttk.Entry(vae_frame, textvariable=self.latent_dim, width=10).grid(row=0, column=1, sticky=tk.W); ttk.Label(vae_frame, text="VAE Intermediate Dim:").grid(row=1, column=0, sticky=tk.W, pady=2); ttk.Entry(vae_frame, textvariable=self.intermediate_dim, width=10).grid(row=1, column=1, sticky=tk.W); ttk.Label(vae_frame, text="VAE Epochs:").grid(row=2, column=0, sticky=tk.W, pady=2); ttk.Entry(vae_frame, textvariable=self.vae_epochs, width=10).grid(row=2, column=1, sticky=tk.W); ttk.Label(vae_frame, text="VAE Dropout:").grid(row=3, column=0, sticky=tk.W, pady=2); ttk.Entry(vae_frame, textvariable=self.vae_dropout, width=10).grid(row=3, column=1, sticky=tk.W); ttk.Separator(vae_frame, orient='horizontal').grid(row=4, column=0, columnspan=2, sticky='ew', pady=10); ttk.Label(vae_frame, text="Num Behavioral Clusters (HMM):").grid(row=5, column=0, sticky=tk.W, pady=2); ttk.Entry(vae_frame, textvariable=self.vae_num_behavioral_clusters, width=10).grid(row=5, column=1, sticky=tk.W); ttk.Label(vae_frame, text="Min Postural Cluster Size (HDBSCAN):").grid(row=6, column=0, sticky=tk.W, pady=2); ttk.Entry(vae_frame, textvariable=self.vae_min_postural_cluster_size, width=10).grid(row=6, column=1, sticky=tk.W); ttk.Separator(vae_frame, orient='horizontal').grid(row=7, column=0, columnspan=2, sticky='ew', pady=10); ttk.Checkbutton(vae_frame, text="Run Bidirectional LSTM after VAE", variable=self.run_lstm).grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=5); ttk.Label(vae_frame, text="LSTM Layers:").grid(row=9, column=0, sticky=tk.W, pady=2); ttk.Entry(vae_frame, textvariable=self.lstm_layers, width=10).grid(row=9, column=1, sticky=tk.W); ttk.Label(vae_frame, text="LSTM Epochs:").grid(row=10, column=0, sticky=tk.W, pady=2); ttk.Entry(vae_frame, textvariable=self.lstm_epochs, width=10).grid(row=10, column=1, sticky=tk.W); ttk.Label(vae_frame, text="LSTM Dropout:").grid(row=11, column=0, sticky=tk.W, pady=2); ttk.Entry(vae_frame, textvariable=self.lstm_dropout, width=10).grid(row=11, column=1, sticky=tk.W)
        
        ttk.Button(parent_tab, text="Suggest Parameters", command=self.suggest_params).grid(row=1, column=0, columnspan=2, pady=10)

    def create_tab3_execute(self, parent_tab):
        parent_tab.columnconfigure(0, weight=1); parent_tab.rowconfigure(1, weight=1)
        exec_frame = ttk.LabelFrame(parent_tab, text="Execution Controls", padding="10"); exec_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10)); exec_frame.columnconfigure(1, weight=1)
        ttk.Label(exec_frame, text="Output Folder:").grid(row=0, column=0, sticky=tk.W, pady=2); ttk.Entry(exec_frame, textvariable=self.output_folder).grid(row=0, column=1, sticky="ew", padx=(0, 5)); ttk.Button(exec_frame, text="Browse...", command=self.select_output_folder).grid(row=0, column=2)
        ttk.Label(exec_frame, text="Plot to Display:").grid(row=1, column=0, sticky=tk.W, pady=2); plot_values = ["transition", "emission", "comparison", "latent_space", "postural_clusters", "behavioral_clusters", "dominant_cluster_map", "vae_comparison"]; ttk.Combobox(exec_frame, textvariable=self.plot_type, values=plot_values, state="readonly").grid(row=1, column=1, sticky=tk.W); self.plot_type.trace_add("write", self.update_plot)
        run_button = ttk.Button(exec_frame, text="Run Analysis", command=self.start_analysis, style="Accent.TButton"); run_button.grid(row=0, column=3, rowspan=2, padx=20, ipady=10); ttk.Style().configure("Accent.TButton", font=('TkDefaultFont', 10, 'bold'))
        status_frame = ttk.Frame(exec_frame); status_frame.grid(row=2, column=0, columnspan=4, sticky='ew', pady=(10, 0)); status_frame.columnconfigure(1, weight=1); ttk.Label(status_frame, text="Status:").grid(row=0, column=0, sticky=tk.W); ttk.Label(status_frame, textvariable=self.status_var).grid(row=0, column=1, sticky=tk.W); ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100).grid(row=1, column=0, columnspan=2, sticky='ew', pady=5)
        
        plot_frame = ttk.LabelFrame(parent_tab, text="Visualization", padding="10"); plot_frame.grid(row=1, column=0, sticky="nsew"); plot_frame.columnconfigure(0, weight=1); plot_frame.rowconfigure(0, weight=1); self.figure = Figure(figsize=(8, 6), dpi=100); self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame); self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True); self.update_plot()

    def create_tab4_export(self, parent_tab):
        parent_tab.columnconfigure(0, weight=1)
        export_frame = ttk.LabelFrame(parent_tab, text="Export Raw Data", padding="20"); export_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=20); export_frame.columnconfigure(0, weight=1)
        info_label = ttk.Label(export_frame, text="This tool aggregates individual detection .txt files into a single CSV per video. The output is a 'wide' table with columns for track ID, frame, and x, y, confidence for each keypoint.", wraplength=500, justify=tk.LEFT); info_label.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 20))
        ttk.Label(export_frame, text="Export to same Output Folder:").grid(row=1, column=0, sticky=tk.E, padx=5); ttk.Entry(export_frame, textvariable=self.output_folder, state='readonly', width=60).grid(row=1, column=1, sticky='ew'); ttk.Button(export_frame, text="Export Keypoints to CSV", command=self.start_csv_export, style="Accent.TButton").grid(row=2, column=0, columnspan=2, pady=20)

    def add_group(self):
        group_name = simpledialog.askstring("Group Name", "Enter group name (e.g., Control):")
        if not group_name or group_name in self.groups:
            if group_name: messagebox.showwarning("Warning", f"Group '{group_name}' already exists.")
            return
        self.groups[group_name] = {'sources': [], 'video_path_map': {}}
        self.group_tree.insert("", "end", text=group_name, iid=group_name, values=("Group", group_name), open=True)

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
        
        data_source = {'pose_dir': pose_dir, 'video_path': video_path}
        
        if group_name not in self.groups: self.groups[group_name] = {'sources': [], 'video_path_map': {}}
        self.groups[group_name]['sources'].append(data_source)
        self.groups[group_name]['video_path_map'][pose_dir] = video_path

        source_display_text = f"{os.path.basename(pose_dir)} | {os.path.basename(video_path)}"
        self.group_tree.insert(group_id, "end", text="", values=("Data Source", source_display_text))
        logger.info(f"Added data source to group {group_name}: {data_source}")

    def remove_selected(self):
        for item_id in reversed(self.group_tree.selection()):
            parent_id = self.group_tree.parent(item_id)
            if not parent_id: # It's a group
                group_name = self.group_tree.item(item_id, "text")
                if group_name in self.groups: del self.groups[group_name]
            else: # It's a data source
                group_name = self.group_tree.item(parent_id, "text")
                pose_dir_to_remove = self.group_tree.item(item_id, "values")[1].split(' | ')[0]
                if group_name in self.groups:
                    self.groups[group_name]['sources'] = [src for src in self.groups[group_name]['sources'] if os.path.basename(src.get('pose_dir', '')) != pose_dir_to_remove]
                    # Also clean up the video_path_map
                    key_to_del = next((k for k in self.groups[group_name]['video_path_map'] if os.path.basename(k) == pose_dir_to_remove), None)
                    if key_to_del: del self.groups[group_name]['video_path_map'][key_to_del]
            self.group_tree.delete(item_id)

    def select_output_folder(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if folder: self.output_folder.set(folder)
    
    def get_all_params(self):
        # Create a dictionary of all parameters to be saved
        params = {
            'groups': self.groups,
            'output_folder': self.output_folder.get(),
            'keypoints_entry_var': self.keypoints_entry_var.get(),
            'behaviors_entry_var': self.behaviors_entry_var.get(),
            'analysis_type': self.analysis_type.get(),
            'use_bbox_var': self.use_bbox_var.get(),
            'location_mode': self.location_mode.get(),
            'location_grid_size': self.location_grid_size.get(),
            'generate_location_videos': self.generate_location_videos.get(),
            'model_type': self.model_type.get(),
            'use_poses': self.use_poses.get(),
            'n_states': self.n_states.get(),
            'max_frame_gap': self.max_frame_gap.get(),
            'min_bout_duration': self.min_bout_duration.get(),
            'conf_threshold': self.conf_threshold.get(),
            'umap_neighbors': self.umap_neighbors.get(),
            'umap_components': self.umap_components.get(),
            'min_cluster_size': self.min_cluster_size.get(),
            'comparison_method': self.comparison_method.get(),
            'latent_dim': self.latent_dim.get(),
            'intermediate_dim': self.intermediate_dim.get(),
            'vae_epochs': self.vae_epochs.get(),
            'vae_dropout': self.vae_dropout.get(),
            'lstm_layers': self.lstm_layers.get(),
            'lstm_epochs': self.lstm_epochs.get(),
            'lstm_dropout': self.lstm_dropout.get(),
            'run_lstm': self.run_lstm.get(),
            'vae_num_behavioral_clusters': self.vae_num_behavioral_clusters.get(),
            'vae_min_postural_cluster_size': self.vae_min_postural_cluster_size.get(),
            'hybrid_mode': self.hybrid_mode.get(),
            'use_gpu': self.use_gpu.get(),
            'social_mode': self.social_mode.get(),
            'skeleton_connections': self.skeleton_connections,
            'normalization_left': self.normalization_left.get(),
            'normalization_right': self.normalization_right.get(),
            'roi_definitions_text': self.roi_text.get("1.0", tk.END)
        }
        return params

    def set_all_params(self, state):
        # Load state from dictionary
        # Handle regular tk.Variables
        for key, value in state.items():
            if hasattr(self, key):
                attr = getattr(self, key)
                if isinstance(attr, tk.Variable):
                    attr.set(value)

        # Handle special cases
        self.groups = state.get('groups', {})
        self.skeleton_connections = state.get('skeleton_connections', [])
        
        # Repopulate GUI widgets that don't use tk.Variable directly
        if 'roi_definitions_text' in state:
            self.roi_text.delete("1.0", tk.END)
            self.roi_text.insert(tk.END, state['roi_definitions_text'])

        self.skeleton_list.delete(0, tk.END)
        for conn in self.skeleton_connections:
            self.skeleton_list.insert(tk.END, f"{conn[0]} -- {conn[1]}")

        # Rebuild the group tree from the loaded data
        self.group_tree.delete(*self.group_tree.get_children())
        for group_name, group_data in self.groups.items():
            self.group_tree.insert("", "end", text=group_name, iid=group_name, values=("Group", group_name), open=True)
            sources = group_data.get('sources', [])
            for source in sources:
                if isinstance(source, dict) and 'pose_dir' in source and 'video_path' in source:
                    source_text = f"{os.path.basename(source['pose_dir'])} | {os.path.basename(source['video_path'])}"
                    self.group_tree.insert(group_name, "end", text="", values=("Data Source", source_text))
        
        # Trigger updates based on loaded data
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
            self.running = True; self.status_var.set("Processing multiple projects..."); self.progress_var.set(0)
            for i, file_path in enumerate(file_paths):
                self.status_var.set(f"Loading project {i+1}/{len(file_paths)}: {os.path.basename(file_path)}")
                self.load_project(file_path)
                self.status_var.set(f"Running analysis for project {i+1}/{len(file_paths)}")
                if self.analysis_type.get() == "HMM": self.run_hmm_analysis()
                else: self.run_vae_lstm_analysis()
                self.progress_var.set(((i+1) / len(file_paths)) * 100)
            self.status_var.set("Batch processing complete"); messagebox.showinfo("Success", "All projects processed!")
        except Exception as e:
            self.status_var.set(f"Error: {e}"); logger.error(f"Batch processing failed: {e}", exc_info=True); messagebox.showerror("Error", f"Batch processing failed: {e}")
        finally:
            self.running = False; self.progress_var.set(0); self.status_var.set("Ready")

    def validate_inputs(self, for_export=False):
        if not self.output_folder.get() or not os.path.isdir(self.output_folder.get()):
            raise ValueError("A valid Output folder must be specified.")
        if not self.groups: raise ValueError("At least one data group must be added.")
        if for_export: return True
        
        # Validate parameters that could cause crashes
        try:
            float(self.conf_threshold.get())
            int(self.max_frame_gap.get())
            int(self.n_states.get())
            int(self.latent_dim.get())
        except ValueError as e:
            raise ValueError(f"Invalid numerical parameter. Please check your inputs. Error: {e}")

        # Validate normalization points
        if not self.normalization_left.get() or not self.normalization_right.get():
            raise ValueError("Both normalization reference keypoints must be selected.")

        # Validate ROI JSON
        if self.location_mode.get() == 'roi':
            try:
                roi_defs = json.loads(self.roi_text.get("1.0", tk.END))
                validate_roi_definitions(roi_defs)
            except json.JSONDecodeError: raise ValueError("ROI Definitions are not valid JSON.")
            except ValidationError as e: raise ValueError(f"ROI Definition Error: {e}")
        
        return True

    def suggest_params(self):
        if not self.groups: return messagebox.showwarning("Warning", "Add data before suggesting parameters.")
        threading.Thread(target=self._suggest_params_worker, daemon=True).start()

    def _suggest_params_worker(self):
        try:
            self.status_var.set("Reading data to suggest parameters..."); self.progress_var.set(10)
            pose_dirs_by_group = {g: [s['pose_dir'] for s in v['sources']] for g, v in self.groups.items()}
            video_path_map = {pd: vp for g in self.groups.values() for pd, vp in g['video_path_map'].items()}

            detections_df, _, _ = read_detections(pose_dirs_by_group, self.keypoints_entry_var.get(), self.behaviors_entry_var.get(), self.use_bbox_var.get(), video_path_map=video_path_map)
            if detections_df.empty: 
                messagebox.showwarning("Warning", "No valid detections found in the specified directories. Cannot suggest parameters.")
                return

            num_samples = len(detections_df)
            self.progress_var.set(50)
            self.status_var.set(f"Found {num_samples} samples. Calculating suggestions...")

            # Suggest HDBSCAN min_cluster_size
            suggested_min_cluster = max(10, min(150, num_samples // 100))
            self.min_cluster_size.set(str(suggested_min_cluster))
            self.vae_min_postural_cluster_size.set(str(suggested_min_cluster))

            # Suggest UMAP neighbors
            suggested_neighbors = max(15, min(75, int(np.sqrt(num_samples))))
            self.umap_neighbors.set(str(suggested_neighbors))
            
            # Suggest latent dimensions based on number of keypoints
            num_kps = len(parse_comma_separated(self.keypoints_entry_var.get()))
            suggested_latent_dim = max(4, min(16, num_kps))
            self.latent_dim.set(str(suggested_latent_dim))

            self.progress_var.set(100)
            messagebox.showinfo("Success", "Parameters suggested based on data size and keypoints.")
        except Exception as e:
            logger.error(f"Failed to suggest parameters: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to suggest parameters: {str(e)}")
        finally:
            self.status_var.set("Ready")
            self.progress_var.set(0)

    def start_analysis(self):
        if self.running: return messagebox.showwarning("Warning", "Analysis is already running.")
        try: self.validate_inputs()
        except ValueError as e: return messagebox.showerror("Input Error", str(e))
        self.running = True
        if self.hybrid_mode.get(): threading.Thread(target=self.run_hybrid_analysis, daemon=True).start()
        elif self.analysis_type.get() == "HMM": threading.Thread(target=self.run_hmm_analysis, daemon=True).start()
        else: threading.Thread(target=self.run_vae_lstm_analysis, daemon=True).start()
            
    def start_csv_export(self):
        if self.running: return messagebox.showwarning("Warning", "Process is already running.")
        try: self.validate_inputs(for_export=True)
        except ValueError as e: return messagebox.showerror("Input Error", str(e))
        self.running = True; threading.Thread(target=self.run_csv_export, daemon=True).start()

    def run_csv_export(self):
        try:
            logger.info("="*20 + " Starting CSV Export " + "="*20)
            self.status_var.set("Starting CSV export..."); self.progress_var.set(0)
            pose_dirs_by_group = {g: [s['pose_dir'] for s in v['sources']] for g, v in self.groups.items()}
            
            self.status_var.set("Loading detection data..."); self.progress_var.set(20)
            detections_df, keypoint_names, _ = read_detections(pose_dirs_by_group, self.keypoints_entry_var.get(), self.behaviors_entry_var.get(), self.use_bbox_var.get())
            if detections_df.empty: raise ValueError("No detections found to export.")
            
            self.status_var.set("Writing CSV files..."); self.progress_var.set(60)
            export_detections_to_csv(detections_df, keypoint_names, self.output_folder.get())
            
            self.status_var.set("Export complete"); self.progress_var.set(100); messagebox.showinfo("Success", "CSV export complete!")
        except Exception as e:
            self.status_var.set(f"Error: {e}"); logger.error(f"CSV export failed: {e}", exc_info=True); messagebox.showerror("Error", f"CSV export failed: {e}")
        finally:
            self.running = False; self.status_var.set("Ready"); self.progress_var.set(0)

    def progress_callback(self, current, total, message):
        if total > 0: self.progress_var.set((current / total) * 100)
        self.status_var.set(message); self.root.update_idletasks()

    def run_hmm_analysis(self):
        try:
            logger.info("="*20 + " Starting New HMM Analysis Run " + "="*20)
            self.status_var.set("Initializing..."); self.progress_var.set(0); params = self.get_all_params()
            roi_defs = json.loads(self.roi_text.get("1.0", tk.END)) if params['location_mode'] == 'roi' else None
            grid_size = int(params['location_grid_size']) if params['location_mode'] == 'unsupervised' else 50
            normalization_ref_points = (self.normalization_left.get(), self.normalization_right.get())
            skeleton_connections = self.skeleton_connections
            
            self.progress_callback(5, 100, "Loading & un-normalizing data...")
            pose_dirs = {g: [s['pose_dir'] for s in v['sources']] for g, v in self.groups.items()}
            video_map = {pd: vp for g in self.groups.values() for pd, vp in g['video_path_map'].items()}
            detections_df, keypoint_names, behavior_names = read_detections(pose_dirs, params['keypoints_entry_var'], params['behaviors_entry_var'], params['use_bbox_var'], video_path_map=video_map)
            self.behavior_names = behavior_names
            if detections_df.empty: raise ValueError("No detections found.")
            
            self.progress_callback(15, 100, "Computing features...")
            detections_df = compute_feature_vectors(detections_df, keypoint_names, float(params['conf_threshold']), normalization_ref_points=normalization_ref_points, use_bbox=params['use_bbox_var'], location_mode=params['location_mode'], location_grid_size=grid_size, roi_definitions=roi_defs, social_mode=params['social_mode'])
            
            sequences_by_group = defaultdict(list)
            bouts = [] # Define bouts here to have it in scope later
            if params['use_poses']:
                self.progress_callback(25, 100, "Clustering poses..."); 
                detections_df, _ = cluster_poses(detections_df, int(params['umap_neighbors']), int(params['umap_components']), int(params['min_cluster_size']))
                detections_df['cluster_label'] = detections_df['pose_category']
                self.progress_callback(35, 100, "Finding pose representatives..."); self.pose_representatives = find_pose_representatives(detections_df)
                self.progress_callback(45, 100, "Grouping frame sequences...")
                for group in self.groups:
                    group_df = detections_df[detections_df['group'] == group]
                    if not group_df.empty: sequences_by_group[group] = split_sequences(group_by_track(group_df.to_dict('records')), int(params['max_frame_gap']))
            else: # Bout-based
                self.progress_callback(25, 100, "Aggregating into behavioral bouts..."); 
                bouts = aggregate_into_bouts(detections_df, int(params['max_frame_gap']), int(params['min_bout_duration']))
                if not bouts: raise ValueError("Bout aggregation resulted in an empty list.")
                for b in bouts: b['pose_category'] = b['class_id']
                self.progress_callback(35, 100, "Grouping bout sequences...")
                bouts_by_track = defaultdict(list); [bouts_by_track[(b['group'], b['video_source'], b['track_id'])].append(b) for b in bouts]
                for (group, _, _), track_bouts in bouts_by_track.items(): sequences_by_group[group].append(track_bouts)

            self.progress_callback(50, 100, "Training HMMs..."); self.group_hmms.clear()
            for i, (group, sequences) in enumerate(sequences_by_group.items()):
                if not sequences: continue
                def hmm_progress(c,t,m): self.progress_callback(50 + (i + c/t) / len(sequences_by_group) * 25, 100, f"HMM ({group}): {m}")
                model, id_map = train_hmm(sequences, int(params['n_states']), model_type=params['model_type'], use_poses=params['use_poses'], progress_callback=hmm_progress)
                if model:
                    emission_df = compute_emission_matrix(model, id_map, behavior_names=behavior_names, use_poses=params['use_poses'])
                    self.group_hmms[group] = {'model': model, 'id_map': id_map, 'transmat': compute_transition_matrix(model), 'emission_df': emission_df, 'state_labels': assign_state_labels(emission_df), 'use_poses': params['use_poses']}
            if not self.group_hmms: raise ValueError("HMM training failed for all groups.")

            self.progress_callback(75, 100, "Comparing groups & saving..."); 
            if len(self.group_hmms) > 1: self.comparison_results = compare_groups(self.group_hmms, params['comparison_method'], sequences_by_group)
            save_results(self.group_hmms, params['output_folder'], params['comparison_method'], keypoint_names, skeleton_connections, pose_representatives=self.pose_representatives, video_path_map=video_map, comparison_results=self.comparison_results)
            
            # --- CORRECTED LOCATION REPORTING ---
            self.progress_callback(85, 100, "Generating location reports...")
            video_path, video_dims = self.get_background_info(detections_df, video_map)
            self.background_frame_info = {'path': video_path, 'dims': video_dims, 'params': params}
            self.location_counts = None

            if video_path and video_dims[0]:
                # Path for Frame-based (Poses) analysis
                if params['use_poses']:
                    self.location_counts = generate_location_report(detections_df, params['location_mode'], params['output_folder'], grid_size, roi_defs)
                    if params['generate_location_videos']: create_location_specific_videos(params['output_folder'], detections_df, video_map, params['location_mode'], grid_size, roi_defs)
                
                # New Path for Event-based (Bouts) analysis
                elif not params['use_poses'] and bouts:
                    logger.info("Generating location report from behavioral bouts...")
                    self.location_counts = defaultdict(lambda: defaultdict(int))
                    for bout in bouts:
                        behavior_label = bout['class_id'] # Use the original behavior class ID
                        center_x, center_y = bout['mean_location']
                        location_id = 'N/A'
                        
                        if params['location_mode'] == 'unsupervised':
                            grid_cols = int(np.ceil(video_dims[0] / grid_size))
                            col = int(center_x // grid_size)
                            row_idx = int(center_y // grid_size)
                            location_id = f"cell_{row_idx * grid_cols + col}"
                        elif params['location_mode'] == 'roi' and roi_defs:
                            location_id = 'outside_roi'
                            for name, (x, y, w, h) in roi_defs.items():
                                if (x <= center_x < x + w) and (y <= center_y < y + h):
                                    location_id = name; break
                        
                        self.location_counts[location_id][behavior_label] += 1
            # --- END CORRECTION ---
            
            self.progress_callback(95, 100, "Plotting results..."); self.update_plot()
            self.progress_callback(100, 100, "Analysis complete"); messagebox.showinfo("Success", "HMM analysis complete!")
        except Exception as e:
            self.status_var.set(f"Error: {e}"); logger.error(f"HMM analysis failed: {e}", exc_info=True); messagebox.showerror("Error", f"HMM analysis failed:\n{e}")
        finally:
            self.running = False; self.status_var.set("Ready"); self.progress_var.set(0)
            
    def run_vae_lstm_analysis(self):
        try:
            logger.info("="*20 + " Starting New VAE/LSTM Analysis Run " + "="*20)
            self.status_var.set("Initializing..."); self.progress_var.set(0); params = self.get_all_params()
            roi_defs = json.loads(self.roi_text.get("1.0", tk.END)) if params['location_mode'] == 'roi' else None
            grid_size = int(params['location_grid_size']) if params['location_mode'] == 'unsupervised' else 50
            normalization_ref_points = (self.normalization_left.get(), self.normalization_right.get())
            skeleton_connections = self.skeleton_connections
            
            self.progress_callback(5, 100, "Loading & un-normalizing data...")
            pose_dirs = {g: [s['pose_dir'] for s in v['sources']] for g, v in self.groups.items()}
            video_map = {pd: vp for g in self.groups.values() for pd, vp in g['video_path_map'].items()}
            detections_df, keypoint_names, _ = read_detections(pose_dirs, params['keypoints_entry_var'], params['behaviors_entry_var'], params['use_bbox_var'], video_path_map=video_map)
            if detections_df.empty: raise ValueError("No detections found.")
            
            self.progress_callback(15, 100, "Computing features...")
            detections_df = compute_feature_vectors(detections_df.copy(), keypoint_names, float(params['conf_threshold']), normalization_ref_points=normalization_ref_points, use_bbox=params['use_bbox_var'], location_mode=params['location_mode'], location_grid_size=grid_size, roi_definitions=roi_defs, social_mode=params['social_mode'])
            detections_df['original_index'] = detections_df.index

            baseline_group_name = list(self.groups.keys())[0]; logger.info(f"Using group '{baseline_group_name}' as baseline for VAE training.")
            baseline_fv = detections_df[detections_df['group'] == baseline_group_name]['feature_vector'].tolist()
            if not baseline_fv: raise ValueError(f"No feature vectors for baseline group '{baseline_group_name}'.")

            device = 'cuda' if params['use_gpu'] and torch.cuda.is_available() else 'cpu'
            vae, _ = train_vae(baseline_fv, int(params['latent_dim']), int(params['intermediate_dim']), int(params['vae_epochs']), float(params['vae_dropout']), device=device, progress_callback=lambda c,t,m: self.progress_callback(25 + (c/t)*10, 100, m))
            
            self.progress_callback(35, 100, "Encoding all sequences...")
            sequences = split_sequences(group_by_track(detections_df.to_dict('records')), int(params['max_frame_gap']))
            encoded_sequences_by_group, all_encoded_sequences = defaultdict(list), []
            all_original_indices = []
            for seq in sequences:
                if not seq: continue
                encoded_seq = encode_sequences(vae, [seq], device=device)[0]; encoded_sequences_by_group[seq[0]['group']].append(encoded_seq); all_encoded_sequences.append(encoded_seq)
                all_original_indices.extend([det['original_index'] for det in seq])
            if not all_encoded_sequences: raise ValueError("Encoding resulted in an empty list.")
            self.all_latent_points = np.concatenate([s for s in all_encoded_sequences if s.ndim == 2 and s.shape[0] > 0])

            self.progress_callback(45, 100, "Postural clustering (HDBSCAN)..."); self.postural_labels = hdbscan.HDBSCAN(min_cluster_size=int(params['vae_min_postural_cluster_size'])).fit_predict(self.all_latent_points)
            
            self.progress_callback(55, 100, "Behavioral clustering (HMM)..."); hmm_sequences = [[{'feature_vector': f} for f in seq] for seq in all_encoded_sequences]
            hmm_model, _ = train_hmm(hmm_sequences, int(params['vae_num_behavioral_clusters']), model_type='gaussian', use_poses=True, progress_callback=lambda c,t,m: self.progress_callback(55 + (c/t)*10, 100, m))
            if hmm_model: self.behavioral_labels = np.concatenate(infer_state_sequences(hmm_model, hmm_sequences, {}, use_poses=True))
            
            self.progress_callback(65, 100, "Comparing groups in latent space...")
            if len(self.groups) > 1:
                self.comparison_results = compare_groups_vae(encoded_sequences_by_group)
                if not self.comparison_results.empty: self.comparison_results.to_csv(os.path.join(params['output_folder'], "vae_group_comparison_report.csv"))
            
            if self.behavioral_labels is not None and self.behavioral_labels.size > 0:
                if len(self.behavioral_labels) == len(all_original_indices):
                    # Map the flat labels back to the original dataframe using the preserved indices
                    label_series = pd.Series(self.behavioral_labels, index=all_original_indices)
                    detections_df['cluster_label'] = label_series.reindex(detections_df.index).fillna(-1).astype(int)
                else:
                    logger.warning("Length mismatch between behavioral labels and original indices. Cannot assign cluster labels.")

            if params['run_lstm']: train_lstm(all_encoded_sequences, int(params['latent_dim']), int(params['lstm_layers']), int(params['lstm_epochs']), float(params['lstm_dropout']), device=device, progress_callback=lambda c,t,m: self.progress_callback(70 + (c/t)*15, 100, m))
            
            self.progress_callback(85, 100, "Generating reports & videos...")
            video_path, video_dims = self.get_background_info(detections_df, video_map); self.background_frame_info = {'path': video_path, 'dims': video_dims, 'params': params}
            if video_path and video_dims[0] and 'cluster_label' in detections_df.columns:
                self.location_counts = generate_location_report(detections_df, params['location_mode'], params['output_folder'], grid_size, roi_defs)
                if params['generate_location_videos']: create_location_specific_videos(params['output_folder'], detections_df, video_map, params['location_mode'], grid_size, roi_defs)
            
            self.progress_callback(95, 100, "Plotting and saving results..."); self.update_plot()
            os.makedirs(params['output_folder'], exist_ok=True)
            fig_ls = Figure(); self.plot_latent_space(ax=fig_ls.add_subplot(111)); fig_ls.tight_layout(); fig_ls.savefig(os.path.join(params['output_folder'], "vae_latent_space.png")); plt.close(fig_ls)
            if self.postural_labels is not None: fig_pc = Figure(); self.plot_postural_clusters(ax=fig_pc.add_subplot(111)); fig_pc.tight_layout(); fig_pc.savefig(os.path.join(params['output_folder'], "vae_postural_clusters.png")); plt.close(fig_pc)
            if self.behavioral_labels is not None and 'cluster_label' in detections_df.columns: fig_bc = Figure(); self.plot_behavioral_clusters(ax=fig_bc.add_subplot(111)); fig_bc.tight_layout(); fig_bc.savefig(os.path.join(params['output_folder'], "vae_behavioral_clusters.png")); plt.close(fig_bc)

            self.progress_callback(100, 100, "Analysis complete"); messagebox.showinfo("Success", "VAE/LSTM analysis complete!")
        except Exception as e:
            self.status_var.set(f"Error: {e}"); logger.error(f"VAE/LSTM analysis failed: {e}", exc_info=True); messagebox.showerror("Error", f"VAE/LSTM analysis failed:\n{e}")
        finally:
            self.running = False; self.status_var.set("Ready"); self.progress_var.set(0)
            
    def run_hybrid_analysis(self):
        try:
            logger.info("="*20 + " Starting Hybrid HMM + VAE Analysis " + "="*20)
            self.progress_callback(0, 100, "Initializing hybrid analysis..."); params = self.get_all_params()
            roi_defs = json.loads(self.roi_text.get("1.0", tk.END)) if params['location_mode'] == 'roi' else None
            grid_size = int(params['location_grid_size']) if params['location_mode'] == 'unsupervised' else 50
            normalization_ref_points = (self.normalization_left.get(), self.normalization_right.get())
            skeleton_connections = self.skeleton_connections
            
            self.progress_callback(5, 100, "Loading data...")
            pose_dirs = {g: [s['pose_dir'] for s in v['sources']] for g, v in self.groups.items()}
            video_map = {pd: vp for g in self.groups.values() for pd, vp in g['video_path_map'].items()}
            detections_df, keypoint_names, behavior_names = read_detections(pose_dirs, params['keypoints_entry_var'], params['behaviors_entry_var'], params['use_bbox_var'], video_path_map=video_map)
            if detections_df.empty: raise ValueError("No detections found.")
            
            self.progress_callback(15, 100, "Computing features...")
            detections_df = compute_feature_vectors(detections_df, keypoint_names, float(params['conf_threshold']), normalization_ref_points=normalization_ref_points, use_bbox=params['use_bbox_var'], location_mode=params['location_mode'], location_grid_size=grid_size, roi_definitions=roi_defs, social_mode=params['social_mode'])
            detections_df['original_index'] = detections_df.index
            
            baseline_fv = detections_df[detections_df['group'] == list(self.groups.keys())[0]]['feature_vector'].tolist()
            device = 'cuda' if params['use_gpu'] and torch.cuda.is_available() else 'cpu'
            vae, _ = train_vae(baseline_fv, int(params['latent_dim']), int(params['intermediate_dim']), int(params['vae_epochs']), float(params['vae_dropout']), device=device, progress_callback=lambda c,t,m: self.progress_callback(25 + (c/t)*15, 100, m))
            
            sequences = split_sequences(group_by_track(detections_df.to_dict('records')), int(params['max_frame_gap']))
            latent_sequences_by_group = defaultdict(list)
            for seq in sequences:
                if not seq: continue
                # We need to create a list of dicts with feature vectors for encode_sequences
                latent_seq_vectors = [det['feature_vector'] for det in seq]
                encoded_latent_seq, _ = vae.encode(torch.tensor(latent_seq_vectors, dtype=torch.float32).to(device))
                
                # Re-package for HMM
                hmm_ready_seq = [{'feature_vector': vec} for vec in encoded_latent_seq.detach().cpu().numpy()]
                latent_sequences_by_group[seq[0]['group']].append(hmm_ready_seq)

            self.progress_callback(40, 100, "Training HMM on VAE latents...")
            self.group_hmms.clear()
            for group, latent_seqs in latent_sequences_by_group.items():
                if not latent_seqs: continue
                model, id_map = train_hmm(latent_seqs, int(params['n_states']), 'gaussian', True, progress_callback=lambda c,t,m: self.progress_callback(40 + (c/t)*20, 100, m))
                if model:
                    emission_df = compute_emission_matrix(model, id_map, behavior_names=behavior_names, use_poses=True)
                    self.group_hmms[group] = {'model': model, 'id_map': id_map, 'transmat': compute_transition_matrix(model), 'emission_df': emission_df, 'state_labels': assign_state_labels(emission_df), 'use_poses': True}
            if not self.group_hmms: raise ValueError("Hybrid HMM training failed.")

            self.progress_callback(60, 100, "Inferring states...")
            inferred_labels, original_indices = [], []
            for group, latent_seqs in latent_sequences_by_group.items():
                model_data = self.group_hmms[group]
                states = infer_state_sequences(model_data['model'], latent_seqs, model_data['id_map'], True)
                inferred_labels.extend(np.concatenate(states))
            # Get original indices in the same order
            all_original_indices = [det['original_index'] for seq in sequences for det in seq]
            if len(inferred_labels) == len(all_original_indices):
                label_series = pd.Series(inferred_labels, index=all_original_indices)
                detections_df['cluster_label'] = label_series.reindex(detections_df.index).fillna(-1).astype(int)

            self.progress_callback(75, 100, "Comparing groups & saving...")
            if len(self.group_hmms) > 1: self.comparison_results = compare_groups(self.group_hmms, params['comparison_method'], latent_sequences_by_group)
            save_results(self.group_hmms, params['output_folder'], params['comparison_method'], keypoint_names, skeleton_connections, video_path_map=video_map, comparison_results=self.comparison_results)
            
            self.progress_callback(85, 100, "Generating location reports...")
            video_path, video_dims = self.get_background_info(detections_df, video_map); self.background_frame_info = {'path': video_path, 'dims': video_dims, 'params': params}
            if video_path and video_dims[0] and 'cluster_label' in detections_df.columns:
                self.location_counts = generate_location_report(detections_df, params['location_mode'], params['output_folder'], grid_size, roi_defs)
                if params['generate_location_videos']: create_location_specific_videos(params['output_folder'], detections_df, video_map, params['location_mode'], grid_size, roi_defs)
            
            self.progress_callback(95, 100, "Plotting..."); self.update_plot()
            self.progress_callback(100, 100, "Hybrid analysis complete"); messagebox.showinfo("Success", "Hybrid HMM + VAE analysis complete!")
        except Exception as e:
            self.status_var.set(f"Error: {e}"); logger.error(f"Hybrid analysis failed: {e}", exc_info=True); messagebox.showerror("Error", f"Hybrid analysis failed:\n{e}")
        finally:
            self.running = False; self.status_var.set("Ready"); self.progress_var.set(0)
            
    def get_background_info(self, df, video_path_map):
        if df.empty or 'directory' not in df.columns: return None, (None, None)
        first_dir = df['directory'].iloc[0]; video_path = video_path_map.get(first_dir)
        if not video_path or not os.path.exists(video_path): return None, (None, None)
        video_dims = (df['video_width'].iloc[0], df['video_height'].iloc[0])
        if not video_dims[0] or pd.isna(video_dims[0]): return video_path, (None, None)
        return video_path, video_dims

    def update_plot(self, *args):
        self.figure.clear(); ax = self.figure.add_subplot(111); plot_type, analysis_type = self.plot_type.get(), self.analysis_type.get()
        try:
            if plot_type == 'dominant_cluster_map':
                if not self.location_counts or not self.background_frame_info.get('path'): ax.text(0.5, 0.5, "Run location-aware analysis first.", ha='center', color='gray')
                else:
                    params = self.background_frame_info['params']
                    roi_defs = json.loads(self.roi_text.get("1.0", tk.END)) if params['location_mode'] == 'roi' else None
                    grid_size = int(params['location_grid_size']) if params['location_mode'] == 'unsupervised' else 50
                    cap = cv2.VideoCapture(self.background_frame_info['path']); ret, frame = cap.read(); cap.release()
                    if ret: 
                        # CORRECTED LINE: All arguments after the first keyword are now also keywords.
                        plot_dominant_cluster_map(ax, self.location_counts, frame, params['location_mode'], behavior_names=self.behavior_names, grid_size=grid_size, roi_definitions=roi_defs)
                    else: ax.text(0.5, 0.5, "Could not load background image.", ha='center', color='red')
            
            elif (analysis_type == "HMM" or self.hybrid_mode.get()) and self.group_hmms:
                key = next(iter(self.group_hmms), None) 
                if not key: raise ValueError("No HMM data to plot.")
                data = self.group_hmms[key]
                labels = [v['label'] for k,v in sorted(data['state_labels'].items())]
                
                if plot_type == 'transition': plot_transition_heatmap(data['transmat'], labels, ax, f'Transition Matrix ({key})')
                elif plot_type == 'emission' and data['emission_df'] is not None: plot_emission_matrix(data['emission_df'], labels, ax, f'Emission Matrix ({key})')
                elif plot_type == 'comparison' and isinstance(self.comparison_results, pd.DataFrame) and not self.comparison_results.empty: 
                    plot_comparison_heatmap(self.comparison_results, self.comparison_method.get(), ax)
                else: ax.text(0.5, 0.5, "Plot not available for current data/selection.", ha='center', color='gray')

            elif analysis_type == "VAE/LSTM" and self.all_latent_points is not None:
                if plot_type == 'latent_space': self.plot_latent_space(ax)
                elif plot_type == 'postural_clusters': self.plot_postural_clusters(ax)
                elif plot_type == 'behavioral_clusters': self.plot_behavioral_clusters(ax)
                elif plot_type == 'vae_comparison' and isinstance(self.comparison_results, pd.DataFrame) and not self.comparison_results.empty: 
                    plot_comparison_heatmap(self.comparison_results, "Latent Space Distances", ax, cmap='magma', annotate=True)
                else: ax.text(0.5, 0.5, "Plot not available for current data/selection.", ha='center', color='gray')
                
            else: ax.text(0.5, 0.5, "Run analysis to see plots.", ha='center', color='gray')
                
        except Exception as e:
            logger.error(f"Failed to update plot: {e}", exc_info=True); ax.text(0.5, 0.5, f"Error displaying plot:\n{e}", ha='center', color='red', wrap=True)
            
        self.figure.tight_layout(); self.canvas.draw()

    def plot_latent_space(self, ax): 
        if self.all_latent_points is None: return
        ax.scatter(self.all_latent_points[:, 0], self.all_latent_points[:, 1], alpha=0.5, s=5, rasterized=True)
        ax.set_title("Latent Space (VAE)"); ax.set_xlabel("Latent Dim 1"); ax.set_ylabel("Latent Dim 2")
        
    def plot_postural_clusters(self, ax): 
        if self.all_latent_points is None or self.postural_labels is None: return
        scatter = ax.scatter(self.all_latent_points[:, 0], self.all_latent_points[:, 1], c=self.postural_labels, cmap='tab20', alpha=0.5, s=5, rasterized=True)
        ax.set_title("Postural Clusters (HDBSCAN)"); ax.set_xlabel("Latent Dim 1"); ax.set_ylabel("Latent Dim 2")
        if len(np.unique(self.postural_labels)) > 1: ax.legend(*scatter.legend_elements(), title="Postures", bbox_to_anchor=(1.05, 1), loc='upper left')

    def plot_behavioral_clusters(self, ax): 
        if self.all_latent_points is None or self.behavioral_labels is None: return
        scatter = ax.scatter(self.all_latent_points[:, 0], self.all_latent_points[:, 1], c=self.behavioral_labels, cmap='viridis', alpha=0.5, s=5, rasterized=True)
        ax.set_title("Behavioral Clusters (HMM)"); ax.set_xlabel("Latent Dim 1"); ax.set_ylabel("Latent Dim 2")
        if len(np.unique(self.behavioral_labels)) > 1: ax.legend(*scatter.legend_elements(), title="Behaviors", bbox_to_anchor=(1.05, 1), loc='upper left')


if __name__ == "__main__":
    root = tk.Tk()
    app = BehaviorAnalysisApp(root)
    root.mainloop()