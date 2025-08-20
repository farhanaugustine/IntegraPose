# Updated data_processing_hdbscan.py with user-defined normalization_ref_points as a parameter

# data_processing_hdbscan.py
# This is a rewritten version for improved robustness and flexibility.

import os
import re
import numpy as np
from collections import defaultdict
import umap
import hdbscan
import pandas as pd
import logging
from scipy.spatial.distance import pdist, squareform
import cv2
import json
from jsonschema import validate, ValidationError, Draft7Validator

logger = logging.getLogger(__name__)


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
        body_ref_dist = median_distances.get(normalization_ref_points)
        if body_ref_dist is None: # Also check reverse order
            body_ref_dist = median_distances.get(tuple(reversed(normalization_ref_points)))

        if body_ref_dist is None or body_ref_dist == 0:
            # Fallback to median of all pairs if user-defined fails
            if median_distances:
                body_ref_dist = np.median(list(median_distances.values()))
                logger.warning(f"Reference key pair {normalization_ref_points} not found or invalid for group '{group_name}'. "
                               f"Using median of all pairs: {body_ref_dist:.2f}")
            else:
                # Critical failure - raise an error
                raise ValueError(f"Could not determine any valid reference distance for group '{group_name}'. "
                                 "Check keypoint data and confidence threshold. Cannot proceed with feature computation.")

        logger.info(f"Group '{group_name}' body reference distance for normalization: {body_ref_dist:.2f}")
        group_ref_distances[group_name] = {'body_ref': body_ref_dist, 'medians': median_distances}

    # ... (rest of the function is identical to the original)
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

        for i in range(N_keypoints):
            for j in range(i + 1, N_keypoints):
                kp1, kp2 = keypoint_names[i], keypoint_names[j]
                if (keypoints[kp1][2] >= conf_threshold and keypoints[kp2][2] >= conf_threshold):
                    dist = np.linalg.norm(np.array(keypoints[kp1][:2]) - np.array(keypoints[kp2][:2]))
                else:
                    dist = median_distances.get((kp1, kp2), 0)
                vec.append(dist)

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

        if social_mode:
            inter_feats = inter_animal_features.get(idx, {'mean_inter_dist': 0.0, 'min_inter_dist': 0.0})
            vec.append(inter_feats['mean_inter_dist']); vec.append(inter_feats['min_inter_dist'])

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


# The rest of the functions (split_sequences, cluster_poses, etc.) are unchanged.
# They are included here for completeness.
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
                    'mean_feature_vector': mean_feature_vector
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