# data_loader.py
import os
import re
import numpy as np
import pandas as pd
import cv2
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

def extract_frame_number(filename):
    """Extracts the frame number from a YOLO-pose output filename."""
    match = re.search(r'_(\d+)\.txt$', filename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot extract frame number from {filename}")

def read_detections(pose_dir, video_path, keypoint_names):
    """
    Reads detections from YOLO-pose text files, converting coordinates.
    """
    n_keypoints = len(keypoint_names)
    if not keypoint_names:
        raise ValueError("Keypoint names must be provided in the config file.")

    detections = []
    expected_pose_values = 3 * n_keypoints
    # class_id + bbox(4) + keypoints(3*n) + track_id
    expected_line_parts = 1 + 4 + expected_pose_values + 1

    logger.info(f"Reading video dimensions from: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    logger.info(f"Video dimensions: {video_width}x{video_height}")

    logger.info(f"Reading detections from: {pose_dir}")
    for filename in sorted(os.listdir(pose_dir)):
        if not filename.endswith(".txt"):
            continue
        try:
            frame_num = extract_frame_number(filename)
            file_path = os.path.join(pose_dir, filename)
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != expected_line_parts:
                        logger.warning(f"Skipping malformed line in {filename}: Expected {expected_line_parts} parts, got {len(parts)}")
                        continue
                    
                    class_id = int(parts[0])
                    bbox_normalized = [float(p) for p in parts[1:5]]
                    pose_data_normalized = [float(p) for p in parts[5:-1]]
                    track_id = int(parts[-1])

                    bbox = [
                        bbox_normalized[0] * video_width,
                        bbox_normalized[1] * video_height,
                        bbox_normalized[2] * video_width,
                        bbox_normalized[3] * video_height,
                    ]
                    keypoints = {
                        keypoint_names[i]: (
                            pose_data_normalized[3 * i] * video_width,
                            pose_data_normalized[3 * i + 1] * video_height,
                            pose_data_normalized[3 * i + 2],
                        )
                        for i in range(n_keypoints)
                    }

                    detections.append({
                        'frame': frame_num,
                        'track_id': track_id,
                        'class_id': class_id,
                        'keypoints': keypoints,
                        'bbox': bbox,
                    })
        except Exception as e:
            logger.error(f"Failed to process file {filename}: {e}")

    if not detections:
        raise ValueError("No valid detections found. Check the pose directory and file format.")
    return pd.DataFrame(detections), keypoint_names

def compute_feature_vectors(df, keypoint_names, angle_definitions, conf_threshold=0.3):
    """
    Computes feature vectors including pairwise distances, angles, velocities, and accelerations.
    Implements robust keypoint imputation using both forward and backward fill.
    """
    logger.info("Computing feature vectors with robust imputation...")
    df = df.sort_values(['track_id', 'frame'])

    # Vectorize the imputation process for efficiency
    for kp_name in keypoint_names:
        # Create columns for x, y, conf
        df[f'{kp_name}_x'] = df['keypoints'].apply(lambda kps: kps[kp_name][0])
        df[f'{kp_name}_y'] = df['keypoints'].apply(lambda kps: kps[kp_name][1])
        df[f'{kp_name}_conf'] = df['keypoints'].apply(lambda kps: kps[kp_name][2])
        
        # Set low-confidence keypoints to NaN to allow for filling
        df.loc[df[f'{kp_name}_conf'] < conf_threshold, [f'{kp_name}_x', f'{kp_name}_y']] = np.nan
        
        # Group by track_id and apply forward fill then backward fill
        df[f'{kp_name}_x'] = df.groupby('track_id')[f'{kp_name}_x'].transform(lambda x: x.ffill().bfill())
        df[f'{kp_name}_y'] = df.groupby('track_id')[f'{kp_name}_y'].transform(lambda x: x.ffill().bfill())

    # Reconstruct the keypoints dictionary from the imputed columns
    def reconstruct_kps(row):
        return {
            kp: (row[f'{kp}_x'], row[f'{kp}_y'], row[f'{kp}_conf'])
            for kp in keypoint_names
        }
    df['keypoints'] = df.apply(reconstruct_kps, axis=1)

    # Calculate velocities (shift 1) and accelerations (shift 2)
    df['prev_kps_t1'] = df.groupby('track_id')['keypoints'].shift(1)
    df['prev_kps_t2'] = df.groupby('track_id')['keypoints'].shift(2)

    feature_vectors = []
    for _, row in df.iterrows():
        kps = row['keypoints']
        prev_kps_t1 = row['prev_kps_t1']
        prev_kps_t2 = row['prev_kps_t2']
        vec = []

        # 1. Pairwise distances
        for i in range(len(keypoint_names)):
            for j in range(i + 1, len(keypoint_names)):
                p1 = np.array(kps[keypoint_names[i]][:2])
                p2 = np.array(kps[keypoint_names[j]][:2])
                dist = np.linalg.norm(p1 - p2) if not (np.isnan(p1).any() or np.isnan(p2).any()) else 0
                vec.append(dist)

        # 2. Angles
        for p1_name, p2_name, p3_name in angle_definitions:
            p1 = np.array(kps[p1_name][:2])
            p2 = np.array(kps[p2_name][:2])
            p3 = np.array(kps[p3_name][:2])
            v1, v2 = p1 - p2, p3 - p2
            dot = np.dot(v1, v2)
            norms = np.linalg.norm(v1) * np.linalg.norm(v2)
            angle = np.arccos(np.clip(dot / (norms + 1e-8), -1.0, 1.0)) if norms > 0 else 0
            vec.append(angle)

        # 3. Velocities and Accelerations
        for kp_name in keypoint_names:
            vel = 0
            accel = 0
            current_pos = np.array(kps[kp_name][:2])
            
            if isinstance(prev_kps_t1, dict):
                prev_pos_t1 = np.array(prev_kps_t1[kp_name][:2])
                if not (np.isnan(current_pos).any() or np.isnan(prev_pos_t1).any()):
                    vel = np.linalg.norm(current_pos - prev_pos_t1)
                
                if isinstance(prev_kps_t2, dict):
                    prev_pos_t2 = np.array(prev_kps_t2[kp_name][:2])
                    if not (np.isnan(prev_pos_t1).any() or np.isnan(prev_pos_t2).any()):
                        prev_vel = np.linalg.norm(prev_pos_t1 - prev_pos_t2)
                        accel = vel - prev_vel

            vec.append(vel)
            vec.append(accel)
        
        feature_vectors.append(vec)

    df['feature_vector'] = feature_vectors
    # Clean up temporary columns
    kp_cols_to_drop = [f'{kp}_{ax}' for kp in keypoint_names for ax in ['x', 'y', 'conf']]
    df.drop(columns=['prev_kps_t1', 'prev_kps_t2'] + kp_cols_to_drop, inplace=True, errors='ignore')
    return df

def create_sequences(df, max_gap=15, strategy='dynamic'):
    """Creates sequences of detections based on track_id and splitting strategy."""
    logger.info(f"Creating sequences with strategy='{strategy}', max_gap={max_gap}...")
    tracks = defaultdict(list)
    # Use to_dict('records') for faster iteration
    for det in df.to_dict('records'):
        tracks[det['track_id']].append(det)

    sequences = []
    discarded_count = 0
    for track_id, dets in tracks.items():
        if not dets:
            continue
        
        sorted_dets = sorted(dets, key=lambda x: x['frame'])
        current_seq = [sorted_dets[0]]
        
        for i in range(1, len(sorted_dets)):
            prev_det = sorted_dets[i - 1]
            curr_det = sorted_dets[i]
            
            # Determine if a split is needed
            is_gap = (curr_det['frame'] - prev_det['frame']) > max_gap
            class_id_changed = curr_det['class_id'] != prev_det['class_id']
            
            split = is_gap or (strategy == 'dynamic' and class_id_changed)
            
            if split:
                if len(current_seq) > 1:
                    sequences.append(current_seq)
                else:
                    discarded_count += 1
                current_seq = [curr_det]
            else:
                current_seq.append(curr_det)
        
        # Add the last sequence
        if len(current_seq) > 1:
            sequences.append(current_seq)
        else:
            discarded_count += 1

    logger.info(f"Generated {len(sequences)} sequences, discarded {discarded_count} single-frame or empty sequences.")
    if not sequences:
        raise ValueError("Data processing resulted in zero valid sequences. Check parameters and data quality.")
    return sequences