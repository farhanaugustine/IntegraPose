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
    match = re.search(r'_(\d+)\.txt$', filename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot extract frame number from {filename}")

def read_detections(pose_dir, keypoint_names):
    n_keypoints = len(keypoint_names)
    detections = []
    expected_line_parts = 1 + 4 + (3 * n_keypoints) + 1

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
                    
                    detections.append({
                        'frame': frame_num,
                        'class_id': int(parts[0]),
                        'pose_data': [float(p) for p in parts[5:-1]],
                        'track_id': int(parts[-1]),
                    })
        except Exception as e:
            logger.error(f"Failed to process file {filename}: {e}")

    if not detections:
        raise ValueError("No valid detections found.")
    return pd.DataFrame(detections)

def compute_feature_vectors(df, keypoint_names, angle_definitions, conf_threshold=0.3):
    logger.info("Computing feature vectors...")
    df = df.sort_values(['track_id', 'frame']).reset_index(drop=True)
    n_keypoints = len(keypoint_names)

    # Convert pose data to structured keypoints
    all_kps = []
    for _, row in df.iterrows():
        kps = {
            keypoint_names[i]: (
                row['pose_data'][3 * i], 
                row['pose_data'][3 * i + 1], 
                row['pose_data'][3 * i + 2]
            )
            for i in range(n_keypoints)
        }
        all_kps.append(kps)
    df['keypoints'] = all_kps
    
    # Imputation for missing keypoints (simplified for clarity)
    # A more robust solution might be needed for very noisy data
    df['keypoints'] = df.groupby('track_id')['keypoints'].transform(
        lambda x: x.fillna(method='ffill').fillna(method='bfill')
    )
    df.dropna(subset=['keypoints'], inplace=True)

    # Calculate velocities and accelerations
    df['prev_kps_t1'] = df.groupby('track_id')['keypoints'].shift(1)
    df['prev_kps_t2'] = df.groupby('track_id')['keypoints'].shift(2)

    feature_vectors = []
    for _, row in df.iterrows():
        kps, prev_kps_t1, prev_kps_t2 = row['keypoints'], row['prev_kps_t1'], row['prev_kps_t2']
        vec = []

        # Pairwise distances
        points = [np.array(kps[name][:2]) for name in keypoint_names]
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                vec.append(np.linalg.norm(points[i] - points[j]))

        # Angles
        for p1_name, p2_name, p3_name in angle_definitions:
            p1, p2, p3 = np.array(kps[p1_name][:2]), np.array(kps[p2_name][:2]), np.array(kps[p3_name][:2])
            v1, v2 = p1 - p2, p3 - p2
            dot = np.dot(v1, v2)
            norms = np.linalg.norm(v1) * np.linalg.norm(v2)
            angle = np.arccos(np.clip(dot / (norms + 1e-8), -1.0, 1.0))
            vec.append(angle)

        # Velocities and Accelerations
        for kp_name in keypoint_names:
            vel, accel = 0, 0
            if isinstance(prev_kps_t1, dict):
                vel = np.linalg.norm(np.array(kps[kp_name][:2]) - np.array(prev_kps_t1[kp_name][:2]))
                if isinstance(prev_kps_t2, dict):
                    prev_vel = np.linalg.norm(np.array(prev_kps_t1[kp_name][:2]) - np.array(prev_kps_t2[kp_name][:2]))
                    accel = vel - prev_vel
            vec.extend([vel, accel])
        
        feature_vectors.append(vec)

    df['feature_vector'] = feature_vectors
    df.drop(columns=['pose_data', 'keypoints', 'prev_kps_t1', 'prev_kps_t2'], inplace=True)
    return df

def create_labeled_sequences(df, sequence_length):
    logger.info(f"Creating labeled sequences of length {sequence_length}...")
    df = df.sort_values(['track_id', 'frame'])
    
    X, y = [], []
    
    for track_id, group in df.groupby('track_id'):
        features = group['feature_vector'].tolist()
        labels = group['class_id'].tolist()
        
        if len(features) >= sequence_length:
            for i in range(len(features) - sequence_length + 1):
                X.append(features[i:i + sequence_length])
                y.append(labels[i + sequence_length - 1]) # Label from the last frame

    if not X:
        raise ValueError("Could not create any sequences. Check data or sequence_length.")
    
    logger.info(f"Generated {len(X)} sequences.")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)