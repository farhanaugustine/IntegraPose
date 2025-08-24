# integrapose_lstm/data_processing/loader.py
import os
import re
import logging
from typing import List

import numpy as np
import pandas as pd

# Import the centralized feature calculation function
from .features import calculate_feature_vector

logger = logging.getLogger(__name__)

def _parse_filename(filename: str) -> tuple[str, int]:
    """
    Parses the filename to extract a unique clip identifier and the frame number.
    Assumes filename format like 'some_video_name_behavior_frames_start_end_framenumber.txt'.
    For example: 'mouse_clip1_Walking_frames_100_250_101.txt'
    It extracts 'mouse_clip1_Walking_frames_100_250' as the clip_id and '101' as the frame number.
    """
    base_name = os.path.splitext(filename)[0]
    # Regex to find the numeric part at the very end of the string
    match = re.match(r'(.+)_(\d+)$', base_name)
    if match:
        clip_id = match.group(1)
        frame_num = int(match.group(2))
        return clip_id, frame_num
    raise ValueError(f"Cannot parse clip ID and frame number from filename: {filename}")


def read_detections(pose_dir: str, keypoint_names: List[str]) -> pd.DataFrame:
    """
    Reads all YOLO pose detection .txt files from a directory into a DataFrame.
    It now creates a unique 'clip_id' for each source file.
    """
    n_keypoints = len(keypoint_names)
    detections = []

    logger.info(f"Reading detections from directory: {pose_dir}")
    if not os.path.isdir(pose_dir):
        raise FileNotFoundError(f"Pose directory not found at path: {pose_dir}")

    # Expected number of parts in each line of the .txt file:
    # class_id + 4 bbox parts + (3 parts per keypoint: x, y, conf)
    expected_parts = 1 + 4 + (3 * n_keypoints)

    for filename in sorted(os.listdir(pose_dir)):
        if not filename.endswith(".txt"):
            continue
        
        try:
            # Correctly parse the filename to get a shared clip_id and frame number
            clip_id, frame_num = _parse_filename(filename)
            file_path = os.path.join(pose_dir, filename)

            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    
                    if len(parts) != expected_parts:
                        logger.warning(
                            f"Skipping malformed line in {filename}: "
                            f"Expected {expected_parts} parts, but got {len(parts)}"
                        )
                        continue
                    
                    detections.append({
                        'frame': frame_num,
                        'class_id': int(parts[0]),
                        'pose_data': [float(p) for p in parts[5:]],
                        'clip_id': clip_id  # Add the new unique clip identifier
                    })
        except Exception as e:
            logger.error(f"Failed to process file {filename}: {e}")

    if not detections:
        raise ValueError(f"No valid detections found in {pose_dir}. Check file format and content.")
        
    return pd.DataFrame(detections)

def compute_features_from_df(
    df: pd.DataFrame,
    keypoint_names: List[str],
    angle_definitions: List[List[str]]
) -> pd.DataFrame:
    """
    Computes feature vectors for each detection in the DataFrame using the centralized function.
    """
    logger.info("Computing feature vectors for the dataset...")
    # Sort by the new clip_id and then frame
    df = df.sort_values(['clip_id', 'frame']).reset_index(drop=True)
    n_keypoints = len(keypoint_names)

    df['keypoints'] = df['pose_data'].apply(
        lambda data: np.array(data).reshape(n_keypoints, 3)
    )

    # Group by clip_id to get previous frames' keypoints correctly
    df['prev_kps_t1'] = df.groupby('clip_id')['keypoints'].shift(1)
    df['prev_kps_t2'] = df.groupby('clip_id')['keypoints'].shift(2)

    # Drop the first two rows of each group as they won't have complete history for acceleration
    df.dropna(subset=['prev_kps_t1', 'prev_kps_t2'], inplace=True)
    
    if df.empty:
        raise ValueError("DataFrame is empty after dropping rows for feature calculation. Ensure clips are long enough (>= 3 frames).")

    feature_vectors = df.apply(
        lambda row: calculate_feature_vector(
            current_kps=row['keypoints'],
            prev_kps_t1=row['prev_kps_t1'],
            prev_kps_t2=row['prev_kps_t2'],
            keypoint_names=keypoint_names,
            angle_definitions=angle_definitions
        ),
        axis=1
    )

    df['feature_vector'] = feature_vectors
    
    df.drop(columns=['pose_data', 'keypoints', 'prev_kps_t1', 'prev_kps_t2'], inplace=True)
    df.dropna(subset=['feature_vector'], inplace=True)
    
    logger.info(f"Successfully computed {len(df)} feature vectors.")
    return df

def create_labeled_sequences(df: pd.DataFrame, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates overlapping sequences of feature vectors and their corresponding labels.
    """
    logger.info(f"Creating labeled sequences of length {sequence_length}...")
    # Group by clip_id to create sequences
    df = df.sort_values(['clip_id', 'frame'])
    
    X, y = [], []
    
    # Process each clip independently
    for _, group in df.groupby('clip_id'):
        features = np.vstack(group['feature_vector'].values)
        labels = group['class_id'].values
        
        if len(features) >= sequence_length:
            for i in range(len(features) - sequence_length + 1):
                X.append(features[i : i + sequence_length])
                y.append(labels[i + sequence_length - 1])

    if not X:
        raise ValueError(
            "Could not create any sequences. This might be due to insufficient "
            "continuous data for the given sequence_length. Check if your video clips are long enough."
        )
    
    logger.info(f"Generated {len(X)} sequences.")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)