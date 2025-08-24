# integrapose_lstm/data_processing/features.py
import numpy as np
from typing import List, Tuple, Optional, Dict

def calculate_feature_vector(
    current_kps: np.ndarray,
    prev_kps_t1: Optional[np.ndarray],
    prev_kps_t2: Optional[np.ndarray],
    keypoint_names: List[str],
    angle_definitions: List[Tuple[str, str, str]]
) -> np.ndarray:
    """
    Calculates a comprehensive feature vector from a set of keypoints and their history.

    This function is the single source of truth for feature engineering. It computes:
    1.  Pairwise distances between all keypoints.
    2.  Angles between specified triplets of keypoints.
    3.  Instantaneous velocity for each keypoint (if t-1 is available).
    4.  Instantaneous acceleration for each keypoint (if t-1 and t-2 are available).

    Args:
        current_kps (np.ndarray): The keypoints for the current frame (t).
                                  Shape: (n_keypoints, 2) or (n_keypoints, 3).
        prev_kps_t1 (Optional[np.ndarray]): The keypoints for the previous frame (t-1).
                                            Can be NaN for the first frame of a track.
        prev_kps_t2 (Optional[np.ndarray]): The keypoints for the frame before previous (t-2).
                                            Can be NaN for early frames of a track.
        keypoint_names (List[str]): An ordered list of names for each keypoint.
        angle_definitions (List[Tuple[str, str, str]]): A list of triplets of keypoint
                                                       names defining the angles to compute.

    Returns:
        np.ndarray: A 1D NumPy array containing the concatenated features.
    """
    feature_vec = []
    
    # Create a dictionary for easy name-to-coordinate mapping
    kps_map: Dict[str, Tuple[float, float]] = {
        name: (kp[0], kp[1]) for name, kp in zip(keypoint_names, current_kps)
    }

    # --- 1. Pairwise Distances ---
    # Use the raw numpy array for efficient distance calculations
    points = current_kps[:, :2] # Ensure we only use (x, y)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = np.linalg.norm(points[i] - points[j])
            feature_vec.append(distance)

    # --- 2. Angles ---
    for p1_name, p2_name, p3_name in angle_definitions:
        # Ensure all keypoints for the angle exist
        if all(name in kps_map for name in [p1_name, p2_name, p3_name]):
            p1 = np.array(kps_map[p1_name])
            p2 = np.array(kps_map[p2_name]) # Vertex of the angle
            p3 = np.array(kps_map[p3_name])
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            dot_product = np.dot(v1, v2)
            norms = np.linalg.norm(v1) * np.linalg.norm(v2)
            
            # Add a small epsilon to avoid division by zero
            angle = np.arccos(np.clip(dot_product / (norms + 1e-8), -1.0, 1.0))
            feature_vec.append(angle)
        else:
            # If a keypoint is missing, append a neutral value (0)
            feature_vec.append(0.0)

    # --- 3. Velocities and Accelerations ---
    # Create maps for previous time steps only if they are valid numpy arrays
    # This robustly handles the NaN values from the .shift() operation
    prev_kps_t1_map = {name: (kp[0], kp[1]) for name, kp in zip(keypoint_names, prev_kps_t1)} if isinstance(prev_kps_t1, np.ndarray) else {}
    prev_kps_t2_map = {name: (kp[0], kp[1]) for name, kp in zip(keypoint_names, prev_kps_t2)} if isinstance(prev_kps_t2, np.ndarray) else {}

    for kp_name in keypoint_names:
        velocity, acceleration = 0.0, 0.0
        
        # Calculate velocity if the previous frame's keypoints are available
        if kp_name in kps_map and kp_name in prev_kps_t1_map:
            current_point = np.array(kps_map[kp_name])
            prev_point_t1 = np.array(prev_kps_t1_map[kp_name])
            velocity = np.linalg.norm(current_point - prev_point_t1)
            
            # Calculate acceleration if the frame before that is also available
            if kp_name in prev_kps_t2_map:
                prev_point_t2 = np.array(prev_kps_t2_map[kp_name])
                prev_velocity = np.linalg.norm(prev_point_t1 - prev_point_t2)
                acceleration = velocity - prev_velocity
                
        feature_vec.extend([velocity, acceleration])
        
    return np.array(feature_vec, dtype=np.float32)