import numpy as np

def compute_pairwise_distances(keypoints: np.ndarray) -> np.ndarray:
    """
    Computes pairwise Euclidean distances between keypoints.
    Input: [T, K, C] or [K, C] where C >= 2 (x, y).
    Output: [T, K*(K-1)/2] or [K*(K-1)/2].
    """
    if keypoints.ndim == 2:
        kpts = keypoints[np.newaxis, ...] # [1, K, C]
    else:
        kpts = keypoints

    T, K, _ = kpts.shape
    if K < 2:
        return np.zeros((T, 0), dtype=kpts.dtype)

    # Extract x, y
    pts = kpts[..., :2] # [T, K, 2]
    
    # Compute differences: [T, K, 1, 2] - [T, 1, K, 2] -> [T, K, K, 2]
    diff = pts[:, :, np.newaxis, :] - pts[:, np.newaxis, :, :]
    
    # Squared distances: [T, K, K]
    dist_sq = np.sum(diff**2, axis=-1)
    
    # Euclidean distances
    dist = np.sqrt(np.maximum(dist_sq, 0.0))
    
    # Extract upper triangle indices (excluding diagonal)
    triu_idx = np.triu_indices(K, k=1)
    
    # [T, N_pairs]
    pairwise_dists = dist[:, triu_idx[0], triu_idx[1]]
    
    if keypoints.ndim == 2:
        return pairwise_dists[0]
    return pairwise_dists

def compute_velocities(keypoints: np.ndarray) -> np.ndarray:
    """
    Computes temporal derivatives (speed magnitude).
    Input: [T, K, C] where C >= 2.
    Output: [T, K].
    """
    # pts: [T, K, 2]
    pts = keypoints[..., :2]
    
    # Delta P
    # Pad first frame with 0 velocity
    delta = np.zeros_like(pts)
    delta[1:] = pts[1:] - pts[:-1]
    
    # Speed magnitude: [T, K]
    speed = np.sqrt(np.sum(delta**2, axis=-1))
    return speed

def center_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Centers keypoints relative to the bounding box center (0.5, 0.5) 
    or their own centroid?
    
    Since keypoints are normalized [0, 1] relative to the crop, 
    subtracting 0.5 centers them in the crop frame.
    
    Input: [T, K, 3] (x, y, conf)
    Output: [T, K, 3] (centered_x, centered_y, conf)
    """
    centered = keypoints.copy()
    centered[..., :2] -= 0.5
    return centered

def extract_rich_pose_features(keypoints: np.ndarray) -> np.ndarray:
    """
    Combines raw centered coords, pairwise distances, and speeds.
    
    Input: keypoints [T, K, 3] (x, y, conf) normalized to [0, 1].
    Output: flattened feature vector [T, FeatDim].
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.ndim != 3 or keypoints.shape[-1] < 3:
        raise ValueError(f"Expected keypoints with shape [T, K, 3], got {keypoints.shape}.")

    # Defensive cleanup: YOLO/confidence streams sometimes include NaNs (e.g., missing conf),
    # and augmentations can push points slightly out-of-bounds.
    keypoints = np.nan_to_num(keypoints, nan=0.0, posinf=0.0, neginf=0.0)

    xy = keypoints[..., :2]
    conf = keypoints[..., 2:3]

    valid = (xy[..., 0:1] >= 0.0) & (xy[..., 0:1] <= 1.0) & (xy[..., 1:2] >= 0.0) & (xy[..., 1:2] <= 1.0)
    conf = np.clip(conf, 0.0, 1.0) * valid.astype(np.float32)
    xy = np.clip(xy, 0.0, 1.0)
    keypoints = np.concatenate([xy, conf], axis=-1)

    # 1. Centered Coords (Translation Invariant within crop) [T, K*3]
    centered = center_keypoints(keypoints)
    flat_coords = centered.reshape(centered.shape[0], -1)
    
    # 2. Pairwise Distances (Rotation/Translation Invariant) [T, N_pairs]
    dists = compute_pairwise_distances(keypoints)
    
    # 3. Keypoint Speeds (Temporal Features) [T, K]
    speeds = compute_velocities(keypoints)
    
    # Concatenate
    # Features = [K*3 + K*(K-1)/2 + K]
    rich_features = np.concatenate([flat_coords, dists, speeds], axis=1)
    
    return rich_features.astype(np.float32)
