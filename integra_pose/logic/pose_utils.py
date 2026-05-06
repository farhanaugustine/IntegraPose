"""
Utility helpers shared across pose analytics components.
"""

from __future__ import annotations

import numpy as np


def normalize_pose(pose: np.ndarray, translation: bool = True, scale: bool = True) -> np.ndarray:
    """
    Normalize a flat pose array of x/y keypoint pairs.

    Args:
        pose: Array shaped (2 * num_keypoints,) with interleaved x/y coords.
        translation: If True, subtract the centroid so the pose is centered.
        scale: If True, divide by global standard deviation to normalize body size.

    Returns:
        Flattened, normalized pose array.
    """
    if pose is None or len(pose) % 2 != 0:
        raise ValueError("Pose array length must be even (x,y pairs).")

    keypoints = pose.reshape(-1, 2).astype(float)
    if translation:
        keypoints -= np.mean(keypoints, axis=0)
    if scale:
        std = np.std(keypoints)
        if std > 0:
            keypoints /= std
    return keypoints.flatten()


__all__ = ["normalize_pose"]
