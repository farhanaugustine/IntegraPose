# utils.py
import cv2
import numpy as np
import json
import os
import logging

logger = logging.getLogger(__name__)

# You can keep this function if you plan to use ROIs non-interactively in the future
def get_rois(video_path, config_path):
    """
    Loads ROIs from a config file.
    """
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                rois_data = json.load(f)
            for roi in rois_data:
                roi['coords'] = np.array(roi['coords'], dtype=np.int32)
            logger.info(f"Loaded {len(rois_data)} ROIs from {config_path}")
            return rois_data
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to load or parse ROIs from {config_path}: {e}. Please fix or delete the file.")
            raise
    return []