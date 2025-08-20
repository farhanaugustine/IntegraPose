# data_loader.py
import os
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def load_yolo_data(txt_dir, video_width, video_height, keypoint_order, conf_threshold, total_frames, video_base_name):
    """
    Loads YOLO .txt files for a video, correctly mapping them to the source video's
    frame numbers. It iterates through all expected frames and only loads data
    from .txt files that actually exist, leaving gaps for frames without detections.
    This version also robustly handles optional bbox confidence and per-keypoint confidence.
    """
    all_detections = []

    if not os.path.isdir(txt_dir):
        logger.error(f"YOLO text directory not found at: {txt_dir}")
        raise FileNotFoundError(f"YOLO text directory not found at: {txt_dir}")

    num_keypoints = len(keypoint_order)

    logger.info(f"Scanning for detections across {total_frames} frames using base name '{video_base_name}'...")

    for frame_number in range(total_frames):
        if frame_number == 0:
            filename = f"{video_base_name}.txt"
        else:
            filename = f"{video_base_name}_{frame_number + 1}.txt"

        filepath = os.path.join(txt_dir, filename)
        if not os.path.exists(filepath):
            continue

        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    parts = line.strip().split()
                    if len(parts) < 5: continue

                    class_id = int(parts[0])
                    track_id = int(parts[-1])
                    x_center, y_center, w, h = map(float, parts[1:5])
                    payload = parts[5:-1]
                    
                    box_conf = 1.0
                    kpt_data_str = []
                    expected_kpt_values = num_keypoints * 3

                    if len(payload) == expected_kpt_values + 1:
                        box_conf = float(payload[0])
                        kpt_data_str = payload[1:]
                    elif len(payload) == expected_kpt_values:
                        kpt_data_str = payload
                    else:
                        continue

                    if box_conf < conf_threshold:
                        continue
                        
                    kpt_data = np.array(list(map(float, kpt_data_str)))
                    kpts_normalized_with_conf = kpt_data.reshape(num_keypoints, 3)
                    
                    kpts_normalized = kpts_normalized_with_conf[:, :2]
                    
                    center_x_px = x_center * video_width
                    center_y_px = y_center * video_height
                    keypoints_px = kpts_normalized * np.array([video_width, video_height])

                    keypoints_px[np.where((keypoints_px[:, 0] == 0) & (keypoints_px[:, 1] == 0))] = np.nan
                    
                    record = {
                        'frame': frame_number,
                        'track_id': track_id,
                        'behavior_id': class_id,
                        'confidence': box_conf,
                        'center_x': center_x_px,
                        'center_y': center_y_px,
                    }
                    
                    for j, name in enumerate(keypoint_order):
                        record[f'{name}_x'] = keypoints_px[j, 0]
                        record[f'{name}_y'] = keypoints_px[j, 1]
                        record[f'{name}_conf'] = kpts_normalized_with_conf[j, 2]
                    
                    all_detections.append(record)

                except (ValueError, IndexError):
                    logger.warning(f"File {filename}, Line {line_num}: Could not parse line. Skipping.")
                    continue

    if not all_detections:
        logger.warning("No valid detections were loaded.")
        return pd.DataFrame()

    logger.info(f"Loaded {len(all_detections)} detections from the video frames.")
    return pd.DataFrame(all_detections)
