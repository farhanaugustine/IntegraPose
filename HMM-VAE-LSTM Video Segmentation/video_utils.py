# This is an updated version of video_utils.py with multi-animal support (drawing multiple detections per frame).

import os
import cv2
import numpy as np
import pandas as pd
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


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