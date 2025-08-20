# video_generator.py
import os
import cv2
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

def draw_on_frame(frame, text):
    """Draws text overlay on a frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Add a semi-transparent background for better readability
    (text_width, text_height), _ = cv2.getTextSize(text, font, 1, 2)
    cv2.rectangle(frame, (10, 10), (20 + text_width, 50 + text_height), (0,0,0), -1)
    cv2.putText(frame, text, (15, 45), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

def draw_detection(frame, bbox, keypoints, conf_threshold=0.3):
    """Draws bounding box and keypoints for a single detection."""
    # Draw bbox (center_x, center_y, width, height)
    tl = (int(bbox[0] - bbox[2] / 2), int(bbox[1] - bbox[3] / 2))
    br = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
    cv2.rectangle(frame, tl, br, (0, 255, 0), 2)  # Green for bbox
    
    # Draw keypoints if confidence >= threshold
    for kp_name, (x, y, conf) in keypoints.items():
        if conf >= conf_threshold:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1) # Red for keypoints
    return frame

def create_cluster_videos(output_dir, video_path, all_detections_df):
    """
    Generates a video for each behavioral sub-cluster with a single efficient pass
    over the source video file.
    """
    logger.info("Starting optimized video generation for behavioral sub-clusters...")
    if 'cluster_label' not in all_detections_df.columns:
        logger.warning("No 'cluster_label' column found. Skipping video generation.")
        return

    # --- 1. Prepare frame-to-cluster mapping ---
    frame_map = defaultdict(lambda: defaultdict(list))
    # Filter for detections that have been successfully clustered (label != -1)
    clustered_df = all_detections_df[all_detections_df['cluster_label'] != -1]
    for _, row in clustered_df.iterrows():
        key = (row['class_id'], row['cluster_label'])
        frame_map[row['frame']][key].append({
            'bbox': row['bbox'],
            'keypoints': row['keypoints']
        })

    if not frame_map:
        logger.warning("No clustered frames to write. No videos will be generated.")
        return

    # --- 2. Initialize Video Writers for each cluster ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open source video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    writers = {}
    unique_clusters = clustered_df[['class_id', 'cluster_label']].drop_duplicates()
    for _, pair in unique_clusters.iterrows():
        class_id, cluster_label = int(pair['class_id']), int(pair['cluster_label'])
        video_out_path = os.path.join(output_dir, f"behavior_{class_id}_subcluster_{cluster_label}.mp4")
        writer = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))
        if writer.isOpened():
            writers[(class_id, cluster_label)] = writer
        else:
            logger.error(f"Failed to open VideoWriter for: {video_out_path}")

    if not writers:
        logger.warning("No valid video writers could be created. Aborting video generation.")
        cap.release()
        return

    # --- 3. Single Sequential Pass Through Video ---
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1 # Frames are 1-indexed in the data
        if frame_idx in frame_map:
            # Iterate through clusters present in this frame
            for key, detections in frame_map[frame_idx].items():
                if key in writers:
                    frame_copy = frame.copy() # Avoid drawing on the original frame
                    # Draw all detections for this cluster in this frame
                    for det in detections:
                        frame_copy = draw_detection(frame_copy, det['bbox'], det['keypoints'])
                    
                    text = f"Behavior: {key[0]} | Sub-Cluster: {key[1]} | Frame: {frame_idx}"
                    frame_copy = draw_on_frame(frame_copy, text)
                    writers[key].write(frame_copy)

    # --- 4. Release all resources ---
    logger.info("Releasing video capture and writers...")
    cap.release()
    for writer in writers.values():
        writer.release()
    logger.info("Finished generating all cluster videos.")