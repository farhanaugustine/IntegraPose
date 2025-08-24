# integrapose_lstm/data_processing/process_clips.py
import logging
from pathlib import Path
import cv2
from ultralytics import YOLO

# Setup logger for this module
logger = logging.getLogger(__name__)

def _save_yolo_txt_output(output_path: Path, results, confidence_threshold: float):
    """Formats and saves pose estimation results to a YOLO .txt file for a single frame."""
    with open(output_path, 'w') as f:
        # Check if detections and keypoints exist
        if results[0].boxes is None or results[0].keypoints is None:
            return  # Save an empty file if no detections

        # Use the robust properties of the ultralytics results objects
        boxes = results[0].boxes
        keypoints = results[0].keypoints

        # Get normalized xy and confidence tensors
        keypoints_xy_norm = keypoints.xyn.cpu().numpy()
        keypoints_conf = keypoints.conf.cpu().numpy()

        for i in range(len(boxes)):  # Iterate through each detected instance
            if boxes.conf[i] < confidence_threshold:
                continue

            # Extract bounding box and class ID
            class_id = int(boxes.cls[i])
            bbox_x_center, bbox_y_center, width, height = boxes.xywhn[i].cpu().numpy()

            # For the i-th detected instance, get its xy coordinates and confidences
            xy_for_instance = keypoints_xy_norm[i]
            conf_for_instance = keypoints_conf[i]

            kpts_flat = []
            # Manually combine xy and conf for each keypoint
            for kp_idx in range(len(xy_for_instance)):
                x, y = xy_for_instance[kp_idx]
                conf = conf_for_instance[kp_idx]
                kpts_flat.extend([f"{x:.6f}", f"{y:.6f}", f"{conf:.6f}"])

            line_parts = [
                str(class_id),
                f"{bbox_x_center:.6f}",
                f"{bbox_y_center:.6f}",
                f"{width:.6f}",
                f"{height:.6f}",
                *kpts_flat
            ]
            f.write(" ".join(line_parts) + "\n")

def process_video(video_path: Path, model: YOLO, confidence_threshold: float):
    """
    Processes a single video file to extract keypoints and saves them as one .txt file PER FRAME.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Could not open video file: {video_path}")
        return

    # Use the parent directory (e.g., "Walking") to save the output files
    output_dir_for_frames = video_path.parent

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO inference
        results = model(frame, verbose=False)
        
        # Define the output filename based on the clip name and frame number
        output_filename = f"{video_path.stem}_{frame_count}.txt"
        output_path = output_dir_for_frames / output_filename
        
        # Save the formatted output
        _save_yolo_txt_output(output_path, results, confidence_threshold)
        
        frame_count += 1

    cap.release()
    logger.info(f"Processed {frame_count} frames for {video_path.name}. Output .txt files saved in {output_dir_for_frames}/")

def process_clips_in_directory(config: dict):
    """
    Processes all video clips in a specified directory to extract and save pose data.
    """
    # Check for the main 'process' key if the whole config is passed
    if 'process' in config:
        process_config = config['process']
    else:
        process_config = config

    source_directory = process_config.get('source_directory')
    model_path = process_config.get('yolo_model_path')
    confidence_threshold = process_config.get('confidence_threshold', 0.5)
    device = process_config.get('device', 'cpu')

    if not all([source_directory, model_path]):
        if not source_directory:
            logger.error("Missing required key in config.json: 'source_directory'. Cannot process clips.")
        if not model_path:
            logger.error("Missing required key in config.json: 'yolo_model_path'. Cannot process clips.")
        return

    source_dir = Path(source_directory)
    if not source_dir.is_dir():
        logger.error(f"Source directory not found: {source_directory}")
        return

    try:
        model = YOLO(model_path)
        model.to(device)
        logger.info(f"YOLO model loaded from {model_path} and moved to device: {device}.")
    except Exception as e:
        logger.error(f"Failed to load YOLO model from {model_path}. Error: {e}")
        return

    video_files = list(source_dir.rglob('*.mp4')) + list(source_dir.rglob('*.avi'))
    if not video_files:
        logger.warning(f"No video files (.mp4, .avi) found in {source_directory}")
        return

    logger.info(f"Found {len(video_files)} video clips to process.")
    for video_path in video_files:
        logger.info(f"Processing video: {video_path.name}...")
        process_video(video_path, model, confidence_threshold)
        # After processing, the source video is no longer needed by the pipeline
        # You can uncomment the next line to automatically delete video clips after processing
        # video_path.unlink()

    logger.info("All video clips have been processed.")