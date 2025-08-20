# main.py
import cv2
import numpy as np
import pandas as pd
import os
import logging
import argparse

# Import project modules
from data_loader import load_yolo_data
from analysis import process_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run(args, config_obj):
    """Main execution function, now accepts a dynamic config object."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_csv_path = os.path.join(args.output_dir, 'final_analysis_data.csv')
    gait_analysis_path = os.path.join(args.output_dir, 'gait_analysis_summary.csv')
    
    yolo_txt_dir = args.yolo_dir
    if not os.path.isdir(yolo_txt_dir):
        logger.error(f"YOLO labels directory not found: {yolo_txt_dir}")
        return

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file: {args.video_path}")
        return
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_base_name = os.path.splitext(os.path.basename(args.video_path))[0]
    cap.release()

    raw_df = load_yolo_data(
        txt_dir=yolo_txt_dir,
        video_width=video_width, video_height=video_height,
        keypoint_order=config_obj['DATASET']['KEYPOINT_ORDER'], 
        conf_threshold=config_obj['GENERAL_PARAMS']['DETECTION_CONF_THRESHOLD'],
        total_frames=total_frames, video_base_name=video_base_name
    )
    
    if raw_df.empty:
        logger.error("Failed to load any data from YOLO files. Exiting.")
        return

    logger.info("Consolidating tracks for single-animal analysis...")
    raw_df = raw_df.sort_values(by=['frame', 'confidence'], ascending=[True, False])
    raw_df = raw_df.drop_duplicates(subset='frame', keep='first')
    raw_df['track_id'] = 1

    final_df, gait_df = process_data(raw_df, config_obj)
    final_df.to_csv(output_csv_path, index=False)
    logger.info(f"Saved final processed data to {output_csv_path}")

    if gait_df is not None and not gait_df.empty:
        gait_df.to_csv(gait_analysis_path, index=False)
        logger.info(f"Saved gait analysis summary to {gait_analysis_path}")
    else:
        logger.warning("No strides were detected or gait analysis was skipped.")
    
    logger.info("Analysis complete for this video.")

if __name__ == "__main__":
    # This part is for standalone testing and is not used by the GUI.
    parser = argparse.ArgumentParser(description="Run behavior analysis on a single video.")
    parser.add_argument("--video_path", required=True, help="Full path to the input video file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save all results.")
    parser.add_argument("--yolo_dir", required=True, help="Directory with the YOLO .txt label files.")
    parser.add_argument("--config_file", required=True, help="Path to the project_config.json file.")
    args = parser.parse_args()
    
    import json
    with open(args.config_file, 'r') as f:
        config_from_file = json.load(f)
    
    run(args, config_from_file)
