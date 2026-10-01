import cv2
import os
import logging
import argparse
from pathlib import Path

from .data_loader import load_yolo_data
from .analysis import process_data
from .scientific_utils import validated_fps
from .profiles import validate_config
from .selection import select_subject
from .review import write_review
from .model_schema import validate_label_mapping
from integra_pose.utils.operation_result import OperationResult

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run(args, config_obj):
    """Analyze one video and return an outcome that cannot imply false success."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_csv_path = os.path.join(args.output_dir, 'final_analysis_data.csv')
    gait_analysis_path = os.path.join(args.output_dir, 'gait_analysis_summary.csv')
    
    yolo_txt_dir = args.yolo_dir
    if not os.path.isdir(yolo_txt_dir):
        message = f"YOLO labels directory not found: {yolo_txt_dir}"
        logger.error(message)
        return OperationResult.failure(message)

    try:
        config_obj = validate_config(config_obj)
        config_obj['MAPPING_VERIFICATION'] = validate_label_mapping(config_obj, yolo_txt_dir)
    except (ValueError, TypeError, KeyError) as exc:
        return OperationResult.failure("Invalid gait configuration.", error=str(exc))

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        message = f"Could not open video file: {args.video_path}"
        logger.error(message)
        return OperationResult.failure(message)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = validated_fps(cap.get(cv2.CAP_PROP_FPS))
    video_base_name = os.path.splitext(os.path.basename(args.video_path))[0]
    cap.release()

    if video_width <= 0 or video_height <= 0 or total_frames <= 0:
        return OperationResult.failure("Video dimensions or frame count are missing or invalid.")

    if video_fps is None:
        message = (
            "Video FPS metadata is missing or invalid; gait speed and duration "
            "cannot be reported in scientific time units."
        )
        logger.error(message)
        return OperationResult.failure(message)

    try:
        raw_df = load_yolo_data(
            txt_dir=yolo_txt_dir,
            video_width=video_width, video_height=video_height,
            keypoint_order=config_obj['DATASET']['KEYPOINT_ORDER'],
            conf_threshold=config_obj['GENERAL_PARAMS']['DETECTION_CONF_THRESHOLD'],
            total_frames=total_frames, video_base_name=video_base_name
        )
    except Exception as exc:
        logger.exception("Failed to load gait pose labels from %s", yolo_txt_dir)
        return OperationResult.failure("Failed to load gait pose labels.", error=str(exc))
    
    if raw_df.empty:
        message = "Failed to load any data from YOLO files."
        logger.error(message)
        return OperationResult.failure(message)

    try:
        # Always save a subject inventory, including when selection fails.
        raw_df.groupby('track_id').agg(
            detections=('frame', 'size'), start_frame=('frame', 'min'), end_frame=('frame', 'max')
        ).to_csv(Path(args.output_dir) / 'available_tracks.csv')
        raw_df = select_subject(raw_df, config_obj, total_frames)
        final_df, gait_df = process_data(raw_df, config_obj, fps=video_fps)
        final_tmp = Path(output_csv_path).with_suffix(".csv.tmp")
        gait_tmp = Path(gait_analysis_path).with_suffix(".csv.tmp")
        final_df.to_csv(final_tmp, index=False)
        # An empty, schema-bearing summary deliberately replaces any stale
        # stride file from a previous run.
        gait_df.to_csv(gait_tmp, index=False)
        os.replace(gait_tmp, gait_analysis_path)
        os.replace(final_tmp, output_csv_path)
        review_artifacts = write_review(
            final_df, gait_df, config_obj, args.video_path, args.output_dir, video_fps, total_frames
        )
    except Exception as exc:
        logger.exception("Gait analysis failed for %s", args.video_path)
        return OperationResult.failure("Gait analysis failed.", error=str(exc))

    logger.info(f"Saved final processed data to {output_csv_path}")
    if gait_df.empty:
        logger.warning("No strides were detected; wrote an empty gait summary with schema.")
    else:
        logger.info(f"Saved gait analysis summary to {gait_analysis_path}")
    
    logger.info("Analysis complete for this video.")
    return OperationResult.success(
        "Gait analysis complete.",
        final_analysis=output_csv_path,
        gait_summary=gait_analysis_path,
        video_fps=video_fps,
        stride_count=len(gait_df),
        **review_artifacts,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run human or animal gait analysis on one video and its YOLO pose labels.")
    parser.add_argument("--video_path", required=True, help="Full path to the input video file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save all results.")
    parser.add_argument("--yolo_dir", required=True, help="Directory with the YOLO .txt label files.")
    parser.add_argument("--config_file", required=True, help="Path to the project_config.json file.")
    args = parser.parse_args()
    
    import json
    with open(args.config_file, 'r') as f:
        config_from_file = json.load(f)
    
    result = run(args, config_from_file)
    if not result.succeeded:
        logger.error("%s%s", result.message, f" ({result.error})" if result.error else "")
        raise SystemExit(1)
