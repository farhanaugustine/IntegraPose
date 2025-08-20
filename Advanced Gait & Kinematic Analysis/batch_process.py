import os
import subprocess
import logging
import sys

# --- ⚙️ Configuration ---
# 1. Set the path to the folder where your VIDEOS and LABEL FOLDERS are located.
#    The script will automatically find matching pairs inside this folder.
SOURCE_DATA_DIR = r"C:\Users\Aegis-MSI\Documents\IntegraPose\Modified_YOLO_C3TR\A2C2f_C3k2_C2PSA_Backbone_L-Model\Trimmed Videos\organized_labels"

# 2. Set the path where you want all your results to be saved.
#    The script will create a subfolder for each video here.
BASE_RESULTS_DIR = r"C:\Users\Aegis-MSI\Documents\IntegraPose\Modified_YOLO_C3TR\A2C2f_C3k2_C2PSA_Backbone_L-Model\Trimmed Videos\Gait_Analysis"

# 3. Add any other video file extensions you want to look for.
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')

# 4. Set the full path to your Python executable. This avoids environment issues.
PYTHON_EXECUTABLE = sys.executable
# Example: PYTHON_EXECUTABLE = r"C:\Users\YourUser\miniconda3\envs\myenv\python.exe"

# --- End of Configuration ---


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def find_matching_files():
    """Scans the source directory to find video files and their matching label folders."""
    logger.info(f"Scanning for videos in: {SOURCE_DATA_DIR}")
    videos_to_process = []

    if not os.path.isdir(SOURCE_DATA_DIR):
        logger.error(f"Source directory not found: {SOURCE_DATA_DIR}")
        return []

    for item_name in sorted(os.listdir(SOURCE_DATA_DIR)):
        # Check if the item is a video file based on its extension
        if item_name.lower().endswith(VIDEO_EXTENSIONS):
            base_name = os.path.splitext(item_name)[0]
            video_path = os.path.join(SOURCE_DATA_DIR, item_name)

            # Check for a corresponding directory with the same base name
            yolo_dir_path = os.path.join(SOURCE_DATA_DIR, base_name)

            if os.path.isdir(yolo_dir_path):
                # If a match is found, add the full paths to our list
                videos_to_process.append({
                    "video_path": video_path,
                    "yolo_dir": yolo_dir_path
                })
                logger.info(f"✅ Match Found: Video='{item_name}', Labels='{base_name}'")
            else:
                logger.warning(f"⚠️ Video '{item_name}' found, but missing label folder '{base_name}'. Skipping.")
    
    return videos_to_process


def run_analysis_for_video(video_path, yolo_labels_path, main_script_path):
    """Constructs and executes the analysis command for a single video."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(BASE_RESULTS_DIR, video_name)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"--- Starting analysis for: {video_name} ---")
    logger.info(f"Output will be saved to: {output_dir}")

    command = [
        PYTHON_EXECUTABLE,
        main_script_path,
        "--video_path", video_path,
        "--output_dir", output_dir,
        "--yolo_dir", yolo_labels_path
    ]

    try:
        # Using capture_output=False so the main.py output appears in real-time
        subprocess.run(command, check=True, capture_output=False)
        logger.info(f"--- Successfully completed analysis for: {video_name} ---\n")
    except subprocess.CalledProcessError as e:
        logger.error(f"--- Analysis FAILED for: {video_name}. Error: {e}\n")
    except FileNotFoundError:
        logger.error(f"Could not find {PYTHON_EXECUTABLE} or main.py. Please check the script paths.")
        sys.exit(1)


def main():
    """Finds videos and initiates batch processing."""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    main_script_path = os.path.join(project_dir, "main.py")

    if not os.path.exists(main_script_path):
        logger.error(f"Could not find main.py at {main_script_path}")
        return

    videos_to_process = find_matching_files()

    if not videos_to_process:
        logger.warning("No valid video/label pairs were found to process. Exiting.")
        return

    logger.info(f"Found {len(videos_to_process)} video/label pairs to process.")
    for video_info in videos_to_process:
        run_analysis_for_video(
            video_info["video_path"],
            video_info["yolo_dir"],
            main_script_path
        )

    logger.info("--- All processing tasks are complete. ---")


if __name__ == "__main__":
    main()