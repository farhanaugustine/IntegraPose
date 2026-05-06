import os
import subprocess
import logging
import sys
from pathlib import Path
from typing import List, Dict

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
PYTHON_EXECUTABLE = sys.executable

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = MODULE_DIR / "sample_inputs"
DEFAULT_RESULTS_DIR = MODULE_DIR / "outputs"

SOURCE_DATA_DIR = Path(
    os.getenv("INTEGRAPOSE_GAIT_SOURCE_DIR", str(DEFAULT_SOURCE_DIR))
).expanduser()
BASE_RESULTS_DIR = Path(
    os.getenv("INTEGRAPOSE_GAIT_RESULTS_DIR", str(DEFAULT_RESULTS_DIR))
).expanduser()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def find_matching_files() -> List[Dict[str, str]]:
    """Scan the source directory for videos with matching YOLO label folders."""
    logger.info("Scanning for videos in: %s", SOURCE_DATA_DIR)
    if not SOURCE_DATA_DIR.is_dir():
        logger.error("Source directory not found: %s", SOURCE_DATA_DIR)
        return []

    matches: List[Dict[str, str]] = []

    for video_path in sorted(SOURCE_DATA_DIR.glob("*")):
        if not video_path.is_file() or not video_path.suffix.lower() in VIDEO_EXTENSIONS:
            continue

        base_name = video_path.stem
        label_dir = SOURCE_DATA_DIR / base_name

        if label_dir.is_dir():
            matches.append(
                {
                    "video_path": str(video_path),
                    "yolo_dir": str(label_dir),
                }
            )
            logger.info("Match found: video='%s', labels='%s'", video_path.name, base_name)
        else:
            logger.warning(
                "Video '%s' found, but label folder '%s' is missing. Skipping.",
                video_path.name,
                base_name,
            )

    return matches


def run_analysis_for_video(video_path: str, yolo_labels_path: str, main_script_path: Path) -> None:
    """Construct and execute the analysis command for a single video."""
    video_name = Path(video_path).stem
    output_dir = BASE_RESULTS_DIR / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("--- Starting analysis for: %s ---", video_name)
    logger.info("Output will be saved to: %s", output_dir)

    command = [
        PYTHON_EXECUTABLE,
        str(main_script_path),
        "--video_path",
        video_path,
        "--output_dir",
        str(output_dir),
        "--yolo_dir",
        yolo_labels_path,
    ]

    try:
        subprocess.run(command, check=True)
        logger.info("--- Successfully completed analysis for: %s ---", video_name)
    except subprocess.CalledProcessError as exc:
        logger.error("Analysis failed for %s. Error: %s", video_name, exc)
    except FileNotFoundError:
        logger.error("Could not find %s or %s.", PYTHON_EXECUTABLE, main_script_path)
        raise


def main() -> None:
    """Find videos and initiate batch processing."""
    main_script_path = MODULE_DIR / "main.py"
    if not main_script_path.exists():
        logger.error("Could not find main.py at %s", main_script_path)
        return

    videos_to_process = find_matching_files()
    if not videos_to_process:
        logger.warning("No valid video/label pairs were found to process. Exiting.")
        return

    logger.info("Found %s video/label pairs to process.", len(videos_to_process))
    for info in videos_to_process:
        run_analysis_for_video(info["video_path"], info["yolo_dir"], main_script_path)

    logger.info("--- All processing tasks are complete. ---")


if __name__ == "__main__":
    main()
