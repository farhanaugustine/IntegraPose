import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def _resolve_path(value: str, base_dir: Path) -> str:
    """Return an absolute path for a config value, preserving environment expansions."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads the JSON configuration file and normalises any relative paths."""
    config_file = Path(config_path).expanduser()
    try:
        with config_file.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError as exc:
        logger.error(f"Configuration file not found at: {config_file}")
        raise FileNotFoundError(f"Configuration file not found at: {config_file}") from exc
    except json.JSONDecodeError as exc:
        logger.error(f"Error decoding JSON from {config_file}. Please check its format.")
        raise ValueError(f"Invalid JSON in configuration file: {config_file}") from exc

    base_dir = config_file.parent

    # Normalise known path-bearing fields so the toolkit works when launched from the main GUI.
    path_fields = [
        ("dataset_preparation", "source_directory"),
        ("data", "pose_directory"),
        ("data", "output_directory"),
        ("process", "source_directory"),
        ("process", "yolo_model_path"),
        ("inference_params", "video_source"),
        ("inference_params", "output_directory"),
        ("inference_params", "yolo_model_path"),
        ("inference_params", "lstm_classifier_path"),
    ]

    for section, field in path_fields:
        section_data = config.get(section)
        if not isinstance(section_data, dict):
            continue
        raw_value = section_data.get(field)
        if not raw_value:
            continue
        try:
            section_data[field] = _resolve_path(str(raw_value), base_dir)
        except Exception:
            logger.warning(
                "Could not resolve path for %s.%s (value=%r). Leaving as-is.",
                section,
                field,
                raw_value,
            )

    return config

def main():
    """Main entry point for the IntegraPose LSTM Toolkit."""
    parser = argparse.ArgumentParser(
        description="IntegraPose LSTM Toolkit for real-time behavior classification.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'action',
        choices=['clip', 'process', 'prepare', 'train', 'infer'],
        help="""The action to perform:
  clip      - Launch the Video Clipper GUI to create labeled video clips.
  process   - Automatically run YOLO pose estimation on all created video clips.
  prepare   - Consolidate all pose data (.txt files) into a final dataset.
  train     - Train the LSTM classifier model on the final dataset.
  infer     - Run real-time inference on a new video using trained models.
"""
    )
    parser.add_argument(
        '--config',
        default='config.json',
        help="Path to the configuration file (default: config.json)."
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.action == 'clip':
        logger.info("Launching the Video Clipper GUI...")
        from .gui.clipper import launch_clipper_gui

        launch_clipper_gui(config)
    elif args.action == 'process':
        logger.info("Starting automated processing of video clips for pose data...")
        from .data_processing.process_clips import process_clips_in_directory

        process_clips_in_directory(config)
    elif args.action == 'prepare':
        logger.info("Starting dataset preparation...")
        from .data_processing.preparation import process_and_relabel_files

        prep_cfg = config['dataset_preparation']
        data_cfg = config['data']
        process_and_relabel_files(
            source_dir=prep_cfg['source_directory'],
            output_dir=data_cfg['pose_directory'],
            class_mapping=prep_cfg['class_mapping']
        )
    elif args.action == 'train':
        logger.info("Starting the model training pipeline...")
        from .run_training import run_training_pipeline

        run_training_pipeline(config)
    elif args.action == 'infer':
        logger.info("Starting real-time inference...")
        from .inference.real_time import run_real_time_inference

        run_real_time_inference(config)

if __name__ == '__main__':
    main()
