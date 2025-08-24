# integrapose_lstm/main.py
import argparse
import json
import logging
import sys
from typing import Dict, Any
import os

# Add the project root to the Python path to allow for absolute imports
# This is a common practice for structuring larger Python projects.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main functions from our modules
from gui.clipper import launch_clipper_gui
from data_processing.process_clips import process_clips_in_directory
from data_processing.preparation import process_and_relabel_files
from run_training import run_training_pipeline
from inference.real_time import run_real_time_inference

# --- Global Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads the JSON configuration file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {config_path}. Please check its format.")
        sys.exit(1)

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

    # --- Execute the chosen action ---
    if args.action == 'clip':
        logger.info("Launching the Video Clipper GUI...")
        launch_clipper_gui(config)
    elif args.action == 'process':
        logger.info("Starting automated processing of video clips for pose data...")
        process_clips_in_directory(config)
    elif args.action == 'prepare':
        logger.info("Starting dataset preparation...")
        prep_cfg = config['dataset_preparation']
        data_cfg = config['data']
        process_and_relabel_files(
            source_dir=prep_cfg['source_directory'],
            output_dir=data_cfg['pose_directory'],
            class_mapping=prep_cfg['class_mapping']
        )
    elif args.action == 'train':
        logger.info("Starting the model training pipeline...")
        run_training_pipeline(config)
    elif args.action == 'infer':
        logger.info("Starting real-time inference...")
        run_real_time_inference(config)

if __name__ == '__main__':
    main()