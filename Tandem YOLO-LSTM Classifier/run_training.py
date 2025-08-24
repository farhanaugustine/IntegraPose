# integrapose_lstm/run_training.py
import json
import logging
import os
import pickle
from typing import Dict
import math

import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Import our modular functions
from data_processing import loader
from training import train

# --- Setup Basic Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def run_training_pipeline(config: Dict):
    """
    Executes the full data loading, processing, and model training pipeline.
    """
    try:
        # --- 1. Unpack Configuration ---
        data_cfg = config['data']
        feature_cfg = config['feature_params']
        seq_cfg = config['sequence_params']
        model_cfg = config['lstm_classifier_params']
        train_cfg = config['training_params']
        prep_cfg = config['dataset_preparation']

        os.makedirs(data_cfg['output_directory'], exist_ok=True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")

        # --- 2. Load and Process Data ---
        detections_df = loader.read_detections(data_cfg['pose_directory'], data_cfg['keypoint_names'])
        features_df = loader.compute_features_from_df(
            detections_df, data_cfg['keypoint_names'], feature_cfg['angles_to_compute']
        )

        # --- 3. Robust Stratified Split by Clip ID ---
        # Determine the primary class for each clip
        stratify_labels = features_df.groupby('clip_id')['class_id'].agg(lambda x: x.mode()[0])
        
        train_clip_ids = []
        val_clip_ids = []

        # Manually ensure each class is represented in the validation set
        logger.info("Performing manual stratified split to guarantee class representation.")
        for class_id in np.unique(stratify_labels.values):
            clips_for_class = stratify_labels[stratify_labels == class_id].index.tolist()
            
            # Calculate how many clips to move to validation (at least 1 if possible)
            num_val_for_class = math.ceil(len(clips_for_class) * train_cfg['test_split_ratio'])
            if len(clips_for_class) > 1 and num_val_for_class == 0:
                num_val_for_class = 1 # Ensure at least one if there's more than one clip
            
            np.random.shuffle(clips_for_class)

            val_clips = clips_for_class[:num_val_for_class]
            train_clips = clips_for_class[num_val_for_class:]

            val_clip_ids.extend(val_clips)
            train_clip_ids.extend(train_clips)

        # Fallback: if validation is empty (e.g., only 1 clip per class), move one from training
        if not val_clip_ids and train_clip_ids:
            logger.warning("Validation set was empty after manual split. Moving one clip from training.")
            val_clip_ids.append(train_clip_ids.pop())

        train_df = features_df[features_df['clip_id'].isin(train_clip_ids)]
        val_df = features_df[features_df['clip_id'].isin(val_clip_ids)]

        logger.info(f"Data split by clip ID: {len(train_clip_ids)} training clips, {len(val_clip_ids)} validation clips.")

        # --- 4. Create Sequences from the Split DataFrames ---
        if train_df.empty or val_df.empty:
            raise ValueError("One of the data splits is empty. Ensure you have enough video clips for both training and validation.")
            
        X_train, y_train = loader.create_labeled_sequences(train_df, seq_cfg['sequence_length'])
        X_val, y_val = loader.create_labeled_sequences(val_df, seq_cfg['sequence_length'])
        
        logger.info(f"Created {len(X_train)} training sequences and {len(X_val)} validation sequences.")
        logger.info(f"Training class distribution: {dict(sorted(zip(*np.unique(y_train, return_counts=True))))}")
        logger.info(f"Validation class distribution: {dict(sorted(zip(*np.unique(y_val, return_counts=True))))}")

        # --- 5. Normalize Features using ONLY Training Set Statistics ---
        num_train_samples, seq_len, num_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, num_features)
        
        mean = np.mean(X_train_reshaped, axis=0)
        std = np.std(X_train_reshaped, axis=0)
        std[std == 0] = 1e-8 # Prevent division by zero
        
        X_train_normalized = (X_train_reshaped - mean) / std
        X_train = X_train_normalized.reshape(num_train_samples, seq_len, num_features)

        num_val_samples = X_val.shape[0]
        X_val_reshaped = X_val.reshape(-1, num_features)
        X_val_normalized = (X_val_reshaped - mean) / std
        X_val = X_val_normalized.reshape(num_val_samples, seq_len, num_features)

        norm_stats_path = os.path.join(data_cfg['output_directory'], 'norm_stats.pkl')
        with open(norm_stats_path, 'wb') as f:
            pickle.dump({'mean': mean, 'std': std}, f)
        logger.info(f"Normalization stats (mean/std) saved to {norm_stats_path}")
        
        # --- 6. Train the Classifier ---
        class_mapping = prep_cfg['class_mapping']
        class_names = sorted(class_mapping.keys(), key=lambda k: class_mapping[k])

        train.train_classifier(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            model_params=model_cfg,
            training_params=train_cfg,
            class_names=class_names,
            output_dir=data_cfg['output_directory'],
            device=device,
            model_filename=data_cfg['model_save_filename']
        )
        
        logger.info("--- Training Pipeline Finished Successfully ---")

    except (KeyError, FileNotFoundError) as e:
        logger.error(f"An error occurred: {e}. Please check your config.json and file paths.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during the training pipeline: {e}", exc_info=True)

def main(config_path: str = 'config.json'):
    """Main entry point to load config and run the training pipeline."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}. Aborting.")
        return
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {config_path}. Please check its format.")
        return
    
    run_training_pipeline(config)


if __name__ == '__main__':
    main()