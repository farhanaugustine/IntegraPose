# run_training.py
import json
import logging
import torch
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from data_loader import read_detections, compute_feature_vectors, create_labeled_sequences
from training import train_classifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    with open('config.json', 'r') as f:
        config = json.load(f)

    data_cfg = config['data']
    prep_cfg = config['preprocessing']
    feature_cfg = config['feature_params']
    seq_cfg = config['sequence_params']
    model_cfg = config['lstm_classifier_params']
    train_cfg = config['training_params']
    
    os.makedirs(data_cfg['output_directory'], exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # 1. Load and Preprocess Data
    detections_df = read_detections(data_cfg['pose_directory'], data_cfg['keypoint_names'])
    features_df = compute_feature_vectors(
        detections_df, data_cfg['keypoint_names'], 
        feature_cfg['angles_to_compute'], prep_cfg['confidence_threshold']
    )
    features_df.dropna(subset=['feature_vector'], inplace=True)

    # 2. Create Labeled Sequences
    X, y = create_labeled_sequences(features_df, seq_cfg['sequence_length'])

    # Block to analyze class distribution
    unique_classes, counts = np.unique(y, return_counts=True)
    class_distribution = dict(zip(unique_classes, counts))
    logger.info(f"Class distribution in sequences: {class_distribution}")

    # 3. Normalize Features
    # Reshape for normalization: (num_sequences * sequence_length, num_features)
    num_samples, seq_len, num_features = X.shape
    X_reshaped = X.reshape(-1, num_features)
    
    mean = np.mean(X_reshaped, axis=0)
    std = np.std(X_reshaped, axis=0)
    std[std == 0] = 1e-8 # Avoid division by zero
    
    X_normalized = (X_reshaped - mean) / std
    X = X_normalized.reshape(num_samples, seq_len, num_features)

    # Save normalization stats for inference
    norm_stats_path = os.path.join(data_cfg['output_directory'], 'norm_stats.pkl')
    with open(norm_stats_path, 'wb') as f:
        pickle.dump({'mean': mean, 'std': std}, f)
    logger.info(f"Normalization stats saved to {norm_stats_path}")

    # 4. Split Data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=train_cfg['test_split_ratio'], random_state=42, stratify=y
    )
    logger.info(f"Data split: {len(X_train)} training samples, {len(X_val)} validation samples.")
    
    # 5. Train Classifier
    train_classifier(
        X_train, y_train, X_val, y_val, 
        model_cfg, train_cfg, 
        data_cfg['output_directory'], device, 
        data_cfg['model_save_filename']
    )
    
    logger.info("Training complete!")

if __name__ == '__main__':
    main()