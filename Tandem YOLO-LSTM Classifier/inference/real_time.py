# integrapose_lstm/inference/real_time.py
import json
import logging
import os
import pickle
from collections import defaultdict, deque
from typing import Dict, Deque, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Import our modular components
from data_processing.features import calculate_feature_vector
from training.models import get_model

logger = logging.getLogger(__name__)

def load_models_and_config(config: Dict) -> Tuple:
    """Loads all necessary models, stats, and configuration for inference."""
    # --- Unpack Config ---
    data_cfg = config['data']
    model_cfg = config['lstm_classifier_params']
    prep_cfg = config['dataset_preparation']
    inf_cfg = config['inference_params']
    feature_cfg = config['feature_params']
    seq_cfg = config['sequence_params']

    # Get device from config (can be an int like 0, or a string like 'cpu')
    device_setting = inf_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device_setting}")

    # --- Load YOLO Pose Model ---
    yolo_model_path = inf_cfg['yolo_model_path']
    if not os.path.exists(yolo_model_path):
        raise FileNotFoundError(f"YOLO model not found at: {yolo_model_path}")
    yolo_model = YOLO(yolo_model_path)
    yolo_model.to(device_setting)
    logger.info(f"YOLO Pose model loaded from {yolo_model_path}")

    # --- Load Normalization Stats ---
    output_dir = data_cfg['output_directory']
    norm_stats_path = os.path.join(output_dir, 'norm_stats.pkl')
    if not os.path.exists(norm_stats_path):
        raise FileNotFoundError(f"Normalization stats not found at: {norm_stats_path}")
    with open(norm_stats_path, 'rb') as f:
        norm_stats = pickle.load(f)
    mean, std = norm_stats['mean'], norm_stats['std']
    feature_vector_size = len(mean)
    logger.info(f"Normalization stats loaded. Feature vector size: {feature_vector_size}")

    # --- Load LSTM Classifier ---
    class_mapping = prep_cfg['class_mapping']
    num_classes = len(class_mapping)
    
    model_type = model_cfg.get('model_type', 'lstm')
    lstm_model = get_model(
        model_name=model_type,
        model_params=model_cfg,
        input_dim=feature_vector_size,
        num_classes=num_classes
    ).to(device_setting)
    
    # --- CORRECTED PART: Create a safe map_location for torch.load ---
    # It requires a string ('cpu', 'cuda:0') or a torch.device object, not a raw integer.
    if isinstance(device_setting, int):
        map_location = f'cuda:{device_setting}'
    else:
        map_location = device_setting

    lstm_model_path = inf_cfg['lstm_classifier_path']
    if not os.path.exists(lstm_model_path):
        raise FileNotFoundError(f"LSTM model not found at: {lstm_model_path}")
    
    lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=map_location))
    lstm_model.eval()
    logger.info(f"LSTM Classifier loaded from {lstm_model_path}")

    # --- Prepare Class Names Mapping ---
    class_names = {idx: name for name, idx in class_mapping.items()}

    return (yolo_model, lstm_model, mean, std, class_names, device_setting,
            data_cfg, feature_cfg, seq_cfg, inf_cfg)


def run_real_time_inference(config: Dict):
    """
    Runs the full real-time inference pipeline on a video source.
    """
    try:
        (yolo_model, lstm_model, mean, std, class_names, device_setting,
         data_cfg, feature_cfg, seq_cfg, inf_cfg) = load_models_and_config(config)

        # --- Setup Inference Buffers ---
        sequence_buffers: Dict[int, Deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=seq_cfg['sequence_length'])
        )
        keypoint_history: Dict[int, Deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=3))
        behavior_predictions: Dict[int, str] = defaultdict(lambda: "Initializing...")

        video_source = inf_cfg['video_source']
        if not os.path.exists(video_source):
             raise FileNotFoundError(f"Inference video not found at: {video_source}")

        # --- Start Video Processing ---
        results_generator = yolo_model.track(
            source=video_source,
            stream=True,
            persist=True,
            conf=inf_cfg.get('yolo_confidence_threshold', 0.5)
        )

        for results in results_generator:
            frame = results.orig_img.copy()

            if results.boxes is not None and results.boxes.id is not None:
                track_ids = results.boxes.id.int().cpu().tolist()
                keypoints = results.keypoints  # The main Keypoints object

                # --- CORRECTED PART: Manually combine xyn and conf ---
                # Get all keypoint data as numpy arrays once per frame
                keypoints_xy_norm = keypoints.xyn.cpu().numpy()
                keypoints_conf = keypoints.conf.cpu().numpy()

                for i, track_id in enumerate(track_ids):
                    # For the i-th detected instance, get its xy coordinates and confidences
                    xy_for_instance = keypoints_xy_norm[i]
                    conf_for_instance = keypoints_conf[i]

                    # Manually combine them into a (num_keypoints, 3) numpy array
                    conf_reshaped = conf_for_instance.reshape(-1, 1)
                    keypoints_for_instance_np = np.concatenate((xy_for_instance, conf_reshaped), axis=1)

                    # Now, append the correctly formatted NumPy array.
                    keypoint_history[track_id].append(keypoints_for_instance_np)

                    if len(keypoint_history[track_id]) < 3:
                        continue

                    # --- Feature Calculation ---
                    current_kps = keypoint_history[track_id][-1]
                    prev_kps_t1 = keypoint_history[track_id][-2]
                    prev_kps_t2 = keypoint_history[track_id][-3]
                    
                    feature_vec = calculate_feature_vector(
                        current_kps, prev_kps_t1, prev_kps_t2,
                        data_cfg['keypoint_names'], feature_cfg['angles_to_compute']
                    )
                    sequence_buffers[track_id].append(feature_vec)

                    # --- Prediction ---
                    if len(sequence_buffers[track_id]) == seq_cfg['sequence_length']:
                        sequence_data = np.array(sequence_buffers[track_id])
                        normalized_sequence = (sequence_data - mean) / (std + 1e-8)
                        seq_tensor = torch.from_numpy(normalized_sequence).unsqueeze(0).float().to(device_setting)

                        with torch.no_grad():
                            output = lstm_model(seq_tensor)
                            probabilities = torch.softmax(output, dim=1)
                            confidence, pred_idx_tensor = torch.max(probabilities, 1)
                        
                        pred_idx = pred_idx_tensor.item()
                        pred_behavior = class_names.get(pred_idx, "Unknown")
                        behavior_predictions[track_id] = f"{pred_behavior} ({confidence.item():.2f})"

            # --- Visualization ---
            annotated_frame = results.plot()  # YOLO's built-in plotting
            if results.boxes is not None and results.boxes.id is not None:
                for box in results.boxes:
                    track_id = int(box.id.item())
                    x1, y1, _, _ = map(int, box.xyxy[0])
                    label = behavior_predictions[track_id]
                    cv2.putText(annotated_frame, f"ID {track_id}: {label}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("IntegraPose - Real-Time LSTM Classification", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except (KeyError, FileNotFoundError) as e:
        logger.error(f"An error occurred: {e}. Please check your config.json and file paths.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during inference: {e}", exc_info=True)
    finally:
        cv2.destroyAllWindows()