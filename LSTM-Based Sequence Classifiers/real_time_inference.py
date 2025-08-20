# real_time_inference.py
import json
import logging
import torch
import numpy as np
import os
import pickle
import cv2
from ultralytics import YOLO
from collections import defaultdict, deque
from models import LSTMClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# This is a simplified version of your feature calculation logic
# It's adapted to work with the live YOLO output format
def calculate_feature_vector(keypoints, keypoint_names, angle_definitions):
    """Calculates a feature vector from a single frame's keypoints."""
    # Note: Velocity/acceleration are omitted for this real-time example,
    # as they require tracking state across frames, which is handled by the buffer.
    # The LSTM will learn these temporal dynamics from the sequence of features.
    
    vec = []
    kps_map = {name: (kp[0], kp[1]) for name, kp in zip(keypoint_names, keypoints)}

    # 1. Pairwise distances
    points = [np.array(kps_map[name]) for name in keypoint_names]
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            vec.append(np.linalg.norm(points[i] - points[j]))

    # 2. Angles
    for p1_name, p2_name, p3_name in angle_definitions:
        p1, p2, p3 = np.array(kps_map[p1_name]), np.array(kps_map[p2_name]), np.array(kps_map[p3_name])
        v1, v2 = p1 - p2, p3 - p2
        dot = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        angle = np.arccos(np.clip(dot / (norms + 1e-8), -1.0, 1.0))
        vec.append(angle)

    # Placeholder for velocities/accelerations (2 features per keypoint)
    # The LSTM learns this from the sequence, so we can use zeros or simple deltas
    # For simplicity, we add placeholders to match the trained input_dim.
    vec.extend([0] * len(keypoint_names) * 2)

    return np.array(vec, dtype=np.float32)


def run_tandem_inference():
    # --- 1. Load Configurations and Models ---
    with open('config.json', 'r') as f:
        config = json.load(f)

    data_cfg = config['data']
    feature_cfg = config['feature_params']
    seq_cfg = config['sequence_params']
    model_cfg = config['lstm_classifier_params']
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    yolo_model_path = "A2C2f_C3k2_C2PSA_Backbone_L-Model.pt"
    yolo_model = YOLO(yolo_model_path)
    logger.info(f"YOLO Pose model loaded from {yolo_model_path}")

    lstm_model_path = os.path.join(data_cfg['output_directory'], data_cfg['model_save_filename'])
    feature_vector_size = 93
    lstm_model = LSTMClassifier(
        input_dim=feature_vector_size,
        hidden_dim=model_cfg['hidden_dim'],
        num_layers=model_cfg['num_layers'],
        num_classes=model_cfg['num_classes']
    ).to(device)
    lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=device))
    lstm_model.eval()
    logger.info(f"LSTM Classifier loaded from {lstm_model_path}")

    norm_stats_path = os.path.join(data_cfg['output_directory'], 'norm_stats.pkl')
    with open(norm_stats_path, 'rb') as f:
        norm_stats = pickle.load(f)
    mean = norm_stats['mean']
    std = norm_stats['std']
    
    # --- 2. Setup Inference Buffers and Names ---
    sequence_buffers = defaultdict(lambda: deque(maxlen=seq_cfg['sequence_length']))
    behavior_predictions = defaultdict(lambda: "Waiting...")
    class_names = {0: "Walking", 1: "Wall-Rearing", 2: "Rearing", 3: "Grooming"}

    # --- 3. Start Real-time Processing ---
    video_source = data_cfg['video_file']
    
    # <-- NEW: Initialize a boolean flag to control YOLO annotations
    show_yolo_annotations = False
    
    results_generator = yolo_model.track(source=video_source, stream=True, persist=True, max_det=1, show_labels=False, show_conf=False, show_boxes=False)
    
    for results in results_generator:
        frame = results.orig_img
        
        if results.boxes.id is not None:
            track_ids = results.boxes.id.int().cpu().tolist()
            keypoints_data = results.keypoints.xy.cpu().numpy()

            for track_id, keypoints in zip(track_ids, keypoints_data):
                feature_vector = calculate_feature_vector(
                    keypoints, data_cfg['keypoint_names'], feature_cfg['angles_to_compute']
                )
                
                sequence_buffers[track_id].append(feature_vector)
                
                if len(sequence_buffers[track_id]) == seq_cfg['sequence_length']:
                    sequence_data = np.array(sequence_buffers[track_id], dtype=np.float32)
                    normalized_sequence = (sequence_data - mean) / std
                    sequence_tensor = torch.from_numpy(normalized_sequence).unsqueeze(0).to(device)
                    length_tensor = torch.tensor([seq_cfg['sequence_length']], dtype=torch.long)
                    
                    with torch.no_grad():
                        output = lstm_model(sequence_tensor, length_tensor)
                        probabilities = torch.softmax(output, dim=1)
                        confidence, predicted_idx = torch.max(probabilities, 1)
                    
                    predicted_behavior = class_names.get(predicted_idx.item(), "Unknown")
                    behavior_predictions[track_id] = f"{predicted_behavior} ({confidence.item():.2f})"

        # --- 4. Visualize the Results ---
        
        # <-- MODIFIED: Conditionally choose the frame to display
        if show_yolo_annotations:
            # Use the frame with YOLO's skeleton drawings
            annotated_frame = results.plot()
        else:
            # Use the original, un-annotated frame. Use .copy() to avoid modifying the original.
            annotated_frame = results.orig_img.copy()

        # Add the LSTM behavior predictions to the chosen frame
        if results.boxes.id is not None:
            for box in results.boxes:
                track_id = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = behavior_predictions[track_id]
                cv2.putText(annotated_frame, f"ID {track_id}: {label}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("YOLO + LSTM Tandem Classification", annotated_frame)
        
        # <-- MODIFIED: Add a key listener to toggle the flag
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("t"): # 't' for toggle
            show_yolo_annotations = not show_yolo_annotations
            logger.info(f"YOLO annotations toggled {'ON' if show_yolo_annotations else 'OFF'}")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_tandem_inference()