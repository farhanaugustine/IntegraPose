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

# --- MODIFIED: This function now accepts previous keypoints to calculate temporal features ---
def calculate_feature_vector(keypoints, prev_kps_t1, prev_kps_t2, keypoint_names, angle_definitions):
    """Calculates a feature vector including velocities and accelerations."""
    vec = []
    kps_map = {name: (kp[0], kp[1]) for name, kp in zip(keypoint_names, keypoints)}

    # 1. Pairwise distances (same as before)
    points = [np.array(kps_map[name]) for name in keypoint_names]
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            vec.append(np.linalg.norm(points[i] - points[j]))

    # 2. Angles (same as before)
    for p1_name, p2_name, p3_name in angle_definitions:
        p1, p2, p3 = np.array(kps_map[p1_name]), np.array(kps_map[p2_name]), np.array(kps_map[p3_name])
        v1, v2 = p1 - p2, p3 - p2
        dot = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        angle = np.arccos(np.clip(dot / (norms + 1e-8), -1.0, 1.0))
        vec.append(angle)

    # --- CORRECTED: Calculate actual velocities and accelerations ---
    for kp_name in keypoint_names:
        vel, accel = 0.0, 0.0
        if prev_kps_t1 is not None:
            prev_kps_t1_map = {name: (kp[0], kp[1]) for name, kp in zip(keypoint_names, prev_kps_t1)}
            vel = np.linalg.norm(np.array(kps_map[kp_name]) - np.array(prev_kps_t1_map[kp_name]))
            if prev_kps_t2 is not None:
                prev_kps_t2_map = {name: (kp[0], kp[1]) for name, kp in zip(keypoint_names, prev_kps_t2)}
                prev_vel = np.linalg.norm(np.array(prev_kps_t1_map[kp_name]) - np.array(prev_kps_t2_map[kp_name]))
                accel = vel - prev_vel
        vec.extend([vel, accel])
        
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
    
    # NOTE: update this path to YOLO model
    yolo_model_path = "A2C2f_C3k2_C2PSA_Backbone_L-Model.pt"
    yolo_model = YOLO(yolo_model_path)
    logger.info(f"YOLO Pose model loaded from {yolo_model_path}")

    lstm_model_path = os.path.join(data_cfg['output_directory'], data_cfg['model_save_filename'])
    
    # Determine feature vector size from the saved normalization stats
    norm_stats_path = os.path.join(data_cfg['output_directory'], 'norm_stats.pkl')
    with open(norm_stats_path, 'rb') as f:
        norm_stats = pickle.load(f)
    mean = norm_stats['mean']
    std = norm_stats['std']
    feature_vector_size = len(mean) # Dynamically get size
    logger.info(f"Feature vector size set to {feature_vector_size} based on normalization stats.")

    lstm_model = LSTMClassifier(
        input_dim=feature_vector_size,
        hidden_dim=model_cfg['hidden_dim'],
        num_layers=model_cfg['num_layers'],
        num_classes=model_cfg['num_classes']
    ).to(device)
    lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=device))
    lstm_model.eval()
    logger.info(f"LSTM Classifier loaded from {lstm_model_path}")
    
    # --- 2. Setup Inference Buffers and Names ---
    sequence_buffers = defaultdict(lambda: deque(maxlen=seq_cfg['sequence_length']))
    
    # NEW: Buffer to store previous keypoints for each track_id to calculate velocity/acceleration
    keypoint_history = defaultdict(lambda: deque(maxlen=3)) # Stores current, t-1, t-2

    behavior_predictions = defaultdict(lambda: "Waiting...")
    class_names = {0: "Walking", 1: "Wall-Rearing", 2: "Grooming", 3: "Hesitation"}

    # --- 3. Start Real-time Processing ---
    video_source = data_cfg['video_file']
    
    show_yolo_annotations = False
    
    results_generator = yolo_model.track(source=video_source, stream=True, persist=True, max_det=1, show_labels=False, show_conf=False, show_boxes=False)
    
    for results in results_generator:
        frame = results.orig_img
        
        if results.boxes.id is not None:
            track_ids = results.boxes.id.int().cpu().tolist()
            keypoints_data = results.keypoints.xy.cpu().numpy()

            for track_id, keypoints in zip(track_ids, keypoints_data):
                # Update keypoint history for this track ID
                keypoint_history[track_id].append(keypoints)

                # Need at least 3 frames of history to calculate acceleration
                if len(keypoint_history[track_id]) < 3:
                    behavior_predictions[track_id] = "Gathering data..."
                    continue 

                # Get keypoints for t, t-1, and t-2
                current_kps = keypoint_history[track_id][-1]
                prev_kps_t1 = keypoint_history[track_id][-2]
                prev_kps_t2 = keypoint_history[track_id][-3]
                
                # --- CALL THE MODIFIED, STATEFUL FUNCTION ---
                feature_vector = calculate_feature_vector(
                    current_kps, prev_kps_t1, prev_kps_t2,
                    data_cfg['keypoint_names'], feature_cfg['angles_to_compute']
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
        if show_yolo_annotations:
            annotated_frame = results.plot()
        else:
            annotated_frame = results.orig_img.copy()

        if results.boxes.id is not None:
            for box in results.boxes:
                track_id = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = behavior_predictions[track_id]
                cv2.putText(annotated_frame, f"ID {track_id}: {label}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("YOLO + LSTM Tandem Classification", annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("t"):
            show_yolo_annotations = not show_yolo_annotations
            logger.info(f"YOLO annotations toggled {'ON' if show_yolo_annotations else 'OFF'}")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_tandem_inference()