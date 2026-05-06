import csv
import logging
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import cv2
import numpy as np
import torch

from ..data_processing.features import calculate_feature_vector
from ..training.models import get_model
from integra_pose.utils.safe_model_io import load_normalization_stats, load_torch_artifact

logger = logging.getLogger(__name__)


def load_models_and_config(config: Dict) -> Tuple:
    """Load YOLO, LSTM, and supporting metadata required for tandem inference."""
    data_cfg = config["data"]
    model_cfg = config["lstm_classifier_params"]
    prep_cfg = config["dataset_preparation"]
    inf_cfg = config["inference_params"]
    feature_cfg = config["feature_params"]
    seq_cfg = config["sequence_params"]

    device_setting = inf_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device_setting)

    yolo_model_path = inf_cfg["yolo_model_path"]
    if not os.path.exists(yolo_model_path):
        raise FileNotFoundError(f"YOLO model not found at: {yolo_model_path}")
    from ultralytics import YOLO

    yolo_model = YOLO(yolo_model_path)
    yolo_model.to(device_setting)
    logger.info("YOLO pose model loaded from %s", yolo_model_path)

    output_dir = data_cfg["output_directory"]
    norm_stats_path = Path(output_dir) / "norm_stats.npz"
    mean, std = load_normalization_stats(norm_stats_path)
    feature_vector_size = len(mean)
    logger.info("Normalization stats loaded (feature vector size: %s)", feature_vector_size)

    class_mapping = prep_cfg["class_mapping"]
    num_classes = len(class_mapping)

    model_type = model_cfg.get("model_type", "lstm")
    lstm_model = get_model(
        model_name=model_type,
        model_params=model_cfg,
        input_dim=feature_vector_size,
        num_classes=num_classes,
    ).to(device_setting)

    if isinstance(device_setting, int):
        map_location = f"cuda:{device_setting}"
    else:
        map_location = device_setting

    lstm_model_path = inf_cfg["lstm_classifier_path"]
    if not os.path.exists(lstm_model_path):
        raise FileNotFoundError(f"LSTM model not found at: {lstm_model_path}")

    lstm_model.load_state_dict(
        load_torch_artifact(lstm_model_path, map_location=map_location, description="tandem LSTM weights")
    )
    lstm_model.eval()
    logger.info("LSTM classifier loaded from %s", lstm_model_path)

    class_names = {idx: name for name, idx in class_mapping.items()}

    return (
        yolo_model,
        lstm_model,
        mean,
        std,
        class_names,
        device_setting,
        data_cfg,
        feature_cfg,
        seq_cfg,
        inf_cfg,
    )


def run_real_time_inference(config: Dict) -> None:
    """Run YOLO pose tracking and LSTM behaviour classification in tandem."""
    video_writer = None
    output_dir: Path | None = None
    video_path: Path | None = None
    csv_records: List[Dict[str, object]] = []

    try:
        (
            yolo_model,
            lstm_model,
            mean,
            std,
            class_names,
            device_setting,
            data_cfg,
            feature_cfg,
            seq_cfg,
            inf_cfg,
        ) = load_models_and_config(config)

        sequence_buffers: Dict[int, Deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=seq_cfg["sequence_length"])
        )
        keypoint_history: Dict[int, Deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=3))
        behavior_predictions: Dict[int, Dict[str, object]] = defaultdict(
            lambda: {"class_id": None, "label": None, "confidence": 0.0}
        )

        video_source = inf_cfg["video_source"]
        if not os.path.exists(video_source):
            raise FileNotFoundError(f"Inference video not found at: {video_source}")

        video_path = Path(video_source)
        output_dir = Path(inf_cfg.get("output_directory", "inference_output")).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        pose_txt_dir = output_dir / "pose_sequences"
        pose_txt_dir.mkdir(parents=True, exist_ok=True)

        frame_index = -1
        save_annotated_video = bool(inf_cfg.get("save_annotated_video"))
        video_fps = 0.0
        frame_size: tuple[int, int] | None = None

        if save_annotated_video:
            capture = cv2.VideoCapture(str(video_source))
            if capture.isOpened():
                video_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                if width > 0 and height > 0:
                    frame_size = (width, height)
            capture.release()

        yolo_names: Dict[int, str] = {}
        try:
            # Ultralytics models expose class names in different locations depending on version
            if hasattr(yolo_model, "names") and isinstance(yolo_model.names, (list, tuple, dict)):
                if isinstance(yolo_model.names, dict):
                    yolo_names = {int(k): v for k, v in yolo_model.names.items()}
                else:
                    yolo_names = {idx: name for idx, name in enumerate(yolo_model.names)}
            elif hasattr(yolo_model.model, "names"):
                yolo_names = {int(k): v for k, v in yolo_model.model.names.items()}
        except Exception:  # pragma: no cover - optional metadata
            yolo_names = {}

        results_generator = yolo_model.track(
            source=video_source,
            stream=True,
            persist=True,
            conf=inf_cfg.get("yolo_confidence_threshold", 0.5),
        )

        behavior_id_to_name = class_names

        for results in results_generator:
            frame = results.orig_img.copy()
            frame_index += 1

            if frame_size is None:
                frame_size = (frame.shape[1], frame.shape[0])
            if save_annotated_video and video_writer is None and frame_size is not None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                annotated_path = output_dir / f"{video_path.stem}_annotated.mp4"
                video_writer = cv2.VideoWriter(
                    str(annotated_path), fourcc, video_fps or 30.0, frame_size
                )

            pose_lines = []
            if results.boxes is not None and results.boxes.id is not None:
                track_ids = results.boxes.id.int().cpu().tolist()
                keypoints = results.keypoints

                keypoints_xy_norm = keypoints.xyn.cpu().numpy()
                keypoints_conf = keypoints.conf.cpu().numpy()
                boxes_xywhn = results.boxes.xywhn.cpu().numpy()
                boxes_conf = results.boxes.conf.cpu().numpy()
                boxes_cls = results.boxes.cls.int().cpu().tolist()

                for idx, track_id in enumerate(track_ids):
                    xy_for_instance = keypoints_xy_norm[idx]
                    conf_for_instance = keypoints_conf[idx]

                    conf_reshaped = conf_for_instance.reshape(-1, 1)
                    keypoints_for_instance_np = np.concatenate(
                        (xy_for_instance, conf_reshaped), axis=1
                    )

                    keypoint_history[track_id].append(keypoints_for_instance_np)

                    if len(keypoint_history[track_id]) < 3:
                        continue

                    current_kps = keypoint_history[track_id][-1]
                    prev_kps_t1 = keypoint_history[track_id][-2]
                    prev_kps_t2 = keypoint_history[track_id][-3]

                    feature_vec = calculate_feature_vector(
                        current_kps,
                        prev_kps_t1,
                        prev_kps_t2,
                        data_cfg["keypoint_names"],
                        feature_cfg["angles_to_compute"],
                    )
                    sequence_buffers[track_id].append(feature_vec)

                    if len(sequence_buffers[track_id]) == seq_cfg["sequence_length"]:
                        sequence_data = np.array(sequence_buffers[track_id])
                        normalized_sequence = (sequence_data - mean) / (std + 1e-8)
                        seq_tensor = (
                            torch.from_numpy(normalized_sequence)
                            .unsqueeze(0)
                            .float()
                            .to(device_setting)
                        )

                        with torch.no_grad():
                            output = lstm_model(seq_tensor)
                            probabilities = torch.softmax(output, dim=1)
                            confidence, pred_idx_tensor = torch.max(probabilities, 1)

                        pred_idx = pred_idx_tensor.item()
                        pred_behavior = behavior_id_to_name.get(pred_idx, "Unknown")
                        behavior_predictions[track_id] = {
                            "class_id": pred_idx,
                            "label": pred_behavior,
                            "confidence": float(confidence.item()),
                        }

                for idx, track_id in enumerate(track_ids):
                    xy_for_instance = keypoints_xy_norm[idx]
                    conf_for_instance = keypoints_conf[idx]
                    bbox = boxes_xywhn[idx]
                    yolo_confidence = float(boxes_conf[idx])
                    yolo_class_id = boxes_cls[idx]
                    yolo_label = yolo_names.get(yolo_class_id, str(yolo_class_id))

                    pred_info = behavior_predictions[track_id]
                    behaviour_class_id = pred_info.get("class_id")
                    behaviour_label = pred_info.get("label")
                    behaviour_confidence = pred_info.get("confidence")

                    effective_class_id = (
                        behaviour_class_id if behaviour_class_id is not None else yolo_class_id
                    )
                    effective_label = behaviour_label if behaviour_label else yolo_label

                    kpts_flat = []
                    for kp_idx in range(len(xy_for_instance)):
                        x_norm, y_norm = xy_for_instance[kp_idx]
                        kp_conf = conf_for_instance[kp_idx]
                        kpts_flat.extend([f"{x_norm:.6f}", f"{y_norm:.6f}", f"{kp_conf:.6f}"])

                    line_parts = [
                        str(effective_class_id),
                        f"{bbox[0]:.6f}",
                        f"{bbox[1]:.6f}",
                        f"{bbox[2]:.6f}",
                        f"{bbox[3]:.6f}",
                        *kpts_flat,
                    ]
                    line_parts.append(str(track_id))
                    pose_lines.append(" ".join(line_parts))

                    csv_records.append(
                        {
                            "frame_index": frame_index,
                            "track_id": track_id,
                            "behavior_class_id": behaviour_class_id,
                            "behavior_label": behaviour_label,
                            "behavior_confidence": behaviour_confidence,
                            "yolo_class_id": yolo_class_id,
                            "yolo_label": yolo_label,
                            "yolo_confidence": yolo_confidence,
                            "bbox_x_center": float(bbox[0]),
                            "bbox_y_center": float(bbox[1]),
                            "bbox_width": float(bbox[2]),
                            "bbox_height": float(bbox[3]),
                        }
                    )

            pose_output_path = pose_txt_dir / f"{video_path.stem}_{frame_index:06d}.txt"
            with open(pose_output_path, "w", encoding="utf-8") as pose_file:
                pose_file.write("\n".join(pose_lines))

            annotated_frame = results.plot()
            if results.boxes is not None and results.boxes.id is not None:
                for box in results.boxes:
                    track_id = int(box.id.item())
                    x1, y1, _, _ = map(int, box.xyxy[0])
                    pred_info = behavior_predictions[track_id]
                    if pred_info.get("label"):
                        label_text = f"{pred_info['label']} ({pred_info['confidence']:.2f})"
                    else:
                        label_text = "Initializing..."
                    cv2.putText(
                        annotated_frame,
                        f"ID {track_id}: {label_text}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            if video_writer is not None:
                video_writer.write(annotated_frame)

            cv2.imshow("IntegraPose - Real-Time LSTM Classification", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except (KeyError, FileNotFoundError) as exc:
        logger.error("An error occurred: %s. Please check your config.json and file paths.", exc)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("An unexpected error occurred during inference: %s", exc, exc_info=True)
    finally:
        cv2.destroyAllWindows()
        if video_writer is not None:
            video_writer.release()
            if output_dir is not None and video_path is not None:
                logger.info(
                    "Annotated video saved to %s",
                    output_dir / f"{video_path.stem}_annotated.mp4",
                )

        if csv_records and output_dir is not None and video_path is not None:
            csv_path = output_dir / f"{video_path.stem}_predictions.csv"
            fieldnames = [
                "frame_index",
                "track_id",
                "behavior_class_id",
                "behavior_label",
                "behavior_confidence",
                "yolo_class_id",
                "yolo_label",
                "yolo_confidence",
                "bbox_x_center",
                "bbox_y_center",
                "bbox_width",
                "bbox_height",
            ]
            with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_records)
            logger.info("Saved unified inference summary to %s", csv_path)
