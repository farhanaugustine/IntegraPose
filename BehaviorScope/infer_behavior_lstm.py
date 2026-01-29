from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Sequence

# [CHANGE 1] Filter out the PyTorch TypedStorage deprecation warning
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

import numpy as np
import cv2
import torch
from torch import nn
from torch.nn import functional as F

if __package__ is None or __package__ == "":  # pragma: no cover - CLI entry
    from cropping import (
        CropDetection,
        iterate_yolo_crops,
        load_yolo_model,
    )
    import quantification
    import review_annotations
    from data import build_frame_transform
    from model import BehaviorSequenceClassifier
    from utils.logging_utils import setup_logger
    from utils.pose_features import extract_rich_pose_features
else:
    from .cropping import CropDetection, iterate_yolo_crops, load_yolo_model
    from . import quantification
    from . import review_annotations
    from .data import build_frame_transform
    from .model import BehaviorSequenceClassifier
    from .utils.logging_utils import setup_logger
    from .utils.pose_features import extract_rich_pose_features


@dataclass
class WindowPrediction:
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    class_idx: int | None
    class_name: str
    confidence: float


class JsonStreamWriter:
    def __init__(self, path: Path | None):
        self.path = path
        self.file = None
        self.first = True
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.file = path.open("w", encoding="utf-8")
            self.file.write("[\n")

    def write(self, prediction: WindowPrediction):
        if not self.file:
            return
        if not self.first:
            self.file.write(",\n")
        # Keep JSON schema stable (top-1 per-window only).
        json.dump(
            {
                "start_time": prediction.start_time,
                "end_time": prediction.end_time,
                "start_frame": prediction.start_frame,
                "end_frame": prediction.end_frame,
                "class_idx": prediction.class_idx,
                "class_name": prediction.class_name,
                "confidence": prediction.confidence,
            },
            self.file,
        )
        self.file.flush()
        self.first = False

    def close(self):
        if not self.file:
            return
        if self.first:
            self.file.write("]\n")
        else:
            self.file.write("\n]\n")
        self.file.close()
        self.file = None


class CsvMergeWriter:
    def __init__(self, path: Path, keep_unknown: bool):
        self.path = path
        self.keep_unknown = keep_unknown
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            [
                "start_time",
                "end_time",
                "start_frame",
                "end_frame",
                "label",
                "confidence",
            ]
        )
        self.current: WindowPrediction | None = None
        self.count = 0
        self.conf_sum = 0.0

    def _flush_current(self):
        if self.current is None:
            return
        self.current.confidence = self.conf_sum / max(self.count, 1)
        self.writer.writerow(
            [
                f"{self.current.start_time:.3f}",
                f"{self.current.end_time:.3f}",
                self.current.start_frame,
                self.current.end_frame,
                self.current.class_name,
                f"{self.current.confidence:.3f}",
            ]
        )
        self.file.flush()
        self.current = None
        self.count = 0
        self.conf_sum = 0.0

    def add(self, prediction: WindowPrediction):
        if not self.keep_unknown and prediction.class_idx is None:
            self._flush_current()
            return

        if (
            self.current is not None
            and self.current.class_idx == prediction.class_idx
            and self.current.class_name == prediction.class_name
        ):
            self.current.end_frame = prediction.end_frame
            self.current.end_time = prediction.end_time
            self.count += 1
            self.conf_sum += prediction.confidence
        else:
            self._flush_current()
            self.current = replace(prediction)
            self.count = 1
            self.conf_sum = prediction.confidence

    def close(self):
        self._flush_current()
        self.file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a trained behavior LSTM model on a long video."
    )
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--video_path", type=Path, required=True)
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Path for the merged behavior annotations (.csv).",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional JSON dump of per-window predictions.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help="Number of frames per inference window (defaults to training num_frames).",
    )
    parser.add_argument(
        "--window_stride",
        type=int,
        default=None,
        help="Stride (in frames) between inference windows. Defaults to window_size//2.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--prob_threshold",
        type=float,
        default=0.5,
        help="Minimum probability to accept a prediction. Otherwise labeled as 'Unknown'.",
    )
    parser.add_argument(
        "--keep_unknown",
        action="store_true",
        help="Include unknown segments in the merged CSV.",
    )
    parser.add_argument(
        "--skip_tail",
        action="store_true",
        help="Skip the final partial window instead of forcing an evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=None,
        help=(
            "Optional: set torch intra-op thread count (CPU only). "
            "Good starting point: physical core count."
        ),
    )
    parser.add_argument(
        "--cpu_interop_threads",
        type=int,
        default=None,
        help=(
            "Optional: set torch inter-op thread count (CPU only). "
            "Good starting point: 1-2."
        ),
    )
    parser.add_argument(
        "--cpu_quantize",
        type=str,
        default="none",
        choices=("none", "dynamic"),
        help=(
            "Optional: apply post-training quantization for CPU inference. "
            "'dynamic' quantizes Linear/LSTM weights to int8 (no calibration "
            "data required)."
        ),
    )
    parser.add_argument("--yolo_weights", type=Path, required=True)
    parser.add_argument(
        "--yolo_mode",
        type=str,
        default="auto",
        choices=("auto", "bbox", "pose"),
        help=(
            "How to interpret YOLO outputs: "
            "'auto' uses keypoints if present, "
            "'bbox' forces bbox-only (ignores keypoints), "
            "'pose' requires keypoints (errors if missing)."
        ),
    )
    parser.add_argument(
        "--require_pose",
        action="store_true",
        help=(
            "If the loaded model expects pose features, require pose to be present for every "
            "inference window. Errors if pose is missing (useful for strict pose-vs-RGB comparisons)."
        ),
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=None,
        help="Resize crop size (defaults to training frame_size).",
    )
    parser.add_argument("--pad_ratio", type=float, default=0.2)
    parser.add_argument("--yolo_conf", type=float, default=0.25)
    parser.add_argument("--yolo_iou", type=float, default=0.45)
    parser.add_argument("--yolo_imgsz", type=int, default=640)
    parser.add_argument(
        "--yolo_max_det",
        type=int,
        default=1,
        help="Maximum detections per frame (locked to 1 for single-animal videos).",
    )
    parser.add_argument(
        "--no_yolo_nms",
        action="store_true",
        help="Disable YOLO non-maximum suppression (ignored; NMS is locked on for single-animal videos).",
    )
    parser.add_argument("--keep_last_box", action="store_true")
    parser.add_argument(
        "--progress_interval",
        type=int,
        default=50,
        help="Print progress every N processed windows (0 to disable).",
    )
    parser.add_argument(
        "--output_xlsx",
        type=Path,
        default=None,
        help="Optional path to export per-frame YOLO+behavior metrics (.xlsx).",
    )
    parser.add_argument(
        "--per_frame_mode",
        type=str,
        default="true",
        choices=["none", "simple", "true"],
        help="Per-frame behavior mapping: none/simple(nearest window center)/true(average window probs).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Top-K behavior probabilities to export (1-5).",
    )
    parser.add_argument(
        "--smoothing",
        type=str,
        default="none",
        choices=["none", "ema", "kalman", "interpolation"],
        help="Smoothing method for bbox centers when computing kinematics.",
    )
    parser.add_argument(
        "--ema_alpha",
        type=float,
        default=0.2,
        help="EMA alpha for smoothing (if smoothing=ema).",
    )
    parser.add_argument(
        "--interp_max_gap",
        type=int,
        default=5,
        help="Maximum gap (frames) to interpolate (if smoothing=interpolation).",
    )
    parser.add_argument(
        "--output_review_video",
        type=Path,
        default=None,
        help="Optional path to export an annotated MP4 review video.",
    )
    parser.add_argument(
        "--no_draw_bbox",
        dest="draw_bbox",
        action="store_false",
        help=(
            "Disable drawing the YOLO bbox/keypoints overlay in review videos. "
            "When disabled, review videos use a label-only overlay."
        ),
    )
    parser.set_defaults(draw_bbox=True)
    parser.add_argument(
        "--font_scale",
        type=float,
        default=0.7,
        help="OpenCV font scale for overlays in review videos.",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=2,
        help="Text/border thickness for overlays in review videos.",
    )
    parser.add_argument(
        "--label_bg_alpha",
        type=float,
        default=1.0,
        help=(
            "Opacity (0-1) for the label background rectangles in review videos. "
            "Lower values reduce occlusion during QA."
        ),
    )
    args = parser.parse_args()
    if args.yolo_max_det <= 0:
        raise SystemExit("--yolo_max_det must be >= 1.")
    if args.label_bg_alpha < 0.0 or args.label_bg_alpha > 1.0:
        raise SystemExit("--label_bg_alpha must be between 0 and 1.")
    if args.thickness <= 0:
        raise SystemExit("--thickness must be >= 1.")
    if args.font_scale <= 0:
        raise SystemExit("--font_scale must be > 0.")
    return args


def apply_cpu_optimizations(
    model: BehaviorSequenceClassifier,
    *,
    quantize: str = "none",
    num_threads: int | None = None,
    num_interop_threads: int | None = None,
) -> BehaviorSequenceClassifier:
    if num_threads is not None and int(num_threads) > 0:
        torch.set_num_threads(int(num_threads))
    if num_interop_threads is not None and int(num_interop_threads) > 0:
        torch.set_num_interop_threads(int(num_interop_threads))

    quantize = str(quantize or "none").strip().lower()
    if quantize == "none":
        return model
    if quantize != "dynamic":
        raise ValueError(f"Unknown CPU quantization mode: {quantize!r}")

    try:
        torch.backends.quantized.engine = "fbgemm"
    except Exception:
        pass

    try:
        from torch.ao.quantization import quantize_dynamic
    except Exception as exc:  # pragma: no cover - torch build dependent
        raise RuntimeError(
            "Dynamic quantization is not available in this PyTorch build."
        ) from exc

    model = quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM},
        dtype=torch.qint8,
    )
    return model


# [FIX] Added logger=None default to fix GUI compatibility
def load_checkpoint(model_path: Path, device: torch.device, logger=None):
    # If called from GUI without a logger, create a simple print-based fallback
    if logger is None:
        class SimpleLogger:
            def info(self, msg): print(f"[INFO] {msg}")
            def warning(self, msg): print(f"[WARNING] {msg}")
            def error(self, msg): print(f"[ERROR] {msg}")
        logger = SimpleLogger()

    logger.info(f"Loading model from {model_path}...")
    
    # 1. Attempt to load the file
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load model file: {e}")

    # 2. Check: Is it a Dictionary (Standard Checkpoint) or a Model Object (Quantized)?
    if isinstance(checkpoint, dict):
        # === OLD PATH: Dictionary Checkpoint ===
        logger.info("Detected standard checkpoint dictionary.")
        hparams = checkpoint.get("hparams", {})
        idx_to_class = checkpoint.get("idx_to_class") or hparams.get("idx_to_class")
        
        if idx_to_class is None:
            raise ValueError("Checkpoint is missing idx_to_class metadata.")

        idx_to_class = {int(k): v for k, v in idx_to_class.items()}
        class_names = [idx_to_class[idx] for idx in sorted(idx_to_class.keys())]
        num_classes = len(class_names)

        # Detect pose params in state_dict if hparams missing them
        state_dict = checkpoint["model_state"]
        pose_input_dim = int(hparams.get("pose_input_dim", 0))
        pose_fusion_dim = int(hparams.get("pose_fusion_dim", 128))
        pose_fusion_strategy = hparams.get("pose_fusion_strategy", "gated_attention")

        if pose_input_dim == 0 and "pose_encoder.0.weight" in state_dict:
            logger.info("Auto-detected OLD pose_encoder in checkpoint weights. Migrating to PoseVisualFusion(concat).")
            w_pose = state_dict["pose_encoder.0.weight"]
            # Linear(in, out) -> weight shape is [out, in]
            pose_fusion_dim = w_pose.shape[0]
            pose_input_dim = w_pose.shape[1]
            pose_fusion_strategy = "concat"
            
            # Remap keys to new module structure
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("pose_encoder."):
                    new_state_dict[f"pose_fusion.{k}"] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
            logger.info(f"Inferred pose config: input_dim={pose_input_dim}, fusion_dim={pose_fusion_dim} (legacy concat)")
            
        elif pose_input_dim == 0 and "pose_fusion.pose_encoder.0.weight" in state_dict:
            logger.info("Auto-detected NEW pose_fusion in checkpoint weights.")
            w_pose = state_dict["pose_fusion.pose_encoder.0.weight"]
            pose_fusion_dim = w_pose.shape[0]
            pose_input_dim = w_pose.shape[1]
            logger.info(f"Inferred pose config: input_dim={pose_input_dim}, fusion_dim={pose_fusion_dim}")

        # Rebuild the architecture
        model = BehaviorSequenceClassifier(
            num_classes=num_classes,
            backbone=hparams.get("backbone", "mobilenet_v3_small"),
            pretrained_backbone=False,
            train_backbone=False,
            hidden_dim=int(hparams.get("hidden_dim", 256)),
            num_layers=int(hparams.get("num_layers", 1)),
            dropout=float(hparams.get("dropout", 0.2)),
            bidirectional=bool(hparams.get("bidirectional", False)),
            sequence_model=hparams.get("sequence_model", "lstm"),
            temporal_attention_layers=int(hparams.get("temporal_attention_layers", 0)),
            attention_heads=int(hparams.get("attention_heads", 4)),
            positional_encoding=hparams.get("positional_encoding", "none"),
            positional_encoding_max_len=int(hparams.get("positional_encoding_max_len", 512)),
            use_attention_pooling=bool(hparams.get("attention_pool", False)),
            use_feature_se=bool(hparams.get("feature_se", False)),
            slowfast_alpha=int(hparams.get("slowfast_alpha", 4)),
            slowfast_fusion_ratio=float(hparams.get("slowfast_fusion_ratio", 0.25)),
            slowfast_base_channels=int(hparams.get("slowfast_base_channels", 48)),
            pose_input_dim=pose_input_dim,
            pose_fusion_dim=pose_fusion_dim,
            pose_fusion_strategy=pose_fusion_strategy,
        ).to(device)
        
        # Load weights
        model.load_state_dict(state_dict)
        model.eval()
        return model, class_names, hparams

    else:
        # === NEW PATH: Quantized Model Object ===
        logger.info("Detected ready-to-use Model Object (likely Quantized).")
        model = checkpoint
        model.to(device)
        model.eval()

        # Attempt to recover class names (metadata is often lost in quantization)
        num_classes = getattr(model, "num_classes", None)
        
        # Fallback if num_classes is missing
        if num_classes is None:
            try:
                if hasattr(model, "head") and isinstance(model.head[-1], nn.Linear):
                     num_classes = model.head[-1].out_features
            except:
                pass
        
        if num_classes is None:
             raise ValueError("Could not determine number of classes from the quantized model. "
                              "Ensure the model object has a 'num_classes' attribute.")

        # Generate generic names if text mapping was lost
        class_names = [f"Class_{i}" for i in range(num_classes)]
        if hasattr(model, "idx_to_class") and model.idx_to_class:
             try:
                 class_names = [model.idx_to_class[i] for i in range(num_classes)]
             except:
                 pass
        else:
             logger.warning(f"[Warning] Class names not found in quantized model. Using generic: {class_names}")

        # Create dummy hparams to satisfy the rest of the script
        hparams = {
            "num_frames": getattr(model, "window_size", 16), # Default assumption
            "frame_size": 224, # Default assumption
            "backbone": "quantized_model" 
        }
        
        return model, class_names, hparams


def probe_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps is None or fps <= 1e-3:
        return 30.0
    return float(fps)


def probe_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if total is None or total <= 0:
        return 0
    return int(total)


def _safe_release(resource) -> None:
    try:
        resource.release()
    except Exception:
        return


def export_review_video(
    video_path: Path,
    annotations_csv: Path,
    output_video_path: Path,
    *,
    frame_meta: list[dict] | None,
    draw_bbox: bool,
    font_scale: float,
    thickness: int,
    label_bg_alpha: float,
) -> None:
    if not annotations_csv.exists():
        raise FileNotFoundError(f"Annotations CSV not found: {annotations_csv}")

    segments = review_annotations.load_segments(annotations_csv)
    if segments:
        segments.sort(key=lambda seg: seg.start_frame)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        _safe_release(cap)
        raise RuntimeError(f"Could not open source video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        _safe_release(cap)
        _safe_release(writer)
        raise RuntimeError(f"Could not create output video file: {output_video_path}")

    frame_idx = 0
    current_seg_idx = 0
    meta_idx = 0

    def _pack_keypoints(
        xy: list[tuple[float, float]] | None,
        conf: list[float] | None,
    ) -> list[tuple[float, float, float | None]] | None:
        if not xy:
            return None
        out: list[tuple[float, float, float | None]] = []
        for k, (x, y) in enumerate(xy):
            c_val: float | None = None
            if conf is not None and k < len(conf):
                try:
                    c_val = float(conf[k])
                except Exception:
                    c_val = None
            out.append((float(x), float(y), c_val))
        return out

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            while (
                current_seg_idx < len(segments)
                and frame_idx > segments[current_seg_idx].end_frame
            ):
                current_seg_idx += 1
            active_segment = None
            if current_seg_idx < len(segments):
                seg = segments[current_seg_idx]
                if seg.start_frame <= frame_idx <= seg.end_frame:
                    active_segment = seg

            label = "Unknown"
            conf = None
            if active_segment is not None:
                label = str(active_segment.label)
                try:
                    conf = float(active_segment.confidence)
                except Exception:
                    conf = None

            meta = None
            if frame_meta is not None:
                while meta_idx < len(frame_meta) and int(
                    frame_meta[meta_idx].get("frame_idx", -1)
                ) < frame_idx:
                    meta_idx += 1
                if meta_idx < len(frame_meta) and int(
                    frame_meta[meta_idx].get("frame_idx", -1)
                ) == frame_idx:
                    meta = frame_meta[meta_idx]

            yolo_status = None
            bbox_xyxy = None
            keypoints = None
            if meta is not None:
                yolo_status = meta.get("yolo_status")
                if draw_bbox:
                    bbox_xyxy = meta.get("bbox_xyxy")
                    keypoints = _pack_keypoints(
                        meta.get("keypoints_xy"),
                        meta.get("keypoints_conf"),
                    )

            annotated = review_annotations.annotate_frame_bbox(
                frame,
                bbox_xyxy if draw_bbox else None,
                label,
                conf,
                yolo_status,
                float(font_scale),
                int(thickness),
                keypoints=keypoints if draw_bbox else None,
                draw_keypoints=bool(draw_bbox),
                label_bg_alpha=float(label_bg_alpha),
            )
            writer.write(annotated)
            frame_idx += 1
    finally:
        _safe_release(cap)
        _safe_release(writer)


def predict_windows(
    model: BehaviorSequenceClassifier,
    class_names: List[str],
    video_path: Path,
    window_size: int,
    stride: int,
    batch_size: int,
    prob_threshold: float,
    device: torch.device,
    yolo_model,
    crop_size: int,
    pad_ratio: float,
    yolo_conf: float,
    yolo_iou: float,
    yolo_imgsz: int,
    keep_last_box: bool,
    skip_tail: bool,
    yolo_max_det: int = 1,
    yolo_nms: bool = True,
    top_k: int = 3,
    frame_meta: list[dict] | None = None,
    window_meta: list[dict] | None = None,
    window_probs: list[np.ndarray] | None = None,
    timing: dict[str, float] | None = None,
    yolo_mode: str = "auto",
    require_pose: bool = False,
) -> List[WindowPrediction]:
    transform = build_frame_transform(crop_size, augment=False)
    fps = probe_fps(video_path)
    batch_frames: List[torch.Tensor] = []
    batch_meta: List[Dict] = []
    buffer: deque[CropDetection] = deque()
    window_idx = 0
    want_window_details = window_meta is not None or window_probs is not None
    top_k = quantification.clamp_top_k(int(top_k), len(class_names))
    yolo_mode = str(yolo_mode or "auto").strip().lower()
    if yolo_mode not in {"auto", "bbox", "pose"}:
        raise ValueError(f"Unknown yolo_mode: {yolo_mode!r}")

    model_pose_dim = int(getattr(model, "pose_input_dim", 0) or 0)
    model_uses_pose = bool(getattr(model, "pose_fusion", None) is not None) and model_pose_dim > 0
    require_pose = bool(require_pose)
    if require_pose and model_uses_pose and yolo_mode == "bbox":
        raise ValueError(
            "require_pose=True but yolo_mode='bbox' disables keypoints/pose. "
            "Use --yolo_mode auto/pose and pose-capable YOLO weights."
        )

    def flush_batch():
        if not batch_frames:
            return []
        batch_len = len(batch_frames)
        if timing is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        to_device_started = time.perf_counter()
        
        # batch_frames is list of (clip, pose) tuples
        clips_list = [x[0] for x in batch_frames]
        poses_list = [x[1] for x in batch_frames]
        
        clips = torch.stack(clips_list, dim=0).to(device)
        pose_present = torch.tensor([p is not None for p in poses_list], dtype=torch.bool, device=device)
        poses = None
        if pose_present.any().item():
            valid_sample = next(p for p in poses_list if p is not None)
            padded = [p if p is not None else torch.zeros_like(valid_sample) for p in poses_list]
            poses = torch.stack(padded, dim=0).to(device)
        if timing is not None:
            timing["windows_with_pose"] = timing.get("windows_with_pose", 0.0) + float(
                pose_present.sum().item()
            )
        
        if timing is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        if timing is not None:
            timing["batch_to_device_ms"] = timing.get("batch_to_device_ms", 0.0) + (
                (time.perf_counter() - to_device_started) * 1000.0
            )
        with torch.no_grad():
            if timing is not None and device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_started = time.perf_counter()
            
            if poses is None and not pose_present.any().item():
                inputs = clips
            else:
                inputs = (clips, poses, pose_present)
            logits = model(inputs)
            
            probs = F.softmax(logits, dim=1)
            if timing is not None and device.type == "cuda":
                torch.cuda.synchronize(device)
            if timing is not None:
                timing["model_forward_ms"] = timing.get("model_forward_ms", 0.0) + (
                    (time.perf_counter() - forward_started) * 1000.0
                )
        top_p, top_idx = probs.max(dim=1)
        topk_idx: list[list[int]] | None = None
        topk_probs: list[list[float]] | None = None
        probs_full: np.ndarray | None = None
        if want_window_details:
            topk_vals_t, topk_idx_t = torch.topk(probs, k=int(top_k), dim=1)
            topk_idx = topk_idx_t.detach().cpu().tolist()
            topk_probs = topk_vals_t.detach().cpu().tolist()
            if window_probs is not None:
                probs_full = probs.detach().cpu().numpy().astype(
                    np.float32, copy=False
                )

        results = []
        nonlocal window_idx
        for row_i, (meta, prob, idx) in enumerate(
            zip(batch_meta, top_p.tolist(), top_idx.tolist())
        ):
            if prob < prob_threshold:
                class_idx = None
                class_name = "Unknown"
            else:
                class_idx = idx
                class_name = class_names[int(idx)]

            if want_window_details and window_meta is not None:
                start_frame = int(meta["start_frame"])
                end_frame = int(meta["end_frame"])
                center_frame = int((start_frame + end_frame) // 2)
                window_meta.append(
                    {
                        "window_idx": int(window_idx),
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "center_frame": center_frame,
                        "start_time": float(meta["start_time"]),
                        "end_time": float(meta["end_time"]),
                        "center_time": float(center_frame / max(fps, 1e-6)),
                        "label": str(class_name),
                        "confidence": float(prob),
                        "topk_indices": list(topk_idx[row_i]) if topk_idx else [],
                        "topk_probs": list(topk_probs[row_i]) if topk_probs else [],
                    }
                )
            if want_window_details and window_probs is not None and probs_full is not None:
                window_probs.append(probs_full[row_i])

            results.append(
                WindowPrediction(
                    start_frame=meta["start_frame"],
                    end_frame=meta["end_frame"],
                    start_time=meta["start_time"],
                    end_time=meta["end_time"],
                    class_idx=class_idx,
                    class_name=class_name,
                    confidence=float(prob),
                )
            )
            window_idx += 1
        batch_frames.clear()
        batch_meta.clear()
        if timing is not None:
            timing["windows_predicted"] = timing.get("windows_predicted", 0.0) + float(
                batch_len
            )
        return results

    def enqueue_window(entries: List[CropDetection]):
        prep_started = time.perf_counter()
        processed = [transform(det.crop_rgb) for det in entries]
        clip_tensor = torch.stack(processed, dim=0)
        
        pose_tensor = None
        if model_uses_pose and yolo_mode != "bbox" and entries:
            # Only compute pose features when ALL frames in the window have pose.
            # If any frame lacks pose, fall back to RGB-only for that window.
            first_xy = entries[0].keypoints_xy
            if first_xy is not None:
                num_kpts = len(first_xy)
                pose_all = True
                for entry in entries:
                    if entry.keypoints_xy is None or len(entry.keypoints_xy) != num_kpts:
                        pose_all = False
                        break
                    if entry.crop_xyxy is None:
                        pose_all = False
                        break
                if pose_all:
                    # Build normalized keypoints array [T, K, 3] in crop coordinates.
                    # (Normalization is invariant to the crop resize-to-square step.)
                    kpts_arr = np.zeros((len(entries), num_kpts, 3), dtype=np.float32)
                    for i, entry in enumerate(entries):
                        xy = entry.keypoints_xy
                        conf = entry.keypoints_conf
                        crop_xyxy = entry.crop_xyxy
                        assert xy is not None
                        assert crop_xyxy is not None
                        crop_x1, crop_y1, crop_x2, crop_y2 = crop_xyxy
                        crop_w = max(int(crop_x2) - int(crop_x1), 1)
                        crop_h = max(int(crop_y2) - int(crop_y1), 1)
                        for k in range(num_kpts):
                            x_raw, y_raw = xy[k]
                            kpts_arr[i, k, 0] = (float(x_raw) - float(crop_x1)) / float(crop_w)
                            kpts_arr[i, k, 1] = (float(y_raw) - float(crop_y1)) / float(crop_h)
                            c = 1.0
                            if conf is not None and k < len(conf):
                                try:
                                    c = float(conf[k])
                                except Exception:
                                    c = 1.0
                            kpts_arr[i, k, 2] = c
                    feats = extract_rich_pose_features(kpts_arr)
                    pose_tensor = torch.from_numpy(feats).float()
                    if pose_tensor.ndim != 2:
                        raise ValueError(f"Pose feature tensor has unexpected shape: {pose_tensor.shape}")
                    if pose_tensor.shape[1] != model_pose_dim:
                        raise ValueError(
                            "Pose feature dim mismatch: "
                            f"model expects {model_pose_dim}, but extractor returned {pose_tensor.shape[1]}. "
                            "This usually means the YOLO keypoint skeleton differs between training and inference."
                        )
        if require_pose and model_uses_pose and pose_tensor is None:
            raise RuntimeError(
                "Pose was required for inference, but pose keypoints/features were missing for at least one window. "
                "Confirm you're using YOLO-pose weights and set --yolo_mode pose (or auto)."
            )

        if timing is not None:
            timing["clip_prep_ms"] = timing.get("clip_prep_ms", 0.0) + (
                (time.perf_counter() - prep_started) * 1000.0
            )
        start_frame = entries[0].frame_idx
        end_frame = entries[-1].frame_idx
        batch_frames.append((clip_tensor, pose_tensor))
        batch_meta.append(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time": start_frame / fps,
                "end_time": end_frame / fps,
            }
        )
        if len(batch_frames) >= batch_size:
            for item in flush_batch():
                yield item

    stride = max(1, stride)
    if device.type == "cuda":
        yolo_device = (
            f"cuda:{device.index}" if device.index is not None else "cuda"
        )
    else:
        yolo_device = device.type

    crop_generator = iterate_yolo_crops(
        model=yolo_model,
        video_path=video_path,
        crop_size=crop_size,
        pad_ratio=pad_ratio,
        conf_threshold=yolo_conf,
        iou_threshold=yolo_iou,
        imgsz=yolo_imgsz,
        device=yolo_device,
        keep_last_box=keep_last_box,
        max_det=yolo_max_det,
        nms=yolo_nms,
        yield_misses=frame_meta is not None,
        timing=timing,
    )

    for detection in crop_generator:
        if frame_meta is not None:
            det_kpts_xy = getattr(detection, "keypoints_xy", None)
            det_kpts_conf = getattr(detection, "keypoints_conf", None)
            if yolo_mode == "bbox":
                det_kpts_xy = None
                det_kpts_conf = None
            frame_meta.append(
                {
                    "frame_idx": int(detection.frame_idx),
                    "bbox_xyxy": detection.bbox_xyxy,
                    "confidence": float(detection.confidence),
                    "yolo_status": str(getattr(detection, "yolo_status", "detected")),
                    "keypoints_xy": det_kpts_xy,
                    "keypoints_conf": det_kpts_conf,
                }
            )
        if detection.crop_rgb is None:
            continue
        buffer.append(detection)
        while len(buffer) >= window_size:
            entries = list(buffer)[:window_size]
            yield from enqueue_window(entries)
            for _ in range(stride):
                if buffer:
                    buffer.popleft()

    if not skip_tail and buffer:
        tail_entries = list(buffer)
        if tail_entries:
            while len(tail_entries) < window_size:
                tail_entries.append(tail_entries[-1])
            yield from enqueue_window(tail_entries[:window_size])

    yield from flush_batch()


def _export_metrics_xlsx(
    output_xlsx: Path,
    *,
    video_path: Path,
    class_names: list[str],
    fps: float,
    prob_threshold: float,
    window_size: int,
    window_stride: int,
    crop_size: int,
    pad_ratio: float,
    yolo_conf: float,
    yolo_iou: float,
    yolo_imgsz: int,
    keep_last_box: bool,
    smoothing: str,
    ema_alpha: float,
    interp_max_gap: int,
    per_frame_mode: str,
    top_k: int,
    frame_meta: list[dict],
    window_meta: list[dict],
    window_probs: list[np.ndarray] | None,
    model_path: Path,
    yolo_weights: Path,
    yolo_mode: str = "auto",
) -> None:
    total_frames = len(frame_meta)
    if total_frames <= 0:
        raise RuntimeError("No frames were processed; cannot export XLSX.")

    bboxes = [row.get("bbox_xyxy") for row in frame_meta]
    statuses = [str(row.get("yolo_status", "detected")) for row in frame_meta]
    yolo_confs = [float(row.get("confidence") or 0.0) for row in frame_meta]
    keypoints_xy = [row.get("keypoints_xy") for row in frame_meta]
    keypoints_conf = [row.get("keypoints_conf") for row in frame_meta]

    num_keypoints = 0
    for kpts in keypoints_xy:
        if kpts is None:
            continue
        try:
            num_keypoints = max(num_keypoints, int(len(kpts)))
        except Exception:
            continue
    has_keypoints = num_keypoints > 0

    fps = float(fps) if fps and fps > 1e-6 else 30.0
    top_k = quantification.clamp_top_k(int(top_k), len(class_names))

    centers_raw = quantification.compute_centers_xy(bboxes)
    centers_smooth = quantification.smooth_centers_xy(
        centers_raw,
        method=str(smoothing),
        ema_alpha=float(ema_alpha),
        interp_max_gap=int(interp_max_gap),
    )
    kinematics = quantification.compute_kinematics(centers_smooth, fps)

    kpt_centers_raw = None
    kpt_centers_smooth = None
    kpt_kinematics = None
    kpt_valid_count = None
    kpt_mean_conf = None
    if has_keypoints:
        kpt_centers_raw = np.full((total_frames, 2), np.nan, dtype=np.float32)
        kpt_valid_count = np.zeros(total_frames, dtype=np.int32)
        kpt_mean_conf = np.full(total_frames, np.nan, dtype=np.float32)
        for i in range(total_frames):
            kxy = keypoints_xy[i]
            if not kxy:
                continue
            kconf = keypoints_conf[i]
            xs: list[float] = []
            ys: list[float] = []
            ws: list[float] = []
            confs: list[float] = []
            for j, pt in enumerate(kxy):
                try:
                    x = float(pt[0])
                    y = float(pt[1])
                except Exception:
                    continue
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                w = 1.0
                if kconf is not None and j < len(kconf):
                    try:
                        c = float(kconf[j])
                        if np.isfinite(c):
                            w = max(c, 0.0)
                            confs.append(c)
                    except Exception:
                        pass
                xs.append(x)
                ys.append(y)
                ws.append(w)
            if not xs:
                continue
            wsum = float(np.sum(ws))
            if wsum > 1e-9:
                cx_k = float(np.sum(np.array(xs) * np.array(ws)) / wsum)
                cy_k = float(np.sum(np.array(ys) * np.array(ws)) / wsum)
            else:
                cx_k = float(np.mean(xs))
                cy_k = float(np.mean(ys))
            kpt_centers_raw[i, 0] = cx_k
            kpt_centers_raw[i, 1] = cy_k
            kpt_valid_count[i] = int(len(xs))
            if confs:
                kpt_mean_conf[i] = float(np.mean(confs))

        kpt_centers_smooth = quantification.smooth_centers_xy(
            kpt_centers_raw,
            method=str(smoothing),
            ema_alpha=float(ema_alpha),
            interp_max_gap=int(interp_max_gap),
        )
        kpt_kinematics = quantification.compute_kinematics(kpt_centers_smooth, fps)

    num_windows = len(window_meta)
    starts = np.array(
        [int(w["start_frame"]) for w in window_meta], dtype=np.int32
    )
    ends = np.array([int(w["end_frame"]) for w in window_meta], dtype=np.int32)
    centers_w = np.array(
        [int(w["center_frame"]) for w in window_meta], dtype=np.int32
    )

    frame_topk_idx = np.full((total_frames, top_k), -1, dtype=np.int32)
    frame_topk_probs = np.full((total_frames, top_k), np.nan, dtype=np.float32)

    cover_count = np.zeros(total_frames, dtype=np.int32)
    per_frame_mode = str(per_frame_mode or "none").strip().lower()
    if per_frame_mode == "true":
        if num_windows <= 0:
            cover_count = np.zeros(total_frames, dtype=np.int32)
        elif window_probs is None or len(window_probs) != num_windows:
            raise RuntimeError(
                "Per-frame mode 'true' requires collecting full window probabilities."
            )
        else:
            probs_w = np.stack(window_probs, axis=0).astype(np.float32, copy=False)
            probs_pf, cover_count = quantification.aggregate_window_probs_average(
                total_frames, starts, ends, probs_w
            )
            idx, vals = quantification.topk_from_probs(probs_pf, top_k)
            frame_topk_idx = idx
            frame_topk_probs = vals
    elif per_frame_mode == "simple":
        if num_windows > 0:
            cover_count = quantification.windows_cover_count(total_frames, starts, ends)
            assignment = quantification.assign_windows_nearest_center(
                total_frames, centers_w
            )
            assignment[cover_count == 0] = -1

            win_topk_idx = np.full((num_windows, top_k), -1, dtype=np.int32)
            win_topk_probs = np.full((num_windows, top_k), np.nan, dtype=np.float32)
            for i, win in enumerate(window_meta):
                idxs = list(win.get("topk_indices") or [])[:top_k]
                vals = list(win.get("topk_probs") or [])[:top_k]
                for j in range(min(len(idxs), top_k)):
                    win_topk_idx[i, j] = int(idxs[j])
                    win_topk_probs[i, j] = float(vals[j])

            valid = assignment >= 0
            frame_topk_idx[valid] = win_topk_idx[assignment[valid]]
            frame_topk_probs[valid] = win_topk_probs[assignment[valid]]
        else:
            cover_count = np.zeros(total_frames, dtype=np.int32)
    elif per_frame_mode == "none":
        cover_count = np.zeros(total_frames, dtype=np.int32)
    else:
        raise ValueError(f"Unknown per_frame_mode: {per_frame_mode}")

    statuses_arr = np.array(statuses, dtype=object)
    unknown_mask = statuses_arr == "missed"
    if per_frame_mode in {"simple", "true"}:
        unknown_mask = unknown_mask | (cover_count <= 0)

    top1 = frame_topk_probs[:, 0]
    unknown_mask = unknown_mask | (~np.isfinite(top1)) | (
        top1 < float(prob_threshold)
    )

    frame_topk_idx[unknown_mask] = -1
    frame_topk_probs[unknown_mask] = np.nan

    frame_label = np.array(["Unknown"] * total_frames, dtype=object)
    frame_conf = np.full(total_frames, np.nan, dtype=np.float32)
    valid = frame_topk_idx[:, 0] >= 0
    if np.any(valid):
        frame_label[valid] = np.array(
            [class_names[int(i)] for i in frame_topk_idx[valid, 0].tolist()],
            dtype=object,
        )
        frame_conf[valid] = frame_topk_probs[valid, 0]

    bouts = quantification.bouts_from_frame_labels(
        frame_label.tolist(),
        fps,
        confidences=frame_conf.tolist(),
        cum_dist_px=kinematics["cum_dist_px"].tolist(),
    )
    bouts_kpt = None
    if has_keypoints and kpt_kinematics is not None:
        bouts_kpt = quantification.bouts_from_frame_labels(
            frame_label.tolist(),
            fps,
            confidences=frame_conf.tolist(),
            cum_dist_px=kpt_kinematics["cum_dist_px"].tolist(),
        )

    def _iter_per_frame_rows():
        for i in range(total_frames):
            bbox = bboxes[i]
            if bbox is None:
                x1 = y1 = x2 = y2 = None
            else:
                x1, y1, x2, y2 = bbox

            cx_raw = (
                float(centers_raw[i, 0]) if np.isfinite(centers_raw[i, 0]) else None
            )
            cy_raw = (
                float(centers_raw[i, 1]) if np.isfinite(centers_raw[i, 1]) else None
            )
            cx = (
                float(centers_smooth[i, 0])
                if np.isfinite(centers_smooth[i, 0])
                else None
            )
            cy = (
                float(centers_smooth[i, 1])
                if np.isfinite(centers_smooth[i, 1])
                else None
            )
            step = (
                float(kinematics["step_dist_px"][i])
                if np.isfinite(kinematics["step_dist_px"][i])
                else None
            )
            spd = (
                float(kinematics["speed_px_s"][i])
                if np.isfinite(kinematics["speed_px_s"][i])
                else None
            )
            acc = (
                float(kinematics["accel_px_s2"][i])
                if np.isfinite(kinematics["accel_px_s2"][i])
                else None
            )
            cum = float(kinematics["cum_dist_px"][i])

            label = str(frame_label[i])
            conf = float(frame_conf[i]) if np.isfinite(frame_conf[i]) else None

            row = [
                int(i),
                float(i / max(fps, 1e-6)),
                str(statuses[i]),
                float(yolo_confs[i]),
                x1,
                y1,
                x2,
                y2,
            ]
            if has_keypoints:
                kxy = keypoints_xy[i]
                kconf = keypoints_conf[i]
                for j in range(int(num_keypoints)):
                    kx = ky = kc = None
                    if kxy is not None and j < len(kxy):
                        try:
                            kx = float(kxy[j][0])
                            ky = float(kxy[j][1])
                            if not (np.isfinite(kx) and np.isfinite(ky)):
                                kx = ky = None
                        except Exception:
                            kx = ky = None
                    if kconf is not None and j < len(kconf):
                        try:
                            kc = float(kconf[j])
                            if not np.isfinite(kc):
                                kc = None
                        except Exception:
                            kc = None
                    row.extend([kx, ky, kc])

            row.extend([cx_raw, cy_raw, cx, cy, step, spd, acc, cum])

            if (
                has_keypoints
                and kpt_centers_raw is not None
                and kpt_centers_smooth is not None
                and kpt_kinematics is not None
                and kpt_valid_count is not None
                and kpt_mean_conf is not None
            ):
                k_valid = int(kpt_valid_count[i])
                k_mc = (
                    float(kpt_mean_conf[i])
                    if np.isfinite(kpt_mean_conf[i])
                    else None
                )
                kcx_raw = (
                    float(kpt_centers_raw[i, 0])
                    if np.isfinite(kpt_centers_raw[i, 0])
                    else None
                )
                kcy_raw = (
                    float(kpt_centers_raw[i, 1])
                    if np.isfinite(kpt_centers_raw[i, 1])
                    else None
                )
                kcx = (
                    float(kpt_centers_smooth[i, 0])
                    if np.isfinite(kpt_centers_smooth[i, 0])
                    else None
                )
                kcy = (
                    float(kpt_centers_smooth[i, 1])
                    if np.isfinite(kpt_centers_smooth[i, 1])
                    else None
                )
                k_step = (
                    float(kpt_kinematics["step_dist_px"][i])
                    if np.isfinite(kpt_kinematics["step_dist_px"][i])
                    else None
                )
                k_spd = (
                    float(kpt_kinematics["speed_px_s"][i])
                    if np.isfinite(kpt_kinematics["speed_px_s"][i])
                    else None
                )
                k_acc = (
                    float(kpt_kinematics["accel_px_s2"][i])
                    if np.isfinite(kpt_kinematics["accel_px_s2"][i])
                    else None
                )
                k_cum = float(kpt_kinematics["cum_dist_px"][i])
                row.extend(
                    [
                        k_valid,
                        k_mc,
                        kcx_raw,
                        kcy_raw,
                        kcx,
                        kcy,
                        k_step,
                        k_spd,
                        k_acc,
                        k_cum,
                    ]
                )

            row.extend([label, conf])
            for k_i in range(1, int(top_k)):
                idx_k = int(frame_topk_idx[i, k_i])
                if idx_k >= 0:
                    row.append(str(class_names[idx_k]))
                    prob_k = float(frame_topk_probs[i, k_i])
                    row.append(prob_k if np.isfinite(prob_k) else None)
                else:
                    row.append(None)
                    row.append(None)
            yield row

    def _iter_lstm_rows():
        for win in window_meta:
            row = [
                int(win["window_idx"]),
                int(win["start_frame"]),
                int(win["end_frame"]),
                int(win["center_frame"]),
                float(win["start_time"]),
                float(win["end_time"]),
                str(win["label"]),
                float(win["confidence"]),
            ]
            idxs = list(win.get("topk_indices") or [])
            vals = list(win.get("topk_probs") or [])
            for k_i in range(1, int(top_k)):
                if k_i < len(idxs):
                    row.append(str(class_names[int(idxs[k_i])]))
                    row.append(float(vals[k_i]))
                else:
                    row.append(None)
                    row.append(None)
            yield row

    per_frame_header = [
        "frame_idx",
        "time_s",
        "yolo_status",
        "yolo_conf",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
    ]
    if has_keypoints:
        digits = max(2, len(str(int(num_keypoints) - 1)))
        for j in range(int(num_keypoints)):
            tag = f"kpt_{j:0{digits}d}"
            per_frame_header += [f"{tag}_x", f"{tag}_y", f"{tag}_conf"]
    per_frame_header += [
        "center_x",
        "center_y",
        "center_x_smooth",
        "center_y_smooth",
        "step_dist_px",
        "speed_px_s",
        "accel_px_s2",
        "cum_dist_px",
    ]
    if has_keypoints:
        per_frame_header += [
            "kpt_valid",
            "kpt_mean_conf",
            "kpt_center_x",
            "kpt_center_y",
            "kpt_center_x_smooth",
            "kpt_center_y_smooth",
            "kpt_step_dist_px",
            "kpt_speed_px_s",
            "kpt_accel_px_s2",
            "kpt_cum_dist_px",
        ]
    per_frame_header += [
        "behavior_label",
        "behavior_conf",
    ]
    for k_i in range(2, int(top_k) + 1):
        per_frame_header += [f"top{k_i}_label", f"top{k_i}_prob"]

    lstm_header = [
        "window_idx",
        "start_frame",
        "end_frame",
        "center_frame",
        "start_time",
        "end_time",
        "label",
        "confidence",
    ]
    for k_i in range(2, int(top_k) + 1):
        lstm_header += [f"top{k_i}_label", f"top{k_i}_prob"]

    bouts_header = [
        "label",
        "start_frame",
        "end_frame",
        "start_time",
        "end_time",
        "duration_s",
        "mean_confidence",
        "distance_px",
    ]
    if has_keypoints:
        bouts_header.append("distance_px_kpt")

    def _iter_bouts_rows():
        for bout_idx, b in enumerate(bouts):
            row = [
                b.get("label"),
                b.get("start_frame"),
                b.get("end_frame"),
                b.get("start_time"),
                b.get("end_time"),
                b.get("duration_s"),
                b.get("mean_confidence"),
                b.get("distance_px"),
            ]
            if has_keypoints:
                dist_kpt = None
                if bouts_kpt is not None and bout_idx < len(bouts_kpt):
                    dist_kpt = bouts_kpt[bout_idx].get("distance_px")
                row.append(dist_kpt)
            yield row

    bouts_rows = _iter_bouts_rows()

    metadata = {
        "video_path": str(video_path),
        "fps": float(fps),
        "total_frames": int(total_frames),
        "model_path": str(model_path),
        "yolo_weights": str(yolo_weights),
        "window_size": int(window_size),
        "window_stride": int(window_stride),
        "crop_size": int(crop_size),
        "pad_ratio": float(pad_ratio),
        "prob_threshold": float(prob_threshold),
        "per_frame_mode": str(per_frame_mode),
        "top_k": int(top_k),
        "smoothing": str(smoothing),
        "ema_alpha": float(ema_alpha),
        "interp_max_gap": int(interp_max_gap),
        "yolo_conf": float(yolo_conf),
        "yolo_iou": float(yolo_iou),
        "yolo_imgsz": int(yolo_imgsz),
        "yolo_max_det": 1,
        "yolo_nms": True,
        "keep_last_box": bool(keep_last_box),
        "yolo_has_keypoints": bool(has_keypoints),
        "yolo_num_keypoints": int(num_keypoints),
        "yolo_mode": str(yolo_mode),
    }

    quantification.write_metrics_xlsx(
        output_xlsx,
        per_frame_header=per_frame_header,
        per_frame_rows=_iter_per_frame_rows(),
        lstm_header=lstm_header,
        lstm_rows=_iter_lstm_rows(),
        bouts_header=bouts_header,
        bouts_rows=bouts_rows,
        metadata=metadata,
    )


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    else:
        device = torch.device(args.device)

    log_path = args.video_path.with_suffix(".log")
    logger = setup_logger(name="Inference", log_file=log_path)

    model, class_names, hparams = load_checkpoint(args.model_path, device, logger)
    model_pose_dim = int(getattr(model, "pose_input_dim", 0) or 0)
    model_uses_pose = bool(getattr(model, "pose_fusion", None) is not None) and model_pose_dim > 0
    logger.info(
        f"[Inference] Model pose fusion: {'ENABLED' if model_uses_pose else 'disabled'} "
        f"(pose_input_dim={model_pose_dim})"
    )
    if bool(getattr(args, "require_pose", False)) and not model_uses_pose:
        logger.warning(
            "[Inference] --require_pose is set, but the loaded model does not use pose features. "
            "Continuing with RGB-only inference."
        )
    if device.type == "cpu":
        model = apply_cpu_optimizations(
            model,
            quantize=args.cpu_quantize,
            num_threads=args.cpu_threads,
            num_interop_threads=args.cpu_interop_threads,
        )
    default_num_frames = int(hparams.get("num_frames", 16))
    default_frame_size = int(hparams.get("frame_size", 160))

    window_size = args.window_size or default_num_frames
    stride = args.window_stride or max(1, window_size // 2)
    crop_size = args.crop_size or default_frame_size

    backbone_name = str(hparams.get("backbone", "")).lower()
    if backbone_name.startswith("vit") and int(crop_size) != 224:
        raise ValueError(
            "ViT backbones require 224x224 crops in this implementation. "
            "Leave --crop_size blank or set it to 224."
        )

    yolo_device = (
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    yolo_task = None
    if args.yolo_mode == "pose":
        yolo_task = "pose"
    elif args.yolo_mode == "bbox":
        yolo_task = "detect"
    yolo_model = load_yolo_model(args.yolo_weights, device=yolo_device, task=yolo_task)

    output_csv = (
        args.output_csv
        if args.output_csv
        else args.video_path.with_suffix(".behavior.csv")
    )

    total_frames = probe_frame_count(args.video_path)
    json_stream = JsonStreamWriter(args.output_json)
    csv_stream = CsvMergeWriter(output_csv, keep_unknown=args.keep_unknown)
    windows_processed = 0
    last_end_frame: int | None = None
    started = time.perf_counter()

    want_review_video = args.output_review_video is not None
    collect_frame_meta = bool(args.output_xlsx) or (
        want_review_video and bool(args.draw_bbox)
    )
    frame_meta: list[dict] | None = [] if collect_frame_meta else None
    window_meta: list[dict] | None = [] if args.output_xlsx else None
    window_probs: list[np.ndarray] | None = None
    if args.output_xlsx and str(args.per_frame_mode).strip().lower() == "true":
        window_probs = []
    timing: dict[str, float] = {}

    if args.yolo_max_det != 1:
        logger.warning(
            f"[Inference] Warning: forcing yolo_max_det=1 (got {args.yolo_max_det})."
        )
    if args.no_yolo_nms:
        logger.warning(
            "[Inference] Warning: forcing yolo_nms=True (ignored --no_yolo_nms)."
        )

    try:
        for prediction in predict_windows(
            model=model,
            class_names=class_names,
            video_path=args.video_path,
            window_size=window_size,
            stride=stride,
            batch_size=args.batch_size,
            prob_threshold=args.prob_threshold,
            device=device,
            yolo_model=yolo_model,
            crop_size=crop_size,
            pad_ratio=args.pad_ratio,
            yolo_conf=args.yolo_conf,
            yolo_iou=args.yolo_iou,
            yolo_imgsz=args.yolo_imgsz,
            keep_last_box=args.keep_last_box,
            skip_tail=args.skip_tail,
            yolo_max_det=1,
            yolo_nms=True,
            top_k=args.top_k,
            frame_meta=frame_meta,
            window_meta=window_meta,
            window_probs=window_probs,
            timing=timing,
            yolo_mode=args.yolo_mode,
            require_pose=bool(getattr(args, "require_pose", False)),
        ):
            windows_processed += 1
            last_end_frame = prediction.end_frame
            json_stream.write(prediction)
            csv_stream.add(prediction)

            if (
                args.progress_interval > 0
                and windows_processed % args.progress_interval == 0
            ):
                elapsed = time.perf_counter() - started
                if total_frames > 0:
                    percent = prediction.end_frame / max(total_frames - 1, 1) * 100.0
                    frame_info = (
                        f"{prediction.end_frame}/{total_frames} "
                        f"({percent:.1f}%)"
                    )
                else:
                    frame_info = f"{prediction.end_frame}"
                speed = ""
                if elapsed > 1e-6:
                    processed_frames = prediction.end_frame + 1
                    fps_effective = processed_frames / elapsed
                    windows_per_sec = windows_processed / elapsed
                    speed = f" speed={fps_effective:.2f}fps ({windows_per_sec:.2f} win/s)"
                logger.info(
                    f"[Inference] windows={windows_processed} frame={frame_info}{speed}"
                )
    finally:
        csv_stream.close()
        json_stream.close()

    if model_uses_pose:
        windows_with_pose = int(round(float(timing.get("windows_with_pose", 0.0))))
        if windows_processed > 0:
            pct = windows_with_pose / float(windows_processed) * 100.0
        else:
            pct = 0.0
        logger.info(
            f"[Inference] Pose features used in {windows_with_pose}/{windows_processed} windows ({pct:.1f}%)."
        )
        if windows_with_pose == 0:
            logger.warning(
                "[Inference] Pose fusion is enabled in the model, but no pose features were used. "
                "This run effectively behaved like RGB-only. Check --yolo_mode and YOLO weights."
            )

    logger.info(f"Merged annotations saved to: {output_csv}")
    if args.output_json:
        logger.info(f"Per-window predictions saved to: {args.output_json}")
    if args.output_xlsx:
        if frame_meta is None or window_meta is None:
            raise RuntimeError("Internal error: missing XLSX export collectors.")
        logger.info(f"[Inference] Exporting metrics workbook: {args.output_xlsx}")
        _export_metrics_xlsx(
            args.output_xlsx,
            video_path=args.video_path,
            class_names=class_names,
            fps=probe_fps(args.video_path),
            prob_threshold=args.prob_threshold,
            window_size=window_size,
            window_stride=stride,
            crop_size=crop_size,
            pad_ratio=args.pad_ratio,
            yolo_conf=args.yolo_conf,
            yolo_iou=args.yolo_iou,
            yolo_imgsz=args.yolo_imgsz,
            keep_last_box=args.keep_last_box,
            smoothing=args.smoothing,
            ema_alpha=args.ema_alpha,
            interp_max_gap=args.interp_max_gap,
            per_frame_mode=args.per_frame_mode,
            top_k=args.top_k,
            frame_meta=frame_meta,
            window_meta=window_meta,
            window_probs=window_probs,
            model_path=args.model_path,
            yolo_weights=args.yolo_weights,
            yolo_mode=args.yolo_mode,
        )
        logger.info(f"Metrics workbook saved to: {args.output_xlsx}")
    if args.output_review_video:
        logger.info(f"[Inference] Exporting review video: {args.output_review_video}")
        export_review_video(
            video_path=args.video_path,
            annotations_csv=output_csv,
            output_video_path=args.output_review_video,
            frame_meta=frame_meta if bool(args.draw_bbox) else None,
            draw_bbox=bool(args.draw_bbox),
            font_scale=float(args.font_scale),
            thickness=int(args.thickness),
            label_bg_alpha=float(args.label_bg_alpha),
        )
        logger.info(f"[Inference] Review video saved to: {args.output_review_video}")
    logger.info(f"Total windows processed: {windows_processed}")
    elapsed_total = time.perf_counter() - started
    if last_end_frame is not None and elapsed_total > 1e-6:
        effective_fps = (last_end_frame + 1) / elapsed_total
        windows_per_sec = windows_processed / elapsed_total
        logger.info(
            f"[Inference] runtime={elapsed_total:.2f}s speed={effective_fps:.2f}fps ({windows_per_sec:.2f} win/s)"
        )


if __name__ == "__main__":
    main()
