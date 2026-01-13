from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import cv2
import torch
from torch.nn import functional as F

if __package__ is None or __package__ == "":  # pragma: no cover - CLI entry
    from cropping import (
        CropDetection,
        iterate_yolo_crops,
        load_yolo_model,
    )
    import quantification
    from data import build_frame_transform
    from model import BehaviorSequenceClassifier
else:
    from .cropping import CropDetection, iterate_yolo_crops, load_yolo_model
    from . import quantification
    from .data import build_frame_transform
    from .model import BehaviorSequenceClassifier


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
    args = parser.parse_args()
    if args.yolo_max_det <= 0:
        raise SystemExit("--yolo_max_det must be >= 1.")
    return args


def load_checkpoint(model_path: Path, device: torch.device):
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - torch<2.0 compatibility
        checkpoint = torch.load(model_path, map_location=device)
    hparams = checkpoint.get("hparams", {})
    idx_to_class = checkpoint.get("idx_to_class") or hparams.get("idx_to_class")
    if idx_to_class is None:
        raise ValueError("Checkpoint is missing idx_to_class metadata.")

    idx_to_class = {int(k): v for k, v in idx_to_class.items()}
    class_names = [idx_to_class[idx] for idx in sorted(idx_to_class.keys())]
    num_classes = len(class_names)

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
        temporal_attention_layers=int(
            hparams.get("temporal_attention_layers", 0)
        ),
        attention_heads=int(hparams.get("attention_heads", 4)),
        use_attention_pooling=bool(hparams.get("attention_pool", False)),
        use_feature_se=bool(hparams.get("feature_se", False)),
        slowfast_alpha=int(hparams.get("slowfast_alpha", 4)),
        slowfast_fusion_ratio=float(hparams.get("slowfast_fusion_ratio", 0.25)),
        slowfast_base_channels=int(hparams.get("slowfast_base_channels", 48)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

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

    def flush_batch():
        if not batch_frames:
            return []
        batch_len = len(batch_frames)
        if timing is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        to_device_started = time.perf_counter()
        clips = torch.stack(batch_frames, dim=0).to(device)
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
            logits = model(clips)
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
        if timing is not None:
            timing["clip_prep_ms"] = timing.get("clip_prep_ms", 0.0) + (
                (time.perf_counter() - prep_started) * 1000.0
            )
        start_frame = entries[0].frame_idx
        end_frame = entries[-1].frame_idx
        batch_frames.append(clip_tensor)
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
        if (
            yolo_mode == "pose"
            and getattr(detection, "yolo_status", None) == "detected"
            and not getattr(detection, "keypoints_xy", None)
        ):
            raise RuntimeError(
                "yolo_mode='pose' requires keypoints, but none were found. "
                "Did you pass YOLO-pose weights?"
            )
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

    model, class_names, hparams = load_checkpoint(args.model_path, device)
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
    yolo_model = load_yolo_model(args.yolo_weights, device=yolo_device)

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

    frame_meta: list[dict] | None = [] if args.output_xlsx else None
    window_meta: list[dict] | None = [] if args.output_xlsx else None
    window_probs: list[np.ndarray] | None = None
    if args.output_xlsx and str(args.per_frame_mode).strip().lower() == "true":
        window_probs = []

    if args.yolo_max_det != 1:
        print(
            f"[Inference] Warning: forcing yolo_max_det=1 (got {args.yolo_max_det}).",
            flush=True,
        )
    if args.no_yolo_nms:
        print(
            "[Inference] Warning: forcing yolo_nms=True (ignored --no_yolo_nms).",
            flush=True,
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
            yolo_mode=args.yolo_mode,
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
                print(
                    f"[Inference] windows={windows_processed} frame={frame_info}{speed}",
                    flush=True,
                )
    finally:
        csv_stream.close()
        json_stream.close()

    print(f"Merged annotations saved to: {output_csv}")
    if args.output_json:
        print(f"Per-window predictions saved to: {args.output_json}")
    if args.output_xlsx:
        if frame_meta is None or window_meta is None:
            raise RuntimeError("Internal error: missing XLSX export collectors.")
        print(f"[Inference] Exporting metrics workbook: {args.output_xlsx}", flush=True)
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
        print(f"Metrics workbook saved to: {args.output_xlsx}")
    print(f"Total windows processed: {windows_processed}")
    elapsed_total = time.perf_counter() - started
    if last_end_frame is not None and elapsed_total > 1e-6:
        effective_fps = (last_end_frame + 1) / elapsed_total
        windows_per_sec = windows_processed / elapsed_total
        print(
            f"[Inference] runtime={elapsed_total:.2f}s speed={effective_fps:.2f}fps ({windows_per_sec:.2f} win/s)"
        )


if __name__ == "__main__":
    main()
