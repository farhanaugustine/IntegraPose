from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Optional, Tuple

import cv2
import numpy as np

try:  # pragma: no cover - optional dependency
    from ultralytics import YOLO
    from ultralytics.utils.ops import scale_boxes
except ImportError:  # pragma: no cover - handled at call site
    YOLO = None
    scale_boxes = None


@dataclass
class CropConfig:
    crop_size: int = 160
    pad_ratio: float = 0.2
    keep_last_box: bool = True


@dataclass
class CropDetection:
    frame_idx: int
    crop_rgb: Optional[np.ndarray]
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int] | None
    crop_xyxy: Tuple[int, int, int, int] | None
    yolo_status: str  # detected | reused | missed
    keypoints_xy: Tuple[Tuple[float, float], ...] | None = None
    keypoints_conf: Tuple[float, ...] | None = None


def _check_ultralytics():
    if YOLO is None:
        raise ImportError(
            "Ultralytics YOLO is required for this operation. "
            "Install it with 'pip install ultralytics'."
        )


def load_yolo_model(weights_path: Path | str, device: str = "cpu", task: str | None = None):
    _check_ultralytics()
    w_str = str(weights_path)
    model = YOLO(w_str, task=task)
    # Only PyTorch models support .to() for device movement.
    # Exported models (ONNX, OpenVINO, TensorRT) handle device during inference.
    if w_str.endswith(".pt") or w_str.endswith(".yaml") or w_str.endswith(".pth"):
        model.to(device)
    return model


def _expand_bbox(
    bbox_xyxy: np.ndarray, pad_ratio: float, image_shape: Tuple[int, int, int]
):
    x1, y1, x2, y2 = bbox_xyxy.astype(float)
    width = x2 - x1
    height = y2 - y1
    cx = x1 + width / 2
    cy = y1 + height / 2
    scale = 1.0 + pad_ratio
    half_w = width * scale / 2
    half_h = height * scale / 2

    new_x1 = max(0.0, cx - half_w)
    new_y1 = max(0.0, cy - half_h)
    # Note: treat the max bounds as exclusive crop endpoints (safe for slicing).
    new_x2 = min(float(image_shape[1]), cx + half_w)
    new_y2 = min(float(image_shape[0]), cy + half_h)
    return np.array([new_x1, new_y1, new_x2, new_y2], dtype=np.float32)

def compute_crop_xyxy(
    frame_bgr: np.ndarray,
    bbox_xyxy: np.ndarray,
    pad_ratio: float,
) -> tuple[int, int, int, int]:
    expanded = _expand_bbox(bbox_xyxy, pad_ratio, frame_bgr.shape)
    x1, y1, x2, y2 = expanded.astype(int)
    h, w = frame_bgr.shape[:2]
    x1 = int(max(0, min(w - 1, x1)))
    y1 = int(max(0, min(h - 1, y1)))
    x2 = int(max(0, min(w, x2)))
    y2 = int(max(0, min(h, y2)))
    # Ensure a non-empty crop window; fallback to the whole frame.
    if x2 <= x1 or y2 <= y1:
        return (0, 0, int(w), int(h))
    return (x1, y1, x2, y2)


def crop_and_resize(
    frame_bgr: np.ndarray,
    bbox_xyxy: np.ndarray,
    pad_ratio: float,
    crop_size: int,
) -> np.ndarray:
    x1, y1, x2, y2 = compute_crop_xyxy(frame_bgr, bbox_xyxy, pad_ratio)
    sub_img = frame_bgr[y1:y2, x1:x2]
    if sub_img.size == 0:
        sub_img = frame_bgr
    resized = cv2.resize(sub_img, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb


def iterate_yolo_crops(
    model,
    video_path: Path | str | int,
    crop_size: int,
    pad_ratio: float,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    imgsz: int = 640,
    device: str = "cpu",
    keep_last_box: bool = True,
    max_det: int = 1,
    nms: bool = True,
    yield_misses: bool = False,
    timing: dict[str, float] | None = None,
) -> Generator[CropDetection, None, None]:
    """
    Run YOLO on a video and yield per-frame padded crops.
    """
    _check_ultralytics()
    source = video_path
    if isinstance(video_path, Path):
        source = str(video_path)
    results = model.predict(
        source=source,
        stream=True,
        device=device,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        max_det=1,  # single-animal only (locked)
        nms=True,  # single-animal only (locked)
        verbose=False,
    )

    last_bbox = None
    last_conf = 0.0
    last_keypoints_xy = None
    last_keypoints_conf = None
    for frame_idx, result in enumerate(results):
        if timing is not None:
            timing["yolo_frames"] = timing.get("yolo_frames", 0.0) + 1.0
            speed = getattr(result, "speed", None)
            if isinstance(speed, dict):
                timing["yolo_preprocess_ms"] = timing.get("yolo_preprocess_ms", 0.0) + float(
                    speed.get("preprocess", 0.0) or 0.0
                )
                timing["yolo_inference_ms"] = timing.get("yolo_inference_ms", 0.0) + float(
                    speed.get("inference", 0.0) or 0.0
                )
                timing["yolo_postprocess_ms"] = timing.get("yolo_postprocess_ms", 0.0) + float(
                    speed.get("postprocess", 0.0) or 0.0
                )
        frame_bgr = result.orig_img
        boxes = result.boxes
        best_bbox = None
        best_conf = None
        best_keypoints_xy = None
        best_keypoints_conf = None
        status = "missed"
        if boxes is not None and boxes.xyxy is not None and len(boxes.xyxy) > 0:
            confs = boxes.conf.cpu().numpy()
            idx = int(confs.argmax())
            best_bbox = boxes.xyxy[idx].cpu().numpy()
            best_conf = float(confs[idx])

            # Auto-scale normalized coordinates (best-effort).
            if (
                best_bbox.size == 4
                and float(best_bbox.min()) >= 0.0
                and float(best_bbox.max()) <= 1.0
            ):
                h_img, w_img = frame_bgr.shape[:2]
                best_bbox[0] *= float(w_img)
                best_bbox[1] *= float(h_img)
                best_bbox[2] *= float(w_img)
                best_bbox[3] *= float(h_img)

            try:
                keypoints = getattr(result, "keypoints", None)
                if keypoints is not None:
                    xy = getattr(keypoints, "xy", None)
                    if xy is not None and len(xy) > idx:
                        xy_i = xy[idx]
                        if hasattr(xy_i, "detach"):
                            xy_i = xy_i.detach()
                        if hasattr(xy_i, "cpu"):
                            xy_i = xy_i.cpu()
                        xy_np = xy_i.numpy() if hasattr(xy_i, "numpy") else np.asarray(xy_i)

                        # Best-effort: ensure keypoints are in original-frame pixel coordinates.
                        # Ultralytics commonly returns pixel coords via keypoints.xy. Some pipelines
                        # may yield normalized coords (0..1); we handle that case.
                        kpt_scale_x, kpt_scale_y = 1.0, 1.0
                        try:
                            if xy_np.size:
                                xy_max = float(np.nanmax(xy_np))
                                xy_min = float(np.nanmin(xy_np))
                                h_img, w_img = frame_bgr.shape[:2]
                                if 0.0 <= xy_min and xy_max <= 1.0:
                                    kpt_scale_x, kpt_scale_y = float(w_img), float(h_img)
                        except Exception:
                            kpt_scale_x, kpt_scale_y = 1.0, 1.0

                        if kpt_scale_x != 1.0 or kpt_scale_y != 1.0:
                            xy_np = xy_np.copy()
                            xy_np[:, 0] *= float(kpt_scale_x)
                            xy_np[:, 1] *= float(kpt_scale_y)

                        best_keypoints_xy = tuple(
                            (float(x), float(y)) for x, y in xy_np.tolist()
                        )
                    conf = getattr(keypoints, "conf", None)
                    if conf is not None and len(conf) > idx:
                        conf_i = conf[idx]
                        if hasattr(conf_i, "detach"):
                            conf_i = conf_i.detach()
                        if hasattr(conf_i, "cpu"):
                            conf_i = conf_i.cpu()
                        conf_np = (
                            conf_i.numpy()
                            if hasattr(conf_i, "numpy")
                            else np.asarray(conf_i)
                        )
                        best_keypoints_conf = tuple(float(c) for c in conf_np.tolist())
            except Exception:
                best_keypoints_xy = None
                best_keypoints_conf = None
            last_bbox = best_bbox
            last_conf = best_conf
            last_keypoints_xy = best_keypoints_xy
            last_keypoints_conf = best_keypoints_conf
            status = "detected"
        elif keep_last_box and last_bbox is not None:
            best_bbox = last_bbox
            best_conf = last_conf
            best_keypoints_xy = last_keypoints_xy
            best_keypoints_conf = last_keypoints_conf
            status = "reused"

        if best_bbox is None:
            if not yield_misses:
                continue
            yield CropDetection(
                frame_idx=frame_idx,
                crop_rgb=None,
                confidence=0.0,
                bbox_xyxy=None,
                crop_xyxy=None,
                yolo_status="missed",
                keypoints_xy=None,
                keypoints_conf=None,
            )
            continue

        crop_started = time.perf_counter()
        crop_xyxy = compute_crop_xyxy(frame_bgr, best_bbox, pad_ratio)
        crop_rgb = crop_and_resize(frame_bgr, best_bbox, pad_ratio, crop_size)
        if timing is not None:
            timing["crop_ms"] = timing.get("crop_ms", 0.0) + (
                (time.perf_counter() - crop_started) * 1000.0
            )
            timing["crop_frames"] = timing.get("crop_frames", 0.0) + 1.0
        yield CropDetection(
            frame_idx=frame_idx,
            crop_rgb=crop_rgb,
            confidence=best_conf if best_conf is not None else 0.0,
            bbox_xyxy=tuple(int(v) for v in best_bbox),
            crop_xyxy=tuple(int(v) for v in crop_xyxy),
            yolo_status=status,
            keypoints_xy=best_keypoints_xy,
            keypoints_conf=best_keypoints_conf,
        )
