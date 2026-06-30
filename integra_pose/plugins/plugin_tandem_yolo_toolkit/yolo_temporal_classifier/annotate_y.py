"""
Per-frame annotation helper for TandemYTC's annotated-video exporter.

Given an `NCropDetection` (with `orig_frame_bgr` populated) and a behavior
prediction for the window the frame belongs to, this module draws:

  * a header strip across the top with:
      - the predicted behavior label and softmax confidence,
      - the frame index (and time in seconds),
      - the YOLO status flag (`ok` / `keep_last_box` / `missing_dets`),
      - the per-class probability bar.
  * per-animal bounding boxes coloured by track-stable slot ID,
  * per-animal keypoints (small filled circles, alpha-faded by confidence),
  * a small label above each bbox: `slot=k  track_id=t  pose=p`.

Designed to be called once per source frame from `infer_n.py`. The frame is
modified in place; the same frame is returned for convenience.

This module has zero coupling to torch / the model itself — it operates on
numpy arrays and `NCropDetection` only, so it is also useful for QC overlays
on raw detection streams (no model inference required).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

# Import lazily so this module can be imported without cropping_n's heavier deps.
try:
    from cropping_y import NCropDetection
except Exception:  # pragma: no cover
    from .cropping_y import NCropDetection  # type: ignore


# A small palette for the first 8 animal slots; cycles past that.
SLOT_COLORS_BGR: Tuple[Tuple[int, int, int], ...] = (
    (0, 220, 0),     # slot 0: green
    (0, 165, 255),   # slot 1: orange
    (255, 0, 255),   # slot 2: magenta
    (255, 255, 0),   # slot 3: cyan
    (0, 0, 255),     # slot 4: red
    (255, 200, 0),   # slot 5: light blue
    (180, 105, 255), # slot 6: pink
    (0, 255, 255),   # slot 7: yellow
)

# A second palette for behavior labels (used for the header strip).
LABEL_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "attack":        (0, 0, 255),       # red
    "mount":         (255, 0, 255),     # magenta
    "investigation": (0, 220, 0),       # green
    "other":         (180, 180, 180),   # grey
}
DEFAULT_LABEL_COLOR_BGR: Tuple[int, int, int] = (255, 255, 255)


def slot_color(slot: int) -> Tuple[int, int, int]:
    return SLOT_COLORS_BGR[int(slot) % len(SLOT_COLORS_BGR)]


def label_color(label: str) -> Tuple[int, int, int]:
    return LABEL_COLORS_BGR.get(label.lower(), DEFAULT_LABEL_COLOR_BGR)


# ----------------------------------------------------------------------------
# Drawing primitives
# ----------------------------------------------------------------------------

def _put_text(
    frame: np.ndarray,
    text: str,
    org: Tuple[int, int],
    *,
    font_scale: float = 0.55,
    thickness: int = 1,
    color_bgr: Tuple[int, int, int] = (255, 255, 255),
    bg_bgr: Optional[Tuple[int, int, int]] = (0, 0, 0),
    pad: int = 3,
) -> None:
    """Draw text with an opaque black backing rectangle for legibility."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = int(org[0]), int(org[1])
    if bg_bgr is not None:
        cv2.rectangle(
            frame,
            (x - pad, y - h - pad),
            (x + w + pad, y + baseline + pad),
            bg_bgr,
            thickness=-1,
        )
    cv2.putText(frame, text, (x, y), font, font_scale, color_bgr, thickness, cv2.LINE_AA)


def _draw_bbox(
    frame: np.ndarray,
    xyxy: Sequence[float],
    color_bgr: Tuple[int, int, int],
    label: str,
    thickness: int = 2,
    font_scale: float = 0.5,
) -> None:
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, thickness)
    if label:
        # Label above the bbox if there is room; otherwise inside.
        ly = y1 - 6 if y1 - 12 > 0 else y1 + 18
        _put_text(
            frame,
            label,
            (x1 + 2, ly),
            font_scale=font_scale,
            thickness=1,
            color_bgr=(255, 255, 255),
            bg_bgr=color_bgr,
            pad=2,
        )


def _draw_keypoints(
    frame: np.ndarray,
    keypoints_xy: np.ndarray,        # [K, 2]
    keypoints_conf: np.ndarray,      # [K]
    color_bgr: Tuple[int, int, int],
    *,
    radius: int = 3,
    conf_threshold: float = 0.10,
) -> None:
    if keypoints_xy is None or keypoints_xy.size == 0:
        return
    K = int(keypoints_xy.shape[0])
    for k in range(K):
        c = float(keypoints_conf[k]) if keypoints_conf is not None else 1.0
        if c < conf_threshold:
            continue
        x, y = float(keypoints_xy[k, 0]), float(keypoints_xy[k, 1])
        if not (np.isfinite(x) and np.isfinite(y)) or x <= 0 or y <= 0:
            continue
        cv2.circle(frame, (int(round(x)), int(round(y))), radius, color_bgr, -1, cv2.LINE_AA)


# ----------------------------------------------------------------------------
# Header strip
# ----------------------------------------------------------------------------

def _draw_header(
    frame: np.ndarray,
    *,
    label: Optional[str],
    confidence: Optional[float],
    frame_idx: int,
    time_s: float,
    yolo_status: str,
    class_probs: Optional[Sequence[float]] = None,
    class_names: Optional[Sequence[str]] = None,
    header_height: int = 56,
    font_scale: float = 0.6,
    thickness: int = 1,
) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_height), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, dst=frame)

    # Left column: label + confidence
    if label is not None:
        col = label_color(label)
        text_main = f"{label.upper()}"
        if confidence is not None:
            text_main += f"  ({confidence:.2f})"
        _put_text(frame, text_main, (10, 24), font_scale=font_scale + 0.05,
                  thickness=thickness + 1, color_bgr=col, bg_bgr=None)
    else:
        _put_text(frame, "(no prediction yet)", (10, 24),
                  font_scale=font_scale + 0.05, thickness=thickness + 1,
                  color_bgr=(180, 180, 180), bg_bgr=None)

    # Center: frame index + time
    text_idx = f"frame {frame_idx}  t={time_s:.2f}s"
    _put_text(frame, text_idx, (10, 46), font_scale=font_scale - 0.05,
              thickness=thickness, color_bgr=(220, 220, 220), bg_bgr=None)

    # Right: YOLO status
    status_color = (180, 180, 180)
    if yolo_status == "ok":
        status_color = (0, 220, 0)
    elif "keep_last" in (yolo_status or ""):
        status_color = (0, 165, 255)
    elif "missing" in (yolo_status or ""):
        status_color = (0, 0, 255)
    text_status = f"yolo: {yolo_status}"
    (tw, _), _ = cv2.getTextSize(text_status, cv2.FONT_HERSHEY_SIMPLEX,
                                 font_scale - 0.05, thickness)
    _put_text(frame, text_status, (max(10, w - tw - 12), 46),
              font_scale=font_scale - 0.05, thickness=thickness,
              color_bgr=status_color, bg_bgr=None)

    # Probability bar (right half of header, row 1)
    if class_probs is not None and class_names is not None and len(class_probs) > 0:
        bar_x0 = w // 2
        bar_x1 = w - 12
        bar_y = 12
        bar_h = 14
        cell_w = max(1, (bar_x1 - bar_x0) // len(class_probs))
        for i, (cname, p) in enumerate(zip(class_names, class_probs)):
            x0 = bar_x0 + i * cell_w
            x1 = x0 + cell_w - 2
            fill_w = max(0, int(round((x1 - x0) * float(p))))
            col = label_color(cname)
            cv2.rectangle(frame, (x0, bar_y),
                          (x0 + fill_w, bar_y + bar_h), col, thickness=-1)
            cv2.rectangle(frame, (x0, bar_y),
                          (x1, bar_y + bar_h), (200, 200, 200), thickness=1)
            _put_text(frame, f"{cname[:4]} {p:.2f}",
                      (x0 + 2, bar_y + bar_h + 12),
                      font_scale=0.4, thickness=1,
                      color_bgr=(220, 220, 220), bg_bgr=None)


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------

def annotate_frame_n(
    det: "NCropDetection",
    *,
    label: Optional[str],
    confidence: Optional[float],
    fps: float,
    class_probs: Optional[Sequence[float]] = None,
    class_names: Optional[Sequence[str]] = None,
    draw_bboxes: bool = True,
    draw_keypoints: bool = True,
    draw_header: bool = True,
    bbox_thickness: int = 2,
    font_scale: float = 0.5,
) -> Optional[np.ndarray]:
    """Return an annotated copy of `det.orig_frame_bgr` (or None if missing).

    Args:
        det: an NCropDetection with `orig_frame_bgr` populated. If
             `orig_frame_bgr` is None, returns None.
        label: predicted behavior label for the window this frame belongs to.
               May be None for frames that fall before the first window emit.
        confidence: softmax probability of `label`. May be None.
        fps: source frame rate (used for timestamp display).
        class_probs / class_names: full probability vector + class names for
            the per-class probability bar in the header.
        draw_bboxes / draw_keypoints / draw_header: toggle individual layers.
    """
    frame = det.orig_frame_bgr
    if frame is None:
        return None
    out = frame.copy()
    h, w = out.shape[:2]

    # Per-animal bboxes + keypoints
    if draw_bboxes or draw_keypoints:
        n_animals = int(det.bbox_xyxy_animals.shape[0])
        for slot in range(n_animals):
            if not bool(det.animal_mask[slot]):
                continue
            color = slot_color(slot)
            xyxy = det.bbox_xyxy_animals[slot]
            tid = int(det.track_ids[slot])
            pose_ok = bool(det.pose_mask[slot])
            label_text = f"#{slot} id={tid}" + ("  pose=ok" if pose_ok else "  pose=miss")
            if draw_bboxes:
                _draw_bbox(out, xyxy, color, label_text,
                           thickness=bbox_thickness, font_scale=font_scale)
            if draw_keypoints and pose_ok:
                _draw_keypoints(
                    out,
                    det.keypoints_xy_raw[slot],
                    det.keypoints_conf_raw[slot],
                    color,
                )

    # Header strip
    if draw_header:
        time_s = float(det.frame_idx) / max(float(fps), 1e-6)
        _draw_header(
            out,
            label=label,
            confidence=confidence,
            frame_idx=int(det.frame_idx),
            time_s=time_s,
            yolo_status=str(det.yolo_status),
            class_probs=class_probs,
            class_names=class_names,
        )

    return out
