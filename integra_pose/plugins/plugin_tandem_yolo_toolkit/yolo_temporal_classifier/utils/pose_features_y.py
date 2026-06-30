"""
Keypoint-agnostic relational pose features for TandemYTC.

The relational stream is fully keypoint-agnostic by default. No body-part
index is hard-coded; all direction-derived features come from weighted PCA
on each animal's keypoints. This means the same feature set works for any
pose model (top-view 7-keypoint, front-view, DeepLabCut configurations of
any K, SLEAP outputs, etc.) without configuration.

Design notes:
- PCA on a 2-D keypoint cloud returns a body-axis direction whose sign is
  ambiguous (no inherent "front" vs "rear"). All axis-derived features are
  computed sign-invariantly so that ambiguity does not leak into the model:
    * axis-vs-direction features use |cos(theta)| (head-on and tail-on are
      indistinguishable without head/tail labels — that is honest)
    * axis-vs-axis features use |sin(angle_delta)| (parallel and
      anti-parallel are equivalent)
- The "nose-to-anogenital" semantic in the original spec is replaced by the
  min keypoint-pair distance, which captures the same "how close are their
  nearest body parts" signal without assuming index conventions.
- Approach velocity uses centroid velocity projected onto the inter-animal
  direction (positive = approaching, negative = retreating), not nose
  velocity.

Feature layout (per ordered pair (i, j), fixed dim = 11):

  Always-on (bbox-derived; no pose required):
    0  centroid distance (body-length normalized)
    1  bbox IoU
    2  bbox overlap area / min(area_i, area_j), capped at 5
    3  relative bbox aspect ratio (aspect_i / aspect_j)
    4  relative centroid x-velocity (body-lengths/sec)
    5  relative centroid y-velocity (body-lengths/sec)

  Pose-conditional (require pose_valid_i AND pose_valid_j; zeroed otherwise):
    6  min keypoint-pair distance (body-length normalized)
    7  PCA axis alignment with i->j direction: |cos(theta)|, range [0, 1]
    8  PCA body-axis delta: |sin(angle_delta)|, range [0, 1]
    9  approach velocity: signed projection of centroid_i velocity onto
       unit i->j direction (body-lengths/sec, positive = approaching)
   10  contact estimate: 1.0 if min keypoint-pair distance < 0.35 body
       lengths, else 0.0
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

try:  # pragma: no cover - package/script dual use
    from utils.pose_features import extract_rich_pose_features
except Exception:  # pragma: no cover
    from .pose_features import extract_rich_pose_features


# ---------------------------------------------------------------------------
# bbox helpers (unchanged from prior version)
# ---------------------------------------------------------------------------

def bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 1e-9 else 0.0


def bbox_overlap_min_area_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = max(min(area_a, area_b), 1e-9)
    return float(min(inter / denom, 5.0))


def bbox_center_xyxy(bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)


def bbox_aspect_xyxy(bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    h = max(y2 - y1, 1e-6)
    return float(max(x2 - x1, 1e-6) / h)


# ---------------------------------------------------------------------------
# Keypoint-agnostic body-axis estimation via weighted PCA
# ---------------------------------------------------------------------------

def _weighted_pca_axis(
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray,
    pose_conf_threshold: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Estimate the principal body axis of an animal from its keypoints via
    confidence-weighted PCA.

    Args:
      kpts_xy: [K, 2] keypoint positions in pixel coordinates.
      kpts_conf: [K] per-keypoint confidence.
      pose_conf_threshold: keypoints below this confidence are ignored.

    Returns:
      (axis_unit_vector, weighted_centroid)
      axis_unit_vector is a [2] unit vector along the principal body axis
      (sign is arbitrary — PCA axes are undirected).
      weighted_centroid is a [2] vector of the confidence-weighted mean
      keypoint position.

      Returns (None, None) if too few confident keypoints are available.
    """
    if kpts_xy.ndim != 2 or kpts_xy.shape[1] < 2:
        return None, None
    K = kpts_xy.shape[0]
    if K < 2 or kpts_conf.shape[0] != K:
        return None, None

    valid = kpts_conf >= float(pose_conf_threshold)
    if int(valid.sum()) < 2:
        return None, None

    pts = kpts_xy[valid, :2].astype(np.float64)
    w = kpts_conf[valid].astype(np.float64)
    w_sum = float(w.sum())
    if w_sum <= 1e-9:
        return None, None

    centroid = (pts * w[:, None]).sum(axis=0) / w_sum
    centered = pts - centroid
    cov = (centered * w[:, None]).T @ centered / w_sum
    try:
        eig_vals, eig_vecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None, None

    # Largest-eigenvalue eigenvector = principal axis (sign arbitrary)
    axis = eig_vecs[:, -1]
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-9:
        return None, None
    axis = axis / axis_norm
    return axis.astype(np.float32), centroid.astype(np.float32)


# ---------------------------------------------------------------------------
# Relational features (per-pair)
# ---------------------------------------------------------------------------

REL_FEATURE_DIM = 11
REL_BBOX_DIM = 6      # indices 0..5
REL_POSE_DIM = 5      # indices 6..10
CONTACT_THRESHOLD_BODY_LENGTHS = 0.35


def extract_relational_features(
    animal_keypoints_raw: np.ndarray,
    animal_bboxes: np.ndarray,
    animal_present: np.ndarray,
    pose_valid: np.ndarray,
    body_length_px: float,
    prev_keypoints_raw: np.ndarray | None = None,
    prev_bboxes: np.ndarray | None = None,
    fps: float = 30.0,
    pose_conf_threshold: float = 0.2,
    keypoints_conf_raw: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build fixed-width pairwise relation features. Keypoint-agnostic.

    Args:
      animal_keypoints_raw: [N, K, 2] keypoint positions in pixel coordinates,
        or [N, K, >=2] (only first two components are used as x, y).
      animal_bboxes: [N, 4] bbox xyxy in pixel coordinates.
      animal_present: [N] bool, hard mask (animal detected at this frame).
      pose_valid: [N] bool, soft gate (animal's pose is reliable).
      body_length_px: scalar, body length in pixels for normalization.
      prev_keypoints_raw, prev_bboxes: previous-frame state for velocity
        computation (None for first frame).
      fps: frame rate, for converting per-frame deltas to per-second velocities.
      pose_conf_threshold: confidence floor for including a keypoint in PCA.
      keypoints_conf_raw: optional [N, K] per-keypoint confidence; if not
        provided, all kept keypoints are weighted equally (1.0).

    Returns:
      relation_features: [N, N, 11] float32
      relation_pose_mask: [N, N] bool (pose half active for this pair)
      relation_present: [N, N] bool (both animals present)
    """
    kpts = np.asarray(animal_keypoints_raw, dtype=np.float32)
    bboxes = np.asarray(animal_bboxes, dtype=np.float32)
    present = np.asarray(animal_present, dtype=bool)
    pose_ok = np.asarray(pose_valid, dtype=bool)
    n = int(bboxes.shape[0])
    denom = max(float(body_length_px), 1.0)
    fps = max(float(fps), 1e-6)

    rel = np.zeros((n, n, REL_FEATURE_DIM), dtype=np.float32)
    rel_pose_mask = np.zeros((n, n), dtype=bool)
    rel_present = np.zeros((n, n), dtype=bool)

    if n == 0:
        return rel, rel_pose_mask, rel_present

    # Per-animal centroids and aspects (always available)
    centers = np.stack([bbox_center_xyxy(bb) for bb in bboxes], axis=0)
    aspects = np.array([bbox_aspect_xyxy(bb) for bb in bboxes], dtype=np.float32)

    # Previous-frame centroids for velocity
    prev_centers = None
    if prev_bboxes is not None:
        pb = np.asarray(prev_bboxes, dtype=np.float32)
        if pb.shape[0] == n:
            prev_centers = np.stack([bbox_center_xyxy(bb) for bb in pb], axis=0)

    # Per-animal PCA axis (computed once per animal where pose is valid)
    pca_axes: list[np.ndarray | None] = [None] * n
    if kpts.ndim == 3 and kpts.shape[1] > 0:
        K = kpts.shape[1]
        if keypoints_conf_raw is None:
            kconf_full = np.ones((n, K), dtype=np.float32)
        else:
            kconf_full = np.asarray(keypoints_conf_raw, dtype=np.float32)
            if kconf_full.shape != (n, K):
                kconf_full = np.ones((n, K), dtype=np.float32)
        for i in range(n):
            if not pose_ok[i]:
                continue
            axis, _ = _weighted_pca_axis(kpts[i, :, :2], kconf_full[i], pose_conf_threshold)
            pca_axes[i] = axis  # may still be None if too few confident keypoints

    for i in range(n):
        for j in range(n):
            if i == j or not (present[i] and present[j]):
                continue
            rel_present[i, j] = True

            # ---- Always-on bbox features (indices 0..5) ----
            ci = centers[i]; cj = centers[j]
            delta = cj - ci
            dist_ij = float(np.linalg.norm(delta))
            rel[i, j, 0] = float(dist_ij / denom)
            rel[i, j, 1] = bbox_iou_xyxy(bboxes[i], bboxes[j])
            rel[i, j, 2] = bbox_overlap_min_area_xyxy(bboxes[i], bboxes[j])
            rel[i, j, 3] = float(aspects[i] / max(aspects[j], 1e-6))
            if prev_centers is not None:
                vi = (ci - prev_centers[i]) * fps / denom
                vj = (cj - prev_centers[j]) * fps / denom
                rel[i, j, 4] = float((vj[0] - vi[0]))
                rel[i, j, 5] = float((vj[1] - vi[1]))

            # ---- Pose-conditional features (indices 6..10) ----
            if not (pose_ok[i] and pose_ok[j]):
                continue
            if kpts.ndim != 3 or kpts.shape[1] == 0:
                continue
            ki = kpts[i]; kj = kpts[j]
            if ki.shape[0] == 0 or kj.shape[0] == 0:
                continue
            rel_pose_mask[i, j] = True

            # Feature 6: min keypoint-pair distance / body_length
            d = ki[:, None, :2].astype(np.float64) - kj[None, :, :2].astype(np.float64)
            pair_dists = np.sqrt(np.sum(d * d, axis=-1))
            min_pair_dist = float(pair_dists.min())
            rel[i, j, 6] = float(min_pair_dist / denom)

            # Features 7, 8: PCA-derived orientation features (sign-invariant)
            axis_i = pca_axes[i]
            axis_j = pca_axes[j]
            if axis_i is not None:
                if dist_ij > 1e-6:
                    direction_to_j = delta / dist_ij  # unit vector i->j
                    cos_theta = float(np.dot(axis_i, direction_to_j))
                    rel[i, j, 7] = float(abs(cos_theta))  # |cos|, range [0, 1]
            if axis_i is not None and axis_j is not None:
                # |sin| of angle between two undirected axes:
                # cross-product magnitude in 2D = a.x*b.y - a.y*b.x
                cross = float(axis_i[0] * axis_j[1] - axis_i[1] * axis_j[0])
                rel[i, j, 8] = float(abs(cross))  # |sin|, range [0, 1]

            # Feature 9: approach velocity (signed projection)
            if prev_centers is not None and dist_ij > 1e-6:
                direction_to_j = delta / dist_ij
                vi = (ci - prev_centers[i]) * fps / denom  # body-lengths/sec
                rel[i, j, 9] = float(np.dot(vi, direction_to_j))

            # Feature 10: contact estimate
            rel[i, j, 10] = 1.0 if (min_pair_dist / denom) < CONTACT_THRESHOLD_BODY_LENGTHS else 0.0

    return rel, rel_pose_mask, rel_present


# ---------------------------------------------------------------------------
# Per-animal pose features (delegate to the existing K-agnostic extractor)
# ---------------------------------------------------------------------------

def extract_per_animal_pose_features(keypoints: np.ndarray) -> np.ndarray:
    """
    Apply the existing rich pose feature extractor per animal.

    Args:
      keypoints: [T, N, K, 3] keypoints in crop-relative normalized coords.

    Returns:
      [T, N, D] per-animal pose features, where D depends on K
      (see utils.pose_features.extract_rich_pose_features).
    """
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"Expected [T, N, K, 3] keypoints, got {arr.shape}.")
    t, n, _, _ = arr.shape
    feats = []
    for i in range(n):
        feats.append(extract_rich_pose_features(arr[:, i, :, :]))
    return np.stack(feats, axis=1).astype(np.float32)  # [T, N, D]
