"""TandemYTC temporal bout splitter shared by training and inference.

Single source of truth for the U-shape splitter algorithm. Both
`train_y.py` (fitting) and `infer_y.py` (application) import from
this module so the algorithm cannot drift between calibration and
deployment.

The splitter rule (U-shape detector):
    For each predicted target-class bout [s, e]:
        Find adjacent pairs of high-confidence runs
        (frames where prob_target >= tau_high).
        Between each such pair, find the longest contiguous sub-run
        where prob_target < tau_low.
        If that sub-run is >= K frames long, mark it as the
        replacement class.
    The bout is thereby split into two (or more) bouts at every
    qualifying U-shape dip.

Why this rule:
    - Naive "punch any sub-tau region" splitters fail because they
      destroy borderline-confidence whole bouts via edge erosion.
    - The U-shape requirement (must reach tau_high on BOTH sides of
      the dip) ensures we only split where there is a clear
      confidence-recovers-confidence pattern.

Configuration schema lives inside config.json under "temporal_splitter".
The legacy "v25_splitter" key is also written for backward compat with
old checkpoints.

    {
      "version": "behaviorscope-y-temporal-splitter-v1",
      "enabled": true | false,
      "target_class": "investigation",
      "target_index": 1,
      "replacement_class": "other",
      "replacement_index": 3,
      "tau_high": 0.85,
      "tau_low": 0.55,
      "K": 5,
      "fit_metric": "val_pooled_bout_f1_50_target",
      "fit_constraint": "val_pooled_frame_f1_macro_behavior >= baseline - 0.005",
      "gt_source": "annot_files | window_label_projection | unavailable",
      "n_val_videos_used": <int>,
      "n_val_target_gt_bouts": <int>,
      "val_baseline_pooled_bout_f1_50": <float>,
      "val_decoded_pooled_bout_f1_50": <float>,
      "val_baseline_pooled_frame_f1_macro_behavior": <float>,
      "val_decoded_pooled_frame_f1_macro_behavior": <float>,
      "disabled_reason": null | <str>
    }
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

# Config / version constants
SPLITTER_VERSION = "behaviorscope-y-temporal-splitter-v1"
DEFAULT_TARGET_CLASS = "investigation"
DEFAULT_REPLACEMENT_CLASS = "other"

# Config key written into checkpoints / config.json
TEMPORAL_SPLITTER_CONFIG_KEY = "temporal_splitter"
LEGACY_V25_SPLITTER_CONFIG_KEY = "v25_splitter"   # kept for reading old checkpoints

# Sweep grid (matches the post-hoc study in scripts/r5_v2_ushape_splitter.py)
TAU_HIGH_GRID: Sequence[float] = (0.75, 0.80, 0.85)
TAU_LOW_GRID:  Sequence[float] = (0.45, 0.50, 0.55, 0.60)
K_GRID:        Sequence[int]   = (3, 5, 8)

# Constraint slack: decoded frame F1 must be >= baseline - EPSILON_FRAME_F1
EPSILON_FRAME_F1 = 0.005

# Minimum-evidence gates for splitter fit per source video.
# These are baseline thresholds for annot-based fitting; projection-based
# fitting bumps them via PROJECTION_GATE_MULTIPLIER below.
MIN_COVERED_FRAMES        = 300   # 10 s at 30 fps
MIN_TARGET_GT_BOUTS_TOTAL = 30    # across all videos used in fit
MIN_VIDEOS_FOR_FIT        = 5     # below this, splitter fit is statistically thin
# Projection-based fitting is noisier (window-label projection over-states
# bout continuity because every window is a block label). Require more
# evidence to trust a projection-fit splitter.
PROJECTION_GATE_MULTIPLIER = 1.5  # 30→45 bouts, 5→8 videos for projection


# ---------------------------------------------------------------------------
# Shared apply function
# ---------------------------------------------------------------------------
def apply_temporal_splitter(
    pred: np.ndarray,
    prob_target: np.ndarray,
    target_index: int,
    replacement_index: int,
    tau_high: float,
    tau_low: float,
    K: int,
) -> np.ndarray:
    """Apply U-shape confidence splitter to per-frame predictions.

    Args:
        pred: [T] int array of class IDs (post-decoder, post-smoothing,
            post-min-duration). Will not be modified in place.
        prob_target: [T] float array — per-frame averaged-window
            probability for the target class only.
        target_index: class index whose bouts are candidates for splitting.
        replacement_index: class index that replaces split-out frames
            (typically the background / "other" class).
        tau_high: confidence flank threshold. Both sides of a dip must
            reach prob_target >= tau_high to qualify.
        tau_low: dip threshold. Frames within a candidate dip must drop
            below prob_target < tau_low.
        K: minimum dip duration in frames; shorter dips do not split.

    Returns:
        [T] int array, same shape as `pred`, with U-shape dip frames
        relabeled as `replacement_index`. Other classes are untouched.
    """
    if pred.shape != prob_target.shape:
        raise ValueError(
            f"apply_temporal_splitter: pred shape {pred.shape} != "
            f"prob_target shape {prob_target.shape}"
        )
    if not (tau_high > tau_low):
        raise ValueError(
            f"apply_temporal_splitter: require tau_high ({tau_high}) > tau_low ({tau_low})"
        )
    if K < 1:
        raise ValueError(f"apply_temporal_splitter: K must be >= 1, got {K}")

    out = pred.copy()
    n = len(out)
    if n == 0:
        return out

    # Find target-class bouts in pred
    bouts: List[tuple] = []
    in_b, b_start = False, None
    for i in range(n):
        if out[i] == target_index:
            if not in_b:
                in_b, b_start = True, i
        else:
            if in_b:
                bouts.append((b_start, i - 1))
                in_b = False
    if in_b:
        bouts.append((b_start, n - 1))

    for s, e in bouts:
        # Build mask of high-confidence frames within [s, e]
        high_mask = prob_target[s:e + 1] >= tau_high
        # Find contiguous high-confidence runs
        high_runs: List[tuple] = []
        in_h, hs = False, None
        for j in range(len(high_mask)):
            if high_mask[j]:
                if not in_h:
                    in_h, hs = True, j
            else:
                if in_h:
                    high_runs.append((hs + s, j - 1 + s))
                    in_h = False
        if in_h:
            high_runs.append((hs + s, len(high_mask) - 1 + s))

        if len(high_runs) < 2:
            continue   # need >= 2 high runs for a U-shape

        # For each adjacent pair of high runs, examine the dip between them
        for (h1s, h1e), (h2s, h2e) in zip(high_runs[:-1], high_runs[1:]):
            dip_s = h1e + 1
            dip_e = h2s - 1
            if dip_e < dip_s:
                continue
            # Find longest contiguous sub-run where prob_target < tau_low
            # within [dip_s, dip_e].
            best_run_s, best_run_e, best_len = None, None, 0
            cur_s, cur_len = None, 0
            for j in range(dip_s, dip_e + 1):
                if prob_target[j] < tau_low:
                    if cur_s is None:
                        cur_s = j
                        cur_len = 1
                    else:
                        cur_len += 1
                else:
                    if cur_s is not None and cur_len > best_len:
                        best_run_s, best_run_e, best_len = cur_s, j - 1, cur_len
                    cur_s, cur_len = None, 0
            if cur_s is not None and cur_len > best_len:
                best_run_s, best_run_e, best_len = cur_s, dip_e, cur_len
            if best_run_s is not None and best_len >= K:
                out[best_run_s:best_run_e + 1] = replacement_index
    return out


def apply_temporal_splitter_from_config(
    pred: np.ndarray,
    prob_target: np.ndarray,
    splitter_config: dict,
) -> np.ndarray:
    """Convenience wrapper: apply splitter using parameters from a
    splitter config dict. If config is None, disabled, or malformed,
    returns `pred` unchanged."""
    if not splitter_config or not splitter_config.get("enabled"):
        return pred
    try:
        return apply_temporal_splitter(
            pred=pred,
            prob_target=prob_target,
            target_index=int(splitter_config["target_index"]),
            replacement_index=int(splitter_config["replacement_index"]),
            tau_high=float(splitter_config["tau_high"]),
            tau_low=float(splitter_config["tau_low"]),
            K=int(splitter_config["K"]),
        )
    except (KeyError, ValueError, TypeError) as exc:
        print(f"[temporal splitter] could not apply (config malformed: {exc}); "
              "leaving predictions unchanged.")
        return pred


# ---------------------------------------------------------------------------
# Bout extraction + matching utilities (shared with eval scripts)
# ---------------------------------------------------------------------------
def extract_bouts(frame_array: np.ndarray, class_id: int) -> List[tuple]:
    bouts, in_b, start = [], False, None
    for i, c in enumerate(frame_array):
        if c == class_id:
            if not in_b:
                in_b, start = True, i
        else:
            if in_b:
                bouts.append((start, i - 1))
                in_b = False
    if in_b:
        bouts.append((start, len(frame_array) - 1))
    return bouts


def tiou(pb: tuple, gb: tuple) -> float:
    ps, pe = pb
    gs, ge = gb
    inter = max(0, min(pe, ge) - max(ps, gs) + 1)
    union = (pe - ps + 1) + (ge - gs + 1) - inter
    return inter / union if union > 0 else 0.0


def bout_match_counts(pred_bouts: List[tuple], gt_bouts: List[tuple],
                      threshold: float) -> tuple:
    tp = fp = 0
    matched = set()
    for pb in pred_bouts:
        best, best_gi = 0.0, -1
        for gi, gb in enumerate(gt_bouts):
            if gi in matched:
                continue
            iv = tiou(pb, gb)
            if iv > best:
                best, best_gi = iv, gi
        if best >= threshold and best_gi >= 0:
            tp += 1
            matched.add(best_gi)
        else:
            fp += 1
    fn = len(gt_bouts) - len(matched)
    return tp, fp, fn


def _f1(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ---------------------------------------------------------------------------
# Disabled-config helper (uniform safety guard)
# ---------------------------------------------------------------------------
def make_disabled_config(
    reason: str,
    target_class: str = DEFAULT_TARGET_CLASS,
    target_index: Optional[int] = None,
    replacement_class: str = DEFAULT_REPLACEMENT_CLASS,
    replacement_index: Optional[int] = None,
    gt_source: str = "unavailable",
) -> dict:
    return {
        "version": SPLITTER_VERSION,
        "enabled": False,
        "target_class": target_class,
        "target_index": target_index,
        "replacement_class": replacement_class,
        "replacement_index": replacement_index,
        "tau_high": None,
        "tau_low": None,
        "K": None,
        "fit_metric": "val_pooled_bout_f1_50_target",
        "fit_constraint": (
            "val_pooled_frame_f1_macro_behavior >= "
            f"baseline - {EPSILON_FRAME_F1}"
        ),
        "gt_source": gt_source,
        "n_val_videos_used": 0,
        "n_val_target_gt_bouts": 0,
        "val_baseline_pooled_bout_f1_50": None,
        "val_decoded_pooled_bout_f1_50": None,
        "val_baseline_pooled_frame_f1_macro_behavior": None,
        "val_decoded_pooled_frame_f1_macro_behavior": None,
        "disabled_reason": reason,
    }


# ---------------------------------------------------------------------------
# Per-video fit input: ONE TemporalSplitterVideo per source video, holding
# the full observed frame span PLUS a boolean valid_mask. The splitter and
# bout metrics work on contiguous valid_mask == True segments — uncovered
# frames are NEVER compressed away, so frame N and frame M cannot become
# falsely adjacent just because the frames between them were not in val.
# ---------------------------------------------------------------------------
@dataclass
class TemporalSplitterVideo:
    """Container for one source video's fit-time data, FULL SPAN preserved.

    Fields:
        source_video: name of the source video (no segmentation suffix).
        pred_decoded: [T_full] int — post-decoder per-frame predictions
            over the full observed span. Frames where valid_mask == False
            were never covered by any val window and should be set to
            the decoder background_index by the caller.
        prob_target: [T_full] float — averaged-window prob for the target
            class. Frames where valid_mask == False are 0.0 (unused).
        gt_frames: [T_full] int — GT class IDs over the full span. May
            be derived from .annot files (which are always full-video)
            or from window-label projection (only filled where val
            windows touched).
        valid_mask: [T_full] bool — True iff at least one val window
            covered this frame. ALL splitter / bout / metric operations
            must restrict to valid regions and refuse to compute across
            invalid gaps.
        global_start: absolute frame index in the source video where
            position 0 of the arrays above corresponds to. Stored for
            provenance / audit only; not used in metric computation.
    """
    source_video: str
    pred_decoded: np.ndarray
    prob_target: np.ndarray
    gt_frames: np.ndarray
    valid_mask: np.ndarray
    global_start: int = 0


# Backward-compat alias so existing code using V25FitVideo still works.
V25FitVideo = TemporalSplitterVideo


def _valid_segments(valid_mask: np.ndarray) -> List[tuple]:
    """Return [(s, e), …] inclusive-index pairs for contiguous True runs."""
    segments: List[tuple] = []
    in_seg, s = False, None
    for i in range(len(valid_mask)):
        if valid_mask[i]:
            if not in_seg:
                in_seg, s = True, i
        else:
            if in_seg:
                segments.append((s, i - 1))
                in_seg = False
    if in_seg:
        segments.append((s, len(valid_mask) - 1))
    return segments


def _measure_segment(
    pred_seg: np.ndarray,
    gt_seg: np.ndarray,
    target_index: int,
    behavior_indices: Sequence[int],
) -> tuple:
    """Return (bout_tp50, bout_fp50, bout_fn50, frame_acc_dict) for ONE segment."""
    pred_bouts = extract_bouts(pred_seg, target_index)
    gt_bouts   = extract_bouts(gt_seg,   target_index)
    tp50, fp50, fn50 = bout_match_counts(pred_bouts, gt_bouts, 0.50)
    frame_acc = {}
    for cid in behavior_indices:
        frame_acc[cid] = {
            "tp": int(np.sum((pred_seg == cid) & (gt_seg == cid))),
            "fp": int(np.sum((pred_seg == cid) & (gt_seg != cid))),
            "fn": int(np.sum((pred_seg != cid) & (gt_seg == cid))),
        }
    return tp50, fp50, fn50, frame_acc


def _pooled_metrics(
    videos: Iterable[TemporalSplitterVideo],
    target_index: int,
    replacement_index: int,
    behavior_indices: Sequence[int],
    splitter_params: Optional[tuple] = None,
    min_segment_frames: int = 100,
) -> dict:
    """Compute pooled metrics across videos for the splitter fit.

    For each video, iterate contiguous valid_mask==True segments and
    accumulate per-segment TP/FP/FN. Pools across all segments and
    videos before computing F1 (NOT mean-of-per-video-F1).

    Args:
        videos: iterable of TemporalSplitterVideo (full span + valid_mask).
        target_index: class id whose bouts the splitter operates on.
        replacement_index: class id substituted into split-out frames.
        behavior_indices: class ids to include in the macro-behavior F1
            (typically attack/investigation/mount, NOT background).
        splitter_params: None for baseline (no splitter), else a tuple
            (tau_high, tau_low, K) of splitter parameters to apply
            inside each segment before measuring.
        min_segment_frames: segments shorter than this are skipped
            (too short for a meaningful bout pair to fit).
    """
    bout_tp50 = bout_fp50 = bout_fn50 = 0
    frame_acc = {cid: {"tp": 0, "fp": 0, "fn": 0} for cid in behavior_indices}
    n_segments_used = 0

    for v in videos:
        for s, e in _valid_segments(v.valid_mask):
            if (e - s + 1) < min_segment_frames:
                continue
            pred_seg = v.pred_decoded[s:e + 1]
            prob_seg = v.prob_target[s:e + 1]
            gt_seg   = v.gt_frames[s:e + 1]
            if splitter_params is not None:
                tau_h, tau_l, K = splitter_params
                pred_seg = apply_temporal_splitter(
                    pred_seg, prob_seg,
                    target_index=target_index,
                    replacement_index=replacement_index,
                    tau_high=tau_h, tau_low=tau_l, K=K,
                )
            tp50, fp50, fn50, seg_frame_acc = _measure_segment(
                pred_seg, gt_seg, target_index, behavior_indices,
            )
            bout_tp50 += tp50; bout_fp50 += fp50; bout_fn50 += fn50
            for cid in behavior_indices:
                frame_acc[cid]["tp"] += seg_frame_acc[cid]["tp"]
                frame_acc[cid]["fp"] += seg_frame_acc[cid]["fp"]
                frame_acc[cid]["fn"] += seg_frame_acc[cid]["fn"]
            n_segments_used += 1

    bout_f1_50_target = _f1(bout_tp50, bout_fp50, bout_fn50)
    per_class_f1 = {cid: _f1(d["tp"], d["fp"], d["fn"]) for cid, d in frame_acc.items()}
    macro_behavior_f1 = float(np.mean(list(per_class_f1.values()))) if per_class_f1 else 0.0
    return {
        "pooled_bout_f1_50_target": bout_f1_50_target,
        "per_class_frame_f1": per_class_f1,
        "pooled_frame_f1_macro_behavior": macro_behavior_f1,
        "n_segments_used": n_segments_used,
    }


def fit_temporal_splitter(
    videos: List[TemporalSplitterVideo],
    target_class: str,
    target_index: int,
    replacement_class: str,
    replacement_index: int,
    behavior_indices: Sequence[int],
    gt_source: str,
    epsilon_frame_f1: float = EPSILON_FRAME_F1,
    tau_high_grid: Sequence[float] = TAU_HIGH_GRID,
    tau_low_grid: Sequence[float] = TAU_LOW_GRID,
    K_grid: Sequence[int] = K_GRID,
    min_segment_frames: int = 100,
) -> dict:
    """Fit the U-shape splitter on val data with valid-mask-aware segmentation.

    Each TemporalSplitterVideo holds a full-span pred/prob/gt set PLUS a
    boolean valid_mask. Inside this function, every video is iterated as a
    sequence of contiguous valid_mask==True SEGMENTS:
        1. Splitter is applied within each segment independently
           (so it can never see falsely-adjacent frames across a gap).
        2. Bout extraction and tIoU happen within each segment
           independently (so no fake bout continuity).
        3. TP / FP / FN are pooled across all segments + all videos
           BEFORE F1 is computed.

    Returns the splitter config dict (enabled True/False with reason).
    """
    if not videos:
        return make_disabled_config(
            "no validation videos available for splitter fit",
            target_class=target_class, target_index=target_index,
            replacement_class=replacement_class, replacement_index=replacement_index,
            gt_source=gt_source,
        )

    # Stronger evidence gates when GT is projection-only (noisier).
    is_projection = (gt_source == "window_label_projection")
    mult = PROJECTION_GATE_MULTIPLIER if is_projection else 1.0
    min_videos_required = int(np.ceil(MIN_VIDEOS_FOR_FIT * mult))
    min_bouts_required  = int(np.ceil(MIN_TARGET_GT_BOUTS_TOTAL * mult))

    if len(videos) < min_videos_required:
        return make_disabled_config(
            f"only {len(videos)} val videos passed coverage gates "
            f"(need >= {min_videos_required} for gt_source={gt_source!r})",
            target_class=target_class, target_index=target_index,
            replacement_class=replacement_class, replacement_index=replacement_index,
            gt_source=gt_source,
        )

    n_target_gt_bouts = 0
    for v in videos:
        for s, e in _valid_segments(v.valid_mask):
            if (e - s + 1) < min_segment_frames:
                continue
            n_target_gt_bouts += len(extract_bouts(v.gt_frames[s:e + 1], target_index))
    if n_target_gt_bouts < min_bouts_required:
        return make_disabled_config(
            f"only {n_target_gt_bouts} target GT bouts in usable val segments "
            f"(need >= {min_bouts_required} for gt_source={gt_source!r})",
            target_class=target_class, target_index=target_index,
            replacement_class=replacement_class, replacement_index=replacement_index,
            gt_source=gt_source,
        )

    base_metrics = _pooled_metrics(
        videos, target_index, replacement_index, behavior_indices,
        splitter_params=None, min_segment_frames=min_segment_frames,
    )
    base_bout_f1  = base_metrics["pooled_bout_f1_50_target"]
    base_frame_f1 = base_metrics["pooled_frame_f1_macro_behavior"]
    n_segments_used = base_metrics["n_segments_used"]

    if n_segments_used == 0:
        return make_disabled_config(
            f"no val segments meet min_segment_frames={min_segment_frames}",
            target_class=target_class, target_index=target_index,
            replacement_class=replacement_class, replacement_index=replacement_index,
            gt_source=gt_source,
        )

    best_config = None
    best_bout_f1 = base_bout_f1
    sweep_records = []
    for tau_h, tau_l, K in itertools.product(tau_high_grid, tau_low_grid, K_grid):
        if tau_h <= tau_l:
            continue
        split_metrics = _pooled_metrics(
            videos, target_index, replacement_index, behavior_indices,
            splitter_params=(float(tau_h), float(tau_l), int(K)),
            min_segment_frames=min_segment_frames,
        )
        rec = {
            "tau_high": float(tau_h), "tau_low": float(tau_l), "K": int(K),
            "pooled_bout_f1_50_target": split_metrics["pooled_bout_f1_50_target"],
            "pooled_frame_f1_macro_behavior": split_metrics["pooled_frame_f1_macro_behavior"],
        }
        sweep_records.append(rec)
        if split_metrics["pooled_frame_f1_macro_behavior"] < base_frame_f1 - epsilon_frame_f1:
            continue
        if split_metrics["pooled_bout_f1_50_target"] > best_bout_f1:
            best_bout_f1 = split_metrics["pooled_bout_f1_50_target"]
            best_config = (tau_h, tau_l, K, split_metrics)

    if best_config is None:
        cfg = make_disabled_config(
            "no splitter config improved val bout F1 @ 0.50 while satisfying "
            f"frame F1 >= baseline - {epsilon_frame_f1}",
            target_class=target_class, target_index=target_index,
            replacement_class=replacement_class, replacement_index=replacement_index,
            gt_source=gt_source,
        )
        cfg["n_val_videos_used"] = len(videos)
        cfg["n_val_segments_used"] = int(n_segments_used)
        cfg["n_val_target_gt_bouts"] = int(n_target_gt_bouts)
        cfg["val_baseline_pooled_bout_f1_50"] = round(base_bout_f1, 6)
        cfg["val_baseline_pooled_frame_f1_macro_behavior"] = round(base_frame_f1, 6)
        cfg["sweep_records"] = sweep_records
        return cfg

    tau_h, tau_l, K, split_metrics = best_config
    return {
        "version": SPLITTER_VERSION,
        "enabled": True,
        "target_class": target_class,
        "target_index": int(target_index),
        "replacement_class": replacement_class,
        "replacement_index": int(replacement_index),
        "tau_high": float(tau_h),
        "tau_low": float(tau_l),
        "K": int(K),
        "fit_metric": "val_pooled_bout_f1_50_target",
        "fit_constraint": (
            f"val_pooled_frame_f1_macro_behavior >= baseline - {epsilon_frame_f1}"
        ),
        "gt_source": gt_source,
        "n_val_videos_used": len(videos),
        "n_val_segments_used": int(n_segments_used),
        "n_val_target_gt_bouts": int(n_target_gt_bouts),
        "min_segment_frames": int(min_segment_frames),
        "val_baseline_pooled_bout_f1_50": round(base_bout_f1, 6),
        "val_decoded_pooled_bout_f1_50": round(split_metrics["pooled_bout_f1_50_target"], 6),
        "val_baseline_pooled_frame_f1_macro_behavior": round(base_frame_f1, 6),
        "val_decoded_pooled_frame_f1_macro_behavior": round(split_metrics["pooled_frame_f1_macro_behavior"], 6),
        "disabled_reason": None,
        "sweep_records": sweep_records,
    }


# Backward-compat aliases so existing code importing from this module still works.
apply_v25_splitter = apply_temporal_splitter
apply_v25_splitter_from_config = apply_temporal_splitter_from_config
fit_v25_splitter = fit_temporal_splitter
