"""
Inference for TandemYTC models.

The YOLO checkpoint passed via --yolo_weights runs pose detection and tracking.
TandemYTC converts those detections into pose, tracking-reliability, and
social-geometry tensors for the temporal classifier.

One script, three modes:

  1. Single-video   --source video.mp4
  2. Batch (folder) --source <directory_of_videos>
  3. Real-time      --source 0  (or any webcam/RTSP index)

The script:
  * loads the trained model from a checkpoint produced by ``train_y.py``
  * runs YOLO-pose + tracking via ``cropping_y.iterate_yolo_n_crops``
  * buffers frames into windows matching the model's ``num_frames`` /
    ``window_stride``
  * forwards each window through ``MultiAnimalBehaviorSequenceClassifier``
  * writes per-window predictions to a CSV
  * optionally writes an annotated MP4 (per-frame bboxes + slot IDs +
    keypoints + behavior label burned into the frame)

Graceful stop:
  * Pressing Ctrl+C in the terminal triggers a clean shutdown that
    flushes the CSV and writes the MP4 moov atom.
  * The GUI Stop button drops a sentinel file (``--stop_flag_file <path>``).
    The loop polls it once per detection and shuts down the same way.
  * In both cases the file handles are closed in the ``finally`` block, so
    partial outputs remain valid (CSV is well-formed, MP4 is playable).

Usage examples:

    # Single video, write CSV next to model
    python infer_y.py ^
        --model_path  best_model_macro_f1.pt ^
        --model_config config.json ^
        --yolo_weights yolo11n-pose.pt ^
        --source experiment.mp4 ^
        --output  experiment_predictions.csv

    # Batch over a folder of videos with annotated MP4s
    python infer_y.py ^
        --model_path  best_model_macro_f1.pt ^
        --yolo_weights yolo11n-pose.pt ^
        --source ./videos/ ^
        --output_dir ./predictions/ ^
        --output_video_dir ./predictions/annotated/

    # Real-time webcam
    python infer_y.py ^
        --model_path  best_model_macro_f1.pt ^
        --yolo_weights yolo11n-pose.pt ^
        --source 0 ^
        --realtime
"""
from __future__ import annotations

import warnings
# Suppress the noisy torch.cuda FutureWarning caused by environments that
# still have the deprecated `pynvml` package installed. Root fix:
# pip uninstall pynvml && pip install nvidia-ml-py.
warnings.filterwarnings(
    "ignore",
    message=".*pynvml.*deprecated.*",
    category=FutureWarning,
)

import argparse
import csv
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

YOLO_TEMPORAL_CLASSIFIER_VERSION = "2.0.0-alpha.1"
YOLO_TEMPORAL_CLASSIFIER_PIPELINE = "TandemYTC"
BEHAVIORSCOPE_Y_VERSION = YOLO_TEMPORAL_CLASSIFIER_VERSION
BEHAVIORSCOPE_Y_PIPELINE = YOLO_TEMPORAL_CLASSIFIER_PIPELINE
TEMPORAL_SPLITTER_CONFIG_KEY = "temporal_splitter"
LEGACY_V25_SPLITTER_CONFIG_KEY = "v25_splitter"


def safe_torch_load(path: Path | str, *, map_location=None):
    """Load local checkpoints without unsafe pickle execution when supported."""
    try:
        return torch.load(str(path), map_location=map_location, weights_only=True)
    except TypeError:
        # PyTorch versions before weights_only support.
        return torch.load(str(path), map_location=map_location)


try:
    from torch.amp import autocast as _autocast
except ImportError:  # torch < 2.0
    from torch.cuda.amp import autocast as _autocast  # type: ignore[no-redef]


# ----------------------------------------------------------------------------
# Graceful stop machinery
# ----------------------------------------------------------------------------
# Two ways for the user / GUI to request a clean halt:
#   1. Press Ctrl+C in the terminal (SIGINT). On Windows this also fires when
#      the GUI sends a CTRL_BREAK_EVENT to the subprocess.
#   2. Touch a sentinel file (`--stop_flag_file <path>`). The main loop polls
#      it once per detection — the GUI's Stop button drops this file so that
#      the subprocess can flush both the CSV (every row is flushed already)
#      and the VideoWriter (so the MP4 has a valid moov atom and isn't
#      truncated mid-frame).
#
# In both cases the loop breaks cleanly into the `finally` block, which
# closes the CSV file handle and calls VideoWriter.release(). This is the
# critical difference from a hard `terminate()` / SIGKILL, which leaves
# OpenCV's MP4 muxer in an invalid state and the file unplayable.
_STOP_REQUESTED = False


def _request_stop(reason: str = "signal") -> None:
    """Mark a graceful stop. Called from signal handlers and the loop's
    sentinel-file poll."""
    global _STOP_REQUESTED
    if not _STOP_REQUESTED:
        print(f"[infer] graceful stop requested ({reason}); "
              "flushing CSV and finalizing MP4 before exiting...",
              flush=True)
    _STOP_REQUESTED = True


def _install_signal_handlers() -> None:
    """Catch SIGINT / SIGTERM (and Windows SIGBREAK) and turn them into a
    graceful-stop request rather than an abrupt KeyboardInterrupt."""
    def _handler(signum, _frame):
        _request_stop(f"signal {signum}")
    for sig_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _handler)
        except Exception:
            # Some platforms (e.g. non-main thread) won't accept handlers;
            # that's fine - the loop's sentinel-file poll still works.
            pass


try:
    from cropping_y import iterate_yolo_n_crops, NCropDetection, load_yolo_model, normalize_device
    from model_y import MultiAnimalBehaviorSequenceClassifier, inspect_yolo_keypoint_count
    from utils.pose_features_y import extract_per_animal_pose_features, REL_FEATURE_DIM
    from utils.inference_metrics import (
        MetricsLogger, NullMetricsLogger, derive_metrics_csv_path,
    )
    from annotate_y import annotate_frame_n
except Exception:  # pragma: no cover
    from .cropping_y import iterate_yolo_n_crops, NCropDetection, load_yolo_model, normalize_device
    from .model_y import MultiAnimalBehaviorSequenceClassifier, inspect_yolo_keypoint_count
    from .utils.pose_features_y import extract_per_animal_pose_features, REL_FEATURE_DIM
    from .utils.inference_metrics import (
        MetricsLogger, NullMetricsLogger, derive_metrics_csv_path,
    )
    from .annotate_y import annotate_frame_n


# ----------------------------------------------------------------------------
# AMP helpers
# ----------------------------------------------------------------------------

def _resolve_amp_dtype(requested: str) -> str:
    """Return 'bf16' or 'fp16' given 'auto', 'bf16', or 'fp16'.

    Called at inference time on the inference machine — independent of
    whatever dtype the checkpoint was trained with.
    """
    if requested == "auto":
        return "bf16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "fp16"
    return requested


# ----------------------------------------------------------------------------
# Temporal smoothing helpers (evaluation-grade inference postprocessing)
# ----------------------------------------------------------------------------

def _median_filter_1d(arr: np.ndarray, window: int) -> np.ndarray:
    """Apply a 1-D median filter to an integer label array.

    Uses edge-padding so boundary frames are not left uncovered.
    Window should be odd for symmetric behaviour, but even values work too.
    Pure-numpy implementation — no scipy dependency.
    """
    if window <= 1 or len(arr) == 0:
        return arr.copy()
    half = window // 2
    padded = np.pad(arr.astype(np.int32), half, mode="edge")
    out = np.empty_like(arr, dtype=np.int32)
    for i in range(len(arr)):
        out[i] = int(np.median(padded[i: i + window]))
    return out


def _apply_min_duration(pred: np.ndarray, min_frames: int) -> np.ndarray:
    """Merge runs shorter than min_frames into the longer adjacent run.

    Iterates until no short runs remain (handles chains of short runs).
    The class of the longer neighbor wins; ties go to the preceding run.
    """
    if min_frames <= 1 or len(pred) == 0:
        return pred.copy()
    pred = pred.copy()
    n = len(pred)
    for _ in range(n):          # at most n passes to reach stability
        # Extract current runs as (class, start, end_exclusive)
        runs: list = []
        i = 0
        while i < n:
            v = int(pred[i])
            j = i + 1
            while j < n and pred[j] == v:
                j += 1
            runs.append([v, i, j])
            i = j

        changed = False
        for k, (v, s, e) in enumerate(runs):
            dur = e - s
            if dur >= min_frames:
                continue
            prev_dur = runs[k - 1][2] - runs[k - 1][1] if k > 0 else 0
            next_dur = runs[k + 1][2] - runs[k + 1][1] if k < len(runs) - 1 else 0
            if prev_dur <= 0 and next_dur <= 0:
                continue
            replace_cls = runs[k - 1][0] if prev_dur >= next_dur else runs[k + 1][0]
            pred[s:e] = replace_cls
            changed = True

        if not changed:
            break
    return pred


def _threshold_decoder_enabled(decoder_config: Optional[dict]) -> bool:
    return bool(isinstance(decoder_config, dict) and decoder_config.get("enabled"))


def apply_threshold_decoder_probs(probs: np.ndarray, decoder_config: Optional[dict]) -> np.ndarray:
    """Decode probabilities with validation-fitted thresholds when available."""
    probs = np.asarray(probs, dtype=np.float64)
    one_dim = probs.ndim == 1
    if one_dim:
        probs = probs[None, :]
    if not _threshold_decoder_enabled(decoder_config):
        preds = probs.argmax(axis=1).astype(np.int64)
        return preds[0] if one_dim else preds

    n_classes = int(probs.shape[1])
    bg_idx = int(decoder_config.get("background_index", -1))
    thresholds = np.asarray(decoder_config.get("thresholds_by_index", []), dtype=np.float64)
    if bg_idx < 0 or bg_idx >= n_classes or thresholds.shape[0] != n_classes:
        preds = probs.argmax(axis=1).astype(np.int64)
        return preds[0] if one_dim else preds

    behavior_indices = [i for i in range(n_classes) if i != bg_idx]
    preds = np.full(probs.shape[0], bg_idx, dtype=np.int64)
    for row_i, row in enumerate(probs):
        passing = [i for i in behavior_indices if float(row[i]) >= float(thresholds[i])]
        if passing:
            preds[row_i] = int(passing[int(np.argmax(row[passing]))])
    return preds[0] if one_dim else preds


def _write_smoothed_frames_csv(
    window_records: list,
    window_csv_path: "Path",
    class_names: list,
    idx_to_class: dict,
    fps: float,
    smoothing_window: int,
    min_duration: int,
    threshold_decoder: Optional[dict] = None,
    v25_splitter_config: Optional[dict] = None,
) -> "Optional[Path]":
    """Convert accumulated window records to a smoothed per-frame CSV.

    Pipeline:
      window_records → per-frame prob average (overlapping windows)
      → median filter on argmax labels
      → min-duration filter
      → write <stem>.smoothed_frames.csv

    The output uses frame_start=frame_end=frame_idx so that
    evaluate_against_annot.predictions_to_per_frame() reads each row as a
    single-frame window with no overlap averaging — it gets the smoothed
    label directly.

    Returns the path of the written CSV, or None if nothing was written.
    """
    if not window_records:
        return None

    max_frame = max(end for _, end, _ in window_records)
    n_frames = max_frame + 1
    n_classes = len(class_names)

    prob_sum = np.zeros((n_frames, n_classes), dtype=np.float64)
    coverage = np.zeros(n_frames, dtype=np.int32)
    for start, end, probs in window_records:
        s = max(0, start)
        e = min(n_frames, end + 1)   # inclusive end → exclusive
        prob_sum[s:e] += probs
        coverage[s:e] += 1

    # Per-frame argmax. Uncovered frames default to the decoder's background
    # class (typically "other"), NOT class 0 (which can be a behavior class).
    # Pre-fix: uncovered frames silently became class-0 — if class-0 is a
    # behavior class (e.g. "attack"), that inflates false positives in any
    # region the sliding window did not reach.
    bg_idx_for_init = 0
    if isinstance(threshold_decoder, dict):
        try:
            bg_idx_for_init = int(threshold_decoder.get("background_index", 0))
        except (TypeError, ValueError):
            bg_idx_for_init = 0
    pred = np.full(n_frames, bg_idx_for_init, dtype=np.int32)
    covered = coverage > 0
    if covered.any():
        avg_covered = prob_sum[covered] / coverage[covered, None]
        pred[covered] = apply_threshold_decoder_probs(
            avg_covered, threshold_decoder
        ).astype(np.int32)

    avg_probs = np.zeros((n_frames, n_classes), dtype=np.float32)
    avg_probs[covered] = (prob_sum[covered] / coverage[covered, None]).astype(np.float32)

    # Temporal smoothing on predicted class labels
    if smoothing_window > 1:
        pred = _median_filter_1d(pred, smoothing_window)

    # Minimum-duration filter
    if min_duration > 1:
        pred = _apply_min_duration(pred, min_duration)

    # TandemYTC temporal splitter (per-target-class only).
    # Applied to CSV output only. MP4 overlay path uses pre-splitter labels
    # because the streaming pipeline cannot buffer arbitrary frame ranges.
    # See `utils/temporal_splitter.py` for full algorithm.
    if v25_splitter_config and v25_splitter_config.get("enabled"):
        try:
            from utils.temporal_splitter import apply_temporal_splitter_from_config as apply_v25_splitter_from_config
        except ImportError:
            from .utils.temporal_splitter import apply_temporal_splitter_from_config as apply_v25_splitter_from_config
        try:
            target_idx = int(v25_splitter_config["target_index"])
            prob_target = avg_probs[:, target_idx]
            pred = apply_v25_splitter_from_config(
                pred, prob_target, v25_splitter_config,
            )
        except (KeyError, IndexError, ValueError, TypeError) as _exc:
            print(f"[temporal splitter] could not apply (config malformed: {_exc}); "
                  "leaving smoothed predictions unchanged.")

    out_path = Path(str(window_csv_path)).with_suffix(".smoothed_frames.csv")
    fieldnames = (
        ["frame_idx", "frame_start", "frame_end", "time_s",
         "predicted_class_id", "predicted_class"]
        + [f"prob_{c}" for c in class_names]
        + ["n_windows_covering"]
    )
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for fi in range(n_frames):
            cid = int(pred[fi])
            row = {
                "frame_idx": fi,
                "frame_start": fi,
                "frame_end": fi,
                "time_s": fi / max(fps, 1e-6),
                "predicted_class_id": cid,
                "predicted_class": idx_to_class.get(cid, str(cid)),
                "n_windows_covering": int(coverage[fi]),
            }
            for ci, cname in enumerate(class_names):
                row[f"prob_{cname}"] = float(avg_probs[fi, ci])
            writer.writerow(row)

    return out_path


# ----------------------------------------------------------------------------
# Window assembly
# ----------------------------------------------------------------------------

def detections_to_window_batch(
    dets: List[NCropDetection],
    n_animals: int,
    num_keypoints: int,
) -> dict:
    """Stack a list of T NCropDetections into the dict format the model expects.

    Output dict has all fields with leading [B=1, T, ...] axes ready for
    `model.forward(batch)`.
    """
    T = len(dets)
    if T == 0:
        raise ValueError("Empty detection window")
    num_keypoints = int(num_keypoints)
    if num_keypoints <= 0:
        raise ValueError(f"num_keypoints must be positive, got {num_keypoints}.")
    animal_keypoints = np.zeros((T, n_animals, num_keypoints, 3), dtype=np.float32)
    animal_kp_raw = np.zeros((T, n_animals, num_keypoints, 2), dtype=np.float32)
    animal_kp_conf = np.zeros((T, n_animals, num_keypoints), dtype=np.float32)
    animal_mask = np.zeros((T, n_animals), dtype=bool)
    pose_mask = np.zeros((T, n_animals), dtype=bool)
    pose_conf = np.zeros((T, n_animals), dtype=np.float32)
    crop_conf = np.zeros((T, n_animals), dtype=np.float32)
    track_conf = np.zeros((T, n_animals), dtype=np.float32)
    track_age = np.zeros((T, n_animals), dtype=np.float32)
    bbox_xyxy_animals = np.zeros((T, n_animals, 4), dtype=np.int32)
    crop_xyxy_animals = np.zeros((T, n_animals, 4), dtype=np.int32)
    track_ids = np.full((T, n_animals), -1, dtype=np.int32)
    rel_features = np.zeros((T, n_animals, n_animals, REL_FEATURE_DIM), dtype=np.float32)
    rel_pose_mask = np.zeros((T, n_animals, n_animals), dtype=bool)
    rel_present = np.zeros((T, n_animals, n_animals), dtype=bool)

    for t, d in enumerate(dets):
        if d.animal_mask is not None:
            N_d = min(int(d.animal_mask.shape[0]), n_animals)
        elif d.keypoints_crop_norm is not None and d.keypoints_crop_norm.ndim == 3:
            N_d = min(int(d.keypoints_crop_norm.shape[0]), n_animals)
        else:
            N_d = n_animals
        K_raw = d.keypoints_crop_norm.shape[1] if d.keypoints_crop_norm.ndim == 3 else 0
        if K_raw not in (0, num_keypoints):
            raise ValueError(
                f"Detection keypoint count {K_raw} does not match the trained "
                f"model's num_keypoints={num_keypoints}. Use the same YOLO "
                "pose keypoint schema for training and inference."
            )
        K_d = min(K_raw, num_keypoints)
        if K_d > 0:
            animal_keypoints[t, :N_d, :K_d] = d.keypoints_crop_norm[:N_d, :K_d]
            animal_kp_raw[t, :N_d, :K_d] = d.keypoints_xy_raw[:N_d, :K_d]
            animal_kp_conf[t, :N_d, :K_d] = d.keypoints_conf_raw[:N_d, :K_d]
        animal_mask[t, :N_d] = d.animal_mask[:N_d]
        pose_mask[t, :N_d] = d.pose_mask[:N_d]
        pose_conf[t, :N_d] = d.pose_conf[:N_d]
        crop_conf[t, :N_d] = d.crop_conf[:N_d]
        track_conf[t, :N_d] = d.track_conf[:N_d]
        track_age[t, :N_d] = d.track_age[:N_d]
        bbox_xyxy_animals[t, :N_d] = d.bbox_xyxy_animals[:N_d]
        crop_xyxy_animals[t, :N_d] = d.crop_xyxy_animals[:N_d]
        track_ids[t, :N_d] = d.track_ids[:N_d]
        if d.relation_features.shape[0] >= n_animals and d.relation_features.shape[1] >= n_animals:
            rel_features[t] = d.relation_features[:n_animals, :n_animals]
            rel_pose_mask[t] = d.relation_pose_mask[:n_animals, :n_animals]
            rel_present[t] = d.relation_present[:n_animals, :n_animals]

    try:
        pose_self_feats = extract_per_animal_pose_features(animal_keypoints)
    except Exception:
        K = num_keypoints
        D = K * 4 + K * (K - 1) // 2
        pose_self_feats = np.zeros((T, n_animals, D), dtype=np.float32)

    rel_pose_conf = (pose_conf[:, :, None] * pose_conf[:, None, :]).astype(np.float32)

    out = {
        "animal_mask": torch.from_numpy(animal_mask).unsqueeze(0),
        "pose_mask": torch.from_numpy(pose_mask).unsqueeze(0),
        "pose_conf": torch.from_numpy(pose_conf).unsqueeze(0),
        "crop_conf": torch.from_numpy(crop_conf).unsqueeze(0),
        "track_conf": torch.from_numpy(track_conf).unsqueeze(0),
        "track_age": torch.from_numpy(track_age).unsqueeze(0),
        "pose_self": torch.from_numpy(pose_self_feats).unsqueeze(0),
        "relation_features": torch.from_numpy(rel_features).unsqueeze(0),
        "relation_pose_mask": torch.from_numpy(rel_pose_mask).unsqueeze(0),
        "relation_pose_conf": torch.from_numpy(rel_pose_conf).unsqueeze(0),
        "relation_present": torch.from_numpy(rel_present).unsqueeze(0),
    }
    return out


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------

def load_model_from_checkpoint(
    checkpoint_path: Path | str,
    config_json: Path | str | None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    yolo_weights_fallback: str | None = None,
):
    """Reconstruct the trained TandemYTC model from a checkpoint.

    yolo_weights_fallback: used only if the embedded config does not record
    `yolo_weights`. Older configs (or third-party checkpoints) may lack this
    key; passing the CLI value here makes the function self-contained
    instead of relying on a free `args` reference at module scope.
    """
    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    checkpoint_cfg = None
    if isinstance(checkpoint, dict):
        raw_cfg = checkpoint.get("config") or checkpoint.get("model_config")
        if isinstance(raw_cfg, dict):
            checkpoint_cfg = raw_cfg

    if config_json is not None:
        cfg_path = Path(config_json)
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Model config not found: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
    elif checkpoint_cfg is not None:
        cfg = checkpoint_cfg
        print("[infer] using model config embedded in checkpoint", flush=True)
    else:
        raise FileNotFoundError(
            "A train_n.py config.json or a checkpoint with embedded config is "
            "required so inference can rebuild the exact multi-animal architecture."
        )

    num_classes = len(cfg.get("class_to_idx", {})) or 4
    n_animals = int(cfg.get("n_animals", 2))
    num_keypoints = int(cfg.get("num_keypoints", 7))
    rel_feature_dim = int(cfg.get("rel_feature_dim", REL_FEATURE_DIM))
    flags = cfg.get("ablation_flags", {})

    # YOLO weights path: prefer config (recorded by train_y.py); fall back to
    # the CLI --yolo_weights flag required for pose detection.
    yolo_weights_path = cfg.get("yolo_weights") or yolo_weights_fallback
    if not yolo_weights_path:
        raise ValueError(
            "No yolo_weights found in checkpoint config and no "
            "yolo_weights_fallback supplied. Pass --yolo_weights on the CLI "
            "or train with train_y.py so the path is recorded in config.json."
        )
    model = MultiAnimalBehaviorSequenceClassifier(
        num_classes=num_classes,
        n_animals=n_animals,
        num_keypoints=num_keypoints,
        rel_feature_dim=rel_feature_dim,
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        dropout=float(cfg.get("dropout", 0.5)),
        sequence_model=str(cfg.get("sequence_model", "lstm")),
        num_lstm_layers=int(cfg.get("num_lstm_layers", 1)),
        bidirectional_lstm=bool(cfg.get("bidirectional_lstm", False)),
        use_attention_pool=bool(cfg.get("use_attention_pool", False)),
        attention_heads=int(cfg.get("attention_heads", 4)),
        positional_encoding=str(cfg.get("positional_encoding", "none")),
        disable_pose_self=bool(flags.get("disable_pose_self", False)),
        disable_relations=bool(flags.get("disable_relations", False)),
        relations_pose_only=bool(flags.get("relations_pose_only", False)),
    )
    state = checkpoint
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint/config mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model.to(device)
    model.eval()
    idx_to_class = {int(v): k for k, v in cfg.get("class_to_idx", {}).items()}
    if not idx_to_class:
        idx_to_class = {i: f"class_{i}" for i in range(num_classes)}
    return model, cfg, idx_to_class, num_classes, n_animals, num_keypoints, checkpoint


def load_threshold_decoder_config(
    checkpoint_path: Path | str,
    config_json: Path | str | None,
    cfg: dict,
    decoder_config_path: Path | str | None,
    disable_decoder: bool = False,
    checkpoint: Optional[dict] = None,
) -> Optional[dict]:
    """Load train-time threshold decoder metadata, preferring explicit CLI path."""
    if disable_decoder:
        print("[infer] threshold decoder disabled by --disable_threshold_decoder", flush=True)
        return None

    if decoder_config_path:
        path = Path(decoder_config_path)
        try:
            with path.open("r", encoding="utf-8") as f:
                dec = json.load(f)
            if dec.get("enabled"):
                print(f"[infer] using threshold decoder {path}", flush=True)
            return dec
        except Exception as exc:
            raise FileNotFoundError(f"Could not load --decoder_config {path}: {exc}") from exc

    if isinstance(cfg, dict) and isinstance(cfg.get("threshold_decoder"), dict):
        dec = cfg.get("threshold_decoder")
        if dec.get("enabled"):
            print("[infer] using threshold decoder from model config", flush=True)
        return dec
    candidates = []
    if config_json is not None:
        candidates.append(Path(config_json).with_name("decoder_config.json"))
    candidates.append(Path(checkpoint_path).parent / "decoder_config.json")

    for path in candidates:
        try:
            if path and path.is_file():
                with path.open("r", encoding="utf-8") as f:
                    dec = json.load(f)
                if dec.get("enabled"):
                    bg = dec.get("background_class")
                    print(f"[infer] using threshold decoder {path}  fallback={bg!r}", flush=True)
                else:
                    print(
                        f"[infer] threshold decoder metadata is disabled in {path}: "
                        f"{dec.get('disabled_reason', 'no reason recorded')}",
                        flush=True,
                    )
                return dec
        except Exception as exc:
            print(f"[warn] could not load threshold decoder {path}: {exc}", flush=True)

    try:
        checkpoint = checkpoint if checkpoint is not None else safe_torch_load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            dec = checkpoint.get("threshold_decoder")
            if isinstance(dec, dict):
                if dec.get("enabled"):
                    print("[infer] using threshold decoder embedded in checkpoint", flush=True)
                return dec
    except Exception as exc:
        print(f"[warn] could not inspect checkpoint for threshold decoder: {exc}", flush=True)

    print("[infer] no threshold decoder found; using raw argmax predictions", flush=True)
    return None


def load_temporal_splitter_config(
    checkpoint_path: Path | str,
    config_json: Path | str | None,
    cfg: dict,
    splitter_config_path: Path | str | None,
    disable_splitter: bool = False,
    cli_overrides: Optional[dict] = None,
    checkpoint: Optional[dict] = None,
) -> Optional[dict]:
    """Load v3 temporal-splitter metadata, mirroring the threshold-decoder loader.

    Resolution order:
        1. --disable_temporal_splitter -> returns None (splitter off)
        2. --temporal_splitter_config <path> -> load from explicit JSON
        3. cfg["temporal_splitter"] or legacy cfg["v25_splitter"]
        4. temporal_splitter.json or legacy v25_splitter.json sidecar
        5. checkpoint embedded metadata
        6. None (silent; older checkpoints work as before)

    CLI overrides (tau_high / tau_low / K) are applied to the loaded config
    before returning. Returns the splitter config dict (with enabled
    True/False) or None if no splitter metadata anywhere.
    """
    if disable_splitter:
        print("[infer] temporal splitter disabled by --disable_temporal_splitter", flush=True)
        return None

    splitter_cfg: Optional[dict] = None

    # Path 2: explicit sidecar via CLI
    if splitter_config_path:
        path = Path(splitter_config_path)
        try:
            with path.open("r", encoding="utf-8") as f:
                splitter_cfg = json.load(f)
            print(f"[infer] using temporal splitter from {path}", flush=True)
        except Exception as exc:
            raise FileNotFoundError(
                f"Could not load --temporal_splitter_config {path}: {exc}"
            ) from exc

    # Path 3: cfg dict
    if splitter_cfg is None and isinstance(cfg, dict) and isinstance(cfg.get(TEMPORAL_SPLITTER_CONFIG_KEY), dict):
        splitter_cfg = cfg[TEMPORAL_SPLITTER_CONFIG_KEY]
        if splitter_cfg.get("enabled"):
            print("[infer] using temporal splitter from model config", flush=True)
    if splitter_cfg is None and isinstance(cfg, dict) and isinstance(cfg.get(LEGACY_V25_SPLITTER_CONFIG_KEY), dict):
        splitter_cfg = cfg[LEGACY_V25_SPLITTER_CONFIG_KEY]
        if splitter_cfg.get("enabled"):
            print("[infer] using temporal splitter from legacy model config", flush=True)

    # Path 4: sidecar next to config / checkpoint
    if splitter_cfg is None:
        candidates = []
        if config_json is not None:
            candidates.append(Path(config_json).with_name("temporal_splitter.json"))
            candidates.append(Path(config_json).with_name("v25_splitter.json"))
        candidates.append(Path(checkpoint_path).parent / "temporal_splitter.json")
        candidates.append(Path(checkpoint_path).parent / "v25_splitter.json")
        for path in candidates:
            try:
                if path and path.is_file():
                    with path.open("r", encoding="utf-8") as f:
                        splitter_cfg = json.load(f)
                    print(f"[infer] using temporal splitter from {path}", flush=True)
                    break
            except Exception as exc:
                print(f"[warn] could not load temporal splitter {path}: {exc}", flush=True)

    # Path 5: checkpoint embedded
    if splitter_cfg is None:
        try:
            checkpoint = checkpoint if checkpoint is not None else safe_torch_load(checkpoint_path, map_location="cpu")
            if isinstance(checkpoint, dict):
                emb = checkpoint.get(TEMPORAL_SPLITTER_CONFIG_KEY)
                if not isinstance(emb, dict):
                    emb = checkpoint.get(LEGACY_V25_SPLITTER_CONFIG_KEY)
                if isinstance(emb, dict):
                    splitter_cfg = emb
                    if splitter_cfg.get("enabled"):
                        print("[infer] using temporal splitter from checkpoint", flush=True)
        except Exception as exc:
            print(f"[warn] could not inspect checkpoint for temporal splitter: {exc}", flush=True)

    if splitter_cfg is None:
        # Silent: older checkpoints just don't have one; that's fine.
        return None

    # Apply CLI overrides + record manual_override flag for provenance.
    # Any later analysis can check splitter_cfg["manual_override"] to
    # distinguish results produced by the val-fit defaults vs by manual
    # tuning.
    if cli_overrides:
        applied_overrides = {}
        for key in ("tau_high", "tau_low", "K"):
            v = cli_overrides.get(key)
            if v is not None:
                splitter_cfg = dict(splitter_cfg)
                splitter_cfg[key] = v
                applied_overrides[key] = v
                print(f"[infer] temporal splitter override {key}={v}", flush=True)
        if applied_overrides:
            splitter_cfg["manual_override"] = True
            splitter_cfg["manual_override_fields"] = applied_overrides
            print(
                "[infer] temporal splitter MANUAL OVERRIDE active - results will not "
                "match the saved val-fit config; downstream metadata records "
                "manual_override=True.",
                flush=True,
            )

    # ---- Config validation: bad config disables the splitter, never half-applies ----
    def _disable(reason: str) -> dict:
        bad = dict(splitter_cfg)
        bad["enabled"] = False
        bad["disabled_reason"] = f"inference-time validation failed: {reason}"
        print(f"[infer] temporal splitter DISABLED - {reason}", flush=True)
        return bad

    if splitter_cfg.get("enabled"):
        try:
            tH = float(splitter_cfg.get("tau_high"))
            tL = float(splitter_cfg.get("tau_low"))
            K = int(splitter_cfg.get("K"))
            tgt_idx = int(splitter_cfg.get("target_index"))
            rep_idx = int(splitter_cfg.get("replacement_index"))
        except (TypeError, ValueError) as exc:
            return _disable(f"could not coerce splitter params to numerics: {exc}")
        if not (tH > tL):
            return _disable(f"tau_high ({tH}) must be > tau_low ({tL})")
        if K < 1:
            return _disable(f"K must be >= 1, got {K}")
        if not (0.0 <= tL <= 1.0) or not (0.0 <= tH <= 1.0):
            return _disable(f"tau_low/tau_high must be in [0, 1], got {tL}/{tH}")
        # Class-index range check (best-effort: cfg may not always carry class_to_idx)
        n_classes_known = None
        if isinstance(cfg, dict) and isinstance(cfg.get("class_to_idx"), dict):
            n_classes_known = len(cfg["class_to_idx"])
        elif isinstance(cfg, dict) and isinstance(cfg.get("idx_to_class"), dict):
            n_classes_known = len(cfg["idx_to_class"])
        if n_classes_known is not None:
            if not (0 <= tgt_idx < n_classes_known):
                return _disable(
                    f"target_index={tgt_idx} out of range "
                    f"[0, {n_classes_known})"
                )
            if not (0 <= rep_idx < n_classes_known):
                return _disable(
                    f"replacement_index={rep_idx} out of range "
                    f"[0, {n_classes_known})"
                )
        # Class-name validation (best-effort): if both names are present
        # in the saved class_to_idx, verify they still resolve to the
        # saved indices. Mismatch means the schema changed under us.
        tgt_name = splitter_cfg.get("target_class")
        rep_name = splitter_cfg.get("replacement_class")
        if isinstance(cfg, dict) and isinstance(cfg.get("class_to_idx"), dict):
            cti = cfg["class_to_idx"]
            if tgt_name and tgt_name in cti and int(cti[tgt_name]) != tgt_idx:
                return _disable(
                    f"target_class {tgt_name!r} maps to index {cti[tgt_name]} "
                    f"in current schema but splitter expects {tgt_idx}"
                )
            if rep_name and rep_name in cti and int(cti[rep_name]) != rep_idx:
                return _disable(
                    f"replacement_class {rep_name!r} maps to index "
                    f"{cti[rep_name]} but splitter expects {rep_idx}"
                )

        print(
            f"[infer] temporal splitter active  target={tgt_name!r} "
            f"tau_high={tH} tau_low={tL} K={K}  "
            f"manual_override={bool(splitter_cfg.get('manual_override'))}  "
            f"(splitter applied to smoothed_frames.csv only; "
            f"MP4 overlay shows pre-splitter labels)",
            flush=True,
        )
    else:
        reason = splitter_cfg.get("disabled_reason", "no reason recorded")
        print(f"[infer] temporal splitter present but disabled: {reason}", flush=True)
    return splitter_cfg


# ----------------------------------------------------------------------------
# Sliding-window inference loop
# ----------------------------------------------------------------------------

def _detection_stream(
    model_yolo,
    source: str | int,
    n_animals: int,
    crop_size: int,
    animal_scale_factor: float,
    group_scale_factor: float,
    body_length_px: float | None,
    device: str,
    keep_last_box: bool,
    pose_conf_threshold: float,
    fps: float,
    yolo_batch: int,
    yield_orig_frame: bool = False,
):
    yield from iterate_yolo_n_crops(
        model_yolo,
        source,
        n_animals=n_animals,
        crop_size=crop_size,
        animal_scale_factor=animal_scale_factor,
        group_scale_factor=group_scale_factor,
        body_length_px=body_length_px,
        crop_strategy="fixed_scale",
        device=device,
        keep_last_box=keep_last_box,
        pose_conf_threshold=pose_conf_threshold,
        fps=fps,
        batch_size=yolo_batch,
        yield_orig_frame=yield_orig_frame,
    )


def run_inference_on_source(
    *,
    model: MultiAnimalBehaviorSequenceClassifier,
    model_yolo,
    source: str | int,
    output_csv: Path,
    idx_to_class: dict,
    num_frames: int,
    window_stride: int,
    n_animals: int,
    num_keypoints: int,
    crop_size: int,
    animal_scale_factor: float,
    group_scale_factor: float,
    body_length_px: float | None,
    device: str,
    keep_last_box: bool,
    pose_conf_threshold: float,
    yolo_batch: int,
    realtime: bool,
    print_each_window: bool,
    fps: float,
    output_video: Optional[Path] = None,
    draw_bboxes: bool = True,
    draw_keypoints: bool = True,
    draw_header: bool = True,
    video_fourcc: str = "mp4v",
    smooth_bouts: bool = False,
    stop_flag_file: Optional[Path] = None,
    metrics: Optional["MetricsLogger | NullMetricsLogger"] = None,
    temporal_smoothing_window: int = 0,
    bout_min_duration_frames: int = 0,
    train_window_stride: Optional[int] = None,
    amp_enabled: bool = False,
    amp_dtype_torch=None,
    threshold_decoder: Optional[dict] = None,
    v25_splitter_config: Optional[dict] = None,
    export_pose: bool = False,
    csv_flush_interval: int = 1,
    preview: bool = False,
    preview_window_name: str = "TandemYTC Live Preview",
) -> dict:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_video = Path(output_video) if output_video is not None else None
    if output_video is not None:
        output_video.parent.mkdir(parents=True, exist_ok=True)

    class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    fieldnames = [
        "window_idx", "frame_start", "frame_end",
        "time_s_start", "time_s_end",
        "predicted_class_id", "predicted_class",
    ]
    fieldnames.extend([f"prob_{c}" for c in class_names])
    fieldnames.extend([
        "n_animals_present_mean", "track_ids_first_frame", "yolo_status_first_frame",
    ])

    csv_f = open(output_csv, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
    writer.writeheader()

    # ---- optional per-frame pose export ----
    pose_csv_f = None
    pose_writer = None
    if export_pose:
        pose_csv_path = output_csv.with_name(output_csv.stem + "_pose.csv")
        _pose_kpt_fields: list[str] = []
        for _ki in range(num_keypoints):
            _pose_kpt_fields += [f"kpt{_ki}_x", f"kpt{_ki}_y", f"kpt{_ki}_conf"]
        _pose_fieldnames = (
            ["frame_idx", "animal_id", "track_id",
             "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "bbox_conf"]
            + _pose_kpt_fields
            + ["predicted_class", "predicted_class_id", "class_conf"]
        )
        pose_csv_f = open(pose_csv_path, "w", newline="", encoding="utf-8")
        pose_writer = csv.DictWriter(pose_csv_f, fieldnames=_pose_fieldnames)
        pose_writer.writeheader()
        print(f"[infer] pose export -> {pose_csv_path}", flush=True)

    buffer: List[NCropDetection] = []
    window_idx = 0
    n_windows_written = 0
    csv_flush_interval = max(1, int(csv_flush_interval))
    t0 = time.time()
    last_log = t0
    stopped_early = False
    # Accumulated for post-loop smoothed frame-level CSV. Each entry is
    # (start_frame, end_frame_inclusive, probs_array). Populated whenever
    # temporal_smoothing_window > 0 or bout_min_duration_frames > 0.
    _need_smoothing = (temporal_smoothing_window > 0 or bout_min_duration_frames > 0)
    window_records: List[Tuple[int, int, np.ndarray]] = []

    if stop_flag_file is None:
        stop_flag_file = output_csv.parent / "stop_inference.flag"
    stop_flag_file = Path(stop_flag_file)
    if stop_flag_file.exists():
        try:
            stop_flag_file.unlink()
        except Exception:
            pass

    current_label: Optional[str] = None
    current_conf: Optional[float] = None
    current_probs: Optional[np.ndarray] = None
    n_frames_written_video = 0
    video_writer: Optional[cv2.VideoWriter] = None
    want_video = output_video is not None
    want_preview = bool(preview)
    want_overlay = want_video or want_preview

    def _preview_frame(frame_bgr: np.ndarray) -> bool:
        if not want_preview:
            return True
        try:
            cv2.imshow(str(preview_window_name or "TandemYTC Live Preview"), frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                _request_stop("live preview close requested")
                return False
        except Exception as exc:
            print(f"[warn] live preview failed and will be disabled: {exc}", flush=True)
            return False
        return True

    # ---- bout-smoothing state (only used if smooth_bouts=True) ----
    # When enabled, we delay writing each frame to the MP4 until ALL windows
    # that will ever cover it have been emitted. We then take the average of
    # those windows' softmax probs and label the frame with their argmax.
    # This eliminates the abrupt label flips you see with "latest wins"
    # when bouts straddle a window-stride boundary. Cost: the MP4 lags
    # ~num_frames behind real-time and adds ~num_frames worth of frame
    # buffering memory.
    pending_dets: List[NCropDetection] = []        # frames waiting to be written
    pending_prob_sum: dict[int, np.ndarray] = {}    # frame_idx -> sum of window probs
    pending_prob_count: dict[int, int] = {}         # frame_idx -> n windows covering it

    src_repr = str(source) if not isinstance(source, int) else f"webcam:{source}"
    print(f"[infer] source={src_repr}  num_frames={num_frames}  stride={window_stride}"
          + ("  +annotated_video" if want_video else "")
          + ("  +live_preview" if want_preview else ""), flush=True)

    # Default to a no-op metrics logger if the caller didn't supply one,
    # so the rest of the loop can call metrics.* unconditionally.
    if metrics is None:
        metrics = NullMetricsLogger()
    metrics.start()

    softmax = torch.nn.Softmax(dim=-1)
    try:
        for det in _detection_stream(
            model_yolo, source, n_animals, crop_size,
            animal_scale_factor, group_scale_factor, body_length_px,
            device, keep_last_box, pose_conf_threshold, fps, yolo_batch,
            yield_orig_frame=want_overlay,
        ):
            metrics.frame_seen()
            # ---- graceful stop check ----
            if _STOP_REQUESTED or stop_flag_file.exists():
                stopped_early = True
                if not _STOP_REQUESTED:
                    _request_stop(f"sentinel file {stop_flag_file}")
                break

            buffer.append(det)

            # ---- write per-frame pose row (if export_pose) ----
            if pose_writer is not None:
                for _ai in range(len(det.animal_mask)):
                    if not det.animal_mask[_ai]:
                        continue
                    _bbox = det.bbox_xyxy_animals[_ai] if _ai < len(det.bbox_xyxy_animals) else [float("nan")] * 4
                    _kxy  = det.keypoints_xy_raw[_ai]   if _ai < len(det.keypoints_xy_raw)   else np.full((num_keypoints, 2), float("nan"))
                    _kc   = det.keypoints_conf_raw[_ai]  if _ai < len(det.keypoints_conf_raw)  else np.full(num_keypoints, float("nan"))
                    _pose_row: dict = {
                        "frame_idx":  int(det.frame_idx),
                        "animal_id":  _ai,
                        "track_id":   int(det.track_ids[_ai]) if _ai < len(det.track_ids) else -1,
                        "bbox_x1":    round(float(_bbox[0]), 2),
                        "bbox_y1":    round(float(_bbox[1]), 2),
                        "bbox_x2":    round(float(_bbox[2]), 2),
                        "bbox_y2":    round(float(_bbox[3]), 2),
                        "bbox_conf":  round(float(det.crop_conf[_ai]) if _ai < len(det.crop_conf) else float("nan"), 4),
                        "predicted_class":    current_label if current_label is not None else "",
                        "predicted_class_id": int(pred_id) if "pred_id" in locals() else -1,
                        "class_conf":         round(float(current_conf), 4) if current_conf is not None else float("nan"),
                    }
                    for _ki in range(num_keypoints):
                        _pose_row[f"kpt{_ki}_x"]    = round(float(_kxy[_ki, 0]), 2) if _ki < len(_kxy) else float("nan")
                        _pose_row[f"kpt{_ki}_y"]    = round(float(_kxy[_ki, 1]), 2) if _ki < len(_kxy) else float("nan")
                        _pose_row[f"kpt{_ki}_conf"] = round(float(_kc[_ki]), 4)     if _ki < len(_kc)  else float("nan")
                    pose_writer.writerow(_pose_row)

            if want_video and video_writer is None and det.orig_frame_bgr is not None:
                h, w = det.orig_frame_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*video_fourcc)
                video_writer = cv2.VideoWriter(
                    str(output_video), fourcc, float(fps), (int(w), int(h))
                )
                if not video_writer.isOpened():
                    print(f"[warn] could not open VideoWriter at {output_video}; "
                          "annotated video will be skipped.", flush=True)
                    video_writer = None
                    want_video = False
                else:
                    print(f"[infer] writing annotated video -> {output_video}  "
                          f"({w}x{h} @ {fps:.2f} fps, fourcc={video_fourcc})", flush=True)

            if want_overlay and det.orig_frame_bgr is not None:
                if not smooth_bouts:
                    # "latest wins" path: write the frame immediately with
                    # whatever the most recent window predicted.
                    annotated = annotate_frame_n(
                        det,
                        label=current_label,
                        confidence=current_conf,
                        fps=fps,
                        class_probs=(current_probs.tolist()
                                     if current_probs is not None else None),
                        class_names=class_names,
                        draw_bboxes=draw_bboxes,
                        draw_keypoints=draw_keypoints,
                        draw_header=draw_header,
                    )
                    if annotated is not None:
                        if video_writer is not None:
                            video_writer.write(annotated)
                            n_frames_written_video += 1
                        if not _preview_frame(annotated):
                            stopped_early = True
                            break
                else:
                    # smoothing path: queue the frame; we'll average all
                    # overlapping windows once they've all been emitted.
                    pending_dets.append(det)
                    pending_prob_count.setdefault(int(det.frame_idx), 0)
                    pending_prob_sum.setdefault(
                        int(det.frame_idx),
                        np.zeros(len(class_names), dtype=np.float64),
                    )
                    # Drain frames that no future window can possibly cover.
                    # A frame at idx F is last touched by the window starting
                    # at the largest stride boundary <= F that is also
                    # >= F - num_frames + 1. So once the current detection's
                    # frame_idx is >= F + num_frames, F is finalized.
                    cutoff = int(det.frame_idx) - num_frames
                    while pending_dets and int(pending_dets[0].frame_idx) <= cutoff:
                        d0 = pending_dets.pop(0)
                        f0 = int(d0.frame_idx)
                        n_cov = pending_prob_count.pop(f0, 0)
                        avg = pending_prob_sum.pop(f0, None)
                        if avg is not None and n_cov > 0:
                            avg = avg / float(n_cov)
                            pid = int(apply_threshold_decoder_probs(avg, threshold_decoder))
                            lbl = idx_to_class.get(pid, str(pid))
                            conf = float(avg[pid])
                            probs_list = avg.tolist()
                        else:
                            lbl, conf, probs_list = None, None, None
                        annotated = annotate_frame_n(
                            d0,
                            label=lbl,
                            confidence=conf,
                            fps=fps,
                            class_probs=probs_list,
                            class_names=class_names,
                            draw_bboxes=draw_bboxes,
                            draw_keypoints=draw_keypoints,
                            draw_header=draw_header,
                        )
                        if annotated is not None:
                            if video_writer is not None:
                                video_writer.write(annotated)
                                n_frames_written_video += 1
                            if not _preview_frame(annotated):
                                stopped_early = True
                                break

            while len(buffer) >= num_frames:
                window_dets = buffer[:num_frames]
                start_frame = int(window_dets[0].frame_idx)
                end_frame = int(window_dets[-1].frame_idx)
                batch = detections_to_window_batch(
                    window_dets,
                    n_animals,
                    num_keypoints,
                )
                _infer_dtype = amp_dtype_torch if amp_dtype_torch is not None else torch.float16
                with metrics.stage("classify"):
                    with torch.inference_mode(), _autocast("cuda", dtype=_infer_dtype, enabled=amp_enabled):
                        logits = model(batch).detach().cpu()
                probs = softmax(logits.float()).squeeze(0).numpy()
                pred_id = int(apply_threshold_decoder_probs(probs, threshold_decoder))

                row = {
                    "window_idx": window_idx,
                    "frame_start": start_frame,
                    "frame_end": end_frame,
                    "time_s_start": start_frame / max(fps, 1e-6),
                    "time_s_end": end_frame / max(fps, 1e-6),
                    "predicted_class_id": pred_id,
                    "predicted_class": idx_to_class.get(pred_id, str(pred_id)),
                }
                for ci, cname in enumerate(class_names):
                    row[f"prob_{cname}"] = float(probs[ci])
                npres = float(np.mean([np.sum(d.animal_mask) for d in window_dets]))
                row["n_animals_present_mean"] = npres
                row["track_ids_first_frame"] = ",".join(str(int(t)) for t in window_dets[0].track_ids.tolist())
                row["yolo_status_first_frame"] = window_dets[0].yolo_status
                writer.writerow(row)
                n_windows_written += 1
                if (n_windows_written % csv_flush_interval) == 0:
                    csv_f.flush()
                metrics.window_done(window_idx)

                if _need_smoothing:
                    window_records.append((start_frame, end_frame, probs.copy()))

                current_label = idx_to_class.get(pred_id, str(pred_id))
                current_conf = float(probs[pred_id])
                current_probs = probs

                # Credit this window's probs to every frame it covers, so the
                # smoothing path can average overlapping windows per frame.
                if smooth_bouts and want_overlay:
                    for f in range(start_frame, end_frame + 1):
                        if f in pending_prob_sum:
                            pending_prob_sum[f] = pending_prob_sum[f] + probs.astype(np.float64)
                            pending_prob_count[f] = pending_prob_count.get(f, 0) + 1

                if realtime or print_each_window:
                    print(
                        f"[infer] win {window_idx:5d}  f{start_frame:6d}-{end_frame:6d}  "
                        f"-> {idx_to_class.get(pred_id, str(pred_id))}  ({probs[pred_id]:.2f})",
                        flush=True,
                    )

                window_idx += 1
                buffer = buffer[window_stride:]

            if not realtime and not print_each_window:
                now = time.time()
                if now - last_log >= 5.0:
                    print(
                        f"[infer] buffered {len(buffer)} dets, windows_emitted={n_windows_written}, elapsed={now - t0:.1f}s",
                        flush=True,
                    )
                    last_log = now
    except KeyboardInterrupt:
        # Belt-and-suspenders: if our SIGINT handler somehow didn't run,
        # still treat Ctrl+C as a graceful stop.
        stopped_early = True
        _request_stop("KeyboardInterrupt")
    finally:
        # Flush + close the CSV. Each row was already flushed at write time,
        # but closing ensures the OS buffers are reclaimed cleanly.
        try:
            csv_f.flush()
        except Exception:
            pass
        try:
            csv_f.close()
        except Exception:
            pass
        try:
            if pose_csv_f is not None:
                pose_csv_f.close()
        except Exception:
            pass
        # Flush + close the metrics CSV (writes a final partial row if the
        # last interval didn't naturally close).
        try:
            metrics.close()
        except Exception:
            pass
        # Drain the smoothing buffer: any remaining frames (the last
        # ~num_frames of the video) should be written using whatever
        # windows did cover them. They have fewer overlapping windows
        # than mid-video frames, so labels there may be slightly noisier
        # — that's a cheap price for avoiding lost tail-of-video frames.
        if smooth_bouts and want_overlay and pending_dets:
            try:
                for d0 in pending_dets:
                    f0 = int(d0.frame_idx)
                    n_cov = pending_prob_count.get(f0, 0)
                    avg = pending_prob_sum.get(f0)
                    if avg is not None and n_cov > 0:
                        avg = avg / float(n_cov)
                        pid = int(apply_threshold_decoder_probs(avg, threshold_decoder))
                        lbl = idx_to_class.get(pid, str(pid))
                        conf = float(avg[pid])
                        probs_list = avg.tolist()
                    else:
                        lbl, conf, probs_list = None, None, None
                    annotated = annotate_frame_n(
                        d0, label=lbl, confidence=conf, fps=fps,
                        class_probs=probs_list, class_names=class_names,
                        draw_bboxes=draw_bboxes,
                        draw_keypoints=draw_keypoints,
                        draw_header=draw_header,
                    )
                    if annotated is not None:
                        if video_writer is not None:
                            video_writer.write(annotated)
                            n_frames_written_video += 1
                        _preview_frame(annotated)
            except Exception as e:
                print(f"[warn] tail-flush of smoothed frames failed: {e}", flush=True)
        # Finalize the MP4. release() writes the moov atom for mp4v / avc1
        # and properly closes the file handle. WITHOUT this, the MP4 has no
        # index and most players (incl. VLC) refuse to play it.
        if video_writer is not None:
            try:
                video_writer.release()
            except Exception:
                pass
        if want_preview:
            try:
                cv2.destroyWindow(str(preview_window_name or "TandemYTC Live Preview"))
            except Exception:
                pass
        # Clean up the sentinel so it does not haunt the next run.
        try:
            if stop_flag_file.exists():
                stop_flag_file.unlink()
        except Exception:
            pass

        # Smoothed frame-level CSV (written post-loop so all window probs are
        # available before smoothing — streaming row-by-row would introduce
        # ordering bugs at boundaries).
        if _need_smoothing and window_records:
            try:
                smoothed_path = _write_smoothed_frames_csv(
                    window_records, output_csv, class_names, idx_to_class,
                    fps, temporal_smoothing_window, bout_min_duration_frames,
                    threshold_decoder=threshold_decoder,
                    v25_splitter_config=v25_splitter_config,
                )
                if smoothed_path:
                    print(f"[infer] smoothed frames CSV -> {smoothed_path}", flush=True)
                    # Loud warning when splitter is active but the user requested an
                    # MP4 overlay: the two outputs will show different bout counts
                    # for the target class. This matters for any qualitative figure.
                    if (v25_splitter_config and v25_splitter_config.get("enabled")
                            and want_video and output_video is not None):
                        tgt = v25_splitter_config.get("target_class", "target")
                        print(
                            f"[temporal splitter] NOTE: smoothed_frames.csv applies the "
                            f"splitter (target={tgt!r}); the annotated MP4 at "
                            f"{output_video} shows PRE-splitter labels (streaming "
                            f"pipeline cannot buffer arbitrary frame ranges). "
                            f"Bout counts in the CSV and MP4 will differ for the "
                            f"target class. Use the CSV for evaluation; the MP4 is "
                            f"a pre-splitter qualitative overlay.",
                            flush=True,
                        )
            except Exception as _e:
                print(f"[warn] failed to write smoothed frames CSV: {_e}", flush=True)

    elapsed = time.time() - t0
    summary_parts = [f"windows={n_windows_written}", f"csv={output_csv}"]
    if want_video and output_video is not None:
        summary_parts.append(f"video={output_video} ({n_frames_written_video} frames)")
    if want_preview:
        summary_parts.append("live_preview=on")
    if stopped_early:
        summary_parts.append("[stopped early]")
    print(f"[infer] done in {elapsed:.1f}s  " + "  ".join(summary_parts), flush=True)
    summary = {"windows": n_windows_written, "elapsed_s": elapsed,
               "csv": str(output_csv), "stopped_early": stopped_early}
    if want_video and output_video is not None:
        summary["video"] = str(output_video)
        summary["video_frames_written"] = n_frames_written_video
    summary["live_preview"] = bool(want_preview)
    return summary


# ----------------------------------------------------------------------------
# Source resolution
# ----------------------------------------------------------------------------

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".seq")


def resolve_sources(source: str) -> List[Tuple]:
    try:
        idx = int(source)
        return [(idx, f"webcam_{idx}")]
    except (TypeError, ValueError):
        pass
    p = Path(source)
    if p.is_file():
        return [(str(p), p.stem)]
    if p.is_dir():
        out: List[Tuple] = []
        for f in sorted(p.rglob("*")):
            if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
                out.append((str(f), f.stem))
        return out
    raise SystemExit(f"--source {source!r} is not an int, file, or directory.")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="TandemYTC inference (single video / batch / real-time).")
    p.add_argument("--model_path", required=True, help="Trained model .pt (from train_y.py).")
    p.add_argument("--model_config", default=None, help="config.json from the same train_y.py run.")
    p.add_argument("--yolo_weights", required=True)
    p.add_argument("--yolo_task", default="pose")
    p.add_argument("--source", required=True, help="Video file, directory of videos, or integer webcam index.")
    p.add_argument("--output", default=None, help="Output CSV (single source mode).")
    p.add_argument("--output_dir", default=None, help="Output directory (batch mode).")
    p.add_argument("--realtime", action="store_true")
    p.add_argument("--print_each_window", action="store_true")
    p.add_argument("--num_frames", type=int, default=0)
    p.add_argument("--window_stride", type=int, default=0)
    p.add_argument("--crop_size", type=int, default=224)
    p.add_argument("--animal_scale_factor", type=float, default=4.0)
    p.add_argument("--group_scale_factor", type=float, default=8.0)
    p.add_argument("--body_length_px", type=float, default=0.0,
                   help="Per-video body length used for crop scaling and "
                        "relation-feature normalisation. If 0 and "
                        "--calibration_json is given, the value is loaded "
                        "from the calibration file by source-video name. "
                        "If neither is supplied, falls back to per-frame "
                        "median (training-time mismatch — see warning).")
    p.add_argument("--calibration_json", default=None,
                   help="Path to calibration.json produced by "
                        "prepare_full_video_npz.py. When set, the "
                        "per-video body_length_px is looked up by matching "
                        "the source video stem against the calibration's "
                        "video_id field. If omitted, infer_y.py first uses "
                        "embedded checkpoint calibration when available, then "
                        "falls back to auto-detection near the training manifest.")
    p.add_argument("--keep_last_box", action="store_true", default=True)
    p.add_argument("--pose_conf_threshold", type=float, default=0.3)
    p.add_argument("--yolo_batch", type=int, default=25)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # ---- Annotated video export ----
    p.add_argument("--output_video", default=None,
                   help="Annotated MP4 path (single-source mode).")
    p.add_argument("--output_video_dir", default=None,
                   help="Annotated MP4 output directory (batch mode).")
    p.add_argument("--preview", action="store_true",
                   help="Show a live OpenCV preview window with the same "
                        "boxes, keypoints, and behavior labels used for the "
                        "annotated MP4. Press q or Esc in the preview window "
                        "to request a clean stop.")
    p.add_argument("--preview_window_name", default="TandemYTC Live Preview",
                   help="Window title used by --preview.")
    p.add_argument("--no_bboxes", action="store_true")
    p.add_argument("--no_keypoints", action="store_true")
    p.add_argument("--no_video_header", action="store_true")
    p.add_argument("--export_pose", action="store_true",
                   help="Write a per-frame pose CSV alongside the classification CSV. "
                        "Contains one row per animal per frame: bounding box (x1,y1,x2,y2) "
                        "in original-frame pixels, per-keypoint (x,y,conf) for each of the "
                        "K YOLO-pose keypoints, and the most-recent window classification. "
                        "Output path: {output_csv_stem}_pose.csv")
    p.add_argument("--video_fourcc", default="mp4v",
                   help="FourCC for the OpenCV VideoWriter.")
    p.add_argument("--smooth_bouts", action="store_true",
                   help="When writing the annotated MP4, average the softmax "
                        "probabilities of all windows that cover each frame "
                        "before picking its displayed label, instead of "
                        "showing the most recent window's prediction. Removes "
                        "the abrupt label flips at window-stride boundaries. "
                        "Only meaningful when window_stride < num_frames.")
    # ---- Graceful stop ----
    p.add_argument("--stop_flag_file", default=None,
                   help="Optional sentinel file path. When the file appears, "
                        "the inference loop exits cleanly: CSV is flushed, "
                        "the MP4 moov atom is written, and the sentinel is "
                        "removed. The loop also honours SIGINT (Ctrl+C) and "
                        "SIGTERM.")
    # ---- Runtime-metrics logger ----
    p.add_argument("--log_metrics", action="store_true",
                   help="Log per-interval runtime metrics (FPS, classifier "
                        "time, CPU/GPU memory and utilization) to a sidecar "
                        "CSV alongside the predictions output. Default off "
                        "for CLI; the GUI defaults this on.")
    p.add_argument("--metrics_csv", default=None,
                   help="Path for the metrics CSV. If omitted (and "
                        "--log_metrics is set), the path is auto-derived "
                        "as <predictions_csv>.metrics.csv next to the "
                        "predictions output.")
    p.add_argument("--metrics_log_interval", type=int, default=30,
                   help="Number of windows between metrics CSV rows. "
                        "Default 30 ~= 1 second of source video at 30 fps "
                        "with stride 16.")
    p.add_argument("--csv_flush_interval", type=int, default=1,
                   help="Flush prediction CSV every N windows. Default 1 preserves "
                        "the most conservative crash-recovery behavior. Higher values "
                        "reduce disk sync overhead for throughput benchmarking; files "
                        "are still flushed and closed on graceful stop.")
    # ---- Temporal smoothing (evaluation-grade; affects frame-level CSV) ----
    p.add_argument("--temporal_smoothing_window", type=int, default=0,
                   help="Median-filter window applied to per-frame predicted class labels "
                        "before writing the smoothed_frames CSV. 0 = disabled. "
                        "Tune this on the validation set before looking at test sets.")
    p.add_argument("--bout_min_duration_frames", type=int, default=0,
                   help="Remove predicted bouts shorter than N frames by merging them into "
                        "the longer adjacent run. 0 = disabled. Applied after median filter. "
                        "Tune on validation data only.")
    # ---- Validation-fitted threshold decoder ----
    p.add_argument("--decoder_config", default=None,
                   help="Optional decoder_config.json from train_y.py. If omitted, infer_y.py "
                        "auto-loads decoder metadata from the model config, checkpoint, or "
                        "run directory when present.")
    p.add_argument("--disable_threshold_decoder", action="store_true",
                   help="Ignore any saved validation-fitted threshold decoder and use raw "
                        "softmax argmax predictions.")
    # ---- TandemYTC temporal bout splitter ----
    # Auto-loaded from the model config/checkpoint, same pattern as the
    # threshold decoder. Behaviorist users do NOT need to set these flags.
    p.add_argument("--temporal_splitter_config", dest="v25_splitter_config",
                   default=None, metavar="PATH",
                   help="Optional temporal_splitter.json sidecar. If omitted, infer_y.py "
                        "auto-loads splitter metadata from the model config or checkpoint "
                        "when present. Older checkpoints simply have no splitter.")
    p.add_argument("--disable_temporal_splitter", dest="disable_v25_splitter",
                   action="store_true",
                   help="Ignore any saved temporal splitter and emit pre-splitter labels.")
    p.add_argument("--temporal_splitter_tau_high", dest="v25_splitter_tau_high",
                   type=float, default=None, metavar="FLOAT",
                   help="Override loaded high-confidence threshold. Advanced use only.")
    p.add_argument("--temporal_splitter_tau_low", dest="v25_splitter_tau_low",
                   type=float, default=None, metavar="FLOAT",
                   help="Override loaded low-confidence threshold. Advanced use only.")
    p.add_argument("--temporal_splitter_K", dest="v25_splitter_K",
                   type=int, default=None, metavar="FRAMES",
                   help="Override loaded K, the minimum dip duration in frames.")
    p.add_argument("--v25_splitter_config", dest="v25_splitter_config",
                   default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.add_argument("--disable_v25_splitter", dest="disable_v25_splitter",
                   action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--v25_splitter_tau_high", dest="v25_splitter_tau_high",
                   type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.add_argument("--v25_splitter_tau_low", dest="v25_splitter_tau_low",
                   type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.add_argument("--v25_splitter_K", dest="v25_splitter_K",
                   type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    # ---- Inference AMP (independent of training AMP) ----
    p.add_argument("--amp", action="store_true", default=False,
                   help="Enable mixed-precision for the classifier forward pass during "
                        "inference. Independent of the training --amp flag. "
                        "Inference AMP precision is recorded in the infer_meta JSON.")
    p.add_argument("--amp_dtype", choices=["auto", "fp16", "bf16"], default="auto",
                   help="Inference AMP dtype. 'auto' selects bf16 on Ampere+, fp16 elsewhere.")
    return p.parse_args()


def _load_calibration_lookup(calibration_json: str) -> dict:
    """Load the per-video body-length calibration produced by
    prepare_full_video_npz.py and return a dict keyed by video_id.
    Returns an empty dict on any failure so the caller can fall back cleanly.
    """
    try:
        with open(calibration_json, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception as exc:
        print(f"[infer] WARNING: could not read calibration_json ({exc}); "
              f"will fall back to per-frame median.", flush=True)
        return {}
    pvc = d.get("per_video_calibration", [])
    out = {}
    for entry in pvc:
        vid = entry.get("video_id")
        bl = entry.get("body_length_px")
        if vid and bl is not None:
            out[str(vid)] = float(bl)
    return out


def _embedded_calibration_lookup(cfg: dict) -> dict:
    cal = cfg.get("calibration") if isinstance(cfg, dict) else None
    if not isinstance(cal, dict) or not cal.get("embedded"):
        return {}
    lookup = cal.get("lookup", {})
    if not isinstance(lookup, dict):
        return {}
    out = {}
    for vid, bl in lookup.items():
        try:
            out[str(vid)] = float(bl)
        except Exception:
            continue
    return out


def _lookup_body_length(calib: dict, source_stem: str):
    """Match a source video stem against the calibration's video_id keys.
    Video filenames may include suffixes (e.g. '_Top_J85') beyond the bare
    video ID used as the calibration key. Match by prefix when exact fails.
    """
    if source_stem in calib:
        return calib[source_stem]
    for vid, bl in calib.items():
        if source_stem.startswith(vid):
            return bl
    return None


def main():
    _install_signal_handlers()
    args = parse_args()

    config_path = args.model_config
    if config_path is None:
        candidate = Path(args.model_path).parent / "config.json"
        if candidate.is_file():
            config_path = str(candidate)
            print(f"[infer] auto-detected model_config={config_path}", flush=True)

    device = normalize_device(args.device) if args.device else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
    model, cfg, idx_to_class, num_classes, n_animals, num_keypoints, checkpoint = load_model_from_checkpoint(
        args.model_path, config_path, device=device,
        yolo_weights_fallback=args.yolo_weights,
    )
    threshold_decoder = load_threshold_decoder_config(
        checkpoint_path=args.model_path,
        config_json=config_path,
        cfg=cfg,
        decoder_config_path=args.decoder_config,
        disable_decoder=bool(args.disable_threshold_decoder),
        checkpoint=checkpoint,
    )
    v25_splitter_config = load_temporal_splitter_config(
        checkpoint_path=args.model_path,
        config_json=config_path,
        cfg=cfg,
        splitter_config_path=args.v25_splitter_config,
        disable_splitter=bool(args.disable_v25_splitter),
        cli_overrides={
            "tau_high": args.v25_splitter_tau_high,
            "tau_low":  args.v25_splitter_tau_low,
            "K":        args.v25_splitter_K,
        },
        checkpoint=checkpoint,
    )
    num_frames = int(args.num_frames or cfg.get("num_frames", 32))
    window_stride = int(args.window_stride or cfg.get("window_stride", max(num_frames // 2, 1)))
    if num_frames <= 0:
        raise SystemExit(f"num_frames={num_frames}; expected a positive integer.")
    if window_stride <= 0:
        raise SystemExit(f"window_stride={window_stride}; expected a positive integer.")

    # Stride metadata: record training stride vs inference stride separately.
    train_window_stride = int(cfg.get("window_stride", max(num_frames // 2, 1)))
    if window_stride != train_window_stride:
        print(
            f"[infer] NOTE: inference stride ({window_stride}) differs from "
            f"training stride ({train_window_stride}). Using a denser stride is valid; "
            f"both are recorded in the infer_meta JSON.",
            flush=True,
        )

    # Inference AMP — resolved independently of training AMP.
    _infer_amp_active = bool(args.amp) and torch.cuda.is_available()
    if args.amp and not _infer_amp_active:
        print("[warn] --amp requested but CUDA not available; inference AMP disabled.", flush=True)
    _infer_amp_dtype_resolved = _resolve_amp_dtype(args.amp_dtype) if _infer_amp_active else None
    _infer_amp_dtype_torch = (
        (torch.bfloat16 if _infer_amp_dtype_resolved == "bf16" else torch.float16)
        if _infer_amp_active else None
    )
    if _infer_amp_active:
        print(f"[infer] AMP enabled  dtype_resolved={_infer_amp_dtype_resolved}", flush=True)

    print(f"[infer] feature_source=pose_social  classes={list(idx_to_class.values())}", flush=True)
    print(f"[infer] num_frames={num_frames}  stride={window_stride}  N={n_animals}  K={num_keypoints}", flush=True)
    if int(args.temporal_smoothing_window) > 0 or int(args.bout_min_duration_frames) > 0:
        print(
            f"[infer] post-processing: smoothing_window={args.temporal_smoothing_window}  "
            f"min_duration={args.bout_min_duration_frames}  "
            f"-> smoothed_frames CSV will be written alongside window CSV",
            flush=True,
        )

    embedded_calib_lookup = _embedded_calibration_lookup(cfg)
    embedded_calib_meta = cfg.get("calibration", {}) if isinstance(cfg, dict) else {}

    if not args.calibration_json and not embedded_calib_lookup and float(args.body_length_px) <= 0:
        manifest_path = cfg.get("manifest_path")
        if manifest_path:
            candidate = Path(manifest_path).parent / "calibration.json"
            if candidate.is_file():
                args.calibration_json = str(candidate)
                print(
                    f"[infer] auto-detected calibration_json={args.calibration_json} "
                    "from the training manifest directory",
                    flush=True,
                )

    model_yolo = load_yolo_model(args.yolo_weights, device=device, task=args.yolo_task)
    yolo_num_keypoints = inspect_yolo_keypoint_count(args.yolo_weights)
    if yolo_num_keypoints is None:
        print(
            "[infer] WARNING: could not inspect YOLO pose keypoint count. "
            "The first classified window will still validate the detected "
            "keypoint tensor shape against the trained model config.",
            flush=True,
        )
    elif int(yolo_num_keypoints) != int(num_keypoints):
        raise SystemExit(
            f"YOLO checkpoint emits {yolo_num_keypoints} keypoints, but the "
            f"trained TandemYTC model expects num_keypoints={num_keypoints}. "
            "Use the same keypoint YOLO model/schema that was used to build "
            "the training NPZs and checkpoint."
        )

    sources = resolve_sources(args.source)
    if not sources:
        print("[infer] no videos found at --source.", flush=True)
        return

    is_single = (
        (len(sources) == 1)
        and (
            (str(args.source).isdigit())
            or (not isinstance(sources[0][0], int) and Path(args.source).is_file())
        )
    )
    if is_single and args.output:
        output_csv = Path(args.output)
        if args.output_video:
            out_pairs = [(sources[0][0], output_csv, Path(args.output_video))]
        elif args.output_video_dir:
            vid_dir = Path(args.output_video_dir)
            label = sources[0][1]
            out_pairs = [(sources[0][0], output_csv,
                          vid_dir / f"{label}.annotated.mp4")]
        else:
            out_pairs = [(sources[0][0], output_csv, None)]
    else:
        out_dir = Path(args.output_dir or "predictions")
        if args.output_video_dir:
            vid_dir = Path(args.output_video_dir)
            out_pairs = [(src,
                          out_dir / f"{label}.behavior.csv",
                          vid_dir / f"{label}.annotated.mp4")
                         for (src, label) in sources]
        else:
            out_pairs = [(src, out_dir / f"{label}.behavior.csv", None)
                         for (src, label) in sources]

    # Per-video body_length_px calibration. Order of precedence:
    #   1. Explicit --body_length_px (if > 0)
    #   2. Lookup in --calibration_json by source video stem
    #   3. Embedded checkpoint/config calibration lookup
    #   4. Auto-detected manifest-side calibration.json
    #   5. None -> per-frame median fallback inside cropping_y (with warning)
    explicit_body_length = float(args.body_length_px) if args.body_length_px > 0 else None
    calib_source = None
    calib_lookup = {}
    if args.calibration_json:
        calib_lookup = _load_calibration_lookup(args.calibration_json)
        calib_source = str(args.calibration_json) if calib_lookup else None
    elif embedded_calib_lookup:
        calib_lookup = embedded_calib_lookup
        calib_source = "embedded_checkpoint"
    if calib_lookup:
        print(f"[infer] loaded {len(calib_lookup)} per-video calibrations from "
              f"{calib_source}", flush=True)
    elif explicit_body_length is None:
        print("[infer] WARNING: no --body_length_px and no --calibration_json supplied. "
              "Inference will use per-frame median bbox size for crop scale, which "
              "differs from training-time per-video calibration. Pass "
              "--calibration_json to reproduce training crops exactly.", flush=True)

    stop_flag_path = Path(args.stop_flag_file) if args.stop_flag_file else None

    summaries = []
    for (src, out_csv, out_video) in out_pairs:
        if _STOP_REQUESTED or (stop_flag_path is not None and stop_flag_path.exists()):
            print(f"[infer] stop requested; skipping {src}", flush=True)
            break

        src_label = str(src) if not isinstance(src, int) else f"webcam:{src}"

        # Build an optional MetricsLogger for this source. The CSV path is
        # auto-derived from the predictions CSV unless overridden. For batch
        # mode, --metrics_csv is treated as a *base path* and we append the
        # source label so each source gets its own file (avoids overwriting).
        if bool(args.log_metrics):
            if args.metrics_csv:
                base = Path(args.metrics_csv)
                if len(out_pairs) > 1:
                    metrics_path = base.with_name(
                        f"{base.stem}_{Path(out_csv).stem}.metrics.csv"
                    )
                else:
                    metrics_path = base
            else:
                metrics_path = derive_metrics_csv_path(out_csv)
            metrics = MetricsLogger(
                metrics_path,
                log_interval_windows=int(args.metrics_log_interval),
                source=src_label,
            )
        else:
            metrics = None

        # Resolve body_length_px per source: explicit CLI value wins; else
        # look up calibration by source video stem; else None (per-frame fallback).
        per_source_body_len = explicit_body_length
        if per_source_body_len is None and calib_lookup:
            stem = Path(str(src)).stem
            per_source_body_len = _lookup_body_length(calib_lookup, stem)
            if per_source_body_len is not None:
                print(f"[infer] {src_label}: using calibration body_length_px="
                      f"{per_source_body_len:.2f}", flush=True)
            else:
                print(
                    f"[infer] WARNING: {src_label}: no calibration entry matched "
                    f"source stem {stem!r}; falling back to per-frame median "
                    "body length. Crop scale may differ from training.",
                    flush=True,
                )

        s = run_inference_on_source(
            model=model,
            model_yolo=model_yolo,
            source=src,
            output_csv=out_csv,
            idx_to_class=idx_to_class,
            num_frames=num_frames,
            window_stride=window_stride,
            n_animals=n_animals,
            num_keypoints=num_keypoints,
            crop_size=int(args.crop_size),
            animal_scale_factor=float(args.animal_scale_factor),
            group_scale_factor=float(args.group_scale_factor),
            body_length_px=per_source_body_len,
            device=device,
            keep_last_box=bool(args.keep_last_box),
            pose_conf_threshold=float(args.pose_conf_threshold),
            yolo_batch=int(args.yolo_batch),
            realtime=bool(args.realtime),
            print_each_window=bool(args.print_each_window),
            fps=float(args.fps),
            output_video=out_video,
            draw_bboxes=not bool(args.no_bboxes),
            draw_keypoints=not bool(args.no_keypoints),
            draw_header=not bool(args.no_video_header),
            video_fourcc=str(args.video_fourcc),
            smooth_bouts=bool(args.smooth_bouts),
            stop_flag_file=stop_flag_path,
            metrics=metrics,
            temporal_smoothing_window=int(args.temporal_smoothing_window),
            bout_min_duration_frames=int(args.bout_min_duration_frames),
            train_window_stride=train_window_stride,
            amp_enabled=_infer_amp_active,
            amp_dtype_torch=_infer_amp_dtype_torch,
            threshold_decoder=threshold_decoder,
            v25_splitter_config=v25_splitter_config,
            export_pose=bool(getattr(args, "export_pose", False)),
            csv_flush_interval=int(args.csv_flush_interval),
            preview=bool(args.preview),
            preview_window_name=str(args.preview_window_name),
        )
        # Write per-source inference metadata alongside the predictions CSV so
        # held-out evaluation runs are fully reproducible (stride, smoothing
        # settings, inference precision are all recorded).
        try:
            infer_meta = {
                "source": str(src),
                "model_path": str(args.model_path),
                "feature_source": "pose_social",
                "num_frames": num_frames,
                "train_window_stride": train_window_stride,
                "infer_window_stride": window_stride,
                "num_keypoints": num_keypoints,
                "yolo_keypoints": yolo_num_keypoints,
                "body_length_px": per_source_body_len,
                "calibration_json": str(args.calibration_json) if args.calibration_json else None,
                "pipeline_version": YOLO_TEMPORAL_CLASSIFIER_PIPELINE,
                "yolo_temporal_classifier_version": YOLO_TEMPORAL_CLASSIFIER_VERSION,
                "behaviorscope_y_version": BEHAVIORSCOPE_Y_VERSION,
                "calibration_source": calib_source,
                "embedded_calibration_available": bool(embedded_calib_lookup),
                "embedded_calibration_num_videos": int(embedded_calib_meta.get("num_videos", 0) or 0),
                "used_calibration": bool(per_source_body_len is not None and explicit_body_length is None),
                "temporal_smoothing_window": int(args.temporal_smoothing_window),
                "bout_min_duration_frames": int(args.bout_min_duration_frames),
                "live_preview": bool(args.preview),
                "threshold_decoder_enabled": bool(
                    isinstance(threshold_decoder, dict) and threshold_decoder.get("enabled")
                ),
                "threshold_decoder": threshold_decoder,
                "temporal_splitter_enabled": bool(
                    isinstance(v25_splitter_config, dict) and v25_splitter_config.get("enabled")
                ),
                "temporal_splitter": v25_splitter_config,
                "temporal_splitter_applied_to": (
                    "smoothed_frames_csv_only"
                    if (isinstance(v25_splitter_config, dict)
                        and v25_splitter_config.get("enabled"))
                    else "not_applied"
                ),
                "temporal_splitter_manual_override": bool(
                    isinstance(v25_splitter_config, dict)
                    and v25_splitter_config.get("manual_override")
                ),
                # Legacy keys kept for tools that read old infer_meta.json files.
                "v25_splitter_enabled": bool(
                    isinstance(v25_splitter_config, dict) and v25_splitter_config.get("enabled")
                ),
                "v25_splitter": v25_splitter_config,
                "amp_enabled": _infer_amp_active,
                "amp_dtype_resolved": _infer_amp_dtype_resolved,
                "csv_flush_interval": int(args.csv_flush_interval),
                "classes": list(idx_to_class[i] for i in sorted(idx_to_class)),
            }
            meta_path = Path(str(out_csv)).with_suffix(".infer_meta.json")
            meta_path.write_text(json.dumps(infer_meta, indent=2), encoding="utf-8")
        except Exception as _me:
            print(f"[warn] could not write infer_meta.json: {_me}", flush=True)
        summaries.append({"source": str(src), **s})

    total_windows = sum(s["windows"] for s in summaries)
    total_elapsed = sum(s["elapsed_s"] for s in summaries)
    stopped_any = any(s.get("stopped_early") for s in summaries) or _STOP_REQUESTED
    tail = "  [stopped early]" if stopped_any else ""
    print(f"[infer] all done. sources={len(summaries)} total_windows={total_windows} elapsed={total_elapsed:.1f}s{tail}", flush=True)
    if stop_flag_path is not None and stop_flag_path.exists():
        try:
            stop_flag_path.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
