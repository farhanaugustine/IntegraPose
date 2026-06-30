"""
TandemYTC training entry point.

Trains the multi-animal four-stream behavior classifier on NPZ windows
produced by ``prepare_full_video_npz.py``.  Outputs best-checkpoint,
training log CSV, confusion matrices, and threshold calibration artifacts.
"""
from __future__ import annotations

import warnings
# Suppress pynvml deprecation warning that fires in each DataLoader worker
# process when torch.cuda is imported.  Root fix: pip uninstall pynvml &&
# pip install nvidia-ml-py.  This filter is belt-and-suspenders.
warnings.filterwarnings(
    "ignore",
    message=".*pynvml.*deprecated.*",
    category=FutureWarning,
)

import argparse
import csv
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from torch.amp import autocast, GradScaler
except ImportError:  # torch < 2.0
    from torch.cuda.amp import autocast, GradScaler  # type: ignore[no-redef]

# matplotlib is only needed for confusion-matrix PNGs and is imported lazily
# inside `save_confusion_matrix_png` so that headless setups without matplotlib
# can still train.

try:
    from data_y import (
        load_n_manifest,
        MultiAnimalSequenceDataset,
        collate_multi_animal,
        compute_class_weights,
        make_weighted_sampler,
        COMPATIBLE_SCHEMA_VERSIONS,
    )
    from model_y import MultiAnimalBehaviorSequenceClassifier, inspect_yolo_keypoint_count
    from utils.pose_features_y import REL_FEATURE_DIM
    from utils.temporal_splitter import (
        TemporalSplitterVideo as V25FitVideo,
        fit_temporal_splitter as fit_v25_splitter,
        make_disabled_config as make_v25_disabled_config,
        SPLITTER_VERSION as V25_SPLITTER_VERSION,
        TEMPORAL_SPLITTER_CONFIG_KEY,
        LEGACY_V25_SPLITTER_CONFIG_KEY,
        DEFAULT_TARGET_CLASS as V25_DEFAULT_TARGET_CLASS,
        DEFAULT_REPLACEMENT_CLASS as V25_DEFAULT_REPLACEMENT_CLASS,
        EPSILON_FRAME_F1 as V25_EPSILON_FRAME_F1,
        MIN_COVERED_FRAMES as V25_MIN_COVERED_FRAMES,
    )
except Exception:  # pragma: no cover
    from .data_y import (
        load_n_manifest,
        MultiAnimalSequenceDataset,
        collate_multi_animal,
        compute_class_weights,
        make_weighted_sampler,
        COMPATIBLE_SCHEMA_VERSIONS,
    )
    from .model_y import MultiAnimalBehaviorSequenceClassifier, inspect_yolo_keypoint_count
    from .utils.pose_features_y import REL_FEATURE_DIM
    from .utils.temporal_splitter import (
        TemporalSplitterVideo as V25FitVideo,
        fit_temporal_splitter as fit_v25_splitter,
        make_disabled_config as make_v25_disabled_config,
        SPLITTER_VERSION as V25_SPLITTER_VERSION,
        TEMPORAL_SPLITTER_CONFIG_KEY,
        LEGACY_V25_SPLITTER_CONFIG_KEY,
        DEFAULT_TARGET_CLASS as V25_DEFAULT_TARGET_CLASS,
        DEFAULT_REPLACEMENT_CLASS as V25_DEFAULT_REPLACEMENT_CLASS,
        EPSILON_FRAME_F1 as V25_EPSILON_FRAME_F1,
        MIN_COVERED_FRAMES as V25_MIN_COVERED_FRAMES,
    )


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


def parse_args():
    p = argparse.ArgumentParser(description="Train TandemYTC.")
    p.add_argument("--manifest_path", required=True)
    p.add_argument("--yolo_weights", required=True,
                   help="Path to the YOLO-pose checkpoint used to generate the training cache.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=3e-4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--scheduler", choices=["plateau", "cosine", "none"], default="plateau")
    # class weighting
    p.add_argument("--class_weighting", choices=["sqrt_inverse", "inverse", "none"], default="sqrt_inverse")
    p.add_argument("--class_weight_clamp", type=float, default=2.0)
    p.add_argument("--train_sampler", choices=["weighted", "random"], default="weighted")
    # model
    p.add_argument("--n_animals", type=int, default=2)
    p.add_argument(
        "--num_keypoints",
        type=int,
        default=0,
        help=(
            "Keypoints per animal. Usually inferred from the manifest. "
            "If the manifest lacks keypoints_per_animal, 0 means infer from "
            "--yolo_weights; falls back to 7 only if the YOLO checkpoint does "
            "not expose a pose keypoint shape."
        ),
    )
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_lstm_layers", type=int, default=1)
    p.add_argument("--bidirectional_lstm", action="store_true")
    p.add_argument(
        "--sequence_model",
        choices=[
            "tcn",
            "gated_attention_lstm",
            "attention_lstm",
            "lstm",
            "tcn_attention",
            "attention",
        ],
        default="tcn",
    )
    p.add_argument("--use_attention_pool", action="store_true")
    p.add_argument("--attention_heads", type=int, default=4)
    p.add_argument("--positional_encoding", default="none")
    # ablation flags
    p.add_argument("--disable_pose_self", action="store_true")
    p.add_argument("--disable_relations", action="store_true")
    p.add_argument("--relations_pose_only", action="store_true")
    p.add_argument("--pose_dropout_p_uniform", type=float, default=0.0,
                   help="R6b: per-window uniform-random pose dropout rate, applied per animal per frame independently.")
    # logging / diagnostics
    p.add_argument("--log_interval", type=int, default=0,
                   help="Print batch-level progress every N steps (0 disables).")
    p.add_argument("--per_class_metrics", action="store_true",
                   help="Print per-class precision/recall/F1 each epoch and store them in the CSV log.")
    p.add_argument("--confusion_matrix", action="store_true",
                   help="Print and save validation confusion matrices each epoch (text + PNG heatmaps in the run dir).")
    p.add_argument("--confusion_matrix_interval", type=int, default=1,
                   help="How often (in epochs) to compute the validation confusion matrix.")
    p.add_argument("--disable_threshold_decoder", action="store_true",
                   help="Do not fit the validation-calibrated per-class threshold decoder "
                        "after training. By default, train_y.py fits thresholds on the "
                        "validation split and saves them for automatic inference use.")
    p.add_argument("--background_class", default="auto",
                   help="Fallback/background class used when no behavior passes its "
                        "confidence threshold. Default 'auto' looks for common names "
                        "such as other/background/none/no_behavior.")
    p.add_argument("--threshold_grid_min", type=float, default=0.30,
                   help="Minimum confidence threshold considered during validation fitting.")
    p.add_argument("--threshold_grid_max", type=float, default=0.95,
                   help="Maximum confidence threshold considered during validation fitting.")
    p.add_argument("--threshold_grid_step", type=float, default=0.05,
                   help="Confidence threshold step used during validation fitting.")
    p.add_argument("--threshold_fit_rounds", type=int, default=3,
                   help="Coordinate-descent passes over behavior classes when fitting "
                        "validation thresholds.")
    p.add_argument("--save_decoder_json", action="store_true",
                   help="Also write decoder_config.json as a standalone sidecar. "
                        "By default the decoder is embedded in config.json and all "
                        "training checkpoints to reduce run-folder clutter.")
    # TandemYTC temporal bout splitter (auto-fit on val data).
    # The old --v25_* flags remain as hidden aliases for old scripts.
    p.add_argument("--disable_temporal_splitter", dest="disable_v25_splitter", action="store_true",
                   help="Disable train-time fitting of the validation-calibrated temporal bout splitter. "
                        "Per-class threshold decoder still runs; only the splitter is skipped.")
    p.add_argument("--temporal_splitter_target_class", dest="v25_splitter_target_class",
                   default="investigation", metavar="CLASS",
                   help="Class whose bouts the temporal splitter is allowed to split. "
                        "The splitter auto-disables if this class is not in the schema.")
    p.add_argument("--temporal_splitter_replacement_class", dest="v25_splitter_replacement_class",
                   default="other", metavar="CLASS",
                   help="Class that replaces split-out frames, typically the background class.")
    p.add_argument("--temporal_splitter_annot_root", dest="v25_splitter_annot_root",
                   default=None, metavar="DIR",
                   help="Optional directory of .annot files for splitter GT reconstruction. "
                        "When omitted, fitting falls back to window-label projection or disables "
                        "if neither source is feasible.")
    p.add_argument("--temporal_splitter_strict_annot", dest="v25_splitter_strict_annot",
                   action="store_true",
                   help="Strict annot mode for manuscript runs. When --temporal_splitter_annot_root "
                        "is provided, videos missing their .annot file are skipped instead of "
                        "back-filled with window-label projection.")
    p.add_argument("--save_temporal_splitter_json", dest="save_v25_splitter_json",
                   action="store_true",
                   help="Also write temporal_splitter.json as a standalone sidecar. "
                        "By default the splitter is embedded in config.json and checkpoints.")
    p.add_argument("--disable_v25_splitter", dest="disable_v25_splitter",
                   action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--v25_splitter_target_class", dest="v25_splitter_target_class",
                   default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.add_argument("--v25_splitter_replacement_class", dest="v25_splitter_replacement_class",
                   default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.add_argument("--v25_splitter_annot_root", dest="v25_splitter_annot_root",
                   default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.add_argument("--v25_splitter_strict_annot", dest="v25_splitter_strict_annot",
                   action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--save_v25_splitter_json", dest="save_v25_splitter_json",
                   action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--resume", action="store_true",
                   help="Resume from last_checkpoint.pt if it exists in the run directory.")
    p.add_argument("--skip_invalid_samples", action="store_true",
                   help="Skip invalid NPZ windows instead of failing fast. Use only for dataset triage.")
    # AMP / mixed-precision
    p.add_argument("--amp", action="store_true", default=False,
                   help="Enable automatic mixed precision during training (default off). "
                        "Recorded in config.json; inference AMP is a separate flag in infer_n.py.")
    p.add_argument("--amp_dtype", choices=["auto", "fp16", "bf16"], default="auto",
                   help="AMP dtype. 'auto' selects bf16 on Ampere+ hardware (no GradScaler), "
                        "fp16 with GradScaler elsewhere. Ignored when --amp is not set.")
    # Split selection (guards against held-out test sets entering training)
    p.add_argument("--train_splits", nargs="+", default=["train"],
                   help="Manifest split key(s) used for training. Default: train. "
                        "The full Stage-2 manifest has keys: train, val, test_1, test_2. "
                        "Do not include test_* here in publication runs.")
    p.add_argument("--val_splits", nargs="+", default=["val"],
                   help="Manifest split key(s) used for model selection and early stopping. "
                        "Default: val.")
    p.add_argument("--allow_test_split_training", action="store_true", default=False,
                   help="DEVELOPMENT ONLY. Allow a split whose name starts with 'test' "
                        "to be used for training or validation. Never pass this in a "
                        "publication-grade run — it breaks the held-out evaluation contract.")
    # bookkeeping
    p.add_argument("--project", default="behavior_lstm_runs")
    p.add_argument("--name", default="run")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _resolve_amp_dtype(requested: str) -> str:
    """Return the concrete AMP dtype string given the requested value.

    'auto' picks bf16 on Ampere+ hardware (GradScaler not needed),
    falling back to fp16 (GradScaler required) everywhere else.
    """
    if requested == "auto":
        return "bf16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "fp16"
    return requested


def per_class_prf1(preds: np.ndarray, labels: np.ndarray, num_classes: int):
    """Return per-class precision, recall, F1 arrays of shape (num_classes,)."""
    prec = np.zeros(num_classes, dtype=np.float64)
    rec = np.zeros(num_classes, dtype=np.float64)
    f1 = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        tp = float(((preds == c) & (labels == c)).sum())
        fp = float(((preds == c) & (labels != c)).sum())
        fn = float(((preds != c) & (labels == c)).sum())
        prec[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[c] = (2 * prec[c] * rec[c] / (prec[c] + rec[c])
                 if (prec[c] + rec[c]) > 0 else 0.0)
    return prec, rec, f1


def macro_f1(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    _, _, f1 = per_class_prf1(preds, labels, num_classes)
    return float(np.mean(f1))


def _normalize_class_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def resolve_background_index(class_to_idx: dict, requested: str = "auto"):
    """Return (index, name, reason) for the fallback/background class."""
    if not class_to_idx:
        return None, None, "class_to_idx is empty"
    by_norm = {_normalize_class_name(k): (int(v), str(k)) for k, v in class_to_idx.items()}
    if requested and str(requested).lower() != "auto":
        key = _normalize_class_name(requested)
        if key not in by_norm:
            return None, None, f"requested background_class={requested!r} was not found"
        idx, name = by_norm[key]
        return idx, name, "explicit"

    candidates = [
        "other", "background", "none", "nobehavior", "no_behavior",
        "no-behavior", "null", "idle", "unknown",
    ]
    for cand in candidates:
        key = _normalize_class_name(cand)
        if key in by_norm:
            idx, name = by_norm[key]
            return idx, name, f"auto matched {name!r}"
    return None, None, "no common fallback class name found"


def threshold_grid_values(lo: float, hi: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("--threshold_grid_step must be > 0")
    if hi < lo:
        raise ValueError("--threshold_grid_max must be >= --threshold_grid_min")
    n = int(np.floor((hi - lo) / step + 1e-9)) + 1
    vals = lo + np.arange(n, dtype=np.float64) * step
    if vals.size == 0 or vals[-1] < hi - 1e-9:
        vals = np.append(vals, hi)
    return np.clip(vals, 0.0, 1.0)


def apply_threshold_decoder_probs(probs: np.ndarray, thresholds_by_index, background_index: int) -> np.ndarray:
    """Decode softmax probabilities with per-class thresholds and background fallback."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim == 1:
        probs = probs[None, :]
    n_classes = probs.shape[1]
    thresholds = np.asarray(thresholds_by_index, dtype=np.float64)
    if thresholds.shape[0] != n_classes:
        raise ValueError(
            "threshold_decoder class-count mismatch: "
            f"got {thresholds.shape[0]} thresholds for {n_classes} classes"
        )
    behavior_indices = [i for i in range(n_classes) if i != int(background_index)]
    preds = np.full(probs.shape[0], int(background_index), dtype=np.int64)
    for row_i, row in enumerate(probs):
        passing = [i for i in behavior_indices if float(row[i]) >= float(thresholds[i])]
        if passing:
            best_local = int(np.argmax(row[passing]))
            preds[row_i] = int(passing[best_local])
    return preds


def fit_threshold_decoder(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    class_to_idx: dict,
    idx_to_class: dict,
    background_class: str = "auto",
    grid_min: float = 0.30,
    grid_max: float = 0.95,
    grid_step: float = 0.05,
    rounds: int = 3,
) -> dict:
    """Fit validation thresholds without hard-coding behavior names.

    The decoder is enabled only if it matches or improves validation macro-F1
    compared with raw softmax argmax, which protects users from a harmful
    post-processing default on datasets that do not need it.
    """
    val_probs = np.asarray(val_probs, dtype=np.float64)
    val_labels = np.asarray(val_labels, dtype=np.int64)
    num_classes = int(val_probs.shape[1]) if val_probs.ndim == 2 else len(class_to_idx)
    class_names = [str(idx_to_class.get(i, f"class_{i}")) for i in range(num_classes)]
    raw_preds = val_probs.argmax(axis=1).astype(np.int64) if val_probs.size else np.array([], dtype=np.int64)
    raw_macro = macro_f1(raw_preds, val_labels, num_classes) if val_labels.size else 0.0
    bg_idx, bg_name, bg_reason = resolve_background_index(class_to_idx, background_class)

    base = {
        "version": "behaviorscope-threshold-decoder-v1",
        "source": "validation_threshold_fit",
        "objective": "validation_window_macro_f1",
        "enabled": False,
        "fallback": "background",
        "background_class": bg_name,
        "background_index": bg_idx,
        "background_resolution": bg_reason,
        "classes": class_names,
        "raw_val_macro_f1": float(raw_macro),
        "decoded_val_macro_f1": float(raw_macro),
        "threshold_grid": {
            "min": float(grid_min),
            "max": float(grid_max),
            "step": float(grid_step),
            "rounds": int(rounds),
        },
        "thresholds": {name: 0.0 for name in class_names},
        "thresholds_by_index": [0.0 for _ in range(num_classes)],
        "per_class": {},
    }
    if val_probs.ndim != 2 or val_probs.shape[0] == 0 or val_labels.shape[0] == 0:
        base["disabled_reason"] = "validation probabilities were empty"
        return base
    if bg_idx is None or not (0 <= int(bg_idx) < num_classes):
        base["disabled_reason"] = bg_reason
        return base
    if num_classes <= 1:
        base["disabled_reason"] = "threshold decoding requires at least one behavior plus a fallback class"
        return base

    grid = threshold_grid_values(grid_min, grid_max, grid_step)
    behavior_indices = [i for i in range(num_classes) if i != int(bg_idx)]
    thresholds = np.zeros(num_classes, dtype=np.float64)
    thresholds[behavior_indices] = float(np.clip(0.50, grid.min(), grid.max()))
    best_preds = apply_threshold_decoder_probs(val_probs, thresholds, int(bg_idx))
    best_score = macro_f1(best_preds, val_labels, num_classes)

    for _ in range(max(1, int(rounds))):
        changed = False
        for c_idx in behavior_indices:
            current_t = float(thresholds[c_idx])
            local_best_t = current_t
            local_best_score = best_score
            local_best_preds = best_preds
            for t in grid:
                thresholds[c_idx] = float(t)
                preds = apply_threshold_decoder_probs(val_probs, thresholds, int(bg_idx))
                score = macro_f1(preds, val_labels, num_classes)
                if score > local_best_score + 1e-12:
                    local_best_score = score
                    local_best_t = float(t)
                    local_best_preds = preds
            thresholds[c_idx] = local_best_t
            if abs(local_best_t - current_t) > 1e-12:
                changed = True
            best_score = local_best_score
            best_preds = local_best_preds
        if not changed:
            break

    decoded_macro = macro_f1(best_preds, val_labels, num_classes)
    raw_p, raw_r, raw_f = per_class_prf1(raw_preds, val_labels, num_classes)
    dec_p, dec_r, dec_f = per_class_prf1(best_preds, val_labels, num_classes)
    thresholds_by_index = [float(x) for x in thresholds.tolist()]
    base["thresholds_by_index"] = thresholds_by_index
    base["thresholds"] = {
        class_names[i]: float(thresholds_by_index[i]) for i in range(num_classes)
    }
    base["decoded_val_macro_f1"] = float(decoded_macro)
    base["enabled"] = bool(decoded_macro >= raw_macro - 1e-12)
    if not base["enabled"]:
        base["disabled_reason"] = "validation macro-F1 did not improve over raw argmax"
    for i, name in enumerate(class_names):
        base["per_class"][name] = {
            "threshold": float(thresholds_by_index[i]),
            "raw_precision": float(raw_p[i]),
            "raw_recall": float(raw_r[i]),
            "raw_f1": float(raw_f[i]),
            "decoded_precision": float(dec_p[i]),
            "decoded_recall": float(dec_r[i]),
            "decoded_f1": float(dec_f[i]),
        }
    return base


def attach_threshold_decoder_to_checkpoints(run_dir: Path, config_dump: dict, decoder_config: dict):
    """Best-effort embedding of decoder metadata into existing checkpoints."""
    for name in ("best_model.pt", "best_model_macro_f1.pt", "last_checkpoint.pt"):
        path = run_dir / name
        if not path.is_file():
            continue
        try:
            ckpt = safe_torch_load(path, map_location="cpu")
            if not isinstance(ckpt, dict):
                continue
            ckpt["threshold_decoder"] = decoder_config
            cfg = ckpt.get("config")
            if isinstance(cfg, dict):
                cfg["threshold_decoder"] = decoder_config
                cfg["threshold_decoder_fit"] = config_dump.get("threshold_decoder_fit", {})
            else:
                ckpt["config"] = config_dump
            torch.save(ckpt, path)
        except Exception as exc:
            print(f"[warn] could not embed threshold decoder in {path.name}: {exc}")


def attach_temporal_splitter_to_checkpoints(run_dir: Path, config_dump: dict, splitter_config: dict):
    """Best-effort embedding of v3 temporal-splitter metadata into checkpoints."""
    for name in ("best_model.pt", "best_model_macro_f1.pt", "last_checkpoint.pt"):
        path = run_dir / name
        if not path.is_file():
            continue
        try:
            ckpt = safe_torch_load(path, map_location="cpu")
            if not isinstance(ckpt, dict):
                continue
            ckpt[TEMPORAL_SPLITTER_CONFIG_KEY] = splitter_config
            ckpt[LEGACY_V25_SPLITTER_CONFIG_KEY] = splitter_config
            cfg = ckpt.get("config")
            if isinstance(cfg, dict):
                cfg["pipeline_version"] = YOLO_TEMPORAL_CLASSIFIER_PIPELINE
                cfg["yolo_temporal_classifier_version"] = YOLO_TEMPORAL_CLASSIFIER_VERSION
                cfg["behaviorscope_y_version"] = BEHAVIORSCOPE_Y_VERSION
                cfg[TEMPORAL_SPLITTER_CONFIG_KEY] = splitter_config
                cfg[LEGACY_V25_SPLITTER_CONFIG_KEY] = splitter_config
            else:
                ckpt["config"] = config_dump
            torch.save(ckpt, path)
        except Exception as exc:
            print(f"[warn] could not embed temporal splitter in {path.name}: {exc}")


def _parse_bento_annot_for_fit(path: str, total_frames: int, fps: float,
                                class_to_idx: dict) -> np.ndarray:
    """Parse a BENTO .annot file into a per-frame class-id array.

    Returns int array [total_frames] with class IDs (or "other" / unmapped fallback)."""
    other_idx = class_to_idx.get("other", 0)
    out = np.full(total_frames, other_idx, dtype=np.int64)
    cur = None
    in_data = False
    try:
        with open(path, "r") as fh:
            for line in fh:
                s = line.strip()
                if s.startswith(">"):
                    cur = s[1:].strip()
                    in_data = False
                    continue
                if "Start" in s and "Stop" in s:
                    in_data = True
                    continue
                if in_data and cur and s:
                    parts = s.split()
                    if len(parts) >= 2:
                        cid = class_to_idx.get(cur, None)
                        if cid is None:
                            continue
                        try:
                            sf = int(round(float(parts[0]) * fps))
                            ef = int(round(float(parts[1]) * fps))
                        except ValueError:
                            continue
                        sf = max(sf, 0); ef = min(ef, total_frames - 1)
                        if ef >= sf:
                            out[sf:ef + 1] = cid
    except Exception as exc:
        print(f"[temporal splitter] could not parse annot {path}: {exc}")
    return out


def _build_v25_fit_videos(
    val_samples: list,
    val_probs: np.ndarray,
    decoder_config: dict,
    class_to_idx: dict,
    target_class: str,
    target_index: int,
    annot_root: Optional[str],
    fps: float = 30.0,
    min_segment_frames: int = 100,
    strict_annot: bool = False,
) -> Tuple[List[V25FitVideo], str, Dict[str, int]]:
    """Reconstruct per-source-video continuous predictions and GT for splitter fitting.

    KEY INVARIANT: bout adjacency is preserved. Each contiguous run of
    covered frames within a source video becomes its OWN V25FitVideo
    segment. Uncovered gaps between val windows are NOT collapsed —
    bouts on either side of a gap are correctly recognized as separate
    bouts in separate segments.

    GT provenance is tracked per source video. The returned `gt_source`
    string is honest about the mix:
      "annot_files"                  — annot_root provided AND every
                                        used video had its annot file
      "annot_files_partial"          — annot_root provided, some videos
                                        used annot, others fell back
                                        to window-label projection
      "annot_files_unavailable"      — annot_root provided but ZERO
                                        videos found their annot file
                                        (fit cannot proceed in this mode)
      "window_label_projection"      — annot_root not provided

    Returns:
        (list of V25FitVideo segments, gt_source string, audit_counts dict)
        audit_counts has keys:
            n_videos_total            (videos with any val coverage)
            n_videos_passing_min      (>= MIN_COVERED_FRAMES across all segments)
            n_segments_emitted        (final V25FitVideo count)
            n_annot_attempted         (videos we tried to load .annot for)
            n_annot_matched           (videos whose .annot loaded successfully)
            n_annot_missing           (videos with no .annot match)
    """
    audit = {
        "n_videos_total": 0,
        "n_videos_passing_min": 0,
        "n_videos_emitted": 0,
        "n_segments_emitted": 0,
        "n_videos_skipped_missing_annot": 0,
        "n_annot_attempted": 0,
        "n_annot_matched": 0,
        "n_annot_missing": 0,
    }

    if len(val_probs) != len(val_samples):
        print(f"[temporal splitter] WARNING: val_probs has {len(val_probs)} rows but "
              f"val_samples has {len(val_samples)} entries. Cannot align — splitter fit aborted.")
        return [], "unavailable", audit

    use_annot = annot_root is not None and Path(annot_root).is_dir()
    other_idx = class_to_idx.get("other", 0)

    decoder_enabled = bool(decoder_config and decoder_config.get("enabled"))
    if decoder_enabled:
        thresholds = np.asarray(decoder_config.get("thresholds_by_index", []), dtype=np.float64)
        bg_idx_dec = int(decoder_config.get("background_index", other_idx))
    else:
        thresholds = None
        bg_idx_dec = other_idx

    # Group val samples by source_video
    by_video: Dict[str, List[Tuple[int, object]]] = {}
    for i, sample in enumerate(val_samples):
        src = getattr(sample, "source_video", None)
        if not src:
            src = getattr(sample, "id", f"unknown_{i}")
        by_video.setdefault(src, []).append((i, sample))

    audit["n_videos_total"] = len(by_video)

    fit_videos: List[V25FitVideo] = []
    n_classes = val_probs.shape[1]

    for source_video, items in by_video.items():
        starts = [int(s.start_frame) for _, s in items]
        ends   = [int(s.end_frame)   for _, s in items]
        if not starts:
            continue
        global_start = min(starts)
        global_end   = max(ends)
        T = global_end - global_start + 1

        prob_sum = np.zeros((T, n_classes), dtype=np.float64)
        coverage = np.zeros(T, dtype=np.int32)
        for i, sample in items:
            sf = int(sample.start_frame) - global_start
            ef = int(sample.end_frame) - global_start
            if ef < sf or sf < 0 or ef >= T:
                continue
            prob_sum[sf:ef + 1] += val_probs[i]
            coverage[sf:ef + 1] += 1

        covered_mask = coverage > 0
        total_covered = int(covered_mask.sum())
        if total_covered < V25_MIN_COVERED_FRAMES:
            continue
        audit["n_videos_passing_min"] += 1

        # Per-frame averaged probs over the FULL span; uncovered frames
        # remain zero in `avg_probs` and the splitter will see
        # prob_target=0 there (irrelevant — segment iteration skips them).
        avg_probs = np.zeros((T, n_classes), dtype=np.float64)
        avg_probs[covered_mask] = prob_sum[covered_mask] / coverage[covered_mask, None]

        # Per-frame decoded predictions over the FULL span.
        # Uncovered frames default to background_index (NEVER class 0).
        pred_decoded = np.full(T, bg_idx_dec, dtype=np.int64)
        if decoder_enabled and thresholds is not None and len(thresholds) == n_classes:
            behavior_indices_for_decode = [j for j in range(n_classes) if j != bg_idx_dec]
            for t_idx in np.where(covered_mask)[0]:
                row = avg_probs[t_idx]
                passing = [j for j in behavior_indices_for_decode if row[j] >= thresholds[j]]
                if passing:
                    pred_decoded[t_idx] = int(passing[int(np.argmax(row[passing]))])
        else:
            pred_decoded[covered_mask] = avg_probs[covered_mask].argmax(axis=1).astype(np.int64)

        # GT per-frame over full span. Track annot match.
        annot_loaded = False
        if use_annot:
            audit["n_annot_attempted"] += 1
            pattern = str(Path(annot_root) / "**" / f"{source_video}_*.annot")
            matches = sorted(glob.glob(pattern, recursive=True))
            if matches:
                fps_values = [
                    float(getattr(sample, "fps", fps))
                    for _, sample in items
                    if float(getattr(sample, "fps", fps) or 0.0) > 1e-6
                ]
                video_fps = float(np.median(fps_values)) if fps_values else float(fps)
                if fps_values and max(fps_values) - min(fps_values) > 1e-3:
                    print(
                        f"[temporal splitter] WARNING: mixed FPS for {source_video}; "
                        f"using median fps={video_fps:.6g} from {len(fps_values)} windows"
                    )
                gt_full = _parse_bento_annot_for_fit(
                    matches[0], total_frames=global_end + 1, fps=video_fps,
                    class_to_idx=class_to_idx,
                )
                gt_local = gt_full[global_start:global_end + 1]
                annot_loaded = True
                audit["n_annot_matched"] += 1
            else:
                audit["n_annot_missing"] += 1
                if strict_annot:
                    # In strict mode, skip this video entirely rather than
                    # fall back to noisier projection.
                    audit["n_videos_skipped_missing_annot"] += 1
                    continue

        if not annot_loaded:
            # Window-label projection (used when annot_root not provided OR
            # when this specific video's annot is missing AND strict_annot
            # is False — i.e., partial mode allows projection per-video).
            gt_local = np.full(T, other_idx, dtype=np.int64)
            for i, sample in items:
                sf = int(sample.start_frame) - global_start
                ef = int(sample.end_frame) - global_start
                if ef < sf:
                    continue
                cid = class_to_idx.get(getattr(sample, "class_name", "other"), other_idx)
                gt_local[max(sf, 0):min(ef, T - 1) + 1] = cid

        # Emit ONE V25FitVideo per source video with full-span arrays +
        # valid_mask. Segmentation happens at apply / metric time inside
        # v25_splitter.py — never at storage time.
        fit_videos.append(V25FitVideo(
            source_video=source_video,
            pred_decoded=pred_decoded.copy(),
            prob_target=avg_probs[:, target_index].astype(np.float64).copy(),
            gt_frames=gt_local.astype(np.int64).copy(),
            valid_mask=covered_mask.copy(),
            global_start=int(global_start),
        ))
        audit["n_videos_emitted"] += 1
        audit["n_segments_emitted"] += 1

    # Provenance string from per-video annot tracking
    if not use_annot:
        gt_source = "window_label_projection"
    else:
        if audit["n_annot_matched"] == 0:
            print(f"[temporal splitter] WARNING: --temporal_splitter_annot_root provided "
                  f"but ZERO of {audit['n_annot_attempted']} videos resolved "
                  f"their .annot file. Splitter fit will be disabled.")
            return [], "annot_files_unavailable", audit
        elif strict_annot and audit["n_annot_missing"] > 0:
            # In strict mode, missing-annot videos were skipped above.
            # All emitted videos used annot. Provenance is clean.
            gt_source = "annot_files"
            print(f"[temporal splitter] strict_annot mode: skipped "
                  f"{audit['n_videos_skipped_missing_annot']} videos missing .annot; "
                  f"{audit['n_annot_matched']} videos used .annot for fit.")
        elif audit["n_annot_missing"] > 0:
            gt_source = "annot_files_partial"
            print(f"[temporal splitter] mixed GT: {audit['n_annot_matched']} videos used "
                  f".annot, {audit['n_annot_missing']} fell back to "
                  f"window-label projection (gt_source='annot_files_partial'). "
                  f"Pass --temporal_splitter_strict_annot to skip missing-annot videos "
                  f"and recover gt_source='annot_files'.")
        else:
            gt_source = "annot_files"

    return fit_videos, gt_source, audit


def load_training_calibration_metadata(manifest_path: Path | str) -> dict:
    """Embed compact body-length calibration metadata for portable inference."""
    path = Path(manifest_path).parent / "calibration.json"
    base = {
        "version": "behaviorscope-y-calibration-v1",
        "embedded": False,
        "source_path": str(path),
        "lookup": {},
    }
    if not path.is_file():
        base["reason"] = "calibration.json not found next to manifest"
        return base
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        base["reason"] = f"could not read calibration.json: {exc}"
        return base
    lookup = {}
    for entry in payload.get("per_video_calibration", []):
        vid = entry.get("video_id")
        bl = entry.get("body_length_px")
        if vid and bl is not None:
            lookup[str(vid)] = float(bl)
    base.update({
        "embedded": bool(lookup),
        "source_path": str(path),
        "source_version": payload.get("version"),
        "calibration_strategy": payload.get("calibration_strategy"),
        "body_length_fallback_px": payload.get("body_length_fallback_px"),
        "body_length_min_valid_frames": payload.get("body_length_min_valid_frames"),
        "num_videos": int(len(lookup)),
        "lookup": lookup,
    })
    if not lookup:
        base["reason"] = "no per_video_calibration body_length_px entries found"
    return base


def save_threshold_decoder_diagnostics(
    decoder_config: dict,
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    idx_to_class: dict,
    out_dir: Path,
) -> None:
    """Write validation threshold audit CSVs and best-effort plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    val_probs = np.asarray(val_probs, dtype=np.float64)
    val_labels = np.asarray(val_labels, dtype=np.int64)
    if val_probs.ndim != 2 or val_probs.shape[0] == 0 or val_labels.shape[0] == 0:
        return

    num_classes = int(val_probs.shape[1])
    class_names = [str(idx_to_class.get(i, f"class_{i}")) for i in range(num_classes)]
    bg_idx = decoder_config.get("background_index")
    thresholds = np.asarray(decoder_config.get("thresholds_by_index", []), dtype=np.float64)
    if bg_idx is None or thresholds.shape[0] != num_classes:
        return
    bg_idx = int(bg_idx)

    raw_preds = val_probs.argmax(axis=1).astype(np.int64)
    decoded_preds = apply_threshold_decoder_probs(val_probs, thresholds, bg_idx)
    raw_p, raw_r, raw_f = per_class_prf1(raw_preds, val_labels, num_classes)
    dec_p, dec_r, dec_f = per_class_prf1(decoded_preds, val_labels, num_classes)
    supports = np.bincount(val_labels, minlength=num_classes)

    raw_cm = confusion_matrix_np(raw_preds, val_labels, num_classes)
    decoded_cm = confusion_matrix_np(decoded_preds, val_labels, num_classes)
    np.savetxt(out_dir / "confusion_matrix_raw_argmax.csv", raw_cm, fmt="%d", delimiter=",")
    np.savetxt(out_dir / "confusion_matrix_threshold_decoded.csv", decoded_cm, fmt="%d", delimiter=",")
    with open(out_dir / "confusion_matrices_raw_vs_threshold.txt", "w", encoding="utf-8") as fh:
        fh.write("=== raw argmax ===\n")
        fh.write(format_confusion_matrix(raw_cm, idx_to_class))
        fh.write("\n\n=== threshold decoded ===\n")
        fh.write(format_confusion_matrix(decoded_cm, idx_to_class))
        fh.write("\n")

    summary_path = out_dir / "raw_vs_threshold_metrics.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "class_index", "class_name", "support",
            "selected_threshold", "raw_precision", "raw_recall", "raw_f1",
            "decoded_precision", "decoded_recall", "decoded_f1",
        ])
        writer.writeheader()
        for i, name in enumerate(class_names):
            writer.writerow({
                "class_index": i,
                "class_name": name,
                "support": int(supports[i]),
                "selected_threshold": float(thresholds[i]),
                "raw_precision": float(raw_p[i]),
                "raw_recall": float(raw_r[i]),
                "raw_f1": float(raw_f[i]),
                "decoded_precision": float(dec_p[i]),
                "decoded_recall": float(dec_r[i]),
                "decoded_f1": float(dec_f[i]),
            })

    grid_cfg = decoder_config.get("threshold_grid", {})
    grid = threshold_grid_values(
        float(grid_cfg.get("min", 0.30)),
        float(grid_cfg.get("max", 0.95)),
        float(grid_cfg.get("step", 0.05)),
    )
    behavior_indices = [i for i in range(num_classes) if i != bg_idx]
    rows = []
    for c_idx in behavior_indices:
        for t in grid:
            trial_thresholds = thresholds.copy()
            trial_thresholds[c_idx] = float(t)
            preds = apply_threshold_decoder_probs(val_probs, trial_thresholds, bg_idx)
            p, r, f = per_class_prf1(preds, val_labels, num_classes)
            rows.append({
                "class_index": c_idx,
                "class_name": class_names[c_idx],
                "threshold": float(t),
                "selected": bool(abs(float(t) - float(thresholds[c_idx])) < 1e-9),
                "precision": float(p[c_idx]),
                "recall": float(r[c_idx]),
                "f1": float(f[c_idx]),
                "macro_f1": float(np.mean(f)),
                "support": int(supports[c_idx]),
            })

    sweep_path = out_dir / "threshold_sweep.csv"
    with open(sweep_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "class_index", "class_name", "threshold", "selected",
            "precision", "recall", "f1", "macro_f1", "support",
        ])
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable, skipping threshold plots: {exc}")
        return

    try:
        save_confusion_matrix_png(
            raw_cm,
            idx_to_class,
            out_dir / "confusion_matrix_raw_argmax.png",
            title="Validation confusion matrix - raw argmax",
        )
        save_confusion_matrix_png(
            decoded_cm,
            idx_to_class,
            out_dir / "confusion_matrix_threshold_decoded.png",
            title="Validation confusion matrix - threshold decoded",
        )
    except Exception as exc:
        print(f"[warn] threshold confusion matrix PNG failed: {exc}")

    if not behavior_indices:
        return
    n_panels = len(behavior_indices)
    fig_h = max(3.2, 2.8 * n_panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(7.5, fig_h), sharex=True)
    if n_panels == 1:
        axes = [axes]
    for ax, c_idx in zip(axes, behavior_indices):
        class_rows = [r for r in rows if int(r["class_index"]) == c_idx]
        xs = [float(r["threshold"]) for r in class_rows]
        ax.plot(xs, [float(r["precision"]) for r in class_rows], label="precision")
        ax.plot(xs, [float(r["recall"]) for r in class_rows], label="recall")
        ax.plot(xs, [float(r["f1"]) for r in class_rows], label="F1")
        ax.plot(xs, [float(r["macro_f1"]) for r in class_rows], label="macro-F1", linestyle="--")
        ax.axvline(float(thresholds[c_idx]), color="black", linestyle=":", linewidth=1.5)
        ax.set_title(f"{class_names[c_idx]} threshold sweep")
        ax.set_ylabel("score")
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower left", ncol=4, fontsize=8)
    axes[-1].set_xlabel("confidence threshold")
    fig.tight_layout()
    fig.savefig(out_dir / "threshold_sweep.png", dpi=160)
    fig.savefig(out_dir / "threshold_sweep.pdf")
    plt.close(fig)


def confusion_matrix_np(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Return a [num_classes x num_classes] integer confusion matrix.

    Rows are true labels, columns are predictions (the standard scikit-learn
    convention).
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def format_confusion_matrix(cm: np.ndarray, idx_to_class) -> str:
    """Render the confusion matrix as an aligned text block (raw + normalized)."""
    n = cm.shape[0]
    names = [str(idx_to_class.get(i, i)) for i in range(n)]
    col_w = max(8, max(len(s) for s in names) + 2)
    row_w = max(10, max(len(s) for s in names) + 2)

    lines = []
    lines.append("Confusion matrix (rows=true, cols=pred) — counts:")
    header = " " * row_w + "".join(s.rjust(col_w) for s in names)
    lines.append(header)
    for i, s in enumerate(names):
        row = s.rjust(row_w) + "".join(f"{cm[i, j]:>{col_w}d}" for j in range(n))
        lines.append(row)

    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    norm = cm.astype(np.float64) / row_sums
    lines.append("")
    lines.append("Confusion matrix — row-normalized (recall):")
    lines.append(header)
    for i, s in enumerate(names):
        row = s.rjust(row_w) + "".join(f"{norm[i, j]:>{col_w}.3f}" for j in range(n))
        lines.append(row)
    return "\n".join(lines)


def save_confusion_matrix_png(cm: np.ndarray, idx_to_class, out_path: Path,
                              title: str = "Validation confusion matrix"):
    """Best-effort PNG of the row-normalized confusion matrix."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib unavailable, skipping confusion matrix PNG: {e}")
        return False
    n = cm.shape[0]
    names = [str(idx_to_class.get(i, i)) for i in range(n)]
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    norm = cm.astype(np.float64) / row_sums
    fig, ax = plt.subplots(figsize=(max(4, 0.7 * n + 2), max(3.5, 0.7 * n + 2)))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            txt = f"{norm[i, j]:.2f}\n({cm[i, j]})"
            color = "white" if norm[i, j] > 0.5 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def evaluate(model, loader, criterion, device, num_classes,
             return_preds: bool = False,
             return_probs: bool = False,
             amp_enabled: bool = False,
             amp_dtype_torch=None):
    model.eval()
    losses = []
    all_preds = []
    all_labels = []
    all_probs = []
    _eval_dtype = amp_dtype_torch if amp_dtype_torch is not None else torch.float16
    with torch.no_grad():
        for item in loader:
            if item is None:
                continue
            inputs, labels = item
            labels = labels.to(device, non_blocking=(device.type == "cuda"))
            with autocast("cuda", dtype=_eval_dtype, enabled=amp_enabled):
                logits = model(inputs)
                loss = criterion(logits, labels)
            losses.append(float(loss.item()))
            probs = torch.softmax(logits.detach().float(), dim=-1).cpu().numpy()
            preds = probs.argmax(axis=1).astype(np.int64)
            all_preds.append(preds)
            all_labels.append(labels.detach().cpu().numpy())
            if return_probs:
                all_probs.append(probs)
    if not losses:
        out = [float("inf"), 0.0, 0.0]
        if return_preds:
            out.extend([np.array([], dtype=np.int64), np.array([], dtype=np.int64)])
        if return_probs:
            out.append(np.empty((0, num_classes), dtype=np.float32))
        return tuple(out)
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc = float((preds == labels).mean())
    f1 = macro_f1(preds, labels, num_classes)
    out = [float(np.mean(losses)), acc, f1]
    if return_preds:
        out.extend([preds, labels])
    if return_probs:
        out.append(np.concatenate(all_probs, axis=0))
    return tuple(out)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    try:
        splits, class_to_idx, idx_to_class, meta = load_n_manifest(args.manifest_path)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise SystemExit(f"[manifest] ERROR: {exc}") from exc
    num_classes = len(class_to_idx)
    schema = meta.get("schema_version", "")
    if schema not in COMPATIBLE_SCHEMA_VERSIONS:
        raise SystemExit(
            f"Manifest schema_version={schema!r}; expected one of "
            f"{sorted(COMPATIBLE_SCHEMA_VERSIONS)!r}. Use a TandemYTC "
            "full-video manifest or a compatible BehaviorScope-Y full-video cache."
        )
    num_frames = int(meta.get("window_size", 32))
    window_stride = int(meta.get("window_stride", max(num_frames // 2, 1)))
    if num_frames <= 0:
        raise SystemExit(f"Manifest window_size={num_frames}; expected a positive integer.")
    if window_stride <= 0:
        raise SystemExit(f"Manifest window_stride={window_stride}; expected a positive integer.")
    yolo_num_keypoints = inspect_yolo_keypoint_count(args.yolo_weights)
    if "keypoints_per_animal" in meta:
        num_keypoints = int(meta["keypoints_per_animal"])
        if yolo_num_keypoints is not None and yolo_num_keypoints != num_keypoints:
            raise SystemExit(
                f"Manifest keypoints_per_animal={num_keypoints}, but YOLO "
                f"checkpoint emits {yolo_num_keypoints} keypoints. Train on "
                "NPZs generated with the same YOLO pose keypoint schema, or "
                "use the matching --yolo_weights checkpoint."
            )
    elif int(args.num_keypoints) > 0:
        num_keypoints = int(args.num_keypoints)
    elif yolo_num_keypoints is not None:
        num_keypoints = int(yolo_num_keypoints)
    else:
        num_keypoints = 7
        print(
            "[manifest] WARNING: keypoints_per_animal missing and YOLO "
            "keypoint count could not be inspected; falling back to K=7.",
            flush=True,
        )
    if num_keypoints <= 0:
        raise SystemExit(
            f"Resolved num_keypoints={num_keypoints}; expected a positive integer."
        )
    manifest_n_animals = int(meta.get("n_animals", args.n_animals))
    if manifest_n_animals != int(args.n_animals):
        raise SystemExit(
            f"Manifest n_animals={manifest_n_animals}, but --n_animals={args.n_animals}. "
            "Regenerate the dataset or train with the matching animal count."
        )

    # ---- Split guard -------------------------------------------------------
    # Any split name beginning with "test" must not enter the training or
    # validation loop. This is the publication contract: test sets are touched
    # only by the evaluation script, never by the trainer.
    _train_set = set(args.train_splits)
    _val_set   = set(args.val_splits)
    overlap = _train_set & _val_set
    if overlap:
        raise SystemExit(
            f"--train_splits and --val_splits overlap: {sorted(overlap)}. "
            "Validation data must not appear in the training set."
        )
    if not args.allow_test_split_training:
        _test_requested = [s for s in list(args.train_splits) + list(args.val_splits)
                           if s.startswith("test")]
        if _test_requested:
            raise SystemExit(
                f"Split(s) {_test_requested!r} begin with 'test' and cannot be used "
                "for training or validation. Pass --allow_test_split_training to "
                "override (development / debugging only — never in publication runs)."
            )
    _manifest_splits = set(splits.keys())
    _missing = [s for s in list(args.train_splits) + list(args.val_splits)
                if s not in _manifest_splits]
    if _missing:
        raise SystemExit(
            f"Requested split(s) {_missing!r} not found in manifest. "
            f"Available manifest splits: {sorted(_manifest_splits)!r}."
        )
    train_samples = [s for key in args.train_splits for s in splits.get(key, [])]
    val_samples   = [s for key in args.val_splits   for s in splits.get(key, [])]
    if not train_samples:
        raise SystemExit(f"No training samples in split(s) {args.train_splits!r}.")
    _unused = sorted(_manifest_splits - _train_set - _val_set)
    if _unused:
        print(f"[manifest] splits not used for training: {_unused} "
              f"(held-out splits remain untouched — this is expected)")
    # ---- end split guard ---------------------------------------------------
    print(
        f"[manifest] schema={schema}  num_frames={num_frames}  "
        f"stride={window_stride}  K={num_keypoints}"
    )
    print(
        f"[manifest] train={len(train_samples)} (splits={args.train_splits})  "
        f"val={len(val_samples)} (splits={args.val_splits})  "
        f"classes={class_to_idx}"
    )
    train_ds = MultiAnimalSequenceDataset(
        samples=train_samples,
        num_frames=num_frames,
        n_animals=args.n_animals,
        num_keypoints=num_keypoints,
        rel_feature_dim=REL_FEATURE_DIM,
        pose_dropout_p_uniform=float(args.pose_dropout_p_uniform),
        strict_schema=True,
        skip_invalid=bool(args.skip_invalid_samples),
        rng_seed=int(args.seed),
    )
    val_ds = MultiAnimalSequenceDataset(
        samples=val_samples,
        num_frames=num_frames,
        n_animals=args.n_animals,
        num_keypoints=num_keypoints,
        rel_feature_dim=REL_FEATURE_DIM,
        pose_dropout_p_uniform=0.0,
        strict_schema=True,
        skip_invalid=bool(args.skip_invalid_samples),
        rng_seed=int(args.seed) + 1,
    )

    # Class weights for the loss (and optional weighted sampler)
    class_weights = compute_class_weights(
        train_samples,
        num_classes=num_classes,
        strategy=args.class_weighting,
        clamp_max=float(args.class_weight_clamp) if args.class_weight_clamp > 0 else None,
    ).to(device)
    print("[class weights]", {idx_to_class[i]: round(float(w), 3) for i, w in enumerate(class_weights.cpu().tolist())})

    sampler = None
    shuffle = True
    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(args.seed))
    if args.train_sampler == "weighted":
        sampler = make_weighted_sampler(train_samples, class_weights.cpu(), generator=loader_generator)
        shuffle = False  # WeightedRandomSampler controls order

    # DataLoader perf flags: persistent_workers keeps NPZ-decompression worker
    # processes alive across epochs (cuts startup cost on each epoch);
    # prefetch_factor=2 (PyTorch default) is plenty for the YOLO trunk's
    # consumption rate. pin_memory + non_blocking transfers in the model
    # forward overlap CPU->GPU copies with compute.
    _persist = int(args.num_workers) > 0
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=shuffle, sampler=sampler,
        num_workers=args.num_workers, collate_fn=collate_multi_animal,
        pin_memory=(device.type == "cuda"),
        persistent_workers=_persist,
        prefetch_factor=2 if _persist else None,
        generator=loader_generator if sampler is None else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_multi_animal,
        pin_memory=(device.type == "cuda"),
        persistent_workers=_persist,
        prefetch_factor=2 if _persist else None,
    )

    model = MultiAnimalBehaviorSequenceClassifier(
        num_classes=num_classes,
        n_animals=args.n_animals,
        num_keypoints=num_keypoints,
        rel_feature_dim=REL_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        sequence_model=args.sequence_model,
        num_lstm_layers=args.num_lstm_layers,
        bidirectional_lstm=args.bidirectional_lstm,
        use_attention_pool=args.use_attention_pool,
        attention_heads=args.attention_heads,
        positional_encoding=args.positional_encoding,
        disable_pose_self=args.disable_pose_self,
        disable_relations=args.disable_relations,
        relations_pose_only=args.relations_pose_only,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[model] total params={n_params/1e6:.2f}M  "
        f"trainable={n_trainable/1e6:.2f}M  "
        f"combined_dim={model.combined_dim}"
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    calibration_metadata = load_training_calibration_metadata(args.manifest_path)
    if calibration_metadata.get("embedded"):
        print(
            f"[calibration] embedded body_length_px lookup for "
            f"{calibration_metadata.get('num_videos', 0)} videos",
            flush=True,
        )
    else:
        print(
            f"[calibration] no embedded calibration lookup "
            f"({calibration_metadata.get('reason', 'unknown')})",
            flush=True,
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # AMP / mixed-precision setup.
    # Computed before config_dump so the resolved dtype is recorded in config.json.
    # Training and inference AMP are independent choices (see infer_n.py --amp).
    _amp_active = args.amp and (device.type == "cuda")
    if args.amp and not _amp_active:
        print("[warn] --amp requested but device is not CUDA; AMP disabled.", flush=True)
    amp_dtype_resolved = _resolve_amp_dtype(args.amp_dtype) if _amp_active else None
    _amp_dtype_torch = (torch.bfloat16 if amp_dtype_resolved == "bf16" else torch.float16) if _amp_active else None
    # GradScaler is only needed for fp16; bf16 has fp32 dynamic range.
    scaler = GradScaler("cuda", enabled=(_amp_active and amp_dtype_resolved == "fp16"))
    if _amp_active:
        print(f"[amp] enabled  dtype_requested={args.amp_dtype}  dtype_resolved={amp_dtype_resolved}", flush=True)

    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
    else:
        scheduler = None

    run_dir = Path(args.project) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    stop_flag_path = run_dir / "stop_training.flag"
    if stop_flag_path.exists():
        try:
            stop_flag_path.unlink()
        except Exception:
            pass

    config_dump = {
        "pipeline_version": YOLO_TEMPORAL_CLASSIFIER_PIPELINE,
        "yolo_temporal_classifier_version": YOLO_TEMPORAL_CLASSIFIER_VERSION,
        "behaviorscope_y_version": BEHAVIORSCOPE_Y_VERSION,
        "manifest_path": str(args.manifest_path),
        "feature_source": "pose_social",
        "yolo_weights": str(args.yolo_weights),
        "n_animals": args.n_animals,
        "num_keypoints": num_keypoints,
        "num_frames": num_frames,
        "window_stride": window_stride,
        "rel_feature_dim": REL_FEATURE_DIM,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "sequence_model": args.sequence_model,
        "num_lstm_layers": args.num_lstm_layers,
        "bidirectional_lstm": args.bidirectional_lstm,
        "use_attention_pool": args.use_attention_pool,
        "attention_heads": args.attention_heads,
        "positional_encoding": args.positional_encoding,
        "class_weighting": args.class_weighting,
        "class_weight_clamp": args.class_weight_clamp,
        "train_sampler": args.train_sampler,
        "ablation_flags": {
            "disable_pose_self": args.disable_pose_self,
            "disable_relations": args.disable_relations,
            "relations_pose_only": args.relations_pose_only,
            "pose_dropout_p_uniform": args.pose_dropout_p_uniform,
        },
        "calibration": calibration_metadata,
        "diagnostics": {
            "per_class_metrics": args.per_class_metrics,
            "confusion_matrix": args.confusion_matrix,
            "confusion_matrix_interval": args.confusion_matrix_interval,
            "log_interval": args.log_interval,
        },
        "threshold_decoder_fit": {
            "enabled": not bool(args.disable_threshold_decoder),
            "background_class": str(args.background_class),
            "grid_min": float(args.threshold_grid_min),
            "grid_max": float(args.threshold_grid_max),
            "grid_step": float(args.threshold_grid_step),
            "rounds": int(args.threshold_fit_rounds),
            "save_decoder_json": bool(args.save_decoder_json),
            "selection": "enabled only when validation macro-F1 is not worse than raw argmax",
        },
        "amp": {
            "amp_enabled": _amp_active,
            "amp_dtype_requested": args.amp_dtype if _amp_active else None,
            "amp_dtype_resolved": amp_dtype_resolved,
        },
        "split_selection": {
            "train_splits": list(args.train_splits),
            "val_splits": list(args.val_splits),
            "allow_test_split_training": args.allow_test_split_training,
        },
        "class_to_idx": class_to_idx,
        "schema_version": schema,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_dump, f, indent=2)

    # ---- hardware stats helper ----
    _hw_device_str = str(getattr(args, "device", "cpu"))

    def _collect_hw_stats() -> dict:
        """Collect CPU/RAM/GPU stats at end of epoch. Gracefully no-ops on missing libs."""
        hw: dict = {
            "cpu_pct": float("nan"), "ram_gb_used": float("nan"),
            "gpu_util_pct": float("nan"), "vram_gb_used": float("nan"),
            "gpu_temp_c": float("nan"), "gpu_power_w": float("nan"),
        }
        try:
            import psutil
            hw["cpu_pct"] = float(psutil.cpu_percent(interval=None))
            vm = psutil.virtual_memory()
            hw["ram_gb_used"] = round(vm.used / 1e9, 2)
        except Exception:
            pass
        if "cuda" in _hw_device_str:
            try:
                import pynvml  # nvidia-ml-py
                dev_idx = 0
                if ":" in _hw_device_str:
                    try:
                        dev_idx = int(_hw_device_str.split(":")[-1])
                    except ValueError:
                        pass
                pynvml.nvmlInit()
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(dev_idx)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem  = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    hw["gpu_util_pct"]  = float(util.gpu)
                    hw["vram_gb_used"]  = round(mem.used / 1e9, 2)
                    hw["gpu_temp_c"]    = float(temp)
                    try:
                        hw["gpu_power_w"] = round(
                            pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0, 1
                        )
                    except Exception:
                        pass
                finally:
                    pynvml.nvmlShutdown()
            except Exception:
                pass
        return hw

    # Prime psutil so cpu_percent(interval=None) returns a real value from epoch 1
    try:
        import psutil as _psutil_prime
        _psutil_prime.cpu_percent(interval=None)
    except Exception:
        pass

    # ---- training log files ----
    history_csv_path = run_dir / "training_log.csv"
    _HW_COLS = ["cpu_pct", "ram_gb_used", "gpu_util_pct", "vram_gb_used", "gpu_temp_c", "gpu_power_w"]
    csv_header = [
        "epoch", "train_loss", "train_acc",
        "val_loss", "val_acc", "val_macroF1",
        "elapsed_s", "lr",
    ] + _HW_COLS
    if args.per_class_metrics:
        for c_idx in range(num_classes):
            cname = str(idx_to_class.get(c_idx, c_idx))
            csv_header += [f"precision_{cname}", f"recall_{cname}", f"f1_{cname}"]

    cm_text_path = run_dir / "confusion_matrices.txt"
    csv_mode = "a" if (args.resume and history_csv_path.exists()) else "w"
    if csv_mode == "w":
        with open(history_csv_path, "w", newline="") as cf:
            csv.writer(cf).writerow(csv_header)

    # ---- optional resume ----
    best_val_loss = float("inf")
    best_val_f1 = -1.0
    epochs_since_improvement = 0
    start_epoch = 1
    history = []

    last_ckpt_path = run_dir / "last_checkpoint.pt"
    if args.resume and last_ckpt_path.exists():
        try:
            ckpt = safe_torch_load(last_ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optim_state"])
            if scheduler is not None and ckpt.get("scheduler_state") is not None:
                try:
                    scheduler.load_state_dict(ckpt["scheduler_state"])
                except Exception as e:
                    print(f"[resume] could not restore scheduler state: {e}")
            best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
            best_val_f1 = float(ckpt.get("best_val_f1", best_val_f1))
            epochs_since_improvement = int(ckpt.get("epochs_since_improvement", 0))
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            history = list(ckpt.get("history", []))
            print(f"[resume] loaded {last_ckpt_path}; starting at epoch {start_epoch}")
        except Exception as e:
            print(f"[resume] failed to load {last_ckpt_path}: {e}; starting fresh.")
            start_epoch = 1
            history = []

    if start_epoch > args.epochs:
        print(f"[resume] start_epoch={start_epoch} > epochs={args.epochs}; nothing to do.")

    for epoch in range(start_epoch, args.epochs + 1):
        if stop_flag_path.exists():
            print(f"[stop] {stop_flag_path} detected; halting before epoch {epoch}.")
            break

        model.train()
        t0 = time.time()
        epoch_losses = []
        epoch_correct = 0
        epoch_seen = 0
        n_batches = 0
        for item in train_loader:
            if item is None:
                continue
            inputs, labels = item
            labels = labels.to(device, non_blocking=(device.type == "cuda"))
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=_amp_dtype_torch or torch.float16, enabled=_amp_active):
                logits = model(inputs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.item()))
            batch_correct = int((logits.argmax(dim=-1) == labels).sum().item())
            batch_size_eff = int(labels.shape[0])
            epoch_correct += batch_correct
            epoch_seen += batch_size_eff
            n_batches += 1
            if args.log_interval and (n_batches % args.log_interval == 0):
                print(f"  [epoch {epoch} batch {n_batches}] loss={float(loss.item()):.4f} "
                      f"acc={batch_correct/max(batch_size_eff,1):.3f}")
            if stop_flag_path.exists():
                print(f"[stop] {stop_flag_path} detected mid-epoch; saving and exiting.")
                break

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        train_acc = float(epoch_correct) / max(epoch_seen, 1)

        run_cm_this_epoch = (
            args.confusion_matrix
            and args.confusion_matrix_interval > 0
            and (epoch % args.confusion_matrix_interval == 0)
        )
        if args.per_class_metrics or run_cm_this_epoch:
            val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(
                model, val_loader, criterion, device, num_classes, return_preds=True,
                amp_enabled=_amp_active, amp_dtype_torch=_amp_dtype_torch,
            )
        else:
            val_loss, val_acc, val_f1 = evaluate(
                model, val_loader, criterion, device, num_classes,
                amp_enabled=_amp_active, amp_dtype_torch=_amp_dtype_torch,
            )
            val_preds, val_labels = None, None

        elapsed = time.time() - t0
        lrs = [g["lr"] for g in optimizer.param_groups]
        lr = lrs[-1] if lrs else float("nan")

        log_line = (f"epoch {epoch:3d}  train_loss {train_loss:.4f} train_acc {train_acc:.3f}  "
                    f"val_loss {val_loss:.4f} val_acc {val_acc:.3f} val_macroF1 {val_f1:.3f}  "
                    f"({elapsed:.1f}s)")
        print(log_line)

        hw = _collect_hw_stats()
        epoch_record = {
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "val_macroF1": val_f1,
            "elapsed_s": elapsed,
            "lr": lr,
            **hw,
        }

        per_class_row = []
        if args.per_class_metrics and val_preds is not None and len(val_preds) > 0:
            prec, rec, f1 = per_class_prf1(val_preds, val_labels, num_classes)
            print("  per-class:")
            for c_idx in range(num_classes):
                cname = str(idx_to_class.get(c_idx, c_idx))
                print(f"    {cname:>14s}  P={prec[c_idx]:.3f}  R={rec[c_idx]:.3f}  F1={f1[c_idx]:.3f}")
                epoch_record[f"precision_{cname}"] = float(prec[c_idx])
                epoch_record[f"recall_{cname}"] = float(rec[c_idx])
                epoch_record[f"f1_{cname}"] = float(f1[c_idx])
                per_class_row += [float(prec[c_idx]), float(rec[c_idx]), float(f1[c_idx])]
        elif args.per_class_metrics:
            for c_idx in range(num_classes):
                cname = str(idx_to_class.get(c_idx, c_idx))
                epoch_record[f"precision_{cname}"] = 0.0
                epoch_record[f"recall_{cname}"] = 0.0
                epoch_record[f"f1_{cname}"] = 0.0
                per_class_row += [0.0, 0.0, 0.0]

        if run_cm_this_epoch and val_preds is not None and len(val_preds) > 0:
            cm = confusion_matrix_np(val_preds, val_labels, num_classes)
            cm_text = format_confusion_matrix(cm, idx_to_class)
            print(cm_text)
            with open(cm_text_path, "a") as f:
                f.write(f"\n=== epoch {epoch} ===\n")
                f.write(cm_text + "\n")
            png_path = run_dir / "confusion_matrices" / f"epoch_{epoch:04d}.png"
            try:
                save_confusion_matrix_png(
                    cm, idx_to_class, png_path,
                    title=f"Validation confusion matrix - epoch {epoch}",
                )
            except Exception as e:
                print(f"[warn] confusion matrix PNG failed: {e}")

        history.append(epoch_record)

        with open(history_csv_path, "a", newline="") as cf:
            row = [
                epoch, train_loss, train_acc,
                val_loss, val_acc, val_f1,
                elapsed, lr,
                hw["cpu_pct"], hw["ram_gb_used"],
                hw["gpu_util_pct"], hw["vram_gb_used"],
                hw["gpu_temp_c"], hw["gpu_power_w"],
            ]
            if args.per_class_metrics:
                row += per_class_row
            csv.writer(cf).writerow(row)

        improved = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "config": config_dump,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "best_val_f1": best_val_f1,
                "selection_metric": "val_loss",
                "class_to_idx": class_to_idx,
            }, run_dir / "best_model.pt")
            if val_preds is not None and len(val_preds) > 0:
                cm_best = confusion_matrix_np(val_preds, val_labels, num_classes)
                try:
                    save_confusion_matrix_png(
                        cm_best, idx_to_class,
                        run_dir / "best_confusion_matrix.png",
                        title=f"Best-val confusion matrix (epoch {epoch})",
                    )
                except Exception:
                    pass
            improved = True
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "model_state": model.state_dict(),
                "config": config_dump,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "best_val_f1": best_val_f1,
                "selection_metric": "val_macroF1",
                "class_to_idx": class_to_idx,
            }, run_dir / "best_model_macro_f1.pt")
            improved = True

        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        try:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "scheduler_state": (scheduler.state_dict() if scheduler is not None else None),
                "best_val_loss": best_val_loss,
                "best_val_f1": best_val_f1,
                "epochs_since_improvement": epochs_since_improvement,
                "history": history,
                "config": config_dump,
                "class_to_idx": class_to_idx,
            }, last_ckpt_path)
        except Exception as e:
            print(f"[warn] failed to save last_checkpoint.pt: {e}")

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if improved:
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= args.patience:
                print(f"[early stop] no improvement for {args.patience} epochs.")
                break

        if stop_flag_path.exists():
            print(f"[stop] {stop_flag_path} detected after epoch {epoch}; exiting.")
            break

    decoder_config = None
    val_probs_fit = None
    if args.disable_threshold_decoder:
        decoder_config = {
            "version": "behaviorscope-threshold-decoder-v1",
            "enabled": False,
            "disabled_reason": "disabled by --disable_threshold_decoder",
            "classes": [str(idx_to_class.get(i, f"class_{i}")) for i in range(num_classes)],
        }
    else:
        selected_ckpt = run_dir / "best_model_macro_f1.pt"
        if not selected_ckpt.is_file():
            selected_ckpt = run_dir / "best_model.pt"
        if selected_ckpt.is_file():
            try:
                ckpt = safe_torch_load(selected_ckpt, map_location=device)
                state = ckpt.get("model_state") if isinstance(ckpt, dict) else None
                if state is not None:
                    model.load_state_dict(state)
                    print(f"[threshold decoder] fitting on validation split using {selected_ckpt.name}")
            except Exception as exc:
                print(f"[warn] could not reload best checkpoint for threshold fitting: {exc}")
        else:
            print("[threshold decoder] no best checkpoint found; fitting with current model state")

        try:
            _, _, _, val_preds_raw, val_labels_fit, val_probs_fit = evaluate(
                model, val_loader, criterion, device, num_classes,
                return_preds=True, return_probs=True,
                amp_enabled=_amp_active, amp_dtype_torch=_amp_dtype_torch,
            )
            decoder_config = fit_threshold_decoder(
                val_probs=val_probs_fit,
                val_labels=val_labels_fit,
                class_to_idx=class_to_idx,
                idx_to_class=idx_to_class,
                background_class=str(args.background_class),
                grid_min=float(args.threshold_grid_min),
                grid_max=float(args.threshold_grid_max),
                grid_step=float(args.threshold_grid_step),
                rounds=int(args.threshold_fit_rounds),
            )
            save_threshold_decoder_diagnostics(
                decoder_config,
                val_probs_fit,
                val_labels_fit,
                idx_to_class,
                run_dir / "threshold_diagnostics",
            )
            if decoder_config.get("enabled"):
                bg_idx = decoder_config.get("background_index")
                thresholds = decoder_config.get("thresholds_by_index", [])
                shown = {
                    str(idx_to_class.get(i, i)): round(float(thresholds[i]), 3)
                    for i in range(min(len(thresholds), num_classes))
                    if i != bg_idx
                }
                print(
                    "[threshold decoder] enabled "
                    f"raw_macroF1={decoder_config.get('raw_val_macro_f1', 0.0):.4f} "
                    f"decoded_macroF1={decoder_config.get('decoded_val_macro_f1', 0.0):.4f} "
                    f"fallback={decoder_config.get('background_class')!r} "
                    f"thresholds={shown}"
                )
            else:
                print(
                    "[threshold decoder] disabled "
                    f"reason={decoder_config.get('disabled_reason', 'unknown')} "
                    f"raw_macroF1={decoder_config.get('raw_val_macro_f1', 0.0):.4f} "
                    f"decoded_macroF1={decoder_config.get('decoded_val_macro_f1', 0.0):.4f}"
                )
        except Exception as exc:
            decoder_config = {
                "version": "behaviorscope-threshold-decoder-v1",
                "enabled": False,
                "disabled_reason": f"threshold fitting failed: {exc}",
                "classes": [str(idx_to_class.get(i, f"class_{i}")) for i in range(num_classes)],
            }
            print(f"[warn] threshold decoder fitting failed: {exc}")

    if decoder_config is not None:
        config_dump["threshold_decoder"] = decoder_config
        try:
            if args.save_decoder_json:
                decoder_path = run_dir / "decoder_config.json"
                with open(decoder_path, "w") as f:
                    json.dump(decoder_config, f, indent=2)
                print(f"[threshold decoder] saved {decoder_path}")
            with open(run_dir / "config.json", "w") as f:
                json.dump(config_dump, f, indent=2)
            attach_threshold_decoder_to_checkpoints(run_dir, config_dump, decoder_config)
            print("[threshold decoder] embedded in config.json and checkpoints")
        except Exception as exc:
            print(f"[warn] could not save threshold decoder metadata: {exc}")

    # ---- TandemYTC temporal splitter (auto-fit on val) ----
    splitter_config = None
    target_class = str(args.v25_splitter_target_class)
    replacement_class = str(args.v25_splitter_replacement_class)
    target_index = class_to_idx.get(target_class)
    replacement_index = class_to_idx.get(replacement_class)

    # Auto-disable rule: clip-folder annotation mode means every NPZ window
    # is a pre-segmented behavior example (a "clip"), not a frame range
    # carved out of a continuous video with natural bout boundaries.
    # Fitting a bout splitter on clip-folder data is meaningless because
    # each clip *is* a bout — there are no merged-bout regions to split.
    # Honor an explicit override via `dense_bout_annotations: true` in
    # manifest metadata for users who genuinely have dense annotations
    # per clip and want the splitter anyway.
    annotation_mode = str(meta.get("annotation_mode", "video_with_bouts")).lower()
    has_dense_bouts = bool(meta.get("dense_bout_annotations", False))

    if args.disable_v25_splitter:
        splitter_config = make_v25_disabled_config(
            "disabled by --disable_temporal_splitter",
            target_class=target_class, target_index=target_index,
            replacement_class=replacement_class, replacement_index=replacement_index,
            gt_source="unavailable",
        )
    elif annotation_mode == "clip_folder" and not has_dense_bouts:
        splitter_config = make_v25_disabled_config(
            f"manifest annotation_mode={annotation_mode!r} (clip-folder) and "
            "dense_bout_annotations is not set — every clip is a pre-segmented "
            "behavior example, not a continuous-video bout sequence. The "
            "U-shape bout splitter has no merged bouts to operate on. "
            "Set dense_bout_annotations: true in manifest metadata if your "
            "clip-folder dataset actually has dense bout-edge GT.",
            target_class=target_class, target_index=target_index,
            replacement_class=replacement_class, replacement_index=replacement_index,
            gt_source="unavailable",
        )
    elif target_index is None or replacement_index is None:
        splitter_config = make_v25_disabled_config(
            f"target_class={target_class!r} or replacement_class={replacement_class!r} "
            f"not in schema (class_to_idx={list(class_to_idx)})",
            target_class=target_class, target_index=target_index,
            replacement_class=replacement_class, replacement_index=replacement_index,
            gt_source="unavailable",
        )
    else:
        try:
            # Reuse the val_probs/val_labels collected for decoder fit; if those
            # didn't run (decoder disabled), do a fresh evaluate.
            val_probs_for_splitter = val_probs_fit
            if val_probs_for_splitter is None:
                _, _, _, _vps_preds, _vps_labels, val_probs_for_splitter = evaluate(
                    model, val_loader, criterion, device, num_classes,
                    return_preds=True, return_probs=True,
                    amp_enabled=_amp_active, amp_dtype_torch=_amp_dtype_torch,
                )
            print(f"[temporal splitter] fitting on val with target={target_class!r} "
                  f"(idx={target_index}) replacement={replacement_class!r} "
                  f"(idx={replacement_index})")
            # Derive fps from val samples so annot-to-frame conversion is
            # correct for non-30-fps datasets.  A single unique fps value is
            # the common case.  Warn on mixed fps.
            _fps_vals = sorted({
                float(s.fps) for s in val_samples
                if hasattr(s, "fps") and float(s.fps) > 0
            })
            if not _fps_vals:
                _annot_fps = 30.0
                print("[temporal splitter] WARNING: no fps found on val samples; "
                      "defaulting to 30.0 fps for annot-to-frame conversion.")
            elif len(_fps_vals) == 1:
                _annot_fps = _fps_vals[0]
            else:
                import statistics as _stats
                _annot_fps = _stats.median(_fps_vals)
                print(
                    f"[temporal splitter] WARNING: val samples have mixed fps "
                    f"values {_fps_vals}; using median {_annot_fps:.1f} fps for "
                    f"annot-to-frame conversion. Check your manifest."
                )
            fit_videos, gt_source, fit_audit = _build_v25_fit_videos(
                val_samples=val_samples,
                val_probs=val_probs_for_splitter,
                decoder_config=decoder_config,
                class_to_idx=class_to_idx,
                target_class=target_class,
                target_index=target_index,
                annot_root=args.v25_splitter_annot_root,
                fps=_annot_fps,
                strict_annot=bool(args.v25_splitter_strict_annot),
            )
            print(
                f"[temporal splitter] gt_source={gt_source!r}  "
                f"videos_total={fit_audit.get('n_videos_total', 0)}  "
                f"videos_passing_min_coverage={fit_audit.get('n_videos_passing_min', 0)}  "
                f"contiguous_segments_emitted={fit_audit.get('n_segments_emitted', fit_audit.get('n_videos_emitted', 0))}  "
                f"annot_matched={fit_audit.get('n_annot_matched', 0)}/"
                f"{fit_audit.get('n_annot_attempted', 0)}"
            )
            # Behavior-class indices for frame F1 macro
            behavior_indices = [
                idx for cls, idx in class_to_idx.items() if cls != replacement_class
            ]
            splitter_config = fit_v25_splitter(
                videos=fit_videos,
                target_class=target_class,
                target_index=int(target_index),
                replacement_class=replacement_class,
                replacement_index=int(replacement_index),
                behavior_indices=behavior_indices,
                gt_source=gt_source,
            )
            # Embed audit counts so reviewers can verify fit provenance
            splitter_config["fit_audit"] = fit_audit
            if splitter_config.get("enabled"):
                print(
                    f"[temporal splitter] enabled  "
                    f"tau_high={splitter_config['tau_high']} "
                    f"tau_low={splitter_config['tau_low']} "
                    f"K={splitter_config['K']}  "
                    f"baseline_bout_F1@0.5={splitter_config['val_baseline_pooled_bout_f1_50']:.4f} "
                    f"-> decoded={splitter_config['val_decoded_pooled_bout_f1_50']:.4f}  "
                    f"baseline_frame_F1={splitter_config['val_baseline_pooled_frame_f1_macro_behavior']:.4f} "
                    f"-> decoded={splitter_config['val_decoded_pooled_frame_f1_macro_behavior']:.4f}"
                )
            else:
                print(
                    f"[temporal splitter] disabled  "
                    f"reason={splitter_config.get('disabled_reason', 'unknown')}"
                )
        except Exception as exc:
            splitter_config = make_v25_disabled_config(
                f"temporal splitter fitting failed: {exc}",
                target_class=target_class, target_index=target_index,
                replacement_class=replacement_class, replacement_index=replacement_index,
                gt_source="unavailable",
            )
            print(f"[warn] temporal splitter fitting failed: {exc}")

    if splitter_config is not None:
        config_dump[TEMPORAL_SPLITTER_CONFIG_KEY] = splitter_config
        config_dump[LEGACY_V25_SPLITTER_CONFIG_KEY] = splitter_config
        try:
            if args.save_v25_splitter_json:
                splitter_path = run_dir / "temporal_splitter.json"
                with open(splitter_path, "w") as f:
                    json.dump(splitter_config, f, indent=2)
                legacy_path = run_dir / "v25_splitter.json"
                with open(legacy_path, "w") as f:
                    json.dump(splitter_config, f, indent=2)
                print(f"[temporal splitter] saved {splitter_path}")
            with open(run_dir / "config.json", "w") as f:
                json.dump(config_dump, f, indent=2)
            attach_temporal_splitter_to_checkpoints(run_dir, config_dump, splitter_config)
            print("[temporal splitter] embedded in config.json and checkpoints")
        except Exception as exc:
            print(f"[warn] could not save temporal splitter metadata: {exc}")

    print(f"[done] best val_loss={best_val_loss:.4f}  best val_macroF1={best_val_f1:.3f}")
    print(f"checkpoints in {run_dir}")
    print(f"training log: {history_csv_path}")
    if args.confusion_matrix:
        print(f"confusion matrices: {cm_text_path}")


if __name__ == "__main__":
    main()
