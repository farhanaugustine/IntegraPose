# python train_behavior_lstm.py --manifest_path "C:\Users\Aegis-MSI\Documents\Lin_Lab_IntegraPose_Foundational_Model\Odor_Threshold\clips_for_labeling\behavior_lstm\TestProject\sequence_manifest.json" --output_dir "runs" --epochs 100 --batch_size 4 --device "cuda:0"

from __future__ import annotations

import argparse
import csv
import json
import random
import signal
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Tuple

# Suppress PyTorch AMP deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.*")

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

if __package__ is None or __package__ == "":  # pragma: no cover - CLI entry
    from config import AUGMENTATION_CONFIG, MODEL_DEFAULTS
    from data import (
        SequenceDataset,
        collate_with_error_handling,
        load_sequence_manifest,
    )
    from model import BehaviorSequenceClassifier
    from utils.logging_utils import setup_logger
    from utils.visualization import render_confusion_heatmap, save_training_curves
    from utils.config_loader import load_config, apply_config_as_parser_defaults
    from utils.training_utils import mixup_data, mixup_criterion
else:
    from .config import AUGMENTATION_CONFIG, MODEL_DEFAULTS
    from .data import (
        SequenceDataset,
        collate_with_error_handling,
        load_sequence_manifest,
    )
    from .model import BehaviorSequenceClassifier
    from .utils.logging_utils import setup_logger
    from .utils.visualization import render_confusion_heatmap, save_training_curves
    from .utils.config_loader import load_config, apply_config_as_parser_defaults
    from .utils.training_utils import mixup_data, mixup_criterion


def _yaml_safe(obj: object) -> object:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_yaml_safe(v) for v in obj]
    return obj


def _write_resolved_config_yaml(run_dir: Path, args: argparse.Namespace, manifest_meta: dict, logger) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    preprocessing = {
        "window_size": int(manifest_meta.get("window_size") or args.num_frames),
        "window_stride": int(manifest_meta.get("window_stride") or max(1, int(args.num_frames) // 2)),
        "crop_size": int(manifest_meta.get("crop_size") or args.frame_size),
        "pad_ratio": float(manifest_meta.get("pad_ratio", 0.2)),
        "yolo_conf": float(manifest_meta.get("yolo_conf", 0.25)),
        "yolo_iou": float(manifest_meta.get("yolo_iou", 0.45)),
        "yolo_imgsz": int(manifest_meta.get("yolo_imgsz", 640)),
        "yolo_task": manifest_meta.get("yolo_task"),
        "has_keypoints": bool(manifest_meta.get("has_keypoints", False)),
        "split_strategy": manifest_meta.get("split_strategy"),
        "source_dataset_root": manifest_meta.get("source_dataset_root"),
    }

    augmentation = dict(AUGMENTATION_CONFIG)

    model_cfg = {
        "backbone": str(getattr(args, "backbone", "")),
        "no_pretrained_backbone": bool(getattr(args, "no_pretrained_backbone", False)),
        "train_backbone": bool(getattr(args, "train_backbone", False)),
        "sequence_model": str(getattr(args, "sequence_model", "")),
        "hidden_dim": int(getattr(args, "hidden_dim", 0)),
        "num_layers": int(getattr(args, "num_layers", 0)),
        "dropout": float(getattr(args, "dropout", 0.0)),
        "bidirectional": bool(getattr(args, "bidirectional", False)),
        "attention_heads": int(getattr(args, "attention_heads", 0)),
        "temporal_attention_layers": int(getattr(args, "temporal_attention_layers", 0)),
        "attention_pool": bool(getattr(args, "attention_pool", False)),
        "positional_encoding": str(getattr(args, "positional_encoding", "none")),
        "positional_encoding_max_len": int(getattr(args, "positional_encoding_max_len", 0)),
        "feature_se": bool(getattr(args, "feature_se", False)),
        "slowfast_alpha": int(getattr(args, "slowfast_alpha", 4)),
        "slowfast_fusion_ratio": float(getattr(args, "slowfast_fusion_ratio", 0.25)),
        "slowfast_base_channels": int(getattr(args, "slowfast_base_channels", 48)),
        # Pose fusion
        "use_pose": bool(getattr(args, "use_pose", False)),
        "pose_fusion_strategy": str(getattr(args, "pose_fusion_strategy", "gated_attention")),
        "pose_fusion_dim": int(getattr(args, "pose_fusion_dim", 128)),
        "pose_input_dim": int(getattr(args, "pose_input_dim", 0) or 0),
    }

    training_cfg = {
        "manifest_path": str(getattr(args, "manifest_path", "")),
        "output_dir": str(getattr(args, "output_dir", "")),
        "run_name": str(getattr(args, "run_name", "")),
        "stop_file": str(getattr(args, "stop_file", "")) if getattr(args, "stop_file", None) else None,
        "resume": bool(getattr(args, "resume", False)),
        "num_frames": int(getattr(args, "num_frames", 0)),
        "frame_size": int(getattr(args, "frame_size", 0)),
        "epochs": int(getattr(args, "epochs", 0)),
        "batch_size": int(getattr(args, "batch_size", 0)),
        "lr": float(getattr(args, "lr", 0.0)),
        "backbone_lr": float(getattr(args, "backbone_lr", 0.0)),
        "weight_decay": float(getattr(args, "weight_decay", 0.0)),
        "scheduler": str(getattr(args, "scheduler", "none")),
        "patience": int(getattr(args, "patience", 0)),
        "seed": int(getattr(args, "seed", 42)),
        "num_workers": int(getattr(args, "num_workers", 0)),
        "device": str(getattr(args, "device", "auto")),
        "disable_augment": bool(getattr(args, "disable_augment", False)),
        "mixup_alpha": float(getattr(args, "mixup_alpha", 0.0)),
        "class_weighting": str(getattr(args, "class_weighting", "none")),
        "class_weight_max": float(getattr(args, "class_weight_max", 0.0)),
        "train_sampler": str(getattr(args, "train_sampler", "shuffle")),
        "log_interval": int(getattr(args, "log_interval", 0)),
        "per_class_metrics": bool(getattr(args, "per_class_metrics", False)),
        "confusion_matrix": bool(getattr(args, "confusion_matrix", False)),
        "confusion_matrix_interval": int(getattr(args, "confusion_matrix_interval", 1)),
        "skip_startup_checks": bool(getattr(args, "skip_startup_checks", False)),
    }

    resolved = {
        "preprocessing": preprocessing,
        "augmentation": augmentation,
        "model": model_cfg,
        "training": training_cfg,
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_config": str(getattr(args, "config", "")) if getattr(args, "config", None) else None,
            "run_dir": str(run_dir),
        },
    }

    out_path = run_dir / "config.yaml"
    try:
        out_path.write_text(yaml.safe_dump(_yaml_safe(resolved), sort_keys=False), encoding="utf-8")
        logger.info(f"Saved resolved run config to: {out_path}")
    except Exception as exc:
        logger.warning(f"Failed to write config.yaml ({exc})")


def get_arg_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent if script_dir.name == "behavior_lstm" else script_dir
    default_manifest = (
        repo_root / "processed_sequences" / "sequence_manifest.json"
    )
    default_output = repo_root / "behavior_lstm_runs"

    parser = argparse.ArgumentParser(
        description="Train the LSTM-based behavior classification model."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML configuration file. Overrides defaults.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=default_output,
        help="Folder where checkpoints and logs will be saved.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=f"behavior_lstm_{int(time.time())}",
        help="Sub-directory name for this training run.",
    )
    parser.add_argument(
        "--stop_file",
        type=Path,
        default=None,
        help=(
            "Graceful stop file path. If this file exists, training stops cleanly "
            "and exports final validation metrics/artifacts before exiting. "
            "If omitted, defaults to <output_dir>/<run_name>/stop_training.flag."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from a prior run directory (loads last_checkpoint.pt if present).",
    )
    parser.add_argument(
        "--manifest_path",
        type=Path,
        default=default_manifest,
        help="Sequence manifest JSON produced by prepare_sequences_from_yolo.py.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Override the sequence length (defaults to manifest metadata).",
    )
    parser.add_argument(
        "--frame_size",
        type=int,
        default=None,
        help="Override the crop size (defaults to manifest metadata).",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument(
        "--mixup_alpha",
        type=float,
        default=0.0,
        help="Alpha for MixUp augmentation (0.0 to disable, e.g., 0.2-0.4 for low data).",
    )
    parser.add_argument(
        "--class_weighting",
        type=str,
        default="none",
        choices=["none", "inverse", "sqrt_inverse"],
        help=(
            "Optional class reweighting for CrossEntropyLoss based on training-set label frequency. "
            "Use 'inverse' (or 'sqrt_inverse' for a milder effect) to encourage learning rare classes."
        ),
    )
    parser.add_argument(
        "--class_weight_max",
        type=float,
        default=0.0,
        help="Clamp computed class weights to this maximum (0 to disable clamping).",
    )
    parser.add_argument(
        "--train_sampler",
        type=str,
        default="shuffle",
        choices=["shuffle", "weighted"],
        help=(
            "Sampling strategy for the training DataLoader. "
            "'weighted' uses WeightedRandomSampler (inverse-frequency) to oversample rare classes."
        ),
    )
    parser.add_argument(
        "--monitor_metric",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--use_pose",
        action="store_true",
        help="Enable pose-feature fusion. Requires keypoints in the dataset.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=MODEL_DEFAULTS["hidden_dim"],
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=MODEL_DEFAULTS["num_layers"],
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=MODEL_DEFAULTS["dropout"],
    )
    parser.add_argument(
        "--bidirectional", action="store_true", help="Use a bi-directional LSTM."
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="mobilenet_v3_small",
        help="CNN backbone to encode frames.",
    )
    parser.add_argument(
        "--sequence_model",
        type=str,
        default="lstm",
        choices=["lstm", "attention"],
        help="Temporal head to aggregate frame embeddings.",
    )
    parser.add_argument(
        "--temporal_attention_layers",
        type=int,
        default=0,
        help="Number of self-attention layers before the temporal head.",
    )
    parser.add_argument(
        "--attention_heads",
        type=int,
        default=MODEL_DEFAULTS["attention_heads"],
        help="Number of attention heads for temporal blocks/pooling.",
    )
    parser.add_argument(
        "--attention_pool",
        action="store_true",
        help="Enable attention pooling over LSTM outputs.",
    )
    parser.add_argument(
        "--positional_encoding",
        type=str,
        default="none",
        choices=["none", "sinusoidal", "learned"],
        help=(
            "Positional encoding for temporal attention (used by temporal self-attn "
            "layers and attention pooling)."
        ),
    )
    parser.add_argument(
        "--positional_encoding_max_len",
        type=int,
        default=MODEL_DEFAULTS["positional_encoding_max_len"],
        help="Max length for learned positional embeddings (ignored for sinusoidal).",
    )
    parser.add_argument(
        "--feature_se",
        action="store_true",
        help="Enable squeeze-excitation gating on per-frame features.",
    )
    parser.add_argument(
        "--no_pretrained_backbone",
        action="store_true",
        help="Disable ImageNet pretraining for the CNN backbone.",
    )
    parser.add_argument(
        "--train_backbone",
        action="store_true",
        help="Unfreeze the CNN backbone during training.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (e.g., 'cuda', 'cpu', 'mps', or 'auto').",
    )
    parser.add_argument(
        "--disable_augment",
        action="store_true",
        help="Turn off training-time frame augmentations.",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=0,
        help="Print batch-level progress every N steps (0 to disable).",
    )
    parser.add_argument(
        "--per_class_metrics",
        action="store_true",
        help="Print per-class validation accuracy each epoch (when a val split exists).",
    )
    parser.add_argument(
        "--confusion_matrix",
        action="store_true",
        help="Print confusion matrices (raw + normalized) and save PNG heatmaps to the run directory.",
    )
    parser.add_argument(
        "--confusion_matrix_interval",
        type=int,
        default=1,
        help="Print validation confusion matrices every N epochs (1 = every epoch).",
    )
    parser.add_argument(
        "--slowfast_alpha",
        type=int,
        default=MODEL_DEFAULTS["slowfast_alpha"],
        help="Temporal stride between fast/slow pathways (slowfast only).",
    )
    parser.add_argument(
        "--slowfast_fusion_ratio",
        type=float,
        default=MODEL_DEFAULTS["slowfast_fusion_ratio"],
        help="Channel ratio for the slow pathway (slowfast only).",
    )
    parser.add_argument(
        "--slowfast_base_channels",
        type=int,
        default=MODEL_DEFAULTS["slowfast_base_channels"],
        help="Base channel width for the slowfast backbone.",
    )
    parser.add_argument(
        "--pose_fusion_strategy",
        type=str,
        default=MODEL_DEFAULTS.get("pose_fusion_strategy", "gated_attention"),
        help="Strategy for fusing pose features ('concat' or 'gated_attention').",
    )
    parser.add_argument(
        "--pose_fusion_dim",
        type=int,
        default=MODEL_DEFAULTS.get("pose_fusion_dim", 128),
        help="Dimension to project pose features to before fusion.",
    )
    parser.add_argument(
        "--backbone_lr",
        type=float,
        default=1e-4,
        help="Separate learning rate for the backbone (differential learning rate; used when --train_backbone is set).",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=["none", "plateau", "cosine"],
        help="Learning rate scheduler to use.",
    )
    parser.add_argument(
        "--skip_startup_checks",
        action="store_true",
        help="Skip startup sanity checks (label histogram, crop preview).",
    )
    return parser


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_args(args: argparse.Namespace, logger: logging.Logger):
    if args.epochs <= 0:
        raise ValueError("--epochs must be >= 1")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be >= 1")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be >= 0")
    if args.patience < 0:
        raise ValueError("--patience must be >= 0")
    if args.num_frames is None or int(args.num_frames) <= 0:
        raise ValueError("--num_frames must be >= 1 (or present in the manifest)")
    if args.frame_size is None or int(args.frame_size) <= 0:
        raise ValueError("--frame_size must be >= 1 (or present in the manifest)")
    if not (0.0 <= float(args.dropout) < 1.0):
        raise ValueError("--dropout must be in [0.0, 1.0)")
    if float(args.lr) <= 0:
        raise ValueError("--lr must be > 0")
    if float(args.weight_decay) < 0:
        raise ValueError("--weight_decay must be >= 0")
    if float(getattr(args, "class_weight_max", 0.0)) < 0:
        raise ValueError("--class_weight_max must be >= 0")
    if int(getattr(args, "confusion_matrix_interval", 1)) <= 0:
        raise ValueError("--confusion_matrix_interval must be >= 1")

    backbone = str(args.backbone).lower()
    if backbone.startswith("slowfast"):
        return

    feature_dim_by_backbone = {
        "mobilenet_v3_small": 576,
        "efficientnet_b0": 1280,
        "vit_b_16": 768,
    }
    feature_dim = feature_dim_by_backbone.get(backbone)
    if feature_dim is None:
        raise ValueError(
            f"Unsupported backbone '{args.backbone}'. "
            "Choose from ['mobilenet_v3_small', 'efficientnet_b0', 'vit_b_16', 'slowfast_tiny', 'slowfast_r50']."
        )

    uses_attention = (
        int(args.temporal_attention_layers) > 0
        or bool(args.attention_pool)
        or str(args.sequence_model).lower() == "attention"
    )
    dropout = float(args.dropout)
    if uses_attention and dropout >= 0.6:
        msg = (
            f"--dropout={dropout:.2f} is also applied inside attention blocks/pooling in this "
            "implementation (not just the classifier head). Values this high often prevent the "
            "temporal attention components from learning."
        )
        logger.warning(msg)

    if args.train_backbone and backbone.startswith("vit") and float(args.lr) > 1e-4:
        msg = (
            f"Fine-tuning ViT (--train_backbone) with lr={float(args.lr):.2e} can be unstable. "
            "If training collapses to predicting a single class, try a smaller LR (e.g., 1e-5..1e-4) "
            "or freeze the backbone for a few epochs."
        )
        logger.warning(msg)
    positional_encoding = str(getattr(args, "positional_encoding", "none")).strip().lower()
    if positional_encoding == "learned" and uses_attention:
        max_len = int(getattr(args, "positional_encoding_max_len", 0))
        if max_len <= 0:
            raise ValueError(
                "--positional_encoding_max_len must be >= 1 when using learned positional encoding"
            )
        if max_len < int(args.num_frames):
            raise ValueError(
                "--positional_encoding_max_len must be >= --num_frames when using learned positional encoding"
            )
    if uses_attention:
        heads = int(args.attention_heads)
        if heads <= 0:
            raise ValueError("--attention_heads must be >= 1")

        if int(args.temporal_attention_layers) > 0 and feature_dim % heads != 0:
            raise ValueError(
                f"Temporal attention uses embed_dim={feature_dim} (from backbone='{args.backbone}'). "
                f"It must be divisible by attention_heads={heads}."
            )

        if str(args.sequence_model).lower() == "attention":
            if feature_dim % heads != 0:
                raise ValueError(
                    f"Attention pooling uses embed_dim={feature_dim} (from backbone='{args.backbone}'). "
                    f"It must be divisible by attention_heads={heads}."
                )
        elif bool(args.attention_pool):
            if int(args.hidden_dim) <= 0:
                raise ValueError("--hidden_dim must be >= 1 when using LSTM")
            if int(args.num_layers) <= 0:
                raise ValueError("--num_layers must be >= 1 when using LSTM")
            seq_dim = int(args.hidden_dim) * (2 if bool(args.bidirectional) else 1)
            if seq_dim % heads != 0:
                raise ValueError(
                    f"Attention pooling uses embed_dim={seq_dim} (hidden_dim x directions). "
                    f"It must be divisible by attention_heads={heads}."
                )
    else:
        if str(args.sequence_model).lower() == "lstm":
            if int(args.hidden_dim) <= 0:
                raise ValueError("--hidden_dim must be >= 1 when using LSTM")
            if int(args.num_layers) <= 0:
                raise ValueError("--num_layers must be >= 1 when using LSTM")


def build_loaders(
    splits: Dict[str, List],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader | None, DataLoader | None]:
    def _make_dataset(samples, augment: bool):
        if not samples:
            return None
        return SequenceDataset(
            samples,
            num_frames=args.num_frames,
            frame_size=args.frame_size,
            augment=augment,
            load_pose=bool(getattr(args, "use_pose", False)),
        )

    train_dataset = _make_dataset(
        splits.get("train", []), augment=not args.disable_augment
    )
    val_dataset = _make_dataset(splits.get("val", []), augment=False)
    test_dataset = _make_dataset(splits.get("test", []), augment=False)

    def _make_loader(dataset, shuffle: bool, sampler=None):
        if dataset is None or len(dataset) == 0:
            return None
        if sampler is not None and shuffle:
            shuffle = False
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_with_error_handling,
        )

    train_sampler = None
    if train_dataset is not None and str(getattr(args, "train_sampler", "shuffle")).lower() == "weighted":
        counts = Counter(int(s.label) for s in splits.get("train", []))
        weights: List[float] = []
        for sample in train_dataset.samples:
            label = int(getattr(sample, "label", -1))
            denom = float(counts.get(label, 0))
            weights.append(1.0 / denom if denom > 0 else 0.0)
        generator = torch.Generator()
        generator.manual_seed(int(getattr(args, "seed", 42)))
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )

    return (
        _make_loader(train_dataset, True, sampler=train_sampler),
        _make_loader(val_dataset, False),
        _make_loader(test_dataset, False),
    )


def _format_label_histogram(
    counts: Counter, idx_to_class: Dict[int, str] | None = None
) -> str:
    parts: List[str] = []
    for label, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        name = None
        if idx_to_class is not None:
            name = idx_to_class.get(int(label))
        if name:
            parts.append(f"{int(label)}({name})={int(n)}")
        else:
            parts.append(f"{int(label)}={int(n)}")
    return ", ".join(parts) if parts else "<empty>"


def _format_class_weights(
    weights: torch.Tensor, idx_to_class: Dict[int, str] | None = None
) -> str:
    weights_cpu = weights.detach().float().cpu().tolist()
    parts: List[str] = []
    for idx, w in enumerate(weights_cpu):
        name = None if idx_to_class is None else idx_to_class.get(int(idx))
        if name:
            parts.append(f"{idx}({name})={w:.4g}")
        else:
            parts.append(f"{idx}={w:.4g}")
    return ", ".join(parts) if parts else "<empty>"


def compute_class_weights_from_train_split(
    *,
    train_samples: List,
    num_classes: int,
    scheme: str,
    idx_to_class: Dict[int, str] | None,
    logger,
    max_weight: float = 0.0,
) -> torch.Tensor | None:
    scheme_norm = str(scheme).strip().lower()
    if scheme_norm in {"none", ""}:
        return None
    if scheme_norm not in {"inverse", "sqrt_inverse"}:
        raise ValueError(f"Unsupported class weighting scheme: {scheme!r}")

    counts = torch.zeros(int(num_classes), dtype=torch.long)
    for sample in train_samples:
        try:
            label = int(sample.label)
        except Exception:
            continue
        if 0 <= label < int(num_classes):
            counts[label] += 1

    present = counts > 0
    num_present = int(present.sum().item())
    if num_present == 0:
        logger.warning("Class weighting requested, but training split appears empty. Disabling weights.")
        return None
    if num_present < int(num_classes):
        missing = [i for i in range(int(num_classes)) if int(counts[i].item()) == 0]
        missing_names = (
            [idx_to_class.get(int(i), str(i)) for i in missing] if idx_to_class else [str(i) for i in missing]
        )
        logger.warning(
            "Class weighting: some classes are missing from the training split; "
            f"weights for them will be 0. Missing: {missing_names}"
        )

    power = 1.0 if scheme_norm == "inverse" else 0.5
    raw = torch.zeros(int(num_classes), dtype=torch.float32)
    raw[present] = 1.0 / torch.pow(counts[present].float(), power)
    weights = torch.zeros(int(num_classes), dtype=torch.float32)
    weights[present] = raw[present] / raw[present].mean()

    if max_weight and float(max_weight) > 0:
        weights = torch.clamp(weights, max=float(max_weight))

    logger.info(
        "Using class-weighted CrossEntropyLoss "
        f"(scheme={scheme_norm}, max={float(max_weight):g}). "
        f"Weights: {_format_class_weights(weights, idx_to_class)}"
    )
    return weights


@torch.no_grad()
def run_startup_sanity_checks(
    *,
    logger,
    run_dir: Path,
    args: argparse.Namespace,
    splits: Dict[str, List],
    idx_to_class: Dict[int, str],
) -> None:
    """
    Catch common training regressions early (bad splits, broken crops, NaNs).

    Errors are reserved for clear "cannot learn" conditions. Everything else is
    flagged via warnings/info so long runs don't silently waste hours.
    """

    train_samples = splits.get("train", [])
    val_samples = splits.get("val", [])

    train_counts = Counter(int(s.label) for s in train_samples)
    val_counts = Counter(int(s.label) for s in val_samples)

    logger.info(
        f"Startup check: train split labels: {_format_label_histogram(train_counts, idx_to_class)}"
    )
    if val_samples:
        logger.info(
            f"Startup check: val split labels: {_format_label_histogram(val_counts, idx_to_class)}"
        )

    train_unique = sorted(train_counts.keys())
    if len(train_unique) < 2:
        only = train_unique[0] if train_unique else None
        only_name = idx_to_class.get(int(only)) if only is not None else None
        raise ValueError(
            "Startup check failed: training split contains <2 classes "
            f"(only label={only}{' (' + only_name + ')' if only_name else ''}). "
            "Check your manifest/splits."
        )

    if val_samples:
        missing_in_val = [k for k in train_unique if k not in val_counts]
        if missing_in_val:
            missing_names = [idx_to_class.get(int(k), str(k)) for k in missing_in_val]
            logger.warning(
                "Startup check: validation split is missing classes present in train: "
                + ", ".join(missing_names)
            )

    meta = getattr(args, "_manifest_meta", None) or {}
    if isinstance(meta, dict):
        yolo_task = str(meta.get("yolo_task") or "").strip().lower()
        if yolo_task:
            logger.info(f"Startup check: preprocessing YOLO task: {yolo_task}")
        if meta.get("has_keypoints") and not bool(args.use_pose):
            logger.info(
                "Startup check: manifest indicates keypoints are available, but --use_pose is disabled (RGB-only training)."
            )
            if yolo_task == "pose":
                logger.warning(
                    "Startup check: preprocessing used a pose model (yolo_task=pose), but training is RGB-only. "
                    "If crops look unstable, consider re-preprocessing with a detection model for tighter/more stable boxes."
                )
        if (not meta.get("has_keypoints")) and bool(args.use_pose):
            logger.warning(
                "Startup check: --use_pose is enabled but the manifest metadata does not indicate keypoints are present. "
                "If training fails with 'no keypoint data found', re-run preprocessing with YOLO-pose weights."
            )
            if yolo_task == "detect":
                logger.warning(
                    "Startup check: preprocessing used a detection model (yolo_task=detect), but training expects pose. "
                    "Re-run preprocessing with YOLO-pose weights."
                )
        ws = meta.get("window_size")
        cs = meta.get("crop_size")
        if ws is not None and int(ws) != int(args.num_frames):
            logger.warning(
                f"Startup check: --num_frames={int(args.num_frames)} differs from manifest window_size={int(ws)}."
            )
        if cs is not None and int(cs) != int(args.frame_size):
            logger.warning(
                f"Startup check: --frame_size={int(args.frame_size)} differs from manifest crop_size={int(cs)}."
            )

    if bool(args.train_backbone) and getattr(args, "backbone_lr", None) is None:
        lr_val = float(args.lr)
        if lr_val >= 5e-4 and not bool(getattr(args, "no_pretrained_backbone", False)):
            logger.warning(
                "Startup check: fine-tuning a pretrained backbone with lr >= 5e-4 can be unstable. "
                "Consider setting --backbone_lr (e.g., 1e-5..1e-4) or lowering --lr."
            )

    # Deterministic preview batch (no augment) to verify crops look reasonable.
    try:
        preview_n = int(min(64, len(train_samples)))
        if preview_n <= 0:
            raise RuntimeError("Training split is empty.")
        preview_ds = SequenceDataset(
            train_samples[:preview_n],
            num_frames=int(args.num_frames),
            frame_size=int(args.frame_size),
            augment=False,
            load_pose=bool(getattr(args, "use_pose", False)),
        )
        preview_loader = DataLoader(
            preview_ds,
            batch_size=int(min(16, args.batch_size)),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_with_error_handling,
        )
        first = next(iter(preview_loader))
        if first is None:
            raise RuntimeError("Preview loader returned an empty batch.")
        inputs, _ = first
        if not (isinstance(inputs, (tuple, list)) and len(inputs) >= 1):
            raise RuntimeError("Preview loader returned unexpected inputs structure.")
        clips = inputs[0]
        if not torch.is_tensor(clips) or clips.ndim != 5:
            raise RuntimeError(
                f"Expected clips tensor [B, T, C, H, W], got {type(clips)} with shape {getattr(clips, 'shape', None)}"
            )
        if not torch.isfinite(clips).all().item():
            raise RuntimeError("Clip preview contains NaN/Inf.")

        try:
            from config import IMAGENET_MEAN, IMAGENET_STD
        except Exception:  # pragma: no cover
            from .config import IMAGENET_MEAN, IMAGENET_STD

        mean = torch.tensor(IMAGENET_MEAN, dtype=clips.dtype, device=clips.device)[
            None, None, :, None, None
        ]
        std = torch.tensor(IMAGENET_STD, dtype=clips.dtype, device=clips.device)[
            None, None, :, None, None
        ]
        disp = (clips * std + mean).clamp(0.0, 1.0)
        disp_min = float(disp.min().item())
        disp_max = float(disp.max().item())
        disp_mean = float(disp.mean().item())
        disp_std = float(disp.std().item())
        logger.info(
            "Startup check: crop preview stats (0..1 space): "
            f"min={disp_min:.3f}, max={disp_max:.3f}, mean={disp_mean:.3f}, std={disp_std:.3f}"
        )
        if disp_std < 0.02:
            logger.warning(
                "Startup check: crops have very low variance (std < 0.02). "
                "This can happen if crops are mostly background/blank; review QA crops."
            )
        if disp_max < 0.10:
            logger.warning(
                "Startup check: crops appear very dark (max < 0.10). Review QA crops / preprocessing."
            )

        try:
            from torchvision.utils import make_grid
            from PIL import Image

            b = int(min(clips.shape[0], 16))
            frames = clips[:b, 0, :, :, :]
            mean2 = torch.tensor(IMAGENET_MEAN, dtype=frames.dtype, device=frames.device)[
                None, :, None, None
            ]
            std2 = torch.tensor(IMAGENET_STD, dtype=frames.dtype, device=frames.device)[
                None, :, None, None
            ]
            frames = (frames * std2 + mean2).clamp(0.0, 1.0)
            grid = make_grid(frames.cpu(), nrow=4, padding=2)
            img = (grid.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
            Image.fromarray(img).save(run_dir / "startup_crops_preview.png")
            logger.info("Startup check: saved crop preview to startup_crops_preview.png")
        except Exception:
            pass
    except Exception as exc:
        logger.warning(f"Startup check: could not generate crop preview ({exc})")

    # Pose sanity previews: verify pose aligns with crops and augmentations.
    if bool(getattr(args, "use_pose", False)):
        try:
            import math
            from PIL import Image, ImageDraw

            try:
                from config import IMAGENET_MEAN, IMAGENET_STD
            except Exception:  # pragma: no cover
                from .config import IMAGENET_MEAN, IMAGENET_STD

            def _draw_kpts(img: Image.Image, kpts_px: np.ndarray | None) -> None:
                if kpts_px is None:
                    return
                try:
                    draw = ImageDraw.Draw(img)
                except Exception:
                    return
                w, h = img.size
                try:
                    arr = np.asarray(kpts_px, dtype=np.float32)
                except Exception:
                    return
                if arr.ndim != 2 or arr.shape[1] < 2:
                    return
                for row in arr:
                    x = float(row[0])
                    y = float(row[1])
                    c = float(row[2]) if arr.shape[1] >= 3 else 1.0
                    if not (np.isfinite(x) and np.isfinite(y)):
                        continue
                    if c <= 0.0:
                        continue
                    if x < -5 or y < -5 or x > w + 5 or y > h + 5:
                        continue
                    r = 2
                    x0 = int(round(x - r))
                    y0 = int(round(y - r))
                    x1 = int(round(x + r))
                    y1 = int(round(y + r))
                    draw.ellipse((x0, y0, x1, y1), outline=(0, 255, 255), width=1)

            def _make_grid(images: list[Image.Image], *, nrow: int = 4, pad: int = 2) -> Image.Image | None:
                if not images:
                    return None
                nrow = max(1, int(nrow))
                pad = max(0, int(pad))
                w, h = images[0].size
                n = len(images)
                ncol = int(math.ceil(n / nrow))
                grid_w = nrow * w + (nrow - 1) * pad
                grid_h = ncol * h + (ncol - 1) * pad
                canvas = Image.new("RGB", (grid_w, grid_h), (0, 0, 0))
                for i, img in enumerate(images):
                    r = i // nrow
                    c = i % nrow
                    x = c * (w + pad)
                    y = r * (h + pad)
                    canvas.paste(img, (x, y))
                return canvas

            def _load_npz(sample) -> tuple[np.ndarray, np.ndarray | None]:
                if not sample.sequence_npz or not Path(sample.sequence_npz).exists():
                    return (None, None)  # type: ignore[return-value]
                with np.load(sample.sequence_npz, allow_pickle=False) as data:
                    frames_np = data["frames"]
                    kpts_np = data["keypoints"] if "keypoints" in data else None
                    return frames_np, kpts_np

            def _unnormalize_frame(frame_t: torch.Tensor) -> Image.Image | None:
                if frame_t.ndim != 3:
                    return None
                mean = torch.tensor(IMAGENET_MEAN, dtype=frame_t.dtype, device=frame_t.device)[:, None, None]
                std = torch.tensor(IMAGENET_STD, dtype=frame_t.dtype, device=frame_t.device)[:, None, None]
                img_t = (frame_t * std + mean).clamp(0.0, 1.0)
                img_np = (img_t.detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype("uint8")
                return Image.fromarray(img_np)

            def _pose_preview_nonaug(samples: list, *, max_items: int, stem: str) -> None:
                imgs: list[Image.Image] = []
                picked = 0
                for sample in samples:
                    if picked >= max_items:
                        break
                    frames_np, kpts_np = _load_npz(sample)
                    if frames_np is None or kpts_np is None:
                        continue
                    if frames_np.ndim != 4 or kpts_np.ndim != 3:
                        continue
                    if frames_np.shape[0] <= 0:
                        continue
                    # Use the first frame of the clip, resized the same way as non-aug training.
                    frame0 = frames_np[0]
                    h0, w0 = int(frame0.shape[0]), int(frame0.shape[1])
                    img = Image.fromarray(frame0).resize(
                        (int(args.frame_size), int(args.frame_size)),
                        resample=Image.BILINEAR,
                    )
                    k0 = kpts_np[0].copy()
                    k0[:, 0] *= float(args.frame_size) / max(float(w0), 1.0)
                    k0[:, 1] *= float(args.frame_size) / max(float(h0), 1.0)
                    _draw_kpts(img, k0)
                    imgs.append(img)
                    picked += 1
                grid = _make_grid(imgs, nrow=4, pad=2)
                if grid is None:
                    logger.warning(f"Startup check: no pose samples found for {stem} (non-aug).")
                    return
                out_path = run_dir / f"{stem}.png"
                grid.save(out_path)
                logger.info(f"Startup check: saved pose preview to {out_path.name}")

            def _pose_preview_aug(samples: list, *, max_items: int, stem: str) -> None:
                imgs: list[Image.Image] = []
                picked = 0
                aug_ds = SequenceDataset(
                    samples,
                    num_frames=int(args.num_frames),
                    frame_size=int(args.frame_size),
                    augment=True,
                    load_pose=True,
                )
                for sample in samples:
                    if picked >= max_items:
                        break
                    frames_np, kpts_np = _load_npz(sample)
                    if frames_np is None or kpts_np is None:
                        continue
                    if frames_np.ndim != 4 or kpts_np.ndim != 3:
                        continue
                    out = aug_ds._process_augmented_clip(frames_np, kpts_np, return_keypoints_px=True)
                    clip_t, _pose_t, kpts_px = out  # type: ignore[misc]
                    if not torch.is_tensor(clip_t) or clip_t.ndim != 4:
                        continue
                    img = _unnormalize_frame(clip_t[0])
                    if img is None:
                        continue
                    if isinstance(kpts_px, np.ndarray) and kpts_px.ndim == 3 and kpts_px.shape[0] > 0:
                        _draw_kpts(img, kpts_px[0])
                    imgs.append(img)
                    picked += 1
                grid = _make_grid(imgs, nrow=4, pad=2)
                if grid is None:
                    logger.warning(f"Startup check: no pose samples found for {stem} (aug).")
                    return
                out_path = run_dir / f"{stem}.png"
                grid.save(out_path)
                logger.info(f"Startup check: saved pose+aug preview to {out_path.name}")

            max_preview = 16
            _pose_preview_nonaug(train_samples, max_items=max_preview, stem="startup_pose_preview_train")
            if val_samples:
                _pose_preview_nonaug(val_samples, max_items=max_preview, stem="startup_pose_preview_val")
            _pose_preview_aug(train_samples, max_items=max_preview, stem="startup_pose_aug_preview_train")
        except Exception as exc:
            logger.warning(f"Startup check: could not generate pose previews ({exc})")

def train_one_epoch(
    model: BehaviorSequenceClassifier,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
    logger: logging.Logger,
    scaler: torch.cuda.amp.GradScaler | None,
    log_interval: int = 0,
    should_stop: Callable[[], bool] | None = None,
    mixup_alpha: float = 0.0,
):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    num_batches = len(loader)
    
    use_amp = (scaler is not None) and (device.type == "cuda")

    for batch_idx, batch in enumerate(loader, start=1):
        if should_stop is not None and should_stop():
            break
        if batch is None:
            continue
        
        # Unpack batch
        inputs, labels = batch
        pose_present = None
        if isinstance(inputs, (tuple, list)) and len(inputs) == 3:
            clips, poses, pose_present = inputs
        else:
            clips, poses = inputs
        
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if poses is not None:
            poses = poses.to(device, non_blocking=True)
        if pose_present is not None:
            pose_present = pose_present.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            if mixup_alpha > 0.0:
                lam = 1.0
                if mixup_alpha > 0:
                    lam = np.random.beta(mixup_alpha, mixup_alpha)
                
                batch_size = clips.size(0)
                index = torch.randperm(batch_size).to(device)

                mixed_clips = lam * clips + (1 - lam) * clips[index, :]
                mixed_poses = None
                mixed_pose_present = pose_present
                if poses is not None:
                    mixed_poses = lam * poses + (1 - lam) * poses[index, :]
                    if pose_present is not None:
                        mixed_pose_present = pose_present & pose_present[index]
                
                labels_a, labels_b = labels, labels[index]
                
                if mixed_poses is None and mixed_pose_present is None:
                    model_input = mixed_clips
                elif mixed_pose_present is None:
                    model_input = (mixed_clips, mixed_poses)
                else:
                    model_input = (mixed_clips, mixed_poses, mixed_pose_present)
                logits = model(model_input)
                loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            else:
                if poses is None and pose_present is None:
                    model_input = clips
                elif pose_present is None:
                    model_input = (clips, poses)
                else:
                    model_input = (clips, poses, pose_present)
                logits = model(model_input)
                loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        running_loss += loss.item() * clips.size(0)
        _, preds = torch.max(logits, dim=1)
        
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if log_interval > 0 and (
            batch_idx % log_interval == 0 or batch_idx == num_batches
        ):
            batch_acc = (preds == labels).float().mean().item()
            logger.info(
                f"    step {batch_idx:04d}/{num_batches:04d} "
                f"batch_loss={loss.item():.4f} batch_acc={batch_acc:.3f} "
                f"running_acc={correct / max(total, 1):.3f}"
            )

    if total == 0:
        if should_stop is not None and should_stop():
            return {"loss": float("nan"), "accuracy": float("nan")}
        raise RuntimeError(
            "No valid training samples were processed. "
            "Check that your manifest paths exist and that --num_frames/--frame_size "
            "match the saved sequences."
        )

    return {
        "loss": running_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


@torch.no_grad()
def evaluate(
    model: BehaviorSequenceClassifier,
    loader: DataLoader | None,
    criterion,
    device: torch.device,
    idx_to_class: Dict[int, str] | None = None,
    compute_confusion: bool = False,
):
    if loader is None:
        return None

    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0

    confusion = None
    num_classes = None
    per_class_correct = None
    per_class_total = None
    if idx_to_class:
        num_classes = max(idx_to_class.keys()) + 1
        per_class_correct = torch.zeros(num_classes, dtype=torch.long)
        per_class_total = torch.zeros(num_classes, dtype=torch.long)
        if compute_confusion:
            confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
            
    use_amp = (device.type == "cuda")

    for batch in loader:
        if batch is None:
            continue
        
        inputs, labels = batch
        pose_present = None
        if isinstance(inputs, (tuple, list)) and len(inputs) == 3:
            clips, poses, pose_present = inputs
        else:
            clips, poses = inputs
        
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if poses is not None:
            poses = poses.to(device, non_blocking=True)
        if pose_present is not None:
            pose_present = pose_present.to(device, non_blocking=True)

        if poses is None and pose_present is None:
            model_input = clips
        elif pose_present is None:
            model_input = (clips, poses)
        else:
            model_input = (clips, poses, pose_present)
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(model_input)
            loss = criterion(logits, labels)
            
        running_loss += loss.item() * clips.size(0)

        _, preds = torch.max(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if per_class_correct is not None and per_class_total is not None:
            labels_cpu = labels.detach().cpu()
            preds_cpu = preds.detach().cpu()
            per_class_total += torch.bincount(labels_cpu, minlength=num_classes)
            per_class_correct += torch.bincount(
                labels_cpu[preds_cpu == labels_cpu],
                minlength=num_classes,
            )

            if confusion is not None:
                indices = labels_cpu * num_classes + preds_cpu
                confusion += torch.bincount(
                    indices,
                    minlength=num_classes * num_classes,
                ).view(num_classes, num_classes)

    if total == 0:
        return None

    metrics: Dict[str, object] = {
        "loss": running_loss / total,
        "accuracy": correct / total,
    }

    if (
        idx_to_class
        and num_classes is not None
        and per_class_correct is not None
        and per_class_total is not None
    ):
        per_class: Dict[str, Dict[str, float | int | None]] = {}
        for idx, name in sorted(idx_to_class.items()):
            class_total = int(per_class_total[idx].item())
            class_correct = int(per_class_correct[idx].item())
            class_acc = (
                class_correct / class_total if class_total > 0 else None  
            )
            per_class[name] = {
                "accuracy": class_acc,
                "correct": class_correct,
                "total": class_total,
            }
        if confusion is not None:
            f1_values: list[tuple[float, int]] = []
            for idx, name in sorted(idx_to_class.items()):
                tp = int(confusion[idx, idx].item())
                support = int(confusion[idx, :].sum().item())
                pred_total = int(confusion[:, idx].sum().item())
                fp = pred_total - tp
                fn = support - tp
                if support <= 0:
                    precision = None
                    recall = None
                    f1 = None
                else:
                    precision = tp / pred_total if pred_total > 0 else 0.0
                    recall = tp / support
                    denom = precision + recall
                    f1 = (2.0 * precision * recall) / denom if denom > 0 else 0.0
                per_class[name].update(
                    {
                        "precision": float(precision) if precision is not None else None,
                        "recall": float(recall) if recall is not None else None,
                        "f1": float(f1) if f1 is not None else None,
                    }
                )
                if f1 is not None and support > 0:
                    f1_values.append((float(f1), support))
        metrics["per_class"] = per_class
        if confusion is not None:
            f1_only = [val for val, _ in f1_values]
            macro_f1 = sum(f1_only) / len(f1_only) if f1_only else None
            total_support = sum(sup for _, sup in f1_values)
            weighted_f1 = (
                sum(val * sup for val, sup in f1_values) / total_support
                if total_support > 0
                else None
            )
            metrics["macro_f1"] = macro_f1
            metrics["weighted_f1"] = weighted_f1
            metrics["micro_f1"] = metrics.get("accuracy")

    if confusion is not None and idx_to_class and num_classes is not None:
        # JSON-serializable matrices for history.json
        metrics["confusion_matrix"] = confusion.tolist()
        row_sums = confusion.sum(dim=1, keepdim=True).clamp(min=1)
        norm = (confusion.float() / row_sums).tolist()
        metrics["confusion_matrix_normalized"] = norm

    return metrics


def _short_label(label: str, max_len: int = 16) -> str:
    label = str(label)
    if len(label) <= max_len:
        return label
    return label[: max_len - 3] + "..."


def _append_metrics_csv(
    path: Path,
    *,
    fieldnames: List[str],
    row: Dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if should_write_header:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def _format_confusion_table(
    matrix: List[List[float | int]],
    class_names: List[str],
    cell_format,
) -> str:
    labels = [_short_label(name) for name in class_names]
    row_label_width = max([len("true\\pred")] + [len(name) for name in labels])
    formatted_rows = [[cell_format(cell) for cell in row] for row in matrix]
    cell_width = max(
        [max(len(name) for name in labels)]
        + [len(cell) for row in formatted_rows for cell in row]
    )
    header = (
        " " * row_label_width
        + " "
        + " ".join(name.rjust(cell_width) for name in labels)
    )
    lines = [header]
    for name, row in zip(labels, formatted_rows):
        lines.append(
            name.ljust(row_label_width)
            + " "
            + " ".join(cell.rjust(cell_width) for cell in row)
        )
    return "\n".join(lines)


def _print_confusion_matrices(
    split_name: str,
    metrics: Dict[str, object],
    idx_to_class: Dict[int, str],
    logger: logging.Logger,
):
    cm = metrics.get("confusion_matrix")
    cm_norm = metrics.get("confusion_matrix_normalized")
    if not isinstance(cm, list) or not isinstance(cm_norm, list):
        return
    num_classes = max(idx_to_class.keys()) + 1
    class_names = [idx_to_class.get(i, f"class_{i}") for i in range(num_classes)]

    logger.info(f"  Confusion matrix ({split_name}, rows=true, cols=pred):")
    logger.info("\n" + _format_confusion_table(cm, class_names, lambda v: str(int(v))))
    logger.info(f"  Normalized confusion ({split_name}, row-normalized):")
    logger.info(
        "\n" + _format_confusion_table(
            cm_norm, class_names, lambda v: f"{float(v):.2f}"
        )
    )


def _save_confusion_plots(
    run_dir: Path,
    split_name: str,
    metrics: Dict[str, object],
    idx_to_class: Dict[int, str],
    logger: logging.Logger,
):
    cm = metrics.get("confusion_matrix")
    cm_norm = metrics.get("confusion_matrix_normalized")
    if not isinstance(cm, list) or not isinstance(cm_norm, list):
        return
    num_classes = max(idx_to_class.keys()) + 1
    class_names = [idx_to_class.get(i, f"class_{i}") for i in range(num_classes)]

    raw_path = run_dir / f"confusion_{split_name}.png"
    norm_path = run_dir / f"confusion_{split_name}_normalized.png"
    render_confusion_heatmap(
        cm,
        class_names,
        raw_path,
        title=f"{split_name} confusion (counts)",
        normalized=False,
        logger=logger,
    )
    render_confusion_heatmap(
        cm_norm,
        class_names,
        norm_path,
        title=f"{split_name} confusion (row-normalized)",
        normalized=True,
        logger=logger,
    )


def _save_confusion_csv(
    run_dir: Path,
    *,
    stem: str,
    metrics: Dict[str, object],
    idx_to_class: Dict[int, str],
) -> None:
    cm = metrics.get("confusion_matrix")
    cm_norm = metrics.get("confusion_matrix_normalized")
    if not isinstance(cm, list) or not isinstance(cm_norm, list):
        return
    num_classes = max(idx_to_class.keys()) + 1
    class_names = [idx_to_class.get(i, f"class_{i}") for i in range(num_classes)]
    header = ["true\\pred"] + class_names

    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / f"{stem}_confusion.csv"
    norm_path = run_dir / f"{stem}_confusion_normalized.csv"

    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for name, row in zip(class_names, cm):
            writer.writerow([name] + [int(v) for v in row])

    with norm_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for name, row in zip(class_names, cm_norm):
            formatted = []
            for v in row:
                try:
                    formatted.append(f"{float(v):.6f}")
                except Exception:
                    formatted.append("")
            writer.writerow([name] + formatted)


def _save_per_class_csv(
    run_dir: Path,
    *,
    stem: str,
    metrics: Dict[str, object],
) -> None:
    per_class = metrics.get("per_class")
    if not isinstance(per_class, dict):
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"{stem}_per_class.csv"
    fields = ["class", "total", "correct", "accuracy", "precision", "recall", "f1"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, stats in per_class.items():
            if not isinstance(stats, dict):
                continue
            writer.writerow(
                {
                    "class": name,
                    "total": stats.get("total", ""),
                    "correct": stats.get("correct", ""),
                    "accuracy": stats.get("accuracy", ""),
                    "precision": stats.get("precision", ""),
                    "recall": stats.get("recall", ""),
                    "f1": stats.get("f1", ""),
                }
            )


def _save_metrics_json(run_dir: Path, *, filename: str, payload: Dict[str, object]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / filename
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    import sys
    
    # 1. Get the parser (defaults are system defaults from config.py)
    parser = get_arg_parser()
    
    # 2. Check for config file manually to set it as NEW defaults
    # This ensures that explicit CLI args (from GUI) override config values.
    # We do a partial scan for --config
    config_path = None
    if "--config" in sys.argv:
        try:
            idx = sys.argv.index("--config") + 1
            if idx < len(sys.argv):
                config_path = Path(sys.argv[idx])
        except ValueError:
            pass
            
    if config_path and config_path.exists():
        config_dict = load_config(config_path)
        apply_config_as_parser_defaults(parser, config_dict)
         
    # 3. Parse args (CLI overrides defaults)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        name="TrainBehaviorLSTM",
        log_file=run_dir / "train.log"
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    manifest_path = args.manifest_path
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. "
            "Run prepare_sequences_from_yolo.py first."
        )
    splits, class_to_idx, idx_to_class, metadata = load_sequence_manifest(
        manifest_path
    )
    setattr(args, "_manifest_meta", metadata)
    if args.num_frames is None:
        if "window_size" not in metadata:
            logger.warning(
                "Manifest metadata is missing window_size; defaulting --num_frames to 30. "
                "If this is unintended, re-run preprocessing or pass --num_frames explicitly."
            )
        args.num_frames = int(metadata.get("window_size", 30))
    if args.frame_size is None:
        if "crop_size" not in metadata:
            logger.warning(
                "Manifest metadata is missing crop_size; defaulting --frame_size to 224. "
                "If this is unintended, re-run preprocessing or pass --frame_size explicitly."
            )
        args.frame_size = int(metadata.get("crop_size", 224))

    validate_args(args, logger)
    logger.info(
        "Best checkpoints: best_model.pt (highest val_accuracy) and "
        "best_model_macro_f1.pt (highest val_macro_f1)."
    )

    backbone_name = str(args.backbone).lower()
    if backbone_name.startswith("slowfast"):
        logger.info(
            "Note: SlowFast is a video backbone; --sequence_model/attention/LSTM-related "
            "settings are ignored for SlowFast."
        )
    if backbone_name.startswith("vit") and int(args.frame_size) != 224:
        raise ValueError(
            "ViT backbones require 224x224 inputs in this implementation. "
            "Use preprocessing --crop_size 224 and training --frame_size 224 (or leave frame_size blank)."
        )

    train_loader, val_loader, test_loader = build_loaders(
        splits, args, device
    )
    if train_loader is None:
        raise RuntimeError("Training split is empty. Nothing to learn from.")

    if str(getattr(args, "train_sampler", "shuffle")).strip().lower() == "weighted":
        logger.info("Training sampler: weighted (WeightedRandomSampler inverse-frequency).")

    if not bool(getattr(args, "skip_startup_checks", False)):
        run_startup_sanity_checks(
            logger=logger,
            run_dir=run_dir,
            args=args,
            splits=splits,
            idx_to_class=idx_to_class,
        )

    # Check for pose availability
    pose_input_dim = 0
    if len(train_loader.dataset) > 0:
        # Check first few samples to find one with pose data
        # Accessing dataset[i] triggers augmentation, but that's fine for shape check
        check_limit = min(len(train_loader.dataset), 100)
        for i in range(check_limit):
            try:
                sample_data, _ = train_loader.dataset[i]
                if sample_data is None:
                    continue
                (clip_sample, pose_sample) = sample_data
                if pose_sample is not None:
                    # pose_sample shape is [T, FeatureDim]
                    pose_input_dim = pose_sample.shape[1]
                    logger.info(f"Found pose data in sample {i}. Input dim: {pose_input_dim}")
                    break
            except Exception:
                continue

    if args.use_pose:
        if pose_input_dim == 0:
            raise ValueError(
                "--use_pose was requested, but no keypoint data found in the training set. "
                "Did you run preprocessing with YOLO-pose weights?"
            )
        logger.info(f"Pose fusion enabled. Input dim: {pose_input_dim}")
    else:
        if pose_input_dim > 0:
            logger.info("Pose data detected but --use_pose not set. Ignoring pose features.")
        pose_input_dim = 0

    # [FIX] Explicitly inject runtime-determined dims into args so they are saved in hparams
    setattr(args, "pose_input_dim", pose_input_dim)
    # pose_fusion_dim is passed via args, but we ensure it's explicitly recorded if needed by legacy code
    # setattr(args, "pose_fusion_dim", args.pose_fusion_dim) # Already in args namespace

    # Save a resolved YAML config early so it can be reused for ablations.
    try:
        _write_resolved_config_yaml(run_dir, args, metadata, logger)
    except Exception as exc:  # pragma: no cover
        logger.warning(f"Could not write resolved config.yaml ({exc})")

    stop_file = args.stop_file if args.stop_file is not None else (run_dir / "stop_training.flag")
    if stop_file.exists():
        try:
            stop_file.unlink()
        except Exception:
            pass

    stop_state = {"requested": False}

    def request_stop(reason: str) -> None:
        if not stop_state["requested"]:
            logger.info(
                f"Stop requested ({reason}). Exporting validation metrics and artifacts before exiting..."
            )
        stop_state["requested"] = True

    def should_stop() -> bool:
        if stop_state["requested"]:
            return True
        if stop_file.exists():
            request_stop(f"stop_file={stop_file}")
            return True
        return False

    def _handle_signal(signum, _frame):  # pragma: no cover - signal wiring
        request_stop(f"signal {signum}")

    for sig_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _handle_signal)
        except Exception:
            continue

    history_path = run_dir / "history.json"
    metrics_csv_path = run_dir / "training_metrics.csv"
    metrics_csv_fields = [
        "epoch",
        "seconds",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "val_weighted_f1",
        "val_micro_f1",
        "lr",
        "improved",
        "best_val_accuracy",
        "best_val_macro_f1",
        "epochs_without_improve",
    ]
    history: List[Dict] = []
    if args.resume and history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except Exception:
            history = []

    model = BehaviorSequenceClassifier(
        num_classes=len(class_to_idx),
        backbone=args.backbone,
        pretrained_backbone=not args.no_pretrained_backbone,
        train_backbone=args.train_backbone,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        sequence_model=args.sequence_model,
        temporal_attention_layers=args.temporal_attention_layers,
        attention_heads=args.attention_heads,
        positional_encoding=args.positional_encoding,
        positional_encoding_max_len=args.positional_encoding_max_len,
        use_attention_pooling=args.attention_pool,
        use_feature_se=args.feature_se,
        slowfast_alpha=args.slowfast_alpha,
        slowfast_fusion_ratio=args.slowfast_fusion_ratio,
        slowfast_base_channels=args.slowfast_base_channels,
        pose_input_dim=pose_input_dim,
        pose_fusion_strategy=args.pose_fusion_strategy,
        pose_fusion_dim=args.pose_fusion_dim,
    ).to(device)

    # Differential Learning Rate Setup
    if args.backbone_lr is not None and args.train_backbone:
        logger.info(f"Using Differential Learning Rates: Backbone={args.backbone_lr}, Head={args.lr}")
        backbone_params = []
        head_params = []
        
        # Identify backbone parameters
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("frame_encoder"):
                backbone_params.append(param)
            else:
                head_params.append(param)
                
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": args.backbone_lr},
                {"params": head_params, "lr": args.lr},
            ],
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    
    # Scheduler Setup
    scheduler = None
    if args.scheduler == "plateau":
        logger.info("Using ReduceLROnPlateau scheduler.")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=args.patience // 2
        )
    elif args.scheduler == "cosine":
        logger.info("Using CosineAnnealingLR scheduler.")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )

    loss_weights = compute_class_weights_from_train_split(
        train_samples=splits.get("train", []),
        num_classes=len(class_to_idx),
        scheme=getattr(args, "class_weighting", "none"),
        idx_to_class=idx_to_class,
        logger=logger,
        max_weight=float(getattr(args, "class_weight_max", 0.0) or 0.0),
    )
    criterion = nn.CrossEntropyLoss(
        weight=(loss_weights.to(device) if loss_weights is not None else None)
    )
    
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_val_acc = -1.0
    best_val_macro_f1 = -1.0
    best_state = None
    best_state_macro_f1 = None
    epochs_without_improve = 0
    start_epoch = 1

    if args.resume:
        last_ckpt = run_dir / "last_checkpoint.pt"
        fallback_ckpt = run_dir / "best_model.pt"
        ckpt_path = last_ckpt if last_ckpt.exists() else fallback_ckpt
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Resume requested but no checkpoint found in {run_dir} "
                "(expected last_checkpoint.pt or best_model.pt)."
            )
        try:
            resume_state = torch.load(
                ckpt_path, map_location=device, weights_only=False
            )
        except TypeError:  # pragma: no cover - torch<2.0 compatibility
            resume_state = torch.load(ckpt_path, map_location=device)

        if "model_state" not in resume_state:
            raise ValueError(
                f"Checkpoint {ckpt_path} is missing 'model_state'; cannot resume."
            )
        model.load_state_dict(resume_state["model_state"])
        if "optimizer_state" in resume_state:
            try:
                optimizer.load_state_dict(resume_state["optimizer_state"])
            except Exception as exc:
                logger.warning(
                    f"Warning: unable to restore optimizer state ({exc}). Continuing with fresh optimizer."
                )
        if "scaler_state" in resume_state and scaler is not None:
            try:
                scaler.load_state_dict(resume_state["scaler_state"])
            except Exception as exc:
                logger.warning(
                    f"Warning: unable to restore scaler state ({exc})."
                )

        best_val_acc = float(resume_state.get("best_val_acc", best_val_acc))
        best_val_macro_f1 = float(resume_state.get("best_val_macro_f1", best_val_macro_f1))
        epochs_without_improve = int(
            resume_state.get("epochs_without_improve", epochs_without_improve)
        )

        ckpt_epoch = resume_state.get("epoch")
        if isinstance(ckpt_epoch, int):
            start_epoch = ckpt_epoch + 1
        elif history:
            try:
                start_epoch = max(int(row.get("epoch", 0)) for row in history) + 1
            except Exception:
                start_epoch = 1

        best_path = run_dir / "best_model.pt"
        if best_path.exists():
            try:
                best_state = torch.load(
                    best_path, map_location=device, weights_only=False
                )
            except TypeError:  # pragma: no cover
                best_state = torch.load(best_path, map_location=device)

        logger.info(
            f"Resuming from {ckpt_path} at epoch {start_epoch} "
            f"(best_val_acc={best_val_acc:.3f}, best_val_macro_f1={best_val_macro_f1:.4g}, "
            f"patience_counter={epochs_without_improve})."
        )

    for epoch in range(start_epoch, args.epochs + 1):
        if should_stop():
            break
        start = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            logger,
            scaler,
            log_interval=args.log_interval,
            should_stop=should_stop,
            mixup_alpha=args.mixup_alpha,
        )
        compute_confusion = bool(val_loader is not None) or bool(args.confusion_matrix) or should_stop()
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            idx_to_class=idx_to_class,
            compute_confusion=compute_confusion,
        )
        elapsed = time.time() - start

        metrics = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "seconds": elapsed,
        }
        history.append(metrics)
        try:
            history_path.write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

        val_acc_value = None
        acc_source = val_metrics if val_metrics is not None else train_metrics
        try:
            val_acc_value = float(acc_source.get("accuracy"))  # type: ignore[arg-type]
        except Exception:
            val_acc_value = None

        val_macro_f1_value = None
        if val_metrics is not None:
            try:
                val_macro_f1_value = float(val_metrics.get("macro_f1"))  # type: ignore[arg-type]
            except Exception:
                val_macro_f1_value = None

        improved_acc = (
            val_acc_value is not None
            and (best_val_acc < 0 or val_acc_value > best_val_acc)
        )
        improved_macro = (
            val_macro_f1_value is not None
            and (best_val_macro_f1 < 0 or val_macro_f1_value > best_val_macro_f1)
        )

        if improved_acc:
            best_val_acc = float(val_acc_value)
            best_state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict() if scaler else None,
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class,
                "hparams": vars(args),
                "best_val_acc": best_val_acc,
                "best_val_macro_f1": best_val_macro_f1,
            }
            if val_metrics:
                best_state["best_val_metrics"] = val_metrics
            torch.save(best_state, run_dir / "best_model.pt")

        if improved_macro:
            best_val_macro_f1 = float(val_macro_f1_value)
            best_state_macro_f1 = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict() if scaler else None,
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class,
                "hparams": vars(args),
                "best_val_acc": best_val_acc,
                "best_val_macro_f1": best_val_macro_f1,
            }
            if val_metrics:
                best_state_macro_f1["best_val_metrics"] = val_metrics
            torch.save(best_state_macro_f1, run_dir / "best_model_macro_f1.pt")

        improved = improved_acc or improved_macro
        if improved:
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        last_state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler else None,
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "hparams": vars(args),
            "best_val_acc": best_val_acc,
            "best_val_macro_f1": best_val_macro_f1,
            "epochs_without_improve": epochs_without_improve,
        }
        torch.save(last_state, run_dir / "last_checkpoint.pt")
        
        # Step Scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                step_loss = None
                if val_metrics:
                    try:
                        step_loss = float(val_metrics.get("loss"))  # type: ignore[arg-type]
                    except Exception:
                        step_loss = None
                if step_loss is None:
                    try:
                        step_loss = float(train_metrics.get("loss", float("nan")))
                    except Exception:
                        step_loss = float("nan")
                scheduler.step(step_loss)
            else:
                scheduler.step()

        val_acc_display = float("nan")
        if val_metrics is not None:
            try:
                val_acc_display = float(val_metrics.get("accuracy"))  # type: ignore[arg-type]
            except Exception:
                pass
        val_macro_f1_display = float("nan")
        if val_macro_f1_value is not None:
            val_macro_f1_display = float(val_macro_f1_value)

        msg = (
            f"Epoch {epoch:03d}/{args.epochs}: "
            f"Train Acc={train_metrics['accuracy']:.3f} | "
            f"Val Acc={val_acc_display:.3f} | "
            f"Val macro_f1={val_macro_f1_display:.3f} | "
            f"Best Acc={best_val_acc:.3f} Best macro_f1={best_val_macro_f1:.3f} (+{epochs_without_improve})"
        )
        if args.scheduler != "none":
            current_lrs = [group['lr'] for group in optimizer.param_groups]
            msg += f" | LRs={current_lrs}"
            
        logger.info(msg)

        val_loss = val_metrics.get("loss") if val_metrics else None
        _append_metrics_csv(
            metrics_csv_path,
            fieldnames=metrics_csv_fields,
            row={
                "epoch": epoch,
                "seconds": float(elapsed),
                "train_loss": float(train_metrics.get("loss", float("nan"))),
                "train_accuracy": float(train_metrics.get("accuracy", float("nan"))),
                "val_loss": float(val_loss) if val_loss is not None else "",
                "val_accuracy": float(val_metrics.get("accuracy")) if val_metrics and val_metrics.get("accuracy") is not None else (float(val_acc_value) if val_acc_value is not None else ""),
                "val_macro_f1": float(val_metrics.get("macro_f1")) if val_metrics and val_metrics.get("macro_f1") is not None else "",
                "val_weighted_f1": float(val_metrics.get("weighted_f1")) if val_metrics and val_metrics.get("weighted_f1") is not None else "",
                "val_micro_f1": float(val_metrics.get("micro_f1")) if val_metrics and val_metrics.get("micro_f1") is not None else "",
                "lr": float(optimizer.param_groups[0].get("lr", args.lr)),
                "improved": bool(improved),
                "best_val_accuracy": float(best_val_acc),
                "best_val_macro_f1": float(best_val_macro_f1),
                "epochs_without_improve": int(epochs_without_improve),
            },
        )

        logger.info(
            f"[{epoch:03d}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.3f} "
            f"val_acc={(val_metrics['accuracy'] if val_metrics else float('nan')):.3f} "
            f"time={elapsed:.1f}s"
        )

        if args.per_class_metrics and val_metrics and val_metrics.get("per_class"):
            per_class = val_metrics["per_class"]
            sortable = []
            for name, stats in per_class.items():
                total_count = int(stats.get("total", 0))
                if total_count <= 0:
                    continue
                acc = stats.get("accuracy")
                sortable.append((acc is None, float(acc) if acc is not None else 0.0, name, stats))
            sortable.sort()
            logger.info("  Per-class val accuracy (worst -> best):")
            for is_none, acc_val, name, stats in sortable:
                acc_str = "n/a" if is_none else f"{acc_val:.3f}"
                logger.info(
                    f"    {name}: acc={acc_str} ({stats['correct']}/{stats['total']})"
                )

        stopping_now = should_stop()
        if (
            (args.confusion_matrix or stopping_now)
            and val_metrics
            and val_metrics.get("confusion_matrix")
            and (
                stopping_now
                or args.confusion_matrix_interval == 1
                or epoch % args.confusion_matrix_interval == 0
                or improved
            )
        ):
            _print_confusion_matrices("val", val_metrics, idx_to_class, logger)

        if stopping_now:
            break

        if args.patience > 0 and epochs_without_improve >= args.patience:       
            logger.info("Early stopping triggered.")
            break

    history_path = run_dir / "history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    mapping_path = run_dir / "class_mapping.json"
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class,
            },
            f,
            indent=2,
        )

    if best_state and "model_state" in best_state:
        model.load_state_dict(best_state["model_state"])

    if val_loader is not None:
        best_val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            idx_to_class=idx_to_class,
            compute_confusion=True,
        )
    else:
        best_val_metrics = None
    best_epoch = (
        int(best_state.get("epoch"))
        if isinstance(best_state, dict) and isinstance(best_state.get("epoch"), int)
        else None
    )

    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        idx_to_class=idx_to_class,
        compute_confusion=args.confusion_matrix,
    )
    if test_metrics:
        logger.info(
            f"Test accuracy: {test_metrics['accuracy']:.3f}, "
            f"loss: {test_metrics['loss']:.4f}"
        )
        if args.per_class_metrics and test_metrics.get("per_class"):
            logger.info("Per-class test accuracy:")
            for name, stats in sorted(test_metrics["per_class"].items()):
                acc = stats.get("accuracy")
                acc_str = "n/a" if acc is None else f"{float(acc):.3f}"
                logger.info(f"  {name}: acc={acc_str} ({stats['correct']}/{stats['total']})")
        if args.confusion_matrix and test_metrics.get("confusion_matrix"):
            _print_confusion_matrices("test", test_metrics, idx_to_class, logger)
            _save_confusion_plots(run_dir, "test", test_metrics, idx_to_class, logger)

    if best_val_metrics and best_val_metrics.get("confusion_matrix"):
        best_val_report = dict(best_val_metrics)
        if best_epoch is not None:
            best_val_report["epoch"] = best_epoch
        _save_metrics_json(
            run_dir,
            filename="best_val_metrics.json",
            payload=best_val_report,
        )
        _save_confusion_csv(
            run_dir,
            stem="best_val",
            metrics=best_val_report,
            idx_to_class=idx_to_class,
        )
        _save_per_class_csv(run_dir, stem="best_val", metrics=best_val_report)
        if args.confusion_matrix or bool(stop_state.get("requested")):
            _save_confusion_plots(run_dir, "val", best_val_metrics, idx_to_class, logger)

    config_path = run_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    save_training_curves(run_dir, history, logger)

    logger.info(f"Training artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
