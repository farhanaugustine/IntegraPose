# python train_behavior_lstm.py --manifest_path "C:\Users\Aegis-MSI\Documents\Lin_Lab_IntegraPose_Foundational_Model\Odor_Threshold\clips_for_labeling\behavior_lstm\TestProject\sequence_manifest.json" --output_dir "runs" --epochs 100 --batch_size 4 --device "cuda:0"

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":  # pragma: no cover - CLI entry
    from data import (
        SequenceDataset,
        collate_with_error_handling,
        load_sequence_manifest,
    )
    from model import BehaviorSequenceClassifier
else:
    from .data import (
        SequenceDataset,
        collate_with_error_handling,
        load_sequence_manifest,
    )
    from .model import BehaviorSequenceClassifier


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
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
        default=4,
        help="Number of attention heads for temporal blocks/pooling.",
    )
    parser.add_argument(
        "--attention_pool",
        action="store_true",
        help="Enable attention pooling over LSTM outputs.",
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
    parser.add_argument("--patience", type=int, default=10)
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
        default=4,
        help="Temporal stride between fast/slow pathways (slowfast only).",
    )
    parser.add_argument(
        "--slowfast_fusion_ratio",
        type=float,
        default=0.25,
        help="Channel ratio for the slow pathway (slowfast only).",
    )
    parser.add_argument(
        "--slowfast_base_channels",
        type=int,
        default=48,
        help="Base channel width for the slowfast backbone.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_args(args: argparse.Namespace):
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
            "Choose from ['mobilenet_v3_small', 'efficientnet_b0', 'vit_b_16', 'slowfast_tiny']."
        )

    uses_attention = (
        int(args.temporal_attention_layers) > 0
        or bool(args.attention_pool)
        or str(args.sequence_model).lower() == "attention"
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
        )

    train_dataset = _make_dataset(
        splits.get("train", []), augment=not args.disable_augment
    )
    val_dataset = _make_dataset(splits.get("val", []), augment=False)
    test_dataset = _make_dataset(splits.get("test", []), augment=False)

    def _make_loader(dataset, shuffle: bool):
        if dataset is None or len(dataset) == 0:
            return None
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_with_error_handling,
        )

    return (
        _make_loader(train_dataset, True),
        _make_loader(val_dataset, False),
        _make_loader(test_dataset, False),
    )


def train_one_epoch(
    model: BehaviorSequenceClassifier,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
    log_interval: int = 0,
):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    num_batches = len(loader)
    for batch_idx, batch in enumerate(loader, start=1):
        if batch is None:
            continue
        clips, labels = batch
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(clips)
        loss = criterion(logits, labels)
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
            print(
                f"    step {batch_idx:04d}/{num_batches:04d} "
                f"batch_loss={loss.item():.4f} batch_acc={batch_acc:.3f} "
                f"running_acc={correct / max(total, 1):.3f}",
                flush=True,
            )

    if total == 0:
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

    for batch in loader:
        if batch is None:
            continue
        clips, labels = batch
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(clips)
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
        metrics["per_class"] = per_class

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
):
    cm = metrics.get("confusion_matrix")
    cm_norm = metrics.get("confusion_matrix_normalized")
    if not isinstance(cm, list) or not isinstance(cm_norm, list):
        return
    num_classes = max(idx_to_class.keys()) + 1
    class_names = [idx_to_class.get(i, f"class_{i}") for i in range(num_classes)]

    print(f"  Confusion matrix ({split_name}, rows=true, cols=pred):")
    print(_format_confusion_table(cm, class_names, lambda v: str(int(v))))
    print(f"  Normalized confusion ({split_name}, row-normalized):")
    print(
        _format_confusion_table(
            cm_norm, class_names, lambda v: f"{float(v):.2f}"
        )
    )


def _render_confusion_heatmap(
    matrix: List[List[float | int]],
    class_names: List[str],
    out_path: Path,
    title: str,
    normalized: bool,
):
    try:  # pragma: no cover - visualization helper
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:  # pragma: no cover
        print(f"Warning: Pillow is required to save confusion plots ({exc}).")
        return

    def _text_size(draw: "ImageDraw.ImageDraw", text: str, font):
        if hasattr(draw, "textbbox"):
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            return right - left, bottom - top
        return draw.textsize(text, font=font)  # type: ignore[attr-defined]

    n = len(class_names)
    if n == 0:
        return

    labels = [_short_label(name, max_len=12) for name in class_names]
    dummy = Image.new("RGB", (10, 10), "white")
    dummy_draw = ImageDraw.Draw(dummy)
    font = ImageFont.load_default()

    padding = 6
    title_w, title_h = _text_size(dummy_draw, title, font)
    label_w = max(_text_size(dummy_draw, name, font)[0] for name in labels)
    label_h = max(_text_size(dummy_draw, name, font)[1] for name in labels)
    rotated_h = max(_text_size(dummy_draw, name, font)[0] for name in labels)

    if n <= 10:
        cell = 30
    elif n <= 20:
        cell = 24
    elif n <= 40:
        cell = 18
    else:
        cell = 14

    title_area_h = title_h + padding
    pred_area_h = label_h + padding
    xlabels_area_h = rotated_h + padding
    top_margin = title_area_h + pred_area_h + xlabels_area_h + padding
    left_margin = label_w + 2 * padding
    grid_left = left_margin
    grid_top = top_margin

    grid_size = cell * n
    width = grid_left + grid_size + padding
    height = grid_top + grid_size + padding

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Title + axis hints
    draw.text((padding, padding), title, fill="black", font=font)
    draw.text((grid_left, title_area_h), "Predicted", fill="black", font=font)
    draw.text((padding, grid_top - label_h - padding), "True", fill="black", font=font)

    # Column labels (rotated)
    xlabels_y = title_area_h + pred_area_h
    for j, name in enumerate(labels):
        x = grid_left + j * cell + cell // 2
        text_img = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
        text_draw = ImageDraw.Draw(text_img)
        tw, th = _text_size(text_draw, name, font)
        text_img = Image.new("RGBA", (tw, th), (255, 255, 255, 0))
        text_draw = ImageDraw.Draw(text_img)
        text_draw.text((0, 0), name, fill="black", font=font)
        rotated = text_img.rotate(90, expand=1)
        rx, ry = rotated.size
        img.paste(
            rotated,
            (int(x - rx / 2), int(xlabels_y + (rotated_h - ry))),
            rotated,
        )

    # Row labels
    for i, name in enumerate(labels):
        y = grid_top + i * cell + cell // 2 - label_h // 2
        draw.text((grid_left - padding - label_w, y), name, fill="black", font=font)

    # Color scaling
    max_val = 1.0 if normalized else 0.0
    if not normalized:
        for row in matrix:
            for val in row:
                try:
                    max_val = max(max_val, float(val))
                except Exception:
                    continue

    show_values = n <= 25
    for i in range(n):
        for j in range(n):
            try:
                raw_val = float(matrix[i][j])
            except Exception:
                raw_val = 0.0
            if normalized:
                v = max(0.0, min(1.0, raw_val))
            else:
                v = raw_val / max(max_val, 1.0)
                v = v**0.5
            shade = int(255 * (1.0 - v))
            fill = (shade, shade, 255)
            x0 = grid_left + j * cell
            y0 = grid_top + i * cell
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle([x0, y0, x1, y1], fill=fill, outline=(200, 200, 200))

            if show_values:
                text = f"{raw_val:.2f}" if normalized else str(int(raw_val))
                tw, th = _text_size(draw, text, font)
                text_color = "white" if v > 0.55 else "black"
                draw.text(
                    (x0 + (cell - tw) / 2, y0 + (cell - th) / 2),
                    text,
                    fill=text_color,
                    font=font,
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def _save_confusion_plots(
    run_dir: Path,
    split_name: str,
    metrics: Dict[str, object],
    idx_to_class: Dict[int, str],
):
    cm = metrics.get("confusion_matrix")
    cm_norm = metrics.get("confusion_matrix_normalized")
    if not isinstance(cm, list) or not isinstance(cm_norm, list):
        return
    num_classes = max(idx_to_class.keys()) + 1
    class_names = [idx_to_class.get(i, f"class_{i}") for i in range(num_classes)]

    raw_path = run_dir / f"confusion_{split_name}.png"
    norm_path = run_dir / f"confusion_{split_name}_normalized.png"
    _render_confusion_heatmap(
        cm,
        class_names,
        raw_path,
        title=f"{split_name} confusion (counts)",
        normalized=False,
    )
    _render_confusion_heatmap(
        cm_norm,
        class_names,
        norm_path,
        title=f"{split_name} confusion (row-normalized)",
        normalized=True,
    )


def main():
    args = parse_args()
    set_seed(args.seed)

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
    if args.num_frames is None:
        args.num_frames = int(metadata.get("window_size", 30))
    if args.frame_size is None:
        args.frame_size = int(metadata.get("crop_size", 224))

    validate_args(args)

    backbone_name = str(args.backbone).lower()
    if backbone_name.startswith("slowfast"):
        print(
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

    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    history_path = run_dir / "history.json"
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
        use_attention_pooling=args.attention_pool,
        use_feature_se=args.feature_se,
        slowfast_alpha=args.slowfast_alpha,
        slowfast_fusion_ratio=args.slowfast_fusion_ratio,
        slowfast_base_channels=args.slowfast_base_channels,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None
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
                print(
                    f"Warning: unable to restore optimizer state ({exc}). Continuing with fresh optimizer."
                )

        best_val_acc = float(resume_state.get("best_val_acc", best_val_acc))
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

        print(
            f"Resuming from {ckpt_path} at epoch {start_epoch} "
            f"(best_val_acc={best_val_acc:.3f}, patience_counter={epochs_without_improve})."
        )

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            log_interval=args.log_interval,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            idx_to_class=idx_to_class,
            compute_confusion=args.confusion_matrix,
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

        if val_metrics:
            val_acc = val_metrics["accuracy"]
            improved = val_acc > best_val_acc
        else:
            improved = train_metrics["accuracy"] > best_val_acc
            val_acc = train_metrics["accuracy"]

        if improved:
            best_val_acc = val_acc
            best_state = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class,
                "hparams": vars(args),
            }
            torch.save(best_state, run_dir / "best_model.pt")
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        last_state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "hparams": vars(args),
            "best_val_acc": best_val_acc,
            "epochs_without_improve": epochs_without_improve,
        }
        torch.save(last_state, run_dir / "last_checkpoint.pt")

        print(
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
            print("  Per-class val accuracy (worst -> best):")
            for is_none, acc_val, name, stats in sortable:
                acc_str = "n/a" if is_none else f"{acc_val:.3f}"
                print(
                    f"    {name}: acc={acc_str} ({stats['correct']}/{stats['total']})"
                )

        if (
            args.confusion_matrix
            and val_metrics
            and val_metrics.get("confusion_matrix")
            and (
                args.confusion_matrix_interval == 1
                or epoch % args.confusion_matrix_interval == 0
                or improved
            )
        ):
            _print_confusion_matrices("val", val_metrics, idx_to_class)

        if args.patience > 0 and epochs_without_improve >= args.patience:
            print("Early stopping triggered.")
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

    if args.confusion_matrix and val_loader is not None:
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

    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        idx_to_class=idx_to_class,
        compute_confusion=args.confusion_matrix,
    )
    if test_metrics:
        print(
            f"Test accuracy: {test_metrics['accuracy']:.3f}, "
            f"loss: {test_metrics['loss']:.4f}"
        )
        if args.per_class_metrics and test_metrics.get("per_class"):
            print("Per-class test accuracy:")
            for name, stats in sorted(test_metrics["per_class"].items()):
                acc = stats.get("accuracy")
                acc_str = "n/a" if acc is None else f"{float(acc):.3f}"
                print(f"  {name}: acc={acc_str} ({stats['correct']}/{stats['total']})")
        if args.confusion_matrix and test_metrics.get("confusion_matrix"):
            _print_confusion_matrices("test", test_metrics, idx_to_class)
            _save_confusion_plots(run_dir, "test", test_metrics, idx_to_class)

    if (
        args.confusion_matrix
        and best_val_metrics
        and best_val_metrics.get("confusion_matrix")
    ):
        _save_confusion_plots(run_dir, "val", best_val_metrics, idx_to_class)

    config_path = run_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print(f"Training artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
