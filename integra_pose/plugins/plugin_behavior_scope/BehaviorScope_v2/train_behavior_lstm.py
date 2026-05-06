# Example:
# python train_behavior_lstm.py --manifest_path "processed_sequences/sequence_manifest.json" --output_dir "runs" --epochs 100 --batch_size 4 --device "cuda:0"

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":  # pragma: no cover - CLI entry
    SCRIPT_DIR = Path(__file__).resolve().parent
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.append(str(SCRIPT_DIR))
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
    repo_root = Path(__file__).resolve().parents[4]
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
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="none",
        choices=["none", "cosine"],
        help="Learning rate scheduler to apply.",
    )
    parser.add_argument(
        "--scheduler_tmax",
        type=int,
        default=None,
        help="Epoch horizon for the cosine scheduler (defaults to --epochs).",
    )
    parser.add_argument(
        "--scheduler_min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate reached by the cosine scheduler.",
    )
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
    torch.cuda.manual_seed_all(seed)


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
):
    if loader is None:
        return None

    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
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

    if total == 0:
        return None

    return {"loss": running_loss / total, "accuracy": correct / total}


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

    train_loader, val_loader, test_loader = build_loaders(
        splits, args, device
    )
    if train_loader is None:
        raise RuntimeError("Training split is empty. Nothing to learn from.")

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
    scheduler = None
    if args.lr_scheduler == "cosine":
        t_max = args.scheduler_tmax or args.epochs
        if t_max <= 0:
            raise ValueError(
                "scheduler_tmax must be positive when using the cosine scheduler."
            )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=args.scheduler_min_lr,
        )
    criterion = nn.CrossEntropyLoss()

    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    history: List[Dict] = []
    best_val_acc = -1.0
    best_state = None
    epochs_without_improve = 0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            log_interval=args.log_interval,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start

        metrics = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "seconds": elapsed,
        }
        history.append(metrics)

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
                "scheduler_state": scheduler.state_dict() if scheduler else None,
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class,
                "hparams": vars(args),
            }
            torch.save(best_state, run_dir / "best_model.pt")
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.3f} "
            f"val_acc={(val_metrics['accuracy'] if val_metrics else float('nan')):.3f} "
            f"time={elapsed:.1f}s"
        )

        if scheduler is not None:
            scheduler.step()

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

    test_metrics = evaluate(model, test_loader, criterion, device)
    if test_metrics:
        print(
            f"Test accuracy: {test_metrics['accuracy']:.3f}, "
            f"loss: {test_metrics['loss']:.4f}"
        )

    config_path = run_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print(f"Training artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
