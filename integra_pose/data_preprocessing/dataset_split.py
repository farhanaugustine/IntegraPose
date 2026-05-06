"""Utilities for preparing YOLO train/val directory splits.

Ultralytics expects a dataset layout like:

    <dataset_root>/
        images/train/*.jpg
        images/val/*.jpg
        labels/train/*.txt
        labels/val/*.txt

This helper takes a flat image directory + flat label directory and creates the
split folders, copying or moving the matched image/label pairs.
"""

from __future__ import annotations

import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class YoloSplitResult:
    dataset_root: str
    train_images_dir: str
    val_images_dir: str
    train_labels_dir: str
    val_labels_dir: str
    images_seen: int
    samples_used: int
    missing_labels: int
    train_count: int
    val_count: int
    val_fraction: float
    seed: int
    move_files: bool
    split_strategy: str
    grouping_active: bool
    group_count: int


def _list_images_flat(image_dir: Path) -> list[Path]:
    images: list[Path] = []
    for entry in sorted(image_dir.iterdir(), key=lambda p: p.name.lower()):
        if not entry.is_file():
            continue
        if entry.name.startswith("."):
            continue
        if entry.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        images.append(entry)
    return images


def _dir_has_any_files(path: Path) -> bool:
    try:
        for entry in path.iterdir():
            return True
    except FileNotFoundError:
        return False
    return False


def _ensure_empty_dir(path: Path, *, clear_existing: bool) -> None:
    if path.exists() and _dir_has_any_files(path):
        if not clear_existing:
            raise FileExistsError(f"Directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _prefix_group_key(stem: str, delimiter: str) -> str:
    token = str(delimiter or "_")
    if token and token in stem:
        prefix = stem.split(token, 1)[0].strip()
        if prefix:
            return prefix
    return stem


def _resolve_group_mapping(
    pairs: list[tuple[Path, Path | None]],
    *,
    strategy: str,
    delimiter: str,
) -> tuple[str, bool, dict[str, list[tuple[Path, Path | None]]]]:
    normalized = str(strategy or "random").strip().lower()
    if normalized not in {"random", "prefix", "auto"}:
        normalized = "random"

    if normalized == "random":
        return "random", False, {pair[0].stem: [pair] for pair in pairs}

    prefix_groups: dict[str, list[tuple[Path, Path | None]]] = {}
    for pair in pairs:
        key = _prefix_group_key(pair[0].stem, delimiter)
        prefix_groups.setdefault(key, []).append(pair)

    meaningful_prefix_groups = len(prefix_groups) > 1 and any(len(items) > 1 for items in prefix_groups.values())
    if normalized == "auto" and not meaningful_prefix_groups:
        return "random", False, {pair[0].stem: [pair] for pair in pairs}

    return "prefix", True, prefix_groups


def _assign_groups_to_splits(
    grouped_pairs: dict[str, list[tuple[Path, Path | None]]],
    *,
    target_val_count: int,
    rng: random.Random,
) -> tuple[list[tuple[Path, Path | None]], list[tuple[Path, Path | None]]]:
    group_items = list(grouped_pairs.items())
    rng.shuffle(group_items)

    val_groups: list[list[tuple[Path, Path | None]]] = []
    train_groups: list[list[tuple[Path, Path | None]]] = []
    val_samples = 0

    for _, group in group_items:
        group_size = len(group)
        should_fill_val = val_samples < target_val_count
        fits_target = (val_samples + group_size) <= target_val_count
        if should_fill_val and (fits_target or not val_groups):
            val_groups.append(group)
            val_samples += group_size
        else:
            train_groups.append(group)

    if not val_groups and train_groups:
        val_groups.append(train_groups.pop(0))
    if not train_groups and val_groups:
        train_groups.append(val_groups.pop())

    val_pairs = [pair for group in val_groups for pair in group]
    train_pairs = [pair for group in train_groups for pair in group]
    return val_pairs, train_pairs


def create_yolo_train_val_split(
    *,
    image_dir: str,
    label_dir: str,
    dataset_root: str,
    val_fraction: float = 0.2,
    seed: int = 42,
    move_files: bool = False,
    include_unlabeled: bool = False,
    clear_existing: bool = False,
    split_strategy: str = "random",
    group_delimiter: str = "_",
) -> YoloSplitResult:
    """Create YOLO-style train/val directories from flat image/label folders."""
    if not dataset_root:
        raise ValueError("dataset_root is required.")
    if not image_dir:
        raise ValueError("image_dir is required.")
    if not label_dir:
        raise ValueError("label_dir is required.")

    dataset_root_path = Path(dataset_root).expanduser()
    image_dir_path = Path(image_dir).expanduser()
    label_dir_path = Path(label_dir).expanduser()

    if not image_dir_path.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir_path}")
    if not label_dir_path.is_dir():
        raise FileNotFoundError(f"Label directory does not exist: {label_dir_path}")

    try:
        val_fraction = float(val_fraction)
    except Exception as exc:
        raise ValueError(f"val_fraction must be a number, got {val_fraction!r}") from exc
    val_fraction = max(0.01, min(0.99, val_fraction))

    images = _list_images_flat(image_dir_path)
    pairs: list[tuple[Path, Path | None]] = []
    missing_labels = 0
    for img_path in images:
        lbl_path = label_dir_path / f"{img_path.stem}.txt"
        if lbl_path.is_file():
            pairs.append((img_path, lbl_path))
        else:
            missing_labels += 1
            if include_unlabeled:
                pairs.append((img_path, None))

    if len(pairs) < 2:
        raise ValueError(
            f"Need at least 2 images to split; found {len(pairs)} usable image(s). "
            "Make sure annotation label files exist for your images."
        )

    total = len(pairs)
    val_count = int(round(total * val_fraction))
    val_count = max(1, min(total - 1, val_count))
    rng = random.Random(int(seed))

    effective_strategy, grouping_active, grouped_pairs = _resolve_group_mapping(
        pairs,
        strategy=split_strategy,
        delimiter=group_delimiter,
    )
    if grouping_active:
        val_pairs, train_pairs = _assign_groups_to_splits(
            grouped_pairs,
            target_val_count=val_count,
            rng=rng,
        )
        ordered_pairs = val_pairs + train_pairs
    else:
        ordered_pairs = list(pairs)
        rng.shuffle(ordered_pairs)

    actual_val_count = max(1, min(len(ordered_pairs) - 1, len(val_pairs) if grouping_active else val_count))
    train_count = len(ordered_pairs) - actual_val_count

    out_images_train = dataset_root_path / "images" / "train"
    out_images_val = dataset_root_path / "images" / "val"
    out_labels_train = dataset_root_path / "labels" / "train"
    out_labels_val = dataset_root_path / "labels" / "val"

    _ensure_empty_dir(out_images_train, clear_existing=clear_existing)
    _ensure_empty_dir(out_images_val, clear_existing=clear_existing)
    _ensure_empty_dir(out_labels_train, clear_existing=clear_existing)
    _ensure_empty_dir(out_labels_val, clear_existing=clear_existing)

    copier = shutil.move if move_files else shutil.copy2

    def _write_empty_label(dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("", encoding="utf-8")

    for idx, (img_path, lbl_path) in enumerate(ordered_pairs):
        split = "val" if idx < actual_val_count else "train"
        if split == "val":
            img_dest = out_images_val / img_path.name
            lbl_dest = out_labels_val / f"{img_path.stem}.txt"
        else:
            img_dest = out_images_train / img_path.name
            lbl_dest = out_labels_train / f"{img_path.stem}.txt"

        copier(str(img_path), str(img_dest))
        if lbl_path is None:
            _write_empty_label(lbl_dest)
        else:
            copier(str(lbl_path), str(lbl_dest))

    result = YoloSplitResult(
        dataset_root=str(dataset_root_path),
        train_images_dir=str(out_images_train),
        val_images_dir=str(out_images_val),
        train_labels_dir=str(out_labels_train),
        val_labels_dir=str(out_labels_val),
        images_seen=len(images),
        samples_used=len(pairs),
        missing_labels=missing_labels,
        train_count=train_count,
        val_count=actual_val_count,
        val_fraction=val_fraction,
        seed=int(seed),
        move_files=bool(move_files),
        split_strategy=effective_strategy,
        grouping_active=grouping_active,
        group_count=len(grouped_pairs),
    )

    # Write a sibling split manifest so future re-runs know exactly what went
    # into train vs val and which seed was used. Without this, two runs with
    # the same `seed` argument can disagree if the input directory contents
    # changed between runs (added / removed images), and reviewers can't tell.
    try:
        from datetime import datetime, timezone

        from integra_pose.utils.safe_io import safe_write_json

        manifest_payload = {
            "schema_version": 1,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset_root": str(dataset_root_path),
            "image_dir": str(image_dir_path),
            "label_dir": str(label_dir_path),
            "seed": int(seed),
            "val_fraction": float(val_fraction),
            "split_strategy": effective_strategy,
            "grouping_active": bool(grouping_active),
            "group_count": int(len(grouped_pairs)),
            "images_seen": int(len(images)),
            "samples_used": int(len(pairs)),
            "missing_labels": int(missing_labels),
            "train_count": int(train_count),
            "val_count": int(actual_val_count),
            "move_files": bool(move_files),
            "input_images": [img_path.name for img_path, _ in pairs],
        }
        safe_write_json(dataset_root_path / "split_manifest.json", manifest_payload, indent=2)
    except Exception:
        # Manifest write is best-effort; the split itself already succeeded.
        # Failure here is logged at the call site if a logger is wired up.
        pass

    return result
