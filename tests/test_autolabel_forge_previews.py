from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import cv2
import numpy as np


def _load_module(rel_path: str, module_name: str):
    repo_root = Path(__file__).resolve().parent.parent
    module_path = repo_root / rel_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generate_preview_overlays_creates_split_previews(tmp_path: Path) -> None:
    module = _load_module(
        "integra_pose/plugins/plugin_autolabel_forge/autolabel_runtime.py",
        "test_autolabel_forge_preview_runtime_split",
    )
    dataset_dir = tmp_path / "dataset"
    train_images = dataset_dir / "train" / "images"
    train_labels = dataset_dir / "train" / "labels"
    valid_images = dataset_dir / "valid" / "images"
    valid_labels = dataset_dir / "valid" / "labels"
    train_images.mkdir(parents=True)
    train_labels.mkdir(parents=True)
    valid_images.mkdir(parents=True)
    valid_labels.mkdir(parents=True)

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(train_images / "sample.jpg"), image)
    cv2.imwrite(str(valid_images / "empty.jpg"), image)
    (train_labels / "sample.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    (valid_labels / "empty.txt").write_text("", encoding="utf-8")

    logged: list[str] = []
    count = module.generate_preview_overlays(dataset_dir, ["mouse"], logged.append)

    assert count == 2
    assert (dataset_dir / "previews" / "train" / "sample.jpg").exists()
    assert (dataset_dir / "previews" / "valid" / "empty.jpg").exists()
    assert any("Generated 1 preview image(s) for train" in message for message in logged)


def test_generate_preview_overlays_creates_flat_previews(tmp_path: Path) -> None:
    module = _load_module(
        "integra_pose/plugins/plugin_autolabel_forge/autolabel_runtime.py",
        "test_autolabel_forge_preview_runtime_flat",
    )
    dataset_dir = tmp_path / "dataset"
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    image = np.zeros((80, 120, 3), dtype=np.uint8)
    cv2.imwrite(str(images_dir / "frame_001.jpg"), image)
    (labels_dir / "frame_001.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    logged: list[str] = []
    count = module.generate_preview_overlays(dataset_dir, ["mouse"], logged.append)

    assert count == 1
    assert (dataset_dir / "previews" / "images" / "frame_001.jpg").exists()
    assert any("Generated 1 preview image(s) for images" in message for message in logged)


def test_generate_preview_overlays_respects_global_limit(tmp_path: Path) -> None:
    module = _load_module(
        "integra_pose/plugins/plugin_autolabel_forge/autolabel_runtime.py",
        "test_autolabel_forge_preview_runtime_limit",
    )
    dataset_dir = tmp_path / "dataset"
    train_images = dataset_dir / "train" / "images"
    train_labels = dataset_dir / "train" / "labels"
    train_images.mkdir(parents=True)
    train_labels.mkdir(parents=True)

    image = np.zeros((40, 40, 3), dtype=np.uint8)
    for idx in range(5):
        image_path = train_images / f"frame_{idx:03d}.jpg"
        cv2.imwrite(str(image_path), image)
        (train_labels / f"frame_{idx:03d}.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")

    logged: list[str] = []
    count = module.generate_preview_overlays(dataset_dir, ["mouse"], logged.append, max_preview_images=2)

    preview_paths = sorted((dataset_dir / "previews" / "train").glob("*.jpg"))
    assert count == 2
    assert len(preview_paths) == 2
    assert any("Generated 2 preview image(s) for train" in message for message in logged)
