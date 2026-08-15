from __future__ import annotations

from pathlib import Path

import pytest

from integra_pose.data_preprocessing import dataset_split


def _write_pair(image_dir: Path, label_dir: Path, stem: str) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / f"{stem}.jpg").write_bytes(f"image-{stem}".encode("ascii"))
    (label_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")


def test_split_rejects_sources_that_alias_output_directories(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    image_dir = root / "images" / "train"
    label_dir = root / "labels" / "train"
    _write_pair(image_dir, label_dir, "frame_001")
    _write_pair(image_dir, label_dir, "frame_002")
    before_image = (image_dir / "frame_001.jpg").read_bytes()

    with pytest.raises(ValueError, match="Unsafe split paths"):
        dataset_split.create_yolo_train_val_split(
            image_dir=str(image_dir),
            label_dir=str(label_dir),
            dataset_root=str(root),
            clear_existing=True,
        )

    assert (image_dir / "frame_001.jpg").read_bytes() == before_image
    assert (label_dir / "frame_001.txt").is_file()


def test_split_rejects_source_parent_of_output(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    image_dir = root / "images"
    label_dir = tmp_path / "flat_labels"
    _write_pair(image_dir, label_dir, "frame_001")
    _write_pair(image_dir, label_dir, "frame_002")

    with pytest.raises(ValueError, match="overlaps output"):
        dataset_split.create_yolo_train_val_split(
            image_dir=str(image_dir),
            label_dir=str(label_dir),
            dataset_root=str(root),
            clear_existing=True,
        )


def test_failed_staging_preserves_existing_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "dataset"
    image_dir = tmp_path / "flat_images"
    label_dir = tmp_path / "flat_labels"
    _write_pair(image_dir, label_dir, "frame_001")
    _write_pair(image_dir, label_dir, "frame_002")
    existing = root / "images" / "train" / "existing.jpg"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_bytes(b"existing-split")

    real_copy2 = dataset_split.shutil.copy2
    copy_count = 0

    def _fail_second_copy(source, destination, *args, **kwargs):
        nonlocal copy_count
        copy_count += 1
        if copy_count == 2:
            raise OSError("simulated copy failure")
        return real_copy2(source, destination, *args, **kwargs)

    monkeypatch.setattr(dataset_split.shutil, "copy2", _fail_second_copy)

    with pytest.raises(OSError, match="simulated copy failure"):
        dataset_split.create_yolo_train_val_split(
            image_dir=str(image_dir),
            label_dir=str(label_dir),
            dataset_root=str(root),
            clear_existing=True,
        )

    assert existing.read_bytes() == b"existing-split"


def test_split_commits_complete_train_and_validation_directories(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    image_dir = tmp_path / "flat_images"
    label_dir = tmp_path / "flat_labels"
    for idx in range(4):
        _write_pair(image_dir, label_dir, f"frame_{idx:03d}")

    result = dataset_split.create_yolo_train_val_split(
        image_dir=str(image_dir),
        label_dir=str(label_dir),
        dataset_root=str(root),
        val_fraction=0.25,
        seed=7,
    )

    assert result.train_count == 3
    assert result.val_count == 1
    assert len(list((root / "images" / "train").glob("*.jpg"))) == 3
    assert len(list((root / "images" / "val").glob("*.jpg"))) == 1
    assert len(list((root / "labels" / "train").glob("*.txt"))) == 3
    assert len(list((root / "labels" / "val").glob("*.txt"))) == 1
