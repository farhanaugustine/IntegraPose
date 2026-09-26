"""Dataset schema and export checks shared by the augmentation workflow."""

from pathlib import Path
import math

import cv2
import yaml


def load_metadata(root: Path) -> dict:
    candidates = [root / "data.yaml", root / "dataset.yaml"]
    existing = [p for p in candidates if p.is_file()]
    if not existing:
        existing = sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml"))
    if len(existing) > 1:
        raise ValueError("Keep one dataset YAML in the input root (data.yaml is recommended).")
    if not existing:
        return {}
    metadata = yaml.safe_load(existing[0].read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Dataset YAML must contain a mapping.")
    names = metadata.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names or any(type(k) is not int for k in names):
        raise ValueError("Dataset YAML needs class names indexed by integer IDs.")
    if sorted(names) != list(range(len(names))):
        raise ValueError("Class IDs must be contiguous starting at zero.")
    metadata["names"] = names
    for split in ("train", "val", "test"):
        value = metadata.get(split)
        if value and (not isinstance(value, str) or
                      value.replace("\\", "/").strip("./") != f"images/{split}" and
                      Path(value).resolve() != (root / "images" / split).resolve()):
            raise ValueError(f"This exporter requires images/{split}, not lists or external split paths.")
    shape = metadata.get("kpt_shape")
    if shape is not None:
        if not isinstance(shape, list) or len(shape) != 2 or type(shape[0]) is not int or shape[0] < 1 or shape[1] != 3:
            raise ValueError("Pose augmentation requires kpt_shape: [number_of_points, 3].")
        flip = metadata.get("flip_idx", [])
        if flip and (not isinstance(flip, list) or any(type(i) is not int for i in flip) or
                     sorted(flip) != list(range(shape[0])) or any(flip[flip[i]] != i for i in range(len(flip)))):
            raise ValueError("flip_idx must be an involutive permutation of all keypoint indices.")
    return metadata


def validate_samples(samples, metadata: dict) -> dict:
    """Reject malformed/mixed schemas instead of silently exporting bbox-only rows."""
    counts = {len(obj.keypoints) for sample in samples for obj in sample.objects}
    if len(counts) > 1:
        raise ValueError("All objects must have the same keypoint count; mixed bbox/pose rows are not supported.")
    count = next(iter(counts), 0)
    shape = metadata.get("kpt_shape")
    if count and not shape:
        raise ValueError("Pose labels require a dataset YAML with kpt_shape and named classes.")
    if shape and count != shape[0]:
        raise ValueError(f"Expected {shape[0]} keypoints per object, found {count}.")
    classes = {obj.class_id for sample in samples for obj in sample.objects}
    if not metadata:
        metadata = {"names": {i: str(i) for i in range(max(classes, default=0) + 1)}}
    if not classes <= set(metadata["names"]):
        raise ValueError("A label class is not present in the dataset YAML.")
    return metadata


def validate_output(root: Path, metadata: dict, *, progress=None) -> dict:
    """Read back every image/label pair and enforce the declared YOLO schema."""
    report = {}
    point_count = metadata.get("kpt_shape", [0, 3])[0]
    pixel_splits = {}
    import hashlib

    for image_dir in sorted((root / "images").iterdir()):
        if not image_dir.is_dir():
            continue
        label_dir = root / "labels" / image_dir.name
        images = sorted(image_dir.iterdir())
        if len({p.stem for p in images}) != len(images):
            raise ValueError(f"Duplicate image stems in {image_dir.name}.")
        if {p.stem for p in images} != {p.stem for p in label_dir.glob("*.txt")}:
            raise ValueError(f"Image/label pairing mismatch in {image_dir.name}.")
        backgrounds = 0
        for image_index, path in enumerate(images, 1):
            if progress and image_index % 1000 == 0:
                progress(f"Validating {image_dir.name}: {image_index:,}/{len(images):,} image/label pairs")
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Unreadable exported image: {path}")
            digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
            previous = pixel_splits.setdefault(digest, image_dir.name)
            if previous != image_dir.name:
                raise ValueError(f"Identical image pixels cross splits: {path}")
            lines = (label_dir / f"{path.stem}.txt").read_text(encoding="utf-8").splitlines()
            backgrounds += not lines
            for line in lines:
                values = [float(x) for x in line.split()]
                if len(values) != 5 + 3 * point_count or not all(math.isfinite(x) for x in values):
                    raise ValueError(f"Invalid label structure: {path.stem}")
                if values[0] != int(values[0]) or int(values[0]) not in metadata["names"]:
                    raise ValueError(f"Invalid class: {path.stem}")
                cx, cy, bw, bh = values[1:5]
                if not (bw > 0 and bh > 0 and 0 <= cx-bw/2+1e-6 and cx+bw/2 <= 1+1e-6 and
                        0 <= cy-bh/2+1e-6 and cy+bh/2 <= 1+1e-6):
                    raise ValueError(f"Invalid bounding box: {path.stem}")
                for i in range(5, len(values), 3):
                    x, y, visibility = values[i:i+3]
                    if not (0 <= x <= 1 and 0 <= y <= 1 and visibility in (0, 1, 2)):
                        raise ValueError(f"Invalid keypoint: {path.stem}")
        report[image_dir.name] = {"images": len(images), "backgrounds": backgrounds}
    return report


def write_metadata(root: Path, metadata: dict) -> None:
    # Preserve pose names/mappings and other dataset metadata, update only paths.
    updated = dict(metadata, path=root.resolve().as_posix(), train="images/train", val="images/val")
    if (root / "images" / "test").is_dir():
        updated["test"] = "images/test"
    else:
        updated.pop("test", None)
    updated.pop("download", None)
    (root / "data.yaml").write_text(yaml.safe_dump(updated, sort_keys=False), encoding="utf-8")


def source_provenance(root: Path) -> dict:
    """Validate optional recording groups and retain original source identifiers."""
    import csv

    path = root / "manifest.csv"
    records = {}
    groups = {}
    if not path.is_file():
        return records
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            image = row.get("image", "").replace("\\", "/")
            split = row.get("split", "")
            group = row.get("source_group", "")
            if image and split:
                if image in records:
                    raise ValueError(f"Duplicate image in source manifest: {image}")
                records[image] = row
            if group and split and groups.setdefault(group, split) != split:
                raise ValueError(f"Recording group crosses source splits: {group}")
    return records

