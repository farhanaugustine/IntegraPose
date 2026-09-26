from __future__ import annotations

import csv
import inspect
import json
import math
import random
import shutil
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import cv2
import numpy as np

from .dataset_io import load_metadata, validate_samples, validate_output, write_metadata, source_provenance


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
BBOX_EPSILON = 1e-6


class DatasetAugmentorCoreError(RuntimeError):
    """Raised when the augmentation engine cannot complete a requested run."""


@dataclass(frozen=True)
class DomainShiftSettings:
    white_mouse: bool = False
    invert_intensity: float = 1.0
    red_light: bool = False
    red_boost: float = 1.5
    bg_suppress: float = 0.2
    bw: bool = False
    bw_prob: float = 0.5


@dataclass(frozen=True)
class ArtifactSettings:
    frame_noise_prob: float = 0.0
    frame_noise_std: float = 12.0
    frame_banding_prob: float = 0.0
    frame_banding_amp: float = 12.0
    regional_noise_prob: float = 0.0
    regional_noise_std: float = 28.0
    occlusion_prob: float = 0.0
    occlusion_count: int = 2
    occlusion_min_frac: float = 0.04
    occlusion_max_frac: float = 0.18
    occlusion_mode: str = "random"


@dataclass(frozen=True)
class TransformSettings:
    horizontal_flip_prob: float = 0.5
    affine_prob: float = 0.9
    scale_min: float = 0.90
    scale_max: float = 1.10
    translate_percent: float = 0.08
    rotate_degrees: float = 12.0
    shear_degrees: float = 6.0
    brightness_contrast_prob: float = 0.5
    hue_saturation_prob: float = 0.4
    blur_prob: float = 0.15
    gauss_noise_prob: float = 0.2


@dataclass(frozen=True)
class AugmentationRecipe:
    dataset_root: Path
    output_root: Optional[Path] = None
    split: str = "train"
    copy_originals: bool = True
    seed: int = 42
    plan: str = "add"
    add_count: int = 1000
    target_total_images: int = 40000
    target_ratio: float = 1.0
    add_per_class: int = 0
    multiplier: float = 1.0
    include_classes: str = "all"
    exclude_classes: str = ""
    preview_count: int = 50
    max_total_augs: int = 20000
    max_augs_per_source_image: int = 200
    min_visibility: float = 0.25
    domain_shift: DomainShiftSettings = DomainShiftSettings()
    artifacts: ArtifactSettings = ArtifactSettings()
    transforms: TransformSettings = TransformSettings()


@dataclass(frozen=True)
class LabelObject:
    class_id: int
    bbox: list[float]
    keypoints: list[tuple[float, float, int]]


@dataclass(frozen=True)
class SourceSample:
    image_path: Path
    label_path: Path
    split: str
    objects: list[LabelObject]


@dataclass(frozen=True)
class AugmentationResult:
    output_root: Path
    augmented_samples: int
    original_counts: dict[int, int]
    target_counts: dict[int, int]
    remaining_need: dict[int, int]
    manifest_path: Path
    recipe_path: Path
    preview_dir: Path


ProgressCallback = Callable[[str], None]


def parse_class_set(value: str) -> Optional[set[int]]:
    raw = str(value or "").strip().lower()
    if raw in {"", "none"}:
        return set()
    if raw == "all":
        return None
    try:
        return {int(part.strip()) for part in raw.split(",") if part.strip()}
    except ValueError as exc:
        raise DatasetAugmentorCoreError(
            "Class filters must be 'all', blank, or comma-separated integer IDs."
        ) from exc


def sanitize_yolo_bbox(cx: float, cy: float, bw: float, bh: float) -> list[float]:
    eps = BBOX_EPSILON
    cx = float(np.clip(cx, 0.0, 1.0))
    cy = float(np.clip(cy, 0.0, 1.0))
    bw = float(np.clip(bw, 0.0, 1.0))
    bh = float(np.clip(bh, 0.0, 1.0))

    x1 = max(0.0, min(1.0, cx - (bw / 2.0)))
    y1 = max(0.0, min(1.0, cy - (bh / 2.0)))
    x2 = max(0.0, min(1.0, cx + (bw / 2.0)))
    y2 = max(0.0, min(1.0, cy + (bh / 2.0)))

    if x2 <= x1:
        midpoint = float(np.clip((x1 + x2) / 2.0, eps, 1.0 - eps))
        x1 = max(0.0, midpoint - eps)
        x2 = min(1.0, midpoint + eps)
    if y2 <= y1:
        midpoint = float(np.clip((y1 + y2) / 2.0, eps, 1.0 - eps))
        y1 = max(0.0, midpoint - eps)
        y2 = min(1.0, midpoint + eps)

    if x1 <= 0.0:
        x1 = eps
    if y1 <= 0.0:
        y1 = eps
    if x2 >= 1.0:
        x2 = 1.0 - eps
    if y2 >= 1.0:
        y2 = 1.0 - eps

    x1 = min(x1, x2 - eps)
    y1 = min(y1, y2 - eps)
    return [
        float(np.clip((x1 + x2) / 2.0, 0.0, 1.0)),
        float(np.clip((y1 + y2) / 2.0, 0.0, 1.0)),
        float(np.clip(x2 - x1, eps, 1.0)),
        float(np.clip(y2 - y1, eps, 1.0)),
    ]


def read_label_objects(label_path: Path, *, img_w: int, img_h: int) -> list[LabelObject]:
    if not label_path.exists():
        raise DatasetAugmentorCoreError(f"Missing label: {label_path}; backgrounds require empty TXT files.")
    objects = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            values = list(map(float, parts))
            if len(values) < 5 or (len(values) - 5) % 3 or not all(math.isfinite(x) for x in values):
                raise ValueError("Malformed row or non-finite values")
            cls, cx, cy, bw, bh = values[:5]
            if cls != int(cls) or cls < 0:
                raise ValueError("Invalid class ID")
            if not (bw > 0 and bh > 0 and cx-bw/2 >= -1e-6 and cx+bw/2 <= 1+1e-6 and
                    cy-bh/2 >= -1e-6 and cy+bh/2 <= 1+1e-6):
                raise ValueError("Bounding box lies outside image")
            keypoints = []
            for i in range(5, len(values), 3):
                x, y, visibility = values[i:i+3]
                if not (0 <= x <= 1 and 0 <= y <= 1 and visibility in (0, 1, 2)):
                    raise ValueError("Invalid keypoint or visibility")
                keypoints.append((x * img_w, y * img_h, int(visibility)))
            objects.append(LabelObject(int(cls), sanitize_yolo_bbox(cx, cy, bw, bh), keypoints))
        except ValueError as exc:
            raise DatasetAugmentorCoreError(f"Invalid label {label_path}: {exc}") from exc
    return objects


def write_label_objects(
    output_path: Path,
    *,
    boxes: Sequence[Sequence[float]],
    classes: Sequence[int],
    keypoints: Sequence[Sequence[float]],
    vis_flat: Sequence[int],
    keypoint_counts: Sequence[int],
    img_w: int,
    img_h: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kp_idx = 0
    vis_idx = 0
    lines: list[str] = []
    for obj_idx, box in enumerate(boxes):
        cls = int(float(classes[obj_idx])) if obj_idx < len(classes) else 0
        cx, cy, bw, bh = sanitize_yolo_bbox(*[float(v) for v in box[:4]])
        row = [f"{cls}", f"{cx:.8f}", f"{cy:.8f}", f"{bw:.8f}", f"{bh:.8f}"]
        kp_count = int(keypoint_counts[obj_idx]) if obj_idx < len(keypoint_counts) else 0
        for _ in range(max(0, kp_count)):
            if kp_idx < len(keypoints):
                kx_px, ky_px = keypoints[kp_idx]
                kx = max(0.0, min(1.0, float(kx_px) / max(img_w, 1)))
                ky = max(0.0, min(1.0, float(ky_px) / max(img_h, 1)))
                kv = int(vis_flat[vis_idx]) if vis_idx < len(vis_flat) else 0
                if kv == 0:
                    kx = ky = 0.0
                row.extend([f"{kx:.6f}", f"{ky:.6f}", str(kv)])
                kp_idx += 1
                vis_idx += 1
            else:
                row.extend(["0.000000", "0.000000", "0"])
        lines.append(" ".join(row))
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def discover_samples(dataset_root: Path, split: str = "all", *, log=None) -> list[SourceSample]:
    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"
    if not images_root.exists() or not labels_root.exists():
        return []

    split_filter = str(split or "all").strip()
    if split_filter.lower() == "all":
        split_dirs = [p for p in sorted(images_root.iterdir()) if p.is_dir()]
    else:
        split_dirs = [images_root / split_filter]

    samples: list[SourceSample] = []
    for split_dir in split_dirs:
        if not split_dir.is_dir():
            continue
        label_split_dir = labels_root / split_dir.name
        for img_path in _find_images(split_dir):
            label_path = label_split_dir / f"{img_path.stem}.txt"
            image = cv2.imread(str(img_path))
            if image is None:
                raise DatasetAugmentorCoreError(f"Unreadable source image: {img_path}")
            h, w = image.shape[:2]
            objects = read_label_objects(label_path, img_w=w, img_h=h)
            samples.append(SourceSample(img_path, label_path, split_dir.name, objects))
            if len(samples) % 1000 == 0:
                _emit(log, f"Scanned {len(samples):,} source images and labels")
    return samples


def compute_instance_counts(samples: Iterable[SourceSample]) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for sample in samples:
        for obj in sample.objects:
            counts[obj.class_id] += 1
    return dict(sorted(counts.items()))


def compute_class_plan(
    counts: dict[int, int],
    *,
    plan: str,
    add_count: int,
    target_ratio: float,
    add_per_class: int,
    multiplier: float,
    include_classes: str,
    exclude_classes: str,
) -> tuple[dict[int, int], dict[int, int], set[int]]:
    if not counts:
        return {}, {}, set()
    classes = sorted(counts)
    include = parse_class_set(include_classes)
    exclude = parse_class_set(exclude_classes)
    augment_classes = set(classes) if include is None else set(include)
    augment_classes = {c for c in augment_classes - set(exclude or set()) if c in counts}
    if not augment_classes:
        raise DatasetAugmentorCoreError(
            "After include/exclude filtering, there are no labeled classes left to augment."
        )

    max_count = max(counts.values())

    def need_balance() -> tuple[dict[int, int], dict[int, int]]:
        if not (0.0 < float(target_ratio) <= 1.0):
            raise DatasetAugmentorCoreError("target_ratio must be in (0, 1].")
        target = {c: int(round(max_count * float(target_ratio))) for c in classes}
        need = {
            c: max(0, target[c] - counts[c]) if c in augment_classes else 0
            for c in classes
        }
        return need, target

    def need_add() -> tuple[dict[int, int], dict[int, int]]:
        if int(add_per_class) < 0:
            raise DatasetAugmentorCoreError("add_per_class must be >= 0.")
        target = {
            c: counts[c] + (int(add_per_class) if c in augment_classes else 0)
            for c in classes
        }
        need = {c: int(add_per_class) if c in augment_classes else 0 for c in classes}
        return need, target

    def need_scale() -> tuple[dict[int, int], dict[int, int]]:
        if float(multiplier) < 1.0:
            raise DatasetAugmentorCoreError("multiplier must be >= 1.0.")
        target = {
            c: int(round(counts[c] * float(multiplier))) if c in augment_classes else counts[c]
            for c in classes
        }
        need = {
            c: max(0, target[c] - counts[c]) if c in augment_classes else 0
            for c in classes
        }
        return need, target

    if plan == "random":
        total = max(1, int(add_count))
        base, remainder = divmod(total, len(augment_classes))
        selected = sorted(augment_classes)
        need = {c: base + (selected.index(c) < remainder) if c in augment_classes else 0 for c in classes}
        target = {c: counts[c] + need[c] for c in classes}
    elif plan == "balance":
        need, target = need_balance()
    elif plan == "add":
        need, target = need_add()
    elif plan == "scale":
        need, target = need_scale()
    elif plan == "balance_add":
        need_b, _target_b = need_balance()
        need_a, _target_a = need_add()
        need = {c: need_b[c] + need_a[c] for c in classes}
        target = {c: counts[c] + need[c] for c in classes}
    elif plan == "balance_scale":
        need_b, _target_b = need_balance()
        need_s, _target_s = need_scale()
        need = {c: max(need_b[c], need_s[c]) for c in classes}
        target = {c: counts[c] + need[c] for c in classes}
    else:
        raise DatasetAugmentorCoreError(
            "plan must be one of: random, balance, add, scale, balance_add, balance_scale."
        )
    return dict(sorted(need.items())), dict(sorted(target.items())), augment_classes


def run_augmentation(
    recipe: AugmentationRecipe,
    *,
    albumentations_module=None,
    log: Optional[ProgressCallback] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> AugmentationResult:
    if albumentations_module is None:
        try:
            import albumentations as A
        except ImportError as exc:
            raise DatasetAugmentorCoreError(
                "Albumentations is required. Use tools/install_albumentations_gui.py."
            ) from exc
    else:
        A = albumentations_module
    dataset_root = Path(recipe.dataset_root).resolve()
    output_root = Path(recipe.output_root or dataset_root.parent / time.strftime("aug_run_%Y%m%d_%H%M%S")).resolve()
    if output_root == dataset_root or output_root in dataset_root.parents or dataset_root in output_root.parents:
        raise DatasetAugmentorCoreError("Input and output must be separate, non-nested dataset folders.")
    if output_root.exists() and any(output_root.iterdir()):
        raise DatasetAugmentorCoreError("Output folder must be new or empty; existing runs are never overwritten.")
    if recipe.max_total_augs < 1 or recipe.max_augs_per_source_image < 1:
        raise DatasetAugmentorCoreError("Augmentation limits must be positive.")
    random.seed(recipe.seed)
    np.random.seed(recipe.seed)
    metadata = load_metadata(dataset_root)
    _emit(log, "Scanning source images and validating label schema...")
    all_samples = discover_samples(dataset_root, "all", log=log)
    if not all_samples:
        raise DatasetAugmentorCoreError("No images/<split> and labels/<split> samples found.")
    metadata = validate_samples(all_samples, metadata)
    provenance = source_provenance(dataset_root)
    selected_splits = {s.split for s in all_samples} if recipe.split == "all" else {recipe.split}
    samples = [s for s in all_samples if s.split in selected_splits]
    labeled_samples = [s for s in samples if s.objects]
    counts = compute_instance_counts(labeled_samples)
    if not counts:
        raise DatasetAugmentorCoreError("No labeled objects in the selected split(s).")
    flip_idx = metadata.get("flip_idx", [])
    if metadata.get("kpt_shape") and recipe.transforms.horizontal_flip_prob > 0 and not flip_idx:
        raise DatasetAugmentorCoreError("Pose horizontal flips require flip_idx in the dataset YAML; otherwise set flip probability to zero.")
    exact = recipe.plan == "target_total"
    plan_name = "random" if exact else recipe.plan
    requested = recipe.target_total_images - len(samples) if exact else recipe.add_count
    if exact and (not recipe.copy_originals or requested < 0):
        raise DatasetAugmentorCoreError("Final image count requires Copy originals and a target at least as large as the selected input splits.")
    if exact and requested > recipe.max_total_augs:
        raise DatasetAugmentorCoreError(f"Target needs {requested} new images; raise Max total augments accordingly.")
    need, target, augment_classes = compute_class_plan(
        counts, plan=plan_name, add_count=max(1, requested), target_ratio=recipe.target_ratio,
        add_per_class=recipe.add_per_class, multiplier=recipe.multiplier,
        include_classes=recipe.include_classes, exclude_classes=recipe.exclude_classes,
    )
    total_requested = requested if exact else min(recipe.max_total_augs, sum(need.values()))
    if exact:
        # The exact plan targets images, independent of how many objects they contain.
        need = {c: requested if c in augment_classes else 0 for c in counts}
        target = {}
    _emit(log, f"Selected {len(samples)} images ({len(labeled_samples)} labeled); generating up to {total_requested} new images.")
    if selected_splits - {"train"}:
        _emit(log, "Evaluation splits selected explicitly: use an untouched holdout for unbiased evaluation.")
    if not exact:
        _emit(log, _format_plan_summary(counts, target, need, augment_classes, recipe.plan))
    transform = build_transform(A, recipe.transforms, recipe.min_visibility)
    if hasattr(transform, "set_random_seed"):
        transform.set_random_seed(recipe.seed)
    output_root.mkdir(parents=True, exist_ok=True)
    marker = output_root / "EXPORT_INCOMPLETE.txt"
    marker.write_text("Augmentation/export validation has not completed. Do not train on this folder.\n", encoding="utf-8")
    preview_dir = output_root / "aug_preview"
    preview_dir.mkdir()
    # Unselected splits are always copied unchanged, including empty-label backgrounds.
    originals = [s for s in all_samples if recipe.copy_originals or s.split not in selected_splits]
    _copy_originals(originals, output_root)
    if (dataset_root / "manifest.csv").is_file():
        shutil.copy2(dataset_root / "manifest.csv", output_root / "source_manifest.csv")
    recipe_path = output_root / "augmentation_recipe.json"
    _write_recipe(recipe_path, recipe, output_root)
    manifest_path = output_root / "aug_manifest.csv"
    augmented_image_paths = []
    per_source_aug_count = defaultdict(int)
    actual_counts = dict(counts)
    lineage = []
    def record_lineage(image_path, sample, augmented):
        source_image = f"images/{sample.split}/{sample.image_path.name}"
        original = provenance.get(source_image, {})
        lineage.append(dict(image=image_path.relative_to(output_root).as_posix(),
                            split=sample.split, source_image=source_image, augmented=int(augmented),
                            source_group=original.get("source_group", ""),
                            source_video=original.get("source_video", ""), frame=original.get("frame", "")))
    for sample in originals:
        record_lineage(output_root / "images" / sample.split / sample.image_path.name, sample, False)
    aug_counter = 0
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        manifest = csv.writer(manifest_file)
        manifest.writerow(["aug_image", "aug_label", "split", "source_image", "source_label"])
        while aug_counter < total_requested and (exact or sum(need.values()) > 0):
            weights = need
            if exact:
                # Favor the least represented selected behaviors without duplicating only one clip.
                ceiling = max(actual_counts[c] for c in augment_classes) + 1
                weights = {c: ceiling - actual_counts[c] if c in augment_classes else 0 for c in counts}
            sample = _pick_source(labeled_samples, weights, augment_classes, per_source_aug_count, recipe)
            if sample is None:
                raise DatasetAugmentorCoreError("Source/retry limit reached before the requested augmentation completed; output remains marked incomplete.")
            per_source_aug_count[sample.image_path] += 1
            image = cv2.imread(str(sample.image_path))
            if image is None:
                raise DatasetAugmentorCoreError(f"Could not read source image: {sample.image_path}")
            transformed = _transform_sample(transform, image, sample,
                                            horizontal_flip_prob=recipe.transforms.horizontal_flip_prob,
                                            flip_idx=flip_idx)
            if transformed is None:
                continue
            aug_img, boxes, classes, keypoints, vis, kp_counts = transformed
            if not any(c in augment_classes and (exact or need.get(c, 0) > 0) for c in classes):
                continue
            aug_img = apply_domain_shift(aug_img, recipe.domain_shift)
            regions = []
            aug_img = apply_frame_and_patch_artifacts(aug_img, recipe.artifacts, occluded_regions=regions)
            for i, (x, y) in enumerate(keypoints):
                if vis[i] == 2 and any(x1 <= x < x2 and y1 <= y < y2 for x1, y1, x2, y2 in regions):
                    vis[i] = 1
            next_index = aug_counter + 1
            stem = f"{sample.image_path.stem}_aug_{next_index:06d}"
            image_path = output_root / "images" / sample.split / f"{stem}{sample.image_path.suffix.lower()}"
            label_path = output_root / "labels" / sample.split / f"{stem}.txt"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            if image_path.exists() or label_path.exists():
                raise DatasetAugmentorCoreError(f"Output name collision: {stem}")
            if not cv2.imwrite(str(image_path), aug_img):
                raise DatasetAugmentorCoreError(f"Image write failed: {image_path}")
            write_label_objects(label_path, boxes=boxes, classes=classes, keypoints=keypoints,
                                vis_flat=vis, keypoint_counts=kp_counts,
                                img_w=aug_img.shape[1], img_h=aug_img.shape[0])
            manifest.writerow([image_path.name, label_path.name, sample.split, sample.image_path.name, sample.label_path.name])
            augmented_image_paths.append(image_path)
            record_lineage(image_path, sample, True)
            aug_counter = next_index
            for c in classes:
                actual_counts[c] = actual_counts.get(c, 0) + 1
                if not exact:
                    need[c] = max(0, need.get(c, 0) - 1)
            if progress is not None and (aug_counter % 25 == 0 or aug_counter == total_requested):
                progress(aug_counter, total_requested)
            if aug_counter % 200 == 0:
                _emit(log, f"Saved {aug_counter}/{total_requested} augmentations")
    _emit(log, "Validating every exported image and label...")
    validation = validate_output(output_root, metadata, progress=lambda message: _emit(log, message))
    if exact and sum(validation[s]["images"] for s in selected_splits) != recipe.target_total_images:
        raise DatasetAugmentorCoreError("Final image count does not match requested target.")
    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(lineage[0]))
        writer.writeheader()
        writer.writerows(lineage)
    source_provenance(output_root)
    write_metadata(output_root, metadata)
    _write_previews(augmented_image_paths, output_root, preview_dir, recipe.preview_count, metadata=metadata)
    (output_root / "augmentation_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    (output_root / "README.txt").write_text(
        "YOLO dataset augmentation export. See data.yaml for training paths and keypoint schema.\n"
        "augmentation_recipe.json records parameters; aug_manifest.csv maps derivatives to source images.\n"
        "Unselected splits are copied unchanged. Derivatives remain in their source split.\n"
        "Empty labels are backgrounds; invisible/occluded keypoints retain their fixed slots.\n"
        "Visibility: 0=unlabeled/outside image, 1=occluded, 2=visible.\n"
        "Validation checks file pairs, coordinates, schema, and exact-pixel duplicates across splits.\n"
        "It cannot establish animal/recording independence without source provenance.\n"
        "If moving this folder, update path in data.yaml. Preserve an untouched evaluation holdout.\n",
        encoding="utf-8",
    )
    marker.unlink()
    _emit(log, f"Complete: {aug_counter} augmentations; validated dataset at {output_root}")
    return AugmentationResult(output_root, aug_counter, counts, target,
                              {c: 0 for c in counts} if exact else dict(sorted(need.items())),
                              manifest_path, recipe_path, preview_dir)


def build_transform(A, settings: TransformSettings, min_visibility: float):
    transforms = []
    # Mirror explicitly in _transform_sample so keypoint identities follow flip_idx.
    if settings.affine_prob > 0:
        transforms.append(
            A.Affine(
                scale=(float(settings.scale_min), float(settings.scale_max)),
                translate_percent=(0.0, max(0.0, float(settings.translate_percent))),
                rotate=(-abs(float(settings.rotate_degrees)), abs(float(settings.rotate_degrees))),
                shear=(-abs(float(settings.shear_degrees)), abs(float(settings.shear_degrees))),
                p=_prob(settings.affine_prob),
            )
        )
    if settings.brightness_contrast_prob > 0:
        transforms.append(A.RandomBrightnessContrast(p=_prob(settings.brightness_contrast_prob)))
    if settings.hue_saturation_prob > 0:
        transforms.append(A.HueSaturationValue(p=_prob(settings.hue_saturation_prob)))
    if settings.blur_prob > 0:
        transforms.append(A.GaussianBlur(blur_limit=(3, 5), p=_prob(settings.blur_prob)))
    if settings.gauss_noise_prob > 0:
        transforms.append(_make_gauss_noise(A, p=_prob(settings.gauss_noise_prob)))

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels", "box_indices"],
            min_visibility=float(min_visibility),
        ),
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def apply_domain_shift(image: np.ndarray, settings: DomainShiftSettings) -> np.ndarray:
    out = image
    if settings.white_mouse:
        intensity = float(settings.invert_intensity)
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        inv_v = 255 - v
        v_final = cv2.addWeighted(inv_v, intensity, v, 1.0 - intensity, 0) if intensity < 1.0 else inv_v
        out = cv2.cvtColor(cv2.merge([h, s, v_final]), cv2.COLOR_HSV2BGR)
    if settings.red_light:
        red_boost = float(settings.red_boost)
        bg_suppress = float(settings.bg_suppress)
        b, g, r = cv2.split(out)
        r = np.clip(r.astype(np.float32) * red_boost, 0, 255).astype(np.uint8)
        g = np.clip(g.astype(np.float32) * bg_suppress, 0, 255).astype(np.uint8)
        b = np.clip(b.astype(np.float32) * bg_suppress, 0, 255).astype(np.uint8)
        out = cv2.merge([b, g, r])
    if settings.bw and random.random() < _prob(settings.bw_prob):
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return out


def apply_frame_and_patch_artifacts(image: np.ndarray, settings: ArtifactSettings, *, occluded_regions=None) -> np.ndarray:
    out = image
    if random.random() < _prob(settings.frame_noise_prob):
        out = _apply_whole_frame_noise(out, settings.frame_noise_std)
    if random.random() < _prob(settings.frame_banding_prob):
        out = _apply_frame_banding(out, settings.frame_banding_amp)
    if random.random() < _prob(settings.regional_noise_prob):
        out = _apply_regional_noise(
            out,
            settings.occlusion_count,
            settings.occlusion_min_frac,
            settings.occlusion_max_frac,
            settings.regional_noise_std,
        )
    if random.random() < _prob(settings.occlusion_prob):
        out = _apply_occlusions(
            out,
            settings.occlusion_count,
            settings.occlusion_min_frac,
            settings.occlusion_max_frac,
            settings.occlusion_mode,
            regions=occluded_regions,
        )
    return out


def _transform_sample(transform, image: np.ndarray, sample: SourceSample, *, horizontal_flip_prob=0.0, flip_idx=()):
    objects = sample.objects
    bboxes = [obj.bbox for obj in objects]
    classes = [obj.class_id for obj in objects]
    original_keypoint_counts = [len(obj.keypoints) for obj in objects]
    mirrored = random.random() < _prob(horizontal_flip_prob)
    if mirrored:
        image = np.ascontiguousarray(image[:, ::-1])
        bboxes = [[1.0-box[0], *box[1:]] for box in bboxes]
    original_vis: list[int] = []
    original_keypoints: list[tuple[float, float]] = []
    object_kp_offsets: list[int] = []
    running_offset = 0
    for obj in objects:
        object_kp_offsets.append(running_offset)
        points = obj.keypoints
        if mirrored and points:
            if len(flip_idx) != len(points):
                raise DatasetAugmentorCoreError("Missing or mismatched flip_idx for pose mirroring")
            points = [(image.shape[1]-1-points[i][0], points[i][1], points[i][2]) for i in flip_idx]
        for kx, ky, kv in points:
            original_keypoints.append((kx, ky))
            original_vis.append(kv)
        running_offset += len(obj.keypoints)

    box_indices = list(range(len(bboxes)))
    result = transform(
        image=image,
        bboxes=bboxes,
        class_labels=classes,
        box_indices=box_indices,
        keypoints=original_keypoints,
    )
    aug_img = result["image"]
    aug_boxes = [list(box) for box in result["bboxes"]]
    aug_classes = [int(float(c)) for c in result["class_labels"]]
    aug_keypoints = list(result["keypoints"])
    surviving_box_indices = [int(idx) for idx in result.get("box_indices", [])]
    if not aug_boxes:
        return None

    aug_h, aug_w = aug_img.shape[:2]
    keypoint_counts: list[int] = []
    vis_flat: list[int] = []
    keypoints_flat: list[tuple[float, float]] = []
    for orig_idx in surviving_box_indices:
        if orig_idx < 0 or orig_idx >= len(original_keypoint_counts):
            continue
        start = object_kp_offsets[orig_idx]
        count = original_keypoint_counts[orig_idx]
        keypoint_counts.append(int(count))
        for k in range(count):
            flat_idx = start + k
            if flat_idx >= len(aug_keypoints):
                keypoints_flat.append((0.0, 0.0))
                vis_flat.append(0)
                continue
            kx_px, ky_px = aug_keypoints[flat_idx]
            original_visibility = original_vis[flat_idx] if flat_idx < len(original_vis) else 0
            in_frame = (0.0 <= float(kx_px) < float(aug_w)) and (0.0 <= float(ky_px) < float(aug_h))
            vis_flat.append(0 if int(original_visibility) == 0 or not in_frame else int(original_visibility))
            keypoints_flat.append((float(kx_px), float(ky_px)))
    return aug_img, aug_boxes, aug_classes, keypoints_flat, vis_flat, keypoint_counts


def _copy_originals(samples: Sequence[SourceSample], output_root: Path) -> None:
    for sample in samples:
        out_img_dir = output_root / "images" / sample.split
        out_lbl_dir = output_root / "labels" / sample.split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample.image_path, out_img_dir / sample.image_path.name)
        if sample.label_path.exists():
            shutil.copy2(sample.label_path, out_lbl_dir / sample.label_path.name)


def _find_images(img_dir: Path) -> list[Path]:
    out: list[Path] = []
    for ext in IMG_EXTS:
        out.extend(sorted(img_dir.glob(f"*{ext}")))
    return out


def _pick_source(
    samples: Sequence[SourceSample],
    need: dict[int, int],
    augment_classes: set[int],
    per_source_aug_count: dict[Path, int],
    recipe: AugmentationRecipe,
) -> Optional[SourceSample]:
    candidates: list[SourceSample] = []
    weights: list[float] = []
    availability = defaultdict(int)
    for sample in samples:
        if per_source_aug_count[sample.image_path] < int(recipe.max_augs_per_source_image):
            for c in {obj.class_id for obj in sample.objects}:
                availability[c] += 1
    for sample in samples:
        if per_source_aug_count[sample.image_path] >= int(recipe.max_augs_per_source_image):
            continue
        present = {obj.class_id for obj in sample.objects}.intersection(augment_classes)
        weight = sum(need.get(class_id, 0) / max(1, availability[class_id]) for class_id in present)
        if weight > 0:
            candidates.append(sample)
            weights.append(weight)
    if not candidates:
        return None
    return random.choices(candidates, weights=weights, k=1)[0]


def _write_recipe(recipe_path: Path, recipe: AugmentationRecipe, output_root: Path) -> None:
    payload = asdict(recipe)
    payload["dataset_root"] = str(recipe.dataset_root)
    payload["output_root"] = str(output_root)
    recipe_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_previews(
    augmented_image_paths: Sequence[Path],
    output_root: Path,
    preview_dir: Path,
    preview_count: int,
    *,
    metadata=None,
) -> None:
    if not augmented_image_paths or int(preview_count) <= 0:
        return
    chosen = random.sample(list(augmented_image_paths), k=min(int(preview_count), len(augmented_image_paths)))
    for image_path in chosen:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        rel_parts = image_path.relative_to(output_root).parts
        split = rel_parts[1] if len(rel_parts) >= 3 else "train"
        label_path = output_root / "labels" / split / f"{image_path.stem}.txt"
        h, w = image.shape[:2]
        objects = read_label_objects(label_path, img_w=w, img_h=h)
        boxes = [obj.bbox for obj in objects]
        classes = [obj.class_id for obj in objects]
        preview = _draw_boxes(image, boxes, classes)
        for obj in objects:
            names_by_class = (metadata or {}).get("kpt_names", {})
            names = names_by_class.get(obj.class_id, names_by_class.get(str(obj.class_id), []))
            for edge in (metadata or {}).get("skeleton", []):
                if len(edge) == 2 and all(isinstance(i, int) and 0 <= i < len(obj.keypoints) for i in edge):
                    a, b = [obj.keypoints[i] for i in edge]
                    if a[2] and b[2]:
                        cv2.line(preview, (round(a[0]), round(a[1])), (round(b[0]), round(b[1])), (200, 200, 0), 1)
            for i, (x, y, visible) in enumerate(obj.keypoints):
                if visible:
                    color = (0, 220, 0) if visible == 2 else (0, 170, 255)
                    cv2.circle(preview, (round(x), round(y)), 4, color, -1 if visible == 2 else 1)
                    name = names[i] if i < len(names) else str(i)
                    cv2.putText(preview, f"{i}:{name}", (round(x)+5, round(y)-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        if not cv2.imwrite(str(preview_dir / f"{image_path.stem}_preview.jpg"), preview):
            raise DatasetAugmentorCoreError("Preview image write failed")


def _draw_boxes(image_bgr: np.ndarray, boxes: Sequence[Sequence[float]], classes: Sequence[int]) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]
    for box, class_id in zip(boxes, classes):
        x1, y1, x2, y2 = _yolo_to_xyxy(box, w, h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            str(int(class_id)),
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return out


def _yolo_to_xyxy(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    cx, cy, bw, bh = [float(v) for v in box[:4]]
    x1 = int((cx - bw / 2.0) * width)
    y1 = int((cy - bh / 2.0) * height)
    x2 = int((cx + bw / 2.0) * width)
    y2 = int((cy + bh / 2.0) * height)
    return (
        max(0, min(x1, width - 1)),
        max(0, min(y1, height - 1)),
        max(0, min(x2, width - 1)),
        max(0, min(y2, height - 1)),
    )


def _make_gauss_noise(A, p: float):
    params = inspect.signature(A.GaussNoise).parameters
    if "var_limit" in params:
        return A.GaussNoise(var_limit=(5.0, 25.0), p=p)
    if "std_range" in params:
        return A.GaussNoise(std_range=(0.01, 0.05), p=p)
    return A.GaussNoise(p=p)


def _clip_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


def _patch_bounds(image_shape, min_frac: float, max_frac: float) -> tuple[int, int, int, int]:
    h, w = image_shape[:2]
    min_frac = max(0.001, min(float(min_frac), 1.0))
    max_frac = max(min_frac, min(float(max_frac), 1.0))
    patch_w = random.randint(max(1, int(w * min_frac)), max(1, int(w * max_frac)))
    patch_h = random.randint(max(1, int(h * min_frac)), max(1, int(h * max_frac)))
    x1 = random.randint(0, max(0, w - patch_w))
    y1 = random.randint(0, max(0, h - patch_h))
    return x1, y1, x1 + patch_w, y1 + patch_h


def _apply_whole_frame_noise(image: np.ndarray, std: float) -> np.ndarray:
    noise = np.random.normal(0.0, float(std), image.shape).astype(np.float32)
    return _clip_uint8(image.astype(np.float32) + noise)


def _apply_frame_banding(image: np.ndarray, amplitude: float) -> np.ndarray:
    h = image.shape[0]
    period = random.uniform(8.0, 28.0)
    phase = random.uniform(0.0, 2.0 * np.pi)
    y = np.arange(h, dtype=np.float32)
    band = np.sin((2.0 * np.pi * y / period) + phase) * float(amplitude)
    return _clip_uint8(image.astype(np.float32) + band.reshape(h, 1, 1))


def _apply_regional_noise(
    image: np.ndarray,
    count: int,
    min_frac: float,
    max_frac: float,
    std: float,
) -> np.ndarray:
    out = image.copy()
    for _ in range(max(1, int(count))):
        x1, y1, x2, y2 = _patch_bounds(out.shape, min_frac, max_frac)
        patch = out[y1:y2, x1:x2].astype(np.float32)
        noise = np.random.normal(0.0, float(std), patch.shape).astype(np.float32)
        out[y1:y2, x1:x2] = _clip_uint8(patch + noise)
    return out


def _apply_occlusions(
    image: np.ndarray,
    count: int,
    min_frac: float,
    max_frac: float,
    mode: str,
    *,
    regions=None,
) -> np.ndarray:
    out = image.copy()
    modes = ["black", "gray", "noise"] if mode == "random" else [mode]
    for _ in range(max(1, int(count))):
        x1, y1, x2, y2 = _patch_bounds(out.shape, min_frac, max_frac)
        if regions is not None:
            regions.append((x1, y1, x2, y2))
        selected = random.choice(modes)
        if selected == "black":
            fill = np.zeros_like(out[y1:y2, x1:x2])
        elif selected == "gray":
            fill = np.full_like(out[y1:y2, x1:x2], random.randint(35, 210))
        else:
            fill = np.random.randint(0, 256, out[y1:y2, x1:x2].shape, dtype=np.uint8)
        out[y1:y2, x1:x2] = fill
    return out


def _format_plan_summary(
    counts: dict[int, int],
    target: dict[int, int],
    need: dict[int, int],
    augment_classes: set[int],
    plan: str,
) -> str:
    lines = [f"Class-aware plan: {plan}"]
    for class_id in sorted(counts):
        flag = " (augment)" if class_id in augment_classes else ""
        lines.append(
            f"  class {class_id}: {counts[class_id]} -> target {target[class_id]} "
            f"(need +{need[class_id]}){flag}"
        )
    return "\n".join(lines)


def _prob(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _emit(log: Optional[ProgressCallback], message: str) -> None:
    if log is not None:
        log(message)


__all__ = [
    "ArtifactSettings",
    "AugmentationRecipe",
    "AugmentationResult",
    "DatasetAugmentorCoreError",
    "DomainShiftSettings",
    "TransformSettings",
    "apply_domain_shift",
    "apply_frame_and_patch_artifacts",
    "compute_class_plan",
    "compute_instance_counts",
    "discover_samples",
    "parse_class_set",
    "read_label_objects",
    "run_augmentation",
    "sanitize_yolo_bbox",
    "write_label_objects",
]
