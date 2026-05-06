"""Dataset quality checks for YOLO-style image/label folders."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DatasetQAIssue:
    severity: str
    check: str
    message: str
    count: int = 0
    examples: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "severity": self.severity,
            "check": self.check,
            "message": self.message,
            "count": int(self.count),
            "examples": list(self.examples),
        }


@dataclass(frozen=True)
class DatasetQAReport:
    created_at_utc: str
    image_dir: str
    label_dir: str
    tiny_box_area_threshold: float
    image_count: int
    label_file_count: int
    matched_pairs: int
    missing_labels: int
    orphan_labels: int
    duplicate_image_stems: int
    total_label_rows: int
    valid_label_rows: int
    malformed_rows: int
    tiny_boxes: int
    duplicate_rows: int
    class_histogram: Dict[int, int]
    imbalance_ratio: float | None
    status: str
    issues: List[DatasetQAIssue]

    def to_dict(self) -> Dict[str, object]:
        return {
            "created_at_utc": self.created_at_utc,
            "image_dir": self.image_dir,
            "label_dir": self.label_dir,
            "tiny_box_area_threshold": float(self.tiny_box_area_threshold),
            "image_count": int(self.image_count),
            "label_file_count": int(self.label_file_count),
            "matched_pairs": int(self.matched_pairs),
            "missing_labels": int(self.missing_labels),
            "orphan_labels": int(self.orphan_labels),
            "duplicate_image_stems": int(self.duplicate_image_stems),
            "total_label_rows": int(self.total_label_rows),
            "valid_label_rows": int(self.valid_label_rows),
            "malformed_rows": int(self.malformed_rows),
            "tiny_boxes": int(self.tiny_boxes),
            "duplicate_rows": int(self.duplicate_rows),
            "class_histogram": {str(k): int(v) for k, v in sorted(self.class_histogram.items())},
            "imbalance_ratio": None if self.imbalance_ratio is None else float(self.imbalance_ratio),
            "status": self.status,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _list_images_flat(image_dir: Path) -> List[Path]:
    images: List[Path] = []
    for entry in sorted(image_dir.iterdir(), key=lambda p: p.name.lower()):
        if not entry.is_file():
            continue
        if entry.name.startswith("."):
            continue
        if entry.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        images.append(entry)
    return images


def _list_labels_flat(label_dir: Path) -> List[Path]:
    labels: List[Path] = []
    for entry in sorted(label_dir.iterdir(), key=lambda p: p.name.lower()):
        if not entry.is_file():
            continue
        if entry.name.startswith("."):
            continue
        if entry.suffix.lower() != ".txt":
            continue
        labels.append(entry)
    return labels


def _fmt_percent(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def _clamp_examples(examples: List[str], value: str, *, limit: int = 8) -> None:
    if len(examples) < limit:
        examples.append(value)


def run_dataset_qc(
    *,
    image_dir: str,
    label_dir: str,
    tiny_box_area_threshold: float = 0.001,
) -> DatasetQAReport:
    """Validate YOLO label quality against a flat image directory."""
    image_dir_path = Path(image_dir).expanduser()
    label_dir_path = Path(label_dir).expanduser()
    if not image_dir_path.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir_path}")
    if not label_dir_path.is_dir():
        raise FileNotFoundError(f"Label directory does not exist: {label_dir_path}")

    try:
        tiny_box_area_threshold = float(tiny_box_area_threshold)
    except Exception as exc:
        raise ValueError(f"tiny_box_area_threshold must be numeric, got {tiny_box_area_threshold!r}") from exc
    if tiny_box_area_threshold <= 0:
        raise ValueError("tiny_box_area_threshold must be > 0.")

    images = _list_images_flat(image_dir_path)
    labels = _list_labels_flat(label_dir_path)

    image_stem_counts: Dict[str, int] = {}
    for image_path in images:
        image_stem_counts[image_path.stem] = image_stem_counts.get(image_path.stem, 0) + 1
    duplicate_image_stems = sum(1 for _stem, count in image_stem_counts.items() if count > 1)

    image_stems = set(image_stem_counts.keys())
    label_stems = {path.stem for path in labels}
    missing_labels = len(image_stems - label_stems)
    orphan_labels = len(label_stems - image_stems)
    matched_pairs = len(image_stems & label_stems)

    class_histogram: Dict[int, int] = {}
    total_label_rows = 0
    valid_label_rows = 0
    malformed_rows = 0
    tiny_boxes = 0
    duplicate_rows = 0

    malformed_examples: List[str] = []
    tiny_examples: List[str] = []
    duplicate_row_examples: List[str] = []
    duplicate_stem_examples = sorted([stem for stem, count in image_stem_counts.items() if count > 1])[:5]
    missing_label_examples = sorted(image_stems - label_stems)[:5]
    orphan_label_examples = sorted(label_stems - image_stems)[:5]

    for label_path in labels:
        try:
            lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception as exc:
            malformed_rows += 1
            _clamp_examples(malformed_examples, f"{label_path.name}: read error ({exc})")
            continue

        seen_rows: set[str] = set()
        for line_idx, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line:
                continue
            total_label_rows += 1

            canonical_row = " ".join(line.split())
            if canonical_row in seen_rows:
                duplicate_rows += 1
                _clamp_examples(duplicate_row_examples, f"{label_path.name}:{line_idx}")
            else:
                seen_rows.add(canonical_row)

            tokens = line.split()
            if len(tokens) < 5:
                malformed_rows += 1
                _clamp_examples(malformed_examples, f"{label_path.name}:{line_idx} (too few columns)")
                continue

            try:
                vals = [float(tok) for tok in tokens]
            except Exception:
                malformed_rows += 1
                _clamp_examples(malformed_examples, f"{label_path.name}:{line_idx} (non-numeric value)")
                continue

            class_raw = vals[0]
            class_id = int(class_raw)
            if class_raw < 0 or abs(class_raw - class_id) > 1e-9:
                malformed_rows += 1
                _clamp_examples(malformed_examples, f"{label_path.name}:{line_idx} (invalid class id)")
                continue

            x, y, w, h = vals[1], vals[2], vals[3], vals[4]
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                malformed_rows += 1
                _clamp_examples(malformed_examples, f"{label_path.name}:{line_idx} (bbox out of range)")
                continue

            extra_cols = len(vals) - 5
            if extra_cols and extra_cols % 3 != 0:
                malformed_rows += 1
                _clamp_examples(malformed_examples, f"{label_path.name}:{line_idx} (pose columns not divisible by 3)")
                continue

            valid_label_rows += 1
            class_histogram[class_id] = class_histogram.get(class_id, 0) + 1

            if (w * h) < tiny_box_area_threshold:
                tiny_boxes += 1
                _clamp_examples(
                    tiny_examples,
                    f"{label_path.name}:{line_idx} (area={w*h:.6f})",
                )

    issues: List[DatasetQAIssue] = []
    if missing_labels > 0:
        example_text = [f"{stem}.txt missing" for stem in missing_label_examples]
        issues.append(
            DatasetQAIssue(
                severity="error",
                check="missing_labels",
                message="Images without matching .txt labels were found.",
                count=missing_labels,
                examples=example_text,
            )
        )
    if malformed_rows > 0:
        issues.append(
            DatasetQAIssue(
                severity="error",
                check="malformed_rows",
                message="One or more YOLO label rows are malformed or out of range.",
                count=malformed_rows,
                examples=malformed_examples,
            )
        )
    if duplicate_image_stems > 0:
        issues.append(
            DatasetQAIssue(
                severity="warning",
                check="duplicate_image_stems",
                message="Duplicate image basenames were found (same stem across files).",
                count=duplicate_image_stems,
                examples=duplicate_stem_examples,
            )
        )
    if orphan_labels > 0:
        example_text = [f"{stem}.txt" for stem in orphan_label_examples]
        issues.append(
            DatasetQAIssue(
                severity="warning",
                check="orphan_labels",
                message="Label files without matching images were found.",
                count=orphan_labels,
                examples=example_text,
            )
        )
    if tiny_boxes > 0:
        issues.append(
            DatasetQAIssue(
                severity="warning",
                check="tiny_boxes",
                message="Tiny bounding boxes were detected.",
                count=tiny_boxes,
                examples=tiny_examples,
            )
        )
    if duplicate_rows > 0:
        issues.append(
            DatasetQAIssue(
                severity="warning",
                check="duplicate_rows",
                message="Duplicate label rows were found inside label files.",
                count=duplicate_rows,
                examples=duplicate_row_examples,
            )
        )

    imbalance_ratio: float | None = None
    total_valid = sum(class_histogram.values())
    if len(class_histogram) >= 2 and total_valid > 0:
        max_class, max_count = max(class_histogram.items(), key=lambda kv: kv[1])
        _min_class, min_count = min(class_histogram.items(), key=lambda kv: kv[1])
        if min_count > 0:
            imbalance_ratio = float(max_count) / float(min_count)
            dominant_share = float(max_count) / float(total_valid)
            if dominant_share >= 0.75 and imbalance_ratio >= 3.0 and total_valid >= 20:
                issues.append(
                    DatasetQAIssue(
                        severity="warning",
                        check="class_imbalance",
                        message=(
                            f"Class imbalance detected. Class {max_class} has {_fmt_percent(dominant_share)} "
                            f"of valid labels (ratio {imbalance_ratio:.2f}:1)."
                        ),
                        count=max_count,
                        examples=[],
                    )
                )
    elif len(class_histogram) == 1 and total_valid >= 20:
        only_class = next(iter(class_histogram.keys()))
        issues.append(
            DatasetQAIssue(
                severity="warning",
                check="single_class_dataset",
                message=f"Only class {only_class} appears in labels; verify this is intentional.",
                count=class_histogram[only_class],
                examples=[],
            )
        )

    has_errors = any(issue.severity == "error" for issue in issues)
    has_warnings = any(issue.severity == "warning" for issue in issues)
    if has_errors:
        status = "fail"
    elif has_warnings:
        status = "warn"
    else:
        status = "pass"
        issues.append(
            DatasetQAIssue(
                severity="info",
                check="qa_pass",
                message="No blocking dataset issues detected.",
                count=0,
                examples=[],
            )
        )

    return DatasetQAReport(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        image_dir=str(image_dir_path),
        label_dir=str(label_dir_path),
        tiny_box_area_threshold=float(tiny_box_area_threshold),
        image_count=len(images),
        label_file_count=len(labels),
        matched_pairs=matched_pairs,
        missing_labels=missing_labels,
        orphan_labels=orphan_labels,
        duplicate_image_stems=duplicate_image_stems,
        total_label_rows=total_label_rows,
        valid_label_rows=valid_label_rows,
        malformed_rows=malformed_rows,
        tiny_boxes=tiny_boxes,
        duplicate_rows=duplicate_rows,
        class_histogram=class_histogram,
        imbalance_ratio=imbalance_ratio,
        status=status,
        issues=issues,
    )


def format_dataset_qc_summary(report: DatasetQAReport) -> str:
    """Build a compact text summary for UI display."""
    lines = [
        f"Dataset QA status: {report.status.upper()}",
        (
            f"Images={report.image_count} | Labels={report.label_file_count} | "
            f"Matched={report.matched_pairs} | Missing labels={report.missing_labels} | "
            f"Orphan labels={report.orphan_labels}"
        ),
        (
            f"Rows: valid={report.valid_label_rows}/{report.total_label_rows}, "
            f"malformed={report.malformed_rows}, tiny_boxes={report.tiny_boxes}, "
            f"duplicate_rows={report.duplicate_rows}"
        ),
        f"Tiny box threshold (area): {report.tiny_box_area_threshold}",
    ]
    if report.class_histogram:
        histogram_preview = ", ".join(
            f"class {cls_id}: {count}" for cls_id, count in sorted(report.class_histogram.items())
        )
        lines.append(f"Class histogram: {histogram_preview}")
    else:
        lines.append("Class histogram: no valid rows detected.")
    return "\n".join(lines)
