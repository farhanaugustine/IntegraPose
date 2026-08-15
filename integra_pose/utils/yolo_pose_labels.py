"""Schema-aware parsing and formatting for Ultralytics YOLO pose labels."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


POSE_LABEL_SCHEMA_FILENAME = "integrapose_pose_labels.schema.json"
POSE_LABEL_FORMAT = "ultralytics-pose"
POSE_LABEL_SCHEMA_VERSION = 1


def _is_int_like(value: float, *, tolerance: float = 1e-6) -> bool:
    return math.isfinite(value) and abs(value - round(value)) <= tolerance


def _parse_nonnegative_int(value: float, *, field: str) -> int:
    if not _is_int_like(value) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer, got {value!r}.")
    return int(round(value))


@dataclass(frozen=True, slots=True)
class YoloPoseLabelSchema:
    """Run-level layout for numeric Ultralytics pose-label rows.

    Bbox confidence is a run-wide setting. Track IDs are optional per row and,
    when present, always follow bbox confidence. This fixed ordering removes the
    otherwise unavoidable ambiguity between a confidence of ``1`` and track ID
    ``1`` while preserving standard numeric Ultralytics labels.
    """

    keypoint_count: int
    keypoint_dimensions: int
    include_bbox: bool = True
    include_bbox_confidence: bool = False
    include_track_id: bool = True

    def __post_init__(self) -> None:
        if self.keypoint_count <= 0:
            raise ValueError("keypoint_count must be positive.")
        if self.keypoint_dimensions not in (2, 3):
            raise ValueError("keypoint_dimensions must be 2 or 3.")

    def to_dict(self) -> dict:
        return {
            "format": POSE_LABEL_FORMAT,
            "schema_version": POSE_LABEL_SCHEMA_VERSION,
            "keypoint_count": self.keypoint_count,
            "keypoint_dimensions": self.keypoint_dimensions,
            "include_bbox": self.include_bbox,
            "include_bbox_confidence": self.include_bbox_confidence,
            "include_track_id": self.include_track_id,
            "field_order": [
                "class_id",
                "bbox_xywh_normalized",
                "keypoints_normalized",
                "bbox_confidence_if_enabled",
                "track_id_if_available",
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "YoloPoseLabelSchema":
        if not isinstance(payload, dict):
            raise ValueError("Pose label schema must be a JSON object.")
        if payload.get("format") != POSE_LABEL_FORMAT:
            raise ValueError(f"Unsupported pose label format: {payload.get('format')!r}.")
        if payload.get("schema_version") != POSE_LABEL_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported pose label schema version: {payload.get('schema_version')!r}."
            )
        return cls(
            keypoint_count=int(payload["keypoint_count"]),
            keypoint_dimensions=int(payload["keypoint_dimensions"]),
            include_bbox=bool(payload.get("include_bbox", True)),
            include_bbox_confidence=bool(payload.get("include_bbox_confidence", False)),
            include_track_id=bool(payload.get("include_track_id", True)),
        )


@dataclass(frozen=True, slots=True)
class YoloPoseLabel:
    """One parsed pose detection."""

    class_id: int
    bbox: tuple[float, float, float, float] | None
    keypoints: tuple[tuple[float, ...], ...]
    keypoint_dimensions: int
    bbox_confidence: float | None = None
    track_id: int | None = None

    def keypoints_xyc(self) -> tuple[tuple[float, float, float], ...]:
        """Return ``(x, y, confidence)`` triples for 2D or 3D keypoints.

        Two-dimensional YOLO labels do not carry confidence. A non-zero point is
        therefore treated as observed (confidence 1); the YOLO missing-point
        sentinel ``0 0`` remains unavailable (confidence 0).
        """

        if self.keypoint_dimensions == 3:
            return tuple((point[0], point[1], point[2]) for point in self.keypoints)
        return tuple(
            (point[0], point[1], 0.0 if point[0] == 0.0 and point[1] == 0.0 else 1.0)
            for point in self.keypoints
        )


def pose_label_schema_path(labels_dir: str | Path) -> Path:
    return Path(labels_dir) / POSE_LABEL_SCHEMA_FILENAME


def write_pose_label_schema(labels_dir: str | Path, schema: YoloPoseLabelSchema) -> Path:
    """Write the IntegraPose run-level schema next to YOLO ``.txt`` labels."""

    target = pose_label_schema_path(labels_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(schema.to_dict(), indent=2) + "\n", encoding="utf-8")
    return target


def load_pose_label_schema(
    labels_dir: str | Path,
    *,
    expected_keypoint_count: int | None = None,
) -> YoloPoseLabelSchema | None:
    """Load an IntegraPose pose-label schema, returning ``None`` for legacy data."""

    target = pose_label_schema_path(labels_dir)
    if not target.is_file():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read pose label schema {target}: {exc}") from exc
    try:
        schema = YoloPoseLabelSchema.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid pose label schema {target}: {exc}") from exc
    if expected_keypoint_count is not None and schema.keypoint_count != expected_keypoint_count:
        raise ValueError(
            f"Pose label schema declares {schema.keypoint_count} keypoints, but "
            f"{expected_keypoint_count} names were provided."
        )
    return schema


def _format_number(value: float, *, precision: int) -> str:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Pose label values must be finite, got {value!r}.")
    return f"{value:.{precision}f}"


def format_yolo_pose_label(
    *,
    class_id: int,
    bbox: Sequence[float],
    keypoints: Sequence[Sequence[float]],
    bbox_confidence: float | None = None,
    track_id: int | None = None,
    precision: int = 6,
) -> str:
    """Format a standard numeric Ultralytics pose-label row."""

    class_id = _parse_nonnegative_int(float(class_id), field="class_id")
    if len(bbox) != 4:
        raise ValueError(f"bbox must contain four xywh values, got {len(bbox)}.")
    if not keypoints:
        raise ValueError("At least one keypoint is required for a pose label.")

    keypoint_dimensions = len(keypoints[0])
    if keypoint_dimensions not in (2, 3):
        raise ValueError("Each keypoint must contain either x,y or x,y,confidence.")
    if any(len(point) != keypoint_dimensions for point in keypoints):
        raise ValueError("All keypoints in one pose row must have the same dimensionality.")
    if keypoint_dimensions == 3:
        invalid = [float(point[2]) for point in keypoints if not 0.0 <= float(point[2]) <= 2.0]
        if invalid:
            raise ValueError(
                f"Keypoint confidence/visibility values must be in [0, 2], got {invalid[0]!r}."
            )

    fields = [str(class_id)]
    fields.extend(_format_number(value, precision=precision) for value in bbox)
    for point in keypoints:
        fields.extend(_format_number(value, precision=precision) for value in point)
    if bbox_confidence is not None:
        bbox_confidence = float(bbox_confidence)
        if not 0.0 <= bbox_confidence <= 1.0:
            raise ValueError(f"bbox_confidence must be between 0 and 1, got {bbox_confidence!r}.")
        fields.append(_format_number(bbox_confidence, precision=precision))
    if track_id is not None:
        fields.append(str(_parse_nonnegative_int(float(track_id), field="track_id")))
    return " ".join(fields)


def _parse_numeric_tokens(values: str | Iterable[str]) -> list[float]:
    tokens = values.strip().split() if isinstance(values, str) else list(values)
    if not tokens:
        raise ValueError("Pose label row is empty.")
    try:
        numeric = [float(token) for token in tokens]
    except (TypeError, ValueError) as exc:
        raise ValueError("Pose label rows must contain only numeric tokens.") from exc
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("Pose label rows must contain only finite values.")
    return numeric


def _parse_suffix(
    suffix: Sequence[float],
    *,
    include_bbox_confidence: bool | None,
    include_track_id: bool | None,
) -> tuple[float | None, int | None]:
    if len(suffix) > 2:
        raise ValueError(f"Pose label has {len(suffix)} unexpected suffix values.")

    bbox_confidence = None
    track_id = None
    remaining = list(suffix)

    if include_bbox_confidence is True:
        if not remaining:
            raise ValueError("Pose label is missing the bbox confidence required by its schema.")
        bbox_confidence = float(remaining.pop(0))
        if not 0.0 <= bbox_confidence <= 1.0:
            raise ValueError(f"bbox confidence must be between 0 and 1, got {bbox_confidence!r}.")
    elif include_bbox_confidence is False:
        if remaining:
            if include_track_id is False:
                raise ValueError("Pose label has an unexpected suffix value.")
            track_id = _parse_nonnegative_int(remaining.pop(0), field="track_id")
        if remaining:
            raise ValueError("Pose label has unexpected trailing values.")
    elif len(remaining) == 2:
        if include_track_id is False:
            raise ValueError("Pose label has an unexpected track ID.")
        bbox_confidence = float(remaining.pop(0))
        if not 0.0 <= bbox_confidence <= 1.0:
            raise ValueError(f"bbox confidence must be between 0 and 1, got {bbox_confidence!r}.")
        track_id = _parse_nonnegative_int(remaining.pop(0), field="track_id")
    elif len(remaining) == 1:
        candidate = float(remaining.pop(0))
        # Standard Ultralytics emits either confidence or ID as a lone suffix.
        # Integer-like values are IDs for backward compatibility with tracked
        # exports; the run-level schema removes this ambiguity for app output.
        if include_track_id is False and 0.0 <= candidate <= 1.0:
            bbox_confidence = candidate
        elif include_track_id is not False and _is_int_like(candidate) and candidate >= 0:
            track_id = int(round(candidate))
        elif 0.0 <= candidate <= 1.0:
            bbox_confidence = candidate
        else:
            raise ValueError(f"Unrecognized pose-label suffix value {candidate!r}.")

    if remaining:
        if include_track_id is False:
            raise ValueError("Pose label has an unexpected track ID.")
        track_id = _parse_nonnegative_int(remaining.pop(0), field="track_id")
    if remaining:
        raise ValueError("Pose label has unexpected trailing values.")
    return bbox_confidence, track_id


def _parse_known_layout(
    numeric: Sequence[float],
    *,
    keypoint_count: int,
    keypoint_dimensions: int,
    include_bbox: bool,
    include_bbox_confidence: bool | None,
    include_track_id: bool | None,
) -> YoloPoseLabel:
    if keypoint_dimensions not in (2, 3):
        raise ValueError("keypoint_dimensions must be 2 or 3.")
    if keypoint_count <= 0:
        raise ValueError("keypoint_count must be positive.")

    class_id = _parse_nonnegative_int(numeric[0], field="class_id")
    cursor = 1
    bbox = None
    if include_bbox:
        if len(numeric) < cursor + 4:
            raise ValueError("Pose label is missing its four bbox values.")
        bbox = tuple(float(value) for value in numeric[cursor : cursor + 4])
        cursor += 4

    pose_value_count = keypoint_count * keypoint_dimensions
    if len(numeric) < cursor + pose_value_count:
        raise ValueError(
            f"Pose label is too short for {keypoint_count} keypoints with "
            f"{keypoint_dimensions} values each."
        )
    pose_values = numeric[cursor : cursor + pose_value_count]
    cursor += pose_value_count
    keypoints = tuple(
        tuple(float(value) for value in pose_values[index : index + keypoint_dimensions])
        for index in range(0, pose_value_count, keypoint_dimensions)
    )
    if keypoint_dimensions == 3:
        invalid = [point[2] for point in keypoints if not 0.0 <= point[2] <= 2.0]
        if invalid:
            raise ValueError(f"Keypoint confidence/visibility values must be in [0, 2], got {invalid[0]!r}.")

    bbox_confidence, track_id = _parse_suffix(
        numeric[cursor:],
        include_bbox_confidence=include_bbox_confidence,
        include_track_id=include_track_id,
    )
    return YoloPoseLabel(
        class_id=class_id,
        bbox=bbox,
        keypoints=keypoints,
        keypoint_dimensions=keypoint_dimensions,
        bbox_confidence=bbox_confidence,
        track_id=track_id,
    )


def parse_yolo_pose_label(
    values: str | Iterable[str],
    *,
    keypoint_count: int,
    schema: YoloPoseLabelSchema | None = None,
    keypoint_dimensions: int | None = None,
    include_bbox: bool | None = None,
    include_bbox_confidence: bool | None = None,
    include_track_id: bool | None = None,
) -> YoloPoseLabel:
    """Parse standard Ultralytics pose rows without slicing from the suffix.

    A schema or explicit ``keypoint_dimensions`` is definitive. Legacy files
    without either are inferred, preferring the standard bbox layout and 3D
    keypoints when a numeric row is mathematically ambiguous.
    """

    numeric = _parse_numeric_tokens(values)
    if schema is not None:
        if schema.keypoint_count != keypoint_count:
            raise ValueError(
                f"Pose schema expects {schema.keypoint_count} keypoints, got {keypoint_count}."
            )
        return _parse_known_layout(
            numeric,
            keypoint_count=keypoint_count,
            keypoint_dimensions=schema.keypoint_dimensions,
            include_bbox=schema.include_bbox,
            include_bbox_confidence=schema.include_bbox_confidence,
            include_track_id=schema.include_track_id,
        )

    dimensions = [keypoint_dimensions] if keypoint_dimensions is not None else [3, 2]
    bbox_options = [include_bbox] if include_bbox is not None else [True, False]
    errors: list[str] = []
    candidates: list[tuple[int, YoloPoseLabel]] = []
    for bbox_enabled in bbox_options:
        for dimensions_value in dimensions:
            try:
                parsed = _parse_known_layout(
                    numeric,
                    keypoint_count=keypoint_count,
                    keypoint_dimensions=int(dimensions_value),
                    include_bbox=bool(bbox_enabled),
                    include_bbox_confidence=include_bbox_confidence,
                    include_track_id=include_track_id,
                )
            except ValueError as exc:
                errors.append(str(exc))
                continue
            suffix_count = int(parsed.bbox_confidence is not None) + int(parsed.track_id is not None)
            score = (20 if bbox_enabled else 0) + (4 if dimensions_value == 3 else 0) - suffix_count
            candidates.append((score, parsed))

    if not candidates:
        detail = errors[0] if errors else "no supported layout matched"
        raise ValueError(f"Could not parse YOLO pose label: {detail}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]
