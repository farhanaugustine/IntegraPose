"""Frame identity contracts for per-frame inference artifacts."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from integra_pose.utils.safe_io import safe_read_json, safe_write_json


FRAME_LABEL_MANIFEST_FILENAME = "integrapose_frame_labels.json"
FRAME_LABEL_SCHEMA_VERSION = 1


class FrameIdentityError(ValueError):
    """Raised when label files cannot be mapped to unique video frames."""


def normalize_class_name_metadata(
    names: Any,
    *,
    context: str = "class-name metadata",
) -> list[str]:
    """Return a dense ID-ordered class-name list or fail on ambiguity."""
    if names is None:
        return []

    if isinstance(names, dict):
        indexed: dict[int, str] = {}
        for raw_id, raw_name in names.items():
            if isinstance(raw_id, bool):
                raise FrameIdentityError(f"Invalid boolean class ID in {context}.")
            try:
                class_id = int(raw_id)
            except (TypeError, ValueError) as exc:
                raise FrameIdentityError(
                    f"Invalid class ID {raw_id!r} in {context}; IDs must be integers."
                ) from exc
            if class_id < 0:
                raise FrameIdentityError(
                    f"Invalid class ID {class_id} in {context}; IDs must be zero or positive."
                )
            if class_id in indexed:
                raise FrameIdentityError(
                    f"Duplicate class ID {class_id} in {context}."
                )
            name = str(raw_name).strip()
            if not name:
                raise FrameIdentityError(
                    f"Class ID {class_id} has an empty name in {context}."
                )
            indexed[class_id] = name
        if not indexed:
            return []
        expected = list(range(max(indexed) + 1))
        missing = [class_id for class_id in expected if class_id not in indexed]
        if missing:
            raise FrameIdentityError(
                f"Non-contiguous class IDs in {context}; missing IDs: "
                + ", ".join(str(class_id) for class_id in missing)
                + "."
            )
        normalized = [indexed[class_id] for class_id in expected]
    elif isinstance(names, (list, tuple)):
        normalized = []
        for class_id, raw_name in enumerate(names):
            name = str(raw_name).strip()
            if not name:
                raise FrameIdentityError(
                    f"Class ID {class_id} has an empty name in {context}."
                )
            normalized.append(name)
    else:
        raise FrameIdentityError(
            f"Invalid {context}; expected an ID-to-name object or ordered list."
        )

    folded = [name.casefold() for name in normalized]
    if len(folded) != len(set(folded)):
        raise FrameIdentityError(
            f"Duplicate class names in {context}; every class ID must have a unique name."
        )
    return normalized


def extract_model_class_metadata(model: Any) -> tuple[list[str], str]:
    """Extract and validate the class mapping embedded in an Ultralytics model."""
    names = normalize_class_name_metadata(
        getattr(model, "names", None),
        context="model.names",
    )
    declared_count = 0
    try:
        declared_count = int(getattr(getattr(model, "model", None), "nc", 0) or 0)
    except (TypeError, ValueError):
        declared_count = 0
    if names and declared_count > 0 and len(names) != declared_count:
        raise FrameIdentityError(
            "Model class schema mismatch: model.names contains "
            f"{len(names)} names but the model declares {declared_count} classes."
        )
    model_task = str(getattr(model, "task", "") or "unknown").strip().lower() or "unknown"
    return names, model_task


def frame_label_filename(source: str | Path, frame_index: int) -> str:
    """Return the canonical zero-based label filename for one source frame."""
    index = int(frame_index)
    if index < 0:
        raise ValueError("frame_index must be zero or positive.")
    stem = Path(str(source)).stem.strip()
    if not stem:
        stem = "source"
    existing_marker = re.search(
        r"(?i)(?:^|[_\-.])(?:frame|frm|image|img)[_-]?(\d+)$",
        stem,
    )
    if existing_marker and int(existing_marker.group(1)) == index:
        return f"{stem}.txt"
    return f"{stem}_frame_{index:06d}.txt"


def frame_artifact_stem(source: str | Path, frame_index: int) -> str:
    """Return the canonical filename stem shared by labels, images, and crops."""
    return Path(frame_label_filename(source, frame_index)).stem


def parse_frame_index(filename: str | Path) -> int | None:
    """Parse an explicit or legacy frame token without using digits in subject IDs."""
    stem = Path(str(filename)).stem
    if not stem:
        return None

    marker = re.search(
        r"(?i)(?:^|[_\-.])(?:frame|frm|image|img)[_-]?(\d+)$",
        stem,
    )
    if marker:
        return int(marker.group(1))
    if re.fullmatch(r"\d+", stem):
        return int(stem)
    suffix = re.search(r"[_-](\d+)$", stem)
    if suffix:
        return int(suffix.group(1))
    return None


def _matches_source_scope(filename: str, source_stem: str) -> bool:
    """Return whether a frame label can belong to ``source_stem``."""
    stem = Path(filename).stem
    if re.fullmatch(r"\d+", stem):
        return True
    if re.fullmatch(r"(?i)(?:frame|frm|image|img)[_-]?\d+", stem):
        return True
    source_key = source_stem.casefold()
    stem_key = stem.casefold()
    if stem_key == source_key:
        return True
    if not stem_key.startswith(source_key):
        return False
    remainder = stem[len(source_stem) :]
    return remainder.startswith(("_", "-", "."))


def resolve_frame_label_indices(
    filenames: Iterable[str | Path],
    *,
    source: str | Path | None = None,
) -> dict[str, int]:
    """Resolve label filenames to unique zero-based frame indices.

    Canonical marker-based names are already zero-based. Two legacy writers are
    recognized deliberately:

    - IntegraPose's former custom writer emitted ``source.txt`` for frame zero
      followed by padded ``source_000001.txt`` names.
    - Native Ultralytics video export emits one-based ``source_1.txt`` names.

    Ambiguous duplicate identities raise instead of silently merging detections.
    """
    names = [Path(str(value)).name for value in filenames]
    names = list(dict.fromkeys(names))
    source_stem = Path(str(source)).stem if source is not None else ""
    if source_stem:
        names = [name for name in names if _matches_source_scope(name, source_stem)]
    stems = {name: Path(name).stem for name in names}
    resolved: dict[str, int] = {}

    # Canonical explicit markers and numeric-only filenames are unambiguous.
    for name, stem in stems.items():
        marker = re.search(
            r"(?i)(?:^|[_\-.])(?:frame|frm|image|img)[_-]?(\d+)$",
            stem,
        )
        if marker:
            resolved[name] = int(marker.group(1))
        elif re.fullmatch(r"\d+", stem):
            resolved[name] = int(stem)

    if source_stem:
        source_key = source_stem.casefold()
        exact_name = next(
            (name for name, stem in stems.items() if stem.casefold() == source_key),
            None,
        )
        suffix_tokens: dict[str, str] = {}
        suffix_pattern = re.compile(rf"(?i)^{re.escape(source_stem)}_(\d+)$")
        for name, stem in stems.items():
            match = suffix_pattern.fullmatch(stem)
            if match:
                suffix_tokens[name] = match.group(1)

        if exact_name is not None:
            # Former IntegraPose custom output: unindexed base is frame zero.
            resolved[exact_name] = 0
            for name, token in suffix_tokens.items():
                resolved.setdefault(name, int(token))
        elif suffix_tokens:
            padded_zero_based = any(len(token) > 1 and token.startswith("0") for token in suffix_tokens.values())
            for name, token in suffix_tokens.items():
                if name in resolved:
                    continue
                raw_index = int(token)
                if padded_zero_based:
                    resolved[name] = raw_index
                elif raw_index >= 1:
                    # Native Ultralytics video labels are one-based.
                    resolved[name] = raw_index - 1

    # Detect the former unindexed-base convention even when no source path is
    # available. This handles numeric video stems such as ``mouse_1`` safely.
    unresolved_names = [name for name in names if name not in resolved]
    for base_name in unresolved_names:
        base_stem = stems[base_name]
        child_pattern = re.compile(rf"^{re.escape(base_stem)}_(\d+)$", re.IGNORECASE)
        children = {
            name: match.group(1)
            for name, stem in stems.items()
            if name != base_name and (match := child_pattern.fullmatch(stem))
        }
        if children:
            resolved[base_name] = 0
            for child_name, token in children.items():
                resolved.setdefault(child_name, int(token))

    # Final compatibility path for suffix-numbered external exports. Without
    # a source path their index base is unknowable, so preserve the token.
    for name in names:
        if name in resolved:
            continue
        parsed = parse_frame_index(name)
        if parsed is not None:
            resolved[name] = parsed

    frame_counts = Counter(resolved.values())
    duplicates = {frame for frame, count in frame_counts.items() if count > 1}
    if duplicates:
        details = []
        for frame in sorted(duplicates):
            colliding = sorted(name for name, index in resolved.items() if index == frame)
            details.append(f"frame {frame}: {', '.join(colliding)}")
        raise FrameIdentityError(
            "Multiple label files resolve to the same frame; refusing to merge them: "
            + "; ".join(details)
        )

    return resolved


def write_frame_label_manifest(
    labels_dir: str | Path,
    *,
    source: str | Path,
    max_det: int,
    class_names: Any = None,
    class_names_source: str = "",
    model_task: str = "",
) -> Path:
    """Write machine-readable provenance for IntegraPose label filenames."""
    directory = Path(labels_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / FRAME_LABEL_MANIFEST_FILENAME
    payload = {
        "schema_version": FRAME_LABEL_SCHEMA_VERSION,
        "frame_index_base": 0,
        "interval_semantics": "one_file_per_source_frame",
        "filename_pattern": "{source_stem}_frame_{frame_index:06d}.txt",
        "source_stem": Path(str(source)).stem,
        "max_det": int(max_det),
    }
    normalized_names = normalize_class_name_metadata(
        class_names,
        context="frame-label manifest class_names",
    )
    if normalized_names:
        payload["class_count"] = len(normalized_names)
        payload["class_names"] = {
            str(class_id): name for class_id, name in enumerate(normalized_names)
        }
        payload["class_names_source"] = (
            str(class_names_source or "model.names").strip() or "model.names"
        )
    normalized_task = str(model_task or "").strip().lower()
    if normalized_task:
        payload["model_task"] = normalized_task
    safe_write_json(target, payload, indent=2)
    return target


def load_frame_label_manifest(labels_dir: str | Path) -> dict:
    """Load and validate an IntegraPose frame-label manifest when present."""
    path = Path(labels_dir) / FRAME_LABEL_MANIFEST_FILENAME
    if not path.is_file():
        return {}
    payload = safe_read_json(path)
    if not isinstance(payload, dict):
        raise FrameIdentityError(f"Invalid frame-label manifest: {path}")
    if int(payload.get("schema_version", -1)) != FRAME_LABEL_SCHEMA_VERSION:
        raise FrameIdentityError(f"Unsupported frame-label manifest schema: {path}")
    if int(payload.get("frame_index_base", -1)) != 0:
        raise FrameIdentityError(f"Frame-label manifest is not zero-based: {path}")
    return payload


def load_frame_label_class_metadata(labels_dir: str | Path) -> dict[str, Any]:
    """Load validated model class metadata from a frame-label manifest."""
    payload = load_frame_label_manifest(labels_dir)
    if not payload or "class_names" not in payload:
        return {}
    names = normalize_class_name_metadata(
        payload.get("class_names"),
        context=f"{FRAME_LABEL_MANIFEST_FILENAME} class_names",
    )
    if not names:
        return {}
    try:
        declared_count = int(payload.get("class_count", len(names)))
    except (TypeError, ValueError) as exc:
        raise FrameIdentityError(
            f"Invalid class_count in {Path(labels_dir) / FRAME_LABEL_MANIFEST_FILENAME}."
        ) from exc
    if declared_count != len(names):
        raise FrameIdentityError(
            "Frame-label class schema mismatch: class_names contains "
            f"{len(names)} names but class_count is {declared_count}."
        )
    return {
        "class_names": names,
        "class_count": declared_count,
        "class_names_source": str(
            payload.get("class_names_source") or "inference label metadata"
        ).strip(),
        "model_task": str(payload.get("model_task") or "unknown").strip().lower(),
    }


def load_frame_label_class_names(labels_dir: str | Path) -> list[str]:
    """Return ID-ordered class names saved beside inference labels."""
    metadata = load_frame_label_class_metadata(labels_dir)
    return list(metadata.get("class_names") or [])


__all__ = [
    "FRAME_LABEL_MANIFEST_FILENAME",
    "FrameIdentityError",
    "extract_model_class_metadata",
    "frame_artifact_stem",
    "frame_label_filename",
    "load_frame_label_class_metadata",
    "load_frame_label_class_names",
    "load_frame_label_manifest",
    "normalize_class_name_metadata",
    "parse_frame_index",
    "resolve_frame_label_indices",
    "write_frame_label_manifest",
]
