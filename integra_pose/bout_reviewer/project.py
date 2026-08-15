from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from .models import (
    BEHAVIOR,
    EVENT_KINDS,
    FINGERPRINT_SCHEME,
    OBJECT_INTERACTION,
    ROI_CONCURRENT,
    ROI_EXCLUSIVE,
    PredictionBout,
    ProjectData,
    ReviewError,
    VideoRecord,
)


@dataclass(frozen=True)
class EventSource:
    event_kind: str
    suffix: str
    label_column: str
    required: bool


EVENT_SOURCES = (
    EventSource(
        ROI_CONCURRENT,
        "_roi_dwell_events.csv",
        "ROI Name",
        False,
    ),
    EventSource(
        ROI_EXCLUSIVE,
        "_roi_exclusive_dwell_events.csv",
        "ROI Name",
        False,
    ),
    EventSource(
        OBJECT_INTERACTION,
        "_object_interactions_dwell_events.csv",
        "Object ROI",
        False,
    ),
)

VIDEO_SUFFIXES = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".wmv",
}


def _inside(root: Path, candidate: Path) -> bool:
    try:
        common = os.path.commonpath((str(root), str(candidate)))
    except ValueError:
        return False
    return os.path.normcase(common) == os.path.normcase(str(root))


def _resolve_inside(
    root: Path,
    path: str | Path,
    *,
    label: str,
    must_exist: bool = True,
) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=must_exist)
    except (FileNotFoundError, OSError) as exc:
        raise ReviewError(f"{label} is missing or inaccessible: {candidate}") from exc
    if not _inside(root, resolved):
        raise ReviewError(
            f"{label} resolves outside the selected project root and was not opened: "
            f"{resolved}"
        )
    return resolved


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewError(f"Could not read valid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ReviewError(f"{label} must contain a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ReviewError(
            f"Could not hash provenance source file: {path}"
        ) from exc
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _display_reference(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if _inside(root, resolved):
        return resolved.relative_to(root).as_posix()
    return f"external:{resolved.as_posix()}"


def _existing_video(path: Path) -> Path | None:
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError):
        return None
    if not resolved.is_file() or resolved.suffix.lower() not in VIDEO_SUFFIXES:
        return None
    return resolved


def _source_video_candidates(
    *,
    root: Path,
    entry: dict[str, Any],
    session_source_path: Any,
    source_video_root: Path | None,
    folder_name: str,
    video_name: str,
) -> list[Path]:
    """Return exact source-video candidates without recursively searching."""

    candidates: list[Path] = []
    if source_video_root is not None:
        candidates.append(source_video_root / video_name)

    project_local = [
        root / video_name,
        root / "source_videos" / video_name,
        root / "raw_videos" / video_name,
        root / "videos" / video_name,
        root / "videos" / folder_name / video_name,
    ]

    local_session_candidates: list[Path] = []
    external_session_candidates: list[Path] = []
    raw_video_path = str(entry.get("video_path", "")).strip()
    if raw_video_path:
        recorded = Path(raw_video_path)
        recorded = recorded if recorded.is_absolute() else root / recorded
        (
            local_session_candidates
            if _inside(root, recorded.resolve(strict=False))
            else external_session_candidates
        ).append(recorded)

    raw_source_path = str(session_source_path or "").strip()
    if raw_source_path:
        source_path = Path(raw_source_path)
        if not source_path.is_absolute():
            source_path = root / source_path
        source_candidate = (
            source_path
            if source_path.suffix.lower() in VIDEO_SUFFIXES
            else source_path / video_name
        )
        (
            local_session_candidates
            if _inside(root, source_candidate.resolve(strict=False))
            else external_session_candidates
        ).append(source_candidate)

    candidates.extend(
        local_session_candidates
        + project_local
        + external_session_candidates
    )

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(os.path.abspath(str(candidate)))
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _select_display_video(
    *,
    root: Path,
    entry: dict[str, Any],
    session_source_path: Any,
    source_video_root: Path | None,
    analytics_dir: Path,
    run_dir: Path,
    video_id: str,
    folder_name: str,
    stem: str,
    video_name: str,
) -> tuple[Path, str, tuple[int, float, int, int], list[str]]:
    rejected: list[str] = []

    def usable(
        candidate: Path | None,
        role: str,
    ) -> tuple[Path, str, tuple[int, float, int, int], list[str]] | None:
        if candidate is None:
            return None
        try:
            metadata = _probe_video(candidate)
        except ReviewError as exc:
            rejected.append(f"{role}: {candidate} ({exc})")
            return None
        warnings = [
            "Skipped an unreadable review-video candidate: " + item
            for item in rejected
        ]
        return candidate, role, metadata, warnings

    analytics_annotated = _find_exact_or_unique(
        analytics_dir,
        f"{stem}_annotated.mp4",
        "*_annotated.mp4",
        required=False,
        label=f"{video_id} analytics review video",
    )
    selected = usable(analytics_annotated, "analytics_annotated")
    if selected is not None:
        return selected

    inference_annotated = _find_exact_or_unique(
        run_dir,
        f"{stem}_annotated.mp4",
        "*_annotated.mp4",
        required=False,
        label=f"{video_id} inference review video",
    )
    selected = usable(inference_annotated, "inference_annotated")
    if selected is not None:
        return selected

    candidates = _source_video_candidates(
        root=root,
        entry=entry,
        session_source_path=session_source_path,
        source_video_root=source_video_root,
        folder_name=folder_name,
        video_name=video_name,
    )
    for candidate in candidates:
        source_video = _existing_video(candidate)
        if source_video is not None:
            selected = usable(source_video, "original_source")
            if selected is not None:
                return selected

    attempted = "\n".join(f"  - {candidate}" for candidate in candidates)
    unreadable = (
        "Unreadable candidates:\n"
        + "\n".join(f"  - {item}" for item in rejected)
        + "\n"
        if rejected
        else ""
    )
    raise ReviewError(
        f"No review video is available for {video_id} ({video_name}). "
        "Neither an analytics/inference annotated MP4 nor an original source "
        "video could be opened. Exact source candidates checked:\n"
        f"{attempted}\n"
        f"{unreadable}"
        "Restore the session-recorded source video, copy it into a supported "
        "project-local location, or choose its directory with File > Set "
        "source-video folder (or --source-video-root)."
    )


def _int_value(raw: Any, field: str, source: Path, row_number: int) -> int:
    text = str(raw).strip()
    try:
        value = int(text)
    except ValueError as exc:
        raise ReviewError(
            f"{source.name} row {row_number}: {field} must be an integer; "
            f"observed {raw!r}."
        ) from exc
    return value


def _find_exact_or_unique(
    folder: Path,
    exact_name: str,
    glob_pattern: str,
    *,
    required: bool,
    label: str,
) -> Path | None:
    exact = folder / exact_name
    if exact.is_file():
        return exact
    candidates = sorted(folder.glob(glob_pattern))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates and not required:
        return None
    if not candidates:
        raise ReviewError(f"Required {label} is missing from {folder}.")
    raise ReviewError(
        f"Ambiguous {label} in {folder}; candidates={[path.name for path in candidates]}."
    )


def _session_output_path(
    root: Path,
    raw_path: Any,
    fallback: Path,
    label: str,
    *,
    required: bool = True,
) -> Path:
    text = str(raw_path or "").strip()
    if text:
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError):
            resolved = None
        if resolved is not None and _inside(root, resolved):
            return resolved
    return _resolve_inside(
        root,
        fallback,
        label=label,
        must_exist=required,
    )


def _recorded_path_matches(
    root: Path,
    raw_path: Any,
    resolved_path: Path,
) -> bool:
    text = str(raw_path or "").strip()
    if not text:
        return False
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        return candidate.resolve(strict=True) == resolved_path.resolve(strict=True)
    except (FileNotFoundError, OSError):
        return False


def _probe_video(path: Path) -> tuple[int, float, int, int]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ReviewError(f"OpenCV could not open review video: {path}")
        frame_count_raw = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width_raw = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_raw = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    values = (frame_count_raw, fps, width_raw, height_raw)
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ReviewError(f"Invalid review-video metadata for {path.name}: {values}")
    frame_count = int(round(frame_count_raw))
    width = int(round(width_raw))
    height = int(round(height_raw))
    if not math.isclose(frame_count_raw, frame_count, abs_tol=1e-6):
        raise ReviewError(
            f"Nonintegral frame count for {path.name}: {frame_count_raw}"
        )
    return frame_count, fps, width, height


def _prediction_id(
    video_id: str,
    event_kind: str,
    label: str,
    track_id: int,
    start_frame: int,
    end_frame: int,
    source_relative: str,
    source_row: int,
) -> str:
    payload = "|".join(
        (
            video_id,
            event_kind,
            label,
            str(track_id),
            str(start_frame),
            str(end_frame),
            source_relative,
            str(source_row),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _behavior_prediction_id(
    video_id: str,
    class_id: int | None,
    label: str,
    track_id: int,
    start_frame: int,
    end_frame: int,
) -> str:
    """Return a path-independent identity for one behavioral bout."""

    payload = "|".join(
        (
            video_id,
            BEHAVIOR,
            "" if class_id is None else str(class_id),
            label,
            str(track_id),
            str(start_frame),
            str(end_frame),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalized_header(value: Any) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _header_map(fieldnames: list[str] | None) -> dict[str, str]:
    return {
        _normalized_header(field): field
        for field in (fieldnames or [])
        if str(field).strip()
    }


def _behavior_class_catalog(session: dict[str, Any]) -> dict[int, str]:
    """Parse the model class-ID/name mapping recorded by IntegraPose."""

    candidates = [
        (
            session.get("model_capabilities", {}).get("class_names")
            if isinstance(session.get("model_capabilities"), dict)
            else None
        ),
        session.get("class_names"),
        session.get("names"),
    ]
    raw = next((candidate for candidate in candidates if candidate is not None), None)
    if raw is None:
        return {}
    catalog: dict[int, str] = {}
    if isinstance(raw, list):
        for class_id, name in enumerate(raw):
            text = str(name).strip()
            if not text:
                raise ReviewError(
                    f"batch_session.json has an empty class name for class_id={class_id}."
                )
            catalog[class_id] = text
    elif isinstance(raw, dict):
        for key, name in raw.items():
            try:
                class_id = int(str(key).strip())
            except ValueError as exc:
                raise ReviewError(
                    "batch_session.json class-name mapping has a noninteger "
                    f"class ID: {key!r}."
                ) from exc
            text = str(name).strip()
            if class_id < 0 or not text:
                raise ReviewError(
                    "batch_session.json class-name mapping contains an invalid "
                    f"entry: {key!r} -> {name!r}."
                )
            catalog[class_id] = text
    else:
        raise ReviewError(
            "batch_session.json class_names must be a list or an ID-to-name object."
        )
    duplicate_names = {
        name
        for name in catalog.values()
        if list(catalog.values()).count(name) > 1
    }
    if duplicate_names:
        raise ReviewError(
            "Behavior class names must be unique so reclassification is "
            f"unambiguous; duplicates={sorted(duplicate_names)}."
        )
    return dict(sorted(catalog.items()))


def _behavior_settings(session: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "single_animal_mode",
        "tracker_enabled",
        "tracker_config_path",
        "min_bout_frames",
        "max_gap_frames",
        "min_bout_seconds",
        "max_gap_seconds",
        "temporal_threshold_unit",
    )
    return {key: session.get(key) for key in keys if key in session}


def _existing_project_file(
    root: Path,
    raw_path: Any,
) -> Path | None:
    text = str(raw_path or "").strip()
    if not text:
        return None
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError):
        return None
    if not resolved.is_file() or not _inside(root, resolved):
        return None
    return resolved


def _first_existing_file(candidates: list[Path | None]) -> Path | None:
    seen: set[str] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        key = os.path.normcase(os.path.abspath(str(candidate)))
        if key in seen:
            continue
        seen.add(key)
        try:
            resolved = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError):
            continue
        if resolved.is_file():
            return resolved
    return None


def _load_detailed_behavior_bouts(
    *,
    root: Path,
    video_id: str,
    path: Path,
    frame_count: int,
    class_catalog: dict[int, str],
) -> tuple[list[PredictionBout], list[str]]:
    required = {
        "trackid",
        "behavior",
        "startframe",
        "endframe",
        "durationframes",
    }
    predictions: list[PredictionBout] = []
    warnings: list[str] = []
    seen: set[tuple[int | None, str, int, int, int]] = set()
    source_relative = _relative(path, root)
    reverse_catalog = {name: class_id for class_id, name in class_catalog.items()}
    warned_unknown: set[str] = set()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = _header_map(reader.fieldnames)
            missing = required - set(columns)
            if missing:
                raise ReviewError(
                    f"{path.name} is missing behavioral-bout columns "
                    f"{sorted(missing)}."
                )
            for row_number, row in enumerate(reader, start=2):
                behavior = str(row.get(columns["behavior"], "")).strip()
                if not behavior:
                    raise ReviewError(
                        f"{path.name} row {row_number}: empty Behavior."
                    )
                track_id = _int_value(
                    row.get(columns["trackid"]), "Track ID", path, row_number
                )
                start = _int_value(
                    row.get(columns["startframe"]), "Start Frame", path, row_number
                )
                end = _int_value(
                    row.get(columns["endframe"]), "End Frame", path, row_number
                )
                duration = _int_value(
                    row.get(columns["durationframes"]),
                    "Duration (Frames)",
                    path,
                    row_number,
                )
                if track_id < 0:
                    raise ReviewError(
                        f"{path.name} row {row_number}: Track ID cannot be negative."
                    )
                if start < 0 or end < start or end >= frame_count:
                    raise ReviewError(
                        f"{path.name} row {row_number}: invalid inclusive behavior "
                        f"interval [{start}, {end}] for a {frame_count}-frame video."
                    )
                if duration != end - start + 1:
                    raise ReviewError(
                        f"{path.name} row {row_number}: Duration (Frames)={duration}, "
                        f"but inclusive boundaries imply {end - start + 1}."
                    )
                class_id: int | None
                if "classid" in columns and str(
                    row.get(columns["classid"], "")
                ).strip():
                    class_id = _int_value(
                        row.get(columns["classid"]), "Class ID", path, row_number
                    )
                    if class_id < 0:
                        raise ReviewError(
                            f"{path.name} row {row_number}: Class ID cannot be negative."
                        )
                    expected_name = class_catalog.get(class_id)
                    if expected_name is not None and behavior != expected_name:
                        warnings.append(
                            f"{path.name} row {row_number}: Behavior {behavior!r} "
                            f"does not match JSON class_id {class_id} name "
                            f"{expected_name!r}; the JSON name was used."
                        )
                        behavior = expected_name
                else:
                    class_id = reverse_catalog.get(behavior)
                    if class_id is None:
                        raise ReviewError(
                            f"{path.name} row {row_number}: Behavior {behavior!r} "
                            "cannot be mapped to a class ID from batch_session.json."
                        )
                if class_id not in class_catalog and behavior not in warned_unknown:
                    warnings.append(
                        f"Class ID {class_id} is not named in batch_session.json; "
                        f"{behavior!r} was retained from {path.name}."
                    )
                    warned_unknown.add(behavior)
                key = (class_id, behavior, track_id, start, end)
                if key in seen:
                    raise ReviewError(
                        f"{path.name} row {row_number}: duplicate behavior bout {key}."
                    )
                seen.add(key)
                predictions.append(
                    PredictionBout(
                        prediction_id=_behavior_prediction_id(
                            video_id,
                            class_id,
                            behavior,
                            track_id,
                            start,
                            end,
                        ),
                        video_id=video_id,
                        event_kind=BEHAVIOR,
                        label=behavior,
                        track_id=track_id,
                        start_frame=start,
                        end_frame=end,
                        source_file=source_relative,
                        source_row=row_number,
                        class_id=class_id,
                    )
                )
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ReviewError(f"Could not read behavioral bout table {path}.") from exc
    return predictions, warnings


def _batch_row_matches_video(
    raw: Any,
    *,
    video_id: str,
    video_name: str,
    stem: str,
) -> bool:
    text = str(raw or "").strip()
    if not text:
        return False
    candidate = Path(text)
    values = {
        text.casefold(),
        candidate.name.casefold(),
        candidate.stem.casefold(),
    }
    return bool(
        values
        & {
            video_id.casefold(),
            video_name.casefold(),
            stem.casefold(),
        }
    )


def _load_frame_behavior_bouts(
    *,
    root: Path,
    video_id: str,
    video_name: str,
    path: Path,
    frame_count: int,
    class_catalog: dict[int, str],
    shared_batch_table: bool,
    batch_video_count: int,
) -> tuple[list[PredictionBout], list[str]]:
    required = {"frame", "classid", "trackid"}
    source_relative = _relative(path, root)
    warnings: list[str] = []
    frames_by_channel: dict[tuple[int, int, str], list[tuple[int, int]]] = {}
    duplicate_count = 0
    seen_rows: set[tuple[int, int, int]] = set()
    unknown_class_ids: set[int] = set()
    stem = Path(video_name).stem
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = _header_map(reader.fieldnames)
            missing = required - set(columns)
            if missing:
                raise ReviewError(
                    f"{path.name} is missing required frame-level behavior "
                    f"columns {sorted(missing)}."
                )
            video_column = next(
                (
                    columns[key]
                    for key in (
                        "videoid",
                        "videoname",
                        "filename",
                        "sourcevideo",
                        "video",
                    )
                    if key in columns
                ),
                None,
            )
            if shared_batch_table and batch_video_count > 1 and video_column is None:
                raise ReviewError(
                    f"Shared {path.name} covers a {batch_video_count}-video batch "
                    "but has no video_id/video_name column for safe attribution."
                )
            for row_number, row in enumerate(reader, start=2):
                if video_column is not None and not _batch_row_matches_video(
                    row.get(video_column),
                    video_id=video_id,
                    video_name=video_name,
                    stem=stem,
                ):
                    continue
                frame = _int_value(
                    row.get(columns["frame"]), "frame", path, row_number
                )
                class_id = _int_value(
                    row.get(columns["classid"]), "class_id", path, row_number
                )
                track_id = _int_value(
                    row.get(columns["trackid"]), "track_id", path, row_number
                )
                if frame < 0 or frame >= frame_count:
                    raise ReviewError(
                        f"{path.name} row {row_number}: frame {frame} is outside "
                        f"the valid range 0..{frame_count - 1}."
                    )
                if class_id < 0 or track_id < 0:
                    raise ReviewError(
                        f"{path.name} row {row_number}: class_id and track_id "
                        "must be nonnegative integers."
                    )
                key = (frame, class_id, track_id)
                if key in seen_rows:
                    duplicate_count += 1
                    continue
                seen_rows.add(key)
                label = class_catalog.get(class_id)
                if label is None:
                    label = f"Class {class_id}"
                    unknown_class_ids.add(class_id)
                frames_by_channel.setdefault(
                    (class_id, track_id, label), []
                ).append((frame, row_number))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ReviewError(f"Could not read frame-level behavior table {path}.") from exc

    if duplicate_count:
        warnings.append(
            f"{path.name}: ignored {duplicate_count} duplicate "
            "(frame, class_id, track_id) row(s)."
        )
    if unknown_class_ids:
        warnings.append(
            f"{path.name}: class IDs {sorted(unknown_class_ids)} have no names "
            "in batch_session.json and are displayed as Class <ID>."
        )

    predictions: list[PredictionBout] = []
    for (class_id, track_id, label), observed in sorted(
        frames_by_channel.items()
    ):
        ordered = sorted(observed)
        start = end = ordered[0][0]
        source_row = ordered[0][1]
        for frame, row_number in ordered[1:]:
            if frame == end + 1:
                end = frame
                continue
            predictions.append(
                PredictionBout(
                    prediction_id=_behavior_prediction_id(
                        video_id,
                        class_id,
                        label,
                        track_id,
                        start,
                        end,
                    ),
                    video_id=video_id,
                    event_kind=BEHAVIOR,
                    label=label,
                    track_id=track_id,
                    start_frame=start,
                    end_frame=end,
                    source_file=source_relative,
                    source_row=source_row,
                    class_id=class_id,
                )
            )
            start = end = frame
            source_row = row_number
        predictions.append(
            PredictionBout(
                prediction_id=_behavior_prediction_id(
                    video_id,
                    class_id,
                    label,
                    track_id,
                    start,
                    end,
                ),
                video_id=video_id,
                event_kind=BEHAVIOR,
                label=label,
                track_id=track_id,
                start_frame=start,
                end_frame=end,
                source_file=source_relative,
                source_row=source_row,
                class_id=class_id,
            )
        )
    return predictions, warnings


def _load_bouts(
    root: Path,
    video_id: str,
    source: EventSource,
    path: Path | None,
    frame_count: int,
) -> list[PredictionBout]:
    if path is None:
        return []
    required_columns = {
        "Track ID",
        source.label_column,
        "Start Frame",
        "End Frame",
        "Duration (Frames)",
    }
    rows: list[PredictionBout] = []
    seen: set[tuple[str, int, int, int]] = set()
    source_relative = _relative(path, root)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = required_columns - set(reader.fieldnames or [])
            if missing:
                raise ReviewError(
                    f"{path.name} is missing required columns {sorted(missing)}."
                )
            for row_number, row in enumerate(reader, start=2):
                label = str(row.get(source.label_column, "")).strip()
                if not label:
                    raise ReviewError(
                        f"{path.name} row {row_number}: empty {source.label_column}."
                    )
                track_id = _int_value(
                    row.get("Track ID"), "Track ID", path, row_number
                )
                start = _int_value(
                    row.get("Start Frame"), "Start Frame", path, row_number
                )
                end = _int_value(
                    row.get("End Frame"), "End Frame", path, row_number
                )
                duration = _int_value(
                    row.get("Duration (Frames)"),
                    "Duration (Frames)",
                    path,
                    row_number,
                )
                if start < 0 or end < start or end >= frame_count:
                    raise ReviewError(
                        f"{path.name} row {row_number}: invalid inclusive interval "
                        f"[{start}, {end}] for a {frame_count}-frame video."
                    )
                if duration != end - start + 1:
                    raise ReviewError(
                        f"{path.name} row {row_number}: Duration (Frames)={duration}, "
                        f"but inclusive boundaries imply {end - start + 1}."
                    )
                key = (label, track_id, start, end)
                if key in seen:
                    raise ReviewError(
                        f"{path.name} row {row_number}: duplicate bout {key}."
                    )
                seen.add(key)
                prediction_id = _prediction_id(
                    video_id,
                    source.event_kind,
                    label,
                    track_id,
                    start,
                    end,
                    source_relative,
                    row_number,
                )
                rows.append(
                    PredictionBout(
                        prediction_id=prediction_id,
                        video_id=video_id,
                        event_kind=source.event_kind,
                        label=label,
                        track_id=track_id,
                        start_frame=start,
                        end_frame=end,
                        source_file=source_relative,
                        source_row=row_number,
                    )
                )
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ReviewError(f"Could not read event table {path}.") from exc
    return rows


def _load_markers(path: Path) -> set[tuple[str, str, int, int]]:
    required = {"Event Type", "Target Name", "Track ID", "Frame"}
    markers: set[tuple[str, str, int, int]] = set()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ReviewError(
                    f"{path.name} is missing marker columns {sorted(missing)}."
                )
            for row_number, row in enumerate(reader, start=2):
                event_type = str(row.get("Event Type", "")).strip().lower()
                target = str(row.get("Target Name", "")).strip()
                if event_type not in {"entry", "exit"} or not target:
                    raise ReviewError(
                        f"{path.name} row {row_number}: invalid event type or target."
                    )
                track_id = _int_value(
                    row.get("Track ID"), "Track ID", path, row_number
                )
                frame = _int_value(row.get("Frame"), "Frame", path, row_number)
                markers.add((event_type, target, track_id, frame))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ReviewError(f"Could not read marker table {path}.") from exc
    return markers


def _crosscheck_markers(
    predictions: list[PredictionBout],
    marker_path: Path | None,
    event_kind: str,
) -> list[str]:
    if marker_path is None:
        if predictions:
            return [
                f"{event_kind}: boundary marker CSV is missing; intervals remain editable."
            ]
        return []
    try:
        observed = _load_markers(marker_path)
    except ReviewError as exc:
        return [
            f"{event_kind}: optional boundary marker CSV could not be validated "
            f"and was ignored ({exc}). Dwell intervals remain editable."
        ]
    expected = {
        boundary
        for bout in predictions
        for boundary in (
            ("entry", bout.label, bout.track_id, bout.start_frame),
            ("exit", bout.label, bout.track_id, bout.end_frame),
        )
    }
    if observed == expected:
        return []
    missing = sorted(expected - observed)[:10]
    extra = sorted(observed - expected)[:10]
    return [
        f"{event_kind}: interval/marker mismatch in {marker_path.name}; "
        f"missing examples={missing}, extra examples={extra}."
    ]


def _catalog(entry: dict[str, Any], key: str) -> list[str]:
    value = entry.get(key)
    if isinstance(value, dict):
        return sorted(str(name) for name in value)
    return []


def _video_fingerprint(
    *,
    video_id: str,
    frame_count: int,
    prediction_source_hashes: dict[str, str],
    predictions: list[PredictionBout],
) -> str:
    """Stable prediction fingerprint independent of drive letter and file mtime."""

    payload = {
        "scheme": FINGERPRINT_SCHEME,
        "video_id": video_id,
        "video_alignment": {
            "frame_count": int(frame_count),
        },
        "prediction_sources": sorted(
            (
                Path(source_reference).name.casefold(),
                digest,
            )
            for source_reference, digest in prediction_source_hashes.items()
        ),
        "bouts": sorted(
            (
                bout.event_kind,
                -1 if bout.class_id is None else bout.class_id,
                bout.label,
                bout.track_id,
                bout.start_frame,
                bout.end_frame,
            )
            for bout in predictions
        ),
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_portable_batch_project(
    root_path: str | Path,
    *,
    source_video_root: str | Path | None = None,
) -> ProjectData:
    try:
        root = Path(root_path).resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ReviewError(f"Project root is missing or inaccessible: {root_path}") from exc
    if not root.is_dir():
        raise ReviewError(f"Project root is not a directory: {root}")
    resolved_source_video_root: Path | None = None
    if source_video_root is not None:
        candidate = Path(source_video_root)
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            resolved_source_video_root = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise ReviewError(
                f"Source-video directory is missing or inaccessible: {candidate}"
            ) from exc
        if not resolved_source_video_root.is_dir():
            raise ReviewError(
                f"Source-video override is not a directory: "
                f"{resolved_source_video_root}"
            )
    session_path = _resolve_inside(
        root,
        "batch_session.json",
        label="batch_session.json",
        must_exist=True,
    )
    session = _read_json(session_path, "batch session")
    session_id = str(session.get("session_id", "")).strip()
    if not session_id:
        raise ReviewError("batch_session.json has no session_id.")
    raw_videos = session.get("videos")
    if not isinstance(raw_videos, list) or not raw_videos:
        raise ReviewError("batch_session.json has no video entries.")
    behavior_class_catalog = _behavior_class_catalog(session)
    behavior_settings = _behavior_settings(session)
    single_animal_mode = bool(session.get("single_animal_mode", True))

    videos: list[VideoRecord] = []
    project_warnings: list[str] = []
    seen_video_ids: set[str] = set()
    seen_stems: set[str] = set()

    for index, entry in enumerate(raw_videos):
        if not isinstance(entry, dict):
            raise ReviewError(f"Session video entry {index} is not an object.")
        if bool(entry.get("excluded", False)):
            continue
        video_id = str(entry.get("video_id", "")).strip()
        video_name = str(entry.get("video_name", "")).strip()
        if not video_id or not video_name:
            raise ReviewError(f"Session video entry {index} lacks ID or name.")
        stem = Path(video_name).stem
        if video_id in seen_video_ids or stem in seen_stems:
            raise ReviewError(
                f"Duplicate video identity in batch session: {video_id!r}, {stem!r}."
            )
        seen_video_ids.add(video_id)
        seen_stems.add(stem)

        folder_name = f"{video_id}_{stem}"
        run_leaf = Path(str(entry.get("run_output_dir", "infer"))).name or "infer"
        run_dir = _session_output_path(
            root,
            entry.get("run_output_dir"),
            Path("videos") / folder_name / "inference" / run_leaf,
            f"{video_id} inference run",
            required=False,
        )
        analytics_dir = _session_output_path(
            root,
            entry.get("analytics_output_dir"),
            Path("videos") / folder_name / "analytics",
            f"{video_id} analytics directory",
            required=False,
        )
        recorded_analytics_path = entry.get("analytics_output_dir")
        used_portable_analytics_fallback = bool(
            str(recorded_analytics_path or "").strip()
        ) and not _recorded_path_matches(
            root,
            recorded_analytics_path,
            analytics_dir,
        )
        (
            display_video,
            display_video_role,
            video_metadata,
            video_selection_warnings,
        ) = _select_display_video(
            root=root,
            entry=entry,
            session_source_path=session.get("source_path"),
            source_video_root=resolved_source_video_root,
            analytics_dir=analytics_dir,
            run_dir=run_dir,
            video_id=video_id,
            folder_name=folder_name,
            stem=stem,
            video_name=video_name,
        )
        frame_count, fps, width, height = video_metadata

        predictions: list[PredictionBout] = []
        source_paths: list[Path] = []
        prediction_source_paths: list[Path] = []
        source_by_kind: dict[str, Path | None] = {}
        for source in EVENT_SOURCES:
            path = _find_exact_or_unique(
                analytics_dir,
                f"{stem}{source.suffix}",
                f"*{source.suffix}",
                required=source.required,
                label=f"{video_id} {source.event_kind} bouts",
            )
            source_by_kind[source.event_kind] = path
            if path is not None:
                source_paths.append(path)
                prediction_source_paths.append(path)
            predictions.extend(
                _load_bouts(root, video_id, source, path, frame_count)
            )

        video_warnings: list[str] = list(video_selection_warnings)
        detailed_behavior_path = _first_existing_file(
            [
                analytics_dir / f"{stem}_detailed_bouts.csv",
                analytics_dir / "detailed_bouts.csv",
                _existing_project_file(root, entry.get("detailed_bouts_csv")),
            ]
        )
        frame_behavior_path: Path | None = None
        behavior_source_kind = ""
        if detailed_behavior_path is not None:
            behavior_predictions, behavior_warnings = (
                _load_detailed_behavior_bouts(
                    root=root,
                    video_id=video_id,
                    path=detailed_behavior_path,
                    frame_count=frame_count,
                    class_catalog=behavior_class_catalog,
                )
            )
            behavior_source_path = detailed_behavior_path
            behavior_source_kind = "finalized_detailed_bouts"
        else:
            frame_behavior_path = _first_existing_file(
                [
                    _existing_project_file(root, entry.get("batch_results_csv")),
                    _existing_project_file(root, session.get("batch_results_csv")),
                    analytics_dir / f"{stem}_batch_results.csv",
                    analytics_dir / "batch_results.csv",
                    run_dir / "batch_results.csv",
                    root / "batch_results.csv",
                ]
            )
            if frame_behavior_path is not None:
                shared_batch_table = (
                    frame_behavior_path.resolve()
                    == (root / "batch_results.csv").resolve(strict=False)
                )
                behavior_predictions, behavior_warnings = (
                    _load_frame_behavior_bouts(
                        root=root,
                        video_id=video_id,
                        video_name=video_name,
                        path=frame_behavior_path,
                        frame_count=frame_count,
                        class_catalog=behavior_class_catalog,
                        shared_batch_table=shared_batch_table,
                        batch_video_count=len(raw_videos),
                    )
                )
                behavior_source_path = frame_behavior_path
                behavior_source_kind = "postprocessed_frame_results"
            else:
                behavior_predictions = []
                behavior_warnings = [
                    "Behavioral bout review is unavailable: neither the "
                    "finalized *_detailed_bouts.csv nor batch_results.csv "
                    "was found in a supported project-relative location."
                ]
                behavior_source_path = None
        predictions.extend(behavior_predictions)
        video_warnings.extend(behavior_warnings)
        if behavior_source_path is not None:
            source_paths.append(behavior_source_path)
            prediction_source_paths.append(behavior_source_path)
        if not any(path is not None for path in source_by_kind.values()) and (
            behavior_source_path is None
        ):
            raise ReviewError(
                f"No supported interval source exists for {video_id}. Retain "
                "at least one ROI/object dwell table, a finalized "
                "*_detailed_bouts.csv, or batch_results.csv."
            )
        if used_portable_analytics_fallback:
            video_warnings.append(
                "The session-recorded analytics path was stale, unavailable, "
                "or outside this batch root. The reviewer used the portable "
                f"project-relative directory: {_relative(analytics_dir, root)}."
            )
        if display_video_role == "original_source":
            video_warnings.append(
                "No annotated review video was found; the original source video "
                "is being used. Pose/ROI overlays are unavailable, but timeline "
                "review and scoring remain fully functional."
            )
            if not _inside(root, display_video):
                video_warnings.append(
                    "The original source video is outside the selected batch root. "
                    "Only the exact path recorded by the session or supplied with "
                    "--source-video-root is opened read-only."
                )
        roi_markers = _find_exact_or_unique(
            analytics_dir,
            f"{stem}_roi_events.csv",
            "*_roi_events.csv",
            required=False,
            label=f"{video_id} ROI markers",
        )
        object_markers = _find_exact_or_unique(
            analytics_dir,
            f"{stem}_object_events.csv",
            "*_object_events.csv",
            required=False,
            label=f"{video_id} object markers",
        )
        if roi_markers is not None:
            source_paths.append(roi_markers)
        if object_markers is not None:
            source_paths.append(object_markers)
        concurrent_predictions = [
            bout for bout in predictions if bout.event_kind == ROI_CONCURRENT
        ]
        object_predictions = [
            bout for bout in predictions if bout.event_kind == OBJECT_INTERACTION
        ]
        video_warnings.extend(
            _crosscheck_markers(
                concurrent_predictions,
                roi_markers,
                ROI_CONCURRENT,
            )
        )
        video_warnings.extend(
            _crosscheck_markers(
                object_predictions,
                object_markers,
                OBJECT_INTERACTION,
            )
        )

        object_per_frame = _find_exact_or_unique(
            analytics_dir,
            f"{stem}_object_interactions_per_frame.csv",
            "*_object_interactions_per_frame.csv",
            required=False,
            label=f"{video_id} object per-frame table",
        )
        if object_per_frame is not None:
            source_paths.append(object_per_frame)
        if (
            source_by_kind[OBJECT_INTERACTION] is None
            and object_markers is None
            and object_per_frame is not None
        ):
            video_warnings.append(
                "No object-interaction dwell/event CSV was emitted; the per-frame "
                "object table exists, so this is treated as zero predicted object bouts."
            )
        elif source_by_kind[OBJECT_INTERACTION] is None and object_per_frame is None:
            video_warnings.append(
                "Object-interaction outputs are unavailable for this video."
            )

        label_catalog = {
            ROI_CONCURRENT: _catalog(entry, "rois"),
            ROI_EXCLUSIVE: _catalog(entry, "rois"),
            OBJECT_INTERACTION: _catalog(entry, "object_rois"),
            BEHAVIOR: [
                behavior_class_catalog[class_id]
                for class_id in sorted(behavior_class_catalog)
            ],
        }
        effective_behavior_catalog = dict(behavior_class_catalog)
        for bout in behavior_predictions:
            if bout.class_id is not None:
                effective_behavior_catalog.setdefault(bout.class_id, bout.label)
        for kind in EVENT_KINDS:
            observed = sorted(
                {bout.label for bout in predictions if bout.event_kind == kind}
            )
            label_catalog[kind] = sorted(set(label_catalog[kind]) | set(observed))
        track_ids = sorted({bout.track_id for bout in predictions})
        if not track_ids and bool(session.get("single_animal_mode", True)):
            track_ids = [0]
        behavior_tracks = sorted(
            {
                bout.track_id
                for bout in behavior_predictions
            }
        )
        if single_animal_mode and len(behavior_tracks) > 1:
            video_warnings.append(
                "batch_session.json declares single_animal_mode=true, but "
                f"behavioral predictions contain tracks {behavior_tracks}."
            )

        unique_source_paths = sorted(
            set(source_paths),
            key=lambda item: item.as_posix().casefold(),
        )
        prediction_source_set = set(prediction_source_paths)
        source_file_hashes: dict[str, str] = {}
        for path in unique_source_paths:
            reference = _relative(path, root)
            try:
                source_file_hashes[reference] = _sha256(path)
            except ReviewError as exc:
                if path in prediction_source_set:
                    raise
                video_warnings.append(
                    "Optional supporting file could not be hashed for "
                    f"provenance and was omitted from the hash manifest: {exc}"
                )
        prediction_source_hashes = {
            _relative(path, root): source_file_hashes[_relative(path, root)]
            for path in prediction_source_paths
        }
        fingerprint = _video_fingerprint(
            video_id=video_id,
            frame_count=frame_count,
            prediction_source_hashes=prediction_source_hashes,
            predictions=predictions,
        )
        path_provenance = {
            "resolution_policy": "project-relative-first-v2",
            "resolved_analytics_dir": _relative(analytics_dir, root),
            "resolved_inference_dir": _display_reference(run_dir, root),
            "resolved_display_video": _display_reference(display_video, root),
            "session_video_path": str(entry.get("video_path", "") or ""),
            "session_source_path": str(session.get("source_path", "") or ""),
            "session_analytics_output_dir": str(
                entry.get("analytics_output_dir", "") or ""
            ),
            "session_run_output_dir": str(
                entry.get("run_output_dir", "") or ""
            ),
            "display_video_size_bytes": str(display_video.stat().st_size),
            "behavior_source": (
                _relative(behavior_source_path, root)
                if behavior_source_path is not None
                else ""
            ),
            "behavior_source_kind": behavior_source_kind,
        }
        videos.append(
            VideoRecord(
                video_id=video_id,
                video_name=video_name,
                video_stem=stem,
                subject_id=str(entry.get("subject_id", "")).strip(),
                group=str(entry.get("group", "")).strip(),
                time_point=str(entry.get("time_point", "")).strip(),
                display_video=display_video,
                display_video_relative=_display_reference(display_video, root),
                analytics_dir=analytics_dir,
                run_dir=run_dir,
                fps=fps,
                frame_count=frame_count,
                width=width,
                height=height,
                source_fingerprint=fingerprint,
                label_catalog=label_catalog,
                track_ids=track_ids,
                predictions=sorted(
                    predictions,
                    key=lambda bout: (
                        bout.event_kind,
                        -1 if bout.class_id is None else bout.class_id,
                        bout.label,
                        bout.track_id,
                        bout.start_frame,
                        bout.end_frame,
                    ),
                ),
                source_files=[
                    _relative(path, root) for path in unique_source_paths
                ],
                behavior_classes=effective_behavior_catalog,
                single_animal_mode=single_animal_mode,
                behavior_settings=behavior_settings,
                display_video_role=display_video_role,
                source_file_hashes=source_file_hashes,
                path_provenance=path_provenance,
                warnings=video_warnings,
            )
        )
        project_warnings.extend(
            f"{video_id}: {warning}" for warning in video_warnings
        )

    if not videos:
        raise ReviewError("No non-excluded videos could be loaded.")
    videos.sort(key=lambda video: video.video_id)
    return ProjectData(
        root=root,
        session_path=session_path,
        session_id=session_id,
        project_label=root.name,
        videos=videos,
        warnings=project_warnings,
    )


def load_project(
    root_path: str | Path,
    *,
    source_video_root: str | Path | None = None,
) -> ProjectData:
    """Load either a portable batch project or one IntegraPose analytics run.

    The integrated reviewer uses ``run_manifest.json`` as its native boundary.
    The portable ``batch_session.json`` loader remains available so the same
    review core can still open previously created standalone projects.
    """

    candidate = Path(root_path).expanduser()
    if candidate.is_file():
        if candidate.name.casefold() != "run_manifest.json":
            raise ReviewError(
                "Select run_manifest.json, its analytics folder, or a portable "
                f"batch root; received {candidate}."
            )
        from .manifest_project import load_run_manifest_project

        return load_run_manifest_project(
            candidate,
            source_video_root=source_video_root,
        )

    direct_manifest = candidate / "run_manifest.json"
    if direct_manifest.is_file():
        from .manifest_project import load_run_manifest_project

        return load_run_manifest_project(
            direct_manifest,
            source_video_root=source_video_root,
        )

    return _load_portable_batch_project(
        candidate,
        source_video_root=source_video_root,
    )
