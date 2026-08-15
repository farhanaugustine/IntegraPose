from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2

from .models import (
    BEHAVIOR,
    FINGERPRINT_SCHEME,
    OBJECT_INTERACTION,
    ROI_CONCURRENT,
    ROI_EXCLUSIVE,
    PredictionBout,
    ProjectData,
    ReviewError,
    VideoRecord,
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


def _reference(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if _inside(root, resolved):
        return resolved.relative_to(root).as_posix()
    return f"external:{resolved.as_posix()}"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewError(f"Could not read a valid run manifest: {path}") from exc
    if not isinstance(value, dict):
        raise ReviewError(f"run_manifest.json must contain a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ReviewError(f"Could not hash review source: {path}") from exc
    return digest.hexdigest()


def _normalized_header(value: Any) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _header_map(fieldnames: Iterable[str] | None) -> dict[str, str]:
    return {
        _normalized_header(field): str(field)
        for field in (fieldnames or ())
        if str(field).strip()
    }


def _integer(
    raw: Any,
    *,
    field: str,
    path: Path,
    row_number: int,
) -> int:
    try:
        numeric = float(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ReviewError(
            f"{path.name} row {row_number}: {field} must be an integer."
        ) from exc
    if not numeric.is_integer():
        raise ReviewError(
            f"{path.name} row {row_number}: {field} must be an integer."
        )
    return int(numeric)


def _resolve_file(
    root: Path,
    raw: Any = "",
    *,
    fallbacks: Iterable[Path] = (),
) -> Path | None:
    candidates: list[Path] = []
    text = str(raw or "").strip()
    if text:
        recorded = Path(text).expanduser()
        candidates.append(recorded if recorded.is_absolute() else root / recorded)
        candidates.append(root / recorded.name)
    candidates.extend(fallbacks)
    seen: set[str] = set()
    for candidate in candidates:
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


def _probe_video(path: Path) -> tuple[int, float, int, int]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ReviewError(f"OpenCV could not open review video: {path}")
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()
    if frame_count <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise ReviewError(
            "Review video metadata is invalid "
            f"(frames={frame_count}, fps={fps}, size={width}x{height}): {path}"
        )
    return frame_count, fps, width, height


def _unique_direct_match(root: Path, pattern: str) -> Path | None:
    matches = sorted(
        (path.resolve() for path in root.glob(pattern) if path.is_file()),
        key=lambda path: path.name.casefold(),
    )
    if len(matches) > 1:
        raise ReviewError(
            f"More than one file matches {pattern!r} in {root}; "
            "the reviewer will not guess which file is authoritative."
        )
    return matches[0] if matches else None


def _select_display_video(
    *,
    root: Path,
    manifest: dict[str, Any],
    base_name: str,
    source_video_root: Path | None,
) -> tuple[Path, str, tuple[int, float, int, int], list[str]]:
    inputs = manifest.get("inputs") if isinstance(manifest.get("inputs"), dict) else {}
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    source_text = str(inputs.get("video_file") or "").strip()
    source_name = Path(source_text).name if source_text else f"{base_name}.mp4"

    exact_annotated = _resolve_file(
        root,
        outputs.get("annotated_video")
        or outputs.get("annotated_video_path"),
        fallbacks=(root / f"{base_name}_annotated.mp4",),
    )
    candidates: list[tuple[Path | None, str]] = [
        (exact_annotated, "analytics_annotated"),
    ]
    # Only attempt the generic discovery rule when an exact, authoritative
    # annotated-video path was not available. This avoids rejecting a run just
    # because unrelated annotated previews also live beside the chosen file.
    if exact_annotated is None:
        candidates.append(
            (_unique_direct_match(root, "*_annotated.mp4"), "analytics_annotated")
        )
    candidates.append(
        (
            _resolve_file(
                root,
                source_text,
                fallbacks=(
                    root / source_name,
                    root / "source_videos" / source_name,
                ),
            ),
            "original_source",
        )
    )
    if source_video_root is not None:
        candidates.append(
            (
                _resolve_file(
                    root,
                    "",
                    fallbacks=(source_video_root / source_name,),
                ),
                "source_video_override",
            )
        )

    warnings: list[str] = []
    seen: set[str] = set()
    for candidate, role in candidates:
        if candidate is None or candidate.suffix.casefold() not in VIDEO_SUFFIXES:
            continue
        key = os.path.normcase(str(candidate))
        if key in seen:
            continue
        seen.add(key)
        try:
            metadata = _probe_video(candidate)
        except ReviewError as exc:
            warnings.append(str(exc))
            continue
        if role != "analytics_annotated":
            warnings.append(
                "No readable annotated analytics video was found. The reviewer "
                "is using the original source video; timeline review remains "
                "available, but analytics overlays are not shown."
            )
        return candidate, role, metadata, warnings

    raise ReviewError(
        "No readable review video was found. Retain an annotated analytics "
        f"video named {base_name}_annotated.mp4, retain the source video "
        "recorded in run_manifest.json, or provide --source-video-root."
    )


def _class_catalog(manifest: dict[str, Any]) -> dict[int, str]:
    inputs = manifest.get("inputs") if isinstance(manifest.get("inputs"), dict) else {}
    raw = inputs.get("behavior_names")
    catalog: dict[int, str] = {}
    if isinstance(raw, list):
        catalog = {
            index: str(name).strip()
            for index, name in enumerate(raw)
            if str(name).strip()
        }
    elif isinstance(raw, dict):
        for key, name in raw.items():
            try:
                class_id = int(str(key).strip())
            except ValueError as exc:
                raise ReviewError(
                    f"run_manifest.json has a noninteger behavior class ID: {key!r}"
                ) from exc
            text = str(name).strip()
            if class_id < 0 or not text:
                raise ReviewError(
                    "run_manifest.json has an invalid behavior class mapping: "
                    f"{key!r} -> {name!r}"
                )
            catalog[class_id] = text
    return dict(sorted(catalog.items()))


def _prediction_id(
    *,
    video_id: str,
    event_kind: str,
    label: str,
    class_id: int | None,
    track_id: int,
    start_frame: int,
    end_frame: int,
) -> str:
    payload = "|".join(
        (
            video_id,
            event_kind,
            "" if class_id is None else str(class_id),
            label,
            str(track_id),
            str(start_frame),
            str(end_frame),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_interval_table(
    *,
    root: Path,
    path: Path,
    video_id: str,
    frame_count: int,
    event_kind: str,
    label_header: str,
) -> list[PredictionBout]:
    predictions: list[PredictionBout] = []
    source_reference = _reference(path, root)
    seen: set[tuple[str, int, int, int]] = set()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = _header_map(reader.fieldnames)
            required = {
                _normalized_header(label_header),
                "trackid",
                "startframe",
                "endframe",
                "durationframes",
            }
            missing = required - set(columns)
            if missing:
                raise ReviewError(
                    f"{path.name} is missing required interval columns: "
                    f"{sorted(missing)}"
                )
            for row_number, row in enumerate(reader, start=2):
                label = str(
                    row.get(columns[_normalized_header(label_header)], "")
                ).strip()
                track_id = _integer(
                    row.get(columns["trackid"]),
                    field="Track ID",
                    path=path,
                    row_number=row_number,
                )
                start = _integer(
                    row.get(columns["startframe"]),
                    field="Start Frame",
                    path=path,
                    row_number=row_number,
                )
                end = _integer(
                    row.get(columns["endframe"]),
                    field="End Frame",
                    path=path,
                    row_number=row_number,
                )
                duration = _integer(
                    row.get(columns["durationframes"]),
                    field="Duration (Frames)",
                    path=path,
                    row_number=row_number,
                )
                if not label or track_id < 0:
                    raise ReviewError(
                        f"{path.name} row {row_number}: label and Track ID are invalid."
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
                identity = (label, track_id, start, end)
                if identity in seen:
                    raise ReviewError(
                        f"{path.name} row {row_number}: duplicate interval {identity}."
                    )
                seen.add(identity)
                predictions.append(
                    PredictionBout(
                        prediction_id=_prediction_id(
                            video_id=video_id,
                            event_kind=event_kind,
                            label=label,
                            class_id=None,
                            track_id=track_id,
                            start_frame=start,
                            end_frame=end,
                        ),
                        video_id=video_id,
                        event_kind=event_kind,
                        label=label,
                        track_id=track_id,
                        start_frame=start,
                        end_frame=end,
                        source_file=source_reference,
                        source_row=row_number,
                    )
                )
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ReviewError(f"Could not read interval table: {path}") from exc
    return predictions


def _load_behavior_bouts(
    *,
    root: Path,
    path: Path,
    video_id: str,
    frame_count: int,
    class_catalog: dict[int, str],
) -> tuple[list[PredictionBout], list[str]]:
    predictions: list[PredictionBout] = []
    warnings: list[str] = []
    source_reference = _reference(path, root)
    reverse_catalog: dict[str, list[int]] = defaultdict(list)
    for class_id, name in class_catalog.items():
        reverse_catalog[name].append(class_id)
    seen: set[tuple[int, str, int, int, int]] = set()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = _header_map(reader.fieldnames)
            required = {
                "trackid",
                "behavior",
                "startframe",
                "endframe",
                "durationframes",
            }
            missing = required - set(columns)
            if missing:
                raise ReviewError(
                    f"{path.name} is missing behavioral-bout columns: "
                    f"{sorted(missing)}"
                )
            for row_number, row in enumerate(reader, start=2):
                label = str(row.get(columns["behavior"], "")).strip()
                track_id = _integer(
                    row.get(columns["trackid"]),
                    field="Track ID",
                    path=path,
                    row_number=row_number,
                )
                start = _integer(
                    row.get(columns["startframe"]),
                    field="Start Frame",
                    path=path,
                    row_number=row_number,
                )
                end = _integer(
                    row.get(columns["endframe"]),
                    field="End Frame",
                    path=path,
                    row_number=row_number,
                )
                duration = _integer(
                    row.get(columns["durationframes"]),
                    field="Duration (Frames)",
                    path=path,
                    row_number=row_number,
                )
                class_id: int | None = None
                if "classid" in columns and str(
                    row.get(columns["classid"], "")
                ).strip():
                    class_id = _integer(
                        row.get(columns["classid"]),
                        field="Class ID",
                        path=path,
                        row_number=row_number,
                    )
                if class_id is None:
                    candidate_ids = reverse_catalog.get(label, [])
                    if len(candidate_ids) > 1:
                        raise ReviewError(
                            f"{path.name} row {row_number}: behavior {label!r} "
                            "maps to more than one Class ID in "
                            "run_manifest.json. Re-run analytics with the "
                            "explicit Class ID column."
                        )
                    class_id = candidate_ids[0] if candidate_ids else None
                if class_id is None:
                    raise ReviewError(
                        f"{path.name} row {row_number}: behavior {label!r} cannot "
                        "be mapped to a Class ID. Re-run analytics with the "
                        "updated detailed-bout schema or restore behavior_names "
                        "in run_manifest.json."
                    )
                canonical = class_catalog.get(class_id)
                if canonical and canonical != label:
                    warnings.append(
                        f"{path.name} row {row_number}: behavior {label!r} did "
                        f"not match Class ID {class_id} ({canonical!r}); the "
                        "manifest class name was used."
                    )
                    label = canonical
                if (
                    not label
                    or class_id < 0
                    or track_id < 0
                    or start < 0
                    or end < start
                    or end >= frame_count
                ):
                    raise ReviewError(
                        f"{path.name} row {row_number}: invalid behavior interval."
                    )
                if duration != end - start + 1:
                    raise ReviewError(
                        f"{path.name} row {row_number}: Duration (Frames)={duration}, "
                        f"but inclusive boundaries imply {end - start + 1}."
                    )
                identity = (class_id, label, track_id, start, end)
                if identity in seen:
                    raise ReviewError(
                        f"{path.name} row {row_number}: duplicate behavior bout "
                        f"{identity}."
                    )
                seen.add(identity)
                predictions.append(
                    PredictionBout(
                        prediction_id=_prediction_id(
                            video_id=video_id,
                            event_kind=BEHAVIOR,
                            label=label,
                            class_id=class_id,
                            track_id=track_id,
                            start_frame=start,
                            end_frame=end,
                        ),
                        video_id=video_id,
                        event_kind=BEHAVIOR,
                        label=label,
                        class_id=class_id,
                        track_id=track_id,
                        start_frame=start,
                        end_frame=end,
                        source_file=source_reference,
                        source_row=row_number,
                    )
                )
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ReviewError(f"Could not read behavioral bout table: {path}") from exc
    return predictions, warnings


def _load_behavior_frames(
    *,
    root: Path,
    path: Path,
    video_id: str,
    frame_count: int,
    class_catalog: dict[int, str],
) -> tuple[list[PredictionBout], list[str]]:
    frames: dict[tuple[int, int, str], set[int]] = defaultdict(set)
    warnings: list[str] = []
    source_reference = _reference(path, root)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = _header_map(reader.fieldnames)
            missing = {"frame", "classid", "trackid"} - set(columns)
            if missing:
                raise ReviewError(
                    f"{path.name} is missing frame-level behavior columns: "
                    f"{sorted(missing)}"
                )
            for row_number, row in enumerate(reader, start=2):
                frame = _integer(
                    row.get(columns["frame"]),
                    field="frame",
                    path=path,
                    row_number=row_number,
                )
                class_id = _integer(
                    row.get(columns["classid"]),
                    field="class_id",
                    path=path,
                    row_number=row_number,
                )
                track_id = _integer(
                    row.get(columns["trackid"]),
                    field="track_id",
                    path=path,
                    row_number=row_number,
                )
                if frame < 0 or frame >= frame_count or class_id < 0 or track_id < 0:
                    raise ReviewError(
                        f"{path.name} row {row_number}: invalid frame/class/track value."
                    )
                label = class_catalog.get(class_id, f"Class_{class_id}")
                if class_id not in class_catalog:
                    warnings.append(
                        f"Class ID {class_id} has no name in run_manifest.json; "
                        f"{label!r} is used."
                    )
                frames[(track_id, class_id, label)].add(frame)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ReviewError(f"Could not read frame-level behavior table: {path}") from exc

    predictions: list[PredictionBout] = []
    for (track_id, class_id, label), values in sorted(frames.items()):
        ordered = sorted(values)
        start = previous = ordered[0]
        intervals: list[tuple[int, int]] = []
        for frame in ordered[1:]:
            if frame != previous + 1:
                intervals.append((start, previous))
                start = frame
            previous = frame
        intervals.append((start, previous))
        for start, end in intervals:
            predictions.append(
                PredictionBout(
                    prediction_id=_prediction_id(
                        video_id=video_id,
                        event_kind=BEHAVIOR,
                        label=label,
                        class_id=class_id,
                        track_id=track_id,
                        start_frame=start,
                        end_frame=end,
                    ),
                    video_id=video_id,
                    event_kind=BEHAVIOR,
                    label=label,
                    class_id=class_id,
                    track_id=track_id,
                    start_frame=start,
                    end_frame=end,
                    source_file=source_reference,
                    source_row=0,
                )
            )
    return predictions, sorted(set(warnings))


def _file_from_mapping(
    root: Path,
    mapping: Any,
    key: str,
    fallback: Path,
) -> Path | None:
    raw = mapping.get(key) if isinstance(mapping, dict) else ""
    return _resolve_file(root, raw, fallbacks=(fallback,))


def load_run_manifest_project(
    manifest_path: str | Path,
    *,
    source_video_root: str | Path | None = None,
) -> ProjectData:
    path = Path(manifest_path).expanduser()
    if path.is_dir():
        path = path / "run_manifest.json"
    try:
        path = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ReviewError(f"run_manifest.json is missing: {path}") from exc
    if not path.is_file() or path.name.casefold() != "run_manifest.json":
        raise ReviewError(f"Expected an IntegraPose run_manifest.json: {path}")

    root = path.parent.resolve()
    manifest = _read_json(path)
    run_id = str(manifest.get("run_id") or root.name).strip()
    video_payload = (
        manifest.get("video") if isinstance(manifest.get("video"), dict) else {}
    )
    base_name = str(video_payload.get("base_name") or root.name).strip() or root.name
    inputs = manifest.get("inputs") if isinstance(manifest.get("inputs"), dict) else {}
    outputs = (
        manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    )
    parameters = (
        manifest.get("parameters")
        if isinstance(manifest.get("parameters"), dict)
        else {}
    )
    provenance = (
        manifest.get("provenance")
        if isinstance(manifest.get("provenance"), dict)
        else {}
    )

    resolved_override: Path | None = None
    if source_video_root is not None:
        try:
            resolved_override = Path(source_video_root).expanduser().resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise ReviewError(
                f"Source-video fallback folder is missing: {source_video_root}"
            ) from exc
        if not resolved_override.is_dir():
            raise ReviewError(
                f"Source-video fallback is not a folder: {resolved_override}"
            )

    display_video, display_role, metadata, warnings = _select_display_video(
        root=root,
        manifest=manifest,
        base_name=base_name,
        source_video_root=resolved_override,
    )
    frame_count, video_fps, width, height = metadata
    configured_fps = parameters.get("fps")
    if isinstance(configured_fps, (int, float)) and float(configured_fps) > 0:
        fps = float(configured_fps)
        if abs(fps - video_fps) > max(0.01, video_fps * 0.001):
            warnings.append(
                "The manifest FPS differs from the selected review video "
                f"({fps:.6g} versus {video_fps:.6g}); manifest FPS is used for "
                "time conversion because it governed analytics."
            )
    else:
        fps = video_fps

    catalog = _class_catalog(manifest)
    predictions: list[PredictionBout] = []
    source_paths: list[Path] = []

    raw_detailed = outputs.get("raw_detailed_bouts_csv")
    behavior_path = _resolve_file(
        root,
        raw_detailed or outputs.get("detailed_bouts_csv"),
        fallbacks=(root / f"{base_name}_detailed_bouts.csv",),
    )
    if behavior_path is not None:
        behavior_predictions, behavior_warnings = _load_behavior_bouts(
            root=root,
            path=behavior_path,
            video_id=run_id,
            frame_count=frame_count,
            class_catalog=catalog,
        )
        predictions.extend(behavior_predictions)
        warnings.extend(behavior_warnings)
        source_paths.append(behavior_path)
        behavior_source_kind = "finalized_detailed_bouts"
    else:
        behavior_frame_path = _resolve_file(
            root,
            outputs.get("batch_results_csv"),
            fallbacks=(root / "batch_results.csv",),
        )
        if behavior_frame_path is not None:
            behavior_predictions, behavior_warnings = _load_behavior_frames(
                root=root,
                path=behavior_frame_path,
                video_id=run_id,
                frame_count=frame_count,
                class_catalog=catalog,
            )
            predictions.extend(behavior_predictions)
            warnings.extend(behavior_warnings)
            source_paths.append(behavior_frame_path)
            behavior_source_kind = "postprocessed_frame_results"
        else:
            warnings.append(
                "Behavioral review is unavailable because neither a detailed "
                "bout CSV nor batch_results.csv was found."
            )
            behavior_source_kind = ""

    raw_roi_files = (
        outputs.get("raw_roi_metrics_files")
        if isinstance(outputs.get("raw_roi_metrics_files"), dict)
        else outputs.get("roi_metrics_files")
        if isinstance(outputs.get("roi_metrics_files"), dict)
        else {}
    )
    concurrent_path = _file_from_mapping(
        root,
        raw_roi_files,
        "dwell_events",
        root / f"{base_name}_roi_dwell_events.csv",
    )
    exclusive_path = _file_from_mapping(
        root,
        raw_roi_files,
        "exclusive_dwell_events",
        root / f"{base_name}_roi_exclusive_dwell_events.csv",
    )
    object_files = (
        outputs.get("raw_object_interaction_files")
        if isinstance(outputs.get("raw_object_interaction_files"), dict)
        else outputs.get("object_interaction_files")
        if isinstance(outputs.get("object_interaction_files"), dict)
        else {}
    )
    object_path = _file_from_mapping(
        root,
        object_files,
        "dwell_events",
        root / f"{base_name}_object_interactions_dwell_events.csv",
    )
    for event_path, event_kind, label_header in (
        (concurrent_path, ROI_CONCURRENT, "ROI Name"),
        (exclusive_path, ROI_EXCLUSIVE, "ROI Name"),
        (object_path, OBJECT_INTERACTION, "Object ROI"),
    ):
        if event_path is None:
            continue
        predictions.extend(
            _load_interval_table(
                root=root,
                path=event_path,
                video_id=run_id,
                frame_count=frame_count,
                event_kind=event_kind,
                label_header=label_header,
            )
        )
        source_paths.append(event_path)

    if not predictions:
        raise ReviewError(
            "No reviewable behavior, ROI, ROI-X, or object-interaction bout "
            f"tables were found beside {path.name}."
        )

    labels: dict[str, list[str]] = {
        ROI_CONCURRENT: sorted(
            str(name) for name in (inputs.get("roi_polygons") or {}) if str(name)
        ),
        ROI_EXCLUSIVE: sorted(
            str(name) for name in (inputs.get("roi_polygons") or {}) if str(name)
        ),
        OBJECT_INTERACTION: sorted(
            str(name)
            for name in (inputs.get("object_roi_polygons") or {})
            if str(name)
        ),
        BEHAVIOR: [catalog[class_id] for class_id in sorted(catalog)],
    }
    for event_kind in labels:
        labels[event_kind] = sorted(
            set(labels[event_kind])
            | {
                prediction.label
                for prediction in predictions
                if prediction.event_kind == event_kind
            }
        )

    unique_sources = sorted(
        set(source_paths), key=lambda item: str(item).casefold()
    )
    hashes = {_reference(source, root): _sha256(source) for source in unique_sources}
    fingerprint_payload = {
        "scheme": FINGERPRINT_SCHEME,
        "run_id": run_id,
        "sources": hashes,
        "predictions": [
            prediction.to_dict()
            for prediction in sorted(
                predictions,
                key=lambda item: (
                    item.event_kind,
                    -1 if item.class_id is None else item.class_id,
                    item.label,
                    item.track_id,
                    item.start_frame,
                    item.end_frame,
                ),
            )
        ],
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    effective_catalog = dict(catalog)
    for prediction in predictions:
        if prediction.class_id is not None:
            effective_catalog.setdefault(prediction.class_id, prediction.label)
    source_video_text = str(inputs.get("video_file") or "")
    path_provenance = {
        "resolution_policy": "run-manifest-relative-first-v1",
        "run_manifest": path.name,
        "resolved_analytics_dir": ".",
        "resolved_display_video": _reference(display_video, root),
        "manifest_source_video": source_video_text,
        "behavior_source_kind": behavior_source_kind,
        "yolo_folder_recorded_for_provenance_only": str(
            inputs.get("yolo_folder") or ""
        ),
    }
    settings = {
        key: parameters.get(key)
        for key in (
            "max_gap_frames",
            "min_bout_frames",
            "configured_min_bout_seconds",
            "configured_max_gap_seconds",
            "temporal_threshold_unit",
            "behavior_bout_class_mode",
        )
        if key in parameters
    }
    video = VideoRecord(
        video_id=run_id,
        video_name=Path(source_video_text).name or f"{base_name}.mp4",
        video_stem=base_name,
        subject_id=str(provenance.get("subject_id") or "").strip(),
        group=str(provenance.get("group") or "").strip(),
        time_point=str(provenance.get("time_point") or "").strip(),
        display_video=display_video,
        display_video_relative=_reference(display_video, root),
        analytics_dir=root,
        run_dir=root,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        source_fingerprint=fingerprint,
        label_catalog=labels,
        track_ids=sorted({prediction.track_id for prediction in predictions}),
        predictions=sorted(
            predictions,
            key=lambda item: (
                item.event_kind,
                -1 if item.class_id is None else item.class_id,
                item.label,
                item.track_id,
                item.start_frame,
                item.end_frame,
            ),
        ),
        source_files=[_reference(source, root) for source in unique_sources],
        behavior_classes=effective_catalog,
        single_animal_mode=bool(parameters.get("single_animal_mode", True)),
        behavior_settings=settings,
        display_video_role=display_role,
        source_file_hashes=hashes,
        path_provenance=path_provenance,
        warnings=warnings,
    )
    return ProjectData(
        root=root,
        session_path=path,
        session_id=run_id,
        project_label=f"{base_name} ({run_id[:8]})",
        videos=[video],
        warnings=[f"{run_id}: {warning}" for warning in warnings],
    )
