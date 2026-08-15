from __future__ import annotations

import copy
import csv
import json
import os
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .models import (
    ACCEPTED,
    ADDED,
    BEHAVIOR,
    MODIFIED,
    OBJECT_INTERACTION,
    ROI_CONCURRENT,
    ROI_EXCLUSIVE,
    PredictionBout,
    ProjectData,
    ReviewBout,
    ReviewError,
)
from .store import ReviewStore


REFERENCE_DECISIONS = {ACCEPTED, MODIFIED, ADDED}
SPATIAL_KINDS = {ROI_CONCURRENT, ROI_EXCLUSIVE, OBJECT_INTERACTION}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _project_reference(root: Path, value: str | Path) -> str:
    """Return a portable relative path when ``value`` is inside the run."""

    text = str(value or "").strip()
    if not text:
        return ""
    try:
        resolved_root = root.resolve()
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = resolved_root / candidate
        resolved = candidate.resolve(strict=False)
        common = os.path.commonpath((str(resolved_root), str(resolved)))
        if os.path.normcase(common) == os.path.normcase(str(resolved_root)):
            return resolved.relative_to(resolved_root).as_posix()
    except (OSError, RuntimeError, ValueError):
        pass
    return text


def _portable_artifact_paths(root: Path, payload: Any) -> Any:
    if isinstance(payload, dict):
        return {
            str(key): _portable_artifact_paths(root, value)
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [_portable_artifact_paths(root, value) for value in payload]
    if isinstance(payload, Path):
        return _project_reference(root, payload)
    if isinstance(payload, str):
        return _project_reference(root, payload)
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewError(f"Could not update run manifest: {path}") from exc
    if not isinstance(value, dict):
        raise ReviewError(f"run_manifest.json is not a JSON object: {path}")
    return value


def _resolve_source(project: ProjectData, reference: str) -> Path | None:
    text = str(reference or "").strip()
    if not text:
        return None
    if text.startswith("external:"):
        candidate = Path(text[len("external:") :])
    else:
        candidate = project.root / text
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError):
        return None
    return resolved if resolved.is_file() else None


def _csv_rows(path: Path) -> tuple[list[str], dict[int, dict[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            return fields, {
                row_number: dict(row)
                for row_number, row in enumerate(reader, start=2)
            }
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ReviewError(f"Could not preserve source columns from {path}") from exc


def _write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fields: Iterable[str],
) -> None:
    field_list = list(dict.fromkeys(str(field) for field in fields))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=field_list,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _reference_bouts(store: ReviewStore, event_kind: str) -> list[ReviewBout]:
    return sorted(
        (
            bout
            for bout in store.list_review_bouts(include_inactive=True)
            if bout.active
            and bout.event_kind == event_kind
            and bout.decision in REFERENCE_DECISIONS
        ),
        key=lambda bout: (
            bout.track_id,
            -1 if bout.class_id is None else bout.class_id,
            bout.label,
            bout.start_frame,
            bout.end_frame,
        ),
    )


def _required_scope_complete(
    store: ReviewStore,
    *,
    video_id: str,
    event_kind: str,
) -> bool:
    predictions = [
        prediction
        for prediction in store.list_predictions(
            video_id=video_id,
            event_kind=event_kind,
        )
    ]
    if not predictions:
        return False
    if event_kind == BEHAVIOR:
        # Include both predicted and manually corrected track IDs. A reviewer
        # who moves a bout to another animal must explicitly complete that
        # animal's scope before the corrected table becomes authoritative.
        tracks = store.behavior_track_ids(video_id)
        return bool(tracks) and all(
            store.scope_complete(video_id, BEHAVIOR, track_id)
            for track_id in tracks
        )
    return store.scope_complete(video_id, event_kind)


def _original_payloads(
    project: ProjectData,
    predictions: Iterable[PredictionBout],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    sources: dict[str, tuple[list[str], dict[int, dict[str, Any]]]] = {}
    fields: list[str] = []
    payloads: dict[str, dict[str, Any]] = {}
    for prediction in predictions:
        if prediction.source_file not in sources:
            source_path = _resolve_source(project, prediction.source_file)
            if source_path is None or prediction.source_row < 2:
                sources[prediction.source_file] = ([], {})
            else:
                sources[prediction.source_file] = _csv_rows(source_path)
        source_fields, rows = sources[prediction.source_file]
        fields.extend(source_fields)
        payloads[prediction.prediction_id] = dict(
            rows.get(prediction.source_row, {})
        )
    return list(dict.fromkeys(fields)), payloads


def _materialize_behavior(
    *,
    project: ProjectData,
    store: ReviewStore,
    output: Path,
) -> dict[str, Any]:
    video = project.videos[0]
    predictions = [
        prediction
        for prediction in store.list_predictions(
            video_id=video.video_id,
            event_kind=BEHAVIOR,
        )
    ]
    prediction_by_id = {
        prediction.prediction_id: prediction for prediction in predictions
    }
    original_fields, original_payloads = _original_payloads(
        project,
        predictions,
    )
    bouts = _reference_bouts(store, BEHAVIOR)
    rows: list[dict[str, Any]] = []
    required_fields = [
        "Run ID",
        "Bout ID",
        "Track ID",
        "Class ID",
        "Behavior",
        "Start Frame",
        "End Frame",
        "Duration (Frames)",
        "Observed Frames",
        "Bridged Frames",
        "Observed Fraction",
        "Maximum Bridged Gap (Frames)",
        "Resolved Class-Conflict Frames",
        "Start Time (s)",
        "End Time (s)",
        "Duration (s)",
        "Interval Semantics",
        "Bout Construction Semantics",
        "Minimum Bout Basis",
        "Detection Max Gap (Frames)",
        "Detection Min Bout (Frames)",
        "Analysis FPS",
        "Review Decision",
        "Review ID",
        "Origin Prediction IDs",
        "Parent Review IDs",
        "Reviewer",
        "Reviewer Notes",
        "Reviewed At (UTC)",
    ]
    for bout in bouts:
        origin = next(
            (
                prediction_by_id[prediction_id]
                for prediction_id in bout.origin_prediction_ids
                if prediction_id in prediction_by_id
            ),
            None,
        )
        payload = dict(
            original_payloads.get(origin.prediction_id, {})
            if origin is not None
            else {}
        )
        duration = bout.frames
        observed_frames = payload.get("Observed Frames", duration)
        try:
            observed_frames = int(float(observed_frames))
        except (TypeError, ValueError):
            observed_frames = duration
        observed_frames = min(duration, max(0, observed_frames))
        payload.update(
            {
                "Run ID": project.session_id,
                "Bout ID": bout.review_id,
                "Track ID": bout.track_id,
                "Class ID": bout.class_id,
                "Behavior": bout.label,
                "Start Frame": bout.start_frame,
                "End Frame": bout.end_frame,
                "Duration (Frames)": duration,
                "Observed Frames": observed_frames,
                "Bridged Frames": max(0, duration - observed_frames),
                "Observed Fraction": (
                    observed_frames / duration if duration else 0.0
                ),
                "Start Time (s)": bout.start_frame / video.fps,
                "End Time (s)": bout.end_frame / video.fps,
                "Duration (s)": duration / video.fps,
                "Interval Semantics": "inclusive_start_and_end_frames",
                "Bout Construction Semantics": (
                    "manual_reference_preserving_integrapose_predictions"
                ),
                "Minimum Bout Basis": payload.get(
                    "Minimum Bout Basis",
                    "manual_review_may_override_detection_minimum",
                ),
                "Detection Max Gap (Frames)": video.behavior_settings.get(
                    "max_gap_frames", ""
                ),
                "Detection Min Bout (Frames)": video.behavior_settings.get(
                    "min_bout_frames", ""
                ),
                "Analysis FPS": video.fps,
                "Review Decision": bout.decision,
                "Review ID": bout.review_id,
                "Origin Prediction IDs": json.dumps(
                    bout.origin_prediction_ids
                ),
                "Parent Review IDs": json.dumps(bout.parent_review_ids),
                "Reviewer": bout.reviewer,
                "Reviewer Notes": bout.note,
                "Reviewed At (UTC)": bout.updated_at,
            }
        )
        rows.append(payload)

    fields = list(dict.fromkeys(original_fields + required_fields))
    authoritative = output / f"{video.video_stem}_reviewed_bouts.csv"
    _write_csv(authoritative, rows, fields)

    summary_groups: dict[
        tuple[str, int, int | None, str, str],
        list[float],
    ] = defaultdict(list)
    for row in rows:
        roi_name = str(row.get("ROI Name") or "")
        summary_groups[
            (
                str(row.get("Run ID") or ""),
                int(row["Track ID"]),
                (
                    int(row["Class ID"])
                    if row.get("Class ID") not in (None, "")
                    else None
                ),
                str(row["Behavior"]),
                roi_name,
            )
        ].append(float(row["Duration (s)"]))
    summary_rows = []
    for (run_id, track_id, class_id, behavior, roi_name), durations in sorted(
        summary_groups.items(),
        key=lambda item: (
            item[0][1],
            -1 if item[0][2] is None else item[0][2],
            item[0][3],
            item[0][4],
        ),
    ):
        row = {
            "Run ID": run_id,
            "Track ID": track_id,
            "Class ID": class_id,
            "Behavior": behavior,
            "Bout_Count": len(durations),
            "Total_Duration_s": sum(durations),
            "Mean_Duration_s": sum(durations) / len(durations),
            "Analysis FPS": video.fps,
            "Review Semantics": "authoritative_manual_reference",
        }
        if roi_name:
            row["ROI Name"] = roi_name
        summary_rows.append(row)
    summary_fields = [
        "Run ID",
        "Track ID",
        "Class ID",
        "Behavior",
        "ROI Name",
        "Bout_Count",
        "Total_Duration_s",
        "Mean_Duration_s",
        "Analysis FPS",
        "Review Semantics",
    ]
    summary = output / f"{video.video_stem}_reviewed_bouts_summary.csv"
    _write_csv(summary, summary_rows, summary_fields)
    return {
        "authoritative_path": str(authoritative),
        "summary_path": str(summary),
        "accepted_bout_count": len(bouts),
        "rejected_bout_count": sum(
            1
            for bout in store.list_review_bouts(include_inactive=True)
            if bout.event_kind == BEHAVIOR
            and not bout.active
            and bout.decision == "rejected"
        ),
    }


def _event_review_frame(bouts: Iterable[ReviewBout]):
    import pandas as pd

    rows: list[dict[str, Any]] = []
    event_id = 0
    for bout in bouts:
        for event_type, frame in (
            ("entry", bout.start_frame),
            ("exit", bout.end_frame),
        ):
            event_id += 1
            rows.append(
                {
                    "Event ID": event_id,
                    "Source": "zone",
                    "Event Type": event_type,
                    "Target Name": bout.label,
                    "Track ID": bout.track_id,
                    "Frame": frame,
                    "Review Status": "confirmed",
                    "Reviewer Notes": bout.note,
                    "Corrected Manually": bout.decision != ACCEPTED,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "Event ID",
            "Source",
            "Event Type",
            "Target Name",
            "Track ID",
            "Frame",
            "Review Status",
            "Reviewer Notes",
            "Corrected Manually",
        ],
    )


def _null_event_review_frame(predictions: Iterable[PredictionBout]):
    import pandas as pd

    rows: list[dict[str, Any]] = []
    event_id = 0
    for prediction in predictions:
        for event_type, frame in (
            ("entry", prediction.start_frame),
            ("exit", prediction.end_frame),
        ):
            event_id += 1
            rows.append(
                {
                    "Event ID": event_id,
                    "Source": "zone",
                    "Event Type": event_type,
                    "Corrected Event Type": "null",
                    "Target Name": prediction.label,
                    "Track ID": prediction.track_id,
                    "Frame": frame,
                    "Review Status": "confirmed",
                    "Reviewer Notes": "Rejected during bout review.",
                    "Corrected Manually": True,
                }
            )
    return pd.DataFrame(rows)


def _materialize_roi(
    *,
    project: ProjectData,
    store: ReviewStore,
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from integra_pose.utils.roi_event_review import save_reviewed_roi_bundle

    video = project.videos[0]
    concurrent = _reference_bouts(store, ROI_CONCURRENT)
    exclusive = _reference_bouts(store, ROI_EXCLUSIVE)
    concurrent_review = (
        _event_review_frame(concurrent)
        if concurrent
        else _null_event_review_frame(
            store.list_predictions(
                video_id=video.video_id,
                event_kind=ROI_CONCURRENT,
            )
        )
    )
    exclusive_review = (
        _event_review_frame(exclusive)
        if exclusive
        else _null_event_review_frame(
            store.list_predictions(
                video_id=video.video_id,
                event_kind=ROI_EXCLUSIVE,
            )
        )
    )
    concurrent_artifacts = save_reviewed_roi_bundle(
        concurrent_review,
        output_dir=output / "Concurrent",
        video_name=video.video_stem,
        fps=video.fps,
        min_dwell_frames=1,
        max_gap_frames=0,
    )
    exclusive_artifacts = save_reviewed_roi_bundle(
        exclusive_review,
        output_dir=output / "Exclusive",
        video_name=f"{video.video_stem}_exclusive",
        fps=video.fps,
        min_dwell_frames=1,
        max_gap_frames=0,
    )
    return concurrent_artifacts, exclusive_artifacts


def _materialize_object(
    *,
    project: ProjectData,
    store: ReviewStore,
    output: Path,
) -> dict[str, str]:
    video = project.videos[0]
    bouts = _reference_bouts(store, OBJECT_INTERACTION)
    dwell_rows: list[dict[str, Any]] = []
    marker_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[int, str], list[ReviewBout]] = defaultdict(list)
    for bout in bouts:
        grouped[(bout.track_id, bout.label)].append(bout)
        dwell_rows.append(
            {
                "Track ID": bout.track_id,
                "Object ROI": bout.label,
                "Start Frame": bout.start_frame,
                "End Frame": bout.end_frame,
                "Duration (Frames)": bout.frames,
                "Duration (s)": bout.frames / video.fps,
                "Review Decision": bout.decision,
                "Review ID": bout.review_id,
                "Origin Prediction IDs": json.dumps(
                    bout.origin_prediction_ids
                ),
                "Reviewer Notes": bout.note,
            }
        )
        marker_rows.extend(
            [
                {
                    "Source": "object",
                    "Event Type": "entry",
                    "Target Name": bout.label,
                    "Track ID": bout.track_id,
                    "Frame": bout.start_frame,
                    "Review ID": bout.review_id,
                },
                {
                    "Source": "object",
                    "Event Type": "exit",
                    "Target Name": bout.label,
                    "Track ID": bout.track_id,
                    "Frame": bout.end_frame,
                    "Review ID": bout.review_id,
                },
            ]
        )
    per_track_rows: list[dict[str, Any]] = []
    summary_accumulator: dict[str, list[ReviewBout]] = defaultdict(list)
    for (track_id, label), group in sorted(grouped.items()):
        summary_accumulator[label].extend(group)
        frames = sum(bout.frames for bout in group)
        per_track_rows.append(
            {
                "Track ID": track_id,
                "Object ROI": label,
                "Entries": len(group),
                "Exits": len(group),
                "Dwell Events": len(group),
                "Qualified Interaction Frames": frames,
                "Qualified Interaction Time (s)": frames / video.fps,
                "Review Semantics": "authoritative_manual_reference",
            }
        )
    summary_rows = []
    for label, group in sorted(summary_accumulator.items()):
        frames = sum(bout.frames for bout in group)
        summary_rows.append(
            {
                "Object ROI": label,
                "Entries": len(group),
                "Exits": len(group),
                "Dwell Events": len(group),
                "Qualified Interaction Frames": frames,
                "Qualified Interaction Time (s)": frames / video.fps,
                "Review Semantics": "authoritative_manual_reference",
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "dwell_events": output
        / f"{video.video_stem}_reviewed_object_interactions_dwell_events.csv",
        "summary": output
        / f"{video.video_stem}_reviewed_object_interactions_summary.csv",
        "per_track": output
        / f"{video.video_stem}_reviewed_object_interactions_per_track.csv",
        "events": output
        / f"{video.video_stem}_reviewed_object_events.csv",
    }
    _write_csv(
        paths["dwell_events"],
        dwell_rows,
        [
            "Track ID",
            "Object ROI",
            "Start Frame",
            "End Frame",
            "Duration (Frames)",
            "Duration (s)",
            "Review Decision",
            "Review ID",
            "Origin Prediction IDs",
            "Reviewer Notes",
        ],
    )
    summary_fields = [
        "Object ROI",
        "Entries",
        "Exits",
        "Dwell Events",
        "Qualified Interaction Frames",
        "Qualified Interaction Time (s)",
        "Review Semantics",
    ]
    _write_csv(paths["summary"], summary_rows, summary_fields)
    _write_csv(
        paths["per_track"],
        per_track_rows,
        ["Track ID", *summary_fields],
    )
    _write_csv(
        paths["events"],
        marker_rows,
        [
            "Source",
            "Event Type",
            "Target Name",
            "Track ID",
            "Frame",
            "Review ID",
        ],
    )
    return {key: str(path) for key, path in paths.items()}


def _register_behavior(
    manifest: dict[str, Any],
    artifacts: dict[str, Any],
    *,
    project: ProjectData,
    store: ReviewStore,
) -> dict[str, Any]:
    from integra_pose.utils.bout_review import (
        register_authoritative_review_in_manifest,
    )

    export_root = Path(artifacts["authoritative_path"]).parents[2]
    source_export = (
        export_root / "Behavior_Bouts" / "Tables" / "original_predictions.csv"
    )
    decisions_export = (
        export_root / "Behavior_Bouts" / "Tables" / "review_decisions.csv"
    )
    artifacts = {
        **artifacts,
        "raw_detected_path": str(source_export),
        "decisions_path": str(decisions_export),
        "workspace_path": str(store.database),
    }
    return register_authoritative_review_in_manifest(manifest, artifacts)


def _register_object(
    manifest: dict[str, Any],
    paths: dict[str, str],
) -> dict[str, Any]:
    updated = copy.deepcopy(manifest)
    outputs = updated.setdefault("outputs", {})
    if not isinstance(outputs, dict):
        outputs = {}
        updated["outputs"] = outputs
    raw_files = (
        copy.deepcopy(outputs.get("raw_object_interaction_files"))
        if isinstance(outputs.get("raw_object_interaction_files"), dict)
        else copy.deepcopy(outputs.get("object_interaction_files"))
        if isinstance(outputs.get("object_interaction_files"), dict)
        else {}
    )
    if raw_files:
        outputs["raw_object_interaction_files"] = raw_files
    if outputs.get("object_events_csv"):
        outputs.setdefault("raw_object_events_csv", outputs["object_events_csv"])
    outputs["reviewed_object_events_csv"] = paths["events"]
    outputs["object_events_csv"] = paths["events"]
    outputs["object_interaction_files"] = {
        **raw_files,
        "summary": paths["summary"],
        "per_track": paths["per_track"],
        "dwell_events": paths["dwell_events"],
    }
    modules = (
        outputs.get("modules") if isinstance(outputs.get("modules"), dict) else {}
    )
    invalidated = (
        copy.deepcopy(outputs.get("invalidated_raw_object_modules"))
        if isinstance(outputs.get("invalidated_raw_object_modules"), dict)
        else {}
    )
    for key in (
        "object_interactions",
        "object_transition_analysis",
        "event_aligned_windows",
        "normalization_summary",
    ):
        if key in modules:
            invalidated[key] = modules.pop(key)
    outputs["modules"] = modules
    if invalidated:
        outputs["invalidated_raw_object_modules"] = invalidated
    notes = updated.setdefault("notes", {})
    if not isinstance(notes, dict):
        notes = {}
        updated["notes"] = notes
    notes["object_interaction_review"] = {
        "status": "complete",
        "authoritative_basis": "reviewed_object_interaction_bouts",
        "invalidated_raw_object_modules": sorted(invalidated),
    }
    return updated


def materialize_integrapose_review(
    project: ProjectData,
    store: ReviewStore,
    export_path: Path,
) -> dict[str, Any]:
    """Materialize completed review scopes and atomically update the run manifest."""

    manifest_path = project.session_path
    if manifest_path.name.casefold() != "run_manifest.json":
        return {
            "manifest_updated": False,
            "reason": "portable_batch_project",
        }
    if len(project.videos) != 1:
        raise ReviewError(
            "One IntegraPose run manifest must resolve to exactly one video."
        )

    video = project.videos[0]
    manifest = _read_manifest(manifest_path)
    status: dict[str, Any] = {
        "schema_version": 1,
        "updated_at_utc": _utc_now(),
        "run_manifest": str(manifest_path),
        "review_database": str(store.database),
        "latest_export": str(export_path),
        "behavior": "not_applicable",
        "roi": "not_applicable",
        "object_interaction": "not_applicable",
        "spatial": "not_applicable",
    }
    kinds = {
        prediction.event_kind
        for prediction in store.list_predictions(video_id=video.video_id)
    }

    materialized_root = export_path / "IntegraPose_Authoritative"
    if BEHAVIOR in kinds:
        if _required_scope_complete(
            store,
            video_id=video.video_id,
            event_kind=BEHAVIOR,
        ):
            behavior_artifacts = _materialize_behavior(
                project=project,
                store=store,
                output=materialized_root / "Behavior_Bouts",
            )
            manifest = _register_behavior(
                manifest,
                behavior_artifacts,
                project=project,
                store=store,
            )
            status["behavior"] = "complete"
            status["behavior_artifacts"] = behavior_artifacts
        else:
            status["behavior"] = "provisional"

    roi_kinds = kinds & {ROI_CONCURRENT, ROI_EXCLUSIVE}
    if roi_kinds:
        roi_complete = roi_kinds == {ROI_CONCURRENT, ROI_EXCLUSIVE} and all(
            _required_scope_complete(
                store,
                video_id=video.video_id,
                event_kind=kind,
            )
            for kind in roi_kinds
        )
        if roi_complete:
            concurrent, exclusive = _materialize_roi(
                project=project,
                store=store,
                output=materialized_root / "ROI_Bouts",
            )
            from integra_pose.utils.roi_event_review import (
                register_authoritative_roi_review_in_manifest,
            )

            manifest = register_authoritative_roi_review_in_manifest(
                manifest,
                concurrent,
                review_workspace_path=str(store.database),
                exclusive_artifacts=exclusive,
            )
            status["roi"] = "complete"
            status["roi_concurrent_artifacts"] = concurrent
            status["roi_exclusive_artifacts"] = exclusive
        else:
            status["roi"] = "provisional"

    if OBJECT_INTERACTION in kinds:
        if _required_scope_complete(
            store,
            video_id=video.video_id,
            event_kind=OBJECT_INTERACTION,
        ):
            object_paths = _materialize_object(
                project=project,
                store=store,
                output=materialized_root / "Object_Interactions",
            )
            manifest = _register_object(manifest, object_paths)
            status["object_interaction"] = "complete"
            status["object_interaction_artifacts"] = object_paths
        else:
            status["object_interaction"] = "provisional"

    spatial_components = [
        str(status[key])
        for key in ("roi", "object_interaction")
        if status.get(key) != "not_applicable"
    ]
    if spatial_components:
        status["spatial"] = (
            "complete"
            if all(value == "complete" for value in spatial_components)
            else "provisional"
        )

    outputs = manifest.setdefault("outputs", {})
    if not isinstance(outputs, dict):
        outputs = {}
        manifest["outputs"] = outputs
    status_path = project.root / "bout_review_workspace" / "last_review_status.json"
    outputs["bout_review_database"] = _project_reference(
        project.root,
        str(store.database),
    )
    outputs["bout_review_latest_export"] = _project_reference(
        project.root,
        export_path,
    )
    outputs["bout_review_status_json"] = _project_reference(
        project.root,
        status_path,
    )
    outputs["bout_review_portable_paths"] = {
        "path_policy": "run_manifest_parent_relative_v1",
        "behavior_artifacts": _portable_artifact_paths(
            project.root,
            status.get("behavior_artifacts", {}),
        ),
        "roi_concurrent_artifacts": _portable_artifact_paths(
            project.root,
            status.get("roi_concurrent_artifacts", {}),
        ),
        "roi_exclusive_artifacts": _portable_artifact_paths(
            project.root,
            status.get("roi_exclusive_artifacts", {}),
        ),
        "object_interaction_artifacts": _portable_artifact_paths(
            project.root,
            status.get("object_interaction_artifacts", {}),
        ),
    }
    notes = manifest.setdefault("notes", {})
    if not isinstance(notes, dict):
        notes = {}
        manifest["notes"] = notes
    notes["integrated_bout_reviewer"] = {
        key: value
        for key, value in status.items()
        if key
        in {
            "schema_version",
            "updated_at_utc",
            "review_database",
            "latest_export",
            "behavior",
            "roi",
            "object_interaction",
            "spatial",
        }
    }
    notes["integrated_bout_reviewer"]["path_policy"] = (
        "Active legacy-compatible output paths retain their recorded values; "
        "bout_review_portable_paths stores run-relative mirrors for relocation."
    )

    # Activate the manifest first. If the secondary status write fails, the
    # authoritative data paths are still correct and can be rediscovered from
    # the manifest; the inverse ordering could falsely report completion while
    # leaving downstream IntegraPose consumers on the old outputs.
    _atomic_json(manifest_path, manifest)
    _atomic_json(status_path, status)
    return {
        **status,
        "manifest_updated": True,
        "status_path": str(status_path),
    }
