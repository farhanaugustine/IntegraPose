"""Validation and authoritative materialization for reviewed ROI events."""

from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


TERMINAL_ROI_REVIEW_STATUSES = frozenset({"confirmed", "corrected"})
ROI_DEPENDENT_MODULE_KEYS = frozenset(
    {
        "preference_indices",
        "latency_metrics",
        "visit_structure",
        "normalization_summary",
        "roi_time_heatmap",
        "roi_context_windows",
        "behavior_transitions",
        "temporal_trends",
        "activity_budgets",
        "event_aligned_windows",
    }
)


class ROIReviewValidationError(ValueError):
    """Raised when reviewed ROI events cannot form valid visit intervals."""


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _integer(value: Any, *, field: str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ROIReviewValidationError(f"{field} must be an integer, got {value!r}.") from exc
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ROIReviewValidationError(f"{field} must be an integer, got {value!r}.")
    return int(numeric)


def materialize_reviewed_roi_events(
    review_df: pd.DataFrame,
    *,
    fps: float,
    min_dwell_frames: int,
    max_gap_frames: int = 0,
) -> dict[str, Any]:
    """Validate corrected event rows, pair visits, and apply minimum dwell."""

    if fps is None or float(fps) <= 0:
        raise ROIReviewValidationError("A positive FPS is required to finalize ROI events.")
    fps = float(fps)
    min_dwell_frames = max(1, int(min_dwell_frames or 1))
    max_gap_frames = max(0, int(max_gap_frames or 0))
    if review_df is None or review_df.empty:
        raise ROIReviewValidationError("The ROI review table is empty.")

    required = {
        "Event ID",
        "Event Type",
        "Target Name",
        "Track ID",
        "Frame",
        "Review Status",
    }
    missing = sorted(required.difference(review_df.columns))
    if missing:
        raise ROIReviewValidationError(
            "ROI review is missing required columns: " + ", ".join(missing)
        )

    event_ids = review_df["Event ID"].map(
        lambda value: _integer(value, field="Event ID")
    )
    if event_ids.duplicated().any():
        duplicates = sorted(set(event_ids[event_ids.duplicated(keep=False)].tolist()))
        raise ROIReviewValidationError(
            f"ROI review contains duplicate Event ID values: {duplicates}."
        )

    statuses = review_df["Review Status"].map(lambda value: _text(value).lower() or "detected")
    incomplete_mask = ~statuses.isin(TERMINAL_ROI_REVIEW_STATUSES)
    if bool(incomplete_mask.any()):
        unresolved = int(incomplete_mask.sum())
        raise ROIReviewValidationError(
            f"{unresolved} ROI event(s) are still unreviewed; confirm or correct every event before finalizing."
        )

    effective_rows: list[dict[str, Any]] = []
    null_count = 0
    for position, (_, row) in enumerate(review_df.iterrows(), start=1):
        event_id = _integer(row.get("Event ID"), field=f"Event ID at row {position}")
        status = _text(row.get("Review Status")).lower()
        event_type = (
            _text(row.get("Corrected Event Type")).lower()
            or _text(row.get("Event Type")).lower()
        )
        if event_type == "null":
            null_count += 1
            continue
        if event_type not in {"entry", "exit"}:
            raise ROIReviewValidationError(
                f"Event {event_id} has invalid corrected event type {event_type!r}."
            )
        target = _text(row.get("Corrected Target Name")) or _text(row.get("Target Name"))
        if not target:
            raise ROIReviewValidationError(f"Event {event_id} has no corrected ROI name.")
        track_id = _integer(
            row.get("Corrected Track ID")
            if _text(row.get("Corrected Track ID"))
            else row.get("Track ID"),
            field=f"Track ID for event {event_id}",
        )
        frame = _integer(
            row.get("Corrected Frame")
            if _text(row.get("Corrected Frame"))
            else row.get("Frame"),
            field=f"Frame for event {event_id}",
        )
        if frame < 0:
            raise ROIReviewValidationError(
                f"Event {event_id} has a negative corrected frame ({frame})."
            )
        effective_rows.append(
            {
                "Event ID": event_id,
                "Source": _text(row.get("Source")) or "zone",
                "Event Type": event_type,
                "ROI Name": target,
                "Track ID": track_id,
                "Frame": frame,
                "Review Status": status,
                "Reviewer Notes": _text(row.get("Reviewer Notes")),
                "Corrected Manually": bool(row.get("Corrected Manually", False)),
                "Interval Semantics": "entry_and_exit_frames_are_inclusive",
            }
        )

    if not effective_rows:
        empty_events = pd.DataFrame(
            columns=[
                "Event ID",
                "Source",
                "Event Type",
                "ROI Name",
                "Track ID",
                "Frame",
                "Review Status",
                "Reviewer Notes",
                "Corrected Manually",
                "Visit ID",
                "Meets Minimum Dwell",
                "Interval Semantics",
            ]
        )
        empty_visits = pd.DataFrame(
            columns=[
                "Visit ID",
                "Track ID",
                "ROI Name",
                "Entry Event ID",
                "Exit Event ID",
                "Start Frame",
                "End Frame",
                "Duration (Frames)",
                "Duration (s)",
                "Meets Minimum Dwell",
                "Qualification Status",
                "Minimum Dwell (Frames)",
                "Maximum Gap (Frames)",
                "Interval Semantics",
            ]
        )
        return {
            "events": empty_events,
            "all_visits": empty_visits,
            "qualified_visits": empty_visits.copy(),
            "overview": pd.DataFrame(),
            "per_track": pd.DataFrame(),
            "transitions": pd.DataFrame(
                columns=["From ROI", "To ROI", "Transition Count"]
            ),
            "validation": {
                "reviewed_event_count": int(len(review_df)),
                "null_event_count": null_count,
                "effective_event_count": 0,
                "visit_count": 0,
                "qualified_visit_count": 0,
                "below_minimum_visit_count": 0,
                "status": "valid_empty",
            },
        }

    events_df = pd.DataFrame(effective_rows)
    visit_rows: list[dict[str, Any]] = []
    event_to_visit: dict[int, tuple[str, bool]] = {}
    visit_sequence = 0

    for (track_id, roi_name), group in events_df.groupby(
        ["Track ID", "ROI Name"], sort=True
    ):
        active_entry: pd.Series | None = None
        grouped_frames = group.groupby("Frame", sort=True)
        for frame, frame_group in grouped_frames:
            frame_group = frame_group.sort_values("Event ID", kind="stable")
            entries = frame_group[frame_group["Event Type"] == "entry"]
            exits = frame_group[frame_group["Event Type"] == "exit"]
            if len(entries) > 1 or len(exits) > 1:
                raise ROIReviewValidationError(
                    f"Track {track_id}, ROI {roi_name!r}, frame {frame} contains duplicate "
                    "entry or exit events."
                )

            ordered_rows: list[pd.Series] = []
            if active_entry is not None:
                ordered_rows.extend(row for _, row in exits.iterrows())
                ordered_rows.extend(row for _, row in entries.iterrows())
            else:
                ordered_rows.extend(row for _, row in entries.iterrows())
                ordered_rows.extend(row for _, row in exits.iterrows())

            for event in ordered_rows:
                event_id = int(event["Event ID"])
                if event["Event Type"] == "entry":
                    if active_entry is not None:
                        raise ROIReviewValidationError(
                            f"Track {track_id}, ROI {roi_name!r} has entry event {event_id} "
                            f"before entry event {int(active_entry['Event ID'])} was closed."
                        )
                    active_entry = event
                    continue
                if active_entry is None:
                    raise ROIReviewValidationError(
                        f"Track {track_id}, ROI {roi_name!r} has exit event {event_id} "
                        "without a preceding entry."
                    )
                start_frame = int(active_entry["Frame"])
                end_frame = int(event["Frame"])
                if end_frame < start_frame:
                    raise ROIReviewValidationError(
                        f"Track {track_id}, ROI {roi_name!r} exits before it enters "
                        f"({end_frame} < {start_frame})."
                    )
                visit_sequence += 1
                visit_id = f"roi-visit-{visit_sequence:08d}"
                duration_frames = int(end_frame - start_frame + 1)
                qualifies = duration_frames >= min_dwell_frames
                visit_rows.append(
                    {
                        "Visit ID": visit_id,
                        "Track ID": int(track_id),
                        "ROI Name": str(roi_name),
                        "Entry Event ID": int(active_entry["Event ID"]),
                        "Exit Event ID": event_id,
                        "Start Frame": start_frame,
                        "End Frame": end_frame,
                        "Duration (Frames)": duration_frames,
                        "Duration (s)": duration_frames / fps,
                        "Meets Minimum Dwell": bool(qualifies),
                        "Qualification Status": (
                            "qualified" if qualifies else "below_minimum_dwell"
                        ),
                        "Minimum Dwell (Frames)": min_dwell_frames,
                        "Maximum Gap (Frames)": max_gap_frames,
                        "Interval Semantics": "inclusive_start_and_end_frames",
                    }
                )
                event_to_visit[int(active_entry["Event ID"])] = (visit_id, qualifies)
                event_to_visit[event_id] = (visit_id, qualifies)
                active_entry = None

        if active_entry is not None:
            raise ROIReviewValidationError(
                f"Track {track_id}, ROI {roi_name!r} has unclosed entry event "
                f"{int(active_entry['Event ID'])}."
            )

    all_visits_df = pd.DataFrame(visit_rows)
    qualified_visits_df = all_visits_df[
        all_visits_df["Meets Minimum Dwell"].astype(bool)
    ].copy()

    events_df["Visit ID"] = events_df["Event ID"].map(
        lambda event_id: event_to_visit.get(int(event_id), ("", False))[0]
    )
    events_df["Meets Minimum Dwell"] = events_df["Event ID"].map(
        lambda event_id: event_to_visit.get(int(event_id), ("", False))[1]
    )
    events_df.sort_values(
        ["Frame", "Track ID", "ROI Name", "Event Type"],
        inplace=True,
        kind="stable",
    )
    events_df.reset_index(drop=True, inplace=True)

    overview_rows: list[dict[str, Any]] = []
    per_track_rows: list[dict[str, Any]] = []
    for roi_name in sorted(events_df["ROI Name"].astype(str).unique()):
        all_target = all_visits_df[all_visits_df["ROI Name"] == roi_name]
        qualified_target = qualified_visits_df[
            qualified_visits_df["ROI Name"] == roi_name
        ]
        durations = pd.to_numeric(
            qualified_target.get("Duration (Frames)", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        qualified_frames = int(durations.sum()) if not durations.empty else 0
        overview_rows.append(
            {
                "ROI Name": roi_name,
                "Entries": int(len(qualified_target)),
                "Exits": int(len(qualified_target)),
                "Dwell Events": int(len(qualified_target)),
                "Reviewed Visit Pairs": int(len(all_target)),
                "Below Minimum Dwell Visits": int(
                    (~all_target["Meets Minimum Dwell"].astype(bool)).sum()
                ),
                "Mean Dwell Duration (frames)": (
                    float(durations.mean()) if not durations.empty else np.nan
                ),
                "Mean Dwell Duration (s)": (
                    float(durations.mean() / fps) if not durations.empty else np.nan
                ),
                "Median Dwell Duration (frames)": (
                    float(durations.median()) if not durations.empty else np.nan
                ),
                "Median Dwell Duration (s)": (
                    float(durations.median() / fps) if not durations.empty else np.nan
                ),
                "Total Dwell Frames": qualified_frames,
                "Total Dwell Time (s)": qualified_frames / fps,
                "Qualified Dwell Frames": qualified_frames,
                "Qualified Dwell Time (s)": qualified_frames / fps,
                "Minimum Dwell (Frames)": min_dwell_frames,
                "Maximum Gap (Frames)": max_gap_frames,
                "Review Semantics": "authoritative_corrected_events",
            }
        )

    for (track_id, roi_name), all_target in all_visits_df.groupby(
        ["Track ID", "ROI Name"], sort=True
    ):
        qualified_target = all_target[
            all_target["Meets Minimum Dwell"].astype(bool)
        ]
        durations = pd.to_numeric(
            qualified_target.get("Duration (Frames)", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        qualified_frames = int(durations.sum()) if not durations.empty else 0
        per_track_rows.append(
            {
                "Track ID": int(track_id),
                "ROI Name": str(roi_name),
                "Entries": int(len(qualified_target)),
                "Exits": int(len(qualified_target)),
                "Dwell Events": int(len(qualified_target)),
                "Reviewed Visit Pairs": int(len(all_target)),
                "Below Minimum Dwell Visits": int(
                    (~all_target["Meets Minimum Dwell"].astype(bool)).sum()
                ),
                "Mean Dwell Duration (frames)": (
                    float(durations.mean()) if not durations.empty else np.nan
                ),
                "Mean Dwell Duration (s)": (
                    float(durations.mean() / fps) if not durations.empty else np.nan
                ),
                "Median Dwell Duration (frames)": (
                    float(durations.median()) if not durations.empty else np.nan
                ),
                "Median Dwell Duration (s)": (
                    float(durations.median() / fps) if not durations.empty else np.nan
                ),
                "Total Dwell Frames": qualified_frames,
                "Total Dwell Time (s)": qualified_frames / fps,
                "Qualified Dwell Frames": qualified_frames,
                "Qualified Dwell Time (s)": qualified_frames / fps,
                "Minimum Dwell (Frames)": min_dwell_frames,
                "Maximum Gap (Frames)": max_gap_frames,
                "Review Semantics": "authoritative_corrected_events",
            }
        )

    transitions = Counter()
    for _track_id, track_visits in qualified_visits_df.groupby(
        "Track ID", sort=True
    ):
        ordered = track_visits.sort_values(
            ["Start Frame", "End Frame"], kind="stable"
        )
        names = ordered["ROI Name"].astype(str).tolist()
        for source, target in zip(names, names[1:]):
            if source and target and source != target:
                transitions[(source, target)] += 1
    transitions_df = pd.DataFrame(
        [
            {
                "From ROI": source,
                "To ROI": target,
                "Transition Count": count,
            }
            for (source, target), count in sorted(transitions.items())
        ],
        columns=["From ROI", "To ROI", "Transition Count"],
    )

    return {
        "events": events_df,
        "all_visits": all_visits_df,
        "qualified_visits": qualified_visits_df.reset_index(drop=True),
        "overview": pd.DataFrame(overview_rows),
        "per_track": pd.DataFrame(per_track_rows),
        "transitions": transitions_df,
        "validation": {
            "reviewed_event_count": int(len(review_df)),
            "null_event_count": null_count,
            "effective_event_count": int(len(events_df)),
            "visit_count": int(len(all_visits_df)),
            "qualified_visit_count": int(len(qualified_visits_df)),
            "below_minimum_visit_count": int(
                (~all_visits_df["Meets Minimum Dwell"].astype(bool)).sum()
            ),
            "status": "valid",
        },
    }


def _merge_raw_occupancy(
    reviewed: pd.DataFrame,
    raw: pd.DataFrame | None,
    *,
    keys: list[str],
) -> pd.DataFrame:
    out = reviewed.copy()
    if raw is None or raw.empty:
        out["Raw Occupancy Frames"] = np.nan
        out["Raw Occupancy Time (s)"] = np.nan
        return out
    raw_work = raw.copy()
    raw_frames_column = next(
        (
            column
            for column in ("Raw Occupancy Frames", "Frames in ROI")
            if column in raw_work.columns
        ),
        None,
    )
    raw_time_column = next(
        (
            column
            for column in ("Raw Occupancy Time (s)", "Time in ROI (s)")
            if column in raw_work.columns
        ),
        None,
    )
    keep = list(keys)
    rename: dict[str, str] = {}
    if raw_frames_column:
        keep.append(raw_frames_column)
        rename[raw_frames_column] = "Raw Occupancy Frames"
    if raw_time_column:
        keep.append(raw_time_column)
        rename[raw_time_column] = "Raw Occupancy Time (s)"
    if len(keep) == len(keys):
        out["Raw Occupancy Frames"] = np.nan
        out["Raw Occupancy Time (s)"] = np.nan
        return out
    raw_work = raw_work[keep].rename(columns=rename).drop_duplicates(keys)
    if out.empty:
        out = raw_work[keys].drop_duplicates().copy()
        for column in (
            "Entries",
            "Exits",
            "Dwell Events",
            "Reviewed Visit Pairs",
            "Below Minimum Dwell Visits",
            "Total Dwell Frames",
            "Total Dwell Time (s)",
            "Qualified Dwell Frames",
            "Qualified Dwell Time (s)",
        ):
            out[column] = 0
        for column in (
            "Mean Dwell Duration (frames)",
            "Mean Dwell Duration (s)",
            "Median Dwell Duration (frames)",
            "Median Dwell Duration (s)",
        ):
            out[column] = np.nan
        out["Review Semantics"] = "authoritative_corrected_events"
    return out.merge(raw_work, on=keys, how="left")


def save_reviewed_roi_bundle(
    review_df: pd.DataFrame,
    *,
    output_dir: str | Path,
    video_name: str,
    fps: float,
    min_dwell_frames: int,
    max_gap_frames: int,
    raw_overview_df: pd.DataFrame | None = None,
    raw_per_track_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Materialize reviewed ROI files and return paths plus validation details."""

    result = materialize_reviewed_roi_events(
        review_df,
        fps=fps,
        min_dwell_frames=min_dwell_frames,
        max_gap_frames=max_gap_frames,
    )
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stem = str(video_name or "video").strip() or "video"

    overview = _merge_raw_occupancy(
        result["overview"],
        raw_overview_df,
        keys=["ROI Name"],
    )
    per_track = _merge_raw_occupancy(
        result["per_track"],
        raw_per_track_df,
        keys=["Track ID", "ROI Name"],
    )
    for table in (overview, per_track):
        table["Minimum Dwell (Frames)"] = int(min_dwell_frames)
        table["Maximum Gap (Frames)"] = int(max_gap_frames)
    paths = {
        "events": root / f"{stem}_reviewed_roi_events.csv",
        "all_visits": root / f"{stem}_reviewed_roi_all_visits.csv",
        "dwell_events": root / f"{stem}_reviewed_roi_dwell_events.csv",
        "overview": root / f"{stem}_reviewed_roi_overview.csv",
        "per_track": root / f"{stem}_reviewed_roi_per_track.csv",
        "transitions": root / f"{stem}_reviewed_roi_transitions.csv",
        "validation": root / f"{stem}_reviewed_roi_validation.json",
    }
    result["events"].to_csv(paths["events"], index=False, encoding="utf-8")
    result["all_visits"].to_csv(paths["all_visits"], index=False, encoding="utf-8")
    result["qualified_visits"].to_csv(
        paths["dwell_events"], index=False, encoding="utf-8"
    )
    overview.to_csv(paths["overview"], index=False, encoding="utf-8")
    per_track.to_csv(paths["per_track"], index=False, encoding="utf-8")
    result["transitions"].to_csv(
        paths["transitions"], index=False, encoding="utf-8"
    )
    paths["validation"].write_text(
        json.dumps(result["validation"], indent=2),
        encoding="utf-8",
    )
    return {
        **result["validation"],
        **{f"{key}_path": str(path) for key, path in paths.items()},
    }


def register_authoritative_roi_review_in_manifest(
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    *,
    review_workspace_path: str = "",
    exclusive_artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Activate reviewed ROI outputs and quarantine stale ROI-dependent modules.

    ``artifacts`` describes concurrent-membership ROI bouts.  New integrated
    reviews pass a second independently materialized ``exclusive_artifacts``
    bundle for ROI-X.  When it is omitted, the legacy marker reviewer keeps
    its historical single-bundle behavior for backward compatibility.
    """

    required_artifacts = {
        "events_path",
        "dwell_events_path",
        "overview_path",
        "per_track_path",
        "transitions_path",
        "validation_path",
    }
    missing = sorted(
        key for key in required_artifacts if not _text(artifacts.get(key))
    )
    if missing:
        raise ROIReviewValidationError(
            "Reviewed ROI bundle is missing artifact paths: " + ", ".join(missing)
        )
    if exclusive_artifacts is not None:
        missing_exclusive = sorted(
            key
            for key in required_artifacts
            if not _text(exclusive_artifacts.get(key))
        )
        if missing_exclusive:
            raise ROIReviewValidationError(
                "Reviewed exclusive ROI bundle is missing artifact paths: "
                + ", ".join(missing_exclusive)
            )

    updated = copy.deepcopy(dict(manifest))
    outputs = updated.setdefault("outputs", {})
    if not isinstance(outputs, dict):
        outputs = {}
        updated["outputs"] = outputs

    raw_events = _text(outputs.get("raw_roi_events_csv") or outputs.get("roi_events_csv"))
    raw_metrics = (
        copy.deepcopy(outputs.get("raw_roi_metrics_files"))
        if isinstance(outputs.get("raw_roi_metrics_files"), dict)
        else copy.deepcopy(outputs.get("roi_metrics_files"))
        if isinstance(outputs.get("roi_metrics_files"), dict)
        else {}
    )
    if raw_events:
        outputs["raw_roi_events_csv"] = raw_events
    if raw_metrics:
        outputs["raw_roi_metrics_files"] = raw_metrics

    outputs["roi_review_workspace_csv"] = _text(review_workspace_path)
    outputs["reviewed_roi_events_csv"] = _text(artifacts.get("events_path"))
    outputs["reviewed_roi_all_visits_csv"] = _text(artifacts.get("all_visits_path"))
    outputs["reviewed_roi_dwell_events_csv"] = _text(
        artifacts.get("dwell_events_path")
    )
    outputs["reviewed_roi_overview_csv"] = _text(artifacts.get("overview_path"))
    outputs["reviewed_roi_per_track_csv"] = _text(artifacts.get("per_track_path"))
    outputs["reviewed_roi_transitions_csv"] = _text(
        artifacts.get("transitions_path")
    )
    outputs["reviewed_roi_validation_json"] = _text(
        artifacts.get("validation_path")
    )
    if exclusive_artifacts is not None:
        outputs["reviewed_roi_exclusive_events_csv"] = _text(
            exclusive_artifacts.get("events_path")
        )
        outputs["reviewed_roi_exclusive_all_visits_csv"] = _text(
            exclusive_artifacts.get("all_visits_path")
        )
        outputs["reviewed_roi_exclusive_dwell_events_csv"] = _text(
            exclusive_artifacts.get("dwell_events_path")
        )
        outputs["reviewed_roi_exclusive_overview_csv"] = _text(
            exclusive_artifacts.get("overview_path")
        )
        outputs["reviewed_roi_exclusive_per_track_csv"] = _text(
            exclusive_artifacts.get("per_track_path")
        )
        outputs["reviewed_roi_exclusive_transitions_csv"] = _text(
            exclusive_artifacts.get("transitions_path")
        )
        outputs["reviewed_roi_exclusive_validation_json"] = _text(
            exclusive_artifacts.get("validation_path")
        )
    outputs["roi_events_csv"] = outputs["reviewed_roi_events_csv"]

    exclusive_overview = (
        outputs["reviewed_roi_exclusive_overview_csv"]
        if exclusive_artifacts is not None
        else outputs["reviewed_roi_overview_csv"]
    )
    exclusive_per_track = (
        outputs["reviewed_roi_exclusive_per_track_csv"]
        if exclusive_artifacts is not None
        else outputs["reviewed_roi_per_track_csv"]
    )
    exclusive_dwell = (
        outputs["reviewed_roi_exclusive_dwell_events_csv"]
        if exclusive_artifacts is not None
        else outputs["reviewed_roi_dwell_events_csv"]
    )
    active_roi_files = dict(raw_metrics)
    active_roi_files.update(
        {
            "entries_exits": outputs["reviewed_roi_overview_csv"],
            "exclusive_entries_exits": exclusive_overview,
            "entries_exits_by_track": outputs["reviewed_roi_per_track_csv"],
            "exclusive_entries_exits_by_track": exclusive_per_track,
            "dwell_events": outputs["reviewed_roi_dwell_events_csv"],
            "exclusive_dwell_events": exclusive_dwell,
            "transitions": outputs["reviewed_roi_transitions_csv"],
        }
    )
    outputs["roi_metrics_files"] = active_roi_files

    modules = outputs.get("modules") if isinstance(outputs.get("modules"), dict) else {}
    invalidated_modules = (
        copy.deepcopy(outputs.get("invalidated_raw_roi_modules"))
        if isinstance(outputs.get("invalidated_raw_roi_modules"), dict)
        else {}
    )
    for key in sorted(ROI_DEPENDENT_MODULE_KEYS):
        if key in modules:
            invalidated_modules[key] = modules.pop(key)
    outputs["modules"] = modules
    if invalidated_modules:
        outputs["invalidated_raw_roi_modules"] = invalidated_modules

    notes = updated.setdefault("notes", {})
    if not isinstance(notes, dict):
        notes = {}
        updated["notes"] = notes
    notes["roi_review"] = {
        "status": "complete",
        "reviewed_event_count": int(artifacts.get("reviewed_event_count", 0) or 0),
        "null_event_count": int(artifacts.get("null_event_count", 0) or 0),
        "visit_count": int(artifacts.get("visit_count", 0) or 0),
        "qualified_visit_count": int(
            artifacts.get("qualified_visit_count", 0) or 0
        ),
        "below_minimum_visit_count": int(
            artifacts.get("below_minimum_visit_count", 0) or 0
        ),
        "exclusive_reviewed_event_count": int(
            exclusive_artifacts.get("reviewed_event_count", 0) or 0
        )
        if exclusive_artifacts is not None
        else int(artifacts.get("reviewed_event_count", 0) or 0),
        "exclusive_qualified_visit_count": int(
            exclusive_artifacts.get("qualified_visit_count", 0) or 0
        )
        if exclusive_artifacts is not None
        else int(artifacts.get("qualified_visit_count", 0) or 0),
        "occupancy_semantics": (
            "separate_concurrent_and_exclusive_authoritative_bundles"
            if exclusive_artifacts is not None
            else "legacy_single_marker_bundle"
        ),
        "invalidated_raw_roi_modules": sorted(invalidated_modules),
        "invalidation_reason": (
            "These outputs were computed from pre-review ROI events and were removed "
            "from active manifest outputs."
        ),
    }
    return updated
