"""Scientific bout-review records and authoritative materialization.

Detected bouts use an inclusive ``[Start Frame, End Frame]`` interval.  Review
never re-runs bout construction: the max-gap and minimum-duration settings that
created the detected table remain part of its provenance.
"""

from __future__ import annotations

import os
import ast
import copy
import json
import numbers
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from integra_pose.utils.operation_result import OperationResult, OperationStatus


INTERVAL_SEMANTICS = "inclusive_start_and_end_frames"
TERMINAL_REVIEW_STATUSES = frozenset({"confirmed", "rejected", "corrected"})
BOUT_DEPENDENT_MODULE_KEYS = frozenset(
    {
        "behavior_transitions",
        "temporal_trends",
        "activity_budgets",
        "multi_animal_descriptors",
        "kinematic_descriptors",
        "detection_quality",
        "inter_bout_intervals",
        "roi_context_windows",
        "bout_timeline_export",
    }
)
EMPTY_BOUT_COLUMNS = [
    "Bout ID",
    "Run ID",
    "Source Video",
    "Track ID",
    "Behavior",
    "Original Behavior",
    "Start Frame",
    "End Frame",
    "Original Start Frame",
    "Original End Frame",
    "Duration (Frames)",
    "Start Time (s)",
    "End Time (s)",
    "Duration (s)",
    "Interval Semantics",
]
DECISION_COLUMNS = [
    "Decision ID",
    "Decision Sequence",
    "Bout ID",
    "Run ID",
    "Source Video",
    "Decision",
    "Corrected Behavior",
    "Corrected Start Frame",
    "Corrected End Frame",
    "Reviewer Notes",
    "Reviewer",
    "Reviewed At (UTC)",
    "Interval Semantics",
]


class BoutReviewError(ValueError):
    """Raised when review data would violate the bout contract."""


class IncompleteBoutReviewError(BoutReviewError):
    """Raised when an authoritative table is requested before review is done."""


@dataclass(frozen=True)
class BoutReviewPaths:
    authoritative: Path
    workspace: Path
    decisions: Path
    raw_detected: Path
    summary: Path

    @classmethod
    def from_authoritative(cls, path: str | os.PathLike[str]) -> "BoutReviewPaths":
        authoritative = Path(path).expanduser().resolve()
        stem = authoritative.stem
        if stem.endswith("_reviewed_bouts"):
            base = stem[: -len("_reviewed_bouts")]
        elif stem.endswith("_authoritative"):
            base = stem[: -len("_authoritative")]
        else:
            base = stem
        suffix = authoritative.suffix or ".csv"
        parent = authoritative.parent
        return cls(
            authoritative=authoritative,
            workspace=parent / f"{base}_bout_review_workspace{suffix}",
            decisions=parent / f"{base}_bout_review_decisions{suffix}",
            raw_detected=parent / f"{base}_detected_bouts_raw{suffix}",
            summary=parent / f"{base}_reviewed_bouts_summary{suffix}",
        )


def _first_present_column(df: pd.DataFrame, *candidates: str) -> str | None:
    return next((candidate for candidate in candidates if candidate in df.columns), None)


def _coerce_frame(value: Any, *, field: str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise BoutReviewError(f"{field} must be an integer frame index.") from exc
    if not numeric.is_integer():
        raise BoutReviewError(f"{field} must be an integer frame index, got {value!r}.")
    frame = int(numeric)
    if frame < 0:
        raise BoutReviewError(f"{field} must be non-negative, got {frame}.")
    return frame


def inclusive_duration_frames(start_frame: Any, end_frame: Any) -> int:
    """Return duration for a closed frame interval."""

    start = _coerce_frame(start_frame, field="Start Frame")
    end = _coerce_frame(end_frame, field="End Frame")
    if end < start:
        raise BoutReviewError(
            f"End Frame ({end}) must be greater than or equal to Start Frame ({start})."
        )
    return end - start + 1


def _clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _clean_identifier(value: Any) -> str:
    text = _clean_text(value)
    try:
        numeric = float(text)
        if numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError):
        pass
    return text


def _stable_bout_id(
    *,
    source_video: str,
    run_id: str,
    track_id: str,
    behavior: str,
    start_frame: int,
    end_frame: int,
    roi_name: str,
    duplicate_ordinal: int,
) -> str:
    token = "|".join(
        (
            source_video,
            run_id,
            track_id,
            behavior,
            str(start_frame),
            str(end_frame),
            roi_name,
            str(duplicate_ordinal),
        )
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, token))


def normalize_detected_bouts(
    raw_df: pd.DataFrame | None,
    *,
    source_video: str = "",
    fps: float | None = None,
) -> pd.DataFrame:
    """Normalize a raw detected table without mutating the caller's DataFrame."""

    if raw_df is None:
        return pd.DataFrame(columns=EMPTY_BOUT_COLUMNS)
    if raw_df.empty:
        columns = list(raw_df.columns)
        for column in EMPTY_BOUT_COLUMNS:
            if column not in columns:
                columns.append(column)
        return pd.DataFrame(columns=columns)
    work = raw_df.copy(deep=True).reset_index(drop=True)
    track_col = _first_present_column(work, "Track ID", "Animal ID")
    behavior_col = _first_present_column(work, "Original Behavior", "Detected Behavior", "Behavior")
    start_col = _first_present_column(work, "Original Start Frame", "Start Frame", "Bout Start Frame")
    end_col = _first_present_column(work, "Original End Frame", "End Frame", "Bout End Frame")
    missing = [
        label
        for label, column in (
            ("Track ID/Animal ID", track_col),
            ("Behavior", behavior_col),
            ("Start Frame", start_col),
            ("End Frame", end_col),
        )
        if column is None
    ]
    if missing:
        raise BoutReviewError(f"Detected bouts are missing required columns: {', '.join(missing)}")

    source_default = _clean_text(source_video)
    run_col = _first_present_column(work, "Run ID")
    source_col = _first_present_column(work, "Source Video")
    bout_id_col = _first_present_column(work, "Bout ID")
    roi_col = _first_present_column(work, "ROI Name")
    rows: list[dict[str, Any]] = []
    duplicate_counts: dict[tuple[str, ...], int] = {}
    candidate_ids: list[str] = []

    for _, row in work.iterrows():
        track_id = _clean_identifier(row.get(track_col))
        behavior = _clean_text(row.get(behavior_col))
        if not behavior:
            raise BoutReviewError("Every detected bout must have a behavior label.")
        start = _coerce_frame(row.get(start_col), field="Start Frame")
        end = _coerce_frame(row.get(end_col), field="End Frame")
        duration_frames = inclusive_duration_frames(start, end)
        source = _clean_text(row.get(source_col)) if source_col else source_default
        source = source or source_default
        run_id = _clean_text(row.get(run_col)) if run_col else ""
        roi_name = _clean_text(row.get(roi_col)) if roi_col else ""
        identity = (source, run_id, track_id, behavior, str(start), str(end), roi_name)
        ordinal = duplicate_counts.get(identity, 0)
        duplicate_counts[identity] = ordinal + 1
        existing_id = _clean_text(row.get(bout_id_col)) if bout_id_col else ""
        bout_id = existing_id or _stable_bout_id(
            source_video=source,
            run_id=run_id,
            track_id=track_id,
            behavior=behavior,
            start_frame=start,
            end_frame=end,
            roi_name=roi_name,
            duplicate_ordinal=ordinal,
        )
        candidate_ids.append(bout_id)
        payload = row.to_dict()
        payload["Bout ID"] = bout_id
        if run_id:
            payload["Run ID"] = run_id
        payload["Source Video"] = source
        payload["Track ID"] = track_id
        payload["Behavior"] = behavior
        payload["Original Behavior"] = behavior
        payload["Start Frame"] = start
        payload["End Frame"] = end
        payload["Original Start Frame"] = start
        payload["Original End Frame"] = end
        payload["Duration (Frames)"] = duration_frames
        payload["Interval Semantics"] = INTERVAL_SEMANTICS
        if fps is not None and float(fps) > 0:
            safe_fps = float(fps)
            payload["Start Time (s)"] = start / safe_fps
            payload["End Time (s)"] = end / safe_fps
            payload["Duration (s)"] = duration_frames / safe_fps
        for review_column in (
            "status",
            "Review Status",
            "Corrected Behavior",
            "Corrected Manually",
            "Reviewer Notes",
            "Reviewer",
            "Reviewed At (UTC)",
        ):
            payload.pop(review_column, None)
        rows.append(payload)

    # Duplicate source IDs cannot safely identify independent decisions. Keep a
    # valid unique ID and preserve the supplied value as provenance.
    seen: dict[str, int] = {}
    for index, payload in enumerate(rows):
        candidate = candidate_ids[index]
        occurrence = seen.get(candidate, 0)
        seen[candidate] = occurrence + 1
        if occurrence:
            payload["Source Bout ID"] = candidate
            payload["Bout ID"] = _stable_bout_id(
                source_video=_clean_text(payload.get("Source Video")),
                run_id=_clean_text(payload.get("Run ID")),
                track_id=_clean_text(payload.get("Track ID")),
                behavior=_clean_text(payload.get("Behavior")),
                start_frame=int(payload["Start Frame"]),
                end_frame=int(payload["End Frame"]),
                roi_name=_clean_text(payload.get("ROI Name")),
                duplicate_ordinal=occurrence,
            )
    return pd.DataFrame(rows).reset_index(drop=True)


def normalize_review_decisions(decisions_df: pd.DataFrame | None) -> pd.DataFrame:
    if decisions_df is None or decisions_df.empty:
        return pd.DataFrame(columns=DECISION_COLUMNS)
    work = decisions_df.copy(deep=True)
    for column in DECISION_COLUMNS:
        if column not in work.columns:
            work[column] = ""
    work["Decision"] = work["Decision"].map(_clean_text).str.lower()
    invalid = sorted(set(work["Decision"]) - TERMINAL_REVIEW_STATUSES)
    if invalid:
        raise BoutReviewError(f"Unsupported bout review decision(s): {', '.join(invalid)}")
    work["Bout ID"] = work["Bout ID"].map(_clean_text)
    if (work["Bout ID"] == "").any():
        raise BoutReviewError("Every review decision must reference a Bout ID.")
    work["Decision ID"] = work["Decision ID"].map(_clean_text)
    if (work["Decision ID"] == "").any():
        raise BoutReviewError("Every review decision must have a Decision ID.")
    if work["Decision ID"].duplicated().any():
        raise BoutReviewError("Review Decision IDs must be unique.")
    sequence = pd.to_numeric(work["Decision Sequence"], errors="coerce")
    fallback = pd.Series(range(1, len(work) + 1), index=work.index, dtype=float)
    work["Decision Sequence"] = sequence.where(sequence.notna(), fallback).astype(int)
    if (work["Decision Sequence"] <= 0).any() or work["Decision Sequence"].duplicated().any():
        raise BoutReviewError("Review Decision Sequence values must be unique positive integers.")
    return work.loc[:, DECISION_COLUMNS].sort_values("Decision Sequence", kind="stable").reset_index(drop=True)


def validate_review_decision_references(
    raw_bouts_df: pd.DataFrame,
    decisions_df: pd.DataFrame | None,
) -> None:
    raw = normalize_detected_bouts(raw_bouts_df)
    decisions = normalize_review_decisions(decisions_df)
    if decisions.empty:
        return
    raw_ids = set(raw["Bout ID"].map(_clean_text))
    decision_ids = set(decisions["Bout ID"].map(_clean_text))
    orphaned = sorted(decision_ids - raw_ids)
    if orphaned:
        preview = ", ".join(orphaned[:3])
        suffix = "..." if len(orphaned) > 3 else ""
        raise BoutReviewError(
            f"Review decisions reference {len(orphaned)} unknown Bout ID(s): {preview}{suffix}"
        )


def decisions_from_workspace(workspace_df: pd.DataFrame | None) -> pd.DataFrame:
    """Convert the legacy one-row-per-bout state table to a decision log."""

    if workspace_df is None or workspace_df.empty:
        return normalize_review_decisions(None)
    rows: list[dict[str, Any]] = []
    for sequence, (_, row) in enumerate(workspace_df.iterrows(), start=1):
        decision = _clean_text(row.get("status") or row.get("Review Status")).lower()
        if decision not in TERMINAL_REVIEW_STATUSES:
            continue
        bout_id = _clean_text(row.get("Bout ID"))
        if not bout_id:
            continue
        rows.append(
            {
                "Decision ID": str(uuid.uuid4()),
                "Decision Sequence": sequence,
                "Bout ID": bout_id,
                "Run ID": _clean_text(row.get("Run ID")),
                "Source Video": _clean_text(row.get("Source Video")),
                "Decision": decision,
                "Corrected Behavior": _clean_text(row.get("Corrected Behavior")),
                "Corrected Start Frame": _clean_text(row.get("Corrected Start Frame")),
                "Corrected End Frame": _clean_text(row.get("Corrected End Frame")),
                "Reviewer Notes": _clean_text(row.get("Reviewer Notes")),
                "Reviewer": _clean_text(row.get("Reviewer")),
                "Reviewed At (UTC)": _clean_text(row.get("Reviewed At (UTC)")),
                "Interval Semantics": INTERVAL_SEMANTICS,
            }
        )
    return normalize_review_decisions(pd.DataFrame(rows, columns=DECISION_COLUMNS))


def migrate_legacy_review_workspace(
    raw_bouts_df: pd.DataFrame,
    legacy_review_df: pd.DataFrame,
    *,
    source_video: str = "",
    fps: float | None = None,
) -> pd.DataFrame:
    """Attach a pre-ID review sidecar to its raw table with an audit marker.

    Stable IDs are preferred. A legacy table without them can only be migrated
    when it has exactly one row per raw bout and any identifying columns it does
    provide agree row-for-row with the immutable source.
    """

    raw = normalize_detected_bouts(raw_bouts_df, source_video=source_video, fps=fps)
    legacy = legacy_review_df.copy(deep=True).reset_index(drop=True)
    if legacy.empty:
        return build_review_workspace(raw, None, fps=fps)
    if len(legacy) != len(raw):
        raise BoutReviewError(
            "Legacy review sidecar cannot be migrated because its row count "
            f"({len(legacy)}) differs from the raw bout count ({len(raw)})."
        )
    if "Bout ID" in legacy.columns and legacy["Bout ID"].map(_clean_text).ne("").all():
        raw_ids = set(raw["Bout ID"].map(_clean_text))
        legacy_ids = set(legacy["Bout ID"].map(_clean_text))
        if raw_ids != legacy_ids:
            raise BoutReviewError("Legacy review Bout IDs do not match the raw detected-bout table.")
        source_by_id = raw.set_index(raw["Bout ID"].map(_clean_text), drop=False)
        legacy = legacy.set_index(legacy["Bout ID"].map(_clean_text), drop=False).loc[source_by_id.index].reset_index(drop=True)
        migration_method = "stable_bout_id"
    else:
        comparisons = (
            ("Track ID", "Track ID"),
            ("Animal ID", "Track ID"),
            ("Original Behavior", "Original Behavior"),
            ("Start Frame", "Original Start Frame"),
            ("End Frame", "Original End Frame"),
        )
        for legacy_column, raw_column in comparisons:
            if legacy_column not in legacy.columns:
                continue
            left = legacy[legacy_column].map(_clean_text).tolist()
            right = raw[raw_column].map(_clean_text).tolist()
            if left != right:
                raise BoutReviewError(
                    f"Legacy review column {legacy_column!r} does not match the raw bouts in row order."
                )
        legacy["Bout ID"] = raw["Bout ID"].tolist()
        migration_method = "validated_row_order"

    migrated = raw.copy(deep=True)
    for column in (
        "status",
        "Review Status",
        "Corrected Behavior",
        "Corrected Manually",
        "Reviewer Notes",
        "Reviewer",
        "Reviewed At (UTC)",
    ):
        if column in legacy.columns:
            migrated[column] = legacy[column].tolist()
    migrated["Review Migration"] = migration_method
    return migrated


def append_review_decision(
    decisions_df: pd.DataFrame | None,
    raw_bout: Mapping[str, Any] | pd.Series,
    decision: str,
    *,
    corrected_behavior: str = "",
    corrected_start_frame: int | None = None,
    corrected_end_frame: int | None = None,
    reviewer_notes: str = "",
    reviewer: str = "",
) -> pd.DataFrame:
    decisions = normalize_review_decisions(decisions_df)
    normalized_decision = _clean_text(decision).lower()
    if normalized_decision not in TERMINAL_REVIEW_STATUSES:
        raise BoutReviewError(f"Unsupported bout review decision: {decision!r}")
    bout_id = _clean_text(raw_bout.get("Bout ID"))
    if not bout_id:
        raise BoutReviewError("Cannot review a bout without a stable Bout ID.")
    behavior = _clean_text(raw_bout.get("Behavior"))
    corrected = _clean_text(corrected_behavior)
    if normalized_decision == "corrected" and not corrected:
        corrected = behavior
    start = int(raw_bout.get("Start Frame", 0)) if corrected_start_frame is None else int(corrected_start_frame)
    end = int(raw_bout.get("End Frame", start)) if corrected_end_frame is None else int(corrected_end_frame)
    inclusive_duration_frames(start, end)
    sequence = int(decisions["Decision Sequence"].max()) + 1 if not decisions.empty else 1
    row = {
        "Decision ID": str(uuid.uuid4()),
        "Decision Sequence": sequence,
        "Bout ID": bout_id,
        "Run ID": _clean_text(raw_bout.get("Run ID")),
        "Source Video": _clean_text(raw_bout.get("Source Video")),
        "Decision": normalized_decision,
        "Corrected Behavior": corrected,
        "Corrected Start Frame": start if normalized_decision == "corrected" else "",
        "Corrected End Frame": end if normalized_decision == "corrected" else "",
        "Reviewer Notes": _clean_text(reviewer_notes),
        "Reviewer": _clean_text(reviewer),
        "Reviewed At (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "Interval Semantics": INTERVAL_SEMANTICS,
    }
    return pd.concat([decisions, pd.DataFrame([row], columns=DECISION_COLUMNS)], ignore_index=True)


def _latest_decisions(decisions_df: pd.DataFrame | None) -> pd.DataFrame:
    decisions = normalize_review_decisions(decisions_df)
    if decisions.empty:
        return decisions
    return decisions.drop_duplicates(subset=["Bout ID"], keep="last").set_index("Bout ID", drop=False)


def build_review_workspace(
    raw_bouts_df: pd.DataFrame,
    decisions_df: pd.DataFrame | None,
    *,
    fps: float | None = None,
) -> pd.DataFrame:
    raw = normalize_detected_bouts(raw_bouts_df, fps=fps)
    validate_review_decision_references(raw, decisions_df)
    if raw.empty:
        workspace = raw.copy()
        for column in (
            "status",
            "Review Status",
            "Corrected Behavior",
            "Corrected Manually",
            "Reviewer Notes",
            "Reviewer",
            "Reviewed At (UTC)",
        ):
            if column not in workspace.columns:
                workspace[column] = pd.Series(dtype="object")
        return workspace
    latest = _latest_decisions(decisions_df)
    rows: list[dict[str, Any]] = []
    for _, raw_row in raw.iterrows():
        row = raw_row.to_dict()
        bout_id = _clean_text(row.get("Bout ID"))
        if bout_id in latest.index:
            decision = latest.loc[bout_id]
            status = _clean_text(decision.get("Decision")).lower()
            corrected_behavior = _clean_text(decision.get("Corrected Behavior"))
            if status == "corrected" and corrected_behavior:
                row["Behavior"] = corrected_behavior
            row["Corrected Behavior"] = corrected_behavior or _clean_text(row.get("Behavior"))
            row["Corrected Manually"] = status == "corrected"
            row["Reviewer Notes"] = _clean_text(decision.get("Reviewer Notes"))
            row["Reviewer"] = _clean_text(decision.get("Reviewer"))
            row["Reviewed At (UTC)"] = _clean_text(decision.get("Reviewed At (UTC)"))
        else:
            status = "unreviewed"
            row["Corrected Behavior"] = _clean_text(row.get("Behavior"))
            row["Corrected Manually"] = False
            row["Reviewer Notes"] = ""
            row["Reviewer"] = ""
            row["Reviewed At (UTC)"] = ""
        row["status"] = status
        row["Review Status"] = status
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def review_is_complete(raw_bouts_df: pd.DataFrame, decisions_df: pd.DataFrame | None) -> bool:
    workspace = build_review_workspace(raw_bouts_df, decisions_df)
    if workspace.empty:
        return True
    return bool(workspace["status"].isin(TERMINAL_REVIEW_STATUSES).all())


def materialize_reviewed_bouts(
    raw_bouts_df: pd.DataFrame,
    decisions_df: pd.DataFrame | None,
    *,
    fps: float,
    require_complete: bool = True,
) -> pd.DataFrame:
    """Apply terminal decisions to raw bouts and return the analysis table."""

    if fps is None or float(fps) <= 0:
        raise BoutReviewError("A positive video FPS is required to materialize reviewed bouts.")
    raw = normalize_detected_bouts(raw_bouts_df, fps=float(fps))
    workspace = build_review_workspace(raw, decisions_df, fps=float(fps))
    if workspace.empty:
        return workspace
    unreviewed_count = int((workspace["status"] == "unreviewed").sum())
    if require_complete and unreviewed_count:
        raise IncompleteBoutReviewError(
            f"{unreviewed_count} detected bout(s) still require a decision before authoritative export."
        )
    latest = _latest_decisions(decisions_df)
    rows: list[dict[str, Any]] = []
    for _, raw_row in raw.iterrows():
        bout_id = _clean_text(raw_row.get("Bout ID"))
        if bout_id not in latest.index:
            if not require_complete:
                continue
            raise IncompleteBoutReviewError(f"Bout {bout_id} has no review decision.")
        decision = latest.loc[bout_id]
        status = _clean_text(decision.get("Decision")).lower()
        if status == "rejected":
            continue
        payload = raw_row.to_dict()
        raw_start = int(payload["Start Frame"])
        raw_end = int(payload["End Frame"])
        if status == "corrected":
            corrected_behavior = _clean_text(decision.get("Corrected Behavior"))
            if corrected_behavior:
                payload["Behavior"] = corrected_behavior
            corrected_start = _clean_text(decision.get("Corrected Start Frame"))
            corrected_end = _clean_text(decision.get("Corrected End Frame"))
            if corrected_start:
                payload["Start Frame"] = _coerce_frame(corrected_start, field="Corrected Start Frame")
            if corrected_end:
                payload["End Frame"] = _coerce_frame(corrected_end, field="Corrected End Frame")
        start = int(payload["Start Frame"])
        end = int(payload["End Frame"])
        duration_frames = inclusive_duration_frames(start, end)
        boundary_corrected = start != raw_start or end != raw_end
        if boundary_corrected:
            configured_minimum = pd.to_numeric(
                pd.Series([payload.get("Detection Min Bout (Frames)")]),
                errors="coerce",
            ).iloc[0]
            if pd.notna(configured_minimum) and duration_frames < max(
                1, int(configured_minimum)
            ):
                raise BoutReviewError(
                    f"Bout {bout_id} has a corrected duration of {duration_frames} "
                    f"frame(s), below its configured minimum of "
                    f"{int(configured_minimum)} frame(s)."
                )
            for column in (
                "ROI Name",
                "ROI Memberships",
                "Raw ROI Name",
                "Raw ROI Memberships",
            ):
                if column in payload:
                    payload[column] = ""
            if "ROI Context Semantics" in payload:
                payload["ROI Context Semantics"] = (
                    "invalidated_by_reviewed_boundary_correction"
                )
            for column in (
                "Observed Frames",
                "Bridged Frames",
                "Observed Fraction",
                "Maximum Bridged Gap (Frames)",
            ):
                if column in payload:
                    payload[column] = pd.NA
            if "Bout Construction Semantics" in payload:
                payload["Bout Construction Semantics"] = (
                    "manual_boundary_correction"
                )
            if "Minimum Bout Basis" in payload:
                payload["Minimum Bout Basis"] = "reviewed_interval_span"
        payload["Duration (Frames)"] = duration_frames
        payload["Start Time (s)"] = start / float(fps)
        payload["End Time (s)"] = end / float(fps)
        payload["Duration (s)"] = duration_frames / float(fps)
        payload["Review Status"] = status
        payload["Reviewer Notes"] = _clean_text(decision.get("Reviewer Notes"))
        payload["Reviewer"] = _clean_text(decision.get("Reviewer"))
        payload["Reviewed At (UTC)"] = _clean_text(decision.get("Reviewed At (UTC)"))
        payload["Interval Semantics"] = INTERVAL_SEMANTICS
        rows.append(payload)
    if rows:
        reviewed = pd.DataFrame(rows).reset_index(drop=True)
        if {
            "Track ID",
            "Start Frame",
            "End Frame",
        }.issubset(reviewed.columns):
            for track_id, track_bouts in reviewed.groupby("Track ID", sort=False):
                ordered = track_bouts.sort_values(
                    ["Start Frame", "End Frame"], kind="stable"
                )
                previous = None
                for _, row in ordered.iterrows():
                    if (
                        previous is not None
                        and int(row["Start Frame"]) <= int(previous["End Frame"])
                    ):
                        raise BoutReviewError(
                            f"Reviewed bouts overlap for track {track_id}: "
                            f"{_clean_text(previous.get('Bout ID'))} ends at "
                            f"{int(previous['End Frame'])}, while "
                            f"{_clean_text(row.get('Bout ID'))} starts at "
                            f"{int(row['Start Frame'])}."
                        )
                    previous = row
        return reviewed
    columns = list(raw.columns)
    for column in ("Review Status", "Reviewer Notes", "Reviewer", "Reviewed At (UTC)"):
        if column not in columns:
            columns.append(column)
    return pd.DataFrame(columns=columns)


def summarize_reviewed_bouts(reviewed_df: pd.DataFrame) -> pd.DataFrame:
    if reviewed_df is None:
        return pd.DataFrame()
    group_columns = [column for column in ("Run ID", "Track ID", "Behavior", "ROI Name") if column in reviewed_df.columns]
    if "Track ID" not in group_columns or "Behavior" not in group_columns:
        raise BoutReviewError("Reviewed bouts require Track ID and Behavior columns for summary export.")
    if reviewed_df.empty:
        return pd.DataFrame(
            columns=group_columns + ["Bout_Count", "Total_Duration_s", "Mean_Duration_s"]
        )
    summary = (
        reviewed_df.groupby(group_columns, dropna=False)
        .agg(
            Bout_Count=("Behavior", "size"),
            Total_Duration_s=("Duration (s)", "sum"),
            Mean_Duration_s=("Duration (s)", "mean"),
        )
        .reset_index()
    )
    return summary


def _canonical_raw_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        text = value
        if text[:1] in {"[", "(", "{"} and text[-1:] in {"]",
            ")",
            "}",
        }:
            try:
                return _canonical_raw_value(ast.literal_eval(text))
            except (SyntaxError, ValueError):
                pass
        return text
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_raw_value(nested)
            for key, nested in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_raw_value(item) for item in value]
    if isinstance(value, set):
        return sorted((_canonical_raw_value(item) for item in value), key=lambda item: repr(item))
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        numeric = float(value)
        if numeric.is_integer():
            return int(numeric)
        return round(numeric, 12)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _raw_signature(df: pd.DataFrame) -> str:
    normalized = normalize_detected_bouts(df)
    records: list[dict[str, Any]] = []
    ordered = normalized.sort_values("Bout ID", kind="stable") if not normalized.empty else normalized
    for _, row in ordered.iterrows():
        records.append(
            {
                str(column): _canonical_raw_value(row.get(column))
                for column in sorted(normalized.columns)
            }
        )
    return json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _stage_csv(df: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    df.to_csv(staged, index=False, encoding="utf-8")
    return staged


def _commit_staged_bundle(staged: Mapping[Path, Path]) -> None:
    backups: dict[Path, Path] = {}
    committed: list[Path] = []
    try:
        for destination, staged_path in staged.items():
            if destination.exists():
                backup = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.bak")
                os.replace(destination, backup)
                backups[destination] = backup
            os.replace(staged_path, destination)
            committed.append(destination)
    except Exception:
        for destination in reversed(committed):
            try:
                destination.unlink(missing_ok=True)
            except Exception:
                pass
        for destination, backup in backups.items():
            try:
                if backup.exists():
                    os.replace(backup, destination)
            except Exception:
                pass
        raise
    else:
        for backup in backups.values():
            try:
                backup.unlink(missing_ok=True)
            except Exception:
                pass


def save_review_bundle(
    raw_bouts_df: pd.DataFrame,
    decisions_df: pd.DataFrame | None,
    *,
    authoritative_path: str | os.PathLike[str],
    fps: float,
) -> OperationResult:
    """Atomically save review progress and, when complete, authoritative data."""

    staged_paths: list[Path] = []
    try:
        paths = BoutReviewPaths.from_authoritative(authoritative_path)
        raw = normalize_detected_bouts(raw_bouts_df, fps=fps)
        decisions = normalize_review_decisions(decisions_df)
        workspace = build_review_workspace(raw, decisions, fps=fps)
        if paths.raw_detected.is_file():
            existing_raw = pd.read_csv(paths.raw_detected)
            if _raw_signature(existing_raw) != _raw_signature(raw):
                return OperationResult.failure(
                    "Detected bouts differ from the immutable review source.",
                    error=(
                        f"Refusing to overwrite {paths.raw_detected}. Start a new review output "
                        "or restore the matching detected-bout table."
                    ),
                )

        complete = review_is_complete(raw, decisions)
        reviewed = pd.DataFrame()
        summary = pd.DataFrame()
        staged: dict[Path, Path] = {}
        if not paths.raw_detected.is_file():
            staged[paths.raw_detected] = _stage_csv(raw, paths.raw_detected)
            staged_paths.append(staged[paths.raw_detected])
        staged[paths.decisions] = _stage_csv(decisions, paths.decisions)
        staged_paths.append(staged[paths.decisions])
        staged[paths.workspace] = _stage_csv(workspace, paths.workspace)
        staged_paths.append(staged[paths.workspace])
        if complete:
            reviewed = materialize_reviewed_bouts(raw, decisions, fps=fps, require_complete=True)
            summary = summarize_reviewed_bouts(reviewed)
            staged[paths.authoritative] = _stage_csv(reviewed, paths.authoritative)
            staged_paths.append(staged[paths.authoritative])
            staged[paths.summary] = _stage_csv(summary, paths.summary)
            staged_paths.append(staged[paths.summary])
        _commit_staged_bundle(staged)
        artifacts = {
            "raw_detected_path": str(paths.raw_detected),
            "decisions_path": str(paths.decisions),
            "workspace_path": str(paths.workspace),
            "review_complete": complete,
            "review_not_required": bool(raw.empty),
        }
        if complete:
            artifacts.update(
                {
                    "authoritative_path": str(paths.authoritative),
                    "summary_path": str(paths.summary),
                    "accepted_bout_count": len(reviewed),
                    "rejected_bout_count": int((workspace["status"] == "rejected").sum()),
                }
            )
            return OperationResult.success("Bout review saved and authoritative outputs materialized.", **artifacts)
        return OperationResult(
            OperationStatus.PARTIAL,
            message="Bout review progress saved; authoritative outputs await remaining decisions.",
            artifacts=artifacts,
        )
    except Exception as exc:
        for staged_path in staged_paths:
            try:
                staged_path.unlink(missing_ok=True)
            except Exception:
                pass
        return OperationResult.failure("Failed to save bout review.", error=str(exc))


def register_authoritative_review_in_manifest(
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Register reviewed outputs and quarantine raw-bout-dependent artifacts."""

    authoritative_path = _clean_text(artifacts.get("authoritative_path"))
    summary_path = _clean_text(artifacts.get("summary_path"))
    if not authoritative_path or not summary_path:
        raise BoutReviewError("Complete authoritative and summary paths are required for manifest registration.")
    updated = copy.deepcopy(dict(manifest))
    outputs = updated.setdefault("outputs", {})
    if not isinstance(outputs, dict):
        outputs = {}
        updated["outputs"] = outputs

    raw_detailed = _clean_text(outputs.get("raw_detailed_bouts_csv") or outputs.get("detailed_bouts_csv"))
    raw_summary = _clean_text(outputs.get("raw_summary_csv") or outputs.get("summary_csv"))
    if raw_detailed:
        outputs["raw_detailed_bouts_csv"] = raw_detailed
    if raw_summary:
        outputs["raw_summary_csv"] = raw_summary
    outputs["raw_detected_bouts_csv"] = _clean_text(artifacts.get("raw_detected_path"))
    outputs["bout_review_decisions_csv"] = _clean_text(artifacts.get("decisions_path"))
    outputs["bout_review_workspace_csv"] = _clean_text(artifacts.get("workspace_path"))
    outputs["reviewed_bouts_csv"] = authoritative_path
    outputs["reviewed_bouts_summary_csv"] = summary_path
    outputs["detailed_bouts_csv"] = authoritative_path
    outputs["summary_csv"] = summary_path

    invalidated_core = (
        copy.deepcopy(outputs.get("invalidated_raw_bout_outputs"))
        if isinstance(outputs.get("invalidated_raw_bout_outputs"), dict)
        else {}
    )
    for key in ("bout_summary_export", "excel_summary"):
        value = outputs.pop(key, None)
        if value:
            invalidated_core[key] = value
    modules = outputs.get("modules") if isinstance(outputs.get("modules"), dict) else {}
    invalidated_modules = (
        copy.deepcopy(outputs.get("invalidated_raw_bout_modules"))
        if isinstance(outputs.get("invalidated_raw_bout_modules"), dict)
        else {}
    )
    for key in sorted(BOUT_DEPENDENT_MODULE_KEYS):
        if key in modules:
            invalidated_modules[key] = modules.pop(key)
    outputs["modules"] = modules
    if invalidated_core:
        outputs["invalidated_raw_bout_outputs"] = invalidated_core
    if invalidated_modules:
        outputs["invalidated_raw_bout_modules"] = invalidated_modules

    notes = updated.setdefault("notes", {})
    if not isinstance(notes, dict):
        notes = {}
        updated["notes"] = notes
    notes["bout_review"] = {
        "status": "complete",
        "interval_semantics": INTERVAL_SEMANTICS,
        "authoritative_basis": "immutable_detected_bouts_plus_append_only_review_decisions",
        "accepted_bout_count": int(artifacts.get("accepted_bout_count", 0) or 0),
        "rejected_bout_count": int(artifacts.get("rejected_bout_count", 0) or 0),
        "review_not_required": bool(artifacts.get("review_not_required", False)),
        "invalidated_raw_bout_outputs": sorted(invalidated_core),
        "invalidated_raw_bout_modules": sorted(invalidated_modules),
        "invalidation_reason": (
            "These outputs were computed from pre-review bouts and were removed from active manifest outputs."
        ),
    }
    return updated


def load_review_decisions(
    authoritative_path: str | os.PathLike[str],
    *,
    legacy_workspace: pd.DataFrame | None = None,
) -> pd.DataFrame:
    paths = BoutReviewPaths.from_authoritative(authoritative_path)
    if paths.decisions.is_file():
        return normalize_review_decisions(pd.read_csv(paths.decisions))
    if paths.workspace.is_file():
        return decisions_from_workspace(pd.read_csv(paths.workspace))
    if legacy_workspace is not None:
        return decisions_from_workspace(legacy_workspace)
    return normalize_review_decisions(None)


def ethogram_window(
    bouts_df: pd.DataFrame,
    *,
    focus_bout_id: str,
    context_frames: int,
    total_frames: int,
) -> tuple[int, int, pd.DataFrame]:
    """Return the inclusive display window and all intersecting bouts."""

    if bouts_df is None or bouts_df.empty:
        return 0, max(0, int(total_frames) - 1), pd.DataFrame()
    work = bouts_df.copy()
    match = work[work["Bout ID"].astype(str) == str(focus_bout_id)]
    if match.empty:
        raise BoutReviewError(f"Unknown focus bout ID: {focus_bout_id}")
    focus = match.iloc[0]
    context = max(0, int(context_frames))
    start = max(0, int(focus["Start Frame"]) - context)
    max_frame = max(0, int(total_frames) - 1)
    end = min(max_frame, int(focus["End Frame"]) + context)
    visible = work[
        (pd.to_numeric(work["End Frame"], errors="coerce") >= start)
        & (pd.to_numeric(work["Start Frame"], errors="coerce") <= end)
    ].copy()
    visible.sort_values(["Track ID", "Behavior", "Start Frame", "End Frame"], inplace=True, kind="stable")
    return start, end, visible.reset_index(drop=True)
