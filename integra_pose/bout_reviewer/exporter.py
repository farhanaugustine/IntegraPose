from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .analytics import (
    behavior_correction_rows,
    behavior_transition_matrix_rows,
    behavior_transition_rows,
)
from .models import (
    ACCEPTED,
    ADDED,
    APP_VERSION,
    BEHAVIOR,
    FINGERPRINT_SCHEME,
    MODIFIED,
    OBJECT_INTERACTION,
    ROI_CONCURRENT,
    ROI_EXCLUSIVE,
    ProjectData,
    ReviewBout,
    ReviewError,
    ScoreRow,
)
from .store import ReviewStore


REFERENCE_DECISIONS = {ACCEPTED, MODIFIED, ADDED}

EXPORT_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ROI_Bouts", (ROI_CONCURRENT, ROI_EXCLUSIVE)),
    ("Object_Interactions", (OBJECT_INTERACTION,)),
    ("Behavior_Bouts", (BEHAVIOR,)),
)

REVIEW_FIELDS = (
    "review_id",
    "video_id",
    "event_kind",
    "label",
    "class_id",
    "track_id",
    "start_frame",
    "end_frame",
    "decision",
    "active",
    "origin_prediction_ids",
    "parent_review_ids",
    "note",
    "reviewer",
    "created_at",
    "updated_at",
    "duration_frames",
)

PREDICTION_FIELDS = (
    "prediction_id",
    "video_id",
    "event_kind",
    "label",
    "track_id",
    "start_frame",
    "end_frame",
    "source_file",
    "source_row",
    "class_id",
)

SUMMARY_FIELDS = (
    "video_id",
    "event_kind",
    "label",
    "class_id",
    "track_id",
    "fps",
    "scope_complete",
    "original_bout_count",
    "reviewed_bout_count",
    "reviewed_to_original_bout_ratio",
    "original_dwell_frames",
    "reviewed_dwell_frames",
    "original_dwell_seconds",
    "reviewed_dwell_seconds",
    "reviewed_to_original_dwell_ratio",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def _write_behavior_frames(
    path: Path,
    bouts: Sequence[ReviewBout],
) -> int:
    fieldnames = (
        "video_id",
        "frame",
        "track_id",
        "class_id",
        "behavior",
        "review_id",
        "review_decision",
    )
    count = 0
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for bout in sorted(
            bouts,
            key=lambda item: (
                item.video_id,
                item.track_id,
                -1 if item.class_id is None else item.class_id,
                item.start_frame,
                item.end_frame,
            ),
        ):
            for frame in range(bout.start_frame, bout.end_frame + 1):
                writer.writerow(
                    {
                        "video_id": bout.video_id,
                        "frame": frame,
                        "track_id": bout.track_id,
                        "class_id": bout.class_id,
                        "behavior": bout.label,
                        "review_id": bout.review_id,
                        "review_decision": bout.decision,
                    }
                )
                count += 1
    return count


def _unique_output_dir(root: Path) -> Path:
    parent = root / "bout_review_exports"
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = parent / f"IntegraPose_bout_review_{stamp}"
    output.mkdir(exist_ok=False)
    return output


def _review_row(bout: ReviewBout) -> dict[str, Any]:
    row = bout.to_dict()
    row["active"] = int(bout.active)
    row["origin_prediction_ids"] = json.dumps(bout.origin_prediction_ids)
    row["parent_review_ids"] = json.dumps(bout.parent_review_ids)
    row["duration_frames"] = bout.frames
    return row


def _dwell_rows(
    bouts: Iterable[ReviewBout],
    fps: float,
    event_kind: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    if event_kind in {ROI_CONCURRENT, ROI_EXCLUSIVE}:
        fields = [
            "Track ID",
            "ROI Name",
            "Start Frame",
            "End Frame",
            "Duration (Frames)",
            "Duration (s)",
            "Occupancy Semantics",
            "Review Decision",
            "Review ID",
            "Origin Prediction IDs",
            "Review Note",
        ]
        semantics = (
            "concurrent_membership"
            if event_kind == ROI_CONCURRENT
            else "exclusive_primary"
        )
        for bout in bouts:
            rows.append(
                {
                    "Track ID": bout.track_id,
                    "ROI Name": bout.label,
                    "Start Frame": bout.start_frame,
                    "End Frame": bout.end_frame,
                    "Duration (Frames)": bout.frames,
                    "Duration (s)": bout.frames / fps,
                    "Occupancy Semantics": semantics,
                    "Review Decision": bout.decision,
                    "Review ID": bout.review_id,
                    "Origin Prediction IDs": json.dumps(
                        bout.origin_prediction_ids
                    ),
                    "Review Note": bout.note,
                }
            )
    else:
        fields = [
            "Track ID",
            "Object ROI",
            "Start Frame",
            "End Frame",
            "Duration (Frames)",
            "Duration (s)",
            "Review Decision",
            "Review ID",
            "Origin Prediction IDs",
            "Review Note",
        ]
        for bout in bouts:
            rows.append(
                {
                    "Track ID": bout.track_id,
                    "Object ROI": bout.label,
                    "Start Frame": bout.start_frame,
                    "End Frame": bout.end_frame,
                    "Duration (Frames)": bout.frames,
                    "Duration (s)": bout.frames / fps,
                    "Review Decision": bout.decision,
                    "Review ID": bout.review_id,
                    "Origin Prediction IDs": json.dumps(
                        bout.origin_prediction_ids
                    ),
                    "Review Note": bout.note,
                }
            )
    return rows, fields


def _write_text(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _count_dwell_summary(
    project: ProjectData,
    store: ReviewStore,
    predictions: Sequence[Any],
    corrected: Sequence[ReviewBout],
) -> list[dict[str, Any]]:
    """Return one before/after row per video, channel, class, and track."""

    fps_by_video = {video.video_id: video.fps for video in project.videos}
    keys = {
        (
            bout.video_id,
            bout.event_kind,
            bout.label,
            bout.class_id,
            bout.track_id,
        )
        for bout in predictions
    } | {
        (
            bout.video_id,
            bout.event_kind,
            bout.label,
            bout.class_id,
            bout.track_id,
        )
        for bout in corrected
    }
    rows: list[dict[str, Any]] = []
    for video_id, event_kind, label, class_id, track_id in sorted(
        keys,
        key=lambda item: (
            item[0],
            item[1],
            item[2].casefold(),
            -1 if item[3] is None else item[3],
            item[4],
        ),
    ):
        selected_predictions = [
            bout
            for bout in predictions
            if (
                bout.video_id,
                bout.event_kind,
                bout.label,
                bout.class_id,
                bout.track_id,
            )
            == (video_id, event_kind, label, class_id, track_id)
        ]
        selected_reviewed = [
            bout
            for bout in corrected
            if (
                bout.video_id,
                bout.event_kind,
                bout.label,
                bout.class_id,
                bout.track_id,
            )
            == (video_id, event_kind, label, class_id, track_id)
        ]
        original_count = len(selected_predictions)
        reviewed_count = len(selected_reviewed)
        original_frames = sum(bout.frames for bout in selected_predictions)
        reviewed_frames = sum(bout.frames for bout in selected_reviewed)
        fps = float(fps_by_video[video_id])
        rows.append(
            {
                "video_id": video_id,
                "event_kind": event_kind,
                "label": label,
                "class_id": "" if class_id is None else class_id,
                "track_id": track_id,
                "fps": fps,
                "scope_complete": int(
                    store.scope_complete(
                        video_id,
                        event_kind,
                        track_id if event_kind == BEHAVIOR else None,
                    )
                ),
                "original_bout_count": original_count,
                "reviewed_bout_count": reviewed_count,
                "reviewed_to_original_bout_ratio": _safe_ratio(
                    reviewed_count,
                    original_count,
                ),
                "original_dwell_frames": original_frames,
                "reviewed_dwell_frames": reviewed_frames,
                "original_dwell_seconds": original_frames / fps,
                "reviewed_dwell_seconds": reviewed_frames / fps,
                "reviewed_to_original_dwell_ratio": _safe_ratio(
                    reviewed_frames,
                    original_frames,
                ),
            }
        )
    return rows


def _write_spatial_per_video_tables(
    *,
    table_dir: Path,
    project: ProjectData,
    corrected: Sequence[ReviewBout],
    event_kinds: Sequence[str],
) -> int:
    per_video_dir = table_dir / "Per_Video_Bout_Tables"
    per_video_dir.mkdir(exist_ok=False)
    marker_rows: list[dict[str, Any]] = []
    suffixes = {
        ROI_CONCURRENT: "_reviewed_roi_dwell_events.csv",
        ROI_EXCLUSIVE: "_reviewed_roi_exclusive_dwell_events.csv",
        OBJECT_INTERACTION: "_reviewed_object_interactions_dwell_events.csv",
    }
    for video in project.videos:
        video_bouts = [
            bout
            for bout in corrected
            if bout.video_id == video.video_id
            and bout.event_kind in event_kinds
        ]
        for event_kind in event_kinds:
            kind_bouts = sorted(
                (
                    bout
                    for bout in video_bouts
                    if bout.event_kind == event_kind
                ),
                key=lambda bout: (
                    bout.label,
                    bout.track_id,
                    bout.start_frame,
                    bout.end_frame,
                ),
            )
            rows, fields = _dwell_rows(kind_bouts, video.fps, event_kind)
            _write_csv(
                per_video_dir / f"{video.video_stem}{suffixes[event_kind]}",
                rows,
                fields,
            )
            for bout in kind_bouts:
                source = (
                    "object"
                    if event_kind == OBJECT_INTERACTION
                    else "zone"
                )
                marker_rows.extend(
                    (
                        {
                            "video_id": video.video_id,
                            "event_kind": event_kind,
                            "Source": source,
                            "Event Type": "entry",
                            "Target Name": bout.label,
                            "Track ID": bout.track_id,
                            "Frame": bout.start_frame,
                            "Review ID": bout.review_id,
                        },
                        {
                            "video_id": video.video_id,
                            "event_kind": event_kind,
                            "Source": source,
                            "Event Type": "exit",
                            "Target Name": bout.label,
                            "Track ID": bout.track_id,
                            "Frame": bout.end_frame,
                            "Review ID": bout.review_id,
                        },
                    )
                )
    _write_csv(
        table_dir / "corrected_entry_exit_markers.csv",
        marker_rows,
        (
            "video_id",
            "event_kind",
            "Source",
            "Event Type",
            "Target Name",
            "Track ID",
            "Frame",
            "Review ID",
        ),
    )
    return len(marker_rows)


def _write_behavior_specific_tables(
    *,
    table_dir: Path,
    project: ProjectData,
    store: ReviewStore,
    behavior_bouts: Sequence[ReviewBout],
) -> tuple[int, int]:
    behavior_frame_rows = _write_behavior_frames(
        table_dir / "corrected_behavior_frames.csv",
        behavior_bouts,
    )
    correction_rows = [
        row.to_dict() for row in behavior_correction_rows(store)
    ]
    _write_csv(
        table_dir / "behavior_correction_metrics.csv",
        correction_rows,
        list(correction_rows[0].keys()) if correction_rows else ["scope"],
    )
    transition_rows = behavior_transition_rows(store)
    _write_csv(
        table_dir / "behavior_bout_transitions.csv",
        transition_rows,
        list(transition_rows[0].keys()) if transition_rows else ["video_id"],
    )
    transition_matrix_rows = behavior_transition_matrix_rows(store)
    _write_csv(
        table_dir / "behavior_class_transition_matrix.csv",
        transition_matrix_rows,
        (
            list(transition_matrix_rows[0].keys())
            if transition_matrix_rows
            else ["original_class_id"]
        ),
    )
    overlap_rows = store.behavior_overlap_rows()
    overlap_fields = (
        list(overlap_rows[0].keys())
        if overlap_rows
        else [
            "signature",
            "video_id",
            "track_id",
            "left_review_id",
            "left_class_id",
            "left_behavior",
            "left_start_frame",
            "left_end_frame",
            "right_review_id",
            "right_class_id",
            "right_behavior",
            "right_start_frame",
            "right_end_frame",
            "overlap_start_frame",
            "overlap_end_frame",
            "overlap_frames",
            "same_class",
            "severity",
            "acknowledged",
            "acknowledged_by",
            "acknowledged_at",
        ]
    )
    _write_csv(
        table_dir / "behavior_overlap_review.csv",
        overlap_rows,
        overlap_fields,
    )

    per_video_dir = table_dir / "Per_Video_Bout_Tables"
    per_video_dir.mkdir(exist_ok=False)
    behavior_fields = (
        "Track ID",
        "Class ID",
        "Behavior",
        "Start Frame",
        "End Frame",
        "Duration (Frames)",
        "Duration (s)",
        "Review Decision",
        "Review ID",
        "Origin Prediction IDs",
        "Review Note",
    )
    for video in project.videos:
        video_behavior_bouts = sorted(
            (
                bout
                for bout in behavior_bouts
                if bout.video_id == video.video_id
            ),
            key=lambda bout: (
                bout.track_id,
                -1 if bout.class_id is None else bout.class_id,
                bout.start_frame,
                bout.end_frame,
            ),
        )
        _write_csv(
            per_video_dir / f"{video.video_stem}_reviewed_behavior_bouts.csv",
            [
                {
                    "Track ID": bout.track_id,
                    "Class ID": bout.class_id,
                    "Behavior": bout.label,
                    "Start Frame": bout.start_frame,
                    "End Frame": bout.end_frame,
                    "Duration (Frames)": bout.frames,
                    "Duration (s)": bout.frames / video.fps,
                    "Review Decision": bout.decision,
                    "Review ID": bout.review_id,
                    "Origin Prediction IDs": json.dumps(
                        bout.origin_prediction_ids
                    ),
                    "Review Note": bout.note,
                }
                for bout in video_behavior_bouts
            ],
            behavior_fields,
        )
    return behavior_frame_rows, len(overlap_rows)


def _write_export_index(
    output: Path,
    category_records: dict[str, dict[str, Any]],
) -> None:
    descriptions = {
        "ROI_Bouts": (
            "Concurrent ROI and exclusive ROI-X entry/exit bouts. The two "
            "semantics remain separate in every table and figure."
        ),
        "Object_Interactions": (
            "Object-interaction bouts, decisions, scores, dwell time, and "
            "entry/exit markers."
        ),
        "Behavior_Bouts": (
            "Class- and track-anchored behavior bouts, correction metrics, "
            "transitions, overlaps, and corrected positive frames."
        ),
    }
    lines = [
        "# IntegraPose Bout Review Export",
        "",
        "This export uses an organized mode-specific layout. Original "
        "IntegraPose inputs were not modified.",
        "",
        "## Where to look",
        "",
        "- `Shared_Audit/` contains the cross-mode action log and review-scope "
        "status.",
    ]
    for category_name, _event_kinds in EXPORT_CATEGORIES:
        if category_name in category_records:
            lines.append(
                f"- `{category_name}/Tables/` and `{category_name}/Figures/`: "
                f"{descriptions[category_name]}"
            )
        else:
            lines.append(
                f"- `{category_name}/` was not generated because this review "
                "contains no corresponding predictions or review rows."
            )
    lines.extend(
        [
            "",
            "## Figure definitions",
            "",
            "- **Counts:** original predicted bouts versus the active final "
            "reviewed reference. Ratio = reviewed / original.",
            "- **Dwell time:** original versus reviewed inclusive-frame "
            "duration converted to seconds using each video's FPS.",
            "- **Temporal IoU:** original predictions versus the final reviewed "
            "reference. A corrected-reference-versus-itself line is intentionally "
            "not produced.",
            "- **Boundary error:** weighted mean absolute start and end error "
            "for one-to-one matched events at tIoU 0.50 (or the only configured "
            "threshold when 0.50 is unavailable).",
            "",
            "Counts and dwell values are pooled only for visualization. The "
            "underlying `bout_count_and_dwell_summary.csv` retains one row per "
            "video, event kind, label/class, and track.",
            "",
            "Scores and figures are provisional wherever the corresponding "
            "review scope is not marked complete.",
        ]
    )
    _write_text(output / "EXPORT_INDEX.md", "\n".join(lines))


def export_review(
    project: ProjectData,
    store: ReviewStore,
    scores: Sequence[ScoreRow],
    *,
    event_iou_thresholds: Sequence[float],
) -> Path:
    output = _unique_output_dir(project.root)
    all_reviews = store.list_review_bouts(include_inactive=True)
    corrected = [
        bout
        for bout in all_reviews
        if bout.active and bout.decision in REFERENCE_DECISIONS
    ]
    predictions = store.list_predictions()
    behavior_bouts = [
        bout for bout in corrected if bout.event_kind == BEHAVIOR
    ]

    score_rows = [score.to_dict() for score in scores]
    score_fields = list(score_rows[0].keys()) if score_rows else ["scope"]

    shared_dir = output / "Shared_Audit"
    shared_dir.mkdir(exist_ok=False)
    scope_rows = store.scope_rows()
    _write_csv(
        shared_dir / "review_scope_status.csv",
        scope_rows,
        (
            "video_id",
            "event_kind",
            "track_id",
            "complete",
            "reviewer",
            "completed_at",
        ),
    )
    action_rows = store.action_rows()
    _write_csv(
        shared_dir / "review_audit_log.csv",
        action_rows,
        (
            "action_id",
            "action_at",
            "reviewer",
            "video_id",
            "action",
            "payload_json",
        ),
    )

    category_records: dict[str, dict[str, Any]] = {}
    behavior_frame_rows = 0
    behavior_overlap_pairs = 0
    for category_name, event_kinds in EXPORT_CATEGORIES:
        category_predictions = [
            bout for bout in predictions if bout.event_kind in event_kinds
        ]
        category_reviews = [
            bout for bout in all_reviews if bout.event_kind in event_kinds
        ]
        if not category_predictions and not category_reviews:
            continue
        category_corrected = [
            bout for bout in corrected if bout.event_kind in event_kinds
        ]
        category_score_rows = [
            row
            for row in score_rows
            if row.get("event_kind") in event_kinds
            and (
                int(row.get("predicted_events", 0))
                or int(row.get("reviewed_events", 0))
            )
        ]
        category_dir = output / category_name
        table_dir = category_dir / "Tables"
        category_dir.mkdir(exist_ok=False)
        table_dir.mkdir(exist_ok=False)

        _write_csv(
            table_dir / "original_predictions.csv",
            [bout.to_dict() for bout in category_predictions],
            PREDICTION_FIELDS,
        )
        _write_csv(
            table_dir / "review_decisions.csv",
            [_review_row(bout) for bout in category_reviews],
            REVIEW_FIELDS,
        )
        _write_csv(
            table_dir / "corrected_bouts.csv",
            [_review_row(bout) for bout in category_corrected],
            REVIEW_FIELDS,
        )
        _write_csv(
            table_dir / "prediction_vs_review_scores.csv",
            category_score_rows,
            score_fields,
        )
        summary_rows = _count_dwell_summary(
            project,
            store,
            category_predictions,
            category_corrected,
        )
        _write_csv(
            table_dir / "bout_count_and_dwell_summary.csv",
            summary_rows,
            SUMMARY_FIELDS,
        )

        marker_count = 0
        if category_name in {"ROI_Bouts", "Object_Interactions"}:
            marker_count = _write_spatial_per_video_tables(
                table_dir=table_dir,
                project=project,
                corrected=category_corrected,
                event_kinds=event_kinds,
            )
        elif category_name == "Behavior_Bouts":
            (
                behavior_frame_rows,
                behavior_overlap_pairs,
            ) = _write_behavior_specific_tables(
                table_dir=table_dir,
                project=project,
                store=store,
                behavior_bouts=category_corrected,
            )

        try:
            from .figures import generate_category_figures

            figure_paths = generate_category_figures(
                category_name=category_name,
                output_dir=category_dir / "Figures",
                project=project,
                summary_rows=summary_rows,
                score_rows=category_score_rows,
                event_iou_thresholds=event_iou_thresholds,
            )
        except Exception as exc:
            raise ReviewError(
                f"Could not generate {category_name} figures. "
                f"A partial export remains at {output}: {exc}"
            ) from exc

        category_records[category_name] = {
            "event_kinds": list(event_kinds),
            "prediction_bouts": len(category_predictions),
            "all_review_rows": len(category_reviews),
            "corrected_reference_bouts": len(category_corrected),
            "entry_exit_marker_rows": marker_count,
            "tables_directory": (
                table_dir.relative_to(output).as_posix()
            ),
            "figures": [
                path.relative_to(output).as_posix()
                for path in figure_paths
            ],
        }

    _write_export_index(output, category_records)

    output_files = sorted(
        path for path in output.rglob("*") if path.is_file()
    )
    manifest = {
        "schema_version": 4,
        "export_layout_version": 2,
        "application_version": APP_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_label": project.project_label,
        "session_id": project.session_id,
        "analysis_root": ".",
        "path_resolution_policy": (
            "Project-relative IntegraPose outputs are preferred; exact "
            "session-recorded external source videos are read-only fallbacks."
        ),
        "prediction_fingerprint_scheme": FINGERPRINT_SCHEME,
        "review_database": (
            str(store.database.name)
            if isinstance(store.database, Path)
            else str(store.database)
        ),
        "temporal_event_iou_thresholds": [
            float(value) for value in event_iou_thresholds
        ],
        "advanced_tiou_sweep": len(event_iou_thresholds) > 1,
        "definitions": {
            "interval_semantics": "inclusive start and end frames",
            "corrected_reference": (
                "active bouts with accepted, modified, or added decisions"
            ),
            "before_after_ratio": (
                "final reviewed reference value divided by the original "
                "prediction value; blank when the original denominator is zero"
            ),
            "event_match": (
                "same video, event kind, class ID, label, and track with "
                "inclusive-frame temporal interval IoU at least the configured "
                "threshold; greedy one-to-one matching"
            ),
            "tiou_figure": (
                "original predictions compared with the final reviewed "
                "reference; corrected reference bouts are not compared against "
                "themselves"
            ),
            "boundary_error_figure": (
                "matched-event-count-weighted mean absolute start and end "
                "error in seconds at tIoU 0.50, or the sole configured "
                "threshold when 0.50 is unavailable"
            ),
            "scope_complete": (
                "all original predicted bouts have a final manual decision; "
                "behavior completion is video-by-track and spatial completion "
                "is video-by-event-kind"
            ),
            "behavior_correct_ratio": (
                "unique accepted-unchanged predictions divided by reviewed "
                "original predictions; repeated actions do not inflate counts"
            ),
            "behavior_overlap": (
                "inclusive temporal overlap between two active behavior bouts "
                "on the same track; warnings do not prohibit legitimate "
                "multi-label co-occurrence"
            ),
        },
        "organized_export_categories": category_records,
        "counts": {
            "prediction_bouts": len(predictions),
            "all_review_rows": len(all_reviews),
            "corrected_reference_bouts": len(corrected),
            "corrected_behavior_bouts": len(behavior_bouts),
            "corrected_behavior_positive_frame_rows": behavior_frame_rows,
            "behavior_overlap_pairs": behavior_overlap_pairs,
            "audit_actions": len(action_rows),
        },
        "source_videos": [
            {
                "video_id": video.video_id,
                "video_stem": video.video_stem,
                "display_video": video.display_video_relative,
                "display_video_role": video.display_video_role,
                "fps": video.fps,
                "frame_count": video.frame_count,
                "single_animal_mode": video.single_animal_mode,
                "behavior_classes": {
                    str(class_id): name
                    for class_id, name in video.behavior_classes.items()
                },
                "behavior_bout_settings": video.behavior_settings,
                "prediction_source_fingerprint": video.source_fingerprint,
                "prediction_and_validation_sources": [
                    {
                        "path": source_path,
                        "sha256": video.source_file_hashes.get(source_path, ""),
                    }
                    for source_path in video.source_files
                ],
                "path_provenance": video.path_provenance,
            }
            for video in project.videos
        ],
        "files": [
            {
                "path": path.relative_to(output).as_posix(),
                "sha256": _sha256(path),
            }
            for path in output_files
        ],
        "limitations": [
            "Scores are final only for scopes marked complete.",
            "Manual correction is treated as the review reference, not an independent blinded ground truth unless the study protocol establishes that process.",
            "Frame-level scoring evaluates interval occupancy and does not measure pose-keypoint localization.",
            "Model–reviewer Cohen kappa is a binary behavior-channel agreement statistic, not human inter-rater reliability.",
            "Potential overlapping behaviors may be legitimate co-occurrences and require experiment-specific interpretation.",
            "Batch-pooled count and dwell figures are descriptive; the accompanying summary CSV preserves video-level analytical units.",
        ],
    }
    _write_json(output / "review_export_manifest.json", manifest)
    return output
