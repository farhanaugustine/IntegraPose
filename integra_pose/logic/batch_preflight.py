"""Pure helpers for batch preflight planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Sequence

from integra_pose.logic.analytics_metric_catalog import AnalyticsMetricSpec
from integra_pose.logic.batch_design import parse_time_point_numeric
from integra_pose.utils.frame_identity import FrameIdentityError, resolve_frame_label_indices

PreflightRow = dict[str, str]
IndexLabelsFn = Callable[[Path], list[tuple[Path, list[str]]]]
FindLabelsFn = Callable[..., Path | None]
MIN_GROUP_REPLICATES = 2
MIN_KPSS_TIMEPOINTS = 5
MIN_REPEATED_SUBJECTS_FOR_MIXED = 2


@dataclass(slots=True)
class PreflightVideoState:
    video_id: str
    video_name: str
    video_path: str
    has_rois: bool
    has_object_rois: bool
    group: str = ""
    subject_id: str = ""
    time_point: str = ""
    metadata_sources: dict[str, str] = field(default_factory=dict)
    metadata_warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AnalysisPreflightConfig:
    roi_strategy: str
    use_existing_labels: bool
    model_path: str
    labels_root: str
    object_interaction_enabled: bool
    object_count: int
    include_kpss: bool
    class_count: int
    enabled_metrics: set[str]
    metric_specs: Sequence[AnalyticsMetricSpec]
    include_mixed_effects: bool = True
    auto_detect_design: bool = True
    categorical_factors: Sequence[str] = ()
    model_preflight_error: str = ""


def _normalized_text(value: Any) -> str:
    return str(value or "").strip()


def _summarize_study_design(videos: Sequence[PreflightVideoState]) -> dict[str, Any]:
    group_sizes: dict[str, int] = {}
    group_units: dict[str, set[str]] = {}
    subject_sizes: dict[str, int] = {}
    subject_time_levels: dict[str, set[str]] = {}
    design_cell_sizes: dict[tuple[str, str, str], int] = {}
    time_levels: set[str] = set()
    numeric_time_levels: set[float] = set()
    group_numeric_time_levels: dict[str, set[float]] = {}
    inferred_field_counts = {"group": 0, "subject_id": 0, "time_point": 0}
    metadata_warning_count = 0
    missing_group_videos = 0
    missing_subject_videos = 0
    missing_time_videos = 0

    for video in videos:
        group = _normalized_text(getattr(video, "group", ""))
        subject_id = _normalized_text(getattr(video, "subject_id", ""))
        time_point = _normalized_text(getattr(video, "time_point", ""))
        video_id = _normalized_text(getattr(video, "video_id", "")) or _normalized_text(
            getattr(video, "video_name", "")
        )
        metadata_sources = dict(getattr(video, "metadata_sources", {}) or {})
        metadata_warning_count += len(list(getattr(video, "metadata_warnings", []) or []))
        for field_name in inferred_field_counts:
            source = _normalized_text(metadata_sources.get(field_name, ""))
            if source and source != "manual":
                inferred_field_counts[field_name] += 1

        if group:
            group_sizes[group] = group_sizes.get(group, 0) + 1
            independent_unit = (
                f"{group}\x1f{subject_id}"
                if subject_id
                else f"{group}\x1f__video__:{video_id}"
            )
            group_units.setdefault(group, set()).add(independent_unit)
        else:
            missing_group_videos += 1

        if subject_id:
            subject_key = f"{group}\x1f{subject_id}" if group else subject_id
            subject_sizes[subject_key] = subject_sizes.get(subject_key, 0) + 1
        else:
            subject_key = ""
            missing_subject_videos += 1

        if time_point:
            time_levels.add(time_point)
            if subject_key:
                subject_time_levels.setdefault(subject_key, set()).add(time_point)
            numeric_time = parse_time_point_numeric(time_point)
            if numeric_time is not None:
                numeric_time_levels.add(float(numeric_time))
                if group:
                    group_numeric_time_levels.setdefault(group, set()).add(
                        float(numeric_time)
                    )
        else:
            missing_time_videos += 1
        if group or subject_id or time_point:
            design_key = (group, subject_id, time_point)
            design_cell_sizes[design_key] = design_cell_sizes.get(design_key, 0) + 1

    subjects_with_repeats = sum(1 for count in subject_sizes.values() if count >= 2)
    subjects_with_multiple_timepoints = sum(1 for times in subject_time_levels.values() if len(times) >= 2)
    labeled_group_videos = sum(group_sizes.values())
    group_unit_sizes = {group: len(units) for group, units in group_units.items()}
    repeated_design_cells = {
        key: count
        for key, count in design_cell_sizes.items()
        if count > 1 and all(key)
    }

    return {
        "group_sizes": group_sizes,
        "group_unit_sizes": group_unit_sizes,
        "group_levels": len(group_sizes),
        "labeled_group_videos": labeled_group_videos,
        "missing_group_videos": missing_group_videos,
        "subject_sizes": subject_sizes,
        "subject_count": len(subject_sizes),
        "subjects_with_repeats": subjects_with_repeats,
        "subjects_with_multiple_timepoints": subjects_with_multiple_timepoints,
        "missing_subject_videos": missing_subject_videos,
        "time_levels": len(time_levels),
        "numeric_time_levels": len(numeric_time_levels),
        "group_numeric_time_levels": {
            group: len(values)
            for group, values in group_numeric_time_levels.items()
        },
        "missing_time_videos": missing_time_videos,
        "repeated_design_cells": repeated_design_cells,
        "inferred_field_counts": inferred_field_counts,
        "metadata_warning_count": metadata_warning_count,
    }


def summarize_study_design(videos: Sequence[PreflightVideoState]) -> dict[str, Any]:
    """Public study-design summary shared by preflight and the wizard UI."""

    return _summarize_study_design(videos)


def _default_index_label_dirs(root_dir: Path) -> list[tuple[Path, list[str]]]:
    grouped: dict[Path, list[str]] = {}
    if not root_dir.exists() or not root_dir.is_dir():
        return []
    for txt_file in root_dir.rglob("*.txt"):
        parent = txt_file.parent
        grouped.setdefault(parent, []).append(txt_file.name.lower())
    rows = []
    for parent, names in grouped.items():
        try:
            resolved = resolve_frame_label_indices(names)
        except FrameIdentityError:
            continue
        if resolved:
            rows.append((parent, sorted(resolved)))
    return rows


def _default_find_labels_dir(
    *,
    video_stem: str,
    preferred_dir: str | None,
    labels_root: Path | None,
    indexed_dirs: list[tuple[Path, list[str]]] | None,
) -> Path | None:
    stem = str(video_stem or "").strip().lower()
    if not stem:
        return None

    def _source_filename_match(name: str) -> bool:
        file_stem = Path(name).stem.casefold()
        return bool(
            file_stem == stem
            or re.match(rf"^{re.escape(stem)}__(?:frame|frm|image|img)_?\d+$", file_stem)
            or re.match(rf"^{re.escape(stem)}_(?:frame|frm|image|img)_?\d+$", file_stem)
            or re.match(rf"^{re.escape(stem)}_\d+$", file_stem)
        )

    def _dir_has_frames(path: Path, *, require_source_match: bool) -> bool:
        if not path.exists() or not path.is_dir():
            return False
        try:
            names = [item.name for item in path.glob("*.txt") if not item.name.startswith(".")]
            if require_source_match:
                names = [name for name in names if _source_filename_match(name)]
            return bool(resolve_frame_label_indices(names, source=video_stem))
        except Exception:
            return False

    preferred = Path(str(preferred_dir or "").strip()).expanduser() if preferred_dir else None
    if preferred and _dir_has_frames(preferred, require_source_match=False):
        return preferred.resolve()

    if labels_root is not None:
        direct_candidates = [labels_root / video_stem, labels_root / video_stem / "labels", labels_root / f"{video_stem}_labels"]
        for candidate in direct_candidates:
            if _dir_has_frames(candidate, require_source_match=True):
                return candidate.resolve()

    if not indexed_dirs:
        return None

    prefixes = (f"{stem}_frame", f"{stem}__frame")
    candidates: list[tuple[int, str, Path]] = []
    for parent, names in indexed_dirs:
        matching_names = [name for name in names if _source_filename_match(name)]
        prefix_hit = any(name.startswith(prefixes) for name in matching_names)
        stem_hit = bool(matching_names)
        parent_name = parent.name.lower()
        parent_hit = parent_name in {stem, f"{stem}_labels"}
        # Avoid false positives by requiring a concrete video-stem signal.
        if not (prefix_hit or stem_hit or parent_hit):
            continue
        score = 0
        if prefix_hit:
            score += 120
        if stem_hit:
            score += 60
        if parent_hit:
            score += 25
        if parent_name == "labels":
            score += 10
        score += min(len(matching_names), 20)
        candidates.append((score, str(parent).casefold(), parent))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        return None
    return candidates[0][2].resolve()


def _roi_ready_counts(videos: Sequence[PreflightVideoState], *, strategy: str, shared_has_rois: bool) -> tuple[int, int]:
    total = len(videos)
    if total <= 0:
        return 0, 0
    if strategy == "single":
        return (total if shared_has_rois else 0), total
    ready = sum(1 for video in videos if bool(video.has_rois))
    return ready, total


def _object_roi_ready_counts(
    videos: Sequence[PreflightVideoState],
    *,
    strategy: str,
    shared_has_object_rois: bool,
) -> tuple[int, int]:
    total = len(videos)
    if total <= 0:
        return 0, 0
    # Object placements may vary independently of the arena-ROI strategy.
    # A shared object set is a fallback for every video; otherwise each
    # video's own object ROI store determines readiness.
    if shared_has_object_rois:
        return total, total
    ready = sum(1 for video in videos if bool(video.has_object_rois))
    return ready, total


def _missing_roi_videos(
    videos: Sequence[PreflightVideoState],
    *,
    strategy: str,
) -> list[PreflightVideoState]:
    if str(strategy or "").strip() == "single":
        return []
    return [video for video in videos if not bool(video.has_rois)]


def _missing_object_roi_videos(
    videos: Sequence[PreflightVideoState],
    *,
    strategy: str,
    shared_has_object_rois: bool = False,
) -> list[PreflightVideoState]:
    if bool(shared_has_object_rois):
        return []
    return [video for video in videos if not bool(video.has_object_rois)]


def _format_preflight_video_scope(video: PreflightVideoState) -> str:
    video_id = _normalized_text(getattr(video, "video_id", ""))
    video_name = _normalized_text(getattr(video, "video_name", ""))
    if video_id and video_name:
        return f"{video_id} | {video_name}"
    return video_id or video_name or Path(str(getattr(video, "video_path", "") or "")).name or "Unknown video"


def _format_preflight_video_context(video: PreflightVideoState) -> str:
    group = _normalized_text(getattr(video, "group", "")) or "-"
    subject_id = _normalized_text(getattr(video, "subject_id", "")) or "-"
    time_point = _normalized_text(getattr(video, "time_point", "")) or "-"
    return f"Group={group} | Subject={subject_id} | Time={time_point}"


def _existing_labels_ready_counts(
    videos: Sequence[PreflightVideoState],
    labels_root: str,
    *,
    index_label_dirs_fn: IndexLabelsFn,
    find_labels_dir_fn: FindLabelsFn,
) -> tuple[int, int, bool, Path | None]:
    total = len(videos)
    root_text = str(labels_root or "").strip()
    if total <= 0 or not root_text:
        return 0, total, False, None
    root_path = Path(root_text).expanduser()
    if not root_path.exists() or (not root_path.is_dir()):
        return 0, total, False, root_path
    try:
        indexed = index_label_dirs_fn(root_path)
    except Exception:
        return 0, total, True, root_path
    ready = 0
    for video in videos:
        stem = Path(str(video.video_path or "")).stem
        found = find_labels_dir_fn(
            video_stem=stem,
            preferred_dir=None,
            labels_root=root_path,
            indexed_dirs=indexed,
        )
        if found is not None:
            ready += 1
    return ready, total, True, root_path


def build_analysis_preflight_rows(
    config: AnalysisPreflightConfig,
    videos: Sequence[PreflightVideoState],
    *,
    shared_has_rois: bool = False,
    shared_has_object_rois: bool = False,
    index_label_dirs_fn: IndexLabelsFn | None = None,
    find_labels_dir_fn: FindLabelsFn | None = None,
) -> list[PreflightRow]:
    rows: list[PreflightRow] = []
    strategy = str(config.roi_strategy or "single").strip() or "single"
    queue_total = len(videos)
    use_existing_labels = bool(config.use_existing_labels)
    model_path = str(config.model_path or "").strip()
    model_path_resolved = Path(model_path).expanduser() if model_path else None
    model_path_exists = bool(model_path_resolved and model_path_resolved.is_file())
    model_preflight_error = _normalized_text(config.model_preflight_error)
    model_ready = bool(model_path_exists and not model_preflight_error)
    object_count = max(0, int(config.object_count or 0))
    enabled_metrics = {str(name).strip() for name in (config.enabled_metrics or set()) if str(name).strip()}

    index_fn = index_label_dirs_fn or _default_index_label_dirs
    find_fn = find_labels_dir_fn or _default_find_labels_dir
    labels_ready, labels_total, labels_root_valid, labels_root_path = _existing_labels_ready_counts(
        videos,
        str(config.labels_root or ""),
        index_label_dirs_fn=index_fn,
        find_labels_dir_fn=find_fn,
    )

    if use_existing_labels:
        inference_run = "No"
        inference_scope = "-"
        inference_reason = "Using saved label files; model inference is skipped."
    elif model_preflight_error:
        inference_run = "No"
        inference_scope = "-"
        inference_reason = f"Model preflight failed: {model_preflight_error}"
    elif model_path and model_ready:
        inference_run = "Yes"
        inference_scope = f"{queue_total} video(s)"
        inference_reason = "Model file found and ready."
    elif model_path and (not model_path_exists):
        inference_run = "No"
        inference_scope = "-"
        inference_reason = f"Model file not found: {model_path_resolved}"
    else:
        inference_run = "No"
        inference_scope = "-"
        inference_reason = "No model file selected."
    rows.append(
        {
            "analysis": "Run model inference",
            "will_run": inference_run,
            "scope": inference_scope,
            "variables": "Animal detections, tracks, body keypoints",
            "reason": inference_reason,
        }
    )

    if use_existing_labels and labels_root_valid and labels_ready >= labels_total and labels_total > 0:
        labels_run = "Yes"
        labels_reason = f"Found usable labels for all queued videos ({labels_ready}/{labels_total})."
    elif use_existing_labels and labels_root_valid and labels_total > 0 and labels_ready > 0:
        labels_run = "Partial"
        labels_reason = f"Found labels for {labels_ready}/{labels_total}; missing labels for the rest."
    elif use_existing_labels and labels_root_valid and labels_total > 0:
        labels_run = "No"
        labels_reason = "No matching label folders were found for queued videos."
    elif use_existing_labels and labels_root_valid:
        labels_run = "Yes"
        labels_reason = "Label folder is valid."
    elif use_existing_labels and not str(config.labels_root or "").strip():
        labels_run = "No"
        labels_reason = "Existing-label mode is on, but no labels folder was selected."
    elif use_existing_labels:
        labels_run = "No"
        labels_reason = f"Labels folder is invalid: {labels_root_path}"
    else:
        labels_run = "No"
        labels_reason = "Not enabled."
    rows.append(
        {
            "analysis": "Reuse existing label files",
            "will_run": labels_run,
            "scope": f"{labels_ready}/{labels_total} video(s)" if use_existing_labels and labels_total > 0 else "-",
            "variables": "Precomputed frame label text files",
            "reason": labels_reason,
        }
    )

    prerequisites_ok = queue_total > 0 and (
        (use_existing_labels and labels_root_valid and labels_ready > 0)
        or ((not use_existing_labels) and model_ready)
    )
    if prerequisites_ok:
        if use_existing_labels and labels_total > 0 and labels_ready < labels_total:
            core_run = "Partial"
            core_reason = (
                f"Behavior summaries can run for {labels_ready}/{labels_total} videos; "
                "missing-label videos will fail unless labels are added."
            )
        else:
            core_run = "Yes"
            core_reason = "Ready to compute behavior bouts and summary tables."
    elif queue_total <= 0:
        core_run = "No"
        core_reason = "No videos discovered."
    else:
        core_run = "No"
        core_reason = "Complete model/labels setup first."
    rows.append(
        {
            "analysis": "Behavior bout summaries",
            "will_run": core_run,
            "scope": f"{queue_total} video(s)" if core_run in {"Yes", "Partial"} else "-",
            "variables": "Behavior starts/stops, durations, event counts",
            "reason": core_reason,
        }
    )

    roi_ready, roi_total = _roi_ready_counts(videos, strategy=strategy, shared_has_rois=bool(shared_has_rois))
    if roi_ready <= 0:
        roi_run = "No"
        roi_reason = "No arena ROIs assigned yet."
    elif roi_ready < roi_total:
        roi_run = "Partial"
        roi_reason = f"ROI-ready videos: {roi_ready}/{roi_total}; others use full-video summaries."
    else:
        roi_run = "Yes"
        roi_reason = "Arena ROI summaries are ready for all videos."
    if core_run == "No":
        roi_run = "No"
        roi_reason = "Behavior bout setup is incomplete."
    rows.append(
        {
            "analysis": "Arena ROI summaries",
            "will_run": roi_run,
            "scope": f"{roi_ready}/{roi_total} video(s)" if roi_total > 0 else "-",
            "variables": "Entries/exits, dwell time, ROI transitions",
            "reason": roi_reason,
        }
    )
    missing_rois = _missing_roi_videos(videos, strategy=strategy)
    if str(strategy).strip() == "single" and queue_total > 0 and not bool(shared_has_rois):
        rows.append(
            {
                "analysis": "Missing arena ROI setup",
                "will_run": "Fix",
                "scope": "Shared ROI set",
                "variables": f"Applies to all {queue_total} queued video(s)",
                "reason": "Draw the shared arena ROI set once; it will be reused across the full batch.",
            }
        )
    elif missing_rois:
        for video in missing_rois:
            rows.append(
                {
                    "analysis": "Missing arena ROI",
                    "will_run": "Fix",
                    "scope": _format_preflight_video_scope(video),
                    "variables": _format_preflight_video_context(video),
                    "reason": "Draw arena ROI(s) for this video before batch run.",
                }
            )

    object_ready, object_total = _object_roi_ready_counts(
        videos,
        strategy=strategy,
        shared_has_object_rois=bool(shared_has_object_rois),
    )
    if not bool(config.object_interaction_enabled):
        object_run = "No"
        object_reason = "Disabled in wizard."
    elif object_count <= 0:
        object_run = "No"
        object_reason = "Object count must be greater than 0."
    elif object_ready <= 0:
        object_run = "No"
        object_reason = "No shared or per-video object ROIs are assigned."
    elif object_ready < object_total:
        object_run = "No"
        object_reason = f"Object ROIs are missing for some videos ({object_ready}/{object_total})."
    else:
        object_run = "Yes"
        object_reason = "Object interaction summaries are ready."
    if core_run == "No" and object_run != "No":
        object_run = "No"
        object_reason = "Behavior bout setup is incomplete."
    rows.append(
        {
            "analysis": "Object interaction summaries",
            "will_run": object_run,
            "scope": f"{object_ready}/{object_total} video(s)" if object_total > 0 else "-",
            "variables": "Object entries/exits, interaction time, proximity traces",
            "reason": object_reason,
        }
    )
    missing_object_rois = _missing_object_roi_videos(
        videos,
        strategy=strategy,
        shared_has_object_rois=bool(shared_has_object_rois),
    )
    if bool(config.object_interaction_enabled) and object_count > 0:
        if (
            str(strategy).strip() == "single"
            and queue_total > 0
            and object_ready <= 0
            and not bool(shared_has_object_rois)
        ):
            rows.append(
                {
                    "analysis": "Missing object ROI setup",
                    "will_run": "Fix",
                    "scope": "Shared object ROI set",
                    "variables": f"Applies to all {queue_total} queued video(s)",
                    "reason": (
                        "Place a shared object ROI set once, or use Place Objects "
                        "Across Queue for per-video placements."
                    ),
                }
            )
        elif missing_object_rois:
            for video in missing_object_rois:
                rows.append(
                    {
                        "analysis": "Missing object ROI",
                        "will_run": "Fix",
                        "scope": _format_preflight_video_scope(video),
                        "variables": _format_preflight_video_context(video),
                        "reason": "Place object ROI(s) for this video before object interaction analytics.",
                    }
                )

    for metric_spec in (config.metric_specs or ()):
        metric_key = str(getattr(metric_spec, "key", "") or "").strip()
        metric_label = str(getattr(metric_spec, "label", metric_key) or metric_key)
        metric_variables = str(getattr(metric_spec, "variables", "") or "")
        metric_on = metric_key in enabled_metrics
        if not metric_on:
            run_state = "No"
            scope = "-"
            reason = "Disabled in analytics settings."
        elif core_run == "No":
            run_state = "No"
            scope = "-"
            reason = "Behavior bout setup is incomplete."
        elif bool(getattr(metric_spec, "requires_rois", False)) and roi_ready <= 0:
            run_state = "No"
            scope = "-"
            reason = "Arena ROIs are required for this metric."
        elif bool(getattr(metric_spec, "requires_rois", False)) and roi_ready < roi_total:
            run_state = "Partial"
            scope = f"{roi_ready}/{roi_total} video(s)"
            reason = "Only videos with arena ROIs will produce this metric."
        elif bool(getattr(metric_spec, "requires_object_interactions", False)) and object_run != "Yes":
            if object_run == "Partial":
                run_state = "Partial"
                scope = f"{object_ready}/{object_total} video(s)" if object_total > 0 else "-"
                reason = "Only videos with valid object interaction setup will produce this metric."
            else:
                run_state = "No"
                scope = "-"
                reason = "Object interaction setup is required for this metric."
        elif bool(getattr(metric_spec, "requires_multiclass_behaviors", False)) and int(config.class_count or 0) <= 1:
            run_state = "No"
            scope = "-"
            reason = "This metric requires at least two behavior classes."
        elif bool(getattr(metric_spec, "requires_multi_animal", False)):
            run_state = "Partial"
            scope = f"{queue_total} video(s)" if queue_total > 0 else "-"
            reason = "This metric only yields outputs when labels contain multiple concurrent tracked animals."
        else:
            run_state = "Yes"
            scope = f"{queue_total} video(s)"
            reason = "Enabled and ready."
        rows.append(
            {
                "analysis": metric_label,
                "will_run": run_state,
                "scope": scope,
                "variables": metric_variables,
                "reason": reason,
            }
        )

    design = _summarize_study_design(videos)
    missing_group_videos = int(design["missing_group_videos"])
    missing_subject_videos = int(design["missing_subject_videos"])
    missing_time_videos = int(design["missing_time_videos"])
    metadata_warning_count = int(design["metadata_warning_count"])
    inferred_counts = dict(design["inferred_field_counts"])
    missing_total = (
        missing_group_videos
        + missing_subject_videos
        + missing_time_videos
    )
    if queue_total <= 0:
        metadata_run = "No"
        metadata_scope = "-"
        metadata_reason = "No queued videos are available for study-design discovery."
    elif missing_total <= 0 and metadata_warning_count <= 0:
        metadata_run = "Yes"
        metadata_scope = f"{queue_total}/{queue_total} video(s)"
        metadata_reason = (
            "Group is assigned as the comparison factor, Subject ID as the "
            "experimental/repeated unit, and Time Point as the time factor."
        )
    else:
        metadata_run = "Partial"
        metadata_scope = f"{queue_total} video(s)"
        metadata_reason = (
            f"Missing labels: group={missing_group_videos}, "
            f"subject={missing_subject_videos}, time={missing_time_videos}."
        )
        if metadata_warning_count:
            metadata_reason += (
                f" {metadata_warning_count} ambiguous metadata inference "
                "warning(s) require manual review."
            )
    if queue_total > 0:
        metadata_reason += (
            " Auto-detected values: "
            f"group={int(inferred_counts.get('group', 0))}, "
            f"subject={int(inferred_counts.get('subject_id', 0))}, "
            f"time={int(inferred_counts.get('time_point', 0))}."
        )
        if not bool(config.auto_detect_design):
            metadata_reason += (
                " Automatic addition of discovered categorical factors to "
                "group statistics is disabled."
            )
    rows.append(
        {
            "analysis": "Study-design metadata",
            "will_run": metadata_run,
            "scope": metadata_scope,
            "variables": "Group=factor | Subject ID=experimental unit | Time Point=time/repeated factor",
            "reason": metadata_reason,
        }
    )

    missing_field_specs = {
        "group": (
            "Group",
            "Assign Group before group comparisons.",
        ),
        "subject_id": (
            "Subject ID",
            (
                "Assign Subject ID so repeated videos are collapsed to the "
                "correct experimental unit and mixed-effects models are valid."
            ),
        ),
        "time_point": (
            "Time Point",
            (
                "Assign Time Point before time-course, repeated-measures, or "
                "KPSS analyses."
            ),
        ),
    }
    for video in videos:
        missing_fields = [
            field_name
            for field_name in missing_field_specs
            if not _normalized_text(getattr(video, field_name, ""))
        ]
        if not missing_fields:
            continue
        missing_labels = [
            missing_field_specs[field_name][0]
            for field_name in missing_fields
        ]
        next_actions = [
            missing_field_specs[field_name][1]
            for field_name in missing_fields
        ]
        rows.append(
            {
                "analysis": (
                    f"Missing {missing_labels[0]}"
                    if len(missing_labels) == 1
                    else "Missing design metadata"
                ),
                "will_run": "Fix",
                "scope": _format_preflight_video_scope(video),
                "variables": (
                    _format_preflight_video_context(video)
                    + " | Missing="
                    + ", ".join(missing_labels)
                ),
                "reason": (
                    "No confident value was discovered from the filename or "
                    "folder structure. "
                    + " ".join(next_actions)
                    + " Per-video analytics can still run."
                ),
            }
        )
    for video in videos:
        for warning in list(getattr(video, "metadata_warnings", []) or []):
            rows.append(
                {
                    "analysis": "Ambiguous design metadata",
                    "will_run": "Fix",
                    "scope": _format_preflight_video_scope(video),
                    "variables": _format_preflight_video_context(video),
                    "reason": str(warning),
                }
            )

    repeated_design_cells = dict(design["repeated_design_cells"])
    for (group, subject_id, time_point), count in sorted(
        repeated_design_cells.items()
    ):
        rows.append(
            {
                "analysis": "Repeated design cell",
                "will_run": "Partial",
                "scope": f"{count} video(s)",
                "variables": (
                    f"Group={group} | Subject={subject_id} | Time={time_point}"
                ),
                "reason": (
                    "These videos share one subject/time design cell and will be "
                    "averaged as technical/session replicates for inferential statistics."
                ),
            }
        )

    factor_aliases = {
        "group": "group",
        "grp": "group",
        "cohort": "group",
        "condition": "group",
        "treatment": "group",
        "time": "time_point",
        "timepoint": "time_point",
        "time_point": "time_point",
        "visit": "time_point",
        "subject": "subject_id",
        "subjectid": "subject_id",
        "subject_id": "subject_id",
        "animal": "subject_id",
        "animal_id": "subject_id",
    }
    for raw_factor in list(config.categorical_factors or ()):
        raw_text = _normalized_text(raw_factor)
        if not raw_text:
            continue
        normalized_factor = re.sub(r"[\s\-]+", "_", raw_text.casefold())
        canonical = factor_aliases.get(normalized_factor, "")
        if canonical == "subject_id":
            assigned_count = queue_total - missing_subject_videos
            if assigned_count <= 0:
                factor_state = "Fix"
                factor_reason = (
                    "No Subject IDs are assigned. Subject ID is the experimental "
                    "unit, not an independent categorical comparison factor."
                )
            elif assigned_count < queue_total:
                factor_state = "Partial"
                factor_reason = (
                    f"Subject ID is available for {assigned_count}/{queue_total} "
                    "video(s) and will be used as the experimental unit where present."
                )
            else:
                factor_state = "Yes"
                factor_reason = (
                    "Subject ID is assigned automatically as the experimental "
                    "unit; it is not treated as an independent categorical factor."
                )
            rows.append(
                {
                    "analysis": "Subject ID assignment",
                    "will_run": factor_state,
                    "scope": (
                        f"{assigned_count}/{queue_total} video(s) | "
                        f"{int(design['subject_count'])} subject(s)"
                    ),
                    "variables": "Experimental unit / mixed-model grouping variable",
                    "reason": factor_reason,
                }
            )
        elif canonical in {"group", "time_point"}:
            if canonical == "group":
                assigned_count = queue_total - missing_group_videos
                level_count = int(design["group_levels"])
            else:
                assigned_count = queue_total - missing_time_videos
                level_count = int(design["time_levels"])
            if assigned_count <= 0:
                factor_state = "Fix"
                factor_reason = f"No {canonical} values are assigned."
            elif level_count < 2:
                factor_state = "Partial"
                factor_reason = (
                    f"Only {level_count} non-empty level is available; at least "
                    "two levels are needed for a comparison."
                )
            elif assigned_count < queue_total:
                factor_state = "Partial"
                factor_reason = (
                    f"Recognized for {assigned_count}/{queue_total} video(s); "
                    "videos with missing values cannot contribute to this factor."
                )
            else:
                factor_state = "Yes"
                factor_reason = "Recognized as a built-in batch design variable."
            rows.append(
                {
                    "analysis": f"Configured factor: {canonical}",
                    "will_run": factor_state,
                    "scope": (
                        f"{assigned_count}/{queue_total} video(s) | "
                        f"{level_count} level(s)"
                    ),
                    "variables": canonical,
                    "reason": factor_reason,
                }
            )
        else:
            rows.append(
                {
                    "analysis": "Unavailable categorical factor",
                    "will_run": "Fix",
                    "scope": raw_text,
                    "variables": "Available: group, subject_id, time_point",
                    "reason": (
                        "This column is not present in the batch metadata table. "
                        "Remove it or assign it through a supported design field before running."
                    ),
                }
            )

    group_sizes = dict(design["group_sizes"])
    group_unit_sizes = dict(design["group_unit_sizes"])
    group_levels = int(design["group_levels"])
    labeled_group_videos = int(design["labeled_group_videos"])
    min_group_size = min(group_unit_sizes.values()) if group_unit_sizes else 0

    if core_run == "No":
        group_run = "No"
        group_reason = "Behavior bout setup is incomplete."
        group_scope = "-"
    elif queue_total <= 0:
        group_run = "No"
        group_reason = "No queued videos."
        group_scope = "-"
    elif group_levels < 2:
        group_run = "No"
        group_reason = "Assign at least two non-empty group labels before running inferential group statistics."
        group_scope = "-"
    elif min_group_size < MIN_GROUP_REPLICATES:
        group_run = "No"
        group_reason = (
            f"Each group needs at least {MIN_GROUP_REPLICATES} independent "
            f"subject/video units. Current smallest group has {min_group_size}."
        )
        group_scope = "-"
    else:
        contributing_videos = labeled_group_videos
        if core_run == "Partial" or missing_group_videos > 0:
            group_run = "Partial"
            exclusions: list[str] = []
            if core_run == "Partial":
                exclusions.append("videos missing reusable labels will be excluded")
            if missing_group_videos > 0:
                exclusions.append(f"{missing_group_videos} video(s) are missing a group label")
            group_reason = (
                f"Inferential group tests are ready for the labeled subset ({contributing_videos}/{queue_total} videos); "
                + "; ".join(exclusions)
                + "."
            )
        else:
            group_run = "Yes"
            group_reason = "Inferential group tests are ready with replicated groups."
        group_scope = (
            f"{contributing_videos}/{queue_total} labeled video(s)"
            if contributing_videos != queue_total
            else f"{queue_total} video(s)"
        )
    if core_run != "No" and int(config.class_count or 0) <= 1:
        group_reason += " Single-class model: behavior-like factors may be skipped."
    rows.append(
        {
            "analysis": "Group comparison statistics",
            "will_run": group_run,
            "scope": group_scope,
            "variables": "Kruskal-Wallis omnibus, Mann-Whitney pairwise, and effect sizes",
            "reason": group_reason,
        }
    )

    subjects_with_repeats = int(design["subjects_with_repeats"])
    subjects_with_multiple_timepoints = int(design["subjects_with_multiple_timepoints"])
    subject_count = int(design["subject_count"])
    numeric_time_levels = int(design["numeric_time_levels"])
    group_numeric_time_levels = dict(design["group_numeric_time_levels"])
    time_levels = int(design["time_levels"])

    mixed_ready = subjects_with_repeats >= MIN_REPEATED_SUBJECTS_FOR_MIXED and (
        group_levels >= 2 or subjects_with_multiple_timepoints >= MIN_REPEATED_SUBJECTS_FOR_MIXED
    )
    kpss_ready_groups = [
        group
        for group, level_count in group_numeric_time_levels.items()
        if int(level_count) >= MIN_KPSS_TIMEPOINTS
    ]
    if not group_numeric_time_levels and numeric_time_levels >= MIN_KPSS_TIMEPOINTS:
        kpss_ready_groups = ["all"]
    kpss_ready = bool(kpss_ready_groups)

    if not bool(config.include_mixed_effects):
        mixed_run = "No"
        mixed_reason = "Disabled in Advanced Statistics."
        mixed_scope = "-"
    elif core_run == "No":
        mixed_run = "No"
        mixed_reason = "Behavior bout setup is incomplete."
        mixed_scope = "-"
    elif mixed_ready:
        mixed_run = "Yes" if core_run == "Yes" else "Partial"
        mixed_reason = (
            "Repeated observations are available for at least "
            f"{subjects_with_repeats} subject(s); Subject ID will be used as "
            "the random grouping variable."
        )
        if core_run == "Partial":
            mixed_reason += " Only videos with completed core analytics contribute."
        mixed_scope = f"{subject_count} subject(s)"
    else:
        mixed_run = "No"
        reasons: list[str] = []
        if subject_count < MIN_REPEATED_SUBJECTS_FOR_MIXED:
            reasons.append("assign Subject IDs for at least 2 subjects")
        elif subjects_with_repeats < MIN_REPEATED_SUBJECTS_FOR_MIXED:
            reasons.append("at least 2 subjects need repeated observations")
        elif time_levels < 2 and group_levels < 2:
            reasons.append("assign at least 2 groups or 2 time points")
        else:
            reasons.append("repeated-subject design metadata is incomplete")
        mixed_reason = "; ".join(reasons).capitalize() + "."
        mixed_scope = "-"
    rows.append(
        {
            "analysis": "Mixed-effects models",
            "will_run": mixed_run,
            "scope": mixed_scope,
            "variables": "Group, Time Point, and Subject ID random grouping",
            "reason": mixed_reason,
        }
    )

    if not bool(config.include_kpss):
        kpss_run = "No"
        kpss_reason = "Disabled in Advanced Statistics (recommended for most routine batches)."
        kpss_scope = "-"
    elif core_run == "No":
        kpss_run = "No"
        kpss_reason = "Behavior bout setup is incomplete."
        kpss_scope = "-"
    elif kpss_ready:
        kpss_run = "Yes" if core_run == "Yes" else "Partial"
        kpss_reason = (
            "Enough ordered time points were discovered. Labels such as Day7, "
            "Week2, and Hour24 are converted to an ordered day-scale value."
        )
        if core_run == "Partial":
            kpss_reason += " Only videos with completed core analytics contribute."
        kpss_scope = ", ".join(kpss_ready_groups)
    else:
        kpss_run = "No"
        kpss_reason = (
            f"KPSS needs at least {MIN_KPSS_TIMEPOINTS} ordered time points "
            "within a group. Assign numeric, Day, Week, Hour, or Visit labels."
        )
        kpss_scope = "-"
    rows.append(
        {
            "analysis": "KPSS stationarity diagnostic",
            "will_run": kpss_run,
            "scope": kpss_scope,
            "variables": "Ordered Time Point series within each group",
            "reason": kpss_reason,
        }
    )
    return rows


def summarize_preflight_counts(rows: Iterable[PreflightRow]) -> tuple[int, int, int]:
    yes_count = 0
    partial_count = 0
    no_count = 0
    for row in rows:
        status = str(row.get("will_run", "")).strip()
        if status == "Yes":
            yes_count += 1
        elif status == "Partial":
            partial_count += 1
        elif status in {"No", "Fix"}:
            no_count += 1
    return yes_count, partial_count, no_count
