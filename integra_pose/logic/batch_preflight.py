"""Pure helpers for batch preflight planning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from integra_pose.logic.analytics_metric_catalog import AnalyticsMetricSpec

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


def _normalized_text(value: Any) -> str:
    return str(value or "").strip()


def _summarize_study_design(videos: Sequence[PreflightVideoState]) -> dict[str, Any]:
    group_sizes: dict[str, int] = {}
    subject_sizes: dict[str, int] = {}
    subject_time_levels: dict[str, set[str]] = {}
    time_levels: set[str] = set()
    numeric_time_levels: set[float] = set()
    missing_group_videos = 0
    missing_subject_videos = 0
    missing_time_videos = 0

    for video in videos:
        group = _normalized_text(getattr(video, "group", ""))
        subject_id = _normalized_text(getattr(video, "subject_id", ""))
        time_point = _normalized_text(getattr(video, "time_point", ""))

        if group:
            group_sizes[group] = group_sizes.get(group, 0) + 1
        else:
            missing_group_videos += 1

        if subject_id:
            subject_sizes[subject_id] = subject_sizes.get(subject_id, 0) + 1
        else:
            missing_subject_videos += 1

        if time_point:
            time_levels.add(time_point)
            if subject_id:
                subject_time_levels.setdefault(subject_id, set()).add(time_point)
            try:
                numeric_time_levels.add(float(time_point))
            except Exception:
                pass
        else:
            missing_time_videos += 1

    subjects_with_repeats = sum(1 for count in subject_sizes.values() if count >= 2)
    subjects_with_multiple_timepoints = sum(1 for times in subject_time_levels.values() if len(times) >= 2)
    labeled_group_videos = sum(group_sizes.values())

    return {
        "group_sizes": group_sizes,
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
        "missing_time_videos": missing_time_videos,
    }


def _default_index_label_dirs(root_dir: Path) -> list[tuple[Path, list[str]]]:
    grouped: dict[Path, list[str]] = {}
    if not root_dir.exists() or not root_dir.is_dir():
        return []
    for txt_file in root_dir.rglob("*.txt"):
        parent = txt_file.parent
        grouped.setdefault(parent, []).append(txt_file.name.lower())
    return [(parent, names) for parent, names in grouped.items() if names]


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

    def _dir_has_txt(path: Path) -> bool:
        if not path.exists() or not path.is_dir():
            return False
        try:
            return any(path.glob("*.txt"))
        except Exception:
            return False

    preferred = Path(str(preferred_dir or "").strip()).expanduser() if preferred_dir else None
    if preferred and _dir_has_txt(preferred):
        return preferred.resolve()

    if labels_root is not None:
        direct_candidates = [labels_root / video_stem, labels_root / video_stem / "labels", labels_root / f"{video_stem}_labels"]
        for candidate in direct_candidates:
            if _dir_has_txt(candidate):
                return candidate.resolve()

    if not indexed_dirs:
        return None

    prefix = f"{stem}_frame"
    best_path = None
    best_score = -1
    for parent, names in indexed_dirs:
        prefix_hit = any(name.startswith(prefix) for name in names)
        stem_hit = any(stem in name for name in names)
        parent_name = parent.name.lower()
        parent_hit = stem in parent_name
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
        score += min(len(names), 20)
        if score > best_score:
            best_score = score
            best_path = parent
    if best_score <= 0:
        return None
    return Path(best_path).resolve() if best_path is not None else None


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
    if strategy == "single":
        return (total if shared_has_object_rois else 0), total
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
) -> list[PreflightVideoState]:
    if str(strategy or "").strip() == "single":
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
    elif model_path and model_path_exists:
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
        or ((not use_existing_labels) and model_path_exists)
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
    elif strategy == "single" and object_ready <= 0:
        object_run = "No"
        object_reason = "Shared object ROIs are missing."
    elif strategy != "single" and object_ready < object_total:
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
    missing_object_rois = _missing_object_roi_videos(videos, strategy=strategy)
    if bool(config.object_interaction_enabled) and object_count > 0:
        if str(strategy).strip() == "single" and queue_total > 0 and not bool(shared_has_object_rois):
            rows.append(
                {
                    "analysis": "Missing object ROI setup",
                    "will_run": "Fix",
                    "scope": "Shared object ROI set",
                    "variables": f"Applies to all {queue_total} queued video(s)",
                    "reason": "Place the shared object ROI set once; it will be reused across the full batch.",
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
    group_sizes = dict(design["group_sizes"])
    group_levels = int(design["group_levels"])
    labeled_group_videos = int(design["labeled_group_videos"])
    missing_group_videos = int(design["missing_group_videos"])
    min_group_size = min(group_sizes.values()) if group_sizes else 0

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
            f"Each group needs at least {MIN_GROUP_REPLICATES} videos. "
            f"Current smallest group has {min_group_size}."
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
            "variables": "Video-level summaries and nonparametric group tests",
            "reason": group_reason,
        }
    )

    subjects_with_repeats = int(design["subjects_with_repeats"])
    subjects_with_multiple_timepoints = int(design["subjects_with_multiple_timepoints"])
    subject_count = int(design["subject_count"])
    numeric_time_levels = int(design["numeric_time_levels"])
    time_levels = int(design["time_levels"])

    mixed_ready = subjects_with_repeats >= MIN_REPEATED_SUBJECTS_FOR_MIXED and (
        group_levels >= 2 or subjects_with_multiple_timepoints >= MIN_REPEATED_SUBJECTS_FOR_MIXED
    )
    kpss_ready = numeric_time_levels >= MIN_KPSS_TIMEPOINTS

    if not bool(config.include_kpss):
        kpss_run = "No"
        kpss_reason = "Disabled in wizard."
        kpss_scope = "-"
    elif core_run == "No":
        kpss_run = "No"
        kpss_reason = "Behavior bout setup is incomplete."
        kpss_scope = "-"
    elif mixed_ready and kpss_ready and core_run == "Yes":
        kpss_run = "Yes"
        kpss_reason = (
            "Mixed-effects diagnostics are ready, and KPSS has enough ordered numeric time points "
            f"({numeric_time_levels} available)."
        )
        kpss_scope = f"{queue_total} video(s)"
    elif mixed_ready or kpss_ready or core_run == "Partial":
        kpss_run = "Partial"
        reasons: list[str] = []
        if mixed_ready:
            reasons.append("mixed-effects is ready")
        else:
            if subject_count < MIN_REPEATED_SUBJECTS_FOR_MIXED:
                reasons.append("need subject IDs for at least 2 subjects")
            elif subjects_with_repeats < MIN_REPEATED_SUBJECTS_FOR_MIXED:
                reasons.append("need repeated observations for at least 2 subjects")
            elif time_levels < 2 and group_levels < 2:
                reasons.append("need either replicated groups or repeated time points")
            else:
                reasons.append("mixed-effects design metadata is incomplete")
        if kpss_ready:
            reasons.append("KPSS is ready on ordered numeric time points")
        else:
            reasons.append(f"KPSS needs at least {MIN_KPSS_TIMEPOINTS} numeric ordered time points")
        if core_run == "Partial":
            reasons.append("only videos with completed core analytics contribute")
        kpss_reason = "; ".join(reasons) + "."
        kpss_scope = f"{queue_total} video(s)"
    else:
        kpss_run = "No"
        kpss_reason = (
            "Need repeated-subject metadata for mixed-effects diagnostics and at least "
            f"{MIN_KPSS_TIMEPOINTS} numeric ordered time points for KPSS."
        )
        kpss_scope = "-"
    rows.append(
        {
            "analysis": "Trend / mixed-effects diagnostics",
            "will_run": kpss_run,
            "scope": kpss_scope,
            "variables": "Time-trend diagnostics and mixed-effects model metadata",
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
        elif status == "No":
            no_count += 1
    return yes_count, partial_count, no_count
