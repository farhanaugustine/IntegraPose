"""Shared assay-aware analytics metric catalog for Tab 6 and batch workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class AnalyticsMetricSpec:
    key: str
    label: str
    description: str
    variables: str
    var_attr: str
    category: str
    module_keys: tuple[str, ...] = ()
    requires_rois: bool = False
    requires_object_interactions: bool = False
    requires_multi_animal: bool = False
    requires_keypoints: bool = False
    requires_multiclass_behaviors: bool = False


@dataclass(frozen=True, slots=True)
class AnalyticsAssayPreset:
    key: str
    label: str
    description: str
    metric_keys: tuple[str, ...]
    enable_rois: bool = False
    enable_object_interaction: bool = False


ANALYTICS_METRIC_SPECS: tuple[AnalyticsMetricSpec, ...] = (
    AnalyticsMetricSpec(
        key="preference_indices",
        label="Preference indices",
        description="Compute zone/object preference, discrimination, bias, and first-choice summaries.",
        variables="Pairwise preference scores, discrimination index, first choice, occupancy bias",
        var_attr="enable_preference_indices_var",
        category="Choice and preference",
        module_keys=("preference_indices",),
        requires_rois=False,
        requires_object_interactions=False,
    ),
    AnalyticsMetricSpec(
        key="latency_metrics",
        label="Latency metrics",
        description="Report time to first zone entry, first object interaction/contact, and early disengagement metrics.",
        variables="Latency to first ROI/object visit, first contact, first exit, revisit delay",
        var_attr="enable_latency_metrics_var",
        category="Choice and preference",
        module_keys=("latency_metrics",),
        requires_rois=False,
        requires_object_interactions=False,
    ),
    AnalyticsMetricSpec(
        key="visit_structure",
        label="Visit structure",
        description="Summarize dwell distributions, visit counts, and visit-duration structure for zones and objects.",
        variables="Visit counts, dwell distributions, median/quantile dwell, per-visit structure",
        var_attr="enable_visit_structure_var",
        category="Choice and preference",
        module_keys=("visit_structure",),
        requires_rois=False,
        requires_object_interactions=False,
    ),
    AnalyticsMetricSpec(
        key="object_transition_analysis",
        label="Object transitions",
        description="Track visit sequences and transitions between stimuli or targets of interest.",
        variables="Object-to-object transitions, revisit loops, alternation-like sequence summaries",
        var_attr="enable_object_transition_analysis_var",
        category="Choice and preference",
        module_keys=("object_transition_analysis",),
        requires_object_interactions=True,
    ),
    AnalyticsMetricSpec(
        key="normalization_summary",
        label="Normalized summaries",
        description="Export per-minute, percent-of-session, and normalized occupancy/interaction summaries.",
        variables="Per-minute rates, percent-of-session, normalized visit and interaction summaries",
        var_attr="enable_normalization_summary_var",
        category="Choice and preference",
        module_keys=("normalization_summary",),
    ),
    AnalyticsMetricSpec(
        key="roi_time_heatmap",
        label="Zone occupancy heatmap",
        description="Summarize time spent in arena zones over session time bins.",
        variables="Zone occupancy by time bin, ROI time heatmaps",
        var_attr="enable_roi_time_heatmap_var",
        category="Zones and mazes",
        module_keys=("roi_time_heatmap",),
        requires_rois=True,
    ),
    AnalyticsMetricSpec(
        key="roi_context_windows",
        label="Zone context windows",
        description="Capture zone state immediately before and after each bout.",
        variables="Pre/post zone labels around each bout",
        var_attr="enable_roi_context_windows_var",
        category="Zones and mazes",
        module_keys=("roi_context_windows",),
        requires_rois=True,
    ),
    AnalyticsMetricSpec(
        key="behavior_transitions",
        label="Behavior transitions",
        description="Summarize how behaviors switch over time and across zones.",
        variables="Behavior-to-behavior transitions, ROI-linked switches, transition gaps",
        var_attr="enable_behavior_transitions_var",
        category="Behavior structure",
        module_keys=("behavior_transitions",),
        requires_multiclass_behaviors=True,
    ),
    AnalyticsMetricSpec(
        key="temporal_trends",
        label="Temporal trends",
        description="Measure how behavior or occupancy changes over the session.",
        variables="Behavior time by bin, counts by bin, cumulative trajectories",
        var_attr="enable_temporal_trends_var",
        category="Behavior structure",
        module_keys=("temporal_trends",),
    ),
    AnalyticsMetricSpec(
        key="activity_budgets",
        label="Activity budgets",
        description="Report overall, per-track, and per-zone behavioral time budgets.",
        variables="Behavior time allocation overall, by track, and by ROI",
        var_attr="enable_activity_budgets_var",
        category="Behavior structure",
        module_keys=("activity_budgets",),
    ),
    AnalyticsMetricSpec(
        key="inter_bout_intervals",
        label="Inter-bout intervals",
        description="Compute time between repeated bouts and latency to first bout.",
        variables="Inter-bout intervals, first-bout latency",
        var_attr="enable_inter_bout_intervals_var",
        category="Behavior structure",
        module_keys=("inter_bout_intervals",),
    ),
    AnalyticsMetricSpec(
        key="event_aligned_windows",
        label="Event-aligned windows",
        description="Align behavior around zone or object entry/exit events.",
        variables="Behavior windows around ROI/object entry and exit events",
        var_attr="enable_event_aligned_windows_var",
        category="Behavior structure",
        module_keys=("event_aligned_windows",),
    ),
    AnalyticsMetricSpec(
        key="bout_timeline_export",
        label="Bout timeline export",
        description="Export timeline tables and Gantt-style bout summaries.",
        variables="Bout timelines and timeline plots",
        var_attr="enable_bout_timeline_export_var",
        category="Behavior structure",
        module_keys=("bout_timeline_export",),
    ),
    AnalyticsMetricSpec(
        key="kinematic_descriptors",
        label="Movement summaries",
        description="Measure speed, acceleration, path length, and pose-derived kinematics.",
        variables="Speed, acceleration, path length, joint-angle summaries",
        var_attr="enable_kinematic_descriptors_var",
        category="Motion and QC",
        module_keys=("kinematic_descriptors",),
        requires_keypoints=False,
    ),
    AnalyticsMetricSpec(
        key="detection_quality",
        label="Detection quality",
        description="Evaluate keypoint confidence, completeness, and bbox stability.",
        variables="Confidence, completeness, bbox stability",
        var_attr="enable_detection_quality_var",
        category="Motion and QC",
        module_keys=("detection_quality",),
    ),
    AnalyticsMetricSpec(
        key="multi_animal_descriptors",
        label="Multi-animal proximity",
        description="Measure proximity and co-occurrence for multi-animal experiments.",
        variables="Pairwise proximity, overlap, co-occurrence",
        var_attr="enable_multi_animal_descriptors_var",
        category="Motion and QC",
        module_keys=("multi_animal_descriptors",),
        requires_multi_animal=True,
    ),
)


ANALYTICS_ASSAY_PRESETS: tuple[AnalyticsAssayPreset, ...] = (
    AnalyticsAssayPreset(
        key="custom",
        label="Custom / mixed",
        description="Start blank and freely combine metrics across paradigms.",
        metric_keys=(),
    ),
    AnalyticsAssayPreset(
        key="open_field",
        label="Open field",
        description="Center/periphery occupancy, locomotion, and session dynamics.",
        metric_keys=(
            "roi_time_heatmap",
            "activity_budgets",
            "kinematic_descriptors",
            "latency_metrics",
            "visit_structure",
            "normalization_summary",
            "inter_bout_intervals",
            "detection_quality",
        ),
        enable_rois=True,
    ),
    AnalyticsAssayPreset(
        key="elevated_plus",
        label="Elevated plus maze",
        description="Open/closed arm occupancy, entry dynamics, and risk-oriented exploration patterns.",
        metric_keys=(
            "preference_indices",
            "latency_metrics",
            "visit_structure",
            "roi_time_heatmap",
            "behavior_transitions",
            "activity_budgets",
            "normalization_summary",
            "detection_quality",
        ),
        enable_rois=True,
    ),
    AnalyticsAssayPreset(
        key="t_maze",
        label="T-maze",
        description="Choice bias, latency, revisits, and arm switching for T-maze assays.",
        metric_keys=(
            "preference_indices",
            "latency_metrics",
            "visit_structure",
            "behavior_transitions",
            "event_aligned_windows",
            "normalization_summary",
            "inter_bout_intervals",
        ),
        enable_rois=True,
    ),
    AnalyticsAssayPreset(
        key="y_maze",
        label="Y-maze",
        description="Arm preference, revisit structure, and temporal switching for Y-maze assays.",
        metric_keys=(
            "preference_indices",
            "latency_metrics",
            "visit_structure",
            "behavior_transitions",
            "event_aligned_windows",
            "normalization_summary",
            "roi_time_heatmap",
        ),
        enable_rois=True,
    ),
    AnalyticsAssayPreset(
        key="barnes_maze",
        label="Barnes maze",
        description="Search latency, visit structure, locomotion, and event-centered search behavior.",
        metric_keys=(
            "latency_metrics",
            "visit_structure",
            "kinematic_descriptors",
            "event_aligned_windows",
            "normalization_summary",
            "behavior_transitions",
            "detection_quality",
        ),
        enable_rois=True,
    ),
    AnalyticsAssayPreset(
        key="object_preference",
        label="Object preference / NOR",
        description="Object interaction, discrimination, and first-choice metrics for novelty or preference assays.",
        metric_keys=(
            "preference_indices",
            "latency_metrics",
            "visit_structure",
            "object_transition_analysis",
            "event_aligned_windows",
            "normalization_summary",
            "behavior_transitions",
            "detection_quality",
        ),
        enable_object_interaction=True,
    ),
)


METRIC_SPEC_BY_KEY = {spec.key: spec for spec in ANALYTICS_METRIC_SPECS}
METRIC_VAR_ATTR_MAP = {spec.key: spec.var_attr for spec in ANALYTICS_METRIC_SPECS}
PRESET_BY_KEY = {preset.key: preset for preset in ANALYTICS_ASSAY_PRESETS}


def iter_metric_specs() -> tuple[AnalyticsMetricSpec, ...]:
    return ANALYTICS_METRIC_SPECS


def iter_assay_presets() -> tuple[AnalyticsAssayPreset, ...]:
    return ANALYTICS_ASSAY_PRESETS


def get_metric_spec(metric_key: str) -> AnalyticsMetricSpec | None:
    return METRIC_SPEC_BY_KEY.get(str(metric_key or "").strip())


def get_assay_preset(preset_key: str) -> AnalyticsAssayPreset | None:
    return PRESET_BY_KEY.get(str(preset_key or "").strip())


def collect_enabled_metric_keys(config_section) -> set[str]:
    enabled: set[str] = set()
    if config_section is None:
        return enabled
    for spec in ANALYTICS_METRIC_SPECS:
        var = getattr(config_section, spec.var_attr, None)
        if var is None:
            continue
        try:
            if bool(var.get()):
                enabled.add(spec.key)
        except Exception:
            continue
    return enabled


def set_enabled_metric_keys(config_section, metric_keys: Iterable[str]) -> None:
    desired = {str(key).strip() for key in (metric_keys or ()) if str(key).strip()}
    if config_section is None:
        return
    for spec in ANALYTICS_METRIC_SPECS:
        var = getattr(config_section, spec.var_attr, None)
        if var is None:
            continue
        try:
            var.set(spec.key in desired)
        except Exception:
            continue


def expand_metrics_to_modules(metric_keys: Iterable[str]) -> list[str]:
    modules: list[str] = []
    seen: set[str] = set()
    for metric_key in metric_keys or ():
        spec = get_metric_spec(metric_key)
        if spec is None:
            continue
        for module_key in spec.module_keys:
            if module_key in seen:
                continue
            seen.add(module_key)
            modules.append(module_key)
    return modules


def apply_assay_preset(config_section, preset_key: str) -> set[str]:
    preset = get_assay_preset(preset_key) or PRESET_BY_KEY["custom"]
    set_enabled_metric_keys(config_section, preset.metric_keys)
    preset_var = getattr(config_section, "assay_preset_var", None)
    if preset_var is not None:
        try:
            preset_var.set(preset.key)
        except Exception:
            pass
    rois_var = getattr(config_section, "use_rois_for_analytics_var", None)
    if preset.enable_rois and rois_var is not None:
        try:
            rois_var.set(True)
        except Exception:
            pass
    object_var = getattr(config_section, "object_interaction_enabled_var", None)
    if preset.enable_object_interaction and object_var is not None:
        try:
            object_var.set(True)
        except Exception:
            pass
    return set(preset.metric_keys)


def category_order_for_specs(specs: Sequence[AnalyticsMetricSpec] | None = None) -> list[str]:
    categories: list[str] = []
    seen: set[str] = set()
    for spec in specs or ANALYTICS_METRIC_SPECS:
        if spec.category in seen:
            continue
        seen.add(spec.category)
        categories.append(spec.category)
    return categories
