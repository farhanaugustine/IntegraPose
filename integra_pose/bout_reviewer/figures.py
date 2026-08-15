from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

from matplotlib import pyplot as plt

from .models import (
    BEHAVIOR,
    EVENT_KIND_LABELS,
    OBJECT_INTERACTION,
    ROI_CONCURRENT,
    ROI_EXCLUSIVE,
    ProjectData,
)


BACKGROUND = "#171a1f"
PANEL = "#232830"
GRID = "#3d4551"
TEXT = "#e5e9ef"
MUTED = "#9da6b2"
ORIGINAL = "#9da6b2"
REVIEWED = "#4cc9f0"
PRECISION = "#4cc9f0"
RECALL = "#9b5de5"
F1 = "#ffd166"
START_ERROR = "#4cc9f0"
END_ERROR = "#f72585"

CATEGORY_TITLES = {
    "ROI_Bouts": "ROI bouts",
    "Object_Interactions": "Object interactions",
    "Behavior_Bouts": "Behavior bouts",
}

CATEGORY_PREFIXES = {
    "ROI_Bouts": "roi",
    "Object_Interactions": "object",
    "Behavior_Bouts": "behavior",
}

EVENT_ORDER = {
    ROI_CONCURRENT: 0,
    ROI_EXCLUSIVE: 1,
    OBJECT_INTERACTION: 2,
    BEHAVIOR: 3,
}

EVENT_SHORT_LABELS = {
    ROI_CONCURRENT: "ROI concurrent",
    ROI_EXCLUSIVE: "ROI-X exclusive",
    OBJECT_INTERACTION: "Object interaction",
    BEHAVIOR: "Behavior",
}

EVENT_STYLES = {
    ROI_CONCURRENT: ("-", "o"),
    ROI_EXCLUSIVE: ("--", "s"),
    OBJECT_INTERACTION: ("-", "D"),
    BEHAVIOR: ("-", "o"),
}


def _style_axis(axis: Any) -> None:
    axis.set_facecolor(PANEL)
    axis.tick_params(colors=TEXT, labelcolor=TEXT)
    axis.xaxis.label.set_color(MUTED)
    axis.yaxis.label.set_color(MUTED)
    axis.title.set_color(TEXT)
    for spine in axis.spines.values():
        spine.set_color(GRID)
    axis.grid(axis="x", color=GRID, alpha=0.55, linewidth=0.8)
    axis.set_axisbelow(True)


def _save_figure(figure: Any, path: Path) -> None:
    try:
        figure.savefig(
            path,
            dpi=240,
            facecolor=figure.get_facecolor(),
            bbox_inches="tight",
        )
    finally:
        plt.close(figure)


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _class_text(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    return text if text and text != "None" else ""


def _channel_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        EVENT_ORDER.get(str(row.get("event_kind", "")), 99),
        str(row.get("event_kind", "")),
        str(row.get("label", "")).casefold(),
        _class_text(row.get("class_id")),
        int(row.get("track_id", 0)),
    )


def _channel_label(row: dict[str, Any]) -> str:
    event_kind = str(row.get("event_kind", ""))
    label = str(row.get("label", "")).replace("_", " ")
    track_id = int(row.get("track_id", 0))
    if event_kind == BEHAVIOR:
        class_id = _class_text(row.get("class_id"))
        class_label = f"[{class_id}] " if class_id else ""
        return f"{class_label}{label} · Track {track_id}"
    if event_kind in {ROI_CONCURRENT, ROI_EXCLUSIVE}:
        return f"{EVENT_SHORT_LABELS[event_kind]} · {label} · Track {track_id}"
    return f"{label} · Track {track_id}"


def _aggregate_before_after(
    summary_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in summary_rows:
        key = (
            str(row.get("event_kind", "")),
            str(row.get("label", "")),
            _class_text(row.get("class_id")),
            int(row.get("track_id", 0)),
        )
        aggregate = grouped.setdefault(
            key,
            {
                "event_kind": key[0],
                "label": key[1],
                "class_id": key[2],
                "track_id": key[3],
                "original_bout_count": 0,
                "reviewed_bout_count": 0,
                "original_dwell_seconds": 0.0,
                "reviewed_dwell_seconds": 0.0,
                "video_ids": set(),
                "scope_complete": True,
            },
        )
        aggregate["original_bout_count"] += int(
            row.get("original_bout_count", 0)
        )
        aggregate["reviewed_bout_count"] += int(
            row.get("reviewed_bout_count", 0)
        )
        aggregate["original_dwell_seconds"] += float(
            row.get("original_dwell_seconds", 0.0)
        )
        aggregate["reviewed_dwell_seconds"] += float(
            row.get("reviewed_dwell_seconds", 0.0)
        )
        aggregate["video_ids"].add(str(row.get("video_id", "")))
        aggregate["scope_complete"] = bool(
            aggregate["scope_complete"]
        ) and bool(row.get("scope_complete", False))
    return sorted(grouped.values(), key=_channel_key)


def _before_after_figure(
    *,
    rows: Sequence[dict[str, Any]],
    original_field: str,
    reviewed_field: str,
    x_label: str,
    title: str,
    footer: str,
    output_path: Path,
) -> Path | None:
    if not rows:
        return None
    labels = [_channel_label(row) for row in rows]
    original_values = [float(row[original_field]) for row in rows]
    reviewed_values = [float(row[reviewed_field]) for row in rows]
    y_positions = list(range(len(rows)))
    figure_height = max(4.8, min(28.0, 1.8 + 0.48 * len(rows)))
    figure, axis = plt.subplots(figsize=(13.5, figure_height))
    figure.patch.set_facecolor(BACKGROUND)
    _style_axis(axis)
    bar_height = 0.34
    axis.barh(
        [value - bar_height / 2 for value in y_positions],
        original_values,
        height=bar_height,
        color=ORIGINAL,
        label="Original prediction",
    )
    axis.barh(
        [value + bar_height / 2 for value in y_positions],
        reviewed_values,
        height=bar_height,
        color=REVIEWED,
        label="Final reviewed reference",
    )
    axis.set_yticks(y_positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel(x_label)
    provisional = not all(bool(row["scope_complete"]) for row in rows)
    axis.set_title(f"{title}{' · PROVISIONAL' if provisional else ''}", pad=14)
    maximum = max([*original_values, *reviewed_values, 1.0])
    axis.set_xlim(0, maximum * 1.32)
    for position, original, reviewed in zip(
        y_positions,
        original_values,
        reviewed_values,
    ):
        ratio = _ratio(reviewed, original)
        ratio_text = "ratio N/A" if ratio is None else f"ratio {ratio:.2f}"
        axis.text(
            max(original, reviewed) + maximum * 0.025,
            position,
            ratio_text,
            color=TEXT,
            va="center",
            fontsize=9,
        )
    legend = axis.legend(loc="lower right", frameon=False)
    for text in legend.get_texts():
        text.set_color(TEXT)
    figure.text(
        0.01,
        0.005,
        footer,
        color=MUTED,
        fontsize=8.5,
        ha="left",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    _save_figure(figure, output_path)
    return output_path


def _score_rows_with_data(
    score_rows: Sequence[dict[str, Any]],
    *,
    scope: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in score_rows
        if row.get("scope") == scope
        and (
            int(row.get("predicted_events", 0))
            or int(row.get("reviewed_events", 0))
        )
    ]


def _tiou_figure(
    *,
    score_rows: Sequence[dict[str, Any]],
    output_path: Path,
    category_title: str,
) -> Path | None:
    rows = _score_rows_with_data(score_rows, scope="batch_event_kind")
    if not rows:
        return None
    metrics = (
        ("event_precision", "Event precision", PRECISION),
        ("event_recall", "Event recall", RECALL),
        ("event_f1", "Event F1", F1),
    )
    event_kinds = sorted(
        {str(row["event_kind"]) for row in rows},
        key=lambda value: EVENT_ORDER.get(value, 99),
    )
    has_value = any(
        row.get(metric) is not None
        for row in rows
        for metric, _label, _color in metrics
    )
    if not has_value:
        return None
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharey=True)
    figure.patch.set_facecolor(BACKGROUND)
    for axis, (metric, metric_label, color) in zip(axes, metrics):
        _style_axis(axis)
        for event_kind in event_kinds:
            selected = sorted(
                (
                    row
                    for row in rows
                    if str(row["event_kind"]) == event_kind
                    and row.get(metric) is not None
                ),
                key=lambda row: float(row["temporal_iou_threshold"]),
            )
            if not selected:
                continue
            x_values = [
                float(row["temporal_iou_threshold"]) for row in selected
            ]
            y_values = [float(row[metric]) for row in selected]
            line_style, marker = EVENT_STYLES.get(
                event_kind,
                ("-", "o"),
            )
            axis.plot(
                x_values,
                y_values,
                marker=marker,
                linestyle=line_style,
                linewidth=2.0,
                markersize=6,
                color=color,
                alpha=0.9,
                label=EVENT_SHORT_LABELS.get(
                    event_kind,
                    EVENT_KIND_LABELS.get(event_kind, event_kind),
                ),
            )
            if len(x_values) == 1:
                axis.text(
                    x_values[0],
                    min(1.02, y_values[0] + 0.055),
                    f"{y_values[0]:.2f}",
                    color=TEXT,
                    ha="center",
                    fontsize=9,
                )
        axis.set_title(metric_label)
        axis.set_xlabel("Temporal IoU threshold")
        axis.set_ylim(0, 1.05)
        thresholds = sorted(
            {float(row["temporal_iou_threshold"]) for row in rows}
        )
        axis.set_xticks(thresholds)
    axes[0].set_ylabel("Agreement score")
    if len(event_kinds) > 1:
        handles, labels = axes[-1].get_legend_handles_labels()
        legend = figure.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(3, len(labels)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.93),
        )
        for text in legend.get_texts():
            text.set_color(TEXT)
        top = 0.84
    else:
        top = 0.9
    provisional = not all(bool(row.get("scope_complete")) for row in rows)
    figure.suptitle(
        f"{category_title}: original predictions vs reviewed reference"
        f"{' · PROVISIONAL' if provisional else ''}",
        color=TEXT,
        fontsize=15,
        y=0.99,
    )
    figure.text(
        0.01,
        0.005,
        "One-to-one event matching; corrected bouts are the review reference "
        "and are not plotted against themselves.",
        color=MUTED,
        fontsize=8.5,
        ha="left",
    )
    figure.tight_layout(rect=(0, 0.06, 1, top))
    _save_figure(figure, output_path)
    return output_path


def _weighted_boundary_rows(
    *,
    project: ProjectData,
    score_rows: Sequence[dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    fps_by_video = {video.video_id: video.fps for video in project.videos}
    grouped: dict[tuple[Any, ...], dict[str, Any]] = defaultdict(
        lambda: {
            "start_weighted_seconds": 0.0,
            "end_weighted_seconds": 0.0,
            "matched_events": 0,
            "scope_complete": True,
        }
    )
    identities: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in score_rows:
        if row.get("scope") != "video_label_track":
            continue
        if abs(float(row["temporal_iou_threshold"]) - threshold) > 1e-9:
            continue
        matched = int(row.get("true_positive_events", 0))
        start_error = row.get("mean_abs_start_error_frames")
        end_error = row.get("mean_abs_end_error_frames")
        if not matched or start_error is None or end_error is None:
            continue
        video_id = str(row["video_id"])
        fps = float(fps_by_video.get(video_id, 0.0))
        if fps <= 0:
            continue
        key = (
            str(row.get("event_kind", "")),
            str(row.get("label", "")),
            _class_text(row.get("class_id")),
            int(row.get("track_id", 0)),
        )
        aggregate = grouped[key]
        aggregate["start_weighted_seconds"] += (
            float(start_error) / fps
        ) * matched
        aggregate["end_weighted_seconds"] += (
            float(end_error) / fps
        ) * matched
        aggregate["matched_events"] += matched
        aggregate["scope_complete"] = bool(
            aggregate["scope_complete"]
        ) and bool(row.get("scope_complete", False))
        identities[key] = {
            "event_kind": key[0],
            "label": key[1],
            "class_id": key[2],
            "track_id": key[3],
        }
    result: list[dict[str, Any]] = []
    for key, aggregate in grouped.items():
        matched = int(aggregate["matched_events"])
        result.append(
            {
                **identities[key],
                "mean_abs_start_error_seconds": (
                    aggregate["start_weighted_seconds"] / matched
                ),
                "mean_abs_end_error_seconds": (
                    aggregate["end_weighted_seconds"] / matched
                ),
                "matched_events": matched,
                "scope_complete": bool(aggregate["scope_complete"]),
            }
        )
    return sorted(result, key=_channel_key)


def _boundary_figure(
    *,
    project: ProjectData,
    score_rows: Sequence[dict[str, Any]],
    thresholds: Sequence[float],
    output_path: Path,
    category_title: str,
) -> Path | None:
    available = sorted({float(value) for value in thresholds})
    if not available:
        return None
    threshold = (
        0.5
        if any(abs(value - 0.5) < 1e-9 for value in available)
        else available[0]
    )
    rows = _weighted_boundary_rows(
        project=project,
        score_rows=score_rows,
        threshold=threshold,
    )
    if not rows:
        return None
    labels = [_channel_label(row) for row in rows]
    start_values = [
        float(row["mean_abs_start_error_seconds"]) for row in rows
    ]
    end_values = [
        float(row["mean_abs_end_error_seconds"]) for row in rows
    ]
    y_positions = list(range(len(rows)))
    figure_height = max(4.8, min(28.0, 1.8 + 0.48 * len(rows)))
    figure, axis = plt.subplots(figsize=(13.5, figure_height))
    figure.patch.set_facecolor(BACKGROUND)
    _style_axis(axis)
    for position, start, end in zip(y_positions, start_values, end_values):
        axis.plot([start, end], [position, position], color=GRID, linewidth=2)
    axis.scatter(
        start_values,
        y_positions,
        color=START_ERROR,
        s=52,
        label="Mean absolute start error",
        zorder=3,
    )
    axis.scatter(
        end_values,
        y_positions,
        color=END_ERROR,
        s=52,
        label="Mean absolute end error",
        zorder=3,
    )
    maximum = max([*start_values, *end_values, 0.001])
    axis.set_xlim(0, maximum * 1.35)
    axis.set_yticks(y_positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Weighted mean absolute error (seconds)")
    provisional = not all(bool(row["scope_complete"]) for row in rows)
    axis.set_title(
        f"{category_title}: boundary error at tIoU ≥ {threshold:.2f}"
        f"{' · PROVISIONAL' if provisional else ''}",
        pad=14,
    )
    for position, row, start, end in zip(
        y_positions,
        rows,
        start_values,
        end_values,
    ):
        axis.text(
            max(start, end) + maximum * 0.025,
            position,
            f"n={int(row['matched_events'])}",
            color=TEXT,
            va="center",
            fontsize=9,
        )
    legend = axis.legend(loc="lower right", frameon=False)
    for text in legend.get_texts():
        text.set_color(TEXT)
    figure.text(
        0.01,
        0.005,
        "Each channel value pools video-level means using matched-event counts "
        "as weights; unmatched events do not have a boundary error.",
        color=MUTED,
        fontsize=8.5,
        ha="left",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    _save_figure(figure, output_path)
    return output_path


def generate_category_figures(
    *,
    category_name: str,
    output_dir: Path,
    project: ProjectData,
    summary_rows: Sequence[dict[str, Any]],
    score_rows: Sequence[dict[str, Any]],
    event_iou_thresholds: Sequence[float],
) -> list[Path]:
    """Generate the compact, mode-specific review figure set."""

    output_dir.mkdir(parents=True, exist_ok=False)
    category_title = CATEGORY_TITLES[category_name]
    prefix = CATEGORY_PREFIXES[category_name]
    pooled = _aggregate_before_after(summary_rows)
    generated: list[Path] = []

    count_path = _before_after_figure(
        rows=pooled,
        original_field="original_bout_count",
        reviewed_field="reviewed_bout_count",
        x_label="Bout count pooled across videos",
        title=f"{category_title}: bout counts before and after review",
        footer=(
            "Ratio = final reviewed bouts / original predicted bouts. "
            "N/A indicates zero original predictions."
        ),
        output_path=output_dir / f"01_{prefix}_bout_counts_before_after.png",
    )
    if count_path is not None:
        generated.append(count_path)

    dwell_path = _before_after_figure(
        rows=pooled,
        original_field="original_dwell_seconds",
        reviewed_field="reviewed_dwell_seconds",
        x_label="Total bout duration (seconds), pooled across videos",
        title=f"{category_title}: dwell time before and after review",
        footer=(
            "Ratio = final reviewed duration / original predicted duration. "
            "Concurrent ROI or overlapping behavior durations may overlap in time."
        ),
        output_path=output_dir / f"02_{prefix}_dwell_time_before_after.png",
    )
    if dwell_path is not None:
        generated.append(dwell_path)

    tiou_path = _tiou_figure(
        score_rows=score_rows,
        output_path=output_dir / f"03_{prefix}_tiou_default_or_sweep.png",
        category_title=category_title,
    )
    if tiou_path is not None:
        generated.append(tiou_path)

    boundary_path = _boundary_figure(
        project=project,
        score_rows=score_rows,
        thresholds=event_iou_thresholds,
        output_path=output_dir / f"04_{prefix}_boundary_errors.png",
        category_title=category_title,
    )
    if boundary_path is not None:
        generated.append(boundary_path)

    return generated
