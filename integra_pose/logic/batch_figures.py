"""Batch-level publication figure export utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import re
import textwrap

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from integra_pose.logic.analytics_metric_catalog import expand_metrics_to_modules, get_assay_preset

FIGURE_MANIFEST_COLUMNS = [
    "scope",
    "figure_type",
    "metric",
    "video_id",
    "video_name",
    "group",
    "subject_id",
    "time_point",
    "group_a",
    "group_b",
    "factor",
    "format",
    "title",
    "source",
    "path",
]

_IMAGE_SUFFIXES = {".png", ".svg", ".pdf", ".jpg", ".jpeg", ".tif", ".tiff"}
_SUMMARY_EXCLUDE_COLUMNS = {"video_id", "video_name", "video_path", "group", "subject_id", "time_point"}
_MODULE_CONTEXT_COLUMNS = {
    "video_id",
    "video_name",
    "video_path",
    "group",
    "subject_id",
    "time_point",
    "source_module",
    "source_file_key",
    "source_file_path",
}


@dataclass(slots=True)
class BatchFigureArtifacts:
    output_dir: str
    manifest_csv: str
    figure_count: int
    assay_manifest_csv: str = ""
    assay_figure_count: int = 0


def _safe_read_csv(path: str | Path) -> pd.DataFrame:
    target = Path(str(path or "")).expanduser()
    if not target.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(target)
    except Exception:
        return pd.DataFrame()


def _safe_read_json(path: str | Path) -> dict[str, Any]:
    target = Path(str(path or "")).expanduser()
    if not target.is_file():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _resolve_numeric_metrics(df: pd.DataFrame, *, exclude: set[str] | None = None, min_non_na: int = 2) -> list[str]:
    excluded = set(exclude or set())
    metrics: list[str] = []
    for column in df.columns:
        if column in excluded:
            continue
        numeric = _coerce_numeric(df[column])
        if numeric.notna().sum() >= max(1, int(min_non_na or 1)):
            metrics.append(column)
    return metrics


def _display_label(value: str) -> str:
    return str(value or "").replace("_", " ").strip().title()


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip()).strip("_")
    return cleaned.lower() or "figure"


def _resolve_file_path(raw_path: str | Path, *, base_dir: Path | None = None) -> Path:
    candidate = Path(str(raw_path or "")).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if base_dir is not None:
        return (base_dir / candidate).resolve()
    return candidate.resolve()


def _manifest_row(
    *,
    scope: str,
    figure_type: str,
    metric: str = "",
    video_id: str = "",
    video_name: str = "",
    group: str = "",
    subject_id: str = "",
    time_point: str = "",
    group_a: str = "",
    group_b: str = "",
    factor: str = "",
    title: str = "",
    source: str = "",
    path: str | Path,
) -> dict[str, str]:
    resolved = _resolve_file_path(path)
    return {
        "scope": str(scope),
        "figure_type": str(figure_type),
        "metric": str(metric),
        "video_id": str(video_id),
        "video_name": str(video_name),
        "group": str(group),
        "subject_id": str(subject_id),
        "time_point": str(time_point),
        "group_a": str(group_a),
        "group_b": str(group_b),
        "factor": str(factor),
        "format": resolved.suffix.lstrip(".").lower(),
        "title": str(title),
        "source": str(source),
        "path": str(resolved),
    }


def _save_figure(fig: plt.Figure, target_base: Path) -> list[Path]:
    saved_paths: list[Path] = []
    png_target = target_base.with_suffix(".png")
    svg_target = target_base.with_suffix(".svg")
    for target, kwargs in (
        (png_target, {"dpi": 360}),
        (svg_target, {}),
    ):
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target, bbox_inches="tight", facecolor="white", **kwargs)
            saved_paths.append(target.resolve())
        except Exception:
            continue
    plt.close(fig)
    return saved_paths


def _apply_publication_theme() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.edgecolor": "#374151",
            "axes.linewidth": 0.9,
            "axes.titlesize": 16,
            "axes.labelsize": 12.5,
            "axes.titlepad": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 9.5,
            "legend.title_fontsize": 10,
            "figure.titlesize": 17,
            "font.family": "DejaVu Sans",
            "font.weight": "regular",
            "figure.dpi": 160,
            "grid.color": "#d7dde6",
            "grid.linestyle": ":",
            "grid.linewidth": 0.75,
            "axes.facecolor": "#fbfcfe",
            "savefig.transparent": False,
        }
    )


def _ordered_labels(series: pd.Series) -> list[str]:
    cleaned = series.fillna("").astype(str).str.strip()
    cleaned = cleaned[cleaned != ""]
    if cleaned.empty:
        return []
    unique = list(dict.fromkeys(cleaned.tolist()))
    numeric_pairs: list[tuple[float, str]] = []
    for label in unique:
        try:
            numeric_pairs.append((float(label), label))
        except Exception:
            return unique
    return [label for _value, label in sorted(numeric_pairs, key=lambda pair: pair[0])]


def _format_value(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    magnitude = abs(float(value))
    if magnitude >= 1000:
        return f"{value:,.1f}"
    if magnitude >= 100:
        return f"{value:,.2f}"
    if magnitude >= 1:
        return f"{value:,.3f}"
    return f"{value:.4f}"


def _wrap_label(value: Any, *, width: int = 18) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return textwrap.fill(text, width=width, break_long_words=False, break_on_hyphens=False)


def _thin_category_labels(labels: list[Any], *, max_labels: int = 8) -> tuple[list[int], list[str]]:
    cleaned = [str(label) for label in labels]
    if len(cleaned) <= max_labels:
        return list(range(len(cleaned))), cleaned
    max_labels = max(2, int(max_labels))
    positions = np.linspace(0, len(cleaned) - 1, num=max_labels, dtype=int)
    unique_positions: list[int] = []
    seen: set[int] = set()
    for pos in positions.tolist():
        if pos in seen:
            continue
        seen.add(pos)
        unique_positions.append(pos)
    return unique_positions, [cleaned[pos] for pos in unique_positions]


def _apply_sparse_x_labels(ax: plt.Axes, labels: list[Any], *, max_labels: int = 8, rotation: int = 30) -> None:
    positions, tick_labels = _thin_category_labels(labels, max_labels=max_labels)
    ax.set_xticks(positions)
    ax.set_xticklabels([_wrap_label(label, width=14) for label in tick_labels], rotation=rotation, ha="right")


def _apply_sparse_numeric_x_ticks(ax: plt.Axes, values: list[Any], *, max_labels: int = 8, rotation: int = 30) -> None:
    positions, tick_labels = _thin_category_labels(values, max_labels=max_labels)
    tick_values = [values[pos] for pos in positions]
    ax.set_xticks(tick_values)
    ax.set_xticklabels([_wrap_label(label, width=14) for label in tick_labels], rotation=rotation, ha="right")


def _apply_sparse_heatmap_ticks(
    ax: plt.Axes,
    *,
    x_labels: list[Any],
    y_labels: list[Any],
    max_x_labels: int = 10,
    max_y_labels: int = 12,
) -> None:
    x_positions, x_tick_labels = _thin_category_labels(x_labels, max_labels=max_x_labels)
    ax.set_xticks([pos + 0.5 for pos in x_positions])
    ax.set_xticklabels([_wrap_label(label, width=14) for label in x_tick_labels], rotation=35, ha="right")
    y_positions, y_tick_labels = _thin_category_labels(y_labels, max_labels=max_y_labels)
    ax.set_yticks([pos + 0.5 for pos in y_positions])
    ax.set_yticklabels([_wrap_label(label, width=16) for label in y_tick_labels], rotation=0)


def _collapse_pivot_columns(
    pivot: pd.DataFrame,
    *,
    max_columns: int = 8,
    other_label: str = "Other",
) -> pd.DataFrame:
    if pivot.empty or len(pivot.columns) <= max_columns:
        return pivot
    kept = list(pivot.columns[: max_columns - 1])
    other = pivot.drop(columns=kept).sum(axis=1)
    collapsed = pivot[kept].copy()
    collapsed[other_label] = other
    return collapsed


def _select_primary_metrics(
    metrics: list[str],
    omnibus_df: pd.DataFrame | None,
    *,
    max_metrics: int,
) -> list[str]:
    ordered = [str(metric) for metric in metrics if str(metric).strip()]
    if len(ordered) <= max_metrics:
        return ordered
    if omnibus_df is not None and not omnibus_df.empty and {"metric", "p_adj"}.issubset(omnibus_df.columns):
        ranked = omnibus_df.copy()
        ranked["metric"] = ranked["metric"].astype(str)
        ranked["p_adj"] = _coerce_numeric(ranked["p_adj"])
        if "epsilon_squared" in ranked.columns:
            ranked["epsilon_squared"] = _coerce_numeric(ranked["epsilon_squared"]).fillna(0.0)
        else:
            ranked["epsilon_squared"] = 0.0
        ranked = ranked[ranked["metric"].isin(ordered)].dropna(subset=["p_adj"])
        if not ranked.empty:
            ranked = ranked.sort_values(by=["p_adj", "epsilon_squared"], ascending=[True, False], kind="stable")
            selected: list[str] = []
            seen: set[str] = set()
            for metric in ranked["metric"].tolist():
                if metric in seen:
                    continue
                seen.add(metric)
                selected.append(metric)
                if len(selected) >= max_metrics:
                    return selected
    return ordered[:max_metrics]


def _rank_omnibus_metrics(omnibus_df: pd.DataFrame | None, *, max_metrics: int = 8) -> list[str]:
    if omnibus_df is None or omnibus_df.empty or "metric" not in omnibus_df.columns or "p_adj" not in omnibus_df.columns:
        return []
    ranked = omnibus_df.copy()
    ranked["metric"] = ranked["metric"].astype(str)
    ranked["p_adj"] = _coerce_numeric(ranked["p_adj"])
    if "epsilon_squared" in ranked.columns:
        ranked["epsilon_squared"] = _coerce_numeric(ranked["epsilon_squared"]).fillna(0.0)
    else:
        ranked["epsilon_squared"] = 0.0
    ranked = ranked.dropna(subset=["p_adj"]).sort_values(by=["p_adj", "epsilon_squared"], ascending=[True, False], kind="stable")
    out: list[str] = []
    seen: set[str] = set()
    for metric in ranked["metric"].tolist():
        if metric in seen:
            continue
        seen.add(metric)
        out.append(metric)
        if len(out) >= max_metrics:
            break
    return out


def _prepare_group_summary_matrix(
    video_summary_df: pd.DataFrame,
    *,
    metrics: list[str],
    group_col: str,
) -> pd.DataFrame:
    if video_summary_df is None or video_summary_df.empty or group_col not in video_summary_df.columns or not metrics:
        return pd.DataFrame()
    work = video_summary_df[[group_col] + [metric for metric in metrics if metric in video_summary_df.columns]].copy()
    work[group_col] = work[group_col].fillna("").astype(str).str.strip().replace("", "Ungrouped")
    agg = work.groupby(group_col, dropna=False).mean(numeric_only=True)
    if agg.empty:
        return pd.DataFrame()
    standardized = agg.copy()
    for column in standardized.columns:
        series = _coerce_numeric(standardized[column])
        mean = float(series.mean()) if not series.dropna().empty else 0.0
        std = float(series.std(ddof=0)) if series.notna().sum() > 1 else 0.0
        standardized[column] = 0.0 if std <= 0 else (series - mean) / std
    standardized = standardized.T
    standardized.index = [_display_label(label) for label in standardized.index]
    return standardized


def _select_dashboard_module_key(module_tables: dict[str, pd.DataFrame]) -> str:
    priority = [
        "event_aligned_windows__summary",
        "temporal_trends__duration",
        "preference_indices__zone_pairwise",
        "preference_indices__object_pairwise",
        "latency_metrics__zone_latency",
        "latency_metrics__object_latency",
        "activity_budgets__global",
        "visit_structure__zone_summary",
        "visit_structure__object_summary",
        "behavior_transitions__global_matrix",
    ]
    for table_key in priority:
        df = module_tables.get(table_key)
        if df is not None and not df.empty:
            return table_key
    return ""


def _render_dashboard_module_panel(
    ax: plt.Axes,
    *,
    module_tables: dict[str, pd.DataFrame],
    group_col: str,
) -> str:
    table_key = _select_dashboard_module_key(module_tables)
    if not table_key:
        ax.axis("off")
        ax.text(0.0, 0.5, "No assay-specific module table available for dashboard rendering.", fontsize=11, ha="left", va="center")
        return "Assay-specific signal unavailable"
    df = module_tables.get(table_key)
    assert df is not None
    if table_key == "event_aligned_windows__summary":
        work = df.copy()
        required = {group_col, "Relative Time (s)", "Behavior Fraction", "Behavior"}
        if not required.issubset(work.columns):
            ax.axis("off")
            return "Event-aligned window"
        work[group_col] = work[group_col].fillna("").astype(str).str.strip()
        work["Behavior"] = work["Behavior"].fillna("").astype(str).str.strip()
        work["Behavior Fraction"] = _coerce_numeric(work["Behavior Fraction"])
        work["Relative Time (s)"] = _coerce_numeric(work["Relative Time (s)"])
        work = work.dropna(subset=["Behavior Fraction", "Relative Time (s)"])
        if work.empty:
            ax.axis("off")
            return "Event-aligned window"
        top_behavior = work.groupby("Behavior", dropna=False)["Behavior Fraction"].mean().sort_values(ascending=False).index[0]
        plot_df = work[work["Behavior"] == top_behavior]
        sns.lineplot(
            data=plot_df,
            x="Relative Time (s)",
            y="Behavior Fraction",
            hue=group_col,
            estimator="mean",
            errorbar=("ci", 95),
            linewidth=2.2,
            ax=ax,
        )
        ax.set_title(f"Event-aligned behavior: {_display_label(str(top_behavior))}", loc="left", fontweight="bold")
        ax.set_xlabel("Relative time (s)")
        ax.set_ylabel("Mean behavior fraction")
        legend = ax.get_legend()
        if legend is not None:
            legend.set_frame_on(False)
        return "Event-aligned module signal"
    if table_key == "temporal_trends__duration":
        work = df.copy()
        required = {group_col, "Time Bin Start (s)", "Total_Duration_s", "Behavior"}
        if not required.issubset(work.columns):
            ax.axis("off")
            return "Temporal trend"
        work[group_col] = work[group_col].fillna("").astype(str).str.strip()
        work["Behavior"] = work["Behavior"].fillna("").astype(str).str.strip()
        work["Time Bin Start (s)"] = _coerce_numeric(work["Time Bin Start (s)"])
        work["Total_Duration_s"] = _coerce_numeric(work["Total_Duration_s"])
        work = work.dropna(subset=["Time Bin Start (s)", "Total_Duration_s"])
        if work.empty:
            ax.axis("off")
            return "Temporal trend"
        top_behavior = work.groupby("Behavior", dropna=False)["Total_Duration_s"].mean().sort_values(ascending=False).index[0]
        plot_df = work[work["Behavior"] == top_behavior]
        sns.lineplot(
            data=plot_df,
            x="Time Bin Start (s)",
            y="Total_Duration_s",
            hue=group_col,
            estimator="mean",
            errorbar=("ci", 95),
            linewidth=2.2,
            ax=ax,
        )
        ax.set_title(f"Temporal trend: {_display_label(str(top_behavior))}", loc="left", fontweight="bold")
        ax.set_xlabel("Time bin start (s)")
        ax.set_ylabel("Mean duration (s)")
        legend = ax.get_legend()
        if legend is not None:
            legend.set_frame_on(False)
        return "Temporal trend module signal"
    if table_key in {"preference_indices__zone_pairwise", "preference_indices__object_pairwise"}:
        work = df.copy()
        metric_column = "Time Preference Index"
        if metric_column not in work.columns or group_col not in work.columns:
            ax.axis("off")
            return "Preference indices"
        target_a = "Target A" if "Target A" in work.columns else ""
        target_b = "Target B" if "Target B" in work.columns else ""
        if target_a and target_b:
            work["comparison_label"] = work[target_a].astype(str) + " vs " + work[target_b].astype(str)
        else:
            work["comparison_label"] = "Preference Index"
        top_label = work["comparison_label"].value_counts().index[0]
        plot_df = work[work["comparison_label"] == top_label].copy()
        plot_df[metric_column] = _coerce_numeric(plot_df[metric_column])
        plot_df = plot_df.dropna(subset=[metric_column])
        sns.boxplot(data=plot_df, x=group_col, y=metric_column, ax=ax, color="#dbeafe", showfliers=False)
        sns.stripplot(data=plot_df, x=group_col, y=metric_column, ax=ax, color="#0f172a", alpha=0.75, size=5)
        ax.axhline(0.0, color="#334155", linewidth=1.0, linestyle="--")
        ax.set_title(f"Preference: {_wrap_label(str(top_label), width=28)}", loc="left", fontweight="bold")
        ax.set_xlabel("Group")
        ax.set_ylabel("Preference index")
        return "Preference index module signal"
    if table_key in {"latency_metrics__zone_latency", "latency_metrics__object_latency"}:
        work = df.copy()
        metric_column = "First Entry Latency (s)" if "First Entry Latency (s)" in work.columns else "First Interaction Latency (s)"
        if metric_column not in work.columns or group_col not in work.columns:
            ax.axis("off")
            return "Latency metrics"
        entity_col = "Target Name" if "Target Name" in work.columns else "Object ROI" if "Object ROI" in work.columns else ""
        if entity_col:
            top_entity = work[entity_col].astype(str).value_counts().index[0]
            work = work[work[entity_col].astype(str) == top_entity].copy()
        work[metric_column] = _coerce_numeric(work[metric_column])
        work = work.dropna(subset=[metric_column])
        sns.boxplot(data=work, x=group_col, y=metric_column, ax=ax, color="#fde68a", showfliers=False)
        sns.stripplot(data=work, x=group_col, y=metric_column, ax=ax, color="#92400e", alpha=0.75, size=5)
        ax.set_title("Latency summary", loc="left", fontweight="bold")
        ax.set_xlabel("Group")
        ax.set_ylabel(_display_label(metric_column))
        return "Latency module signal"
    if table_key == "activity_budgets__global":
        work = df.copy()
        metric_column = "Proportion_of_Session" if "Proportion_of_Session" in work.columns else "Total_Time_s"
        if metric_column not in work.columns or group_col not in work.columns or "Behavior" not in work.columns:
            ax.axis("off")
            return "Activity budget"
        work[metric_column] = _coerce_numeric(work[metric_column])
        summary = work.groupby([group_col, "Behavior"], dropna=False)[metric_column].mean().reset_index()
        pivot = summary.pivot(index=group_col, columns="Behavior", values=metric_column).fillna(0.0)
        if pivot.empty:
            ax.axis("off")
            return "Activity budget"
        pivot = _collapse_pivot_columns(pivot[pivot.sum(axis=0).sort_values(ascending=False).index.tolist()], max_columns=6)
        palette = sns.color_palette("Set2", n_colors=max(3, len(pivot.columns)))
        lefts = np.zeros(len(pivot.index), dtype=float)
        y_positions = np.arange(len(pivot.index))
        for idx, behavior in enumerate(pivot.columns):
            values = pivot[behavior].to_numpy(dtype=float)
            ax.barh(y_positions, values, left=lefts, color=palette[idx % len(palette)], label=str(behavior), edgecolor="white", linewidth=0.8)
            lefts += values
        ax.set_yticks(y_positions)
        ax.set_yticklabels([_wrap_label(label, width=14) for label in pivot.index])
        ax.set_title("Activity budget", loc="left", fontweight="bold")
        ax.set_xlabel("Mean proportion of session")
        legend = ax.legend(frameon=False, loc="lower right", ncol=2, title="Behavior")
        if legend is not None:
            legend.get_title().set_fontsize(9)
        return "Activity budget module signal"
    ax.axis("off")
    ax.text(0.0, 0.5, f"Module preview not implemented for {table_key}.", fontsize=11, ha="left", va="center")
    return _display_label(table_key.replace("__", " "))


def _render_group_stats_overview(
    omnibus_df: pd.DataFrame | None,
    pairwise_df: pd.DataFrame | None,
    *,
    output_dir: Path,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    if (omnibus_df is None or omnibus_df.empty) and (pairwise_df is None or pairwise_df.empty):
        return records

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.6), gridspec_kw={"width_ratios": [1.2, 1.35, 1.1]})
    ax_omnibus, ax_pairwise, ax_rank = axes

    if omnibus_df is not None and not omnibus_df.empty and {"metric", "p_adj"}.issubset(omnibus_df.columns):
        work = omnibus_df.copy()
        if "factor" in work.columns:
            work["factor"] = work["factor"].fillna("group").astype(str)
        else:
            work["factor"] = "group"
        work["metric"] = work["metric"].astype(str)
        work["p_adj"] = _coerce_numeric(work["p_adj"])
        work = work.dropna(subset=["p_adj"])
        if not work.empty:
            top_metrics = _rank_omnibus_metrics(work, max_metrics=8)
            # Do NOT fillna(0.0) here: a missing p_adj means the test was
            # skipped or crashed. Filling with 0 would render as -log10(0)
            # = "totally not significant" — visually identical to a real
            # not-significant test. Leave NaN and let seaborn render those
            # cells as masked/blank so missing data is distinguishable.
            heat = (
                work[work["metric"].isin(top_metrics)]
                .assign(score=lambda df: -np.log10(df["p_adj"].clip(lower=1e-12)))
                .pivot_table(index="metric", columns="factor", values="score", aggfunc="max")
            )
            heat = heat.reindex(top_metrics)
            heat.index = [_display_label(metric) for metric in heat.index]
            sns.heatmap(
                heat,
                cmap="crest",
                linewidths=0.5,
                linecolor="white",
                cbar_kws={"label": "-log10(FDR p)"},
                ax=ax_omnibus,
                mask=heat.isna(),
            )
            ax_omnibus.set_title("Omnibus significance", loc="left", fontweight="bold")
            ax_omnibus.set_xlabel("Factor")
            ax_omnibus.set_ylabel("Metric")
        else:
            ax_omnibus.axis("off")
            ax_omnibus.text(0.0, 0.5, "No omnibus statistics available.", ha="left", va="center")
    else:
        ax_omnibus.axis("off")
        ax_omnibus.text(0.0, 0.5, "No omnibus statistics available.", ha="left", va="center")

    if pairwise_df is not None and not pairwise_df.empty and {"metric", "group_a", "group_b", "cliffs_delta"}.issubset(pairwise_df.columns):
        work = pairwise_df.copy()
        work["metric"] = work["metric"].astype(str)
        work["cliffs_delta"] = _coerce_numeric(work["cliffs_delta"])
        work["pair_label"] = work["group_a"].astype(str) + " vs " + work["group_b"].astype(str)
        work["abs_effect"] = work["cliffs_delta"].abs()
        top_metrics = (
            work.groupby("metric", dropna=False)["abs_effect"]
            .max()
            .sort_values(ascending=False)
            .head(8)
            .index.tolist()
        )
        # Do NOT fillna(0.0): a missing Cliff's delta means the contrast
        # could not be computed (test crashed or pair excluded), which is
        # different from a true effect of zero. Mask NaN cells so they
        # render as blank background, distinguishable from a 0-coloured cell
        # in the centred 'vlag' palette.
        heat = (
            work[work["metric"].isin(top_metrics)]
            .pivot_table(index="metric", columns="pair_label", values="cliffs_delta", aggfunc="mean")
        )
        if not heat.empty:
            heat = heat.reindex(top_metrics)
            heat.index = [_display_label(metric) for metric in heat.index]
            sns.heatmap(
                heat,
                cmap="vlag",
                center=0.0,
                linewidths=0.5,
                linecolor="white",
                cbar_kws={"label": "Cliff's delta"},
                ax=ax_pairwise,
                mask=heat.isna(),
            )
            ax_pairwise.set_title("Pairwise effect sizes", loc="left", fontweight="bold")
            ax_pairwise.set_xlabel("Contrast")
            ax_pairwise.set_ylabel("Metric")
        else:
            ax_pairwise.axis("off")
            ax_pairwise.text(0.0, 0.5, "No pairwise contrasts available.", ha="left", va="center")
    else:
        ax_pairwise.axis("off")
        ax_pairwise.text(0.0, 0.5, "No pairwise contrasts available.", ha="left", va="center")

    ranked_rows: list[tuple[str, float, str]] = []
    if omnibus_df is not None and not omnibus_df.empty and {"metric", "p_adj"}.issubset(omnibus_df.columns):
        for row in omnibus_df.to_dict(orient="records"):
            p_adj = pd.to_numeric(pd.Series([row.get("p_adj")]), errors="coerce").iloc[0]
            if np.isfinite(p_adj):
                ranked_rows.append((_display_label(str(row.get("metric", ""))), float(-np.log10(max(p_adj, 1e-12))), "Omnibus"))
    if pairwise_df is not None and not pairwise_df.empty and {"metric", "cliffs_delta"}.issubset(pairwise_df.columns):
        for row in pairwise_df.to_dict(orient="records"):
            effect = pd.to_numeric(pd.Series([row.get("cliffs_delta")]), errors="coerce").iloc[0]
            if np.isfinite(effect):
                pair_label = f"{row.get('group_a', '')} vs {row.get('group_b', '')}".strip()
                ranked_rows.append((_display_label(str(row.get("metric", ""))), float(abs(effect)), pair_label or "Pairwise"))
    if ranked_rows:
        rank_df = pd.DataFrame(ranked_rows, columns=["metric", "score", "label"])
        rank_df = rank_df.sort_values("score", ascending=False).head(8).iloc[::-1]
        palette = ["#0f766e" if label == "Omnibus" else "#7c3aed" for label in rank_df["label"]]
        ax_rank.barh(np.arange(len(rank_df)), rank_df["score"].to_numpy(dtype=float), color=palette, edgecolor="white", linewidth=0.8)
        ax_rank.set_yticks(np.arange(len(rank_df)))
        ax_rank.set_yticklabels([_wrap_label(label, width=22) for label in rank_df["metric"].tolist()])
        ax_rank.set_title("Top findings", loc="left", fontweight="bold")
        ax_rank.set_xlabel("Score")
        for idx, row in enumerate(rank_df.itertuples(index=False)):
            ax_rank.text(float(row.score), idx, f"  {row.label}", va="center", ha="left", fontsize=9)
    else:
        ax_rank.axis("off")
        ax_rank.text(0.0, 0.5, "No ranked findings available.", ha="left", va="center")

    fig.suptitle("Group Statistics Overview", x=0.04, y=0.99, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    base_path = output_dir / "group_stats_overview"
    for saved_path in _save_figure(fig, base_path):
        records.append(
            _manifest_row(
                scope="intergroup",
                figure_type="group_stats_overview",
                factor="group_stats",
                title="Group Statistics Overview",
                source="group_stats",
                path=saved_path,
            )
        )
    return records


def _render_batch_dashboard(
    *,
    video_summary_df: pd.DataFrame,
    omnibus_df: pd.DataFrame | None,
    pairwise_df: pd.DataFrame | None,
    module_tables: dict[str, pd.DataFrame],
    assay_preset_key: str,
    output_dir: Path,
    group_col: str,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    if video_summary_df is None or video_summary_df.empty:
        return records
    metrics = _resolve_numeric_metrics(video_summary_df, exclude=_SUMMARY_EXCLUDE_COLUMNS, min_non_na=2)
    if not metrics:
        return records
    summary_metrics = _select_primary_metrics(metrics, omnibus_df, max_metrics=6)
    fig = plt.figure(figsize=(18.5, 11.5))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.18)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_omnibus = fig.add_subplot(gs[0, 1])
    ax_pairwise = fig.add_subplot(gs[1, 0])
    ax_module = fig.add_subplot(gs[1, 1])

    group_heat = _prepare_group_summary_matrix(video_summary_df, metrics=summary_metrics, group_col=group_col)
    if not group_heat.empty:
        sns.heatmap(group_heat, cmap="vlag", center=0.0, linewidths=0.5, linecolor="white", cbar_kws={"label": "Within-metric z-score"}, ax=ax_heat)
        ax_heat.set_title("Group summary matrix", loc="left", fontweight="bold")
        ax_heat.set_xlabel("Group")
        ax_heat.set_ylabel("Metric")
    else:
        ax_heat.axis("off")
        ax_heat.text(0.0, 0.5, "Not enough grouped summary metrics for dashboard heatmap.", ha="left", va="center")

    if omnibus_df is not None and not omnibus_df.empty and {"metric", "p_adj"}.issubset(omnibus_df.columns):
        work = omnibus_df.copy()
        work["metric"] = work["metric"].astype(str)
        work["p_adj"] = _coerce_numeric(work["p_adj"])
        work = work.dropna(subset=["p_adj"])
        ranked_metrics = _rank_omnibus_metrics(work, max_metrics=8)
        plot_df = (
            work[work["metric"].isin(ranked_metrics)]
            .groupby("metric", dropna=False)["p_adj"]
            .min()
            .reset_index()
            .assign(score=lambda df: -np.log10(df["p_adj"].clip(lower=1e-12)))
            .sort_values("score", ascending=True)
        )
        sns.barplot(data=plot_df, x="score", y="metric", orient="h", color="#0f766e", ax=ax_omnibus)
        ax_omnibus.set_yticklabels([_wrap_label(_display_label(label.get_text()), width=24) for label in ax_omnibus.get_yticklabels()])
        ax_omnibus.set_title("Top omnibus signals", loc="left", fontweight="bold")
        ax_omnibus.set_xlabel("-log10(FDR p)")
        ax_omnibus.set_ylabel("Metric")
    else:
        ax_omnibus.axis("off")
        ax_omnibus.text(0.0, 0.5, "No omnibus statistics available for dashboard ranking.", ha="left", va="center")

    if pairwise_df is not None and not pairwise_df.empty and {"metric", "group_a", "group_b", "cliffs_delta"}.issubset(pairwise_df.columns):
        work = pairwise_df.copy()
        work["cliffs_delta"] = _coerce_numeric(work["cliffs_delta"])
        work["pair_label"] = work["group_a"].astype(str) + " vs " + work["group_b"].astype(str)
        work["metric"] = work["metric"].astype(str)
        work["abs_effect"] = work["cliffs_delta"].abs()
        plot_df = work.sort_values("abs_effect", ascending=False).head(10).iloc[::-1]
        palette = ["#1d4ed8" if value >= 0 else "#b91c1c" for value in plot_df["cliffs_delta"].to_numpy(dtype=float)]
        ax_pairwise.barh(np.arange(len(plot_df)), plot_df["cliffs_delta"].to_numpy(dtype=float), color=palette, edgecolor="white", linewidth=0.8)
        ax_pairwise.axvline(0.0, color="#111827", linewidth=1.0, alpha=0.8)
        ax_pairwise.set_yticks(np.arange(len(plot_df)))
        ax_pairwise.set_yticklabels([
            _wrap_label(f"{_display_label(metric)} | {pair}", width=28)
            for metric, pair in zip(plot_df["metric"].tolist(), plot_df["pair_label"].tolist())
        ])
        ax_pairwise.set_title("Strongest pairwise effects", loc="left", fontweight="bold")
        ax_pairwise.set_xlabel("Cliff's delta")
        ax_pairwise.set_ylabel("Metric / contrast")
    else:
        ax_pairwise.axis("off")
        ax_pairwise.text(0.0, 0.5, "No pairwise contrasts available for dashboard.", ha="left", va="center")

    module_subtitle = _render_dashboard_module_panel(ax_module, module_tables=module_tables, group_col=group_col)
    assay_label = getattr(get_assay_preset(assay_preset_key), "label", "Custom / mixed")
    fig.suptitle(f"Batch Dashboard | {assay_label}", x=0.04, y=0.99, ha="left", fontweight="bold")
    fig.text(0.04, 0.965, module_subtitle, ha="left", fontsize=11, color="#475569")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    base_path = output_dir / "batch_dashboard_overview"
    for saved_path in _save_figure(fig, base_path):
        records.append(
            _manifest_row(
                scope="group",
                figure_type="batch_dashboard_overview",
                factor="batch_dashboard",
                title=f"Batch Dashboard | {assay_label}",
                source="dashboard",
                path=saved_path,
            )
        )
    return records


def _iter_manifest_paths(payload: Any, *, key_prefix: str = "") -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            prefix = f"{key_prefix}.{key}" if key_prefix else str(key)
            rows.extend(_iter_manifest_paths(value, key_prefix=prefix))
        return rows
    if isinstance(payload, (list, tuple, set)):
        for index, value in enumerate(payload):
            prefix = f"{key_prefix}.{index}" if key_prefix else str(index)
            rows.extend(_iter_manifest_paths(value, key_prefix=prefix))
        return rows
    if isinstance(payload, (str, Path)):
        rows.append((key_prefix, str(payload)))
    return rows


def _collect_existing_analytics_figures(video_results: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for result in list(video_results or []):
        manifest_path = str(result.get("run_manifest_json", "") or "").strip()
        manifest = _safe_read_json(manifest_path)
        if not manifest:
            continue
        base_dir = Path(manifest_path).expanduser().resolve().parent
        outputs = manifest.get("outputs", {})
        for key_path, raw_path in _iter_manifest_paths(outputs):
            resolved = _resolve_file_path(raw_path, base_dir=base_dir)
            suffix = resolved.suffix.lower()
            if suffix not in _IMAGE_SUFFIXES or not resolved.is_file():
                continue
            norm_path = str(resolved).lower()
            if norm_path in seen_paths:
                continue
            seen_paths.add(norm_path)
            title_stub = key_path.replace(".", " / ").replace("_", " ").strip() or resolved.stem
            records.append(
                _manifest_row(
                    scope="individual",
                    figure_type="existing_analytics_plot",
                    video_id=str(result.get("video_id", "") or ""),
                    video_name=str(result.get("video_name", "") or ""),
                    group=str(result.get("group", "") or ""),
                    subject_id=str(result.get("subject_id", "") or ""),
                    time_point=str(result.get("time_point", "") or ""),
                    title=f"{result.get('video_name', '')}: {title_stub}",
                    source="run_manifest",
                    path=resolved,
                )
            )
    return records


def render_video_quicklook_bundle(
    video_result: dict[str, Any],
    *,
    output_dir: str | Path,
) -> list[dict[str, str]]:
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summary_df = _safe_read_csv(video_result.get("summary_bouts_csv", ""))
    detailed_df = _safe_read_csv(video_result.get("detailed_bouts_csv", ""))
    roi_df = _safe_read_csv(video_result.get("roi_overview_csv", ""))
    object_df = _safe_read_csv(video_result.get("object_interactions_csv", ""))
    metrics_df = _safe_read_csv(video_result.get("metrics_summary_by_track_csv", ""))
    if metrics_df.empty:
        metrics_df = _safe_read_csv(video_result.get("metrics_csv", ""))

    if summary_df.empty and detailed_df.empty and roi_df.empty and object_df.empty and metrics_df.empty:
        return []

    _apply_publication_theme()
    fig = plt.figure(figsize=(13.0, 8.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.1, 1.0])
    ax_behavior = fig.add_subplot(grid[0, 0])
    ax_space = fig.add_subplot(grid[0, 1])
    ax_motion = fig.add_subplot(grid[1, 0])
    ax_notes = fig.add_subplot(grid[1, 1])

    plotted_behavior = False
    if not summary_df.empty and {"Class Label", "Number of Bouts"}.issubset(summary_df.columns):
        behavior_df = summary_df[["Class Label", "Number of Bouts"]].copy()
        behavior_df["Number of Bouts"] = _coerce_numeric(behavior_df["Number of Bouts"])
        behavior_df["Class Label"] = behavior_df["Class Label"].fillna("").astype(str).str.strip()
        behavior_df = behavior_df.dropna(subset=["Number of Bouts"])
        behavior_df = behavior_df[behavior_df["Class Label"] != ""]
        behavior_df = behavior_df.groupby("Class Label", dropna=False)["Number of Bouts"].sum().sort_values(ascending=True)
        if not behavior_df.empty:
            ax_behavior.barh(behavior_df.index.astype(str), behavior_df.to_numpy(dtype=float), color="#0f766e")
            ax_behavior.set_title("Bout counts by behavior", loc="left", fontweight="bold")
            ax_behavior.set_xlabel("Number of bouts")
            ax_behavior.set_ylabel("Behavior")
            plotted_behavior = True
    if not plotted_behavior and not detailed_df.empty and "Behavior" in detailed_df.columns:
        behavior_df = (
            detailed_df["Behavior"].fillna("").astype(str).str.strip().loc[lambda series: series != ""].value_counts().sort_values(ascending=True)
        )
        if not behavior_df.empty:
            ax_behavior.barh(behavior_df.index.astype(str), behavior_df.to_numpy(dtype=float), color="#0f766e")
            ax_behavior.set_title("Bout counts by behavior", loc="left", fontweight="bold")
            ax_behavior.set_xlabel("Number of bouts")
            ax_behavior.set_ylabel("Behavior")
            plotted_behavior = True
    if not plotted_behavior:
        ax_behavior.axis("off")
        ax_behavior.text(0.0, 0.6, "No bout summary data", ha="left", va="center", fontsize=11, color="#64748b")

    space_rows: list[tuple[str, float]] = []
    if not roi_df.empty and "ROI Name" in roi_df.columns:
        roi_metric_col = "Time in ROI (s)" if "Time in ROI (s)" in roi_df.columns else "Entries" if "Entries" in roi_df.columns else ""
        if roi_metric_col:
            tmp = roi_df[["ROI Name", roi_metric_col]].copy()
            tmp[roi_metric_col] = _coerce_numeric(tmp[roi_metric_col])
            tmp["ROI Name"] = tmp["ROI Name"].fillna("").astype(str).str.strip()
            tmp = tmp.dropna(subset=[roi_metric_col])
            tmp = tmp[tmp["ROI Name"] != ""]
            for row in tmp.groupby("ROI Name", dropna=False)[roi_metric_col].sum().sort_values(ascending=False).head(5).items():
                space_rows.append((f"ROI: {row[0]}", float(row[1])))
    if not object_df.empty and "Object ROI" in object_df.columns:
        object_metric_col = (
            "Time Interacting (s)"
            if "Time Interacting (s)" in object_df.columns
            else "Entries"
            if "Entries" in object_df.columns
            else ""
        )
        if object_metric_col:
            tmp = object_df[["Object ROI", object_metric_col]].copy()
            tmp[object_metric_col] = _coerce_numeric(tmp[object_metric_col])
            tmp["Object ROI"] = tmp["Object ROI"].fillna("").astype(str).str.strip()
            tmp = tmp.dropna(subset=[object_metric_col])
            tmp = tmp[tmp["Object ROI"] != ""]
            for row in tmp.groupby("Object ROI", dropna=False)[object_metric_col].sum().sort_values(ascending=False).head(5).items():
                space_rows.append((f"Obj: {row[0]}", float(row[1])))
    if space_rows:
        labels = [label for label, _value in space_rows]
        values = [value for _label, value in space_rows]
        ax_space.barh(labels[::-1], values[::-1], color="#7c3aed")
        ax_space.set_title("ROI and object occupancy", loc="left", fontweight="bold")
        ax_space.set_xlabel("Time / count summary")
        ax_space.set_ylabel("ROI / object")
    else:
        ax_space.axis("off")
        ax_space.text(0.0, 0.6, "No ROI/object metrics", ha="left", va="center", fontsize=11, color="#64748b")

    motion_rows: list[tuple[str, float]] = []
    if not metrics_df.empty:
        if "track_id" in metrics_df.columns and "mean_speed_px_per_frame" in metrics_df.columns:
            tmp = metrics_df[["track_id", "mean_speed_px_per_frame"]].copy()
            tmp["mean_speed_px_per_frame"] = _coerce_numeric(tmp["mean_speed_px_per_frame"])
            tmp["track_id"] = pd.to_numeric(tmp["track_id"], errors="coerce")
            tmp = tmp.dropna(subset=["track_id", "mean_speed_px_per_frame"])
            for _, row in tmp.head(6).iterrows():
                motion_rows.append((f"Track {int(row['track_id'])}", float(row["mean_speed_px_per_frame"])))
        elif "mean_speed_px_per_frame" in metrics_df.columns:
            speed_values = _coerce_numeric(metrics_df["mean_speed_px_per_frame"]).dropna()
            if not speed_values.empty:
                motion_rows.append(("Mean speed", float(speed_values.mean())))
        if "turn_count" in metrics_df.columns:
            turn_values = _coerce_numeric(metrics_df["turn_count"]).dropna()
            if not turn_values.empty:
                motion_rows.append(("Mean turns", float(turn_values.mean())))
    if motion_rows:
        labels = [label for label, _value in motion_rows]
        values = [value for _label, value in motion_rows]
        ax_motion.barh(labels[::-1], values[::-1], color="#d97706")
        ax_motion.set_title("Motion quicklook", loc="left", fontweight="bold")
        ax_motion.set_xlabel("Per-track / mean metric")
        ax_motion.set_ylabel("Track / metric")
    else:
        ax_motion.axis("off")
        ax_motion.text(0.0, 0.6, "No kinematic summary", ha="left", va="center", fontsize=11, color="#64748b")

    ax_notes.axis("off")
    total_bouts = 0
    if not summary_df.empty and "Number of Bouts" in summary_df.columns:
        total_bouts = int(_coerce_numeric(summary_df["Number of Bouts"]).fillna(0).sum())
    elif not detailed_df.empty:
        total_bouts = int(len(detailed_df.index))
    total_roi_dwell = float(_coerce_numeric(roi_df["Time in ROI (s)"]).fillna(0).sum()) if not roi_df.empty and "Time in ROI (s)" in roi_df.columns else np.nan
    total_object_time = (
        float(_coerce_numeric(object_df["Time Interacting (s)"]).fillna(0).sum())
        if not object_df.empty and "Time Interacting (s)" in object_df.columns
        else np.nan
    )
    mean_speed = (
        float(_coerce_numeric(metrics_df["mean_speed_px_per_frame"]).dropna().mean())
        if not metrics_df.empty and "mean_speed_px_per_frame" in metrics_df.columns
        else np.nan
    )
    note_lines = [
        f"Video: {video_result.get('video_name', '') or video_result.get('video_id', '')}",
        f"Group: {video_result.get('group', '') or 'n/a'}",
        f"Subject: {video_result.get('subject_id', '') or 'n/a'}",
        f"Time point: {video_result.get('time_point', '') or 'n/a'}",
        "",
        f"Total bouts: {total_bouts}",
        f"Total ROI dwell (s): {_format_value(total_roi_dwell)}",
        f"Total object interaction (s): {_format_value(total_object_time)}",
        f"Mean speed (px/frame): {_format_value(mean_speed)}",
    ]
    ax_notes.text(0.0, 1.0, "\n".join(note_lines), ha="left", va="top", fontsize=10.5, family="DejaVu Sans Mono")

    title = str(video_result.get("video_name", "") or video_result.get("video_id", "") or "Video quicklook")
    fig.suptitle(f"{title} quicklook", x=0.05, y=0.99, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])

    video_key = f"{video_result.get('video_id', '')}_{video_result.get('video_name', '')}"
    base_path = output_root / f"{_slugify(video_key)}_quicklook"
    records: list[dict[str, str]] = []
    for saved_path in _save_figure(fig, base_path):
        records.append(
            _manifest_row(
                scope="individual",
                figure_type="video_quicklook",
                video_id=str(video_result.get("video_id", "") or ""),
                video_name=str(video_result.get("video_name", "") or ""),
                group=str(video_result.get("group", "") or ""),
                subject_id=str(video_result.get("subject_id", "") or ""),
                time_point=str(video_result.get("time_point", "") or ""),
                title=f"{title} quicklook",
                source="batch_quicklook",
                path=saved_path,
            )
        )
    return records


def _render_individual_summary_profiles(
    video_summary_df: pd.DataFrame,
    *,
    metrics: list[str],
    output_dir: Path,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    if video_summary_df is None or video_summary_df.empty or not metrics:
        return records

    cohort_metrics: dict[str, pd.Series] = {metric: _coerce_numeric(video_summary_df[metric]) for metric in metrics}
    for row in video_summary_df.to_dict(orient="records"):
        metric_rows: list[tuple[str, float, float]] = []
        for metric in metrics:
            raw_value = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
            if not np.isfinite(raw_value):
                continue
            cohort_values = cohort_metrics[metric].dropna()
            std = float(cohort_values.std(ddof=0)) if len(cohort_values) > 1 else 0.0
            mean = float(cohort_values.mean()) if not cohort_values.empty else float(raw_value)
            z_score = 0.0 if std <= 0 or not np.isfinite(std) else float((raw_value - mean) / std)
            metric_rows.append((metric, float(raw_value), z_score))

        if not metric_rows:
            continue

        labels = [_display_label(metric) for metric, _value, _z in metric_rows]
        z_scores = [z_score for _metric, _value, z_score in metric_rows]
        colors = ["#0f766e" if value >= 0 else "#b91c1c" for value in z_scores]
        height = max(5.0, 0.5 * len(metric_rows) + 1.5)
        fig, (ax_plot, ax_table) = plt.subplots(
            1,
            2,
            figsize=(12.5, height),
            gridspec_kw={"width_ratios": [1.8, 1.15]},
        )
        y_positions = np.arange(len(metric_rows))
        ax_plot.barh(y_positions, z_scores, color=colors, edgecolor="white", linewidth=0.8)
        ax_plot.axvline(0.0, color="#111827", linewidth=1.0, alpha=0.8)
        ax_plot.set_yticks(y_positions)
        ax_plot.set_yticklabels(labels)
        ax_plot.invert_yaxis()
        ax_plot.set_xlabel("Cohort z-score")
        ax_plot.set_ylabel("Metric")
        ax_plot.set_title("Relative metric profile", loc="left", fontweight="bold")
        for idx, value in enumerate(z_scores):
            ha = "left" if value >= 0 else "right"
            x_offset = 0.05 if value >= 0 else -0.05
            ax_plot.text(value + x_offset, idx, f"{value:+.2f}", va="center", ha=ha, fontsize=9)

        ax_table.axis("off")
        value_lines = [f"{_display_label(metric)}: {_format_value(raw_value)}" for metric, raw_value, _z in metric_rows]
        ax_table.text(
            0.0,
            1.0,
            "\n".join(value_lines),
            va="top",
            ha="left",
            fontsize=10,
            family="DejaVu Sans Mono",
        )

        title = str(row.get("video_name", "") or row.get("video_id", "") or "Video summary")
        subtitle_parts = [
            f"Group: {row.get('group', '') or 'n/a'}",
            f"Subject: {row.get('subject_id', '') or 'n/a'}",
            f"Time: {row.get('time_point', '') or 'n/a'}",
        ]
        fig.suptitle(title, x=0.06, y=0.985, ha="left", fontweight="bold")
        fig.text(0.06, 0.94, " | ".join(subtitle_parts), ha="left", fontsize=10, color="#475569")
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])

        video_key = f"{row.get('video_id', '')}_{row.get('video_name', '')}"
        base_path = output_dir / f"{_slugify(video_key)}_summary_profile"
        for saved_path in _save_figure(fig, base_path):
            records.append(
                _manifest_row(
                    scope="individual",
                    figure_type="video_summary_profile",
                    video_id=str(row.get("video_id", "") or ""),
                    video_name=str(row.get("video_name", "") or ""),
                    group=str(row.get("group", "") or ""),
                    subject_id=str(row.get("subject_id", "") or ""),
                    time_point=str(row.get("time_point", "") or ""),
                    title=title,
                    source="video_summary_df",
                    path=saved_path,
                )
            )
    return records


def _lookup_omnibus_padj(omnibus_df: pd.DataFrame | None, metric: str, *, factor: str = "group") -> float:
    if omnibus_df is None or omnibus_df.empty or "metric" not in omnibus_df.columns:
        return np.nan
    subset = omnibus_df[(omnibus_df["metric"] == metric) & (omnibus_df.get("factor") == factor)]
    if subset.empty or "p_adj" not in subset.columns:
        return np.nan
    return float(pd.to_numeric(subset["p_adj"], errors="coerce").dropna().iloc[0]) if subset["p_adj"].notna().any() else np.nan


def _render_group_distribution_plots(
    video_summary_df: pd.DataFrame,
    omnibus_df: pd.DataFrame,
    *,
    metrics: list[str],
    output_dir: Path,
    group_col: str,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    if video_summary_df is None or video_summary_df.empty or group_col not in video_summary_df.columns:
        return records

    for metric in metrics:
        plot_df = video_summary_df[[group_col, metric]].copy()
        plot_df[metric] = _coerce_numeric(plot_df[metric])
        plot_df[group_col] = plot_df[group_col].fillna("").astype(str).str.strip()
        plot_df = plot_df[(plot_df[group_col] != "") & plot_df[metric].notna()]
        if plot_df.empty or plot_df[group_col].nunique() < 2:
            continue

        order = (
            plot_df.groupby(group_col, dropna=False)[metric]
            .median()
            .sort_values(ascending=True)
            .index.astype(str)
            .tolist()
        )
        counts = plot_df[group_col].value_counts()
        display_order = [f"{label}\n(n={int(counts.get(label, 0))})" for label in order]
        label_map = dict(zip(order, display_order))
        plot_df["__group_label__"] = plot_df[group_col].map(label_map)

        fig, ax = plt.subplots(figsize=(max(7.6, 5.8), max(4.8, 0.72 * len(display_order) + 1.6)))
        sns.boxplot(
            data=plot_df,
            y="__group_label__",
            x=metric,
            ax=ax,
            color="#dbeafe",
            width=0.55,
            showfliers=False,
            order=display_order,
        )
        sns.stripplot(
            data=plot_df,
            y="__group_label__",
            x=metric,
            ax=ax,
            color="#0f172a",
            alpha=0.7,
            size=5,
            jitter=0.18,
            order=display_order,
        )
        mean_series = plot_df.groupby("__group_label__", dropna=False)[metric].mean().reindex(display_order)
        ax.scatter(
            mean_series.to_numpy(dtype=float),
            np.arange(len(display_order)),
            marker="D",
            s=46,
            color="#d97706",
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
            label="Mean",
        )

        p_adj = _lookup_omnibus_padj(omnibus_df, metric, factor=group_col)
        title = _display_label(metric)
        if np.isfinite(p_adj):
            title = f"{title} | FDR p={p_adj:.3g}"
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel(_display_label(metric))
        ax.set_ylabel(_display_label(group_col))
        ax.set_yticks(np.arange(len(display_order)))
        ax.set_yticklabels([_wrap_label(label, width=18) for label in display_order])
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()

        base_path = output_dir / f"{_slugify(metric)}_by_{_slugify(group_col)}"
        for saved_path in _save_figure(fig, base_path):
            records.append(
                _manifest_row(
                    scope="group",
                    figure_type="group_distribution",
                    metric=metric,
                    factor=group_col,
                    title=title,
                    source="video_summary_df",
                    path=saved_path,
                )
            )
    return records


def _render_group_timepoint_plots(
    video_summary_df: pd.DataFrame,
    *,
    metrics: list[str],
    output_dir: Path,
    group_col: str,
    time_col: str,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    if (
        video_summary_df is None
        or video_summary_df.empty
        or group_col not in video_summary_df.columns
        or time_col not in video_summary_df.columns
    ):
        return records

    time_levels = _ordered_labels(video_summary_df[time_col])
    if len(time_levels) < 2:
        return records

    palette = sns.color_palette("colorblind", n_colors=max(2, video_summary_df[group_col].nunique()))
    group_labels = [label for label in _ordered_labels(video_summary_df[group_col]) if label]
    color_map = {label: palette[idx % len(palette)] for idx, label in enumerate(group_labels)}

    for metric in metrics:
        plot_df = video_summary_df[[group_col, time_col, metric]].copy()
        plot_df[metric] = _coerce_numeric(plot_df[metric])
        plot_df[group_col] = plot_df[group_col].fillna("").astype(str).str.strip()
        plot_df[time_col] = plot_df[time_col].fillna("").astype(str).str.strip()
        plot_df = plot_df[(plot_df[group_col] != "") & (plot_df[time_col] != "") & plot_df[metric].notna()]
        if plot_df.empty or plot_df[time_col].nunique() < 2:
            continue

        agg = (
            plot_df.groupby([group_col, time_col], dropna=False)[metric]
            .agg(mean="mean", sem="sem", n="size")
            .reset_index()
        )
        if agg.empty:
            continue

        fig, ax = plt.subplots(figsize=(max(7.6, 0.95 * len(time_levels)), 5.2))
        x_positions = np.arange(len(time_levels))
        any_series = False
        for group_label in group_labels:
            group_df = agg[agg[group_col] == group_label].set_index(time_col).reindex(time_levels)
            if group_df["mean"].notna().sum() == 0:
                continue
            any_series = True
            means = group_df["mean"].to_numpy(dtype=float)
            sems = group_df["sem"].to_numpy(dtype=float)
            color = color_map.get(group_label, "#0f766e")
            ax.plot(x_positions, means, marker="o", linewidth=2.4, label=group_label, color=color, markersize=5.5)
            valid = np.isfinite(means) & np.isfinite(sems)
            if valid.any():
                ax.fill_between(
                    x_positions[valid],
                    means[valid] - sems[valid],
                    means[valid] + sems[valid],
                    color=color,
                    alpha=0.15,
                )

        if not any_series:
            plt.close(fig)
            continue

        _apply_sparse_x_labels(ax, time_levels, max_labels=8, rotation=30)
        ax.set_xlabel(_display_label(time_col))
        ax.set_ylabel(_display_label(metric))
        ax.set_title(f"{_display_label(metric)} over {_display_label(time_col)}", loc="left", fontweight="bold")
        ax.legend(frameon=False, loc="best", title=_display_label(group_col))
        fig.tight_layout()

        base_path = output_dir / f"{_slugify(metric)}_over_{_slugify(time_col)}"
        for saved_path in _save_figure(fig, base_path):
            records.append(
                _manifest_row(
                    scope="group",
                    figure_type="timecourse",
                    metric=metric,
                    factor=f"{group_col}:{time_col}",
                    title=f"{_display_label(metric)} over {_display_label(time_col)}",
                    source="video_summary_df",
                    path=saved_path,
                )
            )
    return records


def _render_omnibus_heatmap(omnibus_df: pd.DataFrame, *, output_dir: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    if omnibus_df is None or omnibus_df.empty or "p_adj" not in omnibus_df.columns or "metric" not in omnibus_df.columns:
        return records

    heat_df = omnibus_df.copy()
    if "factor" in heat_df.columns:
        heat_df = heat_df[heat_df["factor"] == "group"]
    heat_df["p_adj"] = _coerce_numeric(heat_df["p_adj"])
    heat_df = heat_df.dropna(subset=["metric", "p_adj"])
    if heat_df.empty:
        return records

    heat_df["neg_log10_padj"] = -np.log10(heat_df["p_adj"].clip(lower=1e-12))
    matrix = heat_df[["metric", "neg_log10_padj"]].drop_duplicates("metric").set_index("metric")
    annot = heat_df[["metric", "p_adj"]].drop_duplicates("metric").set_index("metric").reindex(matrix.index)
    matrix.index = [_display_label(metric) for metric in matrix.index]
    matrix.columns = ["-log10(FDR p)"]

    fig, ax = plt.subplots(figsize=(6.4, max(3.8, 0.45 * len(matrix))))
    sns.heatmap(
        matrix,
        annot=annot.round(4).to_numpy(),
        fmt="",
        cmap="crest",
        cbar_kws={"label": "-log10(FDR p)"},
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("Omnibus group-level significance", loc="left", fontweight="bold")
    ax.set_xlabel("Omnibus significance column")
    ax.set_ylabel("Metric")
    fig.tight_layout()

    base_path = output_dir / "omnibus_group_significance"
    for saved_path in _save_figure(fig, base_path):
        records.append(
            _manifest_row(
                scope="intergroup",
                figure_type="omnibus_heatmap",
                factor="group",
                title="Omnibus group-level significance",
                source="group_stats",
                path=saved_path,
            )
        )
    return records


def _render_pairwise_contrast_plots(pairwise_df: pd.DataFrame, *, output_dir: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    required_columns = {"metric", "group_a", "group_b", "cliffs_delta"}
    if pairwise_df is None or pairwise_df.empty or not required_columns.issubset(pairwise_df.columns):
        return records

    plot_df = pairwise_df.copy()
    if "factor" in plot_df.columns:
        plot_df = plot_df[plot_df["factor"] == "group"]
    plot_df["cliffs_delta"] = _coerce_numeric(plot_df["cliffs_delta"])
    plot_df["p_adj"] = _coerce_numeric(plot_df["p_adj"]) if "p_adj" in plot_df.columns else np.nan
    plot_df = plot_df.dropna(subset=["metric", "group_a", "group_b", "cliffs_delta"])
    if plot_df.empty:
        return records

    plot_df["pair_label"] = plot_df["group_a"].astype(str) + " vs " + plot_df["group_b"].astype(str)
    for pair_label, pair_df in plot_df.groupby("pair_label", dropna=False):
        pair_data = pair_df.copy()
        pair_data["metric_label"] = pair_data["metric"].map(_display_label)
        pair_data = pair_data.sort_values("cliffs_delta", key=lambda values: np.abs(values), ascending=True)
        height = max(4.6, 0.4 * len(pair_data) + 1.4)
        fig, ax = plt.subplots(figsize=(10.8, height))
        y_positions = np.arange(len(pair_data))
        colors = ["#0f766e" if value < 0.05 else "#64748b" for value in pair_data["p_adj"].fillna(1.0)]
        ax.axvline(0.0, color="#111827", linewidth=1.0, alpha=0.8)
        ax.scatter(
            pair_data["cliffs_delta"].to_numpy(dtype=float),
            y_positions,
            s=78,
            c=colors,
            edgecolors="white",
            linewidth=0.8,
            zorder=5,
        )
        ax.set_yticks(y_positions)
        ax.set_yticklabels([_wrap_label(label, width=26) for label in pair_data["metric_label"].tolist()])
        ax.set_xlabel("Cliff's delta")
        ax.set_ylabel("Metric")
        ax.set_title(f"Intergroup contrast: {pair_label}", loc="left", fontweight="bold")
        min_delta = float(pair_data["cliffs_delta"].min())
        max_delta = float(pair_data["cliffs_delta"].max())
        pad = max(0.12, 0.15 * max(abs(min_delta), abs(max_delta), 1.0))
        ax.set_xlim(min(-1.0, min_delta - pad), max(1.0, max_delta + pad))
        for idx, row in enumerate(pair_data.itertuples(index=False)):
            p_adj = getattr(row, "p_adj", np.nan)
            text = "FDR n/a" if not np.isfinite(p_adj) else f"FDR={p_adj:.3g}"
            ax.text(ax.get_xlim()[1] - 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0]), idx, text, ha="right", va="center", fontsize=9)
        ax.legend(
            handles=[
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#0f766e", markersize=8, label="FDR < 0.05"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#64748b", markersize=8, label="FDR >= 0.05"),
            ],
            frameon=False,
            loc="best",
        )
        fig.tight_layout()

        safe_pair = _slugify(pair_label)
        base_path = output_dir / f"{safe_pair}_cliffs_delta"
        first_group_a = str(pair_data["group_a"].iloc[0])
        first_group_b = str(pair_data["group_b"].iloc[0])
        for saved_path in _save_figure(fig, base_path):
            records.append(
                _manifest_row(
                    scope="intergroup",
                    figure_type="pairwise_contrast",
                    group_a=first_group_a,
                    group_b=first_group_b,
                    factor="group",
                    title=f"Intergroup contrast: {pair_label}",
                    source="group_stats",
                    path=saved_path,
                )
            )
    return records


def _render_temporal_module_figures(
    module_tables: dict[str, pd.DataFrame],
    *,
    output_dir: Path,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    df = module_tables.get("temporal_trends__duration")
    required = {"group", "Behavior", "Time Bin Start (s)", "Total_Duration_s"}
    if df is None or df.empty or not required.issubset(df.columns):
        return records

    plot_df = df[list(required)].copy()
    plot_df["Time Bin Start (s)"] = _coerce_numeric(plot_df["Time Bin Start (s)"])
    plot_df["Total_Duration_s"] = _coerce_numeric(plot_df["Total_Duration_s"])
    plot_df["group"] = plot_df["group"].fillna("").astype(str).str.strip()
    plot_df["Behavior"] = plot_df["Behavior"].fillna("").astype(str).str.strip()
    plot_df = plot_df.dropna(subset=["Time Bin Start (s)", "Total_Duration_s"])
    plot_df = plot_df[(plot_df["group"] != "") & (plot_df["Behavior"] != "")]
    if plot_df.empty:
        return records

    group_labels = [label for label in _ordered_labels(plot_df["group"]) if label]
    palette = sns.color_palette("colorblind", n_colors=max(2, len(group_labels)))
    color_map = {label: palette[idx % len(palette)] for idx, label in enumerate(group_labels)}

    for behavior, behavior_df in plot_df.groupby("Behavior", dropna=False):
        agg = (
            behavior_df.groupby(["group", "Time Bin Start (s)"], dropna=False)["Total_Duration_s"]
            .agg(mean="mean", sem="sem")
            .reset_index()
        )
        if agg.empty:
            continue
        time_bins = sorted(pd.unique(agg["Time Bin Start (s)"]))
        x_positions = np.asarray(time_bins, dtype=float)
        fig, ax = plt.subplots(figsize=(max(7.0, 0.9 * len(time_bins)), 4.8))
        plotted = False
        for group_label in group_labels:
            group_df = agg[agg["group"] == group_label].set_index("Time Bin Start (s)").reindex(time_bins)
            means = group_df["mean"].to_numpy(dtype=float)
            sems = group_df["sem"].to_numpy(dtype=float)
            if np.isfinite(means).sum() == 0:
                continue
            plotted = True
            color = color_map.get(group_label, "#0f766e")
            ax.plot(x_positions, means, marker="o", linewidth=2.4, label=group_label, color=color, markersize=5.5)
            valid = np.isfinite(means) & np.isfinite(sems)
            if valid.any():
                ax.fill_between(
                    x_positions[valid],
                    means[valid] - sems[valid],
                    means[valid] + sems[valid],
                    color=color,
                    alpha=0.15,
                )
        if not plotted:
            plt.close(fig)
            continue
        _apply_sparse_numeric_x_ticks(ax, list(time_bins), max_labels=8, rotation=28)
        ax.set_xlabel("Time bin start (s)")
        ax.set_ylabel("Mean bout duration in bin (s)")
        ax.set_title(f"Temporal trend: {_display_label(behavior)}", loc="left", fontweight="bold")
        ax.legend(frameon=False, loc="best", title="Group")
        fig.tight_layout()

        base_path = output_dir / f"temporal_trend_{_slugify(behavior)}"
        for saved_path in _save_figure(fig, base_path):
            records.append(
                _manifest_row(
                    scope="group",
                    figure_type="module_temporal_trend",
                    metric=str(behavior),
                    factor="temporal_trends",
                    title=f"Temporal trend: {_display_label(behavior)}",
                    source="module_tables.temporal_trends__duration",
                    path=saved_path,
                )
            )
    return records


def _render_activity_budget_module_figures(
    module_tables: dict[str, pd.DataFrame],
    *,
    output_dir: Path,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    df = module_tables.get("activity_budgets__global")
    required = {"group", "Behavior"}
    if df is None or df.empty or not required.issubset(df.columns):
        return records

    metric_column = "Proportion_of_Session" if "Proportion_of_Session" in df.columns else "Total_Time_s" if "Total_Time_s" in df.columns else ""
    if not metric_column:
        return records

    plot_df = df[["group", "Behavior", metric_column]].copy()
    plot_df[metric_column] = _coerce_numeric(plot_df[metric_column])
    plot_df["group"] = plot_df["group"].fillna("").astype(str).str.strip()
    plot_df["Behavior"] = plot_df["Behavior"].fillna("").astype(str).str.strip()
    plot_df = plot_df.dropna(subset=[metric_column])
    plot_df = plot_df[(plot_df["group"] != "") & (plot_df["Behavior"] != "")]
    if plot_df.empty:
        return records

    summary = plot_df.groupby(["group", "Behavior"], dropna=False)[metric_column].mean().reset_index()
    if summary.empty:
        return records
    pivot = summary.pivot(index="group", columns="Behavior", values=metric_column).fillna(0.0)
    if pivot.empty:
        return records

    behavior_order = pivot.sum(axis=0).sort_values(ascending=False).index.tolist()
    pivot = _collapse_pivot_columns(pivot[behavior_order], max_columns=8)
    palette = sns.color_palette("Set2", n_colors=max(3, len(behavior_order)))
    y_positions = np.arange(len(pivot.index))
    lefts = np.zeros(len(pivot.index), dtype=float)
    fig, ax = plt.subplots(figsize=(9.0, max(4.8, 0.75 * len(pivot.index) + 1.4)))
    for idx, behavior in enumerate(pivot.columns):
        values = pivot[behavior].to_numpy(dtype=float)
        ax.barh(
            y_positions,
            values,
            left=lefts,
            label=str(behavior),
            color=palette[idx % len(palette)],
            edgecolor="white",
            linewidth=0.8,
        )
        lefts += values
    ax.set_yticks(y_positions)
    ax.set_yticklabels([_wrap_label(label, width=16) for label in pivot.index])
    ax.set_xlabel("Mean proportion of session" if metric_column == "Proportion_of_Session" else "Mean total time (s)")
    ax.set_xlim(0.0, max(1.0 if metric_column == "Proportion_of_Session" else lefts.max() * 1.08, lefts.max() * 1.08))
    ax.set_title("Activity budget by group", loc="left", fontweight="bold")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left", title="Behavior", ncol=1)
    fig.tight_layout()

    base_path = output_dir / "activity_budget_by_group"
    for saved_path in _save_figure(fig, base_path):
        records.append(
            _manifest_row(
                scope="group",
                figure_type="module_activity_budget",
                factor="activity_budgets",
                title="Activity budget by group",
                source="module_tables.activity_budgets__global",
                path=saved_path,
            )
        )
    return records


def _find_roi_label_column(df: pd.DataFrame) -> str:
    for candidate in ("ROI Name", "Object ROI", "ROI", "Zone"):
        if candidate in df.columns:
            return candidate
    excluded = set(_MODULE_CONTEXT_COLUMNS) | {"Time Bin Start (s)", "Time Bin End (s)", "Time in ROI (s)", "Frame_Count", "Time Bin"}
    for column in df.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            continue
        return column
    return ""


def _render_roi_occupancy_module_figures(
    module_tables: dict[str, pd.DataFrame],
    *,
    output_dir: Path,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    df = module_tables.get("roi_time_heatmap__occupancy")
    if df is None or df.empty or "group" not in df.columns or "Time in ROI (s)" not in df.columns:
        return records
    roi_column = _find_roi_label_column(df)
    if not roi_column or "Time Bin Start (s)" not in df.columns:
        return records

    plot_df = df[["group", roi_column, "Time Bin Start (s)", "Time in ROI (s)"]].copy()
    plot_df["group"] = plot_df["group"].fillna("").astype(str).str.strip()
    plot_df[roi_column] = plot_df[roi_column].fillna("").astype(str).str.strip()
    plot_df["Time Bin Start (s)"] = _coerce_numeric(plot_df["Time Bin Start (s)"])
    plot_df["Time in ROI (s)"] = _coerce_numeric(plot_df["Time in ROI (s)"])
    plot_df = plot_df.dropna(subset=["Time Bin Start (s)", "Time in ROI (s)"])
    plot_df = plot_df[(plot_df["group"] != "") & (plot_df[roi_column] != "")]
    if plot_df.empty:
        return records

    for group_label, group_df in plot_df.groupby("group", dropna=False):
        heat_df = (
            group_df.groupby([roi_column, "Time Bin Start (s)"], dropna=False)["Time in ROI (s)"]
            .mean()
            .reset_index()
            .pivot(index=roi_column, columns="Time Bin Start (s)", values="Time in ROI (s)")
            .fillna(0.0)
        )
        if heat_df.empty:
            continue
        heat_df = heat_df.sort_index()
        heat_df = heat_df.reindex(sorted(heat_df.columns), axis=1)
        fig, ax = plt.subplots(figsize=(max(7.0, 0.65 * len(heat_df.columns)), max(4.6, 0.45 * len(heat_df.index))))
        sns.heatmap(heat_df, cmap="mako", linewidths=0.5, linecolor="white", cbar_kws={"label": "Mean time in ROI (s)"}, ax=ax)
        _apply_sparse_heatmap_ticks(
            ax,
            x_labels=list(heat_df.columns),
            y_labels=list(heat_df.index),
            max_x_labels=10,
            max_y_labels=10,
        )
        ax.set_title(f"ROI occupancy over time: {group_label}", loc="left", fontweight="bold")
        ax.set_xlabel("Time bin start (s)")
        ax.set_ylabel(_display_label(roi_column))
        fig.tight_layout()

        base_path = output_dir / f"roi_occupancy_{_slugify(group_label)}"
        for saved_path in _save_figure(fig, base_path):
            records.append(
                _manifest_row(
                    scope="group",
                    figure_type="module_roi_heatmap",
                    group=group_label,
                    factor="roi_time_heatmap",
                    title=f"ROI occupancy over time: {group_label}",
                    source="module_tables.roi_time_heatmap__occupancy",
                    path=saved_path,
                )
            )
    return records


def _render_behavior_transition_module_figures(
    module_tables: dict[str, pd.DataFrame],
    *,
    output_dir: Path,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    df = module_tables.get("behavior_transitions__global_matrix")
    required = {"group", "From Behavior", "To Behavior"}
    if df is None or df.empty or not required.issubset(df.columns):
        return records
    metric_column = "Transition Percentage" if "Transition Percentage" in df.columns else "Transition Count" if "Transition Count" in df.columns else ""
    if not metric_column:
        return records

    plot_df = df[["group", "From Behavior", "To Behavior", metric_column]].copy()
    plot_df["group"] = plot_df["group"].fillna("").astype(str).str.strip()
    plot_df["From Behavior"] = plot_df["From Behavior"].fillna("").astype(str).str.strip()
    plot_df["To Behavior"] = plot_df["To Behavior"].fillna("").astype(str).str.strip()
    plot_df[metric_column] = _coerce_numeric(plot_df[metric_column])
    plot_df = plot_df.dropna(subset=[metric_column])
    plot_df = plot_df[(plot_df["group"] != "") & (plot_df["From Behavior"] != "") & (plot_df["To Behavior"] != "")]
    if plot_df.empty:
        return records

    for group_label, group_df in plot_df.groupby("group", dropna=False):
        heat_df = (
            group_df.groupby(["From Behavior", "To Behavior"], dropna=False)[metric_column]
            .mean()
            .reset_index()
            .pivot(index="From Behavior", columns="To Behavior", values=metric_column)
            .fillna(0.0)
        )
        if heat_df.empty:
            continue
        row_strength = heat_df.sum(axis=1)
        col_strength = heat_df.sum(axis=0)
        focus_labels = sorted(set(row_strength.nlargest(10).index) | set(col_strength.nlargest(10).index))
        heat_df = heat_df.reindex(index=focus_labels, columns=focus_labels, fill_value=0.0)
        fig, ax = plt.subplots(figsize=(max(6.8, 0.8 * len(heat_df.columns)), max(4.8, 0.55 * len(heat_df.index))))
        sns.heatmap(
            heat_df,
            cmap="rocket_r",
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Mean transition %" if metric_column == "Transition Percentage" else "Mean transition count"},
            ax=ax,
        )
        _apply_sparse_heatmap_ticks(
            ax,
            x_labels=list(heat_df.columns),
            y_labels=list(heat_df.index),
            max_x_labels=10,
            max_y_labels=10,
        )
        ax.set_title(f"Behavior transitions: {group_label}", loc="left", fontweight="bold")
        ax.set_xlabel("To behavior")
        ax.set_ylabel("From behavior")
        fig.tight_layout()

        base_path = output_dir / f"behavior_transitions_{_slugify(group_label)}"
        for saved_path in _save_figure(fig, base_path):
            records.append(
                _manifest_row(
                    scope="group",
                    figure_type="module_behavior_transition_heatmap",
                    group=group_label,
                    factor="behavior_transitions",
                    title=f"Behavior transitions: {group_label}",
                    source="module_tables.behavior_transitions__global_matrix",
                    path=saved_path,
                )
            )
    return records


def _render_group_entity_metric_heatmaps(
    df: pd.DataFrame,
    *,
    entity_col: str,
    metric_cols: list[str],
    output_dir: Path,
    figure_type: str,
    factor: str,
    source: str,
    title_prefix: str,
    value_label_map: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    if df is None or df.empty or "group" not in df.columns or entity_col not in df.columns:
        return records
    value_label_map = dict(value_label_map or {})

    base_df = df.copy()
    base_df["group"] = base_df["group"].fillna("").astype(str).str.strip()
    base_df[entity_col] = base_df[entity_col].fillna("").astype(str).str.strip()
    base_df = base_df[(base_df["group"] != "") & (base_df[entity_col] != "")]
    if base_df.empty:
        return records

    for metric in metric_cols:
        if metric not in base_df.columns:
            continue
        plot_df = base_df[["group", entity_col, metric]].copy()
        plot_df[metric] = _coerce_numeric(plot_df[metric])
        plot_df = plot_df.dropna(subset=[metric])
        if plot_df.empty:
            continue

        heat_df = (
            plot_df.groupby([entity_col, "group"], dropna=False)[metric]
            .mean()
            .reset_index()
            .pivot(index=entity_col, columns="group", values=metric)
        )
        if heat_df.empty:
            continue
        heat_df = heat_df.sort_index()
        heat_df = heat_df.reindex(sorted(heat_df.columns), axis=1)
        if len(heat_df.index) > 12:
            entity_strength = heat_df.abs().max(axis=1).sort_values(ascending=False)
            heat_df = heat_df.loc[entity_strength.head(12).index]

        values = heat_df.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        diverging = bool(np.nanmin(finite) < 0.0 < np.nanmax(finite))
        cmap = "vlag" if diverging else "crest"
        center = 0.0 if diverging else None
        vmin = vmax = None
        if diverging:
            max_abs = float(np.nanmax(np.abs(finite)))
            vmin = -max_abs
            vmax = max_abs

        fig, ax = plt.subplots(figsize=(max(6.8, 1.15 * len(heat_df.columns)), max(4.6, 0.5 * len(heat_df.index))))
        sns.heatmap(
            heat_df,
            cmap=cmap,
            center=center,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": value_label_map.get(metric, _display_label(metric))},
            ax=ax,
        )
        _apply_sparse_heatmap_ticks(
            ax,
            x_labels=list(heat_df.columns),
            y_labels=list(heat_df.index),
            max_x_labels=10,
            max_y_labels=12,
        )
        title = f"{title_prefix}: {_display_label(metric)}"
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Group")
        ax.set_ylabel(_display_label(entity_col))
        fig.tight_layout()

        base_path = output_dir / f"{_slugify(factor)}_{_slugify(metric)}"
        for saved_path in _save_figure(fig, base_path):
            records.append(
                _manifest_row(
                    scope="group",
                    figure_type=figure_type,
                    metric=str(metric),
                    factor=factor,
                    title=title,
                    source=source,
                    path=saved_path,
                )
            )
    return records


def _render_preference_module_figures(
    module_tables: dict[str, pd.DataFrame],
    *,
    output_dir: Path,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    configs = (
        ("preference_indices__zone_pairwise", "Zone preference indices", "module_preference_index"),
        ("preference_indices__object_pairwise", "Object preference indices", "module_preference_index"),
    )
    for table_key, title_prefix, figure_type in configs:
        df = module_tables.get(table_key)
        if df is None or df.empty or not {"group", "Target A", "Target B"}.issubset(df.columns):
            continue
        work_df = df.copy()
        work_df["Preference Pair"] = work_df["Target A"].fillna("").astype(str).str.strip() + " vs " + work_df["Target B"].fillna("").astype(str).str.strip()
        metric_cols = [column for column in work_df.columns if str(column).endswith("Index")]
        metric_cols = [column for column in metric_cols if column not in _MODULE_CONTEXT_COLUMNS]
        records.extend(
            _render_group_entity_metric_heatmaps(
                work_df,
                entity_col="Preference Pair",
                metric_cols=metric_cols,
                output_dir=output_dir / _slugify(table_key),
                figure_type=figure_type,
                factor=table_key,
                source=f"module_tables.{table_key}",
                title_prefix=title_prefix,
            )
        )
    return records


def _render_latency_module_figures(
    module_tables: dict[str, pd.DataFrame],
    *,
    output_dir: Path,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    configs = (
        ("latency_metrics__zone_latency", "Zone latency metrics"),
        ("latency_metrics__object_latency", "Object latency metrics"),
    )
    for table_key, title_prefix in configs:
        df = module_tables.get(table_key)
        if df is None or df.empty or not {"group", "Target Name"}.issubset(df.columns):
            continue
        metric_cols = [column for column in df.columns if "Latency" in str(column)]
        records.extend(
            _render_group_entity_metric_heatmaps(
                df,
                entity_col="Target Name",
                metric_cols=metric_cols,
                output_dir=output_dir / _slugify(table_key),
                figure_type="module_latency_heatmap",
                factor=table_key,
                source=f"module_tables.{table_key}",
                title_prefix=title_prefix,
            )
        )
    return records


def _render_normalization_module_figures(
    module_tables: dict[str, pd.DataFrame],
    *,
    output_dir: Path,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    configs = (
        ("normalization_summary__zone_summary", "ROI Name", "Zone normalized summaries"),
        ("normalization_summary__object_summary", "Object ROI", "Object normalized summaries"),
    )
    for table_key, entity_col, title_prefix in configs:
        df = module_tables.get(table_key)
        if df is None or df.empty or "group" not in df.columns or entity_col not in df.columns:
            continue
        metric_cols = [
            column
            for column in df.columns
            if any(token in str(column) for token in ("per Minute", "Percent of Session"))
        ]
        records.extend(
            _render_group_entity_metric_heatmaps(
                df,
                entity_col=entity_col,
                metric_cols=metric_cols,
                output_dir=output_dir / _slugify(table_key),
                figure_type="module_normalization_heatmap",
                factor=table_key,
                source=f"module_tables.{table_key}",
                title_prefix=title_prefix,
            )
        )
    return records


def _render_visit_structure_module_figures(
    module_tables: dict[str, pd.DataFrame],
    *,
    output_dir: Path,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    configs = (
        ("visit_structure__zone_summary", "ROI Name", "Zone visit structure"),
        ("visit_structure__object_summary", "Object ROI", "Object visit structure"),
    )
    preferred_metrics = ("Visits", "Mean_Duration_s", "Median_Duration_s", "Max_Duration_Frames")
    for table_key, entity_col, title_prefix in configs:
        df = module_tables.get(table_key)
        if df is None or df.empty or "group" not in df.columns or entity_col not in df.columns:
            continue
        metric_cols = [column for column in preferred_metrics if column in df.columns]
        records.extend(
            _render_group_entity_metric_heatmaps(
                df,
                entity_col=entity_col,
                metric_cols=metric_cols,
                output_dir=output_dir / _slugify(table_key),
                figure_type="module_visit_structure_heatmap",
                factor=table_key,
                source=f"module_tables.{table_key}",
                title_prefix=title_prefix,
            )
        )
    return records


def _render_inter_bout_module_figures(
    module_tables: dict[str, pd.DataFrame],
    *,
    output_dir: Path,
) -> list[dict[str, str]]:
    df = module_tables.get("inter_bout_intervals__summary")
    if df is None or df.empty or not {"group", "Behavior"}.issubset(df.columns):
        return []
    metric_cols = [
        column
        for column in (
            "Median Inter-Bout Interval (s)",
            "Mean Inter-Bout Interval (s)",
            "Latency to First Bout (s)",
        )
        if column in df.columns
    ]
    return _render_group_entity_metric_heatmaps(
        df,
        entity_col="Behavior",
        metric_cols=metric_cols,
        output_dir=output_dir,
        figure_type="module_inter_bout_heatmap",
        factor="inter_bout_intervals__summary",
        source="module_tables.inter_bout_intervals__summary",
        title_prefix="Inter-bout summaries",
    )


def _render_object_transition_module_figures(
    module_tables: dict[str, pd.DataFrame],
    *,
    output_dir: Path,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    df = module_tables.get("object_transition_analysis__matrix")
    required = {"group", "From Object", "To Object"}
    if df is None or df.empty or not required.issubset(df.columns):
        return records
    metric_column = "Transition_Count" if "Transition_Count" in df.columns else "Transition Count" if "Transition Count" in df.columns else ""
    if not metric_column:
        return records

    plot_df = df[["group", "From Object", "To Object", metric_column]].copy()
    plot_df["group"] = plot_df["group"].fillna("").astype(str).str.strip()
    plot_df["From Object"] = plot_df["From Object"].fillna("").astype(str).str.strip()
    plot_df["To Object"] = plot_df["To Object"].fillna("").astype(str).str.strip()
    plot_df[metric_column] = _coerce_numeric(plot_df[metric_column])
    plot_df = plot_df.dropna(subset=[metric_column])
    plot_df = plot_df[(plot_df["group"] != "") & (plot_df["From Object"] != "") & (plot_df["To Object"] != "")]
    if plot_df.empty:
        return records

    for group_label, group_df in plot_df.groupby("group", dropna=False):
        heat_df = (
            group_df.groupby(["From Object", "To Object"], dropna=False)[metric_column]
            .mean()
            .reset_index()
            .pivot(index="From Object", columns="To Object", values=metric_column)
            .fillna(0.0)
        )
        if heat_df.empty:
            continue
        heat_df = heat_df.sort_index().reindex(sorted(heat_df.columns), axis=1)
        fig, ax = plt.subplots(figsize=(max(6.8, 0.8 * len(heat_df.columns)), max(4.8, 0.55 * len(heat_df.index))))
        sns.heatmap(
            heat_df,
            cmap="flare",
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Mean transition count"},
            ax=ax,
        )
        title = f"Object transitions: {group_label}"
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("To object")
        ax.set_ylabel("From object")
        fig.tight_layout()

        base_path = output_dir / f"object_transitions_{_slugify(group_label)}"
        for saved_path in _save_figure(fig, base_path):
            records.append(
                _manifest_row(
                    scope="group",
                    figure_type="module_object_transition_heatmap",
                    group=group_label,
                    factor="object_transition_analysis__matrix",
                    title=title,
                    source="module_tables.object_transition_analysis__matrix",
                    path=saved_path,
                )
            )
    return records


def _render_event_aligned_module_figures(
    module_tables: dict[str, pd.DataFrame],
    *,
    output_dir: Path,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    df = module_tables.get("event_aligned_windows__summary")
    required = {"group", "Source", "Event Type", "Target Name", "Relative Time (s)", "Behavior"}
    if df is None or df.empty or not required.issubset(df.columns):
        return records
    metric_column = "Behavior Fraction" if "Behavior Fraction" in df.columns else "Frame_Count" if "Frame_Count" in df.columns else ""
    if not metric_column:
        return records

    plot_df = df[list(required | {metric_column})].copy()
    plot_df["group"] = plot_df["group"].fillna("").astype(str).str.strip()
    plot_df["Source"] = plot_df["Source"].fillna("").astype(str).str.strip()
    plot_df["Event Type"] = plot_df["Event Type"].fillna("").astype(str).str.strip()
    plot_df["Target Name"] = plot_df["Target Name"].fillna("").astype(str).str.strip()
    plot_df["Behavior"] = plot_df["Behavior"].fillna("").astype(str).str.strip()
    plot_df["Relative Time (s)"] = _coerce_numeric(plot_df["Relative Time (s)"])
    plot_df[metric_column] = _coerce_numeric(plot_df[metric_column])
    plot_df = plot_df.dropna(subset=["Relative Time (s)", metric_column])
    plot_df = plot_df[
        (plot_df["group"] != "")
        & (plot_df["Source"] != "")
        & (plot_df["Event Type"] != "")
        & (plot_df["Target Name"] != "")
        & (plot_df["Behavior"] != "")
    ]
    if plot_df.empty:
        return records

    group_labels = [label for label in _ordered_labels(plot_df["group"]) if label]
    for combo, combo_df in plot_df.groupby(["Source", "Event Type", "Target Name"], dropna=False):
        source_label, event_type, target_name = combo
        agg = (
            combo_df.groupby(["group", "Behavior", "Relative Time (s)"], dropna=False)[metric_column]
            .mean()
            .reset_index()
        )
        if agg.empty:
            continue
        behavior_order = (
            agg.groupby("Behavior", dropna=False)[metric_column]
            .mean()
            .sort_values(ascending=False)
            .index.astype(str)
            .tolist()
        )
        behavior_order = behavior_order[:8]
        time_values = sorted(pd.unique(agg["Relative Time (s)"]))
        if not behavior_order or not time_values:
            continue

        present_groups = [label for label in group_labels if label in set(agg["group"])]
        if not present_groups:
            continue
        fig, axes = plt.subplots(
            1,
            len(present_groups),
            figsize=(max(6.8, 4.6 * len(present_groups)), max(4.8, 0.45 * len(behavior_order) + 1.8)),
            sharey=True,
        )
        if len(present_groups) == 1:
            axes = [axes]
        for ax, group_label in zip(axes, present_groups):
            group_df = agg[agg["group"] == group_label]
            heat_df = (
                group_df[group_df["Behavior"].isin(behavior_order)]
                .pivot(index="Behavior", columns="Relative Time (s)", values=metric_column)
                .reindex(index=behavior_order, columns=time_values)
            )
            heat_df = heat_df.fillna(0.0)
            sns.heatmap(
                heat_df,
                cmap="magma",
                linewidths=0.4,
                linecolor="white",
                cbar=len(present_groups) == 1,
                cbar_kws={"label": "Mean behavior fraction" if metric_column == "Behavior Fraction" else "Mean frame count"},
                ax=ax,
            )
            _apply_sparse_heatmap_ticks(
                ax,
                x_labels=list(heat_df.columns),
                y_labels=list(heat_df.index),
                max_x_labels=9,
                max_y_labels=8,
            )
            ax.set_title(group_label, fontweight="bold")
            ax.set_xlabel("Relative time (s)")
            ax.set_ylabel("Behavior")
        title = f"Event-aligned behavior: {_display_label(source_label)} {event_type} | {target_name}"
        fig.suptitle(title, x=0.05, y=0.99, ha="left", fontweight="bold")
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])

        base_path = output_dir / f"{_slugify(source_label)}_{_slugify(event_type)}_{_slugify(target_name)}"
        for saved_path in _save_figure(fig, base_path):
            records.append(
                _manifest_row(
                    scope="group",
                    figure_type="module_event_aligned_heatmap",
                    metric=str(metric_column),
                    factor="event_aligned_windows__summary",
                    title=title,
                    source="module_tables.event_aligned_windows__summary",
                    path=saved_path,
                )
            )
    return records


def _module_key_from_manifest_row(row: dict[str, Any]) -> str:
    source = str(row.get("source", "") or "").strip()
    if source.startswith("module_tables."):
        suffix = source.split("module_tables.", 1)[1]
        return suffix.split("__", 1)[0].strip()
    factor = str(row.get("factor", "") or "").strip()
    if "__" in factor:
        return factor.split("__", 1)[0].strip()
    return ""


def _build_assay_figure_index(
    manifest_df: pd.DataFrame,
    *,
    assay_preset_key: str,
) -> pd.DataFrame:
    if manifest_df is None or manifest_df.empty:
        return pd.DataFrame(
            columns=[
                "assay_preset",
                "assay_label",
                "selection_reason",
                "module_key",
            ]
            + FIGURE_MANIFEST_COLUMNS
        )

    preset_key = str(assay_preset_key or "custom").strip() or "custom"
    preset = get_assay_preset(preset_key)
    allowed_modules = set(expand_metrics_to_modules(getattr(preset, "metric_keys", ())))
    module_order = {
        module_key: index
        for index, module_key in enumerate(expand_metrics_to_modules(getattr(preset, "metric_keys", ())), start=1)
    }
    general_types = {
        "batch_dashboard_overview",
        "group_stats_overview",
        "video_summary_profile",
        "video_quicklook",
        "group_distribution",
        "timecourse",
        "omnibus_heatmap",
        "pairwise_contrast",
    }
    figure_type_order = {
        "batch_dashboard_overview": 0,
        "group_stats_overview": 1,
        "omnibus_heatmap": 0,
        "pairwise_contrast": 2,
        "group_distribution": 3,
        "timecourse": 4,
        "module_preference_index": 10,
        "module_latency_heatmap": 11,
        "module_visit_structure_heatmap": 12,
        "module_normalization_heatmap": 13,
        "module_inter_bout_heatmap": 14,
        "module_event_aligned_heatmap": 15,
        "module_behavior_transition_heatmap": 16,
        "module_object_transition_heatmap": 17,
        "module_roi_heatmap": 18,
        "module_temporal_trend": 19,
        "module_activity_budget": 20,
        "video_quicklook": 30,
        "video_summary_profile": 31,
        "existing_analytics_plot": 40,
    }

    rows: list[dict[str, Any]] = []
    for row in manifest_df.to_dict(orient="records"):
        figure_type = str(row.get("figure_type", "") or "").strip()
        module_key = _module_key_from_manifest_row(row)
        if preset_key == "custom":
            selection_reason = "custom_all_figures"
        elif figure_type in general_types:
            selection_reason = "core_summary"
        elif module_key and module_key in allowed_modules:
            selection_reason = "preset_module"
        else:
            continue

        enriched = {
            "assay_preset": preset_key,
            "assay_label": getattr(preset, "label", "Custom / mixed") if preset is not None else "Custom / mixed",
            "selection_reason": selection_reason,
            "module_key": module_key,
        }
        enriched.update(row)
        rows.append(enriched)

    out = pd.DataFrame(
        rows,
        columns=[
            "assay_preset",
            "assay_label",
            "selection_reason",
            "module_key",
        ]
        + FIGURE_MANIFEST_COLUMNS,
    )
    if out.empty:
        return out
    reason_order = {"core_summary": 0, "preset_module": 1, "custom_all_figures": 0}
    scope_order = {"intergroup": 0, "group": 1, "individual": 2}
    out["__module_order__"] = out["module_key"].map(lambda value: module_order.get(str(value or ""), 999))
    out["__figure_type_order__"] = out["figure_type"].map(lambda value: figure_type_order.get(str(value or ""), 999))
    out["__reason_order__"] = out["selection_reason"].map(lambda value: reason_order.get(str(value), 99))
    out["__scope_order__"] = out["scope"].map(lambda value: scope_order.get(str(value), 99))
    out = out.sort_values(
        by=["__reason_order__", "__module_order__", "__figure_type_order__", "__scope_order__", "metric", "title", "path"],
        kind="stable",
    )
    out["figure_rank"] = np.arange(1, len(out.index) + 1, dtype=int)
    out["priority_tier"] = np.where(
        out["figure_rank"] <= 6,
        "primary",
        np.where(out["figure_rank"] <= 14, "secondary", "supporting"),
    )
    out["recommended_label"] = out.apply(
        lambda row: f"{row['figure_rank']:02d}. {str(row.get('title', '') or row.get('figure_type', '')).strip()}",
        axis=1,
    )
    out["recommended_filename"] = out.apply(
        lambda row: f"{_slugify(row.get('assay_preset', 'custom'))}_{int(row.get('figure_rank', 0)):02d}_{_slugify(row.get('title', '') or row.get('figure_type', 'figure'))}.{str(row.get('format', 'png') or 'png')}",
        axis=1,
    )
    out = out.drop(columns=["__reason_order__", "__scope_order__", "__module_order__", "__figure_type_order__"])
    ordered_columns = [
        "assay_preset",
        "assay_label",
        "figure_rank",
        "priority_tier",
        "selection_reason",
        "recommended_label",
        "recommended_filename",
        "module_key",
    ] + FIGURE_MANIFEST_COLUMNS
    out = out[ordered_columns]
    return out


def export_batch_figure_bundle(
    *,
    video_summary_df: pd.DataFrame,
    omnibus_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    output_dir: str | Path,
    video_results: list[dict[str, Any]] | None = None,
    module_tables: dict[str, pd.DataFrame] | None = None,
    precomputed_figure_records: list[dict[str, str]] | None = None,
    assay_preset_key: str = "custom",
    export_mode: str = "full_bundle",
    export_publication_figures: bool = True,
    export_batch_dashboard: bool = False,
    export_group_stats_overview: bool = False,
    export_individual_profiles: bool = True,
    export_module_archive: bool | None = None,
    group_col: str = "group",
    time_col: str = "time_point",
) -> tuple[pd.DataFrame, BatchFigureArtifacts]:
    output_root = Path(output_dir).expanduser().resolve()
    individual_dir = output_root / "individual"
    group_dir = output_root / "group"
    intergroup_dir = output_root / "intergroup"
    output_root.mkdir(parents=True, exist_ok=True)

    _apply_publication_theme()
    export_mode_key = str(export_mode or "assay_shortlist").strip() or "assay_shortlist"
    if export_module_archive is None:
        export_module_archive = export_mode_key == "full_bundle"
    assay_preset = get_assay_preset(assay_preset_key) or get_assay_preset("custom")
    allowed_modules = set(expand_metrics_to_modules(getattr(assay_preset, "metric_keys", ()) or ()))

    metrics = _resolve_numeric_metrics(
        video_summary_df if video_summary_df is not None else pd.DataFrame(),
        exclude=_SUMMARY_EXCLUDE_COLUMNS,
        min_non_na=2,
    )
    use_curated_metrics = export_mode_key == "assay_shortlist" and not bool(export_module_archive)
    summary_metrics = _select_primary_metrics(metrics, omnibus_df, max_metrics=6) if use_curated_metrics else metrics
    filtered_omnibus_df = omnibus_df
    filtered_pairwise_df = pairwise_df
    if use_curated_metrics:
        if filtered_omnibus_df is not None and not filtered_omnibus_df.empty and "metric" in filtered_omnibus_df.columns:
            filtered_omnibus_df = filtered_omnibus_df[filtered_omnibus_df["metric"].astype(str).isin(summary_metrics)].copy()
        if filtered_pairwise_df is not None and not filtered_pairwise_df.empty and "metric" in filtered_pairwise_df.columns:
            filtered_pairwise_df = filtered_pairwise_df[filtered_pairwise_df["metric"].astype(str).isin(summary_metrics)].copy()

    records: list[dict[str, str]] = list(precomputed_figure_records or [])
    if export_mode_key == "full_bundle" and bool(export_module_archive):
        records.extend(_collect_existing_analytics_figures(video_results))
    module_table_map = dict(module_tables or {})
    if export_batch_dashboard:
        records.extend(
            _render_batch_dashboard(
                video_summary_df=video_summary_df,
                omnibus_df=filtered_omnibus_df,
                pairwise_df=filtered_pairwise_df,
                module_tables=module_table_map,
                assay_preset_key=assay_preset_key,
                output_dir=output_root / "dashboard",
                group_col=group_col,
            )
        )
    if export_group_stats_overview:
        records.extend(_render_group_stats_overview(filtered_omnibus_df, filtered_pairwise_df, output_dir=output_root / "group_stats"))
    if export_individual_profiles:
        records.extend(_render_individual_summary_profiles(video_summary_df, metrics=summary_metrics, output_dir=individual_dir))
    if export_publication_figures:
        records.extend(
            _render_group_distribution_plots(
                video_summary_df,
                filtered_omnibus_df,
                metrics=summary_metrics,
                output_dir=group_dir / "distributions",
                group_col=group_col,
            )
        )
        records.extend(
            _render_group_timepoint_plots(
                video_summary_df,
                metrics=summary_metrics,
                output_dir=group_dir / "timecourse",
                group_col=group_col,
                time_col=time_col,
            )
        )
        records.extend(_render_omnibus_heatmap(filtered_omnibus_df, output_dir=intergroup_dir))
        records.extend(_render_pairwise_contrast_plots(filtered_pairwise_df, output_dir=intergroup_dir))
    render_allowed_modules = bool(export_publication_figures or export_module_archive)
    render_all_modules = bool(export_module_archive)
    if render_allowed_modules and (render_all_modules or "temporal_trends" in allowed_modules):
        records.extend(_render_temporal_module_figures(module_table_map, output_dir=group_dir / "module_temporal"))
    if render_allowed_modules and (render_all_modules or "activity_budgets" in allowed_modules):
        records.extend(_render_activity_budget_module_figures(module_table_map, output_dir=group_dir / "module_activity"))
    if render_allowed_modules and (render_all_modules or "roi_time_heatmap" in allowed_modules):
        records.extend(_render_roi_occupancy_module_figures(module_table_map, output_dir=group_dir / "module_roi"))
    if render_allowed_modules and (render_all_modules or "behavior_transitions" in allowed_modules):
        records.extend(_render_behavior_transition_module_figures(module_table_map, output_dir=group_dir / "module_transitions"))
    if render_allowed_modules and (render_all_modules or "preference_indices" in allowed_modules):
        records.extend(_render_preference_module_figures(module_table_map, output_dir=group_dir / "module_preference"))
    if render_allowed_modules and (render_all_modules or "latency_metrics" in allowed_modules):
        records.extend(_render_latency_module_figures(module_table_map, output_dir=group_dir / "module_latency"))
    if render_allowed_modules and (render_all_modules or "normalization_summary" in allowed_modules):
        records.extend(_render_normalization_module_figures(module_table_map, output_dir=group_dir / "module_normalization"))
    if render_allowed_modules and (render_all_modules or "visit_structure" in allowed_modules):
        records.extend(_render_visit_structure_module_figures(module_table_map, output_dir=group_dir / "module_visit_structure"))
    if render_allowed_modules and (render_all_modules or "inter_bout_intervals" in allowed_modules):
        records.extend(_render_inter_bout_module_figures(module_table_map, output_dir=group_dir / "module_inter_bout"))
    if render_allowed_modules and (render_all_modules or "object_transition_analysis" in allowed_modules):
        records.extend(_render_object_transition_module_figures(module_table_map, output_dir=group_dir / "module_object_transitions"))
    if render_allowed_modules and (render_all_modules or "event_aligned_windows" in allowed_modules):
        records.extend(_render_event_aligned_module_figures(module_table_map, output_dir=group_dir / "module_event_aligned"))

    manifest_df = pd.DataFrame(records, columns=FIGURE_MANIFEST_COLUMNS) if records else pd.DataFrame(columns=FIGURE_MANIFEST_COLUMNS)
    if not manifest_df.empty:
        manifest_df = manifest_df.drop_duplicates(subset=["path"]).sort_values(
            by=["scope", "figure_type", "video_name", "metric", "format", "path"],
            kind="stable",
        )
    full_manifest_df = manifest_df.copy()
    assay_manifest_df = _build_assay_figure_index(full_manifest_df, assay_preset_key=assay_preset_key)
    visible_manifest_df = full_manifest_df
    if export_mode_key == "assay_shortlist":
        visible_manifest_df = assay_manifest_df[FIGURE_MANIFEST_COLUMNS].copy() if not assay_manifest_df.empty else pd.DataFrame(columns=FIGURE_MANIFEST_COLUMNS)
        if not visible_manifest_df.empty:
            visible_manifest_df = visible_manifest_df.drop_duplicates(subset=["path"]).sort_values(
                by=["scope", "figure_type", "video_name", "metric", "format", "path"],
                kind="stable",
            )
    manifest_csv = output_root / "figure_manifest.csv"
    visible_manifest_df.to_csv(manifest_csv, index=False)
    assay_manifest_csv = output_root / f"assay_figure_index_{_slugify(assay_preset_key or 'custom')}.csv"
    assay_manifest_df.to_csv(assay_manifest_csv, index=False)

    artifacts = BatchFigureArtifacts(
        output_dir=str(output_root),
        manifest_csv=str(manifest_csv),
        figure_count=int(len(visible_manifest_df)),
        assay_manifest_csv=str(assay_manifest_csv),
        assay_figure_count=int(len(assay_manifest_df)),
    )
    return visible_manifest_df, artifacts
