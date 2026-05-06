"""Batch-level group statistics utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import kpss
import statsmodels.formula.api as smf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

MIN_FACTOR_REPLICATES = 2
MIN_KPSS_TIMEPOINTS = 5
MIN_REPEATED_SUBJECTS_FOR_MIXED = 2


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _epsilon_squared_kruskal(h_stat: float, n_total: int, n_groups: int) -> float:
    if n_total <= n_groups or n_groups <= 1:
        return np.nan
    return float((h_stat - n_groups + 1) / (n_total - n_groups))


def _cliffs_delta(x: Iterable[float], y: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=float)
    b = np.asarray(list(y), dtype=float)
    if a.size == 0 or b.size == 0:
        return np.nan
    gt = sum(1 for ax in a for by in b if ax > by)
    lt = sum(1 for ax in a for by in b if ax < by)
    denom = a.size * b.size
    if denom == 0:
        return np.nan
    return float((gt - lt) / denom)


def _resolve_numeric_metrics(
    df: pd.DataFrame,
    *,
    exclude: set[str] | None = None,
    min_non_na: int = 3,
) -> list[str]:
    excluded = set(exclude or set())
    min_required = max(1, int(min_non_na or 1))
    metrics: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        numeric = _coerce_numeric(df[col])
        if numeric.notna().sum() >= min_required:
            metrics.append(col)
    return metrics


def _apply_pvalue_correction(df: pd.DataFrame, p_col: str, method: str) -> pd.DataFrame:
    out = df.copy()
    out["p_adj"] = np.nan
    out["reject_h0"] = False
    if out.empty or p_col not in out.columns:
        return out
    valid = out[p_col].notna()
    if valid.sum() == 0:
        return out
    reject, p_adj, _, _ = multipletests(out.loc[valid, p_col].to_numpy(), method=method)
    out.loc[valid, "p_adj"] = p_adj
    out.loc[valid, "reject_h0"] = np.asarray(reject, dtype=bool)
    return out


def _clean_factor_series(series: pd.Series, *, unknown_token: str = "__missing__") -> pd.Series:
    out = series.fillna(unknown_token).astype(str).str.strip()
    out = out.replace({"": unknown_token})
    return out


def _build_factor_frames(
    df: pd.DataFrame,
    *,
    group_col: str,
    subject_col: str = "subject_id",
    time_col: str = "time_point",
    categorical_factors: list[str] | None = None,
) -> list[tuple[str, pd.Series]]:
    def _add_factor(
        name: str,
        column: pd.Series | None,
        store: dict[str, str] | None = None,
    ) -> None:
        if column is None:
            return
        key = name.strip()
        if not key or (store is not None and key in store):
            return
        if store is not None:
            store[key] = key
        factors.append((key, _clean_factor_series(column)))

    factors: list[tuple[str, pd.Series]] = []
    added: dict[str, str] = {}
    if group_col in df.columns:
        _add_factor("group", df[group_col], store=added)

    if categorical_factors:
        for factor_name in categorical_factors:
            if not isinstance(factor_name, str):
                continue
            candidate = factor_name.strip()
            if not candidate:
                continue
            if candidate == subject_col:
                continue
            if candidate in df.columns and candidate not in added:
                _add_factor(candidate, df[candidate], store=added)

    # Global combined strata across available categorical descriptors.
    combo_parts: list[pd.Series] = []
    for _name, col in factors:
        combo_parts.append(col)
    if len(combo_parts) > 1:
        combo = combo_parts[0].astype(str)
        for col in combo_parts[1:]:
            combo = combo + " | " + col.astype(str)
        factors.append(("global_combined", combo))
    return factors


def compute_nonparametric_group_stats(
    video_summary_df: pd.DataFrame,
    *,
    group_col: str = "group",
    subject_col: str = "subject_id",
    time_col: str = "time_point",
    correction_method: str = "fdr_bh",
    categorical_factors: list[str] | None = None,
    log_fn: Callable[[str, str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute Kruskal + Mann-Whitney stats with correction and effect sizes.

    ``log_fn(message, level)`` (optional) is invoked when an individual test
    crashes on degenerate input. Each failure is reported as a ``WARNING`` so
    the user sees it in the run log; analysis continues and the offending row's
    ``note`` column records the cause for later inspection in the CSV.
    """
    kruskal_failures: list[str] = []
    mannwhitney_failures: list[str] = []
    if video_summary_df is None or video_summary_df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    df = video_summary_df.copy()
    factor_frames = _build_factor_frames(
        df,
        group_col=group_col,
        subject_col=subject_col,
        time_col=time_col,
        categorical_factors=categorical_factors,
    )
    if not factor_frames:
        empty = pd.DataFrame()
        return empty, empty, empty
    factor_columns = {name for name, _ in factor_frames if name != "global_combined"}
    excluded_columns = set(
        {
            group_col,
            "video_id",
            "video_name",
            "video_path",
            subject_col,
            time_col,
        }
    ) | factor_columns
    excluded_columns.update({"global_combined"})

    numeric_metrics = _resolve_numeric_metrics(
        df,
        exclude=excluded_columns,
        min_non_na=2,
    )
    if not numeric_metrics:
        empty = pd.DataFrame()
        return empty, empty, empty

    omnibus_rows = []
    pairwise_rows = []
    effect_rows = []
    max_pairwise_levels = 20

    for factor_name, factor_series in factor_frames:
        work = df.copy()
        work["__factor__"] = factor_series
        work = work[work["__factor__"] != "__missing__"]
        if work.empty:
            continue
        levels = sorted(work["__factor__"].unique().tolist())
        if len(levels) < 2:
            continue

        for metric in numeric_metrics:
            grouped_values: list[tuple[str, np.ndarray]] = []
            underpowered_levels: list[str] = []
            for lvl in levels:
                values = _coerce_numeric(work.loc[work["__factor__"] == lvl, metric]).dropna().to_numpy(dtype=float)
                if values.size > 0:
                    grouped_values.append((lvl, values))
                    if values.size < MIN_FACTOR_REPLICATES:
                        underpowered_levels.append(f"{lvl} (n={values.size})")

            if len(grouped_values) < 2:
                continue

            metric_n = int(sum(arr.size for _lvl, arr in grouped_values))
            analysis_scope = (
                "group_level" if factor_name == "group" else ("global_level" if factor_name == "global_combined" else "factor_level")
            )
            if underpowered_levels:
                omnibus_rows.append(
                    {
                        "analysis_scope": analysis_scope,
                        "factor": factor_name,
                        "metric": metric,
                        "groups_compared": len(grouped_values),
                        "n_total": metric_n,
                        "kruskal_h": np.nan,
                        "p_value": np.nan,
                        "epsilon_squared": np.nan,
                        "note": (
                            f"skipped:requires>={MIN_FACTOR_REPLICATES}_replicates_per_level;"
                            f" underpowered_levels={', '.join(underpowered_levels)}"
                        ),
                    }
                )
                continue
            kruskal_note = ""
            try:
                h_stat, p_value = kruskal(*[arr for _lvl, arr in grouped_values])
            except ValueError as exc:
                # scipy.stats.kruskal raises ValueError on degenerate input
                # (all values identical, fewer than two groups, etc.). Record the
                # cause in `note` so missing rows are distinguishable from real
                # not-significant rows in downstream figures and CSVs.
                h_stat, p_value = np.nan, np.nan
                kruskal_note = f"error:kruskal_failed:{type(exc).__name__}: {exc}"
                kruskal_failures.append(f"{factor_name}:{metric}")
                if log_fn is not None:
                    log_fn(
                        f"Kruskal-Wallis failed for factor='{factor_name}' "
                        f"metric='{metric}': {exc}. Analysis continues; row marked "
                        f"with note='{kruskal_note}' in group_stats_overview.csv.",
                        "WARNING",
                    )

            epsilon_sq = _epsilon_squared_kruskal(h_stat, metric_n, len(grouped_values)) if np.isfinite(h_stat) else np.nan
            omnibus_rows.append(
                {
                    "analysis_scope": analysis_scope,
                    "factor": factor_name,
                    "metric": metric,
                    "groups_compared": len(grouped_values),
                    "n_total": metric_n,
                    "kruskal_h": h_stat,
                    "p_value": p_value,
                    "epsilon_squared": epsilon_sq,
                    "note": kruskal_note,
                }
            )
            effect_rows.append(
                {
                    "factor": factor_name,
                    "metric": metric,
                    "effect_type": "epsilon_squared",
                    "value": epsilon_sq,
                    "context": f"{factor_name}:kruskal_omnibus",
                }
            )

            if len(grouped_values) > max_pairwise_levels:
                pairwise_rows.append(
                    {
                        "analysis_scope": analysis_scope,
                        "factor": factor_name,
                        "metric": metric,
                        "group_a": "",
                        "group_b": "",
                        "n_a": np.nan,
                        "n_b": np.nan,
                        "u_stat": np.nan,
                        "p_value": np.nan,
                        "cliffs_delta": np.nan,
                        "note": f"pairwise_skipped:too_many_levels({len(grouped_values)})",
                    }
                )
                continue

            for i in range(len(grouped_values)):
                for j in range(i + 1, len(grouped_values)):
                    g1, a = grouped_values[i]
                    g2, b = grouped_values[j]
                    pair_note = ""
                    try:
                        u_stat, p_pair = mannwhitneyu(a, b, alternative="two-sided")
                    except ValueError as exc:
                        # scipy.stats.mannwhitneyu raises ValueError on degenerate
                        # input (all-equal values, empty array). Distinguish a
                        # crashed pair from a real not-significant one in `note`.
                        u_stat, p_pair = np.nan, np.nan
                        pair_note = f"error:mannwhitney_failed:{type(exc).__name__}: {exc}"
                        mannwhitney_failures.append(f"{factor_name}:{metric}:{g1}-vs-{g2}")
                        if log_fn is not None:
                            log_fn(
                                f"Mann-Whitney U failed for factor='{factor_name}' "
                                f"metric='{metric}' pair='{g1} vs {g2}': {exc}. "
                                f"Analysis continues; row marked with "
                                f"note='{pair_note}' in group_pairwise_tests.csv.",
                                "WARNING",
                            )
                    delta = _cliffs_delta(a, b)
                    pairwise_rows.append(
                        {
                            "analysis_scope": analysis_scope,
                            "factor": factor_name,
                            "metric": metric,
                            "group_a": g1,
                            "group_b": g2,
                            "n_a": int(a.size),
                            "n_b": int(b.size),
                            "u_stat": u_stat,
                            "p_value": p_pair,
                            "cliffs_delta": delta,
                            "note": pair_note,
                        }
                    )
                    effect_rows.append(
                        {
                            "factor": factor_name,
                            "metric": metric,
                            "effect_type": "cliffs_delta",
                            "value": delta,
                            "context": f"{factor_name}:{g1}_vs_{g2}",
                        }
                    )

    omnibus_df = pd.DataFrame(omnibus_rows)
    pairwise_df = pd.DataFrame(pairwise_rows)
    effects_df = pd.DataFrame(effect_rows)

    if log_fn is not None and (kruskal_failures or mannwhitney_failures):
        log_fn(
            (
                f"Group stats completed with degenerate-input failures: "
                f"{len(kruskal_failures)} omnibus and {len(mannwhitney_failures)} pairwise "
                f"tests could not run. Affected rows kept with NaN statistics and an "
                f"'error:' tag in their 'note' column. Inspect "
                f"group_stats_overview.csv / group_pairwise_tests.csv for details."
            ),
            "INFO",
        )

    if not omnibus_df.empty:
        omnibus_df = _apply_pvalue_correction(omnibus_df, "p_value", correction_method)
    if not pairwise_df.empty:
        corrected_chunks = []
        for metric, group_df in pairwise_df.groupby("metric", dropna=False):
            corrected = _apply_pvalue_correction(group_df, "p_value", correction_method)
            corrected_chunks.append(corrected)
        pairwise_df = pd.concat(corrected_chunks, ignore_index=True) if corrected_chunks else pairwise_df

    return omnibus_df, pairwise_df, effects_df


def compute_kpss_and_mixed_effects(
    video_summary_df: pd.DataFrame,
    *,
    group_col: str = "group",
    subject_col: str = "subject_id",
    time_col: str = "time_point",
) -> pd.DataFrame:
    """Compute KPSS and optional mixed-effects trends for numeric metrics."""
    if video_summary_df is None or video_summary_df.empty:
        return pd.DataFrame()
    df = video_summary_df.copy()
    numeric_metrics = _resolve_numeric_metrics(
        df,
        exclude={group_col, "video_id", "video_name", "video_path", subject_col, time_col},
        min_non_na=1,
    )
    if not numeric_metrics:
        return pd.DataFrame()

    rows: list[dict] = []
    has_subject = subject_col in df.columns
    has_time = time_col in df.columns
    has_group = group_col in df.columns

    if has_subject:
        df[subject_col] = df[subject_col].fillna("").astype(str)
    if has_group:
        df[group_col] = df[group_col].fillna("").astype(str)
    if has_time:
        df[time_col] = df[time_col].fillna("").astype(str)
        # Numeric KPSS is time ordered by an encoded value; textual timepoint labels are handled
        # by collapsing to numeric bins where possible.

    for metric in numeric_metrics:
        analysis_group = "all"
        if has_group:
            iterable = df.groupby(group_col, dropna=False)
        else:
            iterable = [(analysis_group, df)]

        for grp, group_df in iterable:
            group_metric = _coerce_numeric(group_df[metric])
            if time_col not in group_df.columns:
                rows.append(
                    {
                        "metric": metric,
                        "group": grp,
                        "analysis": "kpss",
                        "kpss_stat": np.nan,
                        "p_value": np.nan,
                        "n": 0,
                        "series_basis": "timepoint_mean",
                        "note": "kpss_skipped:time_col_missing",
                    }
                )
                continue
            time_numeric = _coerce_numeric(group_df[time_col])
            timed = pd.DataFrame(
                {
                    "metric_value": group_metric,
                    "time_value": time_numeric,
                }
            ).dropna(subset=["metric_value", "time_value"])
            if timed.empty:
                rows.append(
                    {
                        "metric": metric,
                        "group": grp,
                        "analysis": "kpss",
                        "kpss_stat": np.nan,
                        "p_value": np.nan,
                        "n": 0,
                        "series_basis": "timepoint_mean",
                        "note": "kpss_skipped:requires_numeric_timepoints",
                    }
                )
                continue

            # KPSS expects an ordered time series, so aggregate replicates by numeric time bin.
            timed = timed.groupby("time_value", as_index=False)["metric_value"].mean().sort_values("time_value")
            values = timed["metric_value"].to_numpy(dtype=float)
            if values.size < MIN_KPSS_TIMEPOINTS:
                rows.append(
                    {
                        "metric": metric,
                        "group": grp,
                        "analysis": "kpss",
                        "kpss_stat": np.nan,
                        "p_value": np.nan,
                        "n": int(values.size),
                        "series_basis": "timepoint_mean",
                        "note": f"kpss_skipped:requires>={MIN_KPSS_TIMEPOINTS}_numeric_timepoints",
                    }
                )
                continue
            try:
                stat, p_value, _lags, _crit = kpss(values, regression="ct", nlags="auto")
                rows.append(
                    {
                        "metric": metric,
                        "group": grp,
                        "analysis": "kpss",
                        "kpss_stat": float(stat),
                        "p_value": float(p_value),
                        "n": int(values.size),
                        "series_basis": "timepoint_mean",
                        "note": "",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "metric": metric,
                        "group": grp,
                        "analysis": "kpss",
                        "kpss_stat": np.nan,
                        "p_value": np.nan,
                        "n": int(values.size),
                        "series_basis": "timepoint_mean",
                        "note": f"kpss_failed:{exc}",
                    }
                )

    if not has_subject:
        return pd.DataFrame(rows)

    for metric in numeric_metrics:
        mixed_parts = [metric, subject_col]
        if has_group:
            mixed_parts.append(group_col)
        if has_time:
            mixed_parts.append(time_col)
        mixed_df = df[mixed_parts].copy()
        rename_map = {metric: "metric_value", subject_col: "subject"}
        if has_group:
            rename_map[group_col] = "group"
        if has_time:
            rename_map[time_col] = "time_factor"
        mixed_df = mixed_df.rename(columns=rename_map)

        mixed_df["metric_value"] = _coerce_numeric(mixed_df["metric_value"])
        mixed_df["subject"] = mixed_df["subject"].fillna("").astype(str)
        mixed_df = mixed_df[mixed_df["subject"] != ""]
        if has_group:
            mixed_df["group"] = mixed_df["group"].fillna("").astype(str)
            mixed_df = mixed_df[mixed_df["group"] != ""]
        if has_time:
            mixed_df["time_factor"] = mixed_df["time_factor"].fillna("").astype(str)
            mixed_df = mixed_df[mixed_df["time_factor"] != ""]

        mixed_df = mixed_df.dropna(subset=["metric_value"])
        if mixed_df["metric_value"].empty:
            rows.append(
                {
                    "metric": metric,
                    "group": "all",
                    "analysis": "mixedlm",
                    "kpss_stat": np.nan,
                    "p_value": np.nan,
                    "n": int(len(mixed_df)),
                    "note": "mixedlm_skipped:no_valid_rows",
                }
            )
            continue

        if mixed_df["subject"].nunique() < 2:
            rows.append(
                {
                    "metric": metric,
                    "group": "all",
                    "analysis": "mixedlm",
                    "kpss_stat": np.nan,
                    "p_value": np.nan,
                    "n": int(len(mixed_df)),
                    "note": "mixedlm_skipped:requires>=2_subjects",
                }
            )
            continue

        subject_counts = mixed_df["subject"].value_counts()
        repeated_subjects = int((subject_counts >= 2).sum())
        if repeated_subjects < MIN_REPEATED_SUBJECTS_FOR_MIXED:
            rows.append(
                {
                    "metric": metric,
                    "group": "all",
                    "analysis": "mixedlm",
                    "kpss_stat": np.nan,
                    "p_value": np.nan,
                    "n": int(len(mixed_df)),
                    "note": (
                        "mixedlm_skipped:"
                        f"requires>={MIN_REPEATED_SUBJECTS_FOR_MIXED}_subjects_with_repeated_observations"
                    ),
                }
            )
            continue

        if not has_group and not has_time:
            rows.append(
                {
                    "metric": metric,
                    "group": "all",
                    "analysis": "mixedlm",
                    "kpss_stat": np.nan,
                    "p_value": np.nan,
                    "n": int(len(mixed_df)),
                    "note": "mixedlm_skipped:requires_group_or_time",
                }
            )
            continue

        group_n = mixed_df["group"].nunique() if has_group else 1
        time_n = mixed_df["time_factor"].nunique() if has_time else 1
        if has_time and time_n > 1:
            subject_time_counts = mixed_df.groupby("subject")["time_factor"].nunique()
            subjects_with_multi_time = int((subject_time_counts >= 2).sum())
            if subjects_with_multi_time < MIN_REPEATED_SUBJECTS_FOR_MIXED:
                rows.append(
                    {
                        "metric": metric,
                        "group": "all",
                        "analysis": "mixedlm",
                        "kpss_stat": np.nan,
                        "p_value": np.nan,
                        "n": int(len(mixed_df)),
                        "note": (
                            "mixedlm_skipped:"
                            f"requires>={MIN_REPEATED_SUBJECTS_FOR_MIXED}_subjects_spanning_multiple_timepoints"
                        ),
                    }
                )
                continue
        formula = "metric_value ~ 1"
        if has_group and group_n > 1:
            formula = "metric_value ~ C(group)"
        if has_time and time_n > 1:
            formula = f"{formula} + C(time_factor)" if formula != "metric_value ~ 1" else "metric_value ~ C(time_factor)"
        if has_group and has_time and group_n > 1 and time_n > 1:
            formula = "metric_value ~ C(group) * C(time_factor)"

        if formula == "metric_value ~ 1":
            rows.append(
                {
                    "metric": metric,
                    "group": "all",
                    "analysis": "mixedlm",
                    "kpss_stat": np.nan,
                    "p_value": np.nan,
                    "n": int(len(mixed_df)),
                    "note": "mixedlm_skipped:insufficient_group_and_time_variation",
                }
            )
            continue

        try:
            model = smf.mixedlm(formula, data=mixed_df, groups=mixed_df["subject"])
            fit = model.fit(reml=False)
            rows.append(
                {
                    "metric": metric,
                    "group": "all",
                    "analysis": "mixedlm",
                    "kpss_stat": np.nan,
                    "p_value": np.nan,
                    "n": int(len(mixed_df)),
                    "note": f"ok ({formula})",
                    "model_aic": float(getattr(fit, "aic", np.nan)),
                    "model_bic": float(getattr(fit, "bic", np.nan)),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "metric": metric,
                    "group": "all",
                    "analysis": "mixedlm",
                    "kpss_stat": np.nan,
                    "p_value": np.nan,
                    "n": int(len(mixed_df)),
                    "note": f"mixedlm_failed:{exc}",
                }
            )

    return pd.DataFrame(rows)


@dataclass(slots=True)
class GroupStatsArtifacts:
    overview_csv: str
    pairwise_csv: str
    kpss_csv: str
    effects_csv: str
    plots_dir: str


def render_group_plots(
    video_summary_df: pd.DataFrame,
    omnibus_df: pd.DataFrame,
    *,
    out_dir: str | Path,
    group_col: str = "group",
) -> str:
    target_dir = Path(out_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    if video_summary_df is None or video_summary_df.empty or group_col not in video_summary_df.columns:
        return str(target_dir)

    numeric_metrics = _resolve_numeric_metrics(
        video_summary_df,
        exclude={group_col, "video_id", "video_name", "video_path", "subject_id", "time_point"},
    )
    if not numeric_metrics:
        return str(target_dir)

    top_metrics = numeric_metrics[: min(6, len(numeric_metrics))]
    for metric in top_metrics:
        plot_df = video_summary_df[[group_col, metric]].copy()
        plot_df[metric] = _coerce_numeric(plot_df[metric])
        plot_df = plot_df.dropna(subset=[group_col, metric])
        if plot_df.empty:
            continue
        fig = plt.figure(figsize=(7.5, 4.2))
        ax = fig.add_subplot(111)
        sns.boxplot(data=plot_df, x=group_col, y=metric, ax=ax, color="#9ecae1")
        sns.stripplot(data=plot_df, x=group_col, y=metric, ax=ax, color="#1f2937", alpha=0.65, size=4, jitter=0.20)
        ax.set_title(f"{metric} by {group_col}")
        ax.set_xlabel(group_col)
        ax.set_ylabel(metric)
        fig.tight_layout()
        fig.savefig(target_dir / f"{metric}_by_group.png", dpi=170)
        plt.close(fig)

    if omnibus_df is not None and not omnibus_df.empty and "p_adj" in omnibus_df.columns:
        heat_df = omnibus_df[["metric", "p_adj"]].dropna().copy()
        if not heat_df.empty:
            heat_df = heat_df.set_index("metric")
            fig = plt.figure(figsize=(6.2, max(3.5, 0.35 * len(heat_df))))
            ax = fig.add_subplot(111)
            sns.heatmap(
                -np.log10(heat_df[["p_adj"]].clip(lower=1e-12)),
                annot=heat_df[["p_adj"]].round(4),
                fmt="",
                cmap="viridis",
                cbar_kws={"label": "-log10(p_adj)"},
                ax=ax,
            )
            ax.set_title("Adjusted p-values (omnibus)")
            ax.set_xlabel("Omnibus significance column")
            ax.set_ylabel("Metric")
            fig.tight_layout()
            fig.savefig(target_dir / "omnibus_padj_heatmap.png", dpi=170)
            plt.close(fig)

    return str(target_dir)


def export_group_stats_bundle(
    video_summary_df: pd.DataFrame,
    *,
    output_dir: str | Path,
    correction_method: str = "fdr_bh",
    include_kpss: bool = True,
    categorical_factors: list[str] | None = None,
    render_plots: bool = True,
    log_fn: Callable[[str, str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, GroupStatsArtifacts]:
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    omnibus_df, pairwise_df, effects_df = compute_nonparametric_group_stats(
        video_summary_df,
        group_col="group",
        subject_col="subject_id",
        time_col="time_point",
        correction_method=correction_method,
        categorical_factors=categorical_factors,
        log_fn=log_fn,
    )
    kpss_df = compute_kpss_and_mixed_effects(video_summary_df, group_col="group") if include_kpss else pd.DataFrame()

    overview_csv = output_root / "group_stats_overview.csv"
    pairwise_csv = output_root / "group_pairwise_tests.csv"
    kpss_csv = output_root / "group_kpss_mixed_effects.csv"
    effects_csv = output_root / "group_effect_sizes.csv"
    plots_dir = output_root / "plots"

    omnibus_df.to_csv(overview_csv, index=False)
    pairwise_df.to_csv(pairwise_csv, index=False)
    kpss_df.to_csv(kpss_csv, index=False)
    effects_df.to_csv(effects_csv, index=False)
    if render_plots:
        render_group_plots(video_summary_df, omnibus_df, out_dir=plots_dir, group_col="group")

    artifacts = GroupStatsArtifacts(
        overview_csv=str(overview_csv),
        pairwise_csv=str(pairwise_csv),
        kpss_csv=str(kpss_csv),
        effects_csv=str(effects_csv),
        plots_dir=str(plots_dir),
    )
    return omnibus_df, pairwise_df, kpss_df, effects_df, artifacts
