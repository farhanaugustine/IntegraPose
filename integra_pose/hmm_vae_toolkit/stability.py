"""ADP-4 Commit E — Cluster stability audit.

Re-runs ``cluster_per_class`` with N different seeds and computes the
pairwise Adjusted Rand Index (ARI) between the resulting cluster
assignments **per class**. The output answers a question every reviewer
of an unsupervised analysis asks:

    "If you re-ran this with a different seed, would you get the same
    clusters?"

If yes, the clusters are stable and defensible. If no, they're an
artifact of the random initialization and the user shouldn't trust
them. The defensibility report panel will refuse to surface a sub-cluster
as a "candidate novel behavior" without a stability score above a
threshold.

Why ARI specifically:

- ARI is symmetric, bounded in ``[-1, 1]`` (1 = identical clusterings,
  0 = chance, negative = worse than chance).
- ARI is **label-permutation invariant** — HDBSCAN's "cluster 0" in
  seed 42 might be the same group of frames as "cluster 2" in seed 43.
  ARI doesn't care; it only looks at which pairs of frames were
  grouped together. So we don't need Hungarian matching to align
  labels across seeds before scoring.
- Sklearn's ``adjusted_rand_score`` is the canonical implementation;
  we don't reimplement.

Contract with downstream code:

- ``compute_class_stability`` is **pure**: given the same input dataframe
  and same base seed, it returns the same numbers. No global RNG.
- The N additional clustering runs are independent of the user's
  primary run; the primary cluster assignment is preserved. Stability
  is only used for reporting / signal-scoring, never to override the
  primary labels.
- Cost: N additional clusterings per call. Default N=3 means 3×
  the runtime of the main run. Surface this in the UI as opt-in.

Usage:

    from .stability import compute_class_stability
    from .per_class_clustering import cluster_per_class

    primary_df, primary_result = cluster_per_class(detections_df)
    stability = compute_class_stability(detections_df, n_seeds=3)
    # stability.per_class[0].mean_ari → 0.78 (stable)
    # stability.per_class[1].mean_ari → 0.23 (seed-sensitive, suspicious)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# Default seed offset stride — seeds tested are
# ``[base_seed, base_seed+1, ..., base_seed + n_seeds - 1]``.
# Default base seed = 42 to match per_class_clustering and splits.py;
# anyone wanting a different seed family edits the constant in code,
# no UI knob (per user, "ease of use means loyalty later").
DEFAULT_BASE_SEED = 42
DEFAULT_N_SEEDS = 3

# Mean-ARI threshold above which a class is labeled "stable" in the
# report panel. 0.5 is a common heuristic for "moderately stable" in
# the clustering-stability literature; raise to 0.7 if you want to
# only surface very-stable clusters.
STABILITY_THRESHOLD_STABLE = 0.5


@dataclass
class ClassStability:
    """Stability outcome for one YOLO class across N seed runs."""

    class_id: int
    class_name: str = ""
    n_seeds: int = 0
    seeds_used: list = field(default_factory=list)
    # Pairwise ARI matrix shape = (n_seeds, n_seeds), diagonal = 1.0.
    # Stored as a list-of-lists for easy JSON serialization in the
    # defensibility report.
    ari_matrix: list = field(default_factory=list)
    mean_ari: float = float("nan")
    min_ari: float = float("nan")
    n_clusters_per_seed: list = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""

    @property
    def is_stable(self) -> bool:
        """True iff mean off-diagonal ARI ≥ ``STABILITY_THRESHOLD_STABLE``."""
        try:
            return float(self.mean_ari) >= STABILITY_THRESHOLD_STABLE
        except Exception:
            return False


@dataclass
class StabilityReport:
    """Top-level stability outcome across all classes.

    The report panel renders one row per ``ClassStability`` and a
    summary line ("3 of 4 classes are stable").
    """

    per_class: list  # list[ClassStability]
    n_seeds: int
    base_seed: int

    @property
    def n_stable(self) -> int:
        return sum(1 for c in self.per_class if c.is_stable and not c.skipped)

    @property
    def n_total(self) -> int:
        return sum(1 for c in self.per_class if not c.skipped)

    @property
    def overall_summary(self) -> str:
        if self.n_total == 0:
            return "No classes were stability-tested (all skipped)."
        return f"{self.n_stable} of {self.n_total} classes are stable (mean ARI ≥ {STABILITY_THRESHOLD_STABLE})."


def compute_class_stability(
    detections_df,
    *,
    n_seeds: int = DEFAULT_N_SEEDS,
    base_seed: int = DEFAULT_BASE_SEED,
    cluster_kwargs: Optional[dict] = None,
) -> StabilityReport:
    """Re-run per-class clustering N times and compute pairwise ARI.

    For each YOLO class present in ``detections_df``, this function:

      1. Runs ``cluster_per_class`` ``n_seeds`` times with seeds
         ``[base_seed, base_seed+1, ..., base_seed+n_seeds-1]``.
      2. Extracts per-frame cluster labels for that class from each run.
      3. Computes the upper-triangular pairwise ARI between the N
         label vectors.
      4. Records the mean (off-diagonal) and min ARI as the class's
         stability headline.

    Classes that ``cluster_per_class`` skipped (too few frames) are
    skipped here too, with the same skip_reason propagated.

    Args:
        detections_df: Frame-level dataframe (must have ``class_id`` and
            ``feature_vector``). Same input shape as
            ``cluster_per_class``.
        n_seeds: Number of seeds to run. Default 3. Must be ≥ 2 (one
            seed gives no pairwise comparison).
        base_seed: First seed in the series. Default 42.
        cluster_kwargs: Optional kwargs forwarded to ``cluster_per_class``
            (``min_class_size``, ``min_cluster_size``, ``umap_neighbors``,
            ``umap_components``). When None, ``cluster_per_class`` defaults
            apply.

    Returns:
        :class:`StabilityReport` with per-class stability outcomes.

    Raises:
        ValueError: If ``n_seeds < 2`` or input frame is invalid.
    """
    if n_seeds < 2:
        raise ValueError(
            f"compute_class_stability: n_seeds must be ≥ 2 to compute "
            f"pairwise ARI; got {n_seeds}."
        )

    # Late imports: keep this module cheap, and let test code stub
    # cluster_per_class without dragging UMAP/HDBSCAN into the import.
    from .per_class_clustering import cluster_per_class  # noqa: PLC0415

    cluster_kwargs = dict(cluster_kwargs or {})
    cluster_kwargs.pop("seed", None)  # we override per iteration

    seeds_to_test = [base_seed + i for i in range(n_seeds)]
    logger.info(
        "Stability audit: running %d clusterings with seeds %s",
        n_seeds, seeds_to_test,
    )

    # Run all N clusterings and store the per-frame label series.
    all_runs = []
    for seed in seeds_to_test:
        df_i, multi_i = cluster_per_class(
            detections_df, seed=seed, **cluster_kwargs,
        )
        all_runs.append((seed, df_i, multi_i))

    # Build per-class results.
    per_class_results: list[ClassStability] = []
    primary_multi = all_runs[0][2]
    for primary_class in primary_multi.classes:
        class_id = primary_class.class_id
        class_name = primary_class.class_name

        if primary_class.skipped:
            per_class_results.append(
                ClassStability(
                    class_id=class_id,
                    class_name=class_name,
                    n_seeds=n_seeds,
                    seeds_used=seeds_to_test,
                    skipped=True,
                    skip_reason=primary_class.skip_reason,
                )
            )
            continue

        # Pull cluster labels for this class from each seed run.
        per_seed_labels: list = []
        per_seed_n_clusters: list = []
        for seed, df_i, multi_i in all_runs:
            class_mask = df_i["class_id"] == class_id
            labels = list(df_i.loc[class_mask, "cluster_label"].values)
            per_seed_labels.append(labels)
            # Find this class's n_clusters in this seed's multi_result.
            n_clusters_this_seed = 0
            for c in multi_i.classes:
                if c.class_id == class_id and not c.skipped:
                    n_clusters_this_seed = c.n_clusters
                    break
            per_seed_n_clusters.append(n_clusters_this_seed)

        ari_matrix = _pairwise_ari_matrix(per_seed_labels)
        mean_ari, min_ari = _summarize_off_diagonal(ari_matrix)

        per_class_results.append(
            ClassStability(
                class_id=class_id,
                class_name=class_name,
                n_seeds=n_seeds,
                seeds_used=seeds_to_test,
                ari_matrix=ari_matrix,
                mean_ari=mean_ari,
                min_ari=min_ari,
                n_clusters_per_seed=per_seed_n_clusters,
            )
        )

    return StabilityReport(
        per_class=per_class_results,
        n_seeds=n_seeds,
        base_seed=base_seed,
    )


def _pairwise_ari_matrix(per_seed_labels: list) -> list:
    """Compute the symmetric N×N ARI matrix.

    Diagonal is 1.0. Returns nested list (not numpy) for easy JSON
    serialization downstream.
    """
    from sklearn.metrics import adjusted_rand_score  # noqa: PLC0415

    n = len(per_seed_labels)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            try:
                ari = float(adjusted_rand_score(per_seed_labels[i], per_seed_labels[j]))
            except Exception as exc:
                logger.warning("ARI computation failed for seeds %d/%d: %s", i, j, exc)
                ari = float("nan")
            matrix[i][j] = ari
            matrix[j][i] = ari
    return matrix


def _summarize_off_diagonal(matrix: list) -> tuple:
    """Mean and min of the strictly upper-triangular entries."""
    import math

    n = len(matrix)
    values = []
    for i in range(n):
        for j in range(i + 1, n):
            v = matrix[i][j]
            if v is None:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            if math.isnan(fv):
                continue
            values.append(fv)
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    return mean, min(values)


__all__ = [
    "compute_class_stability",
    "ClassStability",
    "StabilityReport",
    "DEFAULT_BASE_SEED",
    "DEFAULT_N_SEEDS",
    "STABILITY_THRESHOLD_STABLE",
]
