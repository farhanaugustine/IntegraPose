"""Per-class sub-behavior clustering for Tab 7.

Detections are grouped by YOLO class and clustered independently. This
preserves the supervised class label while assigning a sub-cluster label to
each frame. UMAP is used for optional dimensionality reduction, followed by
HDBSCAN clustering within each class.

Processing flow:

    detections_df (with `class_id` + `feature_vector`)
        ↓  for each class_id:
    UMAP reduce (only when n >> umap_neighbors)
        ↓
    HDBSCAN cluster
        ↓
    Per-frame `cluster_label` namespaced as f"{class_id}:{local_id}"
        ↓
    Pass to `aggregate_states_into_bouts(state_column='cluster_label')`
        ↓
    Sub-behavior bouts for naming, held-out evaluation, and clip export.

Cluster IDs are namespaced per class (e.g. ``"0:1"``, ``"0:2"``,
``"1:1"``) so two different classes can each have their own
sub-cluster 1 without collision. Noise frames keep ``"-1"`` (string)
so the existing aggregators recognize them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# Default seed for UMAP reproducibility. This seed is independent of the
# subject-split seed defined in splits.py.
DEFAULT_CLUSTERING_SEED = 42

# Classes with fewer frames are skipped and recorded in the result.
# 30 is a heuristic — above it HDBSCAN with min_cluster_size=10 has
# enough room to actually find structure; below it the result is just
# noise. Override per-call via `min_class_size`.
DEFAULT_MIN_CLASS_SIZE = 30

# UMAP / HDBSCAN defaults that work well for normalized pose features.
# These are exposed as kwargs because individual labs may need to tune
# them, but the defaults are good starting points for ~20-50 frame/sec
# pose data.
DEFAULT_UMAP_NEIGHBORS = 15
DEFAULT_UMAP_COMPONENTS = 5
DEFAULT_MIN_CLUSTER_SIZE = 10


@dataclass
class PerClassClusterResult:
    """One class's clustering outcome.

    Returned inside ``MultiClassClusterResult.classes`` so the caller
    can render a "Walking (3 sub-types)" / "Rearing (skipped: 12
    frames)" summary without rederiving anything.
    """

    class_id: int
    class_name: str  # human label, e.g., "walking"
    n_frames: int
    n_clusters: int  # HDBSCAN clusters found, NOT including noise
    n_noise_frames: int
    skipped: bool = False  # true when class had too few frames
    skip_reason: str = ""
    sub_cluster_labels: list = field(default_factory=list)
    # ^^ unique per-class cluster IDs that were found (e.g. [0, 1, 2]).
    # Combined with class_id they form the namespaced keys in the df.


@dataclass
class MultiClassClusterResult:
    """Top-level result of one per-class clustering run.

    The DataFrame is the primary output (rows now have
    ``cluster_label``); this dataclass is the metadata so the report
    panel can describe what happened across all classes.
    """

    classes: list  # list[PerClassClusterResult]
    seed: int
    label_column: str = "cluster_label"
    skipped_classes: list = field(default_factory=list)
    backend_runs: list = field(default_factory=list)

    @property
    def total_clusters(self) -> int:
        return sum(c.n_clusters for c in self.classes if not c.skipped)

    @property
    def total_noise_frames(self) -> int:
        return sum(c.n_noise_frames for c in self.classes if not c.skipped)


def _format_namespaced_label(class_id: int, local_id: int) -> str:
    """Build the cluster_label string for a single frame.

    Noise frames use the literal ``"-1"`` so existing aggregators that
    already special-case noise on the integer or string ``-1`` keep
    working. Normal clusters look like ``"0:0"``, ``"0:1"``, ``"1:0"``,
    ... — the prefix is the YOLO class_id; the suffix is the
    within-class HDBSCAN cluster id.
    """
    if local_id == -1:
        return "-1"
    return f"{int(class_id)}:{int(local_id)}"


def cluster_per_class(
    detections_df,
    *,
    class_names: Optional[dict] = None,
    min_class_size: int = DEFAULT_MIN_CLASS_SIZE,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    umap_neighbors: int = DEFAULT_UMAP_NEIGHBORS,
    umap_components: int = DEFAULT_UMAP_COMPONENTS,
    seed: int = DEFAULT_CLUSTERING_SEED,
    label_column: str = "cluster_label",
):
    """Cluster sub-behaviors **within each YOLO class**.

    For each unique value of ``class_id`` in ``detections_df``, this
    function:

      1. Filters to that class's rows.
      2. Skips the class if there are fewer than ``min_class_size``
         rows (records the skip reason in the result).
      3. UMAP-reduces ``feature_vector`` to ``umap_components`` dims
         using ``umap_neighbors`` (auto-clamped if the class has fewer
         frames than ``umap_neighbors``).
      4. HDBSCAN-clusters the reduced vectors with ``min_cluster_size``.
      5. Writes the namespaced cluster id back to the row's
         ``label_column`` (default ``"cluster_label"``).

    Args:
        detections_df: Frame-level dataframe carrying ``class_id`` and
            ``feature_vector`` columns. Other columns pass through
            untouched.
        class_names: Optional mapping ``{class_id: human_label}`` from
            the YAML/manifest. Used purely for nicer log lines and the
            report dataclass — the actual clustering doesn't depend on
            it.
        min_class_size: Skip any class with fewer rows. Default 30.
        min_cluster_size: HDBSCAN ``min_cluster_size``. Default 10.
        umap_neighbors: UMAP ``n_neighbors``. Use 0 to skip UMAP and
            cluster the normalized feature vectors directly. Otherwise this
            must be at least 2. Default 15.
        umap_components: UMAP target dim. Default 5.
        seed: UMAP ``random_state``. Default 42 (matches splits.py).
        label_column: Column name to write the cluster ids into.
            Default ``"cluster_label"``. Pre-existing values in this
            column are overwritten.

    Returns:
        Tuple[pandas.DataFrame, MultiClassClusterResult]. The first is a
        copy of ``detections_df`` with ``label_column`` populated; the
        second is the metadata describing per-class outcomes.

    Raises:
        ValueError: If ``detections_df`` lacks ``class_id`` or
            ``feature_vector``. Both are produced by upstream
            ``read_detections`` and the feature pipeline.
    """
    # Late imports keep this module cheap to import.
    import numpy as np

    for name, value in (("min_class_size", min_class_size), ("min_cluster_size", min_cluster_size),
                        ("umap_neighbors", umap_neighbors), ("umap_components", umap_components), ("seed", seed)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")

    if min_class_size < 1:
        raise ValueError("cluster_per_class: min_class_size must be at least 1.")
    if min_cluster_size < 2:
        raise ValueError("cluster_per_class: min_cluster_size must be at least 2.")
    if umap_neighbors < 0 or umap_neighbors == 1:
        raise ValueError(
            "cluster_per_class: umap_neighbors must be 0 (disabled) or at least 2."
        )
    if umap_components < 1:
        raise ValueError("cluster_per_class: umap_components must be at least 1.")

    if detections_df is None or len(detections_df) == 0:
        raise ValueError("cluster_per_class: detections_df is empty.")
    for col in ("class_id", "feature_vector"):
        if col not in detections_df.columns:
            raise ValueError(
                f"cluster_per_class: required column {col!r} missing from "
                "detections_df. Run feature computation before clustering."
            )

    original_index = detections_df.index.copy()
    out_df = detections_df.copy().reset_index(drop=True)
    class_values = out_df["class_id"]
    try:
        numeric_ids = np.asarray(class_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("class_id must contain finite non-negative integer IDs.") from exc
    if not np.isfinite(numeric_ids).all() or (numeric_ids < 0).any() or (numeric_ids != np.floor(numeric_ids)).any():
        raise ValueError("class_id must contain finite non-negative integer IDs.")
    if any(isinstance(value, (str, bool)) for value in class_values):
        raise ValueError("class_id must contain numeric integer IDs, not strings or booleans.")
    try:
        all_features = np.asarray(list(out_df["feature_vector"]), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Feature vectors must have one consistent numeric shape.") from exc
    if all_features.ndim != 2 or not all_features.shape[1] or not np.isfinite(all_features).all():
        raise ValueError("Feature vectors must be non-empty, equal-length and finite.")
    # Initialize unassigned and skipped rows with the noise sentinel.
    out_df[label_column] = "-1"
    out_df["cluster_status"] = "not_processed"
    out_df["cluster_backend"] = "not_run"
    out_df['cluster_embedding'] = [None] * len(out_df)
    backend_runs = []

    classes: list[PerClassClusterResult] = []
    skipped: list[int] = []

    unique_class_ids = sorted({int(c) for c in out_df["class_id"].dropna().unique()})

    for class_id in unique_class_ids:
        class_name = ""
        if isinstance(class_names, dict):
            class_name = str(class_names.get(class_id) or class_names.get(str(class_id)) or "")

        class_mask = out_df["class_id"] == class_id
        class_indices = out_df.index[class_mask].tolist()
        n_frames = len(class_indices)

        if n_frames < max(min_class_size, min_cluster_size):
            out_df.loc[class_indices, "cluster_status"] = "insufficient_samples"
            reason = f"Only {n_frames} samples; need at least {max(min_class_size, min_cluster_size)} to cluster."
            logger.warning(
                "Skipping class %d (%s): %s",
                class_id, class_name or "unnamed", reason,
            )
            classes.append(
                PerClassClusterResult(
                    class_id=class_id,
                    class_name=class_name,
                    n_frames=n_frames,
                    n_clusters=0,
                    n_noise_frames=0,
                    skipped=True,
                    skip_reason=reason,
                )
            )
            skipped.append(class_id)
            continue

        # Pull this class's feature matrix.
        feature_vectors = list(out_df.loc[class_indices, "feature_vector"].values)
        try:
            feature_matrix = np.array([np.asarray(fv, dtype=float) for fv in feature_vectors])
        except Exception as exc:
            raise ValueError(
                f"cluster_per_class: could not stack feature vectors for class "
                f"{class_id} ({class_name}): {exc}"
            )

        backend_info = {'capture_embedding': True}
        local_labels = _cluster_one_class(
            feature_matrix,
            min_cluster_size=min_cluster_size,
            umap_neighbors=umap_neighbors,
            umap_components=umap_components,
            seed=seed,
            metadata=backend_info,
        )
        embedding = backend_info.pop('_embedding', None)
        backend_info.pop('capture_embedding', None)
        if embedding is not None:
            for row_index, coordinates in zip(class_indices, embedding):
                out_df.at[row_index, 'cluster_embedding'] = coordinates
        backend_info.update(class_id=class_id)
        backend_runs.append(backend_info)

        # Write namespaced labels back to the dataframe.
        namespaced = [_format_namespaced_label(class_id, int(lab)) for lab in local_labels]
        out_df.loc[class_indices, label_column] = namespaced
        out_df.loc[class_indices, "cluster_status"] = ["noise" if int(lab) == -1 else "clustered" for lab in local_labels]
        out_df.loc[class_indices, "cluster_backend"] = "cpu"

        local_unique = sorted({int(lab) for lab in local_labels if int(lab) != -1})
        n_noise = int(sum(1 for lab in local_labels if int(lab) == -1))
        classes.append(
            PerClassClusterResult(
                class_id=class_id,
                class_name=class_name,
                n_frames=n_frames,
                n_clusters=len(local_unique),
                n_noise_frames=n_noise,
                sub_cluster_labels=local_unique,
            )
        )
        logger.info(
            "Class %d (%s): %d frames → %d sub-cluster(s), %d noise frame(s).",
            class_id, class_name or "unnamed", n_frames, len(local_unique), n_noise,
        )

    result = MultiClassClusterResult(
        classes=classes,
        seed=seed,
        label_column=label_column,
        skipped_classes=skipped,
        backend_runs=backend_runs,
    )
    out_df.index = original_index
    return out_df, result


def _cluster_one_class(
    feature_matrix,
    *,
    min_cluster_size: int,
    umap_neighbors: int,
    umap_components: int,
    seed: int,
    metadata=None,
):
    from .cluster_backend import cluster_features

    return cluster_features(
        feature_matrix, min_cluster_size=min_cluster_size,
        umap_neighbors=umap_neighbors, umap_components=umap_components,
        seed=seed, metadata=metadata,
    )


__all__ = [
    "cluster_per_class",
    "PerClassClusterResult",
    "MultiClassClusterResult",
    "DEFAULT_CLUSTERING_SEED",
    "DEFAULT_MIN_CLASS_SIZE",
    "DEFAULT_MIN_CLUSTER_SIZE",
    "DEFAULT_UMAP_NEIGHBORS",
    "DEFAULT_UMAP_COMPONENTS",
]
