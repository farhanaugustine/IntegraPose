"""ADP-4 Tab 7 — per-class sub-behavior clustering.

This module is the new heart of Tab 7. Instead of pooling all detections
into one HMM, we **filter detections to one YOLO class at a time and
cluster within each class**. The output: per-frame sub-cluster labels
that say "this walking frame belongs to sub-type 2 of walking" — which
is what the user actually wants from Tab 7.

Why per-class:

- Tabs 1–6 already produce trustworthy macro labels (walking, rearing,
  grooming) via supervised YOLO-pose. Tab 7's job isn't to second-guess
  those — it's to dissect *within* each known class and surface
  variations the user couldn't have annotated upfront.
- Pooling across classes (the old HMM approach) blurs the question.
  "State 4 contains some walking and some rearing" tells the researcher
  nothing they can act on. "Walking has 3 sub-types" is something they
  can name, defend, and feed back into a finer-grained classifier.
- Each class clusters in its own feature subspace; HDBSCAN's
  ``min_cluster_size`` and noise label naturally filter rare or
  ambiguous frames.

Why HDBSCAN (no HMM, no VAE/LSTM-AE):

- HDBSCAN handles variable-density clusters without a "how many states"
  hyperparameter. The user doesn't have to guess.
- HMM was the wrong fit for per-class because it pools data across
  classes by construction; running one HMM per class would multiply
  the seeded-randomness sources and complicate the report.
- VAE+LSTM-AE adds 200+ epochs of training for embeddings that, in
  practice, give similar HDBSCAN clusters to the raw normalized
  feature vectors UMAP-reduced. Skip the heavy step until a user
  shows it makes a difference for their data.

The flow is intentionally short:

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
    Sub-behavior bouts ready for naming, held-out evaluation, clip export.

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


# Default seed for UMAP reproducibility. The auto-split helper uses 42
# (defined in splits.py) and we mirror it here. The two seeds are
# independent — the split seed assigns subjects to partitions, this seed
# governs UMAP's initial state during dimensionality reduction.
DEFAULT_CLUSTERING_SEED = 42

# When fewer than this many frames belong to a class, we skip clustering
# for that class entirely and surface a "skipped" entry in the result.
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
        umap_neighbors: UMAP ``n_neighbors``. Default 15.
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
            ``read_detections`` + feature pipeline; if either is absent
            we surface a clear error rather than silently fail.
    """
    # Late imports keep this module cheap to import.
    import numpy as np
    import pandas as pd

    if detections_df is None or len(detections_df) == 0:
        raise ValueError("cluster_per_class: detections_df is empty.")
    for col in ("class_id", "feature_vector"):
        if col not in detections_df.columns:
            raise ValueError(
                f"cluster_per_class: required column {col!r} missing from "
                "detections_df. Run feature computation before clustering."
            )

    out_df = detections_df.copy()
    # Initialize the column as the noise sentinel so any rows we don't
    # explicitly assign (e.g. classes that get skipped) end up in noise.
    out_df[label_column] = "-1"

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

        if n_frames < min_class_size:
            reason = f"Only {n_frames} frame(s); need ≥{min_class_size} to cluster."
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
                    n_noise_frames=n_frames,
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

        local_labels = _cluster_one_class(
            feature_matrix,
            min_cluster_size=min_cluster_size,
            umap_neighbors=umap_neighbors,
            umap_components=umap_components,
            seed=seed,
        )

        # Write namespaced labels back to the dataframe.
        namespaced = [_format_namespaced_label(class_id, int(lab)) for lab in local_labels]
        out_df.loc[class_indices, label_column] = namespaced

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
    )
    return out_df, result


def _cluster_one_class(
    feature_matrix,
    *,
    min_cluster_size: int,
    umap_neighbors: int,
    umap_components: int,
    seed: int,
):
    """Run UMAP + HDBSCAN on one class's features. Returns local labels.

    Handles the "feature space is too small to UMAP" edge case by
    skipping UMAP entirely and feeding raw features to HDBSCAN — this
    keeps small classes (just above ``min_class_size``) from failing.
    """
    import numpy as np

    n = feature_matrix.shape[0]

    # If we have fewer rows than UMAP needs, skip UMAP. HDBSCAN can
    # handle the original feature space directly.
    if n <= umap_neighbors + 1:
        reduced = feature_matrix
        logger.debug(
            "Skipping UMAP for small class (%d rows ≤ umap_neighbors+1=%d); "
            "running HDBSCAN on raw features.",
            n, umap_neighbors + 1,
        )
    else:
        try:
            import umap as _umap_lib  # late import — heavy
        except Exception as exc:
            raise ImportError(
                "cluster_per_class: UMAP is required. "
                "`pip install umap-learn`. (For sandboxes without UMAP, "
                "set umap_neighbors=0 to skip dim-reduction — but for "
                "real data UMAP yields better clusters.)"
            ) from exc

        # Clamp neighbors below n; UMAP errors otherwise.
        effective_neighbors = max(2, min(umap_neighbors, n - 1))
        reducer = _umap_lib.UMAP(
            n_neighbors=effective_neighbors,
            n_components=umap_components,
            random_state=seed,
            metric="euclidean",
        )
        reduced = reducer.fit_transform(feature_matrix)

    # HDBSCAN clamp: min_cluster_size > n means no cluster ever; clamp.
    effective_min_cluster = max(2, min(min_cluster_size, n // 2))
    try:
        import hdbscan as _hdbscan_lib  # late import — heavy
    except Exception as exc:
        raise ImportError(
            "cluster_per_class: HDBSCAN is required. "
            "`pip install hdbscan`."
        ) from exc

    clusterer = _hdbscan_lib.HDBSCAN(
        min_cluster_size=effective_min_cluster,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(reduced)

    # Visibility guard: when a class has very few feature rows, HDBSCAN
    # almost always returns -1 (noise) for everything. The user has no
    # way to distinguish that case from a legitimate "many points, no
    # structure found" result without this warning. Emit at WARNING so
    # it surfaces in the toolkit log alongside other run diagnostics.
    if labels.size and (labels == -1).all():
        if n < max(10, effective_min_cluster * 3):
            logger.warning(
                "All %d frames collapsed to noise (HDBSCAN returned -1 for "
                "every point). The class has too few feature rows for "
                "meaningful clustering at min_cluster_size=%d; consider "
                "lowering min_cluster_size, raising min_class_size to "
                "exclude this class entirely, or merging this class with "
                "another upstream.",
                n, effective_min_cluster,
            )
        else:
            logger.warning(
                "All %d frames collapsed to noise (HDBSCAN returned -1 for "
                "every point). The feature distribution shows no detectable "
                "sub-structure at min_cluster_size=%d; consider lowering "
                "min_cluster_size or revisiting feature selection.",
                n, effective_min_cluster,
            )
    return labels


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
