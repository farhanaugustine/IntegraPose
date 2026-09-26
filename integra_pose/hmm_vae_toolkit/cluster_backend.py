"""CPU-first UMAP/HDBSCAN execution with explicit backend provenance."""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version

import numpy as np


def package_versions():
    names = ["numpy", "scikit-learn", "umap-learn", "hdbscan"]
    result = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            pass
    return result


def cluster_features(matrix, *, min_cluster_size=10, umap_neighbors=15,
                     umap_components=5, seed=42, metadata=None):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("Features must be a non-empty 2D matrix.")
    if not np.isfinite(matrix).all():
        raise ValueError("Features contain NaN or infinity; resolve missing data first.")
    for name, value, minimum in (("min_cluster_size", min_cluster_size, 2),
                                  ("umap_components", umap_components, 1),
                                  ("seed", seed, 0), ("umap_neighbors", umap_neighbors, 0)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if umap_neighbors == 1:
        raise ValueError("umap_neighbors must be 0 (disabled) or at least 2.")

    details = metadata if metadata is not None else {}
    capture_embedding = details.pop('capture_embedding', False)
    details.update(backend="cpu", versions=package_versions(), seed=int(seed),
                   min_cluster_size=int(min_cluster_size), n_samples=len(matrix),
                   n_features=matrix.shape[1], reduction="none", umap_reason="disabled",
                   umap_neighbors=int(umap_neighbors), umap_components=int(umap_components))
    if len(matrix) < min_cluster_size:
        details["status"] = "insufficient_samples"
        return np.full(len(matrix), -1, dtype=int)

    reduced = matrix
    if umap_neighbors and len(matrix) > max(umap_neighbors + 1, umap_components + 1):
        reducer_args = dict(n_neighbors=umap_neighbors, n_components=umap_components,
                            random_state=int(seed), metric="euclidean")
        reducer = importlib.import_module("umap").UMAP(**reducer_args)
        reduced = np.asarray(reducer.fit_transform(matrix))
        if not np.isfinite(reduced).all():
            raise ValueError("UMAP produced non-finite coordinates.")
        details.update(reduction="umap", umap_reason="applied")
    elif umap_neighbors:
        details["umap_reason"] = "insufficient_samples_for_requested_neighborhood_or_dimensions"

    if capture_embedding:
        details['_embedding'] = reduced.tolist()
    cluster_args = dict(min_cluster_size=min_cluster_size, metric="euclidean",
                        cluster_selection_method="eom")
    clusterer = importlib.import_module("hdbscan").HDBSCAN(**cluster_args)
    labels = np.asarray(clusterer.fit_predict(reduced), dtype=int).reshape(-1)
    if len(labels) != len(matrix) or (labels < -1).any():
        raise ValueError("Clustering backend returned invalid assignments.")
    details["status"] = "completed"
    return labels
