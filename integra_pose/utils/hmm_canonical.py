"""Canonical state ordering for hmmlearn HMMs.

The HMM "label-switching" problem: hmmlearn assigns state indices arbitrarily
during EM, so two runs on identical data can produce ``State 0 = grooming`` in
one run and ``State 0 = rearing`` in the next — even with the same seed,
because EM convergence wobble can permute states. This module applies a
deterministic permutation after fit so downstream code (figures, exports,
state-name mappings) sees stable IDs across runs.

The ordering convention is **type-specific** (a single universal ordering like
"by stationary probability" is unstable when two states have similar π — small
EM wobble flips them). Conventions:

- **Categorical** (``CategoricalHMM``): sort by ``argmax(emissionprob_)``
  (i.e. which observation each state most strongly emits). Stable because
  argmax over a discrete probability vector tolerates small wobble in the
  individual entries. Tiebreak by stationary probability, then by original
  state index.

- **Gaussian / GMM** (``GaussianHMM`` / ``GMMHMM``): sort by ``means_[:, 0]``
  ascending (first feature dimension of the mean). For GMM, use the weighted
  mean across mixture components. Tiebreak by stationary probability, then
  by original state index.

If you change this convention, bump ``CANONICAL_ORDERING_VERSION`` and add a
migration path so old artefacts can be re-canonicalised on load.
"""

from __future__ import annotations

import numpy as np

CANONICAL_ORDERING_VERSION = 1


def _stationary_distribution(transmat: np.ndarray) -> np.ndarray:
    """Left eigenvector with eigenvalue 1, normalised to sum to 1.

    Falls back to a uniform distribution if the eigendecomposition is
    degenerate. Never raises — used as a tiebreaker, not load-bearing.
    """
    n = transmat.shape[0]
    try:
        eigvals, eigvecs = np.linalg.eig(transmat.T)
        # Find the eigenvalue closest to 1.
        idx = int(np.argmin(np.abs(eigvals - 1.0)))
        vec = np.real(eigvecs[:, idx])
        if not np.all(np.isfinite(vec)):
            return np.full(n, 1.0 / n)
        # Force non-negative + normalise.
        vec = np.abs(vec)
        total = float(vec.sum())
        if total <= 0 or not np.isfinite(total):
            return np.full(n, 1.0 / n)
        return vec / total
    except (np.linalg.LinAlgError, ValueError):
        return np.full(n, 1.0 / n)


def compute_canonical_permutation(model) -> np.ndarray:
    """Return an integer array ``perm`` such that the new ``state[i]`` is the
    old ``state[perm[i]]``.

    The convention depends on the HMM type (see module docstring). The result
    is deterministic — calling this twice on the same fitted model returns
    the same permutation.
    """
    n_components = int(getattr(model, "n_components", 0))
    if n_components <= 0:
        return np.arange(0)

    transmat = np.asarray(getattr(model, "transmat_", np.eye(n_components)))
    stationary = _stationary_distribution(transmat)

    if hasattr(model, "emissionprob_") and model.emissionprob_ is not None:
        # Categorical: primary key is argmax over emissions.
        emissions = np.asarray(model.emissionprob_)
        primary = np.argmax(emissions, axis=1).astype(float)
    elif hasattr(model, "means_") and model.means_ is not None:
        means = np.asarray(model.means_)
        if means.ndim == 3:
            # GMMHMM: shape (n_components, n_mix, n_features). Collapse over
            # mixture by mixture-weight average if available, else simple mean.
            weights = getattr(model, "weights_", None)
            if weights is not None:
                w = np.asarray(weights)
                # weighted mean: (n_components, n_features)
                means = np.einsum("ij,ijk->ik", w, means)
            else:
                means = means.mean(axis=1)
        if means.ndim == 2 and means.shape[1] >= 1:
            primary = means[:, 0].astype(float)
        else:
            primary = np.zeros(n_components, dtype=float)
    else:
        # Last resort: stationary alone.
        primary = stationary.astype(float)

    # Stable lexicographic sort: primary, then stationary (negated for
    # descending — more-occupied states win ties), then index.
    indices = np.arange(n_components)
    order = np.lexsort((indices, -stationary, primary))
    return order.astype(int)


def apply_permutation(model, perm: np.ndarray) -> None:
    """Re-index every HMM attribute that depends on state order, in place.

    Idempotent in the sense that applying ``perm = identity`` is a no-op.
    Only touches attributes that actually exist on ``model``.
    """
    perm = np.asarray(perm, dtype=int)
    if perm.size == 0:
        return

    if hasattr(model, "transmat_") and model.transmat_ is not None:
        # Permute both rows and columns: new_T[i, j] = old_T[perm[i], perm[j]].
        model.transmat_ = np.asarray(model.transmat_)[np.ix_(perm, perm)]

    if hasattr(model, "startprob_") and model.startprob_ is not None:
        model.startprob_ = np.asarray(model.startprob_)[perm]

    if hasattr(model, "emissionprob_") and model.emissionprob_ is not None:
        model.emissionprob_ = np.asarray(model.emissionprob_)[perm, :]

    if hasattr(model, "means_") and model.means_ is not None:
        model.means_ = np.asarray(model.means_)[perm]

    if hasattr(model, "covars_") and model.covars_ is not None:
        model.covars_ = np.asarray(model.covars_)[perm]

    if hasattr(model, "weights_") and model.weights_ is not None:
        # GMMHMM mixture weights: shape (n_components, n_mix). Permute rows.
        weights = np.asarray(model.weights_)
        if weights.ndim >= 1 and weights.shape[0] == perm.size:
            model.weights_ = weights[perm]


def canonicalise(model) -> dict:
    """Compute the canonical permutation, apply it, and return a small report.

    The report is intended for embedding in run metadata so a reviewer can
    verify that canonical ordering was applied and at what version of the
    convention.
    """
    perm = compute_canonical_permutation(model)
    was_identity = bool(np.array_equal(perm, np.arange(perm.size)))
    if not was_identity:
        apply_permutation(model, perm)
    return {
        "canonical_ordering_version": CANONICAL_ORDERING_VERSION,
        "permutation": perm.tolist(),
        "was_identity": was_identity,
    }
