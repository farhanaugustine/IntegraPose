"""Subject-level train/validation/test splitting for Tab 7.

Held-out evaluation requires subjects to remain in separate partitions.
Splitting frames or bouts from the same subject across partitions would
inflate the reported metrics. The split regime is selected from the
smallest group-level subject count:

    min subjects per group ≥ 5  →  60 / 20 / 20  (train / val / test)
    2 ≤ min subjects per group ≤ 4  →  80 / 20         (train / val, no test)
    min subjects per group == 1  →  100 / 0 / 0       (full data, warning)

The smallest cohort determines the regime. The result records the limiting
group so imbalanced cohorts can be identified from the report.


### Required invariants

1. **Split before HMM training. Segment bouts after inference within each
   partition.** Pre-segmenting bouts can leak boundaries across partitions.

2. **Composite `(group, subject_id)` key.** Two groups can independently
   produce a `subject_id` of `"video_01"` through basename fallback.
   Deduplicate on `(group, subject_id)`, not `subject_id`
   alone, or two distinct subjects collapse into one and end up in the
   same partition.

3. **Use local randomness with `random_state=42`.** The split must not
   modify global random-number state used by HMM, UMAP, or torch.

4. **Subjects are atomic.** All frames from a given (group, subject_id)
   stay in one partition. No frame-level shuffling.


### Return shape

A dataframe with a new ``partition`` column (``"train"``, ``"val"``, or
``"test"``) and a :class:`SplitResult` object containing the regime, seed,
and per-group subject assignments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


# Fixed split seed. The value is recorded in SplitResult.seed and is not
# exposed as a GUI control.
DEFAULT_SPLIT_SEED = 42

# Regime labels — referenced from the report panel and from tests, so they
# get string constants rather than free-form strings sprinkled around.
REGIME_FIVE_PLUS = "5plus"
REGIME_TWO_TO_FOUR = "2to4"
REGIME_SINGLE = "single"

# Required columns. We don't import pandas at module load (heavy, slow);
# the runtime check is performed inside auto_split() instead.
_REQUIRED_COLUMNS = ("group", "subject_id")


@dataclass
class SplitResult:
    """Metadata describing what auto_split decided.

    These fields are consumed by the held-out report panel. Keep the shape
    stable for report and export compatibility.
    """

    regime: str  # one of REGIME_* constants above
    seed: int
    # group -> partition -> list of subject ids in that partition
    subjects_per_partition_per_group: dict
    # group -> partition -> count of subjects (a tiny denormalisation so
    # the report panel doesn't have to len() the lists every render).
    n_subjects_per_partition_per_group: dict
    # group -> count of unique subjects in that group (helps the panel
    # explain *why* a regime was chosen)
    n_subjects_per_group: dict
    # The group with the fewest subjects — the one that limited the
    # regime. Empty string when all groups had the same count.
    limiting_group: str = ""
    min_subjects_per_group: int = 0
    partition_column: str = "partition"
    # When the regime is "single" we add this so the report panel can
    # render its yellow note ("Need ≥2 subjects for held-out validation").
    warnings: list = field(default_factory=list)

    @property
    def has_held_out_test(self) -> bool:
        """True iff this split has a non-empty 'test' partition.

        The held-out evaluation panel is hidden when this is False.
        """
        for parts in self.n_subjects_per_partition_per_group.values():
            if parts.get("test", 0) > 0:
                return True
        return False


def _decide_regime(min_subjects: int) -> str:
    if min_subjects >= 5:
        return REGIME_FIVE_PLUS
    if min_subjects >= 2:
        return REGIME_TWO_TO_FOUR
    return REGIME_SINGLE


def _split_subjects_for_regime(
    subjects: list,
    *,
    regime: str,
    rng,
) -> dict:
    """Return ``{partition_name: [subject, ...]}`` for one group.

    ``rng`` is a numpy ``Generator`` (created with ``default_rng(seed)``).
    Subjects are placed deterministically — sorted, shuffled with the
    local generator, then sliced. We don't use sklearn's
    StratifiedShuffleSplit here because we're already iterating one group
    at a time; sklearn is overkill for a 1-label shuffle and brings a
    heavier import. The deterministic-shuffle approach is just as
    reproducible and clearer to read.
    """
    # Sort first so the shuffle is reproducible across machines (dict
    # iteration order on the upstream code shouldn't matter).
    subjects = sorted({str(s) for s in subjects})
    n = len(subjects)
    if n == 0:
        return {"train": [], "val": [], "test": []}

    # Single-subject regime: all into train, no held-out anything.
    if regime == REGIME_SINGLE:
        return {"train": list(subjects), "val": [], "test": []}

    # Permute deterministically, then slice.
    indices = list(range(n))
    rng.shuffle(indices)
    permuted = [subjects[i] for i in indices]

    if regime == REGIME_FIVE_PLUS:
        # 60 / 20 / 20 with at least one in each non-empty bucket.
        n_test = max(1, round(n * 0.2))
        n_val = max(1, round(n * 0.2))
        # Guarantee train >= 1.
        if n_test + n_val >= n:
            n_test = max(1, n // 5)
            n_val = max(1, n // 5)
        n_train = n - n_val - n_test
        train = permuted[:n_train]
        val = permuted[n_train : n_train + n_val]
        test = permuted[n_train + n_val :]
        return {"train": train, "val": val, "test": test}

    if regime == REGIME_TWO_TO_FOUR:
        # 80 / 20, no test.
        n_val = max(1, round(n * 0.2))
        if n_val >= n:
            n_val = 1
        n_train = n - n_val
        return {"train": permuted[:n_train], "val": permuted[n_train:], "test": []}

    raise ValueError(f"Unknown regime: {regime!r}")


def auto_split(detections_df, *, seed: int = DEFAULT_SPLIT_SEED):
    """Split the detection dataframe into train / val / test partitions.

    Picks the regime from ``min(unique subjects per group)``:
      * ``≥ 5``  → 60/20/20
      * ``2..4`` → 80/20 (no test)
      * ``1``    → all train, ``warnings`` populated

    Subjects are atomic — every frame from a given ``(group, subject_id)``
    pair lands in the same partition. Subjects are deduplicated on the
    composite ``(group, subject_id)`` key to avoid basename collisions
    across groups (see contract #2 in module docstring).

    Args:
        detections_df: Frame-level dataframe with at minimum ``group``
            and ``subject_id`` columns. Other columns pass through
            untouched.
        seed: Reproducibility seed. Defaults to
            :data:`DEFAULT_SPLIT_SEED` (42). Used for a *local* numpy
            generator only; global RNG state is never modified.

    Returns:
        Tuple[pandas.DataFrame, SplitResult]: A copy of
        ``detections_df`` with a ``partition`` column added, and a
        :class:`SplitResult` describing the decision.

    Raises:
        ValueError: If the dataframe is empty or missing a required
            column. Held-out evaluation can't proceed without these.
    """
    # Late imports — keeps module-load cheap and matches the rest of the
    # toolkit's lazy-import pattern.
    import numpy as np
    import pandas as pd

    if detections_df is None or len(detections_df) == 0:
        raise ValueError("auto_split: detections_df is empty; nothing to split.")
    missing = [c for c in _REQUIRED_COLUMNS if c not in detections_df.columns]
    if missing:
        raise ValueError(
            f"auto_split: detections_df is missing required column(s) {missing}. "
            "These should be populated by `_collect_group_sources` + "
            "`read_detections`. Check that the runner "
            "calls auto_split AFTER `_read_runtime_detections` and BEFORE "
            "any HMM training (see contract #1 in splits.py module docstring)."
        )

    # Local generator only. NEVER call np.random.seed(seed) here — that
    # would lock the global RNG and corrupt HMM init / UMAP / torch
    # downstream (see contract #3).
    rng = np.random.default_rng(seed)

    # Compute unique (group, subject_id) per group — composite key.
    pairs_per_group: dict[str, list[str]] = {}
    for group_name, sub_df in detections_df.groupby("group"):
        unique_subjects = sorted({str(s) for s in sub_df["subject_id"].dropna().unique()})
        pairs_per_group[str(group_name)] = unique_subjects

    n_subjects_per_group = {g: len(subs) for g, subs in pairs_per_group.items()}
    if not n_subjects_per_group:
        raise ValueError("auto_split: no groups found in detections_df['group'].")
    min_subjects = min(n_subjects_per_group.values())

    # Pick the limiting group (smallest cohort). When ties, just take the
    # first sorted name — reproducible.
    limiting_candidates = sorted(g for g, n in n_subjects_per_group.items() if n == min_subjects)
    limiting_group = limiting_candidates[0] if len(set(n_subjects_per_group.values())) > 1 else ""

    regime = _decide_regime(min_subjects)

    # Per-group split.
    subjects_per_partition_per_group: dict[str, dict[str, list[str]]] = {}
    for group_name, subjects in pairs_per_group.items():
        subjects_per_partition_per_group[group_name] = _split_subjects_for_regime(
            subjects, regime=regime, rng=rng
        )

    # Build the per-(group,subject) → partition lookup, then emit the
    # partition column.
    lookup: dict[tuple[str, str], str] = {}
    for group_name, parts in subjects_per_partition_per_group.items():
        for partition_name, subj_list in parts.items():
            for subj in subj_list:
                lookup[(group_name, subj)] = partition_name

    out_df = detections_df.copy()
    # Default to "train" so any subject that snuck in without being
    # caught (shouldn't happen given the dedup, but defensive) doesn't
    # break downstream consumers.
    out_df["partition"] = [
        lookup.get((str(g), str(s)), "train")
        for g, s in zip(out_df["group"], out_df["subject_id"])
    ]

    n_subjects_per_partition_per_group = {
        g: {p: len(s) for p, s in parts.items()}
        for g, parts in subjects_per_partition_per_group.items()
    }

    warnings: list[str] = []
    if regime == REGIME_SINGLE:
        warnings.append(
            "Need ≥2 subjects per group for held-out validation. "
            "Falling back to full-data training; defensibility metrics "
            "will be hidden."
        )
    elif regime == REGIME_TWO_TO_FOUR:
        warnings.append(
            "2–4 subjects per group: held-out test set is empty. "
            "Train/val only. Generalization metrics computed on the val "
            "partition; treat as exploratory."
        )

    # When test partitions exist but have ≤1 subject per group, surface a
    # noise warning so the report doesn't oversell tiny-N metrics.
    for group_name, counts in n_subjects_per_partition_per_group.items():
        if regime == REGIME_FIVE_PLUS and counts.get("test", 0) <= 1:
            warnings.append(
                f"Group {group_name!r} test partition has "
                f"{counts.get('test', 0)} subject(s). Held-out metrics on "
                "this group will be noisy."
            )

    result = SplitResult(
        regime=regime,
        seed=seed,
        subjects_per_partition_per_group=subjects_per_partition_per_group,
        n_subjects_per_partition_per_group=n_subjects_per_partition_per_group,
        n_subjects_per_group=n_subjects_per_group,
        limiting_group=limiting_group,
        min_subjects_per_group=min_subjects,
        warnings=warnings,
    )
    return out_df, result


__all__ = [
    "auto_split",
    "SplitResult",
    "DEFAULT_SPLIT_SEED",
    "REGIME_FIVE_PLUS",
    "REGIME_TWO_TO_FOUR",
    "REGIME_SINGLE",
]
