"""ADP-4 Commit F — Sub-cluster signal scoring.

Combines several quality signals into a single 0–1 "candidate strength"
score per sub-cluster, so the novel-behavior review panel can surface
the strongest candidates first. The user reviews ranked candidates,
marks each as Real / Noise / Skip, and the labeled subset becomes
training data for the next iteration of YOLO or BehaviorScope.

The honest framing (per user, 2026-04-26):

  > "How to make sure that only clusters with true behaviors are
  > surfaced? How to pick thresholds, etc."

There is no single threshold that's correct. Algorithmic filters
**rank**; the human **decides**. The score is a tool for prioritising
review attention, not for autonomous gating. If a cluster scores 0.3
that does not mean it's noise — it means *spend less time on it
than the 0.8 candidates*.

Signals and weights (defaults, all overridable):

| Signal             | What it tells you                       | Default weight |
|--------------------|------------------------------------------|----------------|
| Cluster size       | More frames → less likely transient noise | 0.25           |
| Subject coverage   | Real behaviors appear in ≥2 animals       | 0.30           |
| Stability ARI      | Cluster survives seed shuffles            | 0.30           |
| Mean bout duration | Longer bouts → more confident segments    | 0.15           |

When stability hasn't been computed (user didn't tick the audit
checkbox), the stability term is excluded from the weighted sum and
the remaining weights are renormalized. That way a "fast run, no
stability" still gets a sensible score — just with one fewer pillar.

This module is pure logic; the review-panel UI lives in main.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# Signal weights. Documented in the module docstring; keep them as
# constants so a power user can override via monkey-patch in scripts.
DEFAULT_WEIGHTS = {
    "size": 0.25,
    "coverage": 0.30,
    "stability": 0.30,
    "duration": 0.15,
}

# Score reference points — define the "saturation level" of each
# signal. Each signal is mapped to ``[0, 1]`` via min(value/ref, 1).
# Once the user provides feedback we can replace these heuristics
# with empirical calibration; for v1 the values come from typical
# pose-data datasets in the literature.
DEFAULT_SIZE_REF = 100.0          # frames — beyond this size signal saturates
DEFAULT_COVERAGE_REF = 3.0        # subjects — 3 distinct subjects → full credit
DEFAULT_DURATION_REF = 30.0       # frames — ~1s at 30 FPS for full credit
DEFAULT_STABILITY_FLOOR = 0.0     # negative ARIs clamp to 0

# Verdict bins — purely advisory labels rendered next to the score.
# These do NOT gate any decision; users see the underlying numbers.
VERDICT_LIKELY_REAL = "likely_real"
VERDICT_REVIEW = "review"
VERDICT_LIKELY_NOISE = "likely_noise"


@dataclass
class SubClusterCandidate:
    """One ranked sub-cluster — what the review panel renders per row.

    Fields are flat strings/floats so JSON-serialization is trivial and
    the panel's table view doesn't need a nested unpacking step.
    """

    class_id: int
    class_name: str
    sub_id: int  # within-class HDBSCAN cluster id
    namespaced_label: str  # e.g., "0:1" — key into clip exporter / state_names
    n_frames: int
    n_subjects: int
    n_bouts: int
    mean_bout_frames: float
    stability_ari: Optional[float]  # None if audit didn't run
    size_signal: float
    coverage_signal: float
    duration_signal: float
    stability_signal: Optional[float]
    score: float  # weighted combination, [0,1]
    verdict: str
    notes: list = field(default_factory=list)


@dataclass
class SignalScoreReport:
    """Top-level scoring outcome — sorted candidates + bookkeeping."""

    candidates: list  # list[SubClusterCandidate], sorted by score desc
    weights_used: dict
    n_likely_real: int
    n_review: int
    n_likely_noise: int

    @property
    def total(self) -> int:
        return len(self.candidates)


def _verdict_for(score: float) -> str:
    """Map a numeric score to an advisory verdict label.

    The cut-points (0.65, 0.4) are heuristic. They're documented and
    stable so users can include them in methods sections; they aren't
    used to gate anything automatically.
    """
    if score >= 0.65:
        return VERDICT_LIKELY_REAL
    if score >= 0.4:
        return VERDICT_REVIEW
    return VERDICT_LIKELY_NOISE


def compute_signal_scores(
    bouts: list,
    multi_result,
    *,
    stability_report=None,
    weights: Optional[dict] = None,
    size_ref: float = DEFAULT_SIZE_REF,
    coverage_ref: float = DEFAULT_COVERAGE_REF,
    duration_ref: float = DEFAULT_DURATION_REF,
) -> SignalScoreReport:
    """Score every (class_id, sub_id) sub-cluster against its quality signals.

    Args:
        bouts: List of bout dicts produced by
            ``aggregate_states_into_bouts(state_column='cluster_label')``.
            Bouts in HDBSCAN noise (state ``"-1"``) are excluded — noise
            is not a candidate sub-cluster, by definition.
        multi_result: ``MultiClassClusterResult`` from ``cluster_per_class``.
            Used to tag class names and to surface skipped classes.
        stability_report: Optional ``StabilityReport``. When provided,
            mean ARI per class becomes the stability signal. When
            absent, the stability term is dropped from the weighted
            average and the remaining three terms are renormalized.
        weights: Optional override of the default weight dict
            (``{"size", "coverage", "stability", "duration"}``).
        size_ref / coverage_ref / duration_ref: Saturation reference
            points for each signal — see module docstring.

    Returns:
        :class:`SignalScoreReport` with candidates sorted by score
        descending. Empty list if no non-noise bouts existed.
    """
    weights = dict(weights or DEFAULT_WEIGHTS)
    has_stability = stability_report is not None

    # If no stability, drop the term and renormalize. Otherwise normalize
    # to defend against weights-summing-to-not-1 if a user passed custom
    # ones.
    active_weights = dict(weights)
    if not has_stability:
        active_weights.pop("stability", None)
    weight_sum = sum(active_weights.values())
    if weight_sum > 0:
        active_weights = {k: v / weight_sum for k, v in active_weights.items()}

    # Build a class_id → name lookup from the multi_result for nice labels.
    class_name_lookup: dict[int, str] = {}
    for c in (multi_result.classes if multi_result else []):
        class_name_lookup[int(c.class_id)] = c.class_name or f"class_{c.class_id}"

    # Build a class_id → mean_ari lookup for the stability signal.
    stab_lookup: dict[int, Optional[float]] = {}
    if has_stability:
        for c in stability_report.per_class:
            if c.skipped:
                continue
            try:
                v = float(c.mean_ari)
                if v != v:  # NaN check
                    v = None
            except Exception:
                v = None
            stab_lookup[int(c.class_id)] = v

    # Group bouts by (class_id, sub_id).
    grouped: dict[tuple[int, int], list] = {}
    for b in (bouts or []):
        try:
            cid = int(b.get("class_id", -1))
        except Exception:
            continue
        state = b.get("state")
        try:
            sid = int(state)
        except Exception:
            continue
        if sid == -1:  # HDBSCAN noise — not a candidate
            continue
        grouped.setdefault((cid, sid), []).append(b)

    candidates: list[SubClusterCandidate] = []
    for (class_id, sub_id), cluster_bouts in grouped.items():
        n_bouts = len(cluster_bouts)
        n_frames = sum(int(b.get("detection_count", 0)) for b in cluster_bouts)
        subjects = {str(b.get("subject_id") or "") for b in cluster_bouts}
        subjects.discard("")
        n_subjects = len(subjects)
        durations = [int(b.get("duration_frames", 0)) for b in cluster_bouts]
        mean_dur = float(sum(durations) / len(durations)) if durations else 0.0

        size_signal = min(1.0, n_frames / max(1.0, size_ref))
        # Coverage: 1 subject → 0.0, 2 → 0.5, 3+ → 1.0 with the default ref=3.
        coverage_signal = min(1.0, max(0.0, (n_subjects - 1) / max(1.0, coverage_ref - 1)))
        duration_signal = min(1.0, mean_dur / max(1.0, duration_ref))

        stab_value = stab_lookup.get(class_id)
        stab_signal = (
            max(DEFAULT_STABILITY_FLOOR, float(stab_value))
            if (has_stability and stab_value is not None)
            else None
        )

        # Weighted sum — terms only added when the corresponding signal
        # is in active_weights.
        score = 0.0
        if "size" in active_weights:
            score += active_weights["size"] * size_signal
        if "coverage" in active_weights:
            score += active_weights["coverage"] * coverage_signal
        if "duration" in active_weights:
            score += active_weights["duration"] * duration_signal
        if "stability" in active_weights and stab_signal is not None:
            score += active_weights["stability"] * stab_signal

        # Build short qualitative notes for the review panel — these
        # explain WHY the score is what it is.
        notes: list[str] = []
        if n_subjects < 2:
            notes.append("Only 1 subject — generalisation unproven.")
        if mean_dur < 5:
            notes.append("Mean bout < 5 frames — possibly transient.")
        if has_stability and stab_signal is not None and stab_signal < 0.5:
            notes.append("Low stability ARI — seed-sensitive.")
        if n_bouts < 3:
            notes.append("Fewer than 3 bouts — small evidence base.")

        candidates.append(
            SubClusterCandidate(
                class_id=class_id,
                class_name=class_name_lookup.get(class_id, f"class_{class_id}"),
                sub_id=sub_id,
                namespaced_label=f"{class_id}:{sub_id}",
                n_frames=n_frames,
                n_subjects=n_subjects,
                n_bouts=n_bouts,
                mean_bout_frames=mean_dur,
                stability_ari=stab_value,
                size_signal=float(size_signal),
                coverage_signal=float(coverage_signal),
                duration_signal=float(duration_signal),
                stability_signal=(
                    float(stab_signal) if stab_signal is not None else None
                ),
                score=float(score),
                verdict=_verdict_for(score),
                notes=notes,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    n_likely_real = sum(1 for c in candidates if c.verdict == VERDICT_LIKELY_REAL)
    n_review = sum(1 for c in candidates if c.verdict == VERDICT_REVIEW)
    n_likely_noise = sum(1 for c in candidates if c.verdict == VERDICT_LIKELY_NOISE)

    return SignalScoreReport(
        candidates=candidates,
        weights_used=active_weights,
        n_likely_real=n_likely_real,
        n_review=n_review,
        n_likely_noise=n_likely_noise,
    )


__all__ = [
    "compute_signal_scores",
    "SubClusterCandidate",
    "SignalScoreReport",
    "DEFAULT_WEIGHTS",
    "DEFAULT_SIZE_REF",
    "DEFAULT_COVERAGE_REF",
    "DEFAULT_DURATION_REF",
    "VERDICT_LIKELY_REAL",
    "VERDICT_REVIEW",
    "VERDICT_LIKELY_NOISE",
]
