"""Tests for ADP-4 Commit F — sub-cluster signal scoring.

Pure-logic tests. The function takes already-computed bouts and result
objects (no clustering, no UMAP), so these run without any ML deps.

Coverage:
  - Each signal is computed correctly in isolation (size, coverage,
    duration, stability) on hand-crafted inputs.
  - Score combines correctly with and without stability (renormalization).
  - Sort order is descending by score.
  - Noise bouts are excluded.
  - Bins map to verdicts at the documented cut-points.
  - Notes describe the weak spots.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from integra_pose.hmm_vae_toolkit.signal_score import (
    compute_signal_scores,
    DEFAULT_WEIGHTS,
    VERDICT_LIKELY_REAL,
    VERDICT_LIKELY_NOISE,
    VERDICT_REVIEW,
    _verdict_for,
)
from integra_pose.hmm_vae_toolkit.per_class_clustering import (
    PerClassClusterResult,
    MultiClassClusterResult,
)


def _bout(class_id, sub_id, subject_id="subj_a", duration=10, count=10):
    return {
        "class_id": class_id,
        "state": sub_id,
        "subject_id": subject_id,
        "duration_frames": duration,
        "detection_count": count,
    }


def _multi_with(class_id, class_name="walking", n_clusters=2):
    return MultiClassClusterResult(
        classes=[
            PerClassClusterResult(
                class_id=class_id, class_name=class_name,
                n_frames=200, n_clusters=n_clusters, n_noise_frames=10,
                sub_cluster_labels=list(range(n_clusters)),
            )
        ],
        seed=42,
    )


class _FakeStability:
    """Lightweight stand-in for StabilityReport.per_class entries."""

    def __init__(self, class_id, mean_ari, skipped=False):
        self.class_id = class_id
        self.mean_ari = mean_ari
        self.skipped = skipped


class _FakeReport:
    def __init__(self, per_class):
        self.per_class = per_class


class TestVerdictBins(unittest.TestCase):
    def test_high_score_likely_real(self) -> None:
        self.assertEqual(_verdict_for(0.9), VERDICT_LIKELY_REAL)
        self.assertEqual(_verdict_for(0.65), VERDICT_LIKELY_REAL)

    def test_mid_score_review(self) -> None:
        self.assertEqual(_verdict_for(0.5), VERDICT_REVIEW)
        self.assertEqual(_verdict_for(0.4), VERDICT_REVIEW)

    def test_low_score_noise(self) -> None:
        self.assertEqual(_verdict_for(0.2), VERDICT_LIKELY_NOISE)
        self.assertEqual(_verdict_for(0.0), VERDICT_LIKELY_NOISE)


class TestSignalIsolation(unittest.TestCase):
    """Each signal computed alone — set the weights so only one matters."""

    def test_size_signal_saturates_at_ref(self) -> None:
        # 200 frames > size_ref=100 → size signal = 1.0
        bouts = [_bout(0, 1, count=100), _bout(0, 1, count=100)]
        weights = {"size": 1.0, "coverage": 0.0, "stability": 0.0, "duration": 0.0}
        report = compute_signal_scores(
            bouts, _multi_with(0), weights=weights, size_ref=100,
        )
        self.assertEqual(len(report.candidates), 1)
        c = report.candidates[0]
        self.assertEqual(c.size_signal, 1.0)
        self.assertEqual(c.score, 1.0)

    def test_coverage_signal_one_subject_zero(self) -> None:
        bouts = [_bout(0, 1, subject_id="only", count=50)]
        weights = {"size": 0.0, "coverage": 1.0, "stability": 0.0, "duration": 0.0}
        report = compute_signal_scores(
            bouts, _multi_with(0), weights=weights, coverage_ref=3,
        )
        self.assertEqual(report.candidates[0].coverage_signal, 0.0)

    def test_coverage_signal_three_subjects_one(self) -> None:
        bouts = [
            _bout(0, 1, subject_id="a"),
            _bout(0, 1, subject_id="b"),
            _bout(0, 1, subject_id="c"),
        ]
        weights = {"size": 0.0, "coverage": 1.0, "stability": 0.0, "duration": 0.0}
        report = compute_signal_scores(
            bouts, _multi_with(0), weights=weights, coverage_ref=3,
        )
        self.assertEqual(report.candidates[0].coverage_signal, 1.0)

    def test_duration_signal_saturates_at_ref(self) -> None:
        bouts = [_bout(0, 1, duration=60)]  # well above duration_ref=30
        weights = {"size": 0.0, "coverage": 0.0, "stability": 0.0, "duration": 1.0}
        report = compute_signal_scores(
            bouts, _multi_with(0), weights=weights, duration_ref=30,
        )
        self.assertEqual(report.candidates[0].duration_signal, 1.0)

    def test_stability_signal_clamps_negative(self) -> None:
        bouts = [_bout(0, 1)]
        weights = {"size": 0.0, "coverage": 0.0, "stability": 1.0, "duration": 0.0}
        stab = _FakeReport([_FakeStability(class_id=0, mean_ari=-0.2)])
        report = compute_signal_scores(
            bouts, _multi_with(0), stability_report=stab, weights=weights,
        )
        self.assertEqual(report.candidates[0].stability_signal, 0.0)


class TestRenormalization(unittest.TestCase):
    """When stability isn't computed, the remaining weights renormalize."""

    def test_no_stability_renormalizes(self) -> None:
        # 3 strong subjects + long bouts. Without stability, score should
        # come from the OTHER three signals normalized to sum to 1.
        bouts = [
            _bout(0, 1, subject_id="a", duration=30, count=30),
            _bout(0, 1, subject_id="b", duration=30, count=30),
            _bout(0, 1, subject_id="c", duration=30, count=30),
        ]
        report = compute_signal_scores(bouts, _multi_with(0))
        # weights_used should sum to ~1 and not contain "stability"
        self.assertNotIn("stability", report.weights_used)
        self.assertAlmostEqual(sum(report.weights_used.values()), 1.0, places=4)
        # The candidate scored on three signals should be high.
        c = report.candidates[0]
        self.assertGreater(c.score, 0.7)
        self.assertIsNone(c.stability_signal)

    def test_with_stability_includes_term(self) -> None:
        bouts = [_bout(0, 1)]
        stab = _FakeReport([_FakeStability(class_id=0, mean_ari=0.85)])
        report = compute_signal_scores(bouts, _multi_with(0), stability_report=stab)
        self.assertIn("stability", report.weights_used)
        self.assertAlmostEqual(sum(report.weights_used.values()), 1.0, places=4)


class TestNoiseExclusion(unittest.TestCase):
    def test_state_minus_one_dropped(self) -> None:
        bouts = [
            _bout(0, -1, count=200),  # noise — should be dropped
            _bout(0, 1, count=50),
        ]
        report = compute_signal_scores(bouts, _multi_with(0))
        sub_ids = {c.sub_id for c in report.candidates}
        self.assertNotIn(-1, sub_ids)
        self.assertEqual(len(report.candidates), 1)


class TestSortOrder(unittest.TestCase):
    def test_candidates_sorted_descending(self) -> None:
        # Two sub-clusters, one strong one weak.
        bouts = [
            # cluster 1 — many frames, 3 subjects, long bouts → strong
            _bout(0, 1, subject_id="a", duration=30, count=50),
            _bout(0, 1, subject_id="b", duration=30, count=50),
            _bout(0, 1, subject_id="c", duration=30, count=50),
            # cluster 2 — single subject, single short bout → weak
            _bout(0, 2, subject_id="a", duration=2, count=2),
        ]
        report = compute_signal_scores(bouts, _multi_with(0, n_clusters=2))
        scores = [c.score for c in report.candidates]
        # Strictly decreasing.
        self.assertEqual(scores, sorted(scores, reverse=True))
        # The strong cluster should rank first.
        self.assertEqual(report.candidates[0].sub_id, 1)


class TestNotes(unittest.TestCase):
    def test_single_subject_note(self) -> None:
        bouts = [_bout(0, 1, subject_id="only", duration=20)]
        report = compute_signal_scores(bouts, _multi_with(0))
        notes = " ".join(report.candidates[0].notes)
        self.assertIn("subject", notes)

    def test_short_mean_bout_note(self) -> None:
        bouts = [
            _bout(0, 1, subject_id="a", duration=2),
            _bout(0, 1, subject_id="b", duration=3),
        ]
        report = compute_signal_scores(bouts, _multi_with(0))
        notes = " ".join(report.candidates[0].notes)
        self.assertIn("transient", notes)

    def test_low_stability_note(self) -> None:
        bouts = [_bout(0, 1, subject_id="a", duration=20)]
        stab = _FakeReport([_FakeStability(class_id=0, mean_ari=0.3)])
        report = compute_signal_scores(bouts, _multi_with(0), stability_report=stab)
        notes = " ".join(report.candidates[0].notes)
        self.assertIn("seed-sensitive", notes)


class TestEmptyInput(unittest.TestCase):
    def test_no_bouts_returns_empty(self) -> None:
        report = compute_signal_scores([], _multi_with(0))
        self.assertEqual(len(report.candidates), 0)

    def test_only_noise_bouts_returns_empty(self) -> None:
        bouts = [_bout(0, -1, count=200)]
        report = compute_signal_scores(bouts, _multi_with(0))
        self.assertEqual(len(report.candidates), 0)


class TestVerdictDistribution(unittest.TestCase):
    def test_counts_match_candidates(self) -> None:
        # Mix one strong, one borderline, one weak.
        bouts = [
            _bout(0, 1, subject_id="a", duration=30, count=50),
            _bout(0, 1, subject_id="b", duration=30, count=50),
            _bout(0, 1, subject_id="c", duration=30, count=50),
            _bout(0, 2, subject_id="a", duration=15, count=30),
            _bout(0, 2, subject_id="b", duration=15, count=30),
            _bout(0, 3, subject_id="a", duration=2, count=2),
        ]
        report = compute_signal_scores(bouts, _multi_with(0, n_clusters=3))
        total_buckets = report.n_likely_real + report.n_review + report.n_likely_noise
        self.assertEqual(total_buckets, len(report.candidates))


if __name__ == "__main__":
    unittest.main()
