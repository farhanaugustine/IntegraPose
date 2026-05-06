"""Tests for ADP-4 Commit E — cluster stability audit.

The stability code calls ``cluster_per_class`` internally, which calls
UMAP + HDBSCAN — heavy deps. Like the per_class tests, we split into:

  1. **Pure-logic tests** that monkeypatch ``cluster_per_class`` to
     return canned label sequences. These verify the orchestration:
     N seed runs, ARI matrix shape, mean/min computation,
     skip-propagation, the dataclass shape that the report panel
     consumes.

  2. **Pure ARI-helper tests** for ``_pairwise_ari_matrix`` and
     ``_summarize_off_diagonal`` — these only need sklearn, which is a
     light enough dep that they should run almost everywhere.

End-to-end stability with real UMAP/HDBSCAN is left to manual testing
in the user's env (sandbox doesn't have those reliably).
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _have_pandas() -> bool:
    try:
        import pandas  # noqa: F401
        import numpy  # noqa: F401
        return True
    except Exception:
        return False


def _have_sklearn() -> bool:
    try:
        from sklearn.metrics import adjusted_rand_score  # noqa: F401
        return True
    except Exception:
        return False


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        if not _have_pandas():
            self.skipTest("pandas / numpy unavailable")
        from integra_pose.hmm_vae_toolkit.stability import (
            compute_class_stability,
            ClassStability,
            StabilityReport,
            STABILITY_THRESHOLD_STABLE,
            _pairwise_ari_matrix,
            _summarize_off_diagonal,
        )
        self.compute_class_stability = compute_class_stability
        self.ClassStability = ClassStability
        self.StabilityReport = StabilityReport
        self.STABILITY_THRESHOLD_STABLE = STABILITY_THRESHOLD_STABLE
        self._pairwise_ari_matrix = _pairwise_ari_matrix
        self._summarize_off_diagonal = _summarize_off_diagonal


class TestARIHelper(_Base):
    def setUp(self) -> None:
        super().setUp()
        if not _have_sklearn():
            self.skipTest("scikit-learn unavailable")

    def test_identical_labels_yield_one(self) -> None:
        labels = [["a", "a", "b", "b"], ["a", "a", "b", "b"]]
        m = self._pairwise_ari_matrix(labels)
        self.assertEqual(m[0][1], 1.0)
        self.assertEqual(m[1][0], 1.0)
        self.assertEqual(m[0][0], 1.0)

    def test_label_permutation_yields_one(self) -> None:
        # ARI is permutation-invariant. Same partition, different
        # cluster names → ARI = 1.
        labels = [["x", "x", "y", "y"], ["foo", "foo", "bar", "bar"]]
        m = self._pairwise_ari_matrix(labels)
        self.assertEqual(m[0][1], 1.0)

    def test_completely_random_yields_near_zero(self) -> None:
        # Two unrelated label sequences should have ARI near 0.
        # (Actually small positive or negative due to randomness.)
        import random
        rng = random.Random(7)

        labels_a = [rng.choice(["x", "y", "z"]) for _ in range(60)]
        labels_b = [rng.choice(["a", "b", "c"]) for _ in range(60)]
        m = self._pairwise_ari_matrix([labels_a, labels_b])
        # Be generous: just assert NOT highly correlated.
        self.assertLess(abs(m[0][1]), 0.5)

    def test_summarize_off_diagonal(self) -> None:
        # 3x3 symmetric matrix, off-diagonal = [0.8, 0.6, 0.7]
        matrix = [
            [1.0, 0.8, 0.6],
            [0.8, 1.0, 0.7],
            [0.6, 0.7, 1.0],
        ]
        mean, mn = self._summarize_off_diagonal(matrix)
        self.assertAlmostEqual(mean, (0.8 + 0.6 + 0.7) / 3.0)
        self.assertAlmostEqual(mn, 0.6)

    def test_summarize_off_diagonal_with_nan(self) -> None:
        matrix = [
            [1.0, 0.8, float("nan")],
            [0.8, 1.0, 0.7],
            [float("nan"), 0.7, 1.0],
        ]
        mean, mn = self._summarize_off_diagonal(matrix)
        self.assertAlmostEqual(mean, (0.8 + 0.7) / 2.0)
        self.assertAlmostEqual(mn, 0.7)

    def test_summarize_all_nan_returns_nan(self) -> None:
        import math
        matrix = [
            [1.0, float("nan")],
            [float("nan"), 1.0],
        ]
        mean, mn = self._summarize_off_diagonal(matrix)
        self.assertTrue(math.isnan(mean))
        self.assertTrue(math.isnan(mn))


class TestStabilityOrchestration(_Base):
    """Mock cluster_per_class to make these tests independent of UMAP/HDBSCAN."""

    def setUp(self) -> None:
        super().setUp()
        if not _have_sklearn():
            self.skipTest("scikit-learn required for ARI computation")

    def _build_mock_cluster_per_class(self, label_pattern_per_seed: dict):
        """Build a fake cluster_per_class that returns labels keyed by seed.

        ``label_pattern_per_seed`` is a dict ``{seed: {class_id:
        [labels...]}}``. The mock looks up the seed in cluster_kwargs
        and returns labels accordingly.
        """
        import pandas as pd
        from integra_pose.hmm_vae_toolkit.per_class_clustering import (
            PerClassClusterResult,
            MultiClassClusterResult,
        )

        def _mock(detections_df, *, class_names=None, seed=42, **_kw):
            df = detections_df.copy()
            df["cluster_label"] = "-1"

            classes_meta = []
            seed_labels = label_pattern_per_seed.get(seed, {})

            for class_id in sorted(seed_labels.keys()):
                labels = seed_labels[class_id]
                class_indices = df.index[df["class_id"] == class_id].tolist()
                # Tile labels to match number of rows (input may have
                # more rows than the pattern).
                tiled = (labels * ((len(class_indices) // len(labels)) + 1))[
                    : len(class_indices)
                ]
                df.loc[class_indices, "cluster_label"] = [str(t) for t in tiled]
                unique = sorted({int(t) for t in tiled if int(t) != -1})
                classes_meta.append(
                    PerClassClusterResult(
                        class_id=class_id,
                        class_name=str((class_names or {}).get(class_id, "")),
                        n_frames=len(class_indices),
                        n_clusters=len(unique),
                        n_noise_frames=int(sum(1 for t in tiled if int(t) == -1)),
                        sub_cluster_labels=unique,
                    )
                )
            multi = MultiClassClusterResult(
                classes=classes_meta, seed=seed, label_column="cluster_label"
            )
            return df, multi

        return _mock

    def test_perfectly_stable_class_yields_ari_one(self) -> None:
        """Same labels across all seeds → mean_ari = 1.0, is_stable=True."""
        import pandas as pd

        df = pd.DataFrame([
            {"class_id": 0, "subject_id": "s", "frame": i, "feature_vector": [1.0]}
            for i in range(40)
        ])

        # Same pattern for every seed.
        same_pattern = {0: [0, 0, 1, 1]}
        patterns = {42: same_pattern, 43: same_pattern, 44: same_pattern}

        with mock.patch(
            "integra_pose.hmm_vae_toolkit.per_class_clustering.cluster_per_class",
            self._build_mock_cluster_per_class(patterns),
        ):
            report = self.compute_class_stability(df, n_seeds=3, base_seed=42)

        self.assertEqual(report.n_seeds, 3)
        self.assertEqual(report.base_seed, 42)
        self.assertEqual(len(report.per_class), 1)
        c0 = report.per_class[0]
        self.assertFalse(c0.skipped)
        self.assertEqual(c0.mean_ari, 1.0)
        self.assertEqual(c0.min_ari, 1.0)
        self.assertTrue(c0.is_stable)
        self.assertEqual(report.n_stable, 1)

    def test_seeds_used_field_recorded(self) -> None:
        import pandas as pd

        df = pd.DataFrame([
            {"class_id": 0, "subject_id": "s", "frame": i, "feature_vector": [1.0]}
            for i in range(40)
        ])
        patterns = {
            42: {0: [0, 1]},
            43: {0: [0, 1]},
            44: {0: [0, 1]},
        }
        with mock.patch(
            "integra_pose.hmm_vae_toolkit.per_class_clustering.cluster_per_class",
            self._build_mock_cluster_per_class(patterns),
        ):
            report = self.compute_class_stability(df, n_seeds=3, base_seed=42)

        c0 = report.per_class[0]
        self.assertEqual(c0.seeds_used, [42, 43, 44])

    def test_skipped_class_propagates(self) -> None:
        """If cluster_per_class skipped a class in seed 0 (the primary
        run), stability should propagate the skip without trying to
        compute ARI for a class with no labels."""
        import pandas as pd
        from integra_pose.hmm_vae_toolkit.per_class_clustering import (
            PerClassClusterResult,
            MultiClassClusterResult,
        )

        df = pd.DataFrame([
            {"class_id": 0, "subject_id": "s", "frame": i, "feature_vector": [1.0]}
            for i in range(5)  # below default min_class_size
        ])

        def _mock_skipped(detections_df, *, class_names=None, seed=42, **_kw):
            d = detections_df.copy()
            d["cluster_label"] = "-1"
            classes_meta = [
                PerClassClusterResult(
                    class_id=0,
                    class_name="",
                    n_frames=5,
                    n_clusters=0,
                    n_noise_frames=5,
                    skipped=True,
                    skip_reason="Only 5 frame(s); need ≥30 to cluster.",
                )
            ]
            return d, MultiClassClusterResult(
                classes=classes_meta, seed=seed, label_column="cluster_label"
            )

        with mock.patch(
            "integra_pose.hmm_vae_toolkit.per_class_clustering.cluster_per_class",
            _mock_skipped,
        ):
            report = self.compute_class_stability(df, n_seeds=3)

        self.assertEqual(len(report.per_class), 1)
        c0 = report.per_class[0]
        self.assertTrue(c0.skipped)
        self.assertIn("≥30", c0.skip_reason)
        self.assertEqual(report.n_total, 0)
        self.assertEqual(report.n_stable, 0)

    def test_ari_matrix_shape_n_by_n(self) -> None:
        import pandas as pd

        df = pd.DataFrame([
            {"class_id": 0, "subject_id": "s", "frame": i, "feature_vector": [1.0]}
            for i in range(40)
        ])
        patterns = {
            42: {0: [0, 1]},
            43: {0: [0, 1]},
            44: {0: [0, 1]},
            45: {0: [0, 1]},
        }
        with mock.patch(
            "integra_pose.hmm_vae_toolkit.per_class_clustering.cluster_per_class",
            self._build_mock_cluster_per_class(patterns),
        ):
            report = self.compute_class_stability(df, n_seeds=4)
        c0 = report.per_class[0]
        self.assertEqual(len(c0.ari_matrix), 4)
        for row in c0.ari_matrix:
            self.assertEqual(len(row), 4)
        # Diagonal is 1.0
        for i in range(4):
            self.assertEqual(c0.ari_matrix[i][i], 1.0)

    def test_unstable_class_below_threshold(self) -> None:
        """Different label patterns per seed → low ARI → not stable."""
        import pandas as pd

        df = pd.DataFrame([
            {"class_id": 0, "subject_id": "s", "frame": i, "feature_vector": [1.0]}
            for i in range(60)
        ])
        # Each seed produces a totally different partition of the same
        # frames. ARI between unrelated partitions is near 0.
        patterns = {
            42: {0: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]},  # halves
            43: {0: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]},  # alternating
            44: {0: [0, 0, 1, 1, 2, 2, 0, 0, 1, 1]},  # different splits
        }
        with mock.patch(
            "integra_pose.hmm_vae_toolkit.per_class_clustering.cluster_per_class",
            self._build_mock_cluster_per_class(patterns),
        ):
            report = self.compute_class_stability(df, n_seeds=3, base_seed=42)

        c0 = report.per_class[0]
        self.assertLess(c0.mean_ari, self.STABILITY_THRESHOLD_STABLE)
        self.assertFalse(c0.is_stable)


class TestStabilityEdgeCases(_Base):
    def test_n_seeds_one_raises(self) -> None:
        import pandas as pd
        df = pd.DataFrame([{"class_id": 0, "feature_vector": [1.0]}])
        with self.assertRaises(ValueError) as ctx:
            self.compute_class_stability(df, n_seeds=1)
        self.assertIn("n_seeds", str(ctx.exception))

    def test_overall_summary_includes_threshold(self) -> None:
        # Build a tiny report manually and verify the summary string
        # has the threshold the user can quote in their paper.
        import math

        c1 = self.ClassStability(
            class_id=0, class_name="walking", n_seeds=3, mean_ari=0.9, min_ari=0.85,
        )
        c2 = self.ClassStability(
            class_id=1, class_name="rearing", n_seeds=3, mean_ari=0.2, min_ari=0.1,
        )
        report = self.StabilityReport(per_class=[c1, c2], n_seeds=3, base_seed=42)
        self.assertIn("1 of 2", report.overall_summary)
        self.assertIn(str(self.STABILITY_THRESHOLD_STABLE), report.overall_summary)


if __name__ == "__main__":
    unittest.main()
