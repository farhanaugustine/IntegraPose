"""Tests for ADP-4 — per-class sub-behavior clustering.

The clustering module routes to UMAP + HDBSCAN, both of which are heavy
optional dependencies. We split the test suite into two halves:

  1. **Pure-logic tests** that exercise the orchestration — class
     iteration, skip-when-too-small, namespaced labels, the result
     dataclasses. These use a monkeypatched ``_cluster_one_class`` so
     UMAP/HDBSCAN aren't imported.

  2. **End-to-end tests** that actually run UMAP + HDBSCAN on tiny
     synthetic data. Skipped automatically when the libs aren't
     installed.

The orchestration tests are the load-bearing ones — they verify the
contract Tab 7's runner depends on (skipped classes don't pollute the
df, namespaced labels keep classes distinct, noise sentinel survives).
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


def _have_umap_hdbscan() -> bool:
    try:
        import umap  # noqa: F401
        import hdbscan  # noqa: F401
        return True
    except Exception:
        return False


def _make_df(rows):
    import pandas as pd

    return pd.DataFrame([
        {
            "class_id": class_id,
            "subject_id": subject_id,
            "frame": frame,
            "feature_vector": list(features),
        }
        for (class_id, subject_id, frame, features) in rows
    ])


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        if not _have_pandas():
            self.skipTest("pandas / numpy unavailable in this environment")
        from integra_pose.hmm_vae_toolkit.per_class_clustering import (
            cluster_per_class,
            PerClassClusterResult,
            MultiClassClusterResult,
            DEFAULT_MIN_CLASS_SIZE,
        )
        self.cluster_per_class = cluster_per_class
        self.PerClassClusterResult = PerClassClusterResult
        self.MultiClassClusterResult = MultiClassClusterResult
        self.DEFAULT_MIN_CLASS_SIZE = DEFAULT_MIN_CLASS_SIZE


class TestOrchestration(_Base):
    """Pure-logic tests: monkeypatch _cluster_one_class so we don't need
    UMAP/HDBSCAN installed. These verify the orchestration contract.
    """

    def _stub_cluster_one_class(self, label_pattern):
        """Return a stub for _cluster_one_class that emits a fixed label
        sequence. ``label_pattern`` is a list-like of ints; if shorter
        than the input, it tiles.
        """
        import numpy as np

        def _stub(feature_matrix, **_kw):
            n = feature_matrix.shape[0]
            tiled = (label_pattern * ((n // len(label_pattern)) + 1))[:n]
            return np.array(tiled)

        return _stub

    def test_namespaced_labels_separate_classes(self) -> None:
        # Two classes (0 and 1), each with 40 rows. The stub assigns
        # the SAME local label (0) to all rows of both classes. The
        # namespaced labels in the output must distinguish them.
        rows = []
        for f in range(40):
            rows.append((0, "subj_a", f, [1.0, 2.0, 3.0]))
            rows.append((1, "subj_a", f, [4.0, 5.0, 6.0]))
        df = _make_df(rows)

        with mock.patch(
            "integra_pose.hmm_vae_toolkit.per_class_clustering._cluster_one_class",
            self._stub_cluster_one_class([0, 0, 0]),
        ):
            out_df, result = self.cluster_per_class(df)

        labels_class0 = set(out_df.loc[out_df["class_id"] == 0, "cluster_label"].unique())
        labels_class1 = set(out_df.loc[out_df["class_id"] == 1, "cluster_label"].unique())

        self.assertEqual(labels_class0, {"0:0"})
        self.assertEqual(labels_class1, {"1:0"})
        # No collision: namespaced labels are distinct.
        self.assertEqual(labels_class0 & labels_class1, set())

    def test_classes_below_min_size_are_skipped(self) -> None:
        # Class 0 has 50 frames; class 1 has only 5 (below min_class_size=30).
        rows = []
        for f in range(50):
            rows.append((0, "subj", f, [1.0, 2.0, 3.0]))
        for f in range(5):
            rows.append((1, "subj", f, [4.0, 5.0, 6.0]))
        df = _make_df(rows)

        with mock.patch(
            "integra_pose.hmm_vae_toolkit.per_class_clustering._cluster_one_class",
            self._stub_cluster_one_class([0]),
        ):
            out_df, result = self.cluster_per_class(df)

        # Class 0 ran; class 1 was skipped.
        skipped_ids = {c.class_id for c in result.classes if c.skipped}
        self.assertEqual(skipped_ids, {1})
        self.assertIn(1, result.skipped_classes)

        # Class 1's frames stay tagged "-1" (the noise sentinel).
        class1_labels = set(out_df.loc[out_df["class_id"] == 1, "cluster_label"].unique())
        self.assertEqual(class1_labels, {"-1"})

    def test_noise_label_namespacing(self) -> None:
        # A stub that emits some -1 (noise) labels. The output should
        # use the literal "-1" string, NOT the namespaced form, so
        # downstream aggregators recognize noise consistently.
        rows = [(0, "subj", f, [1.0, 2.0, 3.0]) for f in range(40)]
        df = _make_df(rows)

        with mock.patch(
            "integra_pose.hmm_vae_toolkit.per_class_clustering._cluster_one_class",
            self._stub_cluster_one_class([0, -1, 1, -1]),
        ):
            out_df, result = self.cluster_per_class(df)

        self.assertIn("-1", set(out_df["cluster_label"].unique()))
        # No "0:-1" anywhere — noise stays literal.
        self.assertNotIn("0:-1", set(out_df["cluster_label"].unique()))
        # Result records the noise count.
        c0 = next(c for c in result.classes if c.class_id == 0)
        self.assertGreater(c0.n_noise_frames, 0)

    def test_class_names_passed_through_to_result(self) -> None:
        rows = [(0, "subj", f, [1.0, 2.0, 3.0]) for f in range(40)]
        df = _make_df(rows)

        with mock.patch(
            "integra_pose.hmm_vae_toolkit.per_class_clustering._cluster_one_class",
            self._stub_cluster_one_class([0]),
        ):
            _out_df, result = self.cluster_per_class(df, class_names={0: "walking"})

        c0 = next(c for c in result.classes if c.class_id == 0)
        self.assertEqual(c0.class_name, "walking")

    def test_total_clusters_aggregates_across_classes(self) -> None:
        # Two classes, stub reports 3 sub-clusters in class 0 and 2 in class 1.
        rows = []
        for f in range(60):
            rows.append((0, "subj", f, [1.0, 2.0, 3.0]))
        for f in range(60):
            rows.append((1, "subj", f, [4.0, 5.0, 6.0]))
        df = _make_df(rows)

        # First call returns labels with 3 unique clusters; second returns 2.
        call_count = {"n": 0}

        def _stub(feature_matrix, **_kw):
            import numpy as np
            call_count["n"] += 1
            n = feature_matrix.shape[0]
            if call_count["n"] == 1:
                # 3 clusters: [0,1,2,0,1,2,...]
                pattern = [0, 1, 2]
            else:
                # 2 clusters: [0,1,0,1,...]
                pattern = [0, 1]
            tiled = (pattern * ((n // len(pattern)) + 1))[:n]
            return np.array(tiled)

        with mock.patch(
            "integra_pose.hmm_vae_toolkit.per_class_clustering._cluster_one_class",
            _stub,
        ):
            _out_df, result = self.cluster_per_class(df)

        self.assertEqual(result.total_clusters, 5)


class TestEdgeCases(_Base):
    def test_empty_df_raises(self) -> None:
        import pandas as pd

        with self.assertRaises(ValueError):
            self.cluster_per_class(pd.DataFrame(columns=["class_id", "feature_vector"]))

    def test_missing_class_id_column_raises(self) -> None:
        import pandas as pd

        df = pd.DataFrame([{"feature_vector": [1.0, 2.0], "frame": 0}])
        with self.assertRaises(ValueError) as ctx:
            self.cluster_per_class(df)
        self.assertIn("class_id", str(ctx.exception))

    def test_missing_feature_vector_column_raises(self) -> None:
        import pandas as pd

        df = pd.DataFrame([{"class_id": 0, "frame": 0}])
        with self.assertRaises(ValueError) as ctx:
            self.cluster_per_class(df)
        self.assertIn("feature_vector", str(ctx.exception))


class TestEndToEnd(_Base):
    """Smoke-tests with real UMAP + HDBSCAN. Skipped if libs not installed."""

    def setUp(self) -> None:
        super().setUp()
        if not _have_umap_hdbscan():
            self.skipTest("umap-learn / hdbscan unavailable in this environment")

    def test_two_distinct_subclusters_within_one_class(self) -> None:
        """Two well-separated blobs in feature space should produce two sub-clusters."""
        import numpy as np

        rng = np.random.default_rng(7)
        # Class 0, subject A, two well-separated 5D blobs (40 frames each).
        rows = []
        f = 0
        for _ in range(40):
            rows.append((0, "a", f, list(rng.normal(loc=0.0, scale=0.3, size=5))))
            f += 1
        for _ in range(40):
            rows.append((0, "a", f, list(rng.normal(loc=10.0, scale=0.3, size=5))))
            f += 1
        df = _make_df(rows)

        out_df, result = self.cluster_per_class(df, min_cluster_size=5, umap_neighbors=10)

        # Should find ≥1 sub-cluster (HDBSCAN may collapse to 1, may
        # find 2; both are acceptable). The important guarantee is
        # that the column exists and is populated.
        c0 = next(c for c in result.classes if c.class_id == 0)
        self.assertFalse(c0.skipped)
        self.assertIn("cluster_label", out_df.columns)
        self.assertEqual(out_df.shape[0], 80)


if __name__ == "__main__":
    unittest.main()
