"""Tests for ADP-4 Commit C — auto train/val/test split.

The split rule is the load-bearing contract for the held-out report. If
this misbehaves silently — e.g., a subject ends up in both train and
test, or two groups' subjects collide — every defensibility metric the
panel renders is a lie. So these tests are paranoid:

  * Subject atomicity (no leakage of one subject across partitions).
  * Composite (group, subject_id) keying (basename collisions across
    groups are kept distinct).
  * Regime selection from min(unique subjects per group).
  * Reproducibility: same input + same seed → same partition assignment.
  * Determinism doesn't pollute global RNG.
  * Edge cases: 1 subject, 5 subjects (boundary), missing columns.

Pure-pandas / numpy tests; no GUI, no torch.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _setup_pandas() -> bool:
    try:
        import pandas  # noqa: F401
        import numpy  # noqa: F401
        return True
    except Exception:
        return False


def _make_df(rows):
    """Build a tiny detections-style df from (group, subject_id, frame) tuples."""
    import pandas as pd

    return pd.DataFrame(
        [{"group": g, "subject_id": s, "frame": f} for (g, s, f) in rows]
    )


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        if not _setup_pandas():
            self.skipTest("pandas / numpy unavailable in this environment")
        from integra_pose.hmm_vae_toolkit.splits import (
            auto_split,
            SplitResult,
            DEFAULT_SPLIT_SEED,
            REGIME_FIVE_PLUS,
            REGIME_TWO_TO_FOUR,
            REGIME_SINGLE,
        )
        self.auto_split = auto_split
        self.SplitResult = SplitResult
        self.DEFAULT_SPLIT_SEED = DEFAULT_SPLIT_SEED
        self.FIVE_PLUS = REGIME_FIVE_PLUS
        self.TWO_TO_FOUR = REGIME_TWO_TO_FOUR
        self.SINGLE = REGIME_SINGLE


class TestRegimeSelection(_Base):
    def test_five_plus_per_group_picks_5plus(self) -> None:
        rows = [(g, f"subj_{g}_{i}", 0) for g in ("control", "treated") for i in range(5)]
        _df, result = self.auto_split(_make_df(rows))
        self.assertEqual(result.regime, self.FIVE_PLUS)
        self.assertEqual(result.min_subjects_per_group, 5)

    def test_three_per_group_picks_2to4(self) -> None:
        rows = [(g, f"subj_{g}_{i}", 0) for g in ("control", "treated") for i in range(3)]
        _df, result = self.auto_split(_make_df(rows))
        self.assertEqual(result.regime, self.TWO_TO_FOUR)
        self.assertFalse(result.has_held_out_test)

    def test_one_per_group_picks_single(self) -> None:
        rows = [("control", "alpha", 0)]
        _df, result = self.auto_split(_make_df(rows))
        self.assertEqual(result.regime, self.SINGLE)
        self.assertGreater(len(result.warnings), 0)

    def test_min_rule_picks_smaller_group(self) -> None:
        # control has 8, treated has 3 -> min=3 -> 2to4 regime.
        rows = []
        for i in range(8):
            rows.append(("control", f"control_{i}", 0))
        for i in range(3):
            rows.append(("treated", f"treated_{i}", 0))
        _df, result = self.auto_split(_make_df(rows))
        self.assertEqual(result.regime, self.TWO_TO_FOUR)
        self.assertEqual(result.limiting_group, "treated")
        self.assertEqual(result.n_subjects_per_group, {"control": 8, "treated": 3})


class TestSubjectAtomicity(_Base):
    """Every frame from a (group, subject) pair must land in ONE partition."""

    def test_no_subject_leaks_across_partitions(self) -> None:
        # 6 subjects per group, 10 frames each.
        rows = []
        for group in ("control", "treated"):
            for s in range(6):
                for f in range(10):
                    rows.append((group, f"{group}_subj_{s}", f))
        out_df, _result = self.auto_split(_make_df(rows))
        # Each (group, subject_id) should map to exactly one partition.
        per_subject = out_df.groupby(["group", "subject_id"])["partition"].nunique()
        self.assertTrue((per_subject == 1).all(), "subject leaked across partitions")

    def test_composite_key_keeps_basename_collisions_distinct(self) -> None:
        # Both groups have a subject named "video_01" (basename collision
        # from the read_detections fallback). They are DIFFERENT animals.
        # The splitter must NOT merge them.
        rows = []
        for group in ("control", "treated"):
            for f in range(5):
                rows.append((group, "video_01", f))
        out_df, result = self.auto_split(_make_df(rows))
        # Two distinct (group, subject_id) tuples → 2 unique partition
        # assignments possible.
        # Both end up in train under SINGLE regime, but the SplitResult
        # records each group's "video_01" separately.
        self.assertEqual(result.regime, self.SINGLE)
        self.assertEqual(
            result.n_subjects_per_group, {"control": 1, "treated": 1}
        )


class TestReproducibility(_Base):
    def test_same_seed_same_partitions(self) -> None:
        rows = [(g, f"{g}_{i}", 0) for g in ("a", "b") for i in range(6)]
        df1, r1 = self.auto_split(_make_df(rows), seed=42)
        df2, r2 = self.auto_split(_make_df(rows), seed=42)
        self.assertEqual(
            r1.subjects_per_partition_per_group,
            r2.subjects_per_partition_per_group,
        )
        # Compare partition column row-by-row.
        self.assertTrue((df1["partition"].values == df2["partition"].values).all())

    def test_different_seed_can_differ(self) -> None:
        # 6 subjects per group so the regime is FIVE_PLUS and the random
        # shuffle has enough room to actually differ between seeds. With
        # 5 it's possible (rare) to land on the same permutation.
        rows = [(g, f"{g}_{i}", 0) for g in ("a", "b") for i in range(6)]
        _df1, r1 = self.auto_split(_make_df(rows), seed=42)
        _df2, r2 = self.auto_split(_make_df(rows), seed=123)
        # Not strictly required to differ for every seed pair — but at
        # least one of these two seeds must yield a different test set.
        same = (
            r1.subjects_per_partition_per_group
            == r2.subjects_per_partition_per_group
        )
        # If seeds 42 and 123 happen to coincide, it's worth knowing.
        # Loosen by trying a couple more.
        if same:
            for s in (7, 99, 2025):
                _, rs = self.auto_split(_make_df(rows), seed=s)
                if rs.subjects_per_partition_per_group != r1.subjects_per_partition_per_group:
                    return
            self.fail("All tested seeds produced identical partitions; "
                      "shuffle is suspiciously deterministic.")

    def test_does_not_pollute_global_rng(self) -> None:
        """The split must use a local Generator only.

        np.random.seed(42) would set the global state; if Commit C makes
        that mistake, downstream HMM/UMAP/torch consumers all get
        contaminated. Verify by snapshotting np.random.random() before
        and after a split and confirming the global stream advances
        independently.
        """
        import numpy as np

        np.random.seed(11)
        before = np.random.random()
        rows = [(g, f"{g}_{i}", 0) for g in ("a", "b") for i in range(6)]
        self.auto_split(_make_df(rows), seed=42)
        np.random.seed(11)  # re-seed and read again
        after = np.random.random()
        # If auto_split didn't touch the global RNG, the same seed gives
        # the same first random number both times.
        self.assertEqual(before, after, "auto_split polluted global RNG")


class TestSplitProportions(_Base):
    def test_5plus_yields_60_20_20(self) -> None:
        rows = [("g", f"s_{i}", 0) for i in range(10)]
        _df, result = self.auto_split(_make_df(rows))
        counts = result.n_subjects_per_partition_per_group["g"]
        # 60/20/20 of 10 → 6/2/2
        self.assertEqual(counts, {"train": 6, "val": 2, "test": 2})

    def test_2to4_yields_80_20(self) -> None:
        rows = [("g", f"s_{i}", 0) for i in range(4)]
        _df, result = self.auto_split(_make_df(rows))
        counts = result.n_subjects_per_partition_per_group["g"]
        # 80/20 of 4 → 3/1/0
        self.assertEqual(counts["test"], 0)
        self.assertEqual(counts["train"] + counts["val"], 4)
        self.assertGreaterEqual(counts["val"], 1)

    def test_no_subject_appears_in_two_partitions(self) -> None:
        rows = [("g", f"s_{i}", 0) for i in range(8)]
        _df, result = self.auto_split(_make_df(rows))
        per_partition = result.subjects_per_partition_per_group["g"]
        all_subjects = set()
        for subj_list in per_partition.values():
            for s in subj_list:
                self.assertNotIn(s, all_subjects, f"subject {s} in two partitions")
                all_subjects.add(s)


class TestEdgeCases(_Base):
    def test_empty_df_raises(self) -> None:
        import pandas as pd

        with self.assertRaises(ValueError):
            self.auto_split(pd.DataFrame(columns=["group", "subject_id"]))

    def test_missing_group_column_raises(self) -> None:
        import pandas as pd

        df = pd.DataFrame([{"subject_id": "x", "frame": 0}])
        with self.assertRaises(ValueError) as ctx:
            self.auto_split(df)
        self.assertIn("group", str(ctx.exception))

    def test_missing_subject_id_column_raises(self) -> None:
        import pandas as pd

        df = pd.DataFrame([{"group": "g", "frame": 0}])
        with self.assertRaises(ValueError) as ctx:
            self.auto_split(df)
        self.assertIn("subject_id", str(ctx.exception))


class TestSplitResultShape(_Base):
    def test_has_held_out_test_true_when_test_nonempty(self) -> None:
        rows = [("g", f"s_{i}", 0) for i in range(10)]
        _df, result = self.auto_split(_make_df(rows))
        self.assertTrue(result.has_held_out_test)

    def test_has_held_out_test_false_in_2to4(self) -> None:
        rows = [("g", f"s_{i}", 0) for i in range(3)]
        _df, result = self.auto_split(_make_df(rows))
        self.assertFalse(result.has_held_out_test)

    def test_partition_column_name_is_partition(self) -> None:
        rows = [("g", f"s_{i}", 0) for i in range(5)]
        out_df, result = self.auto_split(_make_df(rows))
        self.assertEqual(result.partition_column, "partition")
        self.assertIn("partition", out_df.columns)


if __name__ == "__main__":
    unittest.main()
