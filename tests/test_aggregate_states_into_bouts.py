"""Tests for ADP-4 Commit B — aggregate_states_into_bouts.

Builds a synthetic per-frame dataframe in pandas (no ML, no GUI), then
asserts the aggregator slices it into the expected bouts. Covers:

  - State changes start new bouts.
  - Frame gaps larger than max_gap split bouts even on the same state.
  - min_bout_duration drops short bouts.
  - Tracks are isolated (state on track A doesn't influence track B).
  - Optional columns (subject_id, class_id, feature_vector) pass through
    when present and don't crash when absent.
  - Empty dataframe returns [] without raising.
  - Missing state column raises a clear ValueError.

These tests are deterministic and fast — they hit the fastest single-stage
of the Tab 7 pipeline in isolation, so a regression here is hard to miss.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _setup_pandas():
    try:
        import pandas as pd  # noqa: F401
        import numpy as np  # noqa: F401
        return True
    except Exception:
        return False


class _BaseAggregateTest(unittest.TestCase):
    def setUp(self) -> None:
        if not _setup_pandas():
            self.skipTest("pandas / numpy unavailable in this environment")
        # Import from the standalone module — `main.py` re-exports the
        # same symbol but pulls in matplotlib/torch/etc., which would
        # gate this test on a heavy install.
        from integra_pose.hmm_vae_toolkit.bouts import aggregate_states_into_bouts
        self.fn = aggregate_states_into_bouts


def _row(group, directory, track_id, frame, state, *, class_id=0, subject_id=""):
    """Helper: build one frame-level row dict."""
    return {
        "group": group,
        "video_source": directory,
        "directory": directory,
        "track_id": track_id,
        "subject_id": subject_id,
        "frame": frame,
        "class_id": class_id,
        "behavior": f"class_{class_id}",
        "cluster_label": state,
        "bbox": [10.0, 10.0, 20.0, 20.0],
        "keypoints": {"nose": (10.0, 10.0, 0.9)},
    }


class TestStateChangeSegmentation(_BaseAggregateTest):
    def test_simple_two_state_run(self) -> None:
        import pandas as pd

        df = pd.DataFrame([
            _row("g", "d", 1, frame, state)
            for frame, state in zip(range(10), [0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
        ])
        bouts = self.fn(df, max_frame_gap=2, min_bout_duration=1)
        states = [b["state"] for b in bouts]
        self.assertEqual(states, [0, 1, 0])
        durations = [b["duration_frames"] for b in bouts]
        self.assertEqual(durations, [3, 3, 4])

    def test_min_bout_duration_drops_short_bouts(self) -> None:
        import pandas as pd

        df = pd.DataFrame([
            _row("g", "d", 1, frame, state)
            for frame, state in enumerate([0, 0, 0, 0, 1, 0, 0, 0, 0])
            #                              long-0       short-1     long-0
        ])
        bouts = self.fn(df, max_frame_gap=2, min_bout_duration=2)
        # The single-frame state-1 bout is dropped.
        self.assertEqual([b["state"] for b in bouts], [0, 0])

    def test_singleton_kept_when_min_is_one(self) -> None:
        import pandas as pd

        df = pd.DataFrame([
            _row("g", "d", 1, frame, state)
            for frame, state in enumerate([0, 1, 0])
        ])
        bouts = self.fn(df, max_frame_gap=2, min_bout_duration=1)
        self.assertEqual([b["state"] for b in bouts], [0, 1, 0])
        self.assertEqual([b["duration_frames"] for b in bouts], [1, 1, 1])


class TestFrameGapBreaksBouts(_BaseAggregateTest):
    def test_gap_larger_than_max_breaks_same_state(self) -> None:
        import pandas as pd

        # Same state, but frames 0,1,2 then a 5-frame gap, then 8,9.
        df = pd.DataFrame([
            _row("g", "d", 1, 0, 0),
            _row("g", "d", 1, 1, 0),
            _row("g", "d", 1, 2, 0),
            _row("g", "d", 1, 8, 0),
            _row("g", "d", 1, 9, 0),
        ])
        bouts = self.fn(df, max_frame_gap=2, min_bout_duration=1)
        # max_gap=2 means a diff of 6 (8-2) breaks the bout.
        self.assertEqual(len(bouts), 2)
        self.assertEqual(bouts[0]["start_frame"], 0)
        self.assertEqual(bouts[0]["end_frame"], 2)
        self.assertEqual(bouts[1]["start_frame"], 8)

    def test_gap_within_max_keeps_bout(self) -> None:
        import pandas as pd

        df = pd.DataFrame([
            _row("g", "d", 1, 0, 0),
            _row("g", "d", 1, 1, 0),
            _row("g", "d", 1, 3, 0),  # diff=2, within max_gap=2
            _row("g", "d", 1, 4, 0),
        ])
        bouts = self.fn(df, max_frame_gap=2, min_bout_duration=1)
        self.assertEqual(len(bouts), 1)
        self.assertEqual(bouts[0]["duration_frames"], 5)  # frames 0..4 inclusive
        self.assertEqual(bouts[0]["detection_count"], 4)


class TestTrackIsolation(_BaseAggregateTest):
    def test_two_tracks_segmented_independently(self) -> None:
        import pandas as pd

        rows = []
        # Track 1: stays in state 0
        for f in range(5):
            rows.append(_row("g", "d", 1, f, 0))
        # Track 2: state 1 the whole time, same frames
        for f in range(5):
            rows.append(_row("g", "d", 2, f, 1))
        df = pd.DataFrame(rows)
        bouts = self.fn(df, max_frame_gap=2, min_bout_duration=1)
        self.assertEqual(len(bouts), 2)
        states_by_track = {b["track_id"]: b["state"] for b in bouts}
        self.assertEqual(states_by_track, {1: 0, 2: 1})


class TestPassThroughColumns(_BaseAggregateTest):
    def test_subject_id_carried_onto_bouts(self) -> None:
        import pandas as pd

        df = pd.DataFrame([
            _row("g", "d", 1, f, 0, subject_id="mouse_A") for f in range(4)
        ])
        bouts = self.fn(df, max_frame_gap=2, min_bout_duration=1)
        self.assertEqual(len(bouts), 1)
        self.assertEqual(bouts[0]["subject_id"], "mouse_A")

    def test_state_column_field_records_source_column(self) -> None:
        import pandas as pd

        df = pd.DataFrame([
            _row("g", "d", 1, f, 0) for f in range(3)
        ])
        bouts = self.fn(df, max_frame_gap=2, min_bout_duration=1)
        self.assertEqual(bouts[0]["state_column"], "cluster_label")

    def test_alternate_state_column(self) -> None:
        import pandas as pd

        rows = []
        for f, s in enumerate([0, 0, 1, 1, 1]):
            r = _row("g", "d", 1, f, -1)  # cluster_label is -1 (unused)
            r["pose_category"] = s
            rows.append(r)
        df = pd.DataFrame(rows)
        bouts = self.fn(df, max_frame_gap=2, min_bout_duration=1, state_column="pose_category")
        self.assertEqual([b["state"] for b in bouts], [0, 1])
        self.assertTrue(all(b["state_column"] == "pose_category" for b in bouts))

    def test_dominant_class_id_in_mixed_bout(self) -> None:
        # Class id can fluctuate within a single state bout (e.g., YOLO
        # noise). The bout's class_id should be the mode, not just iloc[0].
        import pandas as pd

        rows = [
            _row("g", "d", 1, 0, 0, class_id=0),
            _row("g", "d", 1, 1, 0, class_id=1),
            _row("g", "d", 1, 2, 0, class_id=1),
            _row("g", "d", 1, 3, 0, class_id=1),
        ]
        df = pd.DataFrame(rows)
        bouts = self.fn(df, max_frame_gap=2, min_bout_duration=1)
        self.assertEqual(len(bouts), 1)
        self.assertEqual(bouts[0]["class_id"], 1)


class TestEdgeCases(_BaseAggregateTest):
    def test_empty_df_returns_empty_list(self) -> None:
        import pandas as pd

        df = pd.DataFrame(columns=["group", "directory", "track_id", "frame", "cluster_label"])
        bouts = self.fn(df, max_frame_gap=2, min_bout_duration=1)
        self.assertEqual(bouts, [])

    def test_missing_state_column_raises_clear_error(self) -> None:
        import pandas as pd

        df = pd.DataFrame([_row("g", "d", 1, f, 0) for f in range(3)])
        df = df.drop(columns=["cluster_label"])
        with self.assertRaises(ValueError) as ctx:
            self.fn(df, max_frame_gap=2, min_bout_duration=1)
        self.assertIn("cluster_label", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
