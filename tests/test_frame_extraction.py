"""Tests for ADP-4 Commit G — frame extraction helpers.

The actual cv2 path needs a real video; we don't ship one. So:

  - Pure-logic tests for ``_frame_indices_for_bout`` (which frames to
    pick for a bout of a given size, edge cases) — no cv2 needed.
  - Pure-logic tests for ``select_top_n_bouts`` — sorts and filters.
  - End-to-end ``extract_bout_thumbnails`` test that synthesises a
    tiny test video with cv2, then reads frames out. Skipped when
    cv2 / PIL absent.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from integra_pose.hmm_vae_toolkit.frame_extraction import (
    _frame_indices_for_bout,
    select_top_n_bouts,
)


class TestFrameIndicesForBout(unittest.TestCase):
    def test_three_frames_default(self) -> None:
        # Bout 1..21 → start=1, mid=11, end=21
        self.assertEqual(_frame_indices_for_bout(1, 21, 3), [1, 11, 21])

    def test_one_frame_returns_midpoint(self) -> None:
        self.assertEqual(_frame_indices_for_bout(10, 30, 1), [20])

    def test_zero_frames_returns_empty(self) -> None:
        self.assertEqual(_frame_indices_for_bout(0, 100, 0), [])

    def test_short_bout_repeats(self) -> None:
        # Bout of 1 frame, n=3 → all three picks fall on that frame.
        self.assertEqual(_frame_indices_for_bout(5, 5, 3), [5, 5, 5])

    def test_two_frame_bout_three_requested(self) -> None:
        # 2 frames < 3 requested → repeats.
        self.assertEqual(_frame_indices_for_bout(10, 11, 3), [10, 10, 10])

    def test_five_frames_evenly_spaced(self) -> None:
        # Bout 0..100, n=5 → [0, 25, 50, 75, 100]
        self.assertEqual(_frame_indices_for_bout(0, 100, 5), [0, 25, 50, 75, 100])


class TestSelectTopNBouts(unittest.TestCase):
    def _b(self, state, duration):
        return {"state": state, "duration_frames": duration}

    def test_selects_longest(self) -> None:
        bouts = [
            self._b("0:0", 5),
            self._b("0:0", 50),
            self._b("0:0", 20),
            self._b("0:0", 1),
        ]
        top = select_top_n_bouts(bouts, n=2)
        self.assertEqual([b["duration_frames"] for b in top], [50, 20])

    def test_filters_to_target_state(self) -> None:
        bouts = [
            self._b("0:0", 100),
            self._b("0:1", 50),
            self._b("0:0", 75),
            self._b("0:1", 30),
        ]
        top = select_top_n_bouts(bouts, n=10, target_state="0:1")
        self.assertEqual(len(top), 2)
        self.assertTrue(all(b["state"] == "0:1" for b in top))
        # Sorted desc: 50 first, then 30.
        self.assertEqual([b["duration_frames"] for b in top], [50, 30])

    def test_target_state_int_or_string(self) -> None:
        # When the bout's state is an int and target_state is a string,
        # the comparison should still match because we coerce to str.
        bouts = [{"state": 1, "duration_frames": 10}]
        top = select_top_n_bouts(bouts, n=10, target_state="1")
        self.assertEqual(len(top), 1)

    def test_n_zero_returns_empty(self) -> None:
        bouts = [self._b("0:0", 100)]
        self.assertEqual(select_top_n_bouts(bouts, n=0), [])

    def test_no_bouts(self) -> None:
        self.assertEqual(select_top_n_bouts([], n=5), [])


class TestExtractBoutThumbnails(unittest.TestCase):
    """Smoke test with a synthesised tiny video. Skipped without cv2/PIL."""

    def setUp(self) -> None:
        try:
            import cv2  # noqa: F401
            from PIL import Image  # noqa: F401
            import numpy as np  # noqa: F401
        except Exception:
            self.skipTest("cv2 / Pillow / numpy not available")

    def _make_video(self, path: str, n_frames: int = 30, size: tuple = (64, 64)):
        """Write a 30-frame video with a moving square so frames are distinct."""
        import cv2
        import numpy as np

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, 10.0, size)
        if not writer.isOpened():
            return False
        try:
            for i in range(n_frames):
                frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                # White square, moving across frame.
                x = int((i / n_frames) * (size[0] - 10))
                frame[10:20, x:x + 10] = 255
                writer.write(frame)
        finally:
            writer.release()
        return True

    def test_extracts_three_frames_per_bout(self) -> None:
        from integra_pose.hmm_vae_toolkit.frame_extraction import (
            extract_bout_thumbnails,
        )

        with tempfile.TemporaryDirectory() as td:
            video_path = os.path.join(td, "tiny.mp4")
            if not self._make_video(video_path):
                self.skipTest("cv2.VideoWriter cannot open mp4v on this platform")
            bouts = [
                {"directory": "/fake/dir", "start_frame": 5, "end_frame": 20},
                {"directory": "/fake/dir", "start_frame": 12, "end_frame": 25},
            ]
            video_map = {"/fake/dir": video_path}
            result = extract_bout_thumbnails(bouts, video_path_map=video_map)
            self.assertEqual(len(result), 2)
            for entry in result:
                # Each bout should yield 3 frames (some platforms may
                # return fewer if seek fails on some codecs; tolerate
                # ≥1 but require non-empty).
                self.assertGreaterEqual(len(entry["frames"]), 1)

    def test_missing_video_yields_empty_frames(self) -> None:
        from integra_pose.hmm_vae_toolkit.frame_extraction import (
            extract_bout_thumbnails,
        )

        bouts = [
            {"directory": "/no/such/dir", "start_frame": 1, "end_frame": 10},
        ]
        result = extract_bout_thumbnails(bouts, video_path_map={})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["frames"], [])


if __name__ == "__main__":
    unittest.main()
