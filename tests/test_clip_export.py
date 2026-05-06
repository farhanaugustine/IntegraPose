"""Tests for clip_export.py — Tab 7 sub-cluster clip generator.

The actual MP4 writing requires cv2; we don't run that path in CI. What
we DO test:

  - Label parsing (namespaced strings → (class_id, sub_id), with sane
    fallbacks for noise / malformed)
  - Filename sanitization (Windows / POSIX safe)
  - Filename selection logic (prefers state_names → falls back to
    class_names → falls back to integer fallback)
  - Error paths: missing video file, missing required df columns,
    cv2 unavailable

These cover ~80% of the "did I refactor it correctly" surface without
needing OpenCV. The end-to-end clip-writing test runs only when cv2
and a tiny test video are available.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from integra_pose.hmm_vae_toolkit.clip_export import (
    _parse_namespaced_label,
    _safe_filename_part,
    _behavior_folder_name,
    CLIP_MANIFEST_FILENAME,
)


class TestNamespacedLabelParsing(unittest.TestCase):
    def test_normal_namespaced(self) -> None:
        self.assertEqual(_parse_namespaced_label("0:1"), (0, 1))
        self.assertEqual(_parse_namespaced_label("3:7"), (3, 7))

    def test_noise_label_returns_none(self) -> None:
        self.assertIsNone(_parse_namespaced_label("-1"))

    def test_none_input_returns_none(self) -> None:
        self.assertIsNone(_parse_namespaced_label(None))

    def test_malformed_returns_none(self) -> None:
        self.assertIsNone(_parse_namespaced_label("not_a_label"))
        self.assertIsNone(_parse_namespaced_label("0:not_int"))
        self.assertIsNone(_parse_namespaced_label(""))

    def test_integer_input_coerced_to_string(self) -> None:
        # Defensive: someone might accidentally pass an int.
        self.assertIsNone(_parse_namespaced_label(0))  # no colon
        self.assertIsNone(_parse_namespaced_label(-1))


class TestFilenameSanitization(unittest.TestCase):
    def test_normal_string_unchanged(self) -> None:
        self.assertEqual(_safe_filename_part("walking"), "walking")

    def test_spaces_become_underscores(self) -> None:
        self.assertEqual(_safe_filename_part("forward locomotion"), "forward_locomotion")

    def test_special_chars_stripped(self) -> None:
        # Keys that aren't allowed on Windows: < > : " / \ | ? *
        result = _safe_filename_part('foo<>:"/\\|?*bar')
        self.assertNotIn("<", result)
        self.assertNotIn(">", result)
        self.assertNotIn(":", result)
        self.assertNotIn("\\", result)
        self.assertNotIn("/", result)

    def test_empty_string_yields_unnamed(self) -> None:
        self.assertEqual(_safe_filename_part(""), "unnamed")
        self.assertEqual(_safe_filename_part("___"), "unnamed")


class TestExportErrorPaths(unittest.TestCase):
    """Errors raised before any cv2 work happens (so they fire even
    in environments without cv2)."""

    def setUp(self) -> None:
        try:
            import pandas as pd
            import numpy  # noqa: F401
            self.pd = pd
        except Exception:
            self.skipTest("pandas / numpy unavailable")

    def test_missing_video_raises_filenotfound(self) -> None:
        from integra_pose.hmm_vae_toolkit.clip_export import export_sub_cluster_clips

        df = self.pd.DataFrame([
            {"frame": 1, "bbox": [10, 10, 20, 20], "keypoints": {}, "cluster_label": "0:0"},
        ])
        with tempfile.TemporaryDirectory() as td:
            # cv2 may or may not be installed. We expect either
            # ImportError (no cv2) or FileNotFoundError (cv2 present
            # but video missing). Either is acceptable; the contract
            # is that the function fails fast and clearly.
            with self.assertRaises((FileNotFoundError, ImportError)):
                export_sub_cluster_clips(
                    df,
                    video_path=str(Path(td) / "no_such_video.mp4"),
                    output_dir=td,
                )

    def test_missing_required_column_raises_value_error(self) -> None:
        from integra_pose.hmm_vae_toolkit.clip_export import export_sub_cluster_clips

        try:
            import cv2  # noqa: F401
        except Exception:
            self.skipTest("cv2 unavailable; can't reach the column-check path")

        df = self.pd.DataFrame([
            {"frame": 1, "cluster_label": "0:0"},  # missing bbox + keypoints
        ])
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                export_sub_cluster_clips(
                    df,
                    video_path=video_path,
                    output_dir=tempfile.mkdtemp(),
                )
            self.assertIn("missing required column", str(ctx.exception))
        finally:
            os.unlink(video_path)


class TestExportEmptyResult(unittest.TestCase):
    """If every frame is noise, return empty dict — don't write zero-byte
    files."""

    def setUp(self) -> None:
        try:
            import pandas as pd
            import cv2  # noqa: F401
            self.pd = pd
        except Exception:
            self.skipTest("pandas + cv2 required for end-to-end exit-clean test")

    def test_all_noise_writes_nothing(self) -> None:
        from integra_pose.hmm_vae_toolkit.clip_export import export_sub_cluster_clips

        # Every frame in noise (-1) → expect no clips written.
        df = self.pd.DataFrame([
            {"frame": i, "bbox": [10, 10, 20, 20], "keypoints": {}, "cluster_label": "-1"}
            for i in range(10)
        ])

        # Create a 1-frame valid video so cv2.VideoCapture opens.
        # We don't actually need a real video; clip_export bails before
        # iterating because there are no non-noise frames.
        import cv2

        tmpdir = tempfile.mkdtemp()
        video_path = os.path.join(tmpdir, "tiny.mp4")
        out = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (32, 32)
        )
        if not out.isOpened():
            self.skipTest("cv2.VideoWriter cannot open mp4v on this platform")
        import numpy as np

        out.write(np.zeros((32, 32, 3), dtype=np.uint8))
        out.release()

        try:
            written = export_sub_cluster_clips(df, video_path=video_path, output_dir=tmpdir)
            self.assertEqual(written, {})
        finally:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)


class TestBehaviorFolderName(unittest.TestCase):
    """Folder picking precedence: state_names → class_names → integer fallback."""

    def test_state_names_wins(self) -> None:
        self.assertEqual(
            _behavior_folder_name(
                0, 1,
                class_names={0: "walking"},
                state_names={"0:1": "forward locomotion"},
            ),
            "forward_locomotion",
        )

    def test_falls_back_to_class_name(self) -> None:
        self.assertEqual(
            _behavior_folder_name(
                0, 2, class_names={0: "walking"}, state_names={},
            ),
            "walking__subcluster_2",
        )

    def test_blank_state_name_falls_through(self) -> None:
        # Empty string in state_names should NOT clobber the class fallback.
        self.assertEqual(
            _behavior_folder_name(
                0, 1, class_names={0: "walking"}, state_names={"0:1": "  "},
            ),
            "walking__subcluster_1",
        )

    def test_no_class_name(self) -> None:
        self.assertEqual(
            _behavior_folder_name(3, 0, class_names={}, state_names={}),
            "class_3__subcluster_0",
        )

    def test_unsafe_chars_sanitized(self) -> None:
        result = _behavior_folder_name(
            0, 1, state_names={"0:1": "fight/flight (sub*type)"},
        )
        # No path separator, no shell wildcards.
        self.assertNotIn("/", result)
        self.assertNotIn("*", result)
        self.assertNotIn("\\", result)


class TestExportBoutClipsLogic(unittest.TestCase):
    """Pure-logic tests for export_bout_clips. We don't actually write
    .mp4s — those need cv2. Instead, exercise the early-return / skip
    paths that fire BEFORE any cv2 work, and the manifest emission."""

    def setUp(self) -> None:
        # cv2 is needed for the `import cv2` at the top of the function.
        # If absent, the function raises ImportError at the FIRST line —
        # that's tested below as `test_no_cv2_raises_importerror`.
        try:
            import cv2  # noqa: F401
            self._has_cv2 = True
        except Exception:
            self._has_cv2 = False

    def test_empty_output_dir_raises(self) -> None:
        if not self._has_cv2:
            self.skipTest("cv2 unavailable; ImportError fires before ValueError")
        from integra_pose.hmm_vae_toolkit.clip_export import export_bout_clips

        with self.assertRaises(ValueError):
            export_bout_clips([], video_path_map={}, output_dir="")

    def test_no_bouts_writes_empty_manifest(self) -> None:
        if not self._has_cv2:
            self.skipTest("cv2 unavailable")
        from integra_pose.hmm_vae_toolkit.clip_export import export_bout_clips

        with tempfile.TemporaryDirectory() as td:
            rows = export_bout_clips(
                [], video_path_map={}, output_dir=td,
            )
            self.assertEqual(rows, [])
            # Manifest written even when empty (header only) so callers
            # can rely on it existing.
            manifest = Path(td) / CLIP_MANIFEST_FILENAME
            self.assertTrue(manifest.exists())
            content = manifest.read_text(encoding="utf-8")
            self.assertIn("status", content)
            self.assertIn("behavior_folder", content)

    def test_bout_with_no_video_marked_skipped(self) -> None:
        if not self._has_cv2:
            self.skipTest("cv2 unavailable")
        from integra_pose.hmm_vae_toolkit.clip_export import export_bout_clips

        bouts = [{
            "directory": "/no/such/dir",
            "start_frame": 1, "end_frame": 10,
            "duration_frames": 10,
            "state": "0:1", "track_id": 0,
            "class_id": 0,
        }]
        with tempfile.TemporaryDirectory() as td:
            rows = export_bout_clips(
                bouts,
                video_path_map={},  # nothing maps the directory
                output_dir=td,
                state_names={"0:1": "forward-locomotion"},
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "skipped")
            self.assertIn("no source video", rows[0]["reason"])
            # Manifest mirrors the in-memory rows.
            manifest = Path(td) / CLIP_MANIFEST_FILENAME
            self.assertTrue(manifest.exists())
            text = manifest.read_text(encoding="utf-8")
            self.assertIn("forward-locomotion", text)
            self.assertIn("skipped", text)

    def test_noise_bouts_skipped_with_reason(self) -> None:
        if not self._has_cv2:
            self.skipTest("cv2 unavailable")
        from integra_pose.hmm_vae_toolkit.clip_export import export_bout_clips

        bouts = [{
            "directory": "/some/dir",
            "start_frame": 1, "end_frame": 10,
            "duration_frames": 10,
            "state": "-1", "track_id": 0,
        }]
        with tempfile.TemporaryDirectory() as td:
            rows = export_bout_clips(
                bouts,
                # video_path_map missing → bout would be skipped for video
                # absence first. So pass a dummy path that exists but cv2
                # won't be able to read; the noise check should still
                # take effect via the "noise label" reason if cv2 opens it.
                video_path_map={"/some/dir": td + "/nope.mp4"},
                output_dir=td,
            )
            # Either skipped because video missing OR skipped because
            # noise — both reasons are valid.
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "skipped")

    def test_progress_callback_invoked(self) -> None:
        if not self._has_cv2:
            self.skipTest("cv2 unavailable")
        from integra_pose.hmm_vae_toolkit.clip_export import export_bout_clips

        bouts = [
            {"directory": "/d1", "start_frame": 1, "end_frame": 5,
             "state": "0:1", "track_id": 0, "duration_frames": 5},
            {"directory": "/d2", "start_frame": 1, "end_frame": 5,
             "state": "0:2", "track_id": 0, "duration_frames": 5},
        ]
        events: list = []

        def cb(done, total, msg):
            events.append((done, total, msg))

        with tempfile.TemporaryDirectory() as td:
            export_bout_clips(
                bouts, video_path_map={},  # both will skip
                output_dir=td, progress_callback=cb,
            )
        # One callback per bout.
        self.assertEqual(len(events), 2)
        # Final event should report 2 of 2.
        self.assertEqual(events[-1][0], 2)
        self.assertEqual(events[-1][1], 2)

    def test_run_id_threaded_into_manifest(self) -> None:
        if not self._has_cv2:
            self.skipTest("cv2 unavailable")
        from integra_pose.hmm_vae_toolkit.clip_export import export_bout_clips

        bouts = [{
            "directory": "/dir",
            "start_frame": 1, "end_frame": 5,
            "state": "0:1", "track_id": 0,
            "duration_frames": 5,
        }]
        with tempfile.TemporaryDirectory() as td:
            export_bout_clips(
                bouts, video_path_map={}, output_dir=td,
                run_id="my-uuid-xyz",
            )
            manifest = (Path(td) / CLIP_MANIFEST_FILENAME).read_text(encoding="utf-8")
            self.assertIn("my-uuid-xyz", manifest)


class TestExportBoutClipsEndToEnd(unittest.TestCase):
    """End-to-end: synthesize a tiny video and write a real bout clip.

    Skipped without cv2 / numpy. This covers the seek + read + write
    path that pure-logic tests can't reach.
    """

    def setUp(self) -> None:
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401
            self._ok = True
        except Exception:
            self._ok = False
            self.skipTest("cv2 / numpy unavailable")

    def _make_video(self, path: str, n_frames: int = 30):
        import cv2
        import numpy as np

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, 10.0, (64, 64))
        if not writer.isOpened():
            return False
        try:
            for i in range(n_frames):
                frame = np.zeros((64, 64, 3), dtype=np.uint8)
                # Moving square — distinct frames.
                x = int((i / n_frames) * 50)
                frame[10:20, x:x + 10] = 255
                writer.write(frame)
        finally:
            writer.release()
        return True

    def test_writes_clips_into_behavior_folders(self) -> None:
        from integra_pose.hmm_vae_toolkit.clip_export import export_bout_clips

        with tempfile.TemporaryDirectory() as td:
            video_path = os.path.join(td, "vid.mp4")
            if not self._make_video(video_path):
                self.skipTest("cv2.VideoWriter cannot open mp4v")
            output_dir = os.path.join(td, "out")
            bouts = [
                {"directory": "/poses", "start_frame": 5, "end_frame": 12,
                 "duration_frames": 8, "state": "0:1", "track_id": 0},
                {"directory": "/poses", "start_frame": 15, "end_frame": 20,
                 "duration_frames": 6, "state": "0:2", "track_id": 0},
                {"directory": "/poses", "start_frame": 22, "end_frame": 28,
                 "duration_frames": 7, "state": "0:1", "track_id": 0},
            ]
            rows = export_bout_clips(
                bouts,
                video_path_map={"/poses": video_path},
                output_dir=output_dir,
                state_names={"0:1": "forward-locomotion"},
                class_names={0: "walking"},
                overlay=False,  # skip overlay so the test doesn't need detections_df
            )
            written = [r for r in rows if r["status"] == "written"]
            # All three bouts should write.
            self.assertEqual(len(written), 3)

            # Folder structure: forward-locomotion/ and walking__subcluster_2/
            forward_dir = Path(output_dir) / "forward-locomotion"
            sub2_dir = Path(output_dir) / "walking__subcluster_2"
            self.assertTrue(forward_dir.is_dir(), "forward-locomotion folder missing")
            self.assertTrue(sub2_dir.is_dir(), "walking__subcluster_2 folder missing")

            forward_clips = list(forward_dir.glob("*.mp4"))
            sub2_clips = list(sub2_dir.glob("*.mp4"))
            self.assertEqual(len(forward_clips), 2)
            self.assertEqual(len(sub2_clips), 1)

            # Manifest at root of output_dir.
            manifest = Path(output_dir) / CLIP_MANIFEST_FILENAME
            self.assertTrue(manifest.exists())


if __name__ == "__main__":
    unittest.main()
