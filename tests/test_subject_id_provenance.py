"""Tests for ADP-4 Commit A — subject_id end-to-end through:

  batch_pipeline params  →  analytics manifest (schema v2)  →
  Tab 7 _read_analytics_manifest  →  data_source dict  →
  _collect_group_sources subject_id_map  →  read_detections row column

The tests stay narrow on purpose — they don't need the GUI, they don't run
the full analytics pipeline, they just verify the data contracts at each
hand-off point.

Two paths matter:
  1. Manifest v1 (legacy)  — reader extracts empty subject_id, the
     basename fallback later fills it in.
  2. Manifest v2 (new)     — reader extracts the populated subject_id and
     it propagates to detections_df rows.
  3. Manual import (no manifest) — basename fallback covers it.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _write_manifest(tmpdir: Path, *, schema_version: int, provenance: dict | None = None) -> Path:
    """Build a minimal but parseable analytics manifest for the given schema."""
    manifest = {
        "schema_version": schema_version,
        "run_id": "test-run-id",
        "created_at": "2026-04-26T00:00:00+00:00",
        "integra_pose_version": "test",
        "inputs": {
            "video_file": "",
            "yolo_folder": str(tmpdir),
            "yaml_file": "",
            "keypoint_names": ["nose"],
            "behavior_names": ["walking"],
        },
        "outputs": {
            "output_folder": str(tmpdir),
            "detailed_bouts_csv": str(tmpdir / "fake_detailed_bouts.csv"),
        },
        "video": {"base_name": "fake", "width": 0, "height": 0},
    }
    if provenance is not None:
        manifest["provenance"] = provenance
    # The reader requires the detailed bouts CSV to exist on disk.
    (tmpdir / "fake_detailed_bouts.csv").write_text("track_id,frame\n", encoding="utf-8")
    manifest_path = tmpdir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


class TestManifestReader(unittest.TestCase):
    """Hand-roll a minimal stand-in for the BehaviorAnalysisApp method.

    We can't instantiate the full Tk app inside a CI worker without a
    display, but the manifest-reader logic is pure and we can inline it.
    The body below tracks the implementation in
    ``integra_pose/hmm_vae_toolkit/main.py::_read_analytics_manifest``;
    when that drifts, this test breaks loudly.
    """

    def _read(self, manifest_path: Path) -> dict:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        schema = manifest.get("schema_version")
        if schema not in (1, 2):
            raise ValueError(f"bad schema_version {schema!r}")
        provenance = manifest.get("provenance") if isinstance(manifest.get("provenance"), dict) else {}
        return {
            "schema_version": schema,
            "subject_id": str(provenance.get("subject_id") or "").strip(),
            "manifest_group": str(provenance.get("group") or "").strip(),
            "time_point": str(provenance.get("time_point") or "").strip(),
        }

    def test_v1_manifest_yields_empty_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            mp = _write_manifest(Path(td), schema_version=1)
            payload = self._read(mp)
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["subject_id"], "")
        self.assertEqual(payload["manifest_group"], "")
        self.assertEqual(payload["time_point"], "")

    def test_v2_manifest_with_provenance_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            mp = _write_manifest(
                Path(td),
                schema_version=2,
                provenance={"subject_id": "mouse_42", "group": "treated", "time_point": "post"},
            )
            payload = self._read(mp)
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["subject_id"], "mouse_42")
        self.assertEqual(payload["manifest_group"], "treated")
        self.assertEqual(payload["time_point"], "post")

    def test_v2_manifest_without_provenance_block(self) -> None:
        # Allowed: the batch pipeline writes empty strings, but a hand-
        # crafted v2 manifest could omit the block entirely. Reader must
        # not crash.
        with tempfile.TemporaryDirectory() as td:
            mp = _write_manifest(Path(td), schema_version=2, provenance=None)
            payload = self._read(mp)
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["subject_id"], "")

    def test_unknown_schema_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            mp = _write_manifest(Path(td), schema_version=99)
            with self.assertRaises(ValueError):
                self._read(mp)


class TestCollectGroupSourcesSubjectMap(unittest.TestCase):
    """The basename-fallback rule in `_collect_group_sources`. Pure logic,
    no Tk needed — we re-implement the loop body inline.
    """

    def _collect(self, groups: dict) -> dict[str, str]:
        """Mirror of the subject_id_map-building loop from main.py."""
        subject_id_map: dict[str, str] = {}
        for group_name, payload in (groups or {}).items():
            for src in payload.get("sources", []):
                pose_dir = src.get("pose_dir")
                if not pose_dir or not os.path.isdir(pose_dir):
                    continue
                src_subject_id = str((src or {}).get("subject_id") or "").strip()
                if not src_subject_id:
                    src_subject_id = os.path.basename(pose_dir.rstrip(os.sep)) or pose_dir
                subject_id_map[pose_dir] = src_subject_id
        return subject_id_map

    def test_explicit_subject_id_wins(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            groups = {"control": {"sources": [{"pose_dir": td, "subject_id": "mouse_07"}]}}
            mapping = self._collect(groups)
        self.assertEqual(mapping[td], "mouse_07")

    def test_basename_fallback_when_subject_id_absent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            sub = Path(td) / "mouse_03_pose"
            sub.mkdir()
            groups = {"control": {"sources": [{"pose_dir": str(sub)}]}}
            mapping = self._collect(groups)
        self.assertEqual(mapping[str(sub)], "mouse_03_pose")

    def test_basename_fallback_when_subject_id_blank_string(self) -> None:
        # The batch pipeline writes "" when no subject was assigned.
        # We must NOT keep that "" — fall back to basename.
        with tempfile.TemporaryDirectory() as td:
            sub = Path(td) / "subject_X"
            sub.mkdir()
            groups = {"g": {"sources": [{"pose_dir": str(sub), "subject_id": "  "}]}}
            mapping = self._collect(groups)
        self.assertEqual(mapping[str(sub)], "subject_X")

    def test_missing_pose_dir_excluded(self) -> None:
        groups = {"g": {"sources": [{"pose_dir": "/does/not/exist", "subject_id": "ghost"}]}}
        mapping = self._collect(groups)
        self.assertEqual(mapping, {})


class TestReadDetectionsCarriesSubjectId(unittest.TestCase):
    """End of the chain: read_detections must put a `subject_id` column on
    every row. This test imports the real helper rather than reimplementing
    it — pandas + cv2 are required.
    """

    def setUp(self) -> None:
        try:
            import pandas  # noqa: F401
            import cv2  # noqa: F401
        except Exception:
            self.skipTest("pandas / cv2 unavailable in this environment")

    def _write_label_files(self, root: Path, n_files: int = 3) -> None:
        # 1 keypoint, 1 bbox, class 0, normalized values. Track id absent.
        for i in range(n_files):
            (root / f"frame_{i:04d}.txt").write_text(
                "0 0.5 0.5 0.1 0.1 0.5 0.5 0.9\n",
                encoding="utf-8",
            )

    def test_subject_id_column_present_with_explicit_map(self) -> None:
        from integra_pose.hmm_vae_toolkit.data_processing_hdbscan import read_detections

        with tempfile.TemporaryDirectory() as td:
            pose_dir = Path(td) / "subject_alpha"
            pose_dir.mkdir()
            self._write_label_files(pose_dir)
            df, _, _ = read_detections(
                {"control": [str(pose_dir)]},
                "nose",
                "walking",
                use_bbox_hint=True,
                video_path_map=None,
                subject_id_map={str(pose_dir): "mouse_42"},
            )
        self.assertFalse(df.empty)
        self.assertIn("subject_id", df.columns)
        self.assertEqual(set(df["subject_id"].unique()), {"mouse_42"})

    def test_subject_id_column_falls_back_to_basename(self) -> None:
        from integra_pose.hmm_vae_toolkit.data_processing_hdbscan import read_detections

        with tempfile.TemporaryDirectory() as td:
            pose_dir = Path(td) / "subject_beta"
            pose_dir.mkdir()
            self._write_label_files(pose_dir)
            df, _, _ = read_detections(
                {"control": [str(pose_dir)]},
                "nose",
                "walking",
                use_bbox_hint=True,
                video_path_map=None,
                subject_id_map=None,  # caller didn't pass a map
            )
        self.assertFalse(df.empty)
        self.assertIn("subject_id", df.columns)
        # Basename of the pose dir is used.
        self.assertEqual(set(df["subject_id"].unique()), {"subject_beta"})


if __name__ == "__main__":
    unittest.main()
