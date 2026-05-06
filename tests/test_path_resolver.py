"""Tests for cross-machine path resolution (PR 2B coherence-pass 2e).

Covers:
    * to_storage_path: paths under the project root → relative; outside → absolute
    * resolve_storage_path: relative → re-anchored absolute; absolute → unchanged
    * is_unc_path: Windows UNC detection (no-op on POSIX)
    * find_missing_paths: filters to entries that don't exist on disk
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from integra_pose.utils.path_resolver import (
    find_missing_paths,
    is_unc_path,
    resolve_storage_path,
    to_storage_path,
)


class TestToStoragePath(unittest.TestCase):
    """Saving: absolute paths under the project root collapse to relative."""

    def test_path_under_root_becomes_relative(self) -> None:
        # Use a real (existing) folder pair so resolve() works deterministically.
        root = Path(__file__).resolve().parent
        # The tests folder contains many files; pick one we know exists.
        target = Path(__file__).resolve()
        stored = to_storage_path(str(target), root)
        # Must be relative (no leading drive letter, no leading slash).
        self.assertFalse(Path(stored).is_absolute(), f"Expected relative, got {stored!r}")
        # Must use POSIX separators.
        self.assertNotIn("\\", stored)

    def test_path_outside_root_stays_absolute(self) -> None:
        # The system temp dir is reliably outside any project tree.
        import tempfile
        outside = Path(tempfile.gettempdir()).resolve()
        # Any sibling of the test file's grandparent is "outside" relative
        # to the test file's parent.
        root = Path(__file__).resolve().parent
        stored = to_storage_path(str(outside), root)
        self.assertTrue(Path(stored).is_absolute(), f"Expected absolute, got {stored!r}")

    def test_empty_string_round_trips_as_empty(self) -> None:
        self.assertEqual(to_storage_path("", Path(".").resolve()), "")
        self.assertEqual(to_storage_path("   ", Path(".").resolve()), "")


class TestResolveStoragePath(unittest.TestCase):
    """Loading: relative paths anchor to project root; absolute paths pass through."""

    def test_relative_anchors_to_project_root(self) -> None:
        root = Path(__file__).resolve().parent
        # Re-anchor a known-relative form against the same root.
        relative = "test_path_resolver.py"
        anchored = resolve_storage_path(relative, root)
        self.assertTrue(Path(anchored).is_absolute())
        # Round-trip: must point back to this very file.
        self.assertEqual(Path(anchored).resolve(), Path(__file__).resolve())

    def test_absolute_passes_through(self) -> None:
        absolute = str(Path(__file__).resolve())
        anchored = resolve_storage_path(absolute, Path("/somewhere/else").resolve())
        # The absolute path should resolve to itself, not be anchored against
        # the irrelevant project_root.
        self.assertEqual(Path(anchored).resolve(), Path(__file__).resolve())

    def test_empty_string_passes_through(self) -> None:
        self.assertEqual(resolve_storage_path("", Path(".").resolve()), "")
        self.assertEqual(resolve_storage_path("   ", Path(".").resolve()), "")

    def test_round_trip_relative_then_absolute(self) -> None:
        """Save → load on the same machine yields the same absolute path."""
        root = Path(__file__).resolve().parent
        original_abs = str(Path(__file__).resolve())
        stored = to_storage_path(original_abs, root)
        anchored = resolve_storage_path(stored, root)
        self.assertEqual(Path(anchored).resolve(), Path(original_abs).resolve())


class TestRoundTripCrossMachineSimulation(unittest.TestCase):
    """Simulate sharing a session between two machines with different roots."""

    def test_same_subtree_different_absolute_root(self) -> None:
        # Researcher A's "lab desktop": project at /machineA/study1/
        # Researcher B's "laptop":      project at /machineB/projects/study1/
        # File of interest sits at <root>/videos/v01.mp4 on each.
        relative_in_jsons = "videos/v01.mp4"

        machine_a_root = Path("/machineA/study1").resolve()
        machine_b_root = Path("/machineB/projects/study1").resolve()

        # On A's machine, the resolved path uses A's root.
        a_resolved = resolve_storage_path(relative_in_jsons, machine_a_root)
        # On B's machine, the same JSON resolves to B's root + same suffix.
        b_resolved = resolve_storage_path(relative_in_jsons, machine_b_root)

        # Both end paths share the same trailing structure.
        self.assertTrue(a_resolved.endswith(os.sep + os.path.join("videos", "v01.mp4"))
                        or a_resolved.endswith("/videos/v01.mp4"))
        self.assertTrue(b_resolved.endswith(os.sep + os.path.join("videos", "v01.mp4"))
                        or b_resolved.endswith("/videos/v01.mp4"))
        # And they differ in the prefix (machineA vs machineB roots).
        self.assertNotEqual(a_resolved, b_resolved)


class TestIsUncPath(unittest.TestCase):
    def test_posix_paths_are_not_unc(self) -> None:
        self.assertFalse(is_unc_path("/home/user/data/video.mp4"))

    def test_empty_is_not_unc(self) -> None:
        self.assertFalse(is_unc_path(""))
        self.assertFalse(is_unc_path("   "))

    def test_unc_detection_on_windows_only(self) -> None:
        # Behavior depends on os.name. On non-Windows: always False.
        # On Windows: True for "\\..." or "//..." prefixes.
        if os.name == "nt":
            self.assertTrue(is_unc_path("\\\\server\\share\\file.mp4"))
            self.assertTrue(is_unc_path("//server/share/file.mp4"))
            self.assertFalse(is_unc_path("C:\\Users\\me\\video.mp4"))
        else:
            self.assertFalse(is_unc_path("\\\\server\\share\\file.mp4"))


class TestFindMissingPaths(unittest.TestCase):
    """Filter to entries whose path doesn't exist on disk."""

    def test_existing_path_is_not_missing(self) -> None:
        # This test file definitely exists.
        candidates = {"self": str(Path(__file__).resolve())}
        self.assertEqual(find_missing_paths(candidates), [])

    def test_nonexistent_path_is_missing(self) -> None:
        candidates = {
            "self": str(Path(__file__).resolve()),
            "ghost": "/nonexistent/path/that/does/not/exist/zzz.mp4",
        }
        out = find_missing_paths(candidates)
        # Only the ghost should appear; the test file itself is found.
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0], "ghost")

    def test_empty_values_are_skipped(self) -> None:
        # Empty strings are not "missing" — they're just unset.
        candidates = {"empty": "", "blank": "   "}
        self.assertEqual(find_missing_paths(candidates), [])

    def test_relative_with_project_root(self) -> None:
        root = Path(__file__).resolve().parent
        # Existing file (this test) referenced relative to its parent dir.
        candidates = {"existing": "test_path_resolver.py"}
        out = find_missing_paths(candidates, project_root=root)
        self.assertEqual(out, [])

        candidates_missing = {"missing": "no_such_test_file.py"}
        out_missing = find_missing_paths(candidates_missing, project_root=root)
        self.assertEqual(len(out_missing), 1)
        self.assertEqual(out_missing[0][0], "missing")


if __name__ == "__main__":
    unittest.main()
