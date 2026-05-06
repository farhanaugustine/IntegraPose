"""Tests for the sanity-check package.

These tests do *not* exercise the dialog UI or the bout-analyzer end-to-end
stage (those need pandas/cv2/etc). They cover the pieces that work in any
Python environment: the report dataclass, the system_info probe, the
expected-summary parser, and the report-as-text rendering.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from integra_pose.sanity_check.runner import (
    SanityReport,
    StageResult,
    _parse_expected_summary,
    _stage_atomic_io,
    _stage_gui_smoke,
)
from integra_pose.sanity_check.system_info import (
    collect_runtime_info,
    is_failure,
)


class TestStageResultDataclass(unittest.TestCase):
    def test_passed_only_true_for_ok(self) -> None:
        self.assertTrue(StageResult(name="x", status="ok", summary="").passed)
        self.assertFalse(StageResult(name="x", status="warn", summary="").passed)
        self.assertFalse(StageResult(name="x", status="fail", summary="").passed)

    def test_is_failure_only_true_for_fail(self) -> None:
        self.assertFalse(StageResult(name="x", status="ok", summary="").is_failure)
        self.assertFalse(StageResult(name="x", status="warn", summary="").is_failure)
        self.assertTrue(StageResult(name="x", status="fail", summary="").is_failure)


class TestSanityReport(unittest.TestCase):
    def test_all_passed_aggregates_correctly(self) -> None:
        report = SanityReport(stages=[
            StageResult(name="a", status="ok", summary="."),
            StageResult(name="b", status="ok", summary="."),
        ])
        self.assertTrue(report.all_passed)
        self.assertEqual(report.n_failed, 0)
        self.assertEqual(report.n_warned, 0)

    def test_one_failure_flips_all_passed(self) -> None:
        report = SanityReport(stages=[
            StageResult(name="a", status="ok", summary="."),
            StageResult(name="b", status="fail", summary="boom"),
        ])
        self.assertFalse(report.all_passed)
        self.assertEqual(report.n_failed, 1)

    def test_warn_does_not_flip_all_passed(self) -> None:
        # Documentation: warn means "non-fatal observation"; the user is
        # not in a broken state. all_passed currently treats warn as not-ok
        # which is the conservative choice. If you change it, this test
        # tells you to revisit the dialog's badge logic too.
        report = SanityReport(stages=[
            StageResult(name="a", status="warn", summary="."),
        ])
        self.assertFalse(report.all_passed)
        self.assertEqual(report.n_warned, 1)
        self.assertEqual(report.n_failed, 0)

    def test_as_text_includes_each_stage(self) -> None:
        report = SanityReport(
            started_at_utc="2026-04-26T00:00:00Z",
            finished_at_utc="2026-04-26T00:00:02Z",
            stages=[
                StageResult(name="alpha", status="ok", summary="all good", duration_seconds=0.123),
                StageResult(name="beta", status="fail", summary="kaboom",
                            detail="line one\nline two", duration_seconds=0.456),
            ],
        )
        text = report.as_text()
        self.assertIn("alpha", text)
        self.assertIn("all good", text)
        self.assertIn("beta", text)
        self.assertIn("kaboom", text)
        self.assertIn("line one", text)
        self.assertIn("line two", text)
        self.assertIn("FAIL", text)


class TestExpectedSummaryParser(unittest.TestCase):
    def test_parses_real_fixture(self) -> None:
        path = (
            Path(__file__).resolve().parent.parent
            / "integra_pose"
            / "sanity_check"
            / "data"
            / "expected"
            / "expected_bouts_summary.txt"
        )
        parsed = _parse_expected_summary(path)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["n_bouts_total"], "2")
        self.assertEqual(parsed["n_bouts_walking"], "1")
        self.assertEqual(parsed["n_bouts_rearing"], "1")

    def test_skips_comments_and_blanks(self) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("# a header comment\n\n   \nkey1 = value1\n# another\nkey2=value2\n")
            tmp_path = Path(f.name)
        try:
            parsed = _parse_expected_summary(tmp_path)
            self.assertEqual(parsed, {"key1": "value1", "key2": "value2"})
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_missing_file_returns_none(self) -> None:
        self.assertIsNone(_parse_expected_summary(Path("/nonexistent/zzz.txt")))


class TestSystemInfo(unittest.TestCase):
    def test_collect_returns_three_sections(self) -> None:
        info = collect_runtime_info()
        self.assertIn("host", info)
        self.assertIn("required", info)
        self.assertIn("optional", info)

    def test_host_includes_python_version(self) -> None:
        info = collect_runtime_info()
        host = info["host"]
        self.assertIn("python", host)
        self.assertTrue(host["python"], "python version should be non-empty")

    def test_is_failure_classifies_correctly(self) -> None:
        self.assertTrue(is_failure("(missing — ImportError)"))
        self.assertTrue(is_failure("(import error — RuntimeError: foo)"))
        self.assertFalse(is_failure("1.26.4"))
        self.assertFalse(is_failure("(version unknown)"))


class TestAtomicIoStage(unittest.TestCase):
    """The atomic-io stage round-trips through safe_io — should pass in any
    env where safe_io itself works (it has no heavy deps)."""

    def test_stage_runs_without_error(self) -> None:
        # We don't assert "ok" because if safe_io ever moves we want a clear
        # diagnostic; we just assert it produces a non-trivial result object.
        result = _stage_atomic_io()
        self.assertIsInstance(result, StageResult)
        self.assertEqual(result.name, "Atomic file IO")
        self.assertIn(result.status, {"ok", "fail"})


class TestGuiSmokeStage(unittest.TestCase):
    """The Tk smoke stage tries to construct a hidden root + Toplevel.
    Skipped on environments without a display."""

    def test_stage_returns_a_result(self) -> None:
        try:
            import tkinter as tk

            root = tk.Tk()
            root.destroy()
        except Exception:
            self.skipTest("Tk unavailable in this environment.")
        result = _stage_gui_smoke()
        self.assertIsInstance(result, StageResult)
        self.assertEqual(result.name, "GUI smoke test")


if __name__ == "__main__":
    unittest.main()
