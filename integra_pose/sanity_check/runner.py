"""Sanity-check runner — five independent stages, each in isolation.

Public API:

    from integra_pose.sanity_check.runner import run_sanity_check

    report = run_sanity_check()
    if report.all_passed:
        ...

The runner deliberately catches any exception inside each stage so a
single broken dependency doesn't prevent later stages from running and
reporting their results. The end-of-run report includes per-stage
status, runtime, and (for the runtime-info stage) a structured
inventory of every dependency's version.

Design constraints
------------------

* No video files, no ML models — only synthetic YOLO label fixtures
  (see ``data/README.md`` for why).
* Each stage runs in <0.5 s so the whole check fits in <2 s on any
  reasonable laptop.
* Failures are *informative*: every error message tells the researcher
  what to do (install X, set Y, file an issue with this report).
"""

from __future__ import annotations

import io
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class StageResult:
    """One sanity-check stage's outcome."""

    name: str
    status: str  # "ok" | "warn" | "fail"
    summary: str
    detail: str = ""
    duration_seconds: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "ok"

    @property
    def is_failure(self) -> bool:
        return self.status == "fail"


@dataclass
class SanityReport:
    """End-to-end sanity check report."""

    stages: List[StageResult] = field(default_factory=list)
    started_at_utc: str = ""
    finished_at_utc: str = ""

    @property
    def all_passed(self) -> bool:
        return all(s.passed for s in self.stages)

    @property
    def n_failed(self) -> int:
        return sum(1 for s in self.stages if s.is_failure)

    @property
    def n_warned(self) -> int:
        return sum(1 for s in self.stages if s.status == "warn")

    def as_text(self) -> str:
        """Plain-text rendering suitable for clipboard / bug reports."""
        out = io.StringIO()
        out.write(f"IntegraPose Sanity Check\n")
        out.write(f"  started:  {self.started_at_utc}\n")
        out.write(f"  finished: {self.finished_at_utc}\n")
        out.write(
            f"  result:   {'PASS' if self.all_passed else 'FAIL'} "
            f"({self.n_failed} failed, {self.n_warned} warned, "
            f"{len(self.stages)} total)\n"
        )
        out.write("\n")
        for stage in self.stages:
            badge = {"ok": "[ OK ]", "warn": "[WARN]", "fail": "[FAIL]"}.get(stage.status, "[ ?? ]")
            out.write(f"{badge} {stage.name}  ({stage.duration_seconds:.3f}s)\n")
            out.write(f"       {stage.summary}\n")
            if stage.detail:
                for line in stage.detail.splitlines():
                    out.write(f"         {line}\n")
        return out.getvalue()


# ---------------------------------------------------------------------------
# Stage 1: Runtime version inventory
# ---------------------------------------------------------------------------


def _stage_runtime_inventory() -> StageResult:
    from integra_pose.sanity_check.system_info import collect_runtime_info, is_failure

    info = collect_runtime_info()
    required = info.get("required", {})
    missing = [name for name, ver in required.items() if is_failure(ver)]

    detail_lines = ["Host:"]
    for k, v in info.get("host", {}).items():
        detail_lines.append(f"  {k}: {v}")
    detail_lines.append("Required modules:")
    for name, ver in required.items():
        detail_lines.append(f"  {name}: {ver}")
    detail_lines.append("Optional modules:")
    for name, ver in info.get("optional", {}).items():
        detail_lines.append(f"  {name}: {ver}")

    if missing:
        return StageResult(
            name="Runtime inventory",
            status="fail",
            summary=f"Missing required: {', '.join(missing)}",
            detail="\n".join(detail_lines),
            extra=info,
        )
    return StageResult(
        name="Runtime inventory",
        status="ok",
        summary=f"All {len(required)} required modules import cleanly.",
        detail="\n".join(detail_lines),
        extra=info,
    )


# ---------------------------------------------------------------------------
# Stage 2: GUI smoke test
# ---------------------------------------------------------------------------


def _stage_gui_smoke() -> StageResult:
    """Open + close a hidden Tk root + a Toplevel.

    Catches: Tk not built into this Python, display server unavailable
    (``$DISPLAY`` unset on headless Linux), Pillow missing for image
    rendering, etc. Tk uses its own private dispatch loop so we don't
    bother running update().
    """
    import tkinter as tk  # PLC0415 — local so an absent Tk fails this stage only

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        # Toplevel exercises a different code path than the root.
        top = tk.Toplevel(root)
        top.withdraw()
        # Pillow is needed everywhere images appear.
        from PIL import Image, ImageTk  # noqa: F401  (imports = the test)

        top.destroy()
        return StageResult(
            name="GUI smoke test",
            status="ok",
            summary="Tk root + Toplevel + Pillow all initialise.",
        )
    except Exception as exc:
        return StageResult(
            name="GUI smoke test",
            status="fail",
            summary=f"{exc.__class__.__name__}: {exc}",
            detail=(
                "Common fixes: (a) on Linux, ensure tkinter is installed "
                "(apt-get install python3-tk); (b) on headless servers, "
                "set $DISPLAY or run with a virtual framebuffer (xvfb); "
                "(c) ensure Pillow is installed (pip install pillow)."
            ),
        )
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Stage 3: YOLO label parsing
# ---------------------------------------------------------------------------


def _stage_yolo_parsing() -> StageResult:
    """Read the synthetic .txt label files and check their detection counts."""
    labels_dir = _data_root() / "synthetic_labels"
    if not labels_dir.exists():
        return StageResult(
            name="YOLO label parsing",
            status="fail",
            summary="Synthetic label folder missing from package.",
            detail=f"Expected: {labels_dir}",
        )
    files = sorted(labels_dir.glob("frame_*.txt"))
    if len(files) != 10:
        return StageResult(
            name="YOLO label parsing",
            status="fail",
            summary=f"Expected 10 label files, found {len(files)}.",
        )
    walking = 0
    rearing = 0
    for path in files:
        try:
            line = path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            return StageResult(
                name="YOLO label parsing",
                status="fail",
                summary=f"Could not read {path.name}: {exc}",
            )
        if not line:
            return StageResult(
                name="YOLO label parsing",
                status="fail",
                summary=f"{path.name} is empty.",
            )
        parts = line.split()
        if len(parts) < 5:
            return StageResult(
                name="YOLO label parsing",
                status="fail",
                summary=f"{path.name}: expected ≥5 columns, got {len(parts)}.",
            )
        try:
            class_id = int(parts[0])
        except ValueError:
            return StageResult(
                name="YOLO label parsing",
                status="fail",
                summary=f"{path.name}: column 0 is not an int.",
            )
        if class_id == 0:
            walking += 1
        elif class_id == 1:
            rearing += 1
    if walking != 5 or rearing != 5:
        return StageResult(
            name="YOLO label parsing",
            status="fail",
            summary=f"Class counts wrong: walking={walking}, rearing={rearing}; expected 5 each.",
        )
    return StageResult(
        name="YOLO label parsing",
        status="ok",
        summary=f"Parsed {len(files)} files: 5 walking + 5 rearing detections.",
    )


# ---------------------------------------------------------------------------
# Stage 4: Bout analyzer end-to-end
# ---------------------------------------------------------------------------


def _stage_bout_analyzer() -> StageResult:
    try:
        from integra_pose.utils.bout_analyzer import (  # noqa: PLC0415
            analyze_bouts,
            load_and_preprocess_data,
        )
    except Exception as exc:
        return StageResult(
            name="Bout analyzer end-to-end",
            status="fail",
            summary=f"Could not import bout_analyzer: {exc.__class__.__name__}: {exc}",
            detail=traceback.format_exc(),
        )

    labels_dir = _data_root() / "synthetic_labels"
    yaml_path = _data_root() / "synthetic_dataset.yaml"
    expected_path = _data_root() / "expected" / "expected_bouts_summary.txt"

    expected = _parse_expected_summary(expected_path)
    if expected is None:
        return StageResult(
            name="Bout analyzer end-to-end",
            status="fail",
            summary=f"Could not load expected summary: {expected_path}",
        )

    try:
        per_frame_df, _roi_polygons, class_names = load_and_preprocess_data(
            yaml_path=str(yaml_path),
            yolo_txt_folder=str(labels_dir),
            single_animal_mode=True,
            class_names={0: "walking", 1: "rearing"},
        )
    except Exception as exc:
        return StageResult(
            name="Bout analyzer end-to-end",
            status="fail",
            summary=f"load_and_preprocess_data failed: {exc.__class__.__name__}: {exc}",
            detail=traceback.format_exc(),
        )

    if per_frame_df is None or per_frame_df.empty:
        return StageResult(
            name="Bout analyzer end-to-end",
            status="fail",
            summary="Preprocessed DataFrame is empty.",
        )
    n_frames = int(per_frame_df["frame"].nunique()) if "frame" in per_frame_df.columns else len(per_frame_df)
    if n_frames != int(expected.get("n_frames_total", 10)):
        return StageResult(
            name="Bout analyzer end-to-end",
            status="warn",
            summary=f"Preprocessed {n_frames} frames; expected {expected.get('n_frames_total', 10)}.",
        )

    try:
        detailed_bouts, _summary = analyze_bouts(
            per_frame_df=per_frame_df,
            class_names=class_names,
            max_gap_frames=2,
            min_bout_frames=2,
            fps=30.0,
        )
    except Exception as exc:
        return StageResult(
            name="Bout analyzer end-to-end",
            status="fail",
            summary=f"analyze_bouts raised {exc.__class__.__name__}: {exc}",
            detail=traceback.format_exc(),
        )

    n_bouts = len(detailed_bouts) if detailed_bouts is not None else 0
    expected_n_bouts = int(expected.get("n_bouts_total", 2))
    if n_bouts != expected_n_bouts:
        return StageResult(
            name="Bout analyzer end-to-end",
            status="fail",
            summary=f"Bout count mismatch: got {n_bouts}, expected {expected_n_bouts}.",
            detail=(
                f"Got {n_bouts} bouts from the synthetic batch. "
                f"This usually means the bout-detection logic changed and "
                f"the expected_bouts_summary.txt fixture needs an update."
            ),
        )

    return StageResult(
        name="Bout analyzer end-to-end",
        status="ok",
        summary=f"Detected {n_bouts} bouts from {n_frames} synthetic frames (matches expected).",
    )


# ---------------------------------------------------------------------------
# Stage 5: Atomic file IO round-trip
# ---------------------------------------------------------------------------


def _stage_atomic_io() -> StageResult:
    try:
        from integra_pose.utils.safe_io import (  # noqa: PLC0415
            safe_read_json,
            safe_read_text,
            safe_write_json,
            safe_write_text,
        )
    except Exception as exc:
        return StageResult(
            name="Atomic file IO",
            status="fail",
            summary=f"Could not import safe_io: {exc}",
            detail=traceback.format_exc(),
        )

    import tempfile  # noqa: PLC0415

    payload = {"sanity": True, "items": [1, 2, 3], "unicode": "Ångström — 30°C"}
    text = "Sanity check round-trip\nLine 2 with non-ASCII: ✓\n"

    with tempfile.TemporaryDirectory(prefix="integrapose_sanity_") as tmpdir:
        tmp = Path(tmpdir)
        json_path = tmp / "sanity_payload.json"
        text_path = tmp / "sanity_text.txt"
        try:
            safe_write_json(json_path, payload, indent=2)
            safe_write_text(text_path, text)
            loaded_json = safe_read_json(json_path)
            loaded_text = safe_read_text(text_path)
        except Exception as exc:
            return StageResult(
                name="Atomic file IO",
                status="fail",
                summary=f"safe_io round-trip raised {exc.__class__.__name__}: {exc}",
                detail=traceback.format_exc(),
            )
        if loaded_json != payload:
            return StageResult(
                name="Atomic file IO",
                status="fail",
                summary="JSON round-trip mismatch.",
                detail=f"Wrote: {payload!r}\nGot:   {loaded_json!r}",
            )
        if loaded_text != text:
            return StageResult(
                name="Atomic file IO",
                status="fail",
                summary="Text round-trip mismatch.",
                detail=f"Wrote: {text!r}\nGot:   {loaded_text!r}",
            )
    return StageResult(
        name="Atomic file IO",
        status="ok",
        summary="JSON + text round-trip clean (UTF-8, atomic write).",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _data_root() -> Path:
    return Path(__file__).parent / "data"


def _parse_expected_summary(path: Path) -> Optional[Dict[str, str]]:
    """Loose key=value parser for the expected-output text file.

    Lines starting with ``#`` and blank lines are ignored. Values are
    returned as strings; callers cast as needed. Returns ``None`` on read
    failure.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None
    out: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------


_DEFAULT_STAGES: Tuple[Tuple[str, Callable[[], StageResult]], ...] = (
    ("Runtime inventory", _stage_runtime_inventory),
    ("GUI smoke test", _stage_gui_smoke),
    ("YOLO label parsing", _stage_yolo_parsing),
    ("Bout analyzer end-to-end", _stage_bout_analyzer),
    ("Atomic file IO", _stage_atomic_io),
)


def run_sanity_check(
    *,
    progress_callback: Optional[Callable[[str, str], None]] = None,
) -> SanityReport:
    """Run every stage in sequence; collect a :class:`SanityReport`.

    ``progress_callback(stage_name, status)`` is invoked once per stage:
    ``status`` is ``"running"`` before, then the final status string
    after. Used by the UI to update its live status strip.
    """
    report = SanityReport(started_at_utc=_utc_now_iso())
    for label, fn in _DEFAULT_STAGES:
        if progress_callback is not None:
            try:
                progress_callback(label, "running")
            except Exception:
                pass
        t0 = time.perf_counter()
        try:
            result = fn()
        except Exception as exc:  # pragma: no cover — every stage already wraps
            result = StageResult(
                name=label,
                status="fail",
                summary=f"Unexpected stage error: {exc.__class__.__name__}: {exc}",
                detail=traceback.format_exc(),
            )
        result.duration_seconds = time.perf_counter() - t0
        report.stages.append(result)
        if progress_callback is not None:
            try:
                progress_callback(label, result.status)
            except Exception:
                pass
    report.finished_at_utc = _utc_now_iso()
    return report


__all__ = [
    "StageResult",
    "SanityReport",
    "run_sanity_check",
]
