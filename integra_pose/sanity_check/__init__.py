"""Sanity-check package — verifies an IntegraPose install actually works.

Public entry points:

    from integra_pose.sanity_check.runner import run_sanity_check
    from integra_pose.sanity_check.ui import open_sanity_check_dialog

The package deliberately bundles **no** pre-trained models (avoids the
Ultralytics AGPL-3.0 license trap for downstream commercial users) and
**no** sample videos. Instead it ships a tiny set of synthetic YOLO
label files plus expected-output references, then exercises the
non-ML parts of the pipeline:

    1. Imports / runtime version inventory
    2. GUI smoke test (Tk root + image library)
    3. YOLO label parsing
    4. Bout analyzer end-to-end
    5. Atomic file IO round-trip

Each stage is independent — one failure does not tank the others, so
researchers see exactly which stage broke and why.
"""

from __future__ import annotations

__all__ = ["run_sanity_check", "open_sanity_check_dialog"]


def __getattr__(name: str):
    """Lazy re-exports so importing the package never triggers Tk."""
    if name == "run_sanity_check":
        from integra_pose.sanity_check.runner import run_sanity_check
        return run_sanity_check
    if name == "open_sanity_check_dialog":
        from integra_pose.sanity_check.ui import open_sanity_check_dialog
        return open_sanity_check_dialog
    raise AttributeError(f"module 'integra_pose.sanity_check' has no attribute {name!r}")
