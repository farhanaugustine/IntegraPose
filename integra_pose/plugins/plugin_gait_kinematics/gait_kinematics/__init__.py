"""Gait kinematics analysis module for the IntegraPose plugin suite."""
from __future__ import annotations

import tkinter as tk
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .gui_launcher import AnalysisGUI


def launch(parent: Optional[tk.Misc]) -> AnalysisGUI:
    """Launch the gait & kinematic analysis window attached to *parent*."""
    from .gui_launcher import AnalysisGUI

    owns_parent = False
    if parent is None:
        parent = tk.Tk()
        parent.withdraw()
        owns_parent = True
        try:
            from integra_pose.gui.theme import apply_global_ttk_theme

            apply_global_ttk_theme(parent)
        except Exception:
            pass
    window = AnalysisGUI(parent)
    window._owns_parent = owns_parent
    if parent is not None:
        try:
            window.transient(parent)
        except Exception:
            pass
    window.focus_set()
    window.lift()
    return window


def __getattr__(name: str):  # pragma: no cover
    if name == "AnalysisGUI":
        from .gui_launcher import AnalysisGUI

        return AnalysisGUI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["launch", "AnalysisGUI"]
