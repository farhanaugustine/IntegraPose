"""HMM-VAE segmentation module for the IntegraPose plugin suite."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import tkinter as tk
from typing import Optional

if TYPE_CHECKING:  # pragma: no cover
    from .main import BehaviorAnalysisApp as BehaviorAnalysisApp


def launch(parent: Optional[tk.Misc]) -> "BehaviorAnalysisApp":
    """Launch the segmentation workflow and return the controller instance."""
    from . import main as _main

    _main.configure_toolkit_logging()
    BehaviorAnalysisApp = _main.BehaviorAnalysisApp

    owns_parent = False
    if parent is None:
        parent = tk.Tk()
        parent.withdraw()
        owns_parent = True
    window = tk.Toplevel(parent)
    app = BehaviorAnalysisApp(window)
    app._owns_parent = owns_parent

    def _on_close() -> None:
        window.destroy()
        if owns_parent and isinstance(parent, tk.Tk):
            parent.destroy()

    window.protocol("WM_DELETE_WINDOW", _on_close)
    window.focus_set()
    window.lift()
    return app


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name == "BehaviorAnalysisApp":
        from . import main as _main

        _main.configure_toolkit_logging()
        return _main.BehaviorAnalysisApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["launch", "BehaviorAnalysisApp"]
