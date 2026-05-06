"""IntegraPose package public API and entry points."""
from __future__ import annotations

from importlib import import_module
from typing import Any, List, Tuple

try:  # pragma: no cover - metadata lookup only works when installed
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version  # type: ignore

__all__: List[str] = [
    "YoloApp",
    "launch",
    "main",
    "__version__",
]

try:  # pragma: no cover
    __version__ = version("IntegraPose")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.dev0"


def _load_gui() -> Tuple[Any, Any]:
    gui_module = import_module("integra_pose.main_gui_app")
    return gui_module.YoloApp, gui_module.main


def launch() -> None:
    """Launch the IntegraPose GUI application."""
    _, run_gui = _load_gui()
    run_gui()


def main() -> None:
    """Entry point used by console scripts."""
    _, run_gui = _load_gui()
    run_gui()


_alias_map = {
    "main_gui_app": "integra_pose.main_gui_app",
    "keypoint_annotator_module": "integra_pose.keypoint_annotator_module",
    "gui": "integra_pose.gui",
    "plugins": "integra_pose.plugins",
    "utils": "integra_pose.utils",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - simple proxy
    if name == "YoloApp":
        yolo_app, _ = _load_gui()
        return yolo_app
    if name in _alias_map:
        module = import_module(_alias_map[name])
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:  # pragma: no cover - helper for dir()
    return sorted(set(__all__ + list(_alias_map)))
