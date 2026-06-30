from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QMessageBox

from .annotation_app.main_window import AnnotationMainWindow
from .annotation_app.store import AnnotationStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TandemYTC full-video annotation workspace."
    )
    parser.add_argument(
        "--project_db",
        type=Path,
        default=Path.home()
        / ".integrapose"
        / "yolo_temporal_classifier"
        / "workspace.sqlite",
        help="Path to the SQLite project workspace database.",
    )
    return parser.parse_args()


def _warn_missing_qt_runtime(window: AnnotationMainWindow) -> None:
    missing: list[str] = []
    try:
        import cv2  # noqa: F401
    except Exception:
        missing.append("opencv-python is required for video probing and import.")
    try:
        import yaml  # noqa: F401
    except Exception:
        missing.append("PyYAML is required for behavior-label YAML imports.")
    if not missing:
        return
    QMessageBox.warning(
        window,
        "Missing optional dependencies",
        "Some workspace features may be unavailable:\n\n" + "\n".join(f"- {item}" for item in missing),
    )


def main() -> int:
    args = parse_args()
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("TandemYTC - Tandem YOLO + Temporal Classifier")

    store = AnnotationStore(args.project_db)
    app.aboutToQuit.connect(store.close)
    project = store.get_or_create_default_project("TandemYTC Project")
    window = AnnotationMainWindow(store, project)
    window.show()
    QTimer.singleShot(0, lambda: _warn_missing_qt_runtime(window))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
