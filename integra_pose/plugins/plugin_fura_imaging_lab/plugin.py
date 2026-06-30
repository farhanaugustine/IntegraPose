from __future__ import annotations

import importlib
import logging
import tkinter as tk
from tkinter import messagebox
from types import ModuleType
from typing import Optional

_ui_module: Optional[ModuleType] = None
_ui_import_error: Optional[BaseException] = None


def _resolve_ui_module() -> Optional[ModuleType]:
    global _ui_module, _ui_import_error
    if _ui_module is not None:
        return _ui_module
    if _ui_import_error is not None:
        return None
    try:
        _ui_module = importlib.import_module(".ui", __package__)
    except ModuleNotFoundError as exc:
        _ui_import_error = exc
        logging.getLogger(__name__).warning(
            "Fura Imaging Lab dependency missing: %s",
            exc.name or exc,
        )
    except Exception as exc:  # pragma: no cover - surfaced to GUI log
        _ui_import_error = exc
        logging.getLogger(__name__).exception("Fura Imaging Lab UI import failed")
    return _ui_module


def _format_import_error() -> str:
    if _ui_import_error is None:
        return "Fura Imaging Lab encountered an unknown import error."
    if isinstance(_ui_import_error, ModuleNotFoundError):
        missing = getattr(_ui_import_error, "name", None)
        if missing:
            return (
                "Fura Imaging Lab is unavailable because the optional dependency "
                f"'{missing}' is not installed."
            )
        return (
            "Fura Imaging Lab is unavailable because a required dependency "
            f"is not installed ({_ui_import_error})."
        )
    return f"Fura Imaging Lab is unavailable due to an import error: {_ui_import_error}"


class FuraImagingLabPlugin:
    def __init__(self) -> None:
        self._main_app = None
        self._window: Optional[tk.Misc] = None
        self._import_error_message: Optional[str] = None

    def register(self, main_app) -> None:
        self._main_app = main_app
        menu = getattr(main_app, "plugins_menu", None)
        if menu is None:
            main_app.log_message(
                "plugins_menu not available; cannot register Fura Imaging Lab.",
                "ERROR",
            )
            return

        if _resolve_ui_module() is None:
            self._import_error_message = _format_import_error()
            main_app.log_message(self._import_error_message, "ERROR")
            menu.add_command(
                label="Fura Imaging Lab (unavailable)",
                command=self._show_import_issue,
            )
            return

        menu.add_command(label="Fura Imaging Lab", command=self._open_window)
        main_app.log_message("Fura Imaging Lab plugin registered", "INFO")

    def _open_window(self) -> None:
        module = _resolve_ui_module()
        if module is None:
            self._show_import_issue()
            return

        window = self._window
        if window is not None and window.winfo_exists():
            window.lift()
            try:
                window.focus_force()
            except Exception:
                pass
            return

        ui_error_type = getattr(module, "FuraImagingLabUIError", RuntimeError)
        if not isinstance(ui_error_type, type) or not issubclass(ui_error_type, Exception):
            ui_error_type = RuntimeError

        window_cls = getattr(module, "FuraImagingLabWindow", None)
        if window_cls is None:
            self._show_error("Fura Imaging Lab UI is missing FuraImagingLabWindow.")
            return

        try:
            parent = getattr(self._main_app, "root", None)
            self._window = window_cls(self._main_app, parent=parent)
            self._window.bind("<Destroy>", lambda _e: self._clear_window(), add="+")
            self._main_app.log_message("Fura Imaging Lab opened.", "INFO")
        except Exception as exc:  # pragma: no cover - surfaced to UI
            if isinstance(exc, ui_error_type):
                self._show_error(str(exc))
                return
            logging.getLogger(__name__).exception("Failed to open Fura Imaging Lab window")
            self._show_error(f"Failed to launch Fura Imaging Lab: {exc}")

    def _clear_window(self) -> None:
        self._window = None

    def _show_import_issue(self) -> None:
        message = self._import_error_message or _format_import_error()
        self._show_error(message)

    def _show_error(self, message: str) -> None:
        if self._main_app is not None:
            self._main_app.log_message(message, "ERROR")
            parent = getattr(self._main_app, "root", None)
        else:
            parent = None
        messagebox.showerror("Fura Imaging Lab", message, parent=parent)


def register_plugin(main_app) -> None:
    existing = getattr(main_app, "_fura_imaging_lab_plugin", None)
    if existing is not None:
        return
    plugin = FuraImagingLabPlugin()
    plugin.register(main_app)
    setattr(main_app, "_fura_imaging_lab_plugin", plugin)


__all__ = ["register_plugin"]
