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
            "Dataset Augmentor Lab dependency missing: %s",
            exc.name or exc,
        )
    except Exception as exc:  # pragma: no cover - defensive; surfaced to GUI log
        _ui_import_error = exc
        logging.getLogger(__name__).exception("Dataset Augmentor Lab UI import failed")
    return _ui_module


def _format_import_error() -> str:
    if _ui_import_error is None:
        return "Dataset Augmentor Lab encountered an unknown import error."
    if isinstance(_ui_import_error, ModuleNotFoundError):
        missing = getattr(_ui_import_error, "name", None)
        if missing:
            return (
                "Dataset Augmentor Lab is unavailable because the optional dependency "
                f"'{missing}' is not installed."
            )
        return (
            "Dataset Augmentor Lab is unavailable because a required dependency "
            f"is not installed ({_ui_import_error})."
        )
    return f"Dataset Augmentor Lab is unavailable due to an import error: {_ui_import_error}"


class DatasetAugmentorLabPlugin:
    def __init__(self) -> None:
        self._main_app = None
        self._window: Optional[tk.Misc] = None
        self._import_error_message: Optional[str] = None

    def attach(self, main_app) -> None:
        self._main_app = main_app

    def _apply_prefill_paths(self, *, dataset_root: Optional[str], output_root: Optional[str]) -> None:
        window = self._window
        if window is None or not window.winfo_exists():
            return
        if dataset_root:
            try:
                window._dataset_root_var.set(dataset_root)
            except Exception:
                pass
        if output_root:
            try:
                window._output_root_var.set(output_root)
            except Exception:
                pass

    def open_window(
        self,
        *,
        prefill_dataset_root: Optional[str] = None,
        prefill_output_root: Optional[str] = None,
    ) -> None:
        if self._main_app is None:
            raise RuntimeError("Dataset Augmentor Lab is not attached to the main app.")
        self._open_window(
            prefill_dataset_root=prefill_dataset_root,
            prefill_output_root=prefill_output_root,
        )

    def register(self, main_app) -> None:
        self.attach(main_app)
        menu = getattr(main_app, "plugins_menu", None)
        if menu is None:
            main_app.log_message(
                "plugins_menu not available; cannot register Dataset Augmentor Lab.",
                "ERROR",
            )
            return

        if _resolve_ui_module() is None:
            self._import_error_message = _format_import_error()
            main_app.log_message(self._import_error_message, "ERROR")
            menu.add_command(
                label="Dataset Augmentor Lab (unavailable)",
                command=self._show_import_issue,
            )
            return

        menu.add_command(label="Dataset Augmentor Lab", command=self.open_window)
        main_app.log_message("Dataset Augmentor Lab plugin registered", "INFO")

    def _open_window(
        self,
        *,
        prefill_dataset_root: Optional[str] = None,
        prefill_output_root: Optional[str] = None,
    ) -> None:
        module = _resolve_ui_module()
        if module is None:
            self._show_import_issue()
            return

        window = self._window
        if window is not None and window.winfo_exists():
            self._apply_prefill_paths(
                dataset_root=prefill_dataset_root,
                output_root=prefill_output_root,
            )
            window.lift()
            try:
                window.focus_force()
            except Exception:
                pass
            return

        ui_error_type = getattr(module, "DatasetAugmentorLabUIError", RuntimeError)
        if not isinstance(ui_error_type, type) or not issubclass(ui_error_type, Exception):
            ui_error_type = RuntimeError

        window_cls = getattr(module, "DatasetAugmentorLabWindow", None)
        if window_cls is None:
            self._show_error("Dataset Augmentor Lab UI is missing DatasetAugmentorLabWindow.")
            return

        try:
            parent = getattr(self._main_app, "root", None)
            self._window = window_cls(self._main_app, parent=parent)
            self._apply_prefill_paths(
                dataset_root=prefill_dataset_root,
                output_root=prefill_output_root,
            )
            self._window.bind("<Destroy>", lambda _e: self._clear_window(), add="+")
            self._main_app.log_message("Dataset Augmentor Lab opened.", "INFO")
        except Exception as exc:  # pragma: no cover - surfaced to UI
            if isinstance(exc, ui_error_type):
                self._show_error(str(exc))
                return
            self._show_error(f"Failed to launch Dataset Augmentor Lab: {exc}")

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
        messagebox.showerror("Dataset Augmentor Lab", message, parent=parent)


def register_plugin(main_app) -> None:
    existing = getattr(main_app, "_dataset_augmentor_lab_plugin", None)
    if existing is not None:
        return
    plugin = DatasetAugmentorLabPlugin()
    plugin.register(main_app)
    setattr(main_app, "_dataset_augmentor_lab_plugin", plugin)


__all__ = ["register_plugin"]
