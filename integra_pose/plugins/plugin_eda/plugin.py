from __future__ import annotations

import importlib
import logging
import tkinter as tk
from tkinter import Menu, messagebox
from types import ModuleType
from typing import Optional

_eda_gui_module: Optional[ModuleType] = None
_eda_import_error: Optional[BaseException] = None


def _resolve_eda_gui() -> Optional[ModuleType]:
    global _eda_gui_module, _eda_import_error
    if _eda_gui_module is not None:
        return _eda_gui_module
    if _eda_import_error is not None:
        return None
    try:
        _eda_gui_module = importlib.import_module('.ui.app', __package__)
    except ModuleNotFoundError as exc:
        _eda_import_error = exc
        logging.getLogger(__name__).warning('Pose EDA plugin dependency missing: %s', exc.name or exc)
    except Exception as exc:  # pragma: no cover - defensive; surfaced to GUI log
        _eda_import_error = exc
        logging.getLogger(__name__).exception('Pose EDA plugin failed to import')
    return _eda_gui_module


def _format_import_error() -> str:
    if _eda_import_error is None:
        return 'EDA plugin encountered an unknown import error.'
    if isinstance(_eda_import_error, ModuleNotFoundError):
        missing = getattr(_eda_import_error, 'name', None)
        if missing:
            return f"EDA plugin is unavailable because the optional dependency '{missing}' is not installed."
        return f"EDA plugin is unavailable because a required dependency is not installed ({_eda_import_error})."
    return f'EDA plugin is unavailable due to an import error: {_eda_import_error}'


class IntegraPosePlugin:
    def __init__(self) -> None:
        self.main_app = None
        self._import_error_message: Optional[str] = None
        self._window: Optional[tk.Misc] = None

    def register(self, main_app) -> None:
        self.main_app = main_app
        self.main_app.log_message('--- Inside Plugin Register Method ---', 'INFO')

        menu: Optional[Menu] = getattr(self.main_app, 'plugins_menu', None)
        if not menu:
            self.main_app.log_message('plugins_menu not found.', 'WARNING')
            return

        module = _resolve_eda_gui()
        if module is None:
            self._import_error_message = _format_import_error()
            self.main_app.log_message(self._import_error_message, 'ERROR')
            menu.add_command(label='Launch EDA Tool (unavailable)', command=self._show_import_issue)
            return

        self.main_app.log_message('plugins_menu found, adding command.', 'INFO')
        menu.add_command(label='Launch EDA Tool', command=self._launch_eda_tool)

    def _show_import_issue(self) -> None:
        message = self._import_error_message or _format_import_error()
        if self.main_app:
            self.main_app.log_message(message, 'ERROR')
        messagebox.showerror('Pose EDA Plugin', message)

    def _launch_eda_tool(self) -> None:
        module = _resolve_eda_gui()
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

        root = getattr(self.main_app, 'root', None)
        parent = root if isinstance(root, (tk.Tk, tk.Toplevel)) else None
        eda_window = tk.Toplevel(parent) if parent else tk.Toplevel()
        try:
            self._window = eda_window
            self._window.bind("<Destroy>", lambda _e: self._clear_window(), add="+")
            module.PoseEDAApp(eda_window, host_app=self.main_app)
        except Exception as exc:  # pragma: no cover - surfaced to UI
            self._clear_window()
            message = f"Failed to launch EDA Tool: {exc}"
            logging.getLogger(__name__).exception("PoseEDAApp failed to initialise")
            if self.main_app:
                self.main_app.log_message(message, "ERROR")
            messagebox.showerror("Pose EDA Plugin", message)

    def _clear_window(self) -> None:
        self._window = None


def register_plugin(main_app) -> None:
    """Alternative registration entry point for compatibility."""
    existing = getattr(main_app, "_eda_plugin", None)
    if existing is not None:
        return
    plugin = IntegraPosePlugin()
    plugin.register(main_app)
    setattr(main_app, "_eda_plugin", plugin)
