from __future__ import annotations

import importlib
import sys
import types
import tkinter as tk
from tkinter import messagebox
from typing import Optional


def _load_local_module(name: str, *, required_attrs: tuple[str, ...] = ()):
    package = __package__ or __name__.rpartition(".")[0]
    fullname = f"{package}.{name}"

    existing = sys.modules.get(fullname)
    if existing is not None:
        return existing

    try:
        module = importlib.import_module(fullname)
    except ModuleNotFoundError as exc:
        placeholder = types.ModuleType(fullname)
        missing = exc.name or "required dependency"
        missing_message = (
            f"Optional dependency '{missing}' is required for '{fullname}' but is not installed."
        )

        def _missing(*_args, _message=missing_message, _exc=exc, **_kwargs):
            raise ModuleNotFoundError(_message) from _exc

        def _on_missing(attr_name, _module_name=fullname):
            raise AttributeError(f"module {_module_name!r} has no attribute {attr_name!r}")

        placeholder.__dict__["__getattr__"] = _on_missing
        for attr in required_attrs:
            placeholder.__dict__[attr] = _missing
        sys.modules.setdefault(fullname, placeholder)
        return placeholder
    else:
        for attr in required_attrs:
            if not hasattr(module, attr):
                raise AttributeError(f"{fullname} missing expected attribute '{attr}'")
        return module


gait_kinematics = _load_local_module("gait_kinematics", required_attrs=("launch",))


class GaitKinematicsPlugin:
    def __init__(self) -> None:
        self._main_app = None
        self._window: Optional[tk.Misc] = None

    def register(self, main_app) -> None:
        self._main_app = main_app
        menu = getattr(main_app, "plugins_menu", None)
        if menu is None:
            main_app.log_message(
                "plugins_menu not available; cannot register Gait & Kinematic Dashboard plugin.",
                "ERROR",
            )
            return
        menu.add_command(label="Gait & Kinematic Dashboard", command=self._open_dashboard)
        main_app.log_message("Gait & Kinematic Dashboard plugin registered", "INFO")

    def _open_dashboard(self) -> None:
        window = self._window
        if window is not None and window.winfo_exists():
            window.lift()
            try:
                window.focus_force()
            except Exception:
                pass
            return

        try:
            parent = getattr(self._main_app, "root", None)
            window = gait_kinematics.launch(parent)
            self._window = self._extract_window(window)
            self._attach_destroy_hook(self._window)
            self._main_app.log_message("Opened Gait & Kinematic Dashboard.", "INFO")
        except ModuleNotFoundError as exc:
            self._show_error(f"Missing optional dependency for gait dashboard: {exc}")
        except Exception as exc:  # pragma: no cover - surfaced to UI log
            self._show_error(f"Failed to launch gait dashboard: {exc}")

    def _extract_window(self, tool) -> Optional[tk.Misc]:
        if tool is None:
            return None
        if isinstance(tool, tk.Misc):
            return tool
        return getattr(tool, "root", None)

    def _attach_destroy_hook(self, window: Optional[tk.Misc]) -> None:
        if window is None:
            return
        try:
            window.bind("<Destroy>", lambda _event: self._clear_window(), add="+")
        except Exception:
            pass

    def _clear_window(self) -> None:
        self._window = None

    def _show_error(self, message: str) -> None:
        if self._main_app is not None:
            self._main_app.log_message(message, "ERROR")
            parent = getattr(self._main_app, "root", None)
        else:
            parent = None
        messagebox.showerror("Gait & Kinematic Dashboard", message, parent=parent)


def register_plugin(main_app) -> None:
    plugin = GaitKinematicsPlugin()
    plugin.register(main_app)


__all__ = ["register_plugin"]
