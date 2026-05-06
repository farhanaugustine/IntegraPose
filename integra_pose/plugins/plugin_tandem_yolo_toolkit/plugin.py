from __future__ import annotations

import importlib
import sys
import types
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
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


tandem_yolo_classifier = _load_local_module(
    "tandem_yolo_classifier",
    required_attrs=("load_config", "process_clips_in_directory", "process_and_relabel_files", "run_training_pipeline", "run_real_time_inference", "launch_clipper_gui"),
)


class TandemYoloToolkitPlugin:
    def __init__(self) -> None:
        self._main_app = None
        self._clipper = None

    def register(self, main_app) -> None:
        self._main_app = main_app
        menu = getattr(main_app, "plugins_menu", None)
        if menu is None:
            main_app.log_message(
                "plugins_menu not available; cannot register Tandem YOLO toolkit plugin.",
                "ERROR",
            )
            return

        if isinstance(menu, tk.Menu) and hasattr(menu, "add_cascade"):
            submenu = tk.Menu(menu, tearoff=0)
            submenu.add_command(
                label="Open Clip Manager",
                command=lambda: self._run_action("clip"),
            )
            submenu.add_separator()
            submenu.add_command(
                label="Process Clips (YOLO batches)",
                command=lambda: self._run_action("process"),
            )
            submenu.add_command(
                label="Prepare Dataset (Relabel)",
                command=lambda: self._run_action("prepare"),
            )
            submenu.add_command(
                label="Train LSTM Classifier",
                command=lambda: self._run_action("train"),
            )
            submenu.add_command(
                label="Run Real-time Inference",
                command=lambda: self._run_action("infer"),
            )

            menu.add_cascade(label="Tandem YOLO Toolkit", menu=submenu)
        else:
            menu.add_command(label="Tandem YOLO Toolkit", command=lambda: self._run_action("clip"))
        main_app.log_message("Tandem YOLO toolkit plugin registered", "INFO")

    def _choose_config(self) -> Optional[str]:
        parent = getattr(self._main_app, "root", None)
        config_dir = Path(__file__).resolve().parent / "tandem_yolo_classifier"
        return filedialog.askopenfilename(
            parent=parent,
            title="Select Tandem YOLO configuration",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
            initialdir=str(config_dir),
        )

    def _run_action(self, action: str) -> None:
        config_path = self._choose_config()
        if not config_path:
            return

        self._main_app.log_message(
            f"Tandem YOLO action '{action}' started with config: {config_path}",
            "INFO",
        )

        if action == "clip":
            self._launch_clipper(config_path)
            return

        def worker(path: str, cmd: str) -> None:
            try:
                config = tandem_yolo_classifier.load_config(path)
                if cmd == "process":
                    tandem_yolo_classifier.process_clips_in_directory(config)
                elif cmd == "prepare":
                    prep_cfg = config["dataset_preparation"]
                    data_cfg = config["data"]
                    tandem_yolo_classifier.process_and_relabel_files(
                        source_dir=prep_cfg["source_directory"],
                        output_dir=data_cfg["pose_directory"],
                        class_mapping=prep_cfg["class_mapping"],
                    )
                elif cmd == "train":
                    tandem_yolo_classifier.run_training_pipeline(config)
                elif cmd == "infer":
                    tandem_yolo_classifier.run_real_time_inference(config)
                else:  # pragma: no cover - defensive
                    raise ValueError(f"Unsupported action: {cmd}")
            except ModuleNotFoundError as exc:
                self._notify_error(f"Missing optional dependency: {exc}")
            except Exception as exc:  # pragma: no cover - surfaced to UI
                self._notify_error(f"Tandem YOLO action '{cmd}' failed: {exc}")
            else:
                self._notify_info(f"Tandem YOLO action '{cmd}' completed successfully.")

        threading.Thread(target=worker, args=(config_path, action), daemon=True).start()

    def _launch_clipper(self, config_path: str) -> None:
        clipper_window = self._resolve_window()
        if clipper_window is not None and clipper_window.winfo_exists():
            clipper_window.lift()
            try:
                clipper_window.focus_force()
            except Exception:
                pass
            return

        try:
            config = tandem_yolo_classifier.load_config(config_path)
            clipper = tandem_yolo_classifier.launch_clipper_gui(
                config, parent=getattr(self._main_app, "root", None)
            )
            self._clipper = clipper
            self._attach_destroy_hook(self._resolve_window())
            self._main_app.log_message("Tandem YOLO clip manager launched.", "INFO")
        except ModuleNotFoundError as exc:
            self._notify_error(f"Missing optional dependency for clip manager: {exc}")
        except Exception as exc:  # pragma: no cover - surfaced to UI
            self._notify_error(f"Failed to launch clip manager: {exc}")

    def _resolve_window(self) -> Optional[tk.Misc]:
        clipper = self._clipper
        if clipper is None:
            return None
        if isinstance(clipper, tk.Misc):
            return clipper
        return getattr(clipper, "root", None)

    def _attach_destroy_hook(self, window: Optional[tk.Misc]) -> None:
        if window is None:
            return
        try:
            window.bind("<Destroy>", lambda _event: self._clear_clipper(), add="+")
        except Exception:
            pass

    def _clear_clipper(self) -> None:
        self._clipper = None

    def _notify_error(self, message: str) -> None:
        self._main_app.log_message(message, "ERROR")
        self._show_message("Tandem YOLO Toolkit", message, error=True)

    def _notify_info(self, message: str) -> None:
        self._main_app.log_message(message, "INFO")
        self._show_message("Tandem YOLO Toolkit", message, error=False)

    def _show_message(self, title: str, message: str, *, error: bool) -> None:
        parent = getattr(self._main_app, "root", None)

        def callback() -> None:
            if error:
                messagebox.showerror(title, message, parent=parent)
            else:
                messagebox.showinfo(title, message, parent=parent)

        if parent is not None:
            try:
                parent.after(0, callback)
                return
            except Exception:
                pass
        callback()


def register_plugin(main_app) -> None:
    existing = getattr(main_app, "_tandem_yolo_toolkit_plugin", None)
    if existing is not None:
        return
    plugin = TandemYoloToolkitPlugin()
    plugin.register(main_app)
    setattr(main_app, "_tandem_yolo_toolkit_plugin", plugin)


__all__ = ["register_plugin"]
