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
            "Assisted Pose Curation dependency missing: %s",
            exc.name or exc,
        )
    except Exception as exc:  # pragma: no cover - surfaced to GUI log
        _ui_import_error = exc
        logging.getLogger(__name__).exception("Assisted Pose Curation UI import failed")
    return _ui_module


def _format_import_error() -> str:
    if _ui_import_error is None:
        return "Assisted Pose Curation encountered an unknown import error."
    if isinstance(_ui_import_error, ModuleNotFoundError):
        missing = getattr(_ui_import_error, "name", None)
        if missing:
            return (
                "Assisted Pose Curation is unavailable because the optional dependency "
                f"'{missing}' is not installed."
            )
        return (
            "Assisted Pose Curation is unavailable because a required dependency "
            f"is not installed ({_ui_import_error})."
        )
    return f"Assisted Pose Curation is unavailable due to an import error: {_ui_import_error}"


class AssistedPoseCurationPlugin:
    def __init__(self) -> None:
        self._main_app = None
        self._window: Optional[tk.Misc] = None
        self._import_error_message: Optional[str] = None

    def attach(self, main_app) -> None:
        self._main_app = main_app

    def _apply_prefill(
        self,
        *,
        project_root: Optional[str] = None,
        image_dir: Optional[str] = None,
        label_dir: Optional[str] = None,
        video_path: Optional[str] = None,
        model_path: Optional[str] = None,
        dataset_yaml_path: Optional[str] = None,
        keypoint_names: Optional[list[str]] = None,
        skeleton_edges: Optional[list[tuple[int, int]]] = None,
        class_name: Optional[str] = None,
        class_names: Optional[list[str]] = None,
        auto_load_images: bool = False,
    ) -> None:
        window = self._window
        if window is None or not window.winfo_exists():
            return
        apply_prefill = getattr(window, "apply_prefill", None)
        if not callable(apply_prefill):
            return
        apply_prefill(
            project_root=project_root,
            image_dir=image_dir,
            label_dir=label_dir,
            video_path=video_path,
            model_path=model_path,
            dataset_yaml_path=dataset_yaml_path,
            keypoint_names=keypoint_names,
            skeleton_edges=skeleton_edges,
            class_name=class_name,
            class_names=class_names,
            auto_load_images=auto_load_images,
        )

    def open_window(
        self,
        *,
        prefill_project_root: Optional[str] = None,
        prefill_image_dir: Optional[str] = None,
        prefill_label_dir: Optional[str] = None,
        prefill_video_path: Optional[str] = None,
        prefill_model_path: Optional[str] = None,
        prefill_dataset_yaml_path: Optional[str] = None,
        prefill_keypoint_names: Optional[list[str]] = None,
        prefill_skeleton_edges: Optional[list[tuple[int, int]]] = None,
        prefill_class_name: Optional[str] = None,
        prefill_class_names: Optional[list[str]] = None,
        auto_load_images: bool = False,
    ) -> None:
        self._open_window(
            prefill_project_root=prefill_project_root,
            prefill_image_dir=prefill_image_dir,
            prefill_label_dir=prefill_label_dir,
            prefill_video_path=prefill_video_path,
            prefill_model_path=prefill_model_path,
            prefill_dataset_yaml_path=prefill_dataset_yaml_path,
            prefill_keypoint_names=prefill_keypoint_names,
            prefill_skeleton_edges=prefill_skeleton_edges,
            prefill_class_name=prefill_class_name,
            prefill_class_names=prefill_class_names,
            auto_load_images=auto_load_images,
        )

    def register(self, main_app) -> None:
        self.attach(main_app)
        menu = getattr(main_app, "plugins_menu", None)
        if menu is None:
            main_app.log_message(
                "plugins_menu not available; cannot register Assisted Pose Curation.",
                "ERROR",
            )
            return

        if _resolve_ui_module() is None:
            self._import_error_message = _format_import_error()
            main_app.log_message(self._import_error_message, "ERROR")
            menu.add_command(
                label="Assisted Pose Curation (unavailable)",
                command=self._show_import_issue,
            )
            return

        menu.add_command(label="Assisted Pose Curation", command=self.open_window)
        main_app.log_message("Assisted Pose Curation plugin registered", "INFO")

    def _open_window(
        self,
        *,
        prefill_project_root: Optional[str] = None,
        prefill_image_dir: Optional[str] = None,
        prefill_label_dir: Optional[str] = None,
        prefill_video_path: Optional[str] = None,
        prefill_model_path: Optional[str] = None,
        prefill_dataset_yaml_path: Optional[str] = None,
        prefill_keypoint_names: Optional[list[str]] = None,
        prefill_skeleton_edges: Optional[list[tuple[int, int]]] = None,
        prefill_class_name: Optional[str] = None,
        prefill_class_names: Optional[list[str]] = None,
        auto_load_images: bool = False,
    ) -> None:
        if self._main_app is None:
            raise RuntimeError("Assisted Pose Curation is not attached to the main app.")
        module = _resolve_ui_module()
        if module is None:
            self._show_import_issue()
            return

        window = self._window
        if window is not None and window.winfo_exists():
            self._apply_prefill(
                project_root=prefill_project_root,
                image_dir=prefill_image_dir,
                label_dir=prefill_label_dir,
                video_path=prefill_video_path,
                model_path=prefill_model_path,
                dataset_yaml_path=prefill_dataset_yaml_path,
                keypoint_names=prefill_keypoint_names,
                skeleton_edges=prefill_skeleton_edges,
                class_name=prefill_class_name,
                class_names=prefill_class_names,
                auto_load_images=auto_load_images,
            )
            window.lift()
            try:
                window.focus_force()
            except Exception:
                pass
            return

        ui_error_type = getattr(module, "AssistedPoseCurationUIError", RuntimeError)
        if not isinstance(ui_error_type, type) or not issubclass(ui_error_type, Exception):
            ui_error_type = RuntimeError

        window_cls = getattr(module, "AssistedPoseCurationWindow", None)
        if window_cls is None:
            self._show_error("Assisted Pose Curation UI is missing AssistedPoseCurationWindow.")
            return

        try:
            parent = getattr(self._main_app, "root", None)
            self._window = window_cls(self._main_app, parent=parent)
            self._apply_prefill(
                project_root=prefill_project_root,
                image_dir=prefill_image_dir,
                label_dir=prefill_label_dir,
                video_path=prefill_video_path,
                model_path=prefill_model_path,
                dataset_yaml_path=prefill_dataset_yaml_path,
                keypoint_names=prefill_keypoint_names,
                skeleton_edges=prefill_skeleton_edges,
                class_name=prefill_class_name,
                class_names=prefill_class_names,
                auto_load_images=auto_load_images,
            )
            self._window.bind("<Destroy>", lambda _e: self._clear_window(), add="+")
            self._main_app.log_message("Assisted Pose Curation opened.", "INFO")
        except Exception as exc:  # pragma: no cover - surfaced to UI
            if isinstance(exc, ui_error_type):
                self._show_error(str(exc))
                return
            logging.getLogger(__name__).exception("Failed to open Assisted Pose Curation window")
            self._show_error(f"Failed to launch Assisted Pose Curation: {exc}")

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
        messagebox.showerror("Assisted Pose Curation", message, parent=parent)


def register_plugin(main_app) -> None:
    existing = getattr(main_app, "_assisted_pose_curation_plugin", None)
    if existing is not None:
        return
    plugin = AssistedPoseCurationPlugin()
    plugin.register(main_app)
    setattr(main_app, "_assisted_pose_curation_plugin", plugin)


__all__ = ["register_plugin"]
