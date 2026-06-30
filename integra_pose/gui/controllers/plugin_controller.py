from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
import re
import subprocess
import sys
import threading
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import tkinter as tk
from tkinter import Menu, messagebox, ttk

from integra_pose.utils import plugin_opt_in, plugin_security


@dataclass(frozen=True)
class DiscoveredPlugin:
    identifier: str  # directory name (e.g., plugin_eda)
    import_name: str  # normalized import name
    path: Path
    source: str  # "bundled" | "user"
    metadata: plugin_security.PluginMetadata


class PluginController:
    """Plugin discovery/launching and VAE+HMM toolkit management."""

    def __init__(self, app: Any, plugins_dir: Path | None = None, user_plugins_dir: Path | None = None):
        self.app = app
        self.plugins_dir = Path(plugins_dir) if plugins_dir else Path(__file__).resolve().parents[2] / "plugins"
        self.user_plugins_dir = (
            Path(user_plugins_dir) if user_plugins_dir else Path.home() / ".integrapose" / "plugins"
        )
        self._hmm_vae_controller = None
        self._hmm_vae_lock = threading.Lock()
        self._plugin_manager_window: Optional[tk.Toplevel] = None
        self._dataset_augmentor_plugin = None
        self._assisted_pose_curation_plugin = None

    def check_hmm_vae_dependencies(self) -> tuple[list[str], list[str]]:
        required = ["torch", "sklearn", "hmmlearn", "pandas", "numpy", "umap", "hdbscan", "matplotlib"]
        present: list[str] = []
        missing: list[str] = []
        for mod in required:
            try:
                __import__(mod)
                present.append(mod)
            except Exception:
                missing.append(mod)
        return missing, present

    def _hmm_vae_window(self):
        controller = getattr(self, "_hmm_vae_controller", None)
        if isinstance(controller, tk.Misc):
            return controller
        root = getattr(controller, "root", None)
        return root if isinstance(root, tk.Misc) else None

    def launch_hmm_vae_toolkit(self) -> None:
        """Launch the integrated VAE+HMM toolkit window."""
        with self._hmm_vae_lock:
            window = self._hmm_vae_window()
            if window is not None and window.winfo_exists():
                try:
                    window.lift()
                    window.focus_force()
                except Exception:
                    pass
                return
            missing, _present = self.check_hmm_vae_dependencies()
            if missing:
                msg = "Missing dependencies for VAE + HMM toolkit:\n- " + "\n- ".join(missing)
                self.app._toast(msg, level="error", duration_ms=3500)
                messagebox.showerror("VAE + HMM Toolkit", msg, parent=self.app.root)
                return
            toolkit_module = None
            toolkit_source = ""
            try:
                toolkit_module = importlib.import_module("integra_pose.hmm_vae_toolkit")
                toolkit_source = "built-in"
            except Exception:
                toolkit_module = None
            if toolkit_module is None:
                msg = "VAE + HMM toolkit files are missing from the install."
                self.app._toast(msg, level="error", duration_ms=4000)
                messagebox.showerror("VAE + HMM Toolkit", msg, parent=self.app.root)
                return
            try:
                controller = toolkit_module.launch(self.app.root)
                self._hmm_vae_controller = controller
                win = self._hmm_vae_window()
                if win is not None:
                    try:
                        win.bind("<Destroy>", lambda _e: self._clear_hmm_vae_controller(), add="+")
                    except Exception:
                        pass
                self.app.log_message(f"Opened VAE + HMM segmentation toolkit ({toolkit_source}).", "INFO")
                self.app._toast("Launched VAE + HMM toolkit.", level="info")
            except ModuleNotFoundError as exc:
                self.app._toast(f"Missing dependency: {exc}", level="error")
                messagebox.showerror("VAE + HMM Toolkit", f"Missing dependency: {exc}", parent=self.app.root)
            except Exception as exc:
                self.app._toast(f"Failed to launch VAE + HMM toolkit: {exc}", level="error")
                self.app.log_message(f"Failed to launch VAE + HMM toolkit: {exc}", "ERROR")
                messagebox.showerror("VAE + HMM Toolkit", f"Failed to launch: {exc}", parent=self.app.root)

    def _clear_hmm_vae_controller(self) -> None:
        with self._hmm_vae_lock:
            self._hmm_vae_controller = None

    def _prompt_trust(self, metadata: plugin_security.PluginMetadata) -> bool:
        details = [
            f"Name: {metadata.name}",
            f"Version: {metadata.version or 'unknown'}",
            f"Author: {metadata.author or 'unknown'}",
            "",
            metadata.description or "No description provided.",
            "",
            f"Location: {metadata.path}",
            "",
            "By enabling a plugin you accept responsibility for obtaining the plugin and its dependencies.",
            "IntegraPose will not download or install packages automatically.",
            "",
            "Enable this plugin? Only approve code you trust.",
        ]
        return messagebox.askyesno(
            title="Enable third-party plugin?",
            message="\n".join(details),
            icon=messagebox.WARNING,
            parent=self.app.root,
        )

    def _prompt_opt_in(self, metadata: plugin_security.PluginMetadata) -> bool:
        details = [
            f"Name: {metadata.name}",
            f"Version: {metadata.version or 'unknown'}",
            f"Author: {metadata.author or 'unknown'}",
            "",
            metadata.description or "No description provided.",
            "",
            f"Location: {metadata.path}",
            "",
            "Plugins are optional. You may need to install extra Python packages for this plugin.",
            "IntegraPose will not download or install packages automatically.",
            "",
            "Enable this plugin?",
        ]
        return messagebox.askyesno(
            title="Enable plugin?",
            message="\n".join(details),
            icon=messagebox.WARNING,
            parent=self.app.root,
        )

    def _ensure_plugins_menu(self) -> None:
        if not hasattr(self.app, "plugins_menu") or self.app.plugins_menu is None:
            if hasattr(self.app, "menu_bar"):
                self.app.plugins_menu = Menu(self.app.menu_bar, tearoff=0)
                self.app.menu_bar.add_cascade(label="Plugins", menu=self.app.plugins_menu)
                self.app.log_message("Plugins menu was missing; created dynamically.", "WARNING")

    def _reset_plugins_menu(self) -> None:
        self._ensure_plugins_menu()
        menu = getattr(self.app, "plugins_menu", None)
        if menu is None:
            return
        try:
            menu.delete(0, "end")
        except Exception:
            pass
        menu.add_command(label="Manage Plugins...", command=self.open_plugin_manager)
        menu.add_command(label="Open Plugins Folder...", command=self.open_user_plugins_folder)
        menu.add_separator()

    def open_user_plugins_folder(self) -> None:
        path = self.user_plugins_dir
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
                return
            if sys.platform == "darwin":
                subprocess.run(["open", str(path)], check=False)
                return
            subprocess.run(["xdg-open", str(path)], check=False)
        except Exception:
            messagebox.showinfo("Plugins Folder", f"Plugins folder:\n{path}", parent=self.app.root)

    def open_plugin_manager(self) -> None:
        window = self._plugin_manager_window
        if window is not None and window.winfo_exists():
            window.lift()
            try:
                window.focus_force()
            except Exception:
                pass
            return

        window = _PluginManagerWindow(self, parent=self.app.root)
        self._plugin_manager_window = window
        window.bind("<Destroy>", lambda _e: self._clear_plugin_manager(), add="+")

    def _clear_plugin_manager(self) -> None:
        self._plugin_manager_window = None

    def launch_dataset_augmentor_lab(
        self,
        *,
        prefill_dataset_root: str | None = None,
        prefill_output_root: str | None = None,
    ) -> bool:
        target_id = "plugin_dataset_augmentor_lab"
        discovered_map = {plugin.identifier: plugin for plugin in self.discover_plugins()}
        plugin = discovered_map.get(target_id)
        if plugin is None:
            msg = (
                "Dataset Augmentor Lab plugin was not found. "
                "Ensure integra_pose/plugins/plugin_dataset_augmentor_lab is present."
            )
            self.app.log_message(msg, "ERROR")
            messagebox.showerror("Dataset Augmentor Lab", msg, parent=self.app.root)
            return False

        if not plugin_opt_in.is_plugin_enabled(plugin.path):
            if not self.enable_plugin(plugin):
                return False
        else:
            decision = plugin_security.ensure_plugin_trusted(plugin.path, prompt=self._prompt_trust)
            self.app.log_message(decision.message, decision.level)
            if not decision.allowed:
                return False

        module_import_path = f"integra_pose.plugins.{plugin.import_name}.plugin"
        plugin_module = self._import_plugin_module(plugin, module_import_path)
        if plugin_module is None:
            msg = "Failed to import Dataset Augmentor Lab plugin module."
            self.app.log_message(msg, "ERROR")
            messagebox.showerror("Dataset Augmentor Lab", msg, parent=self.app.root)
            return False

        plugin_cls = getattr(plugin_module, "DatasetAugmentorLabPlugin", None)
        if plugin_cls is None:
            msg = "Dataset Augmentor Lab plugin module is missing DatasetAugmentorLabPlugin."
            self.app.log_message(msg, "ERROR")
            messagebox.showerror("Dataset Augmentor Lab", msg, parent=self.app.root)
            return False

        try:
            launcher = self._dataset_augmentor_plugin
            if launcher is None or not isinstance(launcher, plugin_cls):
                launcher = plugin_cls()
                self._dataset_augmentor_plugin = launcher

            attach = getattr(launcher, "attach", None)
            if callable(attach):
                attach(self.app)
            else:
                setattr(launcher, "_main_app", self.app)

            open_window = getattr(launcher, "open_window", None)
            if callable(open_window):
                open_window(
                    prefill_dataset_root=prefill_dataset_root,
                    prefill_output_root=prefill_output_root,
                )
            else:
                fallback_open = getattr(launcher, "_open_window", None)
                if callable(fallback_open):
                    fallback_open(
                        prefill_dataset_root=prefill_dataset_root,
                        prefill_output_root=prefill_output_root,
                    )
                else:
                    raise RuntimeError("Plugin launcher has no open method.")

            return True
        except Exception as exc:
            self.app.log_message(f"Failed to launch Dataset Augmentor Lab: {exc}", "ERROR")
            messagebox.showerror("Dataset Augmentor Lab", f"Failed to launch: {exc}", parent=self.app.root)
            return False

    def launch_assisted_pose_curation(
        self,
        *,
        prefill_project_root: str | None = None,
        prefill_image_dir: str | None = None,
        prefill_label_dir: str | None = None,
        prefill_video_path: str | None = None,
        prefill_model_path: str | None = None,
        prefill_dataset_yaml_path: str | None = None,
        prefill_keypoint_names: list[str] | None = None,
        prefill_skeleton_edges: list[tuple[int, int]] | None = None,
        prefill_class_name: str | None = None,
        auto_load_images: bool = False,
    ) -> bool:
        target_id = "plugin_assisted_pose_curation"
        discovered_map = {plugin.identifier: plugin for plugin in self.discover_plugins()}
        plugin = discovered_map.get(target_id)
        if plugin is None:
            msg = (
                "Assisted Pose Curation plugin was not found. "
                "Ensure integra_pose/plugins/plugin_assisted_pose_curation is present."
            )
            self.app.log_message(msg, "ERROR")
            messagebox.showerror("Assisted Pose Curation", msg, parent=self.app.root)
            return False

        if not plugin_opt_in.is_plugin_enabled(plugin.path):
            if not self.enable_plugin(plugin):
                return False
        else:
            decision = plugin_security.ensure_plugin_trusted(plugin.path, prompt=self._prompt_trust)
            self.app.log_message(decision.message, decision.level)
            if not decision.allowed:
                return False

        module_import_path = f"integra_pose.plugins.{plugin.import_name}.plugin"
        plugin_module = self._import_plugin_module(plugin, module_import_path)
        if plugin_module is None:
            msg = "Failed to import Assisted Pose Curation plugin module."
            self.app.log_message(msg, "ERROR")
            messagebox.showerror("Assisted Pose Curation", msg, parent=self.app.root)
            return False

        plugin_cls = getattr(plugin_module, "AssistedPoseCurationPlugin", None)
        if plugin_cls is None:
            msg = "Assisted Pose Curation plugin module is missing AssistedPoseCurationPlugin."
            self.app.log_message(msg, "ERROR")
            messagebox.showerror("Assisted Pose Curation", msg, parent=self.app.root)
            return False

        try:
            launcher = self._assisted_pose_curation_plugin
            if launcher is None or not isinstance(launcher, plugin_cls):
                launcher = plugin_cls()
                self._assisted_pose_curation_plugin = launcher

            attach = getattr(launcher, "attach", None)
            if callable(attach):
                attach(self.app)
            else:
                setattr(launcher, "_main_app", self.app)

            open_window = getattr(launcher, "open_window", None)
            if callable(open_window):
                open_window(
                    prefill_project_root=prefill_project_root,
                    prefill_image_dir=prefill_image_dir,
                    prefill_label_dir=prefill_label_dir,
                    prefill_video_path=prefill_video_path,
                    prefill_model_path=prefill_model_path,
                    prefill_dataset_yaml_path=prefill_dataset_yaml_path,
                    prefill_keypoint_names=prefill_keypoint_names,
                    prefill_skeleton_edges=prefill_skeleton_edges,
                    prefill_class_name=prefill_class_name,
                    auto_load_images=auto_load_images,
                )
            else:
                fallback_open = getattr(launcher, "_open_window", None)
                if callable(fallback_open):
                    fallback_open(
                        prefill_project_root=prefill_project_root,
                        prefill_image_dir=prefill_image_dir,
                        prefill_label_dir=prefill_label_dir,
                        prefill_video_path=prefill_video_path,
                        prefill_model_path=prefill_model_path,
                        prefill_dataset_yaml_path=prefill_dataset_yaml_path,
                        prefill_keypoint_names=prefill_keypoint_names,
                        prefill_skeleton_edges=prefill_skeleton_edges,
                        prefill_class_name=prefill_class_name,
                        auto_load_images=auto_load_images,
                    )
                else:
                    raise RuntimeError("Plugin launcher has no open method.")

            return True
        except Exception as exc:
            self.app.log_message(f"Failed to launch Assisted Pose Curation: {exc}", "ERROR")
            messagebox.showerror("Assisted Pose Curation", f"Failed to launch: {exc}", parent=self.app.root)
            return False

    def discover_plugins(self) -> list[DiscoveredPlugin]:
        candidates: dict[str, tuple[Path, str]] = {}
        roots = [("bundled", self.plugins_dir), ("user", self.user_plugins_dir)]
        for source, root in roots:
            if not root.is_dir():
                continue
            for plugin_path in root.iterdir():
                plugin_name = plugin_path.name
                if not plugin_path.is_dir():
                    continue
                if plugin_name.startswith(".") or plugin_name.startswith("_"):
                    continue
                candidates.setdefault(plugin_name, (plugin_path, source))

        discovered: list[DiscoveredPlugin] = []
        used_import_names: set[str] = set()
        ordered_candidates = sorted(candidates.items(), key=lambda item: (item[0].lower(), item[0]))
        for plugin_name, (plugin_path, source) in ordered_candidates:
            base_import_name = re.sub(r"[^0-9a-zA-Z_]", "_", plugin_name) or "plugin"
            normalized_name = base_import_name
            suffix = 2
            while normalized_name in used_import_names:
                normalized_name = f"{base_import_name}_{suffix}"
                suffix += 1
            used_import_names.add(normalized_name)
            metadata = plugin_security.get_plugin_metadata(plugin_path)
            discovered.append(
                DiscoveredPlugin(
                    identifier=plugin_name,
                    import_name=normalized_name,
                    path=plugin_path,
                    source=source,
                    metadata=metadata,
                )
            )
        return discovered

    def load_plugins(self) -> None:
        self.app.log_message("--- Starting Plugin Loading ---", "INFO")
        self._reset_plugins_menu()
        self.app.log_message(f"Bundled plugins directory: {self.plugins_dir}", "INFO")
        self.app.log_message(f"User plugins directory: {self.user_plugins_dir}", "INFO")

        discovered = self.discover_plugins()
        if discovered and not any(plugin_opt_in.is_plugin_enabled(p.path) for p in discovered):
            self.app.log_message(
                "No plugins are enabled. Open Plugins > Manage Plugins... to enable optional features.",
                "INFO",
            )
        loaded: list[str] = []
        for plugin in discovered:
            if not plugin_opt_in.is_plugin_enabled(plugin.path):
                continue

            decision = plugin_security.ensure_plugin_trusted(plugin.path, prompt=None)
            self.app.log_message(decision.message, decision.level)
            if not decision.allowed:
                continue

            if self._load_plugin(plugin):
                loaded.append(plugin.identifier)

        self.app.log_message(f"Plugins loaded: {loaded}", "INFO")
        self.app.log_message("--- Finished Plugin Loading ---", "INFO")

    def enable_plugin(self, plugin: DiscoveredPlugin) -> bool:
        if plugin_opt_in.is_plugin_enabled(plugin.path):
            return True

        existing_trust = plugin_security.get_trust_state(plugin.path)
        decision = plugin_security.ensure_plugin_trusted(plugin.path, prompt=self._prompt_trust)
        self.app.log_message(decision.message, decision.level)
        if not decision.allowed:
            return False

        prompted_for_trust = (
            existing_trust is None and not plugin_security.is_bundled_plugin_path(plugin.path)
        )
        if not prompted_for_trust and not self._prompt_opt_in(plugin.metadata):
            return False

        if not self._load_plugin(plugin):
            self._toast_plugin_failure(plugin)
            return False

        plugin_opt_in.set_plugin_enabled(
            plugin.path,
            True,
            name=plugin.metadata.name,
            version=plugin.metadata.version,
            source=plugin.source,
        )
        return True

    def disable_plugin(self, plugin: DiscoveredPlugin) -> None:
        plugin_opt_in.set_plugin_enabled(
            plugin.path,
            False,
            name=plugin.metadata.name,
            version=plugin.metadata.version,
            source=plugin.source,
        )
        # Rebuild menu to remove plugin-provided entries.
        self.load_plugins()

    def _toast_plugin_failure(self, plugin: DiscoveredPlugin) -> None:
        toast = getattr(self.app, "_toast", None)
        if callable(toast):
            toast(f"Plugin '{plugin.metadata.name}' failed to load and was not enabled. Check logs.", level="error")

    def _load_plugin(self, plugin: DiscoveredPlugin) -> bool:
        plugin_file = plugin.path / "plugin.py"
        if not plugin_file.is_file():
            self.app.log_message(f"No plugin.py found for {plugin.identifier}; skipping.", "WARNING")
            return False

        module_import_path = f"integra_pose.plugins.{plugin.import_name}.plugin"
        plugin_module = self._import_plugin_module(plugin, module_import_path)
        if plugin_module is None:
            self.app.log_message(f"No plugin module resolved for {plugin.identifier}", "ERROR")
            return False

        try:
            if hasattr(plugin_module, "IntegraPosePlugin"):
                instance = plugin_module.IntegraPosePlugin()
                instance.register(self.app)
                self.app.log_message(f"Loaded plugin class from {plugin.identifier}", "INFO")
                return True
            if hasattr(plugin_module, "register_plugin") and callable(plugin_module.register_plugin):
                plugin_module.register_plugin(self.app)
                self.app.log_message(f"Loaded plugin function from {plugin.identifier}", "INFO")
                return True

            self.app.log_message(f"{plugin.identifier} has no IntegraPosePlugin or register_plugin", "WARNING")
            return False
        except Exception as exc:  # pragma: no cover - UI path
            import traceback as _tb

            self.app.log_message(
                f"Exception initializing plugin {plugin.identifier}: {exc}\n{_tb.format_exc()}",
                "ERROR",
            )
            return False

    def _import_plugin_module(self, plugin: DiscoveredPlugin, module_import_path: str):
        try:
            return importlib.import_module(module_import_path)
        except Exception as exc:
            self.app.log_message(f"import_module failed for {module_import_path}: {exc}", "DEBUG")

        try:
            self._mount_plugin_package(plugin)
            return importlib.import_module(module_import_path)
        except Exception as exc2:  # pragma: no cover - UI path
            self.app.log_message(f"Path-based import failed for {plugin.path}: {exc2}", "ERROR")
            return None

    def _mount_plugin_package(self, plugin: DiscoveredPlugin) -> None:
        package_name = f"integra_pose.plugins.{plugin.import_name}"
        if package_name in sys.modules:
            existing_module = sys.modules[package_name]
            if self._module_matches_plugin_path(existing_module, plugin.path):
                return
            raise RuntimeError(
                f"Plugin import namespace collision for '{plugin.identifier}' at {plugin.path}: {package_name}"
            )

        init_file = plugin.path / "__init__.py"
        if init_file.is_file():
            spec = importlib.util.spec_from_file_location(
                package_name, init_file, submodule_search_locations=[str(plugin.path)]
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[package_name] = module
                spec.loader.exec_module(module)  # type: ignore[attr-defined]
                return

        module = types.ModuleType(package_name)
        spec = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(plugin.path)]
        module.__spec__ = spec
        module.__path__ = list(spec.submodule_search_locations)
        sys.modules[package_name] = module

    @staticmethod
    def _module_matches_plugin_path(module: object, plugin_path: Path) -> bool:
        try:
            resolved_plugin_path = plugin_path.resolve()
        except Exception:
            resolved_plugin_path = plugin_path

        module_paths = getattr(module, "__path__", None)
        if module_paths:
            for candidate in module_paths:
                try:
                    if Path(candidate).resolve() == resolved_plugin_path:
                        return True
                except Exception:
                    if str(candidate) == str(plugin_path):
                        return True

        module_file = getattr(module, "__file__", None)
        if module_file:
            try:
                return Path(module_file).resolve().parent == resolved_plugin_path
            except Exception:
                return str(module_file).startswith(str(plugin_path))
        return False


class _PluginManagerWindow(tk.Toplevel):
    def __init__(self, controller: PluginController, parent: tk.Misc | None):
        super().__init__(parent)
        self.controller = controller
        self.title("Plugin Manager")
        self.geometry("820x420")
        self.minsize(720, 360)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Label(
            self,
            text=(
                "Plugins are disabled by default. Enable only what you intend to use.\n\n"
                "If a plugin is unavailable due to missing dependencies, install optional deps:\n"
                "  pip install \"IntegraPose[plugins]\"   (pip/wheel)\n"
                "  pip install \".[plugins]\"            (source checkout)"
            ),
            wraplength=780,
            justify="left",
        )
        header.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))

        columns = ("id", "name", "version", "source", "trusted", "enabled")
        self.tree = ttk.Treeview(self, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("id", text="ID")
        self.tree.heading("name", text="Name")
        self.tree.heading("version", text="Version")
        self.tree.heading("source", text="Source")
        self.tree.heading("trusted", text="Trusted")
        self.tree.heading("enabled", text="Enabled")
        self.tree.column("id", width=170, anchor="w")
        self.tree.column("name", width=240, anchor="w")
        self.tree.column("version", width=80, anchor="w")
        self.tree.column("source", width=80, anchor="w")
        self.tree.column("trusted", width=70, anchor="center")
        self.tree.column("enabled", width=70, anchor="center")

        scroll = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.grid(row=1, column=0, sticky="nsew", padx=(12, 0), pady=6)
        scroll.grid(row=1, column=1, sticky="ns", pady=6)

        self.details = tk.StringVar(value="Select a plugin to see details.")
        details_label = ttk.Label(self, textvariable=self.details, wraplength=780, justify="left")
        details_label.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 6))

        btns = ttk.Frame(self)
        btns.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 12))
        btns.columnconfigure(5, weight=1)

        ttk.Button(btns, text="Enable", command=self._enable_selected).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(btns, text="Disable", command=self._disable_selected).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(btns, text="Open Folder", command=self._open_selected_folder).grid(row=0, column=2, padx=(0, 8))
        ttk.Button(btns, text="Open Plugins Folder", command=controller.open_user_plugins_folder).grid(
            row=0, column=3, padx=(0, 8)
        )
        ttk.Button(btns, text="Refresh", command=self.refresh).grid(row=0, column=4, padx=(0, 8))
        ttk.Button(btns, text="Close", command=self.destroy).grid(row=0, column=6, sticky="e")

        self.tree.bind("<<TreeviewSelect>>", self._on_select, add="+")
        self._plugins_by_iid: dict[str, DiscoveredPlugin] = {}
        self.refresh()
        self.focus_set()

    def refresh(self) -> None:
        self._plugins_by_iid.clear()
        for item in self.tree.get_children():
            self.tree.delete(item)

        for plugin in self.controller.discover_plugins():
            trusted = plugin_security.get_trust_state(plugin.path) == "allow"
            enabled = plugin_opt_in.is_plugin_enabled(plugin.path)
            iid = plugin_opt_in.plugin_key(plugin.path)
            self._plugins_by_iid[iid] = plugin
            self.tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    plugin.identifier,
                    plugin.metadata.name,
                    plugin.metadata.version or "",
                    plugin.source,
                    "yes" if trusted else "no",
                    "yes" if enabled else "no",
                ),
            )

        if not self.tree.get_children():
            self.details.set(
                "No plugins found. Download plugins into your plugins folder and restart (or press Refresh)."
            )
        else:
            self.details.set("Select a plugin to see details.")

    def _selected_plugin(self) -> Optional[DiscoveredPlugin]:
        sel = self.tree.selection()
        if not sel:
            return None
        return self._plugins_by_iid.get(sel[0])

    def _on_select(self, _event=None) -> None:
        plugin = self._selected_plugin()
        if plugin is None:
            self.details.set("Select a plugin to see details.")
            return
        enabled = plugin_opt_in.is_plugin_enabled(plugin.path)
        trusted = plugin_security.get_trust_state(plugin.path) == "allow"
        desc = plugin.metadata.description or "No description provided."
        self.details.set(
            f"{plugin.metadata.name} ({plugin.identifier})\n"
            f"Enabled: {'yes' if enabled else 'no'}   Trusted: {'yes' if trusted else 'no'}   Source: {plugin.source}\n"
            f"Path: {plugin.path}\n\n{desc}"
        )

    def _enable_selected(self) -> None:
        plugin = self._selected_plugin()
        if plugin is None:
            return
        if self.controller.enable_plugin(plugin):
            self.refresh()

    def _disable_selected(self) -> None:
        plugin = self._selected_plugin()
        if plugin is None:
            return
        if not messagebox.askyesno(
            "Disable plugin?",
            f"Disable '{plugin.metadata.name}'?\n\nThis will remove its menu entries for this session.",
            parent=self,
            icon=messagebox.WARNING,
        ):
            return
        self.controller.disable_plugin(plugin)
        self.refresh()

    def _open_selected_folder(self) -> None:
        plugin = self._selected_plugin()
        if plugin is None:
            return
        path = plugin.path
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
                return
            if sys.platform == "darwin":
                subprocess.run(["open", str(path)], check=False)
                return
            subprocess.run(["xdg-open", str(path)], check=False)
        except Exception:
            messagebox.showinfo("Plugin Folder", f"Plugin folder:\n{path}", parent=self)
