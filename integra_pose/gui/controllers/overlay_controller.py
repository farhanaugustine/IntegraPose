from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
import tkinter as tk
from tkinter import colorchooser, ttk, filedialog, messagebox, simpledialog

from integra_pose.gui.scrollable import create_scrollable_section
from integra_pose.gui.tooltips import CreateToolTip
from integra_pose.utils.overlay_presets import (
    DEFAULT_OVERLAY_ORDER,
    OVERLAY_DISPLAY_NAMES,
    default_presets,
    sanitize_order,
)


class OverlayController:
    _OVERLAY_BOOL_ATTRS = [
        "sv_use_box_var",
        "sv_use_label_var",
        "sv_use_trace_var",
        "sv_show_keypoints_var",
        "sv_show_keypoint_vertices_var",
        "sv_show_heading_arrows_var",
        "sv_use_halo_var",
        "sv_use_blur_var",
        "sv_use_pixelate_var",
        "sv_use_background_overlay_var",
        "sv_use_heatmap_var",
        "sv_enable_edges_var",
        "infer_hide_labels_var",
        "infer_hide_conf_var",
    ]
    _OVERLAY_SCALAR_ATTRS = [
        "sv_skeleton_color_var",
    ]
    _OVERLAY_ADVANCED_ATTRS = [
        "sv_halo_kernel_var",
        "sv_halo_opacity_var",
        "sv_blur_kernel_var",
        "sv_pixelate_size_var",
        "sv_heatmap_radius_var",
        "sv_heatmap_kernel_var",
        "sv_heatmap_opacity_var",
        "sv_heatmap_decay_var",
        "sv_heatmap_source_var",
        "sv_heatmap_anchor_var",
        "sv_heatmap_keypoint_index_var",
        "sv_background_opacity_var",
        "sv_background_force_box_var",
        "sv_vertex_radius_var",
        "sv_edge_thickness_var",
        "sv_keypoint_palette_var",
        "sv_edge_palette_var",
        "sv_trace_length_var",
        "sv_trace_thickness_var",
        "sv_trace_opacity_var",
        "sv_trace_persistent_var",
        "sv_trace_source_var",
        "sv_trace_keypoint_var",
        "sv_trace_color_var",
    ]
    _PERFORMANCE_HEAVY_BOOL_ATTRS = [
        "sv_use_heatmap_var",
        "sv_use_blur_var",
        "sv_use_pixelate_var",
        "sv_use_background_overlay_var",
        "sv_use_halo_var",
    ]

    def __init__(self, app) -> None:
        self.app = app
        self.config = app.config
        self.root = app.root
        self.log_message = app.log_message
        self.update_status = app.update_status
        self.performance_metrics_var = tk.StringVar(master=self.root, value="Idle")
        self.performance_warning_var = tk.StringVar(master=self.root, value="")
        self._supervision_notice_var = tk.StringVar(master=self.root, value="")
        self._overlay_widgets: dict[str, tk.Widget] = {}
        self._supervision_control_widgets: list[tuple[tk.Widget, str]] = []
        self._supervision_annotations_supported = True
        self._supervision_disabled_snapshot: dict[str, object] | None = None
        self._overlay_trace_guard = False
        self._overlay_var_handles: list[tuple[tk.Variable, str]] = []
        self._builtin_overlay_presets = default_presets()
        self._available_overlay_presets: list[dict[str, object]] = []
        self._performance_safe_snapshot: dict[str, object] | None = None
        self._overlay_settings_window = None

    def __getattr__(self, name):
        return getattr(self.app, name)

    @property
    def _active_supervision_runner(self):
        return getattr(self.app, "_active_supervision_runner", None)

    @_active_supervision_runner.setter
    def _active_supervision_runner(self, value):
        setattr(self.app, "_active_supervision_runner", value)

    def _initialize_overlay_state_management(self) -> None:
        cfg = self.config.inference
        if not isinstance(getattr(cfg, "overlay_presets", []), list):
            cfg.overlay_presets = []
        cfg.overlay_presets = list(cfg.overlay_presets or [])
        existing_order = getattr(cfg, "overlay_order", DEFAULT_OVERLAY_ORDER)
        cfg.overlay_order = sanitize_order(existing_order)
        self._available_overlay_presets = []
        self._overlay_widgets = {}
        self._overlay_trace_guard = False
        self._overlay_var_handles = []

        watched_attrs = (
            self._OVERLAY_BOOL_ATTRS
            + self._OVERLAY_SCALAR_ATTRS
            + self._OVERLAY_ADVANCED_ATTRS
        )
        for attr in watched_attrs:
            var = getattr(cfg, attr, None)
            if isinstance(var, tk.Variable):
                handle = var.trace_add("write", self._on_overlay_setting_changed)
                self._overlay_var_handles.append((var, handle))

        perf_var = getattr(cfg, "sv_performance_safe_var", None)
        if isinstance(perf_var, tk.Variable):
            handle = perf_var.trace_add("write", self._on_performance_safe_changed)
            self._overlay_var_handles.append((perf_var, handle))

        self._refresh_overlay_preset_catalog()

    @staticmethod
    def _coerce_for_var(var: tk.Variable, value):
        if isinstance(var, tk.BooleanVar):
            return bool(value)
        if isinstance(var, tk.IntVar):
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
        if isinstance(var, tk.DoubleVar):
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0
        return "" if value is None else value

    def _on_overlay_setting_changed(self, *_):
        if self._overlay_trace_guard:
            return
        self._mark_overlay_state_custom()

    def _mark_overlay_state_custom(self) -> None:
        preset_var = self.config.inference.sv_selected_preset_var
        if preset_var.get() != "Custom":
            preset_var.set("Custom")

    def _capture_overlay_state(self) -> dict:
        cfg = self.config.inference
        state_bools = {}
        for attr in self._OVERLAY_BOOL_ATTRS:
            var = getattr(cfg, attr, None)
            state_bools[attr] = bool(var.get()) if isinstance(var, tk.Variable) else bool(var)

        state_scalars = {}
        for attr in self._OVERLAY_SCALAR_ATTRS:
            var = getattr(cfg, attr, None)
            state_scalars[attr] = var.get() if isinstance(var, tk.Variable) else var

        state_advanced = {}
        for attr in self._OVERLAY_ADVANCED_ATTRS:
            var = getattr(cfg, attr, None)
            state_advanced[attr] = var.get() if isinstance(var, tk.Variable) else var

        return {
            "booleans": state_bools,
            "scalars": state_scalars,
            "advanced": state_advanced,
            "overlay_order": sanitize_order(cfg.overlay_order),
            "performance_safe": bool(getattr(cfg.sv_performance_safe_var, "get", lambda: False)()),
            "capture_metrics": bool(getattr(cfg.capture_metrics_var, "get", lambda: False)()),
        }

    def _apply_overlay_state(self, payload: dict, preset_name: str | None = None, update_selection: bool = True) -> None:
        cfg = self.config.inference
        booleans = dict(payload.get("booleans", {}))
        scalars = dict(payload.get("scalars", {}))
        advanced = dict(payload.get("advanced", {}))
        order = sanitize_order(payload.get("overlay_order", cfg.overlay_order))
        performance_safe = bool(payload.get("performance_safe", cfg.sv_performance_safe_var.get()))
        capture_metrics = bool(payload.get("capture_metrics", cfg.capture_metrics_var.get()))

        self._overlay_trace_guard = True
        try:
            for attr, value in booleans.items():
                var = getattr(cfg, attr, None)
                if isinstance(var, tk.Variable):
                    var.set(self._coerce_for_var(var, value))
            for attr, value in scalars.items():
                var = getattr(cfg, attr, None)
                if isinstance(var, tk.Variable):
                    var.set(self._coerce_for_var(var, value))
            for attr, value in advanced.items():
                var = getattr(cfg, attr, None)
                if isinstance(var, tk.Variable):
                    var.set(self._coerce_for_var(var, value))
            cfg.overlay_order = order
            # Keep legacy and new skeleton edge toggles aligned.
            edges_enabled = False
            if hasattr(cfg, "sv_enable_edges_var") and isinstance(cfg.sv_enable_edges_var, tk.Variable):
                edges_enabled = edges_enabled or bool(cfg.sv_enable_edges_var.get())
            if hasattr(cfg, "sv_show_keypoints_var") and isinstance(cfg.sv_show_keypoints_var, tk.Variable):
                edges_enabled = edges_enabled or bool(cfg.sv_show_keypoints_var.get())
            if hasattr(cfg, "sv_enable_edges_var") and isinstance(cfg.sv_enable_edges_var, tk.Variable):
                cfg.sv_enable_edges_var.set(edges_enabled)
            if hasattr(cfg, "sv_show_keypoints_var") and isinstance(cfg.sv_show_keypoints_var, tk.Variable):
                cfg.sv_show_keypoints_var.set(edges_enabled)
        finally:
            self._overlay_trace_guard = False

        cfg.sv_performance_safe_var.set(performance_safe)
        if hasattr(cfg, "capture_metrics_var"):
            cfg.capture_metrics_var.set(capture_metrics)

        if update_selection and preset_name:
            cfg.sv_selected_preset_var.set(preset_name)

        self._sync_overlay_order_listbox()
        self._sync_overlay_widget_states()

    def _get_overlay_preset_by_name(self, name: str) -> dict | None:
        for entry in self._available_overlay_presets:
            if entry.get("name") == name:
                return entry
        return None

    def _refresh_overlay_preset_catalog(self) -> None:
        combined = []
        for preset in self._builtin_overlay_presets:
            entry = deepcopy(preset)
            entry["locked"] = True
            combined.append(entry)
        for preset in self.config.inference.overlay_presets:
            entry = deepcopy(preset)
            entry["locked"] = bool(entry.get("locked", False))
            combined.append(entry)
        self._available_overlay_presets = combined

        names = [entry.get("name", "") for entry in combined]
        values = names + ["Custom"]
        if hasattr(self, "overlay_preset_combo"):
            self.overlay_preset_combo["values"] = values
        if self.config.inference.sv_selected_preset_var.get() not in names:
            self.config.inference.sv_selected_preset_var.set("Custom")

    def _apply_named_overlay_preset(self, name: str) -> None:
        if name == "Custom":
            return
        preset = self._get_overlay_preset_by_name(name)
        if not preset:
            self.log_message(f"Preset '{name}' not found.", "WARNING")
            return
        values = preset.get("values")
        if not isinstance(values, dict):
            self.log_message(f"Preset '{name}' is malformed.", "ERROR")
            return
        self._apply_overlay_state(deepcopy(values), preset_name=name, update_selection=True)

    def _handle_overlay_preset_selection(self, *_):
        selected = self.config.inference.sv_selected_preset_var.get()
        self._apply_named_overlay_preset(selected)

    def _save_overlay_preset_as(self) -> None:
        current_name = self.config.inference.sv_selected_preset_var.get()
        suggestion = "" if current_name == "Custom" else current_name
        name = simpledialog.askstring("Save Overlay Preset", "Preset name:", initialvalue=suggestion, parent=self.root)
        if not name:
            return
        self._save_overlay_preset(name.strip(), overwrite=False)

    def _update_selected_overlay_preset(self) -> None:
        name = self.config.inference.sv_selected_preset_var.get()
        if name in ("", "Custom"):
            self._save_overlay_preset_as()
            return
        preset = self._get_overlay_preset_by_name(name)
        if not preset or preset.get("locked"):
            messagebox.showinfo("Preset Locked", "Built-in presets cannot be overwritten. Use 'Save As' to create a custom preset.", parent=self.root)
            return
        self._save_overlay_preset(name, overwrite=True)

    def _delete_selected_overlay_preset(self) -> None:
        name = self.config.inference.sv_selected_preset_var.get()
        if name in ("", "Custom"):
            messagebox.showinfo("Delete Preset", "Select a custom preset to delete.", parent=self.root)
            return
        preset = self._get_overlay_preset_by_name(name)
        if not preset or preset.get("locked"):
            messagebox.showinfo("Delete Preset", "Built-in presets cannot be deleted.", parent=self.root)
            return
        if not messagebox.askyesno("Delete Preset", f"Remove preset '{name}'?", parent=self.root):
            return
        cfg = self.config.inference
        cfg.overlay_presets = [entry for entry in cfg.overlay_presets if entry.get("name") != name]
        self._refresh_overlay_preset_catalog()
        cfg.sv_selected_preset_var.set("Custom")
        self.update_status(f"Deleted overlay preset '{name}'.")

    def _export_overlay_preset(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Export Overlay Settings",
            filetypes=[("JSON file", "*.json")],
            defaultextension=".json",
            parent=self.root
        )
        if not path:
            return

        try:
            state = self._capture_overlay_state()
            with open(path, 'w') as f:
                json.dump(state, f, indent=4)
            self.log_message(f"Overlay settings exported to {path}", "INFO")
            messagebox.showinfo("Export Successful", f"Overlay settings exported to {path}", parent=self.root)
        except Exception as e:
            self.log_message(f"Failed to export overlay settings: {e}", "ERROR")
            messagebox.showerror("Export Error", f"Failed to export overlay settings: {e}", parent=self.root)

    def _save_overlay_preset(self, name: str, *, overwrite: bool) -> None:
        name = name.strip()
        if not name:
            messagebox.showinfo("Save Preset", "Preset name cannot be empty.", parent=self.root)
            return
        preset = self._get_overlay_preset_by_name(name)
        if preset and preset.get("locked"):
            messagebox.showinfo("Save Preset", "Built-in presets cannot be overwritten.", parent=self.root)
            return
        cfg = self.config.inference
        state = deepcopy(self._capture_overlay_state())
        new_presets = []
        replaced = False
        for entry in cfg.overlay_presets:
            if entry.get("name") == name:
                if not overwrite:
                    messagebox.showinfo("Save Preset", f"Preset '{name}' already exists. Use update to overwrite.", parent=self.root)
                    return
                entry = deepcopy(entry)
                entry["values"] = state
                replaced = True
            new_presets.append(entry)
        if not replaced:
            new_presets.append({"name": name, "values": state})
        cfg.overlay_presets = new_presets
        self._refresh_overlay_preset_catalog()
        cfg.sv_selected_preset_var.set(name)
        self.update_status(f"Saved overlay preset '{name}'.")

    def _sync_overlay_order_listbox(self) -> None:
        if not hasattr(self, "overlay_order_listbox"):
            return
        lb = self.overlay_order_listbox

        selected_indices = lb.curselection()
        selected_items = [lb.get(i) for i in selected_indices]

        cfg = self.config.inference
        cfg.overlay_order = sanitize_order(cfg.overlay_order)

        lb.delete(0, tk.END)
        for key in cfg.overlay_order:
            label = OVERLAY_DISPLAY_NAMES.get(key, key.replace("_", " ").title())
            lb.insert(tk.END, label)

        for item in selected_items:
            try:
                idx = list(lb.get(0, tk.END)).index(item)
                lb.selection_set(idx)
            except ValueError:
                pass

    def _move_overlay_order(self, direction: int) -> None:
        if not hasattr(self, "overlay_order_listbox"):
            return
        lb = self.overlay_order_listbox
        selection = lb.curselection()
        if not selection:
            return
        idx = selection[0]
        new_idx = idx + direction
        cfg = self.config.inference
        order = sanitize_order(cfg.overlay_order)
        if new_idx < 0 or new_idx >= len(order):
            return
        order[idx], order[new_idx] = order[new_idx], order[idx]
        cfg.overlay_order = order
        self._sync_overlay_order_listbox()
        lb.selection_set(new_idx)
        self._mark_overlay_state_custom()

    def _restore_overlay_order_defaults(self) -> None:
        self.config.inference.overlay_order = DEFAULT_OVERLAY_ORDER.copy()
        self._sync_overlay_order_listbox()
        self._mark_overlay_state_custom()

    def _register_supervision_control(self, widget) -> None:
        if widget is None:
            return
        try:
            default_state = widget.cget("state")
        except tk.TclError:
            default_state = None
        if not default_state:
            if isinstance(widget, ttk.Combobox):
                default_state = "readonly"
            else:
                default_state = tk.NORMAL
        else:
            default_state = str(default_state)
        for existing, _ in self._supervision_control_widgets:
            if existing is widget:
                return
        self._supervision_control_widgets.append((widget, default_state))

    def _set_supervision_controls_state(self, enabled: bool) -> None:
        disabled_state = "disabled"
        for widget, default_state in list(self._supervision_control_widgets):
            if widget is None:
                continue
            try:
                if enabled:
                    state_value = default_state or tk.NORMAL
                else:
                    state_value = disabled_state
                    if isinstance(widget, ttk.Combobox):
                        state_value = "disabled"
                widget.configure(state=state_value)
            except tk.TclError:
                continue

    def _initialize_supervision_model_support(self) -> None:
        var = getattr(self.config.inference, "trained_model_path_infer", None)
        if isinstance(var, tk.StringVar):
            try:
                var.trace_add("write", self._on_inference_model_path_changed)
            except Exception:
                pass
            self._apply_supervision_model_support(var.get())

    def _on_inference_model_path_changed(self, *_):
        var = getattr(self.config.inference, "trained_model_path_infer", None)
        if isinstance(var, tk.StringVar):
            self._apply_supervision_model_support(var.get())

    def _evaluate_supervision_support(self, raw_path: str) -> tuple[bool, str]:
        path = (raw_path or "").strip()
        if not path:
            return True, ""
        suffix = Path(path).suffix.lower()
        if suffix in {".pt", ".pth"}:
            return True, ""
        if not suffix:
            suffix = "unknown"
        message = (
            f"Live overlays and motion metrics require a .pt checkpoint "
            f"(detected '{suffix}'). Load a .pt model to re-enable these options."
        )
        return False, message

    def _disable_supervision_overlay_vars(self) -> None:
        cfg = self.config.inference
        self._overlay_trace_guard = True
        try:
            for attr in self._OVERLAY_BOOL_ATTRS:
                var = getattr(cfg, attr, None)
                if isinstance(var, tk.Variable):
                    var.set(False)
            metrics_var = getattr(cfg, "capture_metrics_var", None)
            if isinstance(metrics_var, tk.Variable):
                metrics_var.set(False)
        finally:
            self._overlay_trace_guard = False
        self._sync_overlay_widget_states()

    def restore_overlay_bools(self, bool_state: dict[str, bool]) -> None:
        """Restore only overlay boolean variables (no presets/metrics)."""
        if not bool_state:
            return
        cfg = self.config.inference
        self._overlay_trace_guard = True
        try:
            for attr, value in bool_state.items():
                var = getattr(cfg, attr, None)
                if isinstance(var, tk.Variable):
                    var.set(bool(value))
        finally:
            self._overlay_trace_guard = False
        self._sync_overlay_widget_states()

    def _apply_supervision_model_support(self, path_value: str) -> None:
        supported, message = self._evaluate_supervision_support(path_value)
        self._supervision_annotations_supported = supported

        if supported:
            self._supervision_notice_var.set("")
            self._set_supervision_controls_state(True)
            if self._supervision_disabled_snapshot:
                snapshot = deepcopy(self._supervision_disabled_snapshot)
                self._supervision_disabled_snapshot = None
                self._overlay_trace_guard = True
                try:
                    self._apply_overlay_state(snapshot, update_selection=False)
                finally:
                    self._overlay_trace_guard = False
        else:
            self._supervision_notice_var.set(message)
            self._set_supervision_controls_state(False)
            if self._supervision_disabled_snapshot is None:
                self._supervision_disabled_snapshot = deepcopy(self._capture_overlay_state())
            self._disable_supervision_overlay_vars()

        self._sync_overlay_widget_states()

    def _register_overlay_widget(self, attr: str, widget) -> None:
        self._overlay_widgets[attr] = widget
        self._register_supervision_control(widget)
        self._sync_overlay_widget_states()

    def _sync_overlay_widget_states(self) -> None:
        if not self._supervision_annotations_supported:
            state = tk.DISABLED
        else:
            state = tk.NORMAL
            if self.config.inference.sv_performance_safe_var.get():
                state = tk.DISABLED
        for attr in self._PERFORMANCE_HEAVY_BOOL_ATTRS:
            widget = self._overlay_widgets.get(attr)
            if widget is not None:
                widget.configure(state=state)

    def _on_performance_safe_changed(self, *_):
        enabled = bool(self.config.inference.sv_performance_safe_var.get())
        heavy_attrs = self._PERFORMANCE_HEAVY_BOOL_ATTRS
        if enabled:
            if self._performance_safe_snapshot is None:
                snapshot = {}
                for attr in heavy_attrs:
                    var = getattr(self.config.inference, attr, None)
                    if isinstance(var, tk.Variable):
                        snapshot[attr] = var.get()
                self._performance_safe_snapshot = snapshot
            self._overlay_trace_guard = True
            try:
                for attr in heavy_attrs:
                    var = getattr(self.config.inference, attr, None)
                    if isinstance(var, tk.Variable):
                        var.set(False)
            finally:
                self._overlay_trace_guard = False
            self.set_performance_warning("Performance-safe mode active; heavy overlays disabled.", color="#007acc")
        else:
            if self._performance_safe_snapshot:
                self._overlay_trace_guard = True
                try:
                    for attr, value in self._performance_safe_snapshot.items():
                        var = getattr(self.config.inference, attr, None)
                        if isinstance(var, tk.Variable):
                            var.set(self._coerce_for_var(var, value))
                finally:
                    self._overlay_trace_guard = False
            self._performance_safe_snapshot = None
            if not self.performance_warning_var.get().startswith("Performance-safe"):
                self.set_performance_warning("")
        self._sync_overlay_widget_states()

    def _reset_overlay_accumulators(self) -> None:
        if self._active_supervision_runner is None:
            self.log_message("No active overlay run to reset.", "INFO")
            return
        self._active_supervision_runner.request_accumulator_reset()
        self.log_message("Requested overlay accumulator reset.", "INFO")

    def set_performance_warning(self, text: str, color: str = "#c25700") -> None:
        self.performance_warning_var.set(text)
        if hasattr(self, 'warning_label'):
            self.warning_label.config(foreground=color)

    def _handle_performance_metrics(self, payload: dict) -> None:
        def _update():
            fps = payload.get("fps")
            frame_ms = payload.get("frame_ms")
            overlay_ms = payload.get("overlay_ms")
            if fps is not None and frame_ms is not None and overlay_ms is not None:
                self.performance_metrics_var.set(f"{fps:.1f} FPS | frame {frame_ms:.1f} ms | overlay {overlay_ms:.1f} ms")
            if self.config.inference.sv_performance_safe_var.get():
                return
            heavy = bool(payload.get("heavy"))
            if heavy:
                self.set_performance_warning("Heavy overlay load detected - consider performance-safe mode.", color="#c25700")
            elif not self.performance_warning_var.get().startswith("Performance-safe"):
                self.set_performance_warning("")
        self.root.after(0, _update)

    def _initialize_inference_overlay_controls(self) -> None:
        self._refresh_overlay_preset_catalog()
        self._sync_overlay_order_listbox()
        self._sync_overlay_widget_states()

    def _open_overlay_settings_dialog(self) -> None:
        existing = getattr(self, "_overlay_settings_window", None)
        if existing is not None:
            try:
                existing.deiconify()
                existing.lift()
                return
            except tk.TclError:
                self._overlay_settings_window = None

        win = tk.Toplevel(self.root)
        win.title("Advanced Overlay Settings")
        # Allow resizing so the scrollable wrapper below has room to grow
        # or shrink with the user's display. Previously fixed-size which
        # could clip notebook tab content on small screens.
        win.resizable(True, True)
        self._overlay_settings_window = win

        def _close_dialog() -> None:
            self._overlay_settings_window = None
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", _close_dialog)

        # Wrap the notebook in a scrollable section so overflow content
        # in any tab remains reachable when the dialog is smaller than the
        # natural notebook size.
        scroll_parent = ttk.Frame(win)
        scroll_parent.grid(row=0, column=0, sticky="nsew")
        win.columnconfigure(0, weight=1)
        win.rowconfigure(0, weight=1)

        _overlay_canvas, main_frame = create_scrollable_section(self.root, scroll_parent)
        main_frame.configure(padding=12)

        notebook = ttk.Notebook(main_frame)
        notebook.pack(expand=True, fill="both", pady=5)

        cfg = self.config.inference

        def _validate_int(value_if_allowed: str, min_val: int, max_val: int) -> bool:
            if not value_if_allowed:
                return True
            try:
                val = int(value_if_allowed)
                return min_val <= val <= max_val
            except ValueError:
                return False

        def _validate_float(value_if_allowed: str, min_val: float, max_val: float) -> bool:
            if not value_if_allowed:
                return True
            try:
                val = float(value_if_allowed)
                return min_val <= val <= max_val
            except ValueError:
                return False

        def add_spin_in_tab(tab, row: int, label: str, var, *, from_, to, increment, validation_cmd):
            ttk.Label(tab, text=label).grid(row=row, column=0, sticky='w', padx=(0, 12), pady=6)
            spin = ttk.Spinbox(
                tab,
                from_=from_,
                to=to,
                increment=increment,
                textvariable=var,
                width=10,
                validate='key',
                validatecommand=validation_cmd,
            )
            spin.grid(row=row, column=1, sticky='ew', pady=6)

        blur_tab = ttk.Frame(notebook, padding=10)
        heatmap_tab = ttk.Frame(notebook, padding=10)
        skeleton_tab = ttk.Frame(notebook, padding=10)
        trace_tab = ttk.Frame(notebook, padding=10)
        other_tab = ttk.Frame(notebook, padding=10)

        notebook.add(blur_tab, text="Blur & Pixelate")
        notebook.add(heatmap_tab, text="Heatmap")
        notebook.add(skeleton_tab, text="Skeleton & Keypoints")
        notebook.add(trace_tab, text="Trace")
        notebook.add(other_tab, text="Other")

        vcmd_blur_kernel = (self.root.register(lambda v: _validate_int(v, 1, 101)), '%P')
        vcmd_pixelate_size = (self.root.register(lambda v: _validate_int(v, 1, 100)), '%P')
        add_spin_in_tab(blur_tab, 0, "Blur kernel size (odd):", cfg.sv_blur_kernel_var, from_=1, to=101, increment=2, validation_cmd=vcmd_blur_kernel)
        add_spin_in_tab(blur_tab, 1, "Pixelate size:", cfg.sv_pixelate_size_var, from_=1, to=100, increment=1, validation_cmd=vcmd_pixelate_size)

        vcmd_radius = (self.root.register(lambda v: _validate_int(v, 1, 200)), '%P')
        vcmd_kernel = (self.root.register(lambda v: _validate_int(v, 1, 101)), '%P')
        vcmd_opacity = (self.root.register(lambda v: _validate_float(v, 0.0, 1.0)), '%P')
        add_spin_in_tab(heatmap_tab, 0, "Heatmap radius:", cfg.sv_heatmap_radius_var, from_=1, to=200, increment=1, validation_cmd=vcmd_radius)
        add_spin_in_tab(heatmap_tab, 1, "Heatmap kernel size:", cfg.sv_heatmap_kernel_var, from_=1, to=101, increment=2, validation_cmd=vcmd_kernel)
        add_spin_in_tab(heatmap_tab, 2, "Heatmap opacity:", cfg.sv_heatmap_opacity_var, from_=0.0, to=1.0, increment=0.05, validation_cmd=vcmd_opacity)
        add_spin_in_tab(heatmap_tab, 3, "Heatmap decay (0=none):", cfg.sv_heatmap_decay_var, from_=0.0, to=1.0, increment=0.05, validation_cmd=vcmd_opacity)

        heatmap_source_label = ttk.Label(heatmap_tab, text="Heatmap source:")
        heatmap_source_label.grid(row=4, column=0, sticky='w', padx=(0, 12), pady=6)
        heatmap_source_combo = ttk.Combobox(
            heatmap_tab,
            state='readonly',
            values=("bbox", "keypoint"),
            textvariable=cfg.sv_heatmap_source_var,
            width=18,
        )
        heatmap_source_combo.grid(row=4, column=1, sticky='ew', pady=6)

        heatmap_anchor_label = ttk.Label(heatmap_tab, text="Bounding box anchor:")
        heatmap_anchor_label.grid(row=5, column=0, sticky='w', padx=(0, 12), pady=6)
        anchor_values = (
            "CENTER",
            "CENTER_OF_MASS",
            "TOP_CENTER",
            "BOTTOM_CENTER",
            "CENTER_LEFT",
            "CENTER_RIGHT",
            "TOP_LEFT",
            "TOP_RIGHT",
            "BOTTOM_LEFT",
            "BOTTOM_RIGHT",
        )
        heatmap_anchor_combo = ttk.Combobox(
            heatmap_tab,
            state='readonly',
            values=anchor_values,
            textvariable=cfg.sv_heatmap_anchor_var,
            width=18,
        )
        heatmap_anchor_combo.grid(row=5, column=1, sticky='ew', pady=6)

        heatmap_keypoint_label = ttk.Label(heatmap_tab, text="Keypoint index:")
        heatmap_keypoint_label.grid(row=6, column=0, sticky='w', padx=(0, 12), pady=6)
        vcmd_keypoint = (self.root.register(lambda v: _validate_int(v, 0, 999)), '%P')
        heatmap_keypoint_spin = ttk.Spinbox(
            heatmap_tab,
            from_=0,
            to=999,
            increment=1,
            textvariable=cfg.sv_heatmap_keypoint_index_var,
            width=10,
            validate='key',
            validatecommand=vcmd_keypoint,
            state='disabled',
        )
        heatmap_keypoint_spin.grid(row=6, column=1, sticky='ew', pady=6)

        def _update_heatmap_controls(*_):
            source = (cfg.sv_heatmap_source_var.get() or 'bbox').lower()
            if source == 'keypoint':
                heatmap_anchor_combo.configure(state='disabled')
                heatmap_keypoint_spin.configure(state='normal')
            else:
                heatmap_anchor_combo.configure(state='readonly')
                heatmap_keypoint_spin.configure(state='disabled')

        cfg.sv_heatmap_source_var.trace_add('write', lambda *_: _update_heatmap_controls())
        heatmap_source_combo.bind('<<ComboboxSelected>>', lambda _event: _update_heatmap_controls())
        _update_heatmap_controls()

        palette_values = ("jet", "summer", "autumn", "winter", "spring", "cool", "hsv", "turbo", "bone", "hot", "pink", "solid")
        vcmd_vertex_radius = (self.root.register(lambda v: _validate_int(v, 1, 25)), '%P')
        vcmd_edge_thickness = (self.root.register(lambda v: _validate_int(v, 1, 12)), '%P')
        add_spin_in_tab(skeleton_tab, 0, "Keypoint marker size:", cfg.sv_vertex_radius_var, from_=1, to=25, increment=1, validation_cmd=vcmd_vertex_radius)
        add_spin_in_tab(skeleton_tab, 1, "Skeleton edge thickness:", cfg.sv_edge_thickness_var, from_=1, to=12, increment=1, validation_cmd=vcmd_edge_thickness)

        ttk.Label(skeleton_tab, text="Keypoint palette:").grid(row=2, column=0, sticky='w', padx=(0, 12), pady=6)
        keypoint_palette_combo = ttk.Combobox(
            skeleton_tab,
            state='readonly',
            values=palette_values,
            textvariable=cfg.sv_keypoint_palette_var,
            width=18,
        )
        keypoint_palette_combo.grid(row=2, column=1, sticky='ew', pady=6)

        ttk.Label(skeleton_tab, text="Skeleton edge palette:").grid(row=3, column=0, sticky='w', padx=(0, 12), pady=6)
        edge_palette_combo = ttk.Combobox(
            skeleton_tab,
            state='readonly',
            values=palette_values,
            textvariable=cfg.sv_edge_palette_var,
            width=18,
        )
        edge_palette_combo.grid(row=3, column=1, sticky='ew', pady=6)

        ttk.Label(skeleton_tab, text="Solid fallback color:").grid(row=4, column=0, sticky='w', padx=(0, 12), pady=6)
        skeleton_color_combo = ttk.Combobox(
            skeleton_tab,
            state='readonly',
            values=("red", "green", "blue", "yellow", "cyan", "magenta", "white"),
            textvariable=cfg.sv_skeleton_color_var,
            width=18,
        )
        skeleton_color_combo.grid(row=4, column=1, sticky='ew', pady=6)
        CreateToolTip(
            skeleton_color_combo,
            "Used when the palette is set to 'solid'. Otherwise the selected colormap drives node or edge colors."
        )

        vcmd_trace_len = (self.root.register(lambda v: _validate_int(v, 1, 500)), '%P')
        vcmd_trace_thick = (self.root.register(lambda v: _validate_int(v, 1, 10)), '%P')
        vcmd_trace_opacity = (self.root.register(lambda v: _validate_float(v, 0.0, 1.0)), '%P')
        add_spin_in_tab(trace_tab, 0, "Trace persistence (frames):", cfg.sv_trace_length_var, from_=1, to=500, increment=5, validation_cmd=vcmd_trace_len)
        add_spin_in_tab(trace_tab, 1, "Trace thickness:", cfg.sv_trace_thickness_var, from_=1, to=10, increment=1, validation_cmd=vcmd_trace_thick)
        add_spin_in_tab(trace_tab, 2, "Trace opacity:", cfg.sv_trace_opacity_var, from_=0.0, to=1.0, increment=0.05, validation_cmd=vcmd_trace_opacity)

        persistent_trace_check = ttk.Checkbutton(
            trace_tab,
            text="Persist trace until reset/end of run",
            variable=cfg.sv_trace_persistent_var,
        )
        persistent_trace_check.grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 8))
        CreateToolTip(
            persistent_trace_check,
            "Keep the full trace history for each tracked object until you reset overlays or the run ends. "
            "Turn this off to keep only the most recent N frames."
        )

        trace_source_label = ttk.Label(trace_tab, text="Trace anchor:")
        trace_source_label.grid(row=4, column=0, sticky='w', padx=(0, 12), pady=6)
        trace_source_combo = ttk.Combobox(
            trace_tab,
            state='readonly',
            values=("bbox", "keypoint"),
            textvariable=cfg.sv_trace_source_var,
            width=18,
        )
        trace_source_combo.grid(row=4, column=1, sticky='ew', pady=6)

        keypoint_label = ttk.Label(trace_tab, text="Keypoint (for trace):")
        keypoint_label.grid(row=5, column=0, sticky='w', padx=(0, 12), pady=6)
        keypoint_choices = getattr(self.app, "keypoint_names", []) or []
        if not keypoint_choices:
            setup_names_var = getattr(self.config.setup, "keypoint_names_str", None)
            if setup_names_var:
                try:
                    parsed = [n.strip() for n in setup_names_var.get().split(",") if n.strip()]
                    keypoint_choices = parsed
                except Exception:
                    keypoint_choices = []
        trace_keypoint_combo = ttk.Combobox(
            trace_tab,
            state="readonly" if keypoint_choices else "disabled",
            values=keypoint_choices,
            textvariable=cfg.sv_trace_keypoint_var,
            width=18,
        )
        trace_keypoint_combo.grid(row=5, column=1, sticky='ew', pady=6)

        color_label = ttk.Label(trace_tab, text="Trace color (name or #hex):")
        color_label.grid(row=6, column=0, sticky='w', padx=(0, 12), pady=6)
        color_frame = ttk.Frame(trace_tab)
        color_frame.grid(row=6, column=1, sticky='ew', pady=6)
        color_frame.columnconfigure(0, weight=1)
        trace_color_entry = ttk.Entry(color_frame, textvariable=cfg.sv_trace_color_var)
        trace_color_entry.grid(row=0, column=0, sticky='ew', padx=(0, 6))

        def _pick_trace_color():
            initial = (cfg.sv_trace_color_var.get() or "").strip() or "#38a3ff"
            selected = colorchooser.askcolor(color=initial, parent=win, title="Choose Trace Color")
            hex_value = selected[1] if selected and len(selected) > 1 else None
            if hex_value:
                cfg.sv_trace_color_var.set(hex_value)

        ttk.Button(color_frame, text="Pick...", command=_pick_trace_color).grid(row=0, column=1, sticky='e', padx=(0, 6))
        ttk.Button(color_frame, text="Use palette", command=lambda: cfg.sv_trace_color_var.set("")).grid(row=0, column=2, sticky='e')
        CreateToolTip(
            trace_color_entry,
            "Optional fixed color for traces. Leave blank to use the track/behavior palette. "
            "Named colors like 'red' and hex values like '#38A3FF' are supported."
        )

        def _update_trace_controls(*_):
            source = (cfg.sv_trace_source_var.get() or "bbox").lower()
            if source == "keypoint":
                trace_keypoint_combo.configure(state="readonly" if keypoint_choices else "disabled")
            else:
                trace_keypoint_combo.configure(state="disabled")

        cfg.sv_trace_source_var.trace_add('write', lambda *_: _update_trace_controls())
        trace_source_combo.bind('<<ComboboxSelected>>', lambda _event: _update_trace_controls())
        _update_trace_controls()

        vcmd_halo_kernel = (self.root.register(lambda v: _validate_int(v, 1, 200)), '%P')
        vcmd_opacity_other = (self.root.register(lambda v: _validate_float(v, 0.0, 1.0)), '%P')
        add_spin_in_tab(other_tab, 0, "Halo kernel size:", cfg.sv_halo_kernel_var, from_=1, to=200, increment=1, validation_cmd=vcmd_halo_kernel)
        add_spin_in_tab(other_tab, 1, "Halo opacity:", cfg.sv_halo_opacity_var, from_=0.0, to=1.0, increment=0.05, validation_cmd=vcmd_opacity_other)
        add_spin_in_tab(other_tab, 2, "Background opacity:", cfg.sv_background_opacity_var, from_=0.0, to=1.0, increment=0.05, validation_cmd=vcmd_opacity_other)

        background_force = ttk.Checkbutton(
            other_tab,
            text="Force background overlay to use boxes",
            variable=cfg.sv_background_force_box_var,
        )
        background_force.grid(row=3, column=0, columnspan=2, sticky='w', pady=(12, 4))

        ttk.Button(main_frame, text="Close", command=_close_dialog).pack(pady=(10, 0))
