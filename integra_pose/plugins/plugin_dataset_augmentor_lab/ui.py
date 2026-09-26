from __future__ import annotations

import os
import subprocess
import threading
import traceback
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from integra_pose.gui.plugin_chrome import apply_plugin_chrome, build_plugin_header
from integra_pose.gui.scrollable import create_scrollable_section
from integra_pose.gui.tooltips import CreateToolTip
from integra_pose.gui.windowing import apply_adaptive_window_geometry
from integra_pose.plugins.plugin_dataset_augmentor_lab.core import (
    ArtifactSettings,
    AugmentationRecipe,
    DatasetAugmentorCoreError,
    DomainShiftSettings,
    TransformSettings,
    run_augmentation,
    parse_class_set,
)


from .dataset_io import load_metadata


class DatasetAugmentorLabUIError(DatasetAugmentorCoreError):
    """Raised when augmentation pipeline input is invalid."""


class DatasetAugmentorLabWindow(tk.Toplevel):
    _BBOX_EPSILON = 1e-6

    def __init__(self, main_app, parent: Optional[tk.Misc] = None) -> None:
        super().__init__(parent)
        self.main_app = main_app
        self.title("Dataset Augmentor Lab")
        apply_adaptive_window_geometry(
            self,
            preferred_size=(1040, 760),
            min_size=(900, 640),
            width_ratio=0.94,
            height_ratio=0.92,
        )
        self.style = apply_plugin_chrome(self, main_app)

        self._worker_lock = threading.Lock()
        self._tooltips: list[CreateToolTip] = []
        self._build_ui()
        self._append_log("Dataset Augmentor Lab ready.")

    def _build_ui(self) -> None:
        # Wrap the lab UI in a scrollable section so the controls plus log
        # stay reachable on smaller displays (the window can be shrunk
        # below the natural content height).
        scroll_parent = ttk.Frame(self)
        scroll_parent.pack(fill="both", expand=True)
        _augmentor_canvas, root = create_scrollable_section(self, scroll_parent)
        root.configure(padding=10)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(6, weight=1)

        header = build_plugin_header(
            root,
            title="Dataset Augmentor Lab",
            summary="Generate augmented pose datasets with class-aware plans, transform controls, artifacts, and appearance variants.",
        )
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self._init_vars()

        path_box = ttk.LabelFrame(root, text="Dataset Paths", padding=10)
        path_box.grid(row=1, column=0, sticky="ew")
        path_box.columnconfigure(1, weight=1)
        self._add_path_row(
            path_box,
            0,
            "Dataset root:",
            self._dataset_root_var,
            lambda: self._browse_dir(self._dataset_root_var),
            "Folder containing images/<split>/ and labels/<split>/.",
        )
        self._add_path_row(
            path_box,
            1,
            "Output root:",
            self._output_root_var,
            lambda: self._browse_dir(self._output_root_var),
            "Destination dataset folder. Leave blank to create a timestamped sibling folder next to the dataset root.",
        )

        options = ttk.Notebook(root)
        options.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self._build_plan_tab(options)
        self._build_transform_tab(options)
        self._build_artifact_tab(options)
        self._build_appearance_tab(options)

        actions = ttk.Frame(root)
        actions.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        self._run_btn = ttk.Button(actions, text="Run Augmentation", command=self._start_augmentation, style="Accent.TButton")
        self._run_btn.pack(side="left")
        self._attach_tooltip(self._run_btn, "Create augmented images, labels, previews, manifest, and recipe files.")
        open_btn = ttk.Button(actions, text="Open Output Folder", command=self._open_output)
        open_btn.pack(side="left", padx=(8, 0))
        self._attach_tooltip(open_btn, "Open the selected output folder in the system file browser.")

        self._progress_var = tk.StringVar(value="Idle")
        ttk.Label(root, textvariable=self._progress_var, style="Status.TLabel").grid(row=4, column=0, sticky="w", pady=(8, 0))

        log_box = ttk.LabelFrame(root, text="Run Log", padding=8)
        log_box.grid(row=5, column=0, sticky="nsew", pady=(8, 0))
        log_box.columnconfigure(0, weight=1)
        log_box.rowconfigure(0, weight=1)
        self._log_text = tk.Text(log_box, height=14, state="disabled", wrap="word")
        self._log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(log_box, command=self._log_text.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self._log_text.configure(yscrollcommand=log_scroll.set)

    def _init_vars(self) -> None:
        self._dataset_root_var = tk.StringVar()
        self._output_root_var = tk.StringVar()
        self._split_var = tk.StringVar(value="train")
        self._plan_var = tk.StringVar(value="balance")
        self._add_count_var = tk.StringVar(value="1000")
        self._target_total_var = tk.StringVar(value="40000")
        self._target_ratio_var = tk.StringVar(value="1.0")
        self._add_per_class_var = tk.StringVar(value="200")
        self._multiplier_var = tk.StringVar(value="1.5")
        self._include_classes_var = tk.StringVar(value="all")
        self._exclude_classes_var = tk.StringVar(value="")
        self._preview_count_var = tk.StringVar(value="50")
        self._max_total_augs_var = tk.StringVar(value="20000")
        self._max_augs_per_source_image_var = tk.StringVar(value="200")
        self._min_visibility_var = tk.StringVar(value="0.25")
        self._copy_originals_var = tk.BooleanVar(value=True)
        self._seed_var = tk.StringVar(value="42")

        self._horizontal_flip_prob_var = tk.StringVar(value="0.5")
        self._affine_prob_var = tk.StringVar(value="0.9")
        self._scale_min_var = tk.StringVar(value="0.90")
        self._scale_max_var = tk.StringVar(value="1.10")
        self._translate_percent_var = tk.StringVar(value="0.08")
        self._rotate_degrees_var = tk.StringVar(value="12")
        self._shear_degrees_var = tk.StringVar(value="6")
        self._brightness_contrast_prob_var = tk.StringVar(value="0.5")
        self._hue_saturation_prob_var = tk.StringVar(value="0.4")
        self._blur_prob_var = tk.StringVar(value="0.15")
        self._gauss_noise_prob_var = tk.StringVar(value="0.2")

        self._frame_noise_prob_var = tk.StringVar(value="0.0")
        self._frame_noise_std_var = tk.StringVar(value="12.0")
        self._frame_banding_prob_var = tk.StringVar(value="0.0")
        self._frame_banding_amp_var = tk.StringVar(value="12.0")
        self._regional_noise_prob_var = tk.StringVar(value="0.0")
        self._regional_noise_std_var = tk.StringVar(value="28.0")
        self._occlusion_prob_var = tk.StringVar(value="0.0")
        self._occlusion_count_var = tk.StringVar(value="2")
        self._occlusion_min_frac_var = tk.StringVar(value="0.04")
        self._occlusion_max_frac_var = tk.StringVar(value="0.18")
        self._occlusion_mode_var = tk.StringVar(value="random")

        self._red_light_var = tk.BooleanVar(value=False)
        self._invert_colors_var = tk.BooleanVar(value=False)
        self._bw_var = tk.BooleanVar(value=False)
        self._red_boost_var = tk.StringVar(value="1.5")
        self._bg_suppress_var = tk.StringVar(value="0.2")
        self._invert_intensity_var = tk.StringVar(value="1.0")
        self._bw_prob_var = tk.StringVar(value="0.5")

    def _build_plan_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=10)
        notebook.add(tab, text="Plan")
        for col in (1, 3):
            tab.columnconfigure(col, weight=1)

        self._combo_row(
            tab,
            0,
            "Plan:",
            self._plan_var,
            ("target_total", "balance", "add", "scale", "balance_add", "balance_scale", "random"),
            "How to choose source images and decide how many labeled instances to add per class.",
        )
        self._entry_row(tab, 0, 2, "Split:", self._split_var, "Choose train, val, test, all, or a custom split. Unselected splits are copied unchanged. Keep an untouched holdout for evaluation.")
        self._entry_row(tab, 1, 0, "Random sample count:", self._add_count_var, "Random-plan object quotas; use target_total for an exact final image count.")
        self._entry_row(tab, 1, 2, "Target ratio:", self._target_ratio_var, "For balance plans, target each selected class to this fraction of the largest class.")
        self._entry_row(tab, 2, 0, "Add per class:", self._add_per_class_var, "For add plans, add this many labeled instances for each selected class.")
        self._entry_row(tab, 2, 2, "Multiplier:", self._multiplier_var, "For scale plans, grow each selected class to original count times this multiplier.")
        self._entry_row(tab, 3, 0, "Include classes:", self._include_classes_var, "Comma-separated class IDs to augment, or 'all'.")
        self._entry_row(tab, 3, 2, "Exclude classes:", self._exclude_classes_var, "Comma-separated class IDs to leave unchanged.")
        self._entry_row(tab, 4, 0, "Preview count:", self._preview_count_var, "Number of previews showing boxes and named keypoints; optional YAML skeleton edges are drawn.")
        self._entry_row(tab, 4, 2, "Max total augments:", self._max_total_augs_var, "Hard cap on generated augmented samples for this run.")
        self._entry_row(tab, 5, 0, "Max per source:", self._max_augs_per_source_image_var, "Prevents one source image from dominating the augmented dataset.")
        self._entry_row(tab, 5, 2, "Min box visibility:", self._min_visibility_var, "Minimum bbox visibility Albumentations requires before keeping a transformed box.")
        self._entry_row(tab, 6, 0, "Seed:", self._seed_var, "Random seed for reproducible source selection and stochastic transforms.")
        copy_check = ttk.Checkbutton(tab, text="Copy originals to output", variable=self._copy_originals_var)
        copy_check.grid(
            row=6, column=2, columnspan=2, sticky="w", padx=(8, 0), pady=(8, 0)
        )
        self._attach_tooltip(copy_check, "Copy source images and labels into the output dataset before adding augmented samples.")
        self._entry_row(tab, 7, 0, "Final image count:", self._target_total_var,
                        "target_total only: total images across selected splits, including originals and backgrounds. Requires Copy originals.")
        behavior_btn = ttk.Button(tab, text="Choose behaviors by name...", command=self._choose_behaviors)
        behavior_btn.grid(row=7, column=2, columnspan=2, sticky="ew", padx=(8, 0), pady=4)
        ttk.Label(tab, text="Train is the default. Other splits may be selected explicitly; derivatives stay with their source.",
                  wraplength=850).grid(row=8, column=0, columnspan=4, sticky="w", pady=4)


    def _build_transform_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=10)
        notebook.add(tab, text="Transforms")
        for col in (1, 3):
            tab.columnconfigure(col, weight=1)

        self._entry_row(tab, 0, 0, "Horizontal flip p:", self._horizontal_flip_prob_var, "Probability of mirroring an image horizontally.")
        self._entry_row(tab, 0, 2, "Affine p:", self._affine_prob_var, "Probability of applying scale, translate, rotate, and shear together.")
        self._entry_row(tab, 1, 0, "Scale min:", self._scale_min_var, "Lower bound for random affine scale.")
        self._entry_row(tab, 1, 2, "Scale max:", self._scale_max_var, "Upper bound for random affine scale.")
        self._entry_row(tab, 2, 0, "Translate %:", self._translate_percent_var, "Maximum random image translation as a fraction of image size.")
        self._entry_row(tab, 2, 2, "Rotate degrees:", self._rotate_degrees_var, "Maximum clockwise/counterclockwise affine rotation.")
        self._entry_row(tab, 3, 0, "Shear degrees:", self._shear_degrees_var, "Maximum affine shear angle.")
        self._entry_row(tab, 3, 2, "Brightness/contrast p:", self._brightness_contrast_prob_var, "Probability of random brightness and contrast changes.")
        self._entry_row(tab, 4, 0, "Hue/saturation p:", self._hue_saturation_prob_var, "Probability of random hue and saturation shifts.")
        self._entry_row(tab, 4, 2, "Blur p:", self._blur_prob_var, "Probability of applying a small Gaussian blur.")
        self._entry_row(tab, 5, 0, "Gaussian noise p:", self._gauss_noise_prob_var, "Probability of Albumentations Gaussian noise in the transform pipeline.")

    def _build_artifact_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=10)
        notebook.add(tab, text="Artifacts")
        for col in (1, 3):
            tab.columnconfigure(col, weight=1)

        self._entry_row(tab, 0, 0, "Frame noise p:", self._frame_noise_prob_var, "Probability of adding whole-frame Gaussian sensor noise.")
        self._entry_row(tab, 0, 2, "Frame noise std:", self._frame_noise_std_var, "Standard deviation for whole-frame Gaussian noise.")
        self._entry_row(tab, 1, 0, "Banding p:", self._frame_banding_prob_var, "Probability of adding horizontal stripe/banding artifacts.")
        self._entry_row(tab, 1, 2, "Banding amp:", self._frame_banding_amp_var, "Brightness amplitude of synthetic banding artifacts.")
        self._entry_row(tab, 2, 0, "Regional noise p:", self._regional_noise_prob_var, "Probability of adding Gaussian noise to random rectangular regions.")
        self._entry_row(tab, 2, 2, "Regional noise std:", self._regional_noise_std_var, "Standard deviation for random regional noise patches.")
        self._entry_row(tab, 3, 0, "Occlusion p:", self._occlusion_prob_var, "Probability of drawing artificial rectangular occluders.")
        self._entry_row(tab, 3, 2, "Occlusion count:", self._occlusion_count_var, "Maximum number of occluder or noisy patches per affected image.")
        self._entry_row(tab, 4, 0, "Patch min frac:", self._occlusion_min_frac_var, "Minimum patch side length as a fraction of image size.")
        self._entry_row(tab, 4, 2, "Patch max frac:", self._occlusion_max_frac_var, "Maximum patch side length as a fraction of image size.")
        self._combo_row(tab, 5, "Occlusion mode:", self._occlusion_mode_var, ("random", "black", "gray", "noise"), "Fill style for artificial occluders.")

    def _build_appearance_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=10)
        notebook.add(tab, text="Appearance")
        for col in (1, 3):
            tab.columnconfigure(col, weight=1)

        red_check = ttk.Checkbutton(tab, text="Red light tint", variable=self._red_light_var)
        red_check.grid(row=0, column=0, sticky="w")
        self._attach_tooltip(red_check, "Tint images toward red and suppress blue/green channels.")
        self._entry_row(tab, 0, 2, "Red boost:", self._red_boost_var, "Multiplier applied to the red channel when red light tint is enabled.")
        self._entry_row(tab, 1, 0, "Background suppress:", self._bg_suppress_var, "Multiplier applied to blue and green channels when red light tint is enabled.")
        invert_check = ttk.Checkbutton(tab, text="Invert colors", variable=self._invert_colors_var)
        invert_check.grid(row=2, column=0, sticky="w", pady=(8, 0))
        self._attach_tooltip(invert_check, "Invert image brightness to simulate high-contrast appearance changes.")
        self._entry_row(tab, 2, 2, "Invert intensity:", self._invert_intensity_var, "Blend strength for color inversion; 1.0 is full inversion.")
        gray_check = ttk.Checkbutton(tab, text="Grayscale", variable=self._bw_var)
        gray_check.grid(row=3, column=0, sticky="w", pady=(8, 0))
        self._attach_tooltip(gray_check, "Randomly convert affected images to grayscale.")
        self._entry_row(tab, 3, 2, "Grayscale probability:", self._bw_prob_var, "Probability of grayscale conversion when enabled.")

    def _entry_row(self, parent, row: int, col: int, label: str, var: tk.StringVar, tooltip: str | None = None) -> None:
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=col, sticky="w", padx=(0 if col == 0 else 14, 4), pady=4)
        entry = ttk.Entry(parent, textvariable=var, width=14)
        entry.grid(row=row, column=col + 1, sticky="ew", padx=(0, 8), pady=4)
        self._attach_tooltip(lbl, tooltip)
        self._attach_tooltip(entry, tooltip)

    def _combo_row(self, parent, row: int, label: str, var: tk.StringVar, values: Sequence[str], tooltip: str | None = None) -> None:
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=0, sticky="w", pady=4)
        combo = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=18)
        combo.grid(row=row, column=1, sticky="ew", padx=(0, 8), pady=4)
        self._attach_tooltip(lbl, tooltip)
        self._attach_tooltip(combo, tooltip)

    def _attach_tooltip(self, widget, text: str | None) -> None:
        if text:
            self._tooltips.append(CreateToolTip(widget, text))

    def _add_path_row(self, parent, row: int, label: str, var: tk.StringVar, browse_fn, tooltip: str) -> None:
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=0, sticky="w")
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", padx=(8, 8), pady=2)
        button = ttk.Button(parent, text="Browse...", command=browse_fn)
        button.grid(row=row, column=2, sticky="e")
        self._attach_tooltip(lbl, tooltip)
        self._attach_tooltip(entry, tooltip)
        self._attach_tooltip(button, f"Browse for {label.rstrip(':').lower()}.")

    def _browse_dir(self, var: tk.StringVar) -> None:
        start = var.get().strip() or str(Path.cwd())
        selected = filedialog.askdirectory(parent=self, initialdir=start)
        if selected:
            var.set(selected)

    def _append_log(self, message: str) -> None:
        if self.main_app is not None:
            try:
                self.main_app.log_message(message, "INFO")
            except Exception:
                pass
        self._log_text.configure(state="normal")
        self._log_text.insert("end", message + "\n")
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def _append_error(self, message: str) -> None:
        if self.main_app is not None:
            try:
                self.main_app.log_message(message, "ERROR")
            except Exception:
                pass
        self._append_log(f"[ERROR] {message}")

    def _open_output(self) -> None:
        value = self._output_root_var.get().strip()
        if not value:
            return
        path = Path(value)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
            else:
                import sys

                if sys.platform == "darwin":
                    subprocess.Popen(["open", str(path)])
                else:
                    subprocess.Popen(["xdg-open", str(path)])
        except Exception:
            pass

    def _start_augmentation(self) -> None:
        if self._worker_lock.locked():
            self._append_log("Augmentation is already running.")
            return

        try:
            recipe = self._collect_recipe()
        except Exception as exc:
            message = str(exc)
            self._append_error(message)
            messagebox.showerror("Dataset Augmentor Lab", message, parent=self)
            return

        self._run_btn.configure(state="disabled")
        self._progress_var.set("Running augmentation...")

        def worker() -> None:
            with self._worker_lock:
                try:
                    result = self._run_augmentation_pipeline(recipe)
                    self.after(
                        0,
                        lambda path=result.output_root: messagebox.showinfo(
                            "Dataset Augmentor Lab",
                            f"Augmentation complete.\n{path}",
                            parent=self,
                        ),
                    )
                except Exception as exc:
                    err_text = str(exc)
                    tb_text = traceback.format_exc()
                    self.after(0, lambda msg=err_text: self._append_error(msg))
                    self.after(0, lambda tb=tb_text: self._append_error(tb))
                    self.after(0, lambda msg=err_text: messagebox.showerror("Dataset Augmentor Lab", msg, parent=self))
                finally:
                    self.after(0, lambda: self._run_btn.configure(state="normal"))
                    self.after(0, lambda: self._progress_var.set("Idle"))

        threading.Thread(target=worker, daemon=True).start()

    def _choose_behaviors(self) -> None:
        try:
            metadata = load_metadata(Path(self._dataset_root_var.get().strip()))
            names = metadata.get("names")
            if not names:
                raise ValueError("Choose a dataset with a YAML containing behavior names first.")
            included = parse_class_set(self._include_classes_var.get())
            excluded = parse_class_set(self._exclude_classes_var.get()) or set()
        except (ValueError, OSError) as exc:
            messagebox.showerror("Choose behaviors", str(exc), parent=self)
            return
        dialog = tk.Toplevel(self)
        dialog.title("Behaviors to augment")
        dialog.transient(self)
        dialog.geometry("440x460")
        body = ttk.Frame(dialog, padding=10)
        body.pack(fill="both", expand=True)
        _, content = create_scrollable_section(dialog, body)
        variables = {}
        for class_id, name in names.items():
            value = (included is None or class_id in included) and class_id not in excluded
            variable = tk.BooleanVar(value=value)
            variables[class_id] = variable
            ttk.Checkbutton(content, text=f"{class_id}: {name}", variable=variable).pack(anchor="w", pady=3)
        actions = ttk.Frame(dialog, padding=8)
        actions.pack(fill="x")
        def apply():
            selected = [str(i) for i, variable in variables.items() if variable.get()]
            self._include_classes_var.set(",".join(selected) if selected else "none")
            self._exclude_classes_var.set("")
            dialog.destroy()
        ttk.Button(actions, text="All", command=lambda: [v.set(True) for v in variables.values()]).pack(side="left")
        ttk.Button(actions, text="None", command=lambda: [v.set(False) for v in variables.values()]).pack(side="left")
        ttk.Button(actions, text="Apply", command=apply).pack(side="right")
        dialog.grab_set()

    def _collect_recipe(self) -> AugmentationRecipe:
        dataset_raw = self._dataset_root_var.get().strip()
        if not dataset_raw:
            raise DatasetAugmentorLabUIError("Choose a dataset root before running augmentation.")
        output_raw = self._output_root_var.get().strip()

        try:
            return AugmentationRecipe(
                dataset_root=Path(dataset_raw),
                output_root=Path(output_raw) if output_raw else None,
                split=self._split_var.get().strip() or "train",
                copy_originals=bool(self._copy_originals_var.get()),
                seed=self._int_var(self._seed_var, "Seed", minimum=0),
                plan=self._plan_var.get().strip() or "balance",
                add_count=self._int_var(self._add_count_var, "Random sample count", minimum=1),
                target_total_images=self._int_var(self._target_total_var, "Final image count", minimum=1),
                target_ratio=self._float_var(self._target_ratio_var, "Target ratio", minimum=0.000001, maximum=1.0),
                add_per_class=self._int_var(self._add_per_class_var, "Add per class", minimum=0),
                multiplier=self._float_var(self._multiplier_var, "Multiplier", minimum=1.0),
                include_classes=self._include_classes_var.get().strip() or "all",
                exclude_classes=self._exclude_classes_var.get().strip(),
                preview_count=self._int_var(self._preview_count_var, "Preview count", minimum=0),
                max_total_augs=self._int_var(self._max_total_augs_var, "Max total augments", minimum=1),
                max_augs_per_source_image=self._int_var(self._max_augs_per_source_image_var, "Max per source", minimum=1),
                min_visibility=self._float_var(self._min_visibility_var, "Min box visibility", minimum=0.0, maximum=1.0),
                transforms=TransformSettings(
                    horizontal_flip_prob=self._prob_var(self._horizontal_flip_prob_var, "Horizontal flip probability"),
                    affine_prob=self._prob_var(self._affine_prob_var, "Affine probability"),
                    scale_min=self._float_var(self._scale_min_var, "Scale min", minimum=0.01),
                    scale_max=self._float_var(self._scale_max_var, "Scale max", minimum=0.01),
                    translate_percent=self._float_var(self._translate_percent_var, "Translate percent", minimum=0.0),
                    rotate_degrees=self._float_var(self._rotate_degrees_var, "Rotate degrees", minimum=0.0),
                    shear_degrees=self._float_var(self._shear_degrees_var, "Shear degrees", minimum=0.0),
                    brightness_contrast_prob=self._prob_var(
                        self._brightness_contrast_prob_var,
                        "Brightness/contrast probability",
                    ),
                    hue_saturation_prob=self._prob_var(self._hue_saturation_prob_var, "Hue/saturation probability"),
                    blur_prob=self._prob_var(self._blur_prob_var, "Blur probability"),
                    gauss_noise_prob=self._prob_var(self._gauss_noise_prob_var, "Gaussian noise probability"),
                ),
                artifacts=ArtifactSettings(
                    frame_noise_prob=self._prob_var(self._frame_noise_prob_var, "Frame noise probability"),
                    frame_noise_std=self._float_var(self._frame_noise_std_var, "Frame noise std", minimum=0.0),
                    frame_banding_prob=self._prob_var(self._frame_banding_prob_var, "Banding probability"),
                    frame_banding_amp=self._float_var(self._frame_banding_amp_var, "Banding amplitude", minimum=0.0),
                    regional_noise_prob=self._prob_var(self._regional_noise_prob_var, "Regional noise probability"),
                    regional_noise_std=self._float_var(self._regional_noise_std_var, "Regional noise std", minimum=0.0),
                    occlusion_prob=self._prob_var(self._occlusion_prob_var, "Occlusion probability"),
                    occlusion_count=self._int_var(self._occlusion_count_var, "Occlusion count", minimum=1),
                    occlusion_min_frac=self._float_var(
                        self._occlusion_min_frac_var,
                        "Patch min frac",
                        minimum=0.001,
                        maximum=1.0,
                    ),
                    occlusion_max_frac=self._float_var(
                        self._occlusion_max_frac_var,
                        "Patch max frac",
                        minimum=0.001,
                        maximum=1.0,
                    ),
                    occlusion_mode=self._occlusion_mode_var.get().strip() or "random",
                ),
                domain_shift=DomainShiftSettings(
                    white_mouse=bool(self._invert_colors_var.get()),
                    invert_intensity=self._float_var(self._invert_intensity_var, "Invert intensity", minimum=0.0),
                    red_light=bool(self._red_light_var.get()),
                    red_boost=self._float_var(self._red_boost_var, "Red boost", minimum=0.0),
                    bg_suppress=self._float_var(self._bg_suppress_var, "Background suppress", minimum=0.0),
                    bw=bool(self._bw_var.get()),
                    bw_prob=self._prob_var(self._bw_prob_var, "Grayscale probability"),
                ),
            )
        except ValueError as exc:
            raise DatasetAugmentorLabUIError(str(exc)) from exc

    def _run_augmentation_pipeline(self, recipe: AugmentationRecipe):
        def log(message: str) -> None:
            self.after(0, lambda msg=message: self._append_log(msg))

        def progress(done: int, total: int) -> None:
            self.after(0, lambda d=done, t=total: self._progress_var.set(f"Generated {d}/{t} samples"))

        return run_augmentation(recipe, log=log, progress=progress)

    def _int_var(self, var: tk.StringVar, label: str, *, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
        value = int(var.get().strip() or "0")
        if minimum is not None and value < minimum:
            raise ValueError(f"{label} must be >= {minimum}.")
        if maximum is not None and value > maximum:
            raise ValueError(f"{label} must be <= {maximum}.")
        return value

    def _float_var(
        self,
        var: tk.StringVar,
        label: str,
        *,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ) -> float:
        value = float(var.get().strip() or "0")
        if minimum is not None and value < minimum:
            raise ValueError(f"{label} must be >= {minimum}.")
        if maximum is not None and value > maximum:
            raise ValueError(f"{label} must be <= {maximum}.")
        return value

    def _prob_var(self, var: tk.StringVar, label: str) -> float:
        return self._float_var(var, label, minimum=0.0, maximum=1.0)


__all__ = ["DatasetAugmentorLabUIError", "DatasetAugmentorLabWindow"]
