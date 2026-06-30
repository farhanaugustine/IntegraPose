from __future__ import annotations

import inspect
import os
import random
import shutil
import subprocess
import threading
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
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
)


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
            "Destination dataset folder. Leave blank to create a timestamped run under the dataset root.",
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
        self._split_var = tk.StringVar(value="all")
        self._plan_var = tk.StringVar(value="balance")
        self._add_count_var = tk.StringVar(value="1000")
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
            ("balance", "add", "scale", "balance_add", "balance_scale", "random"),
            "How to choose source images and decide how many labeled instances to add per class.",
        )
        self._entry_row(tab, 0, 2, "Split:", self._split_var, "Use 'all' or a split folder name such as train, val, or test.")
        self._entry_row(tab, 1, 0, "Random sample count:", self._add_count_var, "Used by the random plan as the approximate number of augmented samples to create.")
        self._entry_row(tab, 1, 2, "Target ratio:", self._target_ratio_var, "For balance plans, target each selected class to this fraction of the largest class.")
        self._entry_row(tab, 2, 0, "Add per class:", self._add_per_class_var, "For add plans, add this many labeled instances for each selected class.")
        self._entry_row(tab, 2, 2, "Multiplier:", self._multiplier_var, "For scale plans, grow each selected class to original count times this multiplier.")
        self._entry_row(tab, 3, 0, "Include classes:", self._include_classes_var, "Comma-separated class IDs to augment, or 'all'.")
        self._entry_row(tab, 3, 2, "Exclude classes:", self._exclude_classes_var, "Comma-separated class IDs to leave unchanged.")
        self._entry_row(tab, 4, 0, "Preview count:", self._preview_count_var, "Number of augmented preview images to save with boxes drawn.")
        self._entry_row(tab, 4, 2, "Max total augments:", self._max_total_augs_var, "Hard cap on generated augmented samples for this run.")
        self._entry_row(tab, 5, 0, "Max per source:", self._max_augs_per_source_image_var, "Prevents one source image from dominating the augmented dataset.")
        self._entry_row(tab, 5, 2, "Min box visibility:", self._min_visibility_var, "Minimum bbox visibility Albumentations requires before keeping a transformed box.")
        self._entry_row(tab, 6, 0, "Seed:", self._seed_var, "Random seed for reproducible source selection and stochastic transforms.")
        copy_check = ttk.Checkbutton(tab, text="Copy originals to output", variable=self._copy_originals_var)
        copy_check.grid(
            row=6, column=2, columnspan=2, sticky="w", padx=(8, 0), pady=(8, 0)
        )
        self._attach_tooltip(copy_check, "Copy source images and labels into the output dataset before adding augmented samples.")

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

    def _run_augmentation_pipeline(self, job: Dict[str, object]) -> None:
        try:
            import albumentations as A
        except Exception as exc:
            raise DatasetAugmentorLabUIError(
                "Albumentations is required for augmentation. Install it with: "
                "python tools/install_albumentations_gui.py "
                "(or manually: python -m pip install --no-deps -r requirements-albumentations-gui.txt)"
            ) from exc

        dataset_root = Path(job["dataset_root"])
        if not dataset_root.exists():
            raise DatasetAugmentorLabUIError(f"Dataset root not found: {dataset_root}")

        output_raw = str(job["output_raw"])
        output_root = Path(output_raw) if output_raw else Path(f"{dataset_root}_augmented")
        add_count = int(job["add_count"])
        copy_originals = bool(job["copy_originals"])
        seed = int(job["seed"])
        domain_shift = dict(job["domain_shift"])
        random.seed(seed)

        samples = self._discover_samples(dataset_root)
        if not samples:
            raise DatasetAugmentorLabUIError(
                f"No samples found in {dataset_root}. Expected images/<split>/ and labels/<split>/ layout."
            )
        self.after(0, lambda: self._append_log(f"Discovered {len(samples)} source samples."))

        transform = self._build_transform(A)
        if copy_originals:
            self._copy_originals(samples, output_root)
            self.after(0, lambda: self._append_log("Copied original samples to output root."))

        for i in range(add_count):
            src_img, src_lbl, split_name = random.choice(samples)
            img = cv2.imread(str(src_img))
            if img is None:
                continue
            h, w = img.shape[:2]
            objects = self._read_label_objects(src_lbl, img_w=w, img_h=h)
            bboxes = [obj["bbox"] for obj in objects]
            classes = [obj["class_id"] for obj in objects]
            original_keypoint_counts = [len(obj["keypoints"]) for obj in objects]
            original_vis: List[int] = []
            original_keypoints: List[Tuple[float, float]] = []
            # Pre-compute the (start, length) slice of the flat keypoint
            # arrays that belongs to each input bbox. We need this when a
            # bbox is dropped by the transform — we use it to drop the
            # matching keypoints / visibility entries.
            object_kp_offsets: List[int] = []
            running_offset = 0
            for obj_kp in (obj["keypoints"] for obj in objects):
                object_kp_offsets.append(running_offset)
                for kp in obj_kp:
                    original_keypoints.append((kp[0], kp[1]))
                    original_vis.append(kp[2])
                running_offset += len(obj_kp)

            box_indices = list(range(len(bboxes)))
            result = transform(
                image=img,
                bboxes=bboxes,
                class_labels=classes,
                box_indices=box_indices,
                keypoints=original_keypoints,
            )
            aug_img = result["image"]
            aug_boxes = list(result["bboxes"])
            aug_classes = list(result["class_labels"])
            aug_keypoints = list(result["keypoints"])
            surviving_box_indices = [int(idx) for idx in result.get("box_indices", [])]

            aug_img = self._apply_domain_shift(aug_img, domain_shift)
            aug_h, aug_w = aug_img.shape[:2]

            # Rebuild parallel arrays keyed off which input bboxes survived
            # the transform. Keypoints belonging to dropped bboxes are
            # discarded; keypoints belonging to surviving bboxes keep their
            # transformed coordinates and original visibility (which we
            # reconcile against the post-transform image bounds below).
            keypoint_counts: List[int] = []
            vis_flat: List[int] = []
            keypoints_flat: List[Tuple[float, float]] = []
            for orig_idx in surviving_box_indices:
                if orig_idx < 0 or orig_idx >= len(original_keypoint_counts):
                    continue
                start = object_kp_offsets[orig_idx]
                count = original_keypoint_counts[orig_idx]
                keypoint_counts.append(int(count))
                for k in range(count):
                    flat_idx = start + k
                    if flat_idx >= len(aug_keypoints):
                        # Albumentations should preserve length when
                        # remove_invisible=False, but guard defensively.
                        keypoints_flat.append((0.0, 0.0))
                        vis_flat.append(0)
                        continue
                    kx_px, ky_px = aug_keypoints[flat_idx]
                    original_visibility = (
                        original_vis[flat_idx] if flat_idx < len(original_vis) else 0
                    )
                    # P0-C1: a keypoint that landed outside the augmented
                    # frame must be flagged invisible. Original code clipped
                    # x/y but kept visibility=2, which is a lie about what
                    # the model is being asked to learn.
                    in_frame = (0.0 <= float(kx_px) <= float(aug_w)) and (
                        0.0 <= float(ky_px) <= float(aug_h)
                    )
                    if int(original_visibility) == 0 or not in_frame:
                        new_visibility = 0
                    else:
                        new_visibility = int(original_visibility)
                    keypoints_flat.append((float(kx_px), float(ky_px)))
                    vis_flat.append(new_visibility)

            if len(surviving_box_indices) != len(box_indices):
                dropped = len(box_indices) - len(surviving_box_indices)
                self.after(
                    0,
                    lambda idx=i, d=dropped: self._append_log(
                        f"[WARN] Sample {idx}: {d} bbox(es) dropped by augmentation; "
                        f"associated keypoints removed in lockstep."
                    ),
                )

            out_img_dir = output_root / "images" / split_name
            out_lbl_dir = output_root / "labels" / split_name
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_lbl_dir.mkdir(parents=True, exist_ok=True)

            out_name = f"{src_img.stem}_aug_{i:06d}{src_img.suffix}"
            out_img_path = out_img_dir / out_name
            out_lbl_path = out_lbl_dir / f"{Path(out_name).stem}.txt"
            cv2.imwrite(str(out_img_path), aug_img)
            self._write_label_objects(
                out_lbl_path,
                boxes=aug_boxes,
                classes=aug_classes,
                keypoints=aug_keypoints,
                vis_flat=vis_flat,
                keypoint_counts=keypoint_counts,
                img_w=aug_img.shape[1],
                img_h=aug_img.shape[0],
            )
            if i % 25 == 0 or i == add_count - 1:
                self.after(0, lambda idx=i + 1, total=add_count: self._progress_var.set(f"Generated {idx}/{total} samples"))

        self.after(0, lambda: self._append_log(f"Augmentation complete. Output: {output_root}"))
        self.after(0, lambda: messagebox.showinfo("Dataset Augmentor Lab", f"Augmentation complete.\n{output_root}", parent=self))

    def _discover_samples(self, dataset_root: Path) -> List[Tuple[Path, Path, str]]:
        images_root = dataset_root / "images"
        labels_root = dataset_root / "labels"
        if not images_root.exists() or not labels_root.exists():
            return []
        out: List[Tuple[Path, Path, str]] = []
        for split_dir in sorted(images_root.iterdir()):
            if not split_dir.is_dir():
                continue
            split = split_dir.name
            label_split_dir = labels_root / split
            for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                for img_path in split_dir.glob(pattern):
                    lbl_path = label_split_dir / f"{img_path.stem}.txt"
                    out.append((img_path, lbl_path, split))
        return out

    def _copy_originals(self, samples: Sequence[Tuple[Path, Path, str]], output_root: Path) -> None:
        for src_img, src_lbl, split in samples:
            out_img_dir = output_root / "images" / split
            out_lbl_dir = output_root / "labels" / split
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_lbl_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_img, out_img_dir / src_img.name)
            if src_lbl.exists():
                shutil.copy2(src_lbl, out_lbl_dir / src_lbl.name)

    def _build_transform(self, A):
        noise_kwargs = {"p": 0.2}
        if "var_limit" in inspect.signature(A.GaussNoise).parameters:
            noise_kwargs["var_limit"] = (10.0, 50.0)
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Affine(scale=(0.8, 1.2), translate_percent=(0.0, 0.1), rotate=(-15, 15), shear=(-10, 10), p=0.85),
                A.GaussNoise(**noise_kwargs),
            ],
            # Pass a parallel `box_indices` field so we can identify *which*
            # bboxes survived a transform. Without this, when Albumentations
            # drops a bbox (e.g. translated entirely off-frame), we have no
            # way to know which input bbox went away — and thus which
            # keypoints / visibility flags belong to the dropped object.
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels", "box_indices"], min_visibility=0.0),
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )

    def _apply_domain_shift(self, image: np.ndarray, domain_shift: Dict[str, object]) -> np.ndarray:
        out = image
        if bool(domain_shift.get("white_mouse")):
            intensity = float(domain_shift.get("invert_intensity", 1.0))
            hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            inv_v = 255 - v
            v_final = cv2.addWeighted(inv_v, intensity, v, 1.0 - intensity, 0) if intensity < 1.0 else inv_v
            out = cv2.cvtColor(cv2.merge([h, s, v_final]), cv2.COLOR_HSV2BGR)
        if bool(domain_shift.get("red_light")):
            red_boost = float(domain_shift.get("red_boost", 1.5))
            bg_suppress = float(domain_shift.get("bg_suppress", 0.2))
            b, g, r = cv2.split(out)
            r = np.clip(r.astype(np.float32) * red_boost, 0, 255).astype(np.uint8)
            g = np.clip(g.astype(np.float32) * bg_suppress, 0, 255).astype(np.uint8)
            b = np.clip(b.astype(np.float32) * bg_suppress, 0, 255).astype(np.uint8)
            out = cv2.merge([b, g, r])
        if bool(domain_shift.get("bw")):
            bw_prob = float(domain_shift.get("bw_prob", 0.5))
            if random.random() < max(0.0, min(1.0, bw_prob)):
                gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return out

    def _sanitize_yolo_bbox(self, cx: float, cy: float, bw: float, bh: float) -> List[float]:
        eps = float(self._BBOX_EPSILON)

        cx = float(np.clip(cx, 0.0, 1.0))
        cy = float(np.clip(cy, 0.0, 1.0))
        bw = float(np.clip(bw, 0.0, 1.0))
        bh = float(np.clip(bh, 0.0, 1.0))

        x1 = max(0.0, min(1.0, cx - (bw / 2.0)))
        y1 = max(0.0, min(1.0, cy - (bh / 2.0)))
        x2 = max(0.0, min(1.0, cx + (bw / 2.0)))
        y2 = max(0.0, min(1.0, cy + (bh / 2.0)))

        if x2 <= x1:
            midpoint = float(np.clip((x1 + x2) / 2.0, eps, 1.0 - eps))
            x1 = max(0.0, midpoint - eps)
            x2 = min(1.0, midpoint + eps)
        if y2 <= y1:
            midpoint = float(np.clip((y1 + y2) / 2.0, eps, 1.0 - eps))
            y1 = max(0.0, midpoint - eps)
            y2 = min(1.0, midpoint + eps)

        if x1 <= 0.0:
            x1 = eps
        if y1 <= 0.0:
            y1 = eps
        if x2 >= 1.0:
            x2 = 1.0 - eps
        if y2 >= 1.0:
            y2 = 1.0 - eps

        x1 = min(x1, x2 - eps)
        y1 = min(y1, y2 - eps)

        safe_cx = float(np.clip((x1 + x2) / 2.0, 0.0, 1.0))
        safe_cy = float(np.clip((y1 + y2) / 2.0, 0.0, 1.0))
        safe_bw = float(np.clip(x2 - x1, eps, 1.0))
        safe_bh = float(np.clip(y2 - y1, eps, 1.0))
        return [safe_cx, safe_cy, safe_bw, safe_bh]

    def _read_label_objects(self, label_path: Path, *, img_w: int, img_h: int) -> List[Dict[str, object]]:
        objects: List[Dict[str, object]] = []
        if not label_path.exists():
            return objects
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:5])
            except Exception:
                continue
            kp_payload = parts[5:]
            keypoints: List[Tuple[float, float, int]] = []
            if kp_payload and len(kp_payload) % 3 == 0:
                for idx in range(0, len(kp_payload), 3):
                    try:
                        kx = float(kp_payload[idx]) * img_w
                        ky = float(kp_payload[idx + 1]) * img_h
                        kv = int(float(kp_payload[idx + 2]))
                    except Exception:
                        kx, ky, kv = 0.0, 0.0, 0
                    keypoints.append((kx, ky, kv))
            safe_bbox = self._sanitize_yolo_bbox(cx, cy, bw, bh)
            objects.append({"class_id": class_id, "bbox": safe_bbox, "keypoints": keypoints})
        return objects

    def _write_label_objects(
        self,
        output_path: Path,
        *,
        boxes: Sequence[Sequence[float]],
        classes: Sequence[int],
        keypoints: Sequence[Sequence[float]],
        vis_flat: Sequence[int],
        keypoint_counts: Sequence[int],
        img_w: int,
        img_h: int,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        kp_idx = 0
        vis_idx = 0
        lines: List[str] = []
        for obj_idx, box in enumerate(boxes):
            cls = int(classes[obj_idx]) if obj_idx < len(classes) else 0
            cx, cy, bw, bh = self._sanitize_yolo_bbox(*[float(v) for v in box[:4]])
            row = [f"{cls}", f"{cx:.8f}", f"{cy:.8f}", f"{bw:.8f}", f"{bh:.8f}"]
            kp_count = int(keypoint_counts[obj_idx]) if obj_idx < len(keypoint_counts) else 0
            for _ in range(max(0, kp_count)):
                if kp_idx < len(keypoints):
                    kx_px, ky_px = keypoints[kp_idx]
                    kx = max(0.0, min(1.0, float(kx_px) / max(img_w, 1)))
                    ky = max(0.0, min(1.0, float(ky_px) / max(img_h, 1)))
                    kv = int(vis_flat[vis_idx]) if vis_idx < len(vis_flat) else 2
                    row.extend([f"{kx:.6f}", f"{ky:.6f}", str(kv)])
                    kp_idx += 1
                    vis_idx += 1
                else:
                    row.extend(["0.000000", "0.000000", "0"])
            lines.append(" ".join(row))
        output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _collect_recipe(self) -> AugmentationRecipe:
        dataset_raw = self._dataset_root_var.get().strip()
        if not dataset_raw:
            raise DatasetAugmentorLabUIError("Choose a dataset root before running augmentation.")
        output_raw = self._output_root_var.get().strip()

        try:
            return AugmentationRecipe(
                dataset_root=Path(dataset_raw),
                output_root=Path(output_raw) if output_raw else None,
                split=self._split_var.get().strip() or "all",
                copy_originals=bool(self._copy_originals_var.get()),
                seed=self._int_var(self._seed_var, "Seed", minimum=0),
                plan=self._plan_var.get().strip() or "balance",
                add_count=self._int_var(self._add_count_var, "Random sample count", minimum=1),
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
