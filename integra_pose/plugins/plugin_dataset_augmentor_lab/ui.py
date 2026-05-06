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
from integra_pose.gui.windowing import apply_adaptive_window_geometry


class DatasetAugmentorLabUIError(RuntimeError):
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
            summary="Generate augmented pose datasets and domain-shift variants without burying the core controls under utility clutter.",
        )
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self._dataset_root_var = tk.StringVar()
        self._output_root_var = tk.StringVar()
        self._add_count_var = tk.StringVar(value="1000")
        self._copy_originals_var = tk.BooleanVar(value=True)
        self._red_light_var = tk.BooleanVar(value=False)
        self._white_mouse_var = tk.BooleanVar(value=False)
        self._bw_var = tk.BooleanVar(value=False)
        self._seed_var = tk.StringVar(value="42")
        self._red_boost_var = tk.StringVar(value="1.5")
        self._bg_suppress_var = tk.StringVar(value="0.2")
        self._invert_intensity_var = tk.StringVar(value="1.0")
        self._bw_prob_var = tk.StringVar(value="0.5")

        path_box = ttk.LabelFrame(root, text="Dataset Paths", padding=10)
        path_box.grid(row=1, column=0, sticky="ew")
        path_box.columnconfigure(1, weight=1)
        self._add_path_row(path_box, 0, "Dataset root:", self._dataset_root_var, lambda: self._browse_dir(self._dataset_root_var))
        self._add_path_row(path_box, 1, "Output root:", self._output_root_var, lambda: self._browse_dir(self._output_root_var))

        opts = ttk.LabelFrame(root, text="Augmentation Options", padding=10)
        opts.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(opts, text="Add augmented samples:").grid(row=0, column=0, sticky="w")
        ttk.Entry(opts, textvariable=self._add_count_var, width=10).grid(row=0, column=1, sticky="w", padx=(6, 16))
        ttk.Label(opts, text="Seed:").grid(row=0, column=2, sticky="w")
        ttk.Entry(opts, textvariable=self._seed_var, width=10).grid(row=0, column=3, sticky="w", padx=(6, 16))
        ttk.Checkbutton(opts, text="Copy originals to output", variable=self._copy_originals_var).grid(
            row=0, column=4, sticky="w", padx=(4, 0)
        )

        domain = ttk.LabelFrame(root, text="Domain Shift Controls", padding=10)
        domain.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        ttk.Checkbutton(domain, text="Simulate red light", variable=self._red_light_var).grid(row=0, column=0, sticky="w")
        ttk.Label(domain, text="red_boost").grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Entry(domain, textvariable=self._red_boost_var, width=8).grid(row=0, column=2, sticky="w", padx=(4, 12))
        ttk.Label(domain, text="bg_suppress").grid(row=0, column=3, sticky="w")
        ttk.Entry(domain, textvariable=self._bg_suppress_var, width=8).grid(row=0, column=4, sticky="w", padx=(4, 0))

        ttk.Checkbutton(domain, text="Simulate white mouse", variable=self._white_mouse_var).grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Label(domain, text="invert_intensity").grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        ttk.Entry(domain, textvariable=self._invert_intensity_var, width=8).grid(row=1, column=2, sticky="w", padx=(4, 0), pady=(6, 0))

        ttk.Checkbutton(domain, text="Simulate black & white", variable=self._bw_var).grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Label(domain, text="bw_prob").grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        ttk.Entry(domain, textvariable=self._bw_prob_var, width=8).grid(row=2, column=2, sticky="w", padx=(4, 0), pady=(6, 0))

        actions = ttk.Frame(root)
        actions.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        self._run_btn = ttk.Button(actions, text="Run Augmentation", command=self._start_augmentation, style="Accent.TButton")
        self._run_btn.pack(side="left")
        ttk.Button(actions, text="Open Output Folder", command=self._open_output).pack(side="left", padx=(8, 0))

        self._progress_var = tk.StringVar(value="Idle")
        ttk.Label(root, textvariable=self._progress_var, style="Status.TLabel").grid(row=5, column=0, sticky="w", pady=(8, 0))

        log_box = ttk.LabelFrame(root, text="Run Log", padding=8)
        log_box.grid(row=6, column=0, sticky="nsew", pady=(8, 0))
        log_box.columnconfigure(0, weight=1)
        log_box.rowconfigure(0, weight=1)
        self._log_text = tk.Text(log_box, height=14, state="disabled", wrap="word")
        self._log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(log_box, command=self._log_text.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self._log_text.configure(yscrollcommand=log_scroll.set)

    def _add_path_row(self, parent, row: int, label: str, var: tk.StringVar, browse_fn) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=(8, 8), pady=2)
        ttk.Button(parent, text="Browse...", command=browse_fn).grid(row=row, column=2, sticky="e")

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

        self._run_btn.configure(state="disabled")
        self._progress_var.set("Running augmentation...")
        job = {
            "dataset_root": Path(self._dataset_root_var.get().strip()),
            "output_raw": self._output_root_var.get().strip(),
            "add_count": max(1, int(self._add_count_var.get().strip() or "1")),
            "copy_originals": bool(self._copy_originals_var.get()),
            "seed": int(self._seed_var.get().strip() or "42"),
            "domain_shift": {
                "white_mouse": bool(self._white_mouse_var.get()),
                "invert_intensity": float(self._invert_intensity_var.get().strip() or "1.0"),
                "red_light": bool(self._red_light_var.get()),
                "red_boost": float(self._red_boost_var.get().strip() or "1.5"),
                "bg_suppress": float(self._bg_suppress_var.get().strip() or "0.2"),
                "bw": bool(self._bw_var.get()),
                "bw_prob": float(self._bw_prob_var.get().strip() or "0.5"),
            },
        }

        def worker() -> None:
            with self._worker_lock:
                try:
                    self._run_augmentation_pipeline(job)
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


__all__ = ["DatasetAugmentorLabUIError", "DatasetAugmentorLabWindow"]
