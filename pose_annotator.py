#!/usr/bin/env python3
"""
Simple Tkinter tool to view and fix YOLO-pose annotations.

It reads .txt files that follow the YOLOv8 pose layout:
class xc yc w h x1 y1 v1 x2 y2 v2 ... (all values normalized to image size).

Features
- Shows image + bounding box + skeleton edges.
- Click + drag a keypoint to reposition it.
- Right-click a keypoint to toggle visibility (v: 0<->2).
- Recompute bbox from visible keypoints with one button.
- Edit class id for the selected detection.
- Navigate frames (prev/next), add/delete detections, and save back to txt.

Usage
  python pose_annotator.py --dir path/to/images

Defaults assume jpg images sitting next to matching txt files.
Modify DEFAULT_EDGES below to match your skeleton order.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# Keypoint order assumed: nose, left_ear, right_ear, head, thorax, mid_back, tail_base (indexes 0-6).
# Update edges if your order changes.
DEFAULT_EDGES: Sequence[Tuple[int, int]] = (
    (0, 3),  # nose -> head
    (1, 3),  # left_ear -> head
    (2, 3),  # right_ear -> head
    (3, 4),  # head -> thorax
    (4, 5),  # thorax -> mid_back
    (5, 6),  # mid_back -> tail_base
)

# Basic color palette for up to a handful of detections.
COLORS = [
    "#e86a92",
    "#5fb0e8",
    "#6dd185",
    "#f7c948",
    "#d28eff",
    "#ff9d66",
]

# Class label list; index = class id.
CLASSES = [
    "ambulatory",
    "wall-rearing",
    "sniffing",
    "digging",
    "grooming",
]

# Keypoint names (index = kp id).
KEYPOINT_NAMES = [
    "nose",
    "left_ear",
    "right_ear",
    "head",
    "thorax",
    "mid_back",
    "tail_base",
]


@dataclass
class Detection:
    class_id: int
    xc: float
    yc: float
    w: float
    h: float
    keypoints: List[Tuple[float, float, float]]  # (x, y, v)

    @classmethod
    def from_line(cls, line: str) -> "Detection":
        parts = [float(x) for x in line.strip().split()]
        if len(parts) < 8:
            raise ValueError(f"Line too short to parse pose annotation: {line!r}")
        class_id = int(parts[0])
        xc, yc, w, h = parts[1:5]
        kps_raw = parts[5:]
        if len(kps_raw) % 3 != 0:
            raise ValueError(f"Keypoint triple mismatch in line: {line!r}")
        keypoints = [
            (kps_raw[i], kps_raw[i + 1], kps_raw[i + 2])
            for i in range(0, len(kps_raw), 3)
        ]
        return cls(class_id, xc, yc, w, h, keypoints)

    def to_line(self) -> str:
        vals = [
            str(int(self.class_id)),
            f"{self.xc:.6f}",
            f"{self.yc:.6f}",
            f"{self.w:.6f}",
            f"{self.h:.6f}",
        ]
        for x, y, v in self.keypoints:
            vals.extend([f"{x:.6f}", f"{y:.6f}", f"{v:.6f}"])
        return " ".join(vals)

    @property
    def num_keypoints(self) -> int:
        return len(self.keypoints)


def load_annotations(label_path: Path, expected_kpts: int | None = None) -> List[Detection]:
    if not label_path.exists():
        return []
    detections: List[Detection] = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            det = Detection.from_line(line)
            detections.append(det)
    if expected_kpts is not None:
        for det in detections:
            if det.num_keypoints != expected_kpts:
                raise ValueError(
                    f"{label_path.name}: expected {expected_kpts} kpts, "
                    f"found {det.num_keypoints}"
                )
    return detections


def save_annotations(label_path: Path, detections: Sequence[Detection]) -> None:
    with label_path.open("w", encoding="utf-8") as f:
        for det in detections:
            f.write(det.to_line().rstrip() + "\n")


def parse_edges(text: str) -> Sequence[Tuple[int, int]]:
    edges: list[Tuple[int, int]] = []
    for chunk in text.split(","):
        if not chunk.strip():
            continue
        a, b = chunk.split("-")
        edges.append((int(a), int(b)))
    return edges


class PoseAnnotator:
    def __init__(self, args: argparse.Namespace):
        self.image_dir = Path(args.dir).resolve()
        if not self.image_dir.exists():
            raise SystemExit(f"Directory not found: {self.image_dir}")
        self.img_ext = args.ext.lower()
        self.edges = list(args.edges)
        self.max_w = args.max_width
        self.max_h = args.max_height

        self.image_files = sorted(self.image_dir.glob(f"*{self.img_ext}"))
        if not self.image_files:
            raise SystemExit(f"No images matching *{self.img_ext} in {self.image_dir}")

        self.index = 0
        self.detections: List[Detection] = []
        self.selected_det = 0
        self.selected_kp: int | None = None
        self.dragging = False
        self.scale = 1.0
        self.img_w = 1
        self.img_h = 1

        # GUI setup
        self.root = tk.Tk()
        self.root.title("YOLO Pose Annotator")
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<Key-s>", lambda e: self.save_current())
        self.root.bind("<Key-S>", lambda e: self.save_current())
        self.root.bind("<Key-b>", lambda e: self.recalc_bbox())
        self.root.bind("<Key-B>", lambda e: self.recalc_bbox())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.canvas = tk.Canvas(self.root, highlightthickness=0)
        self.canvas.grid(row=0, column=0, rowspan=10, sticky="nsew")
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<ButtonPress-3>", self.on_right_click)

        ctrl = ttk.Frame(self.root, padding=6)
        ctrl.grid(row=0, column=1, sticky="nw")

        self.info_var = tk.StringVar(value="")
        ttk.Label(ctrl, textvariable=self.info_var, justify="left").grid(
            row=0, column=0, sticky="w"
        )

        btn_row = ttk.Frame(ctrl)
        btn_row.grid(row=1, column=0, pady=4, sticky="w")
        ttk.Button(btn_row, text="Prev (←)", command=self.prev_image).grid(
            row=0, column=0, padx=2
        )
        ttk.Button(btn_row, text="Next (→)", command=self.next_image).grid(
            row=0, column=1, padx=2
        )
        ttk.Button(btn_row, text="Save (S)", command=self.save_current).grid(
            row=0, column=2, padx=2
        )

        self.class_choice = tk.StringVar(value="0: ambulatory")
        class_row = ttk.Frame(ctrl)
        class_row.grid(row=2, column=0, pady=4, sticky="w")
        ttk.Label(class_row, text="Class:").grid(row=0, column=0, padx=2)
        class_values = [f"{i}: {name}" for i, name in enumerate(CLASSES)]
        class_combo = ttk.Combobox(
            class_row,
            values=class_values,
            textvariable=self.class_choice,
            width=18,
            state="readonly",
        )
        class_combo.grid(row=0, column=1, padx=2)
        class_combo.bind("<<ComboboxSelected>>", lambda e: self.update_class_id())
        ttk.Button(class_row, text="Set", command=self.update_class_id).grid(
            row=0, column=2, padx=2
        )

        ttk.Button(ctrl, text="Recalc bbox (B)", command=self.recalc_bbox).grid(
            row=3, column=0, pady=2, sticky="w"
        )

        add_row = ttk.Frame(ctrl)
        add_row.grid(row=4, column=0, pady=4, sticky="w")
        ttk.Button(add_row, text="Add detection", command=self.add_detection).grid(
            row=0, column=0, padx=2
        )
        ttk.Button(add_row, text="Delete detection", command=self.delete_detection).grid(
            row=0, column=1, padx=2
        )

        help_text = (
            "Controls:\n"
            "  Drag left mouse to move a keypoint\n"
            "  Right click toggles visibility (0/2)\n"
            "  Use class dropdown after clicking a detection to set label\n"
            "  ← / → : prev / next image (auto-saves current)\n"
            "  S : save txt\n"
            "  B : recompute bbox from visible kpts\n"
        )
        ttk.Label(ctrl, text=help_text, justify="left").grid(
            row=5, column=0, sticky="w", pady=(6, 0)
        )

        kpt_text = "Keypoints (index:name): " + ", ".join(
            f"{i} {name}" for i, name in enumerate(KEYPOINT_NAMES)
        )
        ttk.Label(ctrl, text=kpt_text, justify="left").grid(
            row=6, column=0, sticky="w", pady=(6, 0)
        )

        for i in range(2):
            self.root.columnconfigure(i, weight=1 if i == 0 else 0)
        self.root.rowconfigure(0, weight=1)

        self.load_image(0)
        self.root.mainloop()

    # ---------------- GUI + drawing ---------------- #
    def load_image(self, idx: int) -> None:
        idx = max(0, min(idx, len(self.image_files) - 1))
        self.index = idx

        img_path = self.image_files[idx]
        label_path = img_path.with_suffix(".txt")
        self.detections = load_annotations(label_path)
        if self.detections:
            self.selected_det = min(self.selected_det, len(self.detections) - 1)
            self.sync_class_widget()
        else:
            self.selected_det = 0
            self.class_choice.set("0: ambulatory")

        pil = Image.open(img_path).convert("RGB")
        self.img_w, self.img_h = pil.size

        self.scale = min(
            self.max_w / self.img_w,
            self.max_h / self.img_h,
            1.0,
        )
        disp_w = int(self.img_w * self.scale)
        disp_h = int(self.img_h * self.scale)
        if self.scale != 1.0:
            pil = pil.resize((disp_w, disp_h), Image.LANCZOS)

        self.tk_img = ImageTk.PhotoImage(pil)
        self.canvas.config(width=disp_w, height=disp_h)
        self.redraw()
        self.root.title(f"[{idx + 1}/{len(self.image_files)}] {img_path.name}")

    def redraw(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        if not self.detections:
            self.info_var.set("No detections. Add one or just view.")
            return

        disp_lines = []
        for det_idx, det in enumerate(self.detections):
            color = COLORS[det_idx % len(COLORS)]
            selected = det_idx == self.selected_det
            # Bounding box
            x0 = (det.xc - det.w / 2) * self.img_w * self.scale
            y0 = (det.yc - det.h / 2) * self.img_h * self.scale
            x1 = (det.xc + det.w / 2) * self.img_w * self.scale
            y1 = (det.yc + det.h / 2) * self.img_h * self.scale
            self.canvas.create_rectangle(
                x0,
                y0,
                x1,
                y1,
                outline=color,
                width=2 if selected else 1,
            )
            # Skeleton edges
            for a, b in self.edges:
                if a >= det.num_keypoints or b >= det.num_keypoints:
                    continue
                xa, ya, va = det.keypoints[a]
                xb, yb, vb = det.keypoints[b]
                if va <= 0 and vb <= 0:
                    continue
                self.canvas.create_line(
                    xa * self.img_w * self.scale,
                    ya * self.img_h * self.scale,
                    xb * self.img_w * self.scale,
                    yb * self.img_h * self.scale,
                    fill=color,
                    width=2,
                )
            # Keypoints
            for kp_idx, (x, y, v) in enumerate(det.keypoints):
                r = 6 if v > 0 else 6
                px = x * self.img_w * self.scale
                py = y * self.img_h * self.scale
                fill = color if v > 0 else "#ff4444"
                self.canvas.create_oval(
                    px - r,
                    py - r,
                    px + r,
                    py + r,
                    fill=fill,
                    outline="black",
                    width=1,
                )
                self.canvas.create_text(
                    px + 8,
                    py,
                    text=str(kp_idx),
                    anchor="w",
                    fill="black",
                    font=("Arial", 9, "bold"),
                )
            disp_lines.append(f"det {det_idx}: class={det.class_id}, kpts={det.num_keypoints}")

        self.info_var.set("\n".join(disp_lines))

    # ---------------- Mouse interactions ---------------- #
    def pick_keypoint(self, event_x: float, event_y: float) -> Tuple[int, int] | None:
        """Return (det_idx, kp_idx) closest within tolerance."""
        if not self.detections:
            return None
        tx = event_x / self.scale
        ty = event_y / self.scale
        best: Tuple[int, int] | None = None
        best_dist2 = 999999.0
        tol = 15.0
        for det_idx, det in enumerate(self.detections):
            for kp_idx, (x, y, _v) in enumerate(det.keypoints):
                dx = x * self.img_w - tx
                dy = y * self.img_h - ty
                dist2 = dx * dx + dy * dy
                if dist2 < best_dist2 and dist2 <= (tol * tol):
                    best = (det_idx, kp_idx)
                    best_dist2 = dist2
        return best

    def on_press(self, event: tk.Event) -> None:
        picked = self.pick_keypoint(event.x, event.y)
        if picked is None:
            return
        det_idx, kp_idx = picked
        self.selected_det = det_idx
        self.selected_kp = kp_idx
        self.sync_class_widget()
        self.dragging = True

    def on_drag(self, event: tk.Event) -> None:
        if not self.dragging or self.selected_kp is None:
            return
        det = self.detections[self.selected_det]
        u = max(0.0, min(1.0, event.x / (self.img_w * self.scale)))
        v = max(0.0, min(1.0, event.y / (self.img_h * self.scale)))
        x, y, conf = det.keypoints[self.selected_kp]
        det.keypoints[self.selected_kp] = (u, v, conf if conf > 0 else 2.0)
        self.redraw()

    def on_release(self, _event: tk.Event) -> None:
        self.dragging = False

    def on_right_click(self, event: tk.Event) -> None:
        picked = self.pick_keypoint(event.x, event.y)
        if picked is None:
            return
        det_idx, kp_idx = picked
        det = self.detections[det_idx]
        x, y, conf = det.keypoints[kp_idx]
        det.keypoints[kp_idx] = (x, y, 0.0 if conf > 0 else 2.0)
        self.selected_det = det_idx
        self.selected_kp = kp_idx
        self.sync_class_widget()
        self.redraw()

    # ---------------- Actions ---------------- #
    def prev_image(self) -> None:
        self.save_current()
        self.load_image(self.index - 1)

    def next_image(self) -> None:
        self.save_current()
        self.load_image(self.index + 1)

    def save_current(self) -> None:
        img_path = self.image_files[self.index]
        label_path = img_path.with_suffix(".txt")
        save_annotations(label_path, self.detections)
        self.root.title(f"[saved] {self.root.title()}")

    def update_class_id(self) -> None:
        if not self.detections:
            return
        try:
            cid = int(self.class_choice.get().split(":")[0])
        except Exception:
            messagebox.showerror("Invalid class", f"Could not parse class: {self.class_choice.get()}")
            return
        self.detections[self.selected_det].class_id = cid
        self.redraw()

    def sync_class_widget(self) -> None:
        """Sync dropdown to currently selected detection."""
        if not self.detections:
            return
        cid = self.detections[self.selected_det].class_id
        name = CLASSES[cid] if 0 <= cid < len(CLASSES) else f"unk{cid}"
        self.class_choice.set(f"{cid}: {name}")

    def add_detection(self) -> None:
        # New detection starts centered with small box and kpts at center.
        num_kpts = self.detections[0].num_keypoints if self.detections else 7
        keypoints = [(0.5, 0.5, 0.0) for _ in range(num_kpts)]
        det = Detection(class_id=0, xc=0.5, yc=0.5, w=0.2, h=0.2, keypoints=keypoints)
        self.detections.append(det)
        self.selected_det = len(self.detections) - 1
        self.sync_class_widget()
        self.redraw()

    def delete_detection(self) -> None:
        if not self.detections:
            return
        del self.detections[self.selected_det]
        self.selected_det = max(0, len(self.detections) - 1)
        if self.detections:
            self.sync_class_widget()
        else:
            self.class_choice.set("0: ambulatory")
        self.redraw()

    def recalc_bbox(self) -> None:
        if not self.detections:
            return
        det = self.detections[self.selected_det]
        visible = [(x, y) for x, y, v in det.keypoints if v > 0]
        if not visible:
            messagebox.showinfo("No visible keypoints", "Nothing to use for bounding box.")
            return
        xs = [p[0] for p in visible]
        ys = [p[1] for p in visible]
        margin = 0.02
        x0 = max(0.0, min(xs) - margin)
        y0 = max(0.0, min(ys) - margin)
        x1 = min(1.0, max(xs) + margin)
        y1 = min(1.0, max(ys) + margin)
        det.xc = (x0 + x1) / 2
        det.yc = (y0 + y1) / 2
        det.w = max(1e-4, x1 - x0)
        det.h = max(1e-4, y1 - y0)
        self.redraw()

    def on_close(self) -> None:
        self.save_current()
        self.root.destroy()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualize and edit YOLO-pose annotations.")
    p.add_argument(
        "--dir",
        "-d",
        default=".",
        help="Directory with images + labels (default: current directory).",
    )
    p.add_argument(
        "--ext",
        default=".jpg",
        help="Image extension to look for (default: .jpg).",
    )
    p.add_argument(
        "--edges",
        type=parse_edges,
        default=DEFAULT_EDGES,
        help="Skeleton edges as a comma list like '0-1,1-2,3-4'. Default matches DEFAULT_EDGES.",
    )
    p.add_argument(
        "--max-width",
        type=int,
        default=1280,
        help="Max display width (images are scaled down for viewing only).",
    )
    p.add_argument(
        "--max-height",
        type=int,
        default=900,
        help="Max display height (images are scaled down for viewing only).",
    )
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    try:
        PoseAnnotator(args)
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
