from __future__ import annotations

import json
import shlex
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Sequence

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from integra_pose.gui.plugin_chrome import apply_plugin_chrome, build_plugin_header
from integra_pose.gui.windowing import apply_adaptive_window_geometry


class BehaviorScopeUIError(RuntimeError):
    """Raised when the BehaviorScope assets cannot be located."""


def _behavior_scope_root() -> Path:
    plugin_dir = Path(__file__).resolve().parent
    bs_root = plugin_dir / "BehaviorScope_v2"
    if not bs_root.exists():
        raise BehaviorScopeUIError(
            f"BehaviorScope_v2 directory not found at {bs_root}. "
            "Verify that the toolkit is bundled with IntegraPose."
        )
    return bs_root


def _format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


class CommandRunner:
    def __init__(
        self,
        command: Sequence[str],
        cwd: Path,
        log_callback: Callable[[str], None],
        completion_callback: Callable[[int], None],
    ) -> None:
        self._command = list(command)
        self._cwd = cwd
        self._log_callback = log_callback
        self._completion_callback = completion_callback
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
        except Exception as exc:
            self._emit(f"[Runner] Failed to terminate process: {exc}")

    def _emit(self, message: str) -> None:
        try:
            self._log_callback(message)
        except Exception:
            pass

    def _run(self) -> None:
        exit_code = -1
        try:
            process = subprocess.Popen(
                self._command,
                cwd=self._cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
            )
        except Exception as exc:
            self._emit(f"[Runner] Failed to start command: {exc}")
            self._completion_callback(exit_code)
            return
        self._process = process

        assert process.stdout is not None
        try:
            for line in process.stdout:
                self._emit(line.rstrip())
            exit_code = process.wait()
        except Exception as exc:
            exit_code = -1
            self._emit(f"[Runner] Execution error: {exc}")
        finally:
            process.stdout.close()
            self._process = None
            self._completion_callback(exit_code)


class Tooltip:
    def __init__(self, widget: tk.Widget, text: str, *, delay: int = 350) -> None:
        self.widget = widget
        self.text = text
        self.delay = delay
        self._after_id: str | None = None
        self._tip_window: tk.Toplevel | None = None
        self._x = 0
        self._y = 0

        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        widget.bind("<ButtonPress>", self._on_leave, add="+")
        widget.bind("<Motion>", self._on_motion, add="+")

    def _on_enter(self, event) -> None:
        self._record_coords(event)
        self._schedule()

    def _on_motion(self, event) -> None:
        self._record_coords(event)
        if self._tip_window is not None:
            self._tip_window.wm_geometry(f"+{self._x}+{self._y}")

    def _on_leave(self, _event) -> None:
        self._cancel()
        self._hide()

    def _record_coords(self, event) -> None:
        self._x = event.x_root + 12
        self._y = event.y_root + 6

    def _schedule(self) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel(self) -> None:
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self) -> None:
        if self._tip_window is not None or not self.text:
            return
        self._tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{self._x}+{self._y}")
        label = ttk.Label(
            tw,
            text=self.text,
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            padding=(6, 4),
            wraplength=360,
            justify="left",
        )
        label.pack()

    def _hide(self) -> None:
        if self._tip_window is not None:
            try:
                self._tip_window.destroy()
            except Exception:
                pass
            self._tip_window = None


class BehaviorScopeWindow(tk.Toplevel):
    def __init__(self, main_app, parent: tk.Misc | None = None) -> None:
        super().__init__(parent)
        self.title("BehaviorScope Toolkit")
        apply_adaptive_window_geometry(
            self,
            preferred_size=(1040, 760),
            min_size=(900, 640),
        )
        self._main_app = main_app
        self.style = apply_plugin_chrome(self, main_app)
        self._bs_root = _behavior_scope_root()
        self._runner: CommandRunner | None = None
        self._current_task = ""
        self._action_buttons: list[ttk.Button] = []
        self._tooltips: list[Tooltip] = []
        self._scroll_canvases: list[tk.Canvas] = []
        self._mousewheel_bound = False
        self._config_path_var = tk.StringVar(
            value=str(self._bs_root / "behaviorscope_config.json")
        )

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._build_ui()
        self._autoload_config()
        self.protocol("WM_DELETE_WINDOW", self._handle_close)

    def _build_ui(self) -> None:
        container = ttk.Frame(self, padding=12)
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=0)
        container.rowconfigure(1, weight=0)
        container.rowconfigure(2, weight=1)
        container.rowconfigure(3, weight=0)

        header = build_plugin_header(
            container,
            title="BehaviorScope Toolkit",
            summary="Build temporal behavior datasets, train sequence models, run inference, and review the outputs inside one consistent desktop workspace.",
        )
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        config_bar = ttk.Frame(container)
        config_bar.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        config_bar.columnconfigure(1, weight=1)
        ttk.Label(config_bar, text="Config file:").grid(row=0, column=0, sticky="w")
        cfg_entry = ttk.Entry(config_bar, textvariable=self._config_path_var)
        cfg_entry.grid(row=0, column=1, sticky="ew", padx=6)
        browse = ttk.Button(
            config_bar, text="Browse...", command=self._browse_config_path
        )
        browse.grid(row=0, column=2, padx=(0, 4))
        load_btn = ttk.Button(config_bar, text="Load", command=self._load_config)
        load_btn.grid(row=0, column=3, padx=(4, 4))
        save_btn = ttk.Button(config_bar, text="Save", command=self._save_config)
        save_btn.grid(row=0, column=4, padx=(0, 4))
        save_as_btn = ttk.Button(
            config_bar,
            text="Save As",
            command=lambda: self._save_config(ask_path=True),
        )
        save_as_btn.grid(row=0, column=5, padx=(0, 0))

        self._notebook = ttk.Notebook(container)
        self._notebook.grid(row=2, column=0, sticky="nsew")

        self._build_sequences_tab()
        self._build_training_tab()
        self._build_inference_tab()
        self._build_review_tab()

        log_frame = ttk.LabelFrame(container, text="Execution Log")
        log_frame.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self._log_widget = ScrolledText(
            log_frame,
            height=10,
            state="disabled",
            wrap="word",
        )
        self._log_widget.grid(row=0, column=0, sticky="nsew")

    def _browse_config_path(self) -> None:
        path = filedialog.askopenfilename(
            parent=self,
            title="Select BehaviorScope config",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
        )
        if path:
            self._config_path_var.set(path)

    def _default_config_path(self) -> Path:
        return self._bs_root / "behaviorscope_config.json"

    def _collect_config(self) -> dict:
        return {
            "prepare": self._serialize_vars(self._prepare_vars),
            "train": self._serialize_vars(self._train_vars),
            "infer": self._serialize_vars(self._infer_vars),
            "review": self._serialize_vars(self._review_vars),
        }

    def _serialize_vars(self, var_map: dict) -> dict:
        data: dict[str, str | bool] = {}
        for key, var in var_map.items():
            try:
                value = var.get()
            except Exception:
                continue
            if isinstance(var, tk.BooleanVar):
                data[key] = bool(value)
            else:
                data[key] = str(value)
        return data

    def _apply_config(self, data: dict) -> None:
        for key, var_map in (
            ("prepare", self._prepare_vars),
            ("train", self._train_vars),
            ("infer", self._infer_vars),
            ("review", self._review_vars),
        ):
            self._apply_vars(var_map, data.get(key, {}))

    def _apply_sequence_manifest(self, path: Path, data: dict) -> bool:
        if not isinstance(data, dict):
            return False
        splits = data.get("splits")
        if not isinstance(splits, dict):
            return False

        meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
        prepare_values: dict[str, str | bool] = {}
        train_values: dict[str, str | bool] = {"manifest_path": str(path)}
        infer_values: dict[str, str | bool] = {}

        def _set_meta(meta_key: str, target: dict, target_key: str | None = None) -> None:
            if meta_key not in meta:
                return
            value = meta.get(meta_key)
            if value is None:
                return
            target[target_key or meta_key] = value

        _set_meta("window_size", prepare_values)
        _set_meta("window_size", infer_values)
        _set_meta("window_size", train_values, "num_frames")
        _set_meta("window_stride", prepare_values)
        _set_meta("window_stride", infer_values)
        _set_meta("crop_size", prepare_values)
        _set_meta("crop_size", infer_values)
        _set_meta("crop_size", train_values, "frame_size")
        _set_meta("pad_ratio", prepare_values)
        _set_meta("pad_ratio", infer_values)
        _set_meta("yolo_conf", prepare_values)
        _set_meta("yolo_conf", infer_values)
        _set_meta("yolo_iou", prepare_values)
        _set_meta("yolo_iou", infer_values)
        _set_meta("yolo_imgsz", prepare_values)
        _set_meta("yolo_imgsz", infer_values)
        _set_meta("save_frame_crops", prepare_values)
        _set_meta("save_sequence_npz", prepare_values)
        _set_meta("source_dataset_root", prepare_values, "dataset_root")

        self._apply_vars(self._prepare_vars, prepare_values)
        self._apply_vars(self._train_vars, train_values)
        self._apply_vars(self._infer_vars, infer_values)
        self._append_log(f"[Config] Loaded sequence manifest metadata from {path}")
        self._log_to_app(
            f"BehaviorScope manifest metadata loaded from {path}",
            level="INFO",
        )
        return True

    def _apply_vars(self, var_map: dict, values: dict) -> None:
        for key, value in values.items():
            if key not in var_map:
                continue
            var = var_map[key]
            try:
                if isinstance(var, tk.BooleanVar):
                    var.set(bool(value))
                else:
                    var.set("" if value is None else str(value))
            except Exception:
                continue

    def _save_config(self, ask_path: bool = False) -> None:
        path_str = self._config_path_var.get().strip()
        path = Path(path_str) if path_str else self._default_config_path()
        if ask_path or not path_str:
            selected = filedialog.asksaveasfilename(
                parent=self,
                title="Save BehaviorScope config",
                initialfile=path.name,
                defaultextension=".json",
                filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
            )
            if not selected:
                return
            path = Path(selected)
            self._config_path_var.set(selected)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._collect_config(), indent=2), encoding="utf-8")
        except Exception as exc:
            self._show_error(f"Failed to save config: {exc}")
            return
        self._append_log(f"[Config] Saved BehaviorScope settings to {path}")
        self._log_to_app(f"BehaviorScope config saved to {path}", level="INFO")

    def _load_config(self) -> None:
        path_str = self._config_path_var.get().strip()
        path = Path(path_str) if path_str else self._default_config_path()
        self._config_path_var.set(str(path))
        if not path.exists():
            self._show_error(f"Config file not found: {path}")
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._show_error(f"Failed to load config: {exc}")
            return
        if not isinstance(data, dict):
            self._show_error("Config file must contain a JSON object.")
            return
        if self._apply_sequence_manifest(path, data):
            return
        self._apply_config(data)
        self._append_log(f"[Config] Loaded BehaviorScope settings from {path}")
        self._log_to_app(f"BehaviorScope config loaded from {path}", level="INFO")

    def _autoload_config(self) -> None:
        path = Path(self._config_path_var.get())
        if not path.exists():
            path = self._default_config_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                if self._apply_sequence_manifest(path, data):
                    return
                self._apply_config(data)
                self._append_log(f"[Config] Auto-loaded BehaviorScope settings from {path}")
        except Exception as exc:
            self._append_log(f"[Config] Skipped auto-load: {exc}")

    def _build_sequences_tab(self) -> None:
        tab = self._create_scrollable_tab("Sequences")
        tab.columnconfigure(0, weight=1)

        self._prepare_vars = {
            "dataset_root": tk.StringVar(),
            "yolo_weights": tk.StringVar(),
            "output_root": tk.StringVar(),
            "manifest_path": tk.StringVar(),
            "window_size": tk.StringVar(value="30"),
            "window_stride": tk.StringVar(value="15"),
            "crop_size": tk.StringVar(value="224"),
            "pad_ratio": tk.StringVar(value="0.2"),
            "yolo_conf": tk.StringVar(value="0.25"),
            "yolo_iou": tk.StringVar(value="0.45"),
            "yolo_imgsz": tk.StringVar(value="640"),
            "yolo_max_det": tk.StringVar(value="1"),
            "device": tk.StringVar(value="cpu"),
            "train_ratio": tk.StringVar(value="0.8"),
            "val_ratio": tk.StringVar(value="0.2"),
            "test_ratio": tk.StringVar(value="0.0"),
            "seed": tk.StringVar(value="42"),
            "min_sequence_count": tk.StringVar(value="1"),
            "default_fps": tk.StringVar(value="30.0"),
            "save_frame_crops": tk.BooleanVar(value=False),
            "save_sequence_npz": tk.BooleanVar(value=True),
            "keep_last_box": tk.BooleanVar(value=True),
            "skip_existing": tk.BooleanVar(value=True),
        }

        paths = ttk.LabelFrame(tab, text="Paths & Assets")
        paths.grid(row=0, column=0, sticky="ew")
        paths.columnconfigure(1, weight=1)
        self._path_input(
            paths,
            row=0,
            label="Dataset root",
            var=self._prepare_vars["dataset_root"],
            browse="dir",
            tooltip="Folder that contains one subdirectory per behavior with the raw videos to crop.",
        )
        self._path_input(
            paths,
            row=1,
            label="YOLO weights",
            var=self._prepare_vars["yolo_weights"],
            browse="file",
            filetypes=(("PyTorch weights", "*.pt"), ("All files", "*.*")),
            tooltip="Ultralytics YOLO checkpoint used to detect and crop the animal in each frame.",
        )
        self._path_input(
            paths,
            row=2,
            label="Output root",
            var=self._prepare_vars["output_root"],
            browse="dir",
            tooltip="Destination folder for QA crops, NPZ windows, and generated manifests.",
        )
        self._path_input(
            paths,
            row=3,
            label="Manifest override",
            var=self._prepare_vars["manifest_path"],
            browse="save",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
            tooltip="Optional explicit path for the sequence_manifest.json file.",
        )

        yolo = ttk.LabelFrame(tab, text="Cropping & Windows")
        yolo.grid(row=1, column=0, sticky="ew", pady=8)
        for idx, (label, key, tip) in enumerate(
            [
                ("Window size", "window_size", "Number of frames included in each YOLO crop window."),
                ("Stride", "window_stride", "How many frames to slide between successive windows."),
                ("Crop size", "crop_size", "Square resolution (pixels) used when saving cropped frames."),
                ("Pad ratio", "pad_ratio", "Fractional padding added around the detected bounding box."),
                ("YOLO conf", "yolo_conf", "Minimum detection confidence to accept a bounding box."),
                ("YOLO NMS IoU", "yolo_iou", "IOU threshold for non-max suppression during cropping."),
                ("YOLO img size", "yolo_imgsz", "Inference resolution passed to YOLO (must be divisible by 32)."),
                ("YOLO max det", "yolo_max_det", "Maximum detections per frame returned by YOLO."),
                ("Device", "device", "Device string for YOLO inference, e.g., 'cpu', 'cuda:0', or 'auto'."),
            ]
        ):
            self._grid_entry(yolo, idx // 2, idx % 2, label, self._prepare_vars[key], tooltip=tip)

        toggles = ttk.LabelFrame(tab, text="Outputs & Options")
        toggles.grid(row=2, column=0, sticky="ew")
        for idx, (label, key, tip) in enumerate(
            [
                ("Save crops", "save_frame_crops", "Write individual JPG crops for QA review."),
                ("Save NPZ", "save_sequence_npz", "Write NumPy arrays per window for fast training."),
                ("Keep last box", "keep_last_box", "Reuse the previous detection when YOLO misses a frame."),
                ("Skip existing", "skip_existing", "Skip videos that already have QA/NPZ outputs."),
            ]
        ):
            chk = ttk.Checkbutton(
                toggles,
                text=label,
                variable=self._prepare_vars[key],
            )
            chk.grid(row=0, column=idx, padx=8, pady=4, sticky="w")
            self._attach_tooltip(chk, tip)

        splits = ttk.LabelFrame(tab, text="Train/Val/Test Split")
        splits.grid(row=3, column=0, sticky="ew", pady=8)
        for idx, (label, key, tip) in enumerate(
            [
                ("Train ratio", "train_ratio", "Proportion of sequences assigned to the training split."),
                ("Val ratio", "val_ratio", "Proportion of sequences used for validation."),
                ("Test ratio", "test_ratio", "Proportion of sequences held out for testing."),
                ("Seed", "seed", "Random seed for deterministic splitting and augmentations."),
                ("Min sequences", "min_sequence_count", "Discard clips that yield fewer than this many windows."),
                ("Default FPS", "default_fps", "Fallback FPS when the source video lacks metadata."),
            ]
        ):
            self._grid_entry(splits, idx // 3, idx % 3, label, self._prepare_vars[key], tooltip=tip)

        button = ttk.Button(
            tab,
            text="Build Sequence Manifest",
            command=self._run_prepare_sequences,
        )
        self._register_action_button(button)
        button.grid(row=4, column=0, sticky="e", pady=(12, 0))
        self._attach_tooltip(button, "Run the YOLO cropping pipeline and write/update the sequence manifest.")

    def _build_training_tab(self) -> None:
        tab = self._create_scrollable_tab("Training")
        tab.columnconfigure(0, weight=1)

        self._train_vars = {
            "manifest_path": tk.StringVar(),
            "output_dir": tk.StringVar(),
            "run_name": tk.StringVar(),
            "num_frames": tk.StringVar(),
            "frame_size": tk.StringVar(),
            "batch_size": tk.StringVar(value="4"),
            "epochs": tk.StringVar(value="25"),
            "lr": tk.StringVar(value="0.0001"),
            "weight_decay": tk.StringVar(value="0.0001"),
            "num_workers": tk.StringVar(value="0"),
            "patience": tk.StringVar(value="10"),
            "seed": tk.StringVar(value="42"),
            "device": tk.StringVar(value="auto"),
            "backbone": tk.StringVar(value="mobilenet_v3_small"),
            "hidden_dim": tk.StringVar(value="256"),
            "num_layers": tk.StringVar(value="1"),
            "dropout": tk.StringVar(value="0.2"),
            "temporal_attention_layers": tk.StringVar(value="0"),
            "attention_heads": tk.StringVar(value="4"),
            "slowfast_alpha": tk.StringVar(value="4"),
            "slowfast_fusion_ratio": tk.StringVar(value="0.25"),
            "slowfast_base_channels": tk.StringVar(value="48"),
            "log_interval": tk.StringVar(value="0"),
            "sequence_model": tk.StringVar(value="lstm"),
            "lr_scheduler": tk.StringVar(value="none"),
            "scheduler_tmax": tk.StringVar(),
            "scheduler_min_lr": tk.StringVar(value="1e-6"),
            "bidirectional": tk.BooleanVar(value=False),
            "attention_pool": tk.BooleanVar(value=False),
            "feature_se": tk.BooleanVar(value=False),
            "no_pretrained_backbone": tk.BooleanVar(value=False),
            "train_backbone": tk.BooleanVar(value=False),
            "disable_augment": tk.BooleanVar(value=False),
        }

        paths = ttk.LabelFrame(tab, text="Data")
        paths.grid(row=0, column=0, sticky="ew")
        paths.columnconfigure(1, weight=1)
        self._path_input(
            paths,
            row=0,
            label="Manifest",
            var=self._train_vars["manifest_path"],
            browse="file",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
            tooltip="Sequence manifest JSON emitted by the preprocessing step.",
        )
        self._path_input(
            paths,
            row=1,
            label="Output directory",
            var=self._train_vars["output_dir"],
            browse="dir",
            tooltip="Parent folder where each training run will create its subdirectory.",
        )
        self._grid_entry(
            paths,
            2,
            0,
            "Run name",
            self._train_vars["run_name"],
            columnspan=2,
            tooltip="Optional subfolder name for this run; defaults to a timestamp.",
        )

        basics = ttk.LabelFrame(tab, text="Batching & Frames")
        basics.grid(row=1, column=0, sticky="ew", pady=8)
        for idx, (label, key, tip) in enumerate(
            [
                ("Num frames", "num_frames", "Override the sequence length stored in the manifest metadata."),
                ("Frame size", "frame_size", "Override the crop size from the manifest (pixels)."),
                ("Batch size", "batch_size", "Number of sequences per optimization step."),
                ("Epochs", "epochs", "Maximum number of passes over the training set."),
                ("Device", "device", "Device string for model training, e.g., 'cuda', 'cuda:0', 'cpu'."),
            ]
        ):
            self._grid_entry(basics, idx // 2, idx % 2, label, self._train_vars[key], tooltip=tip)

        optim = ttk.LabelFrame(tab, text="Optimization")
        optim.grid(row=2, column=0, sticky="ew")
        for idx, (label, key, tip) in enumerate(
            [
                ("Learning rate", "lr", "Initial learning rate for AdamW."),
                ("Weight decay", "weight_decay", "L2 regularization term applied to trainable weights."),
                ("Num workers", "num_workers", "DataLoader worker threads (set >0 for faster disk I/O)."),
                ("Patience", "patience", "Early stopping patience counted in epochs."),
                ("Seed", "seed", "Random seed for dataset shuffling and parameter initialization."),
                ("Log interval", "log_interval", "Print mini-batch stats every N steps (0 disables verbose logging)."),
            ]
        ):
            self._grid_entry(optim, idx // 3, idx % 3, label, self._train_vars[key], tooltip=tip)

        scheduler_frame = ttk.LabelFrame(tab, text="Scheduler")
        scheduler_frame.grid(row=3, column=0, sticky="ew", pady=8)
        radio_frame = ttk.Frame(scheduler_frame)
        radio_frame.grid(row=0, column=0, columnspan=2, sticky="w")
        scheduler_options = (
            ("None", "none", "Keep the learning rate constant throughout training."),
            ("Cosine", "cosine", "Use CosineAnnealingLR to decay the learning rate smoothly."),
        )
        for text, value, tip in scheduler_options:
            radio = ttk.Radiobutton(
                radio_frame,
                text=text,
                value=value,
                variable=self._train_vars["lr_scheduler"],
            )
            radio.pack(side="left", padx=(0, 12), pady=4)
            self._attach_tooltip(radio, tip)
        self._grid_entry(
            scheduler_frame,
            row=1,
            column=0,
            label="T max",
            var=self._train_vars["scheduler_tmax"],
            tooltip="Number of epochs that define one cosine cycle (defaults to total epochs).",
        )
        self._grid_entry(
            scheduler_frame,
            row=1,
            column=1,
            label="Min LR",
            var=self._train_vars["scheduler_min_lr"],
            tooltip="Floor value the cosine scheduler will approach at the end of the cycle.",
        )

        model = ttk.LabelFrame(tab, text="Architecture")
        model.grid(row=4, column=0, sticky="ew")
        backbone_frame = ttk.Frame(model)
        backbone_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        backbone_frame.columnconfigure(1, weight=1)
        backbone_label = ttk.Label(backbone_frame, text="Backbone")
        backbone_label.grid(row=0, column=0, sticky="w", padx=(0, 6), pady=2)
        backbone_combo = ttk.Combobox(
            backbone_frame,
            textvariable=self._train_vars["backbone"],
            values=("mobilenet_v3_small", "efficientnet_b0", "slowfast_tiny"),
            state="readonly",
        )
        backbone_combo.grid(row=0, column=1, sticky="ew", pady=2)
        backbone_tip = (
            "Frame encoder backbone. Use 'slowfast_tiny' for a 3D SlowFast-style "
            "backbone; sequence-model settings are ignored in that case."
        )
        self._attach_tooltip(backbone_label, backbone_tip)
        self._attach_tooltip(backbone_combo, backbone_tip)

        model_items = [
            ("Hidden dim", "hidden_dim", "Width of the LSTM/attention embeddings."),
            ("Layers", "num_layers", "Number of stacked LSTM layers."),
            ("Dropout", "dropout", "Dropout probability applied inside the temporal model."),
            ("Attention layers", "temporal_attention_layers", "Count of self-attention blocks before the head."),
            ("Attention heads", "attention_heads", "Multi-head attention count for temporal blocks."),
            ("SlowFast alpha", "slowfast_alpha", "Temporal stride ratio between the Slow and Fast pathways (slowfast only)."),
            ("SlowFast fusion", "slowfast_fusion_ratio", "Channel ratio when fusing Slow/Fast features (slowfast only)."),
            ("SlowFast channels", "slowfast_base_channels", "Base channel width for the SlowFast backbone."),
        ]
        row_offset = 1
        for idx, (label, key, tip) in enumerate(
            model_items
        ):
            self._grid_entry(
                model,
                row_offset + idx // 2,
                idx % 2,
                label,
                self._train_vars[key],
                tooltip=tip,
            )

        seq_mode_frame = ttk.Frame(model)
        seq_mode_frame.grid(
            row=row_offset + (len(model_items) + 1) // 2,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(8, 0),
        )
        ttk.Label(seq_mode_frame, text="Sequence model:").pack(side="left", padx=(0, 8))
        for text, value, tip in (
            ("LSTM", "lstm", "Recurrent LSTM head for temporal aggregation (ignored for slowfast_tiny)."),
            ("Attention", "attention", "Attention-only head for global pooling (ignored for slowfast_tiny)."),
        ):
            radio = ttk.Radiobutton(
                seq_mode_frame,
                text=text,
                value=value,
                variable=self._train_vars["sequence_model"],
            )
            radio.pack(side="left", padx=(0, 12))
            self._attach_tooltip(radio, tip)

        toggles = ttk.LabelFrame(tab, text="Feature Options")
        toggles.grid(row=5, column=0, sticky="ew", pady=8)
        for idx, (label, key, tip) in enumerate(
            [
                ("Bidirectional LSTM", "bidirectional", "Process sequences in both forward and reverse directions."),
                ("Attention pool", "attention_pool", "Apply attention pooling over the temporal outputs."),
                ("Feature SE", "feature_se", "Enable squeeze-excitation gating on per-frame features."),
                ("Freeze backbone", "no_pretrained_backbone", "Disable ImageNet weights (train from scratch)."),
                ("Train backbone", "train_backbone", "Unfreeze the CNN backbone for fine-tuning."),
                ("Disable augment", "disable_augment", "Turn off random crops/flips during training."),
            ]
        ):
            chk = ttk.Checkbutton(toggles, text=label, variable=self._train_vars[key])
            chk.grid(row=idx // 2, column=idx % 2, sticky="w", padx=8, pady=2)
            self._attach_tooltip(chk, tip)

        button = ttk.Button(
            tab,
            text="Start Training",
            command=self._run_training,
        )
        self._register_action_button(button)
        button.grid(row=6, column=0, sticky="e")
        self._attach_tooltip(button, "Launch the BehaviorScope training job with the selected hyperparameters.")

    def _build_inference_tab(self) -> None:
        tab = self._create_scrollable_tab("Inference")
        tab.columnconfigure(0, weight=1)

        self._infer_vars = {
            "model_path": tk.StringVar(),
            "video_path": tk.StringVar(),
            "output_csv": tk.StringVar(),
            "output_json": tk.StringVar(),
            "window_size": tk.StringVar(),
            "window_stride": tk.StringVar(),
            "batch_size": tk.StringVar(value="8"),
            "prob_threshold": tk.StringVar(value="0.5"),
            "keep_unknown": tk.BooleanVar(value=True),
            "skip_tail": tk.BooleanVar(value=False),
            "device": tk.StringVar(value="auto"),
            "yolo_weights": tk.StringVar(),
            "crop_size": tk.StringVar(),
            "pad_ratio": tk.StringVar(value="0.2"),
            "yolo_conf": tk.StringVar(value="0.25"),
            "yolo_iou": tk.StringVar(value="0.45"),
            "yolo_imgsz": tk.StringVar(value="640"),
            "yolo_max_det": tk.StringVar(value="1"),
            "keep_last_box": tk.BooleanVar(value=True),
            "progress_interval": tk.StringVar(value="50"),
        }

        paths = ttk.LabelFrame(tab, text="Inputs & Outputs")
        paths.grid(row=0, column=0, sticky="ew")
        paths.columnconfigure(1, weight=1)
        self._path_input(
            paths,
            row=0,
            label="Model checkpoint",
            var=self._infer_vars["model_path"],
            browse="file",
            filetypes=(("PyTorch state", "*.pt"), ("All files", "*.*")),
            tooltip="Best_model.pt checkpoint created during training.",
        )
        self._path_input(
            paths,
            row=1,
            label="Video path",
            var=self._infer_vars["video_path"],
            browse="file",
            filetypes=(("Video files", "*.mp4;*.avi;*.mov;*.mkv"), ("All files", "*.*")),
            tooltip="Full-length video to annotate with predicted behaviors.",
        )
        self._path_input(
            paths,
            row=2,
            label="YOLO weights",
            var=self._infer_vars["yolo_weights"],
            browse="file",
            filetypes=(("PyTorch weights", "*.pt"), ("All files", "*.*")),
            tooltip="YOLO checkpoint used to crop subjects during inference.",
        )
        self._path_input(
            paths,
            row=3,
            label="Output CSV",
            var=self._infer_vars["output_csv"],
            browse="save",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
            tooltip="Destination CSV containing merged bout annotations (defaults to <video>.behavior.csv).",
        )
        self._path_input(
            paths,
            row=4,
            label="Output JSON",
            var=self._infer_vars["output_json"],
            browse="save",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
            tooltip="Optional JSON file storing raw per-window predictions.",
        )

        window_opts = ttk.LabelFrame(tab, text="Windowing")
        window_opts.grid(row=1, column=0, sticky="ew", pady=8)
        for idx, (label, key, tip) in enumerate(
            [
                ("Window size", "window_size", "Override the number of frames per inference window."),
                ("Stride", "window_stride", "Override how many frames to advance between windows."),
                ("Batch size", "batch_size", "Batch size for neural network inference."),
                ("Prob threshold", "prob_threshold", "Minimum class probability required to label a bout."),
                ("Device", "device", "Device for inference (auto selects CUDA when available)."),
                ("Progress interval", "progress_interval", "Emit a log line every N processed windows."),
            ]
        ):
            self._grid_entry(window_opts, idx // 3, idx % 3, label, self._infer_vars[key], tooltip=tip)
        toggles = ttk.Frame(window_opts)
        toggles.grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))
        for text, key, tip in (
            ("Keep unknown", "keep_unknown", "Include low-confidence regions as 'Unknown' rows in the CSV."),
            ("Skip tail", "skip_tail", "Skip incomplete trailing windows instead of padding them."),
        ):
            chk = ttk.Checkbutton(toggles, text=text, variable=self._infer_vars[key])
            chk.pack(side="left", padx=(0, 12))
            self._attach_tooltip(chk, tip)

        yolo = ttk.LabelFrame(tab, text="Cropping")
        yolo.grid(row=2, column=0, sticky="ew")
        for idx, (label, key, tip) in enumerate(
            [
                ("Crop size", "crop_size", "Override frame crop resolution, typically matches training."),
                ("Pad ratio", "pad_ratio", "Padding applied around each detection."),
                ("YOLO conf", "yolo_conf", "Minimum detection confidence during inference."),
                ("YOLO NMS IoU", "yolo_iou", "IOU threshold used during NMS."),
                ("YOLO img size", "yolo_imgsz", "Resolution passed to YOLO during inference."),
                ("YOLO max det", "yolo_max_det", "Maximum detections per frame returned by YOLO."),
            ]
        ):
            self._grid_entry(yolo, idx // 2, idx % 2, label, self._infer_vars[key], tooltip=tip)
        keep_last = ttk.Checkbutton(
            yolo,
            text="Keep last box",
            variable=self._infer_vars["keep_last_box"],
        )
        keep_last.grid(row=3, column=0, columnspan=2, sticky="w", pady=4, padx=4)
        self._attach_tooltip(keep_last, "Reuse the previous bounding box when YOLO misses the subject.")

        button = ttk.Button(
            tab,
            text="Run Inference",
            command=self._run_inference,
        )
        self._register_action_button(button)
        button.grid(row=3, column=0, sticky="e", pady=(8, 0))
        self._attach_tooltip(button, "Launch BehaviorScope inference with the selected settings.")

    def _build_review_tab(self) -> None:
        tab = self._create_scrollable_tab("Review")
        tab.columnconfigure(0, weight=1)
        self._review_vars = {
            "video_path": tk.StringVar(),
            "annotations_csv": tk.StringVar(),
            "output_video": tk.StringVar(),
            "no_display": tk.BooleanVar(value=False),
            "font_scale": tk.StringVar(value="0.7"),
            "thickness": tk.StringVar(value="2"),
            "wait_ms": tk.StringVar(value="1"),
        }

        paths = ttk.LabelFrame(tab, text="Paths")
        paths.grid(row=0, column=0, sticky="ew")
        paths.columnconfigure(1, weight=1)
        self._path_input(
            paths,
            row=0,
            label="Video",
            var=self._review_vars["video_path"],
            browse="file",
            filetypes=(("Video files", "*.mp4;*.avi;*.mov;*.mkv"), ("All files", "*.*")),
            tooltip="Source video used when rendering the annotation overlay.",
        )
        self._path_input(
            paths,
            row=1,
            label="Annotations CSV",
            var=self._review_vars["annotations_csv"],
            browse="file",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
            tooltip="Merged BehaviorScope CSV produced by the inference step.",
        )
        self._path_input(
            paths,
            row=2,
            label="Output video",
            var=self._review_vars["output_video"],
            browse="save",
            filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")),
            tooltip="Optional path for saving the annotated playback.",
        )

        opts = ttk.LabelFrame(tab, text="Overlay Options")
        opts.grid(row=1, column=0, sticky="ew", pady=8)
        for idx, (label, key, tip) in enumerate(
            [
                ("Font scale", "font_scale", "Adjust the OpenCV overlay font size."),
                ("Thickness", "thickness", "Pixel thickness for overlay text and borders."),
                ("Wait (ms)", "wait_ms", "Delay between frames; increase to slow playback."),
            ]
        ):
            self._grid_entry(opts, 0, idx, label, self._review_vars[key], tooltip=tip)
        no_display = ttk.Checkbutton(
            opts,
            text="Disable display (save only)",
            variable=self._review_vars["no_display"],
        )
        no_display.grid(row=1, column=0, columnspan=3, sticky="w", pady=6, padx=4)
        self._attach_tooltip(no_display, "Skip the live preview window and only render the optional output video.")

        button = ttk.Button(
            tab,
            text="Launch Reviewer",
            command=self._run_review,
        )
        self._register_action_button(button)
        button.grid(row=2, column=0, sticky="e")
        self._attach_tooltip(button, "Open the overlay viewer to audit predicted annotations.")

    def _create_scrollable_tab(self, title: str) -> ttk.Frame:
        tab = ttk.Frame(self._notebook)
        self._notebook.add(tab, text=title)
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        canvas = tk.Canvas(tab, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        canvas.configure(yscrollincrement=20)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)

        content = ttk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=content, anchor="nw")
        content.columnconfigure(0, weight=1)

        content.bind(
            "<Configure>",
            lambda event, c=canvas: c.configure(scrollregion=c.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event, c=canvas, w_id=window_id: c.itemconfigure(w_id, width=event.width),
        )

        self._register_scroll_canvas(canvas)
        return content

    def _register_scroll_canvas(self, canvas: tk.Canvas) -> None:
        self._scroll_canvases.append(canvas)
        if self._mousewheel_bound:
            return
        self._mousewheel_bound = True
        for sequence in ("<MouseWheel>", "<Shift-MouseWheel>", "<Button-4>", "<Button-5>"):
            self.bind(sequence, self._on_mousewheel, add="+")

    def _scroll_canvas_for_event(self, event) -> tk.Canvas | None:
        try:
            x_root, y_root = event.x_root, event.y_root
        except Exception:
            return None
        for canvas in self._scroll_canvases:
            if not canvas.winfo_exists():
                continue
            try:
                rel_x = int(x_root - canvas.winfo_rootx())
                rel_y = int(y_root - canvas.winfo_rooty())
                if canvas.winfo_contains(rel_x, rel_y):
                    return canvas
            except Exception:
                continue
        return None

    def _canvas_has_vertical_overflow(self, canvas: tk.Canvas) -> bool:
        try:
            bbox = canvas.bbox("all")
        except Exception:
            return False
        if not bbox:
            return False
        try:
            _, top, _, bottom = bbox
        except Exception:
            return False
        try:
            canvas_height = max(canvas.winfo_height(), 1)
        except Exception:
            return False
        content_height = max(bottom - top, 0)
        return content_height > canvas_height + 1

    def _on_mousewheel(self, event) -> str | None:
        canvas = self._scroll_canvas_for_event(event)
        if not canvas or not canvas.winfo_exists():
            return None
        if not self._canvas_has_vertical_overflow(canvas):
            return None

        widget = event.widget
        if widget not in (canvas, self) and hasattr(widget, "yview"):
            return None

        num = getattr(event, "num", None)
        if num in (4, 5):
            direction = -1 if num == 4 else 1
            first, last = canvas.yview()
            if direction < 0 and first <= 0.0:
                return "break"
            if direction > 0 and last >= 1.0:
                return "break"
            canvas.yview_scroll(direction, "units")
            return "break"

        delta = getattr(event, "delta", 0)
        if delta == 0:
            return None
        step = -1 if delta > 0 else 1
        if abs(delta) >= 120:
            step *= max(1, abs(delta) // 120)
        first, last = canvas.yview()
        if step < 0 and first <= 0.0:
            return "break"
        if step > 0 and last >= 1.0:
            return "break"
        canvas.yview_scroll(step, "units")
        return "break"

    def _path_input(
        self,
        parent: ttk.Frame,
        *,
        row: int,
        label: str,
        var: tk.StringVar,
        browse: str,
        filetypes: Sequence[tuple[str, str]] | None = None,
        tooltip: str | None = None,
        button_tooltip: str | None = None,
    ) -> None:
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", pady=2)
        parent.grid_columnconfigure(1, weight=1)
        self._attach_tooltip(lbl, tooltip)
        self._attach_tooltip(entry, tooltip)

        def _browse() -> None:
            if browse == "dir":
                path = filedialog.askdirectory(parent=self, title=f"Select {label}")
            elif browse == "save":
                path = filedialog.asksaveasfilename(
                    parent=self,
                    title=f"Select {label}",
                    filetypes=filetypes or (("All files", "*.*"),),
                )
            else:
                path = filedialog.askopenfilename(
                    parent=self,
                    title=f"Select {label}",
                    filetypes=filetypes or (("All files", "*.*"),),
                )
            if path:
                var.set(path)

        button = ttk.Button(parent, text="Browse", command=_browse)
        button.grid(
            row=row,
            column=2,
            padx=(8, 0),
            pady=2,
        )
        self._attach_tooltip(button, button_tooltip or tooltip)

    def _grid_entry(
        self,
        parent: ttk.Frame,
        row: int,
        column: int,
        label: str,
        var: tk.StringVar,
        columnspan: int = 1,
        tooltip: str | None = None,
    ) -> None:
        base_col = column * 2
        lbl = ttk.Label(parent, text=label)
        lbl.grid(
            row=row,
            column=base_col,
            sticky="w",
            padx=(0, 6),
            pady=2,
        )
        entry = ttk.Entry(parent, textvariable=var, width=16)
        entry.grid(
            row=row,
            column=base_col + 1,
            columnspan=columnspan,
            sticky="ew",
            padx=(0, 12),
            pady=2,
        )
        parent.grid_columnconfigure(base_col + 1, weight=1)
        self._attach_tooltip(lbl, tooltip)
        self._attach_tooltip(entry, tooltip)

    def _register_action_button(self, button: ttk.Button) -> None:
        self._action_buttons.append(button)

    def _attach_tooltip(self, widget, text: str | None) -> None:
        if not text or widget is None:
            return
        self._tooltips.append(Tooltip(widget, text))

    def _require_path(self, var: tk.StringVar, description: str) -> str | None:
        value = var.get().strip()
        if not value:
            self._show_error(f"{description} is required.")
            return None
        return value

    def _maybe_number(
        self,
        args: list[str],
        flag: str,
        var: tk.StringVar,
        cast: Callable[[str], float | int],
    ) -> bool:
        raw = var.get().strip()
        if not raw:
            return True
        try:
            value = cast(raw)
        except ValueError:
            self._show_error(f"Invalid value for {flag}: {raw}")
            return False
        args.extend([flag, str(value)])
        return True

    def _run_prepare_sequences(self) -> None:
        dataset_root = self._require_path(self._prepare_vars["dataset_root"], "Dataset root")
        yolo_weights = self._require_path(self._prepare_vars["yolo_weights"], "YOLO weights")
        if not dataset_root or not yolo_weights:
            return
        if (
            not self._prepare_vars["save_frame_crops"].get()
            and not self._prepare_vars["save_sequence_npz"].get()
        ):
            self._show_error("Enable at least one storage option (crops or NPZ).")
            return

        args = [
            "--dataset_root",
            dataset_root,
            "--yolo_weights",
            yolo_weights,
        ]
        output_root = self._prepare_vars["output_root"].get().strip()
        if output_root:
            args.extend(["--output_root", output_root])
        manifest_path = self._prepare_vars["manifest_path"].get().strip()
        if manifest_path:
            args.extend(["--manifest_path", manifest_path])

        for flag, key, cast in (
            ("--window_size", "window_size", int),
            ("--window_stride", "window_stride", int),
            ("--crop_size", "crop_size", int),
            ("--pad_ratio", "pad_ratio", float),
            ("--yolo_conf", "yolo_conf", float),
            ("--yolo_iou", "yolo_iou", float),
            ("--yolo_imgsz", "yolo_imgsz", int),
            ("--yolo_max_det", "yolo_max_det", int),
            ("--train_ratio", "train_ratio", float),
            ("--val_ratio", "val_ratio", float),
            ("--test_ratio", "test_ratio", float),
            ("--seed", "seed", int),
            ("--min_sequence_count", "min_sequence_count", int),
            ("--default_fps", "default_fps", float),
        ):
            if not self._maybe_number(args, flag, self._prepare_vars[key], cast):
                return

        device = self._prepare_vars["device"].get().strip()
        if device:
            args.extend(["--device", device])

        for flag, key in (
            ("--save_frame_crops", "save_frame_crops"),
            ("--save_sequence_npz", "save_sequence_npz"),
            ("--keep_last_box", "keep_last_box"),
            ("--skip_existing", "skip_existing"),
        ):
            if self._prepare_vars[key].get():
                args.append(flag)

        self._start_command(
            "prepare_sequences_from_yolo.py",
            args,
            "Sequence preparation",
        )

    def _run_training(self) -> None:
        manifest = self._require_path(self._train_vars["manifest_path"], "Manifest path")
        output_dir = self._require_path(self._train_vars["output_dir"], "Output directory")
        if not manifest or not output_dir:
            return
        args = [
            "--manifest_path",
            manifest,
            "--output_dir",
            output_dir,
        ]
        run_name = self._train_vars["run_name"].get().strip()
        if run_name:
            args.extend(["--run_name", run_name])

        for flag, key, cast in (
            ("--num_frames", "num_frames", int),
            ("--frame_size", "frame_size", int),
            ("--batch_size", "batch_size", int),
            ("--epochs", "epochs", int),
            ("--lr", "lr", float),
            ("--weight_decay", "weight_decay", float),
            ("--num_workers", "num_workers", int),
            ("--patience", "patience", int),
            ("--seed", "seed", int),
            ("--hidden_dim", "hidden_dim", int),
            ("--num_layers", "num_layers", int),
            ("--dropout", "dropout", float),
            ("--temporal_attention_layers", "temporal_attention_layers", int),
            ("--attention_heads", "attention_heads", int),
            ("--slowfast_alpha", "slowfast_alpha", int),
            ("--slowfast_fusion_ratio", "slowfast_fusion_ratio", float),
            ("--slowfast_base_channels", "slowfast_base_channels", int),
            ("--log_interval", "log_interval", int),
        ):
            if not self._maybe_number(args, flag, self._train_vars[key], cast):
                return

        device = self._train_vars["device"].get().strip()
        if device:
            args.extend(["--device", device])

        backbone = self._train_vars["backbone"].get().strip()
        if backbone:
            args.extend(["--backbone", backbone])

        sequence_model = self._train_vars["sequence_model"].get().strip()
        if sequence_model:
            args.extend(["--sequence_model", sequence_model])

        scheduler = self._train_vars["lr_scheduler"].get().strip()
        if scheduler:
            args.extend(["--lr_scheduler", scheduler])
            if scheduler == "cosine":
                if not self._maybe_number(
                    args,
                    "--scheduler_tmax",
                    self._train_vars["scheduler_tmax"],
                    int,
                ):
                    return
                if not self._maybe_number(
                    args,
                    "--scheduler_min_lr",
                    self._train_vars["scheduler_min_lr"],
                    float,
                ):
                    return

        for flag, key in (
            ("--bidirectional", "bidirectional"),
            ("--attention_pool", "attention_pool"),
            ("--feature_se", "feature_se"),
            ("--no_pretrained_backbone", "no_pretrained_backbone"),
            ("--train_backbone", "train_backbone"),
            ("--disable_augment", "disable_augment"),
        ):
            if self._train_vars[key].get():
                args.append(flag)

        self._start_command(
            "train_behavior_lstm.py",
            args,
            "Training run",
        )

    def _run_inference(self) -> None:
        model = self._require_path(self._infer_vars["model_path"], "Model checkpoint")
        video = self._require_path(self._infer_vars["video_path"], "Video file")
        yolo_weights = self._require_path(self._infer_vars["yolo_weights"], "YOLO weights")
        if not model or not video or not yolo_weights:
            return
        args = [
            "--model_path",
            model,
            "--video_path",
            video,
            "--yolo_weights",
            yolo_weights,
        ]
        for key, flag in (
            ("output_csv", "--output_csv"),
            ("output_json", "--output_json"),
        ):
            value = self._infer_vars[key].get().strip()
            if value:
                args.extend([flag, value])

        for flag, key, cast in (
            ("--window_size", "window_size", int),
            ("--window_stride", "window_stride", int),
            ("--batch_size", "batch_size", int),
            ("--prob_threshold", "prob_threshold", float),
            ("--crop_size", "crop_size", int),
            ("--pad_ratio", "pad_ratio", float),
            ("--yolo_conf", "yolo_conf", float),
            ("--yolo_iou", "yolo_iou", float),
            ("--yolo_imgsz", "yolo_imgsz", int),
            ("--yolo_max_det", "yolo_max_det", int),
            ("--progress_interval", "progress_interval", int),
        ):
            if not self._maybe_number(args, flag, self._infer_vars[key], cast):
                return

        device = self._infer_vars["device"].get().strip()
        if device:
            args.extend(["--device", device])

        for flag, key in (
            ("--keep_unknown", "keep_unknown"),
            ("--skip_tail", "skip_tail"),
            ("--keep_last_box", "keep_last_box"),
        ):
            if self._infer_vars[key].get():
                args.append(flag)

        self._start_command(
            "infer_behavior_lstm.py",
            args,
            "Inference run",
        )

    def _run_review(self) -> None:
        video = self._require_path(self._review_vars["video_path"], "Video path")
        csv_path = self._require_path(self._review_vars["annotations_csv"], "Annotations CSV")
        if not video or not csv_path:
            return
        args = [
            "--video_path",
            video,
            "--annotations_csv",
            csv_path,
        ]
        output_video = self._review_vars["output_video"].get().strip()
        if output_video:
            args.extend(["--output_video", output_video])

        for flag, key, cast in (
            ("--font_scale", "font_scale", float),
            ("--thickness", "thickness", int),
            ("--wait_ms", "wait_ms", int),
        ):
            if not self._maybe_number(args, flag, self._review_vars[key], cast):
                return

        if self._review_vars["no_display"].get():
            args.append("--no_display")

        self._start_command(
            "review_annotations.py",
            args,
            "Annotation review",
        )

    def _start_command(self, script_name: str, args: list[str], description: str) -> None:
        if self._runner is not None:
            self._show_error("Another BehaviorScope task is currently running.")
            return

        script_path = self._bs_root / script_name
        if not script_path.exists():
            self._show_error(f"Cannot locate {script_name} under {self._bs_root}.")
            return

        command = [sys.executable, str(script_path)] + args
        self._append_log(f"$ {_format_command(command)}")
        self._log_to_app(f"BehaviorScope task started: {description}")
        self._current_task = description
        self._set_running(True)
        self._runner = CommandRunner(
            command=command,
            cwd=self._bs_root,
            log_callback=self._threadsafe_log,
            completion_callback=self._threadsafe_completion,
        )
        self._runner.start()

    def _threadsafe_log(self, message: str) -> None:
        try:
            self.after(0, self._append_log, message)
        except tk.TclError:
            pass

    def _threadsafe_completion(self, exit_code: int) -> None:
        try:
            self.after(0, self._on_command_complete, exit_code)
        except tk.TclError:
            pass

    def _on_command_complete(self, exit_code: int) -> None:
        task = self._current_task or "BehaviorScope task"
        status = "completed" if exit_code == 0 else f"failed ({exit_code})"
        self._append_log(f"{task} {status}.")
        self._log_to_app(f"{task} {status}.", level="INFO" if exit_code == 0 else "ERROR")
        if exit_code != 0:
            messagebox.showerror(
                "BehaviorScope Toolkit",
                f"{task} failed (code {exit_code}).",
                parent=self,
            )
        self._runner = None
        self._current_task = ""
        self._set_running(False)

    def _append_log(self, message: str) -> None:
        self._log_widget.configure(state="normal")
        self._log_widget.insert("end", message + "\n")
        self._log_widget.see("end")
        self._log_widget.configure(state="disabled")

    def _set_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        for button in self._action_buttons:
            button.configure(state=state)

    def _log_to_app(self, message: str, level: str = "INFO") -> None:
        logger = getattr(self._main_app, "log_message", None)
        if callable(logger):
            try:
                logger(message, level)
            except Exception:
                pass

    def _show_error(self, message: str) -> None:
        self._append_log(f"[ERROR] {message}")
        messagebox.showerror("BehaviorScope Toolkit", message, parent=self)
        self._log_to_app(message, level="ERROR")

    def _handle_close(self) -> None:
        if self._runner is not None:
            if not messagebox.askyesno(
                "BehaviorScope Toolkit",
                "A task is still running. Close the window anyway?",
                parent=self,
            ):
                return
            self._runner.stop()
        self.destroy()
