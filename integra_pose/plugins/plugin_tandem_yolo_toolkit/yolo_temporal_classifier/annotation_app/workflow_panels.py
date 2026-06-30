from __future__ import annotations

import shlex
import subprocess
import sys
import re
import json
from pathlib import Path
from typing import Callable
from urllib.parse import unquote, urlparse

from PySide6.QtCore import QProcess, QProcessEnvironment
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


THIS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = THIS_DIR.parent


def _script_path(name: str) -> Path:
    return THIS_DIR / name


def _split_extra_args(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return shlex.split(text, posix=False)


def _display_command(cmd: list[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in cmd])


def _python_invocation() -> list[str]:
    return [sys.executable, "-u"]


def _local_path_text(text: str) -> str:
    value = text.strip()
    parsed = urlparse(value)
    if parsed.scheme.lower() != "file":
        return value
    if parsed.netloc:
        local = f"//{parsed.netloc}{unquote(parsed.path)}"
    else:
        local = unquote(parsed.path)
    if re.match(r"^/[A-Za-z]:/", local):
        local = local[1:]
    return str(Path(local))


class ProcessRunner(QWidget):
    def __init__(
        self,
        *,
        command_builder: Callable[[], list[str] | None],
        graceful_stop: Callable[[], str | None] | None = None,
        on_success: Callable[[list[str]], None] | None = None,
        show_telemetry: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.command_builder = command_builder
        self.graceful_stop = graceful_stop
        self.on_success = on_success
        self.show_telemetry = bool(show_telemetry)
        self.process: QProcess | None = None
        self.current_cmd: list[str] | None = None
        self._epoch_total: int | None = None
        self._telemetry_labels: dict[str, QLabel] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        button_row = QHBoxLayout()
        self.show_btn = QPushButton("Show command")
        self.show_btn.clicked.connect(self._show_command)
        button_row.addWidget(self.show_btn)
        self.start_btn = QPushButton("Start")
        self.start_btn.setObjectName("PrimaryButton")
        self.start_btn.clicked.connect(self._start)
        button_row.addWidget(self.start_btn)
        self.stop_btn = QPushButton("Stop gracefully")
        self.stop_btn.setToolTip(
            "Request an orderly stop when the workflow supports it. "
            "For training this writes stop_training.flag so the run can finish the current checkpoint path."
        )
        self.stop_btn.clicked.connect(self._stop_gracefully)
        self.stop_btn.setEnabled(False)
        button_row.addWidget(self.stop_btn)
        self.kill_btn = QPushButton("Kill")
        self.kill_btn.setToolTip(
            "Force-kill the process. Use only when graceful stop does not respond; "
            "the current checkpoint or CSV row may be incomplete."
        )
        self.kill_btn.clicked.connect(self._kill)
        self.kill_btn.setEnabled(False)
        button_row.addWidget(self.kill_btn)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        if self.show_telemetry:
            telemetry = QWidget()
            telemetry_layout = QHBoxLayout(telemetry)
            telemetry_layout.setContentsMargins(0, 0, 0, 0)
            telemetry_layout.setSpacing(8)
            for key, label, tip in [
                ("fps", "FPS", "Source frames processed per wall-clock second."),
                ("wps", "Windows/s", "Temporal-classifier windows emitted per wall-clock second."),
                ("t_total", "Window ms", "Average total wall-clock time per emitted prediction window."),
                ("t_cls", "Classifier ms", "Average temporal classifier forward-pass time per window."),
                ("cpu", "CPU", "Process CPU utilization normalized to a 0-100% scale."),
                ("gpu_util", "GPU", "GPU utilization over the latest metrics interval."),
                ("gpu_mem", "GPU Mem", "GPU memory used over the latest metrics interval."),
            ]:
                metric = QLabel(f"{label}: --")
                metric.setObjectName("HintLabel")
                metric.setToolTip(tip)
                telemetry_layout.addWidget(metric)
                self._telemetry_labels[key] = metric
            telemetry_layout.addStretch(1)
            layout.addWidget(telemetry)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(190)
        self.log.setObjectName("ProcessLog")
        layout.addWidget(self.log)

        self.status = QLabel("Ready")
        self.status.setObjectName("HintLabel")
        layout.addWidget(self.status)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("Ready")
        layout.addWidget(self.progress)

    def _command_or_none(self) -> list[str] | None:
        try:
            return self.command_builder()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid command", str(exc))
            return None

    def _show_command(self) -> None:
        cmd = self._command_or_none()
        if not cmd:
            return
        self._append_log("\n" + _display_command(cmd) + "\n")

    def _start(self) -> None:
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            QMessageBox.information(self, "Process already running", "Wait for the current process to finish first.")
            return
        cmd = self._command_or_none()
        if not cmd:
            return
        program = str(cmd[0])
        args = [str(part) for part in cmd[1:]]
        self.current_cmd = [str(part) for part in cmd]
        self._epoch_total = self._cmd_int_value(self.current_cmd, "--epochs")
        self.log.clear()
        self._reset_telemetry()
        self._append_log(_display_command(cmd) + "\n\n")
        process = QProcess(self)
        process.setWorkingDirectory(str(REPO_ROOT))
        process.setProcessChannelMode(QProcess.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        process.setProcessEnvironment(env)
        process.readyReadStandardOutput.connect(self._read_output)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._error)
        self.process = process
        self._set_running(True)
        self.status.setText("Running")
        self.progress.setRange(0, 0)
        self.progress.setFormat("Running...")
        process.start(program, args)
        if not process.waitForStarted(3000):
            self.status.setText("Failed to start")
            self._set_running(False)
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.progress.setFormat("Failed to start")

    def _read_output(self) -> None:
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._append_log(data)
        self._update_progress_from_text(data)
        self._update_telemetry_from_text(data)

    def _finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        self._append_log(f"\n[process exited with code {exit_code}]\n")
        self.status.setText("Finished" if exit_code == 0 else f"Exited with code {exit_code}")
        self.progress.setRange(0, 1)
        self.progress.setValue(1 if exit_code == 0 else 0)
        self.progress.setFormat("Finished" if exit_code == 0 else "Failed")
        self._set_running(False)
        completed_cmd = self.current_cmd
        self.process = None
        self.current_cmd = None
        if exit_code == 0 and completed_cmd is not None and self.on_success is not None:
            try:
                self.on_success(completed_cmd)
            except Exception as exc:
                self._append_log(f"\n[post-run path update failed: {exc}]\n")

    def _error(self, error: QProcess.ProcessError) -> None:
        self.status.setText(f"Process error: {error.name}")

    def _stop_gracefully(self) -> None:
        if self.process is None or self.process.state() == QProcess.NotRunning:
            return
        message: str | None = None
        if self.graceful_stop is not None:
            message = self.graceful_stop()
        if message:
            self._append_log(f"\n[{message}]\n")
            return
        self.process.terminate()
        self._append_log("\n[terminate requested]\n")

    def _kill(self) -> None:
        if self.process is None or self.process.state() == QProcess.NotRunning:
            return
        self.process.kill()
        self._append_log("\n[kill requested]\n")

    def _append_log(self, text: str) -> None:
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QTextCursor.End)

    def _set_running(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.kill_btn.setEnabled(running)

    def _update_progress_from_text(self, text: str) -> None:
        if not self._epoch_total:
            return
        matches = re.findall(r"\bepoch\s+(\d+)\b", text, flags=re.IGNORECASE)
        if not matches:
            return
        epoch = max(int(value) for value in matches)
        total = max(1, int(self._epoch_total))
        epoch = min(epoch, total)
        self.progress.setRange(0, total)
        self.progress.setValue(epoch)
        self.progress.setFormat(f"Epoch {epoch}/{total}")

    def _reset_telemetry(self) -> None:
        if not self._telemetry_labels:
            return
        names = {
            "fps": "FPS",
            "wps": "Windows/s",
            "t_total": "Window ms",
            "t_cls": "Classifier ms",
            "cpu": "CPU",
            "gpu_util": "GPU",
            "gpu_mem": "GPU Mem",
        }
        for key, label in names.items():
            widget = self._telemetry_labels.get(key)
            if widget is not None:
                widget.setText(f"{label}: --")

    def _update_telemetry_from_text(self, text: str) -> None:
        if not self._telemetry_labels or "[metrics]" not in text:
            return
        names = {
            "fps": "FPS",
            "wps": "Windows/s",
            "t_total": "Window ms",
            "t_cls": "Classifier ms",
            "cpu": "CPU",
            "gpu_util": "GPU",
            "gpu_mem": "GPU Mem",
        }
        for line in text.splitlines():
            if "[metrics]" not in line:
                continue
            for key, label in names.items():
                match = re.search(rf"\b{re.escape(key)}=([^\s]+)", line)
                if not match:
                    continue
                widget = self._telemetry_labels.get(key)
                if widget is not None:
                    widget.setText(f"{label}: {match.group(1)}")

    @staticmethod
    def _cmd_int_value(cmd: list[str], flag: str) -> int | None:
        try:
            idx = cmd.index(flag)
        except ValueError:
            return None
        if idx + 1 >= len(cmd):
            return None
        try:
            return int(cmd[idx + 1])
        except ValueError:
            return None


class WorkflowPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.form = QFormLayout()
        self.form.setLabelAlignment(self.form.labelAlignment())
        self.extra_args = QLineEdit()
        self.extra_args.setPlaceholderText("Optional raw CLI flags, e.g. --some_flag value")

    def _path_row(
        self,
        label: str,
        *,
        mode: str,
        file_filter: str = "All files (*.*)",
        save: bool = False,
        default: str = "",
    ) -> QLineEdit:
        edit = QLineEdit(default)
        button = QPushButton("Browse")

        def browse() -> None:
            start_path = _local_path_text(edit.text()) or str(REPO_ROOT)
            if mode == "dir":
                path = QFileDialog.getExistingDirectory(self, f"Select {label}", start_path)
            elif save:
                path, _ = QFileDialog.getSaveFileName(self, f"Select {label}", start_path, file_filter)
            else:
                path, _ = QFileDialog.getOpenFileName(self, f"Select {label}", start_path, file_filter)
            if path:
                edit.setText(_local_path_text(path))

        button.clicked.connect(browse)
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        self.form.addRow(label, row)
        return edit

    def _spin_row(self, label: str, value: int, minimum: int = 0, maximum: int = 100000) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        self.form.addRow(label, spin)
        return spin

    def _double_row(self, label: str, value: float, minimum: float = 0.0, maximum: float = 100000.0, decimals: int = 6) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.setSingleStep(0.01)
        self.form.addRow(label, spin)
        return spin

    def _combo_row(self, label: str, values: list[str], current: str) -> QComboBox:
        combo = QComboBox()
        combo.addItems(values)
        idx = combo.findText(current)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        self.form.addRow(label, combo)
        return combo

    def _line_row(self, label: str, default: str = "") -> QLineEdit:
        edit = QLineEdit(default)
        self.form.addRow(label, edit)
        return edit

    def _checkbox_row(self, label: str, checked: bool = False) -> QCheckBox:
        check = QCheckBox(label)
        check.setChecked(checked)
        self.form.addRow("", check)
        return check

    def _wrap(self, title: str, runner: ProcessRunner) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(14, 14, 14, 14)
        header = QLabel(title)
        header.setObjectName("SectionHeader")
        layout.addWidget(header)
        layout.addLayout(self.form)
        self.form.addRow("Extra CLI flags", self.extra_args)
        layout.addWidget(runner)
        layout.addStretch(1)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

    def _require(self, edit: QLineEdit, label: str) -> str:
        value = _local_path_text(edit.text())
        if not value:
            raise ValueError(f"{label} is required.")
        return value

    def _optional_path(self, edit: QLineEdit) -> str:
        return _local_path_text(edit.text())

    def _extra(self) -> list[str]:
        return _split_extra_args(self.extra_args.text())


_YOLO_WEIGHTS_TOOLTIP = (
    "YOLO-pose checkpoint (.pt) - used for keypoint detection before pose/social feature extraction.\n"
    "\n"
    "- Generic (works out of the box, auto-downloads):\n"
    "    yolo11n-pose.pt\n"
    "\n"
    "- Domain-finetuned (recommended for best accuracy):\n"
    "    Train with Ultralytics on your own pose dataset:\n"
    "      yolo train task=pose model=yolo11n-pose.pt data=your_pose.yaml epochs=100\n"
    "    then point this field at the resulting best.pt.\n"
    "\n"
    "- Pre-trained checkpoints (if available):\n"
    "    Download from the project's GitHub Releases page.\n"
    "\n"
    "See README for details."
)


class PrepareFullVideoDatasetPanel(WorkflowPanel):
    def __init__(self, parent: QWidget | None = None, on_success: Callable[[list[str]], None] | None = None) -> None:
        super().__init__(parent)
        self.source_manifest_csv = self._path_row(
            "source_manifest.csv",
            mode="file",
            file_filter="CSV (*.csv);;All files (*.*)",
        )
        self.source_manifest_csv.setToolTip(
            "Input CSV from Project -> Export full-video annotations. "
            "It lists source videos, annotation files, splits, and class metadata."
        )
        self.class_names_file = self._path_row(
            "class_names.txt",
            mode="file",
            file_filter="Text (*.txt);;All files (*.*)",
        )
        self.yolo_weights = self._path_row("YOLO pose weights", mode="file", file_filter="PyTorch (*.pt);;All files (*.*)")
        self.yolo_weights.setPlaceholderText("Required")
        self.yolo_weights.setToolTip(_YOLO_WEIGHTS_TOOLTIP)
        self.output_root = self._path_row("Output dataset root", mode="dir")
        self.manifest_path = self._path_row(
            "Training manifest output",
            mode="file",
            file_filter="JSON (*.json);;All files (*.*)",
            save=True,
        )
        self.manifest_path.setToolTip(
            "Output JSON written by this Prepare step. "
            "Use this file as the Manifest path in the Train tab. "
            "When left blank, TandemYTC writes <Output dataset root>/sequence_manifest.json."
        )
        self.window_size = self._spin_row("Window size", 32, 1)
        self.window_stride = self._spin_row("Window stride", 16, 1)
        self.n_animals = self._spin_row("Animals", 2, 1)
        self.crop_size = self._spin_row("Pose window size", 224, 16)
        self.yolo_imgsz = self._spin_row("YOLO image size", 640, 32)
        self.yolo_batch = self._spin_row("YOLO batch", 64, 1)
        self.storage_mode = self._combo_row("Storage mode", ["hdf5", "npz"], "hdf5")
        self.storage_mode.setToolTip(
            "hdf5 is the compact default: one per-video pose/social store with manifest windows. "
            "npz writes one pose/social file per window and is mainly useful for debugging."
        )
        self.hdf5_compresslevel = self._spin_row("HDF5 gzip level", 1, 0, 9)
        self.hdf5_compresslevel.setToolTip("Compression level for compact HDF5 stores. Level 1 is usually a good speed/size balance.")
        self.npz_writers = self._spin_row("NPZ writers", 4, 1)
        self.npz_writers.setToolTip("Only used when Storage mode is npz.")
        self.npz_compresslevel = self._spin_row("NPZ deflate level", 1, 0, 9)
        self.npz_compresslevel.setToolTip("Only used when Storage mode is npz.")
        self.label_min_dominance = self._double_row("Label min dominance", 0.5, 0.0, 1.0, 3)
        self.other_subsample = self._double_row("Other subsample", 0.3, 0.0, 1.0, 3)
        self.device = self._line_row("Device", "cuda:0")
        self.skip_existing = self._checkbox_row("Resume completed videos", True)
        self.validate_manifest = self._checkbox_row("Validate manifest after build", True)
        self.validation_sample_limit = self._spin_row("Validation sample limit (0 = all)", 1000, 0)
        runner = ProcessRunner(command_builder=self.build_command, on_success=on_success)
        self._wrap("Prepare full-video annotations into temporal windows", runner)

    def build_command(self) -> list[str]:
        script = _script_path("prepare_full_video_npz.py")
        if not script.exists():
            raise ValueError(f"Missing script: {script}")
        cmd = [
            *_python_invocation(),
            str(script),
            "--source_manifest_csv",
            self._require(self.source_manifest_csv, "source_manifest.csv"),
            "--class_names_file",
            self._require(self.class_names_file, "class_names.txt"),
            "--yolo_weights",
            self._require(self.yolo_weights, "YOLO pose weights"),
            "--window_size",
            str(self.window_size.value()),
            "--window_stride",
            str(self.window_stride.value()),
            "--crop_size",
            str(self.crop_size.value()),
            "--n_animals",
            str(self.n_animals.value()),
            "--yolo_imgsz",
            str(self.yolo_imgsz.value()),
            "--yolo_batch",
            str(self.yolo_batch.value()),
            "--storage_mode",
            self.storage_mode.currentText(),
            "--hdf5_compresslevel",
            str(self.hdf5_compresslevel.value()),
            "--npz_writers",
            str(self.npz_writers.value()),
            "--npz_compresslevel",
            str(self.npz_compresslevel.value()),
            "--label_min_dominance",
            str(self.label_min_dominance.value()),
            "--other_subsample",
            str(self.other_subsample.value()),
            "--device",
            self.device.text().strip() or "cuda:0",
            "--validation_sample_limit",
            str(self.validation_sample_limit.value()),
        ]
        output_root = self._optional_path(self.output_root)
        if output_root:
            cmd.extend(["--output_root", output_root])
        manifest_path = self._optional_path(self.manifest_path)
        if manifest_path:
            cmd.extend(["--manifest_path", manifest_path])
        if self.skip_existing.isChecked():
            cmd.append("--skip_existing")
        if self.validate_manifest.isChecked():
            cmd.append("--validate_manifest")
        else:
            cmd.append("--no-validate_manifest")
        cmd.extend(self._extra())
        return cmd


class TrainPanel(WorkflowPanel):
    def __init__(self, parent: QWidget | None = None, on_success: Callable[[list[str]], None] | None = None) -> None:
        super().__init__(parent)
        self.manifest_path = self._path_row("Training manifest JSON", mode="file", file_filter="JSON (*.json);;All files (*.*)")
        self.yolo_weights = self._path_row("YOLO pose weights", mode="file", file_filter="PyTorch (*.pt);;All files (*.*)")
        self.yolo_weights.setPlaceholderText("Required — see README 'Getting YOLO weights'")
        self.yolo_weights.setToolTip(_YOLO_WEIGHTS_TOOLTIP)
        self.project = self._path_row("Run project folder", mode="dir", default=str(THIS_DIR / "runs"))
        self.name = self._line_row("Run name", "yolo_temporal_run")
        self.epochs = self._spin_row("Epochs", 10, 1)
        self.patience = self._spin_row("Patience", 5, 0)
        self.batch = self._spin_row("Batch size", 24, 1)
        self.num_workers = self._spin_row("Workers", 2, 0)
        self.lr = self._line_row("Learning rate", "5e-5")
        self.weight_decay = self._line_row("Weight decay", "5e-4")
        self.dropout = self._double_row("Dropout", 0.6, 0.0, 1.0, 3)
        self.scheduler = self._combo_row("Scheduler", ["plateau", "cosine", "none"], "plateau")
        self.class_weighting = self._combo_row("Class weighting", ["sqrt_inverse", "inverse", "none"], "sqrt_inverse")
        self.class_weight_clamp = self._double_row("Class weight clamp", 1.5, 0.0, 100.0, 3)
        self.train_sampler = self._combo_row("Train sampler", ["random", "weighted"], "random")
        self.n_animals = self._spin_row("Animals", 2, 1)
        self.num_keypoints = self._spin_row("Keypoints (0 = auto)", 0, 0)
        self.hidden_dim = self._spin_row("Hidden dim", 256, 16)
        self.num_temporal_layers = self._spin_row("Temporal layers", 1, 1, 8)
        self.sequence_model = self._combo_row(
            "Temporal head",
            ["tcn", "gated_attention_lstm", "attention_lstm", "lstm", "tcn_attention"],
            "tcn",
        )
        self.attention_heads = self._spin_row("Attention heads", 4, 1)
        self.positional_encoding = self._combo_row("Positional encoding", ["sinusoidal", "learned", "none"], "sinusoidal")
        self.device = self._line_row("Device", "cuda")
        self.seed = self._spin_row("Seed", 0, 0)
        self.log_interval = self._spin_row("Log interval", 200, 0)
        self.resume = self._checkbox_row("Resume from last checkpoint", False)
        self.amp = self._checkbox_row("AMP", True)
        self.per_class_metrics = self._checkbox_row("Per-class metrics", True)
        self.confusion_matrix = self._checkbox_row("Confusion matrix", True)
        self.disable_threshold_decoder = self._checkbox_row("Disable auto threshold decoder", False)
        # Temporal splitter — manuscript-grade GT fitting
        self.temporal_splitter_annot_root = self._path_row(
            "Splitter annot root (optional)",
            mode="dir",
            default="",
        )
        self.temporal_splitter_strict_annot = self._checkbox_row(
            "Strict annot mode (skip videos missing .annot)", False
        )
        self.disable_temporal_splitter = self._checkbox_row("Disable temporal splitter", False)
        self._install_training_tooltips()
        self._add_training_artifact_row()
        self._add_temporal_preflight_row()
        runner = ProcessRunner(
            command_builder=self.build_command,
            graceful_stop=self.request_stop_after_epoch,
            on_success=on_success,
        )
        self._wrap("Train TandemYTC", runner)

    def _install_training_tooltips(self) -> None:
        self.manifest_path.setToolTip(
            "Full-video sequence_manifest.json produced by the Prepare full-video workflow. "
            "Do not use source_manifest.csv here; that CSV is only the Prepare Full Video input. "
            "Pre-flight reads this JSON to summarize window size, stride, classes, and sample counts."
        )
        self.project.setToolTip("Run output folder. A subfolder named by Run name will hold checkpoints, logs, and config.json.")
        self.name.setToolTip("Run folder name inside the selected project folder.")
        self.epochs.setToolTip("Maximum training epochs. Early stopping may stop sooner based on validation performance.")
        self.patience.setToolTip("Validation epochs without improvement before early stopping.")
        self.batch.setToolTip("Training windows per optimizer step. Lower this if GPU memory is tight.")
        self.hidden_dim.setToolTip("Temporal-head width. Higher values increase capacity and memory/latency.")
        self.num_temporal_layers.setToolTip("Number of recurrent layers for LSTM heads or residual causal blocks for TCN heads.")
        self.sequence_model.setToolTip(
            "Temporal head used after YOLO, pose, and social-geometry features are fused. "
            "Use pre-flight to see the selected head's context and capacity."
        )
        self.attention_heads.setToolTip("Attention heads used by attention pooling heads. Keep this compatible with hidden dim.")
        self.positional_encoding.setToolTip("Optional frame-position signal added before temporal modeling.")
        self.resume.setToolTip("Load last_checkpoint.pt from the selected run folder and append to training_log.csv when present.")
        self.amp.setToolTip("Mixed precision training. Usually faster on CUDA with lower memory use.")
        self.per_class_metrics.setToolTip("Report class-level precision, recall, and F1 during validation.")
        self.confusion_matrix.setToolTip("Write validation confusion matrices during training.")
        self.disable_threshold_decoder.setToolTip("Skip validation-calibrated confidence threshold fitting after training.")
        self.temporal_splitter_annot_root.setToolTip(
            "Optional folder of source annotations used to fit a validation-calibrated bout splitter."
        )
        self.temporal_splitter_strict_annot.setToolTip("Skip videos missing annotation files when fitting the temporal splitter.")
        self.disable_temporal_splitter.setToolTip("Disable validation-calibrated bout-boundary postprocessing.")

    def _add_training_artifact_row(self) -> None:
        self.artifact_summary = QLabel(
            "Run folder writes config.json, training_log.csv, history.json, "
            "last_checkpoint.pt, best_model.pt, and best_model_macro_f1.pt."
        )
        self.artifact_summary.setWordWrap(True)
        self.artifact_summary.setObjectName("HintLabel")
        self.artifact_summary.setToolTip(
            "training_log.csv is appended once per completed epoch. "
            "last_checkpoint.pt is updated every epoch and is the resume target; "
            "best_model_macro_f1.pt is preferred for inference when available."
        )
        self.form.addRow("Training outputs", self.artifact_summary)

    def _add_temporal_preflight_row(self) -> None:
        self.preflight_btn = QPushButton("Check temporal head")
        self.preflight_btn.setToolTip("Summarize selected temporal head capacity and manifest context before training.")
        self.preflight_btn.clicked.connect(self.show_temporal_preflight)
        self.preflight_status = QLabel("Not checked")
        self.preflight_status.setObjectName("HintLabel")
        self.preflight_status.setToolTip("Last temporal-head pre-flight result.")
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.preflight_btn)
        layout.addWidget(self.preflight_status, 1)
        self.form.addRow("Temporal pre-flight", row)

    def show_temporal_preflight(self) -> None:
        summary = self._temporal_preflight_summary()
        self.preflight_status.setText(str(summary["status"]))
        if summary.get("valid"):
            QMessageBox.information(self, "Temporal head pre-flight", str(summary["message"]))
        else:
            QMessageBox.warning(self, "Temporal head pre-flight", str(summary["message"]))

    def _temporal_preflight_summary(self) -> dict[str, object]:
        manifest = self._read_manifest_summary()
        head = self.sequence_model.currentText()
        hidden_dim = int(self.hidden_dim.value())
        layers = int(self.num_temporal_layers.value())
        heads = int(self.attention_heads.value())
        window_size = manifest.get("window_size")
        head_lines = self._head_capacity_lines(head, hidden_dim, layers, heads, window_size)
        manifest_lines = manifest.get("lines", ["Manifest not read. Select a manifest to include dataset context."])
        manifest_valid = bool(manifest.get("valid"))
        if manifest_valid:
            manifest_header = "Manifest"
            status = f"{head}: hidden {hidden_dim}, layers {layers}"
            if window_size:
                status += f", window {window_size}"
        else:
            manifest_header = "Manifest check failed"
            status = "Invalid training manifest"

        message_parts = [
            "Temporal head",
            f"  Head: {head}",
            f"  Hidden dim: {hidden_dim}",
            f"  Temporal layers: {layers}",
            f"  Attention heads: {heads}",
            f"  Positional encoding: {self.positional_encoding.currentText()}",
            "",
            "Capacity",
            *[f"  {line}" for line in head_lines],
            "",
            manifest_header,
            *[f"  {line}" for line in manifest_lines],
        ]
        if not manifest_valid:
            message_parts.extend([
                "",
                "Fix",
                "  Run Prepare Full Video first, then select its Training manifest output JSON.",
                "  Do not use source_manifest.csv in the Train tab.",
            ])
        message_parts.extend([
            "",
            "Runtime note",
            "  TandemYTC training uses full-video windows. Inference still uses a streaming window with bounded latency.",
        ])
        return {"status": status, "message": "\n".join(message_parts), "valid": manifest_valid}

    def _read_manifest_summary(self) -> dict[str, object]:
        path_text = self._optional_path(self.manifest_path)
        if not path_text:
            return {
                "valid": False,
                "lines": [
                    "No training manifest JSON selected.",
                    "The Train tab needs the JSON written by Prepare Full Video.",
                ],
            }
        path = Path(path_text)
        if not path.is_file():
            return {
                "valid": False,
                "lines": [f"Training manifest JSON not found: {path}"],
            }
        try:
            payload = self._read_training_manifest_json(path)
        except Exception as exc:
            return {"valid": False, "lines": [str(exc)]}
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        splits = payload.get("splits", {}) if isinstance(payload, dict) else {}
        class_to_idx = payload.get("class_to_idx", {}) if isinstance(payload, dict) else {}
        window_size = int(meta.get("window_size", 0) or 0)
        window_stride = int(meta.get("window_stride", 0) or 0)
        schema = str(meta.get("schema_version", "?"))
        split_counts = {
            str(name): len(items) if isinstance(items, list) else 0
            for name, items in splits.items()
        }
        total = sum(split_counts.values())
        lines = [
            f"Schema: {schema}",
            f"Window: {window_size or '?'} frames; stride: {window_stride or '?'} frames",
            f"Classes: {len(class_to_idx)}",
            f"Samples: {total} ({', '.join(f'{k}={v}' for k, v in split_counts.items()) or 'no splits'})",
        ]
        return {"valid": True, "window_size": window_size, "window_stride": window_stride, "lines": lines}

    def _read_training_manifest_json(self, path: Path) -> dict:
        if path.suffix.lower() == ".csv":
            raise ValueError(
                "selected file is source_manifest.csv or another CSV. "
                "Train needs the JSON written by Prepare Full Video."
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                "selected file is not valid JSON. Use the Prepare Full Video "
                "Training manifest output, usually sequence_manifest.json."
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError("training manifest JSON must contain an object.")
        missing = [name for name in ("class_to_idx", "splits") if name not in payload]
        if missing:
            raise ValueError(
                f"missing required field(s) {missing}. This does not look like a "
                "Prepare Full Video training manifest."
            )
        return payload

    def _head_capacity_lines(
        self,
        head: str,
        hidden_dim: int,
        layers: int,
        attention_heads: int,
        window_size: object,
    ) -> list[str]:
        window = int(window_size or 0)
        if head == "tcn":
            receptive = 1 + 2 * ((2 ** layers) - 1)
            covered = min(window, receptive) if window else receptive
            return [
                "Causal residual TCN; classifier reads the final temporal token.",
                f"Receptive field: {receptive} frames ({covered} covered in selected manifest window).",
                "Lowest-latency temporal option; good default for real-time inference.",
            ]
        if head == "tcn_attention":
            receptive = 1 + 2 * ((2 ** layers) - 1)
            return [
                "Causal residual TCN followed by attention pooling over the window.",
                f"TCN receptive field before pooling: {receptive} frames.",
                f"Attention pooling heads: {attention_heads}; stronger boundary context than plain TCN.",
            ]
        if head == "gated_attention_lstm":
            return [
                "LSTM over the full training window with learned gated temporal attention pooling.",
                "Highest temporal pooling capacity among the current lightweight heads.",
                "Use when subtle behavior boundaries benefit from frame-level weighting.",
            ]
        if head == "attention_lstm":
            return [
                "LSTM over the full training window with multi-head attention pooling.",
                f"Attention pooling heads: {attention_heads}.",
                "Good when behavior evidence can occur away from the last frame in the window.",
            ]
        if head == "lstm":
            return [
                "Plain LSTM over the full training window; classifier reads the last output token.",
                "Small recurrent baseline with predictable streaming behavior.",
                "Use this for direct comparison with older LSTM-only experiments.",
            ]
        return ["Attention pooling over the fused per-frame tokens."]

    def build_command(self) -> list[str]:
        script = _script_path("train_y.py")
        if not script.exists():
            raise ValueError(f"Missing script: {script}")
        manifest_path = self._require(self.manifest_path, "Training manifest JSON")
        self._read_training_manifest_json(Path(manifest_path))
        cmd = [
            *_python_invocation(),
            str(script),
            "--manifest_path",
            manifest_path,
            "--yolo_weights",
            self._require(self.yolo_weights, "YOLO pose weights"),
            "--epochs",
            str(self.epochs.value()),
            "--patience",
            str(self.patience.value()),
            "--scheduler",
            self.scheduler.currentText(),
            "--batch",
            str(self.batch.value()),
            "--num_workers",
            str(self.num_workers.value()),
            "--lr",
            self.lr.text().strip() or "5e-5",
            "--weight_decay",
            self.weight_decay.text().strip() or "5e-4",
            "--dropout",
            str(self.dropout.value()),
            "--class_weighting",
            self.class_weighting.currentText(),
            "--class_weight_clamp",
            str(self.class_weight_clamp.value()),
            "--train_sampler",
            self.train_sampler.currentText(),
            "--n_animals",
            str(self.n_animals.value()),
            "--hidden_dim",
            str(self.hidden_dim.value()),
            "--sequence_model",
            self.sequence_model.currentText(),
            "--num_lstm_layers",
            str(self.num_temporal_layers.value()),
            "--attention_heads",
            str(self.attention_heads.value()),
            "--positional_encoding",
            self.positional_encoding.currentText(),
            "--log_interval",
            str(self.log_interval.value()),
            "--seed",
            str(self.seed.value()),
            "--device",
            self.device.text().strip() or "cuda",
            "--project",
            self._require(self.project, "Run project folder"),
            "--name",
            self._require(self.name, "Run name"),
        ]
        if self.num_keypoints.value() > 0:
            cmd.extend(["--num_keypoints", str(self.num_keypoints.value())])
        if self.resume.isChecked():
            cmd.append("--resume")
        if self.amp.isChecked():
            cmd.append("--amp")
        if self.per_class_metrics.isChecked():
            cmd.append("--per_class_metrics")
        if self.confusion_matrix.isChecked():
            cmd.extend(["--confusion_matrix", "--confusion_matrix_interval", "1"])
        if self.disable_threshold_decoder.isChecked():
            cmd.append("--disable_threshold_decoder")
        # Temporal splitter
        annot_root = self._optional_path(self.temporal_splitter_annot_root)
        if annot_root:
            cmd.extend(["--temporal_splitter_annot_root", annot_root])
        if self.temporal_splitter_strict_annot.isChecked():
            cmd.append("--temporal_splitter_strict_annot")
        if self.disable_temporal_splitter.isChecked():
            cmd.append("--disable_temporal_splitter")
        cmd.extend(self._extra())
        return cmd

    def request_stop_after_epoch(self) -> str:
        project = Path(self._optional_path(self.project) or ".")
        name = self.name.text().strip()
        if not name:
            return "stop not requested: run name is empty"
        run_dir = project / name
        run_dir.mkdir(parents=True, exist_ok=True)
        flag = run_dir / "stop_training.flag"
        flag.write_text("stop after current epoch\n", encoding="utf-8")
        return f"requested stop after current epoch via {flag}"


class InferencePanel(WorkflowPanel):
    def __init__(self, *, batch_mode: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.batch_mode = batch_mode
        self.model_path = self._path_row("Model .pt", mode="file", file_filter="PyTorch (*.pt);;All files (*.*)")
        self.model_config = self._path_row("Model config.json", mode="file", file_filter="JSON (*.json);;All files (*.*)")
        self.yolo_weights = self._path_row("YOLO pose weights", mode="file", file_filter="PyTorch (*.pt);;All files (*.*)")
        self.yolo_weights.setPlaceholderText("Required — same .pt used during training")
        self.yolo_weights.setToolTip(_YOLO_WEIGHTS_TOOLTIP)
        source_label = "Source video folder" if batch_mode else "Source video/file/folder/camera"
        self.source = self._path_row(source_label, mode="dir" if batch_mode else "file", file_filter="Videos (*.mp4 *.avi *.mov *.mkv *.seq);;All files (*.*)")
        if batch_mode:
            self.source.setPlaceholderText("Folder containing videos to process")
            self.source.setToolTip("Batch mode processes every supported video in this folder.")
        else:
            self.source.setPlaceholderText("Video path, folder, camera index 0, or RTSP/HTTP stream URL")
            self.source.setToolTip(
                "Use a video path for offline inference, a folder for multiple videos, "
                "0 or another integer for a camera, or an RTSP/HTTP URL for a live stream."
            )
        self.output = self._path_row("Output CSV (single video)", mode="file", file_filter="CSV (*.csv);;All files (*.*)", save=True)
        self.output_dir = self._path_row("Output CSV directory", mode="dir")
        self.output_video = self._path_row("Output MP4 (single video)", mode="file", file_filter="MP4 (*.mp4);;All files (*.*)", save=True)
        self.output_video_dir = self._path_row("Output MP4 directory", mode="dir")
        self.device = self._line_row("Device", "cuda")
        self.yolo_batch = self._spin_row("YOLO batch", 64, 1)
        self.fps = self._double_row("Fallback FPS", 30.0, 0.1, 1000.0, 3)
        self.num_frames = self._spin_row("Window frames (0 = model config)", 0, 0)
        self.window_stride = self._spin_row("Window stride (0 = model config)", 0, 0)
        self.temporal_smoothing_window = self._spin_row("Median smoothing window", 0, 0)
        self.bout_min_duration = self._spin_row("Bout min duration frames", 0, 0)
        self.csv_flush_interval = self._spin_row("CSV flush interval", 1, 1)
        self.metrics_interval = self._spin_row("Metrics interval windows", 30, 1)
        self.realtime = self._checkbox_row("Real-time source", False)
        self.realtime.setToolTip(
            "Passes --realtime to the inference backend. Use this for cameras, live streams, "
            "or any run where bounded latency matters more than offline throughput."
        )
        if batch_mode:
            self.realtime.setChecked(False)
            self.realtime.setEnabled(False)
        self.smooth_bouts = self._checkbox_row("Smooth bouts for MP4/frame CSV", True)
        self.smooth_bouts.setToolTip(
            "Improves annotated MP4/frame-level display by averaging overlapping windows. "
            "Disable for the lowest-latency live preview path."
        )
        self.live_preview = self._checkbox_row("Live preview", False)
        self.live_preview.setToolTip(
            "Show a live annotated playback window during inference. Press q or Esc "
            "in the preview window to request a clean stop. For lowest latency, also "
            "disable bout smoothing."
        )
        if batch_mode:
            self.live_preview.setChecked(False)
            self.live_preview.setEnabled(False)
        self.log_metrics = self._checkbox_row("Log runtime metrics", True)
        self.log_metrics.setToolTip(
            "Writes a sidecar metrics CSV and updates the live telemetry strip while inference runs."
        )
        self.amp = self._checkbox_row("Inference AMP", True)
        self.export_pose = self._checkbox_row("Export pose CSV (keypoints + bbox per frame)", False)
        self.no_bboxes = self._checkbox_row("Hide boxes in review MP4", False)
        self.no_keypoints = self._checkbox_row("Hide keypoints in review MP4", False)
        runner = ProcessRunner(
            command_builder=self.build_command,
            graceful_stop=self.request_stop,
            show_telemetry=True,
        )
        title = "Batch process videos with TandemYTC" if batch_mode else "Run TandemYTC inference"
        self._wrap(title, runner)

    def build_command(self) -> list[str]:
        script = _script_path("infer_y.py")
        if not script.exists():
            raise ValueError(f"Missing script: {script}")
        cmd = [
            *_python_invocation(),
            str(script),
            "--model_path",
            self._require(self.model_path, "Model .pt"),
            "--yolo_weights",
            self._require(self.yolo_weights, "YOLO pose weights"),
            "--source",
            self._require(self.source, "Source"),
            "--device",
            self.device.text().strip() or "cuda",
            "--yolo_batch",
            str(self.yolo_batch.value()),
            "--fps",
            str(self.fps.value()),
            "--num_frames",
            str(self.num_frames.value()),
            "--window_stride",
            str(self.window_stride.value()),
            "--temporal_smoothing_window",
            str(self.temporal_smoothing_window.value()),
            "--bout_min_duration_frames",
            str(self.bout_min_duration.value()),
            "--csv_flush_interval",
            str(self.csv_flush_interval.value()),
            "--metrics_log_interval",
            str(self.metrics_interval.value()),
        ]
        model_config = self._optional_path(self.model_config)
        if model_config:
            cmd.extend(["--model_config", model_config])
        output = self._optional_path(self.output)
        if output and not self.batch_mode:
            cmd.extend(["--output", output])
        output_dir = self._optional_path(self.output_dir)
        if output_dir:
            cmd.extend(["--output_dir", output_dir])
        output_video = self._optional_path(self.output_video)
        if output_video and not self.batch_mode:
            cmd.extend(["--output_video", output_video])
        output_video_dir = self._optional_path(self.output_video_dir)
        if output_video_dir:
            cmd.extend(["--output_video_dir", output_video_dir])
        if self.realtime.isChecked() and not self.batch_mode:
            cmd.append("--realtime")
        if self.smooth_bouts.isChecked():
            cmd.append("--smooth_bouts")
        if self.live_preview.isChecked() and not self.batch_mode:
            cmd.append("--preview")
        if self.log_metrics.isChecked():
            cmd.append("--log_metrics")
        if self.amp.isChecked():
            cmd.append("--amp")
        if self.export_pose.isChecked():
            cmd.append("--export_pose")
        if self.no_bboxes.isChecked():
            cmd.append("--no_bboxes")
        if self.no_keypoints.isChecked():
            cmd.append("--no_keypoints")
        stop_flag = self._stop_flag_path()
        cmd.extend(["--stop_flag_file", str(stop_flag)])
        cmd.extend(self._extra())
        return cmd

    def _stop_flag_path(self) -> Path:
        base = Path(self._optional_path(self.output_dir) or self._optional_path(self.output) or REPO_ROOT / ".yolo_temporal_classifier")
        if base.suffix:
            base = base.parent
        base.mkdir(parents=True, exist_ok=True)
        return base / "stop_inference.flag"

    def request_stop(self) -> str:
        flag = self._stop_flag_path()
        flag.write_text("stop inference\n", encoding="utf-8")
        return f"requested graceful inference stop via {flag}"
