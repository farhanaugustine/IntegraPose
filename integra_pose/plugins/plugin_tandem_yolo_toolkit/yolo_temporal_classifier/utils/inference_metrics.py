"""
Runtime-metrics logger for TandemYTC inference.

Writes a sidecar CSV alongside the predictions CSV with per-interval
samples of throughput, per-stage timing, CPU memory, and (when NVIDIA
hardware + pynvml are available) GPU utilization, memory and temperature.

Design goals:
  * Default-off at the CLI level — preserves bit-identical behavior
    of any pipeline calling `infer_n.run_inference_on_source()`.
    The GUI flips it on by default.
  * Graceful degradation: missing psutil or pynvml just blanks those
    columns; the CSV still writes.
  * Cheap: per-stage measurements use perf_counter context managers;
    system metrics are sampled only at flush boundaries (default
    every 30 windows ~= 1 s of source video at 30 fps stride 16).
  * Self-describing CSV: header line + one row per flush. Pandas/Excel
    friendly.

Usage (inside infer_n.py):

    from utils.inference_metrics import MetricsLogger, NullMetricsLogger

    metrics = MetricsLogger(csv_path, log_interval_windows=30,
                            source=src_repr) if args.log_metrics \
              else NullMetricsLogger()
    metrics.start()
    try:
        for det in _detection_stream(...):
            metrics.frame_seen()
            ...
            with metrics.stage("classify"):
                logits = model(batch)
            ...
            metrics.window_done(window_idx)  # may flush a row
    finally:
        metrics.close()
"""
from __future__ import annotations

import csv
import os
import time
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---- optional deps -------------------------------------------------------

try:
    import psutil
    _PSUTIL_OK = True
except ImportError:
    psutil = None
    _PSUTIL_OK = False

# pynvml emits a FutureWarning at import time on recent versions because
# the package was renamed to nvidia-ml-py; the import name is unchanged
# and the API still works. Suppress the warning so it doesn't spam every
# inference run.
try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        import pynvml  # type: ignore
    _PYNVML_OK = True
except Exception:
    pynvml = None  # type: ignore
    _PYNVML_OK = False

try:
    import torch  # type: ignore
    _TORCH_OK = True
except ImportError:
    torch = None  # type: ignore
    _TORCH_OK = False


# ---- main class ---------------------------------------------------------

CSV_FIELDS = [
    "wallclock_iso",
    "elapsed_s",
    "windows_done",
    "frames_done",
    "instant_fps",
    "instant_window_throughput",
    "t_total_per_window_ms",
    "t_classify_per_window_ms",
    "t_classify_share_pct",
    "cpu_percent_normalized",   # 0..100 across all cores (matches Task Manager)
    "cpu_percent_singlecore",   # raw psutil reading (>100 means multi-core)
    "cpu_rss_mb",
    "gpu_util_percent_avg",     # NVML utilization, averaged over the interval
    "gpu_util_percent_max",     # peak NVML utilization seen in the interval
    "gpu_mem_allocated_mb",     # torch's live tensor allocations
    "gpu_mem_reserved_mb",      # torch's caching allocator pool
    "gpu_mem_used_mb",          # NVML system-wide GPU memory in use (matches Task Manager)
    "gpu_temp_c",
    "source",
]


class MetricsLogger:
    """Live runtime-metrics logger; writes one CSV row per N windows."""

    def __init__(
        self,
        csv_path: Path | str,
        *,
        log_interval_windows: int = 30,
        source: str = "",
        echo_to_stdout: bool = True,
    ):
        self.csv_path = Path(csv_path)
        self.log_interval = max(1, int(log_interval_windows))
        self.source = source
        self.echo = bool(echo_to_stdout)

        self._t0: float = 0.0
        self._t_last_flush: float = 0.0

        self._frames_total = 0
        self._windows_total = 0
        self._frames_at_last_flush = 0
        self._windows_at_last_flush = 0

        # Per-stage time accumulators (in seconds), reset at each flush.
        self._stage_acc: dict[str, float] = {"classify": 0.0}
        # Stage-counters tell us how many times each stage was entered.
        self._stage_n: dict[str, int] = {"classify": 0}
        # Active stage timers (for context manager).
        self._stage_active: dict[str, float] = {}

        # GPU-utilization sampling buffer (avg + max over the interval).
        # NVML's GetUtilizationRates returns an instantaneous reading; for
        # CPU-bound workloads with episodic GPU bursts a single end-of-
        # interval sample can systematically miss the peaks. We sample on
        # every window_done() and report avg + max at flush time.
        self._gpu_util_samples: list[int] = []

        # NVML handle (lazy init in start()).
        self._nvml_handle = None

        # Cache CPU count for normalised CPU% (matches Task Manager's
        # 0..100% scale rather than psutil's per-core reading).
        self._cpu_count = psutil.cpu_count(logical=True) if _PSUTIL_OK else None
        if not self._cpu_count or self._cpu_count <= 0:
            self._cpu_count = 1

        self._csv_f = None
        self._writer: Optional[csv.DictWriter] = None
        # Pin a process handle so cpu_percent() returns sane values.
        self._proc = psutil.Process(os.getpid()) if _PSUTIL_OK else None
        if self._proc is not None:
            # First call seeds the internal counter; subsequent calls return
            # CPU% over the elapsed window.
            try:
                self._proc.cpu_percent(interval=None)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_f = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_f, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._t0 = time.perf_counter()
        self._t_last_flush = self._t0

        if _PYNVML_OK:
            try:
                pynvml.nvmlInit()
                # Match torch's active CUDA device so we don't sample the
                # idle integrated GPU when the dGPU is doing the work.
                gpu_idx = 0
                if _TORCH_OK and torch.cuda.is_available():
                    try:
                        gpu_idx = int(torch.cuda.current_device())
                    except Exception:
                        gpu_idx = 0
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
            except Exception:
                self._nvml_handle = None

        if self.echo:
            print(f"[metrics] logging to {self.csv_path}", flush=True)

    def close(self) -> None:
        # Final flush — write a partial row so the user always sees the tail.
        if self._writer is not None and self._windows_total > self._windows_at_last_flush:
            self._flush_row(force=True)
        if self._csv_f is not None:
            try:
                self._csv_f.close()
            except Exception:
                pass
            self._csv_f = None
            self._writer = None

        if self._nvml_handle is not None and _PYNVML_OK:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_handle = None

    # ------------------------------------------------------------------
    # Counters & per-stage timers
    # ------------------------------------------------------------------

    def frame_seen(self) -> None:
        """Call once per source frame consumed by inference."""
        self._frames_total += 1

    def window_done(self, window_idx: int = 0) -> None:
        """Call after writing a window prediction. Maybe triggers a flush."""
        self._windows_total += 1
        # Sample GPU utilization at every window so peaks aren't missed when
        # the workload is bursty (compare to a once-per-flush sample, which
        # would systematically catch CPU-bound idle gaps).
        self._sample_gpu_util()
        if (self._windows_total - self._windows_at_last_flush) >= self.log_interval:
            self._flush_row(force=False)

    def _sample_gpu_util(self) -> None:
        if self._nvml_handle is None or not _PYNVML_OK:
            return
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
            self._gpu_util_samples.append(int(util.gpu))
        except Exception:
            pass

    @contextmanager
    def stage(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t
            self._stage_acc[name] = self._stage_acc.get(name, 0.0) + dt
            self._stage_n[name] = self._stage_n.get(name, 0) + 1

    # ------------------------------------------------------------------
    # Flush a CSV row + optional stdout summary line
    # ------------------------------------------------------------------

    def _sample_system(self) -> dict:
        out = {
            "cpu_percent_normalized": "",
            "cpu_percent_singlecore": "",
            "cpu_rss_mb": "",
            "gpu_util_percent_avg": "",
            "gpu_util_percent_max": "",
            "gpu_mem_allocated_mb": "",
            "gpu_mem_reserved_mb": "",
            "gpu_mem_used_mb": "",
            "gpu_temp_c": "",
        }
        if self._proc is not None:
            try:
                # psutil reports per-core (>100% means multi-core); also
                # report a normalised value matching Task Manager's scale.
                cpu_singlecore = self._proc.cpu_percent(interval=None)
                out["cpu_percent_singlecore"] = round(cpu_singlecore, 1)
                out["cpu_percent_normalized"] = round(cpu_singlecore / self._cpu_count, 1)
                out["cpu_rss_mb"] = round(self._proc.memory_info().rss / (1024 * 1024), 1)
            except Exception:
                pass
        if _TORCH_OK and torch.cuda.is_available():
            try:
                out["gpu_mem_allocated_mb"] = round(
                    torch.cuda.memory_allocated() / (1024 * 1024), 1)
                out["gpu_mem_reserved_mb"] = round(
                    torch.cuda.memory_reserved() / (1024 * 1024), 1)
            except Exception:
                pass
        if self._nvml_handle is not None and _PYNVML_OK:
            try:
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                temp = pynvml.nvmlDeviceGetTemperature(
                    self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
                out["gpu_mem_used_mb"] = round(meminfo.used / (1024 * 1024), 1)
                out["gpu_temp_c"] = int(temp)
                # Avg + max over the interval's samples (collected on every
                # window_done() call) instead of a single instantaneous read.
                if self._gpu_util_samples:
                    out["gpu_util_percent_avg"] = round(
                        sum(self._gpu_util_samples) / len(self._gpu_util_samples), 1)
                    out["gpu_util_percent_max"] = max(self._gpu_util_samples)
                else:
                    # Fallback: sample once now if no per-window samples exist.
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                    out["gpu_util_percent_avg"] = int(util.gpu)
                    out["gpu_util_percent_max"] = int(util.gpu)
            except Exception:
                pass
        return out

    def _flush_row(self, *, force: bool) -> None:
        if self._writer is None:
            return
        now = time.perf_counter()
        elapsed = now - self._t0
        dt_interval = now - self._t_last_flush
        d_frames = self._frames_total - self._frames_at_last_flush
        d_windows = self._windows_total - self._windows_at_last_flush

        # Throughput numbers in this interval.
        instant_fps = d_frames / dt_interval if dt_interval > 1e-9 else 0.0
        instant_wps = d_windows / dt_interval if dt_interval > 1e-9 else 0.0

        # Per-window total wallclock (in this interval).
        t_total_per_window_ms = (
            (dt_interval / d_windows) * 1000.0 if d_windows > 0 else 0.0
        )
        # Per-window classifier-only time (averaged over windows seen this interval).
        # We can't subtract per-interval because stage_acc is cumulative; use the
        # stage counter to derive average.
        n_classify = self._stage_n.get("classify", 0)
        t_classify_avg_ms = (
            (self._stage_acc.get("classify", 0.0) / n_classify) * 1000.0
            if n_classify > 0 else 0.0
        )
        share_pct = (
            (t_classify_avg_ms / t_total_per_window_ms) * 100.0
            if t_total_per_window_ms > 1e-9 else 0.0
        )

        sys_row = self._sample_system()
        row = {
            "wallclock_iso": datetime.now().isoformat(timespec="seconds"),
            "elapsed_s": round(elapsed, 3),
            "windows_done": self._windows_total,
            "frames_done": self._frames_total,
            "instant_fps": round(instant_fps, 2),
            "instant_window_throughput": round(instant_wps, 3),
            "t_total_per_window_ms": round(t_total_per_window_ms, 2),
            "t_classify_per_window_ms": round(t_classify_avg_ms, 2),
            "t_classify_share_pct": round(share_pct, 1),
            "source": self.source,
            **sys_row,
        }
        self._writer.writerow(row)
        try:
            self._csv_f.flush()
        except Exception:
            pass

        if self.echo:
            self._echo_row(row)

        # Reset interval accumulators (stage timers reset; counts already reset
        # via *_at_last_flush trackers).
        self._frames_at_last_flush = self._frames_total
        self._windows_at_last_flush = self._windows_total
        self._t_last_flush = now
        # Stage averages reset per interval — otherwise late-run averages
        # become indistinguishable from cumulative.
        for k in list(self._stage_acc.keys()):
            self._stage_acc[k] = 0.0
            self._stage_n[k] = 0
        # GPU util samples reset per interval — same reasoning.
        self._gpu_util_samples.clear()

    def _echo_row(self, row: dict) -> None:
        bits = [
            f"win={row['windows_done']}",
            f"fps={row['instant_fps']:.1f}",
            f"wps={row['instant_window_throughput']:.2f}",
            f"t_total={row['t_total_per_window_ms']:.1f}ms",
            f"t_cls={row['t_classify_per_window_ms']:.1f}ms",
        ]
        # Prefer the normalised CPU% (0-100, matches Task Manager) for the
        # echo line; the per-core value lives in the CSV for power users.
        if row["cpu_percent_normalized"] != "":
            bits.append(f"cpu={row['cpu_percent_normalized']:.0f}%")
        # GPU util: show "avg / peak" so episodic bursts are visible.
        if row["gpu_util_percent_avg"] != "":
            avg = row["gpu_util_percent_avg"]
            peak = row.get("gpu_util_percent_max", "")
            if peak != "" and peak != avg:
                bits.append(f"gpu_util={avg:.0f}%/{peak}% (avg/peak)")
            else:
                bits.append(f"gpu_util={avg:.0f}%")
        # GPU memory: prefer NVML's system-wide reading (matches Task
        # Manager) over torch's tensor-only reading; fall back to torch
        # if NVML isn't available.
        gpu_mem_used = row.get("gpu_mem_used_mb", "")
        gpu_mem_alloc = row.get("gpu_mem_allocated_mb", "")
        if gpu_mem_used != "":
            bits.append(f"gpu_mem={gpu_mem_used:.0f}MB")
        elif gpu_mem_alloc != "":
            bits.append(f"gpu_mem={gpu_mem_alloc:.0f}MB(torch)")
        print("[metrics] " + "  ".join(bits), flush=True)


class NullMetricsLogger:
    """No-op logger used when --log_metrics is off. Same surface so the
    inference loop doesn't need conditional branches everywhere."""

    def start(self) -> None: pass
    def close(self) -> None: pass
    def frame_seen(self) -> None: pass
    def window_done(self, window_idx: int = 0) -> None: pass

    @contextmanager
    def stage(self, name: str):
        yield


def derive_metrics_csv_path(predictions_csv: Path | str) -> Path:
    """Default sidecar path: <basename>.metrics.csv next to the predictions CSV."""
    p = Path(predictions_csv)
    return p.with_name(p.stem + ".metrics.csv")
