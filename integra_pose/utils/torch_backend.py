"""PyTorch backend detection and device normalization helpers.

ROCm PyTorch exposes AMD GPUs through the same ``torch.cuda`` API used by
NVIDIA CUDA builds. This module keeps that API detail in one place so callers
can ask for a usable device while still reporting whether the active stack is
CPU, NVIDIA CUDA, or AMD ROCm.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


@dataclass(frozen=True)
class TorchBackendInfo:
    """Resolved PyTorch backend and device details."""

    requested: str
    device: str
    backend: str
    available: bool
    device_count: int = 0
    device_index: int | None = None
    device_name: str = ""
    torch_version: str = ""
    cuda_version: str = ""
    rocm_version: str = ""
    reason: str = ""

    @property
    def is_gpu(self) -> bool:
        return self.backend in {"cuda", "rocm", "mps"} and self.available

    @property
    def is_cuda(self) -> bool:
        return self.backend == "cuda" and self.available

    @property
    def is_rocm(self) -> bool:
        return self.backend == "rocm" and self.available

    @property
    def supports_cuda_api(self) -> bool:
        return self.backend in {"cuda", "rocm"} and self.available

    @property
    def supports_pinned_memory(self) -> bool:
        return self.supports_cuda_api

    @property
    def supports_amp(self) -> bool:
        return self.supports_cuda_api

    @property
    def autocast_device_type(self) -> str:
        return "cuda" if self.supports_cuda_api else self.backend

    @property
    def non_blocking_transfer(self) -> bool:
        return self.supports_cuda_api

    @property
    def display_name(self) -> str:
        if self.device_name:
            return self.device_name
        if self.backend == "rocm":
            return "AMD ROCm GPU"
        if self.backend == "cuda":
            return "NVIDIA CUDA GPU"
        if self.backend == "mps":
            return "Apple MPS GPU"
        return "CPU"


def detect_torch_backend(
    preferred: str | int | None = "auto",
    *,
    torch_module: Any | None = None,
    preserve_ultralytics_auto: bool = False,
) -> TorchBackendInfo:
    """Resolve a requested PyTorch device to CUDA, ROCm, MPS, or CPU.

    ``device`` remains a PyTorch-compatible string. On ROCm builds that means
    ``cuda:0`` because PyTorch intentionally exposes HIP devices through the
    CUDA compatibility API.
    """

    requested = "" if preferred is None else str(preferred).strip()
    text = requested.lower()
    if text == "":
        torch = torch_module
        if torch is None:
            try:
                import torch as imported_torch  # noqa: PLC0415

                torch = imported_torch
            except Exception:
                torch = None
        torch_version = str(getattr(torch, "__version__", "")) if torch is not None else ""
        version = getattr(torch, "version", None) if torch is not None else None
        return _cpu_info(
            requested=requested,
            torch_version=torch_version,
            cuda_version=str(getattr(version, "cuda", "") or ""),
            rocm_version=str(getattr(version, "hip", "") or ""),
            reason="Blank device value defaults to CPU.",
        )
    if text in {"auto", "gpu"}:
        text = "auto"

    torch = torch_module
    if torch is None:
        try:
            import torch as imported_torch  # noqa: PLC0415

            torch = imported_torch
        except Exception as exc:
            return TorchBackendInfo(
                requested=requested,
                device="cpu",
                backend="cpu",
                available=True,
                reason=f"torch import failed: {exc}",
            )

    torch_version = str(getattr(torch, "__version__", ""))
    version = getattr(torch, "version", None)
    cuda_version = str(getattr(version, "cuda", "") or "")
    rocm_version = str(getattr(version, "hip", "") or "")
    cuda_attr = getattr(torch, "cuda", None)
    cuda_seen = bool(cuda_attr and _call_bool(cuda_attr, "is_available"))
    device_count = _call_int(cuda_attr, "device_count") if cuda_seen else 0
    cuda_api_usable = cuda_seen and device_count > 0
    cuda_like_backend = "rocm" if rocm_version else "cuda"
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(mps and _call_bool(mps, "is_available"))

    if cuda_seen and device_count == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    index = _requested_cuda_index(text)
    requested_cuda_like = index is not None or text in {"auto", "cuda", "rocm", "amd", "hip"}

    if requested_cuda_like:
        if text in {"rocm", "amd", "hip"} and not rocm_version:
            return _cpu_info(
                requested=requested,
                torch_version=torch_version,
                cuda_version=cuda_version,
                rocm_version=rocm_version,
                reason="ROCm/AMD requested but this PyTorch build is not ROCm-enabled.",
            )

        if cuda_api_usable:
            selected_index = 0 if index is None else index
            if 0 <= selected_index < device_count:
                device = (
                    "-1"
                    if preserve_ultralytics_auto and text in {"-1", "auto", "gpu"}
                    else f"cuda:{selected_index}"
                )
                return TorchBackendInfo(
                    requested=requested,
                    device=device,
                    backend=cuda_like_backend,
                    available=True,
                    device_count=device_count,
                    device_index=selected_index,
                    device_name=_device_name(cuda_attr, selected_index),
                    torch_version=torch_version,
                    cuda_version=cuda_version,
                    rocm_version=rocm_version,
                    reason="GPU backend available.",
                )

            return _cpu_info(
                requested=requested,
                torch_version=torch_version,
                cuda_version=cuda_version,
                rocm_version=rocm_version,
                reason=f"Requested GPU index {selected_index} is outside available device count {device_count}.",
            )

        if text in {"-1", "auto", "gpu"} and mps_available:
            return TorchBackendInfo(
                requested=requested,
                device="mps",
                backend="mps",
                available=True,
                torch_version=torch_version,
                cuda_version=cuda_version,
                rocm_version=rocm_version,
                reason="Apple MPS backend selected automatically.",
            )

        return _cpu_info(
            requested=requested,
            torch_version=torch_version,
            cuda_version=cuda_version,
            rocm_version=rocm_version,
            reason="No CUDA/ROCm GPU is available to PyTorch.",
        )

    if text == "mps":
        if mps_available:
            return TorchBackendInfo(
                requested=requested,
                device="mps",
                backend="mps",
                available=True,
                torch_version=torch_version,
                cuda_version=cuda_version,
                rocm_version=rocm_version,
                reason="MPS backend available.",
            )
        return _cpu_info(
            requested=requested,
            torch_version=torch_version,
            cuda_version=cuda_version,
            rocm_version=rocm_version,
            reason="MPS requested but not available.",
        )

    if text == "cpu":
        return _cpu_info(
            requested=requested,
            torch_version=torch_version,
            cuda_version=cuda_version,
            rocm_version=rocm_version,
            reason="CPU requested.",
        )

    return TorchBackendInfo(
        requested=requested,
        device=text or "cpu",
        backend="literal",
        available=False,
        torch_version=torch_version,
        cuda_version=cuda_version,
        rocm_version=rocm_version,
        reason="Unrecognized device string was preserved for the caller.",
    )


def normalize_torch_device(
    preferred: str | int | None = "auto",
    *,
    torch_module: Any | None = None,
) -> str:
    """Return a safe device string for direct PyTorch use."""

    info = detect_torch_backend(preferred, torch_module=torch_module)
    if info.backend == "literal":
        return "cpu"
    return info.device


def torch_supports_bf16(*, torch_module: Any | None = None) -> bool:
    """Return whether the active CUDA-compatible API supports bf16 AMP."""

    torch = _import_torch(torch_module)
    if torch is None:
        return False
    cuda_attr = getattr(torch, "cuda", None)
    if not cuda_attr or not _call_bool(cuda_attr, "is_available"):
        return False
    return _call_bool(cuda_attr, "is_bf16_supported")


def resolve_amp_dtype(requested: str, *, torch_module: Any | None = None) -> str:
    """Resolve an AMP dtype request to ``bf16`` or ``fp16``."""

    if requested != "auto":
        return requested
    return "bf16" if torch_supports_bf16(torch_module=torch_module) else "fp16"


def enable_cudnn_benchmark_if_available(
    backend: TorchBackendInfo,
    *,
    torch_module: Any | None = None,
) -> bool:
    """Enable cuDNN/MIOpen benchmark mode for CUDA-compatible GPU backends."""

    if not backend.supports_cuda_api:
        return False
    torch = _import_torch(torch_module)
    if torch is None:
        return False
    try:
        torch.backends.cudnn.benchmark = True
        return True
    except Exception:
        return False


def manual_seed_cuda_api_if_available(
    seed: int,
    *,
    torch_module: Any | None = None,
) -> bool:
    """Seed all CUDA-compatible devices, including ROCm HIP devices."""

    torch = _import_torch(torch_module)
    if torch is None:
        return False
    cuda_attr = getattr(torch, "cuda", None)
    if not cuda_attr or not _call_bool(cuda_attr, "is_available"):
        return False
    try:
        cuda_attr.manual_seed_all(int(seed))
        return True
    except Exception:
        return False


def normalize_ultralytics_device(
    preferred: str | int | None = "auto",
    *,
    torch_module: Any | None = None,
    preserve_auto: bool = True,
) -> str:
    """Return a safe device string for Ultralytics model loading/inference.

    Ultralytics accepts ``-1`` as an auto-idle-GPU sentinel. When requested and
    a GPU backend is available, preserve that sentinel so Ultralytics can make
    the final placement decision. If no GPU exists, fall back to CPU.
    """

    text = str(preferred or "").strip().lower()
    preserve = preserve_auto and text in {"-1", "auto", "gpu"}
    info = detect_torch_backend(
        preferred,
        torch_module=torch_module,
        preserve_ultralytics_auto=preserve,
    )
    if info.backend == "literal":
        return info.device
    return info.device


def _requested_cuda_index(text: str) -> int | None:
    if text in {"cuda", "gpu", "rocm", "amd", "hip", "-1"}:
        return 0
    if text.isdigit():
        return int(text)
    if text.startswith("cuda:"):
        tail = text.split(":", 1)[1].strip()
        if tail.isdigit():
            return int(tail)
    return None


def _cpu_info(
    *,
    requested: str,
    torch_version: str = "",
    cuda_version: str = "",
    rocm_version: str = "",
    reason: str = "",
) -> TorchBackendInfo:
    return TorchBackendInfo(
        requested=requested,
        device="cpu",
        backend="cpu",
        available=True,
        torch_version=torch_version,
        cuda_version=cuda_version,
        rocm_version=rocm_version,
        reason=reason,
    )


def _call_bool(obj: Any, name: str) -> bool:
    try:
        fn = getattr(obj, name)
        return bool(fn())
    except Exception:
        return False


def _call_int(obj: Any, name: str) -> int:
    try:
        fn = getattr(obj, name)
        return int(fn())
    except Exception:
        return 0


def _import_torch(torch_module: Any | None = None) -> Any | None:
    if torch_module is not None:
        return torch_module
    try:
        import torch  # noqa: PLC0415

        return torch
    except Exception:
        return None


def _device_name(cuda_attr: Any, index: int) -> str:
    try:
        return str(cuda_attr.get_device_name(index))
    except Exception:
        return ""
