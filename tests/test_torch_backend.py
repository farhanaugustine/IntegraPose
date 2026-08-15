from __future__ import annotations

from types import SimpleNamespace

from integra_pose.utils.torch_backend import (
    detect_torch_backend,
    enable_cudnn_benchmark_if_available,
    manual_seed_cuda_api_if_available,
    normalize_torch_device,
    normalize_ultralytics_device,
    resolve_amp_dtype,
)


class _FakeCuda:
    def __init__(
        self,
        *,
        available: bool = True,
        names: list[str] | None = None,
        bf16: bool = False,
    ) -> None:
        self._available = available
        self._names = names or []
        self._bf16 = bf16
        self.seeded: int | None = None

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return len(self._names)

    def get_device_name(self, index: int) -> str:
        return self._names[index]

    def is_bf16_supported(self) -> bool:
        return self._bf16

    def manual_seed_all(self, seed: int) -> None:
        self.seeded = seed


class _FakeMps:
    def __init__(self, available: bool = False) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


def _fake_torch(
    *,
    cuda_available: bool = True,
    names: list[str] | None = None,
    cuda: str = "12.1",
    hip: str = "",
    bf16: bool = False,
    mps_available: bool = False,
):
    backends = SimpleNamespace(
        cudnn=SimpleNamespace(benchmark=False),
        mps=_FakeMps(mps_available),
    )
    return SimpleNamespace(
        __version__="2.test",
        version=SimpleNamespace(cuda=cuda, hip=hip),
        cuda=_FakeCuda(available=cuda_available, names=names or ["Test GPU"], bf16=bf16),
        backends=backends,
    )


def test_detects_nvidia_cuda_backend() -> None:
    info = detect_torch_backend("auto", torch_module=_fake_torch(names=["RTX Test"]))

    assert info.backend == "cuda"
    assert info.device == "cuda:0"
    assert info.device_name == "RTX Test"
    assert info.is_cuda is True
    assert info.supports_cuda_api is True
    assert info.autocast_device_type == "cuda"


def test_detects_rocm_backend_through_cuda_api() -> None:
    info = detect_torch_backend(
        "rocm",
        torch_module=_fake_torch(cuda="", hip="6.3", names=["Radeon Test"]),
    )

    assert info.backend == "rocm"
    assert info.device == "cuda:0"
    assert info.device_name == "Radeon Test"
    assert info.is_rocm is True
    assert info.supports_cuda_api is True
    assert info.rocm_version == "6.3"


def test_falls_back_to_cpu_when_gpu_unavailable() -> None:
    fake = _fake_torch(cuda_available=False, names=[])

    info = detect_torch_backend("cuda:0", torch_module=fake)

    assert info.backend == "cpu"
    assert info.device == "cpu"
    assert normalize_torch_device("auto", torch_module=fake) == "cpu"
    assert normalize_ultralytics_device("0", torch_module=fake) == "cpu"


def test_blank_device_preserves_existing_cpu_default() -> None:
    fake = _fake_torch(names=["GPU 0"])

    assert normalize_torch_device("", torch_module=fake) == "cpu"
    assert normalize_ultralytics_device(None, torch_module=fake) == "cpu"


def test_preserves_ultralytics_auto_when_gpu_available() -> None:
    fake = _fake_torch(names=["GPU 0", "GPU 1"])

    assert normalize_ultralytics_device("-1", torch_module=fake) == "-1"
    assert normalize_ultralytics_device("auto", torch_module=fake) == "-1"
    assert normalize_torch_device("-1", torch_module=fake) == "cuda:0"


def test_ultralytics_auto_selects_mps_when_cuda_is_unavailable() -> None:
    fake = _fake_torch(cuda_available=False, names=[], cuda="", mps_available=True)

    assert normalize_ultralytics_device("-1", torch_module=fake) == "mps"
    assert normalize_ultralytics_device("auto", torch_module=fake) == "mps"
    assert detect_torch_backend("auto", torch_module=fake).backend == "mps"


def test_rocm_request_on_cuda_build_falls_back_to_cpu() -> None:
    info = detect_torch_backend("rocm", torch_module=_fake_torch(hip="", cuda="12.1"))

    assert info.backend == "cpu"
    assert info.device == "cpu"
    assert "not ROCm-enabled" in info.reason


def test_invalid_gpu_index_falls_back_to_cpu() -> None:
    info = detect_torch_backend("cuda:3", torch_module=_fake_torch(names=["GPU 0"]))

    assert info.backend == "cpu"
    assert info.device == "cpu"
    assert "outside available device count" in info.reason


def test_amp_dtype_and_cuda_api_helpers() -> None:
    fake = _fake_torch(names=["GPU 0"], bf16=True)
    info = detect_torch_backend("auto", torch_module=fake)

    assert resolve_amp_dtype("auto", torch_module=fake) == "bf16"
    assert resolve_amp_dtype("fp16", torch_module=fake) == "fp16"
    assert enable_cudnn_benchmark_if_available(info, torch_module=fake) is True
    assert fake.backends.cudnn.benchmark is True
    assert manual_seed_cuda_api_if_available(123, torch_module=fake) is True
    assert fake.cuda.seeded == 123
