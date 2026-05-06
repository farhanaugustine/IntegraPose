from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module(rel_path: str, module_name: str):
    repo_root = Path(__file__).resolve().parent.parent
    module_path = repo_root / rel_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_c10_dll_error_message_points_to_env_mismatch_risk() -> None:
    module = _load_module(
        "integra_pose/plugins/plugin_autolabel_forge/autolabel_runtime.py",
        "test_autolabel_forge_runtime",
    )
    exc = OSError(
        '[WinError 1114] A dynamic link library (DLL) initialization routine failed. '
        'Error loading "C:\\path\\torch\\lib\\c10.dll" or one of its dependencies.'
    )

    message = module.format_grounding_dino_import_error(exc)

    assert "PyTorch failed to initialize" in message
    assert "Conda/venv" in message
    assert "PATH starts with that environment" in message
    assert "torch/torchvision pair" in message


def test_runtime_can_install_yapf_shim() -> None:
    module = _load_module(
        "integra_pose/plugins/plugin_autolabel_forge/autolabel_runtime.py",
        "test_autolabel_forge_runtime_shim",
    )
    for key in ("yapf", "yapf.yapflib", "yapf.yapflib.yapf_api"):
        sys.modules.pop(key, None)

    module.install_groundingdino_yapf_shim()

    assert "yapf.yapflib.yapf_api" in sys.modules
    formatted, changed = sys.modules["yapf.yapflib.yapf_api"].FormatCode("x=1")
    assert formatted == "x=1"
    assert changed is False


def test_ui_builds_subprocess_worker_command() -> None:
    _load_module(
        "integra_pose/plugins/plugin_autolabel_forge/autolabel_runtime.py",
        "integra_pose.plugins.plugin_autolabel_forge.autolabel_runtime",
    )
    module = _load_module(
        "integra_pose/plugins/plugin_autolabel_forge/ui.py",
        "test_autolabel_forge_ui",
    )
    fake_self = object.__new__(module.AutoLabelForgeWindow)
    command = module.AutoLabelForgeWindow._build_autolabel_subprocess_command(fake_self, Path("job.json"))

    assert command[1:4] == ["-u", "-m", "integra_pose.plugins.plugin_autolabel_forge.autolabel_worker"]
    assert command[-2:] == ["--job", "job.json"]


def test_ui_builds_subprocess_env_with_env_prefix_first() -> None:
    _load_module(
        "integra_pose/plugins/plugin_autolabel_forge/autolabel_runtime.py",
        "integra_pose.plugins.plugin_autolabel_forge.autolabel_runtime",
    )
    module = _load_module(
        "integra_pose/plugins/plugin_autolabel_forge/ui.py",
        "test_autolabel_forge_ui_env",
    )
    fake_self = object.__new__(module.AutoLabelForgeWindow)
    old_exec = module.sys.executable
    old_env = dict(module.os.environ)
    try:
        module.sys.executable = r"C:\Env\python.exe"
        module.os.environ.clear()
        module.os.environ.update({"PATH": r"C:\Other;C:\Else"})
        env = module.AutoLabelForgeWindow._build_autolabel_subprocess_env(fake_self)
    finally:
        module.sys.executable = old_exec
        module.os.environ.clear()
        module.os.environ.update(old_env)

    parts = env["PATH"].split(module.os.pathsep)
    assert parts[:6] == [
        r"C:\Env",
        r"C:\Env\Library\mingw-w64\bin",
        r"C:\Env\Library\usr\bin",
        r"C:\Env\Library\bin",
        r"C:\Env\Scripts",
        r"C:\Env\bin",
    ]
