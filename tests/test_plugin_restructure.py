import importlib
import os
import sys
import tempfile
import types

import pytest
import tkinter as tk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from integra_pose.utils import command_builder
from integra_pose.utils.config_manager import ConfigManager


@pytest.mark.parametrize(
    "plugin_module, stub_module, module_attr, stub_attr, attr_kind",
    [
        (
            "integra_pose.plugins.plugin_gait_kinematics.plugin",
            "integra_pose.plugins.plugin_gait_kinematics.gait_kinematics",
            "gait_kinematics",
            "launch",
            "module",
        ),
    ],
)
def test_plugin_imports_use_local_packages(
    plugin_module: str,
    stub_module: str,
    module_attr: str,
    stub_attr: str,
    attr_kind: str,
) -> None:
    """Ensure each plugin pulls its tooling from the dedicated package."""
    stub = types.ModuleType(stub_module)
    setattr(stub, stub_attr, object())

    previous_stub = sys.modules.get(stub_module)
    sys.modules[stub_module] = stub

    previous_plugin = sys.modules.pop(plugin_module, None)

    try:
        module = importlib.import_module(plugin_module)
        if attr_kind == "module":
            assert getattr(module, module_attr) is stub
        else:
            assert getattr(module, module_attr) is getattr(stub, stub_attr)
    finally:
        sys.modules.pop(plugin_module, None)
        if previous_plugin is not None:
            sys.modules[plugin_module] = previous_plugin

        if previous_stub is not None:
            sys.modules[stub_module] = previous_stub
        else:
            sys.modules.pop(stub_module, None)


def _make_app():
    try:
        root = tk.Tk(useTk=False)
        root.withdraw()
    except tk.TclError:
        pytest.skip("Tk is unavailable in this headless environment")

    class DummyApp:
        pass

    app = DummyApp()
    app.root = root
    app.config = ConfigManager(app)
    return app, root


def test_build_export_command_basic(tmp_path):
    app, root = _make_app()
    try:
        weights_path = tmp_path / "best.pt"
        weights_path.write_text("dummy")

        cfg = app.config.training
        cfg.export_model_path.set(str(weights_path))
        cfg.export_format_var.set("onnx")
        cfg.export_half_var.set(True)
        cfg.export_device_var.set("cuda:0")

        cmd, err = command_builder.build_export_command(app)
        assert err is None
        assert cmd[:2] == ["yolo", "export"]
        assert f"model={weights_path}" in cmd
        assert "format=onnx" in cmd
        assert "half=True" in cmd
        assert "device=cuda:0" in cmd
    finally:
        root.destroy()


def test_build_export_command_int8_requires_supported_format():
    app, root = _make_app()
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp.write(b"stub")
            model_path = tmp.name

        cfg = app.config.training
        cfg.export_model_path.set(model_path)
        cfg.export_format_var.set("onnx")
        cfg.export_int8_var.set(True)

        cmd, err = command_builder.build_export_command(app)
        assert cmd is None
        assert "INT8 quantization" in err
    finally:
        root.destroy()
        os.remove(model_path)


def test_build_inference_command_accepts_engine(tmp_path):
    app, root = _make_app()
    try:
        engine_path = tmp_path / "model.engine"
        engine_path.write_bytes(b"engine")
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"video")

        cfg = app.config.inference
        cfg.trained_model_path_infer.set(str(engine_path))
        cfg.video_infer_path.set(str(video_path))

        cmd, err = command_builder.build_inference_command(app)
        assert err is None
        assert f"model={engine_path}" in cmd
        assert f"source={video_path}" in cmd
    finally:
        root.destroy()


def test_build_inference_command_accepts_directory(tmp_path):
    app, root = _make_app()
    try:
        export_dir = tmp_path / "ncnn_model"
        export_dir.mkdir()
        (export_dir / "model.ncnn.param").write_text("param")
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"video")

        cfg = app.config.inference
        cfg.trained_model_path_infer.set(str(export_dir))
        cfg.video_infer_path.set(str(video_path))

        cmd, err = command_builder.build_inference_command(app)
        assert err is None
        assert f"model={export_dir}" in cmd
    finally:
        root.destroy()


def test_build_webcam_command_accepts_onnx(tmp_path):
    app, root = _make_app()
    try:
        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"onnx")

        cfg = app.config.webcam
        cfg.webcam_model_path.set(str(onnx_path))

        cmd, err = command_builder.build_webcam_command(app)
        assert err is None
        assert f"model={onnx_path}" in cmd
    finally:
        root.destroy()
