import contextlib
import importlib
import sys
import types
from pathlib import Path

import pytest

from integra_pose.utils import plugin_security


def _noop(*_args, **_kwargs) -> None:
    return None


_PLUGIN_IMPORT_STUBS = {
    "integra_pose.plugins.plugin_gait_kinematics.plugin": (
        "integra_pose.plugins.plugin_gait_kinematics.gait_kinematics",
        {"launch": _noop},
    ),
}


@contextlib.contextmanager
def _isolated_plugin_import(module_path: str):
    stub_module_name = None
    previous_stub = None

    stub_spec = _PLUGIN_IMPORT_STUBS.get(module_path)
    if stub_spec is not None:
        stub_module_name, attrs = stub_spec
        stub = types.ModuleType(stub_module_name)
        for attr_name, attr_value in attrs.items():
            setattr(stub, attr_name, attr_value)
        previous_stub = sys.modules.get(stub_module_name)
        sys.modules[stub_module_name] = stub

    previous_plugin = sys.modules.pop(module_path, None)

    try:
        yield importlib.import_module(module_path)
    finally:
        sys.modules.pop(module_path, None)
        if previous_plugin is not None:
            sys.modules[module_path] = previous_plugin
        if stub_module_name is not None:
            if previous_stub is not None:
                sys.modules[stub_module_name] = previous_stub
            else:
                sys.modules.pop(stub_module_name, None)


class FakeMenu:
    def __init__(self) -> None:
        self._entries: list[dict[str, object]] = []

    def add_command(self, *, label: str, command=None) -> None:  # pragma: no cover - simple storage
        self._entries.append({"label": label, "command": command})

    def index(self, key):
        if key == "end":
            return len(self._entries) - 1 if self._entries else None
        return key if 0 <= int(key) < len(self._entries) else None

    def entrycget(self, idx: int, option: str) -> str:
        if option != "label":
            raise KeyError(f"FakeMenu only supports the 'label' option, received: {option}")
        return self._entries[idx]["label"]


@pytest.mark.parametrize(
    "plugin_name",
    [
        "plugin_eda",
        "plugin_dataset_augmentor_lab",
        "plugin_gait_kinematics",
        "plugin_tandem_yolo_toolkit",
        "plugin_zone_counter",
    ],
)
def test_bundled_plugins_are_auto_trusted(plugin_name: str) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    plugin_dir = repo_root / "integra_pose" / "plugins" / plugin_name
    decision = plugin_security.ensure_plugin_trusted(plugin_dir, prompt=None)
    assert decision.allowed, f"{plugin_name} should be trusted by default"
    assert "Trusted bundled plugin" in decision.message


@pytest.mark.parametrize(
    ("module_path", "expected_label"),
    [
        ("integra_pose.plugins.plugin_eda.plugin", "Launch EDA Tool"),
        ("integra_pose.plugins.plugin_dataset_augmentor_lab.plugin", "Dataset Augmentor Lab"),
        ("integra_pose.plugins.plugin_gait_kinematics.plugin", "Gait & Kinematic Dashboard"),
        ("integra_pose.plugins.plugin_tandem_yolo_toolkit.plugin", "TandemYTC - Tandem YOLO + Temporal Classifier"),
        ("integra_pose.plugins.plugin_zone_counter.plugin", "Zone Counter"),
    ],
)
def test_register_plugin_adds_menu_entry(module_path: str, expected_label: str) -> None:
    with _isolated_plugin_import(module_path) as module:
        menu = FakeMenu()

        class DummyApp:
            def __init__(self, tk_menu: FakeMenu) -> None:
                self.plugins_menu = tk_menu
                self.logged = []

            def log_message(self, message: str, level: str = "INFO") -> None:  # pragma: no cover - simple sink
                self.logged.append((level, message))

        app = DummyApp(menu)

        # Ensure heavy optional dependencies are not required during registration.
        if module_path.endswith("plugin_eda.plugin"):
            original_resolver = module._resolve_eda_gui
            module._eda_gui_module = types.SimpleNamespace(PoseEDAApp=object())
            module._eda_import_error = None

            def _patched_resolver():
                return module._eda_gui_module

            module._resolve_eda_gui = _patched_resolver
        else:
            original_resolver = None

        try:
            module.register_plugin(app)

            end_index = menu.index("end")
            labels = []
            if end_index is not None:
                for i in range(end_index + 1):
                    labels.append(menu.entrycget(i, "label"))
            assert expected_label in labels
        finally:
            if original_resolver is not None:
                module._resolve_eda_gui = original_resolver
                module._eda_gui_module = None
