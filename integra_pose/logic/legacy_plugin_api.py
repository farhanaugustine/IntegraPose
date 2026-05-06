import importlib.util
import inspect
import os
import sys
from pathlib import Path

class LegacyPluginManager:
    def __init__(self, app):
        self.app = app

    def load_plugins(self):
        plugins_dir = Path(__file__).parent.parent / "plugins"
        if not plugins_dir.is_dir():
            return

        for plugin_name in os.listdir(plugins_dir):
            plugin_dir = plugins_dir / plugin_name
            if not plugin_dir.is_dir():
                continue

            plugin_file = plugin_dir / "plugin.py"
            if not plugin_file.is_file():
                continue

            try:
                spec = importlib.util.spec_from_file_location(f"integra_pose.plugins.{plugin_name}", plugin_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[spec.name] = module
                    spec.loader.exec_module(module)

                    if hasattr(module, "register_plugin") and callable(getattr(module, "register_plugin")):
                        module.register_plugin(self.app)
                        self.app.log_message(f"Loaded legacy plugin: {plugin_name}", "INFO")
            except Exception as e:
                self.app.log_message(f"Failed to load legacy plugin {plugin_name}: {e}", "ERROR")
