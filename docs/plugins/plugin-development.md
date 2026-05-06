# Plugin Development

IntegraPose plugins are optional Python packages that add commands to the
main **Plugins** menu. They are meant for lab-specific tools, analysis
windows, import/export helpers, and workflow extensions that should not
become part of the core seven-tab interface.

!!! warning "Beta plugin API"
    The plugin loader is available for external use, but the public
    contract is still beta. Keep plugins small, pin IntegraPose to a
    release or commit for active projects, and avoid depending on private
    GUI internals unless you control both the plugin and the IntegraPose
    version.

## Discovery

IntegraPose looks for plugin directories in two places:

| Location | Purpose |
| --- | --- |
| `integra_pose/plugins/` | Plugins bundled with IntegraPose |
| `~/.integrapose/plugins/` | User-installed plugins |

Each immediate child directory is treated as a plugin candidate unless
the directory name starts with `.` or `_`.

Example user plugin layout:

```text
~/.integrapose/plugins/
  my_lab_plugin/
    plugin.json
    plugin.py
    README.md
```

Plugin directory names are also used as internal import identifiers. Use
simple ASCII names with letters, numbers, and underscores when possible:

```text
my_lab_plugin
behavior_qc_tool
roi_export_helper
```

## Required Files

Third-party plugins must include:

```text
plugin.json
plugin.py
```

Bundled plugins also follow this structure, but bundled plugins are
trusted implicitly when they ship inside IntegraPose.

## `plugin.json`

`plugin.json` is the metadata shown in the Plugin Manager and trust
dialogs. The current loader reads these fields:

```json
{
  "name": "My Lab Plugin",
  "version": "0.1.0",
  "description": "Adds a custom analysis window for our lab workflow.",
  "author": "Your Lab"
}
```

| Field | Required | Notes |
| --- | --- | --- |
| `name` | Recommended | Display name in the Plugin Manager |
| `version` | Recommended | Used in the Plugin Manager and opt-in record |
| `description` | Recommended | Shown before the user enables the plugin |
| `author` | Recommended | Helps users evaluate plugin trust |

The loader currently imports `plugin.py` by convention. Do not rely on an
`entry_point` field unless a future IntegraPose release documents it.

## `plugin.py`

The loader supports two registration styles.

### Preferred class style

```python
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk


class IntegraPosePlugin:
    def __init__(self) -> None:
        self.main_app = None
        self._window = None

    def register(self, main_app) -> None:
        self.main_app = main_app

        menu = getattr(main_app, "plugins_menu", None)
        if menu is None:
            self._log("Plugins menu is not available; plugin was not registered.", "ERROR")
            return

        menu.add_command(label="My Lab Plugin", command=self._open_window)
        self._log("My Lab Plugin registered.", "INFO")

    def _open_window(self) -> None:
        root = getattr(self.main_app, "root", None)
        if root is None:
            self._log("Application root is not available.", "ERROR")
            return

        if self._window is not None and self._window.winfo_exists():
            self._window.lift()
            self._window.focus_force()
            return

        self._window = tk.Toplevel(root)
        self._window.title("My Lab Plugin")
        self._window.geometry("420x220")
        self._window.bind("<Destroy>", lambda _event: self._clear_window(), add="+")

        ttk.Label(
            self._window,
            text="Custom plugin window",
            padding=16,
        ).pack(fill="both", expand=True)

    def _clear_window(self) -> None:
        self._window = None

    def _log(self, message: str, level: str = "INFO") -> None:
        logger = getattr(self.main_app, "log_message", None)
        if callable(logger):
            logger(message, level)
```

### Function style

```python
from __future__ import annotations

from tkinter import messagebox


def register_plugin(main_app) -> None:
    menu = getattr(main_app, "plugins_menu", None)
    if menu is None:
        logger = getattr(main_app, "log_message", None)
        if callable(logger):
            logger("Plugins menu is not available; plugin was not registered.", "ERROR")
        return

    menu.add_command(
        label="My Simple Plugin",
        command=lambda: messagebox.showinfo(
            "My Simple Plugin",
            "Plugin command executed.",
            parent=getattr(main_app, "root", None),
        ),
    )
```

Use the class style when the plugin owns a window, background job, or
state that must be reused. Use the function style for a small menu command
or compatibility wrapper.

## Registration Contract

During startup or when a user enables a plugin, IntegraPose:

1. Discovers plugin directories.
2. Reads `plugin.json` metadata.
3. Requires third-party trust approval before import.
4. Imports `<plugin_folder>/plugin.py`.
5. Looks for `IntegraPosePlugin` first.
6. If present, instantiates it and calls `instance.register(main_app)`.
7. Otherwise, looks for a callable `register_plugin(main_app)`.
8. Lets the plugin add entries to `main_app.plugins_menu`.

If neither entry point exists, the plugin is skipped and the failure is
written to the Log tab.

## Safe `main_app` Surface

Plugins receive the active GUI application object as `main_app`. Treat it
as a host object, not as a stable full API. These surfaces are the safest
to use:

| Attribute or method | Use |
| --- | --- |
| `main_app.root` | Parent for `tk.Toplevel`, dialogs, and message boxes |
| `main_app.plugins_menu` | Add plugin menu commands |
| `main_app.log_message(message, level)` | Write to the IntegraPose Log tab |
| `main_app._toast(message, level=...)` | Optional short user notification, when present |
| `main_app.config` | Read current GUI settings when needed |

Use `getattr` and callable checks before calling host methods. This keeps
plugins compatible across small host changes:

```python
logger = getattr(main_app, "log_message", None)
if callable(logger):
    logger("Plugin started.", "INFO")
```

Avoid direct changes to core tab widgets, private controller attributes,
or long-lived global state unless the plugin is tied to a specific
IntegraPose release.

## Window Guidelines

Most plugins should launch their own `tk.Toplevel` window instead of
adding controls directly into the main tabs.

Recommended pattern:

```python
if self._window is not None and self._window.winfo_exists():
    self._window.lift()
    self._window.focus_force()
    return

self._window = tk.Toplevel(self.main_app.root)
self._window.bind("<Destroy>", lambda _event: self._clear_window(), add="+")
```

This prevents duplicate windows and allows the plugin to clean up timers,
background tasks, or references when the window closes.

## Dependencies

IntegraPose does not install plugin dependencies automatically. A plugin
should fail gracefully when an optional dependency is missing:

```python
try:
    import pandas as pd
except ModuleNotFoundError as exc:
    pd = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None
```

Then show a clear message from the plugin command:

```python
if pd is None:
    messagebox.showerror(
        "My Lab Plugin",
        f"Missing optional dependency: {_IMPORT_ERROR.name}",
        parent=getattr(self.main_app, "root", None),
    )
    return
```

For larger plugins, include a plugin-local `requirements.txt`:

```text
my_lab_plugin/
  plugin.json
  plugin.py
  requirements.txt
```

Tell users to install it into the same Python environment that launches
IntegraPose:

```bash
python -m pip install -r ~/.integrapose/plugins/my_lab_plugin/requirements.txt
```

## Trust And Opt-In

Plugins are disabled by default.

Third-party plugins must have a valid `plugin.json` before they can be
enabled. When a user enables a third-party plugin, IntegraPose asks for
trust confirmation and stores the decision under:

```text
~/.integrapose/security/trusted_plugins.json
```

Plugin enablement is stored separately under:

```text
~/.integrapose/plugins/enabled_plugins.json
```

If a user accidentally declines a third-party plugin trust prompt, they
can select the plugin in **Plugins -> Manage Plugins...** and press
**Enable** again to re-open the trust prompt. They can also use
**Reset Trust** to clear the stored allow/deny decision from the Plugin
Manager without manually deleting `trusted_plugins.json`.

IntegraPose fingerprints each enabled plugin's `.py`, `.json`, `.yaml`,
`.yml`, and `.toml` files. If those files change, the plugin must be
enabled again. This protects users from silently running changed plugin
code.

Fingerprinting skips common cache folders such as:

```text
__pycache__/
.git/
.pytest_cache/
.mypy_cache/
```

## Development Workflow

1. Create a folder under `~/.integrapose/plugins/`.
2. Add `plugin.json`.
3. Add `plugin.py`.
4. Launch IntegraPose.
5. Open **Plugins -> Manage Plugins...**.
6. Select the plugin and click **Enable**.
7. Accept the trust prompt if you trust the plugin code.
8. Run the command added to the **Plugins** menu.
9. Watch the Log tab for registration or import errors.

During development, if you edit `plugin.py` or `plugin.json`, refresh the
Plugin Manager and enable the plugin again so the new fingerprint is
accepted.

## Minimal Plugin Template

```text
my_lab_plugin/
  plugin.json
  plugin.py
  README.md
```

`plugin.json`:

```json
{
  "name": "My Lab Plugin",
  "version": "0.1.0",
  "description": "Adds one lab-specific analysis command.",
  "author": "Example Lab"
}
```

`plugin.py`:

```python
from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class IntegraPosePlugin:
    def __init__(self) -> None:
        self.main_app = None
        self._window = None

    def register(self, main_app) -> None:
        self.main_app = main_app
        menu = getattr(main_app, "plugins_menu", None)
        if menu is None:
            self._log("Plugins menu was unavailable.", "ERROR")
            return
        menu.add_command(label="My Lab Plugin", command=self._open)
        self._log("My Lab Plugin registered.", "INFO")

    def _open(self) -> None:
        root = getattr(self.main_app, "root", None)
        if root is None:
            self._log("Cannot open plugin: application root is unavailable.", "ERROR")
            return

        if self._window is not None and self._window.winfo_exists():
            self._window.lift()
            return

        self._window = tk.Toplevel(root)
        self._window.title("My Lab Plugin")
        self._window.bind("<Destroy>", lambda _event: self._clear_window(), add="+")
        ttk.Label(self._window, text="Hello from a user plugin.", padding=20).pack()

    def _clear_window(self) -> None:
        self._window = None

    def _log(self, message: str, level: str = "INFO") -> None:
        logger = getattr(self.main_app, "log_message", None)
        if callable(logger):
            logger(message, level)
```

## Packaging Recommendations

For a plugin you plan to share:

- Include a `README.md` with install steps, expected inputs, and outputs.
- Include `requirements.txt` only for plugin-specific dependencies.
- Keep large models, videos, and generated outputs out of the plugin folder.
- Use relative paths in examples.
- Log meaningful errors with `main_app.log_message`.
- Use dialogs only for user-actionable failures.
- Avoid importing heavy optional dependencies at module import time when the
  plugin can show an "unavailable" menu item instead.

## Compatibility Checklist

Before sharing a plugin:

- The folder has `plugin.json` and `plugin.py`.
- `plugin.json` is valid JSON.
- `plugin.py` defines `IntegraPosePlugin.register(main_app)` or
  `register_plugin(main_app)`.
- The plugin can be enabled from **Plugins -> Manage Plugins...**.
- The plugin command appears under **Plugins**.
- Missing dependencies produce a clear message instead of crashing startup.
- Reopening the plugin command does not create duplicate windows.
- Closing the plugin window releases timers or background references.
- No local absolute paths, credentials, private data, model weights, or test
  outputs are included.
