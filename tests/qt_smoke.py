"""Exercise real Qt child startup and clean close, with model calls prohibited."""
import os
from pathlib import Path
import sys
import tempfile
import tkinter as tk
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from integra_pose.plugins.plugin_assisted_pose_curation.qt_bridge import AssistedPoseCurationWindow, _Controller


def prohibited(*args, **kwargs):
    raise AssertionError("No model execution is allowed in this smoke test")


workspace = Path.cwd()
with tempfile.TemporaryDirectory(prefix="qt_smoke_", dir=workspace / ".qt_preview") as temporary:
    os.chdir(temporary)
    root = tk.Tk()
    root.withdraw()
    outcome = []
    with patch.object(_Controller, "_load_prefill_from_main_app", lambda *a, **k: None), patch.object(_Controller, "_get_pose_model", prohibited), patch.object(_Controller, "_get_embed_model", prohibited):
        window = AssistedPoseCurationWindow(None, parent=root)
        def check():
            if window._fallback:
                outcome.append("Qt failed: " + "\n".join(window._stderr))
                root.quit()
            elif window._ready:
                window._incoming.put(dict(id=99, command="close", data={"choice": "discard"}))
                root.after(200, closed)
            else:
                root.after(100, check)
        def closed():
            if window._process.poll() is not None:
                outcome.append("ok" if window._process.returncode == 0 else str(window._stderr))
                root.quit()
            else:
                root.after(100, closed)
        root.after(100, check)
        root.after(15000, root.quit)
        root.mainloop()
        if window._process.poll() is None:
            window._process.terminate()
            window._process.wait(timeout=5)
        root.destroy()
    os.chdir(workspace)
    assert outcome == ["ok"], outcome
print("Qt child startup, JSON handshake, and clean shutdown passed. No model loaded.")
