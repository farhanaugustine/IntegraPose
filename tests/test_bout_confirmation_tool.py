import importlib.util
from pathlib import Path
import sys

import pandas as pd


def _load_module(rel_path: str, module_name: str):
    repo_root = Path(__file__).resolve().parent.parent
    module_path = repo_root / rel_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_bout_confirmation_mod = _load_module(
    "integra_pose/gui/bout_confirmation_tool.py",
    "test_bout_confirmation_tool_module",
)
normalize_bout_confirmation_dataframe = _bout_confirmation_mod.normalize_bout_confirmation_dataframe


def test_normalize_bout_confirmation_dataframe_adds_correction_columns():
    raw = pd.DataFrame(
        [
            {
                "Track ID": 1,
                "Behavior": "Walking",
                "Start Frame": 10,
                "End Frame": 22,
            }
        ]
    )

    normalized = normalize_bout_confirmation_dataframe(raw)

    assert normalized.iloc[0]["status"] == "unreviewed"
    assert normalized.iloc[0]["Original Behavior"] == "Walking"
    assert normalized.iloc[0]["Corrected Behavior"] == "Walking"
    assert normalized.iloc[0]["Review Status"] == "unreviewed"
    assert bool(normalized.iloc[0]["Corrected Manually"]) is False


def test_normalize_bout_confirmation_dataframe_preserves_corrected_rows():
    raw = pd.DataFrame(
        [
            {
                "Track ID": 1,
                "Behavior": "Grooming",
                "Start Frame": 10,
                "End Frame": 22,
                "status": "corrected",
                "Original Behavior": "Walking",
                "Corrected Behavior": "Grooming",
                "Review Status": "corrected",
                "Corrected Manually": True,
            }
        ]
    )

    normalized = normalize_bout_confirmation_dataframe(raw)

    assert normalized.iloc[0]["status"] == "corrected"
    assert normalized.iloc[0]["Original Behavior"] == "Walking"
    assert normalized.iloc[0]["Corrected Behavior"] == "Grooming"
    assert normalized.iloc[0]["Review Status"] == "corrected"
    assert bool(normalized.iloc[0]["Corrected Manually"]) is True
