from __future__ import annotations

from pathlib import Path


def test_albumentations_gui_requirements_include_eval_type_backport() -> None:
    requirements_path = Path(__file__).resolve().parent.parent / "requirements-albumentations-gui.txt"
    lines = [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert "albumentations==1.4.21" in lines
    assert "eval-type-backport==0.3.1" in lines
