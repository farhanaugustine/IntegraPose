from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tools import install_albumentations_gui


def test_albumentations_gui_requirements_include_eval_type_backport() -> None:
    requirements_path = Path(__file__).resolve().parent.parent / "requirements-albumentations-gui.txt"
    lines = [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert "albumentations==1.4.21" in lines
    assert "eval-type-backport==0.3.1" in lines


def test_verify_install_reports_concise_error(monkeypatch, capsys, tmp_path) -> None:
    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["python", "-c", "<verify>"],
            returncode=2,
            stdout="numpy=1.26.4\nopencv-python=4.9.0.80\n",
            stderr=(
                "Traceback (most recent call last):\n"
                "  File \"<string>\", line 1, in <module>\n"
                "VERIFY_ERROR: cv2 imported, but required OpenCV constants are missing.\n"
            ),
        )

    monkeypatch.setattr(install_albumentations_gui.subprocess, "run", fake_run)

    assert install_albumentations_gui.verify_install(Path(sys.executable), cwd=tmp_path) is False
    captured = capsys.readouterr()

    assert "VERIFY_ERROR: cv2 imported, but required OpenCV constants are missing." in captured.err
    assert "Traceback" not in captured.err
    assert "File \"<string>\"" not in captured.err


def test_filtered_addon_requirements_excludes_core_packages(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "\n".join(
            [
                "numpy==1.26.4",
                "scipy==1.11.4",
                "opencv-python==4.9.0.80",
                "albumentations==1.4.21",
                "albucore==0.0.20",
            ]
        ),
        encoding="utf-8",
    )

    filtered = install_albumentations_gui.filtered_addon_requirements(requirements)
    try:
        lines = filtered.read_text(encoding="utf-8").splitlines()
    finally:
        filtered.unlink(missing_ok=True)

    assert "numpy==1.26.4" not in lines
    assert "scipy==1.11.4" not in lines
    assert "opencv-python==4.9.0.80" not in lines
    assert "albumentations==1.4.21" in lines
    assert "albucore==0.0.20" in lines
