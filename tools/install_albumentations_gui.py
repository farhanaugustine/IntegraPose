#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CORE_PINS = (
    "numpy==1.26.4",
    "scipy==1.11.4",
    "opencv-python==4.9.0.80",
)

VERIFY_SNIPPET = r"""
from importlib.metadata import PackageNotFoundError, version

packages = [
    "numpy",
    "scipy",
    "opencv-python",
    "albumentations",
    "albucore",
    "pydantic",
    "eval-type-backport",
]

for package_name in packages:
    try:
        print(f"{package_name}={version(package_name)}")
    except PackageNotFoundError:
        print(f"{package_name}=MISSING")

import cv2  # noqa: F401
import albumentations  # noqa: F401
"""


def run(cmd: list[str], *, cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_optional(cmd: list[str], *, cwd: Path) -> bool:
    print("+", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd), check=False)
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Install the Albumentations GUI add-on without breaking IntegraPose's "
            "GUI OpenCV environment."
        )
    )
    parser.add_argument(
        "--skip-core-repair",
        action="store_true",
        help=(
            "Skip the defensive reinstall of numpy, scipy, and opencv-python. "
            "Use this only when you know the environment is already clean."
        ),
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the final import and version verification step.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    requirements_path = repo_root / "requirements-albumentations-gui.txt"

    if not requirements_path.exists():
        print(f"Missing requirements file: {requirements_path}", file=sys.stderr)
        return 1

    python_exe = Path(sys.executable)
    print(f"Python executable: {python_exe}")
    print(f"Repository root: {repo_root}")

    pip_prefix = [str(python_exe), "-m", "pip"]

    if not args.skip_core_repair:
        run_optional([*pip_prefix, "uninstall", "-y", "opencv-python-headless"], cwd=repo_root)
        run([*pip_prefix, "install", *CORE_PINS], cwd=repo_root)

    run([*pip_prefix, "install", "--no-deps", "-r", str(requirements_path)], cwd=repo_root)

    headless_still_present = run_optional(
        [*pip_prefix, "show", "opencv-python-headless"],
        cwd=repo_root,
    )
    if headless_still_present:
        print(
            "opencv-python-headless is still installed. Remove it before using the GUI.",
            file=sys.stderr,
        )
        return 1

    if not args.no_verify:
        run([str(python_exe), "-c", VERIFY_SNIPPET], cwd=repo_root)

    print("Albumentations GUI add-on install completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
