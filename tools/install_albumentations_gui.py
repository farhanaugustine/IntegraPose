#!/usr/bin/env python
from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import subprocess
import sys
import tempfile
from pathlib import Path


NUMERIC_PINS = ("numpy==1.26.4", "scipy==1.11.4")
OPENCV_PIN = "opencv-python==4.9.0.80"
CORE_PACKAGE_NAMES = {"numpy", "scipy", "opencv-python"}

VERIFY_SNIPPET = r"""
from importlib.metadata import PackageNotFoundError, version
import sys

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

try:
    import cv2
except Exception as exc:
    print(f"VERIFY_ERROR: cv2 import failed: {exc}", file=sys.stderr)
    raise SystemExit(2)

if not hasattr(cv2, "CV_8U"):
    print(
        "VERIFY_ERROR: cv2 imported, but required OpenCV constants are missing. "
        "This usually means opencv-python-headless was removed after opencv-python "
        "and the GUI OpenCV wheel needs to be reinstalled.",
        file=sys.stderr,
    )
    raise SystemExit(2)

build_info = cv2.getBuildInformation()
gui_lines = [line.strip() for line in build_info.splitlines() if line.strip().startswith("GUI:")]
if not gui_lines or "NONE" in gui_lines[0].upper():
    print(
        "VERIFY_ERROR: OpenCV imported, but GUI/HighGUI support is not enabled. "
        "cv2.imshow requires GUI-enabled opencv-python, not opencv-python-headless.",
        file=sys.stderr,
    )
    raise SystemExit(2)
print(f"opencv-gui={gui_lines[0].split(':', 1)[1].strip()}")

try:
    import albumentations  # noqa: F401
except Exception as exc:
    print(f"VERIFY_ERROR: albumentations import failed: {exc}", file=sys.stderr)
    raise SystemExit(2)
"""


def run(cmd: list[str], *, cwd: Path, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        check=True,
        text=True,
        capture_output=capture,
    )


def run_optional(cmd: list[str], *, cwd: Path, capture: bool = False) -> bool:
    print("+", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        check=False,
        text=True,
        capture_output=capture,
    )
    return result.returncode == 0


def package_installed(package_name: str) -> bool:
    return installed_version(package_name) is not None


def installed_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def filtered_addon_requirements(requirements_path: Path) -> Path:
    lines: list[str] = []
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            lines.append(raw_line)
            continue
        package_name = stripped.split("==", 1)[0].split(">=", 1)[0].split("<", 1)[0].strip().lower()
        if package_name in CORE_PACKAGE_NAMES:
            continue
        lines.append(raw_line)

    handle = tempfile.NamedTemporaryFile(
        "w",
        delete=False,
        suffix="-albumentations-addon-requirements.txt",
        encoding="utf-8",
    )
    with handle:
        handle.write("\n".join(lines))
        handle.write("\n")
    return Path(handle.name)


def verify_install(python_exe: Path, *, cwd: Path) -> bool:
    print("+", str(python_exe), "-c", "<verify albumentations gui add-on>")
    result = subprocess.run(
        [str(python_exe), "-c", VERIFY_SNIPPET],
        cwd=str(cwd),
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode == 0:
        if result.stdout.strip():
            print(result.stdout.rstrip())
        return True

    print("Albumentations GUI add-on verification failed.", file=sys.stderr)
    for stream in (result.stdout, result.stderr):
        for line in stream.splitlines():
            if line.startswith("VERIFY_ERROR:") or "=" in line:
                print(line, file=sys.stderr)
    print(
        "Suggested repair: rerun this installer without --skip-core-repair so it can "
        "remove opencv-python-headless and reinstall GUI-enabled opencv-python.",
        file=sys.stderr,
    )
    return False


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
    existing_opencv_version = installed_version("opencv-python")

    if not args.skip_core_repair:
        headless_was_present = package_installed("opencv-python-headless")
        run_optional([*pip_prefix, "uninstall", "-y", "opencv-python-headless"], cwd=repo_root)
        run([*pip_prefix, "install", *NUMERIC_PINS], cwd=repo_root)
        if headless_was_present or existing_opencv_version is None:
            opencv_requirement = (
                f"opencv-python=={existing_opencv_version}"
                if existing_opencv_version is not None
                else OPENCV_PIN
            )
            run([*pip_prefix, "install", "--force-reinstall", "--no-deps", opencv_requirement], cwd=repo_root)

    filtered_requirements = filtered_addon_requirements(requirements_path)
    try:
        run([*pip_prefix, "install", "--no-deps", "-r", str(filtered_requirements)], cwd=repo_root)
    finally:
        try:
            filtered_requirements.unlink()
        except OSError:
            pass

    if package_installed("opencv-python-headless"):
        print(
            "opencv-python-headless is still installed. Remove it before using the GUI.",
            file=sys.stderr,
        )
        return 1

    if not args.no_verify:
        if not verify_install(python_exe, cwd=repo_root):
            opencv_requirement = (
                f"opencv-python=={existing_opencv_version}"
                if existing_opencv_version is not None
                else OPENCV_PIN
            )
            print(f"Attempting one OpenCV GUI repair with {opencv_requirement}.")
            run([*pip_prefix, "install", "--force-reinstall", "--no-deps", opencv_requirement], cwd=repo_root)
            if not verify_install(python_exe, cwd=repo_root):
                return 1

    print("Albumentations GUI add-on install completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
