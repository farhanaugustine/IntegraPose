from __future__ import annotations

import configparser
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_supported_python_range_matches_slots_based_code() -> None:
    parser = configparser.ConfigParser()
    parser.read(REPO_ROOT / "setup.cfg", encoding="utf-8")

    assert parser["options"]["python_requires"].strip() == ">=3.10,<3.12"
    classifiers = parser["metadata"]["classifiers"]
    assert "Python :: 3.9" not in classifiers
    assert "Python :: 3.10" in classifiers
    assert "Python :: 3.11" in classifiers


def test_standard_install_paths_do_not_pull_headless_opencv() -> None:
    requirement_files = [
        REPO_ROOT / "requirements.txt",
        REPO_ROOT / "integra_pose" / "plugins" / "requirements.txt",
        REPO_ROOT
        / "integra_pose"
        / "plugins"
        / "plugin_autolabel_forge"
        / "requirements.txt",
    ]
    active_requirements = []
    for path in requirement_files:
        active_requirements.extend(
            line.strip().lower()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    parser = configparser.ConfigParser()
    parser.read(REPO_ROOT / "setup.cfg", encoding="utf-8")
    setup_requirements = [parser["options"].get("install_requires", "")]
    setup_requirements.extend(parser["options.extras_require"].values())
    combined = "\n".join([*active_requirements, *setup_requirements]).lower()

    assert "opencv-python-headless" not in combined
    assert "roboflow==" not in combined
    assert "opencv-python==4.9.0.80" in combined


def test_core_logic_and_bundled_plugins_are_explicit_packages() -> None:
    required_package_markers = [
        REPO_ROOT / "integra_pose" / "logic" / "__init__.py",
        REPO_ROOT
        / "integra_pose"
        / "plugins"
        / "plugin_zone_counter"
        / "__init__.py",
    ]
    assert all(path.is_file() for path in required_package_markers)
