from __future__ import annotations

import configparser
from pathlib import Path


AUTOLABEL_FORGE_RUNTIME = {
    "autodistill==0.1.29",
    "autodistill-grounding-dino==0.1.4",
    "roboflow==1.2.13",
    "transformers==4.57.6",
}

CORE_COMPAT_RUNTIME = {
    "eval-type-backport==0.3.1",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _non_comment_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_setup_cfg_plugin_extras_include_autolabel_forge_runtime() -> None:
    config = configparser.ConfigParser()
    config.read(_repo_root() / "setup.cfg", encoding="utf-8")

    plugins_extra = set(_non_comment_lines(_repo_root() / "integra_pose" / "plugins" / "requirements.txt"))
    plugins_cfg = {
        line.strip()
        for line in config["options.extras_require"]["plugins"].splitlines()
        if line.strip()
    }
    dev_cfg = {
        line.strip()
        for line in config["options.extras_require"]["dev"].splitlines()
        if line.strip()
    }

    assert AUTOLABEL_FORGE_RUNTIME.issubset(plugins_cfg)
    assert AUTOLABEL_FORGE_RUNTIME.issubset(dev_cfg)
    assert AUTOLABEL_FORGE_RUNTIME.issubset(plugins_extra)


def test_autolabel_forge_runtime_is_declared_in_local_and_unified_requirements() -> None:
    root_reqs = set(_non_comment_lines(_repo_root() / "requirements.txt"))
    plugin_reqs = set(
        _non_comment_lines(
            _repo_root()
            / "integra_pose"
            / "plugins"
            / "plugin_autolabel_forge"
            / "requirements.txt"
        )
    )

    assert AUTOLABEL_FORGE_RUNTIME.issubset(root_reqs)
    assert AUTOLABEL_FORGE_RUNTIME.issubset(plugin_reqs)


def test_eval_type_backport_is_declared_in_core_install_metadata() -> None:
    config = configparser.ConfigParser()
    config.read(_repo_root() / "setup.cfg", encoding="utf-8")

    install_requires = {
        line.strip()
        for line in config["options"]["install_requires"].splitlines()
        if line.strip()
    }
    root_reqs = set(_non_comment_lines(_repo_root() / "requirements.txt"))

    assert CORE_COMPAT_RUNTIME.issubset(install_requires)
    assert CORE_COMPAT_RUNTIME.issubset(root_reqs)
