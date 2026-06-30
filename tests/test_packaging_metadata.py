from __future__ import annotations

from pathlib import Path

from setuptools.config.setupcfg import read_configuration


def test_dev_install_does_not_pin_pytorch_packages() -> None:
    cfg = read_configuration(str(Path(__file__).resolve().parents[1] / "setup.cfg"))
    options = cfg["options"]
    extras = options["extras_require"]
    groups = {
        "install_requires": options["install_requires"],
        "plugins": extras["plugins"],
        "dev": extras["dev"],
    }
    blocked = ("torch", "torchvision")
    offenders: list[str] = []
    for group, requirements in groups.items():
        for requirement in requirements:
            name = requirement.split(";", 1)[0].strip().lower()
            if name.startswith(blocked):
                offenders.append(f"{group}: {requirement}")
    assert offenders == []
