from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from integra_pose.plugins.plugin_tandem_yolo_toolkit.yolo_temporal_classifier import (
    safe_checkpoint_io,
)


def test_standalone_loader_blocks_old_pytorch_unsafe_fallback(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    load = MagicMock()
    monkeypatch.setattr(safe_checkpoint_io, "_core_loader", None)
    monkeypatch.setattr(safe_checkpoint_io, "_supports_weights_only", lambda: False)
    monkeypatch.setattr(safe_checkpoint_io.torch, "load", load)
    monkeypatch.delenv(safe_checkpoint_io.UNSAFE_MODEL_LOAD_ENV, raising=False)

    with pytest.raises(RuntimeError, match="does not support weights-only"):
        safe_checkpoint_io.safe_torch_load(checkpoint)

    load.assert_not_called()


def test_standalone_loader_requires_opt_in_before_unsafe_load(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    load = MagicMock(return_value={"model": {}})
    monkeypatch.setattr(safe_checkpoint_io, "_core_loader", None)
    monkeypatch.setattr(safe_checkpoint_io, "_supports_weights_only", lambda: False)
    monkeypatch.setattr(safe_checkpoint_io.torch, "load", load)
    monkeypatch.setenv(safe_checkpoint_io.UNSAFE_MODEL_LOAD_ENV, "1")

    result = safe_checkpoint_io.safe_torch_load(checkpoint, map_location="cpu")

    assert result == {"model": {}}
    load.assert_called_once_with(str(checkpoint), map_location="cpu")


def test_standalone_loader_uses_weights_only_when_supported(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    load = MagicMock(return_value={"model": {}})
    monkeypatch.setattr(safe_checkpoint_io, "_core_loader", None)
    monkeypatch.setattr(safe_checkpoint_io, "_supports_weights_only", lambda: True)
    monkeypatch.setattr(safe_checkpoint_io.torch, "load", load)

    safe_checkpoint_io.safe_torch_load(checkpoint, map_location="cpu")

    load.assert_called_once_with(
        str(checkpoint),
        map_location="cpu",
        weights_only=True,
    )
