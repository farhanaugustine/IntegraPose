from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from integra_pose.utils import repro_bundle


class _Var:
    def __init__(self, value: str = "") -> None:
        self.value = value

    def get(self) -> str:
        return self.value

    def set(self, value: str) -> None:
        self.value = value


def _json_bytes(payload) -> bytes:
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def _write_bundle(path: Path, files: dict[str, bytes]) -> None:
    manifest = {
        "schema_version": 1,
        "files": [
            {
                "path": name,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
            for name, content in sorted(files.items())
        ],
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
        archive.writestr("bundle_manifest.json", _json_bytes(manifest))


def _fake_app(import_root: Path, *, apply_side_effect=None):
    config = SimpleNamespace(
        inference=SimpleNamespace(trained_model_path_infer=_Var()),
        webcam=SimpleNamespace(webcam_model_path=_Var()),
        training=SimpleNamespace(export_model_path=_Var()),
        _gather_config_as_dict=MagicMock(return_value={"current": {"value": 1}}),
        _apply_config_from_dict=MagicMock(side_effect=apply_side_effect),
    )
    return SimpleNamespace(
        config=config,
        _repro_import_root=import_root,
        _refresh_model_registry_views=MagicMock(),
    )


@pytest.mark.parametrize(
    "member_name",
    ["../escape.txt", "folder/../../escape.txt", r"C:\escape.txt", r"\\server\share\file.txt"],
)
def test_safe_extraction_rejects_traversal_and_absolute_paths(tmp_path, member_name) -> None:
    archive_path = tmp_path / "invalid.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(member_name, b"bad")

    extraction_root = tmp_path / "extract"
    extraction_root.mkdir()
    with zipfile.ZipFile(archive_path, "r") as archive:
        with pytest.raises(ValueError):
            repro_bundle._extract_archive_safely(archive, extraction_root)

    assert not (tmp_path / "escape.txt").exists()


def test_safe_extraction_enforces_member_size_quota(tmp_path, monkeypatch) -> None:
    archive_path = tmp_path / "large.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("large.bin", b"1234")
    monkeypatch.setattr(repro_bundle, "_MAX_ARCHIVE_MEMBER_BYTES", 3)

    extraction_root = tmp_path / "extract"
    extraction_root.mkdir()
    with zipfile.ZipFile(archive_path, "r") as archive:
        with pytest.raises(ValueError, match="too large"):
            repro_bundle._extract_archive_safely(archive, extraction_root)


def test_import_validates_metadata_paths_before_applying_config(tmp_path) -> None:
    bundle_path = tmp_path / "malicious.integrapose.zip"
    files = {
        "project_config.json": _json_bytes({"foreign": True}),
        "selected_models.json": _json_bytes(
            [
                {
                    "role": "inference",
                    "bundle_path": "../outside.pt",
                    "sha256": "0" * 64,
                }
            ]
        ),
    }
    _write_bundle(bundle_path, files)
    app = _fake_app(tmp_path / "imports")

    with pytest.raises(ValueError, match="traversal"):
        repro_bundle.import_bundle(app, bundle_path)

    app.config._apply_config_from_dict.assert_not_called()
    assert not (tmp_path / "imports").exists()


def test_import_restores_artifacts_to_unique_local_root(tmp_path) -> None:
    bundle_path = tmp_path / "study.integrapose.zip"
    model_bytes = b"model artifact"
    curation_bytes = b'{"reviewed": true}'
    model_path = "artifacts/models/inference_best.pt"
    curation_path = "artifacts/project/.integrapose_assisted_pose_manifest.json"
    files = {
        "project_config.json": _json_bytes(
            {"setup": {"dataset_root_yaml": r"C:\untrusted\foreign-project"}}
        ),
        "selected_models.json": _json_bytes(
            [
                {
                    "role": "inference",
                    "bundle_path": model_path,
                    "sha256": hashlib.sha256(model_bytes).hexdigest(),
                }
            ]
        ),
        "project_artifacts.json": _json_bytes(
            [
                {
                    "role": "assisted_pose_curation_manifest",
                    "bundle_path": curation_path,
                    "sha256": hashlib.sha256(curation_bytes).hexdigest(),
                }
            ]
        ),
        model_path: model_bytes,
        curation_path: curation_bytes,
    }
    _write_bundle(bundle_path, files)
    import_root = tmp_path / "imports"
    app = _fake_app(import_root)

    result = repro_bundle.import_bundle(app, bundle_path)

    final_root = Path(result["import_root"]).resolve()
    assert final_root.parent == import_root.resolve()
    assert final_root.is_dir()
    imported_model = Path(result["imported_models"][0]["path"])
    imported_manifest = Path(result["imported_project_artifacts"][0])
    assert imported_model.read_bytes() == model_bytes
    assert imported_manifest.read_bytes() == curation_bytes
    assert app.config.inference.trained_model_path_infer.get() == str(imported_model)
    assert app.config._apply_config_from_dict.call_count == 1
    applied_payload = app.config._apply_config_from_dict.call_args.args[0]
    assert applied_payload["setup"]["dataset_root_yaml"] == ""


def test_config_reference_redaction_preserves_scientific_values() -> None:
    payload = {
        "setup": {
            "dataset_root_yaml": r"C:\Users\researcher\Study",
            "behaviors_list": [{"id": 0, "name": "Walking"}],
        },
        "inference": {
            "source": "rtsp://user:secret@example.test/camera",
            "infer_conf_var": 0.25,
            "infer_device_var": "-1",
        },
    }

    redacted, fields = repro_bundle._redact_config_references(payload)

    assert redacted["setup"]["dataset_root_yaml"] == ""
    assert redacted["inference"]["source"] == ""
    assert redacted["inference"]["infer_conf_var"] == 0.25
    assert redacted["inference"]["infer_device_var"] == "-1"
    assert redacted["setup"]["behaviors_list"] == [{"id": 0, "name": "Walking"}]
    assert set(fields) == {"setup.dataset_root_yaml", "inference.source"}


def test_import_rolls_back_staging_and_config_after_apply_failure(tmp_path) -> None:
    bundle_path = tmp_path / "broken.integrapose.zip"
    files = {"project_config.json": _json_bytes({"foreign": True})}
    _write_bundle(bundle_path, files)
    previous = {"current": {"value": 1}}

    def _apply(payload):
        if payload != previous:
            raise RuntimeError("cannot apply")

    import_root = tmp_path / "imports"
    app = _fake_app(import_root, apply_side_effect=_apply)

    with pytest.raises(RuntimeError, match="cannot apply"):
        repro_bundle.import_bundle(app, bundle_path)

    assert app.config._apply_config_from_dict.call_args_list[-1].args[0] == previous
    assert import_root.is_dir()
    assert list(import_root.iterdir()) == []


def test_manifest_rejects_undeclared_archive_file(tmp_path) -> None:
    bundle_path = tmp_path / "undeclared.zip"
    files = {"project_config.json": _json_bytes({})}
    _write_bundle(bundle_path, files)
    with zipfile.ZipFile(bundle_path, "a") as archive:
        archive.writestr("artifacts/models/undeclared.pt", b"untrusted")
    app = _fake_app(tmp_path / "imports")

    with pytest.raises(ValueError, match="exactly describe"):
        repro_bundle.import_bundle(app, bundle_path)

    app.config._apply_config_from_dict.assert_not_called()


def test_export_redacts_local_identifiers_and_writes_atomically(tmp_path, monkeypatch) -> None:
    config = SimpleNamespace(
        project_file_path=r"C:\Users\researcher\study.json",
        setup=SimpleNamespace(dataset_root_yaml=_Var(r"C:\Users\researcher\dataset")),
        inference=SimpleNamespace(trained_model_path_infer=_Var()),
        webcam=SimpleNamespace(webcam_model_path=_Var()),
        training=SimpleNamespace(export_model_path=_Var()),
        _gather_config_as_dict=MagicMock(
            return_value={
                "setup": {"dataset_root_yaml": r"C:\Users\researcher\dataset"},
                "analytics": {"min_bout_duration_var": 5},
            }
        ),
    )
    app = SimpleNamespace(config=config, _current_app_version=lambda: "test")

    def _capture(root):
        (root / "environment.yml").write_text("name: test\n", encoding="utf-8")
        return {
            "success": True,
            "bundle_path": "environment.yml",
            "command_used": "test",
            "attempts": [],
        }

    monkeypatch.setattr(repro_bundle, "_capture_conda_environment", _capture)
    monkeypatch.setattr(
        repro_bundle.model_registry,
        "list_records",
        lambda _app: [{"model_path": r"C:\Users\researcher\model.pt", "score": 0.9}],
    )
    destination = tmp_path / "study.zip"

    repro_bundle.export_bundle(app, destination)

    assert destination.is_file()
    assert not list(tmp_path.glob(".study.zip.*.tmp"))
    with zipfile.ZipFile(destination, "r") as archive:
        project_config = json.loads(archive.read("project_config.json"))
        project_context = json.loads(archive.read("project_context.json"))
        registry = json.loads(archive.read("model_registry_snapshot.json"))
        traceability = json.loads(archive.read("traceability_manifest.json"))
    assert project_config["setup"]["dataset_root_yaml"] == ""
    assert project_context["project_file_name"] == "study.json"
    assert project_context["dataset_name"] == "dataset"
    assert registry[0]["model_path"] == "model.pt"
    assert traceability["local_config_references_redacted"] == ["setup.dataset_root_yaml"]
    assert "hostname" not in traceability["runtime_context"]
