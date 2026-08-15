from types import SimpleNamespace

from integra_pose.utils.keypoint_schema import (
    resolve_model_keypoint_schema,
    safe_keypoint_token,
    validate_keypoint_names,
)


def _fake_model(**overrides):
    payload = {
        "kpt_names": None,
        "overrides": {},
        "ckpt": {},
        "model": SimpleNamespace(kpt_names=None, yaml={}),
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def test_resolve_model_keypoint_schema_uses_adjacent_dataset_yaml(tmp_path):
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"checkpoint placeholder")
    schema_path = tmp_path / "dataset.yaml"
    schema_path.write_text(
        "kpt_shape: [3, 3]\nkpt_names: [Nose, CenterSpine, TailBase]\n",
        encoding="utf-8",
    )

    resolution = resolve_model_keypoint_schema(_fake_model(), model_path, 3)

    assert resolution.names == ["Nose", "CenterSpine", "TailBase"]
    assert resolution.source == "dataset_yaml"
    assert resolution.source_path == str(schema_path.resolve())


def test_resolve_model_keypoint_schema_ignores_wrong_embedded_count_then_uses_yaml(tmp_path):
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"checkpoint placeholder")
    (tmp_path / "dataset.yaml").write_text(
        "kpt_names:\n  0: [Nose, Thorax, TailBase]\n",
        encoding="utf-8",
    )
    model = _fake_model(kpt_names=["Nose", "Left_Ear", "Right_Ear", "TailBase"])

    resolution = resolve_model_keypoint_schema(model, model_path, 3)

    assert resolution.names == ["Nose", "Thorax", "TailBase"]
    assert any("expected 3 names, found 4" in warning for warning in resolution.warnings)


def test_validate_keypoint_names_rejects_csv_token_collisions():
    clean, error = validate_keypoint_names(["Center spine", "Center_spine"], 2)

    assert clean == ["Center spine", "Center_spine"]
    assert "collide" in error
    assert safe_keypoint_token("Center spine") == "Center_spine"


def test_unresolved_schema_reports_generic_fallback_warning(tmp_path):
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"checkpoint placeholder")

    resolution = resolve_model_keypoint_schema(_fake_model(), model_path, 3)

    assert resolution.names == []
    assert resolution.source == "unresolved"
    assert any("Generic kp0" in warning for warning in resolution.warnings)
