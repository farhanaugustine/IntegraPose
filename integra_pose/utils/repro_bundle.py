"""Reproducibility bundle export/import helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from integra_pose.utils import model_registry, plugin_opt_in

_ASSISTED_POSE_MANIFEST_FILENAME = ".integrapose_assisted_pose_manifest.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _safe_slug(value: str) -> str:
    keep = [ch if ch.isalnum() or ch in "._-" else "_" for ch in value]
    slug = "".join(keep).strip("._")
    return slug or "artifact"


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _json_dump(path: Path, payload: Any) -> None:
    # Delegate to the shared safe-IO helper so every bundle-component write is
    # atomic (temp file + os.replace). Prevents half-written manifests when a
    # bundle export is interrupted partway through its ~10 sequential writes.
    from integra_pose.utils.safe_io import safe_write_json

    safe_write_json(path, payload, indent=2)


def _json_load(path: Path) -> Any:
    from integra_pose.utils.safe_io import safe_read_json

    return safe_read_json(path)


def _collect_plugin_snapshot(app: Any) -> list[dict[str, str]]:
    controller = getattr(app, "plugin_controller", None)
    if controller is None or not hasattr(controller, "discover_plugins"):
        return []

    out: list[dict[str, str]] = []
    try:
        discovered = controller.discover_plugins()
    except Exception:
        return []

    for plugin in discovered:
        try:
            enabled = plugin_opt_in.is_plugin_enabled(plugin.path)
        except Exception:
            enabled = False
        if not enabled:
            continue
        out.append(
            {
                "id": _as_str(getattr(plugin, "identifier", "")),
                "name": _as_str(getattr(getattr(plugin, "metadata", None), "name", "")),
                "version": _as_str(getattr(getattr(plugin, "metadata", None), "version", "")),
                "source": _as_str(getattr(plugin, "source", "")),
            }
        )
    out.sort(key=lambda item: item.get("id", ""))
    return out


def _gather_selected_model_candidates(app: Any) -> list[dict[str, str]]:
    cfg = getattr(app, "config", None)
    if cfg is None:
        return []
    candidates = [
        ("inference", _as_str(getattr(getattr(cfg, "inference", None), "trained_model_path_infer", "").get() if getattr(getattr(cfg, "inference", None), "trained_model_path_infer", None) else "")),
        ("webcam", _as_str(getattr(getattr(cfg, "webcam", None), "webcam_model_path", "").get() if getattr(getattr(cfg, "webcam", None), "webcam_model_path", None) else "")),
        ("export", _as_str(getattr(getattr(cfg, "training", None), "export_model_path", "").get() if getattr(getattr(cfg, "training", None), "export_model_path", None) else "")),
    ]
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for role, raw_path in candidates:
        if not raw_path:
            continue
        try:
            resolved = str(Path(raw_path).expanduser().resolve())
        except Exception:
            resolved = raw_path
        key = f"{role}:{resolved}"
        if key in seen:
            continue
        seen.add(key)
        out.append({"role": role, "path": resolved})
    return out


def _bundle_file_manifest(bundle_root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(bundle_root.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "bundle_manifest.json":
            continue
        rel = path.relative_to(bundle_root).as_posix()
        entries.append(
            {
                "path": rel,
                "sha256": _sha256(path),
                "size_bytes": int(path.stat().st_size),
            }
        )
    return entries


def _stable_payload_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _capture_conda_environment(bundle_root: Path) -> dict[str, Any]:
    env_path = bundle_root / "environment.yml"
    command_variants: list[list[str]] = [["conda", "env", "export", "--from-history"]]
    conda_env = _as_str(os.environ.get("CONDA_DEFAULT_ENV"))
    if conda_env:
        command_variants.append(["conda", "env", "export", "--from-history", "--name", conda_env])

    attempts: list[dict[str, Any]] = []
    for command in command_variants:
        attempt: dict[str, Any] = {"command": " ".join(command)}
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=60,
            )
            stdout_text = result.stdout or ""
            attempt["return_code"] = int(result.returncode)
            attempt["stderr_tail"] = _as_str((result.stderr or "").strip()[-500:])
            attempt["stdout_length"] = len(stdout_text)
            attempts.append(attempt)
            if result.returncode == 0 and stdout_text.strip():
                if not stdout_text.endswith("\n"):
                    stdout_text = f"{stdout_text}\n"
                env_path.write_text(stdout_text, encoding="utf-8")
                return {
                    "success": True,
                    "bundle_path": "environment.yml",
                    "command_used": attempt["command"],
                    "attempts": attempts,
                }
        except FileNotFoundError as exc:
            attempt["error"] = _as_str(exc)
            attempts.append(attempt)
            break
        except Exception as exc:
            attempt["error"] = _as_str(exc)
            attempts.append(attempt)

    failure_reason = "conda env export --from-history failed."
    if attempts:
        last = attempts[-1]
        failure_reason = _as_str(last.get("error") or last.get("stderr_tail")) or failure_reason

    fallback_yaml = "\n".join(
        [
            "# Conda history export was unavailable during bundle creation.",
            "# Traceability details are captured in traceability_manifest.json.",
            "name: unresolved_environment",
            "channels: []",
            "dependencies: []",
            "integra_pose_bundle:",
            f"  exported_at_utc: {json.dumps(_now_iso())}",
            f"  command_expected: {json.dumps('conda env export --from-history')}",
            f"  status: {json.dumps(failure_reason)}",
            "",
        ]
    )
    env_path.write_text(fallback_yaml, encoding="utf-8")
    return {
        "success": False,
        "bundle_path": "environment.yml",
        "command_used": "conda env export --from-history",
        "attempts": attempts,
        "reason": failure_reason,
    }


def _runtime_traceability_context() -> dict[str, str]:
    return {
        "hostname": _as_str(socket.gethostname()),
        "platform": _as_str(platform.platform()),
        "os_name": _as_str(os.name),
        "machine": _as_str(platform.machine()),
        "processor": _as_str(platform.processor()),
        "python_version": _as_str(sys.version.split(" ")[0]),
        "python_implementation": _as_str(platform.python_implementation()),
        "conda_default_env": _as_str(os.environ.get("CONDA_DEFAULT_ENV")),
    }


def export_bundle(app: Any, bundle_zip_path: Path) -> dict[str, Any]:
    cfg = getattr(app, "config", None)
    if cfg is None or not hasattr(cfg, "_gather_config_as_dict"):
        raise ValueError("Application config is unavailable for bundle export.")

    with tempfile.TemporaryDirectory(prefix="integrapose_bundle_") as tmp_dir:
        bundle_root = Path(tmp_dir) / "bundle"
        bundle_root.mkdir(parents=True, exist_ok=True)
        model_artifact_dir = bundle_root / "artifacts" / "models"
        model_artifact_dir.mkdir(parents=True, exist_ok=True)

        config_payload = cfg._gather_config_as_dict()
        project_context = {
            "exported_at": _now_iso(),
            "app_version": _as_str(getattr(app, "_current_app_version", lambda: "unknown")()),
            "project_file_path": _as_str(getattr(cfg, "project_file_path", "")),
            "dataset_root": _as_str(getattr(getattr(cfg, "setup", None), "dataset_root_yaml", "").get() if getattr(getattr(cfg, "setup", None), "dataset_root_yaml", None) else ""),
        }
        plugin_snapshot = _collect_plugin_snapshot(app)
        registry_snapshot = model_registry.list_records(app)
        environment_capture = _capture_conda_environment(bundle_root)

        copied_models: list[dict[str, Any]] = []
        existing_names: set[str] = set()
        for candidate in _gather_selected_model_candidates(app):
            src = Path(candidate["path"])
            item: dict[str, Any] = {"role": candidate["role"], "original_path": candidate["path"], "exists": src.is_file()}
            if src.is_file():
                base_name = _safe_slug(src.name)
                target_name = f"{candidate['role']}_{base_name}"
                stem, suffix = Path(target_name).stem, Path(target_name).suffix
                dedupe_index = 1
                while target_name in existing_names:
                    target_name = f"{stem}_{dedupe_index}{suffix}"
                    dedupe_index += 1
                existing_names.add(target_name)
                rel_target = Path("artifacts") / "models" / target_name
                dst = bundle_root / rel_target
                shutil.copy2(src, dst)
                item["bundle_path"] = rel_target.as_posix()
                item["sha256"] = _sha256(dst)
            copied_models.append(item)

        copied_project_artifacts: list[dict[str, Any]] = []
        dataset_root = _as_str(project_context.get("dataset_root"))
        if dataset_root:
            try:
                project_root_path = Path(dataset_root).expanduser()
                manifest_path = project_root_path / _ASSISTED_POSE_MANIFEST_FILENAME
                if manifest_path.is_file():
                    rel_target = Path("artifacts") / "project" / manifest_path.name
                    dst = bundle_root / rel_target
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(manifest_path, dst)
                    copied_project_artifacts.append(
                        {
                            "role": "assisted_pose_curation_manifest",
                            "original_path": str(manifest_path),
                            "bundle_path": rel_target.as_posix(),
                            "sha256": _sha256(dst),
                        }
                    )
            except Exception:
                pass

        _json_dump(bundle_root / "project_context.json", project_context)
        _json_dump(bundle_root / "project_config.json", config_payload)
        _json_dump(bundle_root / "plugin_snapshot.json", plugin_snapshot)
        _json_dump(bundle_root / "selected_models.json", copied_models)
        _json_dump(bundle_root / "project_artifacts.json", copied_project_artifacts)
        _json_dump(bundle_root / "model_registry_snapshot.json", registry_snapshot)
        _json_dump(bundle_root / "environment_capture.json", environment_capture)

        included_models = [item for item in copied_models if item.get("bundle_path")]
        missing_models = [item for item in copied_models if not item.get("bundle_path")]
        traceability_manifest = {
            "schema_version": 1,
            "bundle_id": str(uuid.uuid4()),
            "generated_at": _now_iso(),
            "intent": [
                "reproducibility",
                "manuscript_traceability",
                "regulatory_traceability",
                "audit_traceability",
            ],
            "project_context": project_context,
            "runtime_context": _runtime_traceability_context(),
            "config_sha256": _stable_payload_sha256(config_payload),
            "environment_export": {
                "success": bool(environment_capture.get("success")),
                "bundle_path": _as_str(environment_capture.get("bundle_path")),
                "command_used": _as_str(environment_capture.get("command_used")),
                "attempts": environment_capture.get("attempts", []),
            },
            "plugins_enabled_count": len(plugin_snapshot),
            "model_registry_records_count": len(registry_snapshot) if isinstance(registry_snapshot, list) else 0,
            "project_artifacts_count": len(copied_project_artifacts),
            "model_artifacts": {
                "included_count": len(included_models),
                "missing_count": len(missing_models),
                "missing_original_paths": [_as_str(item.get("original_path")) for item in missing_models],
            },
        }
        _json_dump(bundle_root / "traceability_manifest.json", traceability_manifest)

        files_manifest = _bundle_file_manifest(bundle_root)
        manifest = {
            "schema_version": 1,
            "exported_at": _now_iso(),
            "app_version": project_context.get("app_version"),
            "files": files_manifest,
        }
        _json_dump(bundle_root / "bundle_manifest.json", manifest)

        bundle_zip_path = Path(bundle_zip_path)
        bundle_zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(bundle_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(bundle_root.rglob("*")):
                if not path.is_file():
                    continue
                archive.write(path, arcname=path.relative_to(bundle_root).as_posix())

    return {
        "bundle_path": str(bundle_zip_path),
        "models_included": len([m for m in copied_models if m.get("bundle_path")]),
        "model_entries": copied_models,
        "environment_exported": bool(environment_capture.get("success")),
        "environment_file": _as_str(environment_capture.get("bundle_path")),
    }


def _verify_manifest(bundle_root: Path) -> None:
    manifest_path = bundle_root / "bundle_manifest.json"
    if not manifest_path.is_file():
        return
    payload = _json_load(manifest_path)
    if not isinstance(payload, dict):
        raise ValueError("Bundle manifest is malformed.")
    files = payload.get("files", [])
    if not isinstance(files, list):
        raise ValueError("Bundle manifest is malformed.")
    for entry in files:
        if not isinstance(entry, dict):
            continue
        rel = _as_str(entry.get("path"))
        expected = _as_str(entry.get("sha256"))
        if not rel or not expected:
            continue
        target = bundle_root / rel
        if not target.is_file():
            raise ValueError(f"Bundle file is missing: {rel}")
        actual = _sha256(target)
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {rel}")


def _import_model_target_root(app: Any) -> Path:
    cfg = getattr(app, "config", None)
    setup = getattr(cfg, "setup", None) if cfg is not None else None
    root_var = getattr(setup, "dataset_root_yaml", None) if setup is not None else None
    root_value = _as_str(root_var.get()) if root_var is not None and hasattr(root_var, "get") else ""
    if root_value:
        try:
            root_path = Path(root_value).expanduser()
            root_path.mkdir(parents=True, exist_ok=True)
            return root_path / "models" / "repro_imported"
        except Exception:
            pass
    return Path.home() / ".integrapose" / "repro_imported_models"


def import_bundle(app: Any, bundle_zip_path: Path) -> dict[str, Any]:
    bundle_zip_path = Path(bundle_zip_path)
    if not bundle_zip_path.is_file():
        raise ValueError(f"Bundle file does not exist: {bundle_zip_path}")

    with tempfile.TemporaryDirectory(prefix="integrapose_bundle_import_") as tmp_dir:
        bundle_root = Path(tmp_dir) / "bundle"
        bundle_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(bundle_zip_path, "r") as archive:
            archive.extractall(bundle_root)

        _verify_manifest(bundle_root)

        config_path = bundle_root / "project_config.json"
        if not config_path.is_file():
            raise ValueError("Bundle is missing project_config.json")
        config_payload = _json_load(config_path)
        if not isinstance(config_payload, dict):
            raise ValueError("Bundle project config is malformed.")

        cfg = getattr(app, "config", None)
        if cfg is None or not hasattr(cfg, "_apply_config_from_dict"):
            raise ValueError("Application config is unavailable for bundle import.")
        cfg._apply_config_from_dict(config_payload)

        imported_project_artifacts: list[str] = []
        project_artifacts_path = bundle_root / "project_artifacts.json"
        if project_artifacts_path.is_file():
            try:
                artifact_payload = _json_load(project_artifacts_path)
            except Exception:
                artifact_payload = []
            if isinstance(artifact_payload, list):
                dataset_root = _as_str(
                    getattr(getattr(cfg, "setup", None), "dataset_root_yaml", "").get()
                    if getattr(getattr(cfg, "setup", None), "dataset_root_yaml", None)
                    else ""
                )
                project_root = Path(dataset_root).expanduser() if dataset_root else None
                for entry in artifact_payload:
                    if not isinstance(entry, dict):
                        continue
                    rel_bundle_path = _as_str(entry.get("bundle_path"))
                    role = _as_str(entry.get("role"))
                    if role != "assisted_pose_curation_manifest" or not rel_bundle_path or project_root is None:
                        continue
                    src = bundle_root / rel_bundle_path
                    if not src.is_file():
                        continue
                    try:
                        project_root.mkdir(parents=True, exist_ok=True)
                        dst = project_root / _ASSISTED_POSE_MANIFEST_FILENAME
                        shutil.copy2(src, dst)
                        imported_project_artifacts.append(str(dst))
                    except Exception:
                        pass

        selected_models_payload = []
        selected_models_path = bundle_root / "selected_models.json"
        if selected_models_path.is_file():
            loaded = _json_load(selected_models_path)
            if isinstance(loaded, list):
                selected_models_payload = [item for item in loaded if isinstance(item, dict)]

        imported_models: list[dict[str, str]] = []
        target_root = _import_model_target_root(app)
        target_root.mkdir(parents=True, exist_ok=True)

        for entry in selected_models_payload:
            rel_bundle_path = _as_str(entry.get("bundle_path"))
            role = _as_str(entry.get("role"))
            if not rel_bundle_path:
                continue
            src = bundle_root / rel_bundle_path
            if not src.is_file():
                continue

            base_name = _safe_slug(src.name)
            dst = target_root / base_name
            dedupe_index = 1
            while dst.exists():
                dst = target_root / f"{Path(base_name).stem}_{dedupe_index}{Path(base_name).suffix}"
                dedupe_index += 1
            shutil.copy2(src, dst)
            imported_models.append({"role": role, "path": str(dst)})

            if role == "inference":
                app.config.inference.trained_model_path_infer.set(str(dst))
            elif role == "webcam":
                app.config.webcam.webcam_model_path.set(str(dst))
            elif role == "export":
                app.config.training.export_model_path.set(str(dst))

        refresh_registry = getattr(app, "_refresh_model_registry_views", None)
        if callable(refresh_registry):
            try:
                refresh_registry()
            except Exception:
                pass

    return {
        "bundle_path": str(bundle_zip_path),
        "imported_models": imported_models,
        "imported_model_count": len(imported_models),
        "imported_project_artifacts": imported_project_artifacts,
    }
