"""Reproducibility bundle export/import helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from integra_pose.utils import model_registry, plugin_opt_in

_ASSISTED_POSE_MANIFEST_FILENAME = ".integrapose_assisted_pose_manifest.json"
_MAX_ARCHIVE_MEMBERS = 4096
_MAX_ARCHIVE_MEMBER_BYTES = 8 * 1024**3
_MAX_ARCHIVE_TOTAL_BYTES = 20 * 1024**3
_MAX_COMPRESSION_RATIO = 250.0
_MAX_MEMBER_PATH_LENGTH = 512
_MAX_METADATA_BYTES = 16 * 1024**2


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _safe_slug(value: str) -> str:
    keep = [ch if ch.isalnum() or ch in "._-" else "_" for ch in value]
    slug = "".join(keep).strip("._")
    return slug or "artifact"


def _redact_local_text(value: Any) -> str:
    text = _as_str(value)
    candidates = {
        _as_str(Path.home()),
        _as_str(os.environ.get("USERPROFILE")),
        _as_str(os.environ.get("HOME")),
    }
    for candidate in sorted((item for item in candidates if item), key=len, reverse=True):
        text = text.replace(candidate, "<HOME>")
        text = text.replace(candidate.replace("\\", "/"), "<HOME>")
    return text


def _redacted_registry_snapshot(records: Any) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        return []
    redacted: list[dict[str, Any]] = []
    for raw_record in records:
        if not isinstance(raw_record, dict):
            continue
        record = dict(raw_record)
        for key, value in list(record.items()):
            key_lower = str(key).lower()
            if "path" not in key_lower or not isinstance(value, str):
                continue
            record[key] = Path(value).name if value else ""
        redacted.append(record)
    return redacted


def _is_machine_local_reference(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if "://" in text:
        return True
    if text.startswith("~"):
        return True
    windows_path = PureWindowsPath(text)
    posix_path = PurePosixPath(text.replace("\\", "/"))
    return bool(windows_path.drive or windows_path.root or posix_path.is_absolute())


def _redact_config_references(
    payload: Any,
    *,
    prefix: tuple[str, ...] = (),
) -> tuple[Any, list[str]]:
    """Copy a config payload while removing non-portable local references."""

    redacted_fields: list[str] = []
    if isinstance(payload, dict):
        output: dict[Any, Any] = {}
        for key, value in payload.items():
            child, child_fields = _redact_config_references(
                value,
                prefix=(*prefix, str(key)),
            )
            output[key] = child
            redacted_fields.extend(child_fields)
        return output, redacted_fields
    if isinstance(payload, list):
        output_list = []
        for index, value in enumerate(payload):
            child, child_fields = _redact_config_references(
                value,
                prefix=(*prefix, str(index)),
            )
            output_list.append(child)
            redacted_fields.extend(child_fields)
        return output_list, redacted_fields
    if isinstance(payload, tuple):
        child, child_fields = _redact_config_references(list(payload), prefix=prefix)
        return child, child_fields
    if isinstance(payload, str) and _is_machine_local_reference(payload):
        field_name = ".".join(prefix) or "<root>"
        return "", [field_name]
    return payload, redacted_fields


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


def _json_load_bounded(path: Path) -> Any:
    try:
        size = int(path.stat().st_size)
    except OSError as exc:
        raise ValueError(f"Bundle metadata could not be read: {path.name}") from exc
    if size > _MAX_METADATA_BYTES:
        raise ValueError(
            f"Bundle metadata file is too large: {path.name} ({size} bytes)."
        )
    return _json_load(path)


def _validated_relative_path(raw_path: Any) -> Path:
    value = _as_str(raw_path)
    if not value or "\x00" in value:
        raise ValueError("Bundle contains an empty or invalid path.")
    if len(value) > _MAX_MEMBER_PATH_LENGTH:
        raise ValueError("Bundle member path exceeds the supported length limit.")

    windows_path = PureWindowsPath(value)
    normalized = value.replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    if (
        windows_path.drive
        or windows_path.root
        or posix_path.is_absolute()
        or normalized.startswith("//")
    ):
        raise ValueError(f"Bundle contains an absolute path: {value}")

    parts = posix_path.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Bundle contains a path traversal component: {value}")
    if any(":" in part for part in parts):
        raise ValueError(f"Bundle contains a drive-like path component: {value}")
    return Path(*parts)


def _resolve_bundle_path(bundle_root: Path, raw_path: Any) -> Path:
    rel_path = _validated_relative_path(raw_path)
    root = bundle_root.resolve()
    target = (root / rel_path).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Bundle path escapes the extraction directory: {raw_path}") from exc
    return target


def _zip_member_is_link_or_special(info: zipfile.ZipInfo) -> bool:
    unix_mode = (int(info.external_attr) >> 16) & 0xFFFF
    file_type = stat.S_IFMT(unix_mode)
    return file_type not in {0, stat.S_IFREG, stat.S_IFDIR}


def _extract_archive_safely(archive: zipfile.ZipFile, bundle_root: Path) -> None:
    members = archive.infolist()
    if len(members) > _MAX_ARCHIVE_MEMBERS:
        raise ValueError(
            f"Bundle has too many archive members ({len(members)}; maximum {_MAX_ARCHIVE_MEMBERS})."
        )

    planned: list[tuple[zipfile.ZipInfo, Path]] = []
    seen_paths: set[str] = set()
    total_size = 0
    for info in members:
        target = _resolve_bundle_path(bundle_root, info.filename)
        relative_key = target.relative_to(bundle_root.resolve()).as_posix().casefold()
        if relative_key in seen_paths:
            raise ValueError(f"Bundle contains duplicate member paths: {info.filename}")
        seen_paths.add(relative_key)
        if info.flag_bits & 0x1:
            raise ValueError(f"Encrypted bundle members are not supported: {info.filename}")
        if _zip_member_is_link_or_special(info):
            raise ValueError(f"Bundle contains a link or special file: {info.filename}")
        if info.file_size < 0 or info.file_size > _MAX_ARCHIVE_MEMBER_BYTES:
            raise ValueError(f"Bundle member is too large: {info.filename}")
        total_size += int(info.file_size)
        if total_size > _MAX_ARCHIVE_TOTAL_BYTES:
            raise ValueError("Bundle exceeds the total uncompressed size limit.")
        if info.file_size > 1024**2:
            ratio = float(info.file_size) / float(max(1, info.compress_size))
            if ratio > _MAX_COMPRESSION_RATIO:
                raise ValueError(f"Bundle member has a suspicious compression ratio: {info.filename}")
        planned.append((info, target))

    free_bytes = shutil.disk_usage(bundle_root).free
    if total_size > free_bytes:
        raise ValueError("Bundle cannot be extracted because there is insufficient free disk space.")

    for info, target in planned:
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        try:
            with archive.open(info, "r") as source, open(target, "xb") as destination:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > info.file_size or written > _MAX_ARCHIVE_MEMBER_BYTES:
                        raise ValueError(f"Bundle member exceeded its declared size: {info.filename}")
                    destination.write(chunk)
        except (RuntimeError, zipfile.BadZipFile, OSError) as exc:
            raise ValueError(f"Bundle member could not be extracted: {info.filename}") from exc
        if written != int(info.file_size):
            raise ValueError(f"Bundle member size did not match its archive metadata: {info.filename}")


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
        display_command = list(command)
        if "--name" in display_command:
            name_index = display_command.index("--name") + 1
            if name_index < len(display_command):
                display_command[name_index] = "<active-environment>"
        attempt: dict[str, Any] = {"command": " ".join(display_command)}
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
            attempt["stderr_tail"] = _redact_local_text((result.stderr or "").strip()[-500:])
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
            attempt["error"] = _redact_local_text(exc)
            attempts.append(attempt)
            break
        except Exception as exc:
            attempt["error"] = _redact_local_text(exc)
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
        "platform": _as_str(platform.platform()),
        "os_name": _as_str(os.name),
        "machine": _as_str(platform.machine()),
        "python_version": _as_str(sys.version.split(" ")[0]),
        "python_implementation": _as_str(platform.python_implementation()),
        "local_identifiers_redacted": "true",
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

        raw_config_payload = cfg._gather_config_as_dict()
        config_payload, redacted_config_fields = _redact_config_references(raw_config_payload)
        project_file_path = _as_str(getattr(cfg, "project_file_path", ""))
        dataset_root = _as_str(
            getattr(getattr(cfg, "setup", None), "dataset_root_yaml", "").get()
            if getattr(getattr(cfg, "setup", None), "dataset_root_yaml", None)
            else ""
        )
        project_context = {
            "exported_at": _now_iso(),
            "app_version": _as_str(getattr(app, "_current_app_version", lambda: "unknown")()),
            "project_file_name": Path(project_file_path).name if project_file_path else "",
            "dataset_name": Path(dataset_root).name if dataset_root else "",
            "local_paths_redacted": True,
        }
        plugin_snapshot = _collect_plugin_snapshot(app)
        registry_snapshot = _redacted_registry_snapshot(model_registry.list_records(app))
        environment_capture = _capture_conda_environment(bundle_root)

        copied_models: list[dict[str, Any]] = []
        existing_names: set[str] = set()
        for candidate in _gather_selected_model_candidates(app):
            src = Path(candidate["path"])
            item: dict[str, Any] = {
                "role": candidate["role"],
                "source_name": src.name,
                "exists": src.is_file(),
            }
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
                            "source_name": manifest_path.name,
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
            "local_config_references_redacted": sorted(set(redacted_config_fields)),
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
                "missing_source_names": [_as_str(item.get("source_name")) for item in missing_models],
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
        temp_fd, temp_name = tempfile.mkstemp(
            prefix=f".{bundle_zip_path.name}.",
            suffix=".tmp",
            dir=str(bundle_zip_path.parent),
        )
        os.close(temp_fd)
        temp_zip_path = Path(temp_name)
        try:
            with zipfile.ZipFile(temp_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(bundle_root.rglob("*")):
                    if not path.is_file():
                        continue
                    archive.write(path, arcname=path.relative_to(bundle_root).as_posix())
            os.replace(temp_zip_path, bundle_zip_path)
        finally:
            try:
                temp_zip_path.unlink(missing_ok=True)
            except OSError:
                pass

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
        raise ValueError("Bundle is missing bundle_manifest.json")
    payload = _json_load_bounded(manifest_path)
    if not isinstance(payload, dict):
        raise ValueError("Bundle manifest is malformed.")
    if payload.get("schema_version") != 1:
        raise ValueError("Bundle manifest schema version is unsupported.")
    files = payload.get("files", [])
    if not isinstance(files, list):
        raise ValueError("Bundle manifest is malformed.")
    if len(files) > _MAX_ARCHIVE_MEMBERS:
        raise ValueError("Bundle manifest contains too many file entries.")

    declared_paths: set[str] = set()
    declared_casefold: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("Bundle manifest contains a malformed file entry.")
        rel = _as_str(entry.get("path"))
        expected = _as_str(entry.get("sha256"))
        expected_size = entry.get("size_bytes")
        if (
            not rel
            or len(expected) != 64
            or any(ch not in "0123456789abcdefABCDEF" for ch in expected)
            or not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
        ):
            raise ValueError("Bundle manifest contains incomplete file metadata.")
        normalized_rel = _validated_relative_path(rel).as_posix()
        normalized_key = normalized_rel.casefold()
        if normalized_key in declared_casefold:
            raise ValueError(f"Bundle manifest contains a duplicate path: {rel}")
        declared_casefold.add(normalized_key)
        declared_paths.add(normalized_rel)
        target = _resolve_bundle_path(bundle_root, rel)
        if not target.is_file():
            raise ValueError(f"Bundle file is missing: {rel}")
        if int(target.stat().st_size) != expected_size:
            raise ValueError(f"Size mismatch for {rel}")
        actual = _sha256(target)
        if actual.lower() != expected.lower():
            raise ValueError(f"Checksum mismatch for {rel}")

    actual_paths = {
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual_paths != declared_paths:
        undeclared = sorted(actual_paths - declared_paths)
        missing = sorted(declared_paths - actual_paths)
        detail = undeclared[0] if undeclared else missing[0] if missing else "unknown"
        raise ValueError(f"Bundle manifest does not exactly describe the archive files: {detail}")


def _import_target_root(app: Any) -> Path:
    override = getattr(app, "_repro_import_root", None)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".integrapose" / "repro_imports"


def _load_optional_metadata_list(bundle_root: Path, file_name: str) -> list[dict[str, Any]]:
    path = bundle_root / file_name
    if not path.is_file():
        return []
    payload = _json_load_bounded(path)
    if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
        raise ValueError(f"Bundle metadata is malformed: {file_name}")
    return [dict(item) for item in payload]


def _validated_artifact_source(
    bundle_root: Path,
    raw_path: Any,
    *,
    required_prefix: tuple[str, ...],
) -> Path:
    relative_path = _validated_relative_path(raw_path)
    relative_parts = tuple(part.casefold() for part in relative_path.parts)
    expected_parts = tuple(part.casefold() for part in required_prefix)
    if relative_parts[: len(expected_parts)] != expected_parts:
        raise ValueError(
            f"Bundle artifact is outside its expected directory: {relative_path.as_posix()}"
        )
    source = _resolve_bundle_path(bundle_root, relative_path.as_posix())
    if not source.is_file():
        raise ValueError(f"Bundle artifact is missing: {relative_path.as_posix()}")
    return source


def _verify_entry_checksum(entry: dict[str, Any], source: Path) -> None:
    expected = _as_str(entry.get("sha256"))
    if (
        len(expected) != 64
        or any(ch not in "0123456789abcdefABCDEF" for ch in expected)
        or _sha256(source).lower() != expected.lower()
    ):
        raise ValueError(f"Bundle artifact checksum is invalid: {source.name}")


def _preflight_import_metadata(
    bundle_root: Path,
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], Path]], list[tuple[dict[str, Any], Path]]]:
    config_path = bundle_root / "project_config.json"
    if not config_path.is_file():
        raise ValueError("Bundle is missing project_config.json")
    raw_config_payload = _json_load_bounded(config_path)
    if not isinstance(raw_config_payload, dict):
        raise ValueError("Bundle project config is malformed.")
    config_payload, _redacted_fields = _redact_config_references(raw_config_payload)

    model_entries: list[tuple[dict[str, Any], Path]] = []
    for entry in _load_optional_metadata_list(bundle_root, "selected_models.json"):
        raw_bundle_path = _as_str(entry.get("bundle_path"))
        if not raw_bundle_path:
            continue
        role = _as_str(entry.get("role"))
        if role not in {"inference", "webcam", "export"}:
            raise ValueError(f"Bundle model role is unsupported: {role or '<empty>'}")
        source = _validated_artifact_source(
            bundle_root,
            raw_bundle_path,
            required_prefix=("artifacts", "models"),
        )
        _verify_entry_checksum(entry, source)
        model_entries.append((entry, source))

    project_entries: list[tuple[dict[str, Any], Path]] = []
    for entry in _load_optional_metadata_list(bundle_root, "project_artifacts.json"):
        raw_bundle_path = _as_str(entry.get("bundle_path"))
        if not raw_bundle_path:
            continue
        role = _as_str(entry.get("role"))
        if role != "assisted_pose_curation_manifest":
            raise ValueError(f"Bundle project artifact role is unsupported: {role or '<empty>'}")
        source = _validated_artifact_source(
            bundle_root,
            raw_bundle_path,
            required_prefix=("artifacts", "project"),
        )
        _verify_entry_checksum(entry, source)
        project_entries.append((entry, source))

    return config_payload, model_entries, project_entries


def _deduplicated_stage_path(directory: Path, file_name: str, reserved: set[str]) -> Path:
    safe_name = _safe_slug(file_name)
    candidate = safe_name
    index = 1
    while candidate.casefold() in reserved:
        candidate = f"{Path(safe_name).stem}_{index}{Path(safe_name).suffix}"
        index += 1
    reserved.add(candidate.casefold())
    return directory / candidate


def _set_imported_model_path(app: Any, role: str, path: Path) -> None:
    role_targets = {
        "inference": ("inference", "trained_model_path_infer"),
        "webcam": ("webcam", "webcam_model_path"),
        "export": ("training", "export_model_path"),
    }
    section_name, variable_name = role_targets[role]
    section = getattr(app.config, section_name, None)
    variable = getattr(section, variable_name, None) if section is not None else None
    if variable is None or not hasattr(variable, "set"):
        raise ValueError(f"Application config cannot restore the {role} model path.")
    variable.set(str(path))


def import_bundle(app: Any, bundle_zip_path: Path) -> dict[str, Any]:
    bundle_zip_path = Path(bundle_zip_path)
    if not bundle_zip_path.is_file():
        raise ValueError(f"Bundle file does not exist: {bundle_zip_path}")

    with tempfile.TemporaryDirectory(prefix="integrapose_bundle_import_") as tmp_dir:
        bundle_root = Path(tmp_dir) / "bundle"
        bundle_root.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(bundle_zip_path, "r") as archive:
                _extract_archive_safely(archive, bundle_root)
        except zipfile.BadZipFile as exc:
            raise ValueError("Bundle is not a valid ZIP archive.") from exc

        _verify_manifest(bundle_root)
        config_payload, model_entries, project_entries = _preflight_import_metadata(bundle_root)

        cfg = getattr(app, "config", None)
        if (
            cfg is None
            or not hasattr(cfg, "_apply_config_from_dict")
            or not hasattr(cfg, "_gather_config_as_dict")
        ):
            raise ValueError("Application config is unavailable for bundle import.")

        import_root = _import_target_root(app)
        import_root.mkdir(parents=True, exist_ok=True)
        stage_dir = Path(tempfile.mkdtemp(prefix=".bundle_import_", dir=str(import_root)))
        final_dir = import_root / f"{_safe_slug(bundle_zip_path.stem)}_{uuid.uuid4().hex[:8]}"
        staged_models: list[dict[str, Any]] = []
        staged_project_artifacts: list[Path] = []
        committed = False
        config_applied = False
        previous_config = cfg._gather_config_as_dict()
        if not isinstance(previous_config, dict):
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise ValueError("Current application config could not be snapshotted for transactional import.")

        try:
            model_dir = stage_dir / "models"
            project_dir = stage_dir / "project_artifacts"
            model_dir.mkdir(parents=True, exist_ok=True)
            project_dir.mkdir(parents=True, exist_ok=True)
            reserved_models: set[str] = set()
            for entry, source in model_entries:
                destination = _deduplicated_stage_path(model_dir, source.name, reserved_models)
                shutil.copy2(source, destination)
                staged_models.append(
                    {
                        "role": _as_str(entry.get("role")),
                        "relative_path": destination.relative_to(stage_dir),
                    }
                )

            reserved_artifacts: set[str] = set()
            for _entry, source in project_entries:
                destination = _deduplicated_stage_path(project_dir, source.name, reserved_artifacts)
                shutil.copy2(source, destination)
                staged_project_artifacts.append(destination.relative_to(stage_dir))

            config_applied = True
            cfg._apply_config_from_dict(config_payload)
            os.replace(stage_dir, final_dir)
            committed = True

            imported_models: list[dict[str, str]] = []
            for staged in staged_models:
                destination = final_dir / staged["relative_path"]
                _set_imported_model_path(app, staged["role"], destination)
                imported_models.append({"role": staged["role"], "path": str(destination)})
            imported_project_artifacts = [
                str(final_dir / relative_path) for relative_path in staged_project_artifacts
            ]
        except Exception:
            if config_applied:
                try:
                    cfg._apply_config_from_dict(previous_config)
                except Exception:
                    pass
            if committed:
                shutil.rmtree(final_dir, ignore_errors=True)
            else:
                shutil.rmtree(stage_dir, ignore_errors=True)
            raise

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
        "import_root": str(final_dir),
    }
