"""Model registry helpers for training lineage and quick model selection."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from integra_pose.utils import plugin_opt_in

_GLOBAL_REGISTRY_PATH = Path.home() / ".integrapose" / "model_registry.json"
_PROJECT_REGISTRY_FILENAME = ".integrapose_model_registry.json"
_MAX_RECORDS = 1000


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _safe_resolve(path_value: str) -> str:
    if not path_value:
        return ""
    try:
        return str(Path(path_value).expanduser().resolve())
    except Exception:
        return path_value


def _safe_var_get(var_holder: Any) -> str:
    try:
        if hasattr(var_holder, "get"):
            return _as_str(var_holder.get())
    except Exception:
        pass
    return _as_str(var_holder)


def _get_project_root(app: Any) -> str:
    setup = getattr(getattr(app, "config", None), "setup", None)
    if setup is None:
        return ""
    return _safe_var_get(getattr(setup, "dataset_root_yaml", ""))


def _project_registry_path(project_root: str) -> Path | None:
    root = _as_str(project_root)
    if not root:
        return None
    path = Path(root)
    if not path.exists() or not path.is_dir():
        return None
    return path / _PROJECT_REGISTRY_FILENAME


def _load_records(path: Path) -> list[dict[str, Any]]:
    try:
        if not path.is_file():
            return []
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return []

    if isinstance(payload, dict):
        records = payload.get("records")
        if isinstance(records, list):
            return [r for r in records if isinstance(r, dict)]
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    return []


def _save_records(path: Path, records: list[dict[str, Any]]) -> None:
    payload = {
        "version": 1,
        "updated_at": _now_iso(),
        "records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _append_record(
    path: Path,
    record: dict[str, Any],
    *,
    dedupe_keys: list[str] | None = None,
) -> None:
    records = _load_records(path)
    if dedupe_keys:
        def _matches(existing: dict[str, Any]) -> bool:
            for key in dedupe_keys:
                if _as_str(existing.get(key)) != _as_str(record.get(key)):
                    return False
            return True
        records = [r for r in records if not _matches(r)]
    else:
        run_id = _as_str(record.get("run_id"))
        if run_id:
            records = [r for r in records if _as_str(r.get("run_id")) != run_id]
    records.insert(0, dict(record))
    if len(records) > _MAX_RECORDS:
        records = records[:_MAX_RECORDS]
    _save_records(path, records)


def _remove_records_by_match(path: Path, matcher) -> int:
    records = _load_records(path)
    if not records:
        return 0
    kept: list[dict[str, Any]] = []
    removed = 0
    for record in records:
        try:
            if matcher(record):
                removed += 1
                continue
        except Exception:
            pass
        kept.append(record)
    if removed > 0:
        _save_records(path, kept)
    return removed


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


def parse_results_metrics(results_csv_path: Path | None) -> dict[str, float]:
    if results_csv_path is None or not results_csv_path.is_file():
        return {}

    rows: list[dict[str, str]] = []
    try:
        with open(results_csv_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if isinstance(row, dict) and row:
                    rows.append(row)
    except Exception:
        return {}

    if not rows:
        return {}

    latest = rows[-1]
    parsed: dict[str, float] = {}
    for key, value in latest.items():
        key_str = _as_str(key)
        value_str = _as_str(value)
        if not key_str or not value_str:
            continue
        if not (
            key_str == "epoch"
            or key_str == "fitness"
            or key_str.startswith("metrics/")
            or key_str.startswith("val/")
            or key_str.startswith("train/")
        ):
            continue
        try:
            numeric = float(value_str)
        except Exception:
            continue
        if math.isfinite(numeric):
            parsed[key_str] = numeric
    return parsed


def summarize_core_metrics(metrics: dict[str, float]) -> list[tuple[str, float]]:
    ordered_keys = [
        ("metrics/precision(P)", "precision(P)"),
        ("metrics/recall(P)", "recall(P)"),
        ("metrics/mAP50(P)", "mAP50(P)"),
        ("metrics/mAP50-95(P)", "mAP50-95(P)"),
        ("metrics/precision(B)", "precision(B)"),
        ("metrics/recall(B)", "recall(B)"),
        ("metrics/mAP50(B)", "mAP50(B)"),
        ("metrics/mAP50-95(B)", "mAP50-95(B)"),
        ("fitness", "fitness"),
    ]
    out: list[tuple[str, float]] = []
    for key, label in ordered_keys:
        if key in metrics:
            out.append((label, metrics[key]))
    return out


def register_training_run(
    app: Any,
    *,
    model_path: Path,
    results_csv_path: Path | None = None,
    notes: str = "",
) -> dict[str, Any] | None:
    if model_path is None or not model_path.exists():
        return None

    cfg = getattr(app, "config", None)
    training = getattr(cfg, "training", None)
    setup = getattr(cfg, "setup", None)
    if training is None or setup is None:
        return None

    run_id = str(uuid4())
    model_path_resolved = _safe_resolve(str(model_path))
    project_root = _get_project_root(app)
    dataset_yaml = _safe_var_get(getattr(training, "dataset_yaml_path_train", ""))
    model_save_dir = _safe_var_get(getattr(training, "model_save_dir", ""))
    run_name = _safe_var_get(getattr(training, "run_name_var", ""))

    metrics = parse_results_metrics(results_csv_path)
    base_record: dict[str, Any] = {
        "run_id": run_id,
        "created_at": _now_iso(),
        "source": "training",
        "task": "pose",
        "model_path": model_path_resolved,
        "model_name": Path(model_path_resolved).name if model_path_resolved else model_path.name,
        "run_name": run_name,
        "dataset_yaml_path": dataset_yaml,
        "model_save_dir": model_save_dir,
        "project_root": _safe_resolve(project_root) if project_root else "",
        "metrics": metrics,
        "plugin_versions": _collect_plugin_snapshot(app),
        "notes": _as_str(notes),
    }

    _append_record(_GLOBAL_REGISTRY_PATH, {**base_record, "registry_scope": "global"})
    project_registry = _project_registry_path(project_root)
    if project_registry is not None:
        _append_record(project_registry, {**base_record, "registry_scope": "project"})
    return base_record


def register_model_path(
    app: Any,
    *,
    model_path: Path,
    source: str = "manual",
    notes: str = "",
) -> dict[str, Any] | None:
    if model_path is None or not model_path.exists() or not model_path.is_file():
        return None

    cfg = getattr(app, "config", None)
    training = getattr(cfg, "training", None)
    if training is None:
        return None

    run_id = str(uuid4())
    model_path_resolved = _safe_resolve(str(model_path))
    project_root = _get_project_root(app)
    dataset_yaml = _safe_var_get(getattr(training, "dataset_yaml_path_train", ""))
    model_save_dir = _safe_var_get(getattr(training, "model_save_dir", ""))

    run_name = ""
    try:
        parent = model_path.parent
        if parent.name.lower() == "weights" and parent.parent:
            run_name = parent.parent.name
        else:
            run_name = model_path.stem
    except Exception:
        run_name = model_path.stem

    results_csv = None
    try:
        if model_path.parent.name.lower() == "weights":
            candidate = model_path.parent.parent / "results.csv"
            if candidate.is_file():
                results_csv = candidate
    except Exception:
        results_csv = None

    metrics = parse_results_metrics(results_csv)
    base_record: dict[str, Any] = {
        "run_id": run_id,
        "created_at": _now_iso(),
        "source": _as_str(source) or "manual",
        "task": "pose",
        "model_path": model_path_resolved,
        "model_name": Path(model_path_resolved).name if model_path_resolved else model_path.name,
        "run_name": run_name,
        "dataset_yaml_path": dataset_yaml,
        "model_save_dir": model_save_dir,
        "project_root": _safe_resolve(project_root) if project_root else "",
        "metrics": metrics,
        "plugin_versions": _collect_plugin_snapshot(app),
        "notes": _as_str(notes),
    }

    _append_record(
        _GLOBAL_REGISTRY_PATH,
        {**base_record, "registry_scope": "global"},
        dedupe_keys=["source", "model_path", "registry_scope"],
    )
    project_registry = _project_registry_path(project_root)
    if project_registry is not None:
        _append_record(
            project_registry,
            {**base_record, "registry_scope": "project"},
            dedupe_keys=["source", "model_path", "registry_scope"],
        )
    return base_record


def list_records(app: Any) -> list[dict[str, Any]]:
    project_root = _get_project_root(app)
    all_records: list[dict[str, Any]] = []

    project_registry = _project_registry_path(project_root)
    if project_registry is not None:
        for record in _load_records(project_registry):
            merged = dict(record)
            merged["registry_scope"] = "project"
            merged["registry_path"] = str(project_registry)
            all_records.append(merged)

    for record in _load_records(_GLOBAL_REGISTRY_PATH):
        merged = dict(record)
        merged["registry_scope"] = "global"
        merged["registry_path"] = str(_GLOBAL_REGISTRY_PATH)
        all_records.append(merged)

    all_records.sort(key=lambda r: _as_str(r.get("created_at")), reverse=True)
    for record in all_records:
        model_path = _as_str(record.get("model_path"))
        record["model_exists"] = bool(model_path and Path(model_path).is_file())
    return all_records


def format_record_label(record: dict[str, Any]) -> str:
    scope = _as_str(record.get("registry_scope")).lower() or "global"
    scope_token = "P" if scope == "project" else "G"
    raw_ts = _as_str(record.get("created_at"))
    try:
        timestamp = datetime.fromisoformat(raw_ts.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")
    except Exception:
        timestamp = raw_ts or "unknown-time"

    model_path = _as_str(record.get("model_path"))
    model_name = _as_str(record.get("model_name")) or (Path(model_path).name if model_path else "unknown-model")
    run_id = _as_str(record.get("run_id"))[:8] or "unknown"

    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    core = summarize_core_metrics(metrics) if isinstance(metrics, dict) else []
    metric_text = f"{core[0][0]}={core[0][1]:.4f}" if core else "no-metrics"
    missing_text = " | missing-file" if not bool(record.get("model_exists")) else ""

    return f"[{scope_token}] {timestamp} | {model_name} | {metric_text} | {run_id}{missing_text}"


def remove_record(app: Any, record: dict[str, Any], *, remove_linked_scopes: bool = True) -> int:
    """Remove a registry entry. Optionally removes linked copies in other scopes by run_id."""
    if not isinstance(record, dict):
        return 0

    run_id = _as_str(record.get("run_id"))
    model_path = _as_str(record.get("model_path"))
    created_at = _as_str(record.get("created_at"))
    source = _as_str(record.get("source"))
    scoped_registry_path = _as_str(record.get("registry_path"))

    def _matches(candidate: dict[str, Any]) -> bool:
        if run_id and _as_str(candidate.get("run_id")) == run_id:
            return True
        # Fallback for malformed/legacy records missing run_id.
        return (
            _as_str(candidate.get("model_path")) == model_path
            and _as_str(candidate.get("created_at")) == created_at
            and _as_str(candidate.get("source")) == source
        )

    targets: list[Path] = []
    if remove_linked_scopes:
        targets.append(_GLOBAL_REGISTRY_PATH)
        project_registry = _project_registry_path(_get_project_root(app))
        if project_registry is not None:
            targets.append(project_registry)
    elif scoped_registry_path:
        targets.append(Path(scoped_registry_path))
    else:
        targets.append(_GLOBAL_REGISTRY_PATH)

    seen: set[str] = set()
    removed_total = 0
    for target in targets:
        resolved = str(target.expanduser().resolve()) if target else ""
        if resolved and resolved in seen:
            continue
        if resolved:
            seen.add(resolved)
        removed_total += _remove_records_by_match(target, _matches)
    return removed_total
