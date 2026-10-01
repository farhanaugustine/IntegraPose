"""Model-index contracts for gait landmarks; never infer anatomy from count."""
import csv
import ast
import json
from pathlib import Path

from integra_pose.utils.keypoint_schema import (
    coerce_keypoint_names, resolve_model_keypoint_schema, safe_keypoint_token, validate_keypoint_names,
)
from .profiles import HUMAN_KEYPOINTS


def token(name):
    return safe_keypoint_token(name).casefold()


def checked_names(names, count=None):
    names = coerce_keypoint_names(names)
    if not names:
        raise ValueError("No ordered landmark names were found. Import the training dataset YAML or a skeleton JSON.")
    names, error = validate_keypoint_names(names, len(names) if count is None else count)
    if error:
        raise ValueError(error)
    return names


def normalize_edges(edges, names):
    """Named edges or explicitly zero-based index edges; no base guessing."""
    indices = {token(name): index for index, name in enumerate(names)}
    result = []
    for edge in edges or []:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise ValueError("Each skeleton edge must contain two landmark names or zero-based indices.")
        if all(isinstance(value, str) for value in edge):
            try:
                a, b = [indices[token(value)] for value in edge]
            except KeyError as exc:
                raise ValueError(f"Skeleton edge references an unknown landmark: {edge}") from exc
        elif all(isinstance(value, int) and not isinstance(value, bool) for value in edge):
            a, b = edge
        else:
            raise ValueError("Use named edges or integer zero-based index_edges.")
        if min(a, b) < 0 or max(a, b) >= len(names) or a == b:
            raise ValueError(f"Invalid zero-based skeleton edge: {edge}")
        result.append([a, b])
    return result


def read_schema_file(path):
    path = Path(path)
    if path.suffix.lower() == ".py":
        # Read literal metadata without importing/executing supplied Python.
        data = {}
        wanted = {"KEYPOINT_ORDER", "KEYPOINT_NAMES", "KPT_NAMES", "kpt_names", "keypoint_names", "kpt_shape", "SKELETON_EDGES", "index_edges", "edges"}
        for node in ast.parse(path.read_text(encoding="utf-8-sig")).body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and target.id in wanted:
                        try:
                            data[target.id] = ast.literal_eval(node.value)
                        except (ValueError, TypeError):
                            pass
        data["keypoint_names"] = next((data[key] for key in ("KEYPOINT_ORDER", "KEYPOINT_NAMES", "KPT_NAMES", "kpt_names", "keypoint_names") if data.get(key)), [])
        data["index_edges"] = data.get("SKELETON_EDGES", data.get("index_edges", []))
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    else:
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("The schema must contain a YAML/JSON object.")
    raw = data.get("kpt_names") or data.get("keypoint_names") or data.get("keypoints")
    shape = data.get("kpt_shape")
    names = checked_names(raw, int(shape[0]) if shape else None)
    # Skeleton Editor exports both; named edges survive changes of index order.
    edges = normalize_edges(data.get("edges") or data.get("index_edges") or [], names)
    return {"keypoint_names": names, "keypoint_count": len(names), "edges": edges,
            "source": str(path.resolve()), "class_names": data.get("names") or {}}


def inspect_model(path):
    """Inspect a locally selected checkpoint on CPU; do not run inference/download."""
    path = Path(path)
    if not path.is_file() or path.suffix.lower() != ".pt":
        raise ValueError("Select an existing local YOLO pose .pt checkpoint.")
    from ultralytics import YOLO
    model = YOLO(str(path))
    model.to("cpu")
    return schema_from_model(model, path)


def schema_from_model(model, path):
    shape = getattr(getattr(model, "model", None), "kpt_shape", None)
    if getattr(model, "task", None) != "pose" or not shape or int(shape[0]) <= 0:
        raise ValueError("This model is not a pose model with keypoints.")
    count = int(shape[0])
    resolved = resolve_model_keypoint_schema(model, path, count)
    names = resolved.names
    source = resolved.source
    # Official COCO-trained models can have no embedded names. Use the COCO
    # contract only with a matching training-dataset marker and person class,
    # never just a filename or a 17-point output shape.
    ckpt = getattr(model, "ckpt", {}) or {}
    training = ckpt.get("train_args", {}) or {}
    dataset = str(training.get("data", "")).replace("\\", "/").split("/")[-1]
    classes = getattr(model, "names", {}) or {}
    if not names and count == 17 and dataset in ("coco-pose.yaml", "coco8-pose.yaml") and classes == {0: "person"}:
        names, source = list(HUMAN_KEYPOINTS), "COCO pose training metadata"
    return {"keypoint_names": names, "keypoint_count": count, "source": source,
            "model_path": str(Path(path).resolve()), "class_names": classes, "edges": []}


def validate_label_mapping(config, labels_dir):
    """Check available names against configured model-index order before reading TXT."""
    expected = checked_names(config["DATASET"]["KEYPOINT_ORDER"])
    configured = [token(name) for name in expected]
    schema = config.get("MODEL_SCHEMA", {})
    if schema.get("keypoint_count") and schema["keypoint_count"] != len(expected):
        raise ValueError("Selected model keypoint count differs from the configured skeleton.")
    model_names = schema.get("keypoint_names") or []
    if model_names and [token(name) for name in model_names] != configured:
        raise ValueError("Configured landmark order differs from the inspected model/schema. Re-import its schema.")
    labels = Path(labels_dir)
    sources = []
    sidecar = labels / "integrapose_pose_labels.schema.json"
    if sidecar.exists():
        payload = json.loads(sidecar.read_text(encoding="utf-8-sig"))
        if payload.get("keypoint_names"):
            sources.append(("pose label sidecar", payload["keypoint_names"]))
    csv_path = labels / "labels.csv"
    if csv_path.exists():
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
        names = [column[3:-4] for column in header if column.startswith("kp_") and column.endswith("_x_n")]
        if names:
            sources.append(("labels.csv", names))
    verified = False
    for source, names in sources:
        names = checked_names(names, len(expected))
        if all(name == f"kp{i}" for i, name in enumerate(names)):
            continue  # Generic indices do not identify anatomy.
        if [token(name) for name in names] != configured:
            raise ValueError(f"Landmark order mismatch with {source}. Import the matching model/schema; do not relabel coordinates by position.")
        verified = True
    if not verified and config["DATASET"].get("MAPPING_CONFIRMED") is not True:
        raise ValueError("Labels contain no verifiable landmark names. Check the model's exact index order, then enable 'I verified landmark order for these labels'.")
    return "label metadata" if verified else "user-confirmed model index order"
