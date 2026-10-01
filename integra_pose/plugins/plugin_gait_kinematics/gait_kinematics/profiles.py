"""Portable gait presets and validation shared by the GUI and CLI."""
from copy import deepcopy
import math

from . import config as defaults

PRESET_NAMES = ("Mouse (12 landmarks)", "Human (COCO 17)")
HUMAN_KEYPOINTS = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", "Left Knee",
    "Right Knee", "Left Ankle", "Right Ankle",
]


def preset_config(name=PRESET_NAMES[0]):
    if name not in PRESET_NAMES:
        raise ValueError(f"Unknown gait preset: {name}")
    human = name == PRESET_NAMES[1]
    paws = ["Left Ankle", "Right Ankle"] if human else list(defaults.GAIT_PAWS)
    return {
        "PROFILE_NAME": name,
        "SUBJECT_TYPE": "human" if human else "animal",
        "DIRECTORIES": {"SOURCE_DATA_DIR": "", "BASE_RESULTS_DIR": ""},
        "DATASET": {
            "KEYPOINT_ORDER": list(HUMAN_KEYPOINTS if human else defaults.KEYPOINT_ORDER),
            "MAPPING_CONFIRMED": False,
            "SKELETON_EDGES": ([[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9],
                                [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13],
                                [13, 15], [12, 14], [14, 16]] if human else
                               [[0, 3], [1, 3], [2, 3], [3, 4], [3, 5], [3, 6], [6, 7],
                                [6, 8], [6, 9], [9, 10], [10, 11]]),
            # A human detector identifies a person, not their behavior.
            "BEHAVIOR_CLASSES": {0: "Person"} if human else dict(defaults.BEHAVIOR_CLASSES),
        },
        "POSE_METRICS": {
            "ELONGATION_CONNECTION": [] if human else list(defaults.ELONGATION_CONNECTION),
            "BODY_ANGLE_CONNECTION": [] if human else list(defaults.BODY_ANGLE_CONNECTION),
        },
        "GAIT_ANALYSIS": {
            "GAIT_DETECTION_METHOD": "Original", "GAIT_PAWS": paws,
            "PAW_ORDER_HILDEBRAND": list(paws),
            "STRIDE_REFERENCE_PAW": "Left Ankle" if human else "Left Rear Paw",
            "OPPOSING_PAW": "Right Ankle" if human else "Right Rear Paw",
            "PAW_SPEED_THRESHOLD_PX_PER_FRAME": 5.0,
            "BODY_SPEED_THRESHOLD_PX_PER_FRAME": 0.0,
            "MEASURES": ["stride_timing", "body_motion"] if human else ["stride_timing", "body_motion", "spatial_proxies"],
        },
        "GENERAL_PARAMS": {
            "DETECTION_CONF_THRESHOLD": 0.25, "KEYPOINT_CONF_THRESHOLD": 0.25,
            "MIN_BOUT_DURATION_FRAMES": 15, "TARGET_TRACK_ID": None,
            "START_FRAME": None, "END_FRAME": None,
        },
        "REVIEW": {"EXPORT_CLIPS": True, "MAX_CLIPS": 12, "PADDING_SECONDS": 0.5,
                   "EXPORT_COORDINATION": False},
        "ADVANCED_PARAMS": {"CCM_TARGET_BEHAVIOR": "Person" if human else "Grooming"},
    }


def opposing_paw(gait):
    """Explicit mapping wins; recognize exact legacy mouse names only."""
    if "OPPOSING_PAW" in gait:
        return gait["OPPOSING_PAW"] or None
    legacy = {"Left Rear Paw": "Right Rear Paw", "Right Rear Paw": "Left Rear Paw",
              "Left Front Paw": "Right Front Paw", "Right Front Paw": "Left Front Paw"}
    candidate = legacy.get(gait.get("STRIDE_REFERENCE_PAW"))
    return candidate if candidate in gait.get("GAIT_PAWS", []) else None


def advanced_unavailable(config):
    """Explicit scope of the retained legacy animal-behavior workflows."""
    steps = ('Run Advanced Behavioral Analysis (UMAP)', 'Run Decision Dynamics Analysis',
             'Run Convergent Cross-Mapping (CCM)')
    if config.get('SUBJECT_TYPE') == 'human':
        return {step: 'This legacy animal-behavior workflow is not implemented for the human preset. Use individual gait review and gait group comparison.' for step in steps}
    unavailable = {}
    names = set(config['DATASET']['KEYPOINT_ORDER'])
    if not {'Center Spine', 'Nose', 'Base of Neck'}.issubset(names):
        unavailable[steps[0]] = 'Animal UMAP requires Center Spine, Nose and Base of Neck landmarks.'
    behaviors = set(config['DATASET']['BEHAVIOR_CLASSES'].values())
    if 'Walking' not in behaviors or not behaviors.intersection({'Grooming','Wall-Rearing'}):
        unavailable[steps[1]] = 'Decision dynamics requires Walking and Grooming or Wall-Rearing behavior labels.'
    if not config.get('POSE_METRICS',{}).get('BODY_ANGLE_CONNECTION') and not {'Left Front Paw','Right Rear Paw'}.issubset(names):
        unavailable[steps[2]] = 'Legacy CCM requires a body-heading pair or Left Front Paw and Right Rear Paw.'
    return unavailable


def validate_config(config):
    """Return a normalized copy; reject mistakes before loading a recording."""
    result = deepcopy(config)
    subject = result.setdefault("SUBJECT_TYPE", {PRESET_NAMES[0]: "animal", PRESET_NAMES[1]: "human"}.get(result.get("PROFILE_NAME"), "custom"))
    if subject not in ("human", "animal", "custom"):
        raise ValueError("SUBJECT_TYPE must be human, animal or custom.")
    dataset = result.setdefault("DATASET", {})
    points = dataset.get("KEYPOINT_ORDER", [])
    if not isinstance(points, list) or not points or any(not isinstance(p, str) or not p.strip() for p in points):
        raise ValueError("Keypoint order must be a nonempty list of landmark names in model order.")
    if len(set(points)) != len(points):
        raise ValueError("Keypoint names must be unique.")
    from .model_schema import checked_names, normalize_edges
    checked_names(points)
    dataset["SKELETON_EDGES"] = normalize_edges(dataset.get("SKELETON_EDGES", []), points)
    behaviors = dataset.get("BEHAVIOR_CLASSES", {})
    if not behaviors or any(not str(v).strip() for v in behaviors.values()):
        raise ValueError("Provide class IDs and class names; human COCO class 0 is Person.")
    try:
        dataset["BEHAVIOR_CLASSES"] = {int(k): str(v) for k, v in behaviors.items()}
    except (ValueError, TypeError) as exc:
        raise ValueError("Class IDs must be integers.") from exc
    gait = result.setdefault("GAIT_ANALYSIS", {})
    paws = gait.get("GAIT_PAWS", [])
    if not paws or len(set(paws)) != len(paws) or any(p not in points for p in paws):
        raise ValueError("Gait landmarks must be unique names from the keypoint order.")
    if gait.get("STRIDE_REFERENCE_PAW") not in paws:
        raise ValueError("Stride reference landmark must be one of the gait landmarks.")
    opposite = opposing_paw(gait)
    if opposite and (opposite not in paws or opposite == gait["STRIDE_REFERENCE_PAW"]):
        raise ValueError("Opposite landmark must be a different gait landmark, or blank.")
    gait["OPPOSING_PAW"] = opposite or ""
    order = gait.setdefault("PAW_ORDER_HILDEBRAND", list(paws))
    if len(order) != len(paws) or set(order) != set(paws):
        raise ValueError("Limb display order must contain every gait landmark exactly once.")
    if gait.get("GAIT_DETECTION_METHOD", "Original") not in ("Original", "Peak-Based (Advanced)"):
        raise ValueError("Select Original or Peak-Based (Advanced) gait detection.")
    gait.setdefault("GAIT_DETECTION_METHOD", "Original")
    measures = gait.setdefault("MEASURES", ["stride_timing", "body_motion", "spatial_proxies"])
    if not measures or any(m not in {"stride_timing", "body_motion", "spatial_proxies"} for m in measures):
        raise ValueError("Choose at least one measurement: stride timing, body motion, or spatial proxies.")
    pose = result.setdefault("POSE_METRICS", {})
    for key in ("ELONGATION_CONNECTION", "BODY_ANGLE_CONNECTION"):
        pair = pose.get(key) or []
        if pair and (len(pair) != 2 or any(p not in points for p in pair) or pair[0] == pair[1]):
            raise ValueError(f"{key}: choose two different model landmarks, or leave blank.")
        pose[key] = list(pair)
    general = result.setdefault("GENERAL_PARAMS", {})
    review = result.setdefault("REVIEW", {})
    for section, key, default, lower, upper in (
        (general, "DETECTION_CONF_THRESHOLD", .25, 0, 1),
        (general, "KEYPOINT_CONF_THRESHOLD", .25, 0, 1),
        (gait, "PAW_SPEED_THRESHOLD_PX_PER_FRAME", 5., 0, None),
        (gait, "BODY_SPEED_THRESHOLD_PX_PER_FRAME", 0., 0, None),
        (review, "PADDING_SECONDS", .5, 0, 10),
    ):
        value = float(section.get(key, default))
        if not math.isfinite(value) or value < lower or (upper is not None and value > upper):
            raise ValueError(f"Invalid value for {key}.")
        section[key] = value
    for section, key, default, lower in (
        (general, "MIN_BOUT_DURATION_FRAMES", 15, 1),
        (general, "TARGET_TRACK_ID", None, 0),
        (general, "START_FRAME", None, 0), (general, "END_FRAME", None, 0),
        (review, "MAX_CLIPS", 12, 0),
    ):
        value = section.get(key, default)
        if value is None or value == "":
            if default is not None:
                raise ValueError(f"{key} requires an integer.")
            section[key] = None
            continue
        number = float(value)
        if not math.isfinite(number) or not number.is_integer() or number < lower:
            raise ValueError(f"{key} must be an integer >= {lower}.")
        section[key] = int(number)
    start, end = general["START_FRAME"], general["END_FRAME"]
    if start is not None and end is not None and end < start:
        raise ValueError("End frame must be greater than or equal to start frame.")
    review.setdefault("EXPORT_CLIPS", True)
    review.setdefault("EXPORT_COORDINATION", False)
    if not isinstance(review["EXPORT_COORDINATION"], bool):
        raise ValueError("EXPORT_COORDINATION must be true or false.")
    if not isinstance(review["EXPORT_CLIPS"], bool):
        raise ValueError("EXPORT_CLIPS must be true or false.")
    result.setdefault("PROFILE_NAME", "Custom")
    return result


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Write a portable human or mouse gait configuration.")
    parser.add_argument("--preset", choices=["human", "mouse"], required=True)
    parser.add_argument("--output", required=True, help="New JSON file to create; existing files are not overwritten.")
    args = parser.parse_args()
    config = preset_config(PRESET_NAMES[1 if args.preset == "human" else 0])
    with Path(args.output).open("x", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
