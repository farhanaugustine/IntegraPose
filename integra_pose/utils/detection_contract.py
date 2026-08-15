"""Postconditions for Ultralytics detection results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class DetectionContractError(RuntimeError):
    """Raised when a model result cannot satisfy the requested detection cap."""


@dataclass(frozen=True)
class DetectionLimitOutcome:
    result: object
    original_count: int
    retained_count: int

    @property
    def dropped_count(self) -> int:
        return self.original_count - self.retained_count


def _to_numpy(value) -> np.ndarray | None:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)
    except Exception:
        return None


def enforce_ultralytics_max_det(result, max_det: int) -> DetectionLimitOutcome:
    """Return a result containing at most ``max_det`` highest-confidence rows."""
    limit = int(max_det)
    if limit < 1:
        raise ValueError("max_det must be at least 1.")

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return DetectionLimitOutcome(result=result, original_count=0, retained_count=0)
    try:
        count = int(len(boxes))
    except Exception as exc:
        raise DetectionContractError("Could not determine the model detection count.") from exc
    if count <= limit:
        return DetectionLimitOutcome(result=result, original_count=count, retained_count=count)

    confidence = _to_numpy(getattr(boxes, "conf", None))
    if confidence is None or confidence.size < count:
        raise DetectionContractError(
            f"Model returned {count} detections with max_det={limit}, but did not provide "
            "one confidence score per detection. The excess rows cannot be capped objectively."
        )
    scores = confidence.reshape(-1)[:count].astype(float, copy=False)
    if not np.isfinite(scores).all():
        raise DetectionContractError(
            f"Model returned non-finite confidence scores while enforcing max_det={limit}."
        )
    keep = np.argsort(-scores, kind="stable")[:limit]

    try:
        limited = result[keep.tolist()]
    except Exception as exc:
        raise DetectionContractError(
            f"Model returned {count} detections with max_det={limit}, and the result could not be capped safely."
        ) from exc

    limited_boxes = getattr(limited, "boxes", None)
    try:
        retained = int(len(limited_boxes)) if limited_boxes is not None else 0
    except Exception as exc:
        raise DetectionContractError("Could not verify the capped model result.") from exc
    if retained > limit:
        raise DetectionContractError(
            f"Model result still contains {retained} detections after enforcing max_det={limit}."
        )
    return DetectionLimitOutcome(result=limited, original_count=count, retained_count=retained)


__all__ = [
    "DetectionContractError",
    "DetectionLimitOutcome",
    "enforce_ultralytics_max_det",
]
