"""ADP-4 Commit B — bout aggregation helpers, GUI-free.

This module hosts pure-data bout aggregators that do NOT pull in any of
the GUI / ML stack (no matplotlib, no torch, no umap, no hmmlearn). The
function lives here so:

  1. Tests can import and exercise it on a minimal pandas+numpy install
     without dragging in 1 GB of optional deps.
  2. ADP-4's defensibility code (held-out report, stability audit) can
     reuse it from contexts that don't need the full toolkit.

The legacy ``aggregate_into_bouts`` (class-id segmentation) remains in
``main.py`` for now — many existing call sites depend on its surrounding
context and a wholesale move would balloon the diff. When that function
gets extracted, this module is its natural home.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def aggregate_states_into_bouts(
    detections_df,
    max_frame_gap,
    min_bout_duration,
    *,
    state_column: str = "cluster_label",
):
    """Aggregate per-frame HMM/cluster states into bouts.

    Sibling of ``aggregate_into_bouts`` (which lives in ``main.py`` and
    cuts on ``class_id``). This one cuts on the state column — the HMM
    state in ``cluster_label`` by default. Use this **after** HMM
    inference, when each frame has a state assigned.

    Two functions exist side-by-side because:

    - ``aggregate_into_bouts`` (class-id segmentation) is the pre-HMM
      pipeline's contract; many existing call sites depend on it.
    - ``aggregate_states_into_bouts`` (state segmentation) is what
      ADP-4's defensibility / cluster-naming machinery needs — clusters
      are HMM states, and a "cluster bout" is a contiguous run of one
      state on a single track.

    Bout dicts carry the same fields as the class-based aggregator plus
    ``state`` (the run's HMM state) and ``state_column`` (which column
    it came from) — useful when downstream code consumes bouts produced
    from different label columns.

    Args:
        detections_df: Frame-level dataframe carrying at minimum
            ``group``, ``directory``, ``track_id``, ``frame``, the
            chosen ``state_column``. ``feature_vector``, ``bbox``,
            ``keypoints``, ``subject_id``, ``video_source``,
            ``class_id``, ``behavior`` are optional and pass through
            when present.
        max_frame_gap: Frames may be missing within a bout up to this
            count. Larger gaps split the bout — same semantics as
            ``aggregate_into_bouts``.
        min_bout_duration: Bouts shorter than this many detections are
            dropped. Pass ``1`` to keep singletons.
        state_column: Column whose value defines a bout. Default
            ``"cluster_label"`` (the HMM state column).

    Returns:
        list[dict]: One dict per bout, ordered by track and start frame.
        Empty list when input is empty.

    Raises:
        ValueError: If ``state_column`` is missing from the dataframe.
            The error message names the column explicitly so the caller
            doesn't have to guess.
    """
    # Late imports keep this module cheap to import.
    import numpy as np

    logger.info(
        "Aggregating state bouts on column %r (max_gap=%s, min_duration=%s)...",
        state_column, max_frame_gap, min_bout_duration,
    )
    if detections_df.empty:
        logger.warning("Cannot aggregate state bouts from an empty DataFrame.")
        return []
    if state_column not in detections_df.columns:
        raise ValueError(
            f"aggregate_states_into_bouts: column {state_column!r} missing from "
            "detections_df. Run HMM inference first so each frame has a "
            "state assigned, or pass state_column= to point at the column "
            "you want to segment."
        )

    sort_cols = [c for c in ("group", "directory", "track_id", "frame") if c in detections_df.columns]
    detections_df = detections_df.sort_values(by=sort_cols).copy()

    has_features = "feature_vector" in detections_df.columns
    has_bbox = "bbox" in detections_df.columns
    has_keypoints = "keypoints" in detections_df.columns
    has_subject = "subject_id" in detections_df.columns
    has_class = "class_id" in detections_df.columns
    has_behavior = "behavior" in detections_df.columns
    has_video_src = "video_source" in detections_df.columns

    all_bouts: list[dict[str, Any]] = []
    for (group, directory, track_id), track_df in detections_df.groupby(
        ["group", "directory", "track_id"]
    ):
        if track_df.empty:
            continue

        video_source = (
            track_df["video_source"].iloc[0]
            if has_video_src
            else (os.path.basename(directory) or "source")
        )
        subject_id = str(track_df["subject_id"].iloc[0]) if has_subject else ""

        # Bout boundaries: a new bout starts when state changes OR when
        # the frame jumps farther than max_frame_gap.
        track_df = track_df.copy()
        track_df["_frame_diff"] = track_df["frame"].diff()
        track_df["_state_diff"] = (
            track_df[state_column] != track_df[state_column].shift()
        ).astype(int)
        bout_start = (track_df["_state_diff"] > 0) | (track_df["_frame_diff"] > max_frame_gap)
        bout_ids = bout_start.cumsum()

        for _bout_id, bout_df in track_df.groupby(bout_ids):
            if bout_df.empty:
                continue
            duration = len(bout_df)
            if duration < min_bout_duration:
                continue

            # Centroid: prefer bbox, fall back to mean of confident kps.
            bout_centroids = []
            if has_bbox or has_keypoints:
                for _, det in bout_df.iterrows():
                    bbox = det.get("bbox") if has_bbox else None
                    if bbox is not None:
                        try:
                            bout_centroids.append(bbox[:2])
                        except Exception:
                            pass
                        continue
                    kps = det.get("keypoints") if has_keypoints else None
                    if kps:
                        try:
                            confident = np.array([kp[:2] for kp in kps.values() if kp[2] > 0.1])
                            if len(confident) > 0:
                                bout_centroids.append(np.mean(confident, axis=0))
                        except Exception:
                            pass
            mean_bout_location = (
                tuple(np.mean(bout_centroids, axis=0).tolist())
                if bout_centroids
                else (0.0, 0.0)
            )

            start_frame = int(bout_df["frame"].iloc[0])
            end_frame = int(bout_df["frame"].iloc[-1])
            state_value = bout_df[state_column].iloc[0]

            bout: dict[str, Any] = {
                "group": group,
                "video_source": video_source,
                "directory": directory,
                "track_id": track_id,
                "subject_id": subject_id,
                "state": state_value,
                "state_column": state_column,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration_frames": (end_frame - start_frame + 1),
                "detection_count": duration,
                "mean_location": mean_bout_location,
            }
            if has_class:
                # Dominant class_id within the bout — handy for the
                # held-out evaluation panel (Commit D).
                try:
                    bout["class_id"] = int(bout_df["class_id"].mode().iloc[0])
                except Exception:
                    bout["class_id"] = int(bout_df["class_id"].iloc[0])
            if has_behavior:
                try:
                    bout["behavior"] = bout_df["behavior"].mode().iloc[0]
                except Exception:
                    bout["behavior"] = bout_df["behavior"].iloc[0]
            if has_features:
                feature_vectors_seq = list(bout_df["feature_vector"])
                if feature_vectors_seq:
                    try:
                        mean_fv = np.mean(np.array(feature_vectors_seq), axis=0)
                    except Exception:
                        mean_fv = feature_vectors_seq[0]
                    bout["feature_vectors"] = feature_vectors_seq
                    bout["mean_feature_vector"] = mean_fv
                    bout["feature_vector"] = mean_fv

            all_bouts.append(bout)

    logger.info(
        "Aggregated %d detections into %d state bouts on column %r.",
        len(detections_df), len(all_bouts), state_column,
    )
    return all_bouts


__all__ = ["aggregate_states_into_bouts"]
