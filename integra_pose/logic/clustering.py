import os
import re
import importlib
from typing import Dict, List

import numpy as np
import pandas as pd

from integra_pose.logic.pose_utils import normalize_pose

hdbscan = None  # type: ignore[assignment]
umap = None  # type: ignore[assignment]
plt = None  # type: ignore[assignment]
_HDBSCAN_ERROR = None
_UMAP_ERROR = None
_MATPLOTLIB_ERROR = None
_OPTIONAL_DEPS_LOADED = False


def _load_optional_dependencies() -> None:
    """Import heavy clustering dependencies only when clustering is actually used."""
    global _OPTIONAL_DEPS_LOADED, hdbscan, umap, plt
    global _HDBSCAN_ERROR, _UMAP_ERROR, _MATPLOTLIB_ERROR

    if _OPTIONAL_DEPS_LOADED:
        return
    _OPTIONAL_DEPS_LOADED = True

    try:
        hdbscan = importlib.import_module("hdbscan")  # type: ignore[assignment]
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        _HDBSCAN_ERROR = exc

    try:
        umap = importlib.import_module("umap")  # type: ignore[assignment]
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        _UMAP_ERROR = exc

    try:
        plt = importlib.import_module("matplotlib.pyplot")  # type: ignore[assignment]
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        _MATPLOTLIB_ERROR = exc


def _missing_dependencies() -> List[str]:
    _load_optional_dependencies()
    missing: List[str] = []
    if hdbscan is None:
        missing.append("hdbscan")
    if umap is None:
        missing.append("umap-learn")
    if plt is None:
        missing.append("matplotlib")
    return missing


def _sanitize_component(value: str | None) -> str:
    if not value:
        return "value"
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    sanitized = sanitized.strip("_")
    return sanitized or "value"


class PoseClustering:
    def __init__(self, app):
        self.app = app

    def run_clustering(
        self,
        track_data,
        keypoint_count,
        keypoint_names,
        skeleton,
        video_path,
        selected_behaviors,
    ):
        output_folder = os.path.join(os.path.dirname(video_path), "bout_analysis_results")
        os.makedirs(output_folder, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        raw_suffix = (
            selected_behaviors[0]
            if len(selected_behaviors) == 1
            else f"multi_{int(pd.Timestamp.now().timestamp())}"
        )
        safe_base_name = _sanitize_component(base_name)
        suffix = _sanitize_component(raw_suffix)

        missing = _missing_dependencies()
        if missing:
            raise RuntimeError(
                "Pose clustering requires optional dependencies: "
                + ", ".join(missing)
                + ". Install them via 'pip install hdbscan umap-learn matplotlib'."
            )

        all_cluster_data: Dict[int, Dict[str, object]] = {}
        valid_clusters: List[str] = []

        for track_id, data in track_data.items():
            pose_array = np.array(data['poses'], dtype=float)
            frame_array = np.array(data['frame_ids'])
            if pose_array.size == 0:
                self.app.log_message(f"No pose data available for track {track_id}; skipping.", "WARNING")
                continue

            invalid_rows = np.isnan(pose_array).any(axis=1)
            if invalid_rows.any():
                dropped = int(invalid_rows.sum())
                pose_array = pose_array[~invalid_rows]
                frame_array = frame_array[~invalid_rows]
                self.app.log_message(
                    f"Dropped {dropped} poses with missing keypoints for track {track_id}.",
                    "WARNING",
                )
                if pose_array.size == 0:
                    self.app.log_message(
                        f"All poses for track {track_id} contained missing keypoints; skipping.",
                        "WARNING",
                    )
                    continue

            frame_ids = frame_array.tolist()
            self.app.log_message(f"Clustering {len(pose_array)} poses for track {track_id}", "INFO")

            try:
                normalized_poses = np.array(
                    [
                        normalize_pose(
                            pose,
                            bool(self.app.config.pose_clustering.norm_translation_var.get()),
                            bool(self.app.config.pose_clustering.norm_scale_var.get()),
                        )
                        for pose in pose_array
                    ]
                )
            except Exception as e:
                self.app.log_message(f"Normalization error for track {track_id}: {e}", "ERROR")
                continue

            try:
                reducer = umap.UMAP(
                    n_neighbors=int(self.app.config.pose_clustering.umap_neighbors_var.get()),
                    min_dist=0.1,
                    n_components=2,
                    random_state=42,
                )
                embedding = reducer.fit_transform(normalized_poses)
            except Exception as e:
                self.app.log_message(f"UMAP error for track {track_id}: {e}", "ERROR")
                continue

            try:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=int(self.app.config.pose_clustering.hdbscan_min_size_var.get()),
                    min_samples=5,
                )
                labels = clusterer.fit_predict(embedding)
            except Exception as e:
                self.app.log_message(f"HDBSCAN error for track {track_id}: {e}", "ERROR")
                continue

            timeline = self._build_timeline(frame_ids, labels)
            transitions = self._collect_transitions(frame_ids, labels)
            summaries = self._summarize_clusters(normalized_poses, labels, frame_ids)
            morphometrics = self._compute_morphometrics_per_cluster(
                normalized_poses,
                labels,
                skeleton,
            )

            timeline_path = None
            transition_path = None
            morph_path = None

            frame_clusters = {}
            if timeline and 'frame_sequence' in timeline:
                frames_list, cluster_list = timeline['frame_sequence']
                frame_clusters = {int(f): int(c) for f, c in zip(frames_list, cluster_list)}
            if timeline and timeline.get('segments'):
                timeline_df = pd.DataFrame(timeline['segments'])
                timeline_df['duration_frames'] = timeline_df['end'] - timeline_df['start'] + 1
                timeline_df['track_id'] = track_id
                timeline_path = os.path.join(
                    output_folder,
                    f"pose_cluster_timeline_{safe_base_name}_{suffix}_track{track_id}.csv",
                )
                timeline_df.to_csv(timeline_path, index=False)

            if transitions:
                transitions_df = pd.DataFrame(transitions)
                transitions_df['track_id'] = track_id
                if 'previous_frame' in transitions_df.columns:
                    transitions_df['delta_frames'] = transitions_df['frame'] - transitions_df['previous_frame']
                transition_path = os.path.join(
                    output_folder,
                    f"pose_cluster_transitions_{safe_base_name}_{suffix}_track{track_id}.csv",
                )
                transitions_df.to_csv(transition_path, index=False)

            if morphometrics:
                morph_rows = []
                for cluster_id, metrics_dict in morphometrics.items():
                    row = {'cluster': cluster_id, **metrics_dict}
                    morph_rows.append(row)
                morph_df = pd.DataFrame(morph_rows)
                morph_df['track_id'] = track_id
                morph_path = os.path.join(
                    output_folder,
                    f"pose_cluster_morphometrics_{safe_base_name}_{suffix}_track{track_id}.csv",
                )
                morph_df.to_csv(morph_path, index=False)

            all_cluster_data[track_id] = {
                'poses': normalized_poses,
                'labels': labels,
                'embedding': embedding,
                'averages': summaries['averages'],
                'medians': summaries['medians'],
                'medoids': summaries['medoids'],
                'stats': summaries['stats'],
                'envelopes': summaries['envelopes'],
                'morphometrics': morphometrics,
                'timeline': timeline,
                'transitions': transitions,
                'frame_clusters': frame_clusters,
                'export_paths': {
                    'timeline': timeline_path,
                    'transitions': transition_path,
                    'morphometrics': morph_path,
                },
                'skeleton': skeleton,
                'keypoint_names': keypoint_names,
                'frame_ids': frame_ids,
                'behavior': selected_behaviors[0] if len(selected_behaviors) == 1 else 'Multiple',
            }

            cluster_img_path = os.path.join(
                output_folder, f"clusters_{safe_base_name}_{suffix}_track{track_id}.png"
            )
            cluster_csv_path = os.path.join(
                output_folder, f"pose_clusters_{safe_base_name}_{suffix}_track{track_id}.csv"
            )

            plt.figure(figsize=(8, 6))
            plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral')
            plt.colorbar(label='Cluster ID')
            plt.title(f"Pose Clusters for {', '.join(selected_behaviors)} (Track {track_id})")
            plt.savefig(cluster_img_path)
            plt.close()

            results_df = pd.DataFrame(
                {
                    'Frame': frame_ids,
                    'Behavior': [selected_behaviors[0]] * len(frame_ids)
                    if len(selected_behaviors) == 1
                    else ['Multiple'] * len(frame_ids),
                    'Track ID': [track_id] * len(frame_ids),
                    'Cluster': labels,
                }
            )
            results_df.to_csv(cluster_csv_path, index=False)

            centroid_path = None
            if summaries['averages']:
                centroid_df = pd.DataFrame(
                    [
                        [label, *avg_pose]
                        for label, avg_pose in summaries['averages'].items()
                    ],
                    columns=['Cluster']
                    + [f"Keypoint_{i + 1}_{axis}" for i in range(keypoint_count) for axis in ['x', 'y']],
                )
                centroid_df['Track ID'] = track_id
                centroid_path = os.path.join(
                    output_folder, f"pose_cluster_centroids_{safe_base_name}_{suffix}_track{track_id}.csv"
                )
                centroid_df.to_csv(centroid_path, index=False)

            export_paths = all_cluster_data[track_id]['export_paths']
            export_paths['assignments'] = cluster_csv_path
            export_paths['embedding_plot'] = cluster_img_path
            if centroid_path:
                export_paths['centroids'] = centroid_path

            message = f"Saved results for track {track_id} to {cluster_csv_path} and {cluster_img_path}"
            if centroid_path:
                message += f" (centroids: {centroid_path})"
            if export_paths.get('timeline'):
                message += f"; timeline: {export_paths['timeline']}"
            if export_paths.get('transitions'):
                message += f"; transitions: {export_paths['transitions']}"
            if export_paths.get('morphometrics'):
                message += f"; morphometrics: {export_paths['morphometrics']}"
            self.app.log_message(message, "INFO")

            for label in summaries['averages'].keys():
                if label != -1:
                    valid_clusters.append(f"Track {track_id} - Cluster {label}")

        return all_cluster_data, valid_clusters, output_folder, base_name, suffix, safe_base_name

    @staticmethod
    def _summarize_clusters(
        normalized_poses: np.ndarray,
        labels: np.ndarray,
        frame_ids: List[int],
    ) -> Dict[str, Dict[int, np.ndarray]]:
        """Compute aggregate pose representations and summary stats for each cluster."""
        unique_labels = np.unique(labels)

        cluster_averages: Dict[int, np.ndarray] = {}
        cluster_medians: Dict[int, np.ndarray] = {}
        cluster_medoids: Dict[int, np.ndarray] = {}
        cluster_stats: Dict[int, Dict[str, float]] = {}
        cluster_envelopes: Dict[int, List[Dict[str, float]]] = {}

        for label in unique_labels:
            if label == -1:
                continue

            mask = labels == label
            if not np.any(mask):
                continue

            cluster_poses = normalized_poses[mask]
            if cluster_poses.size == 0:
                continue

            avg_pose = np.mean(cluster_poses, axis=0)
            median_pose = np.median(cluster_poses, axis=0)

            distances = np.linalg.norm(cluster_poses - avg_pose, axis=1)
            medoid_local_idx = int(np.argmin(distances))
            medoid_pose = cluster_poses[medoid_local_idx]
            cluster_indices = np.flatnonzero(mask)
            medoid_global_idx = int(cluster_indices[medoid_local_idx])

            cluster_averages[label] = avg_pose
            cluster_medians[label] = median_pose
            cluster_medoids[label] = medoid_pose
            cluster_stats[label] = {
                'num_poses': int(cluster_poses.shape[0]),
                'avg_dist_to_centroid': float(distances.mean()) if distances.size else 0.0,
                'dispersion': float(distances.std()) if distances.size else 0.0,
                'medoid_frame': frame_ids[medoid_global_idx] if frame_ids else None,
                'medoid_index': medoid_global_idx,
            }
            cluster_envelopes[label] = PoseClustering._compute_pose_envelope(cluster_poses)

        return {
            'averages': cluster_averages,
            'medians': cluster_medians,
            'medoids': cluster_medoids,
            'stats': cluster_stats,
            'envelopes': cluster_envelopes,
        }

    @staticmethod
    def _compute_pose_envelope(cluster_poses: np.ndarray) -> List[Dict[str, float]]:
        pose_count, coord_count = cluster_poses.shape
        keypoint_count = coord_count // 2 if coord_count else 0
        envelopes: List[Dict[str, float]] = []
        if pose_count <= 1 or keypoint_count == 0:
            return [{'std_x': 0.0, 'std_y': 0.0} for _ in range(keypoint_count)]
        reshaped = cluster_poses.reshape(pose_count, keypoint_count, 2)
        for kp_idx in range(keypoint_count):
            kp_coords = reshaped[:, kp_idx, :]
            std_x = float(np.std(kp_coords[:, 0]))
            std_y = float(np.std(kp_coords[:, 1]))
            envelopes.append({'std_x': std_x, 'std_y': std_y})
        return envelopes

    @staticmethod
    def _build_timeline(frame_ids: List[int], labels: np.ndarray) -> Dict[str, object]:
        if not frame_ids or labels.size == 0:
            return {}
        order = np.argsort(frame_ids)
        frames_sorted = np.array(frame_ids)[order]
        labels_sorted = labels[order]

        segments: List[Dict[str, int]] = []
        current_label = int(labels_sorted[0])
        start_frame = int(frames_sorted[0])
        prev_frame = start_frame

        for frame, label in zip(frames_sorted[1:], labels_sorted[1:]):
            frame = int(frame)
            label = int(label)
            if label != current_label:
                segments.append({'cluster': current_label, 'start': start_frame, 'end': prev_frame})
                start_frame = frame
                current_label = label
            prev_frame = frame
        segments.append({'cluster': current_label, 'start': start_frame, 'end': int(frames_sorted[-1])})

        return {
            'segments': segments,
            'frame_sequence': (frames_sorted.tolist(), labels_sorted.tolist()),
        }

    @staticmethod
    def _collect_transitions(frame_ids: List[int], labels: np.ndarray) -> List[Dict[str, int]]:
        if not frame_ids or labels.size == 0:
            return []
        order = np.argsort(frame_ids)
        frames_sorted = np.array(frame_ids)[order]
        labels_sorted = labels[order]
        transitions: List[Dict[str, int]] = []
        prev_label = int(labels_sorted[0])
        prev_frame = int(frames_sorted[0])
        for frame, label in zip(frames_sorted[1:], labels_sorted[1:]):
            label = int(label)
            frame = int(frame)
            if label != prev_label and prev_label != -1 and label != -1:
                transitions.append({
                    'from': prev_label,
                    'to': label,
                    'frame': frame,
                    'previous_frame': prev_frame,
                })
            prev_label = label
            prev_frame = frame
        return transitions

    @staticmethod
    def _compute_morphometrics_per_cluster(
        normalized_poses: np.ndarray,
        labels: np.ndarray,
        skeleton: List[List[int]],
    ) -> Dict[int, Dict[str, float]]:
        unique_labels = np.unique(labels)
        metrics: Dict[int, Dict[str, float]] = {}
        for label in unique_labels:
            if label == -1:
                continue
            mask = labels == label
            cluster_poses = normalized_poses[mask]
            if cluster_poses.size == 0:
                continue
            metrics[label] = PoseClustering._compute_pose_metrics(cluster_poses, skeleton)
        return metrics

    @staticmethod
    def _compute_pose_metrics(cluster_poses: np.ndarray, skeleton: List[List[int]]) -> Dict[str, float]:
        pose_count, coord_count = cluster_poses.shape
        keypoint_count = coord_count // 2 if coord_count else 0
        if pose_count == 0 or keypoint_count == 0:
            return {'centroid_dispersion': 0.0, 'bbox_area': 0.0, 'mean_skeleton_length': 0.0, 'orientation_deg': 0.0}

        poses = cluster_poses.reshape(pose_count, keypoint_count, 2)
        centroid = poses.mean(axis=1, keepdims=True)
        centered = poses - centroid
        centroid_dist = np.linalg.norm(centered, axis=2)
        centroid_dispersion = float(centroid_dist.mean())

        extents = poses.ptp(axis=1)
        bbox_area = float(np.mean(extents[:, 0] * extents[:, 1]))

        skeleton_lengths = []
        if skeleton:
            for edge in skeleton:
                if len(edge) != 2:
                    continue
                idx1, idx2 = edge
                if idx1 >= keypoint_count or idx2 >= keypoint_count:
                    continue
                segment = poses[:, [idx1, idx2], :]
                lengths = np.linalg.norm(segment[:, 0, :] - segment[:, 1, :], axis=1)
                skeleton_lengths.extend(lengths.tolist())
        mean_skeleton_length = float(np.mean(skeleton_lengths)) if skeleton_lengths else 0.0

        covariance = np.cov(centered.reshape(-1, 2), rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        principal_vector = eigenvectors[:, np.argmax(eigenvalues)] if eigenvalues.size else np.array([1.0, 0.0])
        orientation_rad = float(np.arctan2(principal_vector[1], principal_vector[0]))

        return {
            'centroid_dispersion': centroid_dispersion,
            'bbox_area': bbox_area,
            'mean_skeleton_length': mean_skeleton_length,
            'orientation_deg': float(np.degrees(orientation_rad)),
        }
