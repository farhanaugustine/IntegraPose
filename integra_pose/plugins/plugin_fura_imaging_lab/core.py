from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageSequence
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

try:  # pragma: no cover - optional dependency
    import tifffile  # type: ignore
except Exception:  # pragma: no cover - fallback path exercised when unavailable
    tifffile = None  # type: ignore[assignment]


@dataclass(frozen=True)
class RecordingData:
    source_label: str
    channel_340: np.ndarray
    channel_380: np.ndarray
    time_340: np.ndarray
    time_380: np.ndarray
    width: int
    height: int
    crop_x: int = 0
    crop_y: int = 0


@dataclass(frozen=True)
class AlignedRecording:
    source_label: str
    channel_340: np.ndarray
    channel_380: np.ndarray
    time_s: np.ndarray
    width: int
    height: int
    crop_x: int = 0
    crop_y: int = 0


@dataclass(frozen=True)
class RoiSeed:
    name: str
    kind: str  # "cell" | "background"
    x: int
    y: int
    w: int
    h: int
    shape: str = "rectangle"  # "rectangle" | "circle" | "polygon"
    points: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class TrackingConfig:
    reference_channel: str = "340"
    reference_average_count: int = 5
    search_radius: int = 18
    adaptive_template_rate: float = 0.10
    apply_drift_correction: bool = True


@dataclass(frozen=True)
class TrackingResult:
    aligned: AlignedRecording
    corrected_340: np.ndarray
    corrected_380: np.ndarray
    reference_frame: np.ndarray
    drift_shifts_xy: np.ndarray
    seeds: list[RoiSeed]
    track_table: pd.DataFrame
    ratio_table: pd.DataFrame
    signal_table: pd.DataFrame


@dataclass(frozen=True)
class AnalysisConfig:
    baseline_periods: list[tuple[float, float]]
    stimulations: list[tuple[float, float]]
    event_time: Optional[float]
    signal_family: str
    normalization_mode: str
    smoothing_method: str
    moving_average_window: int
    savgol_window: int
    savgol_polyorder: int
    analysis_window_dur: float
    auc_short_dur: float


@dataclass(frozen=True)
class AnalysisResult:
    ratio_table: pd.DataFrame
    smoothed_table: pd.DataFrame
    corrected_table: pd.DataFrame
    baseline_table: pd.DataFrame
    metrics_table: pd.DataFrame
    baseline_info: dict[str, dict[str, object]]


class AxiLoader:
    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self.size = self.file_path.stat().st_size
        self.width = 0
        self.height = 0
        self.crop_x = 0
        self.crop_y = 0
        self.header_offset = 124
        self.parse_metadata()

    def parse_metadata(self) -> None:
        with self.file_path.open("rb") as handle:
            header = handle.read(1024 * 1024)
        best = self._select_best_roi_wave_block(header)
        if best is not None:
            self.width = int(best["width"])
            self.height = int(best["height"])
            self.crop_x = int(best["crop_x"])
            self.crop_y = int(best["crop_y"])
        if not self.width or not self.height:
            self.width, self.height = 640, 480
            self.crop_x, self.crop_y = 0, 0

    def extract_data(self) -> RecordingData:
        waves: list[int] = []
        chunk_size = 10 * 1024 * 1024
        with self.file_path.open("rb") as handle:
            while True:
                current_pos = handle.tell()
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                pos = 0
                while True:
                    pos = chunk.find(b"wave", pos)
                    if pos == -1:
                        break
                    waves.append(current_pos + pos)
                    pos += 4
                if len(chunk) == chunk_size:
                    handle.seek(-4, 1)

        data_340: list[np.ndarray] = []
        data_380: list[np.ndarray] = []
        times_340: list[float] = []
        times_380: list[float] = []
        frames_seen: set[int] = set()

        with self.file_path.open("rb") as handle:
            for w_off in waves:
                handle.seek(w_off + 12)
                chan_id = struct.unpack("<I", handle.read(4))[0]
                handle.seek(w_off + 24)
                data_ptr = struct.unpack("<I", handle.read(4))[0]
                if 1024 < data_ptr < self.size and data_ptr not in frames_seen:
                    frames_seen.add(data_ptr)
                    self._refine_dimensions_from_frame_header(handle, data_ptr)
                    handle.seek(data_ptr + 8)
                    timestamp = struct.unpack("<I", handle.read(4))[0] / 1000.0
                    handle.seek(data_ptr + self.header_offset)
                    raw_bytes = handle.read(self.width * self.height * 2)
                    if len(raw_bytes) < self.width * self.height * 2:
                        continue
                    pixels = np.frombuffer(raw_bytes, dtype=np.uint16).reshape((self.height, self.width))
                    if chan_id == 0:
                        data_340.append(pixels)
                        times_340.append(timestamp)
                    elif chan_id == 2:
                        data_380.append(pixels)
                        times_380.append(timestamp)

        return RecordingData(
            source_label=str(self.file_path),
            channel_340=np.asarray(data_340, dtype=np.float32),
            channel_380=np.asarray(data_380, dtype=np.float32),
            time_340=np.asarray(times_340, dtype=np.float32),
            time_380=np.asarray(times_380, dtype=np.float32),
            width=int(self.width),
            height=int(self.height),
            crop_x=int(self.crop_x),
            crop_y=int(self.crop_y),
        )

    def _select_best_roi_wave_block(self, header: bytes) -> Optional[dict[str, int]]:
        positions = self._find_roi_wave_positions(header)
        if not positions:
            return None
        candidates: list[dict[str, int]] = []
        for pos in positions:
            block = self._parse_roi_wave_block(header, pos)
            if block is not None:
                candidates.append(block)
        if not candidates:
            return None

        def _score(item: dict[str, int]) -> tuple[int, int, int]:
            return (
                1 if item["self_pointer"] else 0,
                1 if item["channel"] == 0 else 0,
                item["width"] * item["height"],
            )

        return max(candidates, key=_score)

    def _find_roi_wave_positions(self, header: bytes) -> list[int]:
        positions: list[int] = []
        start = 0
        needle = b"roi wave"
        while True:
            pos = header.find(needle, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        return positions

    def _parse_roi_wave_block(self, header: bytes, pos: int) -> Optional[dict[str, int]]:
        if pos < 0 or pos + 60 > len(header):
            return None
        try:
            channel = struct.unpack("<I", header[pos + 12 : pos + 16])[0]
            height = struct.unpack("<I", header[pos + 16 : pos + 20])[0]
            pointer = struct.unpack("<I", header[pos + 24 : pos + 28])[0]
            crop_x = struct.unpack("<I", header[pos + 48 : pos + 52])[0]
            crop_y = struct.unpack("<I", header[pos + 52 : pos + 56])[0]
            width = struct.unpack("<I", header[pos + 56 : pos + 60])[0]
        except Exception:
            return None
        if not (10 <= height <= 5000 and 10 <= width <= 5000):
            return None
        if crop_x > 100000 or crop_y > 100000:
            return None
        return {
            "pos": int(pos),
            "channel": int(channel),
            "height": int(height),
            "width": int(width),
            "crop_x": int(crop_x),
            "crop_y": int(crop_y),
            "pointer": int(pointer),
            "self_pointer": int(pointer == pos),
        }

    def _refine_dimensions_from_frame_header(self, handle, data_ptr: int) -> None:
        handle.seek(data_ptr + 32)
        header_bytes = handle.read(16)
        if len(header_bytes) < 16:
            return
        payload_candidates = [
            struct.unpack("<I", header_bytes[0:4])[0],
            struct.unpack("<I", header_bytes[8:12])[0],
        ]
        for payload_bytes in payload_candidates:
            if payload_bytes <= 0 or payload_bytes > 200_000_000:
                continue
            if self.height > 0 and payload_bytes % max(self.height * 2, 1) == 0:
                inferred_width = payload_bytes // (self.height * 2)
                if 10 <= inferred_width <= 5000:
                    self.width = int(inferred_width)
                    return


def load_axi_recording(file_path: str | Path) -> RecordingData:
    return AxiLoader(file_path).extract_data()


def load_tiff_pair_recording(
    path_340: str | Path,
    path_380: str | Path,
    frame_interval_s: float,
) -> RecordingData:
    stack_340 = _read_tiff_stack(path_340)
    stack_380 = _read_tiff_stack(path_380)
    if stack_340.shape[1:] != stack_380.shape[1:]:
        raise ValueError("340 nm and 380 nm TIFF stacks must have matching frame sizes.")
    count_340 = stack_340.shape[0]
    count_380 = stack_380.shape[0]
    if frame_interval_s <= 0:
        raise ValueError("Frame interval must be positive for TIFF imports.")
    return RecordingData(
        source_label=f"{path_340} | {path_380}",
        channel_340=stack_340.astype(np.float32, copy=False),
        channel_380=stack_380.astype(np.float32, copy=False),
        time_340=np.arange(count_340, dtype=np.float32) * float(frame_interval_s),
        time_380=np.arange(count_380, dtype=np.float32) * float(frame_interval_s),
        width=int(stack_340.shape[2]),
        height=int(stack_340.shape[1]),
        crop_x=0,
        crop_y=0,
    )


def _read_tiff_stack(path: str | Path) -> np.ndarray:
    path = Path(path)
    if tifffile is not None:
        array = np.asarray(tifffile.imread(path))
    else:
        with Image.open(path) as image:
            frames = [np.asarray(frame.copy()) for frame in ImageSequence.Iterator(image)]
        if not frames:
            raise ValueError(f"No frames found in TIFF: {path}")
        array = np.stack(frames, axis=0)
    if array.ndim == 2:
        array = array[None, ...]
    elif array.ndim == 3:
        if array.shape[-1] in {3, 4}:
            array = cv2.cvtColor(array.astype(np.float32), cv2.COLOR_RGB2GRAY)[None, ...]
    elif array.ndim == 4:
        if array.shape[-1] not in {3, 4}:
            raise ValueError(f"Unsupported TIFF stack shape: {array.shape}")
        gray_frames = []
        for frame in array:
            gray_frames.append(cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_RGB2GRAY))
        array = np.stack(gray_frames, axis=0)
    else:
        raise ValueError(f"Unsupported TIFF stack shape: {array.shape}")
    if array.ndim != 3:
        raise ValueError(f"Expected stack with shape [frames, height, width], got {array.shape}")
    return np.asarray(array, dtype=np.float32)


def align_recording(recording: RecordingData) -> AlignedRecording:
    n_frames = min(len(recording.channel_340), len(recording.channel_380))
    if n_frames <= 0:
        raise ValueError("Recording contains no matched 340/380 frame pairs.")
    return AlignedRecording(
        source_label=recording.source_label,
        channel_340=np.asarray(recording.channel_340[:n_frames], dtype=np.float32),
        channel_380=np.asarray(recording.channel_380[:n_frames], dtype=np.float32),
        time_s=np.asarray(recording.time_340[:n_frames], dtype=np.float32),
        width=recording.width,
        height=recording.height,
        crop_x=recording.crop_x,
        crop_y=recording.crop_y,
    )


def build_reference_frame(
    aligned: AlignedRecording,
    *,
    channel: str = "340",
    average_count: int = 5,
) -> np.ndarray:
    stack = aligned.channel_340 if channel == "340" else aligned.channel_380
    frame_count = max(1, min(int(average_count), stack.shape[0]))
    return np.mean(stack[:frame_count], axis=0)


def estimate_drift_shifts(stack: np.ndarray, reference_frame: np.ndarray) -> np.ndarray:
    ref = _prepare_phase_image(reference_frame)
    if ref.size == 0:
        return np.zeros((stack.shape[0], 2), dtype=np.float32)
    shifts = np.zeros((stack.shape[0], 2), dtype=np.float32)
    for idx, frame in enumerate(stack):
        shifts[idx] = _estimate_phase_shift(ref, _prepare_phase_image(frame))
    return shifts


def _prepare_phase_image(frame: np.ndarray) -> np.ndarray:
    work = np.asarray(frame, dtype=np.float32)
    if work.size == 0:
        return work
    lo, hi = np.percentile(work, [1.0, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    work = np.clip((work - lo) / (hi - lo), 0.0, 1.0)
    win_y = np.hanning(work.shape[0])[:, None]
    win_x = np.hanning(work.shape[1])[None, :]
    return work * (win_y * win_x).astype(np.float32)


def _estimate_phase_shift(reference_frame: np.ndarray, current_frame: np.ndarray) -> np.ndarray:
    if reference_frame.shape != current_frame.shape:
        return np.zeros(2, dtype=np.float32)
    try:
        shift_xy, _response = cv2.phaseCorrelate(reference_frame, current_frame)
        return np.asarray(shift_xy, dtype=np.float32)
    except Exception:
        return np.zeros(2, dtype=np.float32)


def apply_frame_shifts(stack: np.ndarray, shifts_xy: np.ndarray) -> np.ndarray:
    corrected = np.empty_like(stack, dtype=np.float32)
    height, width = stack.shape[1:]
    for idx, frame in enumerate(stack):
        shift_x, shift_y = shifts_xy[idx]
        matrix = np.array([[1.0, 0.0, -float(shift_x)], [0.0, 1.0, -float(shift_y)]], dtype=np.float32)
        corrected[idx] = cv2.warpAffine(
            np.asarray(frame, dtype=np.float32),
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
    return corrected


def run_tracking(
    recording: RecordingData,
    seeds: list[RoiSeed],
    config: TrackingConfig,
) -> TrackingResult:
    if not seeds:
        raise ValueError("Add at least one ROI before running tracking.")
    aligned = align_recording(recording)
    ref_stack = aligned.channel_340 if config.reference_channel == "340" else aligned.channel_380
    reference_frame = build_reference_frame(
        aligned,
        channel=config.reference_channel,
        average_count=config.reference_average_count,
    )
    if config.apply_drift_correction:
        drift_shifts = estimate_drift_shifts(ref_stack, reference_frame)
        corrected_340 = apply_frame_shifts(aligned.channel_340, drift_shifts)
        corrected_380 = apply_frame_shifts(aligned.channel_380, drift_shifts)
    else:
        drift_shifts = np.zeros((aligned.channel_340.shape[0], 2), dtype=np.float32)
        corrected_340 = aligned.channel_340.copy()
        corrected_380 = aligned.channel_380.copy()
    corrected_aligned = AlignedRecording(
        source_label=aligned.source_label,
        channel_340=corrected_340,
        channel_380=corrected_380,
        time_s=aligned.time_s,
        width=aligned.width,
        height=aligned.height,
        crop_x=aligned.crop_x,
        crop_y=aligned.crop_y,
    )
    reference_frame_corrected = build_reference_frame(
        corrected_aligned,
        channel=config.reference_channel,
        average_count=config.reference_average_count,
    )
    track_table, ratio_table, signal_table = _track_on_corrected_frames(
        corrected_aligned,
        reference_frame_corrected,
        seeds,
        config,
    )
    return TrackingResult(
        aligned=corrected_aligned,
        corrected_340=corrected_340,
        corrected_380=corrected_380,
        reference_frame=reference_frame_corrected,
        drift_shifts_xy=drift_shifts,
        seeds=list(seeds),
        track_table=track_table,
        ratio_table=ratio_table,
        signal_table=signal_table,
    )


def _track_on_corrected_frames(
    aligned: AlignedRecording,
    reference_frame: np.ndarray,
    seeds: list[RoiSeed],
    config: TrackingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ref_stack = aligned.channel_340 if config.reference_channel == "340" else aligned.channel_380
    cell_seeds = [seed for seed in seeds if seed.kind == "cell"]
    bg_seeds = [seed for seed in seeds if seed.kind == "background"]
    if not cell_seeds:
        raise ValueError("Add at least one cell ROI before running tracking.")

    templates: dict[str, np.ndarray] = {}
    previous_boxes: dict[str, tuple[int, int, int, int]] = {}
    for seed in cell_seeds:
        seed_box = _clip_box(seed.x, seed.y, seed.w, seed.h, aligned.width, aligned.height)
        previous_boxes[seed.name] = seed_box
        templates[seed.name] = _extract_patch(reference_frame, seed_box)

    rows: list[dict[str, object]] = []
    ratio_table = pd.DataFrame({"Time": aligned.time_s})
    signal_table = pd.DataFrame({"Time": aligned.time_s})
    bg340_series = np.zeros(len(aligned.time_s), dtype=np.float32)
    bg380_series = np.zeros(len(aligned.time_s), dtype=np.float32)

    for frame_idx, time_s in enumerate(aligned.time_s):
        frame_ref = ref_stack[frame_idx]
        frame_340 = aligned.channel_340[frame_idx]
        frame_380 = aligned.channel_380[frame_idx]

        if bg_seeds:
            bg_vals_340 = []
            bg_vals_380 = []
            for seed in bg_seeds:
                box = _clip_box(seed.x, seed.y, seed.w, seed.h, aligned.width, aligned.height)
                bg_vals_340.append(_mean_in_seed(frame_340, seed, box))
                bg_vals_380.append(_mean_in_seed(frame_380, seed, box))
            bg340 = float(np.nanmean(bg_vals_340))
            bg380 = float(np.nanmean(bg_vals_380))
        else:
            bg340 = 0.0
            bg380 = 0.0
        bg340_series[frame_idx] = bg340
        bg380_series[frame_idx] = bg380

        for seed in cell_seeds:
            prev_box = previous_boxes[seed.name]
            template = templates[seed.name]
            if frame_idx == 0:
                box = prev_box
                score = 1.0
            else:
                box, score = _track_template(frame_ref, template, prev_box, config.search_radius)
                box = _clip_box(*box, aligned.width, aligned.height)
                previous_boxes[seed.name] = box
                patch = _extract_patch(frame_ref, box)
                if patch.shape == template.shape and patch.size and config.adaptive_template_rate > 0:
                    rate = float(np.clip(config.adaptive_template_rate, 0.0, 1.0))
                    templates[seed.name] = (template * (1.0 - rate) + patch * rate).astype(np.float32)

            mean_340 = _mean_in_seed(frame_340, seed, box)
            mean_380 = _mean_in_seed(frame_380, seed, box)
            bg_sub_340 = mean_340 - bg340
            bg_sub_380 = mean_380 - bg380
            raw_ratio = mean_340 / max(mean_380, 1e-6)
            bg_sub_ratio = bg_sub_340 / max(bg_sub_380, 1e-6)
            area_px = polygon_area(polygon_for_seed(seed, box))
            x, y, w, h = box
            rows.append(
                {
                    "frame": int(frame_idx),
                    "time_s": float(time_s),
                    "cell_id": seed.name,
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "shape": seed.shape,
                    "area_px": float(area_px),
                    "match_score": float(score),
                    "mean_340_raw": float(mean_340),
                    "mean_380_raw": float(mean_380),
                    "bg_340": float(bg340),
                    "bg_380": float(bg380),
                    "mean_340_bg_sub": float(bg_sub_340),
                    "mean_380_bg_sub": float(bg_sub_380),
                    "ratio_raw": float(raw_ratio),
                    "ratio_bg_sub": float(bg_sub_ratio),
                }
            )

    track_table = pd.DataFrame(rows)
    signal_table["Background_340"] = bg340_series
    signal_table["Background_380"] = bg380_series
    for seed in cell_seeds:
        cell_df = track_table[track_table["cell_id"] == seed.name].sort_values("frame")
        signal_table[f"{seed.name}_Area_px"] = cell_df["area_px"].to_numpy(dtype=np.float32)
        signal_table[f"{seed.name}_340nm_Raw"] = cell_df["mean_340_raw"].to_numpy(dtype=np.float32)
        signal_table[f"{seed.name}_380nm_Raw"] = cell_df["mean_380_raw"].to_numpy(dtype=np.float32)
        signal_table[f"{seed.name}_340nm_BGSub"] = cell_df["mean_340_bg_sub"].to_numpy(dtype=np.float32)
        signal_table[f"{seed.name}_380nm_BGSub"] = cell_df["mean_380_bg_sub"].to_numpy(dtype=np.float32)
        signal_table[f"{seed.name}_Ratio_Raw"] = cell_df["ratio_raw"].to_numpy(dtype=np.float32)
        signal_table[f"{seed.name}_Ratio_BGSub"] = cell_df["ratio_bg_sub"].to_numpy(dtype=np.float32)
        ratio_table[f"{seed.name}_Ratio_Raw"] = cell_df["ratio_raw"].to_numpy(dtype=np.float32)
        ratio_table[f"{seed.name}_Ratio_BGSub"] = cell_df["ratio_bg_sub"].to_numpy(dtype=np.float32)
    return track_table, ratio_table, signal_table


def _track_template(
    image: np.ndarray,
    template: np.ndarray,
    previous_box: tuple[int, int, int, int],
    search_radius: int,
) -> tuple[tuple[int, int, int, int], float]:
    px, py, pw, ph = previous_box
    radius = max(int(search_radius), 2)
    x0 = max(0, px - radius)
    y0 = max(0, py - radius)
    x1 = min(int(image.shape[1]), px + pw + radius)
    y1 = min(int(image.shape[0]), py + ph + radius)
    search = np.asarray(image[y0:y1, x0:x1], dtype=np.float32)
    if search.shape[0] < ph or search.shape[1] < pw or template.size == 0:
        return previous_box, 0.0
    search_norm = _normalize_for_matching(search)
    template_norm = _normalize_for_matching(template)
    result = cv2.matchTemplate(search_norm, template_norm, cv2.TM_CCOEFF_NORMED)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
    new_x = x0 + int(max_loc[0])
    new_y = y0 + int(max_loc[1])
    return (new_x, new_y, pw, ph), float(max_val)


def _normalize_for_matching(image: np.ndarray) -> np.ndarray:
    work = np.asarray(image, dtype=np.float32)
    mean = float(np.mean(work))
    std = float(np.std(work))
    if std < 1e-6:
        std = 1.0
    return (work - mean) / std


def _extract_patch(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    return np.asarray(image[y : y + h, x : x + w], dtype=np.float32)


def polygon_for_seed(seed: RoiSeed, box: Optional[tuple[int, int, int, int]] = None) -> np.ndarray:
    target_box = box if box is not None else (seed.x, seed.y, seed.w, seed.h)
    tx, ty, _tw, _th = target_box
    if seed.points:
        dx = int(tx - seed.x)
        dy = int(ty - seed.y)
        return np.asarray([(int(px + dx), int(py + dy)) for px, py in seed.points], dtype=np.int32)
    return _box_to_polygon(target_box)


def polygon_area(polygon: np.ndarray) -> float:
    poly = np.asarray(polygon, dtype=np.float32)
    if poly.ndim != 2 or poly.shape[0] < 3:
        return 0.0
    return float(abs(cv2.contourArea(poly.reshape(-1, 1, 2))))


def _box_to_polygon(box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    return np.asarray(
        [
            (int(x), int(y)),
            (int(x + w), int(y)),
            (int(x + w), int(y + h)),
            (int(x), int(y + h)),
        ],
        dtype=np.int32,
    )


def _mean_in_seed(image: np.ndarray, seed: RoiSeed, box: tuple[int, int, int, int]) -> float:
    polygon = polygon_for_seed(seed, box)
    return _mean_in_polygon(image, polygon, box)


def _mean_in_polygon(
    image: np.ndarray,
    polygon: np.ndarray,
    box: tuple[int, int, int, int],
) -> float:
    x, y, w, h = box
    patch = _extract_patch(image, box)
    if patch.size == 0:
        return 0.0
    rel_poly = np.asarray(polygon, dtype=np.int32).copy()
    rel_poly[:, 0] -= int(x)
    rel_poly[:, 1] -= int(y)
    mask = np.zeros((h, w), dtype=np.uint8)
    if rel_poly.ndim != 2 or rel_poly.shape[0] < 3:
        return float(np.mean(patch))
    cv2.fillPoly(mask, [rel_poly], 1)
    selected = patch[mask.astype(bool)]
    if selected.size == 0:
        return float(np.mean(patch))
    return float(np.mean(selected))


def _mean_in_box(image: np.ndarray, box: tuple[int, int, int, int]) -> float:
    patch = _extract_patch(image, box)
    if patch.size == 0:
        return 0.0
    return float(np.mean(patch))


def _clip_box(x: int, y: int, w: int, h: int, width: int, height: int) -> tuple[int, int, int, int]:
    x = int(max(0, min(x, max(width - 1, 0))))
    y = int(max(0, min(y, max(height - 1, 0))))
    w = int(max(4, w))
    h = int(max(4, h))
    if x + w > width:
        w = max(1, width - x)
    if y + h > height:
        h = max(1, height - y)
    return x, y, w, h


def parse_period_text(text: str) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.replace(";", ",")
        parts = [part.strip() for part in line.split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError(f"Invalid interval line: '{line}'")
        start, end = float(parts[0]), float(parts[1])
        if end <= start:
            raise ValueError(f"Interval end must be greater than start: '{line}'")
        intervals.append((start, end))
    return intervals


def analyze_tracking_result(result: TrackingResult, config: AnalysisConfig) -> AnalysisResult:
    ratio_table = _build_analysis_input_table(result, config)
    anchor = _resolve_event_anchor(config)
    ratio_table["Time_Aligned"] = ratio_table["Time"] - anchor
    smoothed = pd.DataFrame({"Time": ratio_table["Time"], "Time_Aligned": ratio_table["Time_Aligned"]})
    corrected = pd.DataFrame({"Time": ratio_table["Time"], "Time_Aligned": ratio_table["Time_Aligned"]})
    baseline = pd.DataFrame({"Time": ratio_table["Time"], "Time_Aligned": ratio_table["Time_Aligned"]})
    metrics_rows: dict[str, dict[str, float]] = {}
    baseline_info: dict[str, dict[str, object]] = {}

    signal_columns = [c for c in ratio_table.columns if c not in {"Time", "Time_Aligned"}]
    for col in signal_columns:
        signal = pd.Series(pd.to_numeric(ratio_table[col], errors="coerce"), index=ratio_table.index)
        smoothed_series = _smooth_signal(signal, config)
        corrected_series, baseline_series, base_info = _baseline_correct(
            ratio_table["Time"],
            smoothed_series,
            config.baseline_periods,
        )
        smoothed[col] = smoothed_series
        corrected[col] = corrected_series
        baseline[col] = baseline_series
        baseline_info[col] = base_info
        metrics_rows[col] = _compute_event_metrics(
            ratio_table["Time"],
            corrected_series,
            config.stimulations,
            config.analysis_window_dur,
            config.auc_short_dur,
        )

    metrics_table = pd.DataFrame.from_dict(metrics_rows, orient="index")
    return AnalysisResult(
        ratio_table=ratio_table,
        smoothed_table=smoothed,
        corrected_table=corrected,
        baseline_table=baseline,
        metrics_table=metrics_table,
        baseline_info=baseline_info,
    )


def _build_analysis_input_table(result: TrackingResult, config: AnalysisConfig) -> pd.DataFrame:
    family_suffix_map = {
        "ratio_bg_sub": "_Ratio_BGSub",
        "ratio_raw": "_Ratio_Raw",
        "signal_340_bg_sub": "_340nm_BGSub",
        "signal_380_bg_sub": "_380nm_BGSub",
        "signal_340_raw": "_340nm_Raw",
        "signal_380_raw": "_380nm_Raw",
    }
    suffix = family_suffix_map.get(config.signal_family, "_Ratio_BGSub")
    out = pd.DataFrame({"Time": result.signal_table["Time"]})
    base_periods = config.baseline_periods
    for col in [c for c in result.signal_table.columns if c.endswith(suffix)]:
        values = pd.Series(pd.to_numeric(result.signal_table[col], errors="coerce"), index=result.signal_table.index)
        seed_name = col[: -len(suffix)]
        area_col = f"{seed_name}_Area_px"
        area_series = pd.Series(pd.to_numeric(result.signal_table.get(area_col, pd.Series(dtype=float)), errors="coerce"), index=result.signal_table.index)
        out[col] = _normalize_signal(values, area_series, out["Time"], config.normalization_mode, base_periods)
    return out


def _normalize_signal(
    values: pd.Series,
    area_series: pd.Series,
    time_s: pd.Series,
    mode: str,
    baseline_periods: list[tuple[float, float]],
) -> pd.Series:
    mode = str(mode or "none").strip().lower()
    if mode == "none":
        return values.copy()
    if mode == "divide_by_area":
        denom = area_series.replace(0, np.nan)
        normalized = values / denom
        return normalized.replace([np.inf, -np.inf], np.nan)

    baseline_mask = pd.Series(False, index=values.index)
    for start, end in baseline_periods:
        baseline_mask |= (time_s >= start) & (time_s <= end)
    baseline_values = values[baseline_mask & values.notna()]
    if baseline_values.empty:
        return values.copy()
    baseline_mean = float(baseline_values.mean())
    baseline_std = float(baseline_values.std(ddof=0))
    if mode == "delta_over_baseline":
        if abs(baseline_mean) < 1e-9:
            return values.copy()
        return (values - baseline_mean) / baseline_mean
    if mode == "percent_change_from_baseline":
        if abs(baseline_mean) < 1e-9:
            return values.copy()
        return ((values - baseline_mean) / baseline_mean) * 100.0
    if mode == "zscore_baseline":
        if baseline_std < 1e-9:
            return values - baseline_mean
        return (values - baseline_mean) / baseline_std
    return values.copy()


def _resolve_event_anchor(config: AnalysisConfig) -> float:
    if config.event_time is not None:
        return float(config.event_time)
    if config.stimulations:
        return float(config.stimulations[0][0])
    return 0.0


def _smooth_signal(signal: pd.Series, config: AnalysisConfig) -> pd.Series:
    signal = signal.astype(float)
    method = str(config.smoothing_method or "moving_average").strip().lower()
    if method == "none":
        return signal.copy()
    if method == "moving_average":
        window = max(int(config.moving_average_window), 1)
        return signal.rolling(window=window, center=True, min_periods=1).mean()
    if method == "savitzky_golay":
        window = max(int(config.savgol_window), 3)
        if window % 2 == 0:
            window += 1
        poly = max(1, int(config.savgol_polyorder))
        valid = signal.dropna()
        if len(valid) < window or window <= poly:
            return signal.copy()
        smoothed = signal.copy()
        smoothed.loc[valid.index] = savgol_filter(valid.to_numpy(dtype=float), window_length=window, polyorder=poly)
        return smoothed
    return signal.copy()


def _baseline_correct(
    time_s: pd.Series,
    signal: pd.Series,
    baseline_periods: list[tuple[float, float]],
) -> tuple[pd.Series, pd.Series, dict[str, object]]:
    corrected = signal.copy()
    baseline_series = pd.Series(np.zeros(len(signal), dtype=float), index=signal.index)
    info: dict[str, object] = {
        "applied": False,
        "time_points": pd.Series(dtype=float),
        "indices": pd.Index([]),
    }
    if not baseline_periods:
        return corrected, baseline_series, info

    mask = pd.Series(False, index=signal.index)
    for start, end in baseline_periods:
        mask |= (time_s >= start) & (time_s <= end)
    valid = mask & signal.notna() & time_s.notna()
    if valid.sum() < 4:
        return corrected, baseline_series, info

    base_time = pd.Series(time_s[valid], index=time_s[valid].index)
    base_signal = pd.Series(signal[valid], index=signal[valid].index)
    unique_time, unique_idx = np.unique(base_time.to_numpy(dtype=float), return_index=True)
    if len(unique_time) < 4:
        return corrected, baseline_series, info
    unique_values = base_signal.to_numpy(dtype=float)[unique_idx]
    try:
        spline = UnivariateSpline(
            unique_time,
            unique_values,
            k=min(3, len(unique_time) - 1),
            s=max(len(unique_time) * 1e-6, 0.0),
        )
        baseline_vals = spline(time_s.to_numpy(dtype=float))
        baseline_series = pd.Series(baseline_vals, index=signal.index)
        corrected = signal - baseline_series
        info = {
            "applied": True,
            "time_points": pd.Series(unique_time),
            "indices": base_signal.index,
        }
    except Exception:
        return signal.copy(), pd.Series(np.zeros(len(signal), dtype=float), index=signal.index), info
    return corrected, baseline_series, info


def _compute_event_metrics(
    time_s: pd.Series,
    corrected: pd.Series,
    stimulations: list[tuple[float, float]],
    analysis_window_dur: float,
    auc_short_dur: float,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    time_arr = np.asarray(time_s, dtype=float)
    data_arr = np.asarray(corrected, dtype=float)
    finite_mask = np.isfinite(time_arr) & np.isfinite(data_arr)
    if finite_mask.sum() < 3:
        return metrics

    for idx, (stim_start, stim_end) in enumerate(stimulations):
        label = f"Stim{idx + 1}"
        analysis_end = stim_start + max(float(analysis_window_dur), 0.0)
        short_end = stim_start + max(float(auc_short_dur), 0.0)
        full_mask = finite_mask & (time_arr >= stim_start) & (time_arr <= analysis_end)
        short_mask = finite_mask & (time_arr >= stim_start) & (time_arr <= short_end)
        if not np.any(full_mask):
            continue
        t_full = time_arr[full_mask]
        y_full = data_arr[full_mask]
        start_value = float(np.interp(stim_start, time_arr[finite_mask], data_arr[finite_mask]))
        peak_idx = int(np.nanargmax(y_full))
        peak_time = float(t_full[peak_idx])
        peak_value = float(y_full[peak_idx])
        amp = peak_value - start_value
        metrics[f"StartValue_Abs_{label}"] = start_value
        metrics[f"TimeToPeak_Abs_{label}"] = peak_time
        metrics[f"TimeToPeak_Rel_{label}"] = peak_time - float(stim_start)
        metrics[f"MaxValue_Abs_{analysis_window_dur}s_{label}"] = peak_value
        metrics[f"Amp_Max_{analysis_window_dur}s_{label}"] = amp
        metrics[f"Slope_toPeak_{label}"] = amp / max(peak_time - float(stim_start), 1e-6)
        metrics[f"AUC_{analysis_window_dur}s_{label}"] = float(np.trapz(y_full - start_value, t_full))
        if np.any(short_mask):
            t_short = time_arr[short_mask]
            y_short = data_arr[short_mask]
            metrics[f"AUC_{auc_short_dur}s_{label}"] = float(np.trapz(y_short - start_value, t_short))
        peak_mask = finite_mask & (time_arr >= stim_start) & (time_arr <= peak_time)
        if np.any(peak_mask):
            t_peak = time_arr[peak_mask]
            y_peak = data_arr[peak_mask]
            metrics[f"AUC_toPeak_{label}"] = float(np.trapz(y_peak - start_value, t_peak))
        end_mask = finite_mask & (time_arr >= stim_start) & (time_arr <= stim_end)
        if np.any(end_mask):
            end_t = time_arr[end_mask]
            end_y = data_arr[end_mask]
            metrics[f"AUC_StimWindow_{label}"] = float(np.trapz(end_y - start_value, end_t))
    return metrics


def export_analysis_workbook(
    destination: str | Path,
    tracking: TrackingResult,
    analysis: AnalysisResult,
    parameters: dict[str, object],
) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    params_df = pd.DataFrame(
        {
            "Parameter": list(parameters.keys()),
            "Value": [json.dumps(value) if isinstance(value, (list, dict, tuple)) else value for value in parameters.values()],
        }
    )
    with pd.ExcelWriter(destination, engine="openpyxl") as writer:
        tracking.track_table.to_excel(writer, sheet_name="Track_Table", index=False, float_format="%.4f")
        tracking.signal_table.to_excel(writer, sheet_name="Signals_Raw", index=False, float_format="%.4f")
        tracking.ratio_table.to_excel(writer, sheet_name="Ratios_Raw_BGSub", index=False, float_format="%.4f")
        analysis.ratio_table.to_excel(writer, sheet_name="Analysis_Input", index=False, float_format="%.4f")
        analysis.smoothed_table.to_excel(writer, sheet_name="Smoothed_Data", index=False, float_format="%.4f")
        analysis.corrected_table.to_excel(writer, sheet_name="Corrected_Data", index=False, float_format="%.4f")
        analysis.baseline_table.to_excel(writer, sheet_name="Baseline_Trends", index=False, float_format="%.4f")
        metrics_to_save = analysis.metrics_table.copy()
        metrics_to_save.index.name = "Signal"
        metrics_to_save.reset_index().to_excel(writer, sheet_name="Metrics", index=False, float_format="%.4f")
        params_df.to_excel(writer, sheet_name="Parameters", index=False)
