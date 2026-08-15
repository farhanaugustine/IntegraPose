"""Robust video metadata probing for the annotation workspace.

The annotation timeline places behavior spans by millisecond values and derives
frame numbers as ``round(t * fps)``. That is only scientifically accurate when
the stored ``fps`` / ``total_frames`` / ``duration_ms`` describe the video the
same way the media player clocks it. A single-source probe (OpenCV's
``CAP_PROP_FPS``) is fragile: it can return 0 for some containers -- previously
silently defaulted to 30 fps -- and it cannot flag variable-frame-rate (VFR)
recordings where "frame N" is not at ``N / fps``.

``probe_video_metadata`` cross-checks OpenCV with ``ffprobe`` (whichever the
host has) and returns a reconciled, self-consistent set of numbers plus
human-readable ``warnings`` so the UI can tell the user when a file needs
attention instead of letting the timeline silently disagree with the video.

Key choices:
* ``fps`` prefers ffprobe's measured ``avg_frame_rate`` (the true average, and
  exactly the nominal rate for constant-frame-rate video), then ffprobe's
  ``r_frame_rate``, then OpenCV, then a frames/duration estimate, and only
  falls back to 30 as a last resort -- always with a warning.
* ``duration_ms`` prefers the real container duration so the timeline axis and
  the player's clock share the same length (the playhead reaches the end
  exactly). It falls back to ``total_frames / fps`` when no container duration
  is available.
* ``variable_frame_rate`` is flagged when the nominal and average rates differ,
  or when ``total_frames / fps`` disagrees with the container duration.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Optional

# Hide the transient console window ffprobe/ffmpeg would otherwise flash on
# Windows when launched from a GUI process.
_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

# Relative tolerance (fraction of duration) before a frames/fps vs container
# duration mismatch is treated as "inconsistent timing".
_DURATION_TOLERANCE = 0.02
# Relative tolerance before nominal and average frame rates count as VFR.
_FPS_TOLERANCE = 0.02


def _to_fps(value: object) -> Optional[float]:
    """Parse an ffprobe rate ('30000/1001', '25/1', '29.97') to fps, or None."""
    if value in (None, "", "0/0", "N/A"):
        return None
    try:
        parsed = float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        try:
            parsed = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
    return parsed if parsed > 1e-6 else None


def _to_positive_float(value: object) -> Optional[float]:
    try:
        if value in (None, "", "N/A"):
            return None
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _to_positive_int(value: object) -> Optional[int]:
    try:
        if value in (None, "", "N/A"):
            return None
        parsed = int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _run_ffprobe(path: Path) -> Optional[dict]:
    """Return parsed ffprobe JSON for the first video stream, or None."""
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    try:
        completed = subprocess.run(
            [
                exe,
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries",
                "stream=avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,"
                "duration,codec_name,width,height:format=duration",
                "-of", "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0 or not completed.stdout:
        return None
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None


def _opencv_probe(path: Path) -> dict:
    try:
        import cv2
    except Exception:
        return {}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {}
    try:
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        codec = "".join(chr((fourcc_int >> shift) & 0xFF) for shift in (0, 8, 16, 24)).strip("\x00")
        return {
            "fps": _to_fps(cap.get(cv2.CAP_PROP_FPS)),
            "total_frames": _to_positive_int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            "codec": codec or None,
        }
    finally:
        cap.release()


def _count_frames_opencv(path: Path) -> int:
    try:
        import cv2
    except Exception:
        return 0
    cap = cv2.VideoCapture(str(path))
    counted = 0
    if cap.isOpened():
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            counted += 1
    cap.release()
    return counted


def probe_video_metadata(path: str | Path) -> dict:
    """Return reconciled video metadata.

    Keys: ``fps``, ``total_frames``, ``duration_ms``, ``width``, ``height``,
    ``codec`` (compatible with the previous single-source probe) plus
    ``avg_fps``, ``nominal_fps``, ``container_duration_ms``,
    ``variable_frame_rate`` (True/False/None), ``fps_source``,
    ``duration_source``, ``probe_backends`` and ``warnings`` (list[str]).
    """
    path = Path(path)
    warnings: list[str] = []

    cv = _opencv_probe(path)
    ff = _run_ffprobe(path)
    backends = "opencv" + ("+ffprobe" if ff is not None else "")

    stream = (ff.get("streams") or [{}])[0] if ff else {}
    fmt = ff.get("format", {}) if ff else {}

    avg_fps = _to_fps(stream.get("avg_frame_rate")) if ff else None
    nominal_fps = _to_fps(stream.get("r_frame_rate")) if ff else None
    ff_frames = _to_positive_int(stream.get("nb_frames")) or _to_positive_int(stream.get("nb_read_frames"))
    container_dur_s = _to_positive_float(stream.get("duration")) or _to_positive_float(fmt.get("duration"))

    width = int(cv.get("width") or _to_positive_int(stream.get("width")) or 0)
    height = int(cv.get("height") or _to_positive_int(stream.get("height")) or 0)
    codec = cv.get("codec") or stream.get("codec_name") or "unknown"

    total_frames = ff_frames or cv.get("total_frames")
    if not total_frames:
        total_frames = _count_frames_opencv(path)
        if total_frames:
            warnings.append("Frame count was not in the file header; counted it by decoding the video.")

    # Frame rate: measured average first, then nominal, then OpenCV, then derive.
    fps = avg_fps or nominal_fps or cv.get("fps")
    if avg_fps:
        fps_source = "ffprobe:avg_frame_rate"
    elif nominal_fps:
        fps_source = "ffprobe:r_frame_rate"
    elif cv.get("fps"):
        fps_source = "opencv"
    else:
        fps_source = None

    if not fps and total_frames and container_dur_s:
        fps = float(total_frames) / container_dur_s
        fps_source = "derived:frames/duration"
    if not fps:
        fps = 30.0
        fps_source = "default:30"
        warnings.append(
            "Frame rate could not be read from OpenCV or ffprobe; assumed 30 fps. "
            "If the video and the timeline disagree, re-encode the clip or set the correct fps."
        )

    if container_dur_s:
        duration_ms = int(round(container_dur_s * 1000.0))
        duration_source = "container"
    else:
        duration_ms = int(round((float(total_frames or 0) / max(fps, 1e-6)) * 1000.0))
        duration_source = "frames/fps"
        if ff is not None:
            warnings.append(
                "No container duration was reported; the timeline length is estimated from "
                "frame count and fps and may be slightly off."
            )

    # Variable-frame-rate / timing-consistency detection.
    variable_frame_rate: Optional[bool] = None
    if nominal_fps and avg_fps:
        variable_frame_rate = abs(nominal_fps - avg_fps) / avg_fps > _FPS_TOLERANCE
    if container_dur_s and total_frames and fps:
        implied_s = float(total_frames) / fps
        if abs(implied_s - container_dur_s) > max(2.0 / fps, _DURATION_TOLERANCE * container_dur_s):
            if variable_frame_rate is None:
                variable_frame_rate = True
            warnings.append(
                f"Timing looks inconsistent: {total_frames} frames at {fps:.3f} fps is "
                f"{implied_s:.2f}s, but the container reports {container_dur_s:.2f}s. This usually "
                "means a variable frame rate -- frame numbers will be approximate."
            )

    if variable_frame_rate:
        if nominal_fps and avg_fps:
            headline = (
                f"Variable frame rate detected (nominal {nominal_fps:.3f} fps, average "
                f"{avg_fps:.3f} fps)."
            )
        else:
            headline = "Variable frame rate detected."
        warnings.insert(
            0,
            headline
            + " Playback stays aligned because the timeline uses the true duration, but frame "
            "numbers are approximate. Normalize the clip to a constant frame rate for "
            "frame-accurate annotation and training.",
        )

    if ff is None:
        warnings.append(
            "ffprobe was not found on PATH, so frame rate came from OpenCV only. Install FFmpeg "
            "for a second, authoritative timing source."
        )

    return {
        "fps": float(fps),
        "total_frames": int(total_frames or 0),
        "duration_ms": int(duration_ms),
        "width": int(width),
        "height": int(height),
        "codec": codec or "unknown",
        "avg_fps": float(avg_fps) if avg_fps else float(fps),
        "nominal_fps": float(nominal_fps) if nominal_fps else float(fps),
        "container_duration_ms": int(round(container_dur_s * 1000.0)) if container_dur_s else None,
        "variable_frame_rate": variable_frame_rate,
        "fps_source": fps_source,
        "duration_source": duration_source,
        "probe_backends": backends,
        "warnings": warnings,
    }


def normalize_to_cfr(src: str | Path, dst: str | Path, *, fps: float) -> bool:
    """Re-encode ``src`` to a constant-frame-rate MP4 at ``fps`` via FFmpeg.

    Returns True on success. Requires ``ffmpeg`` on PATH; callers should gate the
    offer on :func:`ffmpeg_available`.
    """
    exe = shutil.which("ffmpeg")
    if not exe:
        return False
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    temp = dst.with_name(f".{dst.stem}.partial{dst.suffix}")
    try:
        completed = subprocess.run(
            [
                exe, "-y",
                "-i", str(src),
                "-vsync", "cfr",
                "-r", f"{float(fps):.6f}",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an",
                str(temp),
            ],
            capture_output=True,
            text=True,
            timeout=60 * 60,
            creationflags=_NO_WINDOW,
        )
        if completed.returncode != 0 or not temp.is_file() or temp.stat().st_size <= 0:
            return False
        temp.replace(dst)
        return True
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        if temp.exists():
            try:
                temp.unlink()
            except OSError:
                pass


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None
