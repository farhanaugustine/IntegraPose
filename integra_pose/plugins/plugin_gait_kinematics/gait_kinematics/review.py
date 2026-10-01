"""Offline gait review: consecutive-stride bouts, visuals and optional clips.

Uses the plugin's validated stride table, independently of coordination models.
"""
from html import escape
import json
import logging
from pathlib import Path
import shutil
import subprocess
from uuid import uuid4

import cv2
import numpy as np
import pandas as pd

from .scientific_utils import contiguous_difference
from .coordination import export_motion_review

logger = logging.getLogger(__name__)
BOUT_COLUMNS = ["bout_id", "track_id", "start_frame", "end_frame", "duration_s", "stride_count", "clip"]


def build_bouts(strides, fps):
    """Join only overlapping/touching accepted strides of the same track."""
    details = strides.sort_values(["track_id", "start_frame", "end_frame"]).reset_index(drop=True).copy()
    details["stride_id"] = [f"S{i + 1:04d}" for i in range(len(details))]
    details["bout_id"] = ""
    bouts = []
    for idx, row in details.iterrows():
        if not bouts or row.track_id != bouts[-1]["track_id"] or row.start_frame > bouts[-1]["end_frame"]:
            bouts.append({"bout_id": f"B{len(bouts) + 1:03d}", "track_id": int(row.track_id),
                          "start_frame": int(row.start_frame), "end_frame": int(row.end_frame),
                          "stride_count": 0, "clip": ""})
        bout = bouts[-1]
        bout["end_frame"] = max(bout["end_frame"], int(row.end_frame))
        bout["stride_count"] += 1
        bout["duration_s"] = (bout["end_frame"] - bout["start_frame"]) / fps
        details.at[idx, "bout_id"] = bout["bout_id"]
    return details, pd.DataFrame(bouts, columns=BOUT_COLUMNS)


def _clip(video, output, start, end, fps, rows, paws, *, names=None, edges=None):
    """Exact decoded frame range, inclusive endpoints; visible source frame ID."""
    cap = cv2.VideoCapture(str(video))
    writer = None
    written = 0
    expected = end - start + 1
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("MJPEG clip writer is unavailable.")
        points = rows.set_index("frame")
        colors = [(190, 110, 30), (90, 170, 40), (70, 100, 220), (190, 60, 180)]
        for frame_id in range(start, end + 1):
            ok, frame = cap.read()
            if not ok:
                break
            if frame_id in points.index:
                row = points.loc[frame_id]
                for a, b in edges or []:
                    ax, ay = row[f'{names[a]}_x'], row[f'{names[a]}_y']
                    bx, by = row[f'{names[b]}_x'], row[f'{names[b]}_y']
                    if all(np.isfinite(v) for v in (ax, ay, bx, by)):
                        cv2.line(frame, (round(ax), round(ay)), (round(bx), round(by)), (190, 210, 190), 2)
                for i, paw in enumerate(paws):
                    x, y = row[f"{paw}_x"], row[f"{paw}_y"]
                    if np.isfinite(x) and np.isfinite(y):
                        cv2.circle(frame, (round(x), round(y)), 5, colors[i % len(colors)], 2)
                        if names:
                            cv2.putText(frame, str(names.index(paw)), (round(x) + 7, round(y)), cv2.FONT_HERSHEY_SIMPLEX, .45, colors[i % len(colors)], 1, cv2.LINE_AA)
            cv2.putText(frame, f"Source frame {frame_id} | {frame_id / fps:.3f} s", (12, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, f"Source frame {frame_id} | {frame_id / fps:.3f} s", (12, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(frame)
            written += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if written != expected:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"Incomplete clip: decoded {written} of {expected} expected frames.")
    check = cv2.VideoCapture(str(output))
    try:
        if not check.isOpened() or int(check.get(cv2.CAP_PROP_FRAME_COUNT)) != expected:
            raise RuntimeError("Written clip did not pass frame-count verification.")
    finally:
        check.release()
    # Optional H.264 makes clips playable in common browsers; AVI remains a
    # usable download when ffmpeg is not installed or encoding fails.
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        mp4 = output.with_suffix(".mp4")
        try:
            subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(output), "-an",
                            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)],
                           check=True, capture_output=True, timeout=120)
            check = cv2.VideoCapture(str(mp4))
            try:
                valid = check.isOpened() and int(check.get(cv2.CAP_PROP_FRAME_COUNT)) == expected
            finally:
                check.release()
            if valid:
                output.unlink()
                return mp4
        except (OSError, subprocess.SubprocessError):
            logger.warning("H.264 encoding unavailable; retaining AVI review clip.")
        mp4.unlink(missing_ok=True)
    return output


def phase_svg(rows, paws, threshold, fps):
    """Frame-resolution motion raster; missing samples remain unknown."""
    if rows.empty:
        return "<p>No observed poses in this interval.</p>"
    rows = rows.sort_values(["track_id", "frame"])
    first, last = int(rows.frame.min()), int(rows.frame.max())
    span = max(1, last - first + 1)
    height = 38 * len(paws) + 56
    pieces = [f'<svg viewBox="0 0 1000 {height}" role="img" aria-label="Limb motion timeline">']
    for i, paw in enumerate(paws):
        y = 38 * i + 12
        pieces.append(f'<text x="0" y="{y + 17}">{escape(paw)}</text>')
        pieces.append(f'<rect x="185" y="{y}" width="800" height="24" fill="#d6dde3"/>')
        speed = np.hypot(contiguous_difference(rows, f"{paw}_x"), contiguous_difference(rows, f"{paw}_y"))
        states = np.where(speed.isna(), "unknown", np.where(speed < threshold, "low", "high"))
        # Run-length compression retains phase boundaries without embedding a
        # rectangle per frame for long recordings.
        runs = []
        for frame, state in zip(rows.frame, states):
            frame = int(frame)
            if runs and runs[-1][2] == state and frame == runs[-1][1] + 1:
                runs[-1][1] = frame
            else:
                runs.append([frame, frame, state])
        for begin, end, state in runs:
            color = {"unknown": "#d6dde3", "low": "#2f7f91", "high": "#d77b44"}[state]
            x, width = 185 + 800 * (begin - first) / span, 800 * (end - begin + 1) / span
            pieces.append(f'<rect x="{x:.3f}" y="{y}" width="{width:.3f}" height="24" fill="{color}">'
                          f'<title>{escape(paw)}: {state} motion, frames {begin}-{end}</title></rect>')
    pieces.append(f'<text x="185" y="{height - 8}">{first / fps:.2f} s</text>'
                  f'<text x="985" y="{height - 8}" text-anchor="end">{last / fps:.2f} s (source time)</text></svg>')
    return "".join(pieces)


def scatter_svg(strides):
    values = strides[["stride_duration_s", "stride_speed_px_per_s"]].dropna()
    if values.empty:
        return "<p>No complete duration/speed pairs to plot.</p>"
    max_x = max(.01, float(values.stride_duration_s.max()))
    max_y = max(1., float(values.stride_speed_px_per_s.max()))
    pieces = ['<svg viewBox="0 0 900 250" role="img" aria-label="Stride duration and body speed">',
              '<path d="M70 15V210H875" stroke="#73818b" fill="none"/>',
              '<text x="450" y="245">Stride duration (s)</text>',
              '<text x="12" y="180" transform="rotate(-90 12 180)">Body speed (px/s)</text>']
    for i in range(5):
        pieces.append(f'<text x="{70 + 800 * i / 4}" y="228">{max_x * i / 4:.2f}</text>')
        pieces.append(f'<text x="20" y="{210 - 190 * i / 4}">{max_y * i / 4:.1f}</text>')
    for duration, speed in values.itertuples(index=False, name=None):
        pieces.append(f'<circle cx="{70 + 800 * duration / max_x:.2f}" cy="{210 - 190 * speed / max_y:.2f}"'
                      f' r="4" fill="#2f7f91" opacity=".65"><title>{duration:.3f} s; {speed:.2f} px/s</title></circle>')
    return "".join(pieces) + "</svg>"


def sequence_svg(strides, column, label):
    values = strides[column].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return "<p>No complete values for this measurement.</p>"
    maximum = max(.001, float(finite.max()))
    pieces = ['<svg viewBox="0 0 900 240" role="img" aria-label="Stride measurement sequence">',
              '<path d="M70 20V200H870" stroke="#73818b" fill="none"/>',
              f'<text x="75" y="16">{escape(label)} (maximum {maximum:.3f})</text>',
              '<text x="380" y="233">Accepted stride index</text>']
    for index, value in enumerate(values):
        if np.isfinite(value):
            x, y = 85 + 770 * index / max(1, len(values) - 1), 200 - 175 * value / maximum
            pieces.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#2f7f91"><title>Stride {index + 1}: {value:.3f}</title></circle>')
    return ''.join(pieces) + '</svg>'


def write_review(rows, strides, config, video_path, output_dir, fps, total_frames):
    output = Path(output_dir)
    details, bouts = build_bouts(strides, fps)
    paws = config["GAIT_ANALYSIS"]["PAW_ORDER_HILDEBRAND"]
    review = config["REVIEW"]
    warnings = []
    motion_artifacts = {}
    motion_html = ''
    if review.get('EXPORT_COORDINATION', False):
        try:
            motion_artifacts = export_motion_review(rows, config, video_path, output, fps)
            warnings.extend(motion_artifacts.get('motion_warnings', []))
            motion_html = '<section class="card"><h2>Synchronized limb motion</h2><p><a href="limb_motion_review.html">Open plugin-generated motion player</a> · <a href="limb_motion_series.csv" download>Download motion measurements</a></p><p>Includes configured limb-motion bands and, for mapped human legs, image-plane knee bend and thigh–shank coordination. Contact is unverified.</p></section>'
        except Exception as exc:
            warnings.append(f'Motion visual unavailable ({exc})')
            logger.exception('Synchronized motion export failed')
    if review["EXPORT_CLIPS"] and not bouts.empty and review["MAX_CLIPS"]:
        assets = output / ("gait_clips_" + uuid4().hex[:8])
        assets.mkdir()
        padding = round(review["PADDING_SECONDS"] * fps)
        for index, bout in bouts.head(review["MAX_CLIPS"]).iterrows():
            # Bound export cost on long recordings. Tables/plots retain the
            # full bout; exported preview starts at the first stride.
            start = max(0, int(bout.start_frame) - padding)
            end = min(total_frames - 1, int(bout.end_frame) + padding, start + round(30 * fps) - 1)
            try:
                path = _clip(video_path, assets / f'{bout.bout_id}_f{start}-{end}.avi', start, end, fps, rows, paws,
                             names=config['DATASET']['KEYPOINT_ORDER'], edges=config['DATASET'].get('SKELETON_EDGES', []))
                bouts.at[index, "clip"] = path.relative_to(output).as_posix()
            except Exception as exc:
                warnings.append(f"{bout.bout_id}: clip unavailable ({exc})")
                logger.warning(warnings[-1])
    details.to_csv(output / "gait_stride_details.csv", index=False)
    mapping = pd.DataFrame({'model_index': range(len(config['DATASET']['KEYPOINT_ORDER'])),
                            'landmark': config['DATASET']['KEYPOINT_ORDER']})
    mapping['gait_landmark'] = mapping.landmark.isin(paws)
    mapping.to_csv(output / 'landmark_mapping.csv', index=False)
    bouts.to_csv(output / "gait_bout_summary.csv", index=False)
    (output / "analysis_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    metadata = {"video": str(Path(video_path).resolve()), "fps": fps, "total_video_frames": total_frames,
                "selected_track_id": int(rows.track_id.iloc[0]), "first_analyzed_frame": int(rows.frame.min()),
                "last_analyzed_frame": int(rows.frame.max()), "observed_frames": len(rows),
                "stride_count": len(strides), "bout_count": len(bouts), "warnings": warnings}
    (output / "gait_review.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    threshold = config["GAIT_ANALYSIS"]["PAW_SPEED_THRESHOLD_PX_PER_FRAME"]
    cards = []
    for bout in bouts.itertuples(index=False):
        excerpt = rows[rows.frame.between(bout.start_frame, bout.end_frame)]
        media = "<p>Clip export disabled, unavailable, or beyond the configured clip limit.</p>"
        if bout.clip:
            url = escape(bout.clip, quote=True)
            media = f'<a href="{url}" download>Download review clip</a>'
            if bout.clip.endswith(".mp4"):
                media = f'<video controls preload="none" src="{url}"></video>' + media
            else:
                media += "<p>AVI: open the downloaded clip in a video player.</p>"
        cards.append(f'<details class="card"><summary>{bout.bout_id} · {bout.stride_count} strides · '
                     f'{bout.duration_s:.2f} s · source frames {bout.start_frame}–{bout.end_frame}</summary>'
                     + phase_svg(excerpt, paws, threshold, fps) + media + '</details>')
    display = details.rename(columns={"paw": "Reference landmark", "stride_duration_s": "Duration (s)",
                                     "stride_speed_px_per_s": "Body speed (px/s)", "stride_length": "Stride displacement (px)",
                                     "step_length": "Step displacement (px)", "step_width": "Step width proxy (px)"})
    columns = ["stride_id", "bout_id", "start_frame", "end_frame", "Reference landmark", "Duration (s)",
               "Body speed (px/s)", "Stride displacement (px)", "Step displacement (px)", "Step width proxy (px)"]
    measures = config['GAIT_ANALYSIS'].get('MEASURES', ['stride_timing', 'body_motion', 'spatial_proxies'])
    if 'spatial_proxies' not in measures:
        columns = [c for c in columns if c not in ['Stride displacement (px)', 'Step displacement (px)', 'Step width proxy (px)']]
    if 'body_motion' not in measures:
        columns.remove('Body speed (px/s)')
    if 'stride_timing' not in measures:
        columns.remove('Duration (s)')
    table = display[columns].to_html(index=False, escape=True, na_rep="—", float_format=lambda v: f"{v:.3f}")
    warning_html = "".join(f"<p>{escape(w)}</p>" for w in warnings)
    report = _PAGE.replace("__TITLE__", escape(Path(video_path).name))
    report = report.replace("__PROFILE__", escape(config.get('SUBJECT_TYPE', 'custom').title() + ' analysis · ' + str(config["PROFILE_NAME"])))
    report = report.replace("__MOTION_REVIEW__", motion_html)
    report = report.replace("__STATS__", f'{len(strides)} strides · {len(bouts)} review bouts · {fps:g} FPS · track {metadata["selected_track_id"]}')
    report = report.replace("__WARNINGS__", warning_html)
    if 'stride_timing' in measures and 'body_motion' in measures:
        plot, plot_title = scatter_svg(strides), 'Stride duration and body speed'
    else:
        column, plot_title = ('stride_duration_s', 'Stride duration (s)') if 'stride_timing' in measures else (
            ('stride_speed_px_per_s', 'Body speed (px/s)') if 'body_motion' in measures else ('stride_length', 'Stride displacement (px)'))
        plot = sequence_svg(strides, column, plot_title)
    report = report.replace("__PLOT__", plot).replace('__PLOT_TITLE__', plot_title)
    report = report.replace("__TIMELINE__", phase_svg(rows, paws, threshold, fps))
    report = report.replace("__BOUTS__", "".join(cards) or "<p>No accepted strides. Review landmark quality, subject selection and thresholds.</p>")
    report = report.replace("__TABLE__", table)
    report = report.replace("__METHOD__", escape(config["GAIT_ANALYSIS"]["GAIT_DETECTION_METHOD"]))
    report = report.replace("__MAPPING__", mapping.to_html(index=False, escape=True))
    report = report.replace("__MEASURES__", escape(', '.join(m.replace('_', ' ') for m in measures)))
    report = report.replace("__THRESHOLD__", f"{threshold:g}")
    path = output / "gait_review_report.html"
    path.write_text(report, encoding="utf-8")
    return {"review_report": str(path), "stride_details": str(output / "gait_stride_details.csv"),
            "bout_summary": str(output / "gait_bout_summary.csv"), "review_warnings": warnings, **motion_artifacts}


_PAGE = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Gait review — __TITLE__</title>
<style>
*{box-sizing:border-box}body{margin:0;background:linear-gradient(135deg,#f8f4ee,#eaf3f4);color:#243447;font:16px "Segoe UI",sans-serif}
main{max-width:1400px;margin:30px auto;padding:0 22px}.card,header{background:#ffffffed;border:1px solid #dde3e4;border-radius:18px;padding:24px;margin:18px 0;box-shadow:0 8px 28px #3040500b}
h1{font-size:2.3rem;margin:10px 0}h2{margin-top:0}p{line-height:1.6;color:#536271}.eyebrow{color:#2f7f91;font-weight:700;letter-spacing:.12em}.stats{font-size:1.35rem;font-weight:650}
a{color:#236e81}nav{display:flex;gap:22px;flex-wrap:wrap}svg{width:100%;height:auto;margin:16px 0}svg text{font:14px "Segoe UI",sans-serif;fill:#334454}
video{display:block;width:min(100%,760px);max-height:600px;margin:16px 0;background:#182a36}summary{cursor:pointer;font-weight:650;line-height:1.6}.scroll{overflow:auto;max-height:600px}
table{border-collapse:collapse;width:100%;white-space:nowrap;font-size:14px}th,td{padding:12px;text-align:right;border-bottom:1px solid #dde3e4}th{position:sticky;top:0;background:#eaf2f3}
input{font:inherit;padding:10px;border:1px solid #aebec5;border-radius:8px;width:min(100%,460px)}.legend span{margin-right:18px}.low{color:#2f7f91}.high{color:#aa5928}
</style><main><header><div class="eyebrow">INTEGRAPOSE · GAIT REVIEW</div><h1>__TITLE__</h1>
<p>__PROFILE__ · Detection method: __METHOD__</p><p class="stats">__STATS__</p>
<nav><a href="#timeline">Limb motion</a><a href="#bouts">Bout review</a><a href="#strides">Stride table</a>
<a href="gait_stride_details.csv" download>Download strides</a><a href="gait_bout_summary.csv" download>Download bouts</a>
<a href="analysis_config.json" download>Analysis settings</a></nav></header>
<section class="card"><h2>How to read this report</h2><p>Events are estimates from image motion and need visual review.
An ankle landmark does not measure heel strike or toe contact directly. Distances are 2D pixel displacements; perspective, treadmill motion and camera movement affect their interpretation.
Body speed uses the detection-box center. No physical-unit or clinical interpretation is implied.</p>
<p>Bouts join touching or overlapping accepted reference-limb strides. They are review segments, not validated walking classifications.
Clip previews include up to 0.5 seconds of padding by default and are capped at 30 seconds; the source frame number is overlaid. No audio is exported.</p>__WARNINGS__</section>
<details class="card"><summary>Selected measurements and model landmark mapping</summary><p>__MEASURES__</p><p>Overlay numbers are zero-based model indices. <a href="landmark_mapping.csv" download>Download mapping</a></p><div class="scroll">__MAPPING__</div></details>
__MOTION_REVIEW__
<section class="card" id="timeline"><h2>Limb motion timeline</h2>
<p class="legend"><span class="low">■ Low motion</span><span class="high">■ High motion</span><span>■ Gray: missing or unavailable</span></p>
<p>Threshold: __THRESHOLD__ px/frame. These motion bands provide a review aid for either detection method; they are not independently verified stance/swing labels.</p>__TIMELINE__</section>
<section class="card"><h2>__PLOT_TITLE__</h2>__PLOT__</section>
<section id="bouts"><h2>Review bouts</h2><p>Expand a bout to inspect its limb-motion pattern and video clip.</p>__BOUTS__</section>
<section class="card" id="strides"><h2>Stride details</h2><label>Filter table <input id="filter" placeholder="Search a stride, bout or landmark"></label>
<p>Frames are zero-based source frames. Elapsed stride time is (end − start) / FPS. A dash means the metric is unavailable.</p><div class="scroll">__TABLE__</div></section>
</main><script>document.getElementById('filter').addEventListener('input',function(){const q=this.value.toLowerCase();document.querySelectorAll('#strides tbody tr').forEach(r=>{r.hidden=!r.textContent.toLowerCase().includes(q)});});</script></html>'''
