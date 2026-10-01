"""Plugin-native, synchronized motion review without inferred contact labels.

Image-plane angles are descriptive. No clinical norms, animal gait templates,
3D lifting, contact ground truth, or causal coupling fits are implied.
"""
from html import escape
import json
import math
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
import pandas as pd

from .scientific_utils import contiguous_difference

BG = (28, 21, 14)
PANEL = (45, 35, 24)
INK = (238, 239, 233)
MUTED = (170, 165, 150)
COLORS = [(188, 212, 70), (80, 160, 245), (235, 155, 150), (80, 215, 220)]
LOW, HIGH, UNKNOWN = (188, 170, 48), (65, 137, 228), (80, 76, 68)


def _token(value):
    return ''.join(c for c in value.lower() if c.isalnum())


def human_chains(config):
    """Names identify anatomy; never assign anatomy from a numeric index alone."""
    if config.get('SUBJECT_TYPE') != 'human':
        return {}
    names = config['DATASET']['KEYPOINT_ORDER']
    lookup = {_token(n): n for n in names}
    if len(lookup) != len(names):
        raise ValueError('Ambiguous landmark names after normalization.')
    chains = {}
    for side in ('Left', 'Right'):
        keys = [_token(side + joint) for joint in ('Hip', 'Knee', 'Ankle')]
        if all(k in lookup for k in keys):
            chains[side] = [lookup[k] for k in keys]
    return chains


def motion_table(rows, config, fps):
    """One selected identity, full frame grid, no interpolation across missing poses."""
    if rows.empty or rows.track_id.nunique() != 1 or rows.frame.duplicated().any():
        raise ValueError('Motion review requires one nonempty, unique selected track.')
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Motion review requires valid FPS.')
    rows = rows.sort_values('frame').copy()
    if not np.isfinite(rows.frame).all() or (rows.frame % 1 != 0).any() or (rows.frame < 0).any():
        raise ValueError('Frame numbers must be nonnegative integers.')
    # Apply confidence gating again so the public helper is safe independently.
    threshold = config['GENERAL_PARAMS']['KEYPOINT_CONF_THRESHOLD']
    for name in config['DATASET']['KEYPOINT_ORDER']:
        cols = [name + '_x', name + '_y']
        conf = rows[name + '_conf']
        bad = ~np.isfinite(conf) | (conf < threshold) | ~np.isfinite(rows[cols]).all(axis=1)
        rows.loc[bad, cols] = np.nan
    result = pd.DataFrame({'frame': range(int(rows.frame.min()), int(rows.frame.max()) + 1)})
    result['time_s'] = result.frame / fps
    result['track_id'] = int(rows.track_id.iloc[0])
    result['subject_type'] = config.get('SUBJECT_TYPE', 'custom')
    result = result.set_index('frame')
    speed_limit = config['GAIT_ANALYSIS']['PAW_SPEED_THRESHOLD_PX_PER_FRAME']
    for limb in config['GAIT_ANALYSIS']['PAW_ORDER_HILDEBRAND']:
        speed = np.hypot(contiguous_difference(rows, limb + '_x'), contiguous_difference(rows, limb + '_y'))
        values = pd.Series(speed.to_numpy(), index=rows.frame)
        result[limb + '_speed_px_per_frame'] = values
        result[limb + '_motion'] = np.where(result[limb + '_speed_px_per_frame'].isna(), 'unknown',
                                           np.where(result[limb + '_speed_px_per_frame'] < speed_limit, 'low', 'high'))
    for side, chain in human_chains(config).items():
        hip, knee, ankle = [rows[[name + '_x', name + '_y']].to_numpy(float) for name in chain]
        thigh, shank = knee - hip, ankle - knee
        norm = np.linalg.norm(thigh, axis=1) * np.linalg.norm(shank, axis=1)
        valid = np.isfinite(norm) & (norm > 1e-8)
        bend = np.full(len(rows), np.nan)
        bend[valid] = np.degrees(np.arccos(np.clip(np.sum(thigh[valid] * shank[valid], axis=1) / norm[valid], -1, 1)))
        for label, value in [('knee_bend_deg', bend), ('thigh_deg', np.degrees(np.arctan2(thigh[:, 1], thigh[:, 0]))),
                             ('shank_deg', np.degrees(np.arctan2(shank[:, 1], shank[:, 0])))]:
            value[~valid] = np.nan
            result[side + '_' + label] = pd.Series(value, index=rows.frame)
    return result.reset_index(), rows


def _text(canvas, label, xy, size=.6, color=INK, thick=1):
    cv2.putText(canvas, str(label), xy, cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)


def _axes(canvas, box, title, xlabel, ylabel, xlim, ylim):
    x, y, w, h = box
    cv2.rectangle(canvas, (x, y), (x + w, y + h), PANEL, -1)
    _text(canvas, title, (x + 18, y + 30), .65, INK, 1)
    area = (x + 65, y + 57, w - 95, h - 110)
    ax, ay, aw, ah = area
    for i in range(3):
        xx, yy = ax + round(aw * i / 2), ay + round(ah * i / 2)
        cv2.line(canvas, (xx, ay), (xx, ay + ah), (67, 57, 44), 1)
        cv2.line(canvas, (ax, yy), (ax + aw, yy), (67, 57, 44), 1)
        _text(canvas, f'{xlim[0] + (xlim[1]-xlim[0])*i/2:.1f}', (xx - 15, ay + ah + 21), .42, MUTED)
        _text(canvas, f'{ylim[1] - (ylim[1]-ylim[0])*i/2:.0f}', (ax - 45, yy + 5), .42, MUTED)
    _text(canvas, xlabel, (ax + 30, y + h - 12), .43, MUTED)
    _text(canvas, ylabel, (ax, y + 49), .43, MUTED)
    return area


def _point(x, y, area, xlim, ylim):
    ax, ay, aw, ah = area
    return (round(ax + aw * (x-xlim[0]) / (xlim[1]-xlim[0])),
            round(ay + ah * (ylim[1]-y) / (ylim[1]-ylim[0])))


def _trace(canvas, xs, ys, area, xlim, ylim, color, angular=False):
    previous = None
    for x, y in zip(xs, ys):
        if not (np.isfinite(x) and np.isfinite(y)):
            previous = None
            continue
        point = _point(x, y, area, xlim, ylim)
        # Wrapped orientation jumps are gaps, not diagonal sweeps.
        if previous is not None and (not angular or max(abs(x-previous[1]), abs(y-previous[2])) < 180):
            cv2.line(canvas, previous[0], point, color, 2, cv2.LINE_AA)
        previous = (point, x, y)


def render_frame(source, frame_id, table, rows, config, fps):
    """Shared renderer used by GUI/CLI exports; no desktop capture or tutorial edits."""
    canvas = np.full((900, 1600, 3), BG, np.uint8)
    subject = config.get('SUBJECT_TYPE', 'custom')
    chains = human_chains(config)
    limbs = config['GAIT_ANALYSIS']['PAW_ORDER_HILDEBRAND']
    current = table.loc[table.frame == frame_id]
    row = rows.loc[rows.frame == frame_id]
    _text(canvas, 'INTEGRAPOSE  /  ' + subject.upper() + ' GAIT', (32, 40), .8, INK, 2)
    _text(canvas, 'Plugin-generated motion review | 2D estimates', (32, 71), .57, MUTED)
    _text(canvas, f'Input frame {frame_id}  |  {frame_id/fps:.3f} s  |  Track {int(table.track_id.iloc[0])}', (880, 42), .58)
    if subject == 'human':
        _text(canvas, 'LEFT', (1250, 73), .55, COLORS[0])
        _text(canvas, 'RIGHT', (1390, 73), .55, COLORS[1])
    # Fit the complete input image; no inferred off-camera anatomy is drawn.
    sw, sh = source.shape[1], source.shape[0]
    scale = min(820 / sw, 470 / sh)
    w, h = round(sw * scale), round(sh * scale)
    x0, y0 = 32 + (820-w)//2, 100 + (470-h)//2
    canvas[y0:y0+h, x0:x0+w] = cv2.resize(source, (w, h))
    if not row.empty:
        r = row.iloc[0]
        draw_chains = [(0 if side == 'Left' else 1, chain) for side, chain in chains.items()] if chains else list(enumerate([[limb] for limb in limbs]))
        for k, chain in draw_chains:
            previous = None
            for name in chain:
                x, y = r[name + '_x'], r[name + '_y']
                if not (np.isfinite(x) and np.isfinite(y)) or not (0 <= x < sw and 0 <= y < sh):
                    previous = None
                    continue
                p = (round(x0 + x*scale), round(y0+y*scale))
                if previous is not None:
                    cv2.line(canvas, previous, p, COLORS[k % 4], 4, cv2.LINE_AA)
                cv2.circle(canvas, p, 6, COLORS[k % 4], -1, cv2.LINE_AA)
                previous = p
    first, last = table.time_s.iloc[0], table.time_s.iloc[-1]
    xlim = (first, max(first + 1/fps, last))
    _text(canvas, 'Hildebrand-style motion bands', (32, 611), .72)
    _text(canvas, 'Low motion / High motion / Unknown  |  contact not verified', (32, 641), .5, MUTED)
    start, n = int(table.frame.iloc[0]), len(table)
    row_h = min(36, 145 / max(1, len(limbs)))
    for i, limb in enumerate(limbs):
        yy = round(660 + i*row_h)
        _text(canvas, limb[:26], (32, yy+17), min(.5, row_h/45), COLORS[i % 4])
        for j, state in enumerate(table[limb + '_motion']):
            a, b = 260 + round(580*j/n), 260 + round(580*(j+1)/n)
            cv2.rectangle(canvas, (a, yy), (max(a,b-1), round(yy+row_h-6)), {'low':LOW,'high':HIGH,'unknown':UNKNOWN}[state], -1)
    xx = 260 + round(580*(frame_id-start+.5)/n)
    cv2.line(canvas, (xx, 655), (xx, round(665+row_h*len(limbs))), INK, 2)
    _text(canvas, f'{first:.2f} s', (260, 834), .5, MUTED)
    _text(canvas, f'{last:.2f} s', (760, 834), .5, MUTED)
    if chains:
        area = _axes(canvas, (880, 100, 686, 325), 'Knee bend in the image plane', 'Input time (s)', 'Degrees | straight = 0', xlim, (0,180))
        for side in chains:
            i = 0 if side == 'Left' else 1
            _trace(canvas, table.time_s, table[side+'_knee_bend_deg'], area, xlim, (0,180), COLORS[i])
            if not current.empty and np.isfinite(current.iloc[0][side+'_knee_bend_deg']):
                cv2.circle(canvas, _point(frame_id/fps,current.iloc[0][side+'_knee_bend_deg'],area,xlim,(0,180)), 6,COLORS[i],-1,cv2.LINE_AA)
        cursor = _point(frame_id/fps,0,area,xlim,(0,180))[0]
        cv2.line(canvas,(cursor,area[1]),(cursor,area[1]+area[3]), INK,1)
        area = _axes(canvas,(880,445,686,370),'Thigh-shank coordination','Thigh orientation (deg)','Shank orientation (deg)',(-180,180),(-180,180))
        trail = table[(table.frame <= frame_id) & (table.frame >= frame_id-round(fps))]
        for side in chains:
            i = 0 if side == 'Left' else 1
            _trace(canvas,trail[side+'_thigh_deg'],trail[side+'_shank_deg'],area,(-180,180),(-180,180),COLORS[i],True)
            if not current.empty:
                x,y=current.iloc[0][side+'_thigh_deg'], current.iloc[0][side+'_shank_deg']
                if np.isfinite(x) and np.isfinite(y):
                    cv2.circle(canvas,_point(x,y,area,(-180,180),(-180,180)),6,COLORS[i],-1,cv2.LINE_AA)
        _text(canvas, 'Orientation: +x right, +y down | trail: 1 input second', (897,840),.48,MUTED)
    else:
        _text(canvas,'Configured limb motion', (900,150),.8)
        _text(canvas,'Human joint angles are disabled.',(900,195),.58,MUTED)
        _text(canvas,'Review anatomy, visibility and motion thresholds.',(900,230),.5,MUTED)
        for i,limb in enumerate(limbs):
            if i >= 10: break
            state = current.iloc[0][limb+'_motion'] if not current.empty else 'unknown'
            _text(canvas,limb+': '+state,(900,290+i*40),.57,COLORS[i%4])
    _text(canvas, 'Image motion is not foot contact. Missing poses remain gaps. Review occlusion and left/right identity.', (32,878), .57, MUTED)
    return canvas


def export_motion_review(rows, config, video_path, output_dir, fps):
    """Export full measurements plus a bounded, synchronized video and HTML player."""
    output = Path(output_dir)
    table, gated = motion_table(rows, config, fps)
    table.to_csv(output/'limb_motion_series.csv', index=False)
    first = int(table.frame.iloc[0])
    last = min(int(table.frame.iloc[-1]), first + max(1,math.floor(30*fps))-1)
    visible = table[table.frame.between(first,last)]
    subject = config.get('SUBJECT_TYPE','custom')
    warnings = []
    if subject == 'human' and len(human_chains(config)) < 2:
        warnings.append('Both named Hip-Knee-Ankle chains are needed for bilateral human joint plots; unavailable sides are omitted.')
    meta = dict(subject_type=subject, generated_by='IntegraPose gait plugin', fps=fps,
                input_video=str(Path(video_path).resolve()), frame_origin='zero-based input video',
                track_id=int(table.track_id.iloc[0]), first_frame=first, last_frame=last,
                full_table_last_frame=int(table.frame.iloc[-1]), contact_validated=False,
                angle_convention='2D knee bend: angle between hip-to-knee and knee-to-ankle; straight=0 deg. Segment orientation: atan2(dy,dx), image +x right and +y down, [-180,180].',
                warnings=warnings)
    ffmpeg = shutil.which('ffmpeg')
    media = ''
    cap = cv2.VideoCapture(str(video_path))
    proc = None
    target = output/'limb_motion_review.mp4'
    temporary = output/'limb_motion_review.tmp.mp4'
    try:
        if not cap.isOpened(): raise RuntimeError('Could not open motion-review source video.')
        if ffmpeg:
            proc = subprocess.Popen([ffmpeg,'-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1600x900','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
        else:
            warnings.append('FFmpeg is unavailable: tables and a static preview exported; install FFmpeg for synchronized MP4 export.')
        # Sequential decode: seeking on some sparse laboratory AVI files is unreliable.
        for f in range(last+1):
            ok, frame = cap.read()
            if not ok: raise RuntimeError(f'Source decode stopped before frame {f}.')
            if f < first: continue
            picture = render_frame(frame,f,visible,gated,config,fps)
            if f == (first+last)//2:
                if not cv2.imwrite(str(output/'limb_motion_preview.png'),picture):
                    raise RuntimeError('Could not write motion preview.')
            if proc: proc.stdin.write(picture.tobytes())
        if proc:
            proc.stdin.close()
            if proc.wait(timeout=180) != 0: raise RuntimeError('FFmpeg motion export failed.')
            check=cv2.VideoCapture(str(temporary))
            valid=check.isOpened() and int(check.get(cv2.CAP_PROP_FRAME_COUNT)) == last-first+1
            check.release()
            if not valid: raise RuntimeError('Motion export failed frame-count verification.')
            temporary.replace(target)
            media='<video id="v" controls preload="metadata" src="limb_motion_review.mp4"></video><p>Playback: <button data-rate="0.25">0.25x</button> <button data-rate="0.5">0.5x</button> <button data-rate="1">1x</button> <span id="rate">1x</span> | Plot time and units always refer to the input video.</p><a href="limb_motion_review.mp4" download>Download video</a>'
        else:
            media='<img src="limb_motion_preview.png" alt="Static motion review">'
    finally:
        cap.release()
        if proc and proc.poll() is None:
            proc.kill(); proc.wait()
        temporary.unlink(missing_ok=True)
    (output/'limb_motion_review.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>__SUBJECT__ limb motion review</title><style>body{margin:24px auto;padding:0 20px;max-width:1600px;background:#0e151c;color:#e9edef;font:17px "Segoe UI",sans-serif}h1{font-size:30px}video,img{width:100%;max-height:76vh;background:#0e151c}a{color:#70d4bc}button{padding:8px 18px;background:#253746;color:white;border:1px solid #617786;border-radius:6px;cursor:pointer}p{line-height:1.6;color:#b7c4cd}details{margin:20px 0}</style><h1>__SUBJECT__ limb motion review</h1><p>Generated by the IntegraPose gait plugin. __NAME__ | Track __TRACK__</p>__MEDIA__<p><a href="limb_motion_series.csv" download>Download full measurement table</a> · <a href="limb_motion_review.json" download>Measurement definitions</a> · <a href="gait_review_report.html">Return to gait report</a></p><details open><summary>Reading the visuals</summary><p>The Hildebrand-style bands show low/high image motion, not verified stance/swing. Gray means missing or unavailable. Threshold: __THRESHOLD__ px/frame. Human knee bend uses hip, knee and ankle points; it is an unsigned image-plane angle, not laboratory 3D flexion. The coordination trail compares thigh and shank orientations, not causal coupling. Orientation axes follow the image: right is +x and down is +y. Missing data and angle-wrap jumps break the trace. Review left/right swaps, cropped landmarks and occlusion before interpretation.</p><p>The video shows up to 30 input seconds from the first selected detection. The CSV contains the entire selected interval. Playback speed does not change measured time. A cropped input's frame numbers are relative to that input. No reference-dataset measurements are substituted. __WARNINGS__</p></details><script>document.querySelectorAll('[data-rate]').forEach(b=>b.onclick=()=>{let v=document.getElementById('v');if(v){v.playbackRate=Number(b.dataset.rate);document.getElementById('rate').textContent=b.dataset.rate+'x'}});</script></html>'''
    for key,value in {'__SUBJECT__':subject.title(),'__NAME__':Path(video_path).name,'__TRACK__':meta['track_id'],'__THRESHOLD__':config['GAIT_ANALYSIS']['PAW_SPEED_THRESHOLD_PX_PER_FRAME'],'__WARNINGS__':' '.join(warnings)}.items():
        page=page.replace(key,escape(str(value)))
    page=page.replace('__MEDIA__',media)
    (output/'limb_motion_review.html').write_text(page,encoding='utf-8')
    return {'motion_review':str(output/'limb_motion_review.html'),'motion_warnings':warnings}
