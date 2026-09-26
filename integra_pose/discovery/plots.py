from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget

from .data import fragments, occupancy


def color(label):
    from matplotlib import colormaps
    import hashlib
    if str(label) == '-1':
        return (0.58, 0.64, 0.72, 1.)
    return colormaps['tab20'](int(hashlib.sha256(str(label).encode()).hexdigest()[:8], 16) % 20)


class Plot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(7, 4), layout='constrained', facecolor='#f8fafc')
        self.canvas = FigureCanvasQTAgg(self.figure)
        QVBoxLayout(self).addWidget(self.canvas)
        self.records = []
        self.construct = {}

    def axes(self, three=False):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111, projection='3d' if three else None)
        return self.ax

    def export(self, path, context):
        path = Path(path)
        self.figure.savefig(path, dpi=180)
        path.with_suffix('.json').write_text(json.dumps(dict(context=context, plot=self.construct), indent=2), encoding='utf-8')
        rows = self.records
        with path.with_suffix('.csv').open('w', newline='', encoding='utf-8') as stream:
            if rows:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    def feature(self, rows, payload, feature, a, b, mode, anchor=0):
        ax = self.axes()
        self.records = []
        item = payload['feature_catalog'][feature]
        for label in dict.fromkeys((a, b)):
            selected = [r for r in rows if r['review_label'] == label]
            selected.sort(key=lambda r: (r['source'], r['track'], r['frame']))
            if not selected:
                continue
            for row in selected:
                self.records.append(dict(row_id=row['row_id'], source=row['source'], track=row['track'],
                                         frame=row['frame'], label=label, value=row['features'][feature],
                                         plot_x=row['features'][feature] if mode=='Distribution' else None,
                                         time_seconds=row['frame']/payload['sources'][row['source']]['fps']))
            values = [r['features'][feature] for r in selected]
            if mode == 'Distribution':
                ax.hist(values, bins=30, alpha=.5, label=f'{label}: {len(values)} observations', color=color(label), density=False)
                ax.set(xlabel=f"{item['name']} ({item['units']})", ylabel='Observed detections (not independent replicates)')
            else:
                for fragment in fragments(selected, max_gap=1):
                    ids = set(fragment['rows'])
                    part = [r for r in selected if r['row_id'] in ids]
                    fps = payload['sources'][fragment['source']]['fps']
                    x = np.asarray([r['frame'] for r in part], float)
                    if mode == 'Interval-relative time':
                        x = (x - anchor) / fps
                    elif mode == 'Bout progress (%)':
                        span = fragment['end'] - fragment['start']
                        x = (x-fragment['start']) / max(span, 1) * 100
                    else:
                        x /= fps
                    positions=dict(zip((r['row_id'] for r in part),x.tolist()))
                    for record in self.records:
                        if record['row_id'] in positions:
                            record['plot_x']=positions[record['row_id']]
                    ax.plot(x, [r['features'][feature] for r in part], '.-', ms=2, lw=.7, color=color(label))
                ax.plot([], [], color=color(label), label=label)
                ax.set(xlabel=mode + (' (seconds)' if mode != 'Bout progress (%)' else ''), ylabel=f"{item['name']} ({item['units']})")
        if self.records:
            ax.legend()
        ax.set_title('Clustering input values: imputed defaults may be present\nDescriptive comparison, not independent validation', fontsize=10)
        self.construct = dict(kind='feature', feature=feature, clusters=[a, b], alignment=mode, anchor_frame=anchor)
        self.canvas.draw_idle()

    def spatial(self, rows, payload, keypoint, a, b, heatmap=False):
        self.figure.clear()
        self.records = []
        heatmaps = []
        for number, label in enumerate((a, b), 1):
            ax = self.figure.add_subplot(1, 2, number)
            values = []
            for row in rows:
                if row['review_label'] != label:
                    continue
                kp = row['keypoints'].get(keypoint)
                if not kp or any(v is None for v in kp) or kp[2] < float(payload['params'].get('conf_threshold', .3)):
                    continue
                source = payload['sources'][row['source']]
                xy = list(kp[:2])
                # read_detections retains coordinate space; normalize only declared pixel sources.
                references = payload.get('diagnostics', {}).get('source_normalization', [])
                ref = next((v for v in references if v.get('directory') == source['directory']), {})
                if ref.get('coordinate_space') == 'pixel':
                    xy = [xy[0]/source['width'], xy[1]/source['height']]
                item = dict(row_id=row['row_id'], label=label, source=row['source'], track=row['track'],
                            frame=row['frame'], x=xy[0], y=xy[1], seconds=1/source['fps'])
                values.append(item)
                self.records.append(item)
            if values:
                if heatmap:
                    hist = ax.hist2d([r['x'] for r in values], [r['y'] for r in values], bins=35,
                                     weights=[r['seconds'] for r in values], range=((0, 1), (0, 1)), cmap='viridis')
                    self.figure.colorbar(hist[3], ax=ax, label='Observed dwell (seconds)')
                    heatmaps.append(hist[3])
                else:
                    artist = ax.scatter([r['x'] for r in values], [r['y'] for r in values],
                                        c=[r['frame']/payload['sources'][r['source']]['fps'] for r in values], s=7, cmap='viridis')
                    self.figure.colorbar(artist, ax=ax, label='Source-video time (s)')
                    for before, after in zip(values, values[1:]):
                        if (before['source'], before['track'], before['frame']+1) == (after['source'], after['track'], after['frame']):
                            ax.plot([before['x'], after['x']], [before['y'], after['y']], color='#64748b', lw=.5)
            ax.set(xlim=(0, 1), ylim=(1, 0), xlabel='Frame-width fraction', ylabel='Frame-height fraction', title=f'{label}: {keypoint}')
            ax.set_aspect('equal')
        if heatmaps:
            maximum = max(float(artist.get_array().max()) for artist in heatmaps)
            for artist in heatmaps:
                artist.set_clim(0, max(maximum, 1e-12))
        self.construct = dict(kind='dwell' if heatmap else 'trajectory', keypoint=keypoint, clusters=[a, b],
                              denominator='observed valid-keypoint frames / source FPS; gaps not filled')
        self.canvas.draw_idle()

    def occupancy(self, rows, payload, a, b):
        ax = self.axes()
        counts = occupancy(rows, payload['sources'])
        self.records = counts.to_dict('records')
        for label in dict.fromkeys((a, b)):
            if counts.empty:
                continue
            part = counts[counts['label'] == label]
            ax.plot(part['bin']*5, part['observed_fraction']*100, 'o-', color=color(label), label=label)
        ax.set(xlabel='Source-video time: 5-second bin start', ylabel='% of observed track time in bin', ylim=(0,100),
               title='Includes noise in denominator; untracked frames are not behavior absence')
        if self.records:
            ax.legend()
        self.construct = dict(kind='occupancy', bin_seconds=5, denominator='all observed detections for selected track/class scope')
        self.canvas.draw_idle()
