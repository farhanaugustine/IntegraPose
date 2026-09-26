from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def feature_catalog(names, params):
    catalog = []
    def add(name, units):
        catalog.append(dict(name=name, units=units, role='clustering input; may include imputed defaults'))
    for a, b in itertools.combinations(names, 2):
        add(f'Distance: {a} / {b}', 'body reference lengths')
    for a, b, c in itertools.combinations(names, 3):
        add(f'Angle: {a} / {b} / {c}', 'radians / pi')
    for name in names:
        for axis in ('x', 'y'):
            add(f'Position: {name} {axis}', 'bbox relative' if params.get('use_bbox_var') else 'centroid relative / body length')
    mode = params.get('location_mode', 'none')
    if mode == 'roi':
        for name in sorted(json.loads(params.get('roi_definitions_text', '{}'))):
            add(f'Location: {name}', 'binary membership')
        add('Location: outside', 'binary membership')
    elif mode == 'unsupervised':
        for axis in ('x', 'y'):
            add(f'Location: {axis}', 'fraction of frame')
    if params.get('social_mode'):
        for kind in ('mean', 'min'):
            add(f'Inter-animal distance: {kind}', 'body reference lengths')
    for name in names:
        add(f'Speed: {name}', 'body reference lengths / frame')
        add(f'Acceleration: {name}', 'body reference lengths / frame squared')
    return catalog


def make_payload(df, params, video_map, run_id, diagnostics):
    import cv2
    names = [s.strip() for s in params['keypoints_entry_var'].split(',') if s.strip()]
    features = np.asarray(df['feature_vector'].tolist(), dtype=float)
    catalog = feature_catalog(names, params)
    if features.ndim != 2 or features.shape[1] != len(catalog) or not np.isfinite(features).all():
        raise ValueError('Feature schema does not match the computed clustering matrix.')
    sources = {}
    rows = []
    for _, row in df.iterrows():
        directory = str(row['directory'])
        group = str(row['group'])
        source_id = hashlib.sha256((group + '\0' + directory).encode()).hexdigest()[:20]
        if source_id not in sources:
            video = str(video_map.get(directory, ''))
            capture = cv2.VideoCapture(video)
            try:
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                width, height = [int(capture.get(prop)) for prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)]
            finally:
                capture.release()
            if not np.isfinite(fps) or fps <= 0 or frames <= 0:
                raise ValueError(f'Cannot read source video timing: {video}. Relink the video before creating the explorer run.')
            sources[source_id] = dict(video=video, directory=directory, group=group,
                                      subject=str(row.get('subject_id', '')), fps=fps, frames=frames,
                                      width=width, height=height, session_offset_seconds=None)
        frame = int(row['frame'])
        if frame < 0 or frame >= sources[source_id]['frames']:
            raise ValueError(f'Observation frame {frame} is outside source video {directory}.')
        track = str(row['track_id'])
        row_id = f'{source_id}:{track}:{frame}'
        raw_points = row.get('keypoints', {})
        points = {str(k): [float(v) if np.isfinite(v) else None for v in values] for k, values in raw_points.items()}
        embedding = row.get('cluster_embedding', None)
        if not isinstance(embedding, (list, tuple, np.ndarray)):
            embedding = []
        rows.append(dict(row_id=row_id, source=source_id, track=track, frame=frame,
                         class_id=int(row['class_id']), behavior=str(row.get('behavior', row['class_id'])),
                         cluster_label=str(row['cluster_label']), cluster_status=str(row['cluster_status']),
                         embedding=list(embedding), keypoints=points))
    if len({r['row_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source/track/frame observations: repair track identity before review.')
    for row, values in zip(rows, features.tolist()):
        row['features'] = values
    return dict(run_id=run_id, rows=rows, sources=sources, feature_catalog=catalog,
                params=params, diagnostics=diagnostics, keypoints=names)


def fragments(rows, label='review_label', max_gap=1):
    ordered = sorted(rows, key=lambda r: (r['source'], r['track'], r['frame']))
    result = []
    for row in ordered:
        key = (row['source'], row['track'], row['class_id'], row[label],
               row.get('review_status') if label == 'review_label' else row['cluster_status'])
        if not result or result[-1]['key'] != key or row['frame'] - result[-1]['end'] > max_gap:
            result.append(dict(key=key, source=row['source'], track=row['track'], label=row[label],
                               start=row['frame'], end=row['frame'], rows=[row['row_id']]))
        else:
            result[-1]['end'] = row['frame']
            result[-1]['rows'].append(row['row_id'])
    return result


def display_embedding(rows):
    embeddings = [r.get('embedding', []) for r in rows]
    classes = {r['class_id'] for r in rows}
    if len(classes) == 1 and embeddings and all(len(e) >= 2 for e in embeddings):
        values = np.asarray(embeddings, dtype=float)
        dimensions = values.shape[1]
        values = values[:, :3]
        return np.pad(values, ((0, 0), (0, max(0, 3-values.shape[1])))), (
            f'Clustering representation: first {min(3, dimensions)} of {dimensions} dimensions; one class')
    from sklearn.decomposition import PCA
    matrix = np.asarray([r['features'] for r in rows], dtype=float)
    components = min(3, *matrix.shape)
    if len(matrix) < 2:
        return np.zeros((len(matrix), 3)), 'Single observation; no meaningful projection'
    projection = PCA(n_components=components).fit_transform(matrix)
    return np.pad(projection, ((0, 0), (0, 3-components))), 'PCA display only (not the clustering space); unscaled input features'


def compare_features(rows, catalog, left, right, label='review_label'):
    a = np.asarray([r['features'] for r in rows if r[label] == left], dtype=float)
    b = np.asarray([r['features'] for r in rows if r[label] == right], dtype=float)
    if not len(a) or not len(b):
        return []
    pooled_scale = np.sqrt((a.var(axis=0) + b.var(axis=0)) / 2)
    difference = np.abs(a.mean(axis=0) - b.mean(axis=0))
    score = np.divide(difference, pooled_scale, out=np.zeros_like(difference), where=pooled_scale > 1e-12)
    result = [dict(feature=i, name=item['name'], units=item['units'], mean_a=float(a[:, i].mean()),
                   mean_b=float(b[:, i].mean()), standardized_difference=float(score[i]),
                   zero_variance=bool(pooled_scale[i] <= 1e-12), n_a=len(a), n_b=len(b))
              for i, item in enumerate(catalog)]
    return sorted(result, key=lambda item: item['standardized_difference'], reverse=True)


def occupancy(rows, sources, bin_seconds=5, label='review_label'):
    if bin_seconds <= 0:
        raise ValueError('Time bins must be positive.')
    records = [dict(source=r['source'], track=r['track'],
                    bin=int(r['frame'] / sources[r['source']]['fps'] / bin_seconds), label=r[label],
                    seconds=1 / sources[r['source']]['fps']) for r in rows]
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    counts = df.groupby(['source', 'track', 'bin', 'label'], sort=True, dropna=False)['seconds'].sum().reset_index()
    completed = []
    for (source, track), part in counts.groupby(['source', 'track'], sort=True):
        labels = sorted(part['label'].unique())
        grid = pd.MultiIndex.from_product([range(int(part['bin'].min()), int(part['bin'].max())+1), labels], names=['bin','label'])
        expanded = part.set_index(['bin','label'])[['seconds']].reindex(grid, fill_value=0).reset_index()
        expanded['source'] = source
        expanded['track'] = track
        expanded['observed_seconds'] = expanded.groupby('bin')['seconds'].transform('sum')
        expanded['observed_fraction'] = expanded['seconds'] / expanded['observed_seconds'].replace(0, np.nan)
        completed.append(expanded)
    return pd.concat(completed, ignore_index=True)
