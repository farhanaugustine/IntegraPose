from __future__ import annotations

import uuid

import pandas as pd
from PySide6.QtCore import QThread, Signal

from .store import Workspace


def recluster(workspace, run_id, settings):
    from integra_pose.hmm_vae_toolkit.per_class_clustering import cluster_per_class
    store = Workspace(workspace)
    payload = store.load_run(run_id)
    params = dict(payload['params'], **settings)
    frame = pd.DataFrame(dict(class_id=r['class_id'], feature_vector=r['features']) for r in payload['rows'])
    result, report = cluster_per_class(frame, min_class_size=int(params.get('min_class_size', 30)),
                                      min_cluster_size=int(params['min_cluster_size']),
                                      umap_neighbors=int(params['umap_neighbors']),
                                      umap_components=int(params['umap_components']))
    for row, (_, assignment) in zip(payload['rows'], result.iterrows()):
        embedding = assignment['cluster_embedding']
        row.update(cluster_label=str(assignment['cluster_label']), cluster_status=str(assignment['cluster_status']),
                   embedding=embedding if isinstance(embedding, list) else [])
    payload.update(run_id=uuid.uuid4().hex, parent_run=run_id, params=params)
    payload['diagnostics']['clustering_backend_runs'] = report.backend_runs
    return store.add_run(payload)


class ReclusterWorker(QThread):
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, workspace, run_id, settings):
        super().__init__()
        self.args = workspace, run_id, settings

    def run(self):
        try:
            self.completed.emit(recluster(*self.args))
        except Exception as exc:
            self.failed.emit(str(exc))
