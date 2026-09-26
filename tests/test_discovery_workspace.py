import numpy as np
import pytest

from integra_pose.discovery.data import compare_features, display_embedding, feature_catalog, fragments, occupancy
from integra_pose.discovery.store import Workspace


def sample():
    rows = [dict(row_id=f's:t:{i}', source='s', track='t', frame=i, class_id=0, behavior='walk',
                 cluster_label='0:0' if i<3 else '0:1', cluster_status='clustered',
                 features=[float(i), 1.], embedding=[float(i),0,0], keypoints={'nose':[.2,.3,1.]})
            for i in (0,1,2,5,6,7)]
    return dict(run_id='r', rows=rows, sources={'s':dict(fps=10,frames=10,video='example.mp4')},
                params={},feature_catalog=[dict(name='Speed',units='body lengths/frame'),dict(name='Constant',units='ratio')])


def test_reviews_are_separate_and_reversible(tmp_path):
    store=Workspace(tmp_path/'discovery.sqlite')
    payload=sample(); store.add_run(payload)
    store.edit('r',['s:t:0','s:t:1'],'walking','AB','video inspection')
    assert store.load_run('r')==payload
    assert store.reviewed_rows('r')[0]['review_label']=='walking'
    store.undo('r')
    assert store.reviewed_rows('r')[0]['review_label']=='0:0'
    store.redo('r')
    assert store.reviewed_rows('r')[0]['review_label']=='walking'
    clone=store.clone(tmp_path/'copy.sqlite')
    clone.edit('r',['s:t:0'],'uncertain','CD',status='uncertain')
    assert store.reviewed_rows('r')[0]['review_label']=='walking'
    assert clone.reviewed_rows('r')[0]['review_status']=='uncertain'


def test_invalid_edits_do_not_commit(tmp_path):
    store=Workspace(tmp_path/'w.sqlite'); store.add_run(sample())
    for ids,reviewer in [(['missing'],'AB'), (['s:t:0'],''), ([], 'AB')]:
        with pytest.raises(ValueError):
            store.edit('r',ids,'name',reviewer)
    assert not store.edits('r')


def test_bout_identity_and_gaps():
    rows=sample()['rows']
    assert len(fragments(rows,'cluster_label',1))==2
    extra=dict(rows[0],row_id='other',track='other')
    assert len(fragments(rows+[extra],'cluster_label',10))==3


def test_time_denominator_does_not_fill_missing_frames():
    payload=sample()
    values=occupancy(payload['rows'],payload['sources'],bin_seconds=1,label='cluster_label')
    assert values['seconds'].sum()==pytest.approx(.6)
    assert values['observed_fraction'].sum()==pytest.approx(1)
    assert values['observed_seconds'].iloc[0]==pytest.approx(.6)


def test_feature_catalog_and_contrast():
    catalog=feature_catalog(['nose','tail'],dict(use_bbox_var=True))
    assert len(catalog)==9
    assert catalog[-4]['name']=='Speed: nose'
    payload=sample()
    rank=compare_features(payload['rows'],payload['feature_catalog'],'0:0','0:1',label='cluster_label')
    assert rank[0]['name']=='Speed'
    assert rank[1]['zero_variance']


def test_class_embeddings_are_not_combined_as_shared_umap():
    rows=sample()['rows']
    points,title=display_embedding(rows)
    assert points.shape==(6,3)
    assert 'Clustering representation' in title
    rows[0]['class_id']=1
    points,title=display_embedding(rows)
    assert 'PCA display only' in title


def test_archive_rename_and_state(tmp_path):
    store=Workspace(tmp_path/'w.sqlite'); store.add_run(sample())
    store.rename('r','Walking comparison'); store.archive('r')
    assert not store.runs(False)
    assert store.runs()[0]['name']=='Walking comparison'
    store.set_state('view:r',dict(frame=5,feature=1))
    assert Workspace(store.path).state('view:r')['frame']==5


def test_project_save_as_isolates_analysis(tmp_path):
    from types import SimpleNamespace
    from integra_pose.discovery.bridge import capture_project
    store=Workspace(tmp_path/'original.sqlite'); store.add_run(sample())
    app=SimpleNamespace(_tab7_project_state=dict(discovery_workspace=str(store.path)))
    state=capture_project(app,tmp_path/'project.json')
    assert state['discovery_workspace']!=str(store.path)
    copied=Workspace(state['discovery_workspace'])
    copied.edit('r',['s:t:0'],'new','AB')
    assert not store.edits('r')


def test_zero_occupancy_differs_from_unobserved_time():
    payload=sample()
    payload['rows']=[dict(payload['rows'][0],frame=0),dict(payload['rows'][1],frame=30,cluster_label='0:1')]
    values=occupancy(payload['rows'],payload['sources'],bin_seconds=1,label='cluster_label')
    assert values.query("bin==0 and label=='0:1'")['observed_fraction'].iloc[0]==0
    assert values.query('bin==1')['observed_fraction'].isna().all()


def test_new_edit_supersedes_redo_branch(tmp_path):
    store=Workspace(tmp_path/'w.sqlite'); store.add_run(sample())
    store.edit('r',['s:t:0'],'first','AB'); store.undo('r')
    store.edit('r',['s:t:0'],'second','AB'); store.redo('r')
    assert store.reviewed_rows('r')[0]['review_label']=='second'
    assert store.edits('r')[0]['superseded']


def test_delete_does_not_touch_active_run_or_sources(tmp_path):
    store=Workspace(tmp_path/'w.sqlite'); store.add_run(sample())
    with pytest.raises(ValueError):
        store.delete_run('r')
    newer=sample(); newer['run_id']='r2'; store.add_run(newer)
    store.delete_run('r')
    assert [r['id'] for r in store.runs()]==['r2']


def test_failed_project_copy_rollback_preserves_original(tmp_path):
    from types import SimpleNamespace
    from integra_pose.discovery.bridge import capture_project, rollback_project_copy
    store=Workspace(tmp_path/'original.sqlite'); store.add_run(sample())
    original=dict(discovery_workspace=str(store.path))
    app=SimpleNamespace(_tab7_project_state=original)
    prepared=capture_project(app,tmp_path/'copy.json')
    assert app._tab7_project_state==original
    rollback_project_copy(original,prepared)
    assert store.path.exists()
    from pathlib import Path
    assert not Path(prepared['discovery_workspace']).exists()
