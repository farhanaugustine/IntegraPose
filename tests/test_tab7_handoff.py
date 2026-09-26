import json
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from integra_pose.hmm_vae_toolkit.main import BehaviorAnalysisApp
from integra_pose.main_gui_app import YoloApp
from integra_pose.gui.batch_processing_wizard import BatchProcessingWizard
from integra_pose.utils.analytics_manifest import CURRENT_SCHEMA_VERSION, validate_schema_version
from integra_pose.utils.batch_session import BatchVideoItem


class Value:
    def __init__(self, value=''):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def make_host(output_folder, manifest_path=''):
    toolkit = SimpleNamespace(groups={}, group_tree=Mock(), keypoints_entry_var=Value(),
                              behaviors_entry_var=Value(), output_folder=Value(str(output_folder)))
    for name in ('_ensure_group', '_read_analytics_manifest', '_resolve_existing_dir',
                 '_resolve_existing_file', '_apply_manifest_defaults', '_add_analytics_source',
                 'import_analytics_manifest_paths', '_collect_group_sources'):
        setattr(toolkit, name, MethodType(getattr(BehaviorAnalysisApp, name), toolkit))
    host = SimpleNamespace(root=None, last_tab6_manifest_path=str(manifest_path),
                           _ensure_tab7_toolkit=lambda: toolkit,
                           _prompt_tab7_group_name=lambda *args, **kwargs: 'Test', _toast=Mock())
    host._sanitize_tab7_manifest_paths = YoloApp._sanitize_tab7_manifest_paths
    for name in ('_get_last_tab6_manifest_path', '_import_tab7_manifest_groups'):
        setattr(host, name, MethodType(getattr(YoloApp, name), host))
    return host, toolkit


def make_wizard(host, items):
    wizard = SimpleNamespace(app=host, queue_items=items)
    wizard._batch_manifest_candidate_paths = BatchProcessingWizard._batch_manifest_candidate_paths
    for name in ('_resolve_batch_manifest_path', '_build_tab7_manifest_groups', '_open_batch_results_in_tab7'):
        setattr(wizard, name, MethodType(getattr(BatchProcessingWizard, name), wizard))
    return wizard


def write_manifest(root, version, name='video', group='Test'):
    folder = root / name
    folder.mkdir()
    labels = folder / 'labels'
    labels.mkdir()
    video = folder / 'source.mp4'
    video.touch()
    yaml = folder / 'dataset.yaml'
    yaml.write_text('names: [Walking, Grooming]\n', encoding='utf-8')
    bouts = folder / 'detailed.csv'
    bouts.write_text('track_id,start_frame,end_frame\n0,1,10\n', encoding='utf-8')
    payload = dict(schema_version=version, run_id=name,
                   inputs=dict(yolo_folder=str(labels), video_file=str(video), yaml_file=str(yaml),
                               keypoint_names=['nose', 'midback', 'tailbase'],
                               behavior_names=['Walking', 'Grooming']),
                   outputs=dict(output_folder=str(folder), detailed_bouts_csv=str(bouts)),
                   video=dict(base_name=name, width=480, height=480),
                   parameters=dict(max_gap_frames=10, min_bout_frames=10))
    if version != 1:
        payload['provenance'] = dict(subject_id=name, group=group, time_point='baseline')
    path = folder / 'run_manifest.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    item = BatchVideoItem(video_id=name, video_name=video.name, video_path=str(video),
                         group=group, analytics_status='completed', analytics_output_dir=str(folder))
    return path, item


@pytest.mark.parametrize('version', [1, 2, 3, 4])
@pytest.mark.parametrize('route', ['tab6', 'batch'])
def test_real_handoff_callbacks(tmp_path, version, route):
    path, item = write_manifest(tmp_path, version)
    host, toolkit = make_host(tmp_path, path)
    with patch('tkinter.messagebox.showerror') as error, patch('tkinter.messagebox.showwarning') as warning:
        if route == 'tab6':
            YoloApp._open_tab7_from_bout_analytics(host)
        else:
            BatchProcessingWizard._open_all_completed_results_in_tab7(make_wizard(host, [item]))
    error.assert_not_called()
    warning.assert_not_called()
    source, = toolkit.groups['Test']['sources']
    assert source['run_id'] == 'video'
    assert source['subject_id'] == ('' if version == 1 else 'video')
    assert source['time_point'] == ('' if version == 1 else 'baseline')
    assert source['tab6_manifest'] == str(path.resolve())
    assert toolkit.keypoints_entry_var.get() == 'nose,midback,tailbase'
    assert toolkit.behaviors_entry_var.get() == 'Walking,Grooming'
    assert not (path.parent / 'tab7_behavior_clustering').exists()


def test_batch_preserves_groups_and_filters_incomplete_rows(tmp_path):
    _, first = write_manifest(tmp_path, 4, 'first', 'Control')
    _, second = write_manifest(tmp_path, 4, 'second', 'Treatment')
    _, pending = write_manifest(tmp_path, 4, 'pending', 'Control')
    pending.analytics_status = 'pending'
    host, toolkit = make_host(tmp_path)
    with patch('tkinter.messagebox.showerror') as error:
        BatchProcessingWizard._open_all_completed_results_in_tab7(make_wizard(host, [first, second, pending]))
    error.assert_not_called()
    assert set(toolkit.groups) == {'Control', 'Treatment'}
    assert [s['subject_id'] for s in toolkit.groups['Control']['sources']] == ['first']
    assert [s['subject_id'] for s in toolkit.groups['Treatment']['sources']] == ['second']


@pytest.mark.parametrize('version', [None, 0, 5, 99, True, 4.0, '4', [], {}])
def test_unknown_or_malformed_versions_rejected(version):
    with pytest.raises(ValueError, match='schema_version'):
        validate_schema_version({'schema_version': version})


def test_current_writer_version_is_supported():
    from integra_pose.logic.analytics import ANALYTICS_MANIFEST_SCHEMA_VERSION
    assert ANALYTICS_MANIFEST_SCHEMA_VERSION == CURRENT_SCHEMA_VERSION
    assert validate_schema_version({'schema_version': CURRENT_SCHEMA_VERSION}) == 4


def test_schema_fix_does_not_bypass_missing_output_validation(tmp_path):
    path, _ = write_manifest(tmp_path, 4)
    payload = json.loads(path.read_text(encoding='utf-8'))
    payload['outputs']['detailed_bouts_csv'] = str(tmp_path / 'missing.csv')
    path.write_text(json.dumps(payload), encoding='utf-8')
    _, toolkit = make_host(tmp_path)
    with pytest.raises(ValueError, match='Detailed bouts CSV'):
        toolkit._read_analytics_manifest(str(path))


def test_failed_batch_import_reports_partial_results(tmp_path):
    _, good = write_manifest(tmp_path, 4, 'good')
    _, bad = write_manifest(tmp_path, 99, 'bad')
    host, toolkit = make_host(tmp_path)
    with patch('tkinter.messagebox.showerror') as error, patch('tkinter.messagebox.showwarning') as warning:
        BatchProcessingWizard._open_all_completed_results_in_tab7(make_wizard(host, [good, bad]))
    error.assert_not_called()
    assert len(toolkit.groups['Test']['sources']) == 1
    assert 'Imported 1 analytics run(s), skipped 1' in warning.call_args.args[1]
    assert 'schema_version' in warning.call_args.args[1]
