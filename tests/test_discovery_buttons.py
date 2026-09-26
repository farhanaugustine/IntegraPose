import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from integra_pose.discovery.project_io import tab7_project_payload
from integra_pose.discovery.bridge import monitor_launch


def state():
    return dict(groups={'Example': {'sources': []}}, keypoints_entry_var='nose,tail',
                behaviors_entry_var='Walking,Rearing', normalization_left='nose', normalization_right='tail')


def test_main_and_legacy_project_formats():
    value=state()
    assert tab7_project_payload(value)==(False,value)
    assert tab7_project_payload(dict(schema_version=2,setup={},tab7_discovery=value))==(True,value)
    for invalid in ({},[],{'schema_version':2,'setup':{}},{'groups':[]}, {'groups':{}}):
        with pytest.raises(ValueError):
            tab7_project_payload(invalid)


def test_failed_child_shows_only_current_launch_details(tmp_path):
    log=tmp_path/'discovery.log'
    old='previous unrelated launch\n'
    log.write_text(old+'PySide6: DLL load failed while importing QtWidgets',encoding='utf-8')
    process=Mock(); process.poll.side_effect=[None,1]
    pending=[]; errors=[]
    monitor_launch(process,log,len(old.encode()),lambda delay,fn: pending.append(fn),errors.append)
    pending.pop(0)(); pending.pop(0)()
    assert len(errors)==1 and 'Repair Qt' in errors[0]
    assert 'previous unrelated launch' not in errors[0]


def test_normal_child_exit_does_not_show_error(tmp_path):
    errors=[]; pending=[]
    process=Mock(); process.poll.return_value=0
    monitor_launch(process,tmp_path/'missing.log',0,lambda delay,fn: pending.append(fn),errors.append)
    pending.pop()()
    assert not errors


def test_load_button_uses_main_loader_and_rejects_unrelated_json(tmp_path):
    import tkinter as tk
    from integra_pose.hmm_vae_toolkit.main import BehaviorAnalysisApp
    root=tk.Tk(); root.withdraw()
    try:
        app=BehaviorAnalysisApp(root)
        app.set_all_params(state())
        source=tmp_path/'project.json'
        source.write_text(json.dumps(dict(schema_version=2,setup={},tab7_discovery=state())),encoding='utf-8')
        host=SimpleNamespace(config=SimpleNamespace(open_project=Mock(return_value={'loaded':True})))
        app.integra_app=host
        assert app.load_project(str(source))
        host.config.open_project.assert_called_once_with(str(source))
        app.integra_app=None
        app.behaviors_entry_var.set('wrong')
        assert app.load_project(str(source))
        assert app.behaviors_entry_var.get()=='Walking,Rearing'
        source.write_text('{}',encoding='utf-8')
        before=app.get_all_params()
        with patch('tkinter.messagebox.showerror') as error:
            assert not app.load_project(str(source))
            assert error.called
        assert app.get_all_params()==before
        source.write_text(json.dumps(state()),encoding='utf-8')
        app.running=True
        assert app.load_project(str(source),during_batch=True)
    finally:
        root.destroy()
