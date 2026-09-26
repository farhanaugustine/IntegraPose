from __future__ import annotations

import os
import subprocess
import sys
import copy
import time
import uuid
from pathlib import Path

from .store import Workspace

_processes = {}


def active(path):
    process = _processes.get(str(Path(path).resolve())) if path else None
    return process is not None and process.poll() is None


def launch(path, *, schedule=None, on_failure=None):
    path = str(Path(path).resolve())
    if not Path(path).is_file():
        raise ValueError('Run Sub-Behavior Discovery first to create an explorer workspace.')
    if active(path):
        raise ValueError('This workspace is already open. Switch to its existing explorer window.')
    env = os.environ.copy()
    source = str(Path(__file__).resolve().parents[2])
    env['PYTHONPATH'] = source + os.pathsep + env.get('PYTHONPATH', '')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    log = Path(path).with_suffix('.log')
    log_offset = log.stat().st_size if log.exists() else 0
    with log.open('a', encoding='utf-8') as stream:
        process = subprocess.Popen([sys.executable, '-B', '-m', 'integra_pose.discovery', path],
                                   cwd=source, env=env, stdout=stream, stderr=stream,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    _processes[path] = process
    if schedule is not None and on_failure is not None:
        monitor_launch(process, log, log_offset, schedule, on_failure)
    return process


def monitor_launch(process, log, offset, schedule, on_failure):
    def check():
        code = process.poll()
        if code is None:
            schedule(300, check)
            return
        if code == 0:
            return
        detail = ''
        try:
            with Path(log).open('rb') as stream:
                stream.seek(offset)
                detail = stream.read().decode('utf-8', errors='replace')[-3000:].strip()
        except OSError:
            pass
        message = f'Discovery Explorer exited with code {code}.\n\nPython: {sys.executable}\nLog: {log}'
        if 'DLL load failed' in detail and ('PySide6' in detail or 'QtWidgets' in detail):
            message += '\n\nThe Qt/PySide6 runtime in this environment could not load. The discovery data are not the cause. Repair Qt in this environment and restart IntegraPose.'
        if detail:
            message += '\n\n' + detail
        on_failure(message)
    schedule(300, check)


def capture_project(app, project_path=None):
    toolkit = getattr(app, '_embedded_behavior_analysis_app', None)
    state = toolkit.get_all_params() if toolkit is not None else getattr(app, '_tab7_project_state', None)
    if not isinstance(state, dict) or not state:
        return None
    state = copy.deepcopy(state)
    workspace = state.get('discovery_workspace')
    if workspace and not Path(workspace).is_file():
        raise ValueError(f'Discovery workspace is missing: {workspace}. Relink it before saving this project.')
    if workspace and active(workspace):
        store = Workspace(workspace)
        token = uuid.uuid4().hex
        store.set_state('checkpoint_request', token)
        deadline = time.monotonic() + 3
        while store.state('checkpoint_ack') != token:
            if time.monotonic() > deadline:
                raise ValueError('Explorer is busy. Pause playback or finish the current operation and save again.')
            time.sleep(.05)
    if workspace and project_path:
        target = str(Path(project_path).with_suffix('.discovery.sqlite').resolve())
        if Path(workspace).resolve() != Path(target):
            if active(workspace):
                raise ValueError('Close the Discovery Explorer before Save As so its review state can be copied safely.')
            Workspace(workspace).clone(target)
            state['discovery_workspace'] = target
    return state


def commit_project(app, state):
    if state:
        app._tab7_project_state = state
        toolkit = getattr(app, '_embedded_behavior_analysis_app', None)
        if toolkit is not None:
            toolkit.discovery_workspace = state.get('discovery_workspace', '')


def rollback_project_copy(previous, prepared):
    old = (previous or {}).get('discovery_workspace')
    new = (prepared or {}).get('discovery_workspace')
    if old and new and Path(old).resolve() != Path(new).resolve():
        # Only the newly prepared copy is disposable; the original remains intact.
        Path(new).unlink(missing_ok=True)


def restore_project(app, state):
    old = getattr(app, '_embedded_behavior_analysis_app', None)
    if old is not None and active(getattr(old, 'discovery_workspace', '')):
        raise ValueError('Close the Discovery Explorer before opening another project.')
    app._tab7_project_state = state
    if state:
        workspace = state.get('discovery_workspace', '')
        if workspace and not Path(workspace).is_file():
            from tkinter import filedialog
            replacement = filedialog.askopenfilename(title='Locate this project’s discovery workspace',
                                                     filetypes=[('Discovery workspace', '*.sqlite')], parent=getattr(app, 'root', None))
            if not replacement:
                raise ValueError('Project opening cancelled: discovery workspace could not be located.')
            state = dict(state, discovery_workspace=replacement)
        toolkit = old or app._ensure_tab7_toolkit()
        if toolkit is None:
            raise ValueError('Could not restore the saved Tab 7 toolkit. Check its dependencies.')
        toolkit.set_all_params(state)
    elif old is not None:
        old.set_all_params(dict(groups={}, discovery_workspace=''))


def register_result(toolkit, df, params, video_map, run_id, diagnostics):
    from .data import make_payload
    path = getattr(toolkit, 'discovery_workspace', '')
    if not path:
        path = str(Path(params['output_folder']) / 'discovery.sqlite')
    if active(path):
        raise ValueError('Close the explorer before running new features from Tk, or recluster inside the explorer.')
    payload = make_payload(df, params, video_map, run_id, diagnostics)
    store = Workspace(path)
    store.add_run(payload)
    toolkit.discovery_workspace = str(store.path)
    return store.path
