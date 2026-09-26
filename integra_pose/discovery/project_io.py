from __future__ import annotations

import copy


def tab7_project_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError('Project JSON must contain an object.')
    main_project = 'tab7_discovery' in payload or ('setup' in payload and 'schema_version' in payload)
    state = payload.get('tab7_discovery') if main_project else payload
    if not isinstance(state, dict) or not isinstance(state.get('groups'), dict):
        raise ValueError('This file has no saved Tab 7 setup. Choose an IntegraPose project saved with Tab 7 data, or a legacy Tab 7 project. Your current setup has not been changed.')
    if not isinstance(state.get('keypoints_entry_var'), str):
        raise ValueError('The saved Tab 7 project is missing its keypoint-name configuration.')
    for name, group in state['groups'].items():
        if not isinstance(name, str) or not isinstance(group, dict) or not isinstance(group.get('sources', []), list):
            raise ValueError('The saved Tab 7 groups or sources have an invalid structure.')
        if any(not isinstance(source, dict) for source in group.get('sources', [])):
            raise ValueError('Each saved Tab 7 source must be an object.')
    return main_project, copy.deepcopy(state)
