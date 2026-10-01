"""Scientific boundaries for human/animal synchronized motion review."""
import json
import shutil

import cv2
import numpy as np
import pandas as pd
import pytest

from integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.coordination import (
    motion_table, human_chains, render_frame, export_motion_review,
)
from integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.profiles import preset_config, PRESET_NAMES, validate_config


def fixture_data():
    config = validate_config(preset_config(PRESET_NAMES[1]))
    data = {'frame': [0, 1, 2, 4], 'track_id': [3]*4}
    for name in config['DATASET']['KEYPOINT_ORDER']:
        data[name+'_x'] = [30.]*4
        data[name+'_y'] = [20.]*4
        data[name+'_conf'] = [1.]*4
    # Straight first frame, 90-degree bend next; no interpolation at frame 3.
    for side in ('Left','Right'):
        data[side+' Knee_y'] = [40.]*4
        data[side+' Ankle_x'] = [30.,50.,50.,50.]
        data[side+' Ankle_y'] = [60.,40.,40.,40.]
    data['Left Knee_conf'][2] = .1
    return pd.DataFrame(data), config


def test_known_geometry_confidence_and_gap():
    rows,config=fixture_data()
    table,gated=motion_table(rows,config,50)
    assert table.Left_knee_bend_deg.iloc[0] == pytest.approx(0)
    assert table.Left_knee_bend_deg.iloc[1] == pytest.approx(90)
    assert np.isnan(table.Left_knee_bend_deg.iloc[2])
    assert np.isnan(table.Right_knee_bend_deg.iloc[3])
    assert table['Left Ankle_motion'].iloc[3] == 'unknown'
    assert table['Left Ankle_motion'].iloc[4] == 'unknown'
    assert table.time_s.iloc[4] == pytest.approx(.08)
    assert np.isnan(gated['Left Knee_x'].iloc[2])
    assert rows['Left Knee_x'].iloc[2] == 30  # caller data unchanged


def test_explicit_subject_type_prevents_animal_human_angle_mix():
    rows,config=fixture_data()
    config['SUBJECT_TYPE']='animal'
    table,_=motion_table(rows,config,50)
    assert not human_chains(config)
    assert not any('knee_bend' in c for c in table)
    assert set(table.subject_type) == {'animal'}
    assert validate_config(preset_config(PRESET_NAMES[0]))['SUBJECT_TYPE']=='animal'


def test_anatomy_mapping_is_name_based_not_index_based():
    _,config=fixture_data()
    config['DATASET']['KEYPOINT_ORDER']=list(reversed(config['DATASET']['KEYPOINT_ORDER']))
    assert human_chains(config)['Left']==['Left Hip','Left Knee','Left Ankle']
    config['DATASET']['KEYPOINT_ORDER']=['kp'+str(i) for i in range(17)]
    assert human_chains(config)=={}


@pytest.mark.parametrize('kind',['duplicate','mixed','fps'])
def test_invalid_identity_or_time_rejected(kind):
    rows,config=fixture_data()
    if kind=='duplicate': rows.loc[1,'frame']=0
    if kind=='mixed': rows.loc[1,'track_id']=4
    with pytest.raises(ValueError): motion_table(rows,config,0 if kind=='fps' else 50)


def test_degenerate_geometry_is_unknown():
    rows,config=fixture_data()
    rows['Left Knee_y']=rows['Left Hip_y']
    table,_=motion_table(rows,config,50)
    assert table.Left_knee_bend_deg.isna().all()


def test_export_player_and_measurement_provenance(tmp_path):
    if not shutil.which('ffmpeg'): pytest.skip('FFmpeg unavailable')
    rows,config=fixture_data()
    video=tmp_path/'source.avi'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),50,(80,80))
    assert writer.isOpened()
    for _ in range(5): writer.write(np.zeros((80,80,3),np.uint8))
    writer.release()
    result=export_motion_review(rows,config,video,tmp_path,50)
    page=(tmp_path/'limb_motion_review.html').read_text()
    assert 'data-rate="0.25"' in page and 'Human limb motion review' in page
    assert '__' not in page
    meta=json.loads((tmp_path/'limb_motion_review.json').read_text())
    assert meta['contact_validated'] is False
    assert meta['first_frame']==0 and meta['last_frame']==4
    cap=cv2.VideoCapture(str(tmp_path/'limb_motion_review.mp4'))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==5
    assert cap.get(cv2.CAP_PROP_FPS)==pytest.approx(50)
    cap.release()
    assert 'motion_review' in result


def test_renderer_handles_unknown_frame_and_animal_mode():
    rows,config=fixture_data()
    config['SUBJECT_TYPE']='animal'
    table,gated=motion_table(rows,config,50)
    picture=render_frame(np.zeros((80,80,3),np.uint8),3,table,gated,config,50)
    assert picture.shape==(900,1600,3)


def test_legacy_config_defaults_without_enabling_costly_export():
    config=preset_config(PRESET_NAMES[1])
    config.pop('SUBJECT_TYPE')
    config['REVIEW'].pop('EXPORT_COORDINATION')
    validated=validate_config(config)
    assert validated['SUBJECT_TYPE']=='human'
    assert validated['REVIEW']['EXPORT_COORDINATION'] is False


def test_group_comparison_rejects_human_animal_mix_and_missing_provenance(tmp_path):
    from integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.compare_gait import validate_group_anatomy
    human=preset_config(PRESET_NAMES[1])
    groups=[{'name':'A','videos':['human']},{'name':'B','videos':['mouse']}]
    for name,config in [('human',human),('mouse',preset_config(PRESET_NAMES[0]))]:
        (tmp_path/name).mkdir()
        (tmp_path/name/'analysis_config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError,match='incompatible subject type'):
        validate_group_anatomy(tmp_path,groups,human)
    (tmp_path/'mouse'/'analysis_config.json').unlink()
    with pytest.raises(ValueError,match='missing analysis_config'):
        validate_group_anatomy(tmp_path,groups,human)
