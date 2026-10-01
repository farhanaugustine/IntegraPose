"""Exercise actual optional analysis engines with a reproducible animal fixture.

This tests executable outputs, not biological validity of simulated trajectories.
"""
import numpy as np
import pandas as pd
import pytest

from integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.profiles import preset_config, PRESET_NAMES, advanced_unavailable


def animal_data(tmp_path):
    config=preset_config()
    groups=[{'name':'A','videos':['mouse_a']},{'name':'B','videos':['mouse_b']}]
    for n,name in enumerate(('mouse_a','mouse_b')):
        t=np.arange(240,dtype=float)
        data={'frame':t.astype(int),'track_id':1,'video_fps':50,
              'smoothed_behavior':np.where(t<90,'Walking','Grooming'),
              'speed_px_per_s':10+np.sin(t*.08+n),
              'turning_speed_rad_per_s':np.cos(t*.06)+.1*np.sin(t*.17),
              'turning_speed_deg_per_s':np.cos(t*.06)*50,
              'elongation':30+np.sin(t*.11)}
        for k,kp in enumerate(config['DATASET']['KEYPOINT_ORDER']):
            data[kp+'_x']=100+k*4+np.sin(t*.08+k*.5)*8
            data[kp+'_y']=80+k*3+np.cos(t*.07+k*.6)*5
        data['Left Front Paw_speed_px_per_s']=8+np.sin(t*.09)
        data['Right Rear Paw_speed_px_per_s']=8+np.sin(t*.09+.4)
        (tmp_path/name).mkdir()
        pd.DataFrame(data).to_csv(tmp_path/name/'final_analysis_data.csv',index=False)
    return config,groups


def test_human_advanced_scope_is_explicit():
    blocked=advanced_unavailable(preset_config(PRESET_NAMES[1]))
    assert len(blocked)==3
    assert all('not implemented for the human preset' in message for message in blocked.values())
    assert advanced_unavailable(preset_config())=={}


def test_real_animal_umap_output_and_finite_sample_policy(tmp_path):
    pytest.importorskip('umap')
    from integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.advanced_behavioral_analysis import main
    config,groups=animal_data(tmp_path)
    main(str(tmp_path),groups,config)
    table=pd.read_csv(tmp_path/'advanced_behavior_plots'/'umap_coordinates.csv')
    assert len(table)==478  # one unavailable first-frame velocity per video
    assert table[['umap_1','umap_2','umap_3']].notna().all().all()
    assert set(table.video_source)=={'mouse_a','mouse_b'}
    assert (tmp_path/'advanced_behavior_plots'/'umap_3d_state_space.html').is_file()
    main(str(tmp_path),groups,config)
    repeated=pd.read_csv(tmp_path/'advanced_behavior_plots'/'umap_coordinates.csv')
    pd.testing.assert_frame_equal(table,repeated)


def test_real_decision_outputs_and_empty_result_is_failure(tmp_path):
    from integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.decision_dynamics_analysis import main
    config,groups=animal_data(tmp_path)
    main(str(tmp_path),groups,config)
    paths=list((tmp_path/'decision_dynamics_plots').glob('*.csv'))
    assert len(paths)==3
    table=pd.read_csv(paths[0])
    assert len(table)==122  # 61 offsets per video
    assert set(table.time_to_transition)==set(range(-30,31))
    for name in ('mouse_a','mouse_b'):
        path=tmp_path/name/'final_analysis_data.csv'
        data=pd.read_csv(path); data['smoothed_behavior']='Walking';data.to_csv(path,index=False)
    with pytest.raises(ValueError,match='No complete eligible'):
        main(str(tmp_path),groups,config)


def test_real_ccm_outputs_have_seed_and_video_identity(tmp_path):
    pytest.importorskip('pyEDM')
    from integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.compare_ccm import main
    config,groups=animal_data(tmp_path)
    main(str(tmp_path),groups,config)
    paths=list((tmp_path/'ccm_plots').glob('*.csv'))
    assert len(paths)==2
    for path in paths:
        table=pd.read_csv(path)
        assert set(table.video_source)=={'mouse_a','mouse_b'}
        assert (table.seed==42).all()
        assert np.isfinite(table['Final Rho']).all()
    originals={path.name:pd.read_csv(path) for path in paths}
    main(str(tmp_path),groups,config)
    for path in paths:
        pd.testing.assert_frame_equal(originals[path.name],pd.read_csv(path))
