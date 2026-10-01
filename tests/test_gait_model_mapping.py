import json
from types import SimpleNamespace

import pytest

from integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.model_schema import (
    read_schema_file, schema_from_model, validate_label_mapping, normalize_edges,
)
from integra_pose.plugins.plugin_gait_kinematics.gait_kinematics.profiles import preset_config, PRESET_NAMES, HUMAN_KEYPOINTS


def test_python_literal_metadata_is_read_without_executing_code(tmp_path):
    path = tmp_path / 'model.py'
    path.write_text("raise RuntimeError('must never execute')\nKEYPOINT_NAMES = ['hip', 'knee', 'ankle']\nSKELETON_EDGES = [(0, 1), (1, 2)]\n")
    result = read_schema_file(path)
    assert result['keypoint_names'] == ['hip', 'knee', 'ankle']
    assert result['edges'] == [[0, 1], [1, 2]]


def test_dynamic_python_names_require_explicit_schema(tmp_path):
    path = tmp_path / 'model.py'
    path.write_text("KEYPOINT_NAMES = get_names_from_network()\n")
    with pytest.raises(ValueError, match='No ordered landmark names'):
        read_schema_file(path)


def test_dataset_yaml_and_named_skeleton_edges(tmp_path):
    path = tmp_path / 'data.yaml'
    path.write_text('kpt_shape: [3, 3]\nkpt_names: [ankle, knee, hip]\nedges: [[hip, knee], [knee, ankle]]\n')
    result = read_schema_file(path)
    assert result['edges'] == [[2, 1], [1, 0]]
    path.write_text('kpt_shape: [4, 3]\nkpt_names: [ankle, knee, hip]\n')
    with pytest.raises(ValueError, match='expected 4'):
        read_schema_file(path)


def fake_model(names=None, data='custom.yaml', count=17):
    return SimpleNamespace(task='pose', names={0: 'person'}, kpt_names=names, overrides={},
                           ckpt={'train_args': {'data': data}},
                           model=SimpleNamespace(kpt_shape=[count, 3], yaml={}, kpt_names=None))


@pytest.mark.parametrize('size', ['n', 's', 'm', 'l', 'x'])
def test_standard_yolo26_coco_metadata_maps_exact_indices(tmp_path, size):
    schema = schema_from_model(fake_model(data='/training/coco-pose.yaml'), tmp_path / f'yolo26{size}-pose.pt')
    assert schema['keypoint_names'][15:17] == ['Left Ankle', 'Right Ankle']
    assert schema['keypoint_names'][11:15] == ['Left Hip', 'Right Hip', 'Left Knee', 'Right Knee']


def test_custom_17_point_model_is_not_assumed_to_be_coco(tmp_path):
    schema = schema_from_model(fake_model(), tmp_path / 'yolo26n-pose.pt')
    assert schema['keypoint_count'] == 17
    assert schema['keypoint_names'] == []


def test_custom_embedded_names_override_coco_and_preserve_order(tmp_path):
    custom = list(reversed(HUMAN_KEYPOINTS))
    schema = schema_from_model(fake_model(names=custom, data='coco-pose.yaml'), tmp_path / 'best.pt')
    assert schema['keypoint_names'] == custom


def test_detection_only_model_rejected(tmp_path):
    model = fake_model()
    model.task = 'detect'
    with pytest.raises(ValueError, match='not a pose model'):
        schema_from_model(model, tmp_path / 'detect.pt')


def test_same_count_wrong_label_order_is_rejected_even_if_confirmed(tmp_path):
    config = preset_config(PRESET_NAMES[1])
    config['DATASET']['MAPPING_CONFIRMED'] = True
    names = HUMAN_KEYPOINTS.copy()
    names[15], names[16] = names[16], names[15]
    (tmp_path / 'labels.csv').write_text(','.join(f'kp_{name.replace(" ", "_")}_x_n' for name in names))
    with pytest.raises(ValueError, match='order mismatch'):
        validate_label_mapping(config, tmp_path)


def test_named_labels_accept_spelling_separator_and_case_equivalence(tmp_path):
    config = preset_config(PRESET_NAMES[1])
    (tmp_path / 'labels.csv').write_text(','.join(f'kp_{name.lower().replace(" ", "_")}_x_n' for name in HUMAN_KEYPOINTS))
    assert validate_label_mapping(config, tmp_path) == 'label metadata'


def test_nameless_labels_require_deliberate_confirmation(tmp_path):
    config = preset_config()
    with pytest.raises(ValueError, match='no verifiable landmark names'):
        validate_label_mapping(config, tmp_path)
    config['DATASET']['MAPPING_CONFIRMED'] = True
    assert validate_label_mapping(config, tmp_path) == 'user-confirmed model index order'


def test_model_schema_mismatch_cannot_be_bypassed(tmp_path):
    config = preset_config()
    config['DATASET']['MAPPING_CONFIRMED'] = True
    config['MODEL_SCHEMA'] = {'keypoint_count': 17}
    with pytest.raises(ValueError, match='count differs'):
        validate_label_mapping(config, tmp_path)
    config['MODEL_SCHEMA'] = {'keypoint_count': 12, 'keypoint_names': list(reversed(config['DATASET']['KEYPOINT_ORDER']))}
    with pytest.raises(ValueError, match='order differs'):
        validate_label_mapping(config, tmp_path)


def test_invalid_edges_do_not_wrap_or_guess_one_based_indices():
    with pytest.raises(ValueError, match='Invalid zero-based'):
        normalize_edges([[1, 3]], ['hip', 'knee', 'ankle'])
    with pytest.raises(ValueError, match='unknown landmark'):
        normalize_edges([['hip', 'toe']], ['hip', 'knee', 'ankle'])
