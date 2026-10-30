import hashlib
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from acp.envs.reference import ReferenceMotion, name_indices, action_to_target, target_to_action
from acp.envs.conversion import prepare_urdf, transform


def test_named_motion_queries_preserve_rotation_and_discrete_labels(tmp_path):
    names = [f'joint_{i}' for i in range(23)]
    bodies = [f'body_{i}' for i in range(9)]
    q = np.stack([np.arange(23), np.arange(23)+2])
    labels = np.zeros((2, 9), dtype=bool);labels[1] = True
    path = tmp_path/'motion.npz'
    np.savez(path, timestamps=[0., 1.], joint_names=names, body_names=bodies,
        base_position_w=[[0., 0., 1.], [2., 0., 1.]],
        base_quaternion_wxyz=[[1., 0., 0., 0.], [0., 0., 0., 1.]],
        base_linear_velocity_w=np.ones((2, 3))*2, base_angular_velocity_w=np.zeros((2, 3)),
        joint_positions=q, joint_velocities=np.ones((2, 23))*2,
        body_landmarks_w=np.zeros((2, 9, 3)), contact_flags=labels,
        target_positions_body_yaw=np.stack([np.zeros((9, 3)), np.ones((9, 3))]), target_occupancy=labels)
    motion = ReferenceMotion(path, names[::-1], bodies[::-1])
    sample = motion.sample([.5, 1.])
    np.testing.assert_allclose(sample['joint_positions'][0], (np.arange(23)+1)[::-1])
    np.testing.assert_allclose(sample['base_quaternion_wxyz'][0], [2**-.5, 0, 0, 2**-.5], atol=1e-12)
    assert not sample['contact_flags'][0].any() and sample['contact_flags'][1].all()
    assert not sample['target_positions_body_yaw'][0].any()
    actions = target_to_action(sample['joint_positions'], np.zeros(23))
    np.testing.assert_allclose(action_to_target(actions, np.zeros(23)), sample['joint_positions'])
    with pytest.raises(ValueError): name_indices(['a', 'a'], ['a', 'b'])


def test_massless_leaf_cleanup_preserves_collision_transform_and_source(tmp_path):
    source = tmp_path/'source.urdf'
    source.write_text('''<robot name="test"><link name="pelvis"><inertial><mass value="2"/></inertial></link>
      <link name="tip"><collision><origin xyz="1 0 0"/><geometry><sphere radius="0.02"/></geometry></collision></link>
      <joint name="tip_fixed" type="fixed"><parent link="pelvis"/><child link="tip"/>
        <origin xyz="2 3 4" rpy="0 0 1.5707963267948966"/></joint></robot>''')
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    out = tmp_path/'clean.urdf'
    assert prepare_urdf(source, out) == ['tip']
    root = ET.parse(out).getroot()
    np.testing.assert_allclose(transform(root.find('link/collision/origin'))[:3, 3], [2, 4, 4])
    assert root.find('link/inertial/mass').get('value') == '2'
    assert root.find('link/collision/geometry/sphere').get('radius') == '0.02'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
