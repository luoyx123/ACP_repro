"""Regressions for the Mixamo frame and support adaptation errors."""
import numpy as np
from acp.data.adapters.mixamo import transform_reference, ground_stance, initial_quaternion, target_mapping
from scipy.spatial.transform import Rotation


def test_elevated_feet_do_not_lower_inverted_pelvis():
    names = ['Hips', 'LeftToeBase', 'RightToeBase']
    source = np.array([[[1., 2, .2], [1, 2, 1.5], [1, 2, 1.6]]])
    result = transform_reference(source, names, .8)
    np.testing.assert_allclose(result[0, 0], [0, 0, .16])
    np.testing.assert_allclose(source[0, 0], [1, 2, .2])
    assert 'sphere_5' in target_mapping()['LeftToeBase']


def test_ground_stance_excludes_elevated_support_and_touchdown_transition():
    names = ['LeftToeBase', 'RightToeBase']
    times = np.arange(20)/50
    positions = np.zeros((20, 2, 3))
    positions[:, 1, 2] = .3  # A ladder rung is not the ground.
    positions[:5, 0, 2] = .2
    flags = ground_stance(positions, names, times)
    assert not flags[:, 1].any()
    assert not flags[:7, 0].any()
    assert flags[10:, 0].all()


def test_initial_root_orientation_handles_inversion():
    names = ['Hips', 'LeftUpLeg', 'RightUpLeg', 'Spine']
    p = np.array([[[0., 0, .5], [0, .1, .5], [0, -.1, .5], [0, 0, .3]]])
    q = initial_quaternion(p, names)
    r = Rotation.from_quat(q[[1, 2, 3, 0]])
    np.testing.assert_allclose(r.apply([0, 0, 1]), [0, 0, -1], atol=1e-12)
