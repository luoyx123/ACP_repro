import numpy as np
from scipy.spatial.transform import Rotation

from acp.data.contacts import filter_segments, measured_force_flags, next_onset_targets, yaw_local
from acp.data.motions import load_omni
from acp.data.motions import load_lafan_csv


def test_segments_exact_minimum_and_episode_edges():
    flags = np.zeros((40, 2), bool)
    flags[:10, 0] = True
    flags[15:24, 0] = True
    flags[-10:, 1] = True
    filtered = filter_segments(flags)
    assert filtered[:, 0].sum() == 10
    assert filtered[:, 1].sum() == 10


def test_onset_is_not_continuous_contact_and_has_inclusive_horizon():
    flags = np.zeros((100, 1), bool)
    flags[5:15] = True
    flags[85:95] = True
    points = np.tile([[[0.4, 0.1, 0.0]]], (100, 1, 1))
    root = np.zeros((100, 3))
    quat = np.tile([1, 0, 0, 0], (100, 1))
    labels = next_onset_targets(flags, points, root, quat)
    assert labels['target_onset_indices'][5, 0] == 5
    assert not labels['target_occupancy'][6, 0]  # Next onset is 79 steps away.
    assert labels['target_onset_indices'][10, 0] == 85  # Exactly +75.
    assert not labels['target_occupancy'][86, 0]
    assert labels['future_complete'][24] and not labels['future_complete'][25]


def test_yaw_frame_ignores_roll_pitch_and_masks_out_of_view():
    q = Rotation.from_euler('xyz', [0.2, -0.3, np.pi / 2]).as_quat()[[3, 0, 1, 2]]
    np.testing.assert_allclose(yaw_local([1, 3, 4], [1, 2, 3], q), [1, 0, 1], atol=1e-12)
    labels = next_onset_targets(np.ones((12, 1), bool), np.full((12, 1, 3), 2.0),
                               np.zeros((12, 3)), np.tile([1, 0, 0, 0], (12, 1)))
    assert not labels['target_occupancy'].any()
    assert (labels['target_positions_body_yaw'] == 0).all()


def test_force_history_vertical_threshold_and_short_impulse():
    force = np.zeros((20, 4, 2, 3))
    force[:10, 2, 0, 2] = -1.1
    force[:, :, 1, 0] = 100  # Horizontal force alone must not establish contact.
    force[15:19, 0, 1, 2] = 2
    assert measured_force_flags(force).sum() == 10


def test_resampling_uses_slerp_and_does_not_extend_clip(tmp_path):
    q = np.zeros((3, 36))
    q[:, :4] = Rotation.from_euler('z', np.array([0, np.pi / 4, np.pi / 2])[:, None]).as_quat()[:, [3, 0, 1, 2]]
    q[:, 4] = [0, 1, 2]
    path = tmp_path / 'clip.npz'
    np.savez(path, qpos=q, fps=2)
    t, motion, fps = load_omni(path, output_fps=4)
    assert fps == 2 and len(t) == 5 and t[-1] == 1
    np.testing.assert_allclose(np.linalg.norm(motion[:, :4], axis=1), 1)
    np.testing.assert_allclose(motion[:, 4], [0, 0.5, 1, 1.5, 2])


def test_lafan_quaternion_and_translation_are_not_confused(tmp_path):
    q = np.zeros((4, 36))
    q[:, :3] = [4, 5, 6]
    q[:, 6] = 1  # XYZW identity quaternion.
    path = tmp_path / 'lafan.csv'
    np.savetxt(path, q, delimiter=',')
    _, result, _ = load_lafan_csv(path)
    np.testing.assert_allclose(result[:, :4], np.tile([1, 0, 0, 0], (len(result), 1)))
    np.testing.assert_allclose(result[:, 4:7], np.tile([4, 5, 6], (len(result), 1)))
