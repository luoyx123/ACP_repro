import numpy as np

from acp.data.adapters.holosoma import holosoma_to_canonical, resample_holosoma
from acp.data.motions import resample_drake


def test_mujoco_qpos_to_canonical_base_order():
    qpos = np.array([[1, 2, 3, 1, 0, 0, 0, 0.2],
                     [2, 3, 4, 1, 0, 0, 0, 0.4]], dtype=float)
    canonical = holosoma_to_canonical(qpos)
    np.testing.assert_allclose(canonical[0], [1, 0, 0, 0, 1, 2, 3, 0.2])


def test_resample_uses_fixed_output_period_and_joint_linear_interpolation():
    qpos = np.array([[0, 0, 1, 1, 0, 0, 0, 0],
                     [1, 0, 1, 1, 0, 0, 0, 1],
                     [2, 0, 1, 1, 0, 0, 0, 2]], dtype=float)
    times, canonical = resample_holosoma(qpos, fps=25, output_fps=50)

    np.testing.assert_allclose(times, [0, 0.02, 0.04, 0.06, 0.08])
    np.testing.assert_allclose(canonical[:, 7], [0, 0.5, 1, 1.5, 2])
    np.testing.assert_allclose(canonical[:, 4], [0, 0.5, 1, 1.5, 2])


def test_same_rate_keeps_all_252_frames_despite_float_roundoff():
    # (251 / 50) * 50 is 250.99999999999997: floor-based re-sampling
    # previously dropped the last frame of every climb_17 export.
    qpos = np.tile([0, 0, 1, 1, 0, 0, 0, 0.], (252, 1))
    qpos[:, -1] = np.arange(252) / 1000
    times, canonical = resample_holosoma(qpos, fps=50, output_fps=50)
    assert len(times) == 252
    np.testing.assert_allclose(canonical[:, -1], qpos[:, -1])
    assert times[-1] == 251 / 50


def test_30_to_50_rate_conversion_retains_the_523_frame_source_endpoint():
    # 522 / 30 * 50 can round just below 870. The endpoint is on the
    # 50 Hz clock and must not disappear from the climb_01 conversion.
    source = np.zeros((523, 36))
    source[:, 0] = 1
    source[:, 4] = np.linspace(0, 1, 523)
    times, converted, _ = resample_drake(source, fps=30, output_fps=50)
    assert len(times) == 871
    np.testing.assert_allclose(converted[-1], source[-1], atol=1e-12)
    assert times[-1] <= (len(source) - 1) / 30 + 1e-12
    np.testing.assert_allclose(np.diff(times), 1 / 50, atol=1e-12)
