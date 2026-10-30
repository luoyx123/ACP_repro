"""Check the dataset's root convention and named 29-to-23 reduction together."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from acp.robot.joints import WRIST_JOINTS, project_joint_subset
from acp.data.motions import LAFAN_G1_JOINT_NAMES, load_lafan_csv, read_lafan_csv


def test_lafan_xyzw_root_and_named_subset_preserve_source_values():
    raw = np.zeros((4, 36))
    raw[:, :3] = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]
    # Nontrivial rotation, with both quaternion signs represented.
    raw[:, 3:7] = [0, .6, 0, .8]
    raw[2:, 3:7] *= -1
    raw[:, 7:] = np.arange(29) / 100
    names = [n for n in LAFAN_G1_JOINT_NAMES if n not in WRIST_JOINTS]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "motion.csv"
        np.savetxt(path, raw, delimiter=",")
        q = read_lafan_csv(path)
        result = project_joint_subset(q, LAFAN_G1_JOINT_NAMES, names)
        np.testing.assert_array_equal(result[:, :4], raw[:, [6, 3, 4, 5]])
        np.testing.assert_array_equal(result[:, 4:7], raw[:, :3])
        np.testing.assert_array_equal(result[:, 7:], raw[:, [7 + LAFAN_G1_JOINT_NAMES.index(n) for n in names]])
        times, sampled, _ = load_lafan_csv(path, 50)
        assert len(times) == 6
        np.testing.assert_allclose(np.linalg.norm(sampled[:, :4], axis=1), 1)
        np.testing.assert_allclose(sampled[:, 4:7][[0, -1]], raw[[0, -1], :3])


@pytest.mark.parametrize("kind", ["columns", "quaternion", "nan"])
def test_lafan_rejects_ambiguous_or_invalid_input(tmp_path, kind):
    raw = np.zeros((3, 36))
    raw[:, 6] = 1
    if kind == "columns":
        raw = raw[:, :-1]
    elif kind == "quaternion":
        raw[:, 3:7] = 0
    else:
        raw[0, 9] = np.nan
    path = tmp_path / "bad.csv"
    np.savetxt(path, raw, delimiter=",")
    with pytest.raises(ValueError):
        read_lafan_csv(path)
