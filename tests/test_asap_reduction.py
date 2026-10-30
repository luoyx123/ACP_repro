import numpy as np
import pytest

from acp.robot.joints import WRIST_JOINTS, project_joint_subset


def test_asap_subset_preserves_root_and_named_joints_under_reordering():
    names = [f"joint_{index}" for index in range(23)] + list(WRIST_JOINTS)
    target = list(reversed(names[:23]))
    q = np.arange(4 * 36, dtype=float).reshape(4, 36)
    result = project_joint_subset(q, names, target)
    np.testing.assert_array_equal(result[:, :7], q[:, :7])
    for index, name in enumerate(target):
        np.testing.assert_array_equal(result[:, 7 + index], q[:, 7 + names.index(name)])
    np.testing.assert_array_equal(q, np.arange(4 * 36).reshape(4, 36))


def test_removing_waist_instead_of_wrists_is_not_asap_subset():
    names = [f"joint_{index}" for index in range(23)] + list(WRIST_JOINTS)
    with pytest.raises(ValueError, match="exactly the six wrist"):
        project_joint_subset(np.zeros((3, 36)), names, names[1:23] + [WRIST_JOINTS[0]])
