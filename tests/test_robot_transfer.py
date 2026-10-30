"""Source FK convention checks, run in the Holosoma environment with local assets."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np

try:
    import mujoco
except ModuleNotFoundError:
    mujoco = None

SOURCE = Path("/home/lyx/train/retarget/dataset/OmniRetarget_Dataset/models/g1/g1_29dof_spherehand.urdf")


@unittest.skipUnless(mujoco is not None and SOURCE.is_file(), "Needs Holosoma MuJoCo and local Omni model assets")
class RobotFkConventionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "src/acp/robot/transfer.py"
        spec = importlib.util.spec_from_file_location("robot_transfer", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)
        cls.model = cls.module.load_source_model(SOURCE)

    def test_source_quaternion_and_translation_have_expected_world_effect(self):
        q = np.zeros((2, 36))
        q[0, 0] = 1
        q[1, :4] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
        q[:, 4:7] = [0.1, 0.2, 1]
        mapping = {"root": "pelvis", "toe": "left_ankle_roll_sphere_5_link",
                   "palm": "left_sphere_hand_link"}
        points = self.module.fk_keypoints(self.model, q, list(mapping), mapping)
        np.testing.assert_allclose(points[:, 0], np.tile([0.1, 0.2, 1], (2, 1)), atol=1e-9)
        yaw_90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        np.testing.assert_allclose(points[1] - points[1, 0], (points[0] - points[0, 0]) @ yaw_90.T, atol=1e-8)

    def test_missing_source_link_is_rejected(self):
        q = np.r_[1., np.zeros(35)][None]
        with self.assertRaisesRegex(ValueError, "absent body"):
            self.module.fk_keypoints(self.model, q, ["palm"], {"palm": "not_a_robot_link"})


if __name__ == "__main__":
    unittest.main()
