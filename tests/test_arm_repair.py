"""Check nonlinear distance derivatives and protected DoFs on local robot assets."""

from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
SCENE = PROJECT / "outputs/omni_asap23/clips/climb_03_z_scale_1.0/scene.xml"
try:
    import mujoco
except ModuleNotFoundError:
    mujoco = None


@unittest.skipUnless(mujoco is not None and SCENE.is_file(), "Needs MuJoCo and generated ASAP baseline")
class ArmRepairTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from acp.data.motions import resample_drake
        from acp.quality.repair import ArmProblem
        cls.problem = ArmProblem(mujoco.MjModel.from_xml_path(str(SCENE)),
                                 mujoco.MjModel.from_xml_path(str(SCENE.parent / "comparison/source_scene.xml")))
        with np.load(SCENE.parent / "labeled/motion.npz") as z:
            cls.nominal = np.c_[z["base_position_w"], z["base_quaternion_wxyz"], z["joint_positions"]]
        source = Path("/home/lyx/train/retarget/dataset/OmniRetarget_Dataset/robot-terrain/climb_03_z_scale_1.0.npz")
        with np.load(source) as z:
            _, q, _ = resample_drake(z["qpos"], float(z["fps"]), 50)
        cls.source_q = np.c_[q[:, 4:7], q[:, :4], q[:, 7:]]

    def test_distance_jacobian_for_separated_and_penetrating_hand(self):
        problem = self.problem
        for frame in (30, 200):
            problem.set_source(self.source_q[frame], self.nominal[frame])
            x = self.nominal[frame, problem.qids].copy()
            problem.update(x)
            a, b = problem.hand_geoms[0], problem.env[1]
            phi, analytical = problem.distance(a, b, True)
            self.assertGreater(phi, 0) if frame == 30 else self.assertLess(phi, 0)
            numeric = np.zeros(8)
            for index in range(8):
                delta = np.eye(8)[index] * 1e-6
                problem.update(x + delta)
                upper = problem.distance(a, b)
                problem.update(x - delta)
                lower = problem.distance(a, b)
                numeric[index] = (upper - lower) / 2e-6
            np.testing.assert_allclose(analytical, numeric, atol=1e-5, rtol=1e-4)

    def test_repaired_contact_passes_exact_geometry_and_preserves_lower_body(self):
        problem = self.problem
        frame = 200
        problem.set_source(self.source_q[frame], self.nominal[frame])
        x = self.nominal[frame, problem.qids].copy()
        before = problem.all_distances(x)
        self.assertTrue(np.any(before < problem.bounds_distance - 2e-5))
        lower = np.maximum(problem.lower, x - .8)
        upper = np.minimum(problem.upper, x + .8)
        candidate, feasible, _, _, _ = problem.solve(x, np.zeros(8), lower, upper)
        self.assertTrue(feasible)
        self.assertTrue(np.all(problem.all_distances(candidate) >= problem.bounds_distance - 2e-5))
        protected = [i for i in range(problem.model.nq) if i not in set(problem.qids)]
        np.testing.assert_array_equal(problem.data.qpos[protected], self.nominal[frame, protected])


if __name__ == "__main__":
    unittest.main()
