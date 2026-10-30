"""Physical invariants on actual local assets; run also in the Holosoma environment."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "src"))
SOURCE = Path("/home/lyx/train/retarget/dataset/OmniRetarget_Dataset/models/g1/g1_29dof_spherehand.urdf")
ASAP = Path("/home/lyx/train/ASAP")
try:
    import mujoco
except ModuleNotFoundError:
    mujoco = None


@unittest.skipUnless(mujoco is not None and SOURCE.is_file() and ASAP.is_dir(), "Needs MuJoCo and local source/ASAP assets")
class AsapFkTest(unittest.TestCase):
    def test_wrist_fixing_preserves_feet_torso_and_zero_wrist_full_robot(self):
        from acp.robot.joints import WRIST_JOINTS, project_joint_subset
        from acp.robot.model import compile_robot, prepare

        with tempfile.TemporaryDirectory() as directory:
            urdf = prepare(SOURCE, ASAP, directory)
            source, target = compile_robot(SOURCE), mujoco.MjModel.from_xml_path(str(urdf.with_suffix(".xml")))
            source_names = [source.joint(i).name for i in range(1, source.njnt)]
            target_names = [target.joint(i).name for i in range(1, target.njnt)]
            src, dst = mujoco.MjData(source), mujoco.MjData(target)
            rng = np.random.default_rng(25)
            q = rng.normal(0, .2, (8, 36))
            q[:, :4] /= np.linalg.norm(q[:, :4], axis=1)[:, None]
            q[:, 4:7] += [1, 2, 1]
            reduced = project_joint_subset(q, source_names, target_names)
            for canonical, projected in zip(q, reduced):
                src.qpos[:] = np.r_[canonical[4:7], canonical[:4], canonical[7:]]
                dst.qpos[:] = np.r_[projected[4:7], projected[:4], projected[7:]]
                mujoco.mj_forward(source, src)
                mujoco.mj_forward(target, dst)
                for name in ["pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link",
                             "left_elbow_link", "right_elbow_link"]:
                    np.testing.assert_allclose(src.xpos[source.body(name).id], dst.xpos[target.body(name).id], atol=1e-12)
                    np.testing.assert_allclose(src.xmat[source.body(name).id], dst.xmat[target.body(name).id], atol=1e-12)
                for name in WRIST_JOINTS:
                    src.qpos[source.joint(name).qposadr[0]] = 0
                mujoco.mj_forward(source, src)
                for index in range(1, source.nbody):
                    name = source.body(index).name
                    np.testing.assert_allclose(src.xpos[index], dst.xpos[target.body(name).id], atol=1e-12)
                    np.testing.assert_allclose(src.xmat[index], dst.xmat[target.body(name).id], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
