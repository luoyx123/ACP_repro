"""29DoF 的关节保真、基座反转，以及同一原生模型在 MuJoCo/Drake 的一致性。"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from acp.robot.joints import WRIST_JOINTS, select_g1_joints

RESOURCE = Path('/home/lyx/train/supermimic-extend/InstinctMJ/src/instinct_mj/assets/resources/unitree_g1')
SOURCE = Path('/home/lyx/train/retarget/dataset/OmniRetarget_Dataset/models/g1/g1_29dof_spherehand.urdf')
try:
    import mujoco
except ImportError:
    mujoco = None


class NamedSelectionTest(unittest.TestCase):
    def test_native_reorders_all_real_wrists_without_padding(self):
        names = [f'joint_{i}' for i in range(23)] + list(WRIST_JOINTS)
        q = np.arange(3 * 36).reshape(3, 36).astype(float)
        target = names[::-1]
        result = select_g1_joints(q, names, target)
        np.testing.assert_array_equal(result[:, :7], q[:, :7])
        for i, name in enumerate(target):
            np.testing.assert_array_equal(result[:, 7 + i], q[:, 7 + names.index(name)])
        with self.assertRaises(ValueError):
            select_g1_joints(q[:, :30], names[:23], target)


@unittest.skipUnless(mujoco is not None and RESOURCE.is_dir() and SOURCE.is_file(), 'Needs local MuJoCo/robot assets')
class NativeModelTest(unittest.TestCase):
    def test_pelvis_to_torso_conversion_preserves_wrists_and_orientation(self):
        from acp.robot.instinct import prepare, joint_names, transfer
        from acp.robot.model import compile_robot
        with tempfile.TemporaryDirectory() as directory:
            urdf = prepare(RESOURCE, directory)
            source = compile_robot(SOURCE)
            target = mujoco.MjModel.from_xml_path(str(urdf.with_suffix('.xml')))
            rng = np.random.default_rng(109)
            q = rng.normal(0, .25, (5, 36))
            q[:, :4] /= np.linalg.norm(q[:, :4], axis=1, keepdims=True)
            q[:, 4:7] += [1, 2, 1]
            native, report = transfer(q, source, target)
            sn, tn = joint_names(source), joint_names(target)
            for name in WRIST_JOINTS:
                np.testing.assert_array_equal(native[:, 7 + tn.index(name)], q[:, 7 + sn.index(name)])
            for name in ('waist_pitch_joint', 'waist_roll_joint', 'waist_yaw_joint'):
                np.testing.assert_array_equal(native[:, 7 + tn.index(name)], -q[:, 7 + sn.index(name)])
            self.assertLess(max(report['maximum_position_error_m'].values()), .03)
            src, dst = mujoco.MjData(source), mujoco.MjData(target)
            for a, b in zip(q, native):
                src.qpos[:] = np.r_[a[4:7], a[:4], a[7:]]
                dst.qpos[:] = np.r_[b[4:7], b[:4], b[7:]]
                mujoco.mj_forward(source, src)
                mujoco.mj_forward(target, dst)
                for body in report['links']:
                    np.testing.assert_allclose(src.xmat[source.body(body).id], dst.xmat[target.body(body).id], atol=1e-5)
                np.testing.assert_allclose(src.xpos[source.body('torso_link').id], b[4:7], atol=1e-12)

    def test_drake_labels_use_native_capsules_and_same_kinematic_tree(self):
        from acp.robot.instinct import prepare
        from acp.robot.model import make_scene
        with tempfile.TemporaryDirectory() as directory:
            urdf = prepare(RESOURCE, directory)
            terrain = Path(directory) / 'ground.urdf'
            terrain.write_text('<robot name="ground"><link name="world"/></robot>')
            scene_path = Path(directory) / 'scene.xml'
            model = make_scene(urdf.with_suffix('.xml'), terrain, scene_path)
            data = mujoco.MjData(model)
            names = [model.body(i).name for i in range(1, model.nbody)]
            rng = np.random.default_rng(19)
            fixtures = []
            for height in (.65, .78, .85):
                data.qpos[:] = model.qpos0
                data.qpos[:3] = [.2, -.1, height]
                data.qpos[7:] = rng.normal(0, .15, 29)
                mujoco.mj_forward(model, data)
                ground = model.geom('ground').id
                distances = [mujoco.mj_geomDistance(model, data, g, ground, .1, None)
                    for g in range(model.ngeom) if model.geom_bodyid[g] != 0
                    and (model.geom_contype[g] or model.geom_conaffinity[g])]
                fixtures.append(dict(q=np.r_[data.qpos[3:7], data.qpos[:3], data.qpos[7:]].tolist(),
                                     positions=data.xpos[1:].tolist(), minimum=min(distances)))
            code = '''
import json,sys,numpy as np
from acp.data.scene import MotionScene
payload=json.load(sys.stdin)
s=MotionScene(payload['urdf'],payload['terrain'])
assert s.base_body_name=='torso_link'
for row in payload['fixtures']:
    s.set_positions(row['q'])
    for name,position in zip(payload['names'],row['positions']):
        frame=s.plant.GetFrameByName(name,s.robot)
        actual=s.plant.CalcRelativeTransform(s.plant_context,s.plant.world_frame(),frame).translation()
        np.testing.assert_allclose(actual,position,atol=1e-9)
    minimum=s.contact_geometry(.1)[2]
    np.testing.assert_allclose(minimum,row['minimum'],atol=1e-6)
print('Native FK and capsule distances agree across both engines')
'''
            result = subprocess.run([str(PROJECT / '.venv/bin/python'), '-c', code], input=json.dumps(dict(
                urdf=str(urdf), terrain=str(terrain), names=names, fixtures=fixtures)),
                env={**os.environ, 'PYTHONPATH': str(PROJECT / 'src')}, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
