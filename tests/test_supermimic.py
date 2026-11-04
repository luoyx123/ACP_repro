"""World-model geometry, replay boundaries, and actual ACP observation parity."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
mujoco = pytest.importorskip('mujoco')
from acp.data.assets import PROJECT
from acp.training.supermimic.assets import compile_model, verify
from acp.training.supermimic.model import Kinematics, Terrain, WorldModel, pack, unpack
from acp.training.supermimic.rotations import quat_to_mat, so3_exp, mat_to_quat, mv
from acp.training.supermimic.runner import Replay, Runner
from acp.training import task as task_math


@pytest.fixture(scope='module')
def package():
    root = PROJECT/'outputs/数据集/supermimic/walk_box_v1'
    if not (root/'package.json').is_file():
        pytest.skip('Use supermimic prepare to create the two-clip fixture')
    return root, verify(root)


def test_replay_ring_masks_resets_and_keeps_terminal_transition():
    replay = Replay(4)
    for step in range(6):
        replay.append(dict(s=torch.full((2, 1), float(step)), s1=torch.full((2, 1), float(step+1)),
                           episode=torch.full((2,), 0 if step<3 else 1)))
    window = replay.sample(8, 4)
    torch.testing.assert_close(window['s'][0, :, 0], torch.tensor([2., 3., 4., 5.]))
    torch.testing.assert_close(window['s1'][0, :, 0], torch.tensor([3., 4., 5., 6.]))
    assert window['mask'][0].tolist()==[True, False, False, False]


def test_multiscene_compiled_world_model_backward_uses_frozen_statistics(package):
    root,_=package
    cfg=json.loads((PROJECT/'configs/supermimic.json').read_text())
    cfg.update(world_model_hidden=[16,16],horizon=2,wm_window=3,wm_batch=4,
               pi_batch=4,buffer_steps=8,warmup_steps=4,compile=False)
    runner=Runner(root,cfg,n=4,device='cpu')
    runner.wm_step=torch.compile(runner.wm.step,backend='aot_eager')
    runner.collect(4)
    loss=runner.wm_loss()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in runner.wm.parameters() if p.grad is not None)


def test_warp_mesh_margin_contract_disables_multiccd_but_keeps_contact_offset(package):
    root, manifest = package
    cfg=manifest['robot_config']
    model=compile_model(root/manifest['clips'][1]['scene'],cfg)
    if hasattr(mujoco.mjtDisableBit,'mjDSBL_MULTICCD'):
        assert model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    else:
        assert not model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_MULTICCD)
    active=(model.geom_contype!=0)|(model.geom_conaffinity!=0)
    np.testing.assert_allclose(model.geom_margin[active],cfg.get('contact_offset',.002))


def test_residual_dynamics_identity_and_zero_rotation_gradients():
    S = (torch.randn(3, 3), so3_exp(torch.randn(3, 3)*.1), torch.zeros(3, 3), torch.zeros(3, 3),
         torch.randn(3, 23)*.1, torch.zeros(3, 23))
    for a, b in zip(S, unpack(pack(S))):
        torch.testing.assert_close(a, b)
    wm = WorldModel(2, [32])
    action = torch.randn(3, 23, requires_grad=True)
    S1, logits = wm.step(S, action, torch.zeros(3, 2), torch.zeros(3, 187))
    for a, b in zip(S, S1):
        torch.testing.assert_close(a, b)
    delta = wm.delta(S, S1)
    torch.testing.assert_close(delta, torch.zeros_like(delta), atol=1e-6, rtol=0)
    (delta.square().sum()+logits.square().sum()).backward()
    assert torch.isfinite(action.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in wm.parameters() if p.grad is not None)


@pytest.mark.parametrize('clip_id', [0, 1])
def test_named_fk_and_terrain_match_mujoco(package, clip_id):
    root, manifest = package
    cfg, clip = manifest['robot_config'], manifest['clips'][clip_id]
    model = compile_model(root/clip['scene'], cfg)
    kin, terrain = Kinematics(model, cfg, 'cpu'), Terrain(root/clip['scene'], 'cpu')
    data = mujoco.MjData(model)
    rng = np.random.default_rng(42)
    q = np.asarray(cfg['default_joint_positions'])+rng.normal(0, .2, 23)
    qids = [model.jnt_qposadr[model.joint(n).id] for n in cfg['joint_names']]
    data.qpos[:7] = [0.3, -.1, .9, 1, 0, 0, 0]
    data.qpos[qids] = q
    mujoco.mj_forward(model, data)
    S = (torch.tensor(data.qpos[:3], dtype=torch.float32)[None], torch.eye(3)[None],
         torch.zeros(1, 3), torch.zeros(1, 3), torch.tensor(q, dtype=torch.float32)[None], torch.zeros(1, 23))
    exact = []
    for name, offset in zip(cfg['body_links'], cfg['landmark_offsets']):
        bid = model.body(name).id
        exact.append(data.xpos[bid]+data.xmat[bid].reshape(3, 3)@offset)
    np.testing.assert_allclose(kin(S)[0].numpy(), exact, atol=2e-6)
    for count in [63, 187]:
        scan = terrain.scan(S, count)[0].numpy()
        for index, point in enumerate(terrain.grids[count].numpy()):
            origin = np.r_[point+data.qpos[:2], data.qpos[2]+20]
            distance = mujoco.mj_ray(model, data, origin, np.array([0., 0., -1.]),
                                     np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8), 1, -1, np.zeros(1, dtype=np.int32))
            assert distance>=0
            assert abs(scan[index]-(20-distance))<2e-5


def tiny_config():
    cfg = json.loads((PROJECT/'configs/supermimic.json').read_text())
    return {**cfg, 'world_model_hidden':[32, 32], 'horizon':2, 'wm_window':2,
            'wm_batch':2, 'pi_batch':2, 'buffer_steps':8, 'warmup_steps':4, 'compile':False,
            'wm_batches':1, 'wm_only_iters':0, 'critic_epochs':1, 'critic_minibatches':1}


def test_real_observations_match_original_stage1_and_resume_physics(package, tmp_path):
    root, _ = package
    torch.set_num_threads(2)
    runner = Runner(root, tiny_config(), n=2, device='cpu')
    # The imagined reward must have finite gradients even at EXACT tracking,
    # including zero root-distance and coincident rotations.
    g = runner.groups[0]
    ref = g.mdp.reference.sample(g.times)
    exact = (ref['base_position_w'].clone().requires_grad_(), quat_to_mat(ref['base_quaternion_wxyz']).requires_grad_(),
             ref['base_linear_velocity_w'].clone().requires_grad_(), ref['base_angular_velocity_w'].clone().requires_grad_(),
             ref['joint_positions'].clone().requires_grad_(), ref['joint_velocities'].clone().requires_grad_())
    zeros = torch.zeros_like(exact[4])
    tr = g.mdp.transition(exact, exact, g.times, zeros, zeros, zeros, torch.zeros(g.n, 9, 3),
                           ref['contact_flags'].float(), zeros, zeros, torch.ones(g.n, dtype=torch.long))
    (tr['reward'].sum()+tr['continuation'].sum()).backward()
    assert all(v.grad is None or torch.isfinite(v.grad).all() for v in exact)
    # Execute the original assembly without importing IsaacLab.
    source = ast.parse((PROJECT/'src/acp/envs/stage1.py').read_text())
    method = next(n for c in source.body if isinstance(c, ast.ClassDef) and c.name=='Stage1Task'
                  for n in c.body if isinstance(n, ast.FunctionDef) and n.name=='observations')
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {name:getattr(task_math, name) for name in ['yaw', 'wrap', 'yaw_local', 'inverse_rotate']}
    namespace['torch'] = torch
    exec(compile(ast.fix_missing_locations(module), '<original-observations>', 'exec'), namespace)
    for g in runner.groups:
        g.training_noise = False
        S = g.sim.state()
        quat = mat_to_quat(S[1])
        data = SimpleNamespace(root_link_pos_w=S[0], root_link_quat_w=quat,
                               root_link_lin_vel_b=mv(S[1].transpose(-1, -2), S[2]))
        def height_scan(planner=False, **_kwargs):
            heights = g.mdp.terrain.scan(S, 187 if planner else 63)
            return None, heights, torch.ones_like(heights, dtype=torch.bool)
        env = SimpleNamespace(robot=SimpleNamespace(data=data), origins=torch.zeros_like(S[0]),
                              proprio=lambda:g.mdp.proprio(S, g.sim.previous_action), height_scan=height_scan)
        mock = SimpleNamespace(env=env, randomization=SimpleNamespace(observed_pose=lambda clean:(S[0], quat)),
                               reference=g.mdp.reference, time=g.times, evaluating=True, cfg=g.cfg,
                               actor_history=g.history, n=g.n)
        expected = namespace['observations'](mock)
        actual = g.observations()
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    runner.collect(4)
    # Check both gradient paths without optimizer updates.
    runner.backward(runner.wm_loss(), runner.wm_optimizer, runner.wm.parameters(), apply=False)
    for p in [*runner.wm.parameters(), *runner.policy.critic.parameters()]:
        p.requires_grad_(False)
    loss, _ = runner.policy_loss()
    runner.backward(loss, runner.pi_optimizer, runner.policy.actor.parameters(), apply=False)
    assert runner.optimizer_steps==0
    runner.save(tmp_path)
    # Deliberately corrupt the fresh simulator to prove checkpoint restoration writes it back.
    kp = runner.groups[0].sim.kp.clone()
    runner.groups[0].sim.kp *= 0
    runner.load(tmp_path/'latest.pt')
    torch.testing.assert_close(runner.groups[0].sim.kp, kp)
    for key, expected in runner.groups[0].sim.physical.items():
        np.testing.assert_allclose(getattr(runner.groups[0].sim.models[0], key), expected[0].numpy(), atol=1e-6)
