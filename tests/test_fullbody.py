"""Full-body support filtering and native geometry repair regressions."""
from pathlib import Path
import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')
mujoco = pytest.importorskip('mujoco')
from acp.data.assets import PROJECT
from acp.training.supermimic.fullbody import reduce_contacts, geometry_depths
from acp.training.supermimic.runner import Runner
from acp.quality.fullbody import GeometryRepair
from acp.quality.evaluate import contact_speed_mask


def test_wall_contacts_are_diagnostics_without_changing_paper_fz_definition():
    # floor=0, left hand=1, right hand=2. The self pair is not support.
    geom = torch.tensor([[0, 1], [1, 0], [1, 2], [-1, 2], [0, 2]])
    world = torch.tensor([0, 0, 0, 0, 7])
    normal = torch.tensor([8., 8., 100., 100., 100.])
    frame = torch.tensor([[1., 0, 0]] * 5)
    vector, scalar = reduce_contacts(geom, world, normal, frame, torch.ones(5, dtype=torch.bool),
                                    torch.tensor([False, True, True]), torch.tensor([-1, 4, 5]), 1)
    assert not (vector[..., 2].abs() > 1.).any()  # ACP vertical threshold stays unchanged.
    assert scalar[0, 4] == 16.  # Opposite normals do not hide physical wall contacts.
    assert scalar[0, 5] == 0.  # No fake hand support from the robot's own body.
    assert vector.norm() == 0.


def test_all_nine_bodies_and_stale_packed_slots():
    geom = torch.tensor([[0, k+1] for k in range(9)] + [[1, 2], [0, 1]])
    active = torch.tensor([True]*10+[False])
    world = torch.zeros(11, dtype=torch.long)
    robots = torch.tensor([False]+[True]*9)
    groups = torch.arange(-1, 9)
    vector, scalar = reduce_contacts(geom, world, torch.full((11,), 5.),
        torch.tensor([[0., 0, 1.]]*11), active, robots, groups, 1)
    torch.testing.assert_close(vector[0, :, 2], torch.full((9,), 5.))
    torch.testing.assert_close(scalar, torch.full((1, 9), 5.))
    depths = geometry_depths(geom, world, torch.tensor([-.002]*9+[-.04, float('nan')]), active, robots, 1)
    assert abs(float(depths[0][0])-.002) < 1e-7
    assert abs(float(depths[1][0])-.04) < 1e-7


def test_links_outside_nine_paper_bodies_are_still_diagnosed():
    args = (torch.tensor([[0, 1]]), torch.tensor([0]), torch.tensor([50.]),
            torch.tensor([[1., 0, 0]]), torch.tensor([True]), torch.tensor([False, True]))
    _, paper = reduce_contacts(*args, torch.tensor([-1, -1]), 1)
    _, diagnostic = reduce_contacts(*args, torch.tensor([-1, 0]), 1, num_groups=1)
    assert paper.sum() == 0  # Extra torso/head links do not change ACP's nine labels.
    assert diagnostic[0, 0] == 50


def test_low_swing_foot_is_not_mislabeled_as_stance_sliding():
    # A swinging foot stays inside the 15mm geometric band but is moving fast.
    contacts = np.ones(20,dtype=bool)
    stance = np.r_[np.ones(10,dtype=bool),np.zeros(10,dtype=bool)]
    speed = np.r_[np.full(10,.03),np.full(10,1.8)]
    assert np.percentile(speed[contact_speed_mask(contacts)],95)>1.
    valid = contact_speed_mask(contacts,stance)
    assert valid.sum()==8
    assert np.percentile(speed[valid],95)<.15


def test_native_collision_jacobian_and_repair_reduce_hand_penetration():
    root = PROJECT/'outputs/数据集/omni/instinct29_v1/clips/climb_00_z_scale_0.9'
    if not root.exists():
        pytest.skip('Native29 fixture not materialized')
    problem = GeometryRepair(mujoco.MjModel.from_xml_path(str(root/'scene.xml')))
    q = np.load(root/'retargeted/climb_00_z_scale_0.9_original.npz')['qpos'][150].copy()
    problem.set(q)
    contact = min(problem.collisions(), key=lambda c:c.dist)
    a, b = int(contact.geom1), int(contact.geom2)
    p = contact.pos.copy()
    normal = contact.frame.reshape(3, 3)[0].copy()
    jac = normal @ (problem.point_jac(problem.model.geom_bodyid[b], p) -
                    problem.point_jac(problem.model.geom_bodyid[a], p))
    numeric = []
    for index in problem.qids:
        values = []
        for sign in (-1., 1.):
            shifted = q.copy(); shifted[index] += sign*1e-6; problem.set(shifted)
            values.append(mujoco.mj_geomDistance(problem.model, problem.data, a, b, .5, None))
        numeric.append((values[1]-values[0])/2e-6)
    np.testing.assert_allclose(jac, numeric, atol=3e-5)
    before = problem.worst(q)
    fixed, status = problem.frame(q, np.zeros(len(problem.qids)))
    assert problem.worst(fixed) >= before - 1e-6
    assert problem.worst(fixed) >= -.003
    np.testing.assert_array_equal(fixed[3:7], q[3:7])
    # A box-climb configuration where linearized contact features oscillated.
    corner = np.load(root/'retargeted/climb_00_z_scale_0.9_original.npz')['qpos'][845].copy()
    fixed, status = problem.frame(corner, np.zeros(len(problem.qids)))
    assert problem.worst(fixed) >= -.003
    assert status['status'] == 'exact_distance_fallback'


def test_initializer_rejection_does_not_become_policy_failure_or_replay():
    root = PROJECT/'outputs/数据集/supermimic/walk_box_v1'
    if not (root/'package.json').exists():
        pytest.skip('SuperMimic fixture not materialized')
    cfg = json.loads((PROJECT/'configs/supermimic.json').read_text())
    cfg.update(world_model_hidden=[16], compile=False)
    runner = Runner(root, cfg, n=2, device='cpu')
    g = runner.groups[0]
    phase, size = g.phase.clone(), runner.buffers[0].size
    g.initialization = {**g.initialization, 'maximum_resamples':1}
    g.sim.geometry_depths = lambda:(torch.full((g.n,), .1), torch.zeros(g.n))
    with pytest.raises(ValueError, match='invalid initializations'):
        g.reset()
    torch.testing.assert_close(g.phase, phase)
    assert runner.buffers[0].size == size
    assert runner.optimizer_steps == 0
