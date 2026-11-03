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
from acp.quality.compare import NativeProbe
from acp.quality.support import geometric_stance, prefer_temporal, write_contact_audit


def test_all_body_review_excludes_lost_knee_contact_without_rewriting_offline_gate(tmp_path):
    names=['left_foot','right_foot','left_knee','right_knee','left_hand','right_hand',
           'left_elbow','right_elbow','pelvis']
    rows=[]
    for clip,retained in [('lost_knee',10),('kept_knee',20)]:
        folder=tmp_path/'clips'/clip
        (folder/'retargeted').mkdir(parents=True);(folder/'validation').mkdir()
        intent=np.zeros((20,9),bool);intent[:,2]=True
        distances=np.full((20,9),.1);distances[:retained,2]=.001
        np.savez_compressed(folder/'retargeted'/f'{clip}_original.npz',fps=50.,
            source_geometric_contact_intent=intent,source_geometric_contact_body_names=names)
        np.savez_compressed(folder/'validation/mujoco_geometry.npz',distances=distances)
        rows.append(dict(clip=clip,frames=20,stage1_offline_eligible=True))
    path=tmp_path/'quality_all_clips.json';path.write_text(json.dumps(rows));original=path.read_bytes()
    summary=write_contact_audit(tmp_path)
    assert summary['review_clips']==1 and summary['conservative_candidates']==1
    assert (tmp_path/'conservative_candidate_clips.txt').read_text()=='kept_knee\n'
    assert (tmp_path/'contact_review_clips.txt').read_text()=='lost_knee\n'
    assert path.read_bytes()==original


def test_temporal_improvement_never_excuses_a_new_physical_failure():
    criteria={'maximum_contact_foot_planar_speed_p95_m_s':.15,
              'minimum_stance_near_surface_fraction_per_foot':.9}
    def quality(speed,reasons):
        return dict(status='needs_repair',reasons=reasons,foot_metrics={
            'left':{'contact_tangential_speed_p95_m_s':speed,'stance_near_surface_fraction':1.}})
    baseline=quality(.3,['left_foot_sliding'])
    assert prefer_temporal(quality(.2,['left_foot_sliding']),baseline,criteria)
    assert not prefer_temporal(quality(.1,['environment_penetration']),baseline,criteria)
    assert not prefer_temporal(quality(.35,['left_foot_sliding']),baseline,criteria)


def test_stationary_airborne_foot_is_not_source_support():
    slow=np.array([[True,True],[True,False],[True,True]])
    distance=np.array([[.002,.12],[-.01,.001],[.016,.015]])
    np.testing.assert_array_equal(geometric_stance(slow,distance),
        [[True,False],[True,False],[False,True]])
    # Source penetration is not silently erased from contact intentions.
    # It must still face the independent post-repair collision gate.
    with pytest.raises(ValueError,match='Nonfinite'):
        geometric_stance(slow,np.full((3,2),np.nan))


@pytest.mark.parametrize('yaw',[0.,np.pi/2])
def test_rolling_contact_point_speed_differs_from_translating_center(yaw):
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
        <geom type="plane" size="2 2 .1"/>
        <body name="left_ankle_roll_link" pos="0 0 .099">
        <freejoint/><geom type="sphere" size=".1"/></body>
        </worldbody></mujoco>''')
    probe=NativeProbe(model)
    q=model.qpos0.copy();q[3:7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
    probe.state(q)
    # MuJoCo's free-joint angular velocity lives in the local body frame.
    velocity=np.array([.2,0,0,2*np.sin(yaw),2*np.cos(yaw),0])
    assert np.linalg.norm(velocity[:2])>.15
    assert probe.tangential_speed(velocity,0)<1e-10
    velocity[3:]=0
    assert probe.tangential_speed(velocity,0)==pytest.approx(.2)
    assert np.isnan(probe.tangential_speed(velocity,1))


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


def test_native_probe_separates_joint_housings_from_distant_body_contacts():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom type="plane" size="2 2 .1"/>
      <body name="a" pos="0 0 .09"><joint/><geom type="sphere" size=".1"/>
        <body name="b" pos="0 0 .01"><joint/><inertial mass=".1" pos="0 0 0" diaginertia=".001 .001 .001"/>
          <body name="c" pos="0 0 .01"><joint/><geom type="sphere" size=".1"/>
            <body name="d" pos="0 0 .01"><joint/><geom type="sphere" size=".1"/></body>
          </body>
        </body>
      </body></worldbody></mujoco>''')
    probe=NativeProbe(model)
    distances,pairs=probe.state(model.qpos0)
    assert distances[0]<-.003
    assert distances[1]<distances[2]<-.003
    assert 'a / d' in pairs
    assert 'a / c' not in pairs  # Two-joint housing overlap remains visible in RAW self.


def test_full_batch_records_one_failure_and_finishes_other_clips(tmp_path,monkeypatch):
    from concurrent.futures import Future
    from types import SimpleNamespace
    from acp.quality import fullbody
    baseline,output=tmp_path/'baseline',tmp_path/'output';baseline.mkdir()
    (baseline/'inventory.json').write_text(json.dumps([{'clip':'broken'},{'clip':'good'}]))
    class Pool:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,fn,row,*args):
            future=Future()
            if row['clip']=='broken':future.set_exception(ValueError('broken input'))
            else:future.set_result(dict(clip='good',quality_status='offline_kinematic_candidate'))
            return future
    monkeypatch.setattr(fullbody,'ProcessPoolExecutor',Pool)
    args=SimpleNamespace(baseline=baseline,output=output,clips=None,workers=1,passes=1,maximum_frames=2)
    with pytest.raises(SystemExit,match='Some clips failed'):
        fullbody.run(args)
    progress=json.loads((output/'progress.json').read_text())
    assert progress['processed']==2 and progress['processing_failed']==1
    assert progress['offline_candidates']==1
    manifest=json.loads((output/'clips/broken/manifest.json').read_text())
    assert manifest['status']=='failed' and not manifest['stage1_training_admitted']


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
