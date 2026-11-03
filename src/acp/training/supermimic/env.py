"""Batched MuJoCo-Warp collection; CPU MuJoCo is a small correctness fixture."""
from copy import deepcopy
from dataclasses import fields
import numpy as np
import torch

from .assets import compile_model
from .rotations import mv, quat_to_mat
from .fullbody import reduce_contacts, geometry_depths


class Simulator:
    def __init__(self, scene, robot_cfg, config, n, device, *, acknowledge_engine_mapping=False):
        import mujoco
        self.cfg, self.robot_cfg, self.n, self.device = config, robot_cfg, n, device
        effective = dict(robot_cfg)
        if config['dynamics_randomization']:
            effective['friction'] = config['randomization']['dynamics']['ground_friction']
        self.model = compile_model(scene, effective)
        self.cpu = str(device) == 'cpu'
        if self.cpu and n>8:
            raise ValueError('CPU fixture supports at most 8 environments per scene')
        self.dt, self.decimation = robot_cfg['physics_dt'], robot_cfg['decimation']
        self.noise = config['observation_noise']
        self.randomized = config['dynamics_randomization']
        if self.randomized and not acknowledge_engine_mapping:
            raise ValueError('MuJoCo friction/restitution differ from PhysX; acknowledge_engine_mapping required')
        self.qids = torch.tensor([self.model.jnt_qposadr[self.model.joint(k).id] for k in robot_cfg['joint_names']], device=device)
        self.vids = torch.tensor([self.model.jnt_dofadr[self.model.joint(k).id] for k in robot_cfg['joint_names']], device=device)
        self.default = torch.tensor(robot_cfg['default_joint_positions'], device=device)
        self.effort = torch.tensor(robot_cfg['effort_limits'], device=device)
        self.kp = torch.tensor(robot_cfg['kp'], device=device, dtype=torch.float32).expand(n, -1).clone()
        self.kd = torch.tensor(robot_cfg['kd'], device=device, dtype=torch.float32).expand(n, -1).clone()
        self.bias = torch.zeros((n, 23), device=device)
        self.delay_ticks = torch.zeros(n, device=device, dtype=torch.long)
        self.samples = {}
        self._sample_dynamics()
        group = np.full(self.model.nbody, -1, dtype=np.int64)
        for b in range(1, self.model.nbody):
            name = self.model.body(b).name
            if name in robot_cfg['body_links']:
                group[b] = robot_cfg['body_links'].index(name)
            elif self.model.body_jntnum[b] == 0:
                group[b] = group[self.model.body_parentid[b]]
        self.geom_group = torch.tensor(group[self.model.geom_bodyid], device=device)
        self.robot_geom = torch.tensor(self.model.geom_group == 1, device=device)
        link_ids = sorted(set(self.model.geom_bodyid[self.model.geom_group == 1].tolist()))
        self.link_names = [self.model.body(b).name for b in link_ids]
        self.geom_link_group = torch.tensor([link_ids.index(int(b)) if int(b) in link_ids else -1
                                             for b in self.model.geom_bodyid], device=device)
        if self.cpu:
            self.models = [deepcopy(self.model) for _ in range(n)]
            self.data = [mujoco.MjData(m) for m in self.models]
            for i, (m, d) in enumerate(zip(self.models, self.data)):
                for key, value in self.physical.items():
                    getattr(m, key)[:] = value[i].cpu().numpy()
                mujoco.mj_setConst(m, d)
            self.qpos = torch.zeros((n, self.model.nq))
            self.qvel = torch.zeros((n, self.model.nv))
        else:
            import warp as wp
            import mujoco_warp as mw
            self.wp, self.mw = wp, mw
            with wp.ScopedDevice(device):
                # Derived mass/inertia constants must be per-world too, not shared.
                batches = {f.name:n for f in fields(mw.Model)
                           if getattr(f.type, 'shape', ()) and f.type.shape[0] == '*'}
                self.wmodel = mw.put_model(self.model, batch_sizes=batches)
                self.wmodel.stat.meaninertia = wp.full(n, float(self.model.stat.meaninertia), dtype=wp.float32)
                self.wdata = mw.make_data(self.model, nworld=n, nconmax=128, njmax=768)
                for key, value in self.physical.items():
                    wp.to_torch(getattr(self.wmodel, key)).copy_(value)
                mw.set_const(self.wmodel, self.wdata)
                self.qpos, self.qvel = wp.to_torch(self.wdata.qpos), wp.to_torch(self.wdata.qvel)
                self.qfrc = wp.to_torch(self.wdata.qfrc_applied)
                self.contact_ids = wp.array(np.arange(self.wdata.naconmax, dtype=np.int32), dtype=wp.int32)
                self.contact_force = wp.zeros(self.wdata.naconmax, dtype=wp.spatial_vector)
                for key, value in self.physical.items():
                    torch.testing.assert_close(wp.to_torch(getattr(self.wmodel, key)), value,
                                               rtol=1e-5, atol=1e-6, msg='Physics readback: '+key)
        self.previous_action = torch.zeros((n, 23), device=device)
        self.force_history = torch.zeros((n, self.decimation, 9, 3), device=device)
        self.normal_force_history = torch.zeros((n, self.decimation, 9), device=device)
        self.link_force_history = torch.zeros((n, self.decimation, len(link_ids)), device=device)
        self.computed_torque = torch.zeros((n, 23), device=device)
        self.applied_torque = torch.zeros_like(self.computed_torque)
        self.report = dict(engine='mujoco-cpu-fixture' if self.cpu else 'mujoco-warp', randomized=self.randomized,
                           physical_readback_fields=list(self.physical),
                           friction='robot priority 1 over terrain 0; sampled dynamic coefficient is single Coulomb coefficient; static sample recorded only',
                           restitution='e -> damping ratio -ln(e)/sqrt(pi^2+ln(e)^2), e=0 -> 1; solver approximation',
                           contacts='robot-environment only; world normal vectors by 9 groups; paper max abs z > 1N; scalar normal sums are diagnostics only')

    def _sample_dynamics(self):
        def sample(name, shape, bounds):
            value = bounds[0]+torch.rand(shape, device=self.device)*(bounds[1]-bounds[0])
            self.samples[name] = value
            return value
        m, n, device = self.model, self.n, self.device
        tensor = lambda value:torch.tensor(value, dtype=torch.float32, device=device).expand(n, *np.shape(value)).clone()
        self.physical = {key:tensor(getattr(m, key)) for key in
                         ('body_mass', 'body_inertia', 'body_ipos', 'dof_armature', 'geom_friction', 'geom_solref')}
        if self.randomized:
            cfg = self.cfg['randomization']
            d = cfg['dynamics']
            torso = m.body(cfg['torso_body']).id
            legs = [m.body(name).id for name in cfg['leg_bodies'] if name in [m.body(i).name for i in range(m.nbody)]]
            masses = self.physical['body_mass']
            nominal = masses.clone()
            masses[:, torso] += sample('torso_added_mass', (n,), d['torso_added_mass'])
            masses[:, legs] *= sample('leg_mass_scale', (n, len(legs)), d['leg_mass_scale'])
            if (masses[:, torso]<=0).any():
                raise ValueError('Nonpositive randomized torso mass')
            self.physical['body_inertia'] *= torch.where(nominal>0, masses/nominal.clamp(min=1e-12), 1.)[..., None]
            delta = sample('torso_com_delta', (n, 3), [-1, 1])*self.default.new_tensor(d['torso_com_half_range'])
            self.physical['body_ipos'][:, torso] += delta
            self.samples['torso_com_delta'] = delta
            static = sample('static_friction', (n,), d['static_friction'])
            dynamic = sample('dynamic_friction', (n,), d['dynamic_friction'])
            restitution = sample('restitution', (n,), d['restitution'])
            robot_geoms = torch.tensor(m.geom_group==1, device=device)
            self.physical['geom_friction'][:, robot_geoms, 0] = dynamic[:, None]
            loge = restitution.clamp(min=1e-9).log()
            self.physical['geom_solref'][:, robot_geoms, 1] = (-loge/(torch.pi**2+loge.square()).sqrt())[:, None]
            self.physical['dof_armature'][:, self.vids] *= sample('armature_scale', (n, 23), d['armature_scale'])
            self.kp *= sample('kp_scale', (n, 23), d['pd_gain_scale'])
            self.kd *= sample('kd_scale', (n, 23), d['pd_gain_scale'])
            ankle = self.default.new_tensor([d['ankle_zero_bias'] if 'ankle' in k else d['joint_zero_bias'] for k in self.robot_cfg['joint_names']])
            self.bias = sample('joint_zero_bias', (n, 23), [-1, 1])*ankle
            self.samples['joint_zero_bias'] = self.bias
            self.delay_ticks = (sample('action_delay_seconds', (n,), d['action_delay_seconds'])/self.dt).round().long()
        # Normalize constant contexts with fixed physical scales; no changing distribution in replay.
        scale = lambda key: self.physical[key]/tensor(getattr(m, key)).clamp(min=1e-9)
        robot_bodies = [i for i in range(1, m.nbody) if not m.body(i).name.startswith('multi_boxes_')]
        robot_geom = int(np.flatnonzero(m.geom_group==1)[0])
        self.context = torch.cat((self.bias/.03, self.kp/self.default.new_tensor(self.robot_cfg['kp'])-1,
                                  self.kd/self.default.new_tensor(self.robot_cfg['kd'])-1,
                                  scale('body_mass')[:, robot_bodies]-1,
                                  (self.physical['body_ipos'][:, m.body('torso_link').id]-tensor(m.body_ipos)[:, m.body('torso_link').id])/.05,
                                  scale('dof_armature')[:, self.vids]-1,
                                  self.physical['geom_friction'][:, robot_geom, :1],
                                  self.physical['geom_solref'][:, robot_geom, 1:2], self.delay_ticks[:, None]/4), -1)

    def state(self):
        R = quat_to_mat(self.qpos[:, 3:7])
        return (self.qpos[:, :3].clone(), R, self.qvel[:, :3].clone(), mv(R, self.qvel[:, 3:6]),
                self.qpos[:, self.qids].clone(), self.qvel[:, self.vids].clone())

    def reset(self, values, ids):
        import mujoco
        self.qpos[ids, :3] = values['base_position_w']
        self.qpos[ids, 3:7] = values['base_quaternion_wxyz']
        self.qpos[ids[:, None], self.qids] = values['joint_positions']
        self.qvel[ids, :3] = values['base_linear_velocity_w']
        R = quat_to_mat(values['base_quaternion_wxyz'])
        self.qvel[ids, 3:6] = mv(R.transpose(-1, -2), values['base_angular_velocity_w'])
        self.qvel[ids[:, None], self.vids] = values['joint_velocities']
        self.previous_action[ids] = (values['joint_positions']-self.default)/self.robot_cfg['action_scale']
        self.force_history[ids] = 0
        self.normal_force_history[ids] = 0
        self.link_force_history[ids] = 0
        if self.cpu:
            for i in ids.tolist():
                mujoco.mj_resetData(self.models[i], self.data[i])
                self.data[i].qpos[:] = self.qpos[i].numpy()
                self.data[i].qvel[:] = self.qvel[i].numpy()
                mujoco.mj_forward(self.models[i], self.data[i])
        else:
            self.wp.to_torch(self.wdata.qacc_warmstart)[ids] = 0
            self.wp.to_torch(self.wdata.qfrc_applied)[ids] = 0
            with self.wp.ScopedDevice(self.device):
                self.mw.forward(self.wmodel, self.wdata)

    def packed_contacts(self):
        """The same shape/validity contract for CPU and Warp packed buffers."""
        if not self.cpu:
            wp, data = self.wp, self.wdata
            count = wp.to_torch(data.nacon).reshape(-1)[0]
            torch._assert_async(count <= data.naconmax,
                                'Contact buffer overflow: contact diagnostics are unreliable')
            geom = wp.to_torch(data.contact.geom).long()
            world = wp.to_torch(data.contact.worldid).long()
            active = torch.arange(len(geom), device=self.device) < count
            return geom, world, wp.to_torch(data.contact.dist), active
        geom, world, dist = [], [], []
        for i, data in enumerate(self.data):
            for contact in data.contact:
                geom.append((int(contact.geom1), int(contact.geom2)))
                world.append(i)
                dist.append(float(contact.dist))
        return (torch.tensor(geom, dtype=torch.long).reshape(-1, 2),
                torch.tensor(world, dtype=torch.long), torch.tensor(dist, dtype=torch.float32),
                torch.ones(len(dist), dtype=torch.bool))

    def geometry_depths(self):
        return geometry_depths(*self.packed_contacts(), self.robot_geom, self.n)

    def measured_contacts(self):
        """ACP Table 3: preserve vertical force > 1 N, including all nine bodies."""
        return (self.force_history[..., 2].abs().amax(1) > 1.).float()

    def forces(self):
        """Existing normal-only sensor mapping; self collisions cannot supply support."""
        import mujoco
        geom, world, _, active = self.packed_contacts()
        if self.cpu:
            frames, forces = [], []
            for i, (m, d) in enumerate(zip(self.models, self.data)):
                for c in range(d.ncon):
                    contact = d.contact[c]
                    f = np.zeros(6)
                    mujoco.mj_contactForce(m, d, c, f)
                    frames.append(contact.frame.reshape(3, 3)[0].copy())
                    forces.append(f[0])
            frame = torch.tensor(np.asarray(frames).reshape(-1, 3), dtype=torch.float32)
            force = torch.tensor(forces, dtype=torch.float32)
            vectors, scalar = reduce_contacts(geom, world, force, frame, active, self.robot_geom, self.geom_group, self.n)
            _, links = reduce_contacts(geom, world, force, frame, active, self.robot_geom, self.geom_link_group, self.n, len(self.link_names))
            return vectors, scalar, links
        wp, mw, data = self.wp, self.mw, self.wdata
        self.contact_force.zero_()
        with wp.ScopedDevice(self.device):
            mw.contact_force(self.wmodel, data, self.contact_ids, False, self.contact_force)
        force = wp.to_torch(self.contact_force)[:, 0]
        frame = wp.to_torch(data.contact.frame)[:, 0]
        vectors, scalar = reduce_contacts(geom, world, force, frame, active, self.robot_geom, self.geom_group, self.n)
        _, links = reduce_contacts(geom, world, force, frame, active, self.robot_geom, self.geom_link_group, self.n, len(self.link_names))
        return vectors, scalar, links

    def step(self, action):
        import mujoco
        if action.shape!=(self.n, 23) or not torch.isfinite(action).all():
            raise ValueError('Invalid physical action')
        action = action.clamp(-100, 100)
        for tick in range(self.decimation):
            delayed = torch.where((tick>=self.delay_ticks)[:, None], action, self.previous_action)
            target = self.default+self.bias+self.robot_cfg['action_scale']*delayed
            computed = self.kp*(target-self.qpos[:, self.qids])-self.kd*self.qvel[:, self.vids]
            applied = computed.clamp(-self.effort, self.effort)
            self.computed_torque, self.applied_torque = computed, applied
            if self.cpu:
                for i, (m, d) in enumerate(zip(self.models, self.data)):
                    d.qfrc_applied[self.vids.numpy()] = applied[i].numpy()
                    mujoco.mj_step(m, d)
                    self.qpos[i] = torch.from_numpy(d.qpos.copy())
                    self.qvel[i] = torch.from_numpy(d.qvel.copy())
            else:
                self.qfrc[:, self.vids] = applied
                with self.wp.ScopedDevice(self.device):
                    self.mw.step(self.wmodel, self.wdata)
            self.force_history[:, tick], self.normal_force_history[:, tick], self.link_force_history[:, tick] = self.forces()
        self.previous_action.copy_(action)
        return self.state()

    def imagined_torque(self, S, action, previous_action, env_ids):
        """Last PD substep approximation; real labels use actual pre-substep q/qd."""
        delayed = torch.where((self.delay_ticks[env_ids]<self.decimation)[:, None], action, previous_action)
        target = self.default+self.bias[env_ids]+self.robot_cfg['action_scale']*delayed
        computed = self.kp[env_ids]*(target-S[4])-self.kd[env_ids]*S[5]
        return computed.clamp(-self.effort, self.effort), computed

    def restore_physics(self, state):
        """Restore the checkpoint's actual randomization, not a fresh random sample."""
        import mujoco
        self.samples = {k:v.to(self.device) for k, v in state.get('samples', {}).items()}
        for key in ('kp', 'kd', 'bias', 'delay_ticks', 'context'):
            value = state[key].to(self.device)
            if value.shape!=getattr(self, key).shape:
                raise ValueError('Resume environment count/context differs')
            getattr(self, key).copy_(value)
        for key, value in state['physical'].items():
            self.physical[key].copy_(value.to(self.device))
        if self.cpu:
            for i, (m, d) in enumerate(zip(self.models, self.data)):
                for key, value in self.physical.items():
                    getattr(m, key)[:] = value[i].numpy()
                mujoco.mj_setConst(m, d)
        else:
            with self.wp.ScopedDevice(self.device):
                for key, value in self.physical.items():
                    self.wp.to_torch(getattr(self.wmodel, key)).copy_(value)
                self.mw.set_const(self.wmodel, self.wdata)
