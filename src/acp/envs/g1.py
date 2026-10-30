"""Minimal batched G1 environment for reference-conditioned control in PhysX.

Import only after IsaacLab AppLauncher has started. No reward or policy training
is implemented here; all reset/step tensors use the canonical data joint order.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema, Usd
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import quat_apply
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
from acp.data.assets import PROJECT, read_record, resolve_path
from acp.envs.reference import ReferenceMotion, name_indices
from acp.envs.conversion import prepare_urdf, body_groups, terrain_triangles, combine_meshes, geom_mesh, load_robot


def mesh_prim(stage, path, vertices, faces, collision=True):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(np.asarray(vertices, dtype=np.float32).tolist())
    mesh.CreateFaceVertexCountsAttr([3]*len(faces))
    mesh.CreateFaceVertexIndicesAttr(np.asarray(faces, dtype=np.int32).ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([(0.2, 0.5, 0.7)])
    if collision:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')
    return mesh.GetPrim()


class G1ReferenceEnv:
    def __init__(self, info, config, num_envs, *, terrain=True):
        import mujoco
        import trimesh
        self.cfg, self.info, self.num_envs = config, info, num_envs
        self.device = 'cuda:0';self.dt = config['physics_dt'];self.decimation = config['decimation']
        self.joint_names, self.body_names = config['joint_names'], config['body_names']
        self.manifest = read_record(info['artifact']/'manifest.json')
        source = resolve_path(self.manifest['target_urdf'])
        digest = hashlib.sha256(source.read_bytes()+json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]
        self.cache = PROJECT/'outputs/缓存/m2'/digest
        self.cache.mkdir(parents=True, exist_ok=True)
        self.urdf = self.cache/'g1.urdf'
        self.removed_massless_links = prepare_urdf(source, self.urdf)
        self.reference = ReferenceMotion(info['motion'], self.joint_names, self.body_names)
        self.model = load_robot(self.urdf)
        self.mjdata = mujoco.MjData(self.model)
        self.groups = body_groups(self.urdf, config['body_links'])
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=self.dt, device=self.device, render_interval=self.decimation, enable_scene_query_support=True,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=config['friction'],
                dynamic_friction=config['friction'], restitution=0.0)))
        self.stage = self.sim.stage
        self.origins = torch.tensor([[16*(i%8), 16*(i//8), 0] for i in range(num_envs)],
                                    device=self.device, dtype=torch.float32)
        for i, origin in enumerate(self.origins.cpu().numpy()):
            x = UsdGeom.Xform.Define(self.stage, f'/World/envs/env_{i}')
            x.AddTranslateOp().Set(Gf.Vec3d(*map(float, origin)))
        floor = trimesh.creation.box(extents=(1000, 1000, 0.1))
        floor.apply_translation((0, 0, -0.05))
        mesh_prim(self.stage, '/World/Ground', floor.vertices, floor.faces)
        parts = terrain_triangles(info['scene']) if terrain else []
        if parts:
            v, f = combine_meshes(parts)
            for i in range(num_envs): mesh_prim(self.stage, f'/World/envs/env_{i}/Terrain', v, f)
        vertices, faces = combine_meshes([(floor.vertices, floor.faces), *parts])
        self.scan_mesh = convert_to_warp_mesh(vertices, faces, self.device)
        # Physical limits/gains are explicit and independent of URDF importer defaults.
        by_name = lambda values: dict(zip(self.joint_names, values))
        cfg = ArticulationCfg(
            prim_path='/World/envs/env_.*/Robot',
            spawn=sim_utils.UrdfFileCfg(
                asset_path=str(self.urdf), usd_dir=str(self.cache/'usd'), usd_file_name='g1.usd',
                fix_base=False, merge_fixed_joints=False, self_collision=config['self_collisions'], make_instanceable=False,
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    target_type='none', gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)),
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=config['self_collisions'], solver_position_iteration_count=8, solver_velocity_iteration_count=4)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0.8),
                joint_pos=by_name(config['default_joint_positions']), joint_vel={'.*': 0.0}),
            actuators={'pd': IdealPDActuatorCfg(joint_names_expr=['.*'],
                stiffness=by_name(config['kp']), damping=by_name(config['kd']),
                effort_limit=by_name(config['effort_limits']), effort_limit_sim=by_name(config['effort_limits']),
                velocity_limit_sim=by_name(config['velocity_limits']), armature=by_name(config['armature']))})
        self.robot = Articulation(cfg)
        # This installed URDF importer creates instanceable collision references
        # even when make_instanceable=False. Deinstance before editing colliders.
        for prim in list(self.stage.Traverse()):
            if prim.IsInstance() and '/Robot/' in str(prim.GetPath()): prim.SetInstanceable(False)
        self.collider_repairs = []
        for prim in self.stage.Traverse():
            if '/Robot/' not in str(prim.GetPath()) or '/collisions/' not in str(prim.GetPath()): continue
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
        # Author collision shapes directly below rigid links from the source
        # geometry. This avoids converter-specific OBJ wrapper/instance schemas.
        self.mjdata.qpos[:] = np.r_[0, 0, 0, 1, 0, 0, 0, config['default_joint_positions']]
        mujoco.mj_forward(self.model, self.mjdata)
        for g in range(self.model.ngeom):
            if not (self.model.geom_contype[g] or self.model.geom_conaffinity[g]): continue
            bid = self.model.geom_bodyid[g];name = self.model.body(bid).name
            v, f = geom_mesh(self.model, self.mjdata, g)
            local = (v-self.mjdata.xpos[bid])@self.mjdata.xmat[bid].reshape(3, 3)
            for i in range(num_envs):
                path = f'/World/envs/env_{i}/Robot/{name}/M2Colliders/geom_{g}'
                if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
                    shape = UsdGeom.Sphere.Define(self.stage, path)
                    shape.CreateRadiusAttr(float(self.model.geom_size[g, 0]))
                    center = (self.mjdata.geom_xpos[g]-self.mjdata.xpos[bid])@self.mjdata.xmat[bid].reshape(3, 3)
                    shape.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
                    UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
                else:
                    prim = mesh_prim(self.stage, path, local, f)
                    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
                self.collider_repairs.append(path)
        for i in range(num_envs):
            sim_utils.modify_collision_properties(f'/World/envs/env_{i}/Robot',
                sim_utils.CollisionPropertiesCfg(contact_offset=config['contact_offset'], rest_offset=config['rest_offset']))
        self.sensor = ContactSensor(ContactSensorCfg(
            prim_path='/World/envs/env_.*/Robot/.*', history_length=4, update_period=0.0,
            max_contact_data_count_per_prim=16))
        self.probe = RigidObject(RigidObjectCfg(
            prim_path='/World/envs/env_.*/Probe',
            spawn=sim_utils.CuboidCfg(size=(0.04, 0.04, 0.04),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, -10))))
        self.sim.reset()
        self.robot.update(0.0);self.sensor.update(0.0, force_recompute=True);self.probe.update(0.0)
        self.joint_ids = torch.tensor(name_indices(self.robot.joint_names, self.joint_names), device=self.device)
        self.body_ids = torch.tensor([self.robot.body_names.index(n) for n in config['body_links']], device=self.device)
        self.group_matrix = torch.zeros((len(self.sensor.body_names), 9), device=self.device)
        for i, name in enumerate(self.sensor.body_names):
            if name in self.groups: self.group_matrix[i, self.groups[name]] = 1
        if not (self.group_matrix.sum(0) > 0).all(): raise ValueError('Missing physical contact body groups')
        self.default = torch.tensor(config['default_joint_positions'], device=self.device)
        self.previous_action = torch.zeros((num_envs, 23), device=self.device)
        self.history = torch.zeros((num_envs, 5, 75), device=self.device)
        self.force_history = torch.zeros((num_envs, 4, 9, 3), device=self.device)
        self.offsets = torch.tensor(config['landmark_offsets'], device=self.device, dtype=torch.float32)
        # Collision vertices in link frames, for a geometric floor penetration audit.
        self.clouds = []
        self.mjdata.qpos[:] = np.r_[0, 0, 0, 1, 0, 0, 0, config['default_joint_positions']]
        mujoco.mj_forward(self.model, self.mjdata)
        for g in range(self.model.ngeom):
            if not (self.model.geom_contype[g] or self.model.geom_conaffinity[g]): continue
            bid = self.model.geom_bodyid[g];name = self.model.body(bid).name
            if name not in self.robot.body_names: continue
            v, _ = geom_mesh(self.model, self.mjdata, g)
            v = (v-self.mjdata.xpos[bid])@self.mjdata.xmat[bid].reshape(3, 3)
            self.clouds.append((self.robot.body_names.index(name), torch.tensor(v, device=self.device, dtype=torch.float32)))
        self.park_probe()

    def park_probe(self):
        state = self.probe.data.default_root_state.clone()
        state[:, :3] = self.origins+torch.tensor([0, 0, -10], device=self.device)
        self.probe.write_root_pose_to_sim(state[:, :7]);self.probe.write_root_velocity_to_sim(state[:, 7:])

    def reset(self, reference_times=None, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else torch.as_tensor(env_ids, device=self.device)
        n = len(ids)
        root = torch.zeros((n, 13), device=self.device);root[:, 3] = 1
        if reference_times is None:
            root[:, :2] = -5.0  # Clear flat patch outside the reference obstacles.
            root[:, 2] = 0.80
            q = self.default.expand(n, -1);qd = torch.zeros_like(q)
        else:
            values = self.reference.sample(reference_times)
            if len(values['joint_positions']) != n: raise ValueError('Reset times and environment count differ')
            root = torch.tensor(np.c_[values['base_position_w'], values['base_quaternion_wxyz'],
                    values['base_linear_velocity_w'], values['base_angular_velocity_w']], device=self.device, dtype=torch.float32)
            q = torch.tensor(values['joint_positions'], device=self.device, dtype=torch.float32)
            qd = torch.tensor(values['joint_velocities'], device=self.device, dtype=torch.float32)
        root[:, :3] += self.origins[ids]
        qp, qvp = torch.zeros((n, 23), device=self.device), torch.zeros((n, 23), device=self.device)
        qp[:, self.joint_ids] = q;qvp[:, self.joint_ids] = qd
        self.robot.write_root_link_pose_to_sim(root[:, :7], env_ids=ids)
        self.robot.write_root_link_velocity_to_sim(root[:, 7:], env_ids=ids)
        self.robot.write_joint_state_to_sim(qp, qvp, env_ids=ids)
        self.robot.reset(ids);self.sensor.reset(ids)
        self.robot.set_joint_position_target(qp, env_ids=ids)
        self.previous_action[ids] = (q-self.default)/self.cfg['action_scale']
        self.history[ids] = self.proprio()[ids, None, :]
        self.force_history[ids] = 0
        return root, q, qd

    def contact_forces(self):
        return torch.einsum('nbc,bk->nkc', self.sensor.data.net_forces_w, self.group_matrix)

    def body_positions(self):
        return self.robot.data.body_link_pos_w[:, self.body_ids]+quat_apply(
            self.robot.data.body_link_quat_w[:, self.body_ids], self.offsets.expand(self.num_envs, -1, -1))

    def proprio(self):
        d = self.robot.data
        return torch.cat((d.root_link_ang_vel_b, d.projected_gravity_b,
            d.joint_pos[:, self.joint_ids]-self.default, d.joint_vel[:, self.joint_ids], self.previous_action), dim=-1)

    def step(self, action):
        if action.shape != (self.num_envs, 23) or not torch.isfinite(action).all(): raise ValueError('Invalid action')
        action = action.clamp(-100, 100)
        targets = torch.zeros_like(self.robot.data.joint_pos)
        targets[:, self.joint_ids] = self.default+self.cfg['action_scale']*action
        self.robot.set_joint_position_target(targets)
        for i in range(self.decimation):
            self.robot.write_data_to_sim();self.sim.step(render=False)
            self.robot.update(self.dt);self.sensor.update(self.dt, force_recompute=True);self.probe.update(self.dt)
            self.force_history[:, i] = self.contact_forces()
        self.previous_action[:] = action
        self.history = torch.cat((self.history[:, 1:], self.proprio()[:, None, :]), dim=1)
        return self.proprio()

    def height_scan(self, planner=False):
        nx, ny = (17, 11) if planner else (9, 7)
        x, y = torch.meshgrid(torch.linspace(-(nx-1)*.05, (nx-1)*.05, nx, device=self.device),
                              torch.linspace(-(ny-1)*.05, (ny-1)*.05, ny, device=self.device), indexing='ij')
        offsets = torch.stack((x.ravel(), y.ravel(), torch.zeros(nx*ny, device=self.device)), -1)
        q = self.robot.data.root_link_quat_w
        yaw = torch.atan2(2*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]), 1-2*(q[:, 2]**2+q[:, 3]**2))
        starts = offsets.expand(self.num_envs, -1, -1).clone()
        starts[:, :, 0] = torch.cos(yaw)[:, None]*x.ravel()-torch.sin(yaw)[:, None]*y.ravel()
        starts[:, :, 1] = torch.sin(yaw)[:, None]*x.ravel()+torch.cos(yaw)[:, None]*y.ravel()
        root = self.robot.data.root_link_pos_w-self.origins
        starts += root[:, None];starts[:, :, 2] += 20
        directions = torch.zeros_like(starts);directions[:, :, 2] = -1
        hits = raycast_mesh(starts.contiguous(), directions, self.scan_mesh, max_dist=100)[0]
        valid = torch.isfinite(hits).all(-1)
        # Coordinates are in each tile's reference world; heights are root-relative.
        return hits, hits[:, :, 2]-root[:, None, 2], valid

    def minimum_geometry_z(self):
        minimum = torch.full((self.num_envs,), float('inf'), device=self.device)
        for body, vertices in self.clouds:
            q = self.robot.data.body_link_quat_w[:, body]
            # z component of R(q)*vertex without materializing full rotated clouds.
            w, x, y, z = q.unbind(-1)
            row = torch.stack((2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1)
            values = (row@vertices.T).amin(-1)+self.robot.data.body_link_pos_w[:, body, 2]
            minimum = torch.minimum(minimum, values)
        return minimum

    def close(self):
        self.sim.clear_all_callbacks()
        sim_utils.SimulationContext.clear_instance()
