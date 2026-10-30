"""Table S2 sampling, explicit PhysX writes and observation/reset disturbances.

No simulator imports: tensor math can be tested in the training Python alone.
Startup dynamics stay fixed across episodes. Sensor drift is episode scoped.
"""
import torch
from acp.training.task import yaw, wrap


def uniform(shape, bounds, device):
    return bounds[0]+torch.rand(shape,device=device)*(bounds[1]-bounds[0])


def summary(value):
    value = value.detach().float()
    return dict(min=float(value.min()),max=float(value.max()),mean=float(value.mean()))


class Randomization:
    def __init__(self, env, config):
        self.env,self.cfg = env,config['randomization']
        self.dynamics = config['dynamics_randomization']
        self.noise = config['observation_noise']
        self.n,self.device = env.num_envs,env.device
        self.samples = {}
        self.expected_physics = {}
        self.report = dict(dynamics_enabled=self.dynamics,observation_noise_enabled=self.noise,
            dynamics_scope='sample once at simulation startup, retain across resets',
            odometry_scope='delay and drift sampled at each episode reset',
            action_delay_discretization='nearest 200Hz physics tick (0,5,10,15,20ms)',
            torso_body=self.cfg['torso_body'],leg_bodies=self.cfg['leg_bodies'])
        self.delay_ticks = torch.zeros(self.n,device=self.device,dtype=torch.long)
        self.zero_bias = torch.zeros((self.n,23),device=self.device)
        if self.dynamics:self.apply_dynamics()
        self.odometry_delay = torch.zeros(self.n,device=self.device)
        self.odometry_drift = torch.zeros((self.n,3),device=self.device)
        # 100ms at 50Hz plus the current sample; interpolation supports 50-100ms.
        self.pose_history = torch.zeros((self.n,7,7),device=self.device)
        self.pose_history[:,:,3] = 1
        self.reset_count = 0

    def sample(self, name, shape, bounds, device=None):
        value = uniform(shape,bounds,device or self.device)
        self.samples[name] = value.detach().cpu().clone()
        return value

    def apply_dynamics(self):
        env = self.env;robot = env.robot;view = robot.root_physx_view
        ids = torch.arange(self.n,dtype=torch.int32,device='cpu')
        cfg = self.cfg['dynamics'];names = robot.body_names
        torso = names.index(self.cfg['torso_body'])
        legs = [names.index(name) for name in self.cfg['leg_bodies']]
        masses = view.get_masses().clone();nominal = masses.clone()
        masses[:,torso] += self.sample('torso_added_mass',(self.n,),cfg['torso_added_mass'],'cpu')
        masses[:,legs] *= self.sample('leg_mass_scale',(self.n,len(legs)),cfg['leg_mass_scale'],'cpu')
        if (masses<=0).any():raise ValueError('Randomized body mass must remain positive')
        view.set_masses(masses,ids)
        # Same shape/inertia scaling convention as IsaacLab's mass event.
        inertia = view.get_inertias().clone()*(masses/nominal)[...,None]
        view.set_inertias(inertia,ids)
        coms = view.get_coms().clone()
        com_delta = self.sample('torso_com_delta',(self.n,3),[-1,1],'cpu')*torch.tensor(cfg['torso_com_half_range'])
        self.samples['torso_com_delta'] = com_delta.clone()
        coms[:,torso,:3] += com_delta;view.set_coms(coms,ids)
        # One material per environment, shared by its shapes: <=N+terrain
        # unique materials, rather than N*shapes (PhysX has a 64k limit).
        material = view.get_material_properties().clone()
        static = self.sample('static_friction',(self.n,),cfg['static_friction'],'cpu')
        dynamic = self.sample('dynamic_friction',(self.n,),cfg['dynamic_friction'],'cpu')
        restitution = self.sample('restitution',(self.n,),cfg['restitution'],'cpu')
        material[:] = torch.stack((static,dynamic,restitution),-1)[:,None]
        view.set_material_properties(material,ids)
        armature = torch.tensor(env.cfg['armature'],device=self.device).expand(self.n,-1).clone()
        armature *= self.sample('armature_scale',armature.shape,cfg['armature_scale'])
        robot.write_joint_armature_to_sim(armature,joint_ids=env.joint_ids)
        # IdealPDActuator is explicit. Change its GPU gains, NOT implicit
        # PhysX drive stiffness (the latter must stay zero).
        actuator = robot.actuators['pd']
        actuator_ids = actuator.joint_indices
        if actuator_ids != slice(None):
            if list(actuator_ids)!=list(range(len(robot.joint_names))):
                raise ValueError('Expected one explicit PD actuator covering every joint')
        kp = actuator.stiffness.clone();kd = actuator.damping.clone()
        kp *= self.sample('kp_scale',kp.shape,cfg['pd_gain_scale'])
        kd *= self.sample('kd_scale',kd.shape,cfg['pd_gain_scale'])
        actuator.stiffness.copy_(kp);actuator.damping.copy_(kd)
        ankle = torch.tensor(['ankle' in name for name in env.joint_names],device=self.device)
        bias = self.sample('joint_zero_bias',(self.n,23),[-1,1])
        bias *= torch.where(ankle,cfg['ankle_zero_bias'],cfg['joint_zero_bias'])
        self.zero_bias.copy_(bias);self.samples['joint_zero_bias'] = bias.cpu().clone()
        delay = self.sample('action_delay_seconds',(self.n,),cfg['action_delay_seconds'])
        self.delay_ticks.copy_((delay/env.dt).round().long())
        env.action_delay_ticks = self.delay_ticks
        env.joint_zero_bias = self.zero_bias
        checks = {
            'mass':(view.get_masses(),masses), 'inertia':(view.get_inertias(),inertia),
            'com':(view.get_coms(),coms),'material':(view.get_material_properties(),material),
            'armature':(view.get_dof_armatures()[:,env.joint_ids.cpu()],armature.cpu()),
            'explicit_kp':(actuator.stiffness,kp),'explicit_kd':(actuator.damping,kd)}
        for name,(actual,expected) in checks.items():
            torch.testing.assert_close(actual.cpu(),expected.cpu(),rtol=1e-5,atol=1e-6,msg='Randomization readback: '+name)
            self.expected_physics[name] = expected.detach().cpu().clone()
        self.report['readback_checks'] = {name:True for name in checks}
        self.report['startup_samples'] = {name:summary(value) for name,value in self.samples.items()}
        self.samples['physical_mass'] = masses.cpu();self.samples['physical_com'] = coms.cpu()
        self.samples['physical_kp'] = kp.cpu();self.samples['physical_kd'] = kd.cpu()
        self.samples['action_delay_ticks'] = self.delay_ticks.cpu()
        self.report['body_names'] = names
        self.report['actuator_joint_names'] = robot.joint_names

    def verify_dynamics(self):
        """Re-read after resets/rollouts: startup parameters must not drift."""
        if not self.dynamics:return {'enabled':False}
        env=self.env;view=env.robot.root_physx_view;actuator=env.robot.actuators['pd']
        actual = dict(mass=view.get_masses(),inertia=view.get_inertias(),com=view.get_coms(),
            material=view.get_material_properties(),armature=view.get_dof_armatures()[:,env.joint_ids.cpu()],
            explicit_kp=actuator.stiffness,explicit_kd=actuator.damping)
        for name,value in actual.items():
            torch.testing.assert_close(value.cpu(),self.expected_physics[name],rtol=1e-5,atol=1e-6,
                msg='Startup dynamics changed after reset: '+name)
        return dict(enabled=True,reset_count=self.reset_count,checks={name:True for name in actual})

    def reset(self, ids, perturb=True):
        env = self.env;d = env.robot.data
        if self.dynamics and perturb:
            cfg = self.cfg['reset']
            nominal_q = d.joint_pos[ids].clone()
            q = d.joint_pos[ids].clone()+uniform((len(ids),23),cfg['joint_position'],self.device)
            limits = d.joint_pos_limits[ids]
            q = q.clamp(limits[...,0],limits[...,1])
            pose = torch.cat((d.root_link_pos_w[ids],d.root_link_quat_w[ids]),-1).clone()
            height_delta = uniform((len(ids),),cfg['base_height'],self.device)
            pose[:,2] += height_delta
            env.robot.write_root_link_pose_to_sim(pose,env_ids=ids)
            env.robot.write_joint_state_to_sim(q,d.joint_vel[ids].clone(),env_ids=ids)
            env.sim.forward();env.robot.update(0.)
            if self.reset_count==0:
                self.samples['initial_reset_joint_delta'] = (q-nominal_q).cpu()
                self.samples['initial_reset_height_delta'] = height_delta.cpu()
                self.report['initial_reset'] = dict(joint_delta=summary(q-nominal_q),height_delta=summary(height_delta))
        cfg = self.cfg['observation']
        if self.noise:
            self.odometry_delay[ids] = uniform((len(ids),),cfg['odometry_delay_seconds'],self.device)
            angle = 2*torch.pi*torch.rand(len(ids),device=self.device)
            radius = cfg['odometry_xy_radius']*torch.rand(len(ids),device=self.device).sqrt()
            self.odometry_drift[ids,0] = radius*angle.cos()
            self.odometry_drift[ids,1] = radius*angle.sin()
            self.odometry_drift[ids,2] = uniform((len(ids),),cfg['odometry_z'],self.device)
            if self.reset_count==0:
                self.samples['initial_odometry_delay_seconds'] = self.odometry_delay.cpu().clone()
                self.samples['initial_odometry_drift'] = self.odometry_drift.cpu().clone()
                self.report['initial_odometry'] = dict(delay=summary(self.odometry_delay),drift=summary(self.odometry_drift))
        pose = torch.cat((d.root_link_pos_w-env.origins,d.root_link_quat_w),-1)
        self.pose_history[ids] = pose[ids,None]
        self.reset_count += len(ids)

    def advance(self):
        env = self.env;d = env.robot.data
        pose = torch.cat((d.root_link_pos_w-env.origins,d.root_link_quat_w),-1)
        self.pose_history = torch.cat((pose[:,None],self.pose_history[:,:-1]),1)

    def observed_pose(self, clean=False):
        if clean or not self.noise:return self.pose_history[:,0,:3],self.pose_history[:,0,3:]
        age = self.odometry_delay/(self.env.dt*self.env.decimation)
        lo = age.floor().long();alpha = (age-lo)[:,None]
        ids = torch.arange(self.n,device=self.device)
        a,b = self.pose_history[ids,lo],self.pose_history[ids,lo+1]
        position = a[:,:3]+alpha*(b[:,:3]-a[:,:3])+self.odometry_drift
        # Only yaw is consumed by scan/goal/contact transforms; interpolate on
        # the circle across +/-pi rather than linearly blending quaternion signs.
        heading = yaw(a[:,3:])+alpha[:,0]*wrap(yaw(b[:,3:])-yaw(a[:,3:]))
        q = torch.zeros((self.n,4),device=self.device)
        q[:,0] = (heading/2).cos();q[:,3] = (heading/2).sin()
        return position,q

    def proprio(self, value, clean=False):
        if clean or not self.noise:return value
        cfg = self.cfg['observation']
        scale = value.new_tensor([cfg['angular_velocity']]*3+[cfg['gravity']]*3+
            [cfg['joint_position']]*23+[cfg['joint_velocity']]*23+[0.]*23)
        return value+(2*torch.rand_like(value)-1)*scale
