"""Reference-anchored Stage-1 task; import after AppLauncher only."""
import numpy as np
import torch
from acp.envs.g1 import G1ReferenceEnv
from acp.training.task import TensorReference,ReferenceBank,yaw,wrap,yaw_local,inverse_rotate,update_csi,rewards
from acp.envs.reference import ReferenceMotion
from acp.envs.randomization import Randomization


class Stage1Task:
    def __init__(self, info, robot_cfg, config, task_name, num_envs):
        self.cfg,self.task_cfg,self.name = config,config['tasks'][task_name],task_name
        self.member_names = self.task_cfg.get('members',[task_name])
        infos = info if isinstance(info,list) else [info]
        self.infos = infos
        self.assignment = torch.arange(num_envs,device='cuda:0')*len(infos)//num_envs
        physical_cfg = dict(robot_cfg)
        if config['dynamics_randomization']:physical_cfg['friction'] = config['randomization']['dynamics']['ground_friction']
        self.env = G1ReferenceEnv(infos[0],physical_cfg,num_envs,
            scenes=[infos[i]['scene'] for i in self.assignment.cpu().tolist()])
        self.device,self.n = self.env.device,num_envs
        self.dt = self.env.dt*self.env.decimation
        references = []
        for name,source in zip(self.member_names,infos):
            spec = config['tasks'][name]
            motion = ReferenceMotion(source['motion'],self.env.joint_names,self.env.body_names)
            references.append(TensorReference(motion,spec['start_seconds'],spec['end_seconds'],self.dt,self.device))
        self.reference = ReferenceBank(references,self.assignment)
        self.duration = self.reference.end-self.reference.start
        self.phase = torch.full((num_envs,),config['curriculum']['initial_phase'],device=self.device)
        self.time = torch.zeros(num_envs,device=self.device)
        self.start_phase = torch.zeros_like(self.time)
        self.previous_action = torch.zeros((num_envs,23),device=self.device)
        self.previous_previous_action = torch.zeros_like(self.previous_action)
        self.previous_body = torch.zeros((num_envs,9,3),device=self.device)
        self.previous_body_velocity = torch.zeros_like(self.previous_body)
        self.age = torch.zeros(num_envs,device=self.device,dtype=torch.long)
        self.actor_history = torch.zeros((num_envs,5,75),device=self.device)
        self.evaluating = False
        self.randomization = Randomization(self.env,config)
        self.reset()

    def noisy_proprio(self):
        return self.randomization.proprio(self.env.proprio(),clean=self.evaluating)

    def reset(self, ids=None, phase=None):
        ids = torch.arange(self.n,device=self.device) if ids is None else ids
        fractions = self.phase[ids] if phase is None else torch.full((len(ids),),phase,device=self.device)
        times = self.reference.start[ids]+fractions*self.duration[ids]
        values = self.reference.sample(times,env_ids=ids)
        cpu_values = {k:v.cpu().numpy() for k,v in values.items() if k in
            ('base_position_w','base_quaternion_wxyz','base_linear_velocity_w','base_angular_velocity_w','joint_positions','joint_velocities')}
        self.env.reset(times.cpu().numpy(),env_ids=ids,reference_values=cpu_values)
        self.env.sim.forward();self.env.robot.update(0.)
        fk_error = float((self.env.body_positions()[ids]-self.env.origins[ids,None]-values['body_landmarks_w']).norm(dim=-1).max())
        if fk_error>.002:raise ValueError(f'Reference and physical robot FK differ by {fk_error}m')
        # Validate the exact reference reset before deliberately perturbing it.
        self.randomization.reset(ids,perturb=not self.evaluating)
        self.time[ids] = times;self.start_phase[ids] = fractions;self.age[ids] = 0
        self.previous_action[ids] = self.env.previous_action[ids]
        self.previous_previous_action[ids] = self.previous_action[ids]
        self.previous_body[ids] = self.env.body_positions()[ids]-self.env.origins[ids,None]
        self.previous_body_velocity[ids] = 0
        self.actor_history[ids] = self.noisy_proprio()[ids,None]

    def observations(self):
        env = self.env;data = env.robot.data
        root = data.root_link_pos_w-env.origins;quat = data.root_link_quat_w
        observed_root,observed_quat = self.randomization.observed_pose(clean=self.evaluating)
        reference = self.reference.sample(self.time)
        targets,occupied = self.reference.contacts(self.time,observed_root,observed_quat)
        lookahead = torch.full_like(self.time,1.) if self.evaluating else .5+torch.rand_like(self.time)
        goal_ref = self.reference.sample(self.time+lookahead)
        direction = yaw_local(goal_ref['base_position_w'][:,None],observed_root,observed_quat)[:,0,:2]
        direction /= direction.norm(dim=-1,keepdim=True).clamp(min=.3)
        goal = torch.cat((direction,wrap(yaw(goal_ref['base_quaternion_wxyz'])-yaw(observed_quat))[:,None]),-1)
        _,heights,tracker_valid = env.height_scan(root=observed_root,quaternion=observed_quat)
        _,planner_height,valid = env.height_scan(planner=True)
        if not valid.all() or not tracker_valid.all(): raise ValueError('Reference task height scan missed terrain')
        if self.cfg['observation_noise'] and not self.evaluating:
            noise = self.cfg['randomization']['observation']
            heights = heights+(2*torch.rand_like(heights)-1)*noise['tracker_height']
            targets = torch.where(occupied[...,None],targets+(2*torch.rand_like(targets)-1)*noise['contact_position'],0)
            angle_noise = (2*torch.rand_like(self.time)-1)*noise['goal_angle']
            x,y = goal[:,0].clone(),goal[:,1].clone()
            goal[:,0] = x*angle_noise.cos()-y*angle_noise.sin();goal[:,1] = x*angle_noise.sin()+y*angle_noise.cos()
            goal[:,2] = wrap(goal[:,2]+(2*torch.rand_like(self.time)-1)*noise['goal_angle'])
        actor = torch.cat((self.actor_history.flatten(1),heights,targets.flatten(1),goal),-1)
        reference_kinematic = torch.cat((inverse_rotate(reference['base_quaternion_wxyz'],reference['base_linear_velocity_w']),
            inverse_rotate(reference['base_quaternion_wxyz'],reference['base_angular_velocity_w']),
            reference['joint_positions'],reference['joint_velocities']),-1)
        # Privileged reference body positions must retain the tracking offset
        # relative to the actual robot. Reward frames are handled separately.
        reference_contacts = yaw_local(reference['body_landmarks_w'],root,quat)
        critic = torch.cat((env.proprio(),planner_height,data.root_link_lin_vel_b,reference_kinematic,
            reference_contacts.flatten(1),reference['contact_flags'].float()),-1)
        if actor.shape!=(self.n,468) or critic.shape!=(self.n,353): raise ValueError('Stage-1 observation contract mismatch')
        if not torch.isfinite(actor).all() or not torch.isfinite(critic).all(): raise ValueError('Nonfinite task observation')
        return actor,critic

    def step(self, action):
        env = self.env;env.step(action);self.age += 1
        self.randomization.advance()
        self.time = self.reference.start+self.start_phase*self.duration+self.age*self.dt
        self.actor_history = torch.cat((self.actor_history[:,1:],self.noisy_proprio()[:,None]),1)
        d = env.robot.data;j = env.joint_ids
        body = env.body_positions()-env.origins[:,None]
        body_velocity = (body-self.previous_body)/self.dt
        body_acc = (body_velocity-self.previous_body_velocity)/self.dt
        body_acc[self.age<3] = 0
        state = dict(root_position=d.root_link_pos_w-env.origins,root_quaternion=d.root_link_quat_w,
            linear_velocity=d.root_link_lin_vel_w,angular_velocity=d.root_link_ang_vel_w,
            joint_positions=d.joint_pos[:,j],joint_velocities=d.joint_vel[:,j],body_positions=body,
            measured_contacts=(env.force_history[...,2].abs().amax(1)>1).float(),
            joint_limits=d.soft_joint_pos_limits[:,j],velocity_limits=d.joint_vel_limits[:,j],
            applied_torque=d.applied_torque[:,j],computed_torque=d.computed_torque[:,j],action=action.clamp(-100,100))
        reference = self.reference.sample(self.time)
        terms,metrics = rewards(state,reference,self.previous_action,self.previous_previous_action,body_acc)
        reward = self.dt*sum(self.cfg['reward_weights'][key]*value for key,value in terms.items())
        self.previous_previous_action.copy_(self.previous_action);self.previous_action.copy_(state['action'])
        self.previous_body.copy_(body);self.previous_body_velocity.copy_(body_velocity)
        terminated = (metrics['root_error']>self.cfg['termination']['root_position_error_m']) | (
            metrics['orientation_error']>self.cfg['termination']['orientation_error_rad'])
        reached_end = self.time>=self.reference.start+self.duration*self.cfg['curriculum']['completion_phase']
        success = reached_end & ~terminated
        done = terminated | reached_end
        if not torch.isfinite(reward).all(): raise ValueError('Nonfinite Stage-1 reward')
        return reward,terminated,done,success,terms,metrics

    def finish(self, done, success):
        ids = done.nonzero(as_tuple=True)[0]
        self.phase[ids] = update_csi(self.phase[ids],success[ids],self.cfg['curriculum']['phase_adjustment'])
        if len(ids): self.reset(ids)

    def close(self): self.env.close()
