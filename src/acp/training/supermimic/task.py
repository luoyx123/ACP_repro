"""ACP observations and analytic rewards shared by real and imagined transitions."""
import torch

from acp.envs.reference import ReferenceMotion
from acp.training.task import TensorReference, inverse_rotate, rewards, wrap, yaw, yaw_local
from .model import Kinematics, Terrain
from .rotations import mat_to_quat, mv, quat_to_mat, rot_angle_sq


class MDP:
    def __init__(self, model, scene, motion, specification, cfg, robot_cfg, device):
        self.cfg, self.robot_cfg, self.device = cfg, robot_cfg, device
        self.dt = robot_cfg['physics_dt']*robot_cfg['decimation']
        reference = ReferenceMotion(motion, robot_cfg['joint_names'], robot_cfg['body_names'])
        self.reference = TensorReference(reference, specification['start_seconds'], specification['end_seconds'], self.dt,
                                         device, cfg.get('oracle_contact_position_source', 'body_landmark'))
        self.kin = Kinematics(model, robot_cfg, device)
        self.terrain = Terrain(scene, device)
        self.default = torch.tensor(robot_cfg['default_joint_positions'], device=device)
        self.kp = torch.tensor(robot_cfg['kp'], device=device)
        self.kd = torch.tensor(robot_cfg['kd'], device=device)
        self.effort = torch.tensor(robot_cfg['effort_limits'], device=device)
        ids = [model.joint(name).id for name in robot_cfg['joint_names']]
        # IsaacLab uses a 0.9 soft-position-limit factor by default.
        ranges = torch.tensor(model.jnt_range[ids], dtype=torch.float32, device=device)
        center, half = ranges.mean(-1), (ranges[:, 1]-ranges[:, 0])*.45
        self.bounds = torch.stack((center-half, center+half), -1)
        self.velocity_limits = torch.tensor(robot_cfg['velocity_limits'], device=device)
        noise = cfg['randomization']['observation']
        self.proprio_noise = self.default.new_tensor([noise['angular_velocity']]*3+[noise['gravity']]*3+
                                                    [noise['joint_position']]*23+[noise['joint_velocity']]*23+[0.]*23)

    def proprio(self, S, previous_action, noise=False):
        _, R, _, w, q, qd = S
        value = torch.cat((mv(R.transpose(-1, -2), w), -R[:, 2], q-self.default, qd, previous_action), -1)
        return value+(2*torch.rand_like(value)-1)*self.proprio_noise if noise else value

    def observations(self, S, history, previous_action, times, pose_history=None, delay=None, drift=None, noise=False):
        p, R, v, _, _, _ = S
        quat = mat_to_quat(R)
        op, oq = p, quat
        if noise and pose_history is not None:
            age = delay/self.dt
            lo = age.floor().long()
            alpha = (age-lo)[:, None]
            ids = torch.arange(len(p), device=p.device)
            a, b = pose_history[ids, lo], pose_history[ids, lo+1]
            op = a[:, :3]+alpha*(b[:, :3]-a[:, :3])+drift
            heading = yaw(a[:, 3:])+alpha[:, 0]*wrap(yaw(b[:, 3:])-yaw(a[:, 3:]))
            oq = torch.stack(((heading/2).cos(), torch.zeros_like(heading), torch.zeros_like(heading), (heading/2).sin()), -1)
        reference = self.reference.sample(times)
        targets, occupied = self.reference.contacts(times, op, oq)
        lookahead = .5+torch.rand_like(times) if noise else torch.ones_like(times)
        goal_ref = self.reference.sample(times+lookahead)
        direction = yaw_local(goal_ref['base_position_w'][:, None], op, oq)[:, 0, :2]
        direction = direction/direction.norm(dim=-1, keepdim=True).clamp(min=.3)
        goal = torch.cat((direction, wrap(yaw(goal_ref['base_quaternion_wxyz'])-yaw(oq))[:, None]), -1)
        heights = self.terrain.scan((op, quat_to_mat(oq)), 63)
        if noise:
            params = self.cfg['randomization']['observation']
            heights = heights+(2*torch.rand_like(heights)-1)*params['tracker_height']
            targets = torch.where(occupied[..., None], targets+(2*torch.rand_like(targets)-1)*params['contact_position'], 0.)
            da = (2*torch.rand_like(times)-1)*params['goal_angle']
            gx, gy = goal[:, 0], goal[:, 1]
            goal = torch.stack((gx*da.cos()-gy*da.sin(), gx*da.sin()+gy*da.cos(),
                                wrap(goal[:, 2]+(2*torch.rand_like(times)-1)*params['goal_angle'])), -1)
        actor = torch.cat((history.flatten(1), heights, targets.flatten(1), goal), -1)
        kinematic = torch.cat((inverse_rotate(reference['base_quaternion_wxyz'], reference['base_linear_velocity_w']),
                               inverse_rotate(reference['base_quaternion_wxyz'], reference['base_angular_velocity_w']),
                               reference['joint_positions'], reference['joint_velocities']), -1)
        critic = torch.cat((self.proprio(S, previous_action), self.terrain.scan(S), mv(R.transpose(-1, -2), v),
                            kinematic, yaw_local(reference['body_landmarks_w'], p, quat).flatten(1),
                            reference['contact_flags'].float()), -1)
        return actor, critic

    def transition(self, S0, S1, times, action, previous_action, previous_previous_action,
                   previous_body_velocity, contacts, torque, computed_torque, age):
        p, R, v, w, q, qd = S1
        body, body0 = self.kin(S1), self.kin(S0)
        body_velocity = (body-body0)/self.dt
        acceleration = (body_velocity-previous_body_velocity)/self.dt
        acceleration = torch.where((age<3)[:, None, None], 0., acceleration)
        reference = self.reference.sample(times)
        # atan2 rotation distance has finite gradients at coincident rotations.
        angle = torch.sqrt(rot_angle_sq(R, quat_to_mat(reference['base_quaternion_wxyz']))+1e-12)
        state = dict(root_position=p, root_quaternion=mat_to_quat(R), orientation_error=angle,
                     linear_velocity=v, angular_velocity=w, joint_positions=q, joint_velocities=qd,
                     body_positions=body, measured_contacts=contacts, joint_limits=self.bounds,
                     velocity_limits=self.velocity_limits, applied_torque=torque, computed_torque=computed_torque,
                     action=action)
        terms, metrics = rewards(state, reference, previous_action, previous_previous_action, acceleration)
        # sqrt(sum(square)) has an undefined backward at exact tracking; the
        # vector-norm primitive defines its zero gradient there.
        metrics['root_error'] = torch.linalg.vector_norm(p-reference['base_position_w'], dim=-1)
        reward = self.dt*sum(self.cfg['reward_weights'][k]*v for k, v in terms.items())
        root_margin = metrics['root_error']-self.cfg['termination']['root_position_error_m']
        angle_margin = angle-self.cfg['termination']['orientation_error_rad']
        terminated = (root_margin>0) | (angle_margin>0)
        end = self.reference.start+(self.reference.end-self.reference.start)*self.cfg['curriculum']['completion_phase']
        complete = times>=end
        # Reaching the known motion boundary truncates the imagined return too.
        failure = 1-(1-torch.sigmoid(root_margin/.02))*(1-torch.sigmoid(angle_margin/.04))
        continuation = (1-failure)*(~complete).float()
        return dict(reward=reward, terminated=terminated, done=terminated|complete, success=complete & ~terminated,
                    continuation=continuation, terms=terms, metrics=metrics, body_velocity=body_velocity)
