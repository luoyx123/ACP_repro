"""ACP SuperMimic: collection -> multi-step dynamics -> BPTT -> lambda critic.

Follows local InstinctMJ SuperMimicRunner (7ca851e); see third_party/instinctmj.
Only the explicit `train` command applies optimizer steps. No PPO is called.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch import nn

from acp.data.assets import PROJECT, atomic_json
from acp.training.ppo import ActorCritic
from acp.training.task import update_csi
from .assets import prepare, sha, verify
from .env import Simulator
from .model import WorldModel, pack, unpack
from .rotations import mat_to_quat
from .task import MDP
from .fullbody import contact_metrics


class Replay:
    """Time/env ring. Save terminal states before reset; mask windows across resets."""
    def __init__(self, capacity):
        self.capacity, self.size, self.ptr, self.data = capacity, 0, 0, {}

    def append(self, values):
        for key, value in values.items():
            if key not in self.data:
                self.data[key] = torch.empty((self.capacity, *value.shape), dtype=value.dtype, device=value.device)
            self.data[key][self.ptr].copy_(value)
        self.ptr = (self.ptr+1)%self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample(self, batch, window=1):
        if self.size<window:
            raise ValueError('Not enough replay steps')
        data = self.data['s']
        start = torch.randint(0, self.size-window+1, (batch,), device=data.device)
        env = torch.randint(0, data.shape[1], (batch,), device=data.device)
        rows = (self.ptr-self.size+start[:, None]+torch.arange(window, device=data.device))%self.capacity
        out = {k:v[rows, env[:, None]] for k, v in self.data.items()}
        out['mask'] = out['episode']==out['episode'][:, :1]
        out['env'] = env
        return out


class Collection:
    def __init__(self, root, clip, package, count, device, acknowledge):
        cfg, robot = package['config'], package['robot_config']
        self.sim = Simulator(root/clip['scene'], robot, cfg, count, device, acknowledge_engine_mapping=acknowledge)
        self.mdp = MDP(self.sim.model, root/clip['scene'], root/clip['motion'], clip['specification'], cfg, robot, device)
        self.cfg, self.n, self.device, self.name = cfg, count, device, clip['name']
        self.phase = torch.full((count,), cfg['curriculum']['initial_phase'], device=device)
        self.episode = torch.zeros(count, dtype=torch.long, device=device)
        self.age = torch.zeros_like(self.episode)
        self.times = torch.zeros(count, device=device)
        self.history = torch.zeros((count, 5, 75), device=device)
        self.poses = torch.zeros((count, 7, 7), device=device)
        self.delay = torch.zeros(count, device=device)
        self.drift = torch.zeros((count, 3), device=device)
        self.action_prev2 = torch.zeros((count, 23), device=device)
        self.body_velocity = torch.zeros((count, 9, 3), device=device)
        self.training_noise = cfg['observation_noise']
        self.evaluating = False
        self.initialization = package.get('supermimic_initialization', {})
        self.reset_rejections = 0
        self.reset()

    @torch.no_grad()
    def reset(self, ids=None, phase=None):
        ids = torch.arange(self.n, device=self.device) if ids is None else ids
        ref = self.mdp.reference
        self.times[ids] = ref.start+(self.phase[ids] if phase is None else phase)*(ref.end-ref.start)
        values = ref.sample(self.times[ids])
        self.sim.reset(values, ids)
        error = float((self.mdp.kin(self.sim.state())[ids]-values['body_landmarks_w']).norm(dim=-1).max())
        if error>.002:
            raise ValueError(f'{self.name}: reference reset FK error {error}m')
        if self.sim.randomized and not self.evaluating:
            initial_command = self.sim.previous_action[ids].clone()
            perturbed = {k:v.clone() for k, v in values.items()}
            reset_cfg = self.cfg['randomization']['reset']
            uniform = lambda shape, bounds:bounds[0]+torch.rand(shape, device=self.device)*(bounds[1]-bounds[0])
            bounds = self.sim.model.jnt_range[[self.sim.model.joint(n).id for n in self.sim.robot_cfg['joint_names']]]
            bounds = torch.tensor(bounds, device=self.device, dtype=torch.float32)
            perturbed['joint_positions'] = (perturbed['joint_positions']+uniform((len(ids), 23), reset_cfg['joint_position'])).clamp(bounds[:, 0], bounds[:, 1])
            perturbed['base_position_w'][:, 2] += uniform((len(ids),), reset_cfg['base_height'])
            self.sim.reset(perturbed, ids)
            self.sim.previous_action[ids] = initial_command
        # Admission happens before observation/replay insertion. It is not a
        # rollout termination, a policy failure, or an adaptive-sampling signal.
        gate = self.initialization
        if gate.get('enabled', False):
            tolerance = gate['maximum_penetration_m']
            pending = ids
            for attempt in range(gate['maximum_resamples'] + 1):
                env_depth, self_depth = self.sim.geometry_depths()
                finite = torch.isfinite(pack(self.sim.state())).all(-1)
                bad = pending[(torch.maximum(env_depth, self_depth)[pending] > tolerance) | ~finite[pending]]
                if not len(bad):
                    break
                self.reset_rejections += len(bad)
                if attempt == gate['maximum_resamples']:
                    raise ValueError(f'{self.name}: {len(bad)} invalid initializations after {attempt} resamples; '
                                     'repair reference/reset configuration; threshold is never relaxed automatically')
                values = ref.sample(self.times[bad])
                if self.sim.randomized and not self.evaluating:
                    values['joint_positions'] = (values['joint_positions'] + uniform((len(bad), 23), reset_cfg['joint_position'])).clamp(bounds[:, 0], bounds[:, 1])
                    values['base_position_w'][:, 2] += uniform((len(bad),), reset_cfg['base_height'])
                initial_command = self.sim.previous_action[bad].clone()
                self.sim.reset(values, bad)
                self.sim.previous_action[bad] = initial_command
                pending = bad
        S = self.sim.state()
        self.age[ids] = 0
        self.episode[ids] += 1
        self.action_prev2[ids] = self.sim.previous_action[ids]
        self.body_velocity[ids] = 0
        self.history[ids] = self.mdp.proprio(S, self.sim.previous_action, self.training_noise)[ids, None]
        self.poses[ids] = torch.cat((S[0], mat_to_quat(S[1])), -1)[ids, None]
        if self.training_noise:
            cfg = self.cfg['randomization']['observation']
            lo, hi = cfg['odometry_delay_seconds']
            self.delay[ids] = lo+torch.rand(len(ids), device=self.device)*(hi-lo)
            angle = torch.rand(len(ids), device=self.device)*2*torch.pi
            radius = cfg['odometry_xy_radius']*torch.rand(len(ids), device=self.device).sqrt()
            self.drift[ids, 0], self.drift[ids, 1] = radius*angle.cos(), radius*angle.sin()
            lo, hi = cfg['odometry_z']
            self.drift[ids, 2] = lo+torch.rand(len(ids), device=self.device)*(hi-lo)

    def observations(self):
        return self.mdp.observations(self.sim.state(), self.history, self.sim.previous_action, self.times,
                                     self.poses, self.delay, self.drift, self.training_noise)

    @torch.no_grad()
    def collect(self, policy, replay, action_noise):
        actor, critic = self.observations()
        policy.actor_norm.update(actor)
        policy.critic_norm.update(critic)
        a = (policy.actor(policy.actor_norm(actor))+action_noise*torch.randn((self.n, 23), device=self.device)).clamp(-100, 100)
        S0, previous = self.sim.state(), self.sim.previous_action.clone()
        values = dict(s=pack(S0), a=a, a_prev=previous, a_prev2=self.action_prev2.clone(),
                      history=self.history.clone(), poses=self.poses.clone(), delay=self.delay.clone(),
                      drift=self.drift.clone(), velocity=self.body_velocity.clone(), time=self.times.clone(),
                      age=self.age.clone(), episode=self.episode.clone(), context=self.sim.context.clone())
        S1 = self.sim.step(a)
        self.age += 1
        self.times += self.mdp.dt
        if not torch.isfinite(pack(S1)).all():
            raise FloatingPointError(f'{self.name}: non-finite state before replay insertion')
        contacts = self.sim.measured_contacts()
        result = self.mdp.transition(S0, S1, self.times, a, previous, self.action_prev2, self.body_velocity,
                                     contacts, self.sim.applied_torque, self.sim.computed_torque, self.age)
        values.update(s1=pack(S1), contacts=contacts, done=result['done'], reward=result['reward'])
        diagnostics = contact_metrics(self.sim, self.sim.robot_cfg['body_names'])
        diagnostics['reset/rejected_candidates_total'] = self.reset_rejections
        diagnostics['end/reference_complete_fraction'] = float(result['success'].float().mean())
        diagnostics['end/failure_fraction'] = float((result['done'] & ~result['success']).float().mean())
        for name, term in result['terms'].items():
            diagnostics['reward_terms/' + name] = float(term.mean())
        replay.append(values)
        self.action_prev2.copy_(previous)
        self.body_velocity.copy_(result['body_velocity'])
        self.history = torch.cat((self.history[:, 1:], self.mdp.proprio(S1, a, self.training_noise)[:, None]), 1)
        self.poses = torch.cat((torch.cat((S1[0], mat_to_quat(S1[1])), -1)[:, None], self.poses[:, :-1]), 1)
        done, success = result['done'], result['success']
        self.phase[done] = update_csi(self.phase[done], success[done], self.cfg['curriculum']['phase_adjustment'])
        ids = done.nonzero(as_tuple=True)[0]
        if len(ids):
            self.reset(ids)
        return {**diagnostics, **{k:float(v.float().mean()) for k, v in dict(reward=result['reward'], root_error=result['metrics']['root_error'],
                                                         done=done, success=success, phase=self.phase).items()}}


class Runner:
    def __init__(self, package, cfg, n=1024, device='cuda:0'):
        self.root, self.cfg, self.device = Path(package), cfg, device
        gate = cfg.get('initialization_validation', {})
        if gate.get('enabled') and (gate['maximum_penetration_m'] <= 0 or gate['maximum_resamples'] < 0):
            raise ValueError('Invalid initialization admission settings')
        counts = ['buffer_steps', 'warmup_steps', 'wm_window', 'wm_batches', 'wm_batch', 'horizon',
                  'pi_batch', 'critic_epochs', 'critic_minibatches', 'save_interval', 'lr_decay_iters']
        if any(cfg[k]<1 for k in counts) or cfg['utd']<=0 or cfg['lr']<=0:
            raise ValueError('Invalid SuperMimic counts or learning rate')
        if cfg['buffer_steps']<max(cfg['warmup_steps'], cfg['wm_window']):
            raise ValueError('Replay must hold warmup and world-model windows')
        if cfg['amp'] not in ('bf16', 'none'):
            raise ValueError('amp must be bf16 or none')
        self.package = verify(self.root)
        self.package = {**self.package, 'supermimic_initialization': gate}
        if n<len(self.package['clips']):
            raise ValueError('At least one environment per selected clip')
        torch.manual_seed(cfg['seed'])
        self.groups, self.buffers = [], []
        for i, clip in enumerate(self.package['clips']):
            count = n//len(self.package['clips'])+(i<n%len(self.package['clips']))
            self.groups.append(Collection(self.root, clip, self.package, count, device, cfg['acknowledge_engine_mapping']))
            self.buffers.append(Replay(cfg['buffer_steps']))
        context_dims = {g.sim.context.shape[1]+23 for g in self.groups}
        if len(context_dims)!=1:
            raise ValueError('Robot context dimensions differ')
        self.policy = ActorCritic().to(device)
        self.policy.log_std.requires_grad_(False)
        self.wm = WorldModel(context_dims.pop(), cfg['world_model_hidden'], cfg['amp']=='bf16').to(device)
        self.wm_step = torch.compile(self.wm.step) if cfg['compile'] and str(device).startswith('cuda') else self.wm.step
        self.pi_optimizer = torch.optim.AdamW(self.policy.actor.parameters(), lr=cfg['lr'], betas=(.9, .95))
        self.wm_optimizer = torch.optim.AdamW(self.wm.parameters(), lr=cfg['lr'], betas=(.9, .95))
        self.v_optimizer = torch.optim.AdamW(self.policy.critic.parameters(), lr=cfg['lr'], betas=(.9, .95))
        self.iteration, self.sim_steps, self.optimizer_steps = 0, 0, 0

    def collect(self, steps):
        rows = {}
        for _ in range(steps):
            for group, buffer in zip(self.groups, self.buffers):
                row = group.collect(self.policy, buffer, self.cfg['action_noise'])
                totals = rows.setdefault(group.name, {})
                for key, value in row.items():
                    if key.endswith(('_max_m', '_total')):
                        totals[key] = max(totals.get(key, value), value)
                    else:
                        totals[key] = totals.get(key, 0.) + value / steps
            self.sim_steps += sum(g.n for g in self.groups)
        return rows

    def wm_loss(self, batch=None):
        total, windows = [], []
        for g, replay in zip(self.groups, self.buffers):
            count = math.ceil((batch or self.cfg['wm_batch'])/len(self.groups))
            w = replay.sample(count, self.cfg['wm_window'])
            windows.append((g,w))
            K = w['mask'].shape[1]
            truth0, truth1 = unpack(w['s'].flatten(0, 1)), unpack(w['s1'].flatten(0, 1))
            context = torch.cat((w['context'], w['a_prev']), -1).flatten(0, 1)
            with torch.no_grad():
                valid = w['mask'].flatten()
                self.wm.input_norm.update(self.wm.inputs(truth0, w['a'].flatten(0, 1), context, g.mdp.terrain.scan(truth0))[valid])
                self.wm.output_norm.update(self.wm.delta(truth0, truth1)[valid])
        # Freeze normalization buffers for every scene's differentiable rollout.
        # Compiled backward saves these buffers; later in-place updates invalidate it.
        for g,w in windows:
            K = w['mask'].shape[1]
            S, losses = unpack(w['s'][:, 0]), []
            for k in range(K):
                ctx = torch.cat((w['context'][:, k], w['a_prev'][:, k]), -1)
                S, logits = self.wm_step(S, w['a'][:, k], ctx, g.mdp.terrain.scan(S))
                error = self.wm.delta(unpack(w['s1'][:, k]), S)
                dynamics = (error/torch.sqrt(self.wm.output_norm.variance+1e-8)).square().mean(-1)
                contacts = nn.functional.binary_cross_entropy_with_logits(logits, w['contacts'][:, k], reduction='none').mean(-1)
                mask = w['mask'][:, k]
                losses.append(((dynamics+contacts)*mask).sum()/mask.sum().clamp(min=1))
            total.append(torch.stack(losses).mean())
        return torch.stack(total).mean()

    def imagination(self, group, replay, batch=None):
        c, mdp = self.cfg, group.mdp
        w = {k:v[:, 0] if k!='env' else v for k, v in replay.sample(batch or c['pi_batch']).items()}
        S, history, poses = unpack(w['s']), w['history'], w['poses']
        previous, previous2, velocity = w['a_prev'], w['a_prev2'], w['velocity']
        rs, continuations, critics = [], [], []
        for step in range(c['horizon']):
            t = w['time']+step*mdp.dt
            actor, critic = mdp.observations(S, history, previous, t, poses, w['delay'], w['drift'], group.training_noise)
            critics.append(critic)
            action = self.policy.actor(self.policy.actor_norm(actor)).clamp(-100, 100)
            S1, logits = self.wm_step(S, action, torch.cat((w['context'], previous), -1), mdp.terrain.scan(S))
            torque, computed = group.sim.imagined_torque(S1, action, previous, w['env'])
            result = mdp.transition(S, S1, t+mdp.dt, action, previous, previous2, velocity,
                                    logits.sigmoid(), torque, computed, w['age']+step+1)
            rs.append(result['reward'])
            continuations.append(result['continuation'])
            velocity = result['body_velocity']
            previous2, previous, S = previous, action, S1
            history = torch.cat((history[:, 1:], mdp.proprio(S, action, group.training_noise)[:, None]), 1)
            poses = torch.cat((torch.cat((S[0], mat_to_quat(S[1])), -1)[:, None], poses[:, :-1]), 1)
        _, critic = mdp.observations(S, history, previous, w['time']+c['horizon']*mdp.dt,
                                     poses, w['delay'], w['drift'], group.training_noise)
        critics.append(critic)
        return torch.stack(rs), torch.stack(continuations), torch.stack(critics)

    def policy_loss(self, batch=None):
        losses, fitting = [], []
        for g, replay in zip(self.groups, self.buffers):
            count = math.ceil((batch or self.cfg['pi_batch'])/len(self.groups))
            r, cont, critic = self.imagination(g, replay, count)
            values = self.policy.value(self.policy.critic_norm(critic))
            alive = torch.cumprod(cont, 0)
            before = torch.cat((torch.ones_like(alive[:1]), alive[:-1]), 0)
            discount = self.cfg['gamma']**torch.arange(len(r), device=r.device)[:, None]
            objective = (discount*before*r).sum(0)+(self.cfg['gamma']**len(r))*alive[-1]*values[-1]
            losses.append(-objective.mean())
            with torch.no_grad():
                target, targets = values[-1].detach(), []
                for t in range(len(r)-1, -1, -1):
                    target = r[t]+self.cfg['gamma']*cont[t]*((1-self.cfg['lam'])*values[t+1]+self.cfg['lam']*target)
                    targets.append(target)
                fitting.append((critic[:-1].detach().flatten(0, 1), torch.stack(targets[::-1]).flatten().detach()))
        return torch.stack(losses).mean(), fitting

    @staticmethod
    def backward(loss, optimizer, params, apply=True):
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite SuperMimic loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = list(params)
        norm = nn.utils.clip_grad_norm_(parameters, 1.)
        if not torch.isfinite(norm):
            raise ValueError('Nonfinite SuperMimic gradient')
        nn.utils.clip_grad_value_(parameters, .1)
        if apply:
            optimizer.step()
        return float(norm)

    def update(self):
        for _ in range(self.cfg['wm_batches']):
            wm_loss = self.wm_loss()
            wm_grad = self.backward(wm_loss, self.wm_optimizer, self.wm.parameters())
            self.optimizer_steps += 1
        row = dict(wm_loss=float(wm_loss.detach()), wm_gradient=wm_grad)
        if self.iteration>=self.cfg['wm_only_iters']:
            for p in [*self.wm.parameters(), *self.policy.critic.parameters()]:
                p.requires_grad_(False)
            loss, fitting = self.policy_loss()
            row.update(policy_loss=float(loss.detach()), policy_gradient=self.backward(loss, self.pi_optimizer, self.policy.actor.parameters()))
            self.optimizer_steps += 1
            for p in [*self.wm.parameters(), *self.policy.critic.parameters()]:
                p.requires_grad_(True)
            X, Y = [torch.cat([pair[i] for pair in fitting]) for i in range(2)]
            self.policy.critic_norm.update(X)
            X = self.policy.critic_norm(X)
            for _ in range(self.cfg['critic_epochs']):
                for ids in torch.randperm(len(X), device=self.device).tensor_split(self.cfg['critic_minibatches']):
                    value_loss = (self.policy.value(X[ids])-Y[ids]).square().mean()
                    self.backward(value_loss, self.v_optimizer, self.policy.critic.parameters())
                    self.optimizer_steps += 1
            row['critic_loss'] = float(value_loss.detach())
        self.iteration += 1
        return row

    def save(self, output):
        record = dict(format='acp-supermimic-v1', iteration=self.iteration, model=self.policy.state_dict(),
                      world_model=self.wm.state_dict(), config=self.cfg, package=self.package,
                      optimizers=[o.state_dict() for o in (self.pi_optimizer, self.wm_optimizer, self.v_optimizer)],
                      phase=[g.phase.clone() for g in self.groups], sim_steps=self.sim_steps, optimizer_steps=self.optimizer_steps,
                      simulation=[dict(context=g.sim.context, physical=g.sim.physical, kp=g.sim.kp, kd=g.sim.kd,
                                       bias=g.sim.bias, delay_ticks=g.sim.delay_ticks, samples=g.sim.samples) for g in self.groups])
        temporary = Path(output)/'checkpoint.tmp'
        torch.save(record, temporary)
        temporary.replace(Path(output)/'latest.pt')

    def load(self, checkpoint):
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        if state.get('format')!='acp-supermimic-v1':
            raise ValueError('Requires a SuperMimic checkpoint; old PPO checkpoints are not resumable here')
        if state['package']!=self.package or state['config']!=self.cfg:
            raise ValueError('Resume package/configuration differs')
        self.policy.load_state_dict(state['model'])
        self.wm.load_state_dict(state['world_model'])
        for opt, values in zip((self.pi_optimizer, self.wm_optimizer, self.v_optimizer), state['optimizers']):
            opt.load_state_dict(values)
        if len(state['phase'])!=len(self.groups):
            raise ValueError('Resume task count differs')
        for g, phase, physics in zip(self.groups, state['phase'], state['simulation']):
            g.sim.restore_physics(physics)
            g.phase.copy_(phase)
            g.reset()
        self.iteration, self.sim_steps, self.optimizer_steps = state['iteration'], state['sim_steps'], state['optimizer_steps']

    @torch.no_grad()
    def evaluate(self, output, phases=(0., .9)):
        """Physical evaluation, no auto-reset or observation-normalizer updates."""
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        rows = []
        for phase in phases:
            for g in self.groups:
                training_noise = g.training_noise
                evaluating = g.evaluating
                g.training_noise = False
                g.evaluating = True
                g.reset(phase=phase)
                alive = torch.ones(g.n, dtype=torch.bool, device=self.device)
                success = torch.zeros_like(alive)
                lengths = torch.zeros(g.n, device=self.device)
                error = torch.zeros_like(lengths)
                frames, references, quaternions, forces, normal_forces, link_forces = [], [], [], [], [], []
                max_steps = math.ceil((g.mdp.reference.end-g.mdp.reference.start)*(1-phase)/g.mdp.dt)+2
                for _ in range(max_steps):
                    actor, _ = g.observations()
                    action = self.policy.actor(self.policy.actor_norm(actor)).clamp(-100, 100)
                    previous = g.sim.previous_action.clone()
                    S0 = g.sim.state()
                    S1 = g.sim.step(torch.where(alive[:, None], action, 0.))
                    g.age += 1
                    g.times += g.mdp.dt
                    contacts = g.sim.measured_contacts()
                    result = g.mdp.transition(S0, S1, g.times, action, previous, g.action_prev2, g.body_velocity,
                                              contacts, g.sim.applied_torque, g.sim.computed_torque, g.age)
                    lengths += alive
                    error += result['metrics']['root_error']*alive
                    success |= alive & result['success']
                    if alive[0]:
                        ref = g.mdp.reference.sample(g.times[:1])
                        frames.append(torch.cat((S1[0], mat_to_quat(S1[1]), S1[2], S1[3], S1[4]), -1)[0].cpu().numpy())
                        references.append(torch.cat((ref['base_position_w'], ref['joint_positions']), -1)[0].cpu().numpy())
                        quaternions.append(ref['base_quaternion_wxyz'][0].cpu().numpy())
                        forces.append(g.sim.force_history[0].cpu().numpy())
                        normal_forces.append(g.sim.normal_force_history[0].cpu().numpy())
                        link_forces.append(g.sim.link_force_history[0].cpu().numpy())
                    alive &= ~result['done']
                    if not alive.any():
                        break
                    g.action_prev2.copy_(previous)
                    g.body_velocity.copy_(result['body_velocity'])
                    g.history = torch.cat((g.history[:, 1:], g.mdp.proprio(S1, action)[:, None]), 1)
                    g.poses = torch.cat((torch.cat((S1[0], mat_to_quat(S1[1])), -1)[:, None], g.poses[:, :-1]), 1)
                label = 'full' if phase==0 else f'phase_{phase:g}'
                np.savez_compressed(output/f'{g.name}_{label}.npz', states=np.asarray(frames),
                                    reference=np.asarray(references), reference_quaternion=np.asarray(quaternions),
                                    contact_force_history=np.asarray(forces), dt=g.mdp.dt,
                                    contact_normal_force_sum_history=np.asarray(normal_forces),
                                    all_link_normal_force_sum_history=np.asarray(link_forces),
                                    all_link_names=g.sim.link_names,
                                    contact_definition='environment-only normal vectors; reward uses max abs Fz > 1N',
                                    joint_names=g.sim.robot_cfg['joint_names'], body_names=g.sim.robot_cfg['body_names'],
                                    physics_engine='mujoco', initial_phase=phase)
                rows.append(dict(task=g.name, initial_phase=phase, success_rate=float(success.float().mean()),
                                 episodes=g.n, mean_root_error_m=float((error/lengths.clamp(min=1)).mean()),
                                 unfinished=int(alive.sum()), observation_noise=False,
                                 randomized_startup_dynamics=g.sim.randomized))
                g.training_noise = training_noise
                g.evaluating = evaluating
                g.reset()
        atomic_json(output/'report.json', dict(iteration=self.iteration, results=rows, optimizer_steps=0))
        return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'check', 'smoke', 'benchmark', 'train', 'evaluate'])
    p.add_argument('--package', type=Path, default=PROJECT/'outputs/数据集/supermimic/walk_box_v1')
    p.add_argument('--config', type=Path, default=PROJECT/'configs/supermimic.json')
    p.add_argument('--stage1-config', type=Path, default=PROJECT/'configs/stage1.json',
                   help='prepare: ACP task configuration, including selected motion groups')
    p.add_argument('--task', choices=['walk', 'box', 'walk_box'], default='walk_box')
    p.add_argument('--num-envs', type=int)
    p.add_argument('--device', choices=['cpu', 'cuda:0'], default='cpu')
    p.add_argument('--gpu', default='0')
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--iterations', type=int, default=10000)
    p.add_argument('--output', type=Path)
    p.add_argument('--checkpoint', type=Path, help='SuperMimic checkpoint; train resumes, evaluate requires it')
    args = p.parse_args()
    if args.mode=='evaluate' and not args.checkpoint:
        p.error('evaluate requires --checkpoint')
    if min(args.steps, args.iterations, args.num_envs or 1)<1:
        p.error('Counts must be positive')
    if args.mode=='prepare':
        print(json.dumps(prepare(args.package, config_path=args.stage1_config, task=args.task), ensure_ascii=False, indent=2))
        return
    package = verify(args.package)
    if args.mode=='check':
        from .assets import compile_model
        for clip in package['clips']:
            compile_model(args.package/clip['scene'], package['robot_config'])
        print(json.dumps(dict(passed=True, task=package['task'], files=len(package['files']), optimizer_steps=0)))
        return
    if args.mode in ('train', 'benchmark') and args.device=='cpu':
        p.error('train/benchmark require --device cuda:0; CPU is a correctness fixture')
    if args.device.startswith('cuda'):
        from acp.management.simulation import select_gpu
        gpu = select_gpu(args.gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu['uuid']
    cfg = json.loads(args.config.read_text())
    torch.set_num_threads(4)
    if args.mode=='smoke':
        cfg = {**cfg, 'world_model_hidden':[64, 64], 'horizon':3, 'wm_window':3,
               'wm_batch':4, 'pi_batch':2, 'buffer_steps':16, 'warmup_steps':6, 'compile':False}
    output = args.output or PROJECT/'outputs/实验归档/stage1/supermimic'/f'{args.mode}_{time.time_ns()}'
    output.mkdir(parents=True, exist_ok=False)
    runner = Runner(args.package, cfg, args.num_envs or (4 if args.mode=='smoke' else 1024), args.device)
    if args.checkpoint:
        runner.load(args.checkpoint)
    provenance = dict(config=cfg, package=package, device=args.device, num_envs=sum(g.n for g in runner.groups),
                      source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(),
                      files={str(f.relative_to(PROJECT)):sha(f) for f in Path(__file__).parent.glob('*.py')},
                      versions={name:__import__('importlib.metadata', fromlist=['version']).version(name)
                                for name in ['torch', 'numpy', 'mujoco']}, physics=[g.sim.report for g in runner.groups])
    atomic_json(output/'provenance.json', provenance)
    torch.save([g.sim.samples for g in runner.groups], output/'randomization_samples.pt')
    if args.mode=='smoke':
        original = {k:v.clone() for k, v in runner.policy.state_dict().items() if 'norm.' not in k}
        contract = []
        for g in runner.groups:
            g.training_noise = False
            a, v = g.observations()
            ref = g.mdp.reference.values['joint_positions']
            g.mdp.reference.values['joint_positions'] = ref+.25
            a1, v1 = g.observations()
            g.mdp.reference.values['joint_positions'] = ref
            assert a.shape==(g.n, 468) and v.shape==(g.n, 353)
            assert torch.equal(a, a1) and not torch.equal(v, v1)
            contract.append(dict(task=g.name, actor=468, critic=353, reference_joint_leak=False))
        collection_metrics = runner.collect(max(6, min(args.steps, 16)))
        wm_grad = runner.backward(runner.wm_loss(), runner.wm_optimizer, runner.wm.parameters(), apply=False)
        for q in [*runner.wm.parameters(), *runner.policy.critic.parameters()]:
            q.requires_grad_(False)
        loss, _ = runner.policy_loss()
        pi_grad = runner.backward(loss, runner.pi_optimizer, runner.policy.actor.parameters(), apply=False)
        for key, value in original.items():
            torch.testing.assert_close(runner.policy.state_dict()[key], value, rtol=0, atol=0)
        result = dict(passed=True, optimizer_steps=runner.optimizer_steps, observation_contract=contract,
                      collection_metrics=collection_metrics,
                      wm_gradient=wm_grad, policy_gradient=pi_grad, device=args.device,
                      limitation='CPU does not validate CUDA, throughput, or convergence' if args.device=='cpu' else 'No convergence test')
    elif args.mode=='benchmark':
        runner.collect(8)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        runner.collect(args.steps)
        torch.cuda.synchronize()
        seconds = time.perf_counter()-start
        result = dict(passed=True, optimizer_steps=0, measured_control_steps=args.steps, seconds=seconds,
                      transitions_per_second=args.steps*sum(g.n for g in runner.groups)/seconds,
                      peak_vram_bytes=torch.cuda.max_memory_allocated(), scope='simulation + observations + actor + replay; no updates')
    elif args.mode=='evaluate':
        result = dict(passed=True, optimizer_steps=0, results=runner.evaluate(output/'evaluation'))
    else:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(output/'tensorboard'), flush_secs=15)
        runner.collect(max(cfg['warmup_steps'], cfg['wm_window']))
        start = time.monotonic()
        for _ in range(args.iterations):
            iteration = runner.iteration
            lr = cfg['lr']*(cfg['lr_final_ratio']+(1-cfg['lr_final_ratio'])*.5*(1+math.cos(math.pi*min(iteration/cfg['lr_decay_iters'], 1))))
            for optimizer in [runner.pi_optimizer, runner.wm_optimizer, runner.v_optimizer]:
                for param in optimizer.param_groups:
                    param['lr'] = lr
            n_collect = int((iteration+1)/cfg['utd'])-int(iteration/cfg['utd'])
            real = runner.collect(n_collect)
            row = dict(iteration=iteration+1, **runner.update(), real=real, elapsed_seconds=time.monotonic()-start,
                       sim_steps=runner.sim_steps, optimizer_steps=runner.optimizer_steps)
            def log_scalars(values, prefix=''):
                for key,value in values.items():
                    tag=f'{prefix}/{key}' if prefix else key
                    if isinstance(value,dict):log_scalars(value,tag)
                    elif isinstance(value,(int,float)):writer.add_scalar(tag,value,iteration+1)
            log_scalars(row)
            writer.add_scalar('learning_rate',lr,iteration+1)
            with (output/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            if (iteration+1)%cfg['save_interval']==0 or iteration+1==args.iterations:
                runner.save(output)
            if (iteration+1)%10==0:
                print(json.dumps(row), flush=True)
            atomic_json(output/'progress.json', row)
        runner.save(output)
        writer.close()
        result = dict(passed=True, iterations=runner.iteration, optimizer_steps=runner.optimizer_steps,
                      note='Completion is not convergence; full-start physical evaluation required')
    atomic_json(output/'result.json', result)
    print(json.dumps(dict(output=str(output), **result), ensure_ascii=False), flush=True)


if __name__=='__main__':
    main()
