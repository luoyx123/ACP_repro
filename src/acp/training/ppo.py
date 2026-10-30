"""Native PyTorch PPO, explicit timeout bootstrap and asymmetric observations.

Paper Table S6 supplies optimization settings. Gaussian std, normalization,
initialization and value clipping are recorded engineering choices.
"""
import math
import torch
from torch import nn
from torch.distributions import Normal


class RunningNormalizer(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.register_buffer('mean', torch.zeros(size))
        self.register_buffer('variance', torch.ones(size))
        self.register_buffer('count', torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x):
        n = x.shape[0];total = self.count+n;delta = x.mean(0)-self.mean
        variance = (self.variance*self.count+x.var(0, unbiased=False)*n+delta.square()*self.count*n/total)/total
        self.mean.add_(delta*n/total);self.variance.copy_(variance);self.count.copy_(total)

    def forward(self, x):
        return ((x-self.mean)/torch.sqrt(self.variance+1e-8)).clamp(-10, 10)


def network(widths, final_gain):
    layers = []
    for i, (a,b) in enumerate(zip(widths[:-1], widths[1:])):
        linear = nn.Linear(a,b)
        nn.init.orthogonal_(linear.weight, gain=final_gain if i == len(widths)-2 else math.sqrt(2))
        nn.init.zeros_(linear.bias);layers.append(linear)
        if i != len(widths)-2: layers.append(nn.ELU())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, initial_std=.5, normalize=True):
        super().__init__()
        self.actor = network([468,512,256,128,23], .01)
        self.critic = network([353,512,256,128,1], 1.)
        self.log_std = nn.Parameter(torch.full((23,), math.log(initial_std)))
        self.actor_norm = RunningNormalizer(468) if normalize else nn.Identity()
        self.critic_norm = RunningNormalizer(353) if normalize else nn.Identity()

    def distribution(self, actor_obs):
        mean = self.actor(actor_obs)
        return Normal(mean, self.log_std.clamp(-5,2).exp().expand_as(mean))

    def value(self, critic_obs):
        return self.critic(critic_obs).squeeze(-1)

    @torch.no_grad()
    def act(self, actor_obs, critic_obs):
        distribution = self.distribution(actor_obs);action = distribution.sample()
        return action, distribution.log_prob(action).sum(-1), self.value(critic_obs), distribution.mean, distribution.stddev


def generalized_advantages(rewards, values, next_values, terminated, done, gamma=.99, lam=.95):
    """Truncation bootstraps V(terminal_obs), but does not cross the reset boundary."""
    advantage = torch.zeros_like(rewards);carry = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t]+gamma*next_values[t]*(~terminated[t])-values[t]
        carry = delta+gamma*lam*(~done[t])*carry
        advantage[t] = carry
    return advantage, advantage+values


class PPO:
    def __init__(self, model, config):
        self.model, self.cfg = model, config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    def update(self, rollout):
        data = {k:torch.stack([x[k] for x in rollout]) for k in rollout[0]}
        advantage, returns = generalized_advantages(data['reward'],data['value'],data['next_value'],
            data['terminated'],data['done'],self.cfg['gamma'],self.cfg['gae_lambda'])
        advantage = (advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
        data = {k:v.flatten(0,1) for k,v in data.items()}
        advantage, returns = advantage.flatten(), returns.flatten()
        count = len(advantage);statistics = []
        for _ in range(self.cfg['epochs']):
            permutation = torch.randperm(count,device=advantage.device)
            for ids in permutation.tensor_split(self.cfg['minibatches']):
                distribution = self.model.distribution(data['actor'][ids])
                value = self.model.value(data['critic'][ids])
                with torch.no_grad():
                    old_mean, old_std = data['mean'][ids], data['std'][ids]
                    kl = (torch.log(distribution.stddev/old_std)+(old_std.square()+
                        (old_mean-distribution.mean).square())/(2*distribution.stddev.square())-.5).sum(-1).mean()
                    rate = self.optimizer.param_groups[0]['lr']
                    if kl > 2*self.cfg['target_kl']: rate = max(1e-5,rate/1.5)
                    elif 0 < kl < .5*self.cfg['target_kl']: rate = min(1e-2,rate*1.5)
                    for group in self.optimizer.param_groups: group['lr'] = rate
                ratio = torch.exp(distribution.log_prob(data['action'][ids]).sum(-1)-data['log_prob'][ids])
                policy_loss = torch.maximum(-advantage[ids]*ratio,
                    -advantage[ids]*ratio.clamp(1-self.cfg['clip'],1+self.cfg['clip'])).mean()
                value_clipped = data['value'][ids]+(value-data['value'][ids]).clamp(-self.cfg['clip'],self.cfg['clip'])
                value_loss = (torch.maximum((value-returns[ids]).square(),(value_clipped-returns[ids]).square()).mean()
                    if self.cfg['value_clipping'] else (value-returns[ids]).square().mean())
                entropy = distribution.entropy().sum(-1).mean()
                loss = policy_loss+self.cfg['value_coefficient']*value_loss-self.cfg['entropy_coefficient']*entropy
                if not torch.isfinite(loss): raise ValueError('Nonfinite PPO objective')
                self.optimizer.zero_grad(set_to_none=True);loss.backward()
                gradient = nn.utils.clip_grad_norm_(self.model.parameters(),self.cfg['max_grad_norm'])
                if not torch.isfinite(gradient): raise ValueError('Nonfinite PPO gradient')
                self.optimizer.step()
                statistics.append(torch.stack([policy_loss.detach(),value_loss.detach(),entropy.detach(),kl.detach(),gradient.detach()]))
        mean = torch.stack(statistics).mean(0).cpu().tolist()
        return dict(zip(('policy_loss','value_loss','entropy','kl','gradient_norm'),mean),
                    learning_rate=self.optimizer.param_groups[0]['lr'], action_std=float(self.model.log_std.exp().mean().detach()))
