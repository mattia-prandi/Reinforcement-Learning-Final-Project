"""
Imitation Learning project — Applied Project 3 (EE-568 RL, EPFL).

Standalone Python script extracted from imitation_learning_project.ipynb.
Same code, same plots, same training. Use this for fast edits

Run: /home/mattia/anaconda3/bin/python3 imitation_learning_project.py
"""

import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib
matplotlib.use('Agg', force=True)


# ==============================================================================
# CODE cell 1
# ==============================================================================
import os
import random
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', message='.*Gym has been unmaintained.*')

import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt

PROJECT_DIR = Path('.').resolve()
ARTIFACTS_DIR = PROJECT_DIR / 'artifacts'
EXPERTS_DIR = ARTIFACTS_DIR / 'experts'
DATASETS_DIR = ARTIFACTS_DIR / 'datasets'
RESULTS_DIR = ARTIFACTS_DIR / 'results'
for d in (EXPERTS_DIR, DATASETS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Project dir: {PROJECT_DIR}')
print(f'Device: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ==============================================================================
# CODE cell 2
# ==============================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(0)

# ==============================================================================
# CODE cell 3
# ==============================================================================
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

ENVS = {
    'Pendulum-v1': {
        'total_timesteps': 300_000,
        'ppo_kwargs': dict(
            learning_rate=1e-3,
            n_steps=1024,
            batch_size=64,
            gamma=0.9,
            gae_lambda=0.95,
        ),
    },
    'MountainCarContinuous-v0': {
        'total_timesteps': 500_000,
        'ppo_kwargs': dict(
            learning_rate=7e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.9999,
            gae_lambda=0.9,
            use_sde=True,
            sde_sample_freq=4,
        ),
    },
}

def train_expert(env_id: str, total_timesteps: int, ppo_kwargs: dict, seed: int = 0):
    """Train PPO on a single environment and return the trained model."""
    env = make_vec_env(env_id, n_envs=4, seed=seed)
    # MlpPolicy: 2 hidden layers of 64 units each, tanh activation, separate heads for actor and critic
    model = PPO('MlpPolicy', env, verbose=0, seed=seed, device=DEVICE, **ppo_kwargs)
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    return model

# ==============================================================================
# CODE cell 4
# ==============================================================================
# Training loop
for env_id, cfg in ENVS.items():
    save_path = EXPERTS_DIR / f'ppo_{env_id}.zip'
    if save_path.exists():
        print(f'[skip] expert already trained for {env_id} — at {save_path}')
        continue
    print(f'[train] PPO on {env_id} for {cfg["total_timesteps"]:,} steps...')
    model = train_expert(env_id, cfg['total_timesteps'], cfg['ppo_kwargs'], seed=0)
    model.save(str(save_path))

    eval_env = gym.make(env_id)
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=20)
    print(f'  -> eval reward over 20 episodes: {mean_r:.2f} ± {std_r:.2f}')
    eval_env.close()

# ==============================================================================
# CODE cell 5
# ==============================================================================
def _pendulum_reward(theta, theta_dot, torque=0.0):
    # Gymnasium Pendulum-v1: r = -(theta_norm^2 + 0.1*theta_dot^2 + 0.001*torque^2)
    theta_norm = ((theta + np.pi) % (2 * np.pi)) - np.pi
    return -(theta_norm ** 2 + 0.1 * theta_dot ** 2 + 0.001 * torque ** 2)


def _mcc_reward(position, velocity, action=0.0, goal_position=0.45):
    # MountainCarContinuous-v0: r = -0.1*action^2 each step, +100 if position >= 0.45
    base = -0.1 * action ** 2 * np.ones_like(position)
    return np.where(position >= goal_position, base + 100.0, base)


def _plot_pendulum_reward_heatmap():
    thetas = np.linspace(-np.pi, np.pi, 300)
    theta_dots = np.linspace(-8.0, 8.0, 300)
    TH, TD = np.meshgrid(thetas, theta_dots)
    R = _pendulum_reward(TH, TD)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.pcolormesh(TH, TD, R, shading='auto', cmap='viridis')
    ax.set_xlabel(r'$\theta$ (rad)')
    ax.set_ylabel(r'$\dot{\theta}$ (rad/s)')
    ax.set_title('Pendulum-v1 — ground-truth reward (torque = 0)')
    ax.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='upright')
    ax.legend(loc='upper right')
    fig.colorbar(im, ax=ax, label='reward')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'reward_heatmap_pendulum.png', dpi=140, bbox_inches='tight')
    plt.show()


def _plot_mcc_reward_heatmap():
    positions = np.linspace(-1.2, 0.6, 300)
    velocities = np.linspace(-0.07, 0.07, 300)
    P, V = np.meshgrid(positions, velocities)
    R = _mcc_reward(P, V)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.pcolormesh(P, V, R, shading='auto', cmap='viridis')
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')
    ax.set_title('MountainCarContinuous-v0 — ground-truth reward (action = 0)')
    ax.axvline(0.45, color='red', linestyle='--', linewidth=1, alpha=0.7, label='goal')
    ax.legend(loc='upper right')
    fig.colorbar(im, ax=ax, label='reward')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'reward_heatmap_mcc.png', dpi=140, bbox_inches='tight')
    plt.show()


_plot_pendulum_reward_heatmap()
_plot_mcc_reward_heatmap()

# ==============================================================================
# CODE cell 6
# ==============================================================================
# Generate the expert dataset

K_MAX = 100 # trajectories per environment

def collect_trajectories(model, env_id: str, n_traj: int, seed: int = 0):
    env = gym.make(env_id)
    rng = np.random.default_rng(seed)
    trajs = []
    for _ in range(n_traj):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        traj = {'obs': [], 'act': [], 'next_obs': [], 'done': [], 'rew': []}
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            traj['obs'].append(obs)
            traj['act'].append(action)
            traj['next_obs'].append(next_obs)
            traj['rew'].append(reward)
            done = bool(terminated or truncated)
            traj['done'].append(done)
            obs = next_obs
        trajs.append({k: np.asarray(v) for k, v in traj.items()})
    env.close()
    return trajs

for env_id in ENVS:
    out_path = DATASETS_DIR / f'expert_{env_id}_K{K_MAX}.npz'
    if out_path.exists():
        print(f'[skip] dataset already exists for {env_id} — at {out_path}')
        continue
    print(f'[collect] {K_MAX} expert trajectories on {env_id}...')
    model = PPO.load(str(EXPERTS_DIR / f'ppo_{env_id}.zip'))
    trajs = collect_trajectories(model, env_id, K_MAX, seed=42)
    obs = np.concatenate([t['obs'] for t in trajs])
    act = np.concatenate([t['act'] for t in trajs])
    next_obs = np.concatenate([t['next_obs'] for t in trajs])
    done = np.concatenate([t['done'] for t in trajs])
    rew = np.concatenate([t['rew'] for t in trajs])
    lengths = np.array([len(t['obs']) for t in trajs])
    np.savez(out_path, obs=obs, act=act, next_obs=next_obs, done=done, rew=rew, lengths=lengths)
    mean_traj_reward = sum(t['rew'].sum() for t in trajs) / K_MAX
    print(f'  -> shape: obs {obs.shape}, mean trajectory reward {mean_traj_reward:.2f}')

# ==============================================================================
# CODE cell 7
# ==============================================================================
# Each bar shows how many trajectories achieved a certain total reward value.
# The x-axis represents the total reward of a trajectory,
# while the y-axis shows how many trajectories obtained that reward.
# The red dashed line represents the average trajectory reward.

fig, axes = plt.subplots(1, len(ENVS), figsize=(11, 3.5))
for ax, env_id in zip(axes, ENVS):
    data = np.load(DATASETS_DIR / f'expert_{env_id}_K{K_MAX}.npz')
    lengths = data['lengths']
    rew = data['rew']
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    traj_returns = np.array([rew[s:s + l].sum() for s, l in zip(starts, lengths)])
    ax.hist(traj_returns, bins=20, color='steelblue', edgecolor='black')
    ax.axvline(traj_returns.mean(), color='red', linestyle='--', label=f'mean = {traj_returns.mean():.1f}')
    ax.set_title(env_id)
    ax.set_xlabel('Trajectory return')
    ax.set_ylabel('Count')
    ax.legend()
plt.tight_layout()
plt.show()

# ==============================================================================
# CODE cell 8
# ==============================================================================
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from matplotlib import animation
from IPython.display import HTML, display


def render_policy(policy_fn, env_id: str, max_steps: int = 500, seed: int = 0):

    env = gym.make(env_id, render_mode='rgb_array')
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    total_reward = 0.0
    for _ in range(max_steps):
        action = policy_fn(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        total_reward += float(reward)
        if terminated or truncated:
            break
    env.close()
    return np.stack(frames), total_reward


def show_video(frames: np.ndarray, fps: int = 30, figsize=(4, 3)):
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    img = ax.imshow(frames[0])

    def update(i):
        img.set_data(frames[i])
        return [img]

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames),
        interval=1000 / fps, blit=True,
    )
    plt.close(fig)  
    return HTML(anim.to_jshtml())


# Render one deterministic episode of each PPO expert.
for env_id in ENVS:
    model = PPO.load(str(EXPERTS_DIR / f'ppo_{env_id}.zip'))
    policy_fn = lambda obs, m=model: m.predict(obs, deterministic=True)[0]
    frames, total_r = render_policy(policy_fn, env_id, max_steps=500, seed=0)
    print(f'{env_id} expert: total reward = {total_r:.2f} ({len(frames)} frames)')
    display(show_video(frames, fps=30))

# ==============================================================================
# CODE cell 9
# ==============================================================================
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, hidden_depth: int) -> nn.Module:
    
    if hidden_depth == 0:
        layers = [nn.Linear(in_dim, out_dim)]
    else:
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(hidden_depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        layers += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*layers)


def _orthogonal_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()


class _TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size: int = 1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def _atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, _TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self._atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # Numerically stable form (TF Probability trick).
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.base_dist = pyd.Normal(loc, scale)
        super().__init__(self.base_dist, [_TanhTransform()])

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class DiagGaussianActor(nn.Module):

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256,
                 hidden_depth: int = 2, log_std_bounds=(-5.0, 2.0)):
        super().__init__()
        self.log_std_bounds = log_std_bounds
        self.trunk = _mlp(obs_dim, hidden_dim, 2 * action_dim, hidden_depth)
        self.apply(_orthogonal_init)

    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        mu = torch.clamp(mu, -9.0, 9.0)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = torch.tanh(log_std)
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        return SquashedNormal(mu, log_std.exp())

    def sample(self, obs):
        dist = self.forward(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob, dist.mean


class DoubleQCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, hidden_depth: int = 2):
        super().__init__()
        self.Q1 = _mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = _mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.apply(_orthogonal_init)

    def forward(self, obs, action, both: bool = False):
        x = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return (q1, q2) if both else torch.min(q1, q2)


@torch.no_grad()
def soft_update(net: nn.Module, target_net: nn.Module, tau: float) -> None:
    for p, p_target in zip(net.parameters(), target_net.parameters()):
        p_target.data.mul_(1.0 - tau).add_(tau * p.data)

# ==============================================================================
# CODE cell 10
# ==============================================================================
# Data utilities used during training.
# The ReplayBuffer stores transitions collected by the current agent while interacting with the environment,
# whereas the ExpertDataset loads the expert demonstrations generated with PPO.

from typing import NamedTuple


class Batch(NamedTuple):
    
    obs: torch.Tensor
    action: torch.Tensor          
    next_obs: torch.Tensor
    done: torch.Tensor            
    is_expert: torch.Tensor       


class ReplayBuffer:

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)
        self.idx = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def add(self, obs, action, next_obs, terminated: bool) -> None:
        self.obs[self.idx] = obs
        self.action[self.idx] = action
        self.next_obs[self.idx] = next_obs
        self.done[self.idx] = float(terminated)
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(self, batch_size: int) -> Batch:
        n = len(self)
        idx = np.random.randint(0, n, size=batch_size)
        to_t = lambda x: torch.as_tensor(x[idx], device=self.device)
        return Batch(
            obs=to_t(self.obs),
            action=to_t(self.action),
            next_obs=to_t(self.next_obs),
            done=to_t(self.done),
            is_expert=torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device),
        )


class ExpertDataset:

    def __init__(self, npz_path, K: int, action_range: float, device: torch.device):
        d = np.load(npz_path)
        lengths = d['lengths']
        if K > len(lengths):
            raise ValueError(f'K={K} larger than dataset size {len(lengths)}')
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        idx = np.concatenate([np.arange(starts[i], starts[i] + lengths[i]) for i in range(K)])
        self.obs = torch.as_tensor(d['obs'][idx], dtype=torch.float32, device=device)
        self.next_obs = torch.as_tensor(d['next_obs'][idx], dtype=torch.float32, device=device)
      
        action_range = float(action_range)
        self.action = torch.as_tensor(d['act'][idx] / action_range, dtype=torch.float32, device=device)
        done_arr = np.zeros(len(idx), dtype=np.float32)
        self.done = torch.as_tensor(done_arr.reshape(-1, 1), device=device)
        self.device = device
        self.size = len(idx)

    def set_done_from_terminated(self, terminated_mask: np.ndarray) -> None:
        self.done = torch.as_tensor(terminated_mask.reshape(-1, 1).astype(np.float32),
                                    device=self.device)

    def sample(self, batch_size: int) -> Batch:
        idx = torch.randint(0, self.size, size=(batch_size,), device=self.device)
        return Batch(
            obs=self.obs[idx],
            action=self.action[idx],
            next_obs=self.next_obs[idx],
            done=self.done[idx],
            is_expert=torch.ones(batch_size, 1, dtype=torch.bool, device=self.device),
        )


def cat_batches(a: Batch, b: Batch) -> Batch:
    return Batch(
        obs=torch.cat([a.obs, b.obs], dim=0),
        action=torch.cat([a.action, b.action], dim=0),
        next_obs=torch.cat([a.next_obs, b.next_obs], dim=0),
        done=torch.cat([a.done, b.done], dim=0),
        is_expert=torch.cat([a.is_expert, b.is_expert], dim=0),
    )

# ==============================================================================
# CODE cell 11
# ==============================================================================
# Implementation of the IQ-Learn agent.
# The agent learns from both expert demonstrations and online interactions with the environment,
# while ignoring the rewards.

from torch.optim import Adam


class IQLearnAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_range: float,           
        device: torch.device,
        gamma: float = 0.99,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        critic_tau: float = 0.005,
        init_temp: float = 1e-2,       # FIXED entropy temperature (auto-tune disabled); OLD-notebook value
        alpha_iq: float = 0.5,         
        actor_betas=(0.9, 0.999),
        critic_betas=(0.9, 0.999),
        hidden_dim: int = 256,
        hidden_depth: int = 2,
        log_std_bounds=(-5.0, 2.0),
    ):
        self.device = device
        self.gamma = gamma
        self.critic_tau = critic_tau
        self.alpha_iq = alpha_iq
        self.action_range = float(action_range)

        self.actor = DiagGaussianActor(obs_dim, action_dim, hidden_dim, hidden_depth,
                                       log_std_bounds).to(device)
        self.critic = DoubleQCritic(obs_dim, action_dim, hidden_dim, hidden_depth).to(device)
        self.critic_target = DoubleQCritic(obs_dim, action_dim, hidden_dim, hidden_depth).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.log_alpha_ent = torch.tensor(math.log(init_temp), device=device, requires_grad=False)
        self.target_entropy = -action_dim

        self.actor_opt = Adam(self.actor.parameters(), lr=actor_lr, betas=actor_betas)
        self.critic_opt = Adam(self.critic.parameters(), lr=critic_lr, betas=critic_betas)

    @property
    def alpha_ent(self) -> torch.Tensor:
        return self.log_alpha_ent.exp()
    

    def choose_action(self, obs_np: np.ndarray, deterministic: bool) -> np.ndarray:
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist = self.actor(obs)
            action_squashed = dist.mean if deterministic else dist.sample()
        action_env = action_squashed.cpu().numpy()[0] * self.action_range
        return np.clip(action_env, -self.action_range, self.action_range).astype(np.float32)


    def getV(self, obs: torch.Tensor) -> torch.Tensor:
        action, log_prob, _ = self.actor.sample(obs)
        q = self.critic(obs, action)
        return q - self.alpha_ent.detach() * log_prob

    def get_targetV(self, obs: torch.Tensor) -> torch.Tensor:
        action, log_prob, _ = self.actor.sample(obs)
        q = self.critic_target(obs, action)
        return q - self.alpha_ent.detach() * log_prob


    def update_critic_iq(self, mixed_batch: Batch) -> dict:
        obs, action, next_obs = mixed_batch.obs, mixed_batch.action, mixed_batch.next_obs
        done, is_expert = mixed_batch.done, mixed_batch.is_expert

        q1, q2 = self.critic(obs, action, both=True)
        
        with torch.no_grad():
            next_v = self.get_targetV(next_obs)
        current_v = self.getV(obs)


        y = (1.0 - done) * self.gamma * next_v

        def _iq_loss(current_q):
            reward = current_q - y
            loss_q = -(reward[is_expert]).mean()
            loss_v = (current_v - y).mean()
            chi2 = ((reward[is_expert]) ** 2).mean() / (4.0 * self.alpha_iq)
            return loss_q + loss_v + chi2, dict(q=loss_q.item(), v=loss_v.item(), chi2=chi2.item())

        l1, log1 = _iq_loss(q1)
        l2, log2 = _iq_loss(q2)
        loss = l1 + l2

        self.critic_opt.zero_grad(set_to_none=True)
        loss.backward()

        self.critic_opt.step()
        return {'critic_total': loss.item(), 'q': log1['q'], 'v': log1['v'], 'chi2': log1['chi2']}


    def update_actor(self, obs: torch.Tensor) -> dict:
        action, log_prob, _ = self.actor.sample(obs)
        q = self.critic(obs, action)
        actor_loss = (self.alpha_ent.detach() * log_prob - q).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # No alpha update — entropy temperature is fixed at init_temp.

        return {'actor_loss': actor_loss.item(), 'entropy': -log_prob.mean().item(), 'alpha': self.alpha_ent.item()}


    def update(self, online_batch: Batch, expert_batch: Batch) -> dict:
        mixed = cat_batches(online_batch, expert_batch)
        critic_logs = self.update_critic_iq(mixed)
        actor_logs = self.update_actor(mixed.obs)
        soft_update(self.critic, self.critic_target, self.critic_tau)
        return {**critic_logs, **actor_logs}

# ==============================================================================
# CODE cell 12
# ==============================================================================
# This cell implements the complete IQ-Learn training pipeline.
# The environment rewards are only used during evaluation and are never used for training updates.

import time
import shutil
from dataclasses import dataclass


_ENV_META = {
    'Pendulum-v1':              dict(action_range=2.0, can_terminate=False),
    'MountainCarContinuous-v0': dict(action_range=1.0, can_terminate=True),
}


@dataclass
class IQConfig:
    total_steps: int = 50_000      
    eval_interval: int = 2_500     
    eval_episodes: int = 10        
    batch_size: int = 256          
    replay_capacity: int = 100_000
    warmup_steps: int = 1_000      
    updates_per_step: int = 1      


@torch.no_grad()
def evaluate_policy_eval(agent, env_id: str, n_episodes: int, seed: int) -> float:
    env = gym.make(env_id)
    rng = np.random.default_rng(seed + 10_000)  
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        ep_ret = 0.0
        done = False
        while not done:
            action = agent.choose_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += float(reward)
            done = bool(terminated or truncated)
        returns.append(ep_ret)
    env.close()
    return float(np.mean(returns))


def train_iq_learn(env_id: str, K: int, seed: int,
                   cfg: 'IQConfig | None' = None,
                   overwrite: bool = False) -> dict:
    cfg = cfg or IQConfig()
    set_seed(seed)
    meta = _ENV_META[env_id]

    out_dir = RESULTS_DIR / 'iq_learn' / f'{env_id}_K{K}_seed{seed}'
    if out_dir.exists():
        if overwrite:
            shutil.rmtree(out_dir)
        elif (out_dir / 'learning_curve.npz').exists():
            print(f'[skip] {out_dir} already has results — pass overwrite=True to redo')
            return {'skipped': True, 'out_path': str(out_dir / 'learning_curve.npz')}
    out_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = float(meta['action_range'])

    agent = IQLearnAgent(
        obs_dim=obs_dim, action_dim=action_dim, action_range=action_range,
        device=DEVICE,
    )

    expert_path = DATASETS_DIR / f'expert_{env_id}_K{K_MAX}.npz'
    expert = ExpertDataset(expert_path, K=K, action_range=action_range, device=DEVICE)
    replay = ReplayBuffer(cfg.replay_capacity, obs_dim, action_dim, DEVICE)

    eval_steps, eval_rewards = [], []
    best_eval = -float('inf')
    best_actor_state = None
    best_critic_state = None
    t0 = time.time()
    obs, _ = env.reset(seed=seed)
    for step in range(1, cfg.total_steps + 1):
        if step <= cfg.warmup_steps:
            env_action = env.action_space.sample().astype(np.float32)
        else:
            env_action = agent.choose_action(obs, deterministic=False)

        next_obs, _reward, terminated, truncated, _ = env.step(env_action)
        terminated = bool(terminated) and meta['can_terminate']

        replay.add(obs, env_action / action_range, next_obs, terminated)

        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs

        
        if len(replay) >= cfg.batch_size and step > cfg.warmup_steps:
            for _ in range(cfg.updates_per_step):
                online_b = replay.sample(cfg.batch_size)
                expert_b = expert.sample(cfg.batch_size)
                agent.update(online_b, expert_b)

        
        if step % cfg.eval_interval == 0 or step == cfg.total_steps:
            r = evaluate_policy_eval(agent, env_id, cfg.eval_episodes, seed)
            eval_steps.append(step)
            eval_rewards.append(r)
            elapsed = time.time() - t0
            marker = '*' if r > best_eval else ' '
            
            with torch.no_grad():
                if len(replay) >= 64:
                    diag_batch = replay.sample(min(256, len(replay)))
                    q_pol = agent.critic(diag_batch.obs, diag_batch.action).mean().item()
                    v_pol = agent.getV(diag_batch.obs).mean().item()
                else:
                    q_pol = float('nan'); v_pol = float('nan')
                alpha = agent.alpha_ent.item()
            print(f'  step {step:>6} | eval reward {r:8.2f} {marker} | '
                  f'α={alpha:7.4f} Q_pol={q_pol:+9.2f} V_pol={v_pol:+9.2f} | {elapsed:6.1f}s')
            if r > best_eval:
                best_eval = r
                best_actor_state = {k: v.detach().cpu().clone() for k, v in agent.actor.state_dict().items()}
                best_critic_state = {k: v.detach().cpu().clone() for k, v in agent.critic.state_dict().items()}

    env.close()

    if best_actor_state is not None:
        agent.actor.load_state_dict(best_actor_state)
    if best_critic_state is not None:
        agent.critic.load_state_dict(best_critic_state)

    np.savez(out_dir / 'learning_curve.npz',
             eval_step=np.asarray(eval_steps),
             eval_reward=np.asarray(eval_rewards),
             env_id=env_id, K=K, seed=seed,
             best_eval=best_eval)
    torch.save(agent.actor.state_dict(), out_dir / 'actor.pt')
    torch.save(agent.critic.state_dict(), out_dir / 'critic.pt')

    return {
        'final_reward': eval_rewards[-1],
        'best_reward': best_eval,
        'out_path': str(out_dir / 'learning_curve.npz'),
        'agent': agent,
    }


def load_iq_agent(env_id: str, K: int, seed: int) -> 'IQLearnAgent':
    meta = _ENV_META[env_id]
    env = gym.make(env_id)
    agent = IQLearnAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        action_range=float(meta['action_range']),
        device=DEVICE,
    )
    env.close()
    out_dir = RESULTS_DIR / 'iq_learn' / f'{env_id}_K{K}_seed{seed}'
    agent.actor.load_state_dict(torch.load(out_dir / 'actor.pt', map_location=DEVICE, weights_only=True))
    agent.critic.load_state_dict(torch.load(out_dir / 'critic.pt', map_location=DEVICE, weights_only=True))
    agent.actor.eval(); agent.critic.eval()
    return agent

# ==============================================================================
# CODE cell 13
# ==============================================================================

class MLPReward(nn.Module):

    def __init__(self, obs_dim: int, hidden_dim: int = 256, hidden_depth: int = 2,
                 clamp_magnitude: float = 10.0):
        super().__init__()
        self.clamp_magnitude = clamp_magnitude
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            *sum(([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()] for _ in range(hidden_depth - 1)), []),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.net(obs)
        return torch.clamp(out, -self.clamp_magnitude, self.clamp_magnitude)

    @torch.no_grad()
    def scalar_reward(self, obs_np: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=next(self.parameters()).device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        return self.forward(obs).cpu().numpy().flatten()


class StateDiscriminator(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128, hidden_depth: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            *sum(([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()] for _ in range(hidden_depth - 1)), []),
            nn.Linear(hidden_dim, 1),
        )
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def log_density_ratio(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs).squeeze(-1)

    def train_step(self, expert_obs: torch.Tensor, policy_obs: torch.Tensor,
                   optimizer: torch.optim.Optimizer, batch_size: int = 128):
        n_e, n_p = expert_obs.shape[0], policy_obs.shape[0]
        e_idx = torch.randint(0, n_e, (batch_size,), device=expert_obs.device)
        p_idx = torch.randint(0, n_p, (batch_size,), device=policy_obs.device)
        e_logits = self.forward(expert_obs[e_idx])
        p_logits = self.forward(policy_obs[p_idx])
        logits = torch.cat([e_logits, p_logits], dim=0)
        targets = torch.cat([
            torch.ones(batch_size, 1, device=logits.device),
            torch.zeros(batch_size, 1, device=logits.device),
        ], dim=0)
        loss = self.bce(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            acc = ((torch.sigmoid(logits) > 0.5).float() == targets).float().mean().item()
        return loss.item(), acc


class QEnsemble(nn.Module):

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256,
                 hidden_depth: int = 2, num_pairs: int = 1):
        super().__init__()
        self.num_pairs = num_pairs
        self.q1_list = nn.ModuleList([
            _mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth) for _ in range(num_pairs)
        ])
        self.q2_list = nn.ModuleList([
            _mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth) for _ in range(num_pairs)
        ])
        self.apply(_orthogonal_init)

    def both(self, obs: torch.Tensor, action: torch.Tensor, pair_idx: int):
        x = torch.cat([obs, action], dim=-1)
        return self.q1_list[pair_idx](x), self.q2_list[pair_idx](x)

    def min_per_pair(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return torch.stack([torch.min(q1(x), q2(x)) for q1, q2 in zip(self.q1_list, self.q2_list)], dim=0)

    def pair_parameters(self, pair_idx: int):
        return list(self.q1_list[pair_idx].parameters()) + list(self.q2_list[pair_idx].parameters())

# ==============================================================================
# CODE cell 14
# ==============================================================================
# This cell implements the SAC-based agent used by f-IRL

from typing import Optional


class FIRLSACAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_range: float,
        device: torch.device,
        reward_net: MLPReward,
        num_q_pairs: int = 1,
        uncertainty_coef: float = 1.0,   
        q_std_clip: float = 1.0,          
        gamma: float = 0.99,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        critic_tau: float = 0.005,
        init_temp: float = 0.1,
        hidden_dim: int = 256,
        hidden_depth: int = 2,
        log_std_bounds=(-5.0, 2.0),
    ):
        self.device = device
        self.gamma = gamma
        self.critic_tau = critic_tau
        self.action_range = float(action_range)
        self.reward_net = reward_net
        self.num_q_pairs = num_q_pairs
        self.uncertainty_coef = uncertainty_coef
        self.q_std_clip = q_std_clip

        self.actor = DiagGaussianActor(obs_dim, action_dim, hidden_dim, hidden_depth, log_std_bounds).to(device)
        self.critic = QEnsemble(obs_dim, action_dim, hidden_dim, hidden_depth, num_q_pairs).to(device)
        self.critic_target = QEnsemble(obs_dim, action_dim, hidden_dim, hidden_depth, num_q_pairs).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.log_alpha_ent = torch.tensor(math.log(init_temp), device=device, requires_grad=True)
        self.target_entropy = -action_dim

        self.actor_opt = Adam(self.actor.parameters(), lr=actor_lr)
        self.alpha_opt = Adam([self.log_alpha_ent], lr=alpha_lr)
       
        self.critic_opts = [
            Adam(self.critic.pair_parameters(i), lr=critic_lr)
            for i in range(num_q_pairs)
        ]

    @property
    def alpha_ent(self) -> torch.Tensor:
        return self.log_alpha_ent.exp()

    def choose_action(self, obs_np: np.ndarray, deterministic: bool) -> np.ndarray:
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist = self.actor(obs)
            action_squashed = dist.mean if deterministic else dist.sample()
        action_env = action_squashed.cpu().numpy()[0] * self.action_range
        return np.clip(action_env, -self.action_range, self.action_range).astype(np.float32)

    def update_critic(self, batch: 'Batch') -> dict:
        obs, action, next_obs, done = batch.obs, batch.action, batch.next_obs, batch.done
        with torch.no_grad():
            r = self.reward_net(obs).detach()
            next_a, next_log_p, _ = self.actor.sample(next_obs)

        total_loss = 0.0
        for i in range(self.num_q_pairs):
            with torch.no_grad():
                q1_targ, q2_targ = self.critic_target.both(next_obs, next_a, i)
                q_targ = torch.min(q1_targ, q2_targ)
                target = r + self.gamma * (1.0 - done) * (q_targ - self.alpha_ent.detach() * next_log_p)
            q1, q2 = self.critic.both(obs, action, i)
            loss_i = ((q1 - target) ** 2).mean() + ((q2 - target) ** 2).mean()
            self.critic_opts[i].zero_grad(set_to_none=True)
            loss_i.backward()
            nn.utils.clip_grad_norm_(list(self.critic.pair_parameters(i)), max_norm=1.0)
            self.critic_opts[i].step()
            total_loss += loss_i.item()
        return {'critic_loss': total_loss / self.num_q_pairs}

    def update_actor(self, obs: torch.Tensor) -> dict:
        action, log_p, _ = self.actor.sample(obs)
        q_mins = self.critic.min_per_pair(obs, action)  
        q_mean = q_mins.mean(dim=0)
        if self.num_q_pairs > 1:
            q_std = q_mins.std(dim=0).clamp(0, self.q_std_clip)
            bonus = self.uncertainty_coef * q_std
        else:
            bonus = 0.0
        actor_loss = (self.alpha_ent.detach() * log_p - (q_mean + bonus)).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha_ent * (log_p.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        return {'actor_loss': actor_loss.item(),
                'q_mean': q_mean.mean().item(),
                'q_std': float(q_mins.std(dim=0).mean().item()) if self.num_q_pairs > 1 else 0.0,
                'alpha': self.alpha_ent.item()}

    def update(self, batch: 'Batch') -> dict:
        critic_logs = self.update_critic(batch)
        actor_logs = self.update_actor(batch.obs)
        soft_update(self.critic, self.critic_target, self.critic_tau)
        return {**critic_logs, **actor_logs}

# ==============================================================================
# CODE cell 15
# ==============================================================================
# Complete f-IRL training pipeline

@dataclass
class FIRLConfig:
    n_outer: int = 30
    sac_steps_per_outer: int = 2_000
    n_traj_per_outer: int = 10
    disc_iters: int = 20              # paper-range; high values overfit on a fixed policy batch
    reward_steps: int = 5             # multiple reward steps per outer help the signal propagate
    sac_batch_size: int = 256
    sac_warmup_steps: int = 2_000
    replay_capacity: int = 100_000
    eval_episodes: int = 10
    eval_every_n_outer: int = 3
    disc_batch_size: int = 128
    disc_lr: float = 3e-4
    reward_lr: float = 1e-4
    uncertainty_coef: float = 1.0
    q_std_clip: float = 1.0


def _collect_rollout_states(agent: 'FIRLSACAgent', env_id: str, n_traj: int,
                            seed: int, max_steps: int):
    env = gym.make(env_id)
    rng = np.random.default_rng(seed)
    trajs, lengths = [], []
    for _ in range(n_traj):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        traj = [obs.astype(np.float32)]
        for _ in range(max_steps - 1):
            action = agent.choose_action(obs, deterministic=False)
            obs, _, terminated, truncated, _ = env.step(action)
            traj.append(obs.astype(np.float32))
            if terminated or truncated:
                break
        trajs.append(np.stack(traj))
        lengths.append(len(traj))
    env.close()
    obs_dim = trajs[0].shape[-1]
    padded = np.zeros((n_traj, max_steps, obs_dim), dtype=np.float32)
    for i, t in enumerate(trajs):
        padded[i, :len(t)] = t
    return padded, np.asarray(lengths, dtype=np.int64)


def _firl_reward_loss_rkl(traj_states: torch.Tensor, lengths: torch.Tensor,
                          disc: 'StateDiscriminator', reward_net: 'MLPReward') -> torch.Tensor:
    N, T_max, d = traj_states.shape
    device = traj_states.device
    mask = (torch.arange(T_max, device=device).unsqueeze(0)
            < lengths.unsqueeze(1)).float()
    s_flat = traj_states.reshape(-1, d)
    with torch.no_grad():
        log_ratio = disc.log_density_ratio(s_flat).reshape(N, T_max)
    t1 = ((-log_ratio) * mask).sum(dim=1)
    r = reward_net(s_flat)
    r = r.squeeze(-1) if r.dim() == 2 else r
    r = r.reshape(N, T_max) * mask
    t2 = r.sum(dim=1)
    T_bar = lengths.float().mean().clamp_min(1.0)
    return ((t1 * t2).mean() - t1.mean() * t2.mean()) / T_bar


def train_firl(env_id: str, K: int, seed: int, num_q_pairs: int,
               cfg=None, overwrite: bool = False) -> dict:
    cfg = cfg or FIRLConfig()
    set_seed(seed)
    meta = _ENV_META[env_id]
    action_range = float(meta['action_range'])

    algo_tag = f'firl_q{num_q_pairs}'
    out_dir = RESULTS_DIR / algo_tag / f'{env_id}_K{K}_seed{seed}'
    if out_dir.exists():
        if overwrite:
            shutil.rmtree(out_dir)
        elif (out_dir / 'learning_curve.npz').exists():
            print(f'[skip] {out_dir} already has results -- pass overwrite=True to redo')
            return {'skipped': True, 'out_path': str(out_dir / 'learning_curve.npz')}
    out_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    expert_data = np.load(DATASETS_DIR / f'expert_{env_id}_K{K_MAX}.npz')
    starts = np.concatenate([[0], np.cumsum(expert_data['lengths'])[:-1]])
    expert_idx = np.concatenate([np.arange(starts[i], starts[i] + expert_data['lengths'][i]) for i in range(K)])
    expert_states_t = torch.as_tensor(expert_data['obs'][expert_idx], dtype=torch.float32, device=DEVICE)

    reward_net = MLPReward(obs_dim).to(DEVICE)
    disc = StateDiscriminator(obs_dim).to(DEVICE)

    reward_opt = Adam(reward_net.parameters(), lr=cfg.reward_lr)
    disc_opt = Adam(disc.parameters(), lr=cfg.disc_lr)

    agent = FIRLSACAgent(
        obs_dim=obs_dim, action_dim=action_dim, action_range=action_range,
        device=DEVICE, reward_net=reward_net,
        num_q_pairs=num_q_pairs,
        uncertainty_coef=cfg.uncertainty_coef, q_std_clip=cfg.q_std_clip,
    )
    replay = ReplayBuffer(cfg.replay_capacity, obs_dim, action_dim, DEVICE)

    obs, _ = env.reset(seed=seed)
    total_env_steps = 0

    eval_steps, eval_rewards = [], []
    best_eval = -float('inf')
    best_actor_state = None
    best_reward_state = None
    t0 = time.time()
    T_traj = env.spec.max_episode_steps or 200

    for outer in range(1, cfg.n_outer + 1):
        for _ in range(cfg.sac_steps_per_outer):
            total_env_steps += 1
            if total_env_steps <= cfg.sac_warmup_steps:
                env_action = env.action_space.sample().astype(np.float32)
            else:
                env_action = agent.choose_action(obs, deterministic=False)
            next_obs, _r_env, terminated, truncated, _ = env.step(env_action)
            terminated = bool(terminated) and meta['can_terminate']
            replay.add(obs, env_action / action_range, next_obs, terminated)
            if terminated or truncated:
                obs, _ = env.reset()
            else:
                obs = next_obs
            if len(replay) >= cfg.sac_batch_size and total_env_steps > cfg.sac_warmup_steps:
                agent.update(replay.sample(cfg.sac_batch_size))

        traj_np, traj_lengths = _collect_rollout_states(agent, env_id, cfg.n_traj_per_outer,
                                                        seed=seed + outer, max_steps=T_traj)
        traj_t = torch.as_tensor(traj_np, dtype=torch.float32, device=DEVICE)
        traj_lengths_t = torch.as_tensor(traj_lengths, dtype=torch.long, device=DEVICE)
        T_max = traj_t.shape[1]
        mask = (torch.arange(T_max, device=DEVICE).unsqueeze(0)
                < traj_lengths_t.unsqueeze(1))
        policy_states_flat = traj_t[mask]

        disc_loss, disc_acc = 0.0, 0.0
        for _ in range(cfg.disc_iters):
            l, a = disc.train_step(expert_states_t, policy_states_flat, disc_opt, cfg.disc_batch_size)
            disc_loss += l; disc_acc += a
        disc_loss /= max(cfg.disc_iters, 1); disc_acc /= max(cfg.disc_iters, 1)

        reward_loss_val = 0.0
        for _ in range(cfg.reward_steps):
            loss = _firl_reward_loss_rkl(traj_t, traj_lengths_t, disc, reward_net)
            reward_opt.zero_grad(set_to_none=True)
            loss.backward()
            reward_opt.step()
            reward_loss_val = float(loss.item())

        if outer % cfg.eval_every_n_outer == 0 or outer == cfg.n_outer:
            r = evaluate_policy_eval(agent, env_id, cfg.eval_episodes, seed)
            eval_steps.append(total_env_steps)
            eval_rewards.append(r)
            elapsed = time.time() - t0
            marker = '*' if r > best_eval else ' '
            print(f'  outer {outer:>3} | steps {total_env_steps:>6} | eval reward {r:8.2f} {marker} | {elapsed:6.1f}s')
            if r > best_eval:
                best_eval = r
                best_actor_state = {k: v.detach().cpu().clone() for k, v in agent.actor.state_dict().items()}
                best_reward_state = {k: v.detach().cpu().clone() for k, v in reward_net.state_dict().items()}

    env.close()

    if best_actor_state is not None:
        agent.actor.load_state_dict(best_actor_state)
    if best_reward_state is not None:
        reward_net.load_state_dict(best_reward_state)

    np.savez(out_dir / 'learning_curve.npz',
             eval_step=np.asarray(eval_steps), eval_reward=np.asarray(eval_rewards),
             env_id=env_id, K=K, seed=seed, num_q_pairs=num_q_pairs,
             best_eval=best_eval)
    torch.save(agent.actor.state_dict(), out_dir / 'actor.pt')
    torch.save(agent.critic.state_dict(), out_dir / 'critic.pt')
    torch.save(reward_net.state_dict(), out_dir / 'reward.pt')
    torch.save(disc.state_dict(), out_dir / 'disc.pt')
    return {
        'final_reward': eval_rewards[-1],
        'best_reward': best_eval,
        'out_path': str(out_dir / 'learning_curve.npz'),
        'agent': agent,
    }


def load_firl_agent(env_id: str, K: int, seed: int, num_q_pairs: int):
    meta = _ENV_META[env_id]
    env = gym.make(env_id)
    obs_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
    env.close()
    reward_net = MLPReward(obs_dim).to(DEVICE)
    agent = FIRLSACAgent(obs_dim=obs_dim, action_dim=action_dim,
                         action_range=float(meta['action_range']),
                         device=DEVICE, reward_net=reward_net, num_q_pairs=num_q_pairs)
    out_dir = RESULTS_DIR / f'firl_q{num_q_pairs}' / f'{env_id}_K{K}_seed{seed}'
    agent.actor.load_state_dict(torch.load(out_dir / 'actor.pt', map_location=DEVICE, weights_only=True))
    agent.critic.load_state_dict(torch.load(out_dir / 'critic.pt', map_location=DEVICE, weights_only=True))
    reward_net.load_state_dict(torch.load(out_dir / 'reward.pt', map_location=DEVICE, weights_only=True))
    agent.actor.eval(); agent.critic.eval()
    return agent


# ==============================================================================
# CODE cell 16
# ==============================================================================
K_VALUES = [5, 20, 100]
SEEDS = [0, 1, 2]

ENV_IDS = ['Pendulum-v1', 'MountainCarContinuous-v0']

ENV_BUDGETS = {
    'Pendulum-v1':              150_000,
    'MountainCarContinuous-v0':  50_000,
}
SAC_STEPS_PER_OUTER = 1_000


def _cfgs_for(env_id):
    total = ENV_BUDGETS[env_id]
    n_outer = total // SAC_STEPS_PER_OUTER
    iq = IQConfig(total_steps=total, eval_interval=5_000, eval_episodes=10)
    firl = FIRLConfig(n_outer=n_outer, sac_steps_per_outer=SAC_STEPS_PER_OUTER,
                      eval_every_n_outer=10, eval_episodes=10)
    return iq, firl

def run_full_sweep():
    """Run the entire 3 x 2 x 3 x 3 sweep. Each call writes to artifacts/results/<algo>/."""
    total = 3 * len(ENV_IDS) * len(K_VALUES) * len(SEEDS)
    done = 0
    sweep_t0 = time.time()
    for env_id in ENV_IDS:
        iq_cfg, firl_cfg = _cfgs_for(env_id)
        for K in K_VALUES:
            for seed in SEEDS:
                done += 1
                print(f'\n[{done}/{total}] IQ-Learn | {env_id} | K={K} | seed={seed} '
                      f'| elapsed {(time.time() - sweep_t0) / 60:.1f} min')
                train_iq_learn(env_id, K=K, seed=seed, cfg=iq_cfg)

                done += 1
                print(f'\n[{done}/{total}] f-IRL base | {env_id} | K={K} | seed={seed} '
                      f'| elapsed {(time.time() - sweep_t0) / 60:.1f} min')
                train_firl(env_id, K=K, seed=seed, num_q_pairs=1, cfg=firl_cfg)

                done += 1
                print(f'\n[{done}/{total}] f-IRL + SOAR | {env_id} | K={K} | seed={seed} '
                      f'| elapsed {(time.time() - sweep_t0) / 60:.1f} min')
                train_firl(env_id, K=K, seed=seed, num_q_pairs=4, cfg=firl_cfg)

    print(f'\n=== Sweep finished in {(time.time() - sweep_t0) / 60:.1f} min ===')

run_full_sweep()

# ==============================================================================
# CODE cell 17
# ==============================================================================
import glob
from functools import lru_cache
import pandas as pd


_ALGOS = [
    ('iq_learn',  'IQ-Learn',     'C0'),
    ('firl_q1',   'f-IRL',        'C1'),
    ('firl_q4',   'f-IRL + SOAR', 'C2'),
]


@lru_cache(maxsize=None)
def _expert_random_baselines(env_id: str) -> tuple:
    data = np.load(DATASETS_DIR / f'expert_{env_id}_K{K_MAX}.npz')
    lengths, rew = data['lengths'], data['rew']
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    expert_returns = np.array([rew[s:s + l].sum() for s, l in zip(starts, lengths)])
    expert_mean = float(expert_returns.mean())

    env = gym.make(env_id)
    rets = []
    rng = np.random.default_rng(0)
    for _ in range(10):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        ret, done = 0.0, False
        while not done:
            obs, r, terminated, truncated, _ = env.step(env.action_space.sample())
            ret += float(r); done = terminated or truncated
        rets.append(ret)
    env.close()
    return expert_mean, float(np.mean(rets))


def load_curves(algo: str, env_id: str, K: int):
    pattern = str(RESULTS_DIR / algo / f'{env_id}_K{K}_seed*' / 'learning_curve.npz')
    rewards, steps = [], None
    for path in sorted(glob.glob(pattern)):
        d = np.load(path)
        rewards.append(d['eval_reward'])
        if steps is None:
            steps = d['eval_step']
    if not rewards:
        return None, None, None
    L = min(len(r) for r in rewards)
    arr = np.stack([r[:L] for r in rewards])
    return steps[:L], arr.mean(0), arr.std(0)


def _panel_has_data(env_id: str, K: int) -> bool:
    for algo, _, _ in _ALGOS:
        if load_curves(algo, env_id, K)[1] is not None:
            return True
    return False


def plot_learning_curves(env_id: str, K: int, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    expert_r, random_r = _expert_random_baselines(env_id)
    any_data = False
    for algo, label, color in _ALGOS:
        x, mean, std = load_curves(algo, env_id, K)
        if mean is None: continue
        any_data = True
        if x[0] > 0:
            x = np.concatenate([[0], x])
            mean = np.concatenate([[random_r], mean])
            std = np.concatenate([[0.0], std])
        ax.plot(x, mean, label=label, color=color, lw=1.8)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)
    ax.axhline(expert_r, ls='--', color='black', lw=1.0, label=f'PPO expert ({expert_r:.0f})')
    ax.axhline(random_r, ls=':',  color='gray',  lw=1.0, label=f'random ({random_r:.0f})')
    ax.set_xlabel('Env steps'); ax.set_ylabel('Eval reward')
    ax.set_title(f'{env_id}, K={K}' + ('' if any_data else '  (no data)'))
    if any_data:
        ax.legend(loc='lower right', fontsize=8)
    else:
        ax.text(0.5, 0.5, f'No results yet for K={K}\nrun train_iq_learn / train_firl with this K',
                ha='center', va='center', transform=ax.transAxes,
                color='gray', fontsize=10, fontstyle='italic')
    ax.grid(alpha=0.3)


def plot_dataset_size_effect(env_id: str, K_values=(5, 20, 100), ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    expert_r, _ = _expert_random_baselines(env_id)
    any_data = False
    for algo, label, color in _ALGOS:
        means, stds, ks = [], [], []
        for K in K_values:
            _, m, s = load_curves(algo, env_id, K)
            if m is None: continue
            means.append(m[-1]); stds.append(s[-1]); ks.append(K)
        if means:
            any_data = True
            ax.errorbar(ks, means, yerr=stds, label=label, color=color, marker='o', capsize=3)
    ax.axhline(expert_r, ls='--', color='black', lw=1.0, label=f'PPO expert ({expert_r:.0f})')
    ax.set_xscale('log'); ax.set_xticks(K_values); ax.set_xticklabels([str(k) for k in K_values])
    ax.set_xlabel('Expert trajectories K'); ax.set_ylabel('Final eval reward')
    ax.set_title(f'{env_id}: dataset-size effect' + ('' if any_data else '  (no data)'))
    if any_data:
        ax.legend(loc='lower right', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No multi-K results yet', ha='center', va='center',
                transform=ax.transAxes, color='gray', fontsize=10, fontstyle='italic')
    ax.grid(alpha=0.3)


def summary_table(K_values=(5, 20, 100)) -> pd.DataFrame:
    """Final reward (mean ± std over seeds) for every (algo, env, K) cell."""
    rows = []
    for env_id in ENV_IDS:
        for K in K_values:
            row = {'env': env_id, 'K': K}
            for algo, label, _ in _ALGOS:
                _, mean, std = load_curves(algo, env_id, K)
                row[label] = '—' if mean is None else f'{mean[-1]:7.1f} ± {std[-1]:5.1f}'
            rows.append(row)
    return pd.DataFrame(rows).set_index(['env', 'K'])

_K_PLOT = (5, 20, 100)
compact = False

if compact:
    panels = [(env_id, K) for env_id in ENV_IDS for K in _K_PLOT if _panel_has_data(env_id, K)]
    if not panels:
        print('No results on disk yet. Call `run_full_sweep()` in Section 5.5 first.')
    else:
        ncols = min(3, len(panels))
        nrows = (len(panels) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
        for i, (env_id, K) in enumerate(panels):
            plot_learning_curves(env_id, K, ax=axes[i // ncols, i % ncols])
        for j in range(len(panels), nrows * ncols):
            axes[j // ncols, j % ncols].axis('off')
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / 'learning_curves.png', dpi=150, bbox_inches='tight')
        plt.show()
        if len(panels) < len(ENV_IDS) * len(_K_PLOT):
            print(f'Showing {len(panels)} / {len(ENV_IDS) * len(_K_PLOT)} panels with data. '
                  'To see them all (with empty placeholders), set `compact = False` above.')
else:
    fig, axes = plt.subplots(len(ENV_IDS), len(_K_PLOT), figsize=(18, 8), squeeze=False)
    for i, env_id in enumerate(ENV_IDS):
        for j, K in enumerate(_K_PLOT):
            plot_learning_curves(env_id, K, ax=axes[i, j])
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'learning_curves.png', dpi=150, bbox_inches='tight')
    plt.show()

# ==============================================================================
# CODE cell 18
# ==============================================================================
fig, axes = plt.subplots(1, len(ENV_IDS), figsize=(13, 4))
for ax, env_id in zip(axes, ENV_IDS):
    plot_dataset_size_effect(env_id, K_values=_K_PLOT, ax=ax)
fig.tight_layout()
plt.savefig(RESULTS_DIR / 'dataset_size_effect.png', dpi=150, bbox_inches='tight')
plt.show()

# ==============================================================================
# CODE cell 19
# ==============================================================================
_STATE_DIMS = {
    'Pendulum-v1':              [('angle',     lambda s: np.arctan2(s[:, 1], s[:, 0])),
                                 ('ang. vel.', lambda s: s[:, 2])],
    'MountainCarContinuous-v0': [('position',  lambda s: s[:, 0]),
                                 ('velocity',  lambda s: s[:, 1])],
}


def _best_run(algo: str, env_id: str):
    best = None
    for K in K_VALUES:
        for seed in SEEDS:
            npz = RESULTS_DIR / algo / f'{env_id}_K{K}_seed{seed}' / 'learning_curve.npz'
            if not npz.exists():
                continue
            d = np.load(npz)
            score = float(d['best_eval']) if 'best_eval' in d.files else float(d['eval_reward'][-1])
            if best is None or score > best[2]:
                best = (int(K), int(seed), score)
    return best


def _rollout_states(env_id: str, agent, seed: int) -> np.ndarray:
    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)
    states = [obs.astype(np.float32)]
    done = False
    while not done:
        action = agent.choose_action(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        states.append(obs.astype(np.float32))
        done = bool(terminated or truncated)
    env.close()
    return np.stack(states)


def _expert_trajectory(env_id: str) -> np.ndarray:
    d = np.load(DATASETS_DIR / f'expert_{env_id}_K{K_MAX}.npz')
    T = int(d['lengths'][0])
    return d['obs'][:T].astype(np.float32)


_LOADERS = {
    'iq_learn': lambda env_id, K, seed: load_iq_agent(env_id, K=K, seed=seed),
    'firl_q1':  lambda env_id, K, seed: load_firl_agent(env_id, K=K, seed=seed, num_q_pairs=1),
    'firl_q4':  lambda env_id, K, seed: load_firl_agent(env_id, K=K, seed=seed, num_q_pairs=4),
}


fig, axes = plt.subplots(len(ENV_IDS), 2, figsize=(13, 7))
for row, env_id in enumerate(ENV_IDS):
    expert_traj = _expert_trajectory(env_id)
    rollouts = {}
    for algo, label, _color in _ALGOS:
        best = _best_run(algo, env_id)
        if best is None:
            print(f'[skip] {label} on {env_id}: no results saved')
            continue
        K, seed, score = best
        agent = _LOADERS[algo](env_id, K, seed)
        states = _rollout_states(env_id, agent, seed=seed + 99_999)
        rollouts[algo] = (states, f'{label} (K={K}, seed={seed}, r={score:.0f})')

    for col, (name, extract) in enumerate(_STATE_DIMS[env_id]):
        ax = axes[row, col]
        ax.plot(extract(expert_traj), color='k', lw=2.0, label='Expert', zorder=5)
        for algo, label, color in _ALGOS:
            if algo not in rollouts:
                continue
            states, full_label = rollouts[algo]
            ax.plot(extract(states), color=color, alpha=0.85, label=full_label)
        ax.set_xlabel('env step')
        ax.set_ylabel(name)
        ax.set_title(f'{env_id} — {name}')
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(fontsize=8, loc='best')

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'trajectory_comparison.png', dpi=150, bbox_inches='tight')
plt.show()


# ==============================================================================
# CODE cell 20
# ==============================================================================
_EXPERT_MASTER_SEED = 42

def _expert_reset_seeds(n: int = 100) -> list[int]:
    """Reproduce the per-trajectory reset seeds used when collecting the expert dataset."""
    rng = np.random.default_rng(_EXPERT_MASTER_SEED)
    return [int(rng.integers(0, 2**31 - 1)) for _ in range(n)]


def _best_seed_for(algo: str, env_id: str, K: int):
    """Return (seed, score) for the best-scoring run of this (algo, env, K), or None."""
    best = None
    for seed in SEEDS:
        npz = RESULTS_DIR / algo / f'{env_id}_K{K}_seed{seed}' / 'learning_curve.npz'
        if not npz.exists():
            continue
        d = np.load(npz)
        score = float(d['best_eval']) if 'best_eval' in d.files else float(d['eval_reward'][-1])
        if best is None or score > best[1]:
            best = (int(seed), score)
    return best


def _rollout_from_seed(env_id: str, agent, reset_seed: int) -> np.ndarray:
    env = gym.make(env_id)
    obs, _ = env.reset(seed=reset_seed)
    states = [obs.astype(np.float32)]
    done = False
    while not done:
        action = agent.choose_action(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        states.append(obs.astype(np.float32))
        done = bool(terminated or truncated)
    env.close()
    return np.stack(states)


# Use one dim per env that captures the task-relevant behavior.
_PRIMARY_DIM = {
    'Pendulum-v1':              ('angle θ',  lambda s: np.arctan2(s[:, 1], s[:, 0])),
    'MountainCarContinuous-v0': ('position', lambda s: s[:, 0]),
}

_XLIM = {
    'Pendulum-v1':              100,
    'MountainCarContinuous-v0': 400,
}

fig, axes = plt.subplots(len(ENV_IDS), len(K_VALUES), figsize=(14, 7), sharex='row')
reset_seeds = _expert_reset_seeds(K_MAX)
expert_reset_seed = reset_seeds[0]

for row, env_id in enumerate(ENV_IDS):
    label, extract = _PRIMARY_DIM[env_id]

    expert_data = np.load(DATASETS_DIR / f'expert_{env_id}_K{K_MAX}.npz')
    T0 = int(expert_data['lengths'][0])
    expert_traj = expert_data['obs'][:T0].astype(np.float32)

    for col, K in enumerate(K_VALUES):
        ax = axes[row, col]
        ax.plot(extract(expert_traj), color='k', lw=2.0, label='Expert', zorder=5)

        for algo, algo_label, color in _ALGOS:
            best = _best_seed_for(algo, env_id, K)
            if best is None:
                continue
            seed, score = best
            agent = _LOADERS[algo](env_id, K, seed)
            states = _rollout_from_seed(env_id, agent, expert_reset_seed)
            ax.plot(extract(states), color=color, alpha=0.85,
                    label=f'{algo_label} (seed={seed}, r={score:.0f})')

        ax.set_xlim(0, _XLIM[env_id])
        ax.set_title(f'{env_id} — K={K}')
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        if row == len(ENV_IDS) - 1:
            ax.set_xlabel('env step')
        if row == 0 and col == 0:
            ax.legend(fontsize=8, loc='best')

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'trajectory_grid_by_K.png', dpi=150, bbox_inches='tight')
plt.show()

# ==============================================================================
# CODE cell 21
# ==============================================================================
_expert_reset_seed_0 = _expert_reset_seeds(K_MAX)[0]


def show_policy_video(env_id: str, algo: str, K: int, seed: int,
                      reset_seed: int = _expert_reset_seed_0,
                      max_steps: int = 300, fps: int = 30):
    agent = _LOADERS[algo](env_id, K, seed)
    policy_fn = lambda obs: agent.choose_action(obs, deterministic=True)
    frames, total_r = render_policy(policy_fn, env_id, max_steps=max_steps, seed=reset_seed)
    print(f'{env_id} | {algo} | K={K} seed={seed} | reset_seed={reset_seed} | '
          f'total reward = {total_r:.2f} ({len(frames)} frames)')
    display(show_video(frames, fps=fps))


# for env_id in ENV_IDS:
#     for K in K_VALUES:
#         for algo, label, _color in _ALGOS:
#             best = _best_seed_for(algo, env_id, K)
#             if best is None:
#                 print(f'[skip] {label} on {env_id} K={K}: no results saved')
#                 continue
#             seed, score = best
#             print(f'--- {label} on {env_id}, K={K} (seed={seed}, r={score:.0f}) ---')
#             show_policy_video(env_id, algo, K, seed)

# ==============================================================================
# CODE cell 22
# ==============================================================================
K_LIST = [5, 20, 100]
ALGOS_REWARD = [
    ('iq_learn', 'IQ-Learn (V)'),
    ('firl_q1',  'f-IRL (r̂)'),
    ('firl_q4',  'f-IRL+SOAR (r̂)'),
]


def _learned_reward_grid(env_id: str, n: int = 80):
    if env_id == 'Pendulum-v1':
        thetas = np.linspace(-np.pi, np.pi, n)
        thetadots = np.linspace(-8.0, 8.0, n)
        TH, THD = np.meshgrid(thetas, thetadots)
        obs_grid = np.stack([np.cos(TH), np.sin(TH), THD], axis=-1).astype(np.float32)
        return thetas, thetadots, obs_grid, 'angle θ', 'ang. vel. θ̇'
    positions = np.linspace(-1.2, 0.6, n)
    velocities = np.linspace(-0.07, 0.07, n)
    P, V = np.meshgrid(positions, velocities)
    obs_grid = np.stack([P, V], axis=-1).astype(np.float32)
    return positions, velocities, obs_grid, 'position', 'velocity'


def _expert_states_2d(env_id: str, n_samples: int = 3000):
    d = np.load(DATASETS_DIR / f'expert_{env_id}_K{K_MAX}.npz')
    s = d['obs']
    if env_id == 'Pendulum-v1':
        xs, ys = np.arctan2(s[:, 1], s[:, 0]), s[:, 2]
    else:
        xs, ys = s[:, 0], s[:, 1]
    n = min(n_samples, len(s))
    idx = np.random.default_rng(0).choice(len(s), n, replace=False)
    return xs[idx], ys[idx]


def _compute_landscape(algo: str, env_id: str, K: int, seed: int):
    """Return (xg, yg, grid_values, xl, yl, cbar_label). grid_values is 2D."""
    obs_dim = gym.make(env_id).observation_space.shape[0]
    xg, yg, obs_grid, xl, yl = _learned_reward_grid(env_id)
    flat = torch.as_tensor(obs_grid.reshape(-1, obs_grid.shape[-1]), device=DEVICE)
    out_dir = RESULTS_DIR / algo / f'{env_id}_K{K}_seed{seed}'
    with torch.no_grad():
        if algo == 'iq_learn':
            agent = load_iq_agent(env_id, K, seed)
            v = agent.getV(flat).cpu().numpy().reshape(len(yg), len(xg))
            return xg, yg, v, xl, yl, 'V(s)'
        else:
            reward_net = MLPReward(obs_dim).to(DEVICE)
            reward_net.load_state_dict(torch.load(out_dir / 'reward.pt', map_location=DEVICE, weights_only=True))
            reward_net.eval()
            r = reward_net(flat).cpu().numpy().reshape(len(yg), len(xg))
            return xg, yg, r, xl, yl, 'r̂(s)'


for env_id in ENV_IDS:
    fig, axes = plt.subplots(len(K_LIST), len(ALGOS_REWARD),
                             figsize=(5 * len(ALGOS_REWARD), 4 * len(K_LIST)),
                             squeeze=False)
    ex, ey = _expert_states_2d(env_id)
    for row, K in enumerate(K_LIST):
        for col, (algo, label) in enumerate(ALGOS_REWARD):
            ax = axes[row, col]
            best = _best_seed_for(algo, env_id, K)
            if best is None:
                ax.set_title(f'{label} | K={K} — no data'); ax.axis('off'); continue
            seed, score = best
            try:
                xg, yg, grid, xl, yl, cb_label = _compute_landscape(algo, env_id, K, seed)
            except Exception as e:
                ax.set_title(f'{label} | K={K} — error: {type(e).__name__}'); ax.axis('off'); continue
            im = ax.pcolormesh(xg, yg, grid, cmap='viridis', shading='auto')
            ax.scatter(ex, ey, c='white', s=2, alpha=0.30)
            plt.colorbar(im, ax=ax, label=cb_label)
            ax.set_xlabel(xl); ax.set_ylabel(yl)
            ax.set_title(f'{label} | K={K} | seed={seed}, r={score:.0f}')
    fig.suptitle(f'{env_id}: learned reward (f-IRL/SOAR) and value (IQ-Learn) per K', y=1.00, fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f'learned_reward_heatmap_{env_id}.png', dpi=140, bbox_inches='tight')
    plt.show()

# ==============================================================================
# CODE cell 23
# ==============================================================================
K_VIZ = 100             # the K at which the f-IRL/SOAR collapse is visible

def _collect_policy_states(env_id: str, policy, n_episodes: int = 30, seed: int = 0):
    env = gym.make(env_id)
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = False
        while not done:
            if policy == 'random':
                action = env.action_space.sample()
            else:
                action = policy.choose_action(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            states.append(obs)
            done = bool(terminated or truncated)
    env.close()
    return np.stack(states)


def _project_2d(env_id, states):
    if env_id == 'Pendulum-v1':
        return np.arctan2(states[:, 1], states[:, 0]), states[:, 2]
    return states[:, 0], states[:, 1]


def _axis_meta(env_id):
    if env_id == 'Pendulum-v1':
        return (-np.pi, np.pi), (-8, 8), 'angle θ', 'ang. vel. θ̇'
    return (-1.2, 0.6), (-0.07, 0.07), 'position', 'velocity'


_cols = [
    ('Expert',     None,        'expert'),
    ('IQ-Learn',   'iq_learn',  None),
    ('f-IRL',      'firl_q1',   None),
    ('f-IRL+SOAR', 'firl_q4',   None),
    ('Random',     None,        'random'),
]

fig, axes = plt.subplots(len(ENV_IDS), len(_cols), figsize=(4 * len(_cols), 6.5))
for row, env_id in enumerate(ENV_IDS):
    xlim, ylim, xl, yl = _axis_meta(env_id)
    for col, (title, algo, kind) in enumerate(_cols):
        ax = axes[row, col]
        if kind == 'expert':
            d = np.load(DATASETS_DIR / f'expert_{env_id}_K{K_MAX}.npz')
            states = d['obs']
            badge = ''
        elif kind == 'random':
            states = _collect_policy_states(env_id, 'random', n_episodes=30, seed=0)
            badge = ''
        else:
            best = _best_seed_for(algo, env_id, K_VIZ)
            if best is None:
                ax.set_title(f'{title} — K={K_VIZ}, no data')
                continue
            seed, score = best
            agent = _LOADERS[algo](env_id, K_VIZ, seed)
            states = _collect_policy_states(env_id, agent, n_episodes=30, seed=seed + 999)
            badge = f' K={K_VIZ}, r={score:.0f}'

        xs, ys = _project_2d(env_id, states)
        ax.hexbin(xs, ys, gridsize=40, cmap='viridis', mincnt=1, extent=(*xlim, *ylim))
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        if row == len(ENV_IDS) - 1:
            ax.set_xlabel(xl)
        if col == 0:
            env_short = {'Pendulum-v1': 'Pendulum', 'MountainCarContinuous-v0': 'MCC-v0'}.get(env_id, env_id)
            ax.set_ylabel(f'{env_short}\n{yl}')
        header = f'{title}{badge}'
        ax.set_title(header)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'state_visitation.png', dpi=150, bbox_inches='tight')
plt.show()
