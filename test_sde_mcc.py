"""
Ablation: PPO on MountainCarContinuous-v0 with vs without gSDE.

Trains two PPO experts (same hyperparameters except `use_sde`), saves weights
in a dedicated folder, records a rollout video for each, and plots the training
curves (episode reward vs timesteps) for comparison.

Run: /home/mattia/anaconda3/bin/python3 test_sde_mcc.py
"""

# ruff: noqa

import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib
matplotlib.use('Agg', force=True)

import warnings
from pathlib import Path

warnings.filterwarnings('ignore', message='.*Gym has been unmaintained.*')

import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
import imageio

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

PROJECT_DIR = Path('.').resolve()
OUT_DIR = PROJECT_DIR / 'artifacts' / 'sde_ablation'
WEIGHTS_DIR = OUT_DIR / 'weights'
VIDEOS_DIR = OUT_DIR / 'videos'
PLOTS_DIR = OUT_DIR / 'plots'
for d in (WEIGHTS_DIR, VIDEOS_DIR, PLOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ENV_ID = 'MountainCarContinuous-v0'
TOTAL_TIMESTEPS = 500_000
SEED = 0

BASE_KWARGS = dict(
    learning_rate=7e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.9999,
    gae_lambda=0.9,
)
SDE_EXTRA = dict(use_sde=True, sde_sample_freq=4)


class EpisodeRewardLogger(BaseCallback):
    """Records per-episode reward and the timestep at which the episode ended."""

    def __init__(self):
        super().__init__()
        self.timesteps = []
        self.rewards = []

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            ep = info.get('episode')
            if ep is not None:
                self.timesteps.append(self.num_timesteps)
                self.rewards.append(ep['r'])
        return True


def train(label: str, extra_kwargs: dict):
    save_path = WEIGHTS_DIR / f'ppo_{label}.zip'
    curves_path = WEIGHTS_DIR / f'ppo_{label}_curves.npz'

    if save_path.exists():
        print(f'\n[load] {label} — found {save_path}, skipping training')
        model = PPO.load(str(save_path), device=DEVICE)
        if curves_path.exists():
            data = np.load(curves_path)
            ts, rew = data['timesteps'], data['rewards']
        else:
            ts, rew = np.array([]), np.array([])
    else:
        print(f'\n[train] {label} — {TOTAL_TIMESTEPS:,} steps')
        env = make_vec_env(ENV_ID, n_envs=4, seed=SEED)
        kwargs = {**BASE_KWARGS, **extra_kwargs}
        model = PPO('MlpPolicy', env, verbose=0, seed=SEED, device=DEVICE, **kwargs)
        cb = EpisodeRewardLogger()
        model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True, callback=cb)
        model.save(str(save_path))
        ts, rew = np.array(cb.timesteps), np.array(cb.rewards)
        np.savez(curves_path, timesteps=ts, rewards=rew)

    eval_env = gym.make(ENV_ID)
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=20)
    eval_env.close()
    print(f'  -> eval reward (20 ep): {mean_r:.2f} ± {std_r:.2f}')

    return model, ts, rew, (mean_r, std_r)


def record_video(model, label: str, n_episodes: int = 2):
    env = gym.make(ENV_ID, render_mode='rgb_array')
    frames = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=SEED + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            frames.append(env.render())
            done = term or trunc
    env.close()
    out_path = VIDEOS_DIR / f'{label}.gif'
    imageio.mimsave(str(out_path), frames, fps=30)
    print(f'  -> video saved: {out_path}')


def smooth(x, window=20):
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='valid')


def plot_curves(curves: dict):
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, (ts, rew) in curves.items():
        if len(rew) == 0:
            continue
        ax.plot(ts, rew, alpha=0.25)
        sm = smooth(rew, window=20)
        ts_sm = ts[len(ts) - len(sm):]
        ax.plot(ts_sm, sm, label=label, linewidth=2)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Episode reward')
    ax.set_title(f'PPO on {ENV_ID} — SDE ablation')
    ax.legend()
    ax.grid(alpha=0.3)
    out_path = PLOTS_DIR / 'training_curves.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'\n[plot] saved: {out_path}')


def main():
    curves = {}
    results = {}
    for label, extra in [('no_sde', {}), ('sde', SDE_EXTRA)]:
        model, ts, rew, (mean_r, std_r) = train(label, extra)
        curves[label] = (ts, rew)
        results[label] = (mean_r, std_r)
        record_video(model, label)

    if any(len(rew) > 0 for _, rew in curves.values()):
        plot_curves(curves)
    else:
        print('\n[plot] skipped (no training curves — models were loaded from disk)')

    print('\n=== Summary ===')
    for label, (m, s) in results.items():
        print(f'  {label:8s}: {m:.2f} ± {s:.2f}')


if __name__ == '__main__':
    main()
