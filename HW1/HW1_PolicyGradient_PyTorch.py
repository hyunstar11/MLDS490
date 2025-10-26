"""
HW1 — Policy Gradient (PyTorch) starter tailored to 2025 spec

Covers:
- Part 1: REINFORCE for CartPole-v1 (gamma=0.95) and Pong-v5 (gamma=0.99, actions [RIGHT, LEFT] only)
- Part 2: Baseline variants (none | moving_avg | value)
- Required plots: reward per episode + 100-ep moving average, and 500-episode rollout histogram
- Final report stats: mean and std of the 500 rollout returns

Usage examples:
# CartPole, no baseline
python HW1_PolicyGradient_PyTorch.py --env cartpole --episodes 1500 --gamma 0.95 --baseline none

# Pong, moving-average baseline
python HW1_PolicyGradient_PyTorch.py --env pong --episodes 2000 --gamma 0.99 --baseline moving_avg

# Pong, value-function baseline (actor-critic style advantage)
python HW1_PolicyGradient_PyTorch.py --env pong --episodes 2000 --gamma 0.99 --baseline value

Notes:
- Install deps:
    pip install torch gymnasium matplotlib numpy
    # For Pong (ALE):
    pip install "gymnasium[atari]" "gymnasium[accept-rom-license]"
- The assignment specifies versions CartPole-v1 and Pong-v5. In Gymnasium, Pong is typically "ALE/Pong-v5".
- For Pong we restrict actions to [RIGHT, LEFT] which correspond to ALE action ids [2, 3].
- Plots are saved under ./plots and models under ./checkpoints
"""

import argparse
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

try:
    import gymnasium as gym
except ImportError:
    import gym as gym  # fallback if needed, but Gymnasium is recommended

# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs():
    os.makedirs("plots", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)


def discount_cumsum(rewards, gamma: float):
    """Compute discounted returns G_t = r_t + gamma * r_{t+1} + ..."""
    G = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        G[t] = running
    return G


def moving_average(x, k=100):
    if len(x) < 1:
        return np.array([])
    ma = []
    q = deque(maxlen=k)
    for v in x:
        q.append(v)
        ma.append(np.mean(q))
    return np.array(ma)

# -----------------------------
# Pong preprocessing (per spec)
# -----------------------------

def preprocess_pong(image: np.ndarray) -> np.ndarray:
    """prepro 210x160x3 uint8 frame into 80x80 float32 array (values in {0,1})."""
    # Crop and downsample
    image = image[35:195]
    image = image[::2, ::2, 0]
    # Erase background
    image[image == 144] = 0
    image[image == 109] = 0
    # Set paddles and ball to 1
    image[image != 0] = 1
    # Return 80x80 array
    return image.astype(np.float32)

# -----------------------------
# Policy / Value Networks
# -----------------------------

class MLPPolicy(nn.Module):
    """Simple MLP policy for low-dim state spaces (e.g., CartPole)."""
    def __init__(self, state_dim: int, hidden: int = 128, action_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x):
        logits = self.net(x)
        return logits


class ConvlessPongPolicy(nn.Module):
    """Simple MLP over flattened 80x80 images (2 actions: RIGHT or LEFT)."""
    def __init__(self, hidden: int = 200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(80 * 80, hidden), nn.ReLU(),
            nn.Linear(hidden, 2)  # logits for RIGHT vs LEFT
        )

    def forward(self, x):
        logits = self.net(x)
        return logits


class ValueNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# -----------------------------
# Agent
# -----------------------------

class REINFORCEAgent:
    def __init__(self, env_name: str, gamma: float, lr: float, baseline: str, device: str):
        self.env_name = env_name
        self.gamma = gamma
        self.lr = lr
        self.baseline = baseline  # 'none' | 'moving_avg' | 'value'
        self.device = device

        if env_name == 'cartpole':
            self.env = gym.make('CartPole-v1')
            obs, _ = self.env.reset()
            self.state_dim = obs.shape[0]
            self.policy = MLPPolicy(self.state_dim, hidden=128, action_dim=2).to(device)
            self.value = ValueNet(self.state_dim, hidden=128).to(device) if baseline == 'value' else None
            self.action_head = lambda logits: torch.distributions.Categorical(logits=logits)
            self.pong = False

        elif env_name == 'pong':
            # ALE Pong v5 in Gymnasium
            # If "ALE/Pong-v5" fails, you may try "Pong-v5" depending on your installation.
            try:
                self.env = gym.make('ALE/Pong-v5', obs_type='rgb')
            except Exception:
                self.env = gym.make('Pong-v5', obs_type='rgb')

            obs, _ = self.env.reset()
            # We'll use 80x80 after preprocessing and then flatten
            self.policy = ConvlessPongPolicy(hidden=200).to(device)
            self.value = ValueNet(80 * 80, hidden=256).to(device) if baseline == 'value' else None
            self.action_head = lambda logits: torch.distributions.Categorical(logits=logits)
            self.pong = True
        else:
            raise ValueError("env must be one of {'cartpole','pong'}")

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=self.lr) if self.value is not None else None

        self.running_baseline = 0.0  # for moving_avg baseline
        self.baseline_beta = 0.99

    def _obs_to_tensor(self, obs):
        if self.pong:
            x = preprocess_pong(obs).reshape(1, -1).astype(np.float32)
            x = torch.tensor(x, dtype=torch.float32)  # [1, 6400]
            return x.to(self.device)
        else:
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            return x.to(self.device)

    def _select_action(self, obs_tensor):
        logits = self.policy(obs_tensor)
        dist = self.action_head(logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        if self.pong:
            # Map 0->RIGHT (2), 1->LEFT (3)
            ale_action = 2 if action.item() == 0 else 3
            return ale_action, logp
        else:
            return action.item(), logp

    def run_episode(self, render: bool = False, train: bool = True):
        obs, _ = self.env.reset()
        done = False
        rewards = []
        logps = []
        states = []  # for value baseline

        ep_reward = 0.0
        while not done:
            if render:
                self.env.render()
            s = self._obs_to_tensor(obs)
            action, logp = self._select_action(s)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            rewards.append(reward)
            logps.append(logp)
            if self.value is not None:
                states.append(s)
            ep_reward += reward
            obs = next_obs

        if train:
            self.update_policy(rewards, logps, states)

        return ep_reward

    def update_policy(self, rewards, logps, states):
        G = discount_cumsum(rewards, self.gamma)
        G_t = torch.tensor(G, dtype=torch.float32, device=self.device)

        if self.baseline == 'moving_avg':
            # Exponential moving average of returns
            batch_mean = G_t.mean().item()
            self.running_baseline = self.baseline_beta * self.running_baseline + (1 - self.baseline_beta) * batch_mean
            b = torch.full_like(G_t, fill_value=self.running_baseline)
            advantages = G_t - b
            value_loss = None

        elif self.baseline == 'value' and self.value is not None:
            # Train value network to predict returns
            with torch.no_grad():
                states_cat = torch.cat(states, dim=0)
            values = self.value(states_cat)
            # Fit V(s) to G_t
            v_loss = nn.MSELoss()(values, G_t)
            self.value_optimizer.zero_grad()
            v_loss.backward()
            self.value_optimizer.step()
            # Advantages
            with torch.no_grad():
                values_detached = self.value(states_cat)
            advantages = G_t - values_detached
            value_loss = v_loss.item()
        else:
            advantages = G_t
            value_loss = None

        # Policy loss
        logps_t = torch.stack(logps)
        policy_loss = -(logps_t * advantages.detach()).sum()
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        return policy_loss.item(), value_loss

    def train(self, episodes: int, render: bool = False):
        rewards = []
        for ep in range(1, episodes + 1):
            r = self.run_episode(render=render, train=True)
            rewards.append(r)
            if ep % 10 == 0:
                print(f"Episode {ep:5d} | Return: {r:.2f} | Avg(100): {np.mean(rewards[-100:]):.2f}")
        return rewards

    def rollout(self, episodes: int, deterministic: bool = False):
        # For simplicity, use sampling as in training; deterministic=True uses argmax at action selection.
        returns = []
        for _ in range(episodes):
            obs, _ = self.env.reset()
            done = False
            ep_r = 0.0
            while not done:
                s = self._obs_to_tensor(obs)
                logits = self.policy(s)
                if deterministic:
                    act_idx = torch.argmax(logits, dim=-1).item()
                    if self.pong:
                        action = 2 if act_idx == 0 else 3
                    else:
                        action = act_idx
                else:
                    dist = self.action_head(logits)
                    a = dist.sample().item()
                    if self.pong:
                        action = 2 if a == 0 else 3
                    else:
                        action = a
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                ep_r += reward
            returns.append(ep_r)
        return np.array(returns, dtype=np.float32)

    def save(self, path: str):
        torch.save(self.policy.state_dict(), path)

# -----------------------------
# Plot helpers
# -----------------------------

def plot_training(rewards, env_tag: str):
    plt.figure()
    plt.plot(rewards, label='Episode return')
    plt.plot(moving_average(rewards, k=100), label='100-ep MA')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title(f'Training: {env_tag}')
    plt.legend()
    out = f'plots/{env_tag}_reward_curve.png'
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_histogram(returns, env_tag: str):
    plt.figure()
    plt.hist(returns, bins=30)
    plt.xlabel('Episode return')
    plt.ylabel('Count')
    plt.title(f'Rollout distribution (n={len(returns)}): {env_tag}')
    out = f'plots/{env_tag}_rollout_hist.png'
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {out}")

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', choices=['cartpole', 'pong'], required=True)
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--gamma', type=float, default=None, help='If None, use per-env default')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--baseline', choices=['none', 'moving_avg', 'value'], default='none')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--deterministic_rollout', action='store_true', help='Use argmax for 500-ep rollouts')
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dirs()

    if args.env == 'cartpole' and (args.gamma is None):
        args.gamma = 0.95
    if args.env == 'pong' and (args.gamma is None):
        args.gamma = 0.99

    tag = f"{args.env}_base-{args.baseline}_lr-{args.lr}_g-{args.gamma}_seed-{args.seed}"
    print(f"Config: {tag} on {args.device}")

    agent = REINFORCEAgent(env_name=args.env, gamma=args.gamma, lr=args.lr,
                           baseline=args.baseline, device=args.device)

    # Train
    rewards = agent.train(episodes=args.episodes, render=args.render)
    plot_training(rewards, tag)

    # Save model
    ckpt_path = f"checkpoints/{tag}_policy.pt"
    agent.save(ckpt_path)
    print(f"Saved policy: {ckpt_path}")

    # 500-episode rollout & histogram
    print("Rolling out 500 episodes with trained policy...")
    returns = agent.rollout(episodes=500, deterministic=args.deterministic_rollout)
    mu, sigma = returns.mean(), returns.std(ddof=1)
    print(f"Rollout mean: {mu:.3f} | std: {sigma:.3f}")
    plot_histogram(returns, tag)

if __name__ == '__main__':
    main()
