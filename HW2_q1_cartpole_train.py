#!/usr/bin/env python3
"""
HW2_q1_cartpole_train.py (fixed)
--------------------------------
- CartPole-v1, Gym 0.26+ API compatible
- DQN (Keras) with replay + target network
- Tracks REQUIRED plots:
    (i)  Max Q-value per episode  -> outputs/cartpole_maxQ.png
    (ii) Rewards + MA(100)        -> outputs/cartpole_rewards.png
- Saves Keras 3-compatible checkpoints as .keras under saved_model/
- Default early stop: v1-style (10 consecutive episodes with return >= 500).
- Optional classic stop: --stop_on_ma195 1 (keeps v0-style criterion).

CLI examples:
  python HW2_q1_cartpole_train.py
  python HW2_q1_cartpole_train.py --episodes 1500 --eps_decay_steps 8000 --train_every 1 --target_sync_every 500 --lr 5e-4 --batch_size 128
"""
import os
import math
import argparse
from dataclasses import dataclass
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt

# NumPy 2.x compatibility for Gym 0.26 checker (harmless on NumPy 1.26.x)
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

def _lazy_import_tf():
    try:
        import tensorflow as tf
        from tensorflow import keras
        return tf, keras
    except Exception as e:
        raise RuntimeError('TensorFlow 2.x required. Install TF 2.10+ (2.13 suggested).') from e

def _lazy_import_gym():
    try:
        import gym
        return gym
    except ModuleNotFoundError as e:
        raise RuntimeError('gym not found. Install with: pip install "gym==0.26.2"') from e

def print_gpu_info():
    tf, _ = _lazy_import_tf()
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except Exception:
            pass
        print('GPU is available')
        print('GPU device:', gpus[0])
    else:
        print('No GPU detected. Running on CPU.')

@dataclass
class DQNConfig:
    env_id: str = 'CartPole-v1'
    gamma: float = 0.95
    episodes: int = 800
    max_steps: int = 1_000_000
    buffer_size: int = 50_000
    batch_size: int = 64
    lr: float = 1e-3
    tau: float = 1.0         # 1.0 = hard copy; <1.0 = Polyak
    target_sync_every: int = 1000
    warmup_steps: int = 1000
    train_every: int = 4
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 40_000
    seed: int = 42
    consec_solve: int = 10   # require 10 consecutive 500s for v1
    stop_on_ma195: int = 0   # if 1: also stop when MA(100) >= 195 (classic v0 criterion)
    save_dir: str = 'saved_model'
    save_prefix: str = 'cartpole_model'
    out_dir: str = 'outputs'

class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple, rng: np.random.RandomState):
        self.capacity = capacity
        self.rng = rng
        self.idx = 0
        self.full = False
        self.states = np.zeros((capacity,) + obs_shape, dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity,) + obs_shape, dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)

    def push(self, s, a, r, ns, done):
        self.states[self.idx] = s
        self.actions[self.idx] = a
        self.rewards[self.idx] = r
        self.next_states[self.idx] = ns
        self.dones[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def __len__(self):
        return self.capacity if self.full else self.idx

    def sample(self, batch_size):
        max_idx = self.capacity if self.full else self.idx
        idxs = self.rng.randint(0, max_idx, size=batch_size)
        return (self.states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs], self.dones[idxs])

def build_q_network(obs_dim: int, n_actions: int, lr: float):
    tf, keras = _lazy_import_tf()
    L = keras.layers
    model = keras.Sequential([
        keras.Input(shape=(obs_dim,)),
        L.Dense(128, activation='relu'),
        L.Dense(128, activation='relu'),
        L.Dense(n_actions, activation=None)  # Q-values
    ])
    opt = keras.optimizers.Adam(learning_rate=lr)
    loss_fn = keras.losses.Huber()
    return model, opt, loss_fn

def select_action(model, obs, n_actions, epsilon, rng):
    if rng.rand() < epsilon:
        return rng.randint(n_actions)
    q = model(np.expand_dims(obs, axis=0), training=False).numpy()[0]
    return int(np.argmax(q))

def polyak_update(target_weights, online_weights, tau: float):
    for tw, w in zip(target_weights, online_weights):
        tw.assign((1.0 - tau) * tw + tau * w)

def train(cfg: DQNConfig):
    print_gpu_info()
    gym = _lazy_import_gym()
    tf, keras = _lazy_import_tf()

    rng = np.random.RandomState(cfg.seed)
    random.seed(cfg.seed)

    env = gym.make(cfg.env_id)
    try:
        reset_out = env.reset(seed=cfg.seed)
    except TypeError:
        reset_out = env.reset()
    state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    obs_dim = state.shape[0] if hasattr(state, 'shape') else len(state)
    n_actions = env.action_space.n

    online_model, optimizer, loss_fn = build_q_network(obs_dim, n_actions, cfg.lr)
    target_model, _, _ = build_q_network(obs_dim, n_actions, cfg.lr)
    target_model.set_weights(online_model.get_weights())

    buffer = ReplayBuffer(cfg.buffer_size, (obs_dim,), rng)

    save_dir = Path(cfg.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(cfg.out_dir);   out_dir.mkdir(parents=True, exist_ok=True)

    ep_returns, moving_avg = [], []
    ep_max_q = []            # REQUIRED: per-episode max Q
    total_steps = 0
    grad_steps  = 0
    best_ma = -1e9
    streak_500 = 0

    def epsilon_by_step(step):
        if step >= cfg.eps_decay_steps:
            return cfg.eps_end
        slope = (cfg.eps_end - cfg.eps_start) / cfg.eps_decay_steps
        return cfg.eps_start + slope * step

    for episode in range(cfg.episodes):
        reset_out = env.reset()
        state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        done = False
        ep_return = 0.0
        q_max_ep = -float('inf')

        while not done:
            epsilon = epsilon_by_step(total_steps)
            action = select_action(online_model, np.asarray(state, dtype=np.float32), n_actions, epsilon, rng)
            step_out = env.step(action)
            if len(step_out) == 5:
                next_state, reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:
                next_state, reward, done, info = step_out

            # Track max Q for this episode
            q_vals = online_model(np.expand_dims(np.asarray(state, dtype=np.float32), axis=0), training=False).numpy()[0]
            q_max_ep = max(q_max_ep, float(np.max(q_vals)))

            buffer.push(np.asarray(state, dtype=np.float32), int(action), float(reward), np.asarray(next_state, dtype=np.float32), bool(done))

            state = next_state
            ep_return += float(reward)
            total_steps += 1

            if len(buffer) >= cfg.warmup_steps and total_steps % cfg.train_every == 0:
                s_b, a_b, r_b, ns_b, d_b = buffer.sample(cfg.batch_size)
                with tf.GradientTape() as tape:
                    q_online = online_model(s_b, training=True)
                    action_mask = tf.one_hot(a_b, n_actions)
                    q_sa = tf.reduce_sum(q_online * action_mask, axis=1)
                    q_next = target_model(ns_b, training=False)
                    max_q_next = tf.reduce_max(q_next, axis=1)
                    target = r_b + cfg.gamma * (1.0 - d_b.astype(np.float32)) * max_q_next
                    loss = loss_fn(target, q_sa)
                grads = tape.gradient(loss, online_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, online_model.trainable_variables))
                grad_steps += 1

                if grad_steps % cfg.target_sync_every == 0:
                    if abs(cfg.tau - 1.0) < 1e-8:
                        target_model.set_weights(online_model.get_weights())
                    else:
                        polyak_update(target_model.weights, online_model.weights, cfg.tau)

            if total_steps >= cfg.max_steps:
                print('Hit max global steps; stopping early.')
                done = True

        # Episode end
        ep_returns.append(ep_return)
        ep_max_q.append(q_max_ep)
        ma100 = float(np.mean(ep_returns[-100:]))
        moving_avg.append(ma100)

        # Early-stop logic
        if ep_return >= 500.0:
            streak_500 += 1
        else:
            streak_500 = 0

        print(f"CartPole-v1 episode {episode}, reward sum: {ep_return:.1f}, last 100 avg: {ma100:.2f}, maxQ: {q_max_ep:.3f}")

        # Save checkpoint every 25 episodes and when MA improves
        if (episode + 1) % 25 == 0 or ma100 > best_ma:
            best_ma = max(best_ma, ma100)
            ckpt = save_dir / f"{cfg.save_prefix}_{episode+1:04d}.keras"
            online_model.save(ckpt.as_posix())

        # Stop conditions
        if cfg.stop_on_ma195 and ma100 >= 195.0:
            print('Stopping as the last 100-episode moving average is greater than 195 (classic v0 criterion).')
            final_path = save_dir / f"{cfg.save_prefix}_ma195.keras"
            online_model.save(final_path.as_posix())
            break
        if streak_500 >= cfg.consec_solve:
            print(f'Solved CartPole-v1: {cfg.consec_solve} consecutive episodes with return >= 500.')
            final_path = save_dir / f"{cfg.save_prefix}_solved.keras"
            online_model.save(final_path.as_posix())
            break

    env.close()

    # Final save
    final_ckpt = save_dir / f"{cfg.save_prefix}_final.keras"
    online_model.save(final_ckpt.as_posix())
    print(f'Saved final model: {final_ckpt}')

    # REQUIRED plots
    out_dir.mkdir(parents=True, exist_ok=True)
    # (ii) Rewards + moving average (100)
    plt.figure(figsize=(9,5))
    plt.plot(ep_returns, label='Episode return')
    if len(moving_avg) > 1:
        plt.plot(moving_avg, label='Moving avg (100)')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('CartPole-v1: Episode rewards')
    plt.legend()
    rewards_png = out_dir / 'cartpole_rewards.png'
    plt.tight_layout(); plt.savefig(rewards_png.as_posix(), dpi=150); plt.close()
    print('Saved:', rewards_png)

    # (i) Max Q per episode
    plt.figure(figsize=(9,5))
    plt.plot(ep_max_q, label='Max Q per episode')
    plt.xlabel('Episode')
    plt.ylabel('Max Q')
    plt.title('CartPole-v1: Maximum Q-value vs Episodes')
    plt.legend()
    maxq_png = out_dir / 'cartpole_maxQ.png'
    plt.tight_layout(); plt.savefig(maxq_png.as_posix(), dpi=150); plt.close()
    print('Saved:', maxq_png)

def parse_args():
    p = argparse.ArgumentParser(description='DQN trainer for CartPole-v1 (Gym 0.26+).')
    p.add_argument('--episodes', type=int, default=800)
    p.add_argument('--gamma', type=float, default=0.95)
    p.add_argument('--buffer_size', type=int, default=50000)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--target_sync_every', type=int, default=1000)
    p.add_argument('--tau', type=float, default=1.0)
    p.add_argument('--warmup_steps', type=int, default=1000)
    p.add_argument('--train_every', type=int, default=4)
    p.add_argument('--eps_start', type=float, default=1.0)
    p.add_argument('--eps_end', type=float, default=0.05)
    p.add_argument('--eps_decay_steps', type=int, default=40000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--consec_solve', type=int, default=10)
    p.add_argument('--stop_on_ma195', type=int, default=0)
    p.add_argument('--save_dir', type=str, default='saved_model')
    p.add_argument('--save_prefix', type=str, default='cartpole_model')
    p.add_argument('--out_dir', type=str, default='outputs')
    p.add_argument('--max_steps', type=int, default=1_000_000)
    args = p.parse_args()
    return DQNConfig(env_id='CartPole-v1', gamma=args.gamma, episodes=args.episodes, max_steps=args.max_steps,
                     buffer_size=args.buffer_size, batch_size=args.batch_size, lr=args.lr, tau=args.tau,
                     target_sync_every=args.target_sync_every, warmup_steps=args.warmup_steps,
                     train_every=args.train_every, eps_start=args.eps_start, eps_end=args.eps_end,
                     eps_decay_steps=args.eps_decay_steps, seed=args.seed, consec_solve=args.consec_solve,
                     stop_on_ma195=args.stop_on_ma195, save_dir=args.save_dir, save_prefix=args.save_prefix,
                     out_dir=args.out_dir)

if __name__ == '__main__':
    cfg = parse_args()
    train(cfg)