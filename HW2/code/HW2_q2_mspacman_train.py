#!/usr/bin/env python3
import os
import argparse
import random
from collections import deque
import numpy as np
import tensorflow as tf
import gym

def set_tf():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPUs available: {gpus}")
        except Exception as e:
            print("Could not set GPU memory growth:", e)

def preprocess(obs):
    x = tf.convert_to_tensor(obs)
    if x.shape.rank == 3 and x.shape[-1] == 3:
        x = tf.image.rgb_to_grayscale(x)
    x = tf.image.resize(x, (88, 80), method="nearest")
    x = tf.cast(x, tf.float32) / 255.0
    return x.numpy()

def build_q_network(num_actions: int):
    inp = tf.keras.Input(shape=(88, 80, 1), name="input_layer")
    x = tf.keras.layers.Conv2D(32, 8, strides=2, activation="relu")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 4, strides=1, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)

    class MeanAdvLayer(tf.keras.layers.Layer):
        def call(self, inputs):
            return tf.reduce_mean(inputs, axis=1, keepdims=True)
        def compute_output_shape(self, input_shape):
            return (input_shape[0], 1)

    value_stream = tf.keras.layers.Dense(1, name="value")(x)
    advantage_stream = tf.keras.layers.Dense(num_actions, name="advantage")(x)
    adv_mean = MeanAdvLayer(name="adv_mean")(advantage_stream)
    adv_centered = tf.keras.layers.Subtract(name="adv_centered")([advantage_stream, adv_mean])
    q_values = tf.keras.layers.Add(name="q_values")([value_stream, adv_centered])

    model = tf.keras.Model(inp, q_values, name="DuelingDQN")
    return model

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        s, a, r, ns, d = [], [], [], [], []
        for i in idxs:
            si, ai, ri, nsi, di = self.buffer[i]
            s.append(si)
            a.append(ai)
            r.append(ri)
            ns.append(nsi)
            d.append(di)
        return (np.array(s, dtype=np.float32),
                np.array(a, dtype=np.int32),
                np.array(r, dtype=np.float32),
                np.array(ns, dtype=np.float32),
                np.array(d, dtype=np.float32))

def moving_average(x, w):
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    ret = np.cumsum(x, dtype=float)
    ret[w:] = ret[w:] - ret[:-w]
    out = np.empty_like(x, dtype=np.float32)
    out[:] = np.nan
    out[w-1:] = (ret[w-1:] / w).astype(np.float32)
    return out

def train(args):
    set_tf()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    env = gym.make("MsPacman-v0")
    num_actions = env.action_space.n

    q_net = build_q_network(num_actions)
    target_net = build_q_network(num_actions)
    target_net.set_weights(q_net.get_weights())

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss_fn = tf.keras.losses.Huber()

    buffer = ReplayBuffer(capacity=100_000)

    epsilon_start, epsilon_end = 1.0, 0.1
    epsilon = epsilon_start
    frame_count = 0

    all_rewards = []
    from collections import deque as dq
    last100 = dq(maxlen=100)
    max_q_per_episode = []

    def select_action(state):
        nonlocal epsilon
        if random.random() < epsilon:
            return env.action_space.sample(), None
        q = q_net.predict(state[None, ...], verbose=0)[0]
        return int(np.argmax(q)), float(np.max(q))

    @tf.function
    def train_step(states, actions, rewards, next_states, dones, gamma):
        next_q = target_net(next_states, training=False)
        next_max = tf.reduce_max(next_q, axis=1)

        target = rewards + (1.0 - dones) * gamma * next_max

        with tf.GradientTape() as tape:
            q_values = q_net(states, training=True)
            action_mask = tf.one_hot(actions, depth=num_actions)
            q_selected = tf.reduce_sum(q_values * action_mask, axis=1)
            loss = loss_fn(target, q_selected)

        grads = tape.gradient(loss, q_net.trainable_variables)
        optimizer.apply_gradients(zip(grads, q_net.trainable_variables))
        return loss

    for ep in range(args.episodes):
        obs, _ = env.reset()
        s = preprocess(obs)
        done = False
        ep_reward = 0.0
        ep_max_q = -1e9

        while not done:
            a, q_max = select_action(s)
            if q_max is not None:
                ep_max_q = max(ep_max_q, q_max)

            next_obs, r, terminated, truncated, _ = env.step(a)
            d = terminated or truncated
            ns = preprocess(next_obs)

            buffer.push(s, a, r, ns, float(d))

            s = ns
            ep_reward += r
            done = d

            frame_count += 1
            frac = min(1.0, frame_count / float(args.eps_decay_steps))
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * frac

            if len(buffer) >= args.batch_size and (frame_count % args.train_every == 0):
                states, actions, rewards, next_states, dones = buffer.sample(args.batch_size)
                loss = train_step(states, actions, rewards, next_states, dones, tf.constant(args.gamma, tf.float32))
                print(f"loss: {float(loss.numpy()):.6f}")

            if frame_count % args.target_sync_every == 0:
                target_net.set_weights(q_net.get_weights())

        all_rewards.append(ep_reward)
        last100.append(ep_reward)
        max_q_per_episode.append(ep_max_q if ep_max_q != -1e9 else np.nan)

        print(f"MsPacman-v0 episode {ep}, transformed reward sum: {ep_reward:.1f}, reward sum: {ep_reward:.1f}, last 100 avg: {np.nanmean(last100):.2f}")
        print(f"Epsilon: {epsilon:.12f}")

        if (ep + 1) % args.checkpoint_every == 0:
            ckpt_base = os.path.join(args.save_dir, f"mspacman_model_{ep+1}")
            q_net.save(f"{ckpt_base}.keras")
            q_net.save_weights(os.path.join(args.save_dir, f"mspacman_weights_{ep+1}.weights.h5"))
            print(f"[CKPT] saved {ckpt_base}.keras and weights")

    q_net.save(os.path.join(args.save_dir, "mspacman_model_final.keras"))
    q_net.save_weights(os.path.join(args.save_dir, "mspacman_weights_final.weights.h5"))
    print("[FINAL] model and weights saved.")

    # Also persist metrics and quick plots every checkpoint
    np.save(os.path.join(args.out_dir, "msp_rewards.npy"),
            np.array(all_rewards, dtype=np.float32))
    np.save(os.path.join(args.out_dir, "msp_maxq.npy"),
            np.array(max_q_per_episode, dtype=np.float32))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Reward curve + 100-ep moving average (so far)
        ma = moving_average(all_rewards, 100)
        plt.figure()
        plt.plot(all_rewards, label="Episode reward")
        plt.plot(ma, label="Moving avg (100)", linewidth=2)
        plt.xlabel("Episode"); plt.ylabel("Reward")
        plt.title("MsPacman: Episode Reward (so far)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "msp_rewards_latest.png"))
        plt.close()

        # Optional: histogram of recent rewards (last 100)
        if len(all_rewards) >= 10:
            plt.figure()
            recent = all_rewards[-100:] if len(all_rewards) >= 100 else all_rewards
            plt.hist(recent, bins=20)
            plt.xlabel("Reward"); plt.ylabel("Count")
            plt.title(f"Reward histogram (last {len(recent)} eps)")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, f"msp_reward_hist_ep{ep+1}.png"))
            plt.close()

        # Optional: max-Q trace so far
        plt.figure()
        plt.plot(max_q_per_episode)
        plt.xlabel("Episode"); plt.ylabel("Max Q")
        plt.title("MsPacman: Max Q vs Episodes (so far)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "msp_maxQ_latest.png"))
        plt.close()

        print(f"[CKPT] metrics & plots saved to {args.out_dir}")
    except Exception as e:
        print(f"[WARN] plotting at ep {ep+1} failed: {e}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ma = moving_average(all_rewards, 100)
        plt.figure()
        plt.plot(all_rewards, label="Episode reward")
        plt.plot(ma, label="Moving avg (100)", linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("MsPacman: Episode Reward & 100-ep Moving Avg")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "msp_rewards.png"))
        plt.close()

        plt.figure()
        plt.plot(max_q_per_episode)
        plt.xlabel("Episode")
        plt.ylabel("Max Q")
        plt.title("MsPacman: Max Q vs Episodes")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "msp_maxQ.png"))
        plt.close()
        print(f"[PLOTS] saved to {args.out_dir}")
    except Exception as e:
        print("Plotting failed:", e)

    env.close()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--train_every", type=int, default=4)
    p.add_argument("--target_sync_every", type=int, default=10_000)
    p.add_argument("--eps_decay_steps", type=int, default=1_000_000)
    p.add_argument("--save_dir", type=str, default="saved_model")
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--checkpoint_every", type=int, default=100)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
