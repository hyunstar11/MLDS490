#!/usr/bin/env python3
import os
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym

def set_tf():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"1 Physical GPUs, {len(gpus)} Logical GPUs")
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

    return tf.keras.Model(inp, q_values, name="DuelingDQN")

def load_policy(model_path, num_actions):
    if model_path.endswith(".weights.h5"):
        print(f"Loading weights: {model_path}")
        model = build_q_network(num_actions)
        model.load_weights(model_path)
        return model
    elif model_path.endswith(".keras"):
        print(f"Loading full model (.keras): {model_path}")
        keras.config.enable_unsafe_deserialization()
        return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
    else:
        raise ValueError(f"Unsupported model_path: {model_path}")

def evaluate(args):
    set_tf()
    os.makedirs(args.out_dir, exist_ok=True)

    env = gym.make(args.env)
    num_actions = env.action_space.n

    policy = load_policy(args.model_path, num_actions)

    rewards = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        s = preprocess(obs)
        done = False
        ep_reward = 0.0
        steps = 0
        while not done:
            q = policy.predict(s[None, ...], verbose=0)[0]
            a = int(np.argmax(q))
            next_obs, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            s = preprocess(next_obs)
            ep_reward += r
            steps += 1
        rewards.append(ep_reward)
        print(f"[Rollout] Episode {ep}: reward={ep_reward:.1f}, steps={steps}")

    rewards = np.array(rewards, dtype=np.float32)
    mean_r = float(np.mean(rewards))
    std_r = float(np.std(rewards))
    print(f"[Rollout] Done {args.episodes} eps. mean={mean_r:.2f}, std={std_r:.2f}")

    np.save(os.path.join(args.out_dir, "mspacman_rollout_rewards.npy"), rewards)
    with open(os.path.join(args.out_dir, "mspacman_rollout_stats.txt"), "w") as f:
        f.write(f"mean: {mean_r:.4f}\nstd: {std_r:.4f}\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure()
        plt.hist(rewards, bins=30)
        plt.xlabel("Episode reward")
        plt.ylabel("Count")
        plt.title(f"MsPacman rollout ({args.episodes} eps)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "mspacman_rollout_hist.png"))
        plt.close()
        print(f"[ARTIFACTS] Saved histogram and stats to {args.out_dir}")
    except Exception as e:
        print("Plotting failed:", e)

    env.close()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="MsPacman-v0")
    p.add_argument("--model_path", type=str, required=True, help="Path to .weights.h5 or .keras")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--out_dir", type=str, default="artifacts")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
