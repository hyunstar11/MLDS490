#!/usr/bin/env python3
"""
HW2_q1_cartpole_rollout.py
---------------------------------
Greedy 500-episode rollout for CartPole-v1 using a trained Keras DQN model.

Key features:
- Auto-detects the latest model checkpoint in `--model_dir` supporting:
  * Native Keras (.keras), legacy HDF5 (.h5), or a SavedModel directory
- Gym 0.26+ API compatible (handles terminated/truncated separately)
- Saves histogram plot and stats JSON to `--out_dir`
- CLI options for episodes, seed, and rendering

Usage (terminal):
  python HW2_q1_cartpole_rollout.py \
    --model_dir saved_model \
    --prefix cartpole_model \
    --episodes 500 \
    --out_dir outputs \
    --render 0

If you want to just run defaults:
  python HW2_q1_cartpole_rollout.py

Outputs:
  outputs/cartpole_rollout_hist.png
  outputs/cartpole_rollout_stats.json
"""
import os
import json
import argparse
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt

def _lazy_import_tf():
    try:
        import tensorflow as tf
        return tf
    except Exception as e:
        raise RuntimeError("TensorFlow is required to run rollout. Install TF 2.10+ (2.13 suggested).") from e

def set_gpu_memory_growth():
    tf = _lazy_import_tf()
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except Exception:
            pass

def gym_make(env_id):
    try:
        import gym
    except ModuleNotFoundError:
        raise RuntimeError("gym is not installed. Try: pip install \"gym==0.26.2\"")
    return gym.make(env_id)

def find_latest_model(model_dir: Path, prefix: str):
    """
    Look for the newest model matching <prefix>* under model_dir.
    Acceptable formats:
      - <prefix>*.keras
      - <prefix>*.h5
      - Directory containing a SavedModel (saved_model.pb)
    Returns a Path or None.
    """
    candidates = []

    # Files with known extensions
    for ext in (".keras", ".h5"):
        for p in model_dir.glob(f"{prefix}*{ext}"):
            candidates.append(p)

    # SavedModel directories
    for p in model_dir.glob(f"{prefix}*"):
        if p.is_dir() and (p / "saved_model.pb").exists():
            candidates.append(p)

    if not candidates:
        return None

    # Sort by mtime (newest first)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

def load_model(model_path: Path):
    tf = _lazy_import_tf()
    from tensorflow import keras

    if model_path.is_dir():
        # SavedModel
        model = keras.models.load_model(model_path.as_posix())
    else:
        # .keras or .h5
        model = keras.models.load_model(model_path.as_posix())
    return model

def predict_action(model, obs: np.ndarray):
    """obs shape (obs_dim,) -> action int via argmax Q."""
    # Ensure batch dimension
    q_values = model(np.expand_dims(obs, axis=0), training=False).numpy()[0]
    return int(np.argmax(q_values))

def rollout(env_id="CartPole-v1",
            model_dir="saved_model",
            prefix="cartpole_model",
            episodes=500,
            seed=42,
            render=False,
            out_dir="outputs"):
    set_gpu_memory_growth()

    model_dir = Path(model_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = find_latest_model(model_dir, prefix)
    if model_path is None:
        raise FileNotFoundError(
            f"No model found in '{model_dir}' with prefix '{prefix}'. "
            f"Expected files like '{prefix}*.keras' or '.h5', or a SavedModel directory.")

    model = load_model(model_path)

    # Gym env
    env = gym_make(env_id)
    try:
        # Seed if available (Gym 0.26+)
        if hasattr(env, "reset"):
            try:
                env.reset(seed=seed)
            except TypeError:
                pass
    except Exception:
        pass

    returns = []
    for ep in range(episodes):
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            obs, info = reset_out
        else:
            obs, info = reset_out, {}

        total_r = 0.0
        while True:
            if render:
                env.render()

            action = predict_action(model, np.asarray(obs, dtype=np.float32))
            step_out = env.step(action)

            # Handle Gym 0.26+ and older APIs
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                obs, reward, done, info = step_out

            total_r += float(reward)
            if done:
                break

        returns.append(total_r)
        if (ep + 1) % 25 == 0 or ep == 0:
            print(f"[{ep+1}/{episodes}] return={total_r:.1f}  mean@{ep+1}={np.mean(returns):.2f}")

    env.close()

    returns = np.array(returns, dtype=np.float32)
    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns))

    # Plot histogram
    plt.figure(figsize=(8,5))
    plt.hist(returns, bins=30, edgecolor="black")
    plt.title(f"{env_id}  {episodes} episodes  (mean={mean_r:.2f}, std={std_r:.2f})")
    plt.xlabel("Episode Return")
    plt.ylabel("Count")
    png_path = out_dir / "cartpole_rollout_hist.png"
    plt.tight_layout()
    plt.savefig(png_path.as_posix(), dpi=150)
    plt.close()

    # Save stats
    stats = {
        "env_id": env_id,
        "episodes": int(episodes),
        "mean_return": mean_r,
        "std_return": std_r,
        "model_path": model_path.as_posix(),
        "timestamp": int(time.time())
    }
    with open(out_dir / "cartpole_rollout_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved histogram to: {png_path}")
    print(f"Saved stats to:     {out_dir/'cartpole_rollout_stats.json'}")
    print(f"Model used:         {model_path}")

def parse_args():
    p = argparse.ArgumentParser(description="CartPole-v1 greedy rollout (500 episodes by default).")
    p.add_argument("--model_dir", type=str, default="saved_model", help="Directory containing trained model files.")
    p.add_argument("--prefix", type=str, default="cartpole_model", help="Filename prefix for model checkpoints.")
    p.add_argument("--episodes", type=int, default=500, help="Number of rollout episodes.")
    p.add_argument("--seed", type=int, default=42, help="Env seed.")
    p.add_argument("--render", type=int, default=0, help="1 to render.")
    p.add_argument("--out_dir", type=str, default="outputs", help="Output directory for plots and stats.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    rollout(env_id="CartPole-v1",
            model_dir=args.model_dir,
            prefix=args.prefix,
            episodes=args.episodes,
            seed=args.seed,
            render=bool(args.render),
            out_dir=args.out_dir)