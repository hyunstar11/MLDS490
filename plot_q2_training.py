#!/usr/bin/env python3
import re, os, argparse, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

def smooth(y, w=200):
    if len(y) < w: return np.array(y, dtype=float)
    k = np.ones(w) / w
    return np.convolve(y, k, mode="valid")

def parse_log(path):
    ep, rew, last100 = [], [], []
    epsilons, losses = [], []
    ep_seen = 0
    epi_re = re.compile(r"MsPacman.*episode\s+(\d+).*?reward sum:\s*([\-0-9.]+).*?last 100 avg:\s*([\-0-9.]+)")
    eps_re = re.compile(r"^Epsilon:\s*([\-0-9.eE]+)")
    loss_re = re.compile(r"^loss:\s*([\-0-9.eE]+)")
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = epi_re.search(line)
            if m:
                ep.append(int(m.group(1)))
                rew.append(float(m.group(2)))
                last100.append(float(m.group(3)))
                continue
            m = eps_re.search(line)
            if m:
                epsilons.append(float(m.group(1)))
                continue
            m = loss_re.search(line)
            if m:
                losses.append(float(m.group(1)))
                continue
    return np.array(ep), np.array(rew), np.array(last100), np.array(epsilons), np.array(losses)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="pacman_train.log")
    ap.add_argument("--out_dir", default="artifacts")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ep, rew, last100, epsilons, losses = parse_log(args.log)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save CSV
    import csv
    csv_path = os.path.join(args.out_dir, f"q2_training_metrics_{ts}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode","reward","last100"])
        for i in range(len(ep)):
            w.writerow([ep[i], rew[i] if i<len(rew) else "", last100[i] if i<len(last100) else ""])

    # Reward plot
    plt.figure(figsize=(8,4.5))
    plt.plot(ep, rew, label="reward per episode", linewidth=1)
    if len(rew) > 20:
        sm = smooth(rew, w=20)
        plt.plot(ep[len(ep)-len(sm):], sm, label="reward (smooth w=20)", linewidth=2)
    plt.xlabel("Episode"); plt.ylabel("Reward"); plt.title("MsPacman — Reward per Episode")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"q2_rewards_{ts}.png"), dpi=150); plt.close()

    # Last 100 avg
    if len(last100):
        plt.figure(figsize=(8,4.5))
        plt.plot(ep[:len(last100)], last100, linewidth=1.5)
        plt.xlabel("Episode"); plt.ylabel("Last-100 Average"); plt.title("MsPacman — Last-100 Avg Reward")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"q2_last100_{ts}.png"), dpi=150); plt.close()

    # Epsilon
    if len(epsilons):
        plt.figure(figsize=(8,4.5))
        plt.plot(range(len(epsilons)), epsilons, linewidth=1.5)
        plt.xlabel("Epsilon step"); plt.ylabel("Epsilon"); plt.title("MsPacman — Epsilon")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"q2_epsilon_{ts}.png"), dpi=150); plt.close()

    # Loss (smoothed)
    if len(losses):
        plt.figure(figsize=(8,4.5))
        plt.plot(range(len(losses)), losses, alpha=0.35, linewidth=0.8, label="raw loss")
        sm = smooth(losses, w=max(20, len(losses)//50))
        x = np.arange(len(losses))[-len(sm):]
        plt.plot(x, sm, linewidth=2, label="smoothed")
        plt.xlabel("Update"); plt.ylabel("Loss"); plt.title("MsPacman — Loss")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"q2_loss_{ts}.png"), dpi=150); plt.close()

    print("Wrote:", args.out_dir)

if __name__ == "__main__":
    main()