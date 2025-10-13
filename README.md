# MLDS490

## HW1 — Policy Gradient (REINFORCE) in PyTorch  
Envs: **CartPole-v1** and **ALE/Pong-v5** with/without a baseline

This repo contains a minimal PyTorch implementation of REINFORCE trained on CartPole (classic control) and Pong (Atari). You can run with either **no baseline** or an **exponential moving-average (EMA) baseline** to reduce variance. Training logs, plots, and rollout stats are saved automatically.

---

## 1) Environment & Dependencies

### Option A — CPU (simple)
```bash
conda create -n rlhw python=3.10 -y
conda activate rlhw

# Avoid NumPy 2.x incompat warnings with some native wheels
pip install "numpy<2" matplotlib

# PyTorch (CPU build)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Gymnasium + ALE
pip install "gymnasium==0.29.1" "gymnasium[atari]==0.29.1" "ale-py==0.8.1"

# Atari ROMs (auto-installs licensed ROM set)
pip install "autorom[accept-rom-license]"
AutoROM --accept-license
```

### Option B — GPU (CUDA 11.8, e.g., Deepdish)
```bash
conda create -n rlhw python=3.10 -y
conda activate rlhw

pip install "numpy<2" matplotlib
pip install "gymnasium==0.29.1" "gymnasium[atari]==0.29.1" "ale-py==0.8.1"
pip install "autorom[accept-rom-license]"
AutoROM --accept-license

# PyTorch CUDA 11.8 wheels
pip install "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1"   --index-url https://download.pytorch.org/whl/cu118
```

Quick sanity check:
```bash
python - <<'PY'
import gymnasium as gym
env = gym.make('ALE/Pong-v5', obs_type='rgb')
obs, _ = env.reset()
print('Pong obs:', obs.shape)   # expected (210, 160, 3)
env.close()
PY
```

---

## 2) How to Run

The main entrypoint is **`HW1_PolicyGradient_PyTorch.py`**.

**Common flags**
- `--env {cartpole|pong}`
- `--episodes INT`
- `--gamma FLOAT` (discount)
- `--baseline {none|moving_avg}`
- `--lr FLOAT`
- `--seed INT`
- `--device {cpu|cuda}`

### Reproducing the reported runs

**Part 1(a) — CartPole (no baseline)**
```bash
python HW1_PolicyGradient_PyTorch.py   --env cartpole --episodes 1500 --gamma 0.95   --baseline none --lr 0.003 --seed 42 --device cpu
```

**Part 1(b) — Pong (no baseline, CPU)**
```bash
python HW1_PolicyGradient_PyTorch.py   --env pong --episodes 2000 --gamma 0.99   --baseline none --lr 0.001 --seed 1 --device cpu
```

**Part 2 — CartPole (moving_avg baseline)**
```bash
python HW1_PolicyGradient_PyTorch.py   --env cartpole --episodes 1500 --gamma 0.95   --baseline moving_avg --lr 0.003 --seed 1 --device cpu
```

**Part 2 — Pong (moving_avg baseline, GPU)**
```bash
export CUDA_VISIBLE_DEVICES=0
python HW1_PolicyGradient_PyTorch.py   --env pong --episodes 2000 --gamma 0.99   --baseline moving_avg --lr 0.001 --seed 1 --device cuda
```

**Outputs**
- Plots → `plots/<config>_reward_curve.png`, `plots/<config>_rollout_hist.png`
- Checkpoints → `checkpoints/<config>_policy.pt`
- Console prints **rollout mean/std** after training

`<config>` encodes env, baseline, lr, gamma, seed (e.g., `pong_base-moving_avg_lr-0.001_g-0.99_seed-1`).

---

## 3) Pseudocode (REINFORCE)

### REINFORCE w/ optional EMA baseline
```text
Initialize policy θ, optimizer
Initialize baseline b ← 0 (used only if baseline = moving_avg)
Choose smoothing β ∈ (0,1) for EMA baseline (e.g., β = 0.99)

for episode = 1..N:
    s = env.reset()
    Trajectory buffers: rewards R = [], logprobs L = []
    done = False
    while not done:
        sample action a ~ πθ(·|s)
        logp = log πθ(a|s); L.append(logp)
        s', r, done, info = env.step(a)
        R.append(r)
        s = s'

    # compute discounted returns G_t
    G = []
    running = 0
    for r in reversed(R):
        running = r + γ * running
        G.insert(0, running)

    if baseline == moving_avg:
        # single scalar EMA baseline per episode
        b = β*b + (1-β)*mean(G)
        advantages A_t = G_t - b
    else:
        advantages A_t = G_t

    # policy gradient loss (maximize expected return → minimize negative)
    L_policy = - Σ_t ( L[t] * stopgrad( A_t ) )

    optimizer.zero_grad()
    L_policy.backward()
    optimizer.step()
```

### Evaluation (rollout)
```text
Given trained policy θ:
for k = 1..500 episodes:
    roll out episode using stochastic policy πθ
    record total return
Report mean and std over 500 episodes
Plot histogram of returns
```

---

## 4) Sample Output

**Final rollout stats (500 eps) from my runs**

- CartPole **moving_avg**: `mean 498.470 | std 17.746`  
- Pong **no baseline**: `mean -20.998 | std 0.045`  
- Pong **moving_avg** (GPU): `mean -17.488 | std 2.056`  

Example tail of the training log (Pong, moving_avg):
```text
Episode 1990 | Return: -19.00 | Avg(100): -16.39
Episode 2000 | Return: -14.00 | Avg(100): -16.19
Saved: plots/pong_base-moving_avg_lr-0.001_g-0.99_seed-1_reward_curve.png
Saved policy: checkpoints/pong_base-moving_avg_lr-0.001_g-0.99_seed-1_policy.pt
Rolling out 500 episodes with trained policy...
Rollout mean: -17.488 | std: 2.056
```

The repository's `plots/` directory contains reward curves and 500-episode rollout histograms for each experiment; checkpoints are in `checkpoints/`.

---

## 5) Notes & Tips

- If you see `Namespace ALE not found`, ensure:
  - `pip install "gymnasium[atari]==0.29.1" "ale-py==0.8.1"`
  - `pip install "autorom[accept-rom-license]" && AutoROM --accept-license`
- If you see NumPy 2 warnings with older wheels, pin: `pip install "numpy<2"`
- Deepdish GPU tip: verify CUDA is visible to PyTorch
  ```bash
  python - <<'PY'
  import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
  PY
  ```
- Training time:
  - CartPole (CPU): minutes
  - Pong (GPU recommended): hours for 2k episodes; Pong learning shows a gradual rise from ~-21 toward -10 to -15 with the moving-average baseline.

---

## 6) Repo Structure

```
.
├── HW1_PolicyGradient_PyTorch.py   # main training script
├── README.md                       # this file
├── checkpoints/                    # saved policies (*.pt)
└── plots/                          # reward curves & rollout histograms
```

---

## 7) Attribution
- Gymnasium & ALE maintainers (Farama Foundation)
- PyTorch team


