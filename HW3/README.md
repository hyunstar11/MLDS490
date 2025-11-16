# HW3 — Federated Learning + Differential Privacy (FL + DP)

This repository contains a full PyTorch implementation of:

## Part 1 — Federated Learning (FedAvg)
- Serial (single‑process) federated training  
- Optional Ray‑based multi‑process federated training  
- Global + per‑client label histograms  
- Training/test accuracy & loss vs. communication rounds  

## Part 2 — Differential Privacy with Laplace Mechanism
- Inject Laplace noise into client datasets  
- Train Federated EMNIST models under different noise scales `b`  
- Compare privacy–utility tradeoff using accuracy vs. noise scale plots  

All outputs (history.csv, logs, plots, model checkpoints) are stored under `output/part1/` and `output/part2/`.

---

# Repository Structure

```
.
├── config.py
├── requirements.txt
│
├── src/
│   ├── data_utils.py
│   ├── dp_noise.py
│   ├── eval_and_plots_torch.py
│   ├── plot_history.py
│   ├── plots.py
│   ├── summarize_runs.py
│   ├── train_serial_torch.py
│   ├── train_ray_torch.py
│
├── scripts/
│   ├── run_part1.sh
│
└── output/
    ├── part1/
    └── part2/
```

---

# Setup

```bash
# Create virtual environment
python -m venv .venv_hw3
source .venv_hw3/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

# Running Part 1 — Federated Learning (Serial)

```bash
python src/train_serial_torch.py   --train_data train_data.npy   --test_data test_data.npy   --rounds 120   --client_frac 0.10   --local_epochs 5   --batch_size 64   --lr 0.001   --optimizer adamw --weight_decay 5e-5   --dropout 0.1 --hidden 768   --use_bn --standardize   --label_smoothing 0.05   --out_dir output/part1/run1
```

Generate accuracy/loss plots:

```bash
python src/plot_history.py --dirs output/part1/run1 --out_prefix part1
```

---

# Running Part 1 — Ray Version (Optional)

```bash
python src/train_ray_torch.py   --rounds 120   --client_frac 0.10   --local_epochs 5   --out_dir output/part1/ray_run
```

---

# Running Part 2 — Differential Privacy (Laplace Noise)

Example for noise `b = 0.10`:

```bash
B=0.10

python src/train_serial_torch.py   --train_data train_data.npy   --test_data test_data.npy   --rounds 120   --client_frac 0.10   --local_epochs 5   --batch_size 64   --lr 0.001   --optimizer adamw --weight_decay 5e-5   --dropout 0.1 --hidden 128   --use_bn --standardize   --label_smoothing 0.05   --noise_scale $B   --out_dir output/part2/b$B
```

Plot accuracy/loss:

```bash
python src/plot_history.py   --dirs output/part2/b0.10   --out_prefix part2_b010
```

Generate noise‑comparison plot:

```bash
python src/summarize_runs.py   --dirs output/part2/b0.00 output/part2/b0.05 output/part2/b0.10 output/part2/b0.20 output/part2/b0.50   --out part2_noise_comparison.png
```

---

# Outputs

```
output/
  ├── part1/
  │    └── runs...
  └── part2/
       └── b0.xx...
```

Plots include:

- `*_acc.png` — accuracy vs rounds  
- `*_loss.png` — loss vs rounds  
- `part2_noise_comparison.png` — accuracy vs noise scale b  

---

# Notes

- Only PyTorch FL implementations included (TF removed).  
- Scripts run on standard machines or Northwestern Deepdish GPU servers.

