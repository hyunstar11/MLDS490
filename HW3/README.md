
# HW3 — Federated Learning & Differential Privacy (FL + DP)

This folder contains a PyTorch-based implementation of:

## Part 1 — Federated Learning (FedAvg)

- Serial (single-process) federated training
- Optional Ray-based multi-process federated training
- Global + per-client label histograms
- Training / test accuracy & loss vs. communication rounds

## Part 2 — Differential Privacy with Laplace Noise

- Adds Laplace noise to client data
- Trains federated EMNIST models under multiple noise scales `b`
- Compares privacy–utility tradeoff using accuracy vs. noise scale plots

All outputs (CSV logs, plots, etc.) are stored under `output/part1/` and `output/part2/`.

---

## 1. Environment Setup

From inside the **HW3** directory:

```bash
# (Optional) create and activate a virtual environment
python -m venv hw3_env
source hw3_env/bin/activate    # macOS / Linux
# .\hw3_env\Scripts\activate  # Windows PowerShell

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

All required packages for both parts are listed in `requirements.txt`.

---

## 2. Project Structure

```text
HW3/
├── config.py                # Global configuration (paths, FL/DP hyperparams, etc.)
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── output/                  # Generated logs and plots
│   ├── part1/
│   └── part2/
├── scripts/
│   └── run_part1.sh         # Convenience script for Part 1 experiments
└── src/
    ├── config.py            # Config helpers (imported by training scripts)
    ├── data_utils.py        # Data loading / preprocessing (EMNIST)
    ├── dp_noise.py          # Differential privacy utilities / Part 2 driver
    ├── eval_and_plots_torch.py
    ├── plot_history.py      # Plot accuracy / loss from history CSVs
    ├── plots.py             # Shared plotting helpers
    ├── summarize_runs.py    # Aggregate multiple runs / summary tables
    ├── train_ray_torch.py   # Ray-based federated training (Part 1, parallel)
    └── train_serial_torch.py# Serial FedAvg training (Part 1 baseline)
```

---

## 3. How to Run Part 1 (Federated Learning)

### 3.1. Recommended: reproduce all Part 1 experiments

From the `HW3/` directory:

```bash
bash scripts/run_part1.sh
```

This script:

- Runs serial FedAvg experiments for multiple `E` (local epochs) and `C` (client fraction) settings
- Optionally runs Ray-based multi-process FedAvg (if Ray is installed)
- Saves logs and plots under `output/part1/` in subfolders such as:
  - `output/part1/serial/`
  - `output/part1/ray/`

Typical outputs include:

- `part1_cf005_e1_acc.png`, `part1_cf005_e1_loss.png`
- `part1_cf005_e3_acc.png`, `part1_cf005_e3_loss.png`
- `part1_cf005_e5_acc.png`, `part1_cf005_e5_loss.png`
- `part1_ray_cf090_e5_acc.png`, `part1_ray_cf090_e5_loss.png`
- Label histograms:
  - `part1_global_label_hist.png`
  - `part1_client_0_4_label_hist.png`

### 3.2. Run a single serial FedAvg experiment manually

From `HW3/`:

```bash
python src/train_serial_torch.py
```

Hyperparameters (number of rounds, local epochs, batch size, etc.) can be edited in:

- `config.py` (top-level), and/or
- `src/config.py` (if the training scripts import from there).

### 3.3. Run a Ray-based FedAvg experiment

If you want to use Ray (parallelism across clients):

```bash
python src/train_ray_torch.py
```

Ray-specific parameters (number of workers, etc.) are also controlled via the config file.

---

## 4. How to Run Part 2 (Differential Privacy)

Part 2 experiments evaluate the effect of Laplace noise scale `b` on model performance.

### 4.1. Run DP experiments across multiple noise scales

From `HW3/`:

```bash
python src/dp_noise.py
```

This script:

- Adds Laplace noise to client data for different values of `b`
- Trains federated models under each noise setting
- Logs accuracy / loss for each `b`
- Writes outputs under `output/part2/` such as:
  - `part2_b010_acc.png`
  - `part2_b010_loss.png`
  - `part2_dp_final_acc_vs_b.png` (accuracy vs. noise scale plot)

Noise scales and DP-related hyperparameters are defined in `config.py`.

---

## 5. Plotting & Evaluation Utilities

### 5.1. Re-plot from existing history files

If you already have `history` CSV files saved under `output/` and want to regenerate plots:

```bash
python src/plot_history.py
```

This script reads the CSV histories (accuracy/loss per round) and regenerates the corresponding plots.

### 5.2. Summarize multiple runs

To aggregate several runs and print/save a summary table:

```bash
python src/summarize_runs.py
```

For example, this can be used to compare different `E`/`C` combinations in Part 1, or different `b` values in Part 2.

---

## 6. Outputs Overview

Key outputs include:

- `output/part1/serial/*.png` — accuracy and loss vs. rounds for serial FedAvg
- `output/part1/ray/*.png` — accuracy and loss vs. rounds for Ray-based FedAvg
- `output/part1/*label_hist*.png` — global and per-client label distributions
- `output/part2/part2_b*_acc.png` — test accuracy vs. rounds for each noise level `b`
- `output/part2/part2_b*_loss.png` — test loss vs. rounds for each `b`
- `output/part2/part2_dp_final_acc_vs_b.png` — final accuracy vs. noise scale plot

---

## 7. Reproducibility

To fully reproduce the main results:

1. Set up the environment with `requirements.txt`
2. Run `bash scripts/run_part1.sh` for Part 1
3. Run `python src/dp_noise.py` for Part 2
4. Use `plot_history.py` and `summarize_runs.py` for additional plots and tables if needed.
