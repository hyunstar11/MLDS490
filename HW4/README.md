# HW4 — AutoML: Hyperparameter Optimization

This folder contains a PyTorch-based implementation of:

## Part 1 — Genetic Algorithm (GA)

- Custom implementation with roulette wheel selection, one-point crossover, and age-based replacement
- Tunes mini-batch size and activation function for EMNIST digit classification
- Tracks fitness (macro F1) over generations

## Part 2 — Bayesian Optimization (BO)

- Uses `bayes_opt` package for hyperparameter tuning
- Optimizes the same hyperparameters as GA
- Compares sample efficiency and final performance with GA

All outputs (hyperparameters, plots, etc.) are stored under `artifacts/hyperparameters/` and `artifacts/plots/`.

---

## 1. Environment Setup

From inside the **mlds_hw4** directory:

```bash
# (Optional) create and activate a virtual environment
python -m venv hw4_env
source hw4_env/bin/activate    # macOS / Linux
# .\hw4_env\Scripts\activate  # Windows PowerShell

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Required packages:
- `torch` (PyTorch)
- `numpy`
- `scikit-learn`
- `matplotlib`
- `pyyaml`
- `bayesian-optimization` (for Part 2)

---

## 2. Project Structure

```text
HW4/
├── README.md                 # This file
├── src/                      # Source code
│   ├── hw4_q1.py             # Genetic Algorithm implementation (Part 1)
│   ├── hw4_q2.py             # Bayesian Optimization implementation (Part 2)
│   └── hw4_test.py           # Final model training & test evaluation
├── configs/                  # Configuration files
│   ├── hw4_q1_config.yaml    # GA configuration (population size, generations, etc.)
│   ├── hw4_q2_config.yaml    # BO configuration (iterations, bounds, etc.)
│   ├── hw4_test_ga.yaml      # Test config using GA-selected hyperparameters
│   └── hw4_test_bo.yaml      # Test config using BO-selected hyperparameters
└── artifacts/                # Output artifacts
    ├── hyperparameters/      # Saved hyperparameters and test F1 scores
    │   ├── best_hyperparameters_gen_10.txt
    │   ├── best_hyperparameters_gen_20.txt
    │   ├── best_hyperparameters_gen_30.txt
    │   ├── best_hyperparameters_BO.txt
    │   ├── ga_test_f1.txt
    │   └── bo_test_f1.txt
    └── plots/                # Generated plots
        ├── ga_fitness_over_generations.png
        ├── ga_train_f1_curve.png
        ├── ga_val_f1_curve.png
        ├── bo_val_f1_over_iterations.png
        ├── bo_train_f1_curve.png
        └── bo_val_f1_curve_epochwise.png
```

**Note**: Data files (`train_data.npy`, `test_data.npy`) are not included in the repository due to size. Place them in the project root when running locally.

---

## 3. Dataset

The dataset is a subset of Federated EMNIST containing **digits only (10 classes: 0-9)**:

- **Training data**: 9,517 samples (split 80% train / 20% validation)
- **Test data**: 1,049 samples
- **Image size**: 28×28 pixels (flattened to 784 features)

The data files (`train_data.npy`, `test_data.npy`) contain federated-style data from 100 users.

---

## 4. How to Run Part 1 (Genetic Algorithm)

### 4.1. Run GA hyperparameter search

From the `HW4/` directory:

```bash
python src/hw4_q1.py
```

This script:
- Initializes a random population of hyperparameter configurations
- Evaluates each individual by training a 2-layer MLP and computing validation F1
- Applies roulette selection, one-point crossover, and mutation
- Uses age-based replacement to maintain population diversity
- Saves best hyperparameters at intervals (gen 10, 20, 30)
- Generates fitness vs. generation plot

**Configuration** (`configs/hw4_q1_config.yaml`):
- `population_size`: 10
- `num_generations`: 30
- `mutation_rate`: 0.2
- `batch_size_range`: [16, 1024]
- `activation_funcs`: [relu, sigmoid, tanh]

**Outputs**:
- `artifacts/hyperparameters/best_hyperparameters_gen_30.txt`
- `artifacts/plots/ga_fitness_over_generations.png`

### 4.2. GA Results

| Hyperparameter | Selected Value |
|----------------|----------------|
| Batch Size | 120 |
| Activation Function | tanh |
| Validation F1 | 0.4923 |

---

## 5. How to Run Part 2 (Bayesian Optimization)

### 5.1. Run BO hyperparameter search

```bash
python src/hw4_q2.py
```

This script:
- Uses Gaussian Process surrogate model with Upper Confidence Bound acquisition
- Explores batch size and activation function space
- Runs 20 iterations (5 random + 15 guided)
- Saves best hyperparameters and progress plot

**Configuration** (`configs/hw4_q2_config.yaml`):
- `n_iter`: 20
- `batch_size_range`: [16, 1024]
- `activation_encoding`: 0=ReLU, 1=Tanh, 2=Sigmoid

**Outputs**:
- `artifacts/hyperparameters/best_hyperparameters_BO.txt`
- `artifacts/plots/bo_val_f1_over_iterations.png`

### 5.2. BO Results

| Hyperparameter | Selected Value |
|----------------|----------------|
| Batch Size | 42 |
| Activation Function | relu |
| Validation F1 | 0.7196 |

---

## 6. Final Model Training & Test Evaluation

After hyperparameter search, train the final model on **combined train+validation data** and evaluate on the test set.

### 6.1. Evaluate GA-selected hyperparameters

```bash
python src/hw4_test.py --config configs/hw4_test_ga.yaml
```

**Output**:
- `artifacts/plots/ga_train_f1_curve.png`
- `artifacts/hyperparameters/ga_test_f1.txt` → **Test F1: 0.8855**

### 6.2. Evaluate BO-selected hyperparameters

```bash
python src/hw4_test.py --config configs/hw4_test_bo.yaml
```

**Output**:
- `artifacts/plots/bo_train_f1_curve.png`
- `artifacts/hyperparameters/bo_test_f1.txt` → **Test F1: 0.8735**

---

## 7. Results Summary

| Method | Best Hyperparameters | Validation F1 | Test F1 |
|--------|---------------------|---------------|---------|
| **Genetic Algorithm** | batch=120, activation=tanh | 0.4923 | **0.8855** |
| **Bayesian Optimization** | batch=42, activation=relu | 0.7196 | 0.8735 |

Both methods achieved ~88% test macro-F1 on the EMNIST digits classification task.

---

## 8. Outputs Overview

### Hyperparameters (`artifacts/hyperparameters/`)

| File | Description |
|------|-------------|
| `best_hyperparameters_gen_10.txt` | GA best at generation 10 |
| `best_hyperparameters_gen_20.txt` | GA best at generation 20 |
| `best_hyperparameters_gen_30.txt` | GA final best hyperparameters |
| `best_hyperparameters_BO.txt` | BO final best hyperparameters |
| `ga_test_f1.txt` | GA test macro-F1 score |
| `bo_test_f1.txt` | BO test macro-F1 score |

### Plots (`artifacts/plots/`)

| File | Description |
|------|-------------|
| `ga_fitness_over_generations.png` | GA: Average & best fitness vs. generation |
| `ga_train_f1_curve.png` | GA: Training F1 vs. epochs (final model) |
| `ga_val_f1_curve.png` | GA: Validation F1 vs. epochs (final model) |
| `bo_val_f1_over_iterations.png` | BO: Validation F1 over optimization iterations |
| `bo_train_f1_curve.png` | BO: Training F1 vs. epochs (final model) |
| `bo_val_f1_curve_epochwise.png` | BO: Validation F1 vs. epochs (final model) |

---

## 9. Reproducibility

To fully reproduce the results:

1. Set up the environment with required packages
2. Run `python hw4_q1.py` for Genetic Algorithm (Part 1)
3. Run `python hw4_q2.py` for Bayesian Optimization (Part 2)
4. Run `python hw4_test.py --config hw4_test_ga.yaml` for GA test evaluation
5. Run `python hw4_test.py --config hw4_test_bo.yaml` for BO test evaluation

---

## 10. Report

The final report will be uploaded on Canvas.

The report includes:
- GA implementation details and results
- BO implementation details and results
- Comparison of GA vs. BO (pros/cons discussion)
- All required plots and metrics
