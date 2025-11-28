# hw4_q1.py

import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import yaml
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


# ============================================================
# Generic loader: train_data.npy / test_data.npy -> (X, y)
# ============================================================

def load_X_y_from_npy(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load EMNIST-like data from train_data.npy / test_data.npy and return (X, y).

    Handles your actual format:
    - np.ndarray of shape (100,), dtype=object
    - each element is a dict with keys like 'images' and 'labels'
    """
    arr = np.load(path, allow_pickle=True)
    print(
        f"[load_X_y_from_npy] Loaded {path}: "
        f"type={type(arr)}, shape={getattr(arr, 'shape', None)}, "
        f"dtype={getattr(arr, 'dtype', None)}"
    )

    # ---- Case 1: 1D object array of dicts (your case) ----
    if (
        isinstance(arr, np.ndarray)
        and arr.ndim == 1
        and arr.dtype == object
        and isinstance(arr[0], dict)
    ):
        X_chunks = []
        y_chunks = []
        for i, d in enumerate(arr):
            if not isinstance(d, dict):
                raise ValueError(
                    f"[load_X_y_from_npy] Expected dict at index {i} in {path}, "
                    f"got {type(d)}"
                )

            keys_lower = {k.lower(): k for k in d.keys()}

            # image key candidates
            img_key = None
            for cand in ("images", "image", "x"):
                if cand in keys_lower:
                    img_key = keys_lower[cand]
                    break

            # label key candidates
            lab_key = None
            for cand in ("labels", "label", "y", "targets", "target"):
                if cand in keys_lower:
                    lab_key = keys_lower[cand]
                    break

            if img_key is None or lab_key is None:
                raise ValueError(
                    f"[load_X_y_from_npy] Could not find image/label keys in dict #{i} "
                    f"keys={list(d.keys())}"
                )

            imgs = np.array(d[img_key], dtype=np.float32)
            labs = np.array(d[lab_key], dtype=int)

            # imgs: (n_i, 28, 28) or (n_i, 784)
            if imgs.ndim == 3:
                n_i, h, w = imgs.shape
                imgs = imgs.reshape(n_i, h * w)
            elif imgs.ndim == 2:
                pass
            else:
                raise ValueError(
                    f"[load_X_y_from_npy] Unexpected image array shape {imgs.shape} "
                    f"for dict #{i} in {path}"
                )

            if imgs.shape[0] != labs.shape[0]:
                raise ValueError(
                    f"[load_X_y_from_npy] Mismatch in dict #{i}: "
                    f"imgs.shape[0]={imgs.shape[0]} vs labels.shape[0]={labs.shape[0]}"
                )

            X_chunks.append(imgs)
            y_chunks.append(labs)

        X = np.concatenate(X_chunks, axis=0)
        y = np.concatenate(y_chunks, axis=0)
        print(f"[load_X_y_from_npy] Final X.shape={X.shape}, y.shape={y.shape}")
        return X, y

    # ---- Case 2: scalar object wrapping (X, y) or dict(X=..., y=...) ----
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
        obj = arr.item()
        if isinstance(obj, dict):
            keys = {k.lower(): k for k in obj.keys()}
            if "x" in keys and "y" in keys:
                X = np.array(obj[keys["x"]])
                y = np.array(obj[keys["y"]])
                print(f"[load_X_y_from_npy] Scalar dict: X.shape={X.shape}, y.shape={y.shape}")
                return X, y
            raise ValueError(
                f"[load_X_y_from_npy] Dict in {path} has keys {obj.keys()} but no X/y"
            )

        if isinstance(obj, (list, tuple)) and len(obj) == 2:
            X = np.array(obj[0])
            y = np.array(obj[1])
            print(f"[load_X_y_from_npy] Scalar tuple/list: X.shape={X.shape}, y.shape={y.shape}")
            return X, y

        arr = np.array(obj)

    # ---- Case 3: numeric 2D array (N, 785) with label column 0 ----
    if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 785:
        y = arr[:, 0].astype(int)
        X = arr[:, 1:].astype(np.float32)
        print(f"[load_X_y_from_npy] Flat matrix: X.shape={X.shape}, y.shape={y.shape}")
        return X, y

    # ---- Fallback ----
    raise ValueError(
        f"[load_X_y_from_npy] Unsupported data format in {path}: "
        f"type={type(arr)}, shape={getattr(arr, 'shape', None)}, "
        f"dtype={getattr(arr, 'dtype', None)}"
    )


# ============================================================
# Config / helpers
# ============================================================

@dataclass
class GAConfig:
    population_size: int
    replacement_proportion: float
    num_generations: int
    mutation_rate: float
    learning_rate: float
    epochs: int
    hidden_size: int
    num_classes: int
    batch_size_min: int
    batch_size_max: int
    activation_funcs: List[str]
    random_seed: int
    plot_save_path: str
    hyperparameters_save_path: str
    save_interval: int


def load_config(path: str = "hw4_q1_config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    ga_cfg = cfg["genetic_algorithm"]

    os.makedirs(cfg["plot_save_path"], exist_ok=True)
    os.makedirs(cfg["hyperparameters_save_path"], exist_ok=True)

    gpu_id = cfg.get("gpu_id", None)
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    ga = GAConfig(
        population_size=ga_cfg["population_size"],
        replacement_proportion=ga_cfg["replacement_proportion"],
        num_generations=ga_cfg["num_generations"],
        mutation_rate=ga_cfg["mutation_rate"],
        learning_rate=ga_cfg["learning_rate"],
        epochs=ga_cfg["epochs"],
        hidden_size=ga_cfg["hidden_layer_size"],
        num_classes=ga_cfg["num_classes"],
        batch_size_min=ga_cfg["batch_size_range"][0],
        batch_size_max=ga_cfg["batch_size_range"][1],
        activation_funcs=ga_cfg["activation_funcs"],
        random_seed=cfg["random_seed"],
        plot_save_path=cfg["plot_save_path"],
        hyperparameters_save_path=cfg["hyperparameters_save_path"],
        save_interval=cfg.get("save_interval", 10),
    )
    train_data_path = cfg["data_paths"]["train_data"]
    return ga, train_data_path


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Data + model
# ============================================================

class TwoLayerClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_classes: int, activation: nn.Module):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


def get_activation_module(name: str) -> nn.Module:
    key = name.lower()
    if key == "relu":
        return nn.ReLU()
    if key == "sigmoid":
        return nn.Sigmoid()
    if key == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation function: {name}")


def load_emnist_data(train_data_path: str, seed: int):
    X, y = load_X_y_from_npy(train_data_path)

    # === HW4: Filter to digits only (classes 0-9) ===
    digit_mask = y < 10
    X = X[digit_mask]
    y = y[digit_mask]
    print(f"[load_emnist_data] Filtered to digits only: {len(y)} samples")

    # Map arbitrary label values to 0..num_classes-1
    unique_labels = np.unique(y)
    label_to_idx = {lab: idx for idx, lab in enumerate(unique_labels)}
    y_mapped = np.array([label_to_idx[lab] for lab in y], dtype=int)
    num_classes = len(unique_labels)
    print(f"[load_emnist_data] unique labels={unique_labels}, num_classes={num_classes}")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y_mapped,
        test_size=0.2,
        random_state=seed,
        stratify=y_mapped,
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # return num_classes so GA can build the right-sized output layer
    return X_train, y_train, X_val, y_val, num_classes


# ============================================================
# GA representation / operators
# ============================================================

@dataclass
class Individual:
    batch_size: int
    activation_name: str
    fitness: float = 0.0
    age: int = 0


def random_individual(cfg: GAConfig) -> Individual:
    batch_size = random.randint(cfg.batch_size_min, cfg.batch_size_max)
    activation_name = random.choice(cfg.activation_funcs)
    return Individual(batch_size=batch_size, activation_name=activation_name)


def clamp_batch_size(bs: int, cfg: GAConfig) -> int:
    return max(cfg.batch_size_min, min(cfg.batch_size_max, bs))


def crossover(p1: Individual, p2: Individual, cfg: GAConfig) -> Individual:
    cut = random.randint(0, 2)
    if cut == 0:
        bs = p2.batch_size
        act = p2.activation_name
    elif cut == 1:
        bs = p1.batch_size
        act = p2.activation_name
    else:
        bs = p1.batch_size
        act = p1.activation_name

    return Individual(
        batch_size=clamp_batch_size(bs, cfg),
        activation_name=act,
        fitness=0.0,
        age=0,
    )


def mutate(ind: Individual, cfg: GAConfig):
    if random.random() < cfg.mutation_rate:
        shift = random.randint(-64, 64)
        ind.batch_size = clamp_batch_size(ind.batch_size + shift, cfg)

    if random.random() < cfg.mutation_rate:
        others = [a for a in cfg.activation_funcs if a != ind.activation_name]
        if others:
            ind.activation_name = random.choice(others)


def roulette_selection(population: List[Individual]) -> Individual:
    fitnesses = [max(ind.fitness, 0.0) for ind in population]
    total = sum(fitnesses)
    if total == 0:
        return random.choice(population)
    r = random.random() * total
    acc = 0.0
    for ind, fit in zip(population, fitnesses):
        acc += fit
        if acc >= r:
            return ind
    return population[-1]


# ============================================================
# Evaluation
# ============================================================

def train_and_eval_individual(
    ind: Individual,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    cfg: GAConfig,
    device: torch.device,
) -> float:
    batch_size = int(ind.batch_size)
    act_module = get_activation_module(ind.activation_name)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = TwoLayerClassifier(
        input_dim=X_train.shape[1],
        hidden_size=cfg.hidden_size,
        num_classes=cfg.num_classes,
        activation=act_module,
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    for _ in range(cfg.epochs):
        model.train()
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

    model.eval()
    all_t, all_p = [], []
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)
            out = model(data)
            preds = out.argmax(dim=1)
            all_t.append(target.cpu())
            all_p.append(preds.cpu())

    y_true = torch.cat(all_t).numpy()
    y_pred = torch.cat(all_p).numpy()
    return f1_score(y_true, y_pred, average="macro")


def evaluate_population(
    pop: List[Individual],
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    cfg: GAConfig,
    device: torch.device,
):
    for ind in pop:
        if ind.fitness <= 0.0:
            ind.fitness = train_and_eval_individual(
                ind, X_train, y_train, X_val, y_val, cfg, device
            )


# ============================================================
# GA main loop
# ============================================================

def save_best_hparams(ind: Individual, generation: int, cfg: GAConfig):
    path = os.path.join(
        cfg.hyperparameters_save_path,
        f"best_hyperparameters_gen_{generation}.txt",
    )
    with open(path, "w") as f:
        f.write(f"Batch Size: {ind.batch_size}\n")
        f.write(f"Activation Function: {ind.activation_name}\n")
        f.write(f"F1 Score: {ind.fitness}\n")
    print(f"Saved best hyperparameters (gen {generation}) to {path}")


def save_ga_plot(best_hist, avg_hist, cfg: GAConfig):
    gens = np.arange(1, len(best_hist) + 1)
    plt.figure()
    plt.plot(gens, best_hist, label="Best F1")
    plt.plot(gens, avg_hist, label="Average F1")
    plt.xlabel("Generation")
    plt.ylabel("Macro F1 (validation)")
    plt.title("GA Fitness Over Generations")
    plt.legend()
    plt.grid(True)

    os.makedirs(cfg.plot_save_path, exist_ok=True)
    out_path = os.path.join(cfg.plot_save_path, "ga_fitness_over_generations.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved GA fitness plot to {out_path}")


def run_ga():
    cfg, train_data_path = load_config()
    set_global_seed(cfg.random_seed)
    device = get_device()
    print(f"Using device: {device}")

    X_tr, y_tr, X_val, y_val, num_classes = load_emnist_data(
        train_data_path, cfg.random_seed
    )
    cfg.num_classes = num_classes  # override based on actual labels
    print(f"[run_ga] Using num_classes={cfg.num_classes}")

    X_tr, y_tr = X_tr.to(device), y_tr.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    population = [random_individual(cfg) for _ in range(cfg.population_size)]
    best_hist, avg_hist = [], []

    for gen in range(1, cfg.num_generations + 1):
        print(f"\n=== Generation {gen}/{cfg.num_generations} ===")
        for ind in population:
            ind.age += 1

        evaluate_population(population, X_tr, y_tr, X_val, y_val, cfg, device)

        fitnesses = [ind.fitness for ind in population]
        best_f = max(fitnesses)
        avg_f = float(np.mean(fitnesses))
        best_ind = max(population, key=lambda x: x.fitness)

        best_hist.append(best_f)
        avg_hist.append(avg_f)

        print(
            f"Best F1: {best_f:.4f}, Avg F1: {avg_f:.4f}, "
            f"Best batch={best_ind.batch_size}, act={best_ind.activation_name}"
        )

        if gen % cfg.save_interval == 0 or gen == cfg.num_generations:
            save_best_hparams(best_ind, gen, cfg)

        num_offspring = int(cfg.population_size * cfg.replacement_proportion)
        offspring = []
        for _ in range(num_offspring):
            p1 = roulette_selection(population)
            p2 = roulette_selection(population)
            child = crossover(p1, p2, cfg)
            mutate(child, cfg)
            offspring.append(child)

        combined = population + offspring
        combined.sort(key=lambda ind: (-ind.fitness, ind.age))
        population = combined[:cfg.population_size]

    evaluate_population(population, X_tr, y_tr, X_val, y_val, cfg, device)
    best_ind = max(population, key=lambda x: x.fitness)
    print(
        f"\nFinal best: F1={best_ind.fitness:.4f}, "
        f"batch={best_ind.batch_size}, act={best_ind.activation_name}"
    )
    save_best_hparams(best_ind, cfg.num_generations, cfg)
    save_ga_plot(best_hist, avg_hist, cfg)


if __name__ == "__main__":
    run_ga()