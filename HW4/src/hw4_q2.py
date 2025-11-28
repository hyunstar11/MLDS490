# hw4_q2.py

import os
from typing import Dict, Tuple

import numpy as np
import yaml
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization


# ------------------------------------------------------------
# Shared loader: train_data.npy -> (X, y)
# (same logic as hw4_q1.py)
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Config / device
# ------------------------------------------------------------

def load_config(path: str = "hw4_q2_config.yaml") -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device_from_config(cfg: Dict) -> torch.device:
    requested = cfg.get("device", "cpu")
    if isinstance(requested, str) and "cuda" in requested:
        if torch.cuda.is_available():
            return torch.device(requested)
        else:
            return torch.device("cpu")
    return torch.device(requested)


# ------------------------------------------------------------
# Data + model
# ------------------------------------------------------------

class TwoLayerNet(nn.Module):
    def __init__(self, num_classes: int, activation: nn.Module):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


def get_activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "relu":
        return nn.ReLU()
    if key == "sigmoid":
        return nn.Sigmoid()
    if key == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")


def load_and_split_data(cfg: Dict, device: torch.device):
    data_cfg = cfg["data"]
    split_cfg = cfg["data_split"]

    X, y = load_X_y_from_npy(data_cfg["train_data_path"])

    # === HW4: Filter to digits only (classes 0-9) ===
    digit_mask = y < 10
    X = X[digit_mask]
    y = y[digit_mask]
    print(f"[load_and_split_data] Filtered to digits only: {len(y)} samples")

    unique_labels = np.unique(y)
    cfg["num_classes"] = len(unique_labels)
    print(f"[load_and_split_data] num_classes={cfg['num_classes']}")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
        stratify=y,
    )

    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.long, device=device)
    X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val = torch.tensor(y_val, dtype=torch.long, device=device)

    return X_train, y_train, X_val, y_val


def train_for_f1(
    batch_size: int,
    activation_name: str,
    cfg: Dict,
    device: torch.device,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
) -> float:
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    num_classes = cfg["num_classes"]
    model = TwoLayerNet(num_classes, get_activation(activation_name)).to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        momentum=cfg["optimizer"]["momentum"],
    )
    criterion = nn.CrossEntropyLoss()

    epochs = cfg["training"]["epochs"]
    for _ in range(epochs):
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


# ------------------------------------------------------------
# BO objective
# ------------------------------------------------------------

def make_objective(cfg: Dict, device: torch.device):
    X_train, y_train, X_val, y_val = load_and_split_data(cfg, device)
    act_list = cfg["activation_functions"]

    def objective(batch_size: float, activation_index: float) -> float:
        bs_min, bs_max = cfg["bayesian_opt"]["batch_size"]
        idx_min, idx_max = cfg["bayesian_opt"]["activation_index"]

        b = int(round(batch_size))
        idx = int(round(activation_index))

        b = max(bs_min, min(bs_max, b))
        idx = max(idx_min, min(idx_max, idx))

        act_name = act_list[idx]
        print(f"Evaluating: batch_size={b}, activation={act_name}")
        f1 = train_for_f1(
            batch_size=b,
            activation_name=act_name,
            cfg=cfg,
            device=device,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )
        print(f"Validation F1: {f1:.4f}")
        return f1

    return objective


def save_bo_plot(optimizer: BayesianOptimization, save_dir: str):
    xs = list(range(1, len(optimizer.space.target) + 1))
    ys = optimizer.space.target

    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Validation F1 (macro)")
    plt.title("Bayesian Optimization Progress")
    plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "bo_val_f1_over_iterations.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved BO plot to {out_path}")


def save_best_hparams(best_params: Dict, cfg: Dict, save_dir: str):
    act_list = cfg["activation_functions"]
    bs = int(round(best_params["batch_size"]))
    idx = int(round(best_params["activation_index"]))
    idx = max(0, min(len(act_list) - 1, idx))
    act_name = act_list[idx]

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "best_hyperparameters_BO.txt")
    with open(out_path, "w") as f:
        f.write(f"Batch Size: {bs}\n")
        f.write(f"Activation Function: {act_name}\n")
    print(f"Saved BO best hyperparameters to {out_path}")


def main():
    cfg = load_config()
    device = get_device_from_config(cfg)
    print(f"Using device: {device}")

    objective = make_objective(cfg, device)

    bo_cfg = cfg["bayesian_opt"]
    pbounds = {
        "batch_size": tuple(bo_cfg["batch_size"]),
        "activation_index": tuple(bo_cfg["activation_index"]),
    }

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=bo_cfg["random_state"],
        verbose=bo_cfg["verbose"],
    )

    optimizer.maximize(
        init_points=bo_cfg["init_points"],
        n_iter=bo_cfg["n_iter"],
    )

    best_params = optimizer.max["params"]
    print("Best parameters:", best_params)

    plot_dir = os.path.join("artifacts", "plots")
    hyper_dir = os.path.join("artifacts", "hyperparameters")

    save_bo_plot(optimizer, plot_dir)
    save_best_hparams(best_params, cfg, hyper_dir)


if __name__ == "__main__":
    main()