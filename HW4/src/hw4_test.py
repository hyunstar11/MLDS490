import argparse
import yaml
import numpy as np
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.metrics import f1_score


# ---------------------------
# Data loading utilities
# ---------------------------

def _find_dict_keys(d, x_candidates, y_candidates):
    x_key = None
    y_key = None

    for k in x_candidates:
        if k in d:
            x_key = k
            break

    for k in y_candidates:
        if k in d:
            y_key = k
            break

    if x_key is None or y_key is None:
        raise ValueError(
            f"Could not infer X/y keys in dict. "
            f"Available keys: {list(d.keys())}"
        )
    return x_key, y_key


def _normalize_X(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    # Ensure shape (N, 28, 28)
    if X.ndim == 2 and X.shape[1] == 28 * 28:
        X = X.reshape(-1, 28, 28)
    elif X.ndim == 3 and X.shape[1:] == (28, 28):
        pass
    else:
        raise ValueError(f"Unexpected X shape {X.shape}, expected (N,28,28) or (N,784)")

    X = X.astype(np.float32)
    # Normalize if in [0,255]
    if X.max() > 1.5:
        X = X / 255.0
    return X


def load_X_y_from_npy(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust loader that handles:
      1) Scalar object npy with dict {'images'/'data', 'labels'/'y'}
      2) Object array of length > 1, each element being:
         - dict with image & label arrays
         - tuple/list (X_client, y_client)
    Returns:
      X: (N, 28, 28) float32
      y: (N,) int64
    """
    arr = np.load(path, allow_pickle=True)

    # Case 1: scalar object (e.g., dict)
    if isinstance(arr, np.ndarray) and arr.shape == () and arr.dtype == object:
        obj = arr.item()
        if isinstance(obj, dict):
            x_candidates = ["X", "x", "images", "image", "data"]
            y_candidates = ["y", "labels", "label", "targets", "target"]
            x_key, y_key = _find_dict_keys(obj, x_candidates, y_candidates)
            X = _normalize_X(obj[x_key])
            y = np.asarray(obj[y_key]).astype(np.int64).reshape(-1)
            return X, y
        else:
            raise ValueError(
                f"Don't know how to handle scalar object in {path}: type {type(obj)}"
            )

    # Case 2: object array with multiple elements (federated-style)
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.ndim == 1:
        N = arr.shape[0]
        if N == 0:
            raise ValueError(f"{path} is an empty object array.")

        first = arr[0]

        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []

        if isinstance(first, dict):
            x_candidates = ["X", "x", "images", "image", "data"]
            y_candidates = ["y", "labels", "label", "targets", "target"]
            x_key, y_key = _find_dict_keys(first, x_candidates, y_candidates)

            for i, d in enumerate(arr):
                if not isinstance(d, dict):
                    raise ValueError(
                        f"Inconsistent element type at index {i}: "
                        f"expected dict, got {type(d)}"
                    )
                xs.append(np.asarray(d[x_key]))
                ys.append(np.asarray(d[y_key]))

        elif isinstance(first, (tuple, list)):
            if len(first) < 2:
                raise ValueError(
                    f"Tuple/list element in {path} must have at least 2 items (X, y)."
                )
            for i, elt in enumerate(arr):
                if not isinstance(elt, (tuple, list)) or len(elt) < 2:
                    raise ValueError(
                        f"Inconsistent element at index {i}: "
                        f"expected (X, y) tuple/list, got {elt}"
                    )
                xs.append(np.asarray(elt[0]))
                ys.append(np.asarray(elt[1]))
        else:
            raise ValueError(
                f"Don't know how to split array from {path} with shape {arr.shape}, "
                f"dtype object and element type {type(first)}"
            )

        X = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
        X = _normalize_X(X)
        y = y.astype(np.int64).reshape(-1)
        return X, y

    # Fallback: plain numeric arrays
    if isinstance(arr, np.ndarray) and arr.dtype != object:
        raise ValueError(
            f"{path} appears to be a plain numeric array. "
            f"load_X_y_from_npy expects federated-style or dict-style npy. "
            f"Call it only on files that contain BOTH X and y."
        )

    raise ValueError(
        f"Don't know how to split array from {path} with shape {arr.shape}, "
        f"dtype {arr.dtype}"
    )


# ---------------------------
# Torch dataset / model
# ---------------------------

class EMNISTDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.shape[0] == y.shape[0]
        self.X = torch.from_numpy(X).float()     # (N, 28, 28)
        self.y = torch.from_numpy(y).long()      # (N,)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Flatten for MLP: (28,28) -> (784,)
        x = self.X[idx].view(-1)
        y = self.y[idx]
        return x, y


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, activation: str):
        super().__init__()
        if activation.lower() == "relu":
            act_layer = nn.ReLU()
        elif activation.lower() == "sigmoid":
            act_layer = nn.Sigmoid()
        elif activation.lower() == "tanh":
            act_layer = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_layer,
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------
# Training / evaluation
# ---------------------------

def train_one_model(
    model,
    device,
    train_loader,
    val_loader,
    optimizer,
    epochs: int,
):
    criterion = nn.CrossEntropyLoss()

    train_f1_history = []
    val_f1_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        all_train_preds = []
        all_train_targets = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            preds = torch.argmax(logits, dim=1)
            all_train_preds.extend(preds.detach().cpu().numpy().tolist())
            all_train_targets.extend(yb.detach().cpu().numpy().tolist())

        train_f1 = f1_score(all_train_targets, all_train_preds, average="macro")
        train_f1_history.append(train_f1)

        # Validation
        model.eval()
        val_losses = []
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())

                preds = torch.argmax(logits, dim=1)
                all_val_preds.extend(preds.detach().cpu().numpy().tolist())
                all_val_targets.extend(yb.detach().cpu().numpy().tolist())

        val_f1 = f1_score(all_val_targets, all_val_preds, average="macro")
        val_f1_history.append(val_f1)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {np.mean(train_losses):.4f}, Train F1: {train_f1:.4f} | "
            f"Val Loss: {np.mean(val_losses):.4f}, Val F1: {val_f1:.4f}"
        )

    return train_f1_history, val_f1_history


def evaluate_on_test(model, device, test_loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_targets.extend(yb.detach().cpu().numpy().tolist())

    test_f1 = f1_score(all_targets, all_preds, average="macro")
    print(f"Test macro-F1: {test_f1:.4f}")
    return test_f1


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="hw4_test_config.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device_str = cfg.get("device", "cpu")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_cfg = cfg["data"]
    train_path = data_cfg["train_data_path"]
    test_path = data_cfg["test_data_path"]

    # Load train & test
    print(f"Loading train from {train_path}")
    X_train_full, y_train_full = load_X_y_from_npy(train_path)
    print(f"Loading test from {test_path}")
    X_test, y_test = load_X_y_from_npy(test_path)

    # === HW4: Filter to digits only (classes 0-9) ===
    digit_mask_train = y_train_full < 10
    X_train_full = X_train_full[digit_mask_train]
    y_train_full = y_train_full[digit_mask_train]
    
    digit_mask_test = y_test < 10
    X_test = X_test[digit_mask_test]
    y_test = y_test[digit_mask_test]
    print(f"Filtered to digits only: train={len(y_train_full)}, test={len(y_test)}")

    print("Train X shape:", X_train_full.shape)
    print("Train y shape:", y_train_full.shape, "min/max:", y_train_full.min(), y_train_full.max())
    print("Test X shape:", X_test.shape)
    print("Test y shape:", y_test.shape, "min/max:", y_test.min(), y_test.max())

    # Infer num_classes from labels
    num_classes = int(max(y_train_full.max(), y_test.max()) + 1)
    print(f"Inferred num_classes: {num_classes}")

    # Train/val split
    split_cfg = cfg.get("data_split", {})
    val_size = split_cfg.get("val_size", 0.2)
    random_state = split_cfg.get("random_state", 42)

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    dataset_full = EMNISTDataset(X_train_full, y_train_full)
    n_full = len(dataset_full)
    n_val = int(n_full * val_size)
    n_train = n_full - n_val
    train_dataset, val_dataset = random_split(
        dataset_full,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(random_state),
    )

    train_batch_size = cfg["training"]["batch_size"]
    test_batch_size = cfg.get("testing", {}).get("batch_size", 128)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False)
    test_dataset = EMNISTDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    # Model config
    model_cfg = cfg["model"]
    hidden_dim = model_cfg["hidden_dim"]
    activation = model_cfg["activation"]

    model = MLP(
        input_dim=28 * 28,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        activation=activation,
    ).to(device)

    # Optimizer
    opt_cfg = cfg["optimizer"]
    lr = opt_cfg["lr"]
    momentum = opt_cfg.get("momentum", 0.9)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    epochs = cfg["training"]["epochs"]

    # Train and evaluate
    train_f1_per_epoch, val_f1_per_epoch = train_one_model(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        epochs=epochs,
    )

    test_f1 = evaluate_on_test(model, device, test_loader)

    # Optional save of curves/metrics
    import os
    os.makedirs("artifacts/plots", exist_ok=True)

    plot_cfg = cfg.get("plot", {})
    if plot_cfg.get("enable", True):
        import matplotlib.pyplot as plt

        epochs_idx = list(range(1, len(train_f1_per_epoch) + 1))

        # Training F1 curve
        plt.figure()
        plt.plot(epochs_idx, train_f1_per_epoch, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1 (train)")
        title = plot_cfg.get("title", "Training macro-F1 vs epoch")
        plt.title(title)
        out_path = plot_cfg.get(
            "train_path",
            "artifacts/plots/train_f1_curve.png",
        )
        plt.savefig(out_path, bbox_inches="tight")
        print(f"Saved train F1 curve to {out_path}")

        # (Optional) validation curve too
        if plot_cfg.get("save_val_curve", False):
            plt.figure()
            plt.plot(epochs_idx, val_f1_per_epoch, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Macro F1 (val)")
            val_title = plot_cfg.get(
                "val_title",
                "Validation macro-F1 vs epoch",
            )
            plt.title(val_title)
            val_out_path = plot_cfg.get(
                "val_path",
                "artifacts/plots/val_f1_curve.png",
            )
            plt.savefig(val_out_path, bbox_inches="tight")
            print(f"Saved val F1 curve to {val_out_path}")

    # Optional: save final test F1 to a txt file
    metrics_cfg = cfg.get("metrics", {})
    if metrics_cfg.get("save_test_f1", False):
        metrics_path = metrics_cfg.get(
            "path",
            "artifacts/hyperparameters/test_f1_from_hw4_test.txt",
        )
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            f.write(f"{test_f1:.6f}\n")
        print(f"Saved test F1 to {metrics_path}")


if __name__ == "__main__":
    main()