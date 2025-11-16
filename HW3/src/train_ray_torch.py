import argparse, os, csv, math, random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import ray


# ----------------- Model ----------------- #

class MLP(nn.Module):
    def __init__(self, input_dim, hidden, n_classes, dropout=0.0, use_bn=False):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ----------------- Ray client actor ----------------- #

@ray.remote
class ClientTrainer:
    def __init__(self, X, y, cfg):
        # Keep tensors on CPU inside actors, and copy to make them writable
        self.X = torch.from_numpy(X.copy())
        self.y = torch.from_numpy(y.copy())
        self.cfg = cfg
        self.device = torch.device("cpu")
        self.has_bn = bool(cfg.get("use_bn", False))

    def train(self, global_state):
        cfg = self.cfg

        model = MLP(
            cfg["input_dim"],
            cfg["hidden"],
            cfg["n_classes"],
            dropout=cfg["dropout"],
            use_bn=cfg["use_bn"],
        )
        # global_state is CPU tensors
        model.load_state_dict(global_state)
        model.to(self.device)

        if cfg["label_smoothing"] > 0:
            ce = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
        else:
            ce = nn.CrossEntropyLoss()

        if cfg["optimizer"] == "sgd":
            opt = torch.optim.SGD(
                model.parameters(),
                lr=cfg["lr"],
                momentum=0.9,
                weight_decay=cfg["weight_decay"],
            )
        elif cfg["optimizer"] == "adam":
            opt = torch.optim.Adam(
                model.parameters(),
                lr=cfg["lr"],
                weight_decay=cfg["weight_decay"],
            )
        else:  # adamw
            opt = torch.optim.AdamW(
                model.parameters(),
                lr=cfg["lr"],
                weight_decay=cfg["weight_decay"],
            )

        ds = TensorDataset(self.X, self.y)
        loader = DataLoader(
            ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=False
        )

        model.train()
        for _ in range(cfg["local_epochs"]):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()

                # Handle tiny batch for BatchNorm
                if self.has_bn and xb.size(0) == 1:
                    model.eval()
                    logits = model(xb)
                    model.train()
                else:
                    logits = model(xb)

                loss = ce(logits, yb)
                loss.backward()
                opt.step()

        # return updated weights + sample count (all CPU tensors)
        return {k: v.cpu() for k, v in model.state_dict().items()}, int(self.y.shape[0])


# ----------------- Eval / helpers ----------------- #

def evaluate(model, X, y, device, label_smoothing=0.0, batch_size=1024):
    model.eval()
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    if label_smoothing > 0:
        ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = ce(logits, yb)

            total_loss += float(loss.item()) * xb.size(0)
            preds = logits.argmax(dim=1)
            total_correct += int((preds == yb).sum().item())
            total += xb.size(0)

    return total_correct / total, total_loss / total


def load_federated_data(train_path, test_path):
    """
    train_data.npy: array of length num_clients, each element is a dict:
        client["images"] -> list/array of shape (n_i, 28, 28)
        client["labels"] -> list/array of shape (n_i,)
    test_data.npy: single dict with same keys
    """
    train_raw = np.load(train_path, allow_pickle=True)
    test_raw = np.load(test_path, allow_pickle=True).item()

    X_clients = []
    y_clients = []
    all_X = []

    for client in train_raw:
        imgs = np.asarray(client["images"], dtype="float32") / 255.0
        labels = np.asarray(client["labels"], dtype="int64")

        n = imgs.shape[0]
        X_flat = imgs.reshape(n, -1)
        X_clients.append(X_flat)
        y_clients.append(labels)
        all_X.append(X_flat)

    # global standardization
    X_all = np.concatenate(all_X, axis=0)
    mean = X_all.mean(axis=0, keepdims=True)
    std = X_all.std(axis=0, keepdims=True) + 1e-6

    X_clients = [(X - mean) / std for X in X_clients]

    Xte = np.asarray(test_raw["images"], dtype="float32") / 255.0
    yte = np.asarray(test_raw["labels"], dtype="int64")
    Xte = Xte.reshape(Xte.shape[0], -1)
    Xte = (Xte - mean) / std

    return X_clients, y_clients, Xte, yte


def fedavg(global_state, client_states, client_counts):
    total = float(sum(client_counts))
    new_state = {}
    for k in global_state.keys():
        agg = None
        for st, n in zip(client_states, client_counts):
            w = n / total
            v = st[k].float()
            if agg is None:
                agg = w * v
            else:
                agg += w * v
        new_state[k] = agg
    return new_state


# ----------------- Main ----------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--clients", type=int, default=None)
    parser.add_argument("--client_frac", type=float, default=0.1)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["sgd", "adam", "adamw"])
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--hidden", type=str, default="256,128")
    parser.add_argument("--use_bn", action="store_true")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--out_dir", type=str, default="./outputs_ray_torch")
    parser.add_argument("--ray_cpus", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # seeds
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    print("Loading data...")
    X_clients, y_clients, Xte, yte = load_federated_data(args.train_data, args.test_data)
    n_clients_total = len(X_clients)
    if args.clients is None or args.clients > n_clients_total:
        args.clients = n_clients_total

    # trim if fewer clients requested
    X_clients = X_clients[:args.clients]
    y_clients = y_clients[:args.clients]

    n_classes = int(max(int(y.max()) for y in y_clients) + 1)
    input_dim = X_clients[0].shape[1]
    print(f"Clients: {len(X_clients)} | Total samples: {sum(len(y) for y in y_clients)}")
    print(f"Input dim: {input_dim}, classes: {n_classes}")

    hidden = [int(h) for h in args.hidden.split(",")] if args.hidden else []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = dict(
        input_dim=input_dim,
        n_classes=n_classes,
        hidden=hidden,
        dropout=args.dropout,
        use_bn=args.use_bn,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
    )

    print("Initializing Ray...")
    ray.init(num_cpus=args.ray_cpus, ignore_reinit_error=True)

    print("Creating client actors...")
    actors = [ClientTrainer.remote(Xc, yc, cfg) for Xc, yc in zip(X_clients, y_clients)]

    # global model (GPU for eval, state on CPU for Ray)
    model = MLP(input_dim, hidden, n_classes,
                dropout=args.dropout, use_bn=args.use_bn).to(device)
    global_state = {k: v.cpu() for k, v in model.state_dict().items()}

    history = []
    m_per_round = max(1, int(math.ceil(args.client_frac * len(actors))))
    print(f"Using {m_per_round} clients per round (client_frac={args.client_frac})")

    for r in range(1, args.rounds + 1):
        selected = random.sample(range(len(actors)), m_per_round)
        futures = [actors[i].train.remote(global_state) for i in selected]
        results = ray.get(futures)
        client_states, client_counts = zip(*results)

        # FedAvg on CPU tensors
        global_state = fedavg(global_state, client_states, client_counts)

        # load into eval model
        model.load_state_dict(global_state)
        model.to(device)

        acc, loss = evaluate(
            model, Xte, yte, device,
            label_smoothing=args.label_smoothing
        )
        history.append((r, acc, loss))
        print(f"[Round {r:03d}] test_acc={acc:.4f} loss={loss:.4f}")

    # write history.csv
    csv_path = os.path.join(args.out_dir, "history.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "acc", "loss"])
        for r, acc, loss in history:
            w.writerow([r, acc, loss])
    print("Wrote history to", csv_path)

    ray.shutdown()


if __name__ == "__main__":
    main()