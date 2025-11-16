import numpy as np
import torch
from dp_noise import add_laplace_noise
# train_serial_torch.py
import os, argparse, math, random, csv
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Helpers
# -------------------------
def _as_ndarray(x):
    return x if isinstance(x, np.ndarray) else np.asarray(x)

def _unwrap_object_array(a):
    if isinstance(a, np.ndarray) and a.dtype == object and a.shape == (1,):
        try: return a.item()
        except Exception: return a
    return a

def _maybe_item(x):
    if isinstance(x, np.ndarray) and x.dtype == object and x.shape == ():
        try: return x.item()
        except Exception: return x
    return x

def _xy_from_obj(obj) -> Tuple[np.ndarray, np.ndarray]:
    obj = _maybe_item(obj)
    if isinstance(obj, dict):
        kx = next((k for k in obj.keys() if k.lower() in ("images","image","x","data","features")), None)
        ky = next((k for k in obj.keys() if k.lower() in ("labels","label","y","target","targets")), None)
        if kx is None or ky is None: raise ValueError("Dict missing image/label keys.")
        return _as_ndarray(obj[kx]), _as_ndarray(obj[ky])
    if isinstance(obj, (list, tuple)) and len(obj) == 2:
        X, y = obj
        return _as_ndarray(X), _as_ndarray(y)
    raise ValueError("Unknown (X,y) container.")

def _flatten2d(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1) if X.ndim > 2 else X

def _cfloat32(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a.astype(np.float32, copy=False))

# -------------------------
# Robust loaders for HW3 files
# -------------------------
def load_clients_from_npy(train_path: str) -> List[Tuple[np.ndarray,np.ndarray]]:
    a = np.load(train_path, allow_pickle=True)
    a = _unwrap_object_array(a)

    if isinstance(a, np.ndarray) and a.dtype == object and a.ndim == 1 and a.size > 1:
        clients = []
        for i in range(a.size):
            Xi, yi = _xy_from_obj(a[i])
            clients.append((Xi, yi))
        return clients

    X, y = _xy_from_obj(a)  # fallback: monolithic
    return split_into_clients(X, y, n_clients=100, seed=42)

def load_test_from_npy(test_path: str) -> Tuple[np.ndarray,np.ndarray]:
    b = np.load(test_path, allow_pickle=True)
    b = _unwrap_object_array(b)
    Xte, yte = _xy_from_obj(b)
    return Xte, yte

def split_into_clients(X: np.ndarray, y: np.ndarray, n_clients: int, seed: int=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    shards = np.array_split(idx, n_clients)
    return [(X[s], y[s]) for s in shards]

def remap_labels_global(clients: List[Tuple[np.ndarray,np.ndarray]], test_y: np.ndarray):
    ys = [c[1].ravel() for c in clients if len(c[1]) > 0]
    ys.append(test_y.ravel())
    all_y = np.concatenate(ys)

    if all_y.dtype.kind in "fc": all_y = np.rint(all_y).astype(np.int64)
    uniq = np.unique(all_y)
    label2new = {int(lbl): i for i, lbl in enumerate(uniq.tolist())}

    def _remap(y):
        y = np.rint(y).astype(np.int64) if y.dtype.kind in "fc" else y.astype(np.int64, copy=False)
        return np.vectorize(lambda t: label2new[int(t)])(y)

    new_clients = [(X, _remap(y)) for (X, y) in clients]
    test_y_new = _remap(test_y)
    K = len(uniq)
    return new_clients, test_y_new, K, label2new

def standardize_train_test(clients, Xte):
    Xs = [_flatten2d(c[0]) for c in clients if len(c[1]) > 0]
    bigX = np.concatenate(Xs, axis=0)
    mean = bigX.mean(axis=0)
    std = bigX.std(axis=0); std[std==0] = 1.0
    new_clients = []
    for Xc, yc in clients:
        X2 = _flatten2d(Xc)
        X2 = _cfloat32((X2 - mean) / std)
        new_clients.append((X2, yc))
    Xte2 = _flatten2d(Xte)
    Xte2 = _cfloat32((Xte2 - mean) / std)
    return new_clients, Xte2

def to_float_clients(clients):
    return [(_cfloat32(_flatten2d(X)), y) for (X, y) in clients]

def to_float_test(Xte):
    return _cfloat32(_flatten2d(Xte))

# -------------------------
# Model
# -------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], out_dim: int, dropout: float=0.0, use_bn: bool=False):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            if use_bn: layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0: layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# -------------------------
# Data
# -------------------------
class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = _cfloat32(_as_ndarray(X))
        self.y = _as_ndarray(y).astype(np.int64, copy=False)
        assert len(self.X) == len(self.y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

def make_loader(X, y, bs, shuffle=True, drop_last=False):
    if len(y) == 0:
        Xd = np.zeros((0, X.shape[1]), dtype=np.float32)
        yd = np.zeros((0,), dtype=np.int64)
        return DataLoader(NumpyDataset(Xd, yd), batch_size=bs, shuffle=False, drop_last=False)
    return DataLoader(NumpyDataset(X, y), batch_size=bs, shuffle=shuffle, drop_last=drop_last)

# -------------------------
# FedAvg
# -------------------------
def local_train(global_state, model_ctor, X, y, local_epochs, args, device):
    if len(y) == 0: return None
    model = model_ctor().to(device)
    model.load_state_dict(global_state, strict=True)

    if args.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing) if args.label_smoothing > 0 else nn.CrossEntropyLoss()
    # BN-safe: drop_last when BN is on
    loader = make_loader(X, y, args.batch_size, shuffle=True, drop_last=args.use_bn)

    model.train()
    for _ in range(local_epochs):
        for xb, yb in loader:
            # BN-safe: skip pathological 1-sample batch if it ever appears
            if args.use_bn and xb.size(0) == 1:
                continue
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = ce(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    st = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return (st, len(y))

@torch.no_grad()
def evaluate(model, Xte, yte, bs, device):
    model.eval()
    loader = make_loader(Xte, yte, bs, shuffle=False, drop_last=False)
    n, correct, total_loss = 0, 0, 0.0
    ce = nn.CrossEntropyLoss(reduction="sum")
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        logits = model(xb)
        total_loss += ce(logits, yb).item()
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        n += len(yb)
    if n == 0: return 0.0, float("nan")
    return correct / n, total_loss / n

@torch.no_grad()
def evaluate_on_clients(model, clients, bs, device):
    """Evaluate model on union of all client training data."""
    model.eval()
    Xs = [c[0] for c in clients if len(c[1]) > 0]
    ys = [c[1] for c in clients if len(c[1]) > 0]
    if not Xs: return 0.0, float("nan")
    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    acc, loss = evaluate(model, X_all, y_all, bs, device)
    return acc, loss

def weighted_average_states(collected: List[Tuple[Dict[str,torch.Tensor], int]]):
    total = sum(n for _, n in collected)
    base = {k: v.clone() for k, v in collected[0][0].items()}
    for k in base.keys():
        if torch.is_floating_point(base[k]):
            base[k].mul_(collected[0][1] / total)
            for st, n_i in collected[1:]:
                base[k].add_(st[k] * (n_i / total))
        else:
            base[k] = collected[0][0][k]
    return base

# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_data", required=True)
    p.add_argument("--test_data",  required=True)
    p.add_argument("--train_labels", default=None)
    p.add_argument("--test_labels",  default=None)

    p.add_argument("--clients", type=int, default=None)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--client_frac", type=float, default=0.5)
    p.add_argument("--local_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optimizer", choices=["sgd","adam","adamw"], default="adam")
    p.add_argument("--weight_decay", type=float, default=0.0)

    p.add_argument("--hidden", type=str, default="256,128")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--use_bn", action="store_true")
    p.add_argument("--standardize", action="store_true")
    p.add_argument("--label_smoothing", type=float, default=0.0)

    p.add_argument("--step_milestones", type=str, default="")
    p.add_argument("--step_gamma", type=float, default=0.5)

    p.add_argument("--noise_scale", type=float, default=0.0,
                   help="Laplace noise scale b for DP (0 = no noise)")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="./outputs_torch/run")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    clients = load_clients_from_npy(args.train_data)
    Xte, yte = load_test_from_npy(args.test_data)

    if args.clients is not None and len(clients) != args.clients and len(clients) == 1:
        X, y = clients[0]
        clients = split_into_clients(X, y, n_clients=args.clients, seed=args.seed)

    clients, yte, n_classes, _ = remap_labels_global(clients, yte)

    if args.standardize:
        clients, Xte = standardize_train_test(clients, Xte)
    else:
        clients = to_float_clients(clients)
        Xte = to_float_test(Xte)

    # Apply Laplace noise to each client's local data for DP (Part 2)
    if args.noise_scale > 0.0:
        print(f"Applying Laplace noise with scale b={args.noise_scale} to all clients")
        for i in range(len(clients)):
            Xc, yc = clients[i]
            Xc_noisy = add_laplace_noise(Xc, args.noise_scale)
            clients[i] = (Xc_noisy, yc)

    sizes = [len(c[1]) for c in clients]
    n_total = int(sum(sizes))
    in_dim = clients[next(i for i,s in enumerate(sizes) if s>0)][0].shape[1] if any(s>0 for s in sizes) else Xte.shape[1]
    print(f"Clients: {len(clients)} | Total samples: {n_total}")
    if sizes:
        print(f"[client sizes] min={min(sizes)} med={int(np.median(sizes))} max={max(sizes)} empties={(np.array(sizes)==0).sum()}")

    hidden = [int(x) for x in args.hidden.split(",") if x.strip()]
    def model_ctor():
        return MLP(in_dim=in_dim, hidden=hidden, out_dim=n_classes, dropout=args.dropout, use_bn=args.use_bn)

    global_model = model_ctor()
    global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}

    milestones = []
    if args.step_milestones.strip():
        milestones = sorted(int(m.strip()) for m in args.step_milestones.split(",") if m.strip().isdigit())

    best_acc = -1.0
    history = []  # to store per-round metrics for CSV

    for rnd in range(1, args.rounds + 1):
        m_passed = sum(1 for m in milestones if rnd >= m)
        lr_now = args.lr * (args.step_gamma ** m_passed)
        args_this = argparse.Namespace(**vars(args)); args_this.lr = lr_now

        m = max(1, int(math.ceil(args.client_frac * len(clients))))
        sel = np.random.choice(len(clients), size=m, replace=False)
        collected = []
        for idx in sel:
            Xc, yc = clients[idx]
            if len(yc) == 0: continue
            res = local_train(global_state, model_ctor, Xc, yc, args.local_epochs, args_this, device)
            if res is not None: collected.append(res)
        if not collected: continue

        global_state = weighted_average_states(collected)
        global_model.load_state_dict(global_state, strict=True)

        train_acc, train_loss = evaluate_on_clients(global_model.to(device), clients, bs=1024, device=device)
        test_acc, test_loss = evaluate(global_model.to(device), Xte, yte, bs=1024, device=device)
        print(f"[Round {rnd:03d}] train_acc={train_acc:.4f} test_acc={test_acc:.4f} loss={test_loss:.4f}")
        
        history.append({
            "round": rnd,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "loss": test_loss
        })
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(global_state, os.path.join(args.out_dir, "best_model.pt"))

    torch.save(global_state, os.path.join(args.out_dir, "last_model.pt"))
    print(f"Done. Best acc={best_acc:.4f}")

    # Write history to CSV
    csv_path = os.path.join(args.out_dir, "history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "train_acc", "test_acc", "loss"])
        writer.writeheader()
        for row in history:
            writer.writerow(row)
    print(f"Wrote history to {csv_path}")

if __name__ == "__main__":
    main()