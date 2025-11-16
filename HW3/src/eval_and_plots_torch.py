# eval_and_plots_torch.py
import argparse, os, json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from train_serial_torch import (
    load_clients_from_npy, load_test_from_npy,
    standardize_train_test, to_float_clients, to_float_test,
    remap_labels_global, MLP, evaluate
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_data", required=True)
    ap.add_argument("--test_data", required=True)
    ap.add_argument("--run_dir",   required=True, help="Folder with best_model.pt and history.csv")
    ap.add_argument("--hidden",    default="768,384")
    ap.add_argument("--use_bn",    action="store_true")
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--batch_size", type=int, default=1024)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load & preprocess consistently with training
    clients = load_clients_from_npy(args.train_data)
    Xte, yte = load_test_from_npy(args.test_data)
    clients, yte, n_classes, _ = remap_labels_global(clients, yte)

    if args.standardize:
        clients, Xte = standardize_train_test(clients, Xte)
    else:
        clients = to_float_clients(clients)
        Xte = to_float_test(Xte)

    # Infer input dimension from any non-empty client
    sizes = [len(c[1]) for c in clients]
    idx = next(i for i,s in enumerate(sizes) if s>0)
    in_dim = clients[idx][0].shape[1]

    hidden = [int(x) for x in args.hidden.split(",") if x.strip()]
    model = MLP(in_dim=in_dim, hidden=hidden, out_dim=n_classes, dropout=0.0, use_bn=args.use_bn).to(device)

    # Load best model
    best_path = os.path.join(args.run_dir, "best_model.pt")
    state = torch.load(best_path, map_location="cpu")
    model.load_state_dict(state, strict=True)

    # Compute metrics and predictions
    model.eval()
    # Build a simple loader
    from train_serial_torch import make_loader
    loader = make_loader(Xte, yte, bs=args.batch_size, shuffle=False, drop_last=False)

    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_true.append(yb.cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    # Overall metrics
    acc = (y_pred == y_true).mean()
    print(f"Final Test Accuracy: {acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    fig = plt.figure(figsize=(7,6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.colorbar(); plt.tight_layout()
    cm_path = os.path.join(args.run_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=160)
    print(f"wrote {cm_path}")

    # Classification report
    report = classification_report(y_true, y_pred, labels=np.arange(n_classes), zero_division=0)
    rep_path = os.path.join(args.run_dir, "eval_report.txt")
    with open(rep_path, "w") as f:
        f.write(f"Final Test Accuracy: {acc:.6f}\n\n")
        f.write(report)
    print(f"wrote {rep_path}")

if __name__ == "__main__":
    main()