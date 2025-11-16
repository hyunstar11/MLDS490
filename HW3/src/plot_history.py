# plot_history.py
import argparse, os, csv
import matplotlib.pyplot as plt


def load_history(path):
    rounds, accs, losses = [], [], []

    csv_path = os.path.join(path, "history.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No history.csv in {path}")

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # Be robust: support either "test_acc" or "acc"
        if "test_acc" in fieldnames:
            acc_col = "test_acc"
        elif "acc" in fieldnames:
            acc_col = "acc"
        else:
            raise KeyError(
                f"history.csv in {path} has no 'test_acc' or 'acc' column. "
                f"Columns are: {fieldnames}"
            )

        for row in reader:
            rounds.append(int(row["round"]))
            accs.append(float(row[acc_col]))
            losses.append(float(row["loss"]))

    return rounds, accs, losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dirs",
        nargs="+",
        required=True,
        help="One or more run folders containing history.csv",
    )
    ap.add_argument(
        "--out_prefix",
        default=None,
        help=(
            "Base name for output plots. "
            "If a single dir is given and this is omitted, "
            "plots are saved inside that dir as <basename>_acc.png and _loss.png."
        ),
    )
    args = ap.parse_args()

    multi_run = len(args.dirs) > 1

    # Decide output prefix & paths
    if multi_run:
        # For multiple runs, write plots in current directory
        prefix = args.out_prefix or "part_multi"
        acc_path = f"{prefix}_acc.png"
        loss_path = f"{prefix}_loss.png"
    else:
        # Single run dir: save into that directory
        run_dir = os.path.normpath(args.dirs[0])
        base = os.path.basename(run_dir)
        prefix_name = args.out_prefix or base  # e.g., part2_b010
        prefix = os.path.join(run_dir, prefix_name)
        acc_path = f"{prefix}_acc.png"
        loss_path = f"{prefix}_loss.png"

    # -------- Accuracy plot --------
    plt.figure()
    for d in args.dirs:
        r, a, l = load_history(d)
        label = os.path.basename(os.path.normpath(d))
        plt.plot(r, a, label=label)
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Federated Learning — Accuracy vs Rounds")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(acc_path, bbox_inches="tight", dpi=160)
    print(f"wrote {acc_path}")

    # -------- Loss plot --------
    plt.figure()
    for d in args.dirs:
        r, a, l = load_history(d)
        label = os.path.basename(os.path.normpath(d))
        plt.plot(r, l, label=label)
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title("Federated Learning — Loss vs Rounds")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(loss_path, bbox_inches="tight", dpi=160)
    print(f"wrote {loss_path}")


if __name__ == "__main__":
    main()