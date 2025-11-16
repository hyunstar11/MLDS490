# summarize_runs.py
import json, sys, re
from pathlib import Path
import csv

def parse_cfg_from_dirname(name):
    # tries to extract cf, e, lr or dp noise s
    d = {}
    m = re.search(r'cf([0-9.]+)', name);      d['client_frac'] = float(m.group(1)) if m else None
    m = re.search(r'e([0-9]+)', name);        d['local_epochs'] = int(m.group(1)) if m else None
    m = re.search(r'lr([0-9.]+)', name);      d['lr'] = float(m.group(1)) if m else None
    m = re.search(r's([0-9.]+)$', name);      d['noise_scale'] = float(m.group(1)) if m else None
    return d

rows = []
for hp in sys.argv[1:]:
    p = Path(hp)
    with open(p, "r") as f:
        hist = json.load(f)
    final = hist[-1]
    cfg = parse_cfg_from_dirname(p.parent.name)
    rows.append({
        "run": p.parent.name,
        **cfg,
        "rounds": final["round"],
        "final_acc": round(final["test_acc"], 4),
        "final_loss": round(final["test_loss"], 4),
        "secs_last_round": round(final["secs"], 2)
    })

rows.sort(key=lambda r: (-(r["final_acc"] or 0), r["final_loss"] or 9e9))
out = "part1_summary.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print(f"Wrote {out} with {len(rows)} rows")