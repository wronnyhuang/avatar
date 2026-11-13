"""Plot train/val curves of one or more runs (metrics.jsonl) -> PNG.

    python -I tools/train/plot_curves.py --out curves.png name=path/to/metrics.jsonl [...]
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(p):
    tr, va = [], []
    for ln in open(p):
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if "flow" in r and "step" in r:
            tr.append(r)
        if "val_loss" in r:
            va.append(r)
    return tr, va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for spec_ in a.runs:
        name, path = spec_.split("=", 1)
        tr, va = load(path)
        if tr:
            ax[0].plot([r["step"] for r in tr], [r["flow"] for r in tr], alpha=0.6, label=f"{name} train flow")
        if va:
            ax[0].plot([r["step"] for r in va], [r["val_flow"] for r in va], "o-", label=f"{name} val flow")
            ax[1].plot([r["step"] for r in va], [r["val_loss"] for r in va], "o-", label=name)
            ax[2].plot([r["step"] for r in va], [r.get("val_resp", float("nan")) for r in va], "o-", label=f"{name} CFG")
            ax[2].plot([r["step"] for r in va], [r.get("val_resp_g1", float("nan")) for r in va], "x--",
                       label=f"{name} g=1")
    ax[0].set_title("flow loss"); ax[1].set_title("val total loss"); ax[2].set_title("val responsiveness")
    for x in ax:
        x.set_xlabel("step"); x.legend(fontsize=7); x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(a.out, dpi=90)
    print(a.out)


if __name__ == "__main__":
    main()
