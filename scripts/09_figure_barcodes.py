"""
H1 persistence barcodes, clonal (r0) vs high recombination (r4)

python scripts/09_figure_barcodes.py --pick
    Inspects features.csv and prints, for r0 and r4 at mu_high, the
    replicates whose H1 Betti number sits closest to the group mean.

Writes figures/barcode_example_r0_vs_r4.png at 300 dpi.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FEATURES = "data/processed/features.csv"
DIAGRAMS = "data/processed/diagrams"
OUT = "figures/barcode_example_r0_vs_r4.png"

KEY = "r1.0_abs0_H1"

MU = "high"

PANELS = [
    ("high_r0", 2, "Clonal, no recombination (r0)"),
    ("high_r4", 10, "High recombination (r4, effective rate 0.060)"),
]

BAR_COLOUR = "#D6403A"

def list_diagrams(n=6):
    names = sorted(os.listdir(DIAGRAMS))
    print(f"{len(names)} files in {DIAGRAMS}")
    print("first few:", names[:n])
    sample = np.load(os.path.join(DIAGRAMS, names[0]))
    keys = list(sample.keys())
    print(f"{len(keys)} arrays per file")
    print("keys containing 'abs0_H1':", [k for k in keys if "abs0_H1" in k][:5])
    return names


# choose representative replicates
def pick():
    df = pd.read_csv(FEATURES)
    sel = df[
        (df["dim"] == 1)
        & (df["sigma"] == 0)
        & (df["sample_ratio"] == 1.0)
        & (df["mu_level"] == MU)
        & (df["rho_level"].isin(["r0", "r4"]))
    ]

    print("\nGroup means (targets to match):")
    print(sel.groupby("rho_level")[["betti", "barcode_mean_len"]].mean().round(2))

    for level, g in sel.groupby("rho_level"):
        target = g["betti"].mean()
        g = g.assign(gap=(g["betti"] - target).abs()).sort_values("gap")
        print(f"\n{level}  (target betti = {target:.1f})")
        print(
            g[["replicate", "betti", "barcode_mean_len", "gap"]]
            .head(4)
            .to_string(index=False)
        )


# load and filter one barcode
def load_bars(prefix, replicate, key=KEY):
    """Return an (n_bars, 2) array of [birth, death], filtered and sorted.

    The filter must match the rule used to compute `betti` in features.csv,
    otherwise the bar count in the figure will not match the numbers quoted
    in the text: drop infinite deaths, drop zero-length bars.
    """
    path = os.path.join(DIAGRAMS, f"{prefix}_{replicate}.npz")
    with np.load(path) as d:
        if key not in d:
            raise KeyError(
                f"{key} not in {path}. Available H1 keys at full sample: "
                f"{[k for k in d.keys() if k.startswith('r1.0') and k.endswith('H1')][:5]}"
            )
        bars = np.asarray(d[key], dtype=float)

    if bars.size == 0:
        return np.empty((0, 2))

    bars = bars[np.isfinite(bars[:, 1])]
    bars = bars[bars[:, 1] > bars[:, 0]]
    return bars[np.argsort(bars[:, 0])]

def build():
    loaded = [(load_bars(p, r), title) for p, r, title in PANELS]

    finite_max = [b[:, 1].max() for b, _ in loaded if len(b)]
    xmax = max(finite_max) * 1.05 if finite_max else 1.0
    ymax = max(len(b) for b, _ in loaded)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, sharey=True)

    for ax, (bars, title) in zip(axes, loaded):
        for i, (birth, death) in enumerate(bars):
            ax.hlines(i, birth, death, lw=1.5, color=BAR_COLOUR)

        if len(bars):
            mean_len = (bars[:, 1] - bars[:, 0]).mean()
            stat = f"$\\beta_1$ = {len(bars)},  mean bar length = {mean_len:.2f}"
        else:
            stat = "$\\beta_1$ = 0  (no surviving loops)"

        ax.set_title(f"{title}\n{stat}", loc="left", fontsize=10, pad=8)
        ax.set_ylabel("bar index")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylim(-1.5, ymax + 1.5)
    axes[-1].set_xlim(0, xmax)
    axes[-1].set_xlabel("Filtration scale $\\epsilon$ (Hamming distance)")

    fig.suptitle(
        "H$_1$ persistence barcodes: $\\mu$ high, noiseless, full sample",
        fontsize=12,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}")
    for (bars, title) in loaded:
        print(f"  {title}: {len(bars)} bars")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="inspect .npz naming and keys")
    ap.add_argument("--pick", action="store_true", help="find representative replicates")
    args = ap.parse_args()

    if args.list:
        list_diagrams()
    elif args.pick:
        pick()
    else:
        build()