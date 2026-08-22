import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FEATURES = "data/processed/features.csv"
OUT = "figures/scarcity_betti_by_mu.png"
OUT_BY_RHO = "figures/scarcity_betti_by_mu_rho.png"

MU_ORDER = ["low", "mid", "high"]
MU_TITLE = {
    "low": r"$\mu$ = 1.0$\times$10$^{-5}$   (63 segregating sites)",
    "mid": r"$\mu$ = 5.0$\times$10$^{-5}$   (353 sites)",
    "high": r"$\mu$ = 2.5$\times$10$^{-4}$   (930 sites)",
}

DIM_COLOUR = {0: "#1F3B73", 1: "#E8873A", 2: "#2A9D8F"}
DIM_LABEL = {
    0: "connected components (H$_0$)",
    1: "loops (H$_1$)",
    2: "voids (H$_2$)",
}

RHO_ORDER = ["r0", "r1", "r2", "r3", "r4"]


def load():
    df = pd.read_csv(FEATURES)
    d = df[
        (df["sigma"] == 0)
        & (df["noise_arm"] == "abs")
    ].copy()

    expected = 150 * 10 * 3
    if len(d) != expected:
        print(f"WARNING: got {len(d)} rows, expected {expected}. Check the filters.")
    return d


def draw_panel(ax, sub, title):
    for dim in (0, 1, 2):
        dd = sub[sub["dim"] == dim]

        # Plot individual replicates.
        for (_, _), trace in dd.groupby(["prefix", "replicate"]):
            trace = trace.sort_values("sample_ratio")
            ax.plot(
                trace["sample_ratio"],
                trace["betti"],
                color=DIM_COLOUR[dim],
                alpha=0.10,
                lw=0.7,
                zorder=1,
            )

        m = dd.groupby("sample_ratio")["betti"].mean().sort_index()
        ax.plot(
            m.index,
            m.values,
            color=DIM_COLOUR[dim],
            lw=2.4,
            marker="o",
            ms=4,
            label=DIM_LABEL[dim],
            zorder=3,
        )

    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xlabel("Sampling ratio")
    ax.set_xticks(np.arange(0.1, 1.01, 0.1))
    ax.spines[["top", "right"]].set_visible(False)


def build(by_rho=False):
    d = load()

    if not by_rho:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
        for ax, mu in zip(axes, MU_ORDER):
            draw_panel(ax, d[d["mu_level"] == mu], MU_TITLE[mu])
        axes[0].set_ylabel("Betti number")
        axes[0].legend(frameon=False, fontsize=9, loc="upper left")
        fig.suptitle(
            "Topological feature counts against sampling ratio (noiseless, "
            "pooled across recombination levels)",
            fontsize=12,
            y=1.02,
        )
        out = OUT
    else:
        fig, axes = plt.subplots(
            len(MU_ORDER),
            len(RHO_ORDER),
            figsize=(18, 9),
            sharex=True,
            sharey="row",
        )
        for i, mu in enumerate(MU_ORDER):
            for j, rho in enumerate(RHO_ORDER):
                sub = d[(d["mu_level"] == mu) & (d["rho_level"] == rho)]
                draw_panel(axes[i, j], sub, f"{mu} / {rho}")
                if j:
                    axes[i, j].set_ylabel("")
                if i < len(MU_ORDER) - 1:
                    axes[i, j].set_xlabel("")
        axes[0, 0].set_ylabel("Betti number")
        axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
        out = OUT_BY_RHO

    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")

    # check the H0 trend
    print("\nSpearman(sample_ratio, H0 betti) by mu:")
    for mu in MU_ORDER:
        sub = d[(d["mu_level"] == mu) & (d["dim"] == 0)]
        rho = sub["sample_ratio"].corr(sub["betti"], method="spearman")
        print(f"  {mu:>4}: {rho:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--by-rho", action="store_true", help="3x5 grid instead of 1x3")
    args = ap.parse_args()
    build(by_rho=args.by_rho)