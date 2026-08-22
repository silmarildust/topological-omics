import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FEATURES = "data/processed/features.csv"
OUT_GRID = "figures/sanity_betti_grid.png"
OUT_FLOOR = "figures/sanity_betti_floor.png"

MU_ORDER = ["low", "mid", "high"]
MU_COLOUR = {"low": "#E69F00", "mid": "#EC52A7", "high": "#56B4E9"}
MU_LABEL = {
    "low": r"$\mu$ = 1.0$\times$10$^{-5}$",
    "mid": r"$\mu$ = 5.0$\times$10$^{-5}$",
    "high": r"$\mu$ = 2.5$\times$10$^{-4}$",
}

RHO_ORDER = ["r0", "r1", "r2", "r3", "r4"]
RHO_EFF = {"r0": 0, "r1": 0.005, "r2": 0.015, "r3": 0.030, "r4": 0.060}

DIM_TITLE = {
    0: "H$_0$: connected components",
    1: "H$_1$: loops",
    2: "H$_2$: voids",
}


def load():
    df = pd.read_csv(FEATURES)
    d = df[
        (df["sigma"] == 0)
        & (df["sample_ratio"] == 1.0)
        & (df["noise_arm"] == "abs")
    ].copy()

    expected = 150 * 3
    if len(d) != expected:
        print(f"WARNING: got {len(d)} rows, expected {expected}. Check filters.")
    return d


def summarise(d):
    g = d.groupby(["dim", "mu_level", "rho_level"])["betti"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["sem"] = out["std"] / np.sqrt(out["count"])
    return out


def build_grid(s):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    x = np.arange(len(RHO_ORDER))
    width = 0.26

    for ax, dim in zip(axes, (0, 1, 2)):
        sub = s[s["dim"] == dim]

        for k, mu in enumerate(MU_ORDER):
            row = (
                sub[sub["mu_level"] == mu]
                .set_index("rho_level")
                .reindex(RHO_ORDER)
            )
            offset = (k - 1) * width
            bars = ax.bar(
                x + offset,
                row["mean"].values,
                width,
                yerr=row["sem"].values,
                capsize=2.5,
                color=MU_COLOUR[mu],
                edgecolor="white",
                linewidth=0.6,
                label=MU_LABEL[mu],
                error_kw=dict(lw=1, ecolor="#444444"),
            )

            if dim in (1, 2):
                v = row["mean"].values[0]
                ax.annotate(
                    f"{v:.1f}",
                    (bars[0].get_x() + bars[0].get_width() / 2, v),
                    textcoords="offset points",
                    xytext=(0, 4),
                    ha="center",
                    fontsize=7.5,
                    color=MU_COLOUR[mu],
                    fontweight="bold",
                )

        ax.set_xticks(x)
        ax.set_xticklabels([f"{r}\n{RHO_EFF[r]:g}" for r in RHO_ORDER])
        ax.set_xlabel("Recombination condition\n(effective rate)")
        ax.set_title(DIM_TITLE[dim], loc="left", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="#EEEEEE", lw=0.8)

    axes[0].set_ylabel("Mean Betti number")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")

    fig.suptitle(
        "Topological feature counts across the simulation grid "
        "(noiseless, full sample, n = 10 per cell)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_GRID), exist_ok=True)
    fig.savefig(OUT_GRID, dpi=300, bbox_inches="tight")
    print(f"wrote {OUT_GRID}")


def build_floor(s):
    sub = s[s["rho_level"] == "r0"]

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    x = np.arange(3)
    width = 0.26

    for k, mu in enumerate(MU_ORDER):
        row = sub[sub["mu_level"] == mu].set_index("dim").reindex([0, 1, 2])
        ax.bar(
            x + (k - 1) * width,
            row["mean"].values,
            width,
            yerr=row["sem"].values,
            capsize=2.5,
            color=MU_COLOUR[mu],
            edgecolor="white",
            linewidth=0.6,
            label=MU_LABEL[mu],
            error_kw=dict(lw=1, ecolor="#444444"),
        )

        for xi, v in zip(x + (k - 1) * width, row["mean"].values):
            ax.annotate(
                f"{v:.1f}",
                (xi, v),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=8,
                color=MU_COLOUR[mu],
            )

    ax.set_xticks(x)
    ax.set_xticklabels(["H$_0$", "H$_1$", "H$_2$"])
    ax.set_ylabel("Mean Betti number")
    ax.set_title(
        "Features present with\nzero recombination (r0)",
        loc="left",
        fontsize=10.5,
    )
    ax.legend(frameon=False, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#EEEEEE", lw=0.8)

    fig.tight_layout()
    fig.savefig(OUT_FLOOR, dpi=300, bbox_inches="tight")
    print(f"wrote {OUT_FLOOR}")


def print_appendix_table(s):
    piv = (
        s.pivot_table(index=["mu_level", "rho_level"], columns="dim", values="mean")
        .reindex(pd.MultiIndex.from_product([MU_ORDER, RHO_ORDER]))
        .round(1)
    )
    piv.columns = ["H0", "H1", "H2"]

    print("\n--- appendix table (markdown) ---\n")
    print("| mu | rho | H0 | H1 | H2 |")
    print("|---|---|---|---|---|")

    for (mu, rho), row in piv.iterrows():
        print(f"| {mu} | {rho} | {row.H0} | {row.H1} | {row.H2} |")


if __name__ == "__main__":
    d = load()
    s = summarise(d)
    build_grid(s)
    build_floor(s)
    print_appendix_table(s)