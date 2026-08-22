import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FEATURES = "data/processed/features.csv"
OUT = "figures/dose_response_lines_clean.png"

MU_ORDER = ["low", "mid", "high"]
MU_COLOUR = {"low": "#E69F00", "mid": "#EC52A7", "high": "#56B4E9"}
MU_LABEL = {
    "low": r"$\mu$ = 1.0$\times$10$^{-5}$",
    "mid": r"$\mu$ = 5.0$\times$10$^{-5}$",
    "high": r"$\mu$ = 2.5$\times$10$^{-4}$",
}

RHO_ORDER = ["r0", "r1", "r2", "r3", "r4"]
RHO_EFF = [0, 0.005, 0.015, 0.030, 0.060]

PANELS = [
    ("betti", r"H$_1$ Betti number ($\beta_1$)", "Loop count"),
    ("barcode_mean_len", "H$_1$ barcode mean length", "Loop persistence"),
]


def load():
    df = pd.read_csv(FEATURES)
    d = df[
        (df["dim"] == 1)
        & (df["sigma"] == 0)
        & (df["sample_ratio"] == 1.0)
        & (df["noise_arm"] == "abs")
    ].copy()

    n = len(d)
    expected = 150
    if n != expected:
        print(f"WARNING: got {n} rows, expected {expected}. Check the filters.")
    return d


def build():
    d = load()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(len(RHO_ORDER))

    for ax, (col, ylabel, subtitle) in zip(axes, PANELS):
        for mu in MU_ORDER:
            sub = d[d["mu_level"] == mu]
            g = sub.groupby("rho_level")[col]
            means = np.array([g.get_group(r).mean() for r in RHO_ORDER])
            sems = np.array(
                [
                    g.get_group(r).std(ddof=1) / np.sqrt(g.get_group(r).size)
                    for r in RHO_ORDER
                ]
            )

            # Show individual replicates.
            for r_i, r in enumerate(RHO_ORDER):
                vals = g.get_group(r).values
                jitter = (
                    np.random.default_rng(0).random(len(vals)) - 0.5
                ) * 0.12
                ax.scatter(
                    np.full(len(vals), r_i) + jitter,
                    vals,
                    s=6,
                    color=MU_COLOUR[mu],
                    alpha=0.18,
                    linewidths=0,
                    zorder=1,
                )

            ax.errorbar(
                x,
                means,
                yerr=sems,
                marker="o",
                ms=5,
                lw=2,
                capsize=3,
                color=MU_COLOUR[mu],
                label=MU_LABEL[mu],
                zorder=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([f"{r}\n{e:g}" for r, e in zip(RHO_ORDER, RHO_EFF)])
        ax.set_xlabel("Recombination condition\n(effective rate)")
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle, loc="left", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.06)

    axes[0].legend(frameon=False, fontsize=9, loc="upper left")

    fig.suptitle(
        "Detection and quantification are done by different statistics "
        "(noiseless, full sample)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}")

    # check against table 13
    chk = (
        d[d["mu_level"] == "high"]
        .groupby("rho_level")[["betti", "barcode_mean_len"]]
        .mean()
        .reindex(RHO_ORDER)
        .round(2)
    )
    print("\nmu_high group means:")
    print(chk)


if __name__ == "__main__":
    build()