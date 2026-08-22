"""
All replication figures use ARM A (absolute noise), because that is TopONE's

protocol. The --arm rel variants of R2 and R4 are the corrected versions and

are the pair that carries the mutation-rate contradiction.

    python scripts/11_figure_topone_replication.py            # all five, arm A
    python scripts/11_figure_topone_replication.py --arm rel  # R2b and R4b

Writes figures/topone_R*.png at 300 dpi.

"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FEATURES = "data/processed/features.csv"
FIGDIR = "figures"

DIM_COLOUR = {0: "#E69F00", 1: "#EC52A7", 2: "#56B4E9"}
DIM_LABEL = {0: "connected components", 1: "loops", 2: "voids"}

MU_ORDER = ["low", "mid", "high"]
MU_TITLE = {
    "low": r"$\mu$ = 1.0$\times$10$^{-5}$",
    "mid": r"$\mu$ = 5.0$\times$10$^{-5}$",
    "high": r"$\mu$ = 2.5$\times$10$^{-4}$",
}

RHO_ORDER = ["r0", "r1", "r2", "r3", "r4"]
RHO_TITLE = {
    "r0": "r0  (rate 0, clonal)",
    "r1": "r1  (rate 0.005)",
    "r2": "r2  (rate 0.015)",
    "r3": "r3  (rate 0.030)",
    "r4": "r4  (rate 0.060)",
}

XLAB = {
    "sample_ratio": "Sampling ratio",
    "abs": "Error variance  ($\\sigma^2$)",
    "rel": "Noise-to-signal ratio  ($\\sigma$ / mean $D$)",
}


def load():
    df = pd.read_csv(FEATURES)
    need = {
        "prefix",
        "replicate",
        "mu_level",
        "rho_level",
        "sample_ratio",
        "noise_arm",
        "noise_level",
        "sigma",
        "dim",
        "betti",
    }
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"features.csv is missing columns: {missing}")
    return df


def noise_slice(df, arm):
    return df[(df["noise_arm"] == arm) & (df["sample_ratio"] == 1.0)]


def scarcity_slice(df):
    return df[(df["sigma"] == 0) & (df["noise_arm"] == "abs")]


def spaghetti(ax, d, xcol, title, legend=False):
    if d.empty:
        ax.text(
            0.5,
            0.5,
            "no data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    for dim in (0, 1, 2):
        dd = d[d["dim"] == dim]

        for _, tr in dd.groupby(["prefix", "replicate"]):
            tr = tr.sort_values(xcol)
            ax.plot(
                tr[xcol],
                tr["betti"],
                color=DIM_COLOUR[dim],
                alpha=0.09,
                lw=0.6,
                zorder=1,
            )

        m = dd.groupby(xcol)["betti"].mean().sort_index()
        ax.plot(
            m.index,
            m.values,
            color=DIM_COLOUR[dim],
            lw=2.2,
            label=DIM_LABEL[dim],
            zorder=3,
        )

    ax.set_title(title, loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#EFEFEF", lw=0.8)

    if legend:
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")


def save(fig, name, suptitle):
    fig.suptitle(suptitle, fontsize=12, y=1.03)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def R1(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    spaghetti(
        axes[0],
        scarcity_slice(df),
        "sample_ratio",
        "Increased sampling",
        legend=True,
    )
    axes[0].set_xlabel(XLAB["sample_ratio"])
    axes[0].set_ylabel("Betti number")

    spaghetti(
        axes[1],
        noise_slice(df, "abs"),
        "noise_level",
        "Increased noise",
    )
    axes[1].set_xlabel(XLAB["abs"])

    save(
        fig,
        "topone_R1_sensitivity.png",
        "Betti numbers vs increasing noise and varying sampling sparsity",
    )


def R2(df, arm="abs"):
    d = noise_slice(df, arm)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)

    for i, (ax, mu) in enumerate(zip(axes, MU_ORDER)):
        spaghetti(
            ax,
            d[d["mu_level"] == mu],
            "noise_level",
            MU_TITLE[mu],
            legend=(i == 0),
        )
        ax.set_xlabel(XLAB[arm])

    axes[0].set_ylabel("Betti number")
    tag = "" if arm == "abs" else "b"

    save(
        fig,
        f"topone_R2{tag}_noise_x_mu_{arm}.png",
        f"Betti numbers vs increasing noise at varying mutation rates "
        f"({'absolute' if arm == 'abs' else 'relative'} noise)",
    )


def R3(df):
    d = scarcity_slice(df)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)

    for i, (ax, mu) in enumerate(zip(axes, MU_ORDER)):
        spaghetti(
            ax,
            d[d["mu_level"] == mu],
            "sample_ratio",
            MU_TITLE[mu],
            legend=(i == 0),
        )
        ax.set_xlabel(XLAB["sample_ratio"])

    axes[0].set_ylabel("Betti number")

    save(
        fig,
        "topone_R3_sparsity_x_mu.png",
        "Betti numbers vs increasing sampled % at varying mutation rates",
    )


def R4(df, arm="abs"):
    d = noise_slice(df, arm)
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.0), sharey=True)

    for i, (ax, rho) in enumerate(zip(axes, RHO_ORDER)):
        spaghetti(
            ax,
            d[d["rho_level"] == rho],
            "noise_level",
            RHO_TITLE[rho],
            legend=(i == 0),
        )
        ax.set_xlabel(XLAB[arm])

    axes[0].set_ylabel("Betti number")
    tag = "" if arm == "abs" else "b"

    save(
        fig,
        f"topone_R4{tag}_noise_x_rho_{arm}.png",
        f"Betti numbers vs increasing noise at varying recombination rates "
        f"({'absolute' if arm == 'abs' else 'relative'} noise)"
    )


def R5(df):
    d = scarcity_slice(df)
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.0), sharey=True)

    for i, (ax, rho) in enumerate(zip(axes, RHO_ORDER)):
        spaghetti(
            ax,
            d[d["rho_level"] == rho],
            "sample_ratio",
            RHO_TITLE[rho],
            legend=(i == 0),
        )
        ax.set_xlabel(XLAB["sample_ratio"])

    axes[0].set_ylabel("Betti number")

    save(
        fig,
        "topone_R5_sparsity_x_rho.png",
        "Betti numbers vs increasing sampled % at varying recombination rates",
    )


def scorecard(df):
    print("\nnumbers behind the replication verdicts")

    sc = scarcity_slice(df)

    print("\nR1a/R3  Spearman(sample_ratio, betti), by dim and mu:")
    for dim in (0, 1, 2):
        vals = []

        for mu in MU_ORDER:
            s = sc[(sc["dim"] == dim) & (sc["mu_level"] == mu)]
            vals.append(
                f"{mu}={s['sample_ratio'].corr(s['betti'], method='spearman'):+.2f}"
            )

        print(f"  H{dim}: " + "  ".join(vals))

    for arm in ("abs", "rel"):
        ns = noise_slice(df, arm)

        print(
            f"\nR1b/R2  Spearman(noise_level, betti), "
            f"arm {arm}, by dim and mu:"
        )

        for dim in (0, 1, 2):
            vals = []

            for mu in MU_ORDER:
                s = ns[(ns["dim"] == dim) & (ns["mu_level"] == mu)]
                vals.append(
                    f"{mu}={s['noise_level'].corr(s['betti'], method='spearman'):+.2f}"
                )

            print(f"  H{dim}: " + "  ".join(vals))

    print(
        "\nR2  mean H1 betti at the noiseless end, by mu "
        "(their claim: mu should not matter):"
    )
    clean = sc[(sc["dim"] == 1) & (sc["sample_ratio"] == 1.0)]
    print(
        clean.groupby("mu_level")["betti"]
        .mean()
        .reindex(MU_ORDER)
        .round(2)
        .to_string()
    )

    print("\nR4  mean H1 betti at the noiseless end, by rho:")
    print(
        clean.groupby("rho_level")["betti"]
        .mean()
        .reindex(RHO_ORDER)
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["abs", "rel", "both"], default="abs")
    args = ap.parse_args()

    df = load()

    if args.arm in ("abs", "both"):
        R1(df)
        R2(df, "abs")
        R3(df)
        R4(df, "abs")
        R5(df)

    if args.arm in ("rel", "both"):
        R2(df, "rel")
        R4(df, "rel")

    scorecard(df)