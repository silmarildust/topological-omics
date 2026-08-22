import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DETECT = "data/processed/detectability_h1.csv"
DOSE = "data/processed/dose_response_h1.csv"
BREAK = "data/processed/detectability_breakdown.csv"

OUT_AUC = "figures/scarcity_auc_curves.png"
OUT_DOSE = "figures/scarcity_dose_curves.png"

FEATURES = ["barcode_mean_len", "barcode_var_len", "betti"]
FEAT_TITLE = {
    "barcode_mean_len": "Barcode mean length",
    "barcode_var_len": "Barcode length variance",
    "betti": "Betti number",
}

MU_ORDER = ["low", "mid", "high"]
MU_COLOUR = {"low": "#E69F00", "mid": "#EC52A7", "high": "#56B4E9"}
MU_LABEL = {
    "low": r"$\mu$ low",
    "mid": r"$\mu$ mid",
    "high": r"$\mu$ high",
}

AUC_TARGET = 0.90
RHO_TARGET = 0.70


def _clean(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(0.1, 1.01, 0.1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#EFEFEF", lw=0.8)


def auc_curves():
    d = pd.read_csv(DETECT)
    # noise_level == 0 exists on the abs arm only (rel duplicate dropped in build)
    d = d[d["noise_level"] == 0]

    try:
        bk = pd.read_csv(BREAK)
        bk = bk[bk["factor"] == "scarcity"]
    except FileNotFoundError:
        bk = None
        print("note: detectability_breakdown.csv not found, skipping markers")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)

    for ax, feat in zip(axes, FEATURES):
        sub = d[d["feature"] == feat]

        for mu in MU_ORDER:
            g = sub[sub["mu_level"] == mu].sort_values("sample_ratio")
            if g.empty:
                continue

            # Hanley-McNeil interval. Wide by construction at n_neg = 10;
            # showing it is the point, not a defect.
            if {"auc_lo", "auc_hi"}.issubset(g.columns):
                ax.fill_between(
                    g["sample_ratio"], g["auc_lo"], g["auc_hi"],
                    color=MU_COLOUR[mu], alpha=0.10, linewidth=0,
                )

            ax.plot(
                g["sample_ratio"], g["auc"],
                color=MU_COLOUR[mu], lw=2, marker="o", ms=5,
                label=MU_LABEL[mu], zorder=3,
            )

            # open marker on the sustained breakdown level
            if bk is not None:
                row = bk[(bk["feature"] == feat) & (bk["mu_level"] == mu)]
                if len(row):
                    lab = str(row.iloc[0]["sustained"])
                    if not lab.startswith("none") and not lab.startswith("fails"):
                        xb = float(lab)
                        yb = g.loc[np.isclose(g["sample_ratio"], xb), "auc"]
                        if len(yb):
                            ax.plot(
                                xb, yb.iloc[0], marker="o", ms=11,
                                mfc="none", mec=MU_COLOUR[mu], mew=2, zorder=4,
                            )

        ax.axhline(AUC_TARGET, color="#333333", lw=1.2)
        ax.axhline(0.5, color="#999999", lw=1, ls="--")
        ax.set_title(FEAT_TITLE[feat], loc="left", fontsize=11)
        _clean(ax, "Sampling ratio", "")
        ax.set_ylim(0.40, 1.03)

    axes[0].set_ylabel("Detection AUC")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    axes[0].annotate(
        "detectability criterion (0.90)", (0.12, AUC_TARGET),
        textcoords="offset points", xytext=(0, 5), fontsize=8, color="#333333",
    )
    axes[0].annotate(
        "chance (0.50)", (0.12, 0.5),
        textcoords="offset points", xytext=(0, 5), fontsize=8, color="#777777",
    )

    fig.suptitle(
        "Detection under sampling scarcity (no noise). "
        "Open circles mark the sustained breakdown level; bands are 95% CI.",
        fontsize=11.5, y=1.03,
    )
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig(OUT_AUC, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_AUC}")


def dose_curves():
    try:
        d = pd.read_csv(DOSE)
    except FileNotFoundError:
        print("note: dose_response_h1.csv not found, skipping T3 companion")
        return

    d = d[d["noise_level"] == 0]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)

    for ax, feat in zip(axes, FEATURES):
        sub = d[d["feature"] == feat]
        for mu in MU_ORDER:
            g = sub[sub["mu_level"] == mu].sort_values("sample_ratio")
            if g.empty:
                continue
            ax.plot(
                g["sample_ratio"], g["rho_all"],
                color=MU_COLOUR[mu], lw=2, marker="o", ms=5,
                label=MU_LABEL[mu], zorder=3,
            )

        ax.axhline(RHO_TARGET, color="#333333", lw=1.2)
        ax.axhline(0.0, color="#999999", lw=1, ls="--")
        ax.set_title(FEAT_TITLE[feat], loc="left", fontsize=11)
        _clean(ax, "Sampling ratio", "")
        ax.set_ylim(-1.05, 1.05)

    axes[0].set_ylabel(r"Spearman $\rho$ (rate ordering)")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    axes[0].annotate(
        "usable dose-response (0.70)", (0.12, RHO_TARGET),
        textcoords="offset points", xytext=(0, 5), fontsize=8, color="#333333",
    )

    fig.suptitle(
        "Rate estimation under sampling scarcity (no noise). "
        "Negative values mean the ordering is reversed.",
        fontsize=11.5, y=1.03,
    )
    fig.tight_layout()
    fig.savefig(OUT_DOSE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DOSE}")


if __name__ == "__main__":
    auc_curves()
    dose_curves()