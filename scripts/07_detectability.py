"""T1 - detectability surface for H1 loops."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "features.csv"
OUT = ROOT / "data" / "processed" / "detectability_h1.csv"
FIGDIR = ROOT / "figures"

# one test per cell of this grid: 3 mu x 31 noise x 10 ratio = 930
KEYS = ["mu_level", "noise_arm", "noise_level", "sample_ratio"]

# all three scored; betti inverts under noise, bar lengths do not
SCORES = ["betti", "barcode_mean_len", "barcode_var_len"]

AUC_TARGET = 0.90          # AUC at or above this = reliable detection
NAN_AS_ZERO = True         # empty barcode = no loops = lowest score


def _hanley_mcneil(auc, n_neg, n_pos):
    # analytic AUC standard error, Hanley & McNeil 1982
    if not np.isfinite(auc) or n_neg < 2 or n_pos < 2:
        return np.nan
    Q1 = auc / (2 - auc)
    Q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc)
           + (n_neg - 1) * (Q1 - auc ** 2)
           + (n_pos - 1) * (Q2 - auc ** 2)) / (n_neg * n_pos)
    return float(np.sqrt(max(var, 0.0)))


def cell_stats(sub, score):
    # positives = any recombination, negatives = clonal control
    pos = sub.loc[sub.rho_level != "r0", score].to_numpy(float)
    neg = sub.loc[sub.rho_level == "r0", score].to_numpy(float)

    # every key pre-seeded so early returns still have full columns
    out = {
        "n_nan_pos": int(np.isnan(pos).sum()),
        "n_nan_neg": int(np.isnan(neg).sum()),
        "auc": np.nan,
        "auc_se": np.nan,
        "auc_lo": np.nan,
        "auc_hi": np.nan,
        "p_raw": np.nan,
        "thresh_max_r0": np.nan,
        "thresh_mean2sd_r0": np.nan,
    }

    if NAN_AS_ZERO:
        pos = np.nan_to_num(pos, nan=0.0)
        neg = np.nan_to_num(neg, nan=0.0)
    else:
        pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]

    if len(pos) == 0 or len(neg) == 0:
        return out

    # false-positive ceilings: loop count above which you'd call recombination
    out["thresh_max_r0"] = float(neg.max())
    if len(neg) > 1:
        # parametric alternative; max of 10 draws is unstable
        out["thresh_mean2sd_r0"] = float(neg.mean() + 2 * neg.std(ddof=1))

    both = np.concatenate([pos, neg])
    if both.min() == both.max():
        # all values identical -> mannwhitneyu returns nan, would poison BH
        out["auc"], out["p_raw"] = 0.5, 1.0
    else:
        # U / (n_pos * n_neg) IS the AUC; its p equals a permutation test
        U, p = mannwhitneyu(pos, neg, alternative="greater")
        out["auc"] = float(U / (len(pos) * len(neg)))
        out["p_raw"] = float(p)

    # CI computed once at the end; inside a branch it silently stays nan
    se = _hanley_mcneil(out["auc"], len(neg), len(pos))
    out["auc_se"] = se
    if np.isfinite(se):
        out["auc_lo"] = float(max(0.0, out["auc"] - 1.96 * se))
        out["auc_hi"] = float(min(1.0, out["auc"] + 1.96 * se))

    return out


def build(df):
    h1 = df[df.dim == 1].copy()

    # sigma=0 is stored under both arms; keep it on abs only or it double-counts
    h1 = h1[~((h1.noise_arm == "rel") & (h1.noise_level == 0))]

    # common noise axis: mean Hamming spans 8.65 to 305.60 across mu
    h1["nsr"] = h1.sigma / h1.mean_hamming

    # grouped once, reused for all three features
    groups = list(h1.groupby(KEYS, sort=False))

    rows = []
    for score in SCORES:
        for key, sub in groups:
            rows.append({
                "feature": score,
                **dict(zip(KEYS, key)),
                "n_pos": int((sub.rho_level != "r0").sum()),
                "n_neg": int((sub.rho_level == "r0").sum()),
                "sigma": float(sub.sigma.mean()),
                "nsr": float(sub.nsr.mean()),
                **cell_stats(sub, score),
            })

    res = pd.DataFrame(rows)

    # BH within feature: three families of 930, not one pool of 2790
    res["p_bh"] = np.nan
    for score in SCORES:
        m = (res.feature == score) & res.p_raw.notna()
        if m.any():
            res.loc[m, "p_bh"] = multipletests(
                res.loc[m, "p_raw"], method="fdr_bh")[1]

    res["detectable"] = (res.auc >= AUC_TARGET) & (res.p_bh < 0.05)

    return res.sort_values(
        ["feature", "noise_arm", "mu_level", "sample_ratio", "noise_level"]
    ).reset_index(drop=True)


def plot(res):
    import matplotlib
    matplotlib.use("Agg")     # no display on Windows headless runs
    import matplotlib.pyplot as plt

    FIGDIR.mkdir(exist_ok=True)

    arms = {"abs": "noise variance (Arm A)", "rel": "noise coefficient c (Arm B)"}

    for feature in SCORES:
        for arm, xlabel in arms.items():
            sub = res[(res.feature == feature) & (res.noise_arm == arm)]
            if sub.empty:
                continue

            fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
            im = None
            for ax, mu in zip(axes, ["low", "mid", "high"]):
                piv = (sub[sub.mu_level == mu]
                       .pivot_table(index="sample_ratio", columns="noise_level",
                                    values="auc"))
                if piv.empty:
                    continue
                vals = piv.values
                # diverging map centred on 0.5 so inversion reads as colour flip
                im = ax.pcolormesh(piv.columns, piv.index, vals,
                                   vmin=0.0, vmax=1.0, shading="nearest",
                                   cmap="RdBu")
                # white solid = edge of reliable-detection region
                if np.nanmin(vals) < AUC_TARGET < np.nanmax(vals):
                    ax.contour(piv.columns, piv.index, vals,
                               levels=[AUC_TARGET], colors="white",
                               linewidths=2)
                # black dashed = chance line, below it the ordering is inverted
                if np.nanmin(vals) < 0.5 < np.nanmax(vals):
                    ax.contour(piv.columns, piv.index, vals,
                               levels=[0.5], colors="black",
                               linewidths=1, linestyles="--")
                ax.set_title(f"mu = {mu}")
                ax.set_xlabel(xlabel)

            axes[0].set_ylabel("sampling ratio")
            if im is not None:
                fig.colorbar(im, ax=axes, label=f"AUC, H1 {feature}")

            path = FIGDIR / f"detectability_{feature}_{arm}.png"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {path}")


def degradation(res, arm, factor, label):
    # breakdown point + Spearman trend; trend is the powered test, not per-cell
    print(f"\ndegradation with {label}, Arm {arm.upper()}, full sample")
    sub = res[(res.noise_arm == arm) & (res.sample_ratio == 1.0)] \
        if factor == "nsr" else res[res.noise_level == 0]

    for (feat, mu), g in sub.groupby(["feature", "mu_level"]):
        g = g.sort_values(factor)
        if len(g) < 3:
            continue

        # point estimate, not CI bound: +/-0.13 interval trips nothing
        passed = (g.auc >= AUC_TARGET).to_numpy()
        idx = np.where(passed)[0]
        if len(idx) == 0:
            bp = "fails at all levels"
        elif idx[-1] == len(g) - 1:
            # never failed within the grid -> censored, report as a bound
            edge = g[factor].max() if factor == "nsr" else g[factor].min()
            bp = f"none ({'>=' if factor == 'nsr' else '<='} {edge:.3f})"
        else:
            # noise degrades upward, scarcity downward
            step = idx[-1] + 1 if factor == "nsr" else idx[0] - 1
            bp = f"{g[factor].to_numpy()[step]:.3f}"

        rho, p = spearmanr(g[factor], g.auc)
        print(f"  {feat:18s} mu={mu:5s}  breakdown={bp:22s} "
              f"trend rho={rho:+.2f} p={p:.3f}  "
              f"AUC {g.auc.iloc[0]:.2f}->{g.auc.iloc[-1]:.2f} "
              f"(+/-{g.auc_se.mean():.2f})")


def summarize(res):
    print(f"\ncells: {len(res)}  (expect 2790 = 930 x 3)")
    print(f"cell sizes: pos {sorted(res.n_pos.unique())}, "
          f"neg {sorted(res.n_neg.unique())}")
    # sanity check: 0 here means the CI block regressed
    print(f"AUC CIs computed: {int(res.auc_se.notna().sum())} / {len(res)}")

    print("\ndetectable cells by feature:")
    print(res.groupby("feature").detectable.agg(["sum", "count"]))

    # Arm B only: nsr = c exactly, so it's the one clean noise axis
    print("\nHEADLINE: Arm B, full sample, AUC by noise level")
    b = res[(res.noise_arm == "rel") & (res.sample_ratio == 1.0)]
    print(b.pivot_table(index="noise_level", columns=["feature", "mu_level"],
                        values="auc").round(3))

    print("\nclean baseline (no noise, full sample)")
    c = res[(res.noise_level == 0) & (res.sample_ratio == 1.0)]
    print(c.pivot_table(index="mu_level", columns="feature",
                        values="auc").round(3))

    print("\nscarcity only (no noise)")
    s = res[res.noise_level == 0]
    print(s.pivot_table(index="sample_ratio", columns=["feature", "mu_level"],
                        values="auc").round(3))

    degradation(res, "rel", "nsr", "noise")
    degradation(res, "abs", "sample_ratio", "scarcity")


def main():
    df = pd.read_csv(FEATURES)
    res = build(df)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    summarize(res)
    plot(res)
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()