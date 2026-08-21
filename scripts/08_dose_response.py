"""T3 - dose-response: do H1 features track recombination RATE, not just presence."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "features.csv"
OUT = ROOT / "data" / "processed" / "dose_response_h1.csv"
OUT_MU = ROOT / "data" / "processed" / "mu_effect_h1.csv"
FIGDIR = ROOT / "figures"

KEYS = ["mu_level", "noise_arm", "noise_level", "sample_ratio"]
SCORES = ["betti", "barcode_mean_len", "barcode_var_len"]

RHO_ORDER = ["r0", "r1", "r2", "r3", "r4"]
RHO_STRONG = 0.70          # |Spearman| at or above this = usable ranking


def _epsilon_sq(H, n, k):
    # KW effect size, 0 to 1; analogous to eta-squared
    if n <= k:
        return np.nan
    return float((H - k + 1) / (n - k))


def _spear(x, y):
    # constant input -> spearmanr returns nan, guard so BH stays clean
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def cell_stats(sub, score):
    out = {
        "rho_all": np.nan, "p_all": np.nan,      # r0-r4: dose-response incl. zero dose
        "rho_pos": np.nan, "p_pos": np.nan,      # r1-r4 only: ranking among recombinants
        "kw_H": np.nan, "kw_p": np.nan, "eps_sq": np.nan,
        "n_inversions": np.nan,                  # 0 = group means strictly increasing
        "spread": np.nan,                        # mean(r4) - mean(r0), raw units
    }

    d = sub[["rho_eff", "rho_level", score]].dropna()
    if len(d) < 10:
        return out

    # full dose-response, all five levels
    out["rho_all"], out["p_all"] = _spear(d.rho_eff.to_numpy(),
                                          d[score].to_numpy())

    # recombinants only: can it rank RATE, separate from detecting presence?
    pos = d[d.rho_level != "r0"]
    out["rho_pos"], out["p_pos"] = _spear(pos.rho_eff.to_numpy(),
                                          pos[score].to_numpy())

    # omnibus across the five conditions
    groups = [g[score].to_numpy() for _, g in d.groupby("rho_level")]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) >= 2 and np.ptp(np.concatenate(groups)) > 0:
        H, p = kruskal(*groups)
        out["kw_H"], out["kw_p"] = float(H), float(p)
        out["eps_sq"] = _epsilon_sq(H, len(d), len(groups))

    # monotonicity of the group means across r0 -> r4
    means = d.groupby("rho_level")[score].mean().reindex(RHO_ORDER)
    if means.notna().all():
        diffs = np.diff(means.to_numpy())
        out["n_inversions"] = int((diffs < 0).sum())
        out["spread"] = float(means.iloc[-1] - means.iloc[0])

    return out


def build(df):
    h1 = df[df.dim == 1].copy()
    # sigma=0 duplicated across arms; keep on abs only
    h1 = h1[~((h1.noise_arm == "rel") & (h1.noise_level == 0))]
    h1["nsr"] = h1.sigma / h1.mean_hamming

    groups = list(h1.groupby(KEYS, sort=False))

    rows = []
    for score in SCORES:
        for key, sub in groups:
            rows.append({
                "feature": score,
                **dict(zip(KEYS, key)),
                "n": len(sub),
                "sigma": float(sub.sigma.mean()),
                "nsr": float(sub.nsr.mean()),
                **cell_stats(sub, score),
            })

    res = pd.DataFrame(rows)

    # BH within feature AND within p-column; separate hypothesis families
    for col in ["p_all", "p_pos", "kw_p"]:
        res[col + "_bh"] = np.nan
        for score in SCORES:
            m = (res.feature == score) & res[col].notna()
            if m.any():
                res.loc[m, col + "_bh"] = multipletests(
                    res.loc[m, col], method="fdr_bh")[1]

    res["ranks_dose"] = (res.rho_all >= RHO_STRONG) & (res.p_all_bh < 0.05)
    res["monotone"] = res.n_inversions == 0

    return res.sort_values(
        ["feature", "noise_arm", "mu_level", "sample_ratio", "noise_level"]
    ).reset_index(drop=True)


def mu_effect(df):
    # fourth factor in the Goal 1 statement; tested at clean baseline only
    h1 = df[(df.dim == 1) & (df.sigma == 0) & (df.sample_ratio == 1.0)].copy()

    rows = []
    for score in SCORES:
        for rho_lvl, sub in h1.groupby("rho_level"):
            groups = [g[score].dropna().to_numpy()
                      for _, g in sub.groupby("mu_level")]
            groups = [g for g in groups if len(g) > 0]
            row = {"feature": score, "rho_level": rho_lvl,
                   "n": len(sub), "kw_H": np.nan, "kw_p": np.nan,
                   "eps_sq": np.nan}
            if len(groups) >= 2 and np.ptp(np.concatenate(groups)) > 0:
                H, p = kruskal(*groups)
                row.update(kw_H=float(H), kw_p=float(p),
                           eps_sq=_epsilon_sq(H, len(sub), len(groups)))
            rows.append(row)

    out = pd.DataFrame(rows)
    m = out.kw_p.notna()
    out["kw_p_bh"] = np.nan
    out.loc[m, "kw_p_bh"] = multipletests(out.loc[m, "kw_p"],
                                          method="fdr_bh")[1]
    return out


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
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
                                    values="rho_all"))
                if piv.empty:
                    continue
                # symmetric about 0: negative rho means the ordering is reversed
                im = ax.pcolormesh(piv.columns, piv.index, piv.values,
                                   vmin=-1, vmax=1, shading="nearest",
                                   cmap="RdBu")
                vals = piv.values
                if np.nanmin(vals) < RHO_STRONG < np.nanmax(vals):
                    ax.contour(piv.columns, piv.index, vals,
                               levels=[RHO_STRONG], colors="white",
                               linewidths=2)
                ax.set_title(f"mu = {mu}")
                ax.set_xlabel(xlabel)

            axes[0].set_ylabel("sampling ratio")
            if im is not None:
                fig.colorbar(im, ax=axes,
                             label=f"Spearman rho(rho_eff, {feature})")

            path = FIGDIR / f"dose_response_{feature}_{arm}.png"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {path}")


def summarize(res, mu_res):
    print(f"\ncells: {len(res)}  (expect 2790 = 930 x 3)")
    print(f"Spearmans computed: {int(res.rho_all.notna().sum())} / {len(res)}")

    print("\ncells with usable dose-response (rho >= 0.70, BH p < 0.05):")
    print(res.groupby("feature").ranks_dose.agg(["sum", "count"]))

    print("\ncells with monotone group means (r0 -> r4):")
    print(res.groupby("feature").monotone.agg(["sum", "count"]))

    print("\nCLEAN BASELINE (no noise, full sample)")
    c = res[(res.noise_level == 0) & (res.sample_ratio == 1.0)]
    print(c[["feature", "mu_level", "rho_all", "p_all_bh", "rho_pos",
             "n_inversions", "eps_sq", "spread"]].to_string(index=False))

    print("\nDOSE-RESPONSE vs NOISE (Arm B, full sample) - rho_all")
    b = res[(res.noise_arm == "rel") & (res.sample_ratio == 1.0)]
    print(b.pivot_table(index="noise_level", columns=["feature", "mu_level"],
                        values="rho_all").round(3))

    print("\nDOSE-RESPONSE vs SCARCITY (no noise) - rho_all")
    s = res[res.noise_level == 0]
    print(s.pivot_table(index="sample_ratio", columns=["feature", "mu_level"],
                        values="rho_all").round(3))

    print("\nRANKING AMONG RECOMBINANTS ONLY (r1-r4, clean baseline) - rho_pos")
    print(c.pivot_table(index="mu_level", columns="feature",
                        values="rho_pos").round(3))

    print("\nMUTATION-RATE EFFECT (clean baseline, KW across low/mid/high)")
    print(mu_res.to_string(index=False))


def main():
    df = pd.read_csv(FEATURES)

    res = build(df)
    mu_res = mu_effect(df)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    mu_res.to_csv(OUT_MU, index=False)

    summarize(res, mu_res)
    plot(res)
    print(f"\nwritten -> {OUT}")
    print(f"written -> {OUT_MU}")


if __name__ == "__main__":
    main()