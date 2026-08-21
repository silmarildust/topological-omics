"""Sanity checks on features.csv before running stats."""
from pathlib import Path
import pandas as pd

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
df = pd.read_csv(PROC / "features.csv")
pd.set_option("display.width", 200)

clean = df[(df.sigma == 0) & (df.sample_ratio == 1.0)]
print("=== NOISELESS, FULL SAMPLE: betti by mu x rho ===")
print(clean.pivot_table(index=["mu_level", "rho_level"],
                        columns="dim", values="betti", aggfunc="mean").round(1))

print("\n=== H1 vs noise, full sample, arm A ===")
a = df[(df.dim == 1) & (df.sample_ratio == 1.0) & (df.noise_arm == "abs")]
print(a.pivot_table(index="noise_level", columns="mu_level",
                    values="betti", aggfunc="mean").round(1).iloc[::4])

print("\n=== H1 vs noise, full sample, arm B ===")
b = df[(df.dim == 1) & (df.sample_ratio == 1.0) & (df.noise_arm == "rel")]
print(b.pivot_table(index="noise_level", columns="mu_level",
                    values="betti", aggfunc="mean").round(1).iloc[::2])

print("\n=== H0 vs sampling ratio, noiseless ===")
c = df[(df.dim == 0) & (df.sigma == 0)]
print(c.pivot_table(index="sample_ratio", columns="mu_level",
                    values="betti", aggfunc="mean").round(1))