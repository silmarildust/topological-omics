import pandas as pd

df = pd.read_csv("data/processed/features.csv")
h1 = df[(df.dim == 1) & (df.sample_ratio == 1.0)].copy()
h1 = h1[~((h1.noise_arm == "rel") & (h1.noise_level == 0))]
h1["nsr"] = (h1.sigma / h1.mean_hamming).round(3)

for mu in ["low", "mid", "high"]:
    sub = h1[h1.mu_level == mu]
    print(f"\n===== mu = {mu} : mean H1 betti by rho =====")
    print(sub.pivot_table(index=["noise_arm", "nsr"], columns="rho_level",
                          values="betti").round(1))
    print(f"----- mu = {mu} : mean bar length -----")
    print(sub.pivot_table(index=["noise_arm", "nsr"], columns="rho_level",
                          values="barcode_mean_len").round(2))