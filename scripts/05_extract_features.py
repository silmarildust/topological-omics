"""Sweep scarcity x noise over all 150 replicates -> features.csv + diagrams."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from recombpy.encode import read_alignment, to_binary
from recombpy.distance import hamming_matrix, subsample, add_noise
from recombpy.topology import diagrams, features

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "data" / "raw" / "grid"
PROC = ROOT / "data" / "processed"
DGMS = PROC / "diagrams"
DGMS.mkdir(parents=True, exist_ok=True)

RATIOS = [round(0.1 * i, 1) for i in range(1, 11)]
NOISE_ABS = list(range(0, 101, 5))            # arm A: variance
NOISE_REL = [round(0.02 * i, 2) for i in range(0, 11)]  # arm B: fraction of mean
MASTER_SEED = 20260821

manifest = json.loads((GRID / "manifest.json").read_text())
rows = []

for entry in tqdm(manifest, desc="configs"):
    for rep in range(1, entry["replicates"] + 1):
        fasta = GRID / f"{entry['prefix']}_{rep}.fasta"
        out_npz = DGMS / f"{entry['prefix']}_{rep}.npz"
        arr, _ = read_alignment(fasta)
        B, _ = to_binary(arr)
        D_full = hamming_matrix(B)
        store = {}

        for ratio in RATIOS:
            rng = np.random.default_rng((MASTER_SEED, hash(entry["prefix"]) % 10**6,
                                         rep, int(ratio * 100)))
            D = subsample(D_full, ratio, rng)
            iu = np.triu_indices(D.shape[0], 1)
            mean_d = D[iu].mean() if iu[0].size else 0.0

            plan = ([("abs", v, np.sqrt(v)) for v in NOISE_ABS] +
                    [("rel", c, c * mean_d) for c in NOISE_REL])

            for arm, level, sigma in plan:
                Dn = add_noise(D, sigma, rng)
                dgms = diagrams(Dn, maxdim=2)
                for dim, dgm in enumerate(dgms):
                    betti, mlen, vlen = features(dgm)
                    rows.append({
                        "prefix": entry["prefix"], "mu_level": entry["mu_level"],
                        "mu": entry["mu"], "rho_level": entry["rho_level"],
                        "rho_eff": entry["rho_eff"], "replicate": rep,
                        "sample_ratio": ratio, "noise_arm": arm,
                        "noise_level": level, "sigma": round(sigma, 4),
                        "dim": dim, "betti": betti,
                        "barcode_mean_len": mlen, "barcode_var_len": vlen,
                        "n_seqs": D.shape[0], "n_sites": B.shape[1],
                        "mean_hamming": round(mean_d, 4),
                    })
                    store[f"r{ratio}_{arm}{level}_H{dim}"] = dgm
        np.savez_compressed(out_npz, **store)

df = pd.DataFrame(rows)
df.to_csv(PROC / "features.csv", index=False)
print(f"\n{len(df):,} rows -> {PROC / 'features.csv'}")
print(df.groupby(['mu_level', 'dim'])['betti'].mean().round(1))