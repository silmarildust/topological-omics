"""Measure what the pilot parameters actually produced."""
import time
from pathlib import Path
import numpy as np
from ripser import ripser
from recombpy.encode import read_alignment, to_binary
from recombpy.distance import hamming_matrix

PILOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "pilot"

for prefix in ["mu_low", "mu_mid", "mu_high", "rho_high"]:
    S, mh, mx, bars, secs = [], [], [], [], []
    for f in sorted(PILOT.glob(f"{prefix}_*.fasta")):
        arr, _ = read_alignment(f)
        B, _ = to_binary(arr)
        D = hamming_matrix(B)
        iu = np.triu_indices(D.shape[0], 1)
        t0 = time.perf_counter()
        dgms = ripser(D, maxdim=2, distance_matrix=True)["dgms"]
        secs.append(time.perf_counter() - t0)
        S.append(B.shape[1]); mh.append(D[iu].mean()); mx.append(D[iu].max())
        bars.append([len(d) for d in dgms])
    b = np.mean(bars, axis=0)
    print(f"{prefix:9s} sites={np.mean(S):7.1f}  meanHam={np.mean(mh):7.2f}  "
          f"maxHam={np.mean(mx):6.1f}  H0/H1/H2={b[0]:.0f}/{b[1]:.0f}/{b[2]:.0f}  "
          f"ripser={np.mean(secs):5.2f}s")