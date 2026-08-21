"""Does the recombination knob do anything at all?"""
from pathlib import Path
import numpy as np
from ripser import ripser
from recombpy.simulate import make_config, run_santa
from recombpy.encode import read_alignment, to_binary
from recombpy.distance import hamming_matrix

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "probe"
OUT = ROOT / "data" / "raw" / "probe"

GRID = []
for mu in [2.5e-5, 2.5e-4]:
    GRID.append((f"mu{mu:.0e}_clonal", mu, None, None))
    for p_rec in [1e-3, 1e-2, 1e-1, 5e-1]:
        GRID.append((f"mu{mu:.0e}_pr{p_rec:g}", mu, 0.5, p_rec))

for prefix, mu, pd, pr in GRID:
    cfg = make_config(CFG / f"{prefix}.xml", prefix, mu, pd, pr, replicates=1)
    run_santa(cfg, OUT)
    arr, _ = read_alignment(OUT / f"{prefix}_1.fasta")
    B, _ = to_binary(arr)
    D = hamming_matrix(B)
    d = ripser(D, maxdim=2, distance_matrix=True)["dgms"]
    iu = np.triu_indices(D.shape[0], 1)
    print(f"{prefix:22s} sites={B.shape[1]:4d} meanHam={D[iu].mean():7.2f} "
          f"H1={len(d[1]):3d} H2={len(d[2]):3d}", flush=True)