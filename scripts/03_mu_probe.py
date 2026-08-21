"""Pick mutation rate levels from measurement."""
from pathlib import Path
import numpy as np
from recombpy.simulate import make_config, run_santa
from recombpy.encode import read_alignment, to_binary
from recombpy.distance import hamming_matrix

ROOT = Path(__file__).resolve().parents[1]
CFG, OUT = ROOT / "configs" / "probe", ROOT / "data" / "raw" / "probe"

for mu in [5e-6, 1e-5, 2.5e-5, 5e-5, 1.25e-4, 2.5e-4]:
    prefix = f"mucal_{mu:.2e}"
    cfg = make_config(CFG / f"{prefix}.xml", prefix, mu, None, None, replicates=1)
    run_santa(cfg, OUT)
    arr, _ = read_alignment(OUT / f"{prefix}_1.fasta")
    B, _ = to_binary(arr)
    D = hamming_matrix(B)
    iu = np.triu_indices(100, 1)
    haps = len(np.unique(B, axis=0))
    print(f"mu={mu:.2e} sites={B.shape[1]:4d} haplotypes={haps:3d} "
          f"meanHam={D[iu].mean():7.2f}", flush=True)