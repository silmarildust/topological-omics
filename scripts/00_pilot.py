"""Pilot: 3 mutation rates + 1 recombination check. 3 replicates each."""
from pathlib import Path
from recombpy.simulate import make_config, run_santa

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "pilot"
OUT = ROOT / "data" / "raw" / "pilot"

CONDITIONS = [
    ("mu_low",   2.5e-6, None, None),
    ("mu_mid",   2.5e-5, None, None),
    ("mu_high",  2.5e-4, None, None),
    ("rho_high", 2.5e-5, 0.05, 1.44e-3),
]

for prefix, mu, p_dual, p_rec in CONDITIONS:
    cfg = make_config(CFG / f"{prefix}.xml", prefix, mu, p_dual, p_rec,
                      replicates=3)
    print(f"running {prefix} ...", flush=True)
    run_santa(cfg, OUT)
    print(f"  done -> {OUT}")