"""Generate and run the full Goal 1 simulation grid."""
import json, time
from pathlib import Path
from recombpy.simulate import make_config, run_santa

ROOT = Path(__file__).resolve().parents[1]
CFG, OUT = ROOT / "configs" / "grid", ROOT / "data" / "raw" / "grid"

MU = {"low": 1.0e-5, "mid": 5.0e-5, "high": 2.5e-4}
RHO = {"r0": (None, None), "r1": (0.5, 0.01), "r2": (0.5, 0.03),
       "r3": (0.5, 0.06), "r4": (0.5, 0.12)}

manifest = []
for mk, mu in MU.items():
    for rk, (pd_, pr) in RHO.items():
        prefix = f"{mk}_{rk}"
        cfg = make_config(CFG / f"{prefix}.xml", prefix, mu, pd_, pr, replicates=10)
        t0 = time.perf_counter()
        run_santa(cfg, OUT)
        dt = time.perf_counter() - t0
        manifest.append({"prefix": prefix, "mu_level": mk, "mu": mu,
                         "rho_level": rk, "p_dual": pd_, "p_rec": pr,
                         "rho_eff": 0.0 if pd_ is None else pd_ * pr,
                         "replicates": 10, "seconds": round(dt, 1)})
        print(f"{prefix:10s} {dt:6.1f}s", flush=True)

(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"\n{len(manifest)} configs, {len(manifest)*10} FASTA files")