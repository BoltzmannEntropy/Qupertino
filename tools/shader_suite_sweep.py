#!/usr/bin/env python3
"""Full-suite pure-MLX vs Metal-shader sweep at a fixed size.

For every gate-based benchmark workload, runs pure-MLX and shader-tier
(MLXQ_METAL_KERNELS=1) executions INTERLEAVED per repeat (pure, metal, pure,
metal, ...) so session drift hits both arms equally, and reports paired
per-repeat ratios. Excluded: steady_state (density-matrix Kraus path, no gate
list) and qft_fft_primitive (primitive reference row, not a gate circuit).

Usage: shader_suite_sweep.py --outdir DIR [--qubits 25] [--repeats 5]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlxq import bench  # noqa: E402

WORKLOADS = [
    "qft", "qft_entangled", "phase_estimation", "phase_estimation_inexact",
    "grover", "ghz", "qaoa", "graph_state", "deutsch_jozsa", "ae",
    "quantum_walk", "quantum_walk_vchain", "wstate",
    "variational", "qcbm", "vqe", "realamp", "su2rand", "qnn",
    "random_circuit", "cuquantum_blueqat",
    "hamiltonian", "tfim_trotter2", "tfim_random_field",
    "heisenberg", "heisenberg_xxz", "heisenberg_random_field",
    "ladder_heisenberg", "long_range_ising",
]


def run_once(fn, n: int, metal: bool) -> float:
    if metal:
        os.environ["MLXQ_METAL_KERNELS"] = "1"
    else:
        os.environ.pop("MLXQ_METAL_KERNELS", None)
    try:
        out = fn(n)
    finally:
        os.environ.pop("MLXQ_METAL_KERNELS", None)
    if "error" in out:
        raise RuntimeError(f"{fn.__name__}: {out['error']}")
    return float(out["wall_ms"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--qubits", type=int, default=25)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    summary = []
    for name in WORKLOADS:
        fn = getattr(bench, f"simulate_{name}")
        # one warmup per arm (kernel compile, cache build)
        run_once(fn, args.qubits, False)
        run_once(fn, args.qubits, True)
        pures, metals = [], []
        for rep in range(1, args.repeats + 1):
            p = run_once(fn, args.qubits, False)
            m = run_once(fn, args.qubits, True)
            pures.append(p)
            metals.append(m)
            rows.append({"benchmark": name, "qubits": args.qubits,
                         "repeat": rep, "pure_ms": p, "metal_ms": m,
                         "ratio": p / m})
        ratios = [p / m for p, m in zip(pures, metals)]
        summary.append({
            "benchmark": name, "qubits": args.qubits,
            "repeats": args.repeats,
            "pure_mean_ms": statistics.fmean(pures),
            "pure_stdev_ms": statistics.stdev(pures) if len(pures) > 1 else 0.0,
            "metal_mean_ms": statistics.fmean(metals),
            "metal_stdev_ms": statistics.stdev(metals) if len(metals) > 1 else 0.0,
            "paired_ratio_mean": statistics.fmean(ratios),
            "paired_ratio_min": min(ratios),
            "paired_ratio_max": max(ratios),
        })
        s = summary[-1]
        print(f"{name} {args.qubits}q: pure {s['pure_mean_ms']:.0f} ms, "
              f"metal {s['metal_mean_ms']:.0f} ms, "
              f"paired ratio {s['paired_ratio_mean']:.2f}x "
              f"[{s['paired_ratio_min']:.2f}, {s['paired_ratio_max']:.2f}]",
              flush=True)

    with (outdir / "shader_sweep_raw.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with (outdir / "shader_sweep_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    manifest = {
        "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "qubits": args.qubits, "repeats": args.repeats,
        "interleaving": "pure/metal alternate within each repeat (paired)",
        "excluded": ["steady_state (no gate list)",
                      "qft_fft_primitive (primitive reference)"],
        "platform": platform.platform(), "python": sys.version,
    }
    (outdir / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
