#!/usr/bin/env python3
"""Pure-MLX vs Metal-shader sweep over the local OpenQASM corpus.

For every .qasm file in datasets/qasm/local: parse once, execute the SAME op
stream through Device twice per repeat (pure, then MLXQ_METAL_KERNELS=1,
interleaved), record wall time, paired ratio, and state parity max|delta|.
This exercises the QASM import path end to end: gate-name normalization
(cx->CNOT, cu1/cp->CPHASE, u1/u2/u3, ccx), barrier skipping, and whether the
fusion detectors engage on externally-authored circuits.

Usage: qasm_shader_sweep.py --outdir DIR [--repeats 3] [--max-qubits 27]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mlx.core as mx  # noqa: E402

from mlxq.device import Device  # noqa: E402
from mlxq.qasm import parse_qasm_file  # noqa: E402


def run_once(n: int, ops, metal: bool) -> tuple[float, mx.array]:
    if metal:
        os.environ["MLXQ_METAL_KERNELS"] = "1"
    else:
        os.environ.pop("MLXQ_METAL_KERNELS", None)
    try:
        dev = Device(n)
        t0 = time.perf_counter()
        dev.execute(ops)
        mx.eval(dev.sim.state)
        dt = (time.perf_counter() - t0) * 1000.0
        return dt, dev.sim.state
    finally:
        os.environ.pop("MLXQ_METAL_KERNELS", None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-qubits", type=int, default=27)
    ap.add_argument("--max-cost", type=int, default=int(3e10),
                    help="skip when gates * 2^n exceeds this")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    corpus = sorted((ROOT / "datasets" / "qasm" / "local").glob("*.qasm"))

    rows = []
    skipped = []
    for path in corpus:
        try:
            n, ops = parse_qasm_file(str(path))
        except Exception as exc:
            skipped.append({"file": path.name, "reason": f"parse: {exc}"})
            continue
        if n == 0 or not ops:
            skipped.append({"file": path.name, "reason": "no gates"})
            continue
        if n > args.max_qubits:
            skipped.append({"file": path.name, "reason": f"n={n} too large"})
            continue
        # Cost cap: est full-state passes * state size. Keeps the sweep to
        # minutes; the skipped few are logged, not silently dropped.
        est_cost = len(ops) * (1 << n)
        if est_cost > args.max_cost:
            skipped.append({"file": path.name,
                            "reason": f"cost cap: {len(ops)} gates at n={n}"})
            print(f"{path.name}: SKIP (cost cap, {len(ops)} gates n={n})",
                  flush=True)
            continue
        try:
            # warmup + parity check
            _, s_pure = run_once(n, ops, False)
            _, s_metal = run_once(n, ops, True)
            parity = float(mx.max(mx.abs(s_pure - s_metal)).item().real)
            del s_pure, s_metal
            pures, metals = [], []
            for _ in range(args.repeats):
                dt, s = run_once(n, ops, False)
                del s
                pures.append(dt)
                dt, s = run_once(n, ops, True)
                del s
                metals.append(dt)
            ratios = [p / m for p, m in zip(pures, metals)]
            row = {"file": path.name, "qubits": n, "gates": len(ops),
                   "pure_mean_ms": statistics.fmean(pures),
                   "metal_mean_ms": statistics.fmean(metals),
                   "paired_ratio_mean": statistics.fmean(ratios),
                   "paired_ratio_min": min(ratios),
                   "paired_ratio_max": max(ratios),
                   "parity_max_delta": parity}
            rows.append(row)
            print(f"{path.name}: n={n} gates={len(ops)} "
                  f"pure {row['pure_mean_ms']:.1f} ms metal "
                  f"{row['metal_mean_ms']:.1f} ms ratio "
                  f"{row['paired_ratio_mean']:.2f}x parity {parity:.1e}",
                  flush=True)
        except Exception as exc:
            skipped.append({"file": path.name, "reason": f"execute: {exc}"})
            print(f"{path.name}: SKIP ({exc})", flush=True)

    with (outdir / "qasm_sweep_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (outdir / "qasm_sweep_manifest.json").write_text(json.dumps({
        "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repeats": args.repeats, "files_run": len(rows),
        "skipped": skipped,
    }, indent=2))
    print(f"wrote {outdir} ({len(rows)} circuits, {len(skipped)} skipped)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
