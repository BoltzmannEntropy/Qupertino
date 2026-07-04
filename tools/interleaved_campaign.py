#!/usr/bin/env python3
"""Interleaved same-session benchmark campaign for the QUANTICS revision.

Addresses the round-2 review must-fixes in one run:
  - >= 10 measured repeats per (workload, size, backend) cell
  - osxQuantum, Qiskit Aer CPU, and PennyLane lightning.qubit interleaved
    within each cell (round-robin per repeat), so session-level drift hits
    all backends equally
  - gate-identical PennyLane QFT (explicit ladder; qml.QFT template retired)
  - paired dense-vs-dispatch ablation re-run under the same idle conditions
  - per-run session manifest with a machine-quietness probe

Run only on an otherwise idle machine; the script probes for quietness and
aborts if the reference workload is slower than PROBE_LIMIT_S.
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
sys.path.insert(0, str(ROOT / "tools"))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from mlxq.device import Device  # noqa: E402
import qiskit_aer_baseline as qab  # noqa: E402
import pennylane_baseline as plb  # noqa: E402
from qiskit import QuantumCircuit, transpile  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402

PROBE_LIMIT_S = 0.95  # clean mlxq qft25 runs ~0.69 s; abort if machine is loaded

WORKLOADS = ["qft", "qaoa_ring", "tfim_trotter", "phase_estimation", "grover_proxy", "ghz"]
SIZES = [15, 20, 25]


def mlxq_ops(bench: str, n: int):
    ops = []
    if bench == "qft":
        for j in range(n):
            ops.append({"name": "H", "wires": [j]})
            for k in range(j + 1, n):
                ops.append({"name": "CPHASE", "wires": [k, j], "parameters": [math.pi / (2 ** (k - j))]})
    elif bench == "qaoa_ring":
        for layer in range(6):
            gamma = 0.6 + 0.1 * layer
            beta = 0.4 + 0.05 * layer
            for i in range(n):
                ops.append({"name": "CPHASE", "wires": [i, (i + 1) % n], "parameters": [gamma]})
            for q in range(n):
                ops.append({"name": "RX", "wires": [q], "parameters": [2.0 * beta]})
    elif bench == "tfim_trotter":
        dt = 1.0 / 20
        for _ in range(20):
            for i in range(n - 1):
                ops.append({"name": "ZZPHASE", "wires": [i, i + 1], "parameters": [-dt * 1.0]})
            for q in range(n):
                ops.append({"name": "RX", "wires": [q], "parameters": [2.0 * 0.5 * dt]})
    elif bench == "phase_estimation":
        t = n - 1
        for p in range(n - 1):
            ops.append({"name": "H", "wires": [p]})
        for p in range(n - 1):
            ops.append({"name": "CPHASE", "wires": [p, t], "parameters": [0.4 * (2 ** p)]})
        for j in range(n - 2, -1, -1):
            for k in range(n - 2, j, -1):
                ops.append({"name": "CPHASE", "wires": [k, j], "parameters": [-math.pi / (2 ** (k - j))]})
            ops.append({"name": "H", "wires": [j]})
    elif bench == "grover_proxy":
        for q in range(n):
            ops.append({"name": "H", "wires": [q]})
        for q in range(n):
            ops.append({"name": "H", "wires": [q]})
            ops.append({"name": "X", "wires": [q]})
        for q in range(n - 1):
            ops.append({"name": "CZ", "wires": [q, q + 1]})
        for q in range(n):
            ops.append({"name": "X", "wires": [q]})
            ops.append({"name": "H", "wires": [q]})
    elif bench == "ghz":
        ops = [{"name": "H", "wires": [0]}] + [{"name": "CNOT", "wires": [q, q + 1]} for q in range(n - 1)]
    else:
        raise ValueError(bench)
    return ops


def run_mlxq(bench: str, n: int) -> float:
    ops = mlxq_ops(bench, n)
    dev = Device(n)
    t0 = time.perf_counter()
    dev.execute(ops)
    mx.eval(dev.sim.probabilities_array())
    mx.eval(mx.array([0.0]))
    metal = getattr(mx, "metal", None)
    sync = getattr(metal, "synchronize", None) if metal is not None else None
    if callable(sync):
        sync()
    return (time.perf_counter() - t0) * 1000.0


def run_mlxq_metal(bench: str, n: int) -> float:
    """Same circuits through the hand-tuned Metal shader tier."""
    os.environ["MLXQ_METAL_KERNELS"] = "1"
    try:
        return run_mlxq(bench, n)
    finally:
        os.environ.pop("MLXQ_METAL_KERNELS", None)


_AER_SIM = AerSimulator(method="statevector", device="CPU")
_AER_CIRC_CACHE: dict = {}


def run_aer(bench: str, n: int) -> float:
    key = (bench, n)
    circ = _AER_CIRC_CACHE.get(key)
    if circ is None:
        circ = transpile(qab._build_circuit(QuantumCircuit, bench, n, 6), _AER_SIM, optimization_level=0)
        _AER_CIRC_CACHE[key] = circ
    t0 = time.perf_counter()
    result = _AER_SIM.run(circ, shots=1).result()
    _ = np.asarray(result.get_statevector(circ)).size
    return (time.perf_counter() - t0) * 1000.0


_PL_QNODE_CACHE: dict = {}


def run_pl(bench: str, n: int) -> float:
    key = (bench, n)
    qnode = _PL_QNODE_CACHE.get(key)
    if qnode is None:
        qnode = plb._build_qnode(bench, "lightning.qubit", n, 6)
        _PL_QNODE_CACHE[key] = qnode
    t0 = time.perf_counter()
    _ = np.asarray(qnode()).size
    return (time.perf_counter() - t0) * 1000.0


RUNNERS = {"mlxq": run_mlxq, "mlxq_metal": run_mlxq_metal,
           "aer": run_aer, "pennylane": run_pl}


def quietness_probe() -> float:
    run_mlxq("qft", 25)  # warm
    vals = [run_mlxq("qft", 25) for _ in range(2)]
    return min(vals) / 1000.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument("--ablation-repeats", type=int, default=5)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    probe_s = quietness_probe()
    print(f"quietness probe: qft25 min {probe_s:.3f} s (limit {PROBE_LIMIT_S})", flush=True)
    if probe_s > PROBE_LIMIT_S and not args.skip_probe:
        print("machine not quiet; aborting", flush=True)
        return 3

    manifest = {
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "probe_qft25_s": probe_s,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "interleaving": "round-robin mlxq->mlxq_metal->aer->pennylane within each repeat of each cell",
        "pennylane_qft": "explicit gate-identical ladder (no qml.QFT template)",
        "platform": platform.platform(),
        "python": sys.version,
    }

    raw_path = outdir / "interleaved_raw_runs.csv"
    rows = []
    for bench in WORKLOADS:
        for n in SIZES:
            for backend, fn in RUNNERS.items():
                for _ in range(args.warmups):
                    fn(bench, n)
            for rep in range(1, args.repeats + 1):
                for backend, fn in RUNNERS.items():
                    ms = fn(bench, n)
                    rows.append({"benchmark": bench, "qubits": n, "backend": backend,
                                 "repeat": rep, "wall_ms": ms})
            done = [r for r in rows if r["benchmark"] == bench and r["qubits"] == n]
            for backend in RUNNERS:
                vals = [r["wall_ms"] for r in done if r["backend"] == backend]
                print(f"{bench} {n}q {backend}: mean {statistics.fmean(vals):.1f} ms "
                      f"stdev {statistics.stdev(vals):.1f}", flush=True)
    with raw_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # summary with t-based 95% CI (df = repeats-1)
    from statistics import fmean, stdev
    T_CRIT = {9: 2.262, 4: 2.776, 2: 4.303}
    tcrit = T_CRIT.get(args.repeats - 1, 2.262)
    summary = []
    for bench in WORKLOADS:
        for n in SIZES:
            for backend in RUNNERS:
                vals = [r["wall_ms"] for r in rows
                        if r["benchmark"] == bench and r["qubits"] == n and r["backend"] == backend]
                s = stdev(vals)
                summary.append({"benchmark": bench, "qubits": n, "backend": backend,
                                "repeats": len(vals), "mean_ms": fmean(vals), "stdev_ms": s,
                                "ci95_ms": tcrit * s / math.sqrt(len(vals)),
                                "min_ms": min(vals), "max_ms": max(vals)})
    with (outdir / "interleaved_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # paired dense-vs-dispatch ablation under the same idle conditions
    ablation = {}
    for bench in ("qft", "qaoa_ring"):
        n = 25
        run_mlxq(bench, n)  # warm dispatch
        disp = [run_mlxq(bench, n) / 1000.0 for _ in range(args.ablation_repeats)]
        os.environ["MLXQ_DENSE_ONLY"] = "1"
        run_mlxq(bench, n)  # warm dense
        dense = [run_mlxq(bench, n) / 1000.0 for _ in range(max(2, args.ablation_repeats // 2))]
        del os.environ["MLXQ_DENSE_ONLY"]
        ablation[bench] = {"dispatch_runs_s": disp, "dense_runs_s": dense,
                           "dispatch_mean_s": fmean(disp), "dense_mean_s": fmean(dense),
                           "ratio": fmean(dense) / fmean(disp)}
        print(f"ablation {bench}25: dispatch {fmean(disp):.2f} s, dense {fmean(dense):.2f} s, "
              f"ratio {ablation[bench]['ratio']:.1f}x", flush=True)
    (outdir / "kernel_dispatch_ablation_idle.json").write_text(json.dumps(ablation, indent=2))

    manifest["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (outdir / "campaign_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("campaign complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
