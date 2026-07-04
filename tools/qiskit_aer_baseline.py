#!/usr/bin/env python3
"""Qiskit Aer CPU baseline capture for the QUANTICS paper.

If qiskit-aer is unavailable, the script still writes a manifest and markdown
artifact recording that fact. This makes baseline coverage auditable rather
than relying on prose in the paper.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import platform
import statistics
import sys
import time
from importlib import metadata
from pathlib import Path


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _stats(values: list[float]) -> dict:
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    stderr = stdev / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "mean_ms": mean,
        "stdev_ms": stdev,
        "stderr_ms": stderr,
        "ci95_ms": 1.96 * stderr,
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _availability() -> dict:
    return {
        "qiskit": {
            "available": importlib.util.find_spec("qiskit") is not None,
            "version": _version("qiskit"),
        },
        "qiskit-aer": {
            "available": importlib.util.find_spec("qiskit_aer") is not None,
            "version": _version("qiskit-aer"),
        },
    }


def _write_unavailable(outdir: Path, availability: dict) -> None:
    payload = {
        "status": "unavailable",
        "reason": "qiskit and/or qiskit-aer is not installed in this Python environment",
        "availability": availability,
        "python": sys.version,
        "platform": platform.platform(),
    }
    (outdir / "qiskit_aer_baseline_manifest.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (outdir / "qiskit_aer_baseline.md").write_text(
        "# Qiskit Aer Baseline\n\n"
        "Status: unavailable in this Python environment.\n\n"
        f"Qiskit available: {availability['qiskit']['available']} "
        f"({availability['qiskit']['version']})\n\n"
        f"Qiskit Aer available: {availability['qiskit-aer']['available']} "
        f"({availability['qiskit-aer']['version']})\n",
        encoding="utf-8",
    )


def _qft_circuit(QuantumCircuit, n: int):
    qc = QuantumCircuit(n)
    for j in range(n):
        qc.h(j)
        for k in range(j + 1, n):
            qc.cp(math.pi / (2 ** (k - j)), k, j)
    qc.save_statevector()
    return qc


def _qaoa_ring_circuit(QuantumCircuit, n: int, layers: int):
    qc = QuantumCircuit(n)
    for layer in range(layers):
        gamma = 0.6 + 0.1 * layer
        beta = 0.4 + 0.05 * layer
        for i in range(n):
            qc.cp(gamma, i, (i + 1) % n)
        for q in range(n):
            qc.rx(2.0 * beta, q)
    qc.save_statevector()
    return qc


def _ghz_circuit(QuantumCircuit, n: int):
    qc = QuantumCircuit(n)
    qc.h(0)
    for q in range(n - 1):
        qc.cx(q, q + 1)
    qc.save_statevector()
    return qc


def _grover_proxy_circuit(QuantumCircuit, n: int):
    """Matches mlxq simulate_grover: uniform init, one diffusion-like step
    with pairwise CZ as the phase-oracle proxy."""
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)
    for q in range(n):
        qc.h(q)
        qc.x(q)
    for q in range(n - 1):
        qc.cz(q, q + 1)
    for q in range(n):
        qc.x(q)
        qc.h(q)
    qc.save_statevector()
    return qc


def _phase_estimation_circuit(QuantumCircuit, n: int):
    """Matches mlxq simulate_phase_estimation (base phase 0.4, target = n-1)."""
    qc = QuantumCircuit(n)
    target = n - 1
    for p in range(n - 1):
        qc.h(p)
    base = 0.4
    for p in range(n - 1):
        qc.cp(base * (2 ** p), p, target)
    for j in range(n - 2, -1, -1):
        for k in range(n - 2, j, -1):
            qc.cp(-math.pi / (2 ** (k - j)), k, j)
        qc.h(j)
    qc.save_statevector()
    return qc


def _tfim_trotter_circuit(QuantumCircuit, n: int, trotter_steps: int = 20,
                          time_total: float = 1.0, J: float = 1.0, h: float = 0.5):
    """Matches mlxq simulate_hamiltonian dense schedule: per step, ZZ phases
    exp(+i*J*dt*ZZ) on each open-boundary bond then RX(2*h*dt) on every site.
    Qiskit RZZ(theta) = exp(-i*theta/2*ZZ), so theta = -2*J*dt."""
    qc = QuantumCircuit(n)
    dt = time_total / float(trotter_steps)
    for _ in range(trotter_steps):
        for i in range(n - 1):
            qc.rzz(-2.0 * J * dt, i, i + 1)
        for q in range(n):
            qc.rx(2.0 * h * dt, q)
    qc.save_statevector()
    return qc


def _build_circuit(QuantumCircuit, benchmark: str, n: int, layers: int):
    if benchmark == "qft":
        return _qft_circuit(QuantumCircuit, n)
    if benchmark == "qaoa_ring":
        return _qaoa_ring_circuit(QuantumCircuit, n, layers)
    if benchmark == "ghz":
        return _ghz_circuit(QuantumCircuit, n)
    if benchmark == "grover_proxy":
        return _grover_proxy_circuit(QuantumCircuit, n)
    if benchmark == "phase_estimation":
        return _phase_estimation_circuit(QuantumCircuit, n)
    if benchmark == "tfim_trotter":
        return _tfim_trotter_circuit(QuantumCircuit, n)
    raise ValueError(f"unsupported benchmark: {benchmark}")


def run(args: argparse.Namespace) -> dict:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    availability = _availability()
    if not (availability["qiskit"]["available"] and availability["qiskit-aer"]["available"]):
        _write_unavailable(outdir, availability)
        print(json.dumps({"outdir": str(outdir), "status": "unavailable", "availability": availability}, indent=2))
        return {"status": "unavailable", "availability": availability}

    from qiskit import QuantumCircuit, transpile  # type: ignore
    from qiskit_aer import AerSimulator  # type: ignore

    qubits = [int(x) for x in args.qubits.split(",") if x.strip()]
    benchmarks = [x.strip() for x in args.benchmarks.split(",") if x.strip()]
    simulator = AerSimulator(method="statevector", device="CPU")
    raw_rows: list[dict] = []
    summaries: list[dict] = []
    for benchmark in benchmarks:
        for n in qubits:
            circuit = transpile(_build_circuit(QuantumCircuit, benchmark, n, args.layers), simulator, optimization_level=0)
            measured: list[float] = []
            for run_index in range(args.warmups + args.repeats):
                warmup = run_index < args.warmups
                t0 = time.perf_counter()
                result = simulator.run(circuit, shots=1).result()
                state = result.get_statevector(circuit)
                checksum = len(state)
                wall_ms = (time.perf_counter() - t0) * 1000.0
                raw_rows.append({
                    "backend": "qiskit_aer_statevector_cpu",
                    "benchmark": benchmark,
                    "qubits": n,
                    "layers": args.layers if benchmark == "qaoa_ring" else "",
                    "run_index": run_index,
                    "warmup": warmup,
                    "wall_ms": wall_ms,
                    "state_size": checksum,
                })
                if not warmup:
                    measured.append(wall_ms)
            summary = {
                "backend": "qiskit_aer_statevector_cpu",
                "benchmark": benchmark,
                "qubits": n,
                "layers": args.layers if benchmark == "qaoa_ring" else "",
                "warmups": args.warmups,
                "repeats": args.repeats,
            }
            summary.update(_stats(measured))
            summaries.append(summary)

    raw_csv = outdir / "qiskit_aer_baseline_raw_runs.csv"
    with raw_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)
    summary_csv = outdir / "qiskit_aer_baseline_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    manifest = {
        "status": "measured",
        "availability": availability,
        "python": sys.version,
        "platform": platform.platform(),
        "benchmarks": benchmarks,
        "qubits": qubits,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "layers": args.layers,
        "files": {"raw_runs": raw_csv.name, "summary": summary_csv.name},
    }
    (outdir / "qiskit_aer_baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Qiskit Aer CPU Baseline",
        "",
        "| Workload | Qubits | Mean s | 95% CI s | Repeats |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {benchmark} | {qubits} | {mean_s:.4f} | {ci_s:.4f} | {repeats} |".format(
                benchmark=row["benchmark"],
                qubits=row["qubits"],
                mean_s=row["mean_ms"] / 1000.0,
                ci_s=row["ci95_ms"] / 1000.0,
                repeats=row["repeats"],
            )
        )
    (outdir / "qiskit_aer_baseline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"outdir": str(outdir), "status": "measured", "summaries": summaries}, indent=2))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--benchmarks", default="qft,qaoa_ring")
    parser.add_argument("--qubits", default="15,20,25")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--layers", type=int, default=6)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
