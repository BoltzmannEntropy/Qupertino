#!/usr/bin/env python3
"""Same-machine PennyLane baseline timings for osxQuantum paper revisions."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import time
from importlib import metadata
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-qn-baseline")

import numpy as np  # noqa: E402
import pennylane as qml  # noqa: E402


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _qft_qnode(device_name: str, n_qubits: int):
    """Explicit gate-identical QFT ladder (H + controlled-phase, no final SWAP
    layer), matching the mlxq and Aer baseline circuits gate-for-gate. The
    qml.QFT template appends a swap layer and therefore differs in both gate
    count and output bit order."""
    import math
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface=None)
    def circuit():
        for j in range(n_qubits):
            qml.Hadamard(wires=j)
            for k in range(j + 1, n_qubits):
                qml.ControlledPhaseShift(math.pi / (2 ** (k - j)), wires=[k, j])
        return qml.state()

    return circuit


def _qaoa_ring_qnode(device_name: str, n_qubits: int, layers: int):
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface=None)
    def circuit():
        for layer in range(layers):
            gamma = 0.6 + 0.1 * layer
            beta = 0.4 + 0.05 * layer
            for i in range(n_qubits):
                qml.ControlledPhaseShift(gamma, wires=[i, (i + 1) % n_qubits])
            for wire in range(n_qubits):
                qml.RX(2.0 * beta, wires=wire)
        return qml.state()

    return circuit


def _ghz_qnode(device_name: str, n_qubits: int):
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface=None)
    def circuit():
        qml.Hadamard(wires=0)
        for q in range(n_qubits - 1):
            qml.CNOT(wires=[q, q + 1])
        return qml.state()

    return circuit


def _grover_proxy_qnode(device_name: str, n_qubits: int):
    """Matches mlxq simulate_grover proxy: uniform init + one diffusion-like
    step with pairwise CZ as the phase-oracle proxy."""
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface=None)
    def circuit():
        for q in range(n_qubits):
            qml.Hadamard(wires=q)
        for q in range(n_qubits):
            qml.Hadamard(wires=q)
            qml.PauliX(wires=q)
        for q in range(n_qubits - 1):
            qml.CZ(wires=[q, q + 1])
        for q in range(n_qubits):
            qml.PauliX(wires=q)
            qml.Hadamard(wires=q)
        return qml.state()

    return circuit


def _phase_estimation_qnode(device_name: str, n_qubits: int):
    """Matches mlxq simulate_phase_estimation (base phase 0.4, target = n-1)."""
    import math
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface=None)
    def circuit():
        target = n_qubits - 1
        for p in range(n_qubits - 1):
            qml.Hadamard(wires=p)
        base = 0.4
        for p in range(n_qubits - 1):
            qml.ControlledPhaseShift(base * (2 ** p), wires=[p, target])
        for j in range(n_qubits - 2, -1, -1):
            for k in range(n_qubits - 2, j, -1):
                qml.ControlledPhaseShift(-math.pi / (2 ** (k - j)), wires=[k, j])
            qml.Hadamard(wires=j)
        return qml.state()

    return circuit


def _tfim_trotter_qnode(device_name: str, n_qubits: int, trotter_steps: int = 20,
                        time_total: float = 1.0, J: float = 1.0, h: float = 0.5):
    """Matches mlxq simulate_hamiltonian dense schedule. PennyLane
    IsingZZ(phi) = exp(-i*phi/2*ZZ), so phi = -2*J*dt."""
    dev = qml.device(device_name, wires=n_qubits)
    dt = time_total / float(trotter_steps)

    @qml.qnode(dev, interface=None)
    def circuit():
        for _ in range(trotter_steps):
            for i in range(n_qubits - 1):
                qml.IsingZZ(-2.0 * J * dt, wires=[i, i + 1])
            for q in range(n_qubits):
                qml.RX(2.0 * h * dt, wires=q)
        return qml.state()

    return circuit


def _build_qnode(benchmark: str, device_name: str, n_qubits: int, layers: int):
    if benchmark == "qft":
        return _qft_qnode(device_name, n_qubits)
    if benchmark == "qaoa_ring":
        return _qaoa_ring_qnode(device_name, n_qubits, layers)
    if benchmark == "ghz":
        return _ghz_qnode(device_name, n_qubits)
    if benchmark == "grover_proxy":
        return _grover_proxy_qnode(device_name, n_qubits)
    if benchmark == "phase_estimation":
        return _phase_estimation_qnode(device_name, n_qubits)
    if benchmark == "tfim_trotter":
        return _tfim_trotter_qnode(device_name, n_qubits)
    raise ValueError(f"unsupported benchmark: {benchmark}")


def _time_one(qnode) -> tuple[float, float, int]:
    t0 = time.perf_counter()
    c0 = time.process_time()
    state = qnode()
    arr = np.asarray(state)
    # Force materialization and keep a cheap integrity check in the artifact.
    checksum = int(arr.size)
    return (time.perf_counter() - t0) * 1000.0, time.process_time() - c0, checksum


def run(args: argparse.Namespace) -> dict:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    qubits = [int(item) for item in args.qubits.split(",") if item.strip()]
    benchmarks = [item.strip() for item in args.benchmarks.split(",") if item.strip()]

    rows: list[dict] = []
    summaries: list[dict] = []
    for benchmark in benchmarks:
        for n_qubits in qubits:
            qnode = _build_qnode(benchmark, args.device, n_qubits, args.layers)
            for run_idx in range(args.warmups + args.repeats):
                warmup = run_idx < args.warmups
                wall_ms, cpu_s, checksum = _time_one(qnode)
                rows.append({
                    "backend": args.device,
                    "benchmark": benchmark,
                    "qubits": n_qubits,
                    "layers": args.layers if benchmark == "qaoa_ring" else "",
                    "run_index": run_idx,
                    "warmup": warmup,
                    "wall_ms": wall_ms,
                    "cpu_s": cpu_s,
                    "state_size": checksum,
                })
            measured = [
                row["wall_ms"] for row in rows
                if row["benchmark"] == benchmark
                and row["qubits"] == n_qubits
                and not row["warmup"]
            ]
            summaries.append({
                "backend": args.device,
                "benchmark": benchmark,
                "qubits": n_qubits,
                "layers": args.layers if benchmark == "qaoa_ring" else "",
                "warmups": args.warmups,
                "repeats": args.repeats,
                "mean_ms": statistics.fmean(measured),
                "stdev_ms": statistics.stdev(measured) if len(measured) > 1 else 0.0,
                "min_ms": min(measured),
                "max_ms": max(measured),
            })

    raw_csv = outdir / "pennylane_baseline_raw_runs.csv"
    with raw_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_csv = outdir / "pennylane_baseline_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    manifest = {
        "device": args.device,
        "benchmarks": benchmarks,
        "qubits": qubits,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "qaoa_layers": args.layers,
        "timing": "wall-clock milliseconds around one QNode execution returning qml.state()",
        "packages": {
            "python": platform.python_version(),
            "pennylane": getattr(qml, "__version__", None),
            "pennylane-lightning": _version("pennylane-lightning"),
            "numpy": np.__version__,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "files": {
            "raw_runs": raw_csv.name,
            "summary": summary_csv.name,
        },
    }
    manifest_path = outdir / "pennylane_baseline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"outdir": str(outdir), "summaries": summaries}, indent=2))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--device", default="lightning.qubit")
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
