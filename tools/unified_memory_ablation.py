#!/usr/bin/env python3
"""Unified-memory ablation evidence for the QUANTICS paper.

This script measures two concrete quantities:

1. Synthetic explicit-copy cost for complex64 state vectors at selected qubit
   counts, plus PCIe 4/5 theoretical round-trip estimates.
2. The runtime effect of forcing host-side amplitude reads after every QAOA
   layer, compared with terminal-only evaluation.

It does not claim to emulate a production CUDA simulator. It is a controlled
artifact for separating setup/read-out copy costs and host-inspection costs
from the normal in-place state-vector path.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MLXQ_PRINT_ASCII", "0")

import mlx.core as mx  # noqa: E402
from mlxq.device import Device  # noqa: E402


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
        "repeats": len(values),
    }


def _copy_ablation(qubits: list[int], repeats: int) -> list[dict]:
    rows: list[dict] = []
    for n in qubits:
        dim = 1 << n
        state = np.ones(dim, dtype=np.complex64) / math.sqrt(float(dim))
        times: list[float] = []
        checksum = 0.0
        for _ in range(repeats):
            t0 = time.perf_counter()
            host = state.copy()
            returned = host.copy()
            checksum += float(np.real(returned[0]))
            times.append((time.perf_counter() - t0) * 1000.0)
        bytes_one_state = int(state.nbytes)
        row = {
            "ablation": "synthetic_explicit_roundtrip_copy",
            "qubits": n,
            "state_mb": bytes_one_state / (1024.0 * 1024.0),
            "roundtrip_mb": (2 * bytes_one_state) / (1024.0 * 1024.0),
            "pcie4_32GBps_roundtrip_ms": (2 * bytes_one_state / 32e9) * 1000.0,
            "pcie5_64GBps_roundtrip_ms": (2 * bytes_one_state / 64e9) * 1000.0,
            "checksum": checksum,
        }
        row.update(_stats(times))
        rows.append(row)
    return rows


def _run_qaoa(n: int, layers: int, force_host_read_each_layer: bool) -> tuple[float, float]:
    dev = Device(n)
    checksum = 0.0
    t0 = time.perf_counter()
    for layer in range(layers):
        gamma = 0.6 + 0.1 * layer
        beta = 0.4 + 0.05 * layer
        ops: list[dict] = []
        for i in range(n):
            ops.append({
                "name": "CPHASE",
                "wires": [i, (i + 1) % n],
                "parameters": [gamma],
            })
        for q in range(n):
            ops.append({"name": "RX", "wires": [q], "parameters": [2.0 * beta]})
        dev.execute(ops)
        if force_host_read_each_layer:
            mx.eval(dev.sim.state)
            val = dev.sim.state[0]
            checksum += abs(complex(val.item() if hasattr(val, "item") else val))
    _ = dev.sim.probabilities()
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return wall_ms, checksum


def _host_read_ablation(qubits: int, layers: int, repeats: int) -> list[dict]:
    rows: list[dict] = []
    for mode, force in [
        ("terminal_evaluation_only", False),
        ("host_read_after_each_layer", True),
    ]:
        times: list[float] = []
        checksum = 0.0
        for _ in range(repeats):
            wall_ms, chk = _run_qaoa(qubits, layers, force)
            times.append(wall_ms)
            checksum += chk
        row = {
            "ablation": mode,
            "qubits": qubits,
            "layers": layers,
            "state_mb": ((1 << qubits) * 8) / (1024.0 * 1024.0),
            "forced_host_reads": layers if force else 0,
            "checksum": checksum,
        }
        row.update(_stats(times))
        rows.append(row)
    if len(rows) == 2 and rows[0]["mean_ms"] > 0:
        overhead = rows[1]["mean_ms"] - rows[0]["mean_ms"]
        rows.append({
            "ablation": "host_read_overhead",
            "qubits": qubits,
            "layers": layers,
            "state_mb": rows[0]["state_mb"],
            "forced_host_reads": layers,
            "mean_ms": overhead,
            "overhead_ratio": rows[1]["mean_ms"] / rows[0]["mean_ms"],
            "repeats": repeats,
        })
    return rows


def _write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Unified-Memory Ablation",
        "",
        "| Ablation | Qubits | Mean ms | 95% CI ms | State MB | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        notes = []
        if "pcie4_32GBps_roundtrip_ms" in row:
            notes.append(
                "PCIe4 round-trip {:.3f} ms; PCIe5 {:.3f} ms".format(
                    row["pcie4_32GBps_roundtrip_ms"],
                    row["pcie5_64GBps_roundtrip_ms"],
                )
            )
        if row.get("forced_host_reads"):
            notes.append(f"forced reads={row['forced_host_reads']}")
        if "overhead_ratio" in row:
            notes.append("ratio {:.3f}x".format(row["overhead_ratio"]))
        fmt = dict(row)
        fmt["ci95_ms"] = float(row.get("ci95_ms", 0.0))
        fmt["notes"] = "; ".join(notes)
        lines.append(
            "| {ablation} | {qubits} | {mean_ms:.3f} | {ci95_ms:.3f} | {state_mb:.3f} | {notes} |".format(
                **fmt,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> list[dict]:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    copy_qubits = [int(x) for x in args.copy_qubits.split(",") if x.strip()]
    rows = _copy_ablation(copy_qubits, args.repeats)
    rows.extend(_host_read_ablation(args.host_read_qubits, args.layers, args.repeats))

    csv_path = outdir / "unified_memory_ablation_summary.csv"
    fieldnames = sorted({k for row in rows for k in row})
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "manifest": {
            "python": sys.version,
            "platform": platform.platform(),
            "copy_qubits": copy_qubits,
            "host_read_qubits": args.host_read_qubits,
            "layers": args.layers,
            "repeats": args.repeats,
            "files": {
                "summary_csv": csv_path.name,
                "summary_md": "unified_memory_ablation.md",
            },
        },
        "rows": rows,
    }
    (outdir / "unified_memory_ablation.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    _write_markdown(outdir / "unified_memory_ablation.md", rows)
    print(json.dumps({"outdir": str(outdir), "rows": rows}, indent=2))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--copy-qubits", default="20,25")
    parser.add_argument("--host-read-qubits", type=int, default=20)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
