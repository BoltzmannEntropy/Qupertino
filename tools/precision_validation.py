#!/usr/bin/env python3
"""Precision validation for the QUANTICS paper revision.

The goal is not to create a new simulator; it is to produce a small,
reproducible evidence artifact that bounds the complex64 MLX state-vector
path against complex128 references at the largest practical sizes:

* QFT at 25 qubits is checked against the analytical complex128 |+...+> state.
* Ring-QAOA at 20 qubits is checked against an independent NumPy complex128
  reshape/contract gate path.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MLXQ_PRINT_ASCII", "0")

import mlx.core as mx  # noqa: E402
from mlxq.device import Device  # noqa: E402


def _qft_ops(n: int) -> list[dict]:
    ops: list[dict] = []
    for j in range(n):
        ops.append({"name": "H", "wires": [j]})
        for k in range(j + 1, n):
            ops.append({
                "name": "CPHASE",
                "wires": [k, j],
                "parameters": [math.pi / (2 ** (k - j))],
            })
    return ops


def _qaoa_ops(n: int, layers: int) -> list[dict]:
    ops: list[dict] = []
    for layer in range(layers):
        gamma = 0.6 + 0.1 * layer
        beta = 0.4 + 0.05 * layer
        for i in range(n):
            ops.append({
                "name": "CPHASE",
                "wires": [i, (i + 1) % n],
                "parameters": [gamma],
            })
        for q in range(n):
            ops.append({"name": "RX", "wires": [q], "parameters": [2.0 * beta]})
    return ops


def _mlx_state(n: int, ops: list[dict]) -> np.ndarray:
    dev = Device(n)
    dev.execute(ops)
    mx.eval(dev.sim.state)
    return np.asarray(dev.sim.state).astype(np.complex128, copy=False)


def _h() -> np.ndarray:
    s = 1.0 / math.sqrt(2.0)
    return np.array([[s, s], [s, -s]], dtype=np.complex128)


def _rx(theta: float) -> np.ndarray:
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def _cphase(phi: float) -> np.ndarray:
    return np.diag([1.0, 1.0, 1.0, np.exp(1j * phi)]).astype(np.complex128)


def _gate_matrix(op: dict) -> np.ndarray:
    name = op["name"].upper()
    params = op.get("parameters", [])
    if name == "H":
        return _h()
    if name == "RX":
        return _rx(float(params[0]))
    if name == "CPHASE":
        return _cphase(float(params[0]))
    raise ValueError(f"unsupported op in NumPy reference: {name}")


def _apply_dense_np(state: np.ndarray, gate: np.ndarray, wires: Iterable[int], n: int) -> np.ndarray:
    qs = list(wires)
    k = len(qs)
    expected = 1 << k
    if gate.shape != (expected, expected):
        raise ValueError(f"gate shape {gate.shape} incompatible with {k} wires")
    tensor = state.reshape([2] * n)
    selected = set(qs)
    perm = [i for i in range(n) if i not in selected] + qs
    inv_perm = np.argsort(perm)
    moved = np.transpose(tensor, perm)
    outer_dim = 1 << (n - k)
    matrix = moved.reshape((outer_dim, expected))
    updated = matrix @ gate.T
    return np.transpose(updated.reshape([2] * n), inv_perm).reshape(1 << n)


def _numpy_state(n: int, ops: list[dict]) -> np.ndarray:
    state = np.zeros(1 << n, dtype=np.complex128)
    state[0] = 1.0 + 0.0j
    for op in ops:
        state = _apply_dense_np(state, _gate_matrix(op), op["wires"], n)
    return state


def _metrics(name: str, n: int, mlx_state: np.ndarray, ref_state: np.ndarray | None) -> dict:
    norm_mlx = float(np.linalg.norm(mlx_state))
    if ref_state is None:
        uniform = 1.0 / math.sqrt(float(1 << n))
        overlap = np.sum(mlx_state) * uniform
        amp_l2 = float(np.sqrt(np.sum(np.abs(mlx_state - uniform) ** 2)))
        prob = np.abs(mlx_state) ** 2
        prob_mae = float(np.mean(np.abs(prob - (1.0 / float(1 << n)))))
        prob_max_abs = float(np.max(np.abs(prob - (1.0 / float(1 << n)))))
        norm_ref = 1.0
    else:
        norm_ref = float(np.linalg.norm(ref_state))
        overlap = np.vdot(ref_state, mlx_state)
        amp_l2 = float(np.linalg.norm(mlx_state - ref_state))
        prob_mlx = np.abs(mlx_state) ** 2
        prob_ref = np.abs(ref_state) ** 2
        prob_mae = float(np.mean(np.abs(prob_mlx - prob_ref)))
        prob_max_abs = float(np.max(np.abs(prob_mlx - prob_ref)))
    denom = max(1e-30, norm_mlx * norm_ref)
    fidelity = float((abs(overlap) / denom) ** 2)
    return {
        "workload": name,
        "qubits": n,
        "amplitude_l2": amp_l2,
        "probability_mae": prob_mae,
        "probability_max_abs": prob_max_abs,
        "norm_mlx": norm_mlx,
        "norm_ref": norm_ref,
        "norm_drift_mlx": abs(norm_mlx - 1.0),
        "fidelity": fidelity,
        "infidelity": max(0.0, 1.0 - fidelity),
    }


def _write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Precision Validation",
        "",
        "| Workload | Qubits | Amplitude L2 | Probability MAE | Max prob. error | Norm drift | Infidelity |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {workload} | {qubits} | {amplitude_l2:.3e} | {probability_mae:.3e} | "
            "{probability_max_abs:.3e} | {norm_drift_mlx:.3e} | {infidelity:.3e} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> list[dict]:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t0 = time.perf_counter()

    qft_n = int(args.qft_qubits)
    qft_state = _mlx_state(qft_n, _qft_ops(qft_n))
    rows.append(_metrics("qft_analytical_complex128", qft_n, qft_state, None))

    qaoa_n = int(args.qaoa_qubits)
    qaoa_ops = _qaoa_ops(qaoa_n, int(args.layers))
    qaoa_mlx = _mlx_state(qaoa_n, qaoa_ops)
    qaoa_np = _numpy_state(qaoa_n, qaoa_ops)
    rows.append(_metrics("ring_qaoa_numpy_complex128", qaoa_n, qaoa_mlx, qaoa_np))

    csv_path = outdir / "precision_validation_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "generated_at_epoch": time.time(),
        "elapsed_s": time.perf_counter() - t0,
        "python": sys.version,
        "platform": platform.platform(),
        "qft_reference": "analytical complex128 uniform state for QFT|0...0>",
        "qaoa_reference": "independent NumPy complex128 reshape/contract gate path",
        "files": {"summary_csv": csv_path.name, "summary_md": "precision_validation.md"},
    }
    (outdir / "precision_validation.json").write_text(
        json.dumps({"manifest": manifest, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    _write_markdown(outdir / "precision_validation.md", rows)
    print(json.dumps({"outdir": str(outdir), "rows": rows}, indent=2))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--qft-qubits", type=int, default=25)
    parser.add_argument("--qaoa-qubits", type=int, default=20)
    parser.add_argument("--layers", type=int, default=6)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
