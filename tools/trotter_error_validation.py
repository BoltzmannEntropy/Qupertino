#!/usr/bin/env python3
"""Small-system TFIM Trotter-error validation for the QUANTICS revision."""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm


I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def _kron_all(mats: list[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for mat in mats[1:]:
        out = np.kron(out, mat)
    return out


def _single(op: np.ndarray, q: int, n: int) -> np.ndarray:
    return _kron_all([op if i == q else I2 for i in range(n)])


def _two(op1: np.ndarray, q1: int, op2: np.ndarray, q2: int, n: int) -> np.ndarray:
    mats = []
    for i in range(n):
        if i == q1:
            mats.append(op1)
        elif i == q2:
            mats.append(op2)
        else:
            mats.append(I2)
    return _kron_all(mats)


def _hamiltonian(n: int, J: float, h: float) -> np.ndarray:
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n - 1):
        H += -J * _two(Z, i, Z, i + 1, n)
    for i in range(n):
        H += h * _single(X, i, n)
    return H


def _rx(theta: float) -> np.ndarray:
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def _zz_phase(angle: float) -> np.ndarray:
    ep = np.exp(1j * angle)
    em = np.exp(-1j * angle)
    return np.diag([em, ep, ep, em]).astype(np.complex128)


def _apply_dense(state: np.ndarray, gate: np.ndarray, wires: list[int], n: int) -> np.ndarray:
    k = len(wires)
    tensor = state.reshape([2] * n)
    selected = set(wires)
    perm = [i for i in range(n) if i not in selected] + wires
    inv_perm = np.argsort(perm)
    moved = np.transpose(tensor, perm)
    updated = moved.reshape((1 << (n - k), 1 << k)) @ gate.T
    return np.transpose(updated.reshape([2] * n), inv_perm).reshape(1 << n)


def _trotter_state(n: int, J: float, h: float, total_time: float, steps: int) -> np.ndarray:
    state = np.zeros(1 << n, dtype=np.complex128)
    state[0] = 1.0
    dt = total_time / float(steps)
    Uzz = _zz_phase(-dt * J)
    Ux = _rx(2.0 * h * dt)
    for _ in range(steps):
        for i in range(n - 1):
            state = _apply_dense(state, Uzz, [i, i + 1], n)
        for q in range(n):
            state = _apply_dense(state, Ux, [q], n)
    return state


def _metrics(exact: np.ndarray, approx: np.ndarray) -> dict:
    overlap = np.vdot(exact, approx)
    fidelity = float(abs(overlap) ** 2)
    prob_mae = float(np.mean(np.abs(np.abs(exact) ** 2 - np.abs(approx) ** 2)))
    return {
        "state_l2": float(np.linalg.norm(exact - approx)),
        "fidelity": fidelity,
        "infidelity": max(0.0, 1.0 - fidelity),
        "probability_mae": prob_mae,
    }


def run(args: argparse.Namespace) -> list[dict]:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    steps_list = [int(x) for x in args.steps.split(",") if x.strip()]
    n = int(args.qubits)
    t0 = time.perf_counter()
    H = _hamiltonian(n, args.J, args.h)
    init = np.zeros(1 << n, dtype=np.complex128)
    init[0] = 1.0
    exact = expm(-1j * H * args.time_total) @ init
    rows: list[dict] = []
    for steps in steps_list:
        approx = _trotter_state(n, args.J, args.h, args.time_total, steps)
        row = {
            "qubits": n,
            "J": args.J,
            "h": args.h,
            "time_total": args.time_total,
            "trotter_steps": steps,
        }
        row.update(_metrics(exact, approx))
        rows.append(row)

    csv_path = outdir / "trotter_error_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# TFIM Trotter Error",
        "",
        "| Qubits | Steps | State L2 | Probability MAE | Infidelity |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {qubits} | {trotter_steps} | {state_l2:.3e} | {probability_mae:.3e} | {infidelity:.3e} |".format(**row)
        )
    (outdir / "trotter_error.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {
        "manifest": {
            "python": sys.version,
            "platform": platform.platform(),
            "elapsed_s": time.perf_counter() - t0,
            "reference": "scipy.linalg.expm complex128 dense Hamiltonian",
            "schedule": "first-order product formula matching src/mlxq/bench.py simulate_hamiltonian",
            "files": {"summary_csv": csv_path.name, "summary_md": "trotter_error.md"},
        },
        "rows": rows,
    }
    (outdir / "trotter_error.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"outdir": str(outdir), "rows": rows}, indent=2))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--qubits", type=int, default=6)
    parser.add_argument("--steps", default="4,8,16,32,64")
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=0.5)
    parser.add_argument("--time-total", type=float, default=1.0)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
