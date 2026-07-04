#!/usr/bin/env python3
"""Re-measure the hardware-efficient ansatz optimization diagnostic (tab:vqe).

Faithful to the archived protocol: basic entangler ansatz (per-wire RZ+RX plus
ring entangler => 2n parameters), <Z_0> objective, parameter-shift gradients
(two forward passes per parameter per iteration), Adam, 100 iterations.
Reports total wall time and time/iteration per qubit count on the current
structured-gate-dispatch backend.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlxq import qml  # noqa: E402


def run_one(n: int, iters: int) -> dict:
    dev = qml.Device(wires=n)

    @qml.qnode(dev)
    def circuit(flat_params):
        pairs = [(flat_params[2 * i], flat_params[2 * i + 1]) for i in range(n)]
        qml.basic_entangler_layers(pairs, wires=list(range(n)))
        return qml.expval(qml.PauliZ(0))

    params = [0.1 + 0.01 * i for i in range(2 * n)]
    # Adam state
    m = [0.0] * len(params)
    v = [0.0] * len(params)
    lr, b1, b2, eps = 0.05, 0.9, 0.999, 1e-8

    t0 = time.perf_counter()
    for it in range(1, iters + 1):
        g = circuit.grad(params)
        for i, gi in enumerate(g):
            m[i] = b1 * m[i] + (1 - b1) * gi
            v[i] = b2 * v[i] + (1 - b2) * gi * gi
            mh = m[i] / (1 - b1 ** it)
            vh = v[i] / (1 - b2 ** it)
            params[i] -= lr * mh / (vh ** 0.5 + eps)
    total = time.perf_counter() - t0
    final = circuit(params)
    return {"qubits": n, "iterations": iters, "n_params": 2 * n,
            "total_s": round(total, 2), "s_per_iter": round(total / iters, 4),
            "final_expval": float(final) if not isinstance(final, (list, tuple)) else float(final[0])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qubits", default="10,11,12,13,14,15")
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    results = []
    for n in [int(x) for x in args.qubits.split(",")]:
        r = run_one(n, args.iterations)
        results.append(r)
        print(f"n={r['qubits']}: total {r['total_s']} s, "
              f"{r['s_per_iter']} s/iter, final <Z0>={r['final_expval']:.4f}",
              flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "protocol": "basic entangler ansatz (2n params), <Z_0> objective, "
                    "parameter-shift gradients, Adam, structured-gate dispatch backend",
        "results": results}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
