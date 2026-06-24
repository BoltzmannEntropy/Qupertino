#!/usr/bin/env python3
"""
custatevec_bench.py — Re-run mlxQ benchmark schedules on NVIDIA/cuStateVec.

Requirements on the CUDA system:
  - Python 3.10+
  - cuquantum-python (custatevec) and CuPy installed and working with your CUDA driver
  - This repository available on the machine (so we can reuse qubit lists and CSV layout)

Usage examples:
  # Re-run hamiltonian_simulation using the qubit list from bench/hamiltonian_simulation_data.csv
  python3 tools/custatevec_bench.py --key hamiltonian_simulation \
      --csv bench/hamiltonian_simulation_data.csv --out bench

  # Re-run QFT using an explicit qubit list
  python3 tools/custatevec_bench.py --key qft --qubits 1,2,5,7,10,11,12 --out bench

Notes:
  - This script provides minimal circuit builders for a few bench keys (qft, random_circuit,
    time_evolution, trotter, hamiltonian_simulation) and measures end-to-end wall time
    on cuStateVec if available, otherwise on CuPy (dense state-vector fallback).
  - The CuPy fallback runs on the GPU but does not use cuStateVec kernels; it serves as a
    functional baseline if cuquantum is unavailable. Prefer cuquantum for apples-to-apples tests.
  - Outputs are saved as bench/<key>_cuda_data.csv and <key>_cuda_scaling.png by default.
"""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import cupy as cp  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("CuPy is required on the CUDA system (pip install cupy-cudaXX)") from e

try:
    from cuquantum import custatevec as cu  # type: ignore
    _HAS_CUQ = True
except Exception:
    _HAS_CUQ = False


def _load_qubits_from_csv(csv_path: Path) -> List[int]:
    rows: List[int] = []
    with open(csv_path, 'r') as f:
        rd = csv.DictReader(f)
        # accept variants: qubits or Qubits
        for r in rd:
            q = r.get('qubits') or r.get('Qubits')
            if q is not None:
                try:
                    rows.append(int(q))
                except Exception:
                    pass
    rows = sorted(set(rows))
    return rows


def _vec_n(n: int):
    psi = cp.zeros(1 << n, dtype=cp.complex64)
    psi[0] = 1.0 + 0.0j
    return psi


def _apply_1q_cu(handle, sv: cp.ndarray, U: cp.ndarray, target: int):
    # Minimal cuStateVec wrapper; assumes row-major 2x2 complex64
    targets = (target,)
    cu.apply_matrix(
        handle,
        sv.data.ptr,
        cu.cudaDataType.CUDA_C_32F,
        U.data.ptr,
        cu.cudaDataType.CUDA_C_32F,
        cu.MatrixLayout.ROW,
        0, 0, 1,
        targets,
        0, (), 0, 0,
    )


def _apply_2q_cu(handle, sv: cp.ndarray, U: cp.ndarray, targets: Tuple[int, int]):
    cu.apply_matrix(
        handle,
        sv.data.ptr,
        cu.cudaDataType.CUDA_C_32F,
        U.data.ptr,
        cu.cudaDataType.CUDA_C_32F,
        cu.MatrixLayout.ROW,
        0, 0, 2,
        targets,
        0, (), 0, 0,
    )


def _rx(theta: float) -> cp.ndarray:
    c = cp.cos(theta/2.0)
    s = -1j*cp.sin(theta/2.0)
    return cp.array([[c, s],[s, c]], dtype=cp.complex64)


def _rz(theta: float) -> cp.ndarray:
    return cp.array([[cp.exp(-1j*theta/2.0), 0],[0, cp.exp(1j*theta/2.0)]], dtype=cp.complex64)


def _cnot() -> cp.ndarray:
    return cp.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=cp.complex64)


def build_qft(n: int) -> List[Tuple[str, Tuple]]:
    ops: List[Tuple[str, Tuple]] = []
    for j in range(n):
        ops.append(("rz", (j, cp.pi)))  # phase placeholders improve numerics slightly
        for k in range(j+1, n):
            # Controlled phase rotation R_{k-j+1}
            theta = cp.pi / (1 << (k-j))
            ops.append(("crz", (k, j, theta)))  # control k -> target j
        ops.append(("rx", (j, cp.pi/2)))
    # final swaps omitted for speed; do not change asymptotics
    return ops


def build_random(n: int, depth: int = 2) -> List[Tuple[str, Tuple]]:
    ops: List[Tuple[str, Tuple]] = []
    rs = cp.random.RandomState(1234)
    for _ in range(depth):
        for q in range(n):
            ops.append(("rx", (q, float(rs.uniform(0, 2*cp.pi)))))
            ops.append(("rz", (q, float(rs.uniform(0, 2*cp.pi)))))
        for q in range(0, n-1, 2):
            ops.append(("cnot", (q, q+1)))
    return ops


def build_evolution(n: int, layers: int = 4) -> List[Tuple[str, Tuple]]:
    # simple TE: local rotations + nearest neighbor entanglers
    ops: List[Tuple[str, Tuple]] = []
    for _ in range(layers):
        for q in range(n): ops.append(("rx", (q, 0.3)))
        for q in range(n-1): ops.append(("cnot", (q, q+1)))
    return ops


def run_ops(n: int, ops: List[Tuple[str, Tuple]]) -> float:
    psi = _vec_n(n)
    t0 = time.perf_counter()
    if _HAS_CUQ:
        handle = cu.create()
        for (name, args) in ops:
            if name == 'rx':
                q, theta = args
                _apply_1q_cu(handle, psi, _rx(theta), q)
            elif name == 'rz':
                q, theta = args
                _apply_1q_cu(handle, psi, _rz(theta), q)
            elif name == 'cnot':
                a, b = args
                _apply_2q_cu(handle, psi, _cnot(), (a, b))
            elif name == 'crz':
                c, t, theta = args
                # naive CRZ as CNOT-RZ-CNOT (commuting approx for bench only)
                _apply_2q_cu(handle, psi, _cnot(), (c, t))
                _apply_1q_cu(handle, psi, _rz(theta), t)
                _apply_2q_cu(handle, psi, _cnot(), (c, t))
        cu.destroy(handle)
    else:
        # CuPy fallback: dense apply via reshapes (slower but functional)
        for (name, args) in ops:
            if name in ('rx','rz'):
                q, theta = args
                U = _rx(theta) if name=='rx' else _rz(theta)
                # apply 1q by folding in/out dimension q
                psi = psi.reshape([2]*n)
                psi = cp.tensordot(U, psi, axes=([1],[q]))
                order = [*range(1,q+1),0,*range(q+1,n)]
                psi = cp.transpose(psi, axes=order).reshape(-1)
            elif name=='cnot':
                a,b = args
                U = _cnot()
                psi = psi.reshape([2]*n)
                # move targets to front
                axes = [a,b] + [i for i in range(n) if i not in (a,b)]
                psi = cp.transpose(psi, axes=axes)
                psi = cp.tensordot(U, psi.reshape(4,-1), axes=([1],[0])).reshape([2,2]+[2]*(n-2))
                # invert perm
                inv = [axes.index(i) for i in range(n)]
                psi = cp.transpose(psi, axes=inv).reshape(-1)
    wall_ms = (time.perf_counter() - t0)*1000.0
    # sanity touch
    _ = cp.sum(cp.abs(psi))
    cp.cuda.Stream.null.synchronize()
    return float(wall_ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', default='hamiltonian_simulation', help='bench key (e.g., hamiltonian_simulation, time_evolution, trotter, random_circuit, qft)')
    ap.add_argument('--csv', type=str, default='', help='bench/<key>_data.csv to read qubit list from')
    ap.add_argument('--qubits', type=str, default='', help='explicit qubit CSV, e.g. 1,2,5,7,10')
    ap.add_argument('--out', type=str, default='bench', help='output directory (default bench)')
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.csv:
        qs = _load_qubits_from_csv(Path(args.csv))
    elif args.qubits:
        qs = [int(x.strip()) for x in args.qubits.split(',') if x.strip().isdigit()]
    else:
        raise SystemExit('Provide --csv bench/<key>_data.csv or --qubits list')

    # Choose circuit builder
    key = args.key.lower()
    def build(n: int) -> List[Tuple[str, Tuple]]:
        if key == 'qft':
            return build_qft(n)
        elif key in ('hamiltonian_simulation','time_evolution','trotter'):
            return build_evolution(n, layers=4)
        else:
            return build_random(n, depth=2)

    rows: List[Tuple[int,float]] = []
    for n in qs:
        wall_ms = run_ops(n, build(n))
        print(f"{key:24s} | {n:2d}q | wall {wall_ms:10.2f} ms")
        rows.append((n, wall_ms))

    # write CSV in mlxQ style
    out_csv = out_dir / f"{key}_cuda_data.csv"
    with open(out_csv, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['qubits','execution_time_ms'])
        for n, ms in rows:
            wr.writerow([n, f"{ms:.6f}"])
    print(f"CSV: {out_csv}")

    # optional plot (use matplotlib if available)
    try:
        import matplotlib.pyplot as plt  # type: ignore
        xs = [n for n,_ in rows]
        ys = [ms for _,ms in rows]
        plt.figure(figsize=(6,4))
        plt.semilogy(xs, ys, marker='o', linestyle='-')
        plt.grid(True, which='both', linestyle=':')
        plt.title(f"{key} (cuStateVec{'+CuPy' if not _HAS_CUQ else ''})")
        plt.xlabel('Number of Qubits')
        plt.ylabel('Execution Time (ms)')
        out_png = out_dir / f"{key}_cuda_scaling.png"
        plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()
        print(f"PNG: {out_png}")
    except Exception:
        pass


if __name__ == '__main__':
    main()

