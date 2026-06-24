#!/usr/bin/env python3
"""
overlay_compare.py — Overlay mlxQ CSVs with external results for visual comparison.

Usage:
  python3 tools/overlay_compare.py \
      --ours bench/hamiltonian_simulation_data.csv \
      --ext external_hamiltonian_simulation.csv \
      --label-ours "mlxQ (M1 Max)" \
      --label-ext  "cuStateVec (A100)" \
      --out bench/hamiltonian_simulation_overlay.png

CSV formats accepted:
  - ours (preferred): qubits,execution_time_ms
  - generic: must include columns: qubits,<time_col>, where time is ms or s.
    Use --ext-time-col and --ext-scale-ms to adapt.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple


def load_simple_csv(path: Path, time_col: str = 'execution_time_ms', scale_ms: float = 1.0) -> List[Tuple[int, float]]:
    rows: List[Tuple[int,float]] = []
    with open(path, 'r') as f:
        rd = csv.DictReader(f)
        for r in rd:
            try:
                q = int(r.get('qubits') or r.get('Qubits') or r.get('n'))
                t = float(r.get(time_col)) * scale_ms
                rows.append((q, t))
            except Exception:
                pass
    rows.sort(key=lambda x: x[0])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ours', required=True)
    ap.add_argument('--ext', required=True)
    ap.add_argument('--label-ours', default='mlxQ')
    ap.add_argument('--label-ext', default='external')
    ap.add_argument('--ext-time-col', default='execution_time_ms')
    ap.add_argument('--ext-scale-ms', type=float, default=1.0, help='multiply ext times by this to get ms (e.g., 1000 if ext is seconds)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    ours = load_simple_csv(Path(args.ours))
    ext = load_simple_csv(Path(args.ext), time_col=args.ext_time_col, scale_ms=args.ext_scale_ms)
    if not ours or not ext:
        raise SystemExit('No data loaded; check CSVs and column names')

    try:
        import matplotlib.pyplot as plt  # type: ignore
        plt.figure(figsize=(6.5,4.2))
        xo = [q for q,_ in ours]; yo = [t for _,t in ours]
        xe = [q for q,_ in ext];  ye = [t for _,t in ext]
        plt.semilogy(xo, yo, marker='o', linestyle='-', linewidth=2, label=args.label_ours)
        plt.semilogy(xe, ye, marker='s', linestyle='--', linewidth=2, label=args.label_ext)
        plt.grid(True, which='both', linestyle=':')
        plt.xlabel('Qubits')
        plt.ylabel('Execution Time (ms)')
        plt.legend(loc='best')
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout(); plt.savefig(out, dpi=300); plt.close()
        print(f'Overlay saved: {out}')
    except Exception as e:
        raise SystemExit(f'Matplotlib plotting failed: {e}')


if __name__ == '__main__':
    main()
