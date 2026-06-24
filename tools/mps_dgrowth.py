#!/usr/bin/env python3
"""
MPS D-growth Tracker
--------------------

Runs a simple TEBD loop for selected Hamiltonians under the MPS backend and
records the bond dimension growth across steps. Outputs a CSV and optional plot.

Supported circuits: tfim, heisenberg, heisenberg_xxz.

Examples:
  python3 tools/mps_dgrowth.py --circuit tfim --n 16 --steps 40 --dt 0.05 \
      --dmax 128 --eps 1e-10 --out bench

Emits: bench/mps_dgrowth_<circuit>_n<N>.csv
Optionally emits: bench/mps_dgrowth_<circuit>_n<N>.png when --plot is set.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import mlx.core as mx  # type: ignore

from mlxq.mps_state import MPSState, MPSOptions
from mlxq.gates import RX
from mlxq.bench import _zz_phase_gate, _xx_phase_gate, _yy_phase_gate


def run_tfim(n: int, steps: int, dt: float, J: float, h: float, opts: MPSOptions):
    sim = MPSState(n, opts)
    Uzz = _zz_phase_gate(-dt*J)
    Ux = RX(2.0*h*dt)
    rows = [(0, sim.bond_max(), sim.bond_mean(), sim.trunc_count(), int(sim.truncated()))]
    for s in range(1, steps+1):
        sim.apply_two_sweep(Uzz)
        sim.apply_single_all(Ux)
        rows.append((s, sim.bond_max(), sim.bond_mean(), sim.trunc_count(), int(sim.truncated())))
    return rows


def run_heisenberg(n: int, steps: int, dt: float, Jx: float, Jy: float, Jz: float, opts: MPSOptions):
    sim = MPSState(n, opts)
    Ux = _xx_phase_gate(-dt*Jx) if abs(Jx) > 0 else None
    Uy = _yy_phase_gate(-dt*Jy) if abs(Jy) > 0 else None
    Uz = _zz_phase_gate(-dt*Jz) if abs(Jz) > 0 else None
    rows = [(0, sim.bond_max(), sim.bond_mean(), sim.trunc_count(), int(sim.truncated()))]
    for s in range(1, steps+1):
        if Ux is not None:
            sim.apply_two_sweep(Ux)
        if Uy is not None:
            sim.apply_two_sweep(Uy)
        if Uz is not None:
            sim.apply_two_sweep(Uz)
        rows.append((s, sim.bond_max(), sim.bond_mean(), sim.trunc_count(), int(sim.truncated())))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--circuit', choices=['tfim', 'heisenberg', 'heisenberg_xxz'], required=True)
    ap.add_argument('--n', type=int, required=True, help='number of qubits')
    ap.add_argument('--steps', type=int, default=20, help='Trotter steps (TEBD)')
    ap.add_argument('--dt', type=float, default=0.05, help='time step per TEBD iteration')
    ap.add_argument('--J', type=float, default=1.0, help='coupling J or Jx')
    ap.add_argument('--Jy', type=float, default=1.0, help='coupling Jy (heisenberg_xxz)')
    ap.add_argument('--Jz', type=float, default=1.0, help='coupling Jz (heisenberg_xxz)')
    ap.add_argument('--h', type=float, default=0.5, help='field h (TFIM)')
    ap.add_argument('--dmax', type=int, default=128, help='MPS bond cap')
    ap.add_argument('--eps', type=float, default=1e-10, help='MPS truncation epsilon')
    ap.add_argument('--out', default='bench', help='output directory')
    ap.add_argument('--plot', action='store_true', help='emit a bond-vs-step plot')
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    opts = MPSOptions(dmax=int(args.dmax), eps=float(args.eps))

    if args.circuit == 'tfim':
        rows = run_tfim(args.n, args.steps, args.dt, args.J, args.h, opts)
    elif args.circuit == 'heisenberg':
        rows = run_heisenberg(args.n, args.steps, args.dt, args.J, args.J, args.J, opts)
    else:  # heisenberg_xxz
        rows = run_heisenberg(args.n, args.steps, args.dt, args.J, args.Jy, args.Jz, opts)

    csv_path = out / f"mps_dgrowth_{args.circuit}_n{args.n}.csv"
    with open(csv_path, 'w') as f:
        f.write('step,bond_max,bond_mean,truncations,truncated\n')
        for s, bmax, bmean, truncs, trunc in rows:
            f.write(f"{s},{int(bmax)},{float(bmean):.6f},{int(truncs)},{int(trunc)}\n")
    print(f"D-growth CSV: {csv_path}")

    if args.plot:
        try:
            from mlxq.plotting import plot_scaling
            xs = [r[0] for r in rows]
            ys = [r[1] for r in rows]
            png = out / f"mps_dgrowth_{args.circuit}_n{args.n}.png"
            notes = [f"dmax={opts.dmax}, eps={opts.eps}"]
            plot_scaling(xs, ys, title=f"MPS D-growth ({args.circuit}, n={args.n})", out=str(png), logy=True,
                         annotate=True, ylabel='Max Bond', extra_notes=notes)
            print(f"D-growth plot: {png}")
        except Exception as e:
            print(f"plot skipped: {e}")


if __name__ == '__main__':
    main()

