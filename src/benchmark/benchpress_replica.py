#!/usr/bin/env python3
"""
Generate Benchpress-like summary figures and table from existing mlxQ bench outputs.

Outputs under bench/:
  - benchpress_fig1.png   (Fourier/variational/synthetic overview)
  - benchpress_fig2.png   (Evolution models overview)
  - benchpress_table2.tex (LaTeX table of per-benchmark coverage and caps)
"""
from __future__ import annotations
import csv
import statistics as stats
from pathlib import Path
from typing import Dict, List, Tuple


def load_csv_rows(p: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    try:
        with open(p, 'r') as f:
            rd = csv.DictReader(f)
            for r in rd:
                rows.append(r)
    except Exception:
        pass
    return rows


def pick_data(bench_dir: Path, keys: List[str]) -> Dict[str, List[Dict[str, str]]]:
    data: Dict[str, List[Dict[str, str]]] = {}
    for k in keys:
        # Prefer unsuffixed; fallback to first *_data.csv match
        p = bench_dir / f"{k}_data.csv"
        if not p.exists():
            cands = sorted(bench_dir.glob(f"{k}_*_data.csv"))
            if cands:
                p = cands[0]
        if p.exists():
            data[k] = load_csv_rows(p)
    return data


def to_xy(rows: List[Dict[str, str]]) -> Tuple[List[int], List[float]]:
    xs: List[int] = []
    ys: List[float] = []
    for r in rows:
        try:
            xs.append(int(r.get('qubits') or r.get('Qubits') or '0'))
            ys.append(float(r.get('execution_time_ms') or r.get('exec(ms)') or '0'))
        except Exception:
            pass
    return xs, ys


def fig_overview_1(bench_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    keys = ['qft', 'qaoa', 'vqe', 'random_circuit', 'phase_estimation', 'grover']
    data = pick_data(bench_dir, keys)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), squeeze=False)
    for i, k in enumerate(keys):
        r, c = divmod(i, 3)
        ax = axes[r][c]
        rows = data.get(k) or []
        xs, ys = to_xy(rows)
        if xs and ys:
            ax.semilogy(xs, ys, marker='o', linewidth=2)
        ax.grid(True, which='both', linestyle=':', alpha=0.5)
        ax.set_title(k)
        ax.set_xlabel('qubits'); ax.set_ylabel('exec (ms)')
    fig.tight_layout()
    out = bench_dir / 'benchpress_fig1.png'
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def fig_overview_2(bench_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    keys = ['hamiltonian_simulation', 'time_evolution', 'trotter', 'heisenberg', 'tfim', 'long_range_ising']
    data = pick_data(bench_dir, keys)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), squeeze=False)
    for i, k in enumerate(keys):
        r, c = divmod(i, 3)
        ax = axes[r][c]
        rows = data.get(k) or []
        xs, ys = to_xy(rows)
        if xs and ys:
            ax.semilogy(xs, ys, marker='o', linewidth=2)
        ax.grid(True, which='both', linestyle=':', alpha=0.5)
        ax.set_title(k)
        ax.set_xlabel('qubits'); ax.set_ylabel('exec (ms)')
    fig.tight_layout()
    out = bench_dir / 'benchpress_fig2.png'
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def table2(bench_dir: Path, keys: List[str]) -> Path:
    # Summarize presence, rows, max qubits, median exec at max
    lines: List[str] = []
    lines.append('\\begin{tabular}{lrrr}')
    lines.append('\\toprule')
    lines.append('Benchmark & Rows & MaxQ & Med(ms)@MaxQ \\\\')
    lines.append('\\midrule')
    for k in keys:
        cands = list(bench_dir.glob(f"{k}_data.csv")) + list(bench_dir.glob(f"{k}_*_data.csv"))
        rows = []
        if cands:
            rows = load_csv_rows(sorted(cands)[0])
        if not rows:
            lines.append(f"{k} & 0 & -- & -- \\\")
            continue
        xs, ys = to_xy(rows)
        if not xs or not ys:
            lines.append(f"{k} & {len(rows)} & -- & -- \\\")
            continue
        maxq = max(xs)
        ys_at = [ys[i] for i in range(len(xs)) if xs[i] == maxq]
        med = stats.median(ys_at) if ys_at else 0.0
        lines.append(f"{k} & {len(rows)} & {maxq} & {med:.1f} \\\")
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    out = bench_dir / 'benchpress_table2.tex'
    out.write_text('\n'.join(lines))
    return out


def main() -> None:
    bench = Path('bench')
    bench.mkdir(parents=True, exist_ok=True)
    f1 = fig_overview_1(bench)
    f2 = fig_overview_2(bench)
    keys = [
        'qft','qaoa','vqe','random_circuit','phase_estimation','grover',
        'hamiltonian_simulation','time_evolution','trotter','heisenberg','tfim','long_range_ising',
    ]
    t2 = table2(bench, keys)
    print(f"Generated: {f1}\nGenerated: {f2}\nGenerated: {t2}")


if __name__ == '__main__':
    main()

