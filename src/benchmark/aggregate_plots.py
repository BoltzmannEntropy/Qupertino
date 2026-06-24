#!/usr/bin/env python3
"""
Aggregate per-benchmark scaling CSVs into a single comparison figure.

Reads <bench_dir>/<key>_data.csv for known bench keys and saves
<bench_dir>/all_benchmarks_comparison.png. If assets/benchmarks-frozen/latest
exists (or can be created), copies figures there as well.
"""
import os
import csv
from pathlib import Path
from typing import Dict, List

from mlxq.vendor import BENCH_KEYS


def load_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    try:
        with open(path, 'r') as f:
            rd = csv.DictReader(f)
            for r in rd:
                rows.append(r)
    except Exception:
        pass
    return rows


def main() -> None:
    bench_dir = Path(os.environ.get('MLXQ_BENCH_OUT_DIR', 'bench'))
    bench_dir.mkdir(parents=True, exist_ok=True)
    data: Dict[str, List[Dict[str, str]]] = {}
    for key in BENCH_KEYS:
        # Prefer unsuffixed file
        p = bench_dir / f"{key}_data.csv"
        if p.exists():
            data[key] = load_csv(p)
            continue
        # Fallback: pick a hardware-suffixed CSV if present
        candidates = list(bench_dir.glob(f"{key}_*_data.csv"))
        if candidates:
            # Choose the first in sorted order for determinism
            p2 = sorted(candidates)[0]
            data[key] = load_csv(p2)
    if not data:
        print(f"No CSVs found in {bench_dir}. Run bench.py first.")
        return
    # Plot
    try:
        # Use the shared plotting theme
        import math
        import matplotlib.pyplot as plt  # type: ignore
        from mlxq.plotting import set_theme
        set_theme()
        keys = sorted(data.keys())
        n = len(keys)
        cols = 3
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.8, rows * 3.6), squeeze=False)
        for idx, key in enumerate(keys):
            r = idx // cols
            c = idx % cols
            ax = axes[r][c]
            xs = []
            ys = []
            for row in data[key]:
                # Accept multiple header variants
                q = row.get('qubits') or row.get('Qubits') or row.get('n')
                t = row.get('execution_time_ms') or row.get('exec(ms)') or row.get('time_ms')
                try:
                    xs.append(int(q))
                    ys.append(float(t))
                except Exception:
                    continue
            if xs and ys:
                if any(v > 0 for v in ys):
                    ax.semilogy(xs, ys, marker='o', linestyle='-')
                else:
                    ax.plot(xs, ys, marker='o', linestyle='-')
            ax.set_title(key)
            ax.set_xlabel('Qubits')
            ax.set_ylabel('Exec (ms)')
            ax.grid(True, which='both')
        # Hide unused axes
        for k in range(n, rows * cols):
            r = k // cols
            c = k % cols
            fig.delaxes(axes[r][c])
        fig.tight_layout()
        out_png = bench_dir / 'all_benchmarks_comparison.png'
        fig.savefig(out_png, dpi=300)
        print(f"Comparison plot: {out_png}")
        # Optional copy to frozen latest assets
        assets = Path('assets') / 'benchmarks-frozen' / 'latest'
        try:
            assets.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copyfile(out_png, assets / out_png.name)
            print(f"Copied to {assets / out_png.name}")
        except Exception:
            pass
        # Additionally, if MPS summaries exist, make a bonds comparison grid
        bonds_keys = []
        bonds_data: Dict[str, List[Dict[str, str]]] = {}
        for key in BENCH_KEYS:
            p = bench_dir / f"{key}_mps_summary.csv"
            if p.exists():
                bonds_data[key] = load_csv(p)
                bonds_keys.append(key)
        if bonds_keys:
            rows2 = math.ceil(len(bonds_keys) / cols)
            fig2, axes2 = plt.subplots(rows2, cols, figsize=(cols * 4.8, rows2 * 3.6), squeeze=False)
            for idx, key in enumerate(sorted(bonds_keys)):
                r = idx // cols
                c = idx % cols
                ax = axes2[r][c]
                xs = []
                ys = []
                for row in bonds_data[key]:
                    q = row.get('qubits')
                    d = row.get('bond_max')
                    try:
                        xs.append(int(q))
                        ys.append(float(d))
                    except Exception:
                        continue
                if xs and ys:
                    if any(v > 0 for v in ys):
                        ax.semilogy(xs, ys, marker='o', linestyle='-')
                    else:
                        ax.plot(xs, ys, marker='o', linestyle='-')
                ax.set_title(f"{key} (MPS bonds)")
                ax.set_xlabel('Qubits')
                ax.set_ylabel('Max Bond')
                ax.grid(True, which='both')
            # Hide unused axes
            for k in range(len(bonds_keys), rows2 * cols):
                r = k // cols
                c = k % cols
                fig2.delaxes(axes2[r][c])
            fig2.tight_layout()
            out_bonds = bench_dir / 'all_mps_bonds_comparison.png'
            fig2.savefig(out_bonds, dpi=300)
            print(f"MPS bonds comparison plot: {out_bonds}")
            try:
                import shutil
                shutil.copyfile(out_bonds, assets / out_bonds.name)
                print(f"Copied to {assets / out_bonds.name}")
            except Exception:
                pass
    except Exception as e:
        print(f"plotting failed: {e}")


if __name__ == '__main__':
    main()
