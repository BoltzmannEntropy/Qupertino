#!/usr/bin/env python3
"""Generate truthful performance charts from the campaign/sweep CSV artifacts.

Outputs PNGs consumed by the README and the website. Every value is read
straight from the evidence artifacts, so the charts cannot drift from the
claim-audited tables.

  chart_4way_25q.png       grouped bars, 4 backends x 6 workloads @ 25q (log)
  chart_scaling_qft.png    QFT wall time vs qubits, 4 backends (log-log)
  chart_scaling_tfim.png   TFIM Trotter wall time vs qubits, 4 backends
  chart_speedup_sweep.png  full-suite Metal-vs-pure-MLX speedup (horizontal)

Usage: gen_perf_charts.py --evidence DIR --outdir DIR
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Brand palette (Apple-silicon inspired: graphite + accent)
C_METAL = "#0a84ff"   # Metal shader tier (hero)
C_MLX = "#30d158"     # pure MLX
C_PL = "#ff9f0a"      # PennyLane
C_AER = "#ff453a"     # Aer CPU
BG = "#0b0f14"
FG = "#e6edf3"
GRID = "#26303b"

WL_LABELS = {
    "qft": "QFT", "qaoa_ring": "Ring-QAOA", "tfim_trotter": "TFIM Trotter",
    "phase_estimation": "Phase est.", "grover_proxy": "Grover", "ghz": "GHZ",
}
WL_ORDER = ["qft", "qaoa_ring", "tfim_trotter", "phase_estimation",
            "grover_proxy", "ghz"]
BACKENDS = [("mlxq_metal", "Qupertino Metal", C_METAL),
            ("mlxq", "Qupertino MLX", C_MLX),
            ("aer", "Qiskit Aer CPU", C_AER),
            ("pennylane", "PennyLane", C_PL)]

SWEEP_LABELS = {
    "hamiltonian": "TFIM Trotter (1st)", "tfim_trotter2": "TFIM Trotter (2nd)",
    "long_range_ising": "Long-range Ising", "variational": "Variational",
    "heisenberg": "Heisenberg", "heisenberg_xxz": "Heisenberg XXZ",
    "cuquantum_blueqat": "cuQuantum proxy", "tfim_random_field": "TFIM rand. field",
    "qcbm": "QCBM", "quantum_walk_vchain": "Quantum walk (vchain)",
    "grover": "Grover", "quantum_walk": "Quantum walk",
    "su2rand": "EfficientSU2", "ghz": "GHZ", "heisenberg_random_field": "Heisenberg rand.",
    "graph_state": "Graph state", "phase_estimation_inexact": "Phase est. (inexact)",
    "qft": "QFT", "qaoa": "QAOA", "deutsch_jozsa": "Deutsch-Jozsa",
    "phase_estimation": "Phase est.", "qnn": "QNN", "qft_entangled": "QFT (entangled)",
    "realamp": "RealAmplitudes", "random_circuit": "Random circuit",
    "vqe": "VQE (+energy)", "ladder_heisenberg": "Heisenberg ladder",
    "ae": "Amplitude est.", "wstate": "W state",
}


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, fontsize=13, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, color=FG, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=FG, fontsize=10)
    ax.tick_params(colors=FG, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)


def _read_summary(path):
    d = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            d[(r["benchmark"], int(r["qubits"]), r["backend"])] = float(r["mean_ms"])
    return d


def chart_4way(summary, out):
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=140)
    fig.patch.set_facecolor(BG)
    import numpy as np
    x = np.arange(len(WL_ORDER))
    w = 0.20
    for i, (key, label, color) in enumerate(BACKENDS):
        vals = [summary[(wl, 25, key)] for wl in WL_ORDER]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=label, color=color,
                      edgecolor=BG, linewidth=0.5)
        if key == "mlxq_metal":
            for b, v in zip(bars, vals):
                ax.annotate(f"{v:.0f}" if v >= 10 else f"{v:.0f}",
                            (b.get_x() + b.get_width() / 2, v), ha="center",
                            va="bottom", color=C_METAL, fontsize=7.5,
                            fontweight="bold", xytext=(0, 1),
                            textcoords="offset points")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([WL_LABELS[w] for w in WL_ORDER], rotation=12)
    _style(ax, "Wall time at 25 qubits, four backends (lower is better, log scale)",
           None, "Time (ms)")
    leg = ax.legend(loc="upper right", fontsize=8.5, facecolor="#121820",
                    edgecolor=GRID, labelcolor=FG, ncol=2)
    fig.tight_layout()
    fig.savefig(out, facecolor=BG)
    plt.close(fig)


def chart_scaling(summary, wl, title, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=140)
    fig.patch.set_facecolor(BG)
    sizes = [15, 20, 25]
    for key, label, color in BACKENDS:
        ys = [summary[(wl, n, key)] for n in sizes]
        ax.plot(sizes, ys, marker="o", color=color, label=label, linewidth=2,
                markersize=6)
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    _style(ax, title, "Qubits", "Time (ms)")
    ax.legend(loc="upper left", fontsize=8.5, facecolor="#121820",
              edgecolor=GRID, labelcolor=FG)
    fig.tight_layout()
    fig.savefig(out, facecolor=BG)
    plt.close(fig)


def chart_speedup(sweep_csv, out):
    rows = list(csv.DictReader(open(sweep_csv)))
    rows.sort(key=lambda r: float(r["paired_ratio_mean"]))
    labels = [SWEEP_LABELS.get(r["benchmark"], r["benchmark"]) for r in rows]
    ratios = [float(r["paired_ratio_mean"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8.2, 9.2), dpi=140)
    fig.patch.set_facecolor(BG)
    import numpy as np
    y = np.arange(len(labels))
    colors = [C_METAL if r >= 4 else (C_MLX if r >= 1.1 else "#6e7681")
              for r in ratios]
    ax.barh(y, ratios, color=colors, edgecolor=BG, linewidth=0.4)
    for yi, r in zip(y, ratios):
        ax.annotate(f"{r:.1f}x", (r, yi), ha="left", va="center", color=FG,
                    fontsize=8, xytext=(3, 0), textcoords="offset points")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(0, max(ratios) * 1.12)
    _style(ax, "Metal shaders vs pure MLX at 25q (29 workloads, paired)",
           "Speedup factor", None)
    fig.tight_layout()
    fig.savefig(out, facecolor=BG)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    ev = Path(args.evidence)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    summary = _read_summary(ev / "interleaved_4way_20260704"
                            / "interleaved_summary.csv")
    chart_4way(summary, out / "chart_4way_25q.png")
    chart_scaling(summary, "qft", "QFT scaling, four backends",
                  out / "chart_scaling_qft.png")
    chart_scaling(summary, "tfim_trotter", "TFIM Trotter scaling, four backends",
                  out / "chart_scaling_tfim.png")
    chart_speedup(ev / "shader_sweep_20260704" / "shader_sweep_summary.csv",
                  out / "chart_speedup_sweep.png")
    print(f"wrote 4 charts to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
