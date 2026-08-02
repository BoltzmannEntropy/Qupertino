"""Build the runtime-comparison figure and results tables from raw run JSONL.

Reads every results/raw/*.jsonl produced by src/benchmark/bench_monoprop_tfim.py
and reduces the repeated runs to one **median** per (scenario, size, engine,
tier), with the min-max spread reported alongside. Emits:

  runtime_comparison.png   two-panel log-runtime figure (median + min-max bars)
  RESULTS.md               the same numbers as markdown tables

Selection protocol (the audit trail for every number):
  1. A raw file that contains BOTH engines' rows records a shared-process run.
     Cross-engine contamination is real and measured: monoprop's 28-qubit TFIM
     sweep timed 2.93 s right after mlxQ's memory-heavy run in the same
     process vs a 2.1-2.5 s band in dedicated processes. mlxQ rows are kept
     from such files (mlxQ always ran first, so it was unaffected); monoprop
     rows are DROPPED from the statistics.
  2. Two early mlxQ measurements predate the >=26-qubit MLX buffer-pool guard
     and measured pool thrashing, not the simulator (262 s / 121 s vs the
     post-fix 41-50 s band). They are excluded by name below and retained in
     results/raw/ for inspection.
  3. Everything surviving 1-2 is aggregated as median with min-max spread.
     Single-shot times at sub-second scale vary +-20-50% run to run on this
     machine, which is why medians of repeated dedicated-process runs are
     reported rather than upstream's rounds=1 protocol.

Run:  .runtime-venv/bin/python bench/monoprop_compare/make_figure.py
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RAW = HERE / "results" / "raw"

# Rule 2: pre-fix mlxQ runs (measured MLX pool thrashing, not the simulator).
PREFIX_EXCLUDE = {
    ("tfim_20260801_211318.jsonl", "mlxq", 28),   # before TFIM pool guard
    ("kicked_20260801_213034.jsonl", "mlxq", 28),  # before kicked pool guard
}

BLUE = "#2a78d6"    # series 1: mlxQ
ORANGE = "#eb6834"  # series 2: monoprop
INK = "#0b0b0b"
INK2 = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e6e5e1"


def total_runtime(row):
    rt = row["runtime_s"]
    return sum(rt) if isinstance(rt, list) else rt


def load_samples():
    """{(scenario, engine, tier, num_qubits): [rows...]} after rules 1-2."""
    samples = defaultdict(list)
    for path in sorted(RAW.glob("*.jsonl")):
        with open(path) as f:
            rows = [json.loads(line) for line in f]
        shared = ({r["label"].startswith("mlxQ") for r in rows} == {True, False})
        for r in rows:
            is_mlxq = r["label"].startswith("mlxQ")
            engine = "mlxq" if is_mlxq else "monoprop"
            if shared and engine == "monoprop":
                continue  # rule 1: contaminated by the preceding mlxQ run
            if (path.name, engine, r["num_qubits"]) in PREFIX_EXCLUDE:
                continue  # rule 2
            tier = ("metal" if r.get("metal") == "1" else "pure") if is_mlxq else ""
            samples[(r["scenario"], engine, tier, r["num_qubits"])].append(r)
    return samples


def stats_for(samples, scenario, engine, tier=""):
    """{num_qubits: (median, min, max, n, representative_row)} sorted by size."""
    out = {}
    for (scn, eng, t, nq), rows in samples.items():
        if scn != scenario or eng != engine or t != tier:
            continue
        totals = sorted(total_runtime(r) for r in rows)
        out[nq] = (statistics.median(totals), totals[0], totals[-1],
                   len(totals), rows[-1])
    return dict(sorted(out.items()))


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def draw_series(ax, st, color, label, marker, sizes=None):
    qs = [q for q in st if sizes is None or q in sizes]
    med = [st[q][0] for q in qs]
    lo = [st[q][0] - st[q][1] for q in qs]
    hi = [st[q][2] - st[q][0] for q in qs]
    ax.errorbar(qs, med, yerr=[lo, hi], color=color, linewidth=2, marker=marker,
                markersize=6, capsize=3, elinewidth=1, label=label, zorder=3)
    return qs, med


def main():
    samples = load_samples()
    tfim_mlxq = stats_for(samples, "tfim", "mlxq", "metal")
    tfim_pure = stats_for(samples, "tfim", "mlxq", "pure")
    tfim_mono = stats_for(samples, "tfim", "monoprop")
    kick_mlxq = stats_for(samples, "kicked", "mlxq", "metal")
    kick_mono = stats_for(samples, "kicked", "monoprop")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)

    # Panel 1: TFIM
    q, med = draw_series(ax1, tfim_mlxq, BLUE, "mlxQ (Metal tier)", "o")
    ax1.annotate("mlxQ", xy=(q[-1], med[-1]), xytext=(6, 4),
                 textcoords="offset points", color=BLUE, fontsize=9)
    shared_sizes = [x for x in tfim_mono if x <= max(q)]
    qm, mm = draw_series(ax1, tfim_mono, ORANGE, "monoprop", "s",
                         sizes=shared_sizes)
    ax1.annotate("monoprop", xy=(qm[-1], mm[-1]), xytext=(6, -12),
                 textcoords="offset points", color=ORANGE, fontsize=9)
    solo = [x for x in tfim_mono if x > max(q)]
    if solo:
        draw_series(ax1, tfim_mono, ORANGE, None, "s", sizes=solo)
        ax1.annotate("6x6 upstream setting\n(statevector: ~0.5 TB)",
                     xy=(solo[0], tfim_mono[solo[0]][0]), xytext=(-8, 14),
                     textcoords="offset points", ha="right",
                     color=INK2, fontsize=8)
    ax1.set_title("2D tilted-field Ising (monoprop third-party scenario)\n"
                  "21 timed points: 1 Trotter step + $\\langle Z_aZ_b\\rangle$ each",
                  fontsize=10, color=INK, loc="left")
    ax1.set_xlabel("qubits", fontsize=9, color=INK2)
    ax1.set_ylabel("total runtime (s), log scale — median of runs, min–max bars",
                   fontsize=9, color=INK2)

    # Panel 2: kicked Ising
    q, med = draw_series(ax2, kick_mlxq, BLUE, "mlxQ (Metal tier)", "o")
    ax2.annotate("mlxQ", xy=(q[-1], med[-1]), xytext=(6, 0),
                 textcoords="offset points", color=BLUE, fontsize=9)
    shared_sizes = [x for x in kick_mono if x <= max(q)]
    qm, mm = draw_series(ax2, kick_mono, ORANGE, "monoprop", "s",
                         sizes=shared_sizes)
    ax2.annotate("monoprop", xy=(qm[-1], mm[-1]), xytext=(6, -12),
                 textcoords="offset points", color=ORANGE, fontsize=9)
    if 127 in kick_mono:
        med127, lo127, hi127, n127, _ = kick_mono[127]
        ax2.text(0.03, 0.60,
                 f"monoprop, full 127-qubit Eagle: {med127:.2f} s median\n"
                 "(statevector infeasible)",
                 transform=ax2.transAxes, ha="left", color=INK2, fontsize=8)
    ax2.set_title("Kicked Ising, heavy-hex (monoprop bench_models scenario)\n"
                  "20 layers + $\\langle Z_{mid}\\rangle$, induced subgraphs",
                  fontsize=10, color=INK, loc="left")
    ax2.set_xlabel("qubits", fontsize=9, color=INK2)

    for ax in (ax1, ax2):
        ax.set_yscale("log")
        style_axis(ax)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi + 0.12 * (hi - lo))  # room for direct labels

    fig.text(0.008, 0.030,
             "Apple M1 Max, 32 GB · macOS 26.3 · mlx 0.30.6 (complex64) · "
             "monoprop 0.8.0 PyPI wheel (default threads) · identical "
             "circuits, observables, and timing brackets per monoprop benches/.",
             fontsize=7, color=INK2)
    fig.text(0.008, 0.008,
             "Median of dedicated-process runs, min–max bars. Kicked panel: at "
             "the upstream cutoff=8 / atol=1e-4 settings, monoprop expectation "
             "values deviate 0.02–0.32 from exact at 16–28 qubits (RESULTS.md).",
             fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = HERE / "runtime_comparison.png"
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")

    # ---------------- RESULTS.md ----------------
    def cell(st, nq):
        if nq not in st:
            return "—"
        med, lo, hi, n, _ = st[nq]
        return f"{med:.3f} [{lo:.3f}–{hi:.3f}, n={n}]"

    def verdict(st_a, st_b, nq, name_a="mlxQ", name_b="monoprop"):
        if nq not in st_a or nq not in st_b:
            return f"{name_b} only" if nq in st_b else f"{name_a} only"
        a, b = st_a[nq][0], st_b[nq][0]
        return (f"{name_a} {b / a:.1f}x" if a < b else f"{name_b} {a / b:.1f}x")

    lines = [
        "# Measured results\n",
        "Median total runtime in seconds over dedicated-process repeats, "
        "with [min–max, n] spread. Selection protocol (shared-process and "
        "pre-fix exclusions) is documented in `make_figure.py`; every raw "
        "run is in `results/raw/`.\n",
        "\n## 2D TFIM (upstream settings: dt=0.05, atol=1e-6, no cutoff, "
        "21 timed points)\n",
        "| Grid | Qubits | mlxQ Metal (s) | mlxQ pure (s) | monoprop (s) "
        "| Faster (median) |",
        "|---|---|---|---|---|---|",
    ]
    for nq in sorted(set(tfim_mono) | set(tfim_mlxq)):
        row = (tfim_mono.get(nq) or tfim_mlxq[nq])[4]
        grid = f"{row['nx']}x{row['ny']}"
        lines.append(f"| {grid} | {nq} | {cell(tfim_mlxq, nq)} | "
                     f"{cell(tfim_pure, nq)} | {cell(tfim_mono, nq)} | "
                     f"{verdict(tfim_mlxq, tfim_mono, nq)} |")
    lines += [
        "\n## Kicked Ising (upstream settings: 20 layers, cutoff=8, "
        "atol=1e-4)\n",
        "| Qubits | mlxQ Metal (s) | monoprop (s) | Faster (median) | "
        "Δexpval (trunc. error) |",
        "|---|---|---|---|---|",
    ]
    for nq in sorted(set(kick_mono) | set(kick_mlxq)):
        gap = "—"
        if nq in kick_mlxq and nq in kick_mono:
            gap = "%.3f" % abs(kick_mlxq[nq][4]["expval"]
                               - kick_mono[nq][4]["expval"])
        lines.append(f"| {nq} | {cell(kick_mlxq, nq)} | {cell(kick_mono, nq)} | "
                     f"{verdict(kick_mlxq, kick_mono, nq)} | {gap} |")
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {HERE / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
