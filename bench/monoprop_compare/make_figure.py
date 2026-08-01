"""Build the runtime-comparison figure and results tables from raw run JSONL.

Reads every results/raw/*.jsonl produced by src/benchmark/bench_monoprop_tfim.py,
selects one canonical measurement per (scenario, size, engine) — documented
below — and emits:

  runtime_comparison.png   two-panel log-runtime figure
  RESULTS.md               the same numbers as markdown tables

Selection rules (kept deliberately dumb and auditable):
  - For duplicate measurements the LATEST run wins. This matters once: the
    28-qubit mlxQ TFIM point was first measured before the MLX buffer-pool
    flush landed in the bench (262 s, memory-thrashed); the post-fix rerun
    (40.7 s) supersedes it. Both raw files are retained.
  - mlxQ series are the Metal-shader tier (metal == "1"); the pure-MLX tier
    is tabulated separately where measured.

Run:  .runtime-venv/bin/python bench/monoprop_compare/make_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RAW = HERE / "results" / "raw"

BLUE = "#2a78d6"    # series 1: mlxQ
ORANGE = "#eb6834"  # series 2: monoprop
INK = "#0b0b0b"
INK2 = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e6e5e1"


def load_rows():
    rows = []
    for path in sorted(RAW.glob("*.jsonl")):  # sorted => later timestamp wins
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                r["_run"] = path.name
                rows.append(r)
    return rows


def canonical(rows, scenario, engine, metal=None):
    """Latest row per size for one engine; returns {num_qubits: row}."""
    out = {}
    for r in rows:
        if r["scenario"] != scenario:
            continue
        is_mlxq = r["label"].startswith("mlxQ")
        if engine == "mlxq" and not is_mlxq:
            continue
        if engine == "monoprop" and is_mlxq:
            continue
        if metal is not None and is_mlxq and r.get("metal") != metal:
            continue
        out[r["num_qubits"]] = r  # later files overwrite earlier ones
    return out


def total_runtime(row):
    rt = row["runtime_s"]
    return sum(rt) if isinstance(rt, list) else rt


def series(canon):
    qs = sorted(canon)
    return qs, [total_runtime(canon[q]) for q in qs]


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    rows = load_rows()
    tfim_mlxq = canonical(rows, "tfim", "mlxq", metal="1")
    tfim_pure = canonical(rows, "tfim", "mlxq", metal="0")
    tfim_mono = canonical(rows, "tfim", "monoprop")
    kick_mlxq = canonical(rows, "kicked", "mlxq", metal="1")
    kick_mono = canonical(rows, "kicked", "monoprop")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)

    # Panel 1: TFIM
    q, t = series(tfim_mlxq)
    ax1.plot(q, t, color=BLUE, linewidth=2, marker="o", markersize=6,
             label="mlxQ (Metal tier)", zorder=3)
    ax1.annotate("mlxQ", xy=(q[-1], t[-1]), xytext=(6, 4),
                 textcoords="offset points", color=BLUE, fontsize=9)
    qm, tm = series(tfim_mono)
    both = [x for x in qm if x <= max(q)]
    ax1.plot(both, tm[:len(both)], color=ORANGE, linewidth=2, marker="s",
             markersize=6, label="monoprop", zorder=3)
    ax1.annotate("monoprop", xy=(both[-1], tm[len(both) - 1]), xytext=(6, -12),
                 textcoords="offset points", color=ORANGE, fontsize=9)
    if max(qm) > max(q):  # monoprop-only upstream size
        ax1.plot([qm[-1]], [tm[-1]], color=ORANGE, marker="s", markersize=6,
                 linestyle="none", zorder=3)
        ax1.annotate("6x6 upstream setting\n(statevector: ~0.5 TB)",
                     xy=(qm[-1], tm[-1]), xytext=(-8, 14),
                     textcoords="offset points", ha="right",
                     color=INK2, fontsize=8)
    ax1.set_title("2D tilted-field Ising (monoprop third-party scenario)\n"
                  "21 timed points: 1 Trotter step + $\\langle Z_aZ_b\\rangle$ each",
                  fontsize=10, color=INK, loc="left")
    ax1.set_xlabel("qubits", fontsize=9, color=INK2)
    ax1.set_ylabel("total runtime (s), log scale", fontsize=9, color=INK2)

    # Panel 2: kicked Ising
    q, t = series(kick_mlxq)
    ax2.plot(q, t, color=BLUE, linewidth=2, marker="o", markersize=6,
             label="mlxQ (Metal tier)", zorder=3)
    ax2.annotate("mlxQ", xy=(q[-1], t[-1]), xytext=(6, 0),
                 textcoords="offset points", color=BLUE, fontsize=9)
    qm, tm = series(kick_mono)
    both = [x for x in qm if x <= max(q)]
    ax2.plot(both, [tm[qm.index(x)] for x in both], color=ORANGE, linewidth=2,
             marker="s", markersize=6, label="monoprop", zorder=3)
    ax2.annotate("monoprop", xy=(both[-1], tm[qm.index(both[-1])]),
                 xytext=(6, -12), textcoords="offset points",
                 color=ORANGE, fontsize=9)
    if 127 in kick_mono:
        ax2.text(0.03, 0.60,
                 f"monoprop, full 127-qubit Eagle: "
                 f"{total_runtime(kick_mono[127]):.2f} s\n"
                 "(statevector infeasible)",
                 transform=ax2.transAxes, ha="left", color=INK2, fontsize=8)
    ax2.set_title("Kicked Ising, heavy-hex (monoprop bench_models scenario)\n"
                  "20 layers + $\\langle Z_{mid}\\rangle$, induced subgraphs",
                  fontsize=10, color=INK, loc="left")
    ax2.set_xlabel("qubits", fontsize=9, color=INK2)

    for ax in (ax1, ax2):
        ax.set_yscale("log")
        style_axis(ax)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK,
                  loc="upper left")
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi + 0.12 * (hi - lo))  # room for direct labels

    fig.text(0.008, 0.030,
             "Apple M1 Max, 32 GB · macOS 26.3 · mlx 0.30.6 (complex64) · "
             "monoprop 0.8.0 PyPI wheel (default threads) · identical "
             "circuits, observables, and timing brackets per monoprop benches/.",
             fontsize=7, color=INK2)
    fig.text(0.008, 0.008,
             "Kicked panel: at the upstream cutoff=8 / atol=1e-4 settings, "
             "monoprop expectation values deviate 0.02–0.32 from exact at "
             "16–28 qubits (see RESULTS.md).",
             fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = HERE / "runtime_comparison.png"
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")

    # RESULTS.md
    lines = ["# Measured results\n",
             "Regenerate with `make_figure.py`; raw series in `results/raw/`.\n",
             "\n## 2D TFIM (upstream settings: dt=0.05, atol=1e-6, no cutoff, "
             "21 timed points)\n",
             "| Grid | Qubits | mlxQ Metal (s) | mlxQ pure (s) | monoprop (s) "
             "| Faster |",
             "|---|---|---|---|---|---|"]
    for nq in sorted(set(tfim_mono) | set(tfim_mlxq)):
        r = tfim_mono.get(nq) or tfim_mlxq[nq]
        grid = f"{r['nx']}x{r['ny']}"
        t_metal = total_runtime(tfim_mlxq[nq]) if nq in tfim_mlxq else None
        t_pure = total_runtime(tfim_pure[nq]) if nq in tfim_pure else None
        t_mono = total_runtime(tfim_mono[nq]) if nq in tfim_mono else None
        if t_metal and t_mono:
            fast = ("mlxQ %.1fx" % (t_mono / t_metal)) if t_metal < t_mono \
                else ("monoprop %.1fx" % (t_metal / t_mono))
        else:
            fast = "monoprop only" if t_mono else "mlxQ only"
        lines.append(f"| {grid} | {nq} | "
                     f"{('%.3f' % t_metal) if t_metal else '—'} | "
                     f"{('%.3f' % t_pure) if t_pure else '—'} | "
                     f"{('%.3f' % t_mono) if t_mono else '—'} | {fast} |")
    lines += ["\n## Kicked Ising (upstream settings: 20 layers, cutoff=8, "
              "atol=1e-4)\n",
              "| Qubits | mlxQ Metal (s) | monoprop (s) | Faster | "
              "Δexpval (trunc. error) |",
              "|---|---|---|---|---|"]
    for nq in sorted(set(kick_mono) | set(kick_mlxq)):
        t_metal = total_runtime(kick_mlxq[nq]) if nq in kick_mlxq else None
        t_mono = total_runtime(kick_mono[nq]) if nq in kick_mono else None
        gap = "—"
        if nq in kick_mlxq and nq in kick_mono:
            gap = "%.3f" % abs(kick_mlxq[nq]["expval"] - kick_mono[nq]["expval"])
            fast = ("mlxQ %.1fx" % (t_mono / t_metal)) if t_metal < t_mono \
                else ("monoprop %.1fx" % (t_metal / t_mono))
        else:
            fast = "monoprop only" if t_mono else "mlxQ only"
        lines.append(f"| {nq} | {('%.3f' % t_metal) if t_metal else '—'} | "
                     f"{('%.3f' % t_mono) if t_mono else '—'} | {fast} | {gap} |")
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {HERE / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
