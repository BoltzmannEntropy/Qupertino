"""
High-quality plotting utilities for mlxQ benchmarks (GNUplot-inspired style).

Provides a consistent theme, color palette, and annotation helpers for
per‑benchmark scaling plots and convergence plots.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple


def _mpl():
    import matplotlib as mpl  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    return mpl, plt


def set_theme() -> None:
    """Apply a GNUplot‑inspired, publication‑ready Matplotlib theme."""
    mpl, _ = _mpl()
    mpl.rcParams.update({
        "figure.figsize": (6.5, 4.2),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.color": "#c5c5c5",
        "grid.alpha": 0.7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.fancybox": True,
        "legend.borderpad": 0.3,
        "legend.borderaxespad": 0.6,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
        "axes.prop_cycle": mpl.cycler(
            color=[
                "#1f77b4",  # blue
                "#d62728",  # red
                "#2ca02c",  # green
                "#9467bd",  # purple
                "#ff7f0e",  # orange
                "#8c564b",  # brown
                "#17becf",  # cyan
                "#e377c2",  # pink
            ]
        ),
    })


def _annotate_endpoints(ax, xs: Sequence[float], ys: Sequence[float]) -> None:
    """Annotate first and last data points with values."""
    if not xs or not ys:
        return
    # First point
    ax.annotate(
        f"{ys[0]:.2f} ms",
        xy=(xs[0], ys[0]),
        xytext=(5, 6), textcoords="offset points",
        fontsize=9, color="#444",
        bbox=dict(boxstyle="round,pad=0.2", fc="#f5f5f5", ec="#bbbbbb", lw=0.5),
    )
    # Last/max point
    ax.annotate(
        f"{ys[-1]:.2f} ms",
        xy=(xs[-1], ys[-1]),
        xytext=(6, -14), textcoords="offset points",
        fontsize=9, color="#444",
        bbox=dict(boxstyle="round,pad=0.2", fc="#f5f5f5", ec="#bbbbbb", lw=0.5),
    )


def plot_scaling(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    title: str,
    xlabel: str = "Qubits",
    ylabel: str = "Execution Time (ms)",
    out: Path | str,
    logy: bool = True,
    label: Optional[str] = None,
    annotate: bool = True,
    extra_notes: Optional[Iterable[str]] = None,
) -> None:
    """Render a high‑quality scaling plot and save PNG (+optional PDF/SVG).

    - xs, ys: data
    - title/xlabel/ylabel: labels
    - out: output path (.png); also writes .pdf if MLXQ_SAVE_PDF=1
    - logy: semilogy by default
    - label: legend label (single series)
    - annotate: annotate endpoints with values
    - extra_notes: optional footnotes under the title
    """
    _, plt = _mpl()
    set_theme()
    fig, ax = plt.subplots()
    if logy:
        ax.semilogy(xs, ys, marker="o", linestyle="-", label=label)
    else:
        ax.plot(xs, ys, marker="o", linestyle="-", label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both")
    if label:
        ax.legend(loc="best")
    if annotate:
        _annotate_endpoints(ax, xs, ys)
    # Layout: first tighten, then add bottom caption space if needed
    fig.tight_layout()
    if extra_notes:
        # Place notes as a caption below the axes to avoid overlapping the title.
        # Increase bottom margin to accommodate multiple lines.
        pad = 0.14 + 0.06 * max(0, len(list(extra_notes)) - 1)
        try:
            fig.subplots_adjust(bottom=pad)
        except Exception:
            pass
        y0 = -0.10
        for idx, note in enumerate(extra_notes):
            ax.text(0.0, y0 - 0.06 * idx, note,
                    transform=ax.transAxes, fontsize=8, color="#666",
                    ha="left", va="top")
        # Move xlabel to the right to avoid any visual collision with the caption
        try:
            ax.set_xlabel(xlabel, ha="right", x=1.0)
        except Exception:
            pass
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    if os.environ.get("MLXQ_SAVE_PDF", "0") == "1":
        fig.savefig(out_path.with_suffix(".pdf"))
    if os.environ.get("MLXQ_SAVE_SVG", "0") == "1":
        fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def plot_convergence(
    xs: Sequence[float], ys: Sequence[float], *, title: str, out: Path | str,
    xlabel: str = "Iteration", ylabel: str = "Energy",
    label: Optional[str] = None,
) -> None:
    """Render a VQE convergence plot with consistent theme."""
    _, plt = _mpl()
    set_theme()
    fig, ax = plt.subplots()
    ax.plot(xs, ys, marker="o", linestyle="-", label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both")
    if label:
        ax.legend(loc="best")
    fig.tight_layout()
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    if os.environ.get("MLXQ_SAVE_PDF", "0") == "1":
        fig.savefig(out_path.with_suffix(".pdf"))
    if os.environ.get("MLXQ_SAVE_SVG", "0") == "1":
        fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)
