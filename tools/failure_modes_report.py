#!/usr/bin/env python3
"""Generate a failure-mode report for osxQuantum paper artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QASM_DIR = ROOT / "datasets" / "qasm" / "local"


def _state_mb(n: int, bytes_per_amp: int = 8) -> float:
    return (float(1 << n) * bytes_per_amp) / (1024.0 * 1024.0)


def _memory_rows(total_gb: float, workspace_multipliers: list[float]) -> list[dict]:
    rows: list[dict] = []
    for n in range(20, 37):
        state_gb = _state_mb(n) / 1024.0
        row = {"qubits": n, "state_gb_complex64": state_gb}
        for mult in workspace_multipliers:
            row[f"fits_{mult:g}x_workspace"] = (state_gb * mult) < total_gb
            row[f"workspace_gb_{mult:g}x"] = state_gb * mult
        rows.append(row)
    return rows


def _qasm_diagnostics() -> list[dict]:
    rows: list[dict] = []
    if not QASM_DIR.is_dir():
        return rows
    skipped_patterns = {
        "measure": re.compile(r"\bmeasure\b"),
        "reset": re.compile(r"\breset\b"),
        "classical_if": re.compile(r"\bif\s*\("),
        "barrier": re.compile(r"\bbarrier\b"),
        "opaque": re.compile(r"\bopaque\b"),
    }
    for path in sorted(QASM_DIR.glob("*.qasm")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        row = {"file": path.name}
        for name, pattern in skipped_patterns.items():
            row[name] = len(pattern.findall(text))
        row["semantic_caveat"] = any(row[name] for name in skipped_patterns)
        rows.append(row)
    return rows


def run(args: argparse.Namespace) -> dict:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    memory = _memory_rows(args.memory_gb, [1.0, 2.0, 4.0])
    qasm = _qasm_diagnostics()
    failure_modes = [
        {
            "mode": "out_of_memory_or_swap",
            "status": "bounded_by_capacity_model",
            "finding": (
                "A single complex64 state first exceeds 32 GB at 32 qubits; "
                "a conservative 4x-workspace model first exceeds 32 GB at 30 qubits."
            ),
            "evidence": "failure_modes_memory.csv",
        },
        {
            "mode": "small_qubit_overhead",
            "status": "known_risk_not_claimed_as_speedup",
            "finding": (
                "For small circuits, Python dispatch and MLX graph/synchronization overhead can dominate; "
                "the paper therefore does not claim low-qubit throughput leadership."
            ),
            "evidence": "benchmark timing summaries and external baselines",
        },
        {
            "mode": "OpenQASM_nonunitary_semantics",
            "status": "diagnosed",
            "finding": (
                "The parser executes the unitary subset. measure/reset/if/barrier/opaque lines are skipped "
                "or treated as non-executable metadata, so dynamic-circuit semantics are outside scope."
            ),
            "evidence": "failure_modes_qasm.csv",
        },
        {
            "mode": "precision_drift",
            "status": "measured_for_representative_20_25q_cases",
            "finding": (
                "See precision_validation_summary.csv for QFT 25q analytical complex128 and ring-QAOA "
                "20q NumPy complex128 checks."
            ),
            "evidence": "precision_validation_summary.csv",
        },
        {
            "mode": "variational_nonconvergence",
            "status": "scope_limited",
            "finding": (
                "The variational benchmark is relabeled as hardware-efficient ansatz optimization; "
                "no molecular VQE convergence claim is made."
            ),
            "evidence": "manuscript scope text",
        },
    ]

    with (outdir / "failure_modes_memory.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(memory[0].keys()))
        writer.writeheader()
        writer.writerows(memory)
    if qasm:
        with (outdir / "failure_modes_qasm.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(qasm[0].keys()))
            writer.writeheader()
            writer.writerows(qasm)

    payload = {
        "manifest": {
            "python": sys.version,
            "platform": platform.platform(),
            "assumed_memory_gb": args.memory_gb,
            "qasm_dir": str(QASM_DIR),
        },
        "failure_modes": failure_modes,
        "memory_capacity": memory,
        "qasm_diagnostics": qasm,
    }
    (outdir / "failure_modes.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Failure Modes",
        "",
        "| Mode | Status | Finding | Evidence |",
        "|---|---|---|---|",
    ]
    for row in failure_modes:
        lines.append("| {mode} | {status} | {finding} | {evidence} |".format(**row))
    lines.extend([
        "",
        "## Capacity Thresholds",
        "",
        "| Qubits | State GB | 4x workspace GB | Fits 4x in assumed memory |",
        "|---:|---:|---:|---|",
    ])
    for row in memory:
        if row["qubits"] >= 28:
            lines.append(
                "| {qubits} | {state_gb_complex64:.2f} | {workspace_gb_4x:.2f} | {fits_4x_workspace} |".format(**row)
            )
    (outdir / "failure_modes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"outdir": str(outdir), "failure_modes": failure_modes}, indent=2))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--memory-gb", type=float, default=32.0)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
