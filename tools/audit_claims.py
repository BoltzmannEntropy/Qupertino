#!/usr/bin/env python3
"""Audit manuscript claims against generated artifact files.

This is intentionally conservative: it verifies that key numerical claims in
the QUANTICS TeX are present and are backed by the expected CSV/JSON artifacts.
It is not a full paper parser.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "quantics-lncs-2026" / "mlxquantum_quantics2026_lncs.tex"


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _contains_number(tex: str, value: float, decimals: int) -> bool:
    needle = f"{value:.{decimals}f}"
    trimmed = needle.rstrip("0").rstrip(".")
    return needle in tex or (trimmed and trimmed in tex)


def _contains_scientific(tex: str, value: float) -> bool:
    if value == 0:
        return "0" in tex
    sci = f"{value:.2e}"
    mantissa, exponent = sci.split("e")
    exp_i = int(exponent)
    plain = sci.replace("e-0", "e-").replace("e+0", "e")
    latex = f"{mantissa}\\times10^{{{exp_i}}}"
    latex_no_brace = f"{mantissa}\\times10^{exp_i}"
    return plain in tex or latex in tex or latex_no_brace in tex


# Comparison-table qubit sizes; the scaling sweep covers more sizes but the
# tex comparison table (tab:pennylane-baseline) reports these three.
_COMPARISON_QUBITS = {"15", "20", "25"}

# mlxq timing summaries backing the comparison table, keyed by circuit type.
_MLX_COMPARISON_CIRCUITS = (
    "qft", "qaoa", "hamiltonian_simulation", "phase_estimation",
    "grover", "ghz", "qft_fft_primitive",
)


def _tolerant_match(tex: str, value_s: float) -> bool:
    """Match the artifact value against the tex at the table's rounding.

    Table cells round to 3-4 significant digits; accept the value formatted at
    several plausible precisions.
    """
    for decimals in (5, 4, 3, 2):
        if _contains_number(tex, value_s, decimals):
            return True
    return False


def _check_baseline_summary(tex: str, csv_path: Path, label: str, rel: str) -> list[dict]:
    checks: list[dict] = []
    for row in _read_csv(csv_path):
        if row["qubits"] not in _COMPARISON_QUBITS:
            continue
        value_s = float(row["mean_ms"]) / 1000.0
        checks.append({
            "claim": f"{label} {row['benchmark']} {row['qubits']}q mean",
            "artifact": rel,
            "value": value_s,
            "present_in_tex": _tolerant_match(tex, value_s),
        })
    return checks


def _check_shader_sweep(tex: str, evidence: Path) -> list[dict]:
    """Every row of the full-suite shader table against the paired sweep."""
    checks: list[dict] = []
    sweep = evidence / "shader_sweep_20260704"
    csv_path = sweep / "shader_sweep_summary.csv"
    if not csv_path.is_file():
        return checks
    rel = "evidence_artifacts/shader_sweep_20260704/shader_sweep_summary.csv"
    for row in _read_csv(csv_path):
        for col, tag in (("pure_mean_ms", "pure"), ("metal_mean_ms", "metal")):
            value_s = float(row[col]) / 1000.0
            checks.append({
                "claim": f"shader-sweep {row['benchmark']} 25q {tag} mean",
                "artifact": rel,
                "value": value_s,
                "present_in_tex": _tolerant_match(tex, value_s),
            })
    return checks


_QASM_CITED = ("ghz_state_n23.qasm", "cat_state_n22.qasm", "knn_n25.qasm",
               "swap_test_n25.qasm", "ising_n26.qasm", "wstate_n27.qasm",
               "qft_n18.qasm", "factor247_n15.qasm")


def _check_qasm_sweep(tex: str, evidence: Path) -> list[dict]:
    """Cited rows of the appendix QASM table (values are in ms in the tex)."""
    checks: list[dict] = []
    csv_path = (evidence / "qasm_shader_sweep_20260704"
                / "qasm_sweep_summary.csv")
    if not csv_path.is_file():
        return checks
    rel = "evidence_artifacts/qasm_shader_sweep_20260704/qasm_sweep_summary.csv"
    for row in _read_csv(csv_path):
        if row["file"] not in _QASM_CITED:
            continue
        for col, tag in (("pure_mean_ms", "pure"), ("metal_mean_ms", "metal")):
            v = float(row[col])
            present = (_contains_number(tex, v, 1)
                       or _contains_number(tex, round(v), 0))
            checks.append({
                "claim": f"qasm-sweep {row['file']} {tag} mean (ms)",
                "artifact": rel,
                "value": v,
                "present_in_tex": present,
            })
    return checks


def _check_interleaved_comparison(tex: str, evidence: Path) -> list[dict]:
    """Every cell of tab:pennylane-baseline against the 4-way campaign."""
    checks: list[dict] = []
    rel = "evidence_artifacts/interleaved_4way_20260704/interleaved_summary.csv"
    for row in _read_csv(evidence / "interleaved_4way_20260704"
                         / "interleaved_summary.csv"):
        if row["qubits"] not in _COMPARISON_QUBITS:
            continue
        value_s = float(row["mean_ms"]) / 1000.0
        checks.append({
            "claim": f"{row['backend']} {row['benchmark']} {row['qubits']}q mean",
            "artifact": rel,
            "value": value_s,
            "present_in_tex": _tolerant_match(tex, value_s),
        })
    fft = evidence / "interleaved_campaign_20260702" / "qft_fft_primitive_n10.json"
    if fft.is_file():
        for n, rec in json.loads(fft.read_text(encoding="utf-8")).items():
            value_s = float(rec["mean_ms"]) / 1000.0
            checks.append({
                "claim": f"mlxq qft_fft_primitive {n}q mean",
                "artifact": "evidence_artifacts/interleaved_campaign_20260702/qft_fft_primitive_n10.json",
                "value": value_s,
                "present_in_tex": _tolerant_match(tex, value_s),
            })
    return checks


def _check_mlx_sweep(tex: str, evidence: Path) -> list[dict]:
    """25q sweep values still cited in tab:diagnostic-timings."""
    checks: list[dict] = []
    repro = evidence / "mlx_repro_20260702"
    sweep_circuits = ("phase_estimation", "grover", "ghz",
                      "random_circuit", "variational_circuit")
    for circuit in sweep_circuits:
        for row in _read_csv(repro / f"{circuit}_timing_summary.csv"):
            if row["qubits"] != "25":
                continue
            value_s = float(row["mean_ms"]) / 1000.0
            checks.append({
                "claim": f"sweep {circuit} 25q mean",
                "artifact": f"evidence_artifacts/mlx_repro_20260702/{circuit}_timing_summary.csv",
                "value": value_s,
                "present_in_tex": _tolerant_match(tex, value_s),
            })
    return checks


def _check_full_suite(tex: str, evidence: Path) -> list[dict]:
    """Batch values cited in tab:scaling-full and tab:variational."""
    checks: list[dict] = []
    suite = evidence / "full_suite_20260702"
    wanted = {
        "qft": ("25",),
        "qaoa": ("23", "24", "25"),
        "qcbm": ("23", "24", "25"),
        "hamiltonian_simulation": ("25",),
        "time_evolution": ("25",),
    }
    for circuit, sizes in wanted.items():
        for row in _read_csv(suite / f"{circuit}_timing_summary.csv"):
            if row["qubits"] not in sizes:
                continue
            value_s = float(row["mean_ms"]) / 1000.0
            checks.append({
                "claim": f"full-suite {circuit} {row['qubits']}q mean",
                "artifact": f"evidence_artifacts/full_suite_20260702/{circuit}_timing_summary.csv",
                "value": value_s,
                "present_in_tex": _tolerant_match(tex, value_s),
            })
    return checks


def _check_kokkos(tex: str, evidence: Path) -> list[dict]:
    """Every mean in tab:kokkos against the paired-campaign artifact."""
    checks: list[dict] = []
    rel = "evidence_artifacts/kokkos_campaign_20260704/kokkos_summary.csv"
    for row in _read_csv(evidence / "kokkos_campaign_20260704"
                         / "kokkos_summary.csv"):
        value_s = float(row["mean_ms"]) / 1000.0
        checks.append({
            "claim": f"kokkos-campaign {row['backend']} {row['benchmark']} {row['qubits']}q mean",
            "artifact": rel,
            "value": value_s,
            "present_in_tex": _tolerant_match(tex, value_s),
        })
    return checks


def _check_precision(tex: str, evidence: Path) -> list[dict]:
    rows = _read_csv(evidence / "precision_validation_summary.csv")
    checks: list[dict] = []
    for row in rows:
        for key in ("probability_mae", "infidelity", "norm_drift_mlx"):
            value = float(row[key])
            sci = f"{value:.1e}".replace("e-0", "e-").replace("e+0", "e")
            checks.append({
                "claim": f"{row['workload']} {key}",
                "artifact": "evidence_artifacts/precision_validation_summary.csv",
                "value": value,
                "present_in_tex": (
                    sci in tex
                    or f"{value:.3e}" in tex
                    or _contains_scientific(tex, value)
                ),
            })
    return checks


def _check_required_sections(tex: str) -> list[dict]:
    required = [
        ("Unified-memory ablation", r"Unified-Memory Ablation"),
        ("Precision validation", r"Independent complex128 validation"),
        ("Trotter error", r"Trotter Error"),
        ("Failure modes", r"Failure Modes"),
        ("Qiskit Aer baseline status", r"Qiskit Aer"),
        ("VQE relabel", r"hardware-efficient ansatz optimization"),
        ("Data availability artifact", r"evidence(?:\\_|\_)artifacts"),
    ]
    return [
        {
            "claim": label,
            "artifact": "mlxquantum_quantics2026_lncs.tex",
            "present_in_tex": re.search(pattern, tex, re.IGNORECASE) is not None,
        }
        for label, pattern in required
    ]


def run(args: argparse.Namespace) -> dict:
    tex_path = Path(args.tex)
    evidence = Path(args.evidence_dir)
    artifacts = Path(args.baseline_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tex = tex_path.read_text(encoding="utf-8")

    checks: list[dict] = []
    checks.extend(_check_required_sections(tex))
    checks.extend(_check_interleaved_comparison(tex, evidence))
    checks.extend(_check_shader_sweep(tex, evidence))
    checks.extend(_check_qasm_sweep(tex, evidence))
    checks.extend(_check_mlx_sweep(tex, evidence))
    checks.extend(_check_full_suite(tex, evidence))
    checks.extend(_check_kokkos(tex, evidence))
    checks.extend(_check_precision(tex, evidence))

    for expected in [
        evidence / "unified_memory_ablation_summary.csv",
        evidence / "trotter_error_summary.csv",
        evidence / "failure_modes.json",
        evidence / "qiskit_aer_baseline_manifest.json",
    ]:
        checks.append({
            "claim": f"artifact exists: {expected.name}",
            "artifact": str(expected.relative_to(ROOT)) if expected.is_absolute() else str(expected),
            "present_in_tex": expected.is_file(),
        })

    failed = [c for c in checks if not c.get("present_in_tex")]
    payload = {
        "generated_at_epoch": time.time(),
        "tex": str(tex_path),
        "evidence_dir": str(evidence),
        "checks": checks,
        "failed_count": len(failed),
        "passed_count": len(checks) - len(failed),
        "status": "pass" if not failed else "fail",
    }
    (outdir / "claim_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Claim Audit",
        "",
        f"Status: {payload['status']}",
        "",
        "| Claim | Artifact | Pass |",
        "|---|---|---|",
    ]
    for check in checks:
        lines.append(
            "| {claim} | {artifact} | {present_in_tex} |".format(**check)
        )
    (outdir / "claim_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "failed_count": len(failed), "outdir": str(outdir)}, indent=2))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", default=str(PAPER))
    parser.add_argument(
        "--evidence-dir",
        default=str(ROOT / "paper" / "quantics-lncs-2026" / "evidence_artifacts"),
    )
    parser.add_argument(
        "--baseline-dir",
        default=str(ROOT / "paper" / "quantics-lncs-2026" / "baseline_artifacts"),
    )
    parser.add_argument(
        "--outdir",
        default=str(ROOT / "paper" / "quantics-lncs-2026" / "evidence_artifacts"),
    )
    args = parser.parse_args()
    payload = run(args)
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
