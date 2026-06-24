#!/usr/bin/env python3
"""
Job worker script for QuantumStudio benchmark execution.

This script is invoked as a subprocess by the main API server.
It executes benchmarks and writes output to stdout (which is redirected to log file).

Usage: python job_worker.py <run_id> <payload_json>
"""
from __future__ import annotations

import json
import os
import sys
import traceback
import contextlib
import platform
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STUDIO_ROOT = SCRIPT_DIR.parent
MLX_ROOT = Path(
    os.environ.get("QUANTUMSTUDIO_MLX_ROOT", str(DEFAULT_STUDIO_ROOT.parent)),
)
BENCH_DIR = Path(
    os.environ.get("QUANTUMSTUDIO_BENCH_DIR", str(MLX_ROOT / "bench")),
)
MLX_PYTHON = Path(
    os.environ.get("QUANTUMSTUDIO_MLX_PYTHON", str(MLX_ROOT / "src")),
)

BENCH_DIR.mkdir(parents=True, exist_ok=True)

if str(MLX_PYTHON) not in sys.path:
    sys.path.insert(0, str(MLX_PYTHON))

os.environ.setdefault("MPLBACKEND", "Agg")


def _parse_qubits_spec(spec: str) -> List[int]:
    """Parse qubit specification like '1-12' or '1,2,5,7,10'."""
    cleaned = spec.replace(" ", "")
    if not cleaned:
        raise ValueError("empty qubit spec")
    parts = cleaned.split(",")
    qubits: List[int] = []
    for part in parts:
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if start <= 0 or end <= 0:
                raise ValueError("qubits must be positive integers")
            if start > end:
                raise ValueError("range start must be <= end")
            qubits.extend(range(start, end + 1))
        else:
            value = int(part)
            if value <= 0:
                raise ValueError("qubits must be positive integers")
            qubits.append(value)
    seen = set()
    ordered: List[int] = []
    for q in qubits:
        if q not in seen:
            seen.add(q)
            ordered.append(q)
    return ordered


def _default_qubits_for(name: str, max_qubits: int) -> List[int]:
    """Get default qubit list for a benchmark."""
    env_key = f"MLXQ_LIST_{name.upper()}"
    env_val = os.environ.get(env_key)
    if env_val:
        return [q for q in _parse_qubits_spec(env_val) if q <= max_qubits]
    list25 = [1, 2, 5, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    list30 = list25 + [26, 27, 28]
    vqe_list = [1, 2, 5, 7, 10, 11, 12, 13, 14, 15]
    steady_list = [1, 2, 5, 7, 10, 11, 12]
    list30_keys = {
        "hamiltonian_simulation", "time_evolution", "trotter", "heisenberg",
        "heisenberg_xxz", "heisenberg_random_field", "tfim", "tfim_trotter2",
        "tfim_random_field", "long_range_ising", "ladder_heisenberg",
    }
    if name == "vqe":
        base = vqe_list
    elif name == "steady_state":
        base = steady_list
    elif name in list30_keys:
        base = list30
    else:
        base = list25
    return [q for q in base if q <= max_qubits]


@contextlib.contextmanager
def _temp_environ(overrides: Dict[str, Optional[str]]):
    """Temporarily set environment variables."""
    previous: Dict[str, Optional[str]] = {}
    for key, value in overrides.items():
        previous[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _detect_hw_label() -> tuple:
    """Detect hardware label for file naming."""
    gen = "Unknown"
    var = "Base"
    if platform.system() == "Darwin":
        brand = ""
        try:
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
            brand = out.decode("utf-8", "ignore").strip()
        except Exception:
            brand = ""
        if "M1" in brand:
            gen = "M1"
        elif "M2" in brand:
            gen = "M2"
        elif "M3" in brand:
            gen = "M3"
        elif "M4" in brand:
            gen = "M4"
        if "Max" in brand:
            var = "Max"
        elif "Pro" in brand:
            var = "Pro"
        elif "Ultra" in brand:
            var = "Ultra"
    return f"{gen}_{var}", f"{gen} {var}"


def _generate_ghz_distributions() -> None:
    """Generate GHZ distribution plots."""
    try:
        import matplotlib.pyplot as plt
        from mlxq.mlxQdevice import Device
    except Exception:
        print("GHZ plot generation unavailable (missing deps).")
        traceback.print_exc()
        return

    def ghz_ops(n: int):
        ops = [{"name": "H", "wires": [0]}]
        for i in range(1, n):
            ops.append({"name": "CNOT", "wires": [i - 1, i]})
        return ops

    outdir = BENCH_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    hw_prefix, hw_label = _detect_hw_label()
    for n in (4, 5, 6):
        dev = Device(n, shots=10000)
        dev.execute(ghz_ops(n))
        counts = dev.counts(shots=10000)
        total = max(1, sum(counts.values()))
        csv_path = outdir / f"ghz{n}_distribution_{hw_prefix}.csv"
        with csv_path.open("w") as f:
            f.write("bitstring,count,probability\n")
            for k in sorted(counts.keys()):
                v = counts[k]
                f.write(f"{k},{v},{v/total:.6f}\n")
        keys = sorted(counts.keys())
        probs = [counts[k] / total for k in keys]
        plt.figure(figsize=(12, 6))
        plt.bar(keys, probs, color="#00AA00")
        plt.axhline(0.5, color="red", linewidth=2, linestyle="--")
        plt.ylim(0, 0.6)
        plt.grid(True, axis="y", linestyle=":", alpha=0.6)
        plt.title(f"GHZ-{n} Measurement Distribution (10000 shots on {hw_label})")
        plt.xlabel("Measurement Outcome")
        plt.ylabel("Probability")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        png_path = outdir / f"ghz{n}_distribution_{hw_prefix}.png"
        plt.savefig(png_path, dpi=100)
        plt.close()
    print("GHZ distributions generated.")


def run_job(payload: Dict[str, Any]) -> int:
    """Execute the benchmark job. Returns exit code."""
    benchmark_configs = payload.get("benchmark_configs", [])
    save_plots = payload.get("save_plots", True)
    max_qubits = payload.get("max_qubits") or 25
    env_overrides = payload.get("env_overrides", {})
    run_env_overrides = {k: v for k, v in env_overrides.items() if str(v).strip()}

    os.environ["MLXQ_SAVE_PLOTS"] = "1" if save_plots else "0"
    exit_code = 0

    try:
        from mlxq import bench as mlxq_bench
    except Exception:
        print("Failed to import mlxq benchmarks.")
        traceback.print_exc()
        return 1

    for cfg in benchmark_configs:
        name = cfg.get("name")
        qubits_spec = cfg.get("qubits_spec", "default")
        backend = cfg.get("backend", "sv")
        simulate_cap = cfg.get("simulate_cap")

        try:
            if qubits_spec.lower() in {"default", "auto"}:
                qubits = _default_qubits_for(name, max_qubits)
            else:
                qubits = _parse_qubits_spec(qubits_spec)
        except Exception as exc:
            print(f"Invalid qubit spec '{qubits_spec}': {exc}")
            exit_code = 1
            break

        env_for_run: Dict[str, Optional[str]] = {
            **run_env_overrides,
            "MLXQ_BACKEND": backend or None,
        }

        print(f"\n=== Running {name} (qubits={qubits_spec}, backend={backend}) ===")
        sys.stdout.flush()

        try:
            with _temp_environ(env_for_run):
                if name == "qasm":
                    qasm_max = payload.get("qasm_max_qubits") or simulate_cap or (max(qubits) if qubits else 18)
                    qasm_env = {
                        "QASM_MAX_QUBITS": str(qasm_max),
                        "QASM_TIMEOUT_MS": str(payload.get("qasm_timeout_ms")) if payload.get("qasm_timeout_ms") else None,
                        "QASM_MAX_MEM_MB": str(payload.get("qasm_max_mem_mb")) if payload.get("qasm_max_mem_mb") else None,
                        "QASM_INCLUDE_LARGE": "1" if payload.get("qasm_include_large") else None,
                        "QASM_SIMULATE_LIMIT": str(payload.get("qasm_simulate_limit")) if payload.get("qasm_simulate_limit") else None,
                    }
                    with _temp_environ(qasm_env):
                        mlxq_bench.run_qasm_suite(
                            csv_out=str(BENCH_DIR / "qasm_MLX_python.csv"),
                            json_out=str(BENCH_DIR / "qasm_MLX_python.json"),
                        )
                else:
                    mlxq_bench.run_scaling_benchmark(
                        name,
                        qubits,
                        simulate_cap=simulate_cap,
                        out_prefix=str(BENCH_DIR),
                    )
        except Exception:
            print("\nBenchmark execution failed:")
            traceback.print_exc()
            exit_code = 1
            break

    # Post-processing
    if exit_code == 0 and save_plots:
        print("\n=== Post-processing (aggregate plots & reports) ===")
        sys.stdout.flush()

        # Run aggregate plots
        aggregate_script = MLX_ROOT / "src" / "benchmark" / "aggregate_plots.py"
        if aggregate_script.exists():
            try:
                subprocess.run(
                    [sys.executable, str(aggregate_script)],
                    cwd=str(MLX_ROOT),
                    check=False,
                )
            except Exception:
                print(f"Warning: aggregate_plots.py failed")
                traceback.print_exc()

        # Run MPS report
        mps_script = MLX_ROOT / "tools" / "mps_report.py"
        if mps_script.exists():
            try:
                subprocess.run(
                    [sys.executable, str(mps_script), "--bench", str(BENCH_DIR)],
                    cwd=str(MLX_ROOT),
                    check=False,
                )
            except Exception:
                print(f"Warning: mps_report.py failed")
                traceback.print_exc()

        # Benchpress report
        if payload.get("benchpress"):
            benchpress_script = MLX_ROOT / "src" / "benchmark" / "benchpress_replica.py"
            if benchpress_script.exists():
                try:
                    subprocess.run(
                        [sys.executable, str(benchpress_script)],
                        cwd=str(MLX_ROOT),
                        check=False,
                    )
                except Exception:
                    print(f"Warning: benchpress_replica.py failed")
                    traceback.print_exc()

        # GHZ distributions
        _generate_ghz_distributions()

    return exit_code


def main():
    if len(sys.argv) < 3:
        print("Usage: job_worker.py <run_id> <payload_json>")
        sys.exit(1)

    run_id = sys.argv[1]
    payload_json = sys.argv[2]

    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError as e:
        print(f"Invalid payload JSON: {e}")
        sys.exit(1)

    print(f"Starting job: {run_id}")
    print(f"Benchmarks: {[c.get('name') for c in payload.get('benchmark_configs', [])]}")
    sys.stdout.flush()

    exit_code = run_job(payload)
    print(f"\nJob completed with exit code: {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
