#!/usr/bin/env python3
"""
Python replica of bench.sh – runs mlxq benchmarks end-to-end with no estimates.
Outputs CSV/JSON under bench/ compatible with src/scripts/generate_*.py.
"""

import os
import sys
import importlib.util
from pathlib import Path

# Ensure local package path before importing mlxq
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # add python/

from mlxq.pretty import info, success, warn, error, table
from mlxq.mlxQbench import run_scaling_benchmark, run_qasm_suite
from mlxq.vendor import VENDOR_BENCHMARKS, ALGORITHM_BENCHMARKS, BENCH_KEYS


def _run_core_tests():
    info("=== Testing mlxQCoreTest (Python) ===")
    tests_path = Path(__file__).resolve().parents[1] / 'tests' / 'run_core_tests.py'
    spec = importlib.util.spec_from_file_location("mlxQuantumCoreTestRunner", str(tests_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)


def _call_reports():
    # Trigger plot/report generation if dependencies are available
    from runpy import run_path
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / 'scripts',          # new layout: src/python/scripts -> parents[2]=src
        here.parents[2] / 'src' / 'scripts',  # old layout: python/scripts -> parents[2]=repo
        here.parents[3] / 'src' / 'scripts',  # fallback
    ]
    base = None
    for c in candidates:
        if c.exists():
            base = c; break
    if base is None:
        base = here.parents[2] / 'scripts'
    for script in [
        'generate_coretest_report.py',
        'generate_plots.py',
        'plot_all_benchmarks.py',
        'generate_report.py',
    ]:
        p = base / script
        if p.exists():
            try:
                info(f"Running {script}")
                run_path(str(p))
            except SystemExit:
                pass
            except Exception as e:
                warn(f"{script} skipped: {e}")


def main():
    os.makedirs('bench', exist_ok=True)
    # 1) Core tests
    _run_core_tests()

    # 2) Full benchmark suite (no estimates; we only run actual qubits lists)
    info("=== Running full benchmark suite ===")
    # Global cap can be overridden via MLXQ_MAX_QUBITS
    try:
        max_q = int(os.environ.get('MLXQ_MAX_QUBITS', '25'))
    except Exception:
        max_q = 25
    info("Vendor groups:")
    for v, ks in VENDOR_BENCHMARKS.items():
        print(f"  - {v}: {', '.join(ks)}")
    info("Algorithm groups:")
    for g, ks in ALGORITHM_BENCHMARKS.items():
        print(f"  - {g}: {', '.join(ks)}")
    # Compose canonical lists up to the global cap
    base = [1,2,5,7,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    pub_qubits = [q for q in base if q <= max_q]
    vqe_qubits = [q for q in [1,2,5,7,10,11,12,13,14,15] if q <= max_q]
    # Match legacy/bench.sh exactly for steady_state: 1,2,5,7,10,11,12
    steady_qubits = [q for q in [1,2,5,7,10,11,12] if q <= max_q]

    try:
        run_scaling_benchmark('hamiltonian_simulation', pub_qubits, simulate_cap=min(max_q, 30))
        run_scaling_benchmark('time_evolution', pub_qubits, simulate_cap=min(max_q, 30))
        run_scaling_benchmark('trotter', pub_qubits, simulate_cap=min(max_q, 30))
        run_scaling_benchmark('steady_state', steady_qubits, simulate_cap=min(max_q, 12))
        run_scaling_benchmark('heisenberg', pub_qubits, simulate_cap=min(max_q, 30))
        run_scaling_benchmark('heisenberg_xxz', pub_qubits, simulate_cap=max_q)
        run_scaling_benchmark('heisenberg_random_field', pub_qubits, simulate_cap=max_q)
        run_scaling_benchmark('tfim', pub_qubits, simulate_cap=max_q)
        run_scaling_benchmark('tfim_trotter2', pub_qubits, simulate_cap=max_q)
        run_scaling_benchmark('tfim_random_field', pub_qubits, simulate_cap=max_q)
        run_scaling_benchmark('long_range_ising', pub_qubits, simulate_cap=max_q)
        run_scaling_benchmark('ladder_heisenberg', pub_qubits, simulate_cap=max_q)
        run_scaling_benchmark('random_circuit', pub_qubits, simulate_cap=max_q)
        run_scaling_benchmark('qcbm', pub_qubits, simulate_cap=max_q)
        run_scaling_benchmark('phase_estimation', pub_qubits, simulate_cap=min(max_q, 15))
        run_scaling_benchmark('qft', pub_qubits, simulate_cap=max_q)
        run_scaling_benchmark('qaoa', pub_qubits, simulate_cap=max_q)
        run_scaling_benchmark('vqe', vqe_qubits, simulate_cap=min(max_q, 15))
        run_scaling_benchmark('variational_circuit', pub_qubits, simulate_cap=max_q)
        run_scaling_benchmark('grover', pub_qubits, simulate_cap=max_q)
        run_scaling_benchmark('ghz', pub_qubits, simulate_cap=max_q)

        # 3) QASM suite
        info("=== OpenQASM Circuit Benchmarks ===")
        os.environ.setdefault('QASM_MAX_QUBITS', '18')
        os.environ.setdefault('QASM_MAX_MEM_MB', '4096')
        # Unlimited timeout by default; set QASM_TIMEOUT_MS to cap
        os.environ.setdefault('QASM_TIMEOUT_MS', '0')
        run_qasm_suite()
    except KeyboardInterrupt:
        warn("Aborted by user (Ctrl+C). Partial results saved.")
        return

    # 3b) Optional vendor/algorithm group runs
    if os.environ.get('MLXQ_VENDOR_SUITE', '0') == '1':
        info("=== Running vendor groups (MLXQ_VENDOR_SUITE=1) ===")
        for vendor in VENDOR_BENCHMARKS.keys():
            run_vendor_group(vendor, pub_qubits, simulate_cap=max_q)
    if os.environ.get('MLXQ_ALGO_GROUPS', '0') == '1':
        info("=== Running algorithm groups (MLXQ_ALGO_GROUPS=1) ===")
        for group in ALGORITHM_BENCHMARKS.keys():
            run_algorithm_group(group, pub_qubits, simulate_cap=max_q)

    # 4) Reports/plots
    info("=== Generating core test reports & plots ===")
    _call_reports()

    success("Benchmark suite complete. See bench/ for outputs.")


if __name__ == '__main__':
    main()
