#!/usr/bin/env python3
"""
Python replica of bench_with_logging.sh – runs full suite and logs to bench/.
Generates BENCH_REPORT.md and copies plots to paper/images and assets/benchmarks.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from runpy import run_path

# Ensure local package path before importing mlxq
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlxq.pretty import info, success, warn, error, table
from mlxq.mlxQbench import run_scaling_benchmark, run_qasm_suite


class Tee:
    def __init__(self, path: Path):
        self.file = open(path, 'a', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()
    def flush(self):
        self.stdout.flush()
        self.file.flush()


def _run_core_tests():
    from importlib import util
    tests_path = Path(__file__).resolve().parents[1] / 'tests' / 'run_core_tests.py'
    spec = util.spec_from_file_location("mlxQCoreTestRunner", str(tests_path))
    mod = util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)


def _call_reports():
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / 'scripts',
        here.parents[2] / 'src' / 'scripts',
        here.parents[3] / 'src' / 'scripts',
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


def _copy_plots():
    src = Path('bench')
    dests = [
        Path('paper') / 'images',                 # historical default
        Path('paper') / 'prx-quantum' / 'images', # PRX paper layout
        Path('assets') / 'benchmarks',            # web assets
    ]
    for d in dests:
        d.mkdir(parents=True, exist_ok=True)
    # Per-benchmark scaling plots
    for p in src.glob('*_scaling.png'):
        for d in dests:
            try:
                (d / p.name).write_bytes(p.read_bytes())
            except Exception:
                pass
    # Aggregate comparison
    cmp = src / 'all_benchmarks_comparison.png'
    if cmp.exists():
        for d in dests:
            try:
                (d / cmp.name).write_bytes(cmp.read_bytes())
            except Exception:
                pass
    # Visualization appendix artifacts (mlxQ vs PennyLane) and GHZ distributions
    for p in list(src.glob('vis_*_side_by_side.png')) + \
             list(src.glob('vis_*_mlxq.png')) + \
             list(src.glob('vis_*_pl.png')) + \
             list(src.glob('vis_*_hist_side_by_side.png')) + \
             list(src.glob('vqe_convergence_*.png')) + \
             list(src.glob('ghz*_distribution_*.png')):
        for d in dests:
            try:
                (d / p.name).write_bytes(p.read_bytes())
            except Exception:
                pass


def _write_report(logfile: Path):
    report = Path('BENCH_REPORT.md')
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(report, 'w', encoding='utf-8') as f:
        f.write(f"# mlxQ Benchmark Report\n\n")
        f.write(f"**Generated:** {ts}\n\n")
        f.write(f"**Log File:** {logfile}\n\n---\n\n## Benchmark Results\n\n")
        f.write("```\n")
        try:
            f.write(Path(logfile).read_text(encoding='utf-8'))
        except Exception as e:
            f.write(f"<log not available: {e}>\n")
        f.write("```\n\n---\n\n## Generated Files\n\n### Data Files\n- JSON: `bench/*_mlx_quantum.json`\n- CSV: `bench/*_data.csv`\n\n### Visualizations\n- Plots: `bench/*_scaling.png`\n- Comparison: `bench/all_benchmarks_comparison.png`\n\n### Locations\n- Paper images: `paper/images/`\n- Asset benchmarks: `assets/benchmarks/`\n\n---\n\n")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _cap_for(key: str, fallback: int) -> int:
    uc = key.upper()
    env_name = f"MLXQ_CAP_{uc}"
    try:
        val = os.environ.get(env_name)
        if val is not None and val != "":
            return int(val)
    except Exception:
        pass
    return fallback


def main():
    Path('bench').mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = Path('bench') / f"BENCHMARK_RUN_{ts}.log"
    latest = Path('bench') / "LATEST_BENCHMARK.log"
    # Tee stdout
    tee = Tee(log_file)
    sys.stdout = tee
    try:
        print("📊 Starting mlxQ Benchmark Suite")
        print(f"📝 Logging to: {log_file}")
        print(f"📝 Also updating: {latest}")
        print("")
        print("===================================")
        print("mlxQ Benchmark Run")
        print(f"Started: {datetime.now()}")
        print("===================================")
        print("")

        # Core tests
        _run_core_tests()

        # Single-circuit mode (if requested via env)
        one = os.environ.get('MLXQ_ONE_CIRCUIT', '').strip()
        if one:
            def _parse_qubit_spec(spec: str):
                spec = (spec or '').strip()
                if not spec:
                    return None
                if '-' in spec and all(p.isdigit() for p in spec.split('-',1)):
                    a, b = spec.split('-', 1)
                    a = int(a); b = int(b)
                    return list(range(min(a,b), max(a,b)+1))
                out = []
                for tok in spec.split(','):
                    tok = tok.strip()
                    if tok.isdigit():
                        out.append(int(tok))
                return out or None
            qs = _parse_qubit_spec(os.environ.get('MLXQ_ONE_QUBITS', '')) or \
                 _parse_qubit_spec(os.environ.get('MLXQ_PUB_QUBITS', ''))
            if not qs:
                max_q = _env_int('MLXQ_MAX_QUBITS', 25)
                base = [1,2,5,7,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
                qs = [q for q in base if q <= max_q]
            try:
                cap = os.environ.get('MLXQ_ONE_CAP')
                cap_i = int(cap) if cap else None
            except Exception:
                cap_i = None
            print(f"=== Single-circuit run: {one} (qubits: {qs}, cap: {cap_i}) ===")
            run_scaling_benchmark(one, qs, simulate_cap=cap_i)
            print("")
            print("=== Generating core test reports ===")
            _call_reports()
            print("")
            print("=== Copying plots to paper/images and assets/benchmarks ===")
            _copy_plots()
            print("✅ Plots copied.")
            return

        # Full suite
        print("")
        print("=== Running full benchmark suite ===")
        # Compose canonical lists with env-configurable global cap
        max_q = _env_int('MLXQ_MAX_QUBITS', 25)
        base = [1,2,5,7,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
        pub_qubits = [q for q in base if q <= max_q]
        vqe_qubits = [q for q in [1,2,5,7,10,11,12,13,14,15] if q <= max_q]
        # Match legacy steady_state list exactly
        steady_qubits = [q for q in [1,2,5,7,10,11,12] if q <= max_q]

        # Allow explicit override via env: MLXQ_PUB_QUBITS, MLXQ_VQE_QUBITS, MLXQ_STEADY_QUBITS
        def _parse_qubit_spec(spec: str):
            spec = (spec or '').strip()
            if not spec:
                return None
            if '-' in spec and all(p.isdigit() for p in spec.split('-',1)):
                a, b = spec.split('-', 1)
                a = int(a); b = int(b)
                if a <= b:
                    return list(range(a, b+1))
                else:
                    return list(range(b, a+1))
            out = []
            for tok in spec.split(','):
                tok = tok.strip()
                if tok.isdigit():
                    out.append(int(tok))
            return out or None

        env_pub = os.environ.get('MLXQ_PUB_QUBITS', '')
        env_vqe = os.environ.get('MLXQ_VQE_QUBITS', '')
        env_steady = os.environ.get('MLXQ_STEADY_QUBITS', '')
        qp = _parse_qubit_spec(env_pub)
        qv = _parse_qubit_spec(env_vqe)
        qs = _parse_qubit_spec(env_steady)
        if qp:
            pub_qubits = qp
        if qv:
            vqe_qubits = qv
        if qs:
            steady_qubits = qs

        # Per-benchmark caps (overridable via MLXQ_CAP_* env vars)
        cap_pub = min(max_q, 30)  # hamiltonian_simulation default cap
        cap_time_evolution = min(max_q, 30)
        cap_trotter = min(max_q, 30)
        cap_random = max_q
        cap_qcbm = max_q
        cap_phase = _cap_for('phase_estimation', min(max_q, 15))
        cap_qft = max_q
        cap_qaoa = max_q
        cap_variational = max_q
        cap_grover = max_q
        cap_ghz = max_q
        # Hard benches defaults tuned for 32GB M1
        cap_vqe = _cap_for('vqe', min(max_q, 15))
        cap_steady = _cap_for('steady_state', min(max_q, 12))
        cap_heis = _cap_for('heisenberg', min(max_q, 30))

        try:
            # Main suite (VQE moved to hard benches section below)
            run_scaling_benchmark('hamiltonian_simulation', pub_qubits, simulate_cap=_cap_for('hamiltonian_simulation', cap_pub))
            run_scaling_benchmark('time_evolution',        pub_qubits, simulate_cap=_cap_for('time_evolution', cap_time_evolution))
            run_scaling_benchmark('trotter',               pub_qubits, simulate_cap=_cap_for('trotter', cap_trotter))
            run_scaling_benchmark('steady_state',          steady_qubits, simulate_cap=cap_steady)
            run_scaling_benchmark('heisenberg',            pub_qubits, simulate_cap=cap_heis)
            run_scaling_benchmark('heisenberg_xxz',        pub_qubits, simulate_cap=_cap_for('heisenberg_xxz', cap_trotter))
            run_scaling_benchmark('heisenberg_random_field', pub_qubits, simulate_cap=_cap_for('heisenberg_random_field', cap_trotter))
            run_scaling_benchmark('tfim',                  pub_qubits, simulate_cap=_cap_for('tfim', cap_trotter))
            run_scaling_benchmark('tfim_trotter2',         pub_qubits, simulate_cap=_cap_for('tfim_trotter2', cap_trotter))
            run_scaling_benchmark('tfim_random_field',     pub_qubits, simulate_cap=_cap_for('tfim_random_field', cap_trotter))
            run_scaling_benchmark('long_range_ising',      pub_qubits, simulate_cap=_cap_for('long_range_ising', cap_trotter))
            run_scaling_benchmark('ladder_heisenberg',     pub_qubits, simulate_cap=_cap_for('ladder_heisenberg', cap_trotter))
            run_scaling_benchmark('random_circuit',        pub_qubits, simulate_cap=_cap_for('random_circuit', cap_random))
            run_scaling_benchmark('qcbm',                  pub_qubits, simulate_cap=_cap_for('qcbm', cap_qcbm))
            run_scaling_benchmark('phase_estimation',      pub_qubits, simulate_cap=cap_phase)
            run_scaling_benchmark('qft',                   pub_qubits, simulate_cap=_cap_for('qft', cap_qft))
            run_scaling_benchmark('qaoa',                  pub_qubits, simulate_cap=_cap_for('qaoa', cap_qaoa))
            run_scaling_benchmark('variational_circuit',   pub_qubits, simulate_cap=_cap_for('variational_circuit', cap_variational))
            run_scaling_benchmark('grover',                pub_qubits, simulate_cap=_cap_for('grover', cap_grover))
            run_scaling_benchmark('ghz',                   pub_qubits, simulate_cap=_cap_for('ghz', cap_ghz))

            # Hard benches (32GB M1 safe caps)
            print("")
            print("=== Hard Benches (32GB M1) ===")
            run_scaling_benchmark('vqe', vqe_qubits, simulate_cap=cap_vqe)

            # QASM
            print("")
            print("=== OpenQASM Circuit Benchmarks ===")
            os.environ.setdefault('QASM_MAX_QUBITS', '18')
            os.environ.setdefault('QASM_MAX_MEM_MB', '4096')
            os.environ.setdefault('QASM_TIMEOUT_MS', '0')  # unlimited by default
            run_qasm_suite()
        except KeyboardInterrupt:
            print("\n⚠️  Aborted by user (Ctrl+C). Partial results saved.")
            return

        # Reports
        print("")
        print("=== Generating core test reports ===")
        _call_reports()

        # Copy plots
        print("")
        print("=== Copying plots to paper/images and assets/benchmarks ===")
        _copy_plots()
        print("✅ Plots copied.")

        print("")
        print("===================================")
        print(f"Benchmark Run Completed: {datetime.now()}")
        print("===================================")
    finally:
        # Mirror to latest log
        try:
            latest.write_text(log_file.read_text(encoding='utf-8'), encoding='utf-8')
        except Exception:
            pass
        # restore stdout
        sys.stdout = tee.stdout

    # BENCH_REPORT.md
    print("\n📝 Generating BENCH_REPORT.md from log...")
    _write_report(log_file)
    print(f"✅ Report saved to: BENCH_REPORT.md")
    print(f"✅ Full log saved to: {log_file}")
    print(f"✅ Latest log: {latest}")
    print("✅ Done!")


if __name__ == '__main__':
    # Ensure package path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
