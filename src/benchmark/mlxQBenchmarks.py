import os
import sys
# Ensure local package path so `python3 python/tests/mlxQuantumBenchmarks.py` works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List
from mlxq.pretty import info, success, warn, error, table, console
from mlxq.mlxQbench import (
    bench_gate_suite, run_qasm_suite, run_scaling_benchmark,
)
from mlxq.metrics import now_ms
from mlxq.device import Device


def _parse_qubits_env(var: str, default: List[int]) -> List[int]:
    val = os.environ.get(var)
    if not val:
        return default
    try:
        return [int(x) for x in val.split(',') if x.strip()]
    except Exception:
        return default


def generate_ghz_distributions(qubits_list: List[int] = [4,5,6], shots: int = 10000):
    os.makedirs('bench', exist_ok=True)
    for n in qubits_list:
        info(f"GHZ distribution: {n}q, {shots} shots")
        dev = Device(n, shots=shots)
        dev.execute([{ "name": "H", "wires": [0] }] + [ {"name": "CNOT", "wires": [q, q+1]} for q in range(n-1)])
        counts = dev.counts()
        total = sum(counts.values())
        # write CSV
        csv_path = f"bench/ghz{n}_distribution.csv"
        with open(csv_path, 'w') as f:
            f.write("bitstring,count,freq\n")
            for k,v in sorted(counts.items()):
                f.write(f"{k},{v},{v/total:.6f}\n")
        success(f"CSV: {csv_path}")


def main():
    start = now_ms()
    info("=== Running full benchmark suite ===")
    # 1) Gate micro-benchmarks
    bench_gate_suite(reps=int(os.environ.get('GATE_REPS', '1000')))

    # 2) Circuit sanity checks
    q_small = list(range(1, min(12, int(os.environ.get('MAX_QFT_Q', '12'))) + 1))
    run_scaling_benchmark('qft', q_small)
    run_scaling_benchmark('qcbm', _parse_qubits_env('QCBM_QUBITS', [1,2,5,7,10,11,12]))

    # 3) Scaling benchmarks (caps approximate the C++ defaults)
    publication_qubits = _parse_qubits_env('BENCH_QUBITS', [1,2,5,7,10,11,12,13,14,15,16,17,18,19,20])
    run_scaling_benchmark('vqe', publication_qubits, simulate_cap=15)
    run_scaling_benchmark('qcbm', publication_qubits, simulate_cap=20)
    run_scaling_benchmark('variational_circuit', publication_qubits, simulate_cap=20)
    run_scaling_benchmark('hamiltonian_simulation', publication_qubits, simulate_cap=20)
    run_scaling_benchmark('random_circuit', publication_qubits, simulate_cap=20)
    run_scaling_benchmark('qaoa', publication_qubits, simulate_cap=20)
    run_scaling_benchmark('grover', publication_qubits, simulate_cap=20)
    run_scaling_benchmark('ghz', publication_qubits, simulate_cap=25)
    run_scaling_benchmark('trotter', [2,3,4,5,6,7,8], simulate_cap=8)
    run_scaling_benchmark('time_evolution', publication_qubits, simulate_cap=12)
    run_scaling_benchmark('steady_state', publication_qubits, simulate_cap=8)
    run_scaling_benchmark('qft', publication_qubits, simulate_cap=12)
    run_scaling_benchmark('phase_estimation', publication_qubits, simulate_cap=12)

    # 4) QASM suite
    run_qasm_suite()

    # 5) Measurement distributions (tutorial plots)
    generate_ghz_distributions([4,5,6], shots=int(os.environ.get('GHZ_SHOTS', '10000')))

    success(f"Done in {now_ms()-start:.2f} ms. See bench/ for outputs.")


if __name__ == '__main__':
    main()
