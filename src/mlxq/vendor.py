"""Vendor and algorithm benchmark catalogs for mlxQ.

This module defines two orthogonal catalogs:
- VENDOR_BENCHMARKS: workloads grouped by upstream framework/vendor
- ALGORITHM_BENCHMARKS: workloads grouped by algorithm subject

These names correspond to keys accepted by run_scaling_benchmark() in mlxq.bench
and the QASM suite via run_qasm_suite().
"""

from __future__ import annotations
from typing import Dict, List

# Canonical bench keys supported by our bench runner
BENCH_KEYS: List[str] = [
    'hamiltonian_simulation', 'time_evolution', 'trotter', 'steady_state',
    'heisenberg', 'heisenberg_xxz', 'heisenberg_random_field',
    'tfim', 'tfim_trotter2', 'tfim_random_field', 'long_range_ising', 'ladder_heisenberg',
    'random_circuit', 'qcbm', 'phase_estimation', 'qft', 'qaoa', 'vqe',
    'variational_circuit', 'grover', 'ghz', 'cuquantum_blueqat',
    # MQTBench additions
    'deutsch_jozsa', 'graph_state', 'qft_entangled', 'phase_estimation_inexact',
    'ae', 'quantum_walk', 'quantum_walk_vchain',
    # Paper-2504/MQTBench additions
    'wstate', 'realamp', 'su2rand', 'qnn', 'qpeexact', 'qpeinexact', 'qftentangled', 'graphstate', 'qwalk', 'random',
]

VENDOR_BENCHMARKS: Dict[str, List[str]] = {
    # Yao.jl: synthetic/random circuits, QFT, evolution
    'yao': ['random_circuit', 'qft', 'time_evolution', 'trotter'],
    # PennyLane: variational workloads and templates
    'pennylane': ['vqe', 'qaoa', 'variational_circuit'],
    # Qulacs: reference shallow/deep circuit families
    'qulacs': ['qft', 'random_circuit', 'grover'],
    # NVIDIA cuQuantum: parity workloads comparable to cuStateVec (and QASM)
    'cuquantum': ['qft', 'phase_estimation', 'hamiltonian_simulation', 'random_circuit', 'cuquantum_blueqat'],
    # QuantumInformation.jl (evolution-focused plus Fourier/synthetic basics)
    'quantuminformation': [
        # Many-body evolution (spin chains, TFIM)
        'heisenberg', 'heisenberg_xxz', 'heisenberg_random_field',
        'tfim', 'tfim_trotter2', 'tfim_random_field', 'long_range_ising',
        # Canonical information-theory circuits
        'ghz', 'grover',
        # Fourier/synthetic references
        'qft', 'random_circuit',
        # Open-system/density-matrix toy
        'steady_state',
    ],
    # MQTBench mapping (subset we support)
    'mqtbench': [
        'ghz', 'wstate', 'qft', 'qft_entangled', 'qftentangled',
        'phase_estimation', 'phase_estimation_inexact', 'qpeexact', 'qpeinexact',
        'random_circuit', 'random', 'qaoa', 'vqe', 'realamp', 'su2rand', 'qnn',
        'graph_state', 'graphstate', 'deutsch_jozsa', 'ae',
        'quantum_walk', 'quantum_walk_vchain', 'qwalk'
    ],
    # QASM bench suite (not a scaling key, but a suite)
    # Use run_qasm_suite() for this group
}

ALGORITHM_BENCHMARKS: Dict[str, List[str]] = {
    'fourier': ['qft', 'phase_estimation'],
    'variational': ['vqe', 'qaoa', 'variational_circuit', 'qcbm'],
    'evolution': [
        'hamiltonian_simulation', 'time_evolution', 'trotter', 'steady_state',
        'heisenberg', 'heisenberg_xxz', 'tfim', 'tfim_trotter2', 'tfim_random_field',
        'heisenberg_random_field', 'long_range_ising', 'ladder_heisenberg'
    ],
    'canonical': ['ghz', 'grover'],
    'synthetic': ['random_circuit'],
}

__all__ = ['BENCH_KEYS', 'VENDOR_BENCHMARKS', 'ALGORITHM_BENCHMARKS']
