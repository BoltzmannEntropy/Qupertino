#!/usr/bin/env python3
"""
Run a small MQTBench-like VQE subset using mlxQ's VQE benchmark.

Outputs under bench/mqtbench/ as vqe_data.csv and vqe_scaling.png
"""
from __future__ import annotations
import os
from pathlib import Path

from mlxq.mlxQbench import run_scaling_benchmark


def main() -> None:
    Path('bench/mqtbench').mkdir(parents=True, exist_ok=True)
    # Default qubit set (align with our VQE default): 1,2,5,7,10,11,12,13,14,15
    qs = [1,2,5,7,10,11,12,13,14,15]
    cap = 15
    try:
        env = os.environ.get('MQTBENCH_VQE_QUBITS')
        if env:
            tmp = []
            for t in env.split(','):
                t = t.strip()
                if t.isdigit():
                    tmp.append(int(t))
            if tmp:
                qs = tmp
    except Exception:
        pass
    try:
        cap = int(os.environ.get('MQTBENCH_VQE_CAP', str(cap)))
    except Exception:
        pass
    print(f"=== MQTBench VQE subset: qubits={qs}, cap={cap} ===")
    run_scaling_benchmark('vqe', qs, simulate_cap=cap, out_prefix='bench/mqtbench')


if __name__ == '__main__':
    main()

