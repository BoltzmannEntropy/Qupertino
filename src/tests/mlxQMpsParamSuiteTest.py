import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math

from mlxq.mlxQdevice import Device
from mlxq.mlxQgates import H, CNOT, RX, RZ
from mlxq.mps_state import MPSOptions
from mlxq.mlxQpretty import info, table


def _close_vec(a, b, tol=1e-6):
    if len(a) != len(b):
        return False
    return all(abs(a[i] - b[i]) <= tol for i in range(len(a)))


def test_mps_param_ghz_parity():
    info("MPS param suite: GHZ parity vs SV (n=3,4)")
    for n in (3, 4):
        # SV
        dev_sv = Device(n, backend='sv')
        ops = [{"name": "H", "wires": [0]}]
        for q in range(n-1):
            ops.append({"name": "CNOT", "wires": [q, q+1]})
        dev_sv.execute(ops)
        p_sv = dev_sv.sim.probabilities()
        # MPS (exact settings)
        dev_mps = Device(n, backend='mps', mps_opts=MPSOptions(dmax=256, eps=1e-12))
        dev_mps.execute(ops)
        p_mps = dev_mps.sim.probabilities()
        delta = max(abs(a-b) for a,b in zip(p_sv, p_mps))
        table("GHZ parity", ("n","max|Δ|"), [(str(n), f"{delta:.2e}")])
        assert _close_vec(p_sv, p_mps, tol=1e-6)


def test_mps_param_random_layer_parity():
    info("MPS param suite: random 1-layer parity vs SV (n=4)")
    n = 4
    # Build a single layer RX/RZ then NN CNOT
    ops = []
    base = 0.37
    for q in range(n):
        ops.append({"name": "RX", "wires": [q], "parameters": [base + 0.1*q]})
        ops.append({"name": "RZ", "wires": [q], "parameters": [-0.2 - 0.05*q]})
    for i in range(n-1):
        ops.append({"name": "CNOT", "wires": [i, i+1]})
    # SV
    dev_sv = Device(n, backend='sv')
    dev_sv.execute(ops)
    p_sv = dev_sv.sim.probabilities()
    # MPS (exact settings)
    dev_mps = Device(n, backend='mps', mps_opts=MPSOptions(dmax=512, eps=1e-12))
    dev_mps.execute(ops)
    p_mps = dev_mps.sim.probabilities()
    delta = max(abs(a-b) for a,b in zip(p_sv, p_mps))
    table("Random1 parity", ("n","max|Δ|"), [(str(n), f"{delta:.2e}")])
    assert _close_vec(p_sv, p_mps, tol=2e-5)


def test_mps_param_sampling_ghz_counts():
    info("MPS param suite: GHZ sampling counts (n=3)")
    n = 3
    dev = Device(n, backend='mps', mps_opts=MPSOptions(dmax=64, eps=1e-12))
    ops = [{"name": "H", "wires": [0]}]
    for q in range(n-1): ops.append({"name": "CNOT", "wires": [q, q+1]})
    dev.execute(ops)
    counts = dev.counts(shots=2000)
    p00 = counts.get('000', 0)
    p11 = counts.get('111', 0)
    # Expect dominant mass on 000 and 111
    assert p00 + p11 > 1500


def test_mps_param_bond_growth_tfim():
    info("MPS param suite: TFIM bond growth (n=10, few steps)")
    n = 10; J = 1.0; h = 0.5; dt = 0.1
    dev = Device(n, backend='mps', mps_opts=MPSOptions(dmax=256, eps=1e-12))
    sim = dev.sim
    from mlxq.mlxQgates import RX as _RX
    from mlxq.bench import _zz_phase_gate as ZZ
    Uzz = ZZ(-dt*J)
    Ux = _RX(2.0*h*dt)
    # a few TEBD steps
    for _ in range(4):
        sim.apply_two_sweep(Uzz)
        sim.apply_single_all(Ux)
    assert sim.bond_max() >= 2

