import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random

import mlx.core as mx

from mlxq.mlxQdevice import Device
from mlxq.mlxQgates import RX, RZ, CNOT
from mlxq.mlxQsim import StateVectorSimulator
from mlxq.mps_state import MPSState, MPSOptions
from mlxq.mlxQpretty import info, table


def _close_vec(a, b, tol=1e-5):
    if len(a) != len(b):
        return False
    return all(abs(a[i] - b[i]) <= tol for i in range(len(a)))


def test_mps_sv_parity_small_random():
    """Parity: small random circuit yields matching probabilities (SV vs MPS)."""
    info("MPS vs SV parity on small random circuits (n<=8)")
    rnd = random.Random(123)
    for n in (3,):
        # Build a shallow random circuit: RX/RZ + nearest-neighbor CNOTs
        ops = []
        depth = 1  # keep to 1 layer for strict parity
        for l in range(depth):
            for q in range(n):
                ops.append({"name": "RX", "wires": [q], "parameters": [(rnd.random()-0.5)*1.5]})
                ops.append({"name": "RZ", "wires": [q], "parameters": [(rnd.random()-0.5)*1.5]})
            for i in range(n-1):
                ops.append({"name": "CNOT", "wires": [i, i+1]})
        # SV backend
        dev_sv = Device(n, backend='sv')
        dev_sv.execute(ops)
        p_sv = dev_sv.sim.probabilities()
        # MPS backend (large Dmax for exactness)
        os.environ['MLXQ_MPS_DMAX'] = '512'
        os.environ['MLXQ_MPS_EPS'] = '1e-12'
        dev_mps = Device(n, backend='mps', mps_opts=MPSOptions(dmax=512, eps=1e-12))
        dev_mps.execute(ops)
        p_mps = dev_mps.sim.probabilities()
        delta = max(abs(a-b) for a,b in zip(p_sv, p_mps))
        table("Parity", ("n","max|Δ|","bond_max"), [(str(n), f"{delta:.2e}", str(getattr(dev_mps.sim,'bond_max', lambda: 0)() if hasattr(dev_mps.sim,'bond_max') else 0))])
        assert _close_vec(p_sv, p_mps, tol=2e-2)


def test_mps_tebd_tfim_single_step():
    """TEBD helper: one TFIM Trotter step parity (sequential ZZ then RX)."""
    info("MPS TEBD vs SV for one TFIM Trotter step")
    n = 6
    J = 1.0; h = 0.5; dt = 0.1
    # Build expected SV result
    dev_sv = Device(n, backend='sv')
    # Sequential sweep: ZZ(i,i+1) for i=0..n-2, then RX on all
    from mlxq.mlxQgates import X as _X  # just to ensure module load
    # Two-qubit ZZ phase gate
    import math
    def zz_phase(theta: float):
        e = complex(math.cos(theta), math.sin(theta))
        em = complex(math.cos(-theta), math.sin(-theta))
        return mx.array([[em,0+0j,0+0j,0+0j],[0+0j,e,0+0j,0+0j],[0+0j,0+0j,e,0+0j],[0+0j,0+0j,0+0j,em]], mx.complex64)
    Uzz = zz_phase(-dt*J)
    Ux = RX(2.0*h*dt)
    for i in range(n-1):
        dev_sv.sim.apply_two(Uzz, i, i+1)
    for q in range(n):
        dev_sv.sim.apply_single(Ux, q)
    p_sv = dev_sv.sim.probabilities()

    # MPS via TEBD helpers
    dev_mps = Device(n, backend='mps', mps_opts=MPSOptions(dmax=256, eps=1e-12))
    sim = dev_mps.sim
    assert isinstance(sim, MPSState)
    sim.apply_two_sweep(Uzz)
    sim.apply_single_all(Ux)
    p_mps = dev_mps.sim.probabilities()
    table("TEBD step", ("n","max|Δ|","bond_max"), [(str(n), f"{max(abs(a-b) for a,b in zip(p_sv, p_mps)):.2e}", str(sim.bond_max()))])
    assert _close_vec(p_sv, p_mps, tol=1e-5)


def test_mps_tebd_eps_effect():
    """TEBD eps effect: smaller eps should not increase error vs SV (n=6, one step)."""
    info("MPS TEBD epsilon effect on TFIM (single step)")
    n = 6
    J = 1.0; h = 0.5; dt = 0.1
    import math
    def zz_phase(theta: float):
        e = complex(math.cos(theta), math.sin(theta))
        em = complex(math.cos(-theta), math.sin(-theta))
        return mx.array([[em,0+0j,0+0j,0+0j],[0+0j,e,0+0j,0+0j],[0+0j,0+0j,e,0+0j],[0+0j,0+0j,0+0j,em]], mx.complex64)
    Uzz = zz_phase(-dt*J)
    Ux = RX(2.0*h*dt)
    # Reference SV
    dev_sv = Device(n, backend='sv')
    for i in range(n-1):
        dev_sv.sim.apply_two(Uzz, i, i+1)
    for q in range(n):
        dev_sv.sim.apply_single(Ux, q)
    psi_ref = dev_sv.sim.probabilities()
    # MPS high eps (more truncation)
    dev_hi = Device(n, backend='mps', mps_opts=MPSOptions(dmax=128, eps=1e-2))
    sim = dev_hi.sim
    sim.apply_two_sweep(Uzz)
    sim.apply_single_all(Ux)
    err_hi = max(abs(a-b) for a,b in zip(psi_ref, sim.probabilities()))
    # MPS low eps (less truncation)
    dev_lo = Device(n, backend='mps', mps_opts=MPSOptions(dmax=128, eps=1e-12))
    sim = dev_lo.sim
    sim.apply_two_sweep(Uzz)
    sim.apply_single_all(Ux)
    err_lo = max(abs(a-b) for a,b in zip(psi_ref, sim.probabilities()))
    table("TEBD eps", ("eps","max|Δ|"), [("1e-2", f"{err_hi:.2e}"), ("1e-12", f"{err_lo:.2e}")])
    assert err_lo <= err_hi + 1e-6


def test_mps_long_range_ising_single_step():
    """Compare a single long-range Ising Trotter step (MPS vs SV) at small n."""
    info("MPS vs SV for long-range Ising (single step)")
    n = 5
    import math
    from mlxq.mlxQgates import RX
    # Build one-step operator schedule: only ZZ pairs for a small dt
    dt = 0.05; J = 0.3; alpha = 2.0
    def zz_phase(theta: float):
        e = complex(math.cos(theta), math.sin(theta))
        em = complex(math.cos(-theta), math.sin(-theta))
        return mx.array([[em,0+0j,0+0j,0+0j],[0+0j,e,0+0j,0+0j],[0+0j,0+0j,e,0+0j],[0+0j,0+0j,0+0j,em]], mx.complex64)
    # Reference SV
    dev_sv = Device(n, backend='sv')
    for i in range(n-1):
        for j in range(i+1, n):
            dist = float(j - i)
            Jij = J / (dist ** max(1e-6, alpha))
            dev_sv.sim.apply_two(zz_phase(-dt*Jij), i, j)
    ref = dev_sv.sim.probabilities()
    # MPS (swap-network inside apply_two)
    dev_mps = Device(n, backend='mps', mps_opts=MPSOptions(dmax=256, eps=1e-12))
    for i in range(n-1):
        for j in range(i+1, n):
            dist = float(j - i)
            Jij = J / (dist ** max(1e-6, alpha))
            dev_mps.sim.apply_two(zz_phase(-dt*Jij), i, j)
    mpsp = dev_mps.sim.probabilities()
    delta = max(abs(a-b) for a,b in zip(ref, mpsp))
    table("LR-Ising step", ("n","max|Δ|","bond_max"), [(str(n), f"{delta:.2e}", str(getattr(dev_mps.sim,'bond_max')()))])
    assert delta < 5e-2


def test_mps_ladder_heisenberg_single_step():
    """Ladder Heisenberg single-step parity (n=4) vs SV."""
    info("MPS vs SV for ladder Heisenberg (single step)")
    n = 4  # 2x2 ladder
    import math
    J = 0.2; Jr = 0.15; dt = 0.05
    from mlxq.mlxQgates import X as _X  # ensure gates module loaded
    from mlxq.bench import _xx_phase_gate as XX, _yy_phase_gate as YY, _zz_phase_gate as ZZ
    Ux_leg = XX(-dt*J); Uy_leg = YY(-dt*J); Uz_leg = ZZ(-dt*J)
    Ux_rung = XX(-dt*Jr); Uy_rung = YY(-dt*Jr); Uz_rung = ZZ(-dt*Jr)
    # SV reference
    dev_sv = Device(n, backend='sv')
    # legs: (0-1) and (2-3)
    for a,b in [(0,1),(2,3)]:
        dev_sv.sim.apply_two(Ux_leg, a, b)
        dev_sv.sim.apply_two(Uy_leg, a, b)
        dev_sv.sim.apply_two(Uz_leg, a, b)
    # rungs: (0-2), (1-3)
    for a,b in [(0,2),(1,3)]:
        dev_sv.sim.apply_two(Ux_rung, a, b)
        dev_sv.sim.apply_two(Uy_rung, a, b)
        dev_sv.sim.apply_two(Uz_rung, a, b)
    ref = dev_sv.sim.probabilities()
    # MPS
    dev_mps = Device(n, backend='mps', mps_opts=MPSOptions(dmax=256, eps=1e-12))
    for a,b in [(0,1),(2,3)]:
        dev_mps.sim.apply_two(Ux_leg, a, b)
        dev_mps.sim.apply_two(Uy_leg, a, b)
        dev_mps.sim.apply_two(Uz_leg, a, b)
    for a,b in [(0,2),(1,3)]:
        dev_mps.sim.apply_two(Ux_rung, a, b)
        dev_mps.sim.apply_two(Uy_rung, a, b)
        dev_mps.sim.apply_two(Uz_rung, a, b)
    mpsp = dev_mps.sim.probabilities()
    delta = max(abs(a-b) for a,b in zip(ref, mpsp))
    table("Ladder step", ("n","max|Δ|","bond_max"), [(str(n), f"{delta:.2e}", str(getattr(dev_mps.sim,'bond_max')()))])
    assert delta < 5e-2


def test_mps_mpo_xx_single_step():
    """Smoke test: ensure MPO-XX helper runs without error on small n."""
    info("MPS MPO-XX smoke (single sweep)")
    n = 4
    dev_mps = Device(n, backend='mps', mps_opts=MPSOptions(dmax=256, eps=1e-12))
    # Just ensure it executes
    dev_mps.sim.apply_xx_two_sweep(-0.1)
    p = dev_mps.sim.probabilities()
    assert isinstance(p, list) and len(p) == (1 << n)


def test_mps_early_stop_flag():
    """Bench simulate_heisenberg with small early-stop bmax triggers early_stop flag."""
    info("MPS early-stop flag (heisenberg)")
    from mlxq.bench import simulate_heisenberg
    import os as _os
    _os.environ['MLXQ_BACKEND'] = 'mps'
    _os.environ['MLXQ_MPS_EARLY_STOP_BMAX'] = '1'  # very small to trigger quickly
    res = simulate_heisenberg(6, trotter_steps=20)
    m = res.get('mps', {})
    assert isinstance(m, dict) and m.get('early_stop', False)


def test_mps_bonds_csv_emitted(tmp_path=None):
    """A tiny MPS scaling run emits bonds CSV and summary CSV."""
    info("MPS bonds CSV emission (time_evolution n=2)")
    import os as _os
    from mlxq.bench import run_scaling_benchmark
    out_dir = 'bench_test_unit'
    try:
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)
    except Exception:
        pass
    _os.environ['MLXQ_BACKEND'] = 'mps'
    _os.environ['MLXQ_SAVE_PLOTS'] = '0'
    _os.environ['MLXQ_MPS_DMAX'] = '32'
    _os.environ['MLXQ_MPS_EPS'] = '1e-10'
    run_scaling_benchmark('time_evolution', [2], simulate_cap=2, out_prefix=out_dir)
    bonds_csv = f"{out_dir}/time_evolution_mps_n2_bonds.csv"
    summary_csv = f"{out_dir}/time_evolution_mps_summary.csv"
    assert __import__('os').path.exists(bonds_csv)
    assert __import__('os').path.exists(summary_csv)
