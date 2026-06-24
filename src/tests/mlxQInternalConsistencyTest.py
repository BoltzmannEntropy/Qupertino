import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import mlx.core as mx

from mlxq.mlxQpretty import info, success, warn, error, table, console
from mlxq.mlxQsim import StateVectorSimulator
from mlxq.mlxQgates import H, X, Y, Z, CNOT
from mlxq.mlxQtensor import kron


def _vec(arr):
    a = mx.array(arr, mx.complex64)
    # normalize
    nrm = mx.sqrt(mx.sum(mx.abs(a) ** 2))
    return a / nrm


def test_consist_apply_matrix_single_qubit():
    info("Internal consistency: applyMatrix (1q H vs dense)")
    simA = StateVectorSimulator(1)
    simB = StateVectorSimulator(1)
    # via built-in
    simA.apply_single(H(), 0)
    # via dense matrix
    Hm = H()
    simB.apply_dense_gate(Hm, [0])
    diff = mx.max(mx.abs(simA.state - simB.state))
    console.print(f"  • max|H| = [bold]{float(diff.item()):.2e}[/bold]")
    assert float(diff.item()) < 1e-6


def test_consist_apply_matrix_two_qubit():
    info("Internal consistency: applyMatrix (2q CNOT vs dense)")
    simA = StateVectorSimulator(2)
    simB = StateVectorSimulator(2)
    simA.cnot(0, 1)
    C = CNOT()
    simB.apply_dense_gate(C, [0, 1])
    diff = mx.max(mx.abs(simA.state - simB.state))
    console.print(f"  • max|CNOT| = [bold]{float(diff.item()):.2e}[/bold]")
    assert float(diff.item()) < 1e-6


def test_consist_apply_swap_dense_vs_builtin_2q():
    info("Internal consistency: dense SWAP vs built-in SWAP (2q)")
    from mlxq.mlxQgates import SWAP as SW
    simA = StateVectorSimulator(2)
    simB = StateVectorSimulator(2)
    # Prepare |01>
    simA.apply_single(X(), 1)
    simB.apply_single(X(), 1)
    # Dense SWAP
    simA.apply_dense_gate(SW(), [0, 1])
    # Built-in via device execute path (prepare same initial state |01>)
    from mlxq.mlxQdevice import Device
    dev = Device(2)
    dev.execute([{ 'name': 'X', 'wires': [1]}])
    dev.execute([{ 'name': 'SWAP', 'wires': [0,1]}])
    # Compare simA.state to dev.sim.state
    diff = mx.max(mx.abs(simA.state - dev.sim.state))
    console.print(f"  • max|SWAP| = [bold]{float(diff.item()):.2e}[/bold]")
    assert float(diff.item()) < 1e-6


def test_consist_apply_cphase_dense_vs_builtin():
    info("Internal consistency: dense CPHASE vs built-in (2q)")
    from mlxq.mlxQgates import CPHASE as CP
    theta = 0.7
    # Dense apply
    simA = StateVectorSimulator(2)
    simA.apply_dense_gate(CP(theta), [0, 1])
    # Built-in via two-qubit op
    simB = StateVectorSimulator(2)
    simB.cphase(0, 1, theta)
    diff = mx.max(mx.abs(simA.state - simB.state))
    console.print(f"  • max|CPHASE(θ)| = [bold]{float(diff.item()):.2e}[/bold]")
    assert float(diff.item()) < 1e-6


def test_consist_apply_crz_dense_vs_builtin():
    info("Internal consistency: dense CRZ vs built-in (2q)")
    from mlxq.mlxQgates import CRZ as CRZgate
    theta = -0.45
    # Dense apply
    simA = StateVectorSimulator(2)
    simA.apply_dense_gate(CRZgate(theta), [0, 1])
    # Built-in via Device path
    from mlxq.mlxQdevice import Device
    dev = Device(2)
    dev.execute([{ 'name': 'CRZ', 'wires': [0,1], 'parameters': [theta]}])
    diff = mx.max(mx.abs(simA.state - dev.sim.state))
    console.print(f"  • max|CRZ(θ)| = [bold]{float(diff.item()):.2e}[/bold]")
    assert float(diff.item()) < 1e-6


def test_consist_apply_sequence_dense_vs_builtin():
    info("Internal consistency: dense sequence (H on q0; CNOT 0->1)")
    from mlxq.mlxQgates import H as Hgate
    simA = StateVectorSimulator(2)
    # Dense: apply H on q0 then CNOT(0,1)
    simA.apply_dense_gate(Hgate(), [0])
    simA.apply_dense_gate(CNOT(), [0,1])
    # Built-in path
    simB = StateVectorSimulator(2)
    simB.h(0)
    simB.cnot(0,1)
    diff = mx.max(mx.abs(simA.state - simB.state))
    console.print(f"  • max|seq(H;CNOT)| = [bold]{float(diff.item()):.2e}[/bold]")
    assert float(diff.item()) < 1e-6


def test_consist_compute_expectation_pauli_string():
    info("Internal consistency: computeExpectation on Pauli strings (Bell)")
    # Build Bell |Φ+> = (|00>+|11>)/√2
    sim = StateVectorSimulator(2)
    sim.apply_single(H(), 0)
    sim.cnot(0, 1)
    # <Z⊗Z> = 1, <X⊗X> = 1, <Y⊗Y> = -1
    Zm = Z(); Xm = mx.array([[0+0j,1+0j],[1+0j,0+0j]], mx.complex64)
    Ym = mx.array([[0+0j,-1j],[1j,0+0j]], mx.complex64)
    rho = mx.outer(sim.state, mx.conj(sim.state))
    def exp(opA, opB):
        O = mx.kron(opA, opB)
        v = mx.matmul(mx.matmul(mx.conj(mx.transpose(sim.state)), O), sim.state)
        return float(mx.real(v).item())
    eZZ = exp(Zm, Zm); eXX = exp(Xm, Xm); eYY = exp(Ym, Ym)
    table("Bell ⟨P⊗P⟩", ("P", "⟨⋅⟩"), [("Z", f"{eZZ:.3f}"),("X", f"{eXX:.3f}"),("Y", f"{eYY:.3f}")])
    assert abs(eZZ - 1.0) < 1e-6 and abs(eXX - 1.0) < 1e-6 and abs(eYY + 1.0) < 1e-6


def test_consist_measure_collapse_project():
    info("Internal consistency: collapse/measure (project on |10⟩)")
    sim = StateVectorSimulator(2)
    # create superposition
    sim.apply_single(H(), 0)
    sim.apply_single(H(), 1)
    # project to |10>
    sim.project_measure([0,1], [1,0])
    probs = sim.probabilities()
    # Only index |10> = 2 should remain
    console.print(f"  • P = [bold]{[round(p,3) for p in probs]}[/bold]")
    assert all((i==2 and abs(p-1.0)<1e-6) or (i!=2 and p<1e-6) for i,p in enumerate(probs))


def test_consist_abs2sum_on_z_basis_mask():
    info("Internal consistency: abs2 sum on Z-basis mask (wires=1->bit=0)")
    sim = StateVectorSimulator(2)
    sim.apply_single(H(), 0)
    sim.apply_single(H(), 1)
    P = sim.probabilities()
    # Sum probabilities where qubit-1 (LSB by our MSB-first convention) == 0 → indices with bit1=0
    n=2
    s = 0.0
    for idx, p in enumerate(P):
        b1 = (idx >> (n-1-1)) & 1
        if b1 == 0:
            s += p
    console.print(f"  • Σ|amp|²[bit1=0] = [bold]{s:.3f}[/bold]")
    assert abs(s-0.5) < 1e-6


def test_consist_set_state_vector_custom():
    info("Internal consistency: setStateVector (custom |01⟩)")
    sim = StateVectorSimulator(2)
    v = _vec([0,1,0,0])
    sim.state = v
    P = sim.probabilities()
    console.print(f"  • P = [bold]{[round(p,3) for p in P]}[/bold]")
    assert abs(P[1]-1.0) < 1e-6


def test_consist_set_state_vector_random_norm():
    info("Internal consistency: setStateVector (random normalized 3q)")
    import random
    rnd = random.Random(7)
    vec = [complex(rnd.random()-0.5, rnd.random()-0.5) for _ in range(8)]
    a = mx.array(vec, mx.complex64)
    nrm = mx.sqrt(mx.sum(mx.abs(a) ** 2)); mx.eval(nrm)
    a = a / float(nrm.item())
    sim = StateVectorSimulator(3)
    sim.state = a
    P = sim.probabilities()
    s = sum(P)
    console.print(f"  • ΣP = [bold]{s:.6f}[/bold]")
    assert abs(s - 1.0) < 1e-6


def test_consist_apply_pauli_xyz_probabilities():
    info("Internal consistency: applyPauli X/Y/Z (probabilities)")
    # Start in |0>
    sim = StateVectorSimulator(1)
    # X|0> = |1>
    sim.apply_single(X(), 0)
    P = sim.probabilities()
    console.print(f"  • after X: P = [bold]{[round(p,3) for p in P]}[/bold]")
    assert abs(P[1]-1.0) < 1e-6
    # Reset and Y|0> = i|1> (global phase), same probabilities
    sim = StateVectorSimulator(1)
    sim.apply_single(Y(), 0)
    P = sim.probabilities()
    console.print(f"  • after Y: P = [bold]{[round(p,3) for p in P]}[/bold]")
    assert abs(P[1]-1.0) < 1e-6
    # Reset and Z|0> = |0>
    sim = StateVectorSimulator(1)
    sim.apply_single(Z(), 0)
    P = sim.probabilities()
    console.print(f"  • after Z: P = [bold]{[round(p,3) for p in P]}[/bold]")
    assert abs(P[0]-1.0) < 1e-6


def test_consist_measure_on_z_basis_sample_counts():
    info("Internal consistency: measureOnZBasis (sampling |+>)")
    sim = StateVectorSimulator(1)
    sim.apply_single(H(), 0)
    counts = sim.sample_counts(2000)
    total = sum(counts.values())
    p0 = counts.get('0', 0) / total
    p1 = counts.get('1', 0) / total
    table("Counts (H|0>, ~0.5 each)", ("bit","count","freq"), [(k, v, f"{v/total:.3f}") for k,v in sorted(counts.items())])
    assert abs(p0 - 0.5) < 0.1 and abs(p1 - 0.5) < 0.1


def test_consist_uniform_sampling_n3():
    info("Internal consistency: uniform sampling on H⊗H⊗H")
    sim = StateVectorSimulator(3)
    for q in range(3):
        sim.apply_single(H(), q)
    counts = sim.sample_counts(4000)
    total = sum(counts.values())
    # All 8 bitstrings near 1/8
    ok = True
    for b in [format(i,'03b') for i in range(8)]:
        p = counts.get(b, 0) / total
        if abs(p - 0.125) > 0.06:
            ok = False
    assert ok


def test_consist_measure_collapse_then_expectation():
    info("Internal consistency: measure collapse then expectation on remaining qubit")
    sim = StateVectorSimulator(2)
    sim.apply_single(H(), 0)
    sim.apply_single(H(), 1)
    # Collapse qubit 0 to |1>
    sim.project_measure([0], [1])
    # Expectation of Z on qubit 1 should be ~0 (still in |+>)
    Zm = Z()
    # Build Z acting on wire 1
    # Using kron(I, Z)
    Im = mx.array([[1+0j,0+0j],[0+0j,1+0j]], mx.complex64)
    O = mx.kron(Im, Zm)
    v = mx.matmul(mx.conj(mx.transpose(sim.state)), mx.matmul(O, sim.state))
    e = float(mx.real(v).item())
    console.print(f"  • ⟨I⊗Z⟩ ≈ {e:.3f}")
    assert abs(e) < 0.2


def test_consist_abs2sum_multiwire_mask():
    info("Internal consistency: abs2 sum multiwire mask (3q uniform)")
    sim = StateVectorSimulator(3)
    sim.apply_single(H(), 0)
    sim.apply_single(H(), 1)
    sim.apply_single(H(), 2)
    P = sim.probabilities()
    n=3
    # mask: bit2=0 and bit0=1 (MSB-first indexing); states: 0b1?0 -> indices 0b100, 0b110 (4 and 6)
    s = 0.0
    for idx, p in enumerate(P):
        b0 = (idx >> (n-1-0)) & 1
        b2 = (idx >> (n-1-2)) & 1
        if b2==0 and b0==1:
            s += p
    console.print(f"  • Σ P[bit2=0 & bit0=1] = [bold]{s:.3f}[/bold] (expected 0.25)")
    assert abs(s - 0.25) < 1e-6


def test_consist_compute_expectation_custom_1q():
    info("Internal consistency: computeExpectation custom 1q (⟨X⟩ on |+>)")
    sim = StateVectorSimulator(1)
    sim.apply_single(H(), 0)
    Xm = mx.array([[0+0j,1+0j],[1+0j,0+0j]], mx.complex64)
    v = mx.matmul(mx.conj(mx.transpose(sim.state)), mx.matmul(Xm, sim.state))
    eX = float(mx.real(v).item())
    console.print(f"  • ⟨X⟩(|+>) = [bold]{eX:.3f}[/bold]")
    assert abs(eX - 1.0) < 1e-6


def test_consist_apply_matrix_noncontiguous_cnot_20():
    info("Internal consistency: applyMatrix non-contiguous wires (CNOT on wires [2,0])")
    # Build a non-trivial 3q state
    simA = StateVectorSimulator(3)
    simA.apply_single(H(), 0)
    simA.apply_single(H(), 1)
    simA.apply_single(H(), 2)
    # Copy baseline
    import copy
    simB = copy.deepcopy(simA)
    # Dense apply CNOT on wires [2,0]
    simA.apply_dense_gate(CNOT(), [2, 0])
    # Built-in two-qubit op on same wires
    simB.cnot(2, 0)
    diff = mx.max(mx.abs(simA.state - simB.state))
    console.print(f"  • max|CNOT(2,0)| = [bold]{float(diff.item()):.2e}[/bold]")
    assert float(diff.item()) < 1e-6


def test_consist_sample_subset_wires_bell_marginals():
    info("Internal consistency: sample() on subset wires (Bell marginals)")
    sim = StateVectorSimulator(2)
    sim.apply_single(H(), 0)
    sim.cnot(0, 1)
    counts0 = sim.sample_counts(1000, wires=[0])
    counts1 = sim.sample_counts(1000, wires=[1])
    # Each marginal should be ~0.5 / 0.5
    for cts, label in [(counts0, 'wire0'), (counts1, 'wire1')]:
        total = sum(cts.values())
        p0 = cts.get('0', 0) / total
        p1 = cts.get('1', 0) / total
        console.print(f"  • {label} marginals: p0={p0:.2f}, p1={p1:.2f}")
        assert abs(p0 - 0.5) < 0.15 and abs(p1 - 0.5) < 0.15


def test_consist_collapse_multi_qubits_and_normalize():
    info("Internal consistency: collapse on two qubits and renormalize")
    sim = StateVectorSimulator(3)
    sim.apply_single(H(), 0)
    sim.apply_single(H(), 1)
    sim.apply_single(H(), 2)
    # Project to |q0 q2> = |1 0>
    sim.project_measure([0, 2], [1, 0])
    P = sim.probabilities()
    s = sum(P)
    console.print(f"  • ΣP after collapse = [bold]{s:.3f}[/bold]")
    assert abs(s - 1.0) < 1e-6
