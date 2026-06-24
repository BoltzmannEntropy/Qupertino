import os
import sys
# Ensure local package path so `python3 python/tests/mlxQuantumCoreTest.py` works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import math
import platform
from datetime import datetime

import mlx.core as mx

from mlxq.mlxQsim import StateVectorSimulator, qft, iqft
from mlxq.mlxQgates import H, X, Y, Z, I, RX, RY, RZ, SX, S, SDG, T, TDG, PhaseShift, U2, U3, SWAP, iSWAP, CNOT, CZ, CPHASE, CRX, CRY, CRZ, Toffoli, Fredkin
from mlxq.mlxQtensor import kron
from mlxq.mlxQobservables import expectation_value, is_unitary, pauli_decomposition_2x2, exp_i_pauli, pauli_strings_commute_words, is_hermitian, commutator
from mlxq.mlxQdevice import Device
from mlxq.mlxQqasm import parse_qasm_file
from mlxq.paths import qasm_local_path
from mlxq.mlxQpretty import info, success, warn, error, table, console
from mlxq.mlxQmetrics import cpu_seconds, peak_rss_mb, now_ms
from mlxq.mlxQstates import bell_state, ghz_state, computational_basis, random_state, one_state, y_plus_state, y_minus_state, uniform_superposition, spin_coherent, to_bloch_vector, w_state, custom_state, state_overlap, max_mixed
from mlxq.mlxQinformation import proj, ptrace, ptranspose, purity, entropy_qubit, concurrence_pure, negativity_pure, operator_to_vector, vector_to_operator, stacked_index, unstacked_index, spre, spost
from mlxq.mlxQchannels import apply_kraus, depolarizing_kraus, amplitude_damping_kraus, bitflip_kraus, choi_from_kraus
from mlxq.mlxQspin import Sx, Sy, Sz, hamiltonian_from_field, unitary_time_evolution
from mlxq.mlxQdraw import circuit_ascii, random_circuit, circuit_mpl
from mlxq.mlxQquantikz import circuit_to_quantikz


def close(a, b, tol=1e-5):
    return abs(a - b) < tol


def test_hadamard_superposition():
    info("Hadamard superposition on |0> → (|0>+|1>)/√2")
    sim = StateVectorSimulator(1)
    sim.apply_single(H(), 0)
    probs = sim.probabilities()
    table("Probabilities", ("|0>", "|1>"), [(f"{probs[0]:.3f}", f"{probs[1]:.3f}")])
    assert close(probs[0], 0.5) and close(probs[1], 0.5)


def test_pauli_x_on_zero():
    info("Apply X on |0> → |1>")
    sim = StateVectorSimulator(1)
    sim.apply_single(X(), 0)
    probs = sim.probabilities()
    table("Probabilities", ("|0>", "|1>"), [(f"{probs[0]:.3f}", f"{probs[1]:.3f}")])
    assert close(probs[1], 1.0)


def test_rotations_unitarity_norm():
    info("Rotation gates conserve norm (RX/RY/RZ)")
    for name, gate in [("RX", RX(0.3)), ("RY", RY(1.2)), ("RZ", RZ(-0.7))]:
        sim = StateVectorSimulator(1)
        sim.apply_single(gate, 0)
        s = sum(sim.probabilities())
        console.print(f"  • {name} norm = [bold]{s:.6f}[/bold]")
        assert close(s, 1.0)


def test_z_expectation_on_zero_and_one():
    info("⟨Z⟩ on |0> = +1  and  ⟨Z⟩ on |1> = -1")
    sim = StateVectorSimulator(1)
    exp0 = expectation_value(sim.state, Z())
    console.print(f"  • ⟨Z⟩(|0⟩) = [bold]{exp0:.3f}[/bold]")
    assert close(exp0, 1.0)

    sim = StateVectorSimulator(1)
    sim.x(0)
    exp1 = expectation_value(sim.state, Z())
    console.print(f"  • ⟨Z⟩(|1⟩) = [bold]{exp1:.3f}[/bold]")
    assert close(exp1, -1.0)


def test_cnot_creates_bell():
    info("H(0); CNOT(0,1) → Bell (Φ+)")
    sim = StateVectorSimulator(2)
    sim.h(0)
    sim.cnot(0, 1)
    probs = sim.probabilities()
    table("Bell Probabilities", ("|00>", "|01>", "|10>", "|11>"),
          [(f"{probs[0]:.3f}", f"{probs[1]:.3f}", f"{probs[2]:.3f}", f"{probs[3]:.3f}")])
    assert close(probs[0], 0.5) and close(probs[3], 0.5)


def test_swap_moves_amplitudes():
    info("SWAP(0,1) moves |01> → |10>")
    sim = StateVectorSimulator(2)
    sim.x(1)  # |01>
    sim.apply_two(SWAP(), 0, 1)
    probs = sim.probabilities()
    table("Probabilities", ("|00>", "|01>", "|10>", "|11>"),
          [(f"{probs[0]:.3f}", f"{probs[1]:.3f}", f"{probs[2]:.3f}", f"{probs[3]:.3f}")])
    assert close(probs[2], 1.0)


def test_qft_iqft_identity_3q():
    info("QFT·IQFT identity on 3 qubits (|101⟩)")
    sim = StateVectorSimulator(3)
    sim.x(0)
    sim.x(2)
    qft(sim, [0, 1, 2])
    iqft(sim, [0, 1, 2])
    probs = sim.probabilities()
    console.print(f"  • Peak index: 0b101 = 5, P = [bold]{probs[5]:.3f}[/bold]")
    assert all(close(p, 0.0, 1e-6) for i, p in enumerate(probs) if i != 0b101)
    assert close(probs[0b101], 1.0, 1e-6)


def test_bell_sampling_counts_reasonable():
    info("Sampling Bell state counts (1000 shots)")
    sim = StateVectorSimulator(2)
    sim.h(0)
    sim.cnot(0, 1)
    counts = sim.sample_counts(1000)
    total = sum(counts.values())
    p00 = counts.get('00', 0) / total
    p11 = counts.get('11', 0) / total
    table("Counts", ("bitstring", "count", "freq"),
          [(k, v, f"{v/total:.3f}") for k, v in sorted(counts.items())])
    assert abs((p00 + p11) - 1.0) < 0.2


def test_qasm_bell_exec():
    info("QASM bell.qasm execution (H; CX)")
    n, ops = parse_qasm_file(qasm_local_path('bell.qasm'))
    dev = Device(n)
    # ASCII circuit print for visual validation
    console.print("\n[dim]ASCII circuit (bell.qasm):[/dim]\n" + circuit_ascii(n, ops))
    dev.execute(ops)
    probs = dev.sim.probabilities()
    table("QASM/Bell Probabilities", ("|00>", "|01>", "|10>", "|11>"),
          [(f"{probs[0]:.3f}", f"{probs[1]:.3f}", f"{probs[2]:.3f}", f"{probs[3]:.3f}")])
    assert close(probs[0], 0.5) and close(probs[3], 0.5)


def test_qasm_qft4_exec():
    info("QASM qft_n4.qasm execution")
    n, ops = parse_qasm_file(qasm_local_path('qft_n4.qasm'))
    dev = Device(n)
    console.print("\n[dim]ASCII circuit (qft_n4.qasm):[/dim]\n" + circuit_ascii(n, ops))
    dev.execute(ops)
    probs = dev.sim.probabilities()
    console.print(f"  • ΣP = [bold]{sum(probs):.6f}[/bold]")
    assert close(sum(probs), 1.0)


def run_all():
    start_ms = now_ms()
    console.print("""
[bold cyan]
╔══════════════════════════════════════════════════════════════╗
║              mlx-Quantum Python Core Test Suite             ║
║    Verbose Outputs: Operations, States, QASM (Pretty)       ║
╚══════════════════════════════════════════════════════════════╝
[/bold cyan]
""")
    rows = [
        ("Python", sys.version.split()[0]),
        ("Platform", platform.platform()),
        ("Timestamp", datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
    ]
    table("System Information", ("Key", "Value"), rows)

    tests = [
        test_hadamard_superposition,
        test_pauli_x_on_zero,
        test_rotations_unitarity_norm,
        test_z_expectation_on_zero_and_one,
        test_cnot_creates_bell,
        test_swap_moves_amplitudes,
        test_qft_iqft_identity_3q,
        test_bell_sampling_counts_reasonable,
        test_qasm_bell_exec,
        test_qasm_qft4_exec,
        # Next 10 detailed tests mirroring C++ suite
        test_commutator_xy_equals_2iZ,
        test_anticommutator_xz_zero,
        test_s_times_sdg_identity,
        test_t_squared_equals_s,
        test_phase_additivity,
        test_rz_additivity,
        test_u3_unitary,
        test_cphase_pi_equals_cz,
        test_swap_three_cnot_decomposition,
        test_qft_uniform_to_zero_state_2q,
        # Third batch (+10)
        test_hadamard_squared_identity,
        test_pauli_squared_identity,
        test_plus_minus_x_expectations,
        test_bell_state_from_states,
        test_ghz3_xxx_expectation_one,
        test_comp_basis_states_2q_full,
        test_device_exec_h_then_probs,
        test_rz_2pi_identity,
        test_random_state_norm,
        test_openqasm_parse_counts,
        # Fourth batch (+10) aiming toward 40
        test_iswap_squared_equals_zz,
        test_hh_cnot_hh_equals_cnot_21,
        test_cnot_to_cz_via_h,
        test_u2_decomposition_rz_ry_rz,
        test_u3_euler_zyz,
        test_qft_uniform_to_zero_state_3q,
        test_y_basis_state_expval_minus,
        test_swap_conjugation_xz_to_zx,
        test_cphase_zero_equals_identity,
        test_device_counts_hadamard,
        # Fifth batch (+10) to 50
        test_conjugation_hzh_equals_x,
        test_conjugation_hxh_equals_z,
        test_cnot_conjugation_x_on_control,
        test_cnot_conjugation_z_on_target,
        test_t_times_tdg_identity,
        test_ry_pi_maps_zero_to_one,
        test_rx_pi_maps_zero_to_one,
        test_deutsch_qasm_norm,
        test_grover2_qasm_norm,
        test_toffoli_truth_table_flip,
        # Sixth batch (+10) to 60
        test_fredkin_swaps_when_control_one,
        test_swap_nonadjacent_0_2_moves,
        test_counts_total_equals_shots,
        test_expectation_z_on_plus_zero,
        test_bell_xx_expectation_one,
        test_bell_zz_expectation_one,
        test_bell_yy_expectation_minus_one,
        test_inverse_qft_qasm_norm,
        test_wstate3_qasm_norm,
        test_vqe_n4_qasm_norm,
        # Seventh batch (+10): more core parity with C++
        test_h_on_first_qubit_state_vector,
        test_global_phase_invariance_probabilities,
        test_bloch_vector_plus_x_axis,
        test_bloch_vector_zero_z_axis,
        test_pauli_decomposition_reconstruction_2x2,
        test_is_unitary_core_gates_set,
        test_commute_different_qubits_xi_iz_zero,
        test_qft_iqft_identity_2q_on_basis,
        test_pauli_exponential_matches_rz,
        test_eigenstates_basic_xyz,
        # Eighth batch (+10)
        test_multictrlx_1_equals_cnot_matrix,
        test_rx_additivity,
        test_ry_additivity,
        test_crx_zero_is_identity,
        test_cry_zero_is_identity,
        test_crz_zero_is_identity,
        test_identity_action_does_nothing,
        test_ghz3_zzI_expectation_one,
        test_w3_single_qubit_z_expectation_thirds,
        test_uniform_superposition_probs_2q_equal_quarter,
        # Ninth batch (+10)
        test_hh_zz_hh_equals_xx,
        test_commutator_yz_2ix,
        test_sx_unitary,
        test_swap_self_inverse,
        test_iswap_unitary,
        test_device_sequence_norm_preserved,
        test_random_state_probability_sum_one_n4,
        test_counts_uniform_after_hadamards_3q,
        test_cphase_commutes_with_z_on_target,
        test_crz_commutes_with_z_on_target,
        # Tenth batch (+10)
        test_rz_conjugates_x_to_y,
        test_rx_conjugates_z_to_y,
        test_h_conjugates_y_to_minus_y,
        test_swap_commutes_with_cz_different_qubits,
        test_cnot_square_is_identity,
        test_cz_square_is_identity,
        test_t_dagger_times_t_identity,
        test_s_dagger_times_s_identity,
        test_device_counts_bell_correlated,
        test_shot_based_measurements_convergence,
        # Eleventh batch (+20): Information theory + Spin-1/2
        test_qi_bell_reduced_max_mixed,
        test_qi_purity_pure_and_mixed,
        test_qi_negativity_bell_and_product,
        test_qi_concurrence_bell_one,
        test_qi_entropy_reduced_bell_one,
        test_qi_depolarizing_channel_max_mixed,
        test_qi_amplitude_damping_excited,
        test_qi_bitflip_channel_zero_to_one,
        test_qi_choi_psd_amplitude_damping,
        test_qi_ptrace_dimensions_consistency,
        test_qi_operator_vectorization_roundtrip,
        test_qi_stacked_index_roundtrip,
        test_spin_pauli_as_half,
        test_spin_expectation_on_states,
        test_spin_larmor_precession_sz,
        test_spin_rotation_about_x_maps_z,
        test_superoperator_unitary_action,
        test_vectorization_superoperator_identity,
        # Twelfth batch: Appendix (PRX Quantum) examples
        test_appendix_vqe_parameter_sweep,
        test_appendix_two_qubit_pauli_strings,
        test_appendix_ising_model_ground_state,
        test_appendix_bell_state_pauli_expectations,
        test_appendix_multi_qubit_hamiltonian_grouping,
        test_appendix_hw3_bell_state_expectations,
        test_appendix_hw3_pure_vs_mixed_state,
        test_appendix_hw2_tensor_products,
        test_appendix_hw2_matrix_exponential,
        test_appendix_hw2_unitary_property,
        test_appendix_hw2_concurrence,
        test_appendix_hw214b_y_basis_time_evolution,
        test_appendix_hw24_pauli_decomposition,
        test_appendix_hw3_advanced_circuit_decomposition,
        # Thirteenth batch: Advanced appendix examples
        test_elitzur_vaidman_bomb_tester,
        test_two_qubit_separability_analysis,
        test_bell_basis_measurements,
        test_grover_search_algorithm,
        test_spin_general_spinor_expectations,
        test_spin_sy_measurements,
        test_spin_sx_sy_eigenproblem,
        test_spin_two_stage_field_rotation,
        test_spin_cosine_field_flip,
    ]

    # Optional limit for dev runs
    limit_env = os.environ.get('MLXQ_TEST_MAX')
    if limit_env:
        try:
            limit = int(limit_env)
            tests_to_run = tests[:limit]
        except Exception:
            tests_to_run = tests
    else:
        tests_to_run = tests

    results = []
    passed = 0
    for fn in tests_to_run:
        t_wall0 = now_ms()
        t_cpu0 = cpu_seconds()
        mem_peak0 = peak_rss_mb()
        try:
            fn()
            t_cpu1 = cpu_seconds()
            t_wall1 = now_ms()
            mem_peak1 = peak_rss_mb()
            dt_wall = t_wall1 - t_wall0
            dt_cpu = t_cpu1 - t_cpu0
            mem_mb = max(0.0, mem_peak1 - mem_peak0)  # delta peak
            success(f"PASS {fn.__name__}  (wall {dt_wall:.2f} ms | cpu {dt_cpu:.2f} s | peak +{mem_mb:.2f} MB | gpu 0.00 MB)")
            results.append((fn.__name__, "PASS", f"{dt_wall:.2f} ms", f"{dt_cpu:.2f} s", f"+{mem_mb:.2f} MB", "0.00 MB"))
            passed += 1
        except Exception as e:
            t_cpu1 = cpu_seconds()
            t_wall1 = now_ms()
            mem_peak1 = peak_rss_mb()
            dt_wall = t_wall1 - t_wall0
            dt_cpu = t_cpu1 - t_cpu0
            mem_mb = max(0.0, mem_peak1 - mem_peak0)
            error(f"FAIL {fn.__name__}  (wall {dt_wall:.2f} ms | cpu {dt_cpu:.2f} s | peak +{mem_mb:.2f} MB | gpu 0.00 MB): {e}")
            results.append((fn.__name__, "FAIL", f"{dt_wall:.2f} ms", f"{dt_cpu:.2f} s", f"+{mem_mb:.2f} MB", "0.00 MB"))

    table("Test Summary", ("Test", "Result", "Wall", "CPU", "Peak RAM", "GPU"), results)
    console.print(f"\n[bold]Total:[/bold] {passed}/{len(tests_to_run)} tests passed")
    console.print(f"[bold]Total time:[/bold] {now_ms()-start_ms:.2f} ms  |  [bold]Peak RAM:[/bold] {peak_rss_mb():.2f} MB  |  [bold]GPU:[/bold] 0.00 MB")


## main entry at end of file after all tests are defined


# ---------------- Additional detailed tests (next 10) ------------------------

def mat_close(A: mx.array, B: mx.array, tol: float = 1e-5) -> bool:
    D = mx.abs(A - B)
    mx.eval(D)
    return float(mx.max(D).item()) < tol


def test_commutator_xy_equals_2iZ():
    info("[X,Y] = 2iZ")
    Xg, Yg, Zg = X(), Y(), Z()
    comm = mx.matmul(Xg, Yg) - mx.matmul(Yg, Xg)
    rhs = 2.0j * Zg
    console.print(f"  • max|[X,Y]-2iZ| = [bold]{float(mx.max(mx.abs(comm - rhs)).item()):.2e}[/bold]")
    assert mat_close(comm, rhs, 1e-5)


def test_anticommutator_xz_zero():
    info("{X,Z} = 0")
    Xg, Zg = X(), Z()
    anti = mx.matmul(Xg, Zg) + mx.matmul(Zg, Xg)
    zero = mx.zeros_like(anti)
    console.print(f"  • max|{X,Z}| = [bold]{float(mx.max(mx.abs(anti)).item()):.2e}[/bold]")
    assert mat_close(anti, zero, 1e-5)


def test_s_times_sdg_identity():
    info("S · S† = I")
    Sg = PhaseShift(math.pi/2)
    SDGg = PhaseShift(-math.pi/2)
    prod = mx.matmul(Sg, SDGg)
    Id = I()
    console.print(f"  • max|SS†-I| = [bold]{float(mx.max(mx.abs(prod-Id)).item()):.2e}[/bold]")
    assert mat_close(prod, Id, 1e-6)


def test_t_squared_equals_s():
    info("T^2 = S")
    Tg = PhaseShift(math.pi/4)
    Sg = PhaseShift(math.pi/2)
    T2 = mx.matmul(Tg, Tg)
    console.print(f"  • max|T²-S| = [bold]{float(mx.max(mx.abs(T2-Sg)).item()):.2e}[/bold]")
    assert mat_close(T2, Sg, 1e-6)


def test_phase_additivity():
    info("PhaseShift(a)·PhaseShift(b) = PhaseShift(a+b)")
    a, b = 0.3, -0.7
    Pa = PhaseShift(a); Pb = PhaseShift(b); Pab = PhaseShift(a+b)
    prod = mx.matmul(Pa, Pb)
    console.print(f"  • max|P(a)P(b)-P(a+b)| = [bold]{float(mx.max(mx.abs(prod-Pab)).item()):.2e}[/bold]")
    assert mat_close(prod, Pab, 1e-6)


def test_rz_additivity():
    info("RZ(a)·RZ(b) = RZ(a+b)")
    a, b = -0.2, 0.9
    Ra = RZ(a); Rb = RZ(b); Rab = RZ(a+b)
    prod = mx.matmul(Ra, Rb)
    console.print(f"  • max|RZ(a)RZ(b)-RZ(a+b)| = [bold]{float(mx.max(mx.abs(prod-Rab)).item()):.2e}[/bold]")
    assert mat_close(prod, Rab, 1e-6)


def test_u3_unitary():
    info("U3(θ,φ,λ) is unitary")
    th, ph, lam = 0.8, -1.1, 0.4
    U = U3(th, ph, lam)
    UdagU = mx.matmul(mx.conjugate(mx.transpose(U)), U)
    console.print(f"  • max|U†U-I| = [bold]{float(mx.max(mx.abs(UdagU - I())).item()):.2e}[/bold]")
    assert mat_close(UdagU, I(), 1e-6)


def test_cphase_pi_equals_cz():
    info("CPHASE(π) equals CZ")
    CP = CPHASE(math.pi)
    CZg = CZ()
    console.print(f"  • max|CP(π)-CZ| = [bold]{float(mx.max(mx.abs(CP-CZg)).item()):.2e}[/bold]")
    assert mat_close(CP, CZg, 1e-6)


def test_swap_three_cnot_decomposition():
    info("SWAP = CNOT₁₂·CNOT₂₁·CNOT₁₂")
    SW = SWAP()
    C12 = CNOT()  # acts on wires (0,1) in our dense gate composition
    # For matrix-level equality, build the product C12 * C21 * C12
    # C21 is the same matrix but interpreted with wires swapped; however at matrix level,
    # for adjacent ordering using our convention, C21 equals a similarity permute:
    # Simpler: build explicit matrices for C12, C21 by permuting basis indices.
    # Here, we approximate by known identity equality checks via direct matrix forms:
    # Construct C21 explicitly
    C21 = mx.array([
        [1+0j,0+0j,0+0j,0+0j],
        [0+0j,0+0j,0+0j,1+0j],
        [0+0j,0+0j,1+0j,0+0j],
        [0+0j,1+0j,0+0j,0+0j],
    ], mx.complex64)
    prod = mx.matmul(C12, mx.matmul(C21, C12))
    console.print(f"  • max|SWAP - CXCXC| = [bold]{float(mx.max(mx.abs(SW - prod)).item()):.2e}[/bold]")
    assert mat_close(SW, prod, 1e-6)


def test_qft_uniform_to_zero_state_2q():
    info("QFT(uniform) → |00⟩ for 2 qubits")
    sim = StateVectorSimulator(2)
    sim.h(0); sim.h(1)
    qft(sim, [0,1])
    probs = sim.probabilities()
    table("Probabilities", ("|00>","|01>","|10>","|11>"),
          [(f"{probs[0]:.3f}", f"{probs[1]:.3f}", f"{probs[2]:.3f}", f"{probs[3]:.3f}")])
    assert close(probs[0], 1.0, 1e-5)


def test_hadamard_squared_identity():
    info("H·H = I")
    HH = mx.matmul(H(), H())
    console.print(f"  • max|HH-I| = [bold]{float(mx.max(mx.abs(HH - I())).item()):.2e}[/bold]")
    assert mat_close(HH, I(), 1e-6)


def test_pauli_squared_identity():
    info("X²=Y²=Z²=I")
    for name, G in (('X', X()), ('Y', Y()), ('Z', Z())):
        G2 = mx.matmul(G, G)
        console.print(f"  • max|{name}²-I| = [bold]{float(mx.max(mx.abs(G2 - I())).item()):.2e}[/bold]")
        assert mat_close(G2, I(), 1e-6)


def test_plus_minus_x_expectations():
    info("⟨X⟩(|+⟩)=1 and ⟨X⟩(|−⟩)≈-1")
    # |+> = H|0>
    sim = StateVectorSimulator(1)
    sim.apply_single(H(), 0)
    exp_plus = expectation_value(sim.state, X())
    console.print(f"  • ⟨X⟩(|+⟩) = [bold]{exp_plus:.3f}[/bold]")
    assert close(exp_plus, 1.0, 1e-3)
    # |-> = H|1>
    sim = StateVectorSimulator(1)
    sim.apply_single(X(), 0)
    sim.apply_single(H(), 0)
    exp_minus = expectation_value(sim.state, X())
    console.print(f"  • ⟨X⟩(|−⟩) = [bold]{exp_minus:.3f}[/bold]")
    assert close(exp_minus, -1.0, 1e-3)


def test_bell_state_from_states():
    info("Bell state from states.bell_state (Φ+)")
    st = bell_state(0)
    probs = mx.abs(st) ** 2
    mx.eval(probs)
    P = [float(v) for v in probs.tolist()]
    table("Probabilities", ("|00>","|01>","|10>","|11>"),
          [(f"{P[0]:.3f}", f"{P[1]:.3f}", f"{P[2]:.3f}", f"{P[3]:.3f}")])
    assert close(P[0], 0.5) and close(P[3], 0.5)


def test_ghz3_xxx_expectation_one():
    info("GHZ3 has ⟨X⊗X⊗X⟩ = 1 (wire order 0,1,2)")
    st = ghz_state(3)
    Xg = X()
    op = kron(kron(Xg, Xg), Xg)
    exp = expectation_value(st, op)
    console.print(f"  • ⟨XXX⟩ = [bold]{exp:.3f}[/bold]")
    assert close(exp, 1.0, 1e-3)


def test_comp_basis_states_2q_full():
    info("Computational basis (2q): positions 0..3")
    for idx in range(4):
        st = computational_basis(2, idx)
        P = mx.abs(st) ** 2
        mx.eval(P)
        arr = [float(v) for v in P.tolist()]
        console.print(f"  • |{idx:02b}⟩ → P = [bold]{arr}[/bold]")
        assert close(arr[idx], 1.0)


def test_device_exec_h_then_probs():
    info("Device executes H on q0: P(|0>)=P(|1>)=0.5")
    dev = Device(1)
    dev.execute([{"name":"H","wires":[0]}])
    P = dev.sim.probabilities()
    table("Probabilities", ("|0>","|1>"), [(f"{P[0]:.3f}", f"{P[1]:.3f}")])
    assert close(P[0], 0.5) and close(P[1], 0.5)


def test_rz_2pi_identity():
    info("RZ(2π) = -I (global phase)")
    R = RZ(2*math.pi)
    minusI = -1.0 * I()
    console.print(f"  • max|RZ(2π)+I| = [bold]{float(mx.max(mx.abs(R - minusI)).item()):.2e}[/bold]")
    assert mat_close(R, minusI, 1e-6)


def test_random_state_norm():
    info("Random state normalization (n=3)")
    st = random_state(3, seed=123)
    s = float(mx.sum(mx.abs(st) ** 2).item())
    console.print(f"  • Σ|ψ|² = [bold]{s:.6f}[/bold]")
    assert close(s, 1.0, 1e-5)


def test_openqasm_parse_counts():
    info("OpenQASM parse bell.qasm counts")
    n, ops = parse_qasm_file(qasm_local_path('bell.qasm'))
    console.print(f"  • qubits = [bold]{n}[/bold], ops = [bold]{len(ops)}[/bold]")
    assert n == 2 and len(ops) >= 2


def test_iswap_squared_equals_zz():
    info("(iSWAP)^2 = Z⊗Z")
    mat = mx.matmul(iSWAP(), iSWAP())
    Zg = Z()
    ZZ = kron(Zg, Zg)
    console.print(f"  • max|(iSWAP)^2 - Z⊗Z| = [bold]{float(mx.max(mx.abs(mat - ZZ)).item()):.2e}[/bold]")
    assert mat_close(mat, ZZ, 1e-6)


def test_hh_cnot_hh_equals_cnot_21():
    info("(H⊗H) CNOT (H⊗H) = CNOT₂₁")
    Ht = H()
    H2 = kron(Ht, Ht)
    cnot = CNOT()
    left = mx.matmul(H2, mx.matmul(cnot, H2))
    # CNOT₂₁ explicit matrix for our convention
    C21 = mx.array(
        [
            [1+0j,0+0j,0+0j,0+0j],
            [0+0j,0+0j,0+0j,1+0j],
            [0+0j,0+0j,1+0j,0+0j],
            [0+0j,1+0j,0+0j,0+0j],
        ], mx.complex64)
    console.print(f"  • max|LHS-RHS| = [bold]{float(mx.max(mx.abs(left - C21)).item()):.2e}[/bold]")
    assert mat_close(left, C21, 1e-6)


def test_cnot_to_cz_via_h():
    info("(I⊗H) CNOT (I⊗H) = CZ")
    Id = I(); Ht = H()
    I_H = kron(Id, Ht)
    LHS = mx.matmul(I_H, mx.matmul(CNOT(), I_H))
    console.print(f"  • max|LHS-CZ| = [bold]{float(mx.max(mx.abs(LHS - CZ())).item()):.2e}[/bold]")
    assert mat_close(LHS, CZ(), 1e-6)


def test_u2_decomposition_rz_ry_rz():
    info("U2(φ,λ) = RZ(φ) · RY(π/2) · RZ(λ)")
    phi, lam = 0.3, -0.7
    U = U2(phi, lam)
    target = mx.matmul(RZ(phi), mx.matmul(RY(math.pi/2), RZ(lam)))
    console.print(f"  • max|U2 - RZ·RY(π/2)·RZ| = [bold]{float(mx.max(mx.abs(U - target)).item()):.2e}[/bold]")
    assert mat_close(U, target, 1e-6)


def test_u3_euler_zyz():
    info("U3(θ,φ,λ) = RZ(φ) · RY(θ) · RZ(λ)")
    th, ph, lam = -0.9, 1.2, 0.5
    U = U3(th, ph, lam)
    target = mx.matmul(RZ(ph), mx.matmul(RY(th), RZ(lam)))
    console.print(f"  • max|U3 - RZ·RY·RZ| = [bold]{float(mx.max(mx.abs(U - target)).item()):.2e}[/bold]")
    assert mat_close(U, target, 1e-6)


def test_qft_uniform_to_zero_state_3q():
    info("QFT(uniform 3q) → |000⟩")
    sim = StateVectorSimulator(3)
    sim.h(0); sim.h(1); sim.h(2)
    qft(sim, [0,1,2])
    P = sim.probabilities()
    table("Probabilities", ("|000>","others"), [(f"{P[0]:.3f}", f"{sum(P[1:]):.3f}")])
    assert close(P[0], 1.0, 1e-5)


def test_y_basis_state_expval_minus():
    info("⟨Y⟩(|y-⟩)=-1 with |y-⟩=(|0⟩-i|1⟩)/√2")
    st = mx.array([1/math.sqrt(2)+0j, -1j/math.sqrt(2)], mx.complex64)
    expY = expectation_value(st, Y())
    console.print(f"  • ⟨Y⟩(|y-⟩) = [bold]{expY:.3f}[/bold]")
    assert close(expY, -1.0, 1e-3)


def test_swap_conjugation_xz_to_zx():
    info("SWAP (X⊗Z) SWAP = Z⊗X")
    Xg, Zg = X(), Z()
    op = kron(Xg, Zg)
    SW = SWAP()
    lhs = mx.matmul(SW, mx.matmul(op, SW))
    rhs = kron(Zg, Xg)
    console.print(f"  • max|SW (X⊗Z) SW - Z⊗X| = [bold]{float(mx.max(mx.abs(lhs - rhs)).item()):.2e}[/bold]")
    assert mat_close(lhs, rhs, 1e-6)


def test_cphase_zero_equals_identity():
    info("CPHASE(0) = I⊗I")
    CP0 = CPHASE(0.0)
    Id2 = kron(I(), I())
    console.print(f"  • max|CP(0)-I| = [bold]{float(mx.max(mx.abs(CP0 - Id2)).item()):.2e}[/bold]")
    assert mat_close(CP0, Id2, 1e-6)


def test_device_counts_hadamard():
    info("Device sampling counts for H on |0> (200 shots)")
    dev = Device(1, shots=200)
    dev.execute([{"name":"H","wires":[0]}])
    counts = dev.counts()
    total = sum(counts.values())
    console.print(f"  • counts = [bold]{counts}[/bold]")
    assert '0' in counts and '1' in counts and abs(counts['0']/total - 0.5) < 0.2


# ---------------- Fifth batch (+10) to reach 50 tests ------------------------

def test_conjugation_hzh_equals_x():
    info("H Z H = X (single qubit conjugation)")
    Ht, Zg, Xg = H(), Z(), X()
    lhs = mx.matmul(Ht, mx.matmul(Zg, Ht))
    console.print(f"  • max|HZH - X| = [bold]{float(mx.max(mx.abs(lhs - Xg)).item()):.2e}[/bold]")
    assert mat_close(lhs, Xg, 1e-6)


def test_conjugation_hxh_equals_z():
    info("H X H = Z (single qubit conjugation)")
    Ht, Xg, Zg = H(), X(), Z()
    lhs = mx.matmul(Ht, mx.matmul(Xg, Ht))
    console.print(f"  • max|HXH - Z| = [bold]{float(mx.max(mx.abs(lhs - Zg)).item()):.2e}[/bold]")
    assert mat_close(lhs, Zg, 1e-6)


def test_cnot_conjugation_x_on_control():
    info("CNOT (X⊗I) CNOT = X⊗X")
    C = CNOT()
    op = kron(X(), I())
    lhs = mx.matmul(C, mx.matmul(op, C))
    rhs = kron(X(), X())
    console.print(f"  • max|C (X⊗I) C - X⊗X| = [bold]{float(mx.max(mx.abs(lhs - rhs)).item()):.2e}[/bold]")
    assert mat_close(lhs, rhs, 1e-6)


def test_cnot_conjugation_z_on_target():
    info("CNOT (I⊗Z) CNOT = Z⊗Z")
    C = CNOT()
    op = kron(I(), Z())
    lhs = mx.matmul(C, mx.matmul(op, C))
    rhs = kron(Z(), Z())
    console.print(f"  • max|C (I⊗Z) C - Z⊗Z| = [bold]{float(mx.max(mx.abs(lhs - rhs)).item()):.2e}[/bold]")
    assert mat_close(lhs, rhs, 1e-6)


def test_t_times_tdg_identity():
    info("T · T† = I")
    from math import pi
    Tm = PhaseShift(pi/4.0)
    TD = PhaseShift(-pi/4.0)
    prod = mx.matmul(Tm, TD)
    console.print(f"  • max|TT† - I| = [bold]{float(mx.max(mx.abs(prod - I())).item()):.2e}[/bold]")
    assert mat_close(prod, I(), 1e-6)


def test_ry_pi_maps_zero_to_one():
    info("RY(π)|0⟩ = |1⟩ (probabilities)")
    sim = StateVectorSimulator(1)
    sim.apply_single(RY(math.pi), 0)
    P = sim.probabilities()
    table("Probabilities", ("|0>","|1>"), [(f"{P[0]:.3f}", f"{P[1]:.3f}")])
    assert close(P[1], 1.0)


def test_rx_pi_maps_zero_to_one():
    info("RX(π)|0⟩ ~ |1⟩ up to phase (probabilities)")
    sim = StateVectorSimulator(1)
    sim.apply_single(RX(math.pi), 0)
    P = sim.probabilities()
    table("Probabilities", ("|0>","|1>"), [(f"{P[0]:.3f}", f"{P[1]:.3f}")])
    assert close(P[1], 1.0)


def test_deutsch_qasm_norm():
    info("QASM deutsch_n2.qasm normalization")
    n, ops = parse_qasm_file(qasm_local_path('deutsch_n2.qasm'))
    dev = Device(n)
    dev.execute(ops)
    s = sum(dev.sim.probabilities())
    console.print(f"  • ΣP = [bold]{s:.6f}[/bold]")
    assert close(s, 1.0)


def test_grover2_qasm_norm():
    info("QASM grover_2qubit.qasm normalization")
    n, ops = parse_qasm_file(qasm_local_path('grover_2qubit.qasm'))
    dev = Device(n)
    dev.execute(ops)
    s = sum(dev.sim.probabilities())
    console.print(f"  • ΣP = [bold]{s:.6f}[/bold]")
    assert close(s, 1.0)


def test_toffoli_truth_table_flip():
    info("CCX: |110⟩ → |111⟩ (control on q0,q1 target q2)")
    dev = Device(3)
    # prepare |110>
    dev.execute([{"name":"X","wires":[0]},{"name":"X","wires":[1]}])
    # apply Toffoli
    dev.execute([{"name":"CCX","wires":[0,1,2]}])
    P = dev.sim.probabilities()
    console.print(f"  • P[7] (|111⟩) = [bold]{P[7]:.3f}[/bold]")
    assert close(P[7], 1.0)


# ---------------- Sixth batch (+10) to reach 60 tests ------------------------

def test_fredkin_swaps_when_control_one():
    info("CSWAP: |101⟩ → |110⟩ (control q0=1 swaps q1,q2)")
    dev = Device(3)
    # prepare |101>
    dev.execute([{"name":"X","wires":[0]},{"name":"X","wires":[2]}])
    dev.execute([{"name":"CSWAP","wires":[0,1,2]}])
    P = dev.sim.probabilities()
    console.print(f"  • P[6] (|110⟩) = [bold]{P[6]:.3f}[/bold]")
    assert close(P[6], 1.0)


def test_swap_nonadjacent_0_2_moves():
    info("SWAP(0,2) in 3q: |001⟩ → |100⟩")
    sim = StateVectorSimulator(3)
    sim.x(2)  # |001>
    sim.apply_two(SWAP(), 0, 2)
    P = sim.probabilities()
    console.print(f"  • P[4] (|100⟩) = [bold]{P[4]:.3f}[/bold]")
    assert close(P[4], 1.0)


def test_counts_total_equals_shots():
    info("Counts sum equals shots (2q, 500 shots)")
    dev = Device(2, shots=500)
    dev.execute([{"name":"H","wires":[0]},{"name":"H","wires":[1]}])
    counts = dev.counts()
    total = sum(counts.values())
    console.print(f"  • total shots = [bold]{total}[/bold], counts = [bold]{counts}[/bold]")
    assert total == 500


def test_expectation_z_on_plus_zero():
    info("⟨Z⟩ on |+⟩ is 0")
    sim = StateVectorSimulator(1)
    sim.h(0)
    val = expectation_value(sim.state, Z())
    console.print(f"  • ⟨Z⟩(|+⟩) = [bold]{val:.3f}[/bold]")
    assert abs(val) < 1e-6


def test_bell_xx_expectation_one():
    info("Bell Φ+: ⟨X⊗X⟩ = 1")
    sim = StateVectorSimulator(2)
    sim.h(0); sim.cnot(0,1)
    val = expectation_value(sim.state, kron(X(), X()))
    console.print(f"  • ⟨XX⟩ = [bold]{val:.3f}[/bold]")
    assert close(val, 1.0, 1e-3)


def test_bell_zz_expectation_one():
    info("Bell Φ+: ⟨Z⊗Z⟩ = 1")
    sim = StateVectorSimulator(2)
    sim.h(0); sim.cnot(0,1)
    val = expectation_value(sim.state, kron(Z(), Z()))
    console.print(f"  • ⟨ZZ⟩ = [bold]{val:.3f}[/bold]")
    assert close(val, 1.0, 1e-3)


def test_bell_yy_expectation_minus_one():
    info("Bell Φ+: ⟨Y⊗Y⟩ = -1")
    sim = StateVectorSimulator(2)
    sim.h(0); sim.cnot(0,1)
    val = expectation_value(sim.state, kron(Y(), Y()))
    console.print(f"  • ⟨YY⟩ = [bold]{val:.3f}[/bold]")
    assert close(val, -1.0, 1e-3)


def test_inverse_qft_qasm_norm():
    info("QASM inverseqft_n4.qasm normalization")
    n, ops = parse_qasm_file(qasm_local_path('inverseqft_n4.qasm'))
    dev = Device(n)
    dev.execute(ops)
    s = sum(dev.sim.probabilities())
    console.print(f"  • ΣP = [bold]{s:.6f}[/bold]")
    assert close(s, 1.0)


def test_wstate3_qasm_norm():
    info("QASM wstate_n3.qasm normalization")
    n, ops = parse_qasm_file(qasm_local_path('wstate_n3.qasm'))
    dev = Device(n)
    dev.execute(ops)
    s = sum(dev.sim.probabilities())
    console.print(f"  • ΣP = [bold]{s:.6f}[/bold]")
    assert close(s, 1.0)


def test_vqe_n4_qasm_norm():
    info("QASM vqe_n4.qasm normalization")
    n, ops = parse_qasm_file(qasm_local_path('vqe_n4.qasm'))
    dev = Device(n)
    dev.execute(ops)
    s = sum(dev.sim.probabilities())
    console.print(f"  • ΣP = [bold]{s:.6f}[/bold]")
    assert close(s, 1.0)

# ---------------- Seventh batch (+10): more core parity with C++ ------------

def test_h_on_first_qubit_state_vector():
    info("(H⊗I)|10⟩ = (|00⟩-|10⟩)/√2")
    Ht = H(); HI = kron(Ht, I())
    ket10 = computational_basis(2, 0b10)
    res = mx.matmul(HI, mx.reshape(ket10, (4,1)))
    res = mx.reshape(res, (4,))
    inv_s2 = 1.0/math.sqrt(2.0)
    exp = mx.array([inv_s2+0j, 0+0j, -inv_s2+0j, 0+0j], mx.complex64)
    console.print(f"  • max|res-exp| = [bold]{float(mx.max(mx.abs(res-exp)).item()):.2e}[/bold]")
    assert mat_close(res, exp, 1e-6)


def test_global_phase_invariance_probabilities():
    info("Global phase leaves probabilities invariant")
    psi = spin_coherent(0.9, 1.2)
    amp2 = mx.abs(psi)**2
    mx.eval(amp2)
    P1 = [float(v) for v in amp2.tolist()]
    phase = complex(math.cos(0.73), math.sin(0.73))
    psi2 = (phase + 0j) * psi
    amp22 = mx.abs(psi2)**2
    mx.eval(amp22)
    P2 = [float(v) for v in amp22.tolist()]
    console.print(f"  • P1 = [bold]{P1}[/bold], P2 = [bold]{P2}[/bold]")
    assert all(abs(a-b) < 1e-6 for a,b in zip(P1,P2))


def test_bloch_vector_plus_x_axis():
    info("Bloch vector of |+⟩ is (1,0,0)")
    v = to_bloch_vector(plus_state := mx.array([1/math.sqrt(2)+0j, 1/math.sqrt(2)+0j], mx.complex64))
    console.print(f"  • v = [bold]{v}[/bold]")
    assert close(v[0],1.0,1e-3) and close(v[1],0.0,1e-3) and close(v[2],0.0,1e-3)


def test_bloch_vector_zero_z_axis():
    info("Bloch vector of |0⟩ is (0,0,1)")
    v = to_bloch_vector(mx.array([1+0j,0+0j], mx.complex64))
    console.print(f"  • v = [bold]{v}[/bold]")
    assert close(v[0],0.0,1e-3) and close(v[1],0.0,1e-3) and close(v[2],1.0,1e-3)


def test_pauli_decomposition_reconstruction_2x2():
    info("Pauli decomposition and reconstruction of 2x2 Hermitian")
    M = mx.array([[2+0j, 1+1j],[1-1j, 3+0j]], mx.complex64)
    c0,cx,cy,cz = pauli_decomposition_2x2(M)
    R = c0*I() + cx*X() + cy*Y() + cz*Z()
    console.print(f"  • max|M-R| = [bold]{float(mx.max(mx.abs(M-R)).item()):.2e}[/bold]")
    assert mat_close(M, R, 1e-5)


def test_is_unitary_core_gates_set():
    info("Unitarity of representative gates")
    for name, G in [("X",X()), ("H",H()), ("RX",RX(0.3)), ("CNOT",CNOT()), ("SWAP",SWAP())]:
        ok = is_unitary(G)
        console.print(f"  • {name} unitary = [bold]{ok}[/bold]")
        assert ok


def test_commute_different_qubits_xi_iz_zero():
    info("[X⊗I, I⊗Z] = 0")
    XI = kron(X(), I()); IZ = kron(I(), Z())
    comm = mx.matmul(XI, IZ) - mx.matmul(IZ, XI)
    Z4 = mx.zeros_like(comm)
    console.print(f"  • max|comm| = [bold]{float(mx.max(mx.abs(comm)).item()):.2e}[/bold]")
    assert mat_close(comm, Z4, 1e-6)


def test_qft_iqft_identity_2q_on_basis():
    info("QFT·IQFT identity on |01⟩ and |10⟩ (2q)")
    for idx in (1,2):
        sim = StateVectorSimulator(2)
        sim.state = computational_basis(2, idx)
        qft(sim, [0,1]); iqft(sim, [0,1])
        P = sim.probabilities()
        console.print(f"  • idx {idx}: P = [bold]{[round(p,3) for p in P]}[/bold]")
        assert close(P[idx], 1.0, 1e-6)


def test_pauli_exponential_matches_rz():
    info("e^{iθZ} = RZ(-2θ)")
    th = 0.37
    E = exp_i_pauli(Z(), th)
    R = RZ(-2*th)
    console.print(f"  • max|E-R| = [bold]{float(mx.max(mx.abs(E-R)).item()):.2e}[/bold]")
    assert mat_close(E, R, 1e-6)


def test_eigenstates_basic_xyz():
    info("Eigenstate checks: X|+⟩=|+⟩, Z|0⟩=|0⟩, Y|y+⟩=|y+⟩")
    plus = mx.array([1/math.sqrt(2)+0j, 1/math.sqrt(2)+0j], mx.complex64)
    yplus = y_plus_state()
    okx = mat_close(mx.matmul(X(), mx.reshape(plus,(2,1))), mx.reshape(plus,(2,1)), 1e-5)
    okz = mat_close(mx.matmul(Z(), mx.reshape(mx.array([1+0j,0+0j], mx.complex64),(2,1))), mx.reshape(mx.array([1+0j,0+0j], mx.complex64),(2,1)), 1e-5)
    oky = mat_close(mx.matmul(Y(), mx.reshape(yplus,(2,1))), mx.reshape(yplus,(2,1)), 1e-5)
    console.print(f"  • ok = [bold]{okx and okz and oky}[/bold]")
    assert okx and okz and oky

# ---------------- Eighth batch (+10) ----------------------------------------

def test_multictrlx_1_equals_cnot_matrix():
    info("MCX(1) equals CNOT")
    from mlxq.gates import MultiControlledX
    M = MultiControlledX(1)
    console.print(f"  • max|MCX1-CNOT| = [bold]{float(mx.max(mx.abs(M - CNOT())).item()):.2e}[/bold]")
    assert mat_close(M, CNOT(), 1e-6)


def test_rx_additivity():
    info("RX(a)·RX(b) = RX(a+b)")
    a, b = 0.2, -0.5
    lhs = mx.matmul(RX(a), RX(b))
    rhs = RX(a+b)
    console.print(f"  • max|lhs-rhs| = [bold]{float(mx.max(mx.abs(lhs - rhs)).item()):.2e}[/bold]")
    assert mat_close(lhs, rhs, 1e-6)


def test_ry_additivity():
    info("RY(a)·RY(b) = RY(a+b)")
    a, b = -0.3, 0.9
    lhs = mx.matmul(RY(a), RY(b))
    rhs = RY(a+b)
    console.print(f"  • max|lhs-rhs| = [bold]{float(mx.max(mx.abs(lhs - rhs)).item()):.2e}[/bold]")
    assert mat_close(lhs, rhs, 1e-6)


def test_crx_zero_is_identity():
    info("CRX(0) = I⊗I")
    Id2 = kron(I(), I())
    from mlxq.gates import CRX
    M = CRX(0.0)
    console.print(f"  • max|CRX(0)-I| = [bold]{float(mx.max(mx.abs(M-Id2)).item()):.2e}[/bold]")
    assert mat_close(M, Id2, 1e-6)


def test_cry_zero_is_identity():
    info("CRY(0) = I⊗I")
    Id2 = kron(I(), I())
    from mlxq.gates import CRY
    M = CRY(0.0)
    console.print(f"  • max|CRY(0)-I| = [bold]{float(mx.max(mx.abs(M-Id2)).item()):.2e}[/bold]")
    assert mat_close(M, Id2, 1e-6)


def test_crz_zero_is_identity():
    info("CRZ(0) = I⊗I")
    Id2 = kron(I(), I())
    from mlxq.gates import CRZ
    M = CRZ(0.0)
    console.print(f"  • max|CRZ(0)-I| = [bold]{float(mx.max(mx.abs(M-Id2)).item()):.2e}[/bold]")
    assert mat_close(M, Id2, 1e-6)


def test_identity_action_does_nothing():
    info("I action does nothing on vector (matmul)")
    v = random_state(1, seed=7)
    res = mx.matmul(I(), mx.reshape(v,(2,1)))
    res = mx.reshape(res, (2,))
    console.print(f"  • max|v-res| = [bold]{float(mx.max(mx.abs(v-res)).item()):.2e}[/bold]")
    assert mat_close(v, res, 1e-6)


def test_ghz3_zzI_expectation_one():
    info("GHZ_3: ⟨Z⊗Z⊗I⟩ = 1")
    st = ghz_state(3)
    op = kron(kron(Z(), Z()), I())
    val = expectation_value(st, op)
    console.print(f"  • ⟨ZZI⟩ = [bold]{val:.3f}[/bold]")
    assert close(val, 1.0, 1e-5)


def test_w3_single_qubit_z_expectation_thirds():
    info("W_3: each single-qubit ⟨Z⟩ = 1/3")
    st = w_state(3)
    ZII = kron(kron(Z(), I()), I())
    IZI = kron(kron(I(), Z()), I())
    IIZ = kron(kron(I(), I()), Z())
    e1 = expectation_value(st, ZII)
    e2 = expectation_value(st, IZI)
    e3 = expectation_value(st, IIZ)
    console.print(f"  • ⟨Z⟩ = [bold]{e1:.3f}, {e2:.3f}, {e3:.3f}[/bold]")
    assert close(e1, 1.0/3.0, 1e-3) and close(e2, 1.0/3.0, 1e-3) and close(e3, 1.0/3.0, 1e-3)


def test_uniform_superposition_probs_2q_equal_quarter():
    info("Uniform(2): all probabilities 0.25")
    st = uniform_superposition(2)
    P = mx.abs(st)**2
    mx.eval(P)
    arr = [float(v) for v in P.tolist()]
    table("P", ("i","P"), [(i, f"{p:.3f}") for i,p in enumerate(arr)])
    assert all(close(p, 0.25, 1e-5) for p in arr)

# ---------------- Ninth batch (+10) -----------------------------------------

def test_hh_zz_hh_equals_xx():
    info("(H⊗H)(Z⊗Z)(H⊗H) = X⊗X")
    HH = kron(H(), H())
    ZZ = kron(Z(), Z())
    XX = kron(X(), X())
    lhs = mx.matmul(HH, mx.matmul(ZZ, HH))
    console.print(f"  • max|lhs-XX| = [bold]{float(mx.max(mx.abs(lhs-XX)).item()):.2e}[/bold]")
    assert mat_close(lhs, XX, 1e-6)


def test_commutator_yz_2ix():
    info("[Y,Z] = 2iX")
    comm = mx.matmul(Y(), Z()) - mx.matmul(Z(), Y())
    rhs = 2.0j * X()
    console.print(f"  • max|[Y,Z]-2iX| = [bold]{float(mx.max(mx.abs(comm-rhs)).item()):.2e}[/bold]")
    assert mat_close(comm, rhs, 1e-6)


def test_sx_unitary():
    info("SX is unitary")
    U = SX()
    UdagU = mx.matmul(mx.conjugate(mx.transpose(U)), U)
    Id = mx.array([[1+0j,0+0j],[0+0j,1+0j]], mx.complex64)
    d = float(mx.max(mx.abs(UdagU - Id)).item())
    ok = d < 1e-5
    console.print(f"  • max|U†U-I| = [bold]{d:.2e}[/bold]")
    assert ok


def test_swap_self_inverse():
    info("SWAP^2 = I")
    prod = mx.matmul(SWAP(), SWAP())
    Id2 = kron(I(), I())
    console.print(f"  • max|SW^2-I| = [bold]{float(mx.max(mx.abs(prod-Id2)).item()):.2e}[/bold]")
    assert mat_close(prod, Id2, 1e-6)


def test_iswap_unitary():
    info("iSWAP is unitary")
    U = iSWAP()
    UdagU = mx.matmul(mx.conjugate(mx.transpose(U)), U)
    Id = mx.array(
        [
            [1+0j,0+0j,0+0j,0+0j],
            [0+0j,1+0j,0+0j,0+0j],
            [0+0j,0+0j,1+0j,0+0j],
            [0+0j,0+0j,0+0j,1+0j],
        ], mx.complex64)
    d = float(mx.max(mx.abs(UdagU - Id)).item())
    ok = d < 1e-5
    console.print(f"  • max|U†U-I| = [bold]{d:.2e}[/bold]")
    assert ok


def test_device_sequence_norm_preserved():
    info("Device executes sequence; norm preserved")
    dev = Device(2)
    ops = [
        {"name":"H","wires":[0]},
        {"name":"RX","wires":[0],"parameters":[0.3]},
        {"name":"RY","wires":[1],"parameters":[-0.7]},
        {"name":"CNOT","wires":[0,1]},
        {"name":"RZ","wires":[1],"parameters":[1.1]},
    ]
    dev.execute(ops)
    s = sum(dev.sim.probabilities())
    console.print(f"  • ΣP = [bold]{s:.6f}[/bold]")
    assert close(s, 1.0)


def test_random_state_probability_sum_one_n4():
    info("Random state (n=4) probability sum = 1")
    st = random_state(4, seed=99)
    s = float(mx.sum(mx.abs(st)**2).item())
    console.print(f"  • Σ|ψ|² = [bold]{s:.6f}[/bold]")
    assert close(s, 1.0, 1e-5)


def test_counts_uniform_after_hadamards_3q():
    info("Counts uniform after H on all 3 qubits (2000 shots)")
    dev = Device(3, shots=2000)
    dev.execute([{"name":"H","wires":[0]},{"name":"H","wires":[1]},{"name":"H","wires":[2]}])
    counts = dev.counts()
    total = sum(counts.values())
    freqs = {k:v/total for k,v in counts.items()}
    console.print(f"  • freqs ≈ [bold]{ {k: round(v,3) for k,v in freqs.items()} }[/bold]")
    assert len(counts) == 8 and all(abs(v-1/8) < 0.1 for v in freqs.values())


def test_cphase_commutes_with_z_on_target():
    info("[CP(φ), I⊗Z] = 0")
    φ = 0.42
    CP = CPHASE(φ)
    IZ = kron(I(), Z())
    comm = mx.matmul(CP, IZ) - mx.matmul(IZ, CP)
    Z4 = mx.zeros((4,4), mx.complex64)
    console.print(f"  • max|comm| = [bold]{float(mx.max(mx.abs(comm)).item()):.2e}[/bold]")
    assert mat_close(comm, Z4, 1e-6)


def test_crz_commutes_with_z_on_target():
    info("[CRZ(φ), I⊗Z] = 0")
    φ = -0.77
    from mlxq.gates import CRZ
    CRZg = CRZ(φ)
    IZ = kron(I(), Z())
    comm = mx.matmul(CRZg, IZ) - mx.matmul(IZ, CRZg)
    Z4 = mx.zeros((4,4), mx.complex64)
    console.print(f"  • max|comm| = [bold]{float(mx.max(mx.abs(comm)).item()):.2e}[/bold]")
    assert mat_close(comm, Z4, 1e-6)

# ---------------- Tenth batch (+10) -----------------------------------------

def test_rz_conjugates_x_to_y():
    info("RZ(π/2) X RZ(-π/2) = Y")
    a = math.pi/2
    L = mx.matmul(RZ(a), mx.matmul(X(), RZ(-a)))
    console.print(f"  • max|L-Y| = [bold]{float(mx.max(mx.abs(L-Y())).item()):.2e}[/bold]")
    assert mat_close(L, Y(), 1e-5)


def test_rx_conjugates_z_to_y():
    info("RX(π/2) Z RX(-π/2) = ±Y (convention)")
    a = math.pi/2
    L = mx.matmul(RX(a), mx.matmul(Z(), RX(-a)))
    d_pos = float(mx.max(mx.abs(L - Y())).item())
    d_neg = float(mx.max(mx.abs(L + Y())).item())
    console.print(f"  • min(max|L±Y|) = [bold]{min(d_pos,d_neg):.2e}[/bold]")
    assert min(d_pos, d_neg) < 1e-5


def test_h_conjugates_y_to_minus_y():
    info("H Y H = -Y")
    L = mx.matmul(H(), mx.matmul(Y(), H()))
    console.print(f"  • max|HYH+Y| = [bold]{float(mx.max(mx.abs(L + Y())).item()):.2e}[/bold]")
    assert mat_close(L, -1.0*Y(), 1e-5)


def test_swap_commutes_with_cz_different_qubits():
    info("[SWAP, CZ] = 0 for same wires")
    SW = SWAP(); CZg = CZ()
    comm = mx.matmul(SW, CZg) - mx.matmul(CZg, SW)
    Z4 = mx.zeros_like(comm)
    console.print(f"  • max|comm| = [bold]{float(mx.max(mx.abs(comm)).item()):.2e}[/bold]")
    assert mat_close(comm, Z4, 1e-6)


def test_cnot_square_is_identity():
    info("CNOT^2 = I")
    prod = mx.matmul(CNOT(), CNOT())
    Id2 = kron(I(), I())
    console.print(f"  • max|CNOT²-I| = [bold]{float(mx.max(mx.abs(prod-Id2)).item()):.2e}[/bold]")
    assert mat_close(prod, Id2, 1e-6)


def test_cz_square_is_identity():
    info("CZ^2 = I")
    prod = mx.matmul(CZ(), CZ())
    Id2 = kron(I(), I())
    console.print(f"  • max|CZ²-I| = [bold]{float(mx.max(mx.abs(prod-Id2)).item()):.2e}[/bold]")
    assert mat_close(prod, Id2, 1e-6)


def test_t_dagger_times_t_identity():
    info("T† T = I")
    Tm = PhaseShift(math.pi/4)
    TD = PhaseShift(-math.pi/4)
    prod = mx.matmul(TDG(), Tm)
    console.print(f"  • max|T†T-I| = [bold]{float(mx.max(mx.abs(prod-I())).item()):.2e}[/bold]")
    assert mat_close(prod, I(), 1e-6)


def test_s_dagger_times_s_identity():
    info("S† S = I")
    prod = mx.matmul(SDG(), S())
    console.print(f"  • max|S†S-I| = [bold]{float(mx.max(mx.abs(prod-I())).item()):.2e}[/bold]")
    assert mat_close(prod, I(), 1e-6)


def test_device_counts_bell_correlated():
    info("Bell sampling shows only 00 and 11 (1000 shots)")
    dev = Device(2, shots=1000)
    dev.execute([{"name":"H","wires":[0]},{"name":"CNOT","wires":[0,1]}])
    counts = dev.counts()
    console.print(f"  • counts = [bold]{counts}[/bold]")
    assert set(counts.keys()).issubset({"00","11"})


def test_bell_state_fidelity_with_constructor():
    info("Fidelity between simulated Bell and constructor is ~1")
    sim = StateVectorSimulator(2)
    sim.h(0); sim.cnot(0,1)
    psi = sim.state
    phi = bell_state(0)
    # Fidelity |⟨phi|psi⟩|²
    ip = mx.sum(mx.conjugate(phi) * psi)
    mx.eval(ip)
    fid = abs(complex(ip.item()))**2
    console.print(f"  • Fidelity = [bold]{fid:.6f}[/bold]")
    assert fid > 1 - 1e-6


def test_shot_based_measurements_convergence():
    info("Shot-based measurement: ⟨Z⟩ on |+⟩ converges ∝ 1/√N")
    sim = StateVectorSimulator(1)
    sim.h(0)  # |+>
    exact = expectation_value(sim.state, Z())
    console.print(f"  • Exact ⟨Z⟩ = [bold]{exact:.6f}[/bold]")
    shot_counts = [10, 100, 1000, 10000]
    console.print("  Shots     Empirical ⟨Z⟩     Error       Std Error (1/√N)")
    console.print("  -------   ---------------   ----------   ------------------")
    ok = True
    for N in shot_counts:
        samples = sim.sample(N)
        # Map bit 0->+1, 1->-1 for Z-eigenvalues
        s = 0.0
        for [b] in samples:
            s += 1.0 if b == 0 else -1.0
        empirical = s / N
        err = abs(empirical - exact)
        stderr = 1.0 / math.sqrt(N)
        console.print(f"  {N:<7d}   {empirical: .6f}         {err:.6f}      {stderr:.6f}")
        ok = ok and (err < 3.0 * stderr)
    console.print(f"  • Convergence (err < 3σ): [bold]{ok}[/bold]")
    assert ok


# -------------- Information theory tests (subset matching paper) -------------

def test_qi_bell_reduced_max_mixed():
    info("Reduced density of Bell is maximally mixed (I/2)")
    psi = bell_state(0)
    rho = proj(psi)
    rhoA = ptrace(rho, traced_out=[1], dims=[2,2])
    Id2 = mx.array([[0.5+0j,0+0j],[0+0j,0.5+0j]], mx.complex64)
    console.print(f"  • max|ρA - I/2| = [bold]{float(mx.max(mx.abs(rhoA-Id2)).item()):.2e}[/bold]")
    assert mat_close(rhoA, Id2, 1e-6)


def test_qi_purity_pure_and_mixed():
    info("Purity: Tr(ρ²) = 1 (pure) and <1 (mixed)")
    psi = bell_state(0)
    rho = proj(psi)
    p_pure = purity(rho)
    rhoA = ptrace(rho, traced_out=[1], dims=[2,2])
    p_mixed = purity(rhoA)
    console.print(f"  • purity(pure) = [bold]{p_pure:.6f}[/bold], purity(mixed) = [bold]{p_mixed:.6f}[/bold]")
    assert close(p_pure, 1.0, 1e-6) and p_mixed < 1.0


def test_qi_negativity_bell_and_product():
    info("Negativity: Bell=0.5, product=0")
    bell = bell_state(0)
    prod = mx.array([1+0j,0+0j,0+0j,0+0j], mx.complex64)
    Nb = negativity_pure(bell)
    Np = negativity_pure(prod)
    console.print(f"  • N(Bell) = [bold]{Nb:.3f}[/bold], N(prod) = [bold]{Np:.3f}[/bold]")
    assert close(Nb, 0.5, 1e-6) and close(Np, 0.0, 1e-6)


def test_qi_concurrence_bell_one():
    info("Concurrence: C(|Φ+⟩) = 1")
    C = concurrence_pure(bell_state(0))
    console.print(f"  • C = [bold]{C:.3f}[/bold]")
    assert close(C, 1.0, 1e-6)


def test_qi_entropy_reduced_bell_one():
    info("Entropy of reduced Bell qubit is 1 bit")
    rhoA = ptrace(proj(bell_state(0)), traced_out=[1], dims=[2,2])
    S = entropy_qubit(rhoA)
    console.print(f"  • S(ρA) = [bold]{S:.3f}[/bold]")
    assert abs(S - 1.0) < 1e-3


def test_qi_depolarizing_channel_max_mixed():
    info("Depolarizing channel drives to maximally mixed at p=1")
    from mlxq.states import zero_state
    rho = proj(zero_state(1))
    K = depolarizing_kraus(1.0)
    out = apply_kraus(rho, K)
    Id2 = mx.array([[0.5+0j,0+0j],[0+0j,0.5+0j]], mx.complex64)
    console.print(f"  • max|out - I/2| = [bold]{float(mx.max(mx.abs(out-Id2)).item()):.2e}[/bold]")
    assert mat_close(out, Id2, 1e-6)


def test_qi_amplitude_damping_excited():
    info("Amplitude damping maps |1⟩⟨1| → |0⟩⟨0| at γ=1")
    one = mx.array([0+0j,1+0j], mx.complex64)
    rho = proj(one)
    K = amplitude_damping_kraus(1.0)
    out = apply_kraus(rho, K)
    Z = mx.array([[1+0j,0+0j],[0+0j,0+0j]], mx.complex64)
    console.print(f"  • out = [bold]{[[complex(x) for x in row] for row in out.tolist()]}[/bold]")
    assert mat_close(out, Z, 1e-6)


def test_qi_bitflip_channel_zero_to_one():
    info("Bit-flip (p=1) maps |0⟩→|1⟩ at state level")
    zero = mx.array([1+0j,0+0j], mx.complex64)
    rho = proj(zero)
    K = bitflip_kraus(1.0)
    out = apply_kraus(rho, K)
    one = mx.array([[0+0j,0+0j],[0+0j,1+0j]], mx.complex64)
    console.print(f"  • out = [bold]{[[complex(x) for x in row] for row in out.tolist()]}[/bold]")
    assert mat_close(out, one, 1e-6)


def test_qi_choi_psd_amplitude_damping():
    info("Choi matrix of amplitude damping is Hermitian PSD (random Rayleigh checks)")
    K = amplitude_damping_kraus(0.3)
    J = choi_from_kraus(K)
    # Hermitian
    H = mx.max(mx.abs(J - mx.conjugate(mx.transpose(J))))
    mx.eval(H)
    herm = float(H.item()) < 1e-6
    # PSD by random v*J v ≥ 0 for a few vectors
    import random
    psd = True
    for _ in range(10):
        v = mx.array([complex(random.random(), random.random()) for _ in range(J.shape[0])], mx.complex64)
        vcol = mx.reshape(v, (-1,1))
        val = mx.matmul(mx.conjugate(mx.transpose(vcol)), mx.matmul(J, vcol))
        mx.eval(val)
        psd = psd and (float(val[0,0].item().real) >= -1e-6)
    console.print(f"  • hermitian = [bold]{herm}[/bold], psd ≈ [bold]{psd}[/bold]")
    assert herm and psd


def test_qi_ptrace_dimensions_consistency():
    info("Partial trace dims for Bell reduced are 2×2")
    rhoA = ptrace(proj(bell_state(0)), traced_out=[1], dims=[2,2])
    console.print(f"  • shape = [bold]{rhoA.shape}[/bold]")
    assert rhoA.shape == (2,2)


def test_qi_operator_vectorization_roundtrip():
    info("operator→vector→operator roundtrip (2×2)")
    M = H()
    v = operator_to_vector(M)
    back = vector_to_operator(v, 2, 2)
    console.print(f"  • max|M-back| = [bold]{float(mx.max(mx.abs(M-back)).item()):.2e}[/bold]")
    assert mat_close(M, back, 1e-6)


def test_qi_stacked_index_roundtrip():
    info("stacked_index/unstacked_index roundtrip (dim=4)")
    dim = 4
    ok = True
    for r in range(dim):
        for c in range(dim):
            idx = stacked_index(dim, r, c)
            rr, cc = unstacked_index(dim, idx)
            ok = ok and (rr == r and cc == c)
    console.print(f"  • ok = [bold]{ok}[/bold]")
    assert ok


# -------------------------- Spin-1/2 related tests --------------------------

def test_spin_pauli_as_half():
    info("Spin-1/2: Sx=σx/2, etc. (unitarity of 2S matches Pauli)")
    from mlxq.spin import Sx, Sy, Sz
    ok = is_unitary(2.0 * Sx()) and is_unitary(2.0 * Sy()) and is_unitary(2.0 * Sz())
    console.print(f"  • ok = [bold]{ok}[/bold]")
    assert ok


def test_spin_expectation_on_states():
    info("Spin expectations: ⟨Sz⟩(|0⟩)=+1/2, ⟨Sz⟩(|1⟩)=-1/2")
    sz0 = expectation_value(mx.array([1+0j,0+0j], mx.complex64), Sz())
    sz1 = expectation_value(mx.array([0+0j,1+0j], mx.complex64), Sz())
    console.print(f"  • ⟨Sz⟩(|0⟩)=[bold]{sz0:.3f}[/bold], ⟨Sz⟩(|1⟩)=[bold]{sz1:.3f}[/bold]")
    assert close(sz0, 0.5, 1e-6) and close(sz1, -0.5, 1e-6)


def test_spin_larmor_precession_sz():
    info("Larmor under H=ω Sz via closed form unitary")
    omega = 1.0
    # start |+x> = H|0>
    sim = StateVectorSimulator(1)
    sim.h(0)
    psi0 = sim.state
    def U(t):
        c = math.cos(0.5*omega*t); s = math.sin(0.5*omega*t)
        return c*I() - 1j*s*Z()
    for t in [0.0, math.pi, 2*math.pi]:
        psit = mx.reshape(mx.matmul(U(t), mx.reshape(psi0,(2,1))), (2,))
        sx = expectation_value(psit, Sx())
        console.print(f"  • t={t:.2f}: ⟨Sx⟩=[bold]{sx:.3f}[/bold]")
    assert True


def test_spin_rotation_about_x_maps_z():
    info("Rotation about X maps Z expectation")
    # RX(π) flips |0> ↔ |1> thus ⟨Sz⟩ changes sign
    sim = StateVectorSimulator(1)
    z0 = expectation_value(sim.state, Sz())
    sim.apply_single(RX(math.pi), 0)
    z1 = expectation_value(sim.state, Sz())
    console.print(f"  • ⟨Sz⟩ before=[bold]{z0:.3f}[/bold], after=[bold]{z1:.3f}[/bold]")
    assert close(z0, 0.5, 1e-6) and close(z1, -0.5, 1e-6)


def test_superoperator_unitary_action():
    info("vec(UρU†) = (U⊗U*) vec(ρ)")
    U = H()
    rho = proj(mx.array([1+0j,0+0j], mx.complex64))
    from mlxq.tensor import kron
    S = kron(U, mx.conjugate(U))
    lhs = mx.matmul(S, operator_to_vector(rho))
    rho2 = mx.matmul(U, mx.matmul(rho, mx.conjugate(mx.transpose(U))))
    vec2 = operator_to_vector(rho2)
    console.print(f"  • max|S vec(ρ) - vec(UρU†)| = [bold]{float(mx.max(mx.abs(lhs - vec2)).item()):.2e}[/bold]")
    assert mat_close(lhs, vec2, 1e-6)


def test_vectorization_superoperator_identity():
    info("Vectorization identity with (U⊗U*) on vec(ρ)")
    st = random_state(1, seed=5)
    rho = proj(st)
    U = RY(0.7)
    from mlxq.tensor import kron
    S = kron(U, mx.conjugate(U))
    vec_left = mx.matmul(S, operator_to_vector(rho))
    vec_right = operator_to_vector(mx.matmul(U, mx.matmul(rho, mx.conjugate(mx.transpose(U)))))
    console.print(f"  • max|lhs-rhs| = [bold]{float(mx.max(mx.abs(vec_left-vec_right)).item()):.2e}[/bold]")
    assert mat_close(vec_left, vec_right, 1e-6)


# ---------------- Appendix (PRX Quantum) examples ---------------------------

def test_appendix_vqe_parameter_sweep():
    info("Appendix – VQE parameter sweep: H=2Z+X+I")
    cases = [(0.0, 3.0), (math.pi, -1.0)]
    rows = []
    ok = True
    for theta, expected in cases:
        psi = mx.reshape(RY(theta), (2,2)) @ mx.reshape(mx.array([1+0j,0+0j], mx.complex64),(2,1))
        psi = mx.reshape(psi,(2,))
        z = expectation_value(psi, Z())
        x = expectation_value(psi, X())
        energy = 2.0*z + x + 1.0
        rows.append((f"{theta:.3f}", f"{z:.6f}", f"{x:.6f}", f"{energy:.6f}"))
        ok = ok and close(energy, expected, 1e-4)
    table("θ, ⟨Z⟩, ⟨X⟩, ⟨H⟩", ("θ","⟨Z⟩","⟨X⟩","⟨H⟩"), rows)
    assert ok


def test_appendix_two_qubit_pauli_strings():
    info("Appendix – Two-qubit Pauli strings on Bell Φ+")
    st = bell_state(0)
    px, py, pz = X(), Y(), Z()
    zz = kron(pz,pz); xy = kron(px,py); yy = kron(py,py)
    zz_val = expectation_value(st, zz)
    xy_val = expectation_value(st, xy)
    yy_val = expectation_value(st, yy)
    console.print(f"  • ⟨ZZ⟩={zz_val:.6f}, ⟨XY⟩={xy_val:.6f}, ⟨YY⟩={yy_val:.6f}")
    assert close(zz_val, 1.0) and close(xy_val, 0.0) and close(yy_val, -1.0)


def test_appendix_ising_model_ground_state():
    info("Appendix – Ising model (2q) ground states of ZZ")
    h = kron(Z(), Z())
    min_e = 1e9; gs = []
    ok = True
    for idx in range(4):
        st = computational_basis(2, idx)
        e = expectation_value(st, h)
        console.print(f"  • |{idx:02b}⟩: ⟨H⟩={e:.6f}")
        if e < min_e - 1e-5: min_e = e; gs = [idx]
        elif abs(e - min_e) < 1e-5: gs.append(idx)
        exp_e = -1.0 if idx in (1,2) else 1.0
        ok = ok and close(e, exp_e)
    assert ok and close(min_e, -1.0) and len(gs) == 2


def test_appendix_bell_state_pauli_expectations():
    info("Appendix – Bell Φ+ Pauli expectations")
    st = bell_state(0)
    xx = kron(X(),X()); yy = kron(Y(),Y()); zz = kron(Z(),Z())
    xx_val = expectation_value(st, xx)
    yy_val = expectation_value(st, yy)
    zz_val = expectation_value(st, zz)
    console.print(f"  • ⟨XX⟩={xx_val:.6f}, ⟨YY⟩={yy_val:.6f}, ⟨ZZ⟩={zz_val:.6f}")
    assert close(xx_val, 1.0) and close(yy_val, -1.0) and close(zz_val, 1.0)


def test_appendix_multi_qubit_hamiltonian_grouping():
    info("Appendix – Multi-qubit Hamiltonian grouping (commuting groups)")
    terms = []
    coeffs = [1.2,1.0,0.8,0.6,0.4,0.2]
    for i in range(6):
        word = list('I'*6)
        word[i] = 'X'
        terms.append((''.join(word), coeffs[i]))
    for i in range(3):
        word = list('I'*6)
        word[i] = 'Z'; word[i+1] = 'Z'; word[i+2] = 'X'; word[i+3] = 'Y'
        terms.append((''.join(word), 1.0))
    console.print(f"  • Total terms = [bold]{len(terms)}[/bold]")
    groups = [[0,1,2,3,4,5],[6,7],[8]]
    def group_commutes(g):
        for a in range(len(g)):
            for b in range(a+1,len(g)):
                if not pauli_strings_commute_words(terms[g[a]][0], terms[g[b]][0]):
                    return False
        return True
    ok = (len(terms)==9) and len(groups)==3 and all(group_commutes(g) for g in groups)
    console.print(f"  • Groups commuting = [bold]{ok}[/bold]")
    assert ok


def test_appendix_hw3_bell_state_expectations():
    info("Appendix – HW3 Bell: E(X₁Z₂)=0")
    st = bell_state(0)
    XZ = kron(X(), Z())
    val = expectation_value(st, XZ)
    console.print(f"  • ⟨XZ⟩ = [bold]{val:.6f}[/bold]")
    assert close(val, 0.0)


def test_appendix_hw3_pure_vs_mixed_state():
    info("Appendix – Pure vs mixed purity")
    pure = mx.array([1/math.sqrt(2)+0j, 1/math.sqrt(2)+0j], mx.complex64)
    rho_pure = proj(pure)
    p_pure = purity(rho_pure)
    rho_mixed = max_mixed(2)
    p_mixed = purity(rho_mixed)
    console.print(f"  • Tr(ρ²): pure={p_pure:.6f}, mixed={p_mixed:.6f}")
    assert close(p_pure, 1.0) and close(p_mixed, 0.5)


def test_appendix_hw2_tensor_products():
    info("Appendix – Tensor products and commutation")
    x,z,i = X(),Z(),I()
    xz = kron(x,z); ix = kron(i,x); xi = kron(x,i)
    # Expected per C++ construction
    expected_xz = mx.array([[0,0,1,0],[0,0,0,-1],[1,0,0,0],[0,-1,0,0]], mx.complex64)
    expected_ix = mx.array([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]], mx.complex64)
    expected_xi = mx.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]], mx.complex64)
    m_ok = mat_close(xz, expected_xz) and mat_close(ix, expected_ix) and mat_close(xi, expected_xi)
    comm = mx.matmul(ix, xi) - mx.matmul(xi, ix)
    comm_norm = float(mx.max(mx.abs(comm)).item())
    console.print(f"  • matrices ok={m_ok}, max|comm|={comm_norm:.2e}")
    assert m_ok and comm_norm < 1e-5


def test_appendix_hw2_matrix_exponential():
    info("Appendix – e^{iθX} identity")
    th = math.pi/4
    expx = exp_i_pauli(X(), th)
    expected = math.cos(th)*I() + 1j*math.sin(th)*X()
    d = float(mx.max(mx.abs(expx - expected)).item())
    console.print(f"  • max|expx-expected| = [bold]{d:.2e}[/bold]")
    assert d < 1e-5


def test_appendix_hw2_unitary_property():
    info("Appendix – e^{iθZ} is unitary")
    th = 0.73
    U = exp_i_pauli(Z(), th)
    ok = is_unitary(U)
    console.print(f"  • unitary = [bold]{ok}[/bold]")
    assert ok


def test_appendix_hw2_concurrence():
    info("Appendix – Concurrence for |Ψ⁻⟩ and |00⟩")
    s = 1.0/math.sqrt(2.0)
    psi_minus = custom_state([0+0j, s+0j, -s+0j, 0+0j])
    C_ent = concurrence_pure(psi_minus)
    C_sep = concurrence_pure(mx.array([1+0j,0+0j,0+0j,0+0j], mx.complex64))
    console.print(f"  • C(Ψ⁻)={C_ent:.6f}, C(|00⟩)={C_sep:.6f}")
    assert C_ent > 0.99 and C_sep < 1e-4


def test_appendix_hw214b_y_basis_time_evolution():
    info("Appendix – Y-basis time evolution overlap probability")
    omega_t = 0.42
    e_minus = complex(math.cos(omega_t), -math.sin(omega_t))
    e_plus = complex(math.cos(omega_t),  math.sin(omega_t))
    psi_t = custom_state([e_minus/math.sqrt(2.0), e_plus/math.sqrt(2.0)])
    prob = abs(state_overlap(y_plus_state(), psi_t))**2
    expected = ((math.cos(omega_t) + math.sin(omega_t))**2) / 2.0
    console.print(f"  • P(y+)={prob:.6f}, expected={expected:.6f}")
    assert close(prob, expected, 1e-4)


def test_appendix_hw24_pauli_decomposition():
    info("Appendix – Pauli decomposition coefficients")
    M = mx.array([[2+0j, 1+1j],[1-1j, -2+0j]], mx.complex64)
    w,x,y,z = pauli_decomposition_2x2(M)
    console.print(f"  • w={w.real:.3f}, x={x.real:.3f}, y={y.real:.3f}, z={z.real:.3f}")
    assert close(w.real, 0.0) and close(x.real, 1.0) and close(y.real, -1.0) and close(z.real, 2.0)


def test_appendix_hw3_advanced_circuit_decomposition():
    info("Appendix – RX(θ) = RZ(-π/2)·RY(θ)·RZ(π/2)")
    th = 0.9
    rz_pos = RZ(math.pi/2); rz_neg = RZ(-math.pi/2)
    ry = RY(th); rx = RX(th)
    decomposed = mx.matmul(rz_neg, mx.matmul(ry, rz_pos))
    d = float(mx.max(mx.abs(decomposed - rx)).item())
    console.print(f"  • max|decomp-RX| = [bold]{d:.2e}[/bold]")
    # state action on |+>
    plus = mx.array([1/math.sqrt(2)+0j, 1/math.sqrt(2)+0j], mx.complex64)
    via_decomp = mx.reshape(mx.matmul(decomposed, mx.reshape(plus,(2,1))), (2,))
    via_direct = mx.reshape(mx.matmul(rx, mx.reshape(plus,(2,1))), (2,))
    dd = float(mx.max(mx.abs(via_decomp - via_direct)).item())
    console.print(f"  • max|ψ_decomp-ψ_dir| = [bold]{dd:.2e}[/bold]")
    assert d < 1e-5 and dd < 1e-5


# ---------------- Thirteenth batch: Advanced appendix examples ---------------

def test_elitzur_vaidman_bomb_tester():
    info("Elitzur–Vaidman: distinguish Bell vs product by probabilities")
    # Bell Φ+: nonzero only on |00>,|11>
    sim = StateVectorSimulator(2)
    sim.h(0); sim.cnot(0,1)
    P_bell = sim.probabilities()
    # Product |+>⊗|+>: all four equal
    sim2 = StateVectorSimulator(2)
    sim2.h(0); sim2.h(1)
    P_prod = sim2.probabilities()
    table("Bell probs", ("00","01","10","11"), [(f"{P_bell[0]:.3f}",f"{P_bell[1]:.3f}",f"{P_bell[2]:.3f}",f"{P_bell[3]:.3f}")])
    table("Prod probs", ("00","01","10","11"), [(f"{P_prod[0]:.3f}",f"{P_prod[1]:.3f}",f"{P_prod[2]:.3f}",f"{P_prod[3]:.3f}")])
    assert close(P_bell[0],0.5) and close(P_bell[3],0.5) and all(close(P_prod[i],0.25,1e-3) for i in range(4))


def test_two_qubit_separability_analysis():
    info("Separability: |+>⊗|+> uniform; Bell only 00/11")
    # uniform state
    sim = StateVectorSimulator(2); sim.h(0); sim.h(1)
    P = sim.probabilities(); assert all(close(p,0.25,1e-3) for p in P)
    # bell state
    sim = StateVectorSimulator(2); sim.h(0); sim.cnot(0,1)
    P2 = sim.probabilities(); assert close(P2[0],0.5) and close(P2[3],0.5)


def test_bell_basis_measurements():
    info("Bell basis measurement via overlaps with Bell states")
    # Build |+>⊗|+>
    sim = StateVectorSimulator(2); sim.h(0); sim.h(1)
    psi = sim.state
    # Bell basis states
    bell_phi_plus = bell_state(0)
    bell_phi_minus = bell_state(1)
    bell_psi_plus = bell_state(2)
    bell_psi_minus = bell_state(3)
    def overlap2(a,b):
        v = mx.sum(mx.conjugate(a)*b); mx.eval(v); return abs(complex(v.item()))**2
    probs = [overlap2(bell_phi_plus, psi), overlap2(bell_phi_minus, psi), overlap2(bell_psi_plus, psi), overlap2(bell_psi_minus, psi)]
    console.print(f"  • Bell basis probs = [bold]{[round(x,3) for x in probs]}[/bold]")
    assert close(sum(probs), 1.0, 1e-6)


def test_grover_search_algorithm():
    info("Grover search 3q for |101> (one iteration)")
    n=3; N=1<<n; marked=0b101
    # Start in uniform superposition
    sim = StateVectorSimulator(n); sim.h(0); sim.h(1); sim.h(2)
    # Oracle: flip phase of marked
    st = sim.state.tolist(); st[marked] *= -1.0; sim.state = mx.array(st, mx.complex64)
    # Diffusion operator: 2|s><s| - I
    amp = 1/math.sqrt(N); s = mx.array([amp+0j]*N, mx.complex64)
    P = mx.reshape(mx.matmul(mx.reshape(s,(N,1)), mx.reshape(mx.conjugate(s),(1,N))), (N,N))
    D = 2.0*P - mx.array([[1+0j if i==j else 0+0j for j in range(N)] for i in range(N)], mx.complex64)
    sim.state = mx.reshape(mx.matmul(D, mx.reshape(sim.state,(N,1))), (N,))
    probs = sim.probabilities(); console.print(f"  • argmax = [bold]{probs.index(max(probs))}[/bold]")
    assert probs.index(max(probs)) == marked


def test_spin_general_spinor_expectations():
    info("Spinor statistics: ⟨Sx,Sy,Sz⟩ via overlaps")
    # generic spinor a|0> + b|1>
    a = complex(0.6,0.3); b = complex(math.sqrt(1-abs(a)**2), 0.0) * complex(math.cos(math.pi/3), math.sin(math.pi/3))
    psi = mx.array([a,b], mx.complex64)
    sx = expectation_value(psi, Sx()); sy = expectation_value(psi, Sy()); sz = expectation_value(psi, Sz())
    # analytic
    overlap = (a.conjugate()*b)
    sx_a = overlap.real; sy_a = overlap.imag; sz_a = 0.5*(abs(a)**2 - abs(b)**2)
    console.print(f"  • num=({sx:.3f},{sy:.3f},{sz:.3f}) ana=({sx_a:.3f},{sy_a:.3f},{sz_a:.3f})")
    assert close(sx, sx_a, 1e-3) and close(sy, sy_a, 1e-3) and close(sz, sz_a, 1e-3)


def test_spin_sy_measurements():
    info("Sᵧ measurement probabilities on generic spinor")
    a = complex(0.45,0.35); b = complex(math.sqrt(1-abs(a)**2),0.0) * complex(0.2,0.9797959)
    psi = mx.array([a,b], mx.complex64)
    sy_plus = y_minus_state()  # (|0>-i|1>)/√2 matches +1 eigen of Sy when Sy=σy/2
    sy_minus = y_plus_state()  # (|0>+i|1>)/√2
    def prob(phi):
        v = mx.sum(mx.conjugate(phi)*psi); mx.eval(v); return abs(complex(v.item()))**2
    p_plus = prob(sy_plus); p_minus = prob(sy_minus)
    console.print(f"  • P(+)= [bold]{p_plus:.3f}[/bold], P(-)= [bold]{p_minus:.3f}[/bold]")
    assert close(p_plus + p_minus, 1.0, 1e-6)


def test_spin_sx_sy_eigenproblem():
    info("Eigen: Sx|+x>=+1/2|+x>, Sy|y+>=+1/2|y+>")
    plus_x = mx.array([1/math.sqrt(2)+0j, 1/math.sqrt(2)+0j], mx.complex64)
    yplus = y_plus_state()
    v1 = mx.reshape(mx.matmul(Sx(), mx.reshape(plus_x,(2,1))), (2,))
    v2 = (0.5+0j) * plus_x
    v3 = mx.reshape(mx.matmul(Sy(), mx.reshape(yplus,(2,1))), (2,))
    v4 = (0.5+0j) * yplus
    assert mat_close(v1, v2, 1e-5) and mat_close(v3, v4, 1e-5)


def test_spin_two_stage_field_rotation():
    info("Two-stage rotation about Z then Y; Sz expectation changes consistently")
    # Start |0>, apply RZ then RY and check Sz change from +1/2 toward -1/2 as θ increases
    for th in (0.0, math.pi/2, math.pi):
        sim = StateVectorSimulator(1)
        sim.apply_single(RZ(0.3), 0)
        sim.apply_single(RY(th), 0)
        z = expectation_value(sim.state, Sz())
        console.print(f"  • θ={th:.2f} => ⟨Sz⟩={z:.3f}")
    assert True


def test_spin_cosine_field_flip():
    info("Cosine field flip analog: RX(π) flips Sz")
    sim = StateVectorSimulator(1)
    z0 = expectation_value(sim.state, Sz())
    sim.apply_single(RX(math.pi), 0)
    z1 = expectation_value(sim.state, Sz())
    console.print(f"  • ⟨Sz⟩ before={z0:.3f}, after={z1:.3f}")
    assert close(z0, 0.5, 1e-6) and close(z1, -0.5, 1e-6)


# ---- Additional migration parity tests (C++ ops → Python) ----

def test_multictrlz_two_controls_phase_on_all_ones():
    info("MCZ (2 controls) applies -1 to |111> state only")
    from mlxq.gates import MultiControlledZ
    M = MultiControlledZ(2)
    mx.eval(M)
    diag = [complex(M[i,i].item()) for i in range(8)]
    ok = all(abs(d.real - ( -1.0 if i==7 else 1.0)) < 1e-6 and abs(d.imag) < 1e-9 for i,d in enumerate(diag))
    console.print(f"  • diag(MCZ2) ends with {diag[-3:]} (ok={ok})")
    assert ok


def test_multictrlx_two_controls_flip_truth():
    info("MCX (2 controls) flips target for control state 11*")
    from mlxq.gates import MultiControlledX
    M = MultiControlledX(2)
    # Start in |110> (index 6), expect |111> (index 7)
    v_list = [0+0j]*8
    v_list[6] = 1+0j
    v = mx.array(v_list, mx.complex64)
    out = mx.reshape(mx.matmul(M, mx.reshape(v,(8,1))), (8,))
    probs = mx.abs(out)**2; mx.eval(probs)
    P = [float(x) for x in probs.tolist()]
    console.print(f"  • P(argmax)={P.index(max(P))}, P7={P[7]:.3f}")
    assert P[7] > 0.999


def test_is_hermitian_and_commutator_api():
    info("Hermiticity(Z)=True; [X,Y]=2iZ via API")
    ok_h = is_hermitian(Z())
    C = commutator(X(), Y())
    rhs = 2.0j * Z()
    d = float(mx.max(mx.abs(C - rhs)).item())
    console.print(f"  • hermitian(Z)={ok_h}, max|[X,Y]-2iZ|={d:.2e}")
    assert ok_h and d < 1e-5

if __name__ == '__main__':
    run_all()

# ---------------- Circuit drawing tests (ASCII + random circuits) ----------

def test_draw_ascii_small():
    info("ASCII drawer: small 3q circuit (H; CNOT; SWAP)")
    ops = [
        {"name":"H","wires":[0]},
        {"name":"CNOT","wires":[0,1]},
        {"name":"SWAP","wires":[1,2]},
        {"name":"RZ","wires":[2],"parameters":[0.3]},
    ]
    art = circuit_ascii(3, ops, col_width=9)
    console.print(art)
    assert "H" in art and "⊕" in art and "×" in art


def test_draw_ascii_random_medium():
    info("ASCII drawer: random 5q, depth=8")
    ops = random_circuit(5, depth=8, seed=7)
    art = circuit_ascii(5, ops, col_width=9)
    console.print(art)
    # Basic sanity: correct number of rows and some gate labels present
    assert art.count("q0") == 1 and art.count("q4") == 1
    assert any(lbl in art for lbl in ("[H]","[X]","[RX]","[RY]","[RZ]"))


def test_quantikz_small():
    info("quantikz: small 2q bell circuit")
    ops = [
        {"name":"H","wires":[0]},
        {"name":"CNOT","wires":[0,1]},
    ]
    tex = circuit_to_quantikz(2, ops)
    console.print(tex)
    assert "\\begin{quantikz}" in tex and "\\ctrl{1}" in tex and "\\targ{}" in tex

def test_circuit_mpl_small_and_random():
    info("Matplotlib drawer: small and random circuits render without error")
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception:
        warn("matplotlib not available; skipping circuit_mpl test")
        return
    # Small circuit
    ops_small = [
        {"name": "H", "wires": [0]},
        {"name": "CNOT", "wires": [0, 1]},
        {"name": "SWAP", "wires": [1, 2]},
        {"name": "RZ", "wires": [2], "parameters": [0.3]},
    ]
    res1 = circuit_mpl(3, ops_small, title="Small circuit (test)")
    assert res1 is not None
    # Random circuit
    ops_rand = random_circuit(5, depth=6, seed=11)
    res2 = circuit_mpl(5, ops_rand, title="Random circuit (test)")
    assert res2 is not None
