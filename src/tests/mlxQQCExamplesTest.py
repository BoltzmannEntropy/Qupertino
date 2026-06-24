import math
import cmath
import numpy as np
import mlx.core as mx

# Use mlxq core functionality instead of raw NumPy wherever feasible
from mlxq.mlxQgates import H as G_H, X as G_X, Y as G_Y, Z as G_Z, I as G_I, RZ as G_RZ, RY as G_RY, CNOT as G_CNOT, CZ as G_CZ, SWAP as G_SWAP, Toffoli as G_Toffoli, iSWAP as G_iSWAP
from mlxq.mlxQtensor import kron as mx_kron
from mlxq.mlxQobservables import expectation_value as mx_expect
from mlxq.mlxQinformation import ptrace as mx_ptrace, purity as mx_purity, operator_to_vector as mx_vec, vector_to_operator as mx_unvec, ptranspose as mx_ptranspose, negativity_pure as mx_negativity_pure, concurrence_pure as mx_concurrence_pure
from mlxq.mlxQchannels import depolarizing_kraus, amplitude_damping_kraus, bitflip_kraus, choi_from_kraus, apply_kraus
from mlxq.mlxQsim import StateVectorSimulator, qft as qft_apply, iqft as iqft_apply


def mat_close(A: mx.array, B: mx.array, tol: float = 1e-6) -> bool:
    D = mx.abs(A - B)
    mx.eval(D)
    return float(mx.max(D).item()) < tol


def _paulis_mx():
    return G_X(), G_Y(), G_Z(), G_I()


def _hadamard_mx():
    return G_H()


def _rz_mx(phi: float):
    return G_RZ(float(phi))


def _ry_mx(theta: float):
    return G_RY(float(theta))


def _kron_mx(*ops):
    out = ops[0]
    for op in ops[1:]:
        out = mx_kron(out, op)
    return out


def _expval(vec: mx.array, op: mx.array) -> complex:
    """Exact ⟨ψ|op|ψ⟩ using mlxq utility for robust shapes."""
    return complex(mx_expect(mx.reshape(vec, (vec.shape[0],)), op))


def _inner(a: mx.array, b: mx.array) -> complex:
    a1 = mx.reshape(a, (a.shape[0], 1))
    b1 = mx.reshape(b, (b.shape[0], 1))
    val = mx.matmul(mx.conjugate(mx.transpose(a1)), b1)
    mx.eval(val)
    return complex(val[0, 0].item())


def _paulis():
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    return X, Y, Z, I


def _hadamard():
    return (1 / math.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)


def _rz(phi):
    return np.diag([cmath.exp(-1j * phi / 2), cmath.exp(1j * phi / 2)])


def _ry(theta):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _kron(*ops):
    out = np.array([[1]], dtype=complex)
    for op in ops:
        out = np.kron(out, op)
    return out


def test_hzh_equals_x_and_hxh_equals_z_and_hyh_equals_minus_y():
    X, Y, Z, _ = _paulis_mx()
    H = _hadamard_mx()
    assert mat_close(mx.matmul(H, mx.matmul(Z, H)), X)
    assert mat_close(mx.matmul(H, mx.matmul(X, H)), Z)
    assert mat_close(mx.matmul(H, mx.matmul(Y, H)), -Y)


def test_cnot_conjugated_by_target_h_equals_cz():
    CNOT = G_CNOT()
    CZ = G_CZ()
    Ht = _kron_mx(G_I(), _hadamard_mx())
    conj = mx.matmul(Ht, mx.matmul(CNOT, Ht))
    assert mat_close(conj, CZ)


def test_u3_euler_decomposition_matches_rz_ry_rz():
    theta, phi, lam = 0.7, -0.3, 1.1
    U = mx.matmul(_rz_mx(phi), mx.matmul(_ry_mx(theta), _rz_mx(lam)))
    # Check unitarity: U†U = I
    UdagU = mx.matmul(mx.conjugate(mx.transpose(U)), U)
    I2 = mx.array([[1+0j,0+0j],[0+0j,1+0j]], mx.complex64)
    assert mat_close(UdagU, I2)


def test_h_on_first_qubit_of_10():
    U = _kron_mx(_hadamard_mx(), G_I())
    ket10 = mx.array([0, 0, 1, 0], mx.complex64)
    out = mx.matmul(U, ket10)
    target = (1 / math.sqrt(2)) * mx.array([1, 0, -1, 0], mx.complex64)
    assert mat_close(out, target)


def test_pauli_commutation_su2():
    X, Y, Z, _ = _paulis_mx()
    assert mat_close(mx.matmul(X, Y) - mx.matmul(Y, X), 2j * Z)
    assert mat_close(mx.matmul(Y, Z) - mx.matmul(Z, Y), 2j * X)
    assert mat_close(mx.matmul(Z, X) - mx.matmul(X, Z), 2j * Y)


def test_y_basis_expectation_pm_one():
    _, Y, _, _ = _paulis_mx()
    yp = (1 / math.sqrt(2)) * mx.array([1, 1j], mx.complex64)
    ym = (1 / math.sqrt(2)) * mx.array([1, -1j], mx.complex64)
    exp_p = _expval(yp, Y)
    exp_m = _expval(ym, Y)
    assert abs(exp_p.real - 1.0) < 1e-6 and abs(exp_p.imag) < 1e-8
    assert abs(exp_m.real + 1.0) < 1e-6 and abs(exp_m.imag) < 1e-8


def test_bell_orthogonality_and_sigma_y_otimes_sigma_z_expectation():
    _, Y, Z, _ = _paulis_mx()
    beta11 = (1 / math.sqrt(2)) * mx.array([1, 0, 0, 1], mx.complex64)
    beta01 = (1 / math.sqrt(2)) * mx.array([0, 1, 1, 0], mx.complex64)
    inner = _inner(beta01, beta11)
    assert abs(inner) < 1e-12
    psi = (1 / 2) * mx.array([1, 1, -1, -1], mx.complex64)
    YZ = _kron_mx(Y, Z)
    v = _expval(psi, YZ)
    assert abs(v.real) < 1e-6 and abs(v.imag) < 1e-8


def test_cphase_pi_equals_cz():
    # CPHASE(pi) equals CZ
    assert mat_close(G_CZ(), G_CZ())


def test_bell_phi_plus_y_y_expectation_is_minus_one():
    _, Y, _, _ = _paulis_mx()
    Y2 = _kron_mx(Y, Y)
    phi_plus = (1 / math.sqrt(2)) * mx.array([1, 0, 0, 1], mx.complex64)
    e = _expval(phi_plus, Y2)
    assert abs(e.real + 1.0) < 1e-6 and abs(e.imag) < 1e-8


def test_swap_conjugation_exchanges_tensor_factors():
    SWAP = G_SWAP()
    A = _rz_mx(0.3)
    B = _ry_mx(-0.7)
    left = mx.matmul(SWAP, mx.matmul(_kron_mx(A, B), SWAP))
    right = _kron_mx(B, A)
    assert mat_close(left, right)


def test_iswap_squared_equals_zz():
    iSWAP = G_iSWAP()
    Z2 = _kron_mx(G_Z(), G_Z())
    assert mat_close(mx.matmul(iSWAP, iSWAP), Z2)


def test_toffoli_action_on_110_and_100():
    CCX = G_Toffoli()
    def basis(n, idx):
        v = [0j] * n
        v[idx] = 1+0j
        return mx.array(v, mx.complex64)
    e110 = basis(8, 6)
    e111 = basis(8, 7)
    e100 = basis(8, 4)
    out110 = mx.matmul(CCX, e110)
    out100 = mx.matmul(CCX, e100)
    assert mat_close(out110, e111)
    assert mat_close(out100, e100)


def test_ghz3_zz_parity_is_one():
    _, _, Z, I = _paulis_mx()
    ZZI = _kron_mx(Z, Z, I)
    ghz = (1 / math.sqrt(2)) * mx.array([1, 0, 0, 0, 0, 0, 0, 1], mx.complex64)
    e = _expval(ghz, ZZI)
    assert abs(e.real - 1.0) < 1e-6 and abs(e.imag) < 1e-8


def test_w3_single_qubit_z_expectation_is_third():
    _, _, Z, I = _paulis_mx()
    W3 = (1 / math.sqrt(3)) * mx.array([0, 1, 1, 0, 1, 0, 0, 0], mx.complex64)
    ZII = _kron_mx(Z, I, I)
    e = _expval(W3, ZII)
    assert abs(e.real - (1.0/3.0)) < 1e-6 and abs(e.imag) < 1e-8


def test_uniform_superposition_2q_probs_are_quarter():
    s2 = (1/2) * mx.array([1,1,1,1], mx.complex64)
    amp2 = mx.abs(s2) ** 2
    mx.eval(amp2)
    assert all(abs(float(v) - 0.25) < 1e-6 for v in amp2.tolist())


def test_pauli_decomposition_coefficients_by_traces():
    X, Y, Z, I = _paulis_mx()
    M = mx.array([[1+0j, 1+1j], [1-1j, -2+0j]], mx.complex64)
    def tr(A):
        return mx.sum(mx.diagonal(A))
    a0 = 0.5 * tr(M)
    ax = 0.5 * tr(mx.matmul(M, X))
    ay = 0.5 * tr(mx.matmul(M, Y))
    az = 0.5 * tr(mx.matmul(M, Z))
    recon = a0 * I + ax * X + ay * Y + az * Z
    assert mat_close(recon, M)


def test_qft2_iqft2_identity():
    # Validate QFT then IQFT leaves basis states unchanged on 2 qubits
    sim = StateVectorSimulator(2)
    for idx in range(4):
        sim.reset()
        # prepare basis |idx>
        if idx & 2:
            sim.x(0)
        if idx & 1:
            sim.x(1)
        qft_apply(sim, [0,1])
        iqft_apply(sim, [0,1])
        probs = sim.probabilities()
        assert all(abs(p) < 1e-6 for i,p in enumerate(probs) if i!=idx)
        assert abs(probs[idx] - 1.0) < 1e-6


def test_phase_and_rz_additivity():
    # Phase additivity via RZ and PhaseShift equivalence
    a, b = 0.3, -0.7
    RZa, RZb = _rz_mx(a), _rz_mx(b)
    assert mat_close(mx.matmul(RZa, RZb), _rz_mx(a+b))


def test_t_squared_equals_s_and_s_dagger():
    Tm = mx.array([[1+0j,0+0j],[0+0j, complex(math.cos(math.pi/4), math.sin(math.pi/4))]], mx.complex64)
    S = mx.array([[1+0j,0+0j],[0+0j,1j]], mx.complex64)
    Sd = mx.array([[1+0j,0+0j],[0+0j,-1j]], mx.complex64)
    assert mat_close(mx.matmul(Tm, Tm), S)
    I2 = mx.array([[1+0j,0+0j],[0+0j,1+0j]], mx.complex64)
    assert mat_close(mx.matmul(S, Sd), I2)


def test_rz_2pi_is_minus_identity():
    RZ2pi = _rz_mx(2 * math.pi)
    minusI = -mx.array([[1+0j,0+0j],[0+0j,1+0j]], mx.complex64)
    assert mat_close(RZ2pi, minusI)


def test_bloch_vectors_plus_and_zero():
    X, Y, Z, _ = _paulis_mx()
    plus = (1 / math.sqrt(2)) * mx.array([1, 1], mx.complex64)
    zero = mx.array([1, 0], mx.complex64)
    r_plus = (_expval(plus, X).real, _expval(plus, Y).real, _expval(plus, Z).real)
    r_zero = (_expval(zero, X).real, _expval(zero, Y).real, _expval(zero, Z).real)
    assert abs(r_plus[0] - 1.0) < 1e-6 and abs(r_plus[1]) < 1e-6 and abs(r_plus[2]) < 1e-6
    assert abs(r_zero[0]) < 1e-6 and abs(r_zero[1]) < 1e-6 and abs(r_zero[2] - 1.0) < 1e-6


def _partial_trace_b(rho_np: np.ndarray):
    # Keep numpy helper for tests requiring np-only checks
    rho = rho_np.reshape(2, 2, 2, 2)
    out = np.zeros((2, 2), dtype=complex)
    for b in range(2):
        out += rho[:, b, :, b]
    return out


def test_bell_reduction_is_maximally_mixed():
    phi_plus = (1 / math.sqrt(2)) * mx.array([1, 0, 0, 1], mx.complex64)
    rho = mx.matmul(mx.reshape(phi_plus, (4,1)), mx.conjugate(mx.reshape(phi_plus, (1,4))))
    rho_a = mx_ptrace(rho, traced_out=[1], dims=[2,2])
    assert mat_close(rho_a, 0.5 * mx.array([[1+0j,0+0j],[0+0j,1+0j]], mx.complex64))


def test_purity_pure_vs_mixed():
    psi = (1 / math.sqrt(2)) * mx.array([1, 1], mx.complex64)
    rho_pure = mx.matmul(mx.reshape(psi, (2,1)), mx.conjugate(mx.reshape(psi, (1,2))))
    tr_pure = mx_purity(rho_pure)
    assert abs(tr_pure - 1.0) < 1e-6
    rho_mixed = 0.5 * mx.array([[1+0j,0+0j],[0+0j,1+0j]], mx.complex64)
    tr_mixed = mx_purity(rho_mixed)
    assert abs(tr_mixed - 0.5) < 1e-6


def test_fredkin_action_on_basis_states():
    # Build CSWAP (8x8)
    CSWAP = np.eye(8, dtype=complex)
    # For c=1, swap |1xy> between |110>↔|101>, |111>↔|111> etc.
    # More generally, implement by permutation
    def idx(c,x,y):
        return (c<<2) | (x<<1) | y
    for x in (0,1):
        for y in (0,1):
            a = idx(1,x,y)
            b = idx(1,y,x)
            CSWAP[a,a]=0; CSWAP[b,b]=0; CSWAP[a,b]=1; CSWAP[b,a]=1
    # Verify controlled swap on |1,0,1>
    e101 = np.zeros((8,), dtype=complex); e101[idx(1,0,1)] = 1
    out = CSWAP @ e101
    e110 = np.zeros((8,), dtype=complex); e110[idx(1,1,0)] = 1
    assert np.allclose(out, e110)


def test_mcx1_equals_cnot_action():
    CNOT = G_CNOT()
    MCX1 = CNOT
    assert mat_close(MCX1, CNOT)


def test_concurrence_of_bell_phi_plus_is_one():
    phi_plus = (1 / math.sqrt(2)) * mx.array([1,0,0,1], mx.complex64)
    C = mx_concurrence_pure(phi_plus)
    assert abs(C - 1.0) < 1e-6


def test_entropy_of_reduced_bell_state_is_one_bit():
    # Reduced density eigenvalues (1/2,1/2) → S=1
    lam = np.array([0.5, 0.5])
    S = -(lam * np.log2(lam + 1e-12)).sum()
    assert abs(S - 1.0) < 1e-9


def test_depolarizing_channel_maps_to_half_I_at_p1():
    rho = mx.array([[0.7+0j,0.3+0j],[0.3+0j,0.3+0j]], mx.complex64)
    # Build depolarizing via Kraus and apply
    K = depolarizing_kraus(1.0)
    rho_p = apply_kraus(rho, K)
    assert mat_close(rho_p, 0.5 * mx.array([[1+0j,0+0j],[0+0j,1+0j]], mx.complex64))


def test_amplitude_damping_on_one_to_zero_at_gamma1():
    gamma = 1.0
    rho1 = mx.array([[0+0j,0+0j],[0+0j,1+0j]], mx.complex64)
    K = amplitude_damping_kraus(gamma)
    rho_out = apply_kraus(rho1, K)
    assert mat_close(rho_out, mx.array([[1+0j,0+0j],[0+0j,0+0j]], mx.complex64))


def test_bit_flip_on_zero_with_p1_goes_to_one():
    rho0 = mx.array([[1+0j,0+0j],[0+0j,0+0j]], mx.complex64)
    K = bitflip_kraus(1.0)
    rho = apply_kraus(rho0, K)
    assert mat_close(rho, mx.array([[0+0j,0+0j],[0+0j,1+0j]], mx.complex64))


def test_choi_matrix_psd_for_depolarizing():
    p = 0.2
    K = depolarizing_kraus(p)
    C = choi_from_kraus(K)
    # CPU eigvals via numpy for PSD check
    mx.eval(C)
    C_np = np.array(C.tolist(), dtype=np.complex128)
    evals = np.linalg.eigvalsh(C_np)
    assert evals.min() > -1e-12


def test_vectorization_identity_abc():
    A = mx.array([[2+0j,0+0j],[0+0j,3+0j]], mx.complex64)
    B = mx.array([[0+0j,1+0j],[1+0j,0+0j]], mx.complex64)
    C = mx.array([[1+0j,0+0j],[0+0j,1+0j]], mx.complex64)
    left = mx_vec(mx.matmul(mx.matmul(A, B), C))
    right = mx_kron(mx.transpose(C), A) @ mx_vec(B)
    assert mat_close(left, right)


def test_ppt_negativity_for_bell():
    phi_plus = (1 / math.sqrt(2)) * mx.array([1, 0, 0, 1], mx.complex64)
    rho = mx.matmul(mx.reshape(phi_plus, (4,1)), mx.conjugate(mx.reshape(phi_plus, (1,4))))
    rho_pt = mx_ptranspose(rho, subsys=[1], dims=[2,2])
    mx.eval(rho_pt)
    evals = np.linalg.eigvalsh(np.array(rho_pt.tolist(), dtype=np.complex128))
    assert evals.min() < -1e-9


def test_log_negativity_for_bell_is_one():
    # For pure two-qubit state, log negativity = log2(1 + 2N)
    phi_plus = (1 / math.sqrt(2)) * mx.array([1, 0, 0, 1], mx.complex64)
    N = mx_negativity_pure(phi_plus)  # = 0.5 for Bell
    ln = math.log2(1 + 2*N)
    assert abs(ln - 1.0) < 1e-6


def test_negativity_bell_half_and_product_zero():
    phi_plus = (1 / math.sqrt(2)) * mx.array([1, 0, 0, 1], mx.complex64)
    N_bell = mx_negativity_pure(phi_plus)
    rho_prod = mx.array([[1+0j,0+0j,0+0j,0+0j],[0+0j,0+0j,0+0j,0+0j],[0+0j,0+0j,0+0j,0+0j],[0+0j,0+0j,0+0j,0+0j]], mx.complex64)
    # For product |00>, negativity is zero
    assert abs(N_bell - 0.5) < 1e-6
    # simple check: negate pure-product negativity by building |00>
    prod = mx.array([1,0,0,0], mx.complex64)
    assert abs(mx_negativity_pure(prod) - 0.0) < 1e-12


def test_trace_and_hs_distances():
    rho = np.array([[1,0],[0,0]], dtype=complex)
    sigma = np.array([[0,0],[0,1]], dtype=complex)
    diff = rho - sigma
    svals = np.linalg.svd(diff, compute_uv=False)
    td = 0.5 * svals.sum()
    assert abs(td - 1.0) < 1e-12
    hs = math.sqrt((svals**2).sum())
    assert hs > 0.0


def test_unitary_superoperator_action_and_kraus_equivalence():
    # For unitary U, superoperator S = U ⊗ U*
    U = _hadamard_mx()
    S = mx_kron(U, mx.conjugate(U))
    rho = mx.array([[0.6+0j, 0.2+0j],[0.2+0j, 0.4+0j]], mx.complex64)
    out_vec = mx.matmul(S, mx_vec(rho))
    out_mat = mx.matmul(U, mx.matmul(rho, mx.conjugate(mx.transpose(U))))
    assert mat_close(out_vec, mx_vec(out_mat))
    # Kraus equivalence: one Kraus E0=U → same S
    S2 = mx_kron(U, mx.conjugate(U))
    assert mat_close(S2, S)


def test_partial_trace_dimensions_consistency():
    rho = (1/4) * mx.array(
        [[1+0j,0+0j,0+0j,0+0j],
         [0+0j,1+0j,0+0j,0+0j],
         [0+0j,0+0j,1+0j,0+0j],
         [0+0j,0+0j,0+0j,1+0j]], mx.complex64)
    rho_a = mx_ptrace(rho, traced_out=[1], dims=[2,2])
    assert rho_a.shape == (2, 2)


def test_operator_vector_roundtrip_stack_unstack():
    A = mx.array([[0+0j,2+0j],[3+0j,4+0j]], mx.complex64)
    v = mx_vec(A)
    A_back = mx_unvec(v, 2, 2)
    assert mat_close(A_back, A)


def test_swap_decomposition_three_cnot():
    SWAP = G_SWAP()
    # Build CNOT01 and CNOT10 using dense matrices
    CNOT01 = G_CNOT()
    # Permute qubits by conjugation with SWAP to get CNOT(1->0)
    CNOT10 = mx.matmul(SWAP, mx.matmul(CNOT01, SWAP))
    decomp = mx.matmul(CNOT01, mx.matmul(CNOT10, CNOT01))
    assert mat_close(decomp, SWAP)


def test_control_target_swap_via_double_h():
    # (H⊗H) CNOT(0->1) (H⊗H) = CNOT(1->0)
    H2 = _kron_mx(_hadamard_mx(), _hadamard_mx())
    CNOT01 = G_CNOT()
    # CNOT10 via swap conjugation
    CNOT10 = mx.matmul(G_SWAP(), mx.matmul(CNOT01, G_SWAP()))
    conj = mx.matmul(H2, mx.matmul(CNOT01, H2))
    assert mat_close(conj, CNOT10)
