from typing import List
import math
import mlx.core as mx
from .gates import X as _X, Y as _Y, Z as _Z, I as _I


def probabilities(state: mx.array) -> List[float]:
    amp2 = mx.abs(state) ** 2
    mx.eval(amp2)
    return [float(v) for v in amp2.tolist()]


def expectation_value(state: mx.array, observable: mx.array) -> float:
    """Return ⟨ψ|A|ψ⟩ (real part), handling 1D state properly for MLX matmul."""
    n = state.shape[0]
    ket = mx.reshape(state, (n, 1))
    bra = mx.reshape(mx.conjugate(state), (1, n))
    tmp = mx.matmul(observable, ket)
    val = mx.matmul(bra, tmp)  # shape (1,1)
    mx.eval(val)
    return float(val[0, 0].item().real)


def pauli_x():
    return _X()


def pauli_y():
    return _Y()


def pauli_z():
    return _Z()


def identity(n_qubits: int):
    return _I()  # single-qubit; for n-qubit, caller can kron if needed


def is_unitary(U: mx.array, tol: float = 1e-5) -> bool:
    UdagU = mx.matmul(mx.conjugate(mx.transpose(U)), U)
    n = U.shape[0]
    # Build identity of matching size without using mx.eye (GPU-friendly)
    Id = mx.array([[1+0j if i==j else 0+0j for j in range(n)] for i in range(n)], mx.complex64)
    D = mx.abs(UdagU - Id)
    mx.eval(D)
    return float(mx.max(D).item()) < tol


def is_hermitian(H: mx.array, tol: float = 1e-8) -> bool:
    """Check Hermiticity: H† == H within tolerance."""
    Hd = mx.conjugate(mx.transpose(H))
    D = mx.abs(Hd - H)
    mx.eval(D)
    return float(mx.max(D).item()) < tol


def commutator(A: mx.array, B: mx.array) -> mx.array:
    """Compute [A, B] = AB - BA."""
    return mx.matmul(A, B) - mx.matmul(B, A)


def pauli_decomposition_2x2(M: mx.array):
    """Return coefficients c = [c0,cx,cy,cz] such that M = c0 I + cx X + cy Y + cz Z.
    For general 2x2 complex matrix; exact for Hermitian.
    c_k = 1/2 Tr(M P_k) for P_k in {I, X, Y, Z}.
    """
    I, X, Y, Z = _I(), _X(), _Y(), _Z()
    def tr(A):
        return mx.sum(mx.diagonal(A))
    c0 = 0.5 * tr(M).item()
    cx = 0.5 * tr(mx.matmul(M, X)).item()
    cy = 0.5 * tr(mx.matmul(M, Y)).item()
    cz = 0.5 * tr(mx.matmul(M, Z)).item()
    return [complex(c0), complex(cx), complex(cy), complex(cz)]


def exp_i_pauli(P: mx.array, theta: float) -> mx.array:
    """Compute e^{i θ P} for 2x2 Pauli matrix P: cos θ I + i sin θ P."""
    I = _I()
    c = math.cos(theta)
    s = math.sin(theta)
    return c * I + (1j * s) * P


def pauli_strings_commute_words(word_a: str, word_b: str) -> bool:
    """Check commutation of two Pauli strings given as words over {I,X,Y,Z}.
    They commute iff the number of sites where local Paulis anticommute is even.
    """
    a = word_a.upper(); b = word_b.upper()
    n = min(len(a), len(b))
    def anticomm(p: str, q: str) -> bool:
        if p == 'I' or q == 'I' or p == q:
            return False
        pairs = {('X','Y'),('Y','X'),('X','Z'),('Z','X'),('Y','Z'),('Z','Y')}
        return (p,q) in pairs
    k = 0
    for i in range(n):
        if anticomm(a[i], b[i]):
            k += 1
    return (k % 2) == 0
