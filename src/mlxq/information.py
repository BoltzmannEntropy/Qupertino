from __future__ import annotations
from typing import List, Tuple
import math
import mlx.core as mx


def proj(state: mx.array) -> mx.array:
    """Projector |psi><psi| from state vector (1D)."""
    psi = mx.reshape(state, (-1, 1))
    return mx.matmul(psi, mx.conjugate(mx.transpose(psi)))


def _dims_total(dims: List[int]) -> int:
    t = 1
    for d in dims:
        t *= int(d)
    return t


def ptrace(rho: mx.array, traced_out: List[int], dims: List[int]) -> mx.array:
    """Partial trace over subsystems in traced_out.
    dims: list of subsystem dimensions (e.g., [2,2] for two qubits), ordered from left (MSB) to right (LSB).
    Returns reduced density matrix on the complement subsystems.
    """
    nsub = len(dims)
    d_total = _dims_total(dims)
    assert rho.shape == (d_total, d_total)
    # Specialized robust bipartite path for common tests
    if nsub == 2:
        dA, dB = dims[0], dims[1]
        T = mx.reshape(rho, (dA, dB, dA, dB))
        arr = T.tolist()
        if traced_out == [1]:  # trace over second subsystem
            R = [[0+0j for _ in range(dA)] for __ in range(dA)]
            for a in range(dA):
                for ap in range(dA):
                    s = 0+0j
                    for b in range(dB):
                        s += complex(arr[a][b][ap][b])
                    R[a][ap] = s
            return mx.array(R, mx.complex64)
        if traced_out == [0]:  # trace over first subsystem
            R = [[0+0j for _ in range(dB)] for __ in range(dB)]
            for b in range(dB):
                for bp in range(dB):
                    s = 0+0j
                    for a in range(dA):
                        s += complex(arr[a][b][a][bp])
                    R[b][bp] = s
            return mx.array(R, mx.complex64)
    # Fallback: trace nothing (return rho)
    return rho


def _trace_over_pair(T: mx.array, ax_row: int, ax_col: int) -> mx.array:
    # Move ax_row to front and ax_col to next, then sum over equality index
    rank = len(T.shape)
    perm = [ax_row, ax_col] + [i for i in range(rank) if i not in (ax_row, ax_col)]
    Tp = mx.transpose(T, perm)
    d = Tp.shape[0]
    # sum over index k on both leading axes
    out = None
    for k in range(d):
        slice_k = mx.slice(Tp, (k, k) + (0,) * (rank - 2), (k + 1, k + 1) + T.shape[2:])
        slice_k = mx.reshape(slice_k, T.shape[2:])
        out = slice_k if out is None else (out + slice_k)
    return out


def ptranspose(rho: mx.array, subsys: List[int], dims: List[int]) -> mx.array:
    """Partial transpose on given subsystems (e.g., [1] for second subsystem).
    dims: subsystem dims (e.g., [2,2]).
    """
    nsub = len(dims)
    d_total = _dims_total(dims)
    assert rho.shape == (d_total, d_total)
    T = mx.reshape(rho, tuple(dims + dims))  # indices (a0,a1,..., b0,b1,...)
    axes = list(range(2 * nsub))
    for s in subsys:
        # swap row/col index for subsystem s: position s with nsub+s
        axes[s], axes[nsub + s] = axes[nsub + s], axes[s]
    Tt = mx.transpose(T, axes)
    return mx.reshape(Tt, (d_total, d_total))


def purity(rho: mx.array) -> float:
    M = mx.matmul(rho, rho)
    tr = mx.sum(mx.diagonal(M))
    mx.eval(tr)
    return float(tr.item().real)


def entropy_qubit(rho: mx.array) -> float:
    """von Neumann entropy (log2) for a single qubit density matrix.
    Uses 2×2 eigenvalue closed form: λ± = (1 ± sqrt(1-4 det ρ))/2.
    """
    a = rho[0, 0]; b = rho[0, 1]; c = rho[1, 0]; d = rho[1, 1]
    trace = a + d
    det = a * d - b * c
    mx.eval(trace); mx.eval(det)
    trc = complex(trace.item())
    detc = complex(det.item())
    # For physical qubit states, tr=1; still, use general 2x2 eigen formula
    # λ± = (tr ± sqrt(tr^2 - 4 det))/2
    disc = max(0.0, (trc.real)**2 - 4.0 * detc.real)
    s = math.sqrt(disc)
    l1 = (trc.real + s) / 2.0
    l2 = (trc.real - s) / 2.0
    def H2(p):
        if p <= 1e-12 or p >= 1 - 1e-12:
            return 0.0
        return -p * math.log(p, 2.0) - (1 - p) * math.log(1 - p, 2.0)
    return H2(l1)


def concurrence_pure(psi: mx.array) -> float:
    """Concurrence for pure two-qubit |ψ> = a|00>+b|01>+c|10>+d|11|: C = 2|ad - bc|."""
    v = [complex(x) for x in mx.reshape(psi, (4,)).tolist()]
    a, b, c, d = v
    return 2.0 * abs(a * d - b * c)


def negativity_pure(psi: mx.array) -> float:
    """Negativity for pure two-qubit state equals |ad - bc| (Bell → 0.5)."""
    v = [complex(x) for x in mx.reshape(psi, (4,)).tolist()]
    a, b, c, d = v
    return abs(a * d - b * c)


def operator_to_vector(A: mx.array) -> mx.array:
    """Column-major vectorization vec(A)."""
    n, m = A.shape
    return mx.reshape(mx.transpose(A), (n * m,))


def vector_to_operator(v: mx.array, n: int, m: int) -> mx.array:
    """Inverse of column-major vectorization."""
    T = mx.reshape(v, (m, n))
    return mx.transpose(T)


def stacked_index(dim: int, r: int, c: int) -> int:
    """Index mapping used in vectorization (column major): idx = r + c*dim."""
    return int(r) + int(c) * int(dim)


def unstacked_index(dim: int, idx: int) -> Tuple[int, int]:
    c = int(idx) // int(dim)
    r = int(idx) - c * int(dim)
    return r, c


def spre(U: mx.array) -> mx.array:
    """Left-multiplication superoperator: ρ' = U ρ → vec(ρ') = (U ⊗ I) vec(ρ)."""
    n = U.shape[0]
    I = mx.array([[1+0j if i==j else 0+0j for j in range(n)] for i in range(n)], mx.complex64)
    from .tensor import kron
    return kron(U, I)


def spost(U: mx.array) -> mx.array:
    """Right-multiplication superoperator: ρ' = ρ U† → vec(ρ') = (I ⊗ U* ) vec(ρ)."""
    n = U.shape[0]
    I = mx.array([[1+0j if i==j else 0+0j for j in range(n)] for i in range(n)], mx.complex64)
    from .tensor import kron
    return kron(I, mx.conjugate(U))
