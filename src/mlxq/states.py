from typing import List
import math
import random
import mlx.core as mx


def zero_state(n_qubits: int) -> mx.array:
    dim = 1 << int(n_qubits)
    data = [0j] * dim
    data[0] = 1+0j
    return mx.array(data, mx.complex64)


def computational_basis(n_qubits: int, index: int) -> mx.array:
    dim = 1 << int(n_qubits)
    data = [0j] * dim
    data[int(index)] = 1+0j
    return mx.array(data, mx.complex64)


def one_state() -> mx.array:
    return mx.array([0+0j, 1+0j], mx.complex64)


def plus_state() -> mx.array:
    s = 1.0 / math.sqrt(2.0)
    return mx.array([complex(s,0.0), complex(s,0.0)], mx.complex64)


def minus_state() -> mx.array:
    s = 1.0 / math.sqrt(2.0)
    return mx.array([complex(s,0.0), complex(-s,0.0)], mx.complex64)


def bell_state(kind: int = 0) -> mx.array:
    # |Φ+⟩ = (|00⟩+|11⟩)/√2 by default
    s = 1.0 / math.sqrt(2.0)
    if kind == 0:  # Phi+
        data = [s+0j, 0+0j, 0+0j, s+0j]
    elif kind == 1:  # Phi-
        data = [s+0j, 0+0j, 0+0j, -s+0j]
    elif kind == 2:  # Psi+
        data = [0+0j, s+0j, s+0j, 0+0j]
    else:  # Psi-
        data = [0+0j, s+0j, -s+0j, 0+0j]
    return mx.array(data, mx.complex64)


def ghz_state(n_qubits: int) -> mx.array:
    if n_qubits < 2:
        return zero_state(n_qubits)
    dim = 1 << int(n_qubits)
    s = 1.0 / math.sqrt(2.0)
    data = [0j] * dim
    data[0] = s+0j
    data[dim-1] = s+0j
    return mx.array(data, mx.complex64)


def w_state(n_qubits: int) -> mx.array:
    if n_qubits < 2:
        return zero_state(n_qubits)
    dim = 1 << int(n_qubits)
    s = 1.0 / math.sqrt(n_qubits)
    data = [0j] * dim
    for pos in range(n_qubits):
        idx = 1 << (n_qubits - 1 - pos)
        data[idx] = s+0j
    return mx.array(data, mx.complex64)


def random_state(n_qubits: int, seed: int = 42) -> mx.array:
    random.seed(seed)
    dim = 1 << int(n_qubits)
    # generate random complex amplitudes
    data = []
    for _ in range(dim):
        a = random.random()*2-1
        b = random.random()*2-1
        data.append(complex(a, b))
    st = mx.array(data, mx.complex64)
    # normalize
    norm = mx.sqrt(mx.sum(mx.abs(st) ** 2))
    mx.eval(norm)
    val = float(norm.item()) if hasattr(norm, 'item') else float(norm)
    if val > 0:
        st = st / val
    return st


def y_plus_state() -> mx.array:
    s = 1.0 / math.sqrt(2.0)
    return mx.array([complex(s, 0.0), complex(0.0, s)], mx.complex64)


def y_minus_state() -> mx.array:
    s = 1.0 / math.sqrt(2.0)
    return mx.array([complex(s, 0.0), complex(0.0, -s)], mx.complex64)


def uniform_superposition(n_qubits: int) -> mx.array:
    dim = 1 << int(n_qubits)
    amp = 1.0 / math.sqrt(dim)
    data = [complex(amp, 0.0)] * dim
    return mx.array(data, mx.complex64)


def spin_coherent(theta: float, phi: float) -> mx.array:
    # |ψ(θ,φ)⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩
    c = math.cos(theta/2.0)
    s = math.sin(theta/2.0)
    e = complex(math.cos(phi), math.sin(phi))
    return mx.array([complex(c, 0.0), e * s], mx.complex64)


def to_bloch_vector(state: mx.array):
    from .gates import X, Y, Z
    from .observables import expectation_value
    x = expectation_value(state, X())
    y = expectation_value(state, Y())
    z = expectation_value(state, Z())
    return (x, y, z)


def custom_state(amplitudes) -> mx.array:
    """Build a state vector from a Python list of complex amplitudes."""
    return mx.array([complex(a) for a in amplitudes], mx.complex64)


def state_overlap(phi: mx.array, psi: mx.array) -> complex:
    """Return ⟨phi|psi⟩."""
    v = mx.sum(mx.conjugate(phi) * psi)
    mx.eval(v)
    return complex(v.item())


def max_mixed(dim: int) -> mx.array:
    """Return maximally mixed density matrix I/dim for a single subsystem of size dim."""
    Id = [[1.0 if i == j else 0.0 for j in range(dim)] for i in range(dim)]
    return (1.0/float(dim)) * mx.array(Id, mx.complex64)
