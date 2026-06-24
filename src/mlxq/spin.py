from __future__ import annotations
import math
import mlx.core as mx
from .gates import X, Y, Z, I


def Sx():
    return 0.5 * X()


def Sy():
    return 0.5 * Y()


def Sz():
    return 0.5 * Z()


def hamiltonian_from_field(omega: float, nx: float, ny: float, nz: float) -> mx.array:
    n2 = nx*nx + ny*ny + nz*nz
    if n2 <= 0:
        return mx.zeros((2,2), mx.complex64)
    # normalize
    inv = 1.0 / math.sqrt(n2)
    nx, ny, nz = nx*inv, ny*inv, nz*inv
    return float(omega) * (nx * Sx() + ny * Sy() + nz * Sz())


def unitary_time_evolution(H: mx.array, t: float) -> mx.array:
    """Compute U = exp(-i t H) for 2×2 Hermitian H via Pauli expansion.
    H = a I + b·σ / 2 → U = e^{-i t a} [cos(|b| t/2) I - i sin(|b| t/2) (b̂·σ)].
    """
    # Extract coefficients: a = tr(H)/2, and vector components via traces
    tr = mx.sum(mx.diagonal(H))
    mx.eval(tr)
    a = 0.5 * complex(tr.item())
    # b_k = tr(H σ_k)
    def tr_prod(A, B):
        T = mx.matmul(A, B)
        s = mx.sum(mx.diagonal(T)); mx.eval(s)
        return complex(s.item())
    bx = tr_prod(H, X())
    by = tr_prod(H, Y())
    bz = tr_prod(H, Z())
    bvec = (bx.real, by.real, bz.real)
    bmag = math.sqrt(bvec[0]**2 + bvec[1]**2 + bvec[2]**2) + 1e-12
    nx, ny, nz = (bvec[0]/bmag, bvec[1]/bmag, bvec[2]/bmag)
    phase = complex(math.cos(-t*a), math.sin(-t*a))
    c = math.cos(0.5 * bmag * t)
    s = math.sin(0.5 * bmag * t)
    term = nx * X() + ny * Y() + nz * Z()
    U = (c * I()) + (-1j * s) * term
    return phase * U
