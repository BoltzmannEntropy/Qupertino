from __future__ import annotations
from typing import List
import mlx.core as mx


def apply_kraus(rho: mx.array, K: List[mx.array]) -> mx.array:
    out = mx.zeros_like(rho)
    for E in K:
        term = mx.matmul(E, mx.matmul(rho, mx.conjugate(mx.transpose(E))))
        out = out + term
    return out


def depolarizing_kraus(p: float) -> List[mx.array]:
    from .gates import I, X, Y, Z
    a0 = (1.0 - 3.0 * p / 4.0) ** 0.5
    a = (p / 4.0) ** 0.5
    return [a0 * I(), a * X(), a * Y(), a * Z()]


def bitflip_kraus(p: float) -> List[mx.array]:
    from .gates import I, X
    a0 = (1.0 - p) ** 0.5
    a1 = p ** 0.5
    return [a0 * I(), a1 * X()]


def amplitude_damping_kraus(gamma: float) -> List[mx.array]:
    g = float(gamma)
    k0 = mx.array([[1+0j, 0+0j], [0+0j, (1.0 - g) ** 0.5 + 0j]], mx.complex64)
    k1 = mx.array([[0+0j, (g) ** 0.5 + 0j], [0+0j, 0+0j]], mx.complex64)
    return [k0, k1]


def choi_from_kraus(K: List[mx.array]) -> mx.array:
    """Choi matrix J = sum_k vec(Ek) vec(Ek)† for column-major vec."""
    from .information import operator_to_vector
    v = None
    for E in K:
        e = operator_to_vector(E)
        col = mx.reshape(e, (-1, 1))
        term = mx.matmul(col, mx.conjugate(mx.transpose(col)))
        v = term if v is None else (v + term)
    return v
