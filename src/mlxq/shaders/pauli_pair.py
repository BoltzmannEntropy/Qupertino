"""Fused XX / YY Trotter layers via basis conjugation.

All XX bond terms mutually commute (a shared X commutes with itself), so
exp(-i*theta*sum X_a X_b) diagonalizes globally: conjugating by H on every
qubit maps it to the ZZ layer, which zz.py already applies in ONE pass.
YY uses the Y-basis rotation V = S*H (V Z V^dagger = Y), decomposed into
diagonal S/Sdag passes around the same Walsh-Hadamard layers:

    XX layer = H^(x)n . ZZ-layer . H^(x)n
    YY layer = (S^(x)n H^(x)n) . ZZ-layer . (H^(x)n Sdag^(x)n)

H^(x)n runs as radix-4 Walsh passes (4 qubits per pass, codex round 3);
S^(x)n is one popcount-LUT diagonal pass. NOTE: XX and YY terms on
overlapping bonds do NOT commute with each other (X_j Y_j = i Z_j), so only
same-family runs may be fused; the device detector respects gate order
across families.
"""
from __future__ import annotations

import mlx.core as mx

from .single_qubit import hadamard_layer_all, s_phase_layer
from .zz import zz_chain_layer

__all__ = ["xx_layer", "yy_layer"]


def xx_layer(state: mx.array, theta: float, n: int, bonds) -> mx.array:
    """exp(-i*theta*sum XX) over `bonds`: H-layer, ZZ LUT pass, H-layer."""
    state = hadamard_layer_all(state, n)
    state = zz_chain_layer(state, theta, n, bonds)
    return hadamard_layer_all(state, n)


def yy_layer(state: mx.array, theta: float, n: int, bonds) -> mx.array:
    """exp(-i*theta*sum YY) over `bonds` with V = S*H per qubit:
    V^dag = H*Sdag applies Sdag first, then H; V applies H first, then S."""
    state = s_phase_layer(state, n, dagger=True)
    state = hadamard_layer_all(state, n)
    state = zz_chain_layer(state, theta, n, bonds)
    state = hadamard_layer_all(state, n)
    return s_phase_layer(state, n, dagger=False)
