"""
Experimental MPO helpers for MPS TEBD.

These stubs provide minimal constructors for two-site MPO factors for XX/YY/ZZ
terms. They currently return full 4x4 unitaries to be used directly with
MPSState.apply_two[_sweep]. In future, replace with true rank-3/4 MPO slices
to reduce factorization overhead when chaining many steps.

Public:
  - mpo_xx(theta) -> 4x4 unitary (exp(-i theta X⊗X))
  - mpo_yy(theta) -> 4x4 unitary (exp(-i theta Y⊗Y))
  - mpo_zz(theta) -> 4x4 unitary (exp(-i theta Z⊗Z))
"""
from __future__ import annotations

import math
import mlx.core as mx
from .gates import X, Y
from .tensor import kron


def _i() -> mx.array:
    return mx.array([[1+0j,0+0j],[0+0j,1+0j]], mx.complex64)


def mpo_xx(theta: float) -> mx.array:
    X2 = kron(X(), X())
    I4 = kron(_i(), _i())
    c = math.cos(theta)
    s = math.sin(theta)
    return c * I4 + (-1j * s) * X2


def mpo_yy(theta: float) -> mx.array:
    Y2 = kron(Y(), Y())
    I4 = kron(_i(), _i())
    c = math.cos(theta)
    s = math.sin(theta)
    return c * I4 + (-1j * s) * Y2


def mpo_zz(theta: float) -> mx.array:
    e = complex(math.cos(theta), math.sin(theta))
    em = complex(math.cos(-theta), math.sin(-theta))
    return mx.array([[em,0+0j,0+0j,0+0j],[0+0j,e,0+0j,0+0j],[0+0j,0+0j,e,0+0j],[0+0j,0+0j,0+0j,em]], mx.complex64)

