"""Hand-written Metal shaders for structured gate layers (opt-in).

This package is the single home for every hand-tuned Metal kernel in mlxQ.
The kernels quantify — and, when enabled, deliver — the headroom above the
pure-MLX structured dispatch path. They are OFF by default: set
MLXQ_METAL_KERNELS=1 to route supported layers through them. The pure-MLX
path is untouched when the flag is unset.

Modules:
  zz            fused exp(-i*theta*sum Z_a Z_b) layers (parity + phase LUT)
  qft           radix-4 fused QFT stages (+ single-stage fallback)
  single_qubit  fused all-qubit single-qubit layers (RX-specialized and
                generic U (x) U tensor-product passes)

Design notes, derivations, measured numbers, and codex-review verdicts for
each shader live in shaders/README.md.
"""
from __future__ import annotations

import os

from .zz import zz_chain_layer, zz_weighted_layer
from .qft import qft_stage_all, qft_stage_sub
from .single_qubit import (rx_layer_all, u2_layer_all, u2_list_layer_all,
                           hadamard_layer_all, s_phase_layer)
from .diag import diag_pair_layer, diag_weighted_layer
from .xor_affine import compose_inverse_affine, xor_affine_gather
from .pauli_pair import xx_layer, yy_layer

__all__ = [
    "metal_enabled",
    "zz_chain_layer",
    "zz_weighted_layer",
    "qft_stage_all",
    "qft_stage_sub",
    "rx_layer_all",
    "u2_layer_all",
    "u2_list_layer_all",
    "hadamard_layer_all",
    "s_phase_layer",
    "diag_pair_layer",
    "diag_weighted_layer",
    "compose_inverse_affine",
    "xor_affine_gather",
    "xx_layer",
    "yy_layer",
]


def metal_enabled() -> bool:
    return os.environ.get("MLXQ_METAL_KERNELS", "0") == "1"
