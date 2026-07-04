"""Fused QFT stages: radix-4 two-stage butterflies + single-stage fallback.

Pair-thread layout: one thread per butterfly pair, so every amplitude is read
once and written once per stage. The stage's controlled-phase ladder collapses
to the closed form angle = pi * low / 2^shift_j (no per-bit loop). The radix-4
kernel fuses two consecutive stages per launch, halving global traffic; its
body is the Codex round-2 corrected derivation (round 1 was algebraically
wrong and retracted).
"""
from __future__ import annotations

import math

import mlx.core as mx

__all__ = ["qft_stage_all", "qft_stage_sub"]

# Generalized single stage: Hadamard butterfly on qubit j plus the fused
# controlled-phase ladder toward qubits j+1..m-1 of a SUBREGISTER [j..m-1]
# (m = n recovers the full-register stage). The ladder collapses to
# angle = phase_scale * V with V = (i >> lad_shift) & lad_mask, the integer
# value of the ladder bits; phase_scale = +/- pi / 2^(m-1-j) selects the
# forward or inverse transform. Inverse order is phase-first, then butterfly
# (the adjoint of the forward stage).
_QFT_STAGE_GEN_SRC = """
    uint p = thread_position_in_grid.x;
    if (p >= n_pairs) return;
    uint bit = 1u << shift_j;
    uint low = p & (bit - 1u);
    uint high = p >> shift_j;
    uint i0 = (high << (shift_j + 1u)) | low;
    uint i1 = i0 | bit;

    complex64_t a0 = state[i0];
    complex64_t a1 = state[i1];
    uint lad = (i0 >> lad_shift) & lad_mask;
    float c = 1.0f, s = 0.0f;
    if (lad != 0u) {
        float angle = phase_scale * (float)lad;
        c = metal::precise::cos(angle);
        s = metal::precise::sin(angle);
    }
    const float inv_sqrt2 = 0.70710678118654752f;
    if (inverse != 0u) {
        // a1' = e^{i*angle} * a1 first, then butterfly
        float b1r = a1.real * c - a1.imag * s;
        float b1i = a1.real * s + a1.imag * c;
        out[i0] = complex64_t((a0.real + b1r) * inv_sqrt2,
                              (a0.imag + b1i) * inv_sqrt2);
        out[i1] = complex64_t((a0.real - b1r) * inv_sqrt2,
                              (a0.imag - b1i) * inv_sqrt2);
    } else {
        float h1r = (a0.real - a1.real) * inv_sqrt2;
        float h1i = (a0.imag - a1.imag) * inv_sqrt2;
        out[i0] = complex64_t((a0.real + a1.real) * inv_sqrt2,
                              (a0.imag + a1.imag) * inv_sqrt2);
        out[i1] = complex64_t(h1r * c - h1i * s, h1r * s + h1i * c);
    }
"""

# Radix-4 QFT: two consecutive stages (bit s, then bit s-1) fused in one
# pass. One thread per quad; each amplitude is read and written once for BOTH
# stages. Derivation (validated against the pair kernel): with
# alpha = pi*low'/2^s,
#   out00 = (p0+p1)/2,            out01 = (p0-p1)/2 * e^{i2a},
#   out10 = (m0+i*m1)/2 * e^{ia}, out11 = (m0-i*m1)/2 * e^{i3a},
# where p_x = a0x+a1x, m_x = a0x-a1x.
_QFT_RADIX4_SRC = """
    uint q = thread_position_in_grid.x;
    if (q >= n_quads) return;

    uint bit_lo = 1u << (shift_s - 1u);
    uint bit_hi = 1u << shift_s;
    uint low = q & (bit_lo - 1u);
    uint high = q >> (shift_s - 1u);

    uint i00 = (high << (shift_s + 1u)) | low;
    uint i01 = i00 | bit_lo;
    uint i10 = i00 | bit_hi;
    uint i11 = i00 | bit_hi | bit_lo;

    complex64_t a00 = state[i00], a01 = state[i01];
    complex64_t a10 = state[i10], a11 = state[i11];

    float p0r = a00.real + a10.real, p0i = a00.imag + a10.imag;
    float m0r = a00.real - a10.real, m0i = a00.imag - a10.imag;
    float p1r = a01.real + a11.real, p1i = a01.imag + a11.imag;
    float m1r = a01.real - a11.real, m1i = a01.imag - a11.imag;

    float c = 1.0f, sn = 0.0f;
    if (low != 0u) {
        float alpha = phase_scale * (float)low;   // phase_scale = pi / 2^shift_s
        c = metal::precise::cos(alpha);
        sn = metal::precise::sin(alpha);
    }
    float c2 = c * c - sn * sn;
    float s2 = 2.0f * c * sn;
    float c3 = c * c2 - sn * s2;
    float s3 = c * s2 + sn * c2;

    const float h = 0.5f;
    out[i00] = complex64_t((p0r + p1r) * h, (p0i + p1i) * h);
    float u01r = (p0r - p1r) * h, u01i = (p0i - p1i) * h;
    out[i01] = complex64_t(u01r * c2 - u01i * s2, u01r * s2 + u01i * c2);
    float u10r = (m0r - m1i) * h, u10i = (m0i + m1r) * h;
    out[i10] = complex64_t(u10r * c - u10i * sn, u10r * sn + u10i * c);
    float u11r = (m0r + m1i) * h, u11i = (m0i - m1r) * h;
    out[i11] = complex64_t(u11r * c3 - u11i * s3, u11r * s3 + u11i * c3);
"""

_QFT_STAGE_SRC = """
    uint p = thread_position_in_grid.x;
    if (p >= n_pairs) return;
    uint bit = 1u << shift_j;
    uint low = p & (bit - 1u);
    uint high = p >> shift_j;
    uint i0 = (high << (shift_j + 1u)) | low;
    uint i1 = i0 | bit;

    complex64_t a0 = state[i0];
    complex64_t a1 = state[i1];
    const float inv_sqrt2 = 0.70710678118654752f;
    float h0r = (a0.real + a1.real) * inv_sqrt2;
    float h0i = (a0.imag + a1.imag) * inv_sqrt2;
    float h1r = (a0.real - a1.real) * inv_sqrt2;
    float h1i = (a0.imag - a1.imag) * inv_sqrt2;

    out[i0] = complex64_t(h0r, h0i);
    if (low != 0u) {
        float angle = phase_scale * (float)low;
        float c = metal::precise::cos(angle);
        float s = metal::precise::sin(angle);
        out[i1] = complex64_t(h1r * c - h1i * s, h1r * s + h1i * c);
    } else {
        out[i1] = complex64_t(h1r, h1i);
    }
"""

_qft_kernel = None
_qft_radix4_kernel = None
_qft_stage_gen_kernel = None


def qft_stage_sub(state: mx.array, n: int, j: int, m: int,
                  inverse: bool = False) -> mx.array:
    """One QFT stage on qubit j of subregister [j..m-1] in ONE pass:
    Hadamard butterfly plus the whole controlled-phase ladder toward qubits
    j+1..m-1 (closed form over the ladder bits). inverse=True applies the
    adjoint (phase first, then butterfly, negated angle)."""
    global _qft_stage_gen_kernel
    if not (0 <= j < m <= n):
        raise ValueError(f"qft_stage_sub: invalid j={j}, m={m}, n={n}")
    if _qft_stage_gen_kernel is None:
        _qft_stage_gen_kernel = mx.fast.metal_kernel(
            name="mlxq_qft_stage_gen",
            input_names=["state", "shift_j", "lad_shift", "lad_mask",
                         "phase_scale", "inverse", "n_pairs"],
            output_names=["out"],
            source=_QFT_STAGE_GEN_SRC,
        )
    shift_j = n - 1 - j
    width = m - 1 - j                    # ladder bit count
    lad_shift = n - m
    lad_mask = (1 << width) - 1
    scale = math.pi / float(1 << width) if width > 0 else 0.0
    if inverse:
        scale = -scale
    n_pairs = 1 << (n - 1)
    (state,) = _qft_stage_gen_kernel(
        inputs=[state, mx.array(shift_j, dtype=mx.uint32),
                mx.array(lad_shift, dtype=mx.uint32),
                mx.array(lad_mask, dtype=mx.uint32),
                mx.array(scale, dtype=mx.float32),
                mx.array(1 if inverse else 0, dtype=mx.uint32),
                mx.array(n_pairs, dtype=mx.uint32)],
        grid=(n_pairs, 1, 1),
        threadgroup=(min(256, n_pairs), 1, 1),
        output_shapes=[state.shape],
        output_dtypes=[mx.complex64],
    )
    return state


def _qft_single_stage(state: mx.array, n: int, shift_j: int) -> mx.array:
    global _qft_kernel
    if _qft_kernel is None:
        _qft_kernel = mx.fast.metal_kernel(
            name="mlxq_qft_stage_pair",
            input_names=["state", "shift_j", "phase_scale", "n_pairs"],
            output_names=["out"],
            source=_QFT_STAGE_SRC,
        )
    phase_scale = math.pi / float(1 << shift_j) if shift_j > 0 else 0.0
    n_pairs = 1 << (n - 1)
    (state,) = _qft_kernel(
        inputs=[state, mx.array(shift_j, dtype=mx.uint32),
                mx.array(phase_scale, dtype=mx.float32),
                mx.array(n_pairs, dtype=mx.uint32)],
        grid=(n_pairs, 1, 1),
        threadgroup=(min(256, n_pairs), 1, 1),
        output_shapes=[state.shape],
        output_dtypes=[mx.complex64],
    )
    return state


def qft_stage_all(state: mx.array, n: int) -> mx.array:
    """Full-register QFT via radix-4 fused stage pairs plus a final single
    stage when n is odd. Matches the gate-decomposed qft() convention (no
    final swap layer). Each radix-4 launch executes TWO stages with one read
    and one write per amplitude.
    """
    global _qft_radix4_kernel
    if _qft_radix4_kernel is None:
        _qft_radix4_kernel = mx.fast.metal_kernel(
            name="mlxq_qft_radix4",
            input_names=["state", "shift_s", "phase_scale", "n_quads"],
            output_names=["out"],
            source=_QFT_RADIX4_SRC,
        )
    shift = n - 1  # stages run from the MSB-first top bit downward
    n_quads = 1 << (n - 2) if n >= 2 else 0
    while shift >= 1:
        (state,) = _qft_radix4_kernel(
            inputs=[state, mx.array(shift, dtype=mx.uint32),
                    mx.array(math.pi / float(1 << shift), dtype=mx.float32),
                    mx.array(n_quads, dtype=mx.uint32)],
            grid=(n_quads, 1, 1),
            threadgroup=(min(256, n_quads), 1, 1),
            output_shapes=[state.shape],
            output_dtypes=[mx.complex64],
        )
        shift -= 2
    if shift == 0:
        state = _qft_single_stage(state, n, 0)
    return state
