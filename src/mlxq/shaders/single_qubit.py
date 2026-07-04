"""Fused all-qubit single-qubit layers.

Single-qubit gates on distinct wires commute, so an all-qubit layer can be
applied as floor(n/2) fused two-qubit passes (tensor product U (x) U on an
adjacent bit pair) plus one single-qubit pass when n is odd. Two variants:

  rx_layer_all  RX(theta)-specialized (real cos/sin coefficients in registers)
  u2_layer_all  generic: the same arbitrary 2x2 unitary on every qubit
"""
from __future__ import annotations

import math

import mlx.core as mx

__all__ = ["rx_layer_all", "u2_layer_all", "u2_list_layer_all",
           "hadamard_layer_all", "s_phase_layer"]

# Fused RX layer on two adjacent qubits: (c_hi I - i s_hi X) tensor
# (c_lo I - i s_lo X) applied per quad.
_RX_PAIR_SRC = """
    uint q = thread_position_in_grid.x;
    if (q >= n_quads) return;

    uint bit_lo = 1u << (shift_hi - 1u);
    uint bit_hi = 1u << shift_hi;
    uint low = q & (bit_lo - 1u);
    uint high = q >> (shift_hi - 1u);

    uint i00 = (high << (shift_hi + 1u)) | low;
    uint i01 = i00 | bit_lo;
    uint i10 = i00 | bit_hi;
    uint i11 = i00 | bit_hi | bit_lo;

    complex64_t a00 = state[i00], a01 = state[i01];
    complex64_t a10 = state[i10], a11 = state[i11];

    float A = c_hi * c_lo;
    float B = c_hi * s_lo;
    float C = s_hi * c_lo;
    float D = s_hi * s_lo;

    out[i00] = complex64_t(
        A*a00.real + B*a01.imag + C*a10.imag - D*a11.real,
        A*a00.imag - B*a01.real - C*a10.real - D*a11.imag);
    out[i01] = complex64_t(
        B*a00.imag + A*a01.real - D*a10.real + C*a11.imag,
       -B*a00.real + A*a01.imag - D*a10.imag - C*a11.real);
    out[i10] = complex64_t(
        C*a00.imag - D*a01.real + A*a10.real + B*a11.imag,
       -C*a00.real - D*a01.imag + A*a10.imag - B*a11.real);
    out[i11] = complex64_t(
       -D*a00.real + C*a01.imag + B*a10.imag + A*a11.real,
       -D*a00.imag - C*a01.real - B*a10.real + A*a11.imag);
"""

_RX_SINGLE_SRC = """
    uint p = thread_position_in_grid.x;
    if (p >= n_pairs) return;
    uint bit = 1u << shift_j;
    uint low = p & (bit - 1u);
    uint high = p >> shift_j;
    uint i0 = (high << (shift_j + 1u)) | low;
    uint i1 = i0 | bit;
    complex64_t a0 = state[i0];
    complex64_t a1 = state[i1];
    // RX(theta): [[c, -i s], [-i s, c]]
    out[i0] = complex64_t(cth*a0.real + sth*a1.imag, cth*a0.imag - sth*a1.real);
    out[i1] = complex64_t(sth*a0.imag + cth*a1.real, -sth*a0.real + cth*a1.imag);
"""

# Generic uniform single-qubit layer: the same 2x2 unitary applied to two
# adjacent qubits per pass via the tensor product (U x U). Covers H, X, RY,
# RZ, arbitrary U2 layers; u holds [u00, u01, u10, u11] as complex64.
_U2_PAIR_SRC = """
    uint q = thread_position_in_grid.x;
    if (q >= n_quads) return;

    uint bit_lo = 1u << (shift_hi - 1u);
    uint bit_hi = 1u << shift_hi;
    uint low = q & (bit_lo - 1u);
    uint high = q >> (shift_hi - 1u);

    uint i00 = (high << (shift_hi + 1u)) | low;
    uint i01 = i00 | bit_lo;
    uint i10 = i00 | bit_hi;
    uint i11 = i00 | bit_hi | bit_lo;

    complex64_t a[4] = {state[i00], state[i01], state[i10], state[i11]};
    uint idx[4] = {i00, i01, i10, i11};
    for (uint r = 0; r < 4u; ++r) {
        uint rh = r >> 1, rl = r & 1u;
        float accr = 0.0f, acci = 0.0f;
        for (uint c2 = 0; c2 < 4u; ++c2) {
            uint ch = c2 >> 1, cl = c2 & 1u;
            complex64_t uh = u[rh * 2u + ch];
            complex64_t ul = u[rl * 2u + cl];
            float wr = uh.real * ul.real - uh.imag * ul.imag;
            float wi = uh.real * ul.imag + uh.imag * ul.real;
            accr += wr * a[c2].real - wi * a[c2].imag;
            acci += wr * a[c2].imag + wi * a[c2].real;
        }
        out[idx[r]] = complex64_t(accr, acci);
    }
"""

_U2_SINGLE_SRC = """
    uint p = thread_position_in_grid.x;
    if (p >= n_pairs) return;
    uint bit = 1u << shift_j;
    uint low = p & (bit - 1u);
    uint high = p >> shift_j;
    uint i0 = (high << (shift_j + 1u)) | low;
    uint i1 = i0 | bit;
    complex64_t a0 = state[i0], a1 = state[i1];
    complex64_t u00 = u[0], u01 = u[1], u10 = u[2], u11 = u[3];
    out[i0] = complex64_t(
        u00.real*a0.real - u00.imag*a0.imag + u01.real*a1.real - u01.imag*a1.imag,
        u00.real*a0.imag + u00.imag*a0.real + u01.real*a1.imag + u01.imag*a1.real);
    out[i1] = complex64_t(
        u10.real*a0.real - u10.imag*a0.imag + u11.real*a1.real - u11.imag*a1.imag,
        u10.real*a0.imag + u10.imag*a0.real + u11.real*a1.imag + u11.imag*a1.real);
"""

# Per-qubit-varying variant: DIFFERENT 2x2 unitaries on the hi/lo qubit of
# each pair pass (ua acts on the hi bit, ub on the lo bit).
_U2L_PAIR_SRC = """
    uint q = thread_position_in_grid.x;
    if (q >= n_quads) return;

    uint bit_lo = 1u << (shift_hi - 1u);
    uint bit_hi = 1u << shift_hi;
    uint low = q & (bit_lo - 1u);
    uint high = q >> (shift_hi - 1u);

    uint i00 = (high << (shift_hi + 1u)) | low;
    uint i01 = i00 | bit_lo;
    uint i10 = i00 | bit_hi;
    uint i11 = i00 | bit_hi | bit_lo;

    complex64_t a[4] = {state[i00], state[i01], state[i10], state[i11]};
    uint idx[4] = {i00, i01, i10, i11};
    for (uint r = 0; r < 4u; ++r) {
        uint rh = r >> 1, rl = r & 1u;
        float accr = 0.0f, acci = 0.0f;
        for (uint c2 = 0; c2 < 4u; ++c2) {
            uint ch = c2 >> 1, cl = c2 & 1u;
            complex64_t uh = ua[rh * 2u + ch];
            complex64_t ul = ub[rl * 2u + cl];
            float wr = uh.real * ul.real - uh.imag * ul.imag;
            float wi = uh.real * ul.imag + uh.imag * ul.real;
            accr += wr * a[c2].real - wi * a[c2].imag;
            acci += wr * a[c2].imag + wi * a[c2].real;
        }
        out[idx[r]] = complex64_t(accr, acci);
    }
"""

# Radix-4 Walsh-Hadamard: H on FOUR adjacent qubits per pass, one thread per
# 16-tuple; out[r] = (1/4) * sum_c (-1)^popcount(r&c) a[c]. Halves the
# full-state passes of an H layer vs the pair kernel (codex S5 review
# suggestion). Signs are add/sub only — no complex multiplies.
_WALSH4_SRC = """
    uint q = thread_position_in_grid.x;
    if (q >= n_hexads) return;
    uint bit_low = 1u << (shift_hi - 3u);
    uint low = q & (bit_low - 1u);
    uint high = q >> (shift_hi - 3u);
    uint base = (high << (shift_hi + 1u)) | low;

    complex64_t a[16];
    uint idx[16];
    for (uint c = 0; c < 16u; ++c) {
        idx[c] = base | (c << (shift_hi - 3u));
        a[c] = state[idx[c]];
    }
    const float s = 0.25f;
    for (uint r = 0; r < 16u; ++r) {
        float accr = 0.0f, acci = 0.0f;
        for (uint c = 0; c < 16u; ++c) {
            if (metal::popcount(r & c) & 1u) {
                accr -= a[c].real; acci -= a[c].imag;
            } else {
                accr += a[c].real; acci += a[c].imag;
            }
        }
        out[idx[r]] = complex64_t(accr * s, acci * s);
    }
"""

# S^(x)n (dagger optional) is diagonal with phase i^popcount(i): a 4-entry
# LUT indexed by popcount mod 4. Used to build the Y-basis rotation
# V^(x)n = (S H)^(x)n around the Walsh-Hadamard kernel.
_PHASE_POPCOUNT_SRC = """
    uint i = thread_position_in_grid.x;
    if (i >= n_state) return;
    complex64_t p = lut4[metal::popcount(i) & 3u];
    complex64_t a = state[i];
    out[i] = complex64_t(a.real * p.real - a.imag * p.imag,
                         a.real * p.imag + a.imag * p.real);
"""

_rx_pair_kernel = None
_rx_single_kernel = None
_u2_pair_kernel = None
_u2_single_kernel = None
_u2l_pair_kernel = None
_walsh4_kernel = None
_phase_popcount_kernel = None

_H_FLAT_C64 = None


def _h_flat():
    global _H_FLAT_C64
    if _H_FLAT_C64 is None:
        s = 1.0 / math.sqrt(2.0)
        _H_FLAT_C64 = mx.array([s, s, s, -s], dtype=mx.complex64)
    return _H_FLAT_C64


def hadamard_layer_all(state: mx.array, n: int) -> mx.array:
    """H on every qubit: radix-4 Walsh passes (4 qubits each) plus a pair /
    single tail for n mod 4 leftover qubits."""
    global _walsh4_kernel, _u2_pair_kernel, _u2_single_kernel
    if _walsh4_kernel is None:
        _walsh4_kernel = mx.fast.metal_kernel(
            name="mlxq_walsh4",
            input_names=["state", "shift_hi", "n_hexads"],
            output_names=["out"],
            source=_WALSH4_SRC,
        )
    n_hexads = 1 << (n - 4) if n >= 4 else 0
    shift = n - 1
    while shift >= 3:
        (state,) = _walsh4_kernel(
            inputs=[state, mx.array(shift, dtype=mx.uint32),
                    mx.array(n_hexads, dtype=mx.uint32)],
            grid=(n_hexads, 1, 1),
            threadgroup=(min(256, n_hexads), 1, 1),
            output_shapes=[state.shape],
            output_dtypes=[mx.complex64],
        )
        shift -= 4
    # tail: 0-3 remaining qubits via the pair/single H kernels
    h = _h_flat()
    if _u2_pair_kernel is None:
        _u2_pair_kernel = mx.fast.metal_kernel(
            name="mlxq_u2_pair",
            input_names=["state", "shift_hi", "u", "n_quads"],
            output_names=["out"],
            source=_U2_PAIR_SRC,
        )
    n_quads = 1 << (n - 2) if n >= 2 else 0
    while shift >= 1:
        (state,) = _u2_pair_kernel(
            inputs=[state, mx.array(shift, dtype=mx.uint32), h,
                    mx.array(n_quads, dtype=mx.uint32)],
            grid=(n_quads, 1, 1),
            threadgroup=(min(256, n_quads), 1, 1),
            output_shapes=[state.shape],
            output_dtypes=[mx.complex64],
        )
        shift -= 2
    if shift == 0:
        if _u2_single_kernel is None:
            _u2_single_kernel = mx.fast.metal_kernel(
                name="mlxq_u2_single",
                input_names=["state", "shift_j", "u", "n_pairs"],
                output_names=["out"],
                source=_U2_SINGLE_SRC,
            )
        n_pairs = 1 << (n - 1)
        (state,) = _u2_single_kernel(
            inputs=[state, mx.array(0, dtype=mx.uint32), h,
                    mx.array(n_pairs, dtype=mx.uint32)],
            grid=(n_pairs, 1, 1),
            threadgroup=(min(256, n_pairs), 1, 1),
            output_shapes=[state.shape],
            output_dtypes=[mx.complex64],
        )
    return state


def s_phase_layer(state: mx.array, n: int, dagger: bool = False) -> mx.array:
    """S (or S-dagger) on every qubit: one diagonal pass, phase
    (+/-i)^popcount(index)."""
    global _phase_popcount_kernel
    if _phase_popcount_kernel is None:
        _phase_popcount_kernel = mx.fast.metal_kernel(
            name="mlxq_phase_popcount",
            input_names=["state", "n_state", "lut4"],
            output_names=["out"],
            source=_PHASE_POPCOUNT_SRC,
        )
    if dagger:
        lut = mx.array([1, -1j, -1, 1j], dtype=mx.complex64)
    else:
        lut = mx.array([1, 1j, -1, -1j], dtype=mx.complex64)
    (out,) = _phase_popcount_kernel(
        inputs=[state, mx.array(1 << n, dtype=mx.uint32), lut],
        grid=(1 << n, 1, 1),
        threadgroup=(min(256, 1 << n), 1, 1),
        output_shapes=[state.shape],
        output_dtypes=[mx.complex64],
    )
    return out


def u2_layer_all(state: mx.array, n: int, u2x2: mx.array) -> mx.array:
    """Apply the SAME 2x2 unitary to every qubit: floor(n/2) fused pair
    passes plus one single-qubit pass when n is odd. `u2x2` is a flat
    complex64 array [u00, u01, u10, u11]."""
    global _u2_pair_kernel, _u2_single_kernel
    if _u2_pair_kernel is None:
        _u2_pair_kernel = mx.fast.metal_kernel(
            name="mlxq_u2_pair",
            input_names=["state", "shift_hi", "u", "n_quads"],
            output_names=["out"],
            source=_U2_PAIR_SRC,
        )
    n_quads = 1 << (n - 2) if n >= 2 else 0
    shift = n - 1
    while shift >= 1:
        (state,) = _u2_pair_kernel(
            inputs=[state, mx.array(shift, dtype=mx.uint32), u2x2,
                    mx.array(n_quads, dtype=mx.uint32)],
            grid=(n_quads, 1, 1),
            threadgroup=(min(256, n_quads), 1, 1),
            output_shapes=[state.shape],
            output_dtypes=[mx.complex64],
        )
        shift -= 2
    if shift == 0:
        if _u2_single_kernel is None:
            _u2_single_kernel = mx.fast.metal_kernel(
                name="mlxq_u2_single",
                input_names=["state", "shift_j", "u", "n_pairs"],
                output_names=["out"],
                source=_U2_SINGLE_SRC,
            )
        n_pairs = 1 << (n - 1)
        (state,) = _u2_single_kernel(
            inputs=[state, mx.array(0, dtype=mx.uint32), u2x2,
                    mx.array(n_pairs, dtype=mx.uint32)],
            grid=(n_pairs, 1, 1),
            threadgroup=(min(256, n_pairs), 1, 1),
            output_shapes=[state.shape],
            output_dtypes=[mx.complex64],
        )
    return state


def u2_list_layer_all(state: mx.array, n: int, mats: mx.array,
                      active=None) -> mx.array:
    """Apply a DIFFERENT 2x2 unitary to every qubit in floor(n/2) fused pair
    passes (+ one single pass for odd n). `mats` is an (n, 4) complex64 array;
    row q holds [u00, u01, u10, u11] for qubit q (identity rows are fine).
    `active`, when given, lists qubits with non-identity matrices; pair
    passes where both qubits are identity are skipped entirely."""
    global _u2l_pair_kernel, _u2_single_kernel
    if _u2l_pair_kernel is None:
        _u2l_pair_kernel = mx.fast.metal_kernel(
            name="mlxq_u2_list_pair",
            input_names=["state", "shift_hi", "ua", "ub", "n_quads"],
            output_names=["out"],
            source=_U2L_PAIR_SRC,
        )
    act = set(active) if active is not None else None
    n_quads = 1 << (n - 2) if n >= 2 else 0
    shift = n - 1
    while shift >= 1:
        q_hi = n - 1 - shift          # qubit on the high bit of this pass
        q_lo = q_hi + 1               # adjacent qubit on the low bit
        if act is not None and q_hi not in act and q_lo not in act:
            shift -= 2
            continue
        (state,) = _u2l_pair_kernel(
            inputs=[state, mx.array(shift, dtype=mx.uint32),
                    mats[q_hi], mats[q_lo],
                    mx.array(n_quads, dtype=mx.uint32)],
            grid=(n_quads, 1, 1),
            threadgroup=(min(256, n_quads), 1, 1),
            output_shapes=[state.shape],
            output_dtypes=[mx.complex64],
        )
        shift -= 2
    if shift == 0 and (act is None or (n - 1) in act):
        if _u2_single_kernel is None:
            _u2_single_kernel = mx.fast.metal_kernel(
                name="mlxq_u2_single",
                input_names=["state", "shift_j", "u", "n_pairs"],
                output_names=["out"],
                source=_U2_SINGLE_SRC,
            )
        n_pairs = 1 << (n - 1)
        (state,) = _u2_single_kernel(
            inputs=[state, mx.array(0, dtype=mx.uint32), mats[n - 1],
                    mx.array(n_pairs, dtype=mx.uint32)],
            grid=(n_pairs, 1, 1),
            threadgroup=(min(256, n_pairs), 1, 1),
            output_shapes=[state.shape],
            output_dtypes=[mx.complex64],
        )
    return state


def rx_layer_all(state: mx.array, n: int, theta: float) -> mx.array:
    """RX(theta) on every qubit: floor(n/2) fused two-qubit passes plus one
    single-qubit pass when n is odd. RX gates on distinct qubits commute, so
    ordering is free."""
    global _rx_pair_kernel, _rx_single_kernel
    if _rx_pair_kernel is None:
        _rx_pair_kernel = mx.fast.metal_kernel(
            name="mlxq_rx_pair",
            input_names=["state", "shift_hi", "c_hi", "s_hi", "c_lo", "s_lo",
                         "n_quads"],
            output_names=["out"],
            source=_RX_PAIR_SRC,
        )
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    n_quads = 1 << (n - 2) if n >= 2 else 0
    shift = n - 1
    while shift >= 1:
        (state,) = _rx_pair_kernel(
            inputs=[state, mx.array(shift, dtype=mx.uint32),
                    mx.array(c, dtype=mx.float32), mx.array(s, dtype=mx.float32),
                    mx.array(c, dtype=mx.float32), mx.array(s, dtype=mx.float32),
                    mx.array(n_quads, dtype=mx.uint32)],
            grid=(n_quads, 1, 1),
            threadgroup=(min(256, n_quads), 1, 1),
            output_shapes=[state.shape],
            output_dtypes=[mx.complex64],
        )
        shift -= 2
    if shift == 0:
        if _rx_single_kernel is None:
            _rx_single_kernel = mx.fast.metal_kernel(
                name="mlxq_rx_single",
                input_names=["state", "shift_j", "cth", "sth", "n_pairs"],
                output_names=["out"],
                source=_RX_SINGLE_SRC,
            )
        n_pairs = 1 << (n - 1)
        (state,) = _rx_single_kernel(
            inputs=[state, mx.array(0, dtype=mx.uint32),
                    mx.array(c, dtype=mx.float32), mx.array(s, dtype=mx.float32),
                    mx.array(n_pairs, dtype=mx.uint32)],
            grid=(n_pairs, 1, 1),
            threadgroup=(min(256, n_pairs), 1, 1),
            output_shapes=[state.shape],
            output_dtypes=[mx.complex64],
        )
    return state
