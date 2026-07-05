"""Fused ZZ Trotter layers: exp(-i*theta*sum Z_a Z_b) in one Metal pass.

Chain fast path (bonds are all adjacent qubit pairs): one XOR + popcount per
amplitude, phase from a tiny lookup table indexed by the mismatch count — no
per-bond loop and no trig in the kernel. LUT trick follows a Codex CLI kernel
review (2026-07-04).
"""
from __future__ import annotations

import math

import mlx.core as mx

__all__ = ["zz_chain_layer", "zz_weighted_layer"]

_ZZ_CHAIN_SRC = """
    uint i = thread_position_in_grid.x;
    if (i >= n_state) return;
    uint mismatches = metal::popcount((i ^ (i >> 1)) & chain_mask);
    complex64_t p = phase_lut[mismatches];
    complex64_t a = state[i];
    out[i] = complex64_t(a.real * p.real - a.imag * p.imag,
                         a.real * p.imag + a.imag * p.real);
"""

# Arbitrary-bond fallback: per-bond two-bit masks, mismatch count, phase LUT.
_ZZ_GENERIC_SRC = """
    uint i = thread_position_in_grid.x;
    if (i >= n_state) return;
    uint mismatches = 0u;
    for (uint b = 0; b < n_bonds; ++b) {
        mismatches += metal::popcount(i & bond_masks[b]) & 1u;
    }
    complex64_t p = phase_lut[mismatches];
    complex64_t a = state[i];
    out[i] = complex64_t(a.real * p.real - a.imag * p.imag,
                         a.real * p.imag + a.imag * p.real);
"""

# Weighted variant: per-bond couplings differ (long-range Ising J/d^alpha,
# arbitrary weighted graphs). Bonds are GROUPED by equal angle on the host;
# each group contributes one LUT phase indexed by its mismatch count, and the
# kernel multiplies one unit phase PER GROUP (~24 distance groups for
# long-range Ising at 25q) instead of per bond. Codex round-4 flagged the
# all-distinct case (n_groups == n_bonds) as a possible float32 drift;
# measured directly it stays ~5e-7 vs the ideal even at 300 distinct-angle
# bonds, since the double-precision LUT plus GPU FMA keep the unit-phase
# product well under the 5e-6 parity budget (regression:
# test_metal_zz_weighted_all_distinct_angles). LUTs are built in double
# precision.
_ZZ_WEIGHTED_SRC = """
    uint i = thread_position_in_grid.x;
    if (i >= n_state) return;
    float pr = 1.0f, pim = 0.0f;
    uint off = 0u;
    for (uint g = 0; g < n_groups; ++g) {
        uint k = group_size[g];
        uint mism = 0u;
        for (uint b = 0; b < k; ++b) {
            mism += metal::popcount(i & bond_masks[off + b]) & 1u;
        }
        complex64_t p = phase_lut[lut_start[g] + mism];
        float nr = pr * p.real - pim * p.imag;
        pim = pr * p.imag + pim * p.real;
        pr = nr;
        off += k;
    }
    complex64_t a = state[i];
    out[i] = complex64_t(a.real * pr - a.imag * pim,
                         a.real * pim + a.imag * pr);
"""

_zz_chain_kernel = None
_zz_generic_kernel = None
_zz_weighted_kernel = None


def _phase_lut(theta: float, n_bonds: int) -> mx.array:
    """phase_lut[m] = exp(-i*theta*(n_bonds - 2m)) for mismatch count m."""
    vals = []
    for m in range(n_bonds + 1):
        ang = -theta * (n_bonds - 2 * m)
        vals.append(complex(math.cos(ang), math.sin(ang)))
    return mx.array(vals, dtype=mx.complex64)


def _is_full_chain(bonds, n: int) -> bool:
    return sorted(bonds) == [(i, i + 1) for i in range(n - 1)]


def _group_by_theta(bonds, thetas):
    """Order-free grouping (all terms commute): list of (theta, [bonds])."""
    groups = {}
    for bond, t in zip(bonds, thetas):
        groups.setdefault(t, []).append(bond)
    return list(groups.items())


def zz_weighted_layer(state: mx.array, n: int, bonds, thetas) -> mx.array:
    """One-pass ZZ layer with a DIFFERENT angle per bond (grouped LUTs)."""
    global _zz_weighted_kernel
    if len(bonds) != len(thetas):
        raise ValueError("zz_weighted_layer: bonds/thetas length mismatch")
    for a, b in bonds:
        if a == b or not (0 <= a < n and 0 <= b < n):
            raise ValueError(f"zz_weighted_layer: invalid bond ({a}, {b})")
    if _zz_weighted_kernel is None:
        _zz_weighted_kernel = mx.fast.metal_kernel(
            name="mlxq_zz_weighted_layer",
            input_names=["state", "n_state", "n_groups", "group_size",
                         "lut_start", "bond_masks", "phase_lut"],
            output_names=["out"],
            source=_ZZ_WEIGHTED_SRC,
        )
    groups = _group_by_theta(bonds, thetas)
    flat_masks = []
    sizes = []
    lut_starts = []
    lut_vals = []
    for theta, gb in groups:
        sizes.append(len(gb))
        lut_starts.append(len(lut_vals))
        for a, b in gb:
            flat_masks.append((1 << (n - 1 - a)) | (1 << (n - 1 - b)))
        for m in range(len(gb) + 1):
            ang = -theta * (len(gb) - 2 * m)
            lut_vals.append(complex(math.cos(ang), math.sin(ang)))
    (out,) = _zz_weighted_kernel(
        inputs=[state, mx.array(1 << n, dtype=mx.uint32),
                mx.array(len(groups), dtype=mx.uint32),
                mx.array(sizes, dtype=mx.uint32),
                mx.array(lut_starts, dtype=mx.uint32),
                mx.array(flat_masks, dtype=mx.uint32),
                mx.array(lut_vals, dtype=mx.complex64)],
        grid=(1 << n, 1, 1),
        threadgroup=(min(256, 1 << n), 1, 1),
        output_shapes=[state.shape],
        output_dtypes=[mx.complex64],
    )
    return out


def zz_chain_layer(state: mx.array, theta: float, n: int, bonds) -> mx.array:
    """One-pass fused ZZ layer via Metal; returns the new state array.

    Chain fast path (all adjacent bonds): XOR/popcount parity + phase LUT.
    Arbitrary bonds: per-bond two-bit masks + the same LUT.
    """
    global _zz_chain_kernel, _zz_generic_kernel
    lut = _phase_lut(theta, len(bonds))
    if _is_full_chain(bonds, n):
        if _zz_chain_kernel is None:
            _zz_chain_kernel = mx.fast.metal_kernel(
                name="mlxq_zz_chain_layer",
                input_names=["state", "n_state", "chain_mask", "phase_lut"],
                output_names=["out"],
                source=_ZZ_CHAIN_SRC,
            )
        chain_mask = mx.array((1 << (n - 1)) - 1, dtype=mx.uint32)
        (out,) = _zz_chain_kernel(
            inputs=[state, mx.array(1 << n, dtype=mx.uint32), chain_mask, lut],
            grid=(1 << n, 1, 1),
            threadgroup=(min(256, 1 << n), 1, 1),
            output_shapes=[state.shape],
            output_dtypes=[mx.complex64],
        )
        return out
    if _zz_generic_kernel is None:
        _zz_generic_kernel = mx.fast.metal_kernel(
            name="mlxq_zz_generic_layer",
            input_names=["state", "n_state", "n_bonds", "bond_masks", "phase_lut"],
            output_names=["out"],
            source=_ZZ_GENERIC_SRC,
        )
    masks = mx.array([(1 << (n - 1 - a)) | (1 << (n - 1 - b))
                      for a, b in bonds], dtype=mx.uint32)
    (out,) = _zz_generic_kernel(
        inputs=[state, mx.array(1 << n, dtype=mx.uint32),
                mx.array(len(bonds), dtype=mx.uint32), masks, lut],
        grid=(1 << n, 1, 1),
        threadgroup=(min(256, 1 << n), 1, 1),
        output_shapes=[state.shape],
        output_dtypes=[mx.complex64],
    )
    return out
