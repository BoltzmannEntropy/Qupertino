"""Fused uniform diagonal pair layers: CZ / CPHASE over an arbitrary bond list.

A layer of CPHASE(theta) gates (CZ is theta = pi) over bonds B multiplies each
amplitude by exp(i*theta*m(i)) where m(i) counts the bonds whose two qubits are
BOTH 1 in basis index i. All such gates are diagonal and mutually commute, so
any run of them — chain, ring, brickwork, arbitrary graph — collapses to ONE
pass: per-bond two-bit mask test, count, phase LUT of size |B|+1. Same
machinery as the ZZ mismatch LUT (zz.py) with the parity test replaced by a
both-bits-set test.
"""
from __future__ import annotations

import math

import mlx.core as mx

__all__ = ["diag_pair_layer", "diag_weighted_layer"]

_DIAG_PAIR_SRC = """
    uint i = thread_position_in_grid.x;
    if (i >= n_state) return;
    uint count = 0u;
    for (uint b = 0; b < n_bonds; ++b) {
        uint m = bond_masks[b];
        count += ((i & m) == m) ? 1u : 0u;
    }
    complex64_t p = phase_lut[count];
    complex64_t a = state[i];
    out[i] = complex64_t(a.real * p.real - a.imag * p.imag,
                         a.real * p.imag + a.imag * p.real);
"""

# Chain/ring fast path (codex S1 review): bit p of (i & (i >> 1)) is 1 iff
# bits p and p+1 of i are both 1, so a nearest-neighbor CZ/CPHASE layer needs
# ONE popcount instead of a per-bond loop; the ring's wrap bond (n-1, 0) adds
# a two-bit test.
_DIAG_CHAIN_SRC = """
    uint i = thread_position_in_grid.x;
    if (i >= n_state) return;
    uint count = metal::popcount((i & (i >> 1)) & edge_mask);
    if (wrap_mask != 0u) {
        count += ((i & wrap_mask) == wrap_mask) ? 1u : 0u;
    }
    complex64_t p = phase_lut[count];
    complex64_t a = state[i];
    out[i] = complex64_t(a.real * p.real - a.imag * p.imag,
                         a.real * p.imag + a.imag * p.real);
"""

# Weighted variant: per-bond ANGLES differ (QPE controlled-power ladders,
# inline-iQFT phase ladders, mixed CZ/CPHASE runs). Bonds are grouped by
# equal angle on the host; each group contributes one LUT phase indexed by
# its count of both-bits-set bonds, and the kernel multiplies one unit phase
# per group. LUTs are built in double precision (QPE angles reach base*2^p
# ~ 3e6 rad where float32 trig is meaningless), and grouping bounds the
# float32 product chain (codex round 3: 300 per-bond products drift ~1.25e-5,
# over the 5e-6 parity budget).
_DIAG_WEIGHTED_SRC = """
    uint i = thread_position_in_grid.x;
    if (i >= n_state) return;
    float pr = 1.0f, pim = 0.0f;
    uint off = 0u;
    for (uint g = 0; g < n_groups; ++g) {
        uint k = group_size[g];
        uint cnt = 0u;
        for (uint b = 0; b < k; ++b) {
            uint m = bond_masks[off + b];
            cnt += ((i & m) == m) ? 1u : 0u;
        }
        complex64_t p = phase_lut[lut_start[g] + cnt];
        float nr = pr * p.real - pim * p.imag;
        pim = pr * p.imag + pim * p.real;
        pr = nr;
        off += k;
    }
    complex64_t a = state[i];
    out[i] = complex64_t(a.real * pr - a.imag * pim,
                         a.real * pim + a.imag * pr);
"""

_diag_pair_kernel = None
_diag_chain_kernel = None
_diag_weighted_kernel = None


def _phase_lut(theta: float, n_bonds: int) -> mx.array:
    """phase_lut[m] = exp(i*theta*m) for m bonds with both qubits set.
    theta = pi (CZ) uses exact +/-1 entries (no float trig drift)."""
    if abs(theta - math.pi) < 1e-12:
        vals = [complex(1.0 if m % 2 == 0 else -1.0, 0.0)
                for m in range(n_bonds + 1)]
    else:
        vals = [complex(math.cos(theta * m), math.sin(theta * m))
                for m in range(n_bonds + 1)]
    return mx.array(vals, dtype=mx.complex64)


def _chain_ring_masks(bonds, n: int):
    """(edge_mask, wrap_mask) when bonds are nearest-neighbor (optionally
    with the ring wrap bond), else None. Bond (q, q+1) maps to bit n-2-q of
    (i & (i >> 1))."""
    norm = [tuple(sorted(b)) for b in bonds]
    if len(norm) != len(set(norm)):
        return None  # duplicate bonds must count multiply; generic path
    adj = set()
    wrap = 0
    for lo, hi in norm:
        if hi - lo == 1:
            adj.add(lo)
        elif lo == 0 and hi == n - 1:
            wrap = (1 << (n - 1)) | 1
        else:
            return None
    edge_mask = 0
    for q in adj:
        edge_mask |= 1 << (n - 2 - q)
    return edge_mask, wrap


def diag_weighted_layer(state: mx.array, n: int, bonds, thetas) -> mx.array:
    """One-pass CPHASE layer with a DIFFERENT angle per bond (CZ = pi).
    Phases are precomputed on the host in double precision, so arbitrarily
    large angles (QPE's base*2^p) stay exact."""
    global _diag_weighted_kernel
    if len(bonds) != len(thetas):
        raise ValueError("diag_weighted_layer: bonds/thetas length mismatch")
    for a, b in bonds:
        if a == b or not (0 <= a < n and 0 <= b < n):
            raise ValueError(f"diag_weighted_layer: invalid bond ({a}, {b})")
    if _diag_weighted_kernel is None:
        _diag_weighted_kernel = mx.fast.metal_kernel(
            name="mlxq_diag_weighted_layer",
            input_names=["state", "n_state", "n_groups", "group_size",
                         "lut_start", "bond_masks", "phase_lut"],
            output_names=["out"],
            source=_DIAG_WEIGHTED_SRC,
        )
    groups = {}
    for bond, t in zip(bonds, thetas):
        groups.setdefault(t, []).append(bond)
    flat_masks = []
    sizes = []
    lut_starts = []
    lut_vals = []
    for theta, gb in groups.items():
        sizes.append(len(gb))
        lut_starts.append(len(lut_vals))
        for a, b in gb:
            flat_masks.append((1 << (n - 1 - a)) | (1 << (n - 1 - b)))
        for m in range(len(gb) + 1):
            lut_vals.append(complex(math.cos(theta * m), math.sin(theta * m)))
    (out,) = _diag_weighted_kernel(
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


def diag_pair_layer(state: mx.array, theta: float, n: int, bonds) -> mx.array:
    """One-pass fused CPHASE(theta) layer over `bonds` (CZ when theta = pi)."""
    global _diag_pair_kernel, _diag_chain_kernel
    for a, b in bonds:
        if a == b or not (0 <= a < n and 0 <= b < n):
            raise ValueError(f"diag_pair_layer: invalid bond ({a}, {b})")
    cr = _chain_ring_masks(bonds, n)
    if cr is not None:
        edge_mask, wrap_mask = cr
        if _diag_chain_kernel is None:
            _diag_chain_kernel = mx.fast.metal_kernel(
                name="mlxq_diag_chain_layer",
                input_names=["state", "n_state", "edge_mask", "wrap_mask",
                             "phase_lut"],
                output_names=["out"],
                source=_DIAG_CHAIN_SRC,
            )
        lut = _phase_lut(theta, len(bonds))
        (out,) = _diag_chain_kernel(
            inputs=[state, mx.array(1 << n, dtype=mx.uint32),
                    mx.array(edge_mask, dtype=mx.uint32),
                    mx.array(wrap_mask, dtype=mx.uint32), lut],
            grid=(1 << n, 1, 1),
            threadgroup=(min(256, 1 << n), 1, 1),
            output_shapes=[state.shape],
            output_dtypes=[mx.complex64],
        )
        return out
    if _diag_pair_kernel is None:
        _diag_pair_kernel = mx.fast.metal_kernel(
            name="mlxq_diag_pair_layer",
            input_names=["state", "n_state", "n_bonds", "bond_masks",
                         "phase_lut"],
            output_names=["out"],
            source=_DIAG_PAIR_SRC,
        )
    lut = _phase_lut(theta, len(bonds))
    masks = mx.array([(1 << (n - 1 - a)) | (1 << (n - 1 - b))
                      for a, b in bonds], dtype=mx.uint32)
    (out,) = _diag_pair_kernel(
        inputs=[state, mx.array(1 << n, dtype=mx.uint32),
                mx.array(len(bonds), dtype=mx.uint32), masks, lut],
        grid=(1 << n, 1, 1),
        threadgroup=(min(256, 1 << n), 1, 1),
        output_shapes=[state.shape],
        output_dtypes=[mx.complex64],
    )
    return out
