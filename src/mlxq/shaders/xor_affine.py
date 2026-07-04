"""GF(2) affine permutation: any run of CNOT/X/SWAP gates in ONE gather pass.

X, CNOT, and SWAP act on computational-basis indices as affine maps over
GF(2): the circuit block sends |x> to |f(x)> with f(x) = Mx (+) c, where M is
an invertible bit-matrix and c a flip vector. Applying the block to a state
is therefore a pure permutation of amplitudes: out[y] = state[f^{-1}(y)].
Because each of these gates is an involution, f^{-1} is composed by replaying
the gate run in reverse order onto an identity map. The kernel computes the
source index per thread with n popcount-parities (compute-light; the pass is
bandwidth-bound like any gather).

Convention: qubit q is bit position n-1-q (MSB-first), matching sim.py.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

import mlx.core as mx

__all__ = ["compose_inverse_affine", "xor_affine_gather"]

_XOR_GATHER_SRC = """
    uint y = thread_position_in_grid.x;
    if (y >= n_state) return;
    uint src = 0u;
    for (uint k = 0; k < n_bits; ++k) {
        uint p = metal::popcount(rows[k] & y) & 1u;
        src |= p << (n_bits - 1u - k);
    }
    src ^= cvec;
    out[y] = state[src];
"""

# Pure bit-flip fast path (M = identity, codex S2 review): X-only blocks need
# no matrix rows, just src = y ^ c.
_XOR_FLIP_SRC = """
    uint y = thread_position_in_grid.x;
    if (y >= n_state) return;
    out[y] = state[y ^ cvec];
"""

_xor_gather_kernel = None
_xor_flip_kernel = None


def compose_inverse_affine(ops: Iterable[dict], n: int) -> Tuple[List[int], int]:
    """Compose a CNOT/X/SWAP run into the INVERSE affine map (rows, c).

    Returns per-qubit input-bit masks `rows` (index k = qubit k) and packed
    flip constant `c` such that f^{-1}(y) has qubit-k bit
    parity(rows[k] & y) XOR bit_k(c). Built by replaying the run in reverse
    (each gate is its own inverse) onto the identity map; a gate acting on the
    map's OUTPUT bits updates rows/flips row-wise.
    """
    rows = [1 << (n - 1 - k) for k in range(n)]
    c = 0
    for op in reversed(list(ops)):
        name = str(op.get("name", "")).upper()
        w = list(op.get("wires", []))
        if name == "X":
            c ^= 1 << (n - 1 - w[0])
        elif name in ("CNOT", "CX"):
            a, b = w
            rows[b] ^= rows[a]
            if (c >> (n - 1 - a)) & 1:
                c ^= 1 << (n - 1 - b)
        elif name == "SWAP":
            a, b = w
            rows[a], rows[b] = rows[b], rows[a]
            ca = (c >> (n - 1 - a)) & 1
            cb = (c >> (n - 1 - b)) & 1
            if ca != cb:
                c ^= (1 << (n - 1 - a)) | (1 << (n - 1 - b))
        else:
            raise ValueError(f"Not a GF(2) affine gate: {name}")
    return rows, c


def xor_affine_gather(state: mx.array, n: int, rows: List[int], c: int) -> mx.array:
    """Apply the permutation out[y] = state[M'y (+) c'] in one gather pass."""
    global _xor_gather_kernel, _xor_flip_kernel
    if all(rows[k] == 1 << (n - 1 - k) for k in range(n)):
        if _xor_flip_kernel is None:
            _xor_flip_kernel = mx.fast.metal_kernel(
                name="mlxq_xor_flip",
                input_names=["state", "n_state", "cvec"],
                output_names=["out"],
                source=_XOR_FLIP_SRC,
            )
        (out,) = _xor_flip_kernel(
            inputs=[state, mx.array(1 << n, dtype=mx.uint32),
                    mx.array(c, dtype=mx.uint32)],
            grid=(1 << n, 1, 1),
            threadgroup=(min(256, 1 << n), 1, 1),
            output_shapes=[state.shape],
            output_dtypes=[mx.complex64],
        )
        return out
    if _xor_gather_kernel is None:
        _xor_gather_kernel = mx.fast.metal_kernel(
            name="mlxq_xor_affine_gather",
            input_names=["state", "n_state", "n_bits", "rows", "cvec"],
            output_names=["out"],
            source=_XOR_GATHER_SRC,
        )
    (out,) = _xor_gather_kernel(
        inputs=[state, mx.array(1 << n, dtype=mx.uint32),
                mx.array(n, dtype=mx.uint32),
                mx.array(rows, dtype=mx.uint32),
                mx.array(c, dtype=mx.uint32)],
        grid=(1 << n, 1, 1),
        threadgroup=(min(256, 1 << n), 1, 1),
        output_shapes=[state.shape],
        output_dtypes=[mx.complex64],
    )
    return out
