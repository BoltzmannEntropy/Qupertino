#!/usr/bin/env python3
"""MLX array ops vs hand-written Metal shader: fused TFIM ZZ Trotter layer.

Three implementations of exp(-i*theta*sum_i Z_i Z_{i+1}) applied to a 2^n
state, compared for correctness and speed at 25 qubits:

  A. per-gate    — (n-1) sequential broadcast diagonal multiplies (the
                   structured-dispatch path without layer fusion)
  B. mlx-fused   — one cached phase-vector multiply (apply_zz_layer); the
                   vector build costs a few integer/trig passes ONCE per
                   (theta, bonds) and is amortized across Trotter steps
  C. metal-fused — hand-written Metal kernel via mx.fast.metal_kernel that
                   computes the chain parity and phase per amplitude
                   in-kernel: one pass, no cache, no phase vector in memory
"""
from __future__ import annotations

import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from mlxq.sim import StateVectorSimulator  # noqa: E402

N = 25
THETA = -0.05  # matches TFIM dt*J at r=20, t=1, J=1


def random_state(n: int, seed: int = 3) -> mx.array:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(1 << n) + 1j * rng.standard_normal(1 << n)
    v /= np.linalg.norm(v)
    s = mx.array(v.astype(np.complex64))
    mx.eval(s)
    return s


# --- A. per-gate ------------------------------------------------------------

def per_gate(sim: StateVectorSimulator, theta: float):
    for i in range(sim.n - 1):
        sim.apply_zz_phase(theta, i, i + 1)


# --- C. hand-written Metal kernel -------------------------------------------

_KERNEL_SOURCE = """
    uint i = thread_position_in_grid.x;
    uint idx = i;
    // chain parity: number of disagreeing adjacent bit pairs among (n-1) bonds
    uint x = idx ^ (idx >> 1);              // bit k set <=> bits k,k+1 of idx differ
    uint mask = (1u << (n - 1)) - 1u;       // keep the n-1 adjacent pairs
    uint n_disagree = popcount(x & mask);
    // sum of s_ab = (n-1) - 2*n_disagree; total angle = -theta * sum
    float angle = -theta * (float)((int)(n - 1) - 2 * (int)n_disagree);
    float c = metal::precise::cos(angle);
    float s = metal::precise::sin(angle);
    complex64_t amp = state[i];
    out[i] = complex64_t(amp.real * c - amp.imag * s,
                         amp.real * s + amp.imag * c);
"""

_kernel = None


def metal_fused(state: mx.array, theta: float, n: int) -> mx.array:
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="zz_chain_layer",
            input_names=["state", "theta", "n"],
            output_names=["out"],
            source=_KERNEL_SOURCE,
        )
    (out,) = _kernel(
        inputs=[state, mx.array(theta, dtype=mx.float32), mx.array(n, dtype=mx.uint32)],
        template=[],
        grid=(1 << n, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[state.shape],
        output_dtypes=[mx.complex64],
    )
    return out


def bench(label, fn, repeats=10):
    fn()  # warmup
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    print(f"{label:34s} mean {statistics.fmean(ts):8.2f} ms  "
          f"min {min(ts):8.2f}  stdev {statistics.stdev(ts):6.2f}")
    return statistics.fmean(ts)


def main() -> int:
    bonds = [(i, i + 1) for i in range(N - 1)]
    base = random_state(N)

    # ---- correctness: all three agree ----
    sA = StateVectorSimulator(N); sA.state = base
    per_gate(sA, THETA); mx.eval(sA.state)
    sB = StateVectorSimulator(N); sB.state = base
    sB.apply_zz_layer(THETA, bonds); mx.eval(sB.state)
    outC = metal_fused(base, THETA, N); mx.eval(outC)
    errB = float(mx.max(mx.abs(sA.state - sB.state)).item().real)
    errC = float(mx.max(mx.abs(sA.state - outC)).item().real)
    print(f"correctness vs per-gate: mlx-fused err {errB:.2e}, metal err {errC:.2e}")
    assert errB < 5e-6 and errC < 5e-6, "implementations disagree"

    # ---- timing: one full ZZ layer at 25q ----
    print(f"\n== single ZZ layer, n={N}, {N-1} bonds ==")
    simA = StateVectorSimulator(N)
    def runA():
        simA.state = base
        per_gate(simA, THETA)
        mx.eval(simA.state)
    tA = bench("A per-gate (24 diagonal multiplies)", runA)

    simB = StateVectorSimulator(N)
    simB.apply_zz_layer(THETA, bonds)  # populate cache before timing
    def runB():
        simB.state = base
        simB.apply_zz_layer(THETA, bonds)
        mx.eval(simB.state)
    tB = bench("B mlx-fused (cached phase vector)", runB)

    def runC():
        out = metal_fused(base, THETA, N)
        mx.eval(out)
    tC = bench("C metal shader (in-kernel phase)", runC)

    # ---- cold-build cost for B's cache ----
    t0 = time.perf_counter()
    StateVectorSimulator._ZZ_LAYER_CACHE.clear()
    s = StateVectorSimulator(N); s.state = base
    s.apply_zz_layer(THETA, bonds); mx.eval(s.state)
    print(f"\nB cold build (first layer incl. cache construction): "
          f"{(time.perf_counter()-t0)*1000:.1f} ms")

    print(f"\nspeedups vs per-gate: mlx-fused {tA/tB:.1f}x, metal {tA/tC:.1f}x; "
          f"metal vs mlx-fused {tB/tC:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
