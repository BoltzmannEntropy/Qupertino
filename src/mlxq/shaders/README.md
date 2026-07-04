# mlxQ Metal shaders

Every hand-tuned Metal kernel in mlxQ lives in this package. All kernels are
**opt-in** (`MLXQ_METAL_KERNELS=1`); with the flag unset the simulator uses the
pure-MLX structured dispatch path unchanged. Each kernel entry records the
design, the derivation where non-obvious, measured numbers at 25 qubits on an
idle M1 Max (mean over ≥10 repeats after warmup), the parity error vs the
pure-MLX path, and the codex CLI review outcome.

Per-shader workflow (enforced for every entry below): implement → measure →
codex CLI review → re-test → document.

## Shipped shaders

### zz.py — fused ZZ Trotter layer (`zz_chain_layer`)
- exp(−iθ·ΣZZ) over an arbitrary bond list in ONE pass: per-amplitude mismatch
  count (XOR+popcount for chains; per-bond two-bit masks generically) indexes a
  phase LUT of size bonds+1. No trig, no phase vector in memory.
- 25q, 24 bonds: per-gate 35.9 ms → MLX fused 2.5 ms → **Metal 1.7 ms**.
  At the memory floor; codex review round 1 predicted and confirmed no further
  headroom. Parity ≤7e-8.
- Codex: LUT + popcount design adopted from round-1 review (2026-07-04).

### qft.py — radix-4 fused QFT stages (`qft_stage_all`)
- One thread per quad executes TWO stages (Hadamard butterfly + full
  controlled-phase ladder, closed-form angle α=π·low/2^s) with one read and
  one write per amplitude; single-stage pair kernel finishes odd n.
- 25q: per-gate 748 ms → MLX fused ladder 255 ms → **Metal 21.1 ms**
  (3.7× faster than `mx.fft` at 77 ms). Parity ≤5e-6 (float32 trig).
- Codex: round-1 radix-4 body was algebraically wrong and retracted by codex
  itself in round 2; the corrected derivation was verified independently
  before implementation. History: `paper/tqc-acm-2026/reviews_20260704_shaders/`.

### single_qubit.py — fused all-qubit 1q layers (`rx_layer_all`, `u2_layer_all`)
- Layers of the same 1q gate on every qubit run as ⌊n/2⌋ tensor-product
  (U⊗U) pair passes + one single pass for odd n. RX has a specialized
  real-coefficient kernel; `u2_layer_all` takes an arbitrary 2×2.
- 25q: RX layer 194 ms → **20.9 ms (9.3×)**; H layer 698 ms → **19.8 ms (35×)**.
  End-to-end: TFIM 4.14→0.45 s, Grover 1.22→0.15 s, quantum walk 3.60→0.75 s,
  QCBM 1.70→0.81 s. Parity ≤5.2e-8 (Grover circuit), 5e-10 (H layer).
- Device detector: any maximal window of consecutive 1q ops where every wire
  sees the identical gate sequence fuses into a stack of layers (ops on
  distinct wires commute) — catches interleaved patterns (Grover H,X pairs).
- Codex: RX pair-kernel body from round-2 review, verified then adopted.

### diag.py — fused CZ/CPHASE diagonal layer (`diag_pair_layer`) [S1]
- A run of CZ or equal-θ CPHASE gates over ANY bond topology multiplies each
  amplitude by exp(iθ·m), m = number of bonds with both qubits 1. One pass:
  chain/ring closed form `popcount((i & (i>>1)) & edge_mask)` (+ wrap-bond
  test), generic per-bond mask loop otherwise; LUT exact ±1 for CZ.
- 25q end-to-end: QAOA 10.0×, quantum walk 10.3×, graph_state 8.1×,
  deutsch_jozsa 6.7× (with S2: Grover 13.4×). Parity exact. `ae` unchanged
  (single CZs, no run).
- Codex round 1 (reviews_shaders_v2/round1_s1_diag_s2_xor_affine.md):
  CORRECT; adopted wire validation, chain/ring fast path, exact CZ LUT.

### xor_affine.py — GF(2) affine permutation (`xor_affine_gather`) [S2]
- Any run of CNOT/X/SWAP gates acts on basis indices as f(x)=Mx⊕c; the block
  is ONE amplitude permutation, out[y]=state[f⁻¹(y)], with f⁻¹ composed by
  replaying the run reversed (each gate an involution). Kernel: n parities +
  gather; identity-M runs (X layers) use the `src = y ^ c` fast path.
- 25q end-to-end: GHZ 152→22.7 ms (6.7×), QCBM 12.2×, Grover 14.4× combined
  with S1. Parity exact + pure-Python composition-inverse property test.
- Codex round 1: CORRECT; adopted wire validation and bit-flip fast path;
  rejected absorbing CZ into runs (unsound — data-dependent phase), agreed.

### single_qubit.py — per-qubit-varying layer (`u2_list_layer_all`) [S3]
- Any window of consecutive 1q ops collapses per wire into ONE 2×2 product
  (complex128, at fusion time); the layer runs as ⌊n/2⌋ (ua⊗ub) pair passes
  with identity pairs skipped. Exact-semantics collapse tests (rtol=0,
  atol=1e-12): all-identity windows emit nothing, all-X products emit one
  bit-flip gather — Grover's init-H + H,X prologue becomes a single gather.
- 25q end-to-end: Grover 20.2×, su2rand 17.9×, QCBM 14.9×, variational
  14.0×, realamp 9.6×, qnn 9.3×, random_circuit 8.4×, vqe 2.6× (residual is
  the expectation-value stage). Parity 1.3e-7.
- Codex round 2 (reviews_shaders_v2/round2_s3_u2_list.md) caught a HIGH bug:
  float32 products + numpy default rtol silently dropped RZ(1e-6)-scale
  gates. Fixed (complex128 + rtol=0); regression test added. Confirmed
  product order, tensor order, and no detector double-consume.

### diag.py — weighted diagonal layer (`diag_weighted_layer`) [S4]
- CPHASE runs with per-bond angles (QPE controlled-power ladder base·2^p,
  inline-iQFT ladders; CZ joins as θ=π). Bonds group by equal angle; each
  group contributes one LUT phase (host double precision) indexed by its
  both-bits-set count — one pass total, bounded float32 product chain.
- 25q: phase_estimation 3.3×, phase_estimation_inexact 8.7×. QPE parity
  test covers angles to 0.4·2^p.
- Codex round 3 (reviews_shaders_v2/round3_s4_s5_s6.md): CORRECT; grouping
  adopted after codex measured 1.25e-5 drift for 300 per-bond float32
  products (budget 5e-6).

### pauli_pair.py — XX/YY Trotter layers (`xx_layer`, `yy_layer`) [S5]
- Same-family XX (or YY) terms all commute, so the layer diagonalizes:
  XX = H⊗ⁿ·ZZ-LUT·H⊗ⁿ; YY conjugates by V=S·H, run as S/S† popcount-LUT
  diagonal passes around the H layers. H⊗ⁿ uses a radix-4 Walsh–Hadamard
  kernel (4 qubits per pass, add/sub only): H layer 19.8 → **12.7 ms**.
- 25q: heisenberg 9.5×, heisenberg_xxz 9.5×, heisenberg_random_field 5.5×.
  ladder_heisenberg stays 1.0×: its XX,YY,ZZ per-bond interleave has no
  same-family runs, and cross-family reordering would change the Trotterized
  operator — refused for correctness.
- Codex round 3: CORRECT (V Z V†=Y verified numerically); Walsh radix-4 was
  codex's top perf suggestion, adopted and confirmed.

### zz.py — weighted ZZ layer (`zz_weighted_layer`) [S6]
- Per-bond ZZ couplings (long-range Ising J/d^α, arbitrary weighted graphs):
  bonds group by equal angle, per-group mismatch-count LUT, one complex
  product per group. All-to-all at 25q (300 bonds, 24 distance groups) runs
  in ONE pass vs 300 sequential passes.
- 25q: long_range_ising **30.5×** (13.9 s → 0.46 s), tfim_trotter2 22.3×,
  tfim_random_field 10.8×.
- Codex round 3: CORRECT sign convention; grouped-LUT precision fix applied.

### qft.py — gate-stream QFT stage detector (`qft_stage_sub`) [S7]
- Gate-identical benchmark streams spell QFT as H + CPHASE ladders; the
  generalized stage kernel executes one whole stage (butterfly + closed-form
  ladder, angle = ±π·V/2^(m−1−j) over the ladder-bit value V) per pass, on
  any subregister [j..m−1]. Forward detector matches H(j) then the ascending
  exact-angle ladder; inverse detector matches QPE's descending negative
  ladder followed by H(j) (the adjoint stage: phase first, then butterfly).
  Same gate stream every backend receives; only the routing differs.
- Parity at n=8: full gate-stream QFT and full QPE circuits.

## Coverage summary

Every gate-based benchmark workload routes through at least one shader;
measured non-wins and their structural reasons: `ae` (isolated CZs between
1q blocks, no runs), `wstate` (strictly alternating RY/CNOT, no runs),
`ladder_heisenberg` (cross-family interleave, reordering unsound).
Excluded: `steady_state` (density-matrix Kraus path, no gate list) and
`qft_fft_primitive` (primitive reference row).
