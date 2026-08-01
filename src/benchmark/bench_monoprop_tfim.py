"""Head-to-head: mlxQ statevector vs monoprop Pauli propagation.

Re-implements the benchmark scenarios published in Algorithmiq's monoprop
repository (github.com/algorithmiq/monoprop, benches/) so both engines run
the *same* circuits, observables, and harness semantics on this machine:

  tfim    benches/third_party/pauli_prop: Trotterized 2D tilted-field Ising
          model. Upstream commits 6x6 (36 qubits), hx=hz=1.0, j=1.5, dt=0.05,
          steps 0..40 by 2, lower_atol=1e-6, no weight cutoff, observable ZZ
          on the central horizontal bond. A 36-qubit statevector needs ~0.5 TB,
          so the head-to-head sweeps grids both engines can hold; monoprop can
          additionally be run alone at the upstream 6x6 size.

  kicked  benches/bench_models.py kicked Ising: RX/RZZ layers on the IBM Eagle
          127-qubit heavy-hex topology, 20 layers, theta=coupling=pi/4,
          observable Z on the central qubit, cutoff=8, lower_atol=1e-4.
          Here run on the induced subgraph over the first n qubits.

  check   Correctness cross-check: monoprop at near-zero truncation must agree
          with the exact statevector per-step expectation values.

Harness fidelity (mirrors monoprop's benches/third_party/pauli_prop):
  - each measured TFIM point = one application of the single-step circuit
    followed by the expectation value, with the timer bracketing both
    (their `_run_steps`); the kicked scenario times one propagation of the
    full 20-layer circuit plus the expectation value (their `bench_models`).
  - monoprop's ExpGate parameter p means exp(-i*p*P) (verified numerically:
    Z through an X-gate with parameter p returns cos(2p)), so Qiskit's
    RZZ/RZ/RX(theta) used upstream become monoprop parameters theta/2.
    The mlxQ side keeps the Qiskit angles. `check` proves the mapping.

Usage (from the repo root, with the project venv):
  PYTHONPATH=src .runtime-venv/bin/python src/benchmark/bench_monoprop_tfim.py check
  PYTHONPATH=src .runtime-venv/bin/python src/benchmark/bench_monoprop_tfim.py tfim --sizes 3x3,4x4,4x5,5x5,4x7
  PYTHONPATH=src .runtime-venv/bin/python src/benchmark/bench_monoprop_tfim.py tfim --sizes 6x6 --backends monoprop
  PYTHONPATH=src .runtime-venv/bin/python src/benchmark/bench_monoprop_tfim.py kicked --qubits 16,20,24,28
Set MLXQ_METAL_KERNELS=1 to route the mlxQ side through the Metal shader tier.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

RESULTS_DIR = REPO_ROOT / "bench" / "monoprop_compare"


# ---------------------------------------------------------------------------
# Scenario definitions, copied from monoprop's benches so numbers line up.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TfimSettings:
    """One TFIM benchmark point (upstream settings.json defaults)."""

    nx: int = 6
    ny: int = 6
    hx: float = 1.0
    hz: float = 1.0
    j: float = 1.5
    dt: float = 0.05
    step_min: int = 0
    step_max: int = 40
    step_size: int = 2
    lower_atol: float = 1e-6
    cutoff: Optional[int] = None  # None = no weight cutoff (spelled num_qubits)

    @property
    def num_qubits(self) -> int:
        return self.nx * self.ny

    @property
    def step_range(self) -> range:
        return range(self.step_min, self.step_max + 1, self.step_size)

    @property
    def theta_zz(self) -> float:
        return self.dt * self.j

    @property
    def theta_z(self) -> float:
        return self.dt * self.hz

    @property
    def theta_x(self) -> float:
        return self.dt * self.hx

    @property
    def observable_qubits(self) -> Tuple[int, int]:
        return central_bond(self.nx, self.ny)


def grid_edges(nx: int, ny: int) -> List[Tuple[int, int]]:
    """Nearest-neighbor edges of an nx-by-ny grid, row-major qubit indexing."""
    edges = []
    for row in range(ny):
        for col in range(nx):
            idx = row * nx + col
            if col + 1 < nx:
                edges.append((idx, idx + 1))
            if row + 1 < ny:
                edges.append((idx, idx + nx))
    return edges


def central_bond(nx: int, ny: int) -> Tuple[int, int]:
    """The horizontally-adjacent pair nearest the grid center ((20,21) for 6x6)."""
    row = ny // 2
    col = max(nx // 2 - 1, 0)
    idx = row * nx + col
    return idx, idx + 1


# IBM Eagle heavy-hex coupling map, copied from monoprop benches/_builders.py.
HEAVY_HEX_TOPOLOGY: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
    (9, 10), (10, 11), (11, 12), (12, 13), (0, 14), (4, 15), (8, 16), (12, 17),
    (14, 18), (15, 22), (16, 26), (17, 30), (18, 19), (19, 20), (20, 21),
    (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27), (27, 28),
    (28, 29), (29, 30), (30, 31), (31, 32), (20, 33), (24, 34), (28, 35),
    (32, 36), (33, 39), (34, 43), (35, 47), (36, 51), (37, 38), (38, 39),
    (39, 40), (40, 41), (41, 42), (42, 43), (43, 44), (44, 45), (45, 46),
    (46, 47), (47, 48), (48, 49), (49, 50), (50, 51), (37, 52), (41, 53),
    (45, 54), (49, 55), (52, 56), (53, 60), (54, 64), (55, 68), (56, 57),
    (57, 58), (58, 59), (59, 60), (60, 61), (61, 62), (62, 63), (63, 64),
    (64, 65), (65, 66), (66, 67), (67, 68), (68, 69), (69, 70), (58, 71),
    (62, 72), (66, 73), (70, 74), (71, 77), (72, 81), (73, 85), (74, 89),
    (75, 76), (76, 77), (77, 78), (78, 79), (79, 80), (80, 81), (81, 82),
    (82, 83), (83, 84), (84, 85), (85, 86), (86, 87), (87, 88), (88, 89),
    (75, 90), (79, 91), (83, 92), (87, 93), (90, 94), (91, 98), (92, 102),
    (93, 106), (94, 95), (95, 96), (96, 97), (97, 98), (98, 99), (99, 100),
    (100, 101), (101, 102), (102, 103), (103, 104), (104, 105), (105, 106),
    (106, 107), (107, 108), (96, 109), (100, 110), (104, 111), (108, 112),
    (109, 114), (110, 118), (111, 122), (112, 126), (113, 114), (114, 115),
    (115, 116), (116, 117), (117, 118), (118, 119), (119, 120), (120, 121),
    (121, 122), (122, 123), (123, 124), (124, 125), (125, 126),
]


@dataclass(frozen=True)
class KickedSettings:
    """Kicked-Ising point (upstream KickedIsingConfig defaults, resizable n)."""

    num_qubits: int = 127
    num_layers: int = 20
    theta: float = math.pi / 4
    coupling: float = math.pi / 4
    cutoff: int = 8
    lower_atol: float = 1e-4

    @property
    def observable_qubit(self) -> int:
        # 62 is the center of the 127-qubit Eagle; scale with the subgraph.
        return 62 if self.num_qubits == 127 else self.num_qubits // 2

    @property
    def edges(self) -> List[Tuple[int, int]]:
        n = self.num_qubits
        return [(i, j) for i, j in HEAVY_HEX_TOPOLOGY if i < n and j < n]

    # ExpGate parameters, exactly as monoprop's _xlayer/_zzlayer pass them
    # (p means exp(-i*p*P)): X gates get -(theta/2), ZZ gates get -coupling.
    @property
    def x_param(self) -> float:
        return -(self.theta / 2)

    @property
    def zz_param(self) -> float:
        return -self.coupling


# ---------------------------------------------------------------------------
# Harness (mirrors monoprop benches/third_party/pauli_prop/backends.py)
# ---------------------------------------------------------------------------

@dataclass
class BackendResult:
    """Per-step series for one backend over one model."""

    label: str
    runtime: List[float] = field(default_factory=list)
    expvals: List[float] = field(default_factory=list)
    num_terms: List[int] = field(default_factory=list)
    memory_mb: List[float] = field(default_factory=list)


def _run_steps(settings: TfimSettings, label: str,
               step: Callable[[int], Tuple[float, int, float]]) -> BackendResult:
    """Drive `step` once per Trotter point; the timer brackets propagation and
    the expectation value, matching what every upstream backend reports."""
    result = BackendResult(label=label)
    for step_idx, _ in enumerate(settings.step_range):
        t1 = time.perf_counter()
        expval, num_terms, memory_mb = step(step_idx)
        t2 = time.perf_counter()
        result.runtime.append(t2 - t1)
        result.expvals.append(expval)
        result.num_terms.append(num_terms)
        result.memory_mb.append(memory_mb)
    return result


# ---------------------------------------------------------------------------
# monoprop backend (native API; no qiskit dependency)
# ---------------------------------------------------------------------------

def _monoprop_tfim_circuit(s: TfimSettings):
    from monoprop import Circuit, ExpGate, Pauli, PauliOperator

    n = s.num_qubits
    gates, params = [], []
    # ExpGate parameter p = exp(-i*p*P): Qiskit's RZZ/RZ/RX(theta) -> p = theta/2.
    for i, k in grid_edges(s.nx, s.ny):
        gates.append(ExpGate(PauliOperator({Pauli("ZZ", (i, k)): 1.0}, num_qubits=n)))
        params.append(s.theta_zz / 2.0)
    for i in range(n):
        gates.append(ExpGate(PauliOperator({Pauli("Z", (i,)): 1.0}, num_qubits=n)))
        params.append(s.theta_z / 2.0)
    for i in range(n):
        gates.append(ExpGate(PauliOperator({Pauli("X", (i,)): 1.0}, num_qubits=n)))
        params.append(s.theta_x / 2.0)
    return Circuit(gates=gates, parameters=params, initial_state=[])


def run_monoprop_tfim(s: TfimSettings, lower_atol: Optional[float] = None) -> BackendResult:
    from monoprop import Pauli, PauliOperator, PauliPropagator

    n = s.num_qubits
    circ = _monoprop_tfim_circuit(s)
    a, b = s.observable_qubits
    observable = PauliOperator({Pauli("ZZ", (a, b)): 1.0}, num_qubits=n)
    propagator = PauliPropagator(
        observable,
        circ.initial_state,
        cutoff=n if s.cutoff is None else s.cutoff,
        lower_atol=s.lower_atol if lower_atol is None else lower_atol,
    )

    def step(_step_idx: int) -> Tuple[float, int, float]:
        propagator.propagate(circ)
        expval = propagator.expectation_value()
        mem = propagator._simulator.operator_memory_bytes() / 1024**2
        return expval, propagator.size(), mem

    return _run_steps(s, "monoprop", step)


def run_monoprop_kicked(s: KickedSettings) -> Tuple[float, float, int, float]:
    """Returns (runtime_s, expval, num_terms, memory_mb) for one full run."""
    from monoprop import Circuit, ExpGate, Pauli, PauliOperator, PauliPropagator

    n = s.num_qubits
    gates, params = [], []
    for _ in range(s.num_layers):
        for i in range(n):
            gates.append(ExpGate(PauliOperator({Pauli("X", (i,)): 1.0}, num_qubits=n)))
            params.append(s.x_param)
        for i, j in s.edges:
            gates.append(ExpGate(PauliOperator({Pauli("ZZ", (i, j)): 1.0}, num_qubits=n)))
            params.append(s.zz_param)
    circ = Circuit(gates=gates, parameters=params, initial_state=[])
    observable = PauliOperator(
        {Pauli("Z", (s.observable_qubit,)): 1.0}, num_qubits=n)
    propagator = PauliPropagator(
        observable, circ.initial_state, cutoff=s.cutoff, lower_atol=s.lower_atol)

    t1 = time.perf_counter()
    propagator.propagate(circ)
    expval = propagator.expectation_value()
    t2 = time.perf_counter()
    mem = propagator._simulator.operator_memory_bytes() / 1024**2
    return t2 - t1, expval, propagator.size(), mem


# ---------------------------------------------------------------------------
# mlxQ backend
# ---------------------------------------------------------------------------

def _mlxq_modules():
    import mlx.core as mx
    from mlxq import shaders
    from mlxq.gates import RX, RZ
    from mlxq.sim import StateVectorSimulator
    return mx, shaders, RX, RZ, StateVectorSimulator


def _z_layer_phase(mx, n: int, theta: float):
    """Cached diagonal for RZ(theta) on every qubit: one multiply per layer.

    Same construction as StateVectorSimulator.apply_zz_layer's cache: the
    total phase per basis state is exp(-i*(theta/2)*sum_q s_q), s_q = +-1.
    """
    idx = mx.arange(1 << n, dtype=mx.uint32)
    acc = mx.zeros((1 << n,), dtype=mx.int32)
    for q in range(n):
        bit = ((idx >> (n - 1 - q)) & 1).astype(mx.int32)
        acc = acc + (1 - 2 * bit)
    ang = (-theta / 2.0) * acc.astype(mx.float32)
    phase = mx.cos(ang).astype(mx.complex64) + 1j * mx.sin(ang).astype(mx.complex64)
    mx.eval(phase)
    return phase


def _zz_expectation(mx, state, n: int, a: int, b: int) -> float:
    """<psi| Z_a Z_b |psi> = P(bits agree) - P(bits disagree)."""
    q0, q1 = sorted((a, b))
    a_dim = 1 << q0
    m_dim = 1 << (q1 - q0 - 1)
    b_dim = 1 << (n - q1 - 1)
    t = mx.reshape(state, (a_dim, 2, m_dim, 2, b_dim))
    p = mx.abs(t) ** 2
    agree = mx.sum(p[:, 0, :, 0, :]) + mx.sum(p[:, 1, :, 1, :])
    disagree = mx.sum(p[:, 0, :, 1, :]) + mx.sum(p[:, 1, :, 0, :])
    return float(agree - disagree)


def _z_expectation(mx, state, n: int, q: int) -> float:
    a_dim = 1 << q
    b_dim = 1 << (n - q - 1)
    t = mx.reshape(state, (a_dim, 2, b_dim))
    p = mx.abs(t) ** 2
    return float(mx.sum(p[:, 0, :]) - mx.sum(p[:, 1, :]))


def run_mlxq_tfim(s: TfimSettings) -> BackendResult:
    mx, shaders, RX, RZ, StateVectorSimulator = _mlxq_modules()

    n = s.num_qubits
    sim = StateVectorSimulator(n)
    edges = grid_edges(s.nx, s.ny)
    a, b = s.observable_qubits
    metal = shaders.metal_enabled()
    label = "mlxQ (Metal shaders)" if metal else "mlxQ (pure MLX)"
    z_phase = _z_layer_phase(mx, n, s.theta_z)
    rx_gate = RX(s.theta_x)
    state_mb = (1 << n) * 8 / 1024**2  # complex64 statevector

    def step(_step_idx: int) -> Tuple[float, int, float]:
        sim.apply_zz_layer(s.theta_zz / 2.0, edges)
        sim.state = sim.state * z_phase
        if metal:
            sim.state = shaders.rx_layer_all(sim.state, n, s.theta_x)
        else:
            for q in range(n):
                sim.apply_single(rx_gate, q)
        expval = _zz_expectation(mx, sim.state, n, a, b)  # forces eval
        if n >= 26:
            # Each step churns several state-sized temporaries; above ~1 GB per
            # buffer the cached pool outgrows unified memory and the run starts
            # swapping (measured: 30x per-step degradation at 28 qubits).
            mx.clear_cache()
        return expval, 1 << n, state_mb

    return _run_steps(s, label, step)


def run_mlxq_kicked(s: KickedSettings) -> Tuple[float, float, int, float]:
    """Returns (runtime_s, expval, num_amplitudes, memory_mb) for one full run."""
    mx, shaders, RX, RZ, StateVectorSimulator = _mlxq_modules()

    n = s.num_qubits
    sim = StateVectorSimulator(n)
    edges = s.edges
    metal = shaders.metal_enabled()
    # monoprop param p = exp(-i*p*P): mlxQ RX(theta) has theta = 2p, and
    # apply_zz_layer(theta) = exp(-i*theta*sum ZZ) takes theta = p directly.
    rx_theta = 2.0 * s.x_param
    rx_gate = RX(rx_theta)
    zz_layer_theta = s.zz_param

    t1 = time.perf_counter()
    for _ in range(s.num_layers):
        if metal:
            sim.state = shaders.rx_layer_all(sim.state, n, rx_theta)
        else:
            for q in range(n):
                sim.apply_single(rx_gate, q)
        sim.apply_zz_layer(zz_layer_theta, edges)
        if n >= 26:
            mx.eval(sim.state)
            mx.clear_cache()  # same pool-growth guard as the TFIM path
    expval = _z_expectation(mx, sim.state, n, s.observable_qubit)
    t2 = time.perf_counter()
    return t2 - t1, expval, 1 << n, (1 << n) * 8 / 1024**2


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def _parse_sizes(spec: str) -> List[Tuple[int, int]]:
    sizes = []
    for token in spec.split(","):
        nx, ny = token.lower().split("x")
        sizes.append((int(nx), int(ny)))
    return sizes


def _print_series(result: BackendResult, settings: TfimSettings) -> None:
    total = sum(result.runtime)
    print(f"  {result.label}: total {total:.3f}s over {len(result.runtime)} points, "
          f"final expval {result.expvals[-1]:+.6f}, "
          f"final terms/amps {result.num_terms[-1]}, "
          f"final memory {result.memory_mb[-1]:.1f} MB")


def _jsonl_write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def cmd_tfim(args) -> None:
    out = RESULTS_DIR / f"tfim_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    backends = args.backends.split(",")
    for nx, ny in _parse_sizes(args.sizes):
        s = TfimSettings(nx=nx, ny=ny, step_max=args.steps)
        print(f"\nTFIM {nx}x{ny} ({s.num_qubits} qubits), dt={s.dt}, "
              f"atol={s.lower_atol}, cutoff={s.cutoff or 'none'}, "
              f"obs=ZZ{list(s.observable_qubits)}, "
              f"{len(list(s.step_range))} points")
        results = []
        if "mlxq" in backends:
            results.append(run_mlxq_tfim(s))
        if "monoprop" in backends:
            results.append(run_monoprop_tfim(s))
        for r in results:
            _print_series(r, s)
            _jsonl_write(out, {
                "scenario": "tfim", "nx": nx, "ny": ny,
                "num_qubits": s.num_qubits, "label": r.label,
                "steps": list(s.step_range),
                "runtime_s": r.runtime, "expvals": r.expvals,
                "num_terms": r.num_terms, "memory_mb": r.memory_mb,
                "metal": os.environ.get("MLXQ_METAL_KERNELS", "0"),
                "monoprop_threads": os.environ.get("monoprop_NUM_THREADS", "default"),
            })
        if len(results) == 2:
            t_mlxq, t_mono = sum(results[0].runtime), sum(results[1].runtime)
            drift = max(abs(x - y) for x, y in
                        zip(results[0].expvals, results[1].expvals))
            faster = "mlxQ" if t_mlxq < t_mono else "monoprop"
            ratio = max(t_mlxq, t_mono) / max(min(t_mlxq, t_mono), 1e-12)
            print(f"  -> {faster} faster by {ratio:.1f}x; "
                  f"max expval drift {drift:.2e}")
    print(f"\nresults: {out}")


def cmd_kicked(args) -> None:
    out = RESULTS_DIR / f"kicked_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    backends = args.backends.split(",")
    for n in [int(x) for x in args.qubits.split(",")]:
        s = KickedSettings(num_qubits=n)
        print(f"\nKicked Ising n={n} (heavy-hex induced subgraph, "
              f"{len(s.edges)} edges), {s.num_layers} layers, "
              f"obs=Z[{s.observable_qubit}], cutoff={s.cutoff}, "
              f"atol={s.lower_atol}")
        rows = []
        if "mlxq" in backends and n <= args.max_sv_qubits:
            rt, ev, terms, mem = run_mlxq_kicked(s)
            metal = os.environ.get("MLXQ_METAL_KERNELS", "0") == "1"
            rows.append(("mlxQ (Metal shaders)" if metal else "mlxQ (pure MLX)",
                         rt, ev, terms, mem))
        if "monoprop" in backends:
            rt, ev, terms, mem = run_monoprop_kicked(s)
            rows.append(("monoprop", rt, ev, terms, mem))
        for label, rt, ev, terms, mem in rows:
            print(f"  {label}: {rt:.3f}s, expval {ev:+.6f}, "
                  f"terms/amps {terms}, memory {mem:.1f} MB")
            _jsonl_write(out, {
                "scenario": "kicked", "num_qubits": n, "label": label,
                "runtime_s": rt, "expval": ev, "num_terms": terms,
                "memory_mb": mem,
                "metal": os.environ.get("MLXQ_METAL_KERNELS", "0"),
            })
        if len(rows) == 2:
            faster = "mlxQ" if rows[0][1] < rows[1][1] else "monoprop"
            ratio = max(rows[0][1], rows[1][1]) / max(min(rows[0][1], rows[1][1]), 1e-12)
            print(f"  -> {faster} faster by {ratio:.1f}x; "
                  f"expval delta {abs(rows[0][2] - rows[1][2]):.2e} "
                  f"(monoprop truncates at atol={s.lower_atol}, cutoff={s.cutoff})")
    print(f"\nresults: {out}")


def cmd_check(args) -> None:
    """monoprop at near-zero truncation must match the exact statevector."""
    ok = True
    for nx, ny in _parse_sizes(args.sizes):
        s = TfimSettings(nx=nx, ny=ny, step_max=args.steps, lower_atol=1e-14)
        mlxq_r = run_mlxq_tfim(s)
        mono_r = run_monoprop_tfim(s)
        drift = max(abs(x - y) for x, y in zip(mlxq_r.expvals, mono_r.expvals))
        status = "OK" if drift < 1e-5 else "MISMATCH"
        ok = ok and drift < 1e-5
        print(f"TFIM {nx}x{ny}: max |mlxQ - monoprop| = {drift:.2e}  [{status}]")
    n = args.kicked_qubits
    s = KickedSettings(num_qubits=n, cutoff=n, lower_atol=1e-14)
    _, ev_m, _, _ = run_mlxq_kicked(s)
    _, ev_p, _, _ = run_monoprop_kicked(s)
    drift = abs(ev_m - ev_p)
    status = "OK" if drift < 1e-5 else "MISMATCH"
    ok = ok and drift < 1e-5
    print(f"Kicked n={n}: |mlxQ - monoprop| = {drift:.2e}  [{status}]")
    sys.exit(0 if ok else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_tfim = sub.add_parser("tfim", help="2D TFIM Trotter benchmark")
    p_tfim.add_argument("--sizes", default="3x3,4x4,4x5,5x5,4x7")
    p_tfim.add_argument("--steps", type=int, default=40)
    p_tfim.add_argument("--backends", default="mlxq,monoprop")
    p_tfim.set_defaults(func=cmd_tfim)

    p_kicked = sub.add_parser("kicked", help="heavy-hex kicked Ising benchmark")
    p_kicked.add_argument("--qubits", default="16,20,24,28")
    p_kicked.add_argument("--backends", default="mlxq,monoprop")
    p_kicked.add_argument("--max-sv-qubits", type=int, default=30)
    p_kicked.set_defaults(func=cmd_kicked)

    p_check = sub.add_parser("check", help="cross-engine correctness check")
    p_check.add_argument("--sizes", default="3x3,3x4")
    p_check.add_argument("--steps", type=int, default=10)
    p_check.add_argument("--kicked-qubits", type=int, default=10)
    p_check.set_defaults(func=cmd_check)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
