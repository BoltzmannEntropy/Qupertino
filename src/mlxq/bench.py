import os
import json
import time
from typing import List, Dict, Any, Tuple

import math
import random
from typing import List, Dict, Any, Tuple, Optional
from .qasm import parse_qasm_file
from .device import Device
from .metrics import now_ms, cpu_seconds, peak_rss_mb
from .pretty import console, table, info, success, warn, error
from .gates import H, X, Y, Z, RX, RY, RZ, CNOT, CZ, CPHASE, Toffoli
from .tensor import kron
from .observables import expectation_value
from .states import zero_state
from .channels import amplitude_damping_kraus, apply_kraus
import mlx.core as mx


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _gate_stats(ops: List[Dict[str, Any]]) -> Tuple[int, Dict[str, int]]:
    total = len(ops)
    by = {}
    for op in ops:
        g = op.get("name", "?")
        by[g] = by.get(g, 0) + 1
    return total, by

def _estimate_depth(n_qubits: int, ops: List[Dict[str, Any]]) -> int:
    """Greedy layer scheduling depth estimate.

    Assign each op to the earliest layer with no wire conflicts.
    Returns the number of layers used (an upper bound on depth).
    """
    layers: List[set] = []  # set of occupied wires per layer
    for op in ops:
        wires = tuple(sorted(op.get('wires', [])))
        if not wires:
            continue
        placed = False
        for layer in layers:
            if all((w not in layer) for w in wires):
                for w in wires:
                    layer.add(w)
                placed = True
                break
        if not placed:
            layers.append(set(wires))
    return len(layers)


from pathlib import Path
import platform
import subprocess


def _detect_hardware_info() -> tuple[str, str]:
    """Detect Apple Silicon generation/variant for filename/labels.

    Returns (prefix, label) like ("M4_Max", "M4 Max"). Falls back to
    ("Unknown_Base", "Unknown Base") if not on macOS or cannot detect.
    """
    gen = "Unknown"; var = "Base"; label = "Unknown Base"
    if platform.system() == "Darwin":
        brand = ""
        try:
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], stderr=subprocess.DEVNULL)
            brand = out.decode("utf-8", "ignore").strip()
        except Exception:
            brand = platform.processor() or ""
        b = brand
        if "M1" in b:
            gen = "M1"
        elif "M2" in b:
            gen = "M2"
        elif "M3" in b:
            gen = "M3"
        elif "M4" in b:
            gen = "M4"
        else:
            gen = "Unknown"
        if "Max" in b:
            var = "Max"
        elif "Pro" in b:
            var = "Pro"
        elif "Ultra" in b:
            var = "Ultra"
        else:
            var = "Base"
        label = f"{gen} {var}"
    prefix = f"{gen}_{var}"
    return prefix, label


def _resolve_qasm_dir(qasm_dir: str) -> Optional[Path]:
    p = Path(qasm_dir)
    if p.is_dir():
        return p
    here = Path(__file__).resolve()
    # Try new and old repo layouts
    candidates = [
        # New preferred layout
        here.parents[3] / 'datasets' / 'qasm' / 'local',
        here.parents[2] / 'datasets' / 'qasm' / 'local',
        Path.cwd() / 'datasets' / 'qasm' / 'local',
        # Legacy locations for backward compatibility
        here.parents[2] / 'qasm_circuits',
        here.parents[3] / 'qasm_circuits',
        here.parents[2] / 'src' / 'qasm_circuits',
        Path.cwd() / 'qasm_circuits',
        Path.cwd() / 'src' / 'qasm_circuits',
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


def run_qasm_suite(qasm_dir: str = "datasets/qasm/local",
                   csv_out: str = "bench/qasm_MLX_python.csv",
                   json_out: str = "bench/qasm_MLX_python.json") -> Dict[str, Any]:
    out_dir_env = os.environ.get("MLXQ_BENCH_OUT_DIR", "").strip()
    if out_dir_env:
        # When CLI/UI sets a run-specific output directory, keep QASM outputs there.
        if csv_out == "bench/qasm_MLX_python.csv":
            csv_out = str(Path(out_dir_env) / "qasm_MLX_python.csv")
        if json_out == "bench/qasm_MLX_python.json":
            json_out = str(Path(out_dir_env) / "qasm_MLX_python.json")
    max_qubits = _env_int("QASM_MAX_QUBITS", 18)
    # Default timeout: unlimited (0 or negative → no timeout)
    timeout_ms = _env_int("QASM_TIMEOUT_MS", 0)
    simulate_limit = _env_int("QASM_SIMULATE_LIMIT", 0)  # 0 → unlimited
    mem_cap_mb = _env_int("QASM_MAX_MEM_MB", 0)  # 0 → unlimited

    os.makedirs(os.path.dirname(csv_out), exist_ok=True)

    qdir = _resolve_qasm_dir(qasm_dir)
    if qdir is None:
        warn(f"QASM directory not found: {qasm_dir} (tried src/qasm_circuits as well); skipping")
        return {"rows": [], "suite": []}
    files = [f for f in sorted(os.listdir(qdir)) if f.endswith('.qasm')]
    # Identify an optional large-scale group: files immediately after known anchors
    large_set = set()
    def _add_next_after(anchor: str):
        try:
            i = files.index(anchor)
            if i + 1 < len(files):
                large_set.add(files[i + 1])
        except ValueError:
            return
    _add_next_after('dnn_n16.qasm')
    _add_next_after('vqe_ising.qasm')
    # Also include explicit heavy names if present
    for heavy in ['vqe_n24.qasm', 'factor247_n15.qasm']:
        if heavy in files:
            large_set.add(heavy)
    large_group: List[str] = sorted(large_set)
    include_large = os.environ.get('QASM_INCLUDE_LARGE', '0') == '1'
    # Split files into base suite and optional large group
    base_files = [f for f in files if f not in set(large_group)]
    # Announce optional large-scale set up-front
    if large_group:
        info("Optional large-scale QASM group: " + ", ".join(large_group) +
             (" (ENABLED)" if include_large else " (disabled; set QASM_INCLUDE_LARGE=1)"))
    rows = []
    suite = []
    t0 = now_ms(); cpu0 = cpu_seconds(); mem0 = peak_rss_mb()
    passed = 0

    info(f"QASM Bench: {len(files)} files | max_qubits={max_qubits} | timeout={timeout_ms}ms | simulate_limit={simulate_limit}")
    table("Suite", ("Circuit","Qubits","Gates","Depth","Result","Wall ms","CPU s","Peak ΔMB"), [])

    # Helper to process a single file
    def _process_one(fname: str):
        nonlocal passed
        path = str(qdir / fname)
        try:
            n, ops = parse_qasm_file(path)
        except Exception as e:
            error(f"Parse error {fname}: {e}")
            suite.append({"file": fname, "status": "parse_error", "error": str(e)})
            return

        total_gates, by_gate = _gate_stats(ops)
        depth = _estimate_depth(n, ops)
        if n <= 0:
            warn(f"Skipping {fname}: no qubits parsed")
            suite.append({"file": fname, "qubits": n, "gates": total_gates, "status": "skipped_no_qubits"})
            return
        if n > max_qubits:
            warn(f"Skipping {fname}: {n}q > max {max_qubits}")
            suite.append({"file": fname, "qubits": n, "gates": total_gates, "status": "skipped_qubits"})
            return

        dev = Device(n)
        t_wall0 = now_ms(); t_cpu0 = cpu_seconds(); m0 = peak_rss_mb()
        status = "ok"; err = None
        try:
            if simulate_limit and total_gates > simulate_limit:
                ops_to_run = ops[:simulate_limit]
            else:
                ops_to_run = ops
            # chunk execution with timeout checks (if enabled)
            step = 64
            for i in range(0, len(ops_to_run), step):
                dev.execute(ops_to_run[i:i+step])
                if timeout_ms > 0 and (now_ms() - t_wall0) > timeout_ms:
                    status = "timeout"; break
                if mem_cap_mb and (peak_rss_mb() - m0) > float(mem_cap_mb):
                    status = "memcap"; break
            if status == "ok":
                _ = dev.sim.probabilities()  # force eval
        except Exception as e:
            err = str(e)
            status = "unsupported" if "Unsupported" in err else "error"

        t_wall1 = now_ms(); t_cpu1 = cpu_seconds(); m1 = peak_rss_mb()
        dt_wall = t_wall1 - t_wall0; dt_cpu = t_cpu1 - t_cpu0; dmem = max(0.0, m1 - m0)

        if status == "ok":
            success(f"{fname:28} | {n:2d}q | {total_gates:5d} gates | depth {depth:4d} | {dt_wall:.2f}ms | +{dmem:.2f}MB")
            passed += 1
        elif status == "timeout":
            warn(f"{fname:28} | {n:2d}q | {total_gates:5d} gates | TIMEOUT at {dt_wall:.2f}ms")
        elif status == "unsupported":
            warn(f"{fname:28} | {n:2d}q | {total_gates:5d} gates | UNSUPPORTED: {err}")
        elif status == "memcap":
            warn(f"{fname:28} | {n:2d}q | {total_gates:5d} gates | MEMCAP over {mem_cap_mb}MB")
        elif status == "error":
            error(f"{fname:28} | {n:2d}q | {total_gates:5d} gates | ERROR: {err}")

        rows.append((fname, str(n), str(total_gates), str(depth), status, f"{dt_wall:.2f}", f"{dt_cpu:.2f}", f"{dmem:.2f}"))
        suite.append({
            "file": fname,
            "qubits": n,
            "gates": total_gates,
            "depth": depth,
            "by_gate": by_gate,
            "status": status,
            "wall_ms": dt_wall,
            "cpu_s": dt_cpu,
            "peak_delta_mb": dmem,
            **({"error": err} if err else {})
        })

    # Temporarily disable ASCII dump while running QASM
    _prev_ascii = os.environ.get('MLXQ_PRINT_ASCII')
    os.environ['MLXQ_PRINT_ASCII'] = '0'
    try:
        # Run base files first
        for fname in base_files:
            _process_one(fname)

    # Handle optional large-scale group
        if large_group:
            if include_large:
                info("=== Large-scale QASM group enabled ===")
                for fname in large_group:
                    _process_one(fname)
            else:
                for fname in large_group:
                    warn(f"Skipping {fname}: large-scale group disabled (set QASM_INCLUDE_LARGE=1 to enable)")
                    suite.append({"file": fname, "status": "skipped_large_group"})
    finally:
        if _prev_ascii is None:
            os.environ.pop('MLXQ_PRINT_ASCII', None)
        else:
            os.environ['MLXQ_PRINT_ASCII'] = _prev_ascii

    # Summary
    t1 = now_ms(); cpu1 = cpu_seconds(); mem1 = peak_rss_mb()
    total = len(rows)
    info(f"QASM Suite: {passed}/{total} completed without error | Total time: {t1-t0:.2f}ms")

    # Write CSV
    try:
        with open(csv_out, 'w') as f:
            f.write("circuit,qubits,gates,depth,status,wall_ms,cpu_s,peak_delta_mb\n")
            for r in rows:
                f.write(",".join(r) + "\n")
        success(f"CSV data: {csv_out}")
    except Exception as e:
        error(f"CSV write error: {e}")

    # Write JSON
    try:
        with open(json_out, 'w') as f:
            json.dump({
                "suite": suite,
                "summary": {
                    "passed": passed,
                    "total": total,
                    "wall_ms": t1 - t0,
                    "cpu_s": cpu1 - cpu0,
                    "peak_rss_mb": mem1,
                }
            }, f, indent=2)
        success(f"JSON: {json_out}")
    except Exception as e:
        error(f"JSON write error: {e}")

    return {"rows": rows, "suite": suite}


if __name__ == "__main__":
    run_qasm_suite()

# ---------------------------------------------------------------------------
# Additional benchmark suite (ported from C++ QuantumBenchmarks)
# ---------------------------------------------------------------------------

def _bench_result(name: str, qubits: int, gates: int, t0: float, c0: float, m0: float) -> Dict[str, Any]:
    return {
        "name": name,
        "qubits": qubits,
        "gates": gates,
        "wall_ms": now_ms() - t0,
        "cpu_s": cpu_seconds() - c0,
        "delta_mb": max(0.0, peak_rss_mb() - m0),
    }


def bench_gate_suite(reps: int = 1000) -> Dict[str, Any]:
    info(f"Gate micro-benchmarks, reps={reps}")
    rows = []
    suite = []
    # Single-qubit gates
    for name, gate in [("X", X()), ("H", H()), ("T", RZ(math.pi/4.0)), ("RX", RX(0.3)), ("RZ", RZ(-0.7))]:
        dev = Device(1)
        t0, c0, m0 = now_ms(), cpu_seconds(), peak_rss_mb()
        for _ in range(reps):
            if name in ("RX", "RZ"):
                dev.execute([{"name": name, "wires": [0], "parameters": [0.3 if name=="RX" else -0.7]}])
            else:
                dev.execute([{"name": name, "wires": [0]}])
        _ = dev.sim.probabilities()
        r = _bench_result(name, 1, reps, t0, c0, m0)
        rows.append((name, str(r["qubits"]), str(r["gates"]), f"{r['wall_ms']:.2f}", f"{r['cpu_s']:.2f}", f"{r['delta_mb']:.2f}"))
        suite.append(r)
    # Two- and three-qubit
    dev2 = Device(2); t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
    for _ in range(reps):
        dev2.execute([{"name": "CNOT", "wires": [0,1]}])
    _ = dev2.sim.probabilities()
    r = _bench_result("CNOT", 2, reps, t0, c0, m0)
    rows.append(("CNOT", "2", str(reps), f"{r['wall_ms']:.2f}", f"{r['cpu_s']:.2f}", f"{r['delta_mb']:.2f}")); suite.append(r)

    dev3 = Device(3); t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
    for _ in range(reps):
        dev3.execute([{"name": "CCX", "wires": [0,1,2]}])
    _ = dev3.sim.probabilities()
    r = _bench_result("Toffoli", 3, reps, t0, c0, m0)
    rows.append(("Toffoli", "3", str(reps), f"{r['wall_ms']:.2f}", f"{r['cpu_s']:.2f}", f"{r['delta_mb']:.2f}")); suite.append(r)

    table("Gate Benchmarks", ("gate","qubits","reps","wall(ms)","cpu(s)","ΔMB"), rows)
    return {"rows": rows, "suite": suite}


def simulate_qft(n: int) -> Dict[str, Any]:
    dev = Device(n)
    ops = []
    for j in range(n):
        ops.append({"name": "H", "wires": [j]})
        for k in range(j + 1, n):
            phi = math.pi / (2 ** (k - j))
            ops.append({"name": "CPHASE", "wires": [k, j], "parameters": [phi]})
    t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
    dev.execute(ops)
    _ = dev.sim.probabilities()
    return _bench_result("qft", n, len(ops), t0, c0, m0)


def simulate_phase_estimation(n: int) -> Dict[str, Any]:
    if n < 2:
        return {"error": "n<2", "name": "phase_estimation", "qubits": n}
    dev = Device(n)
    target = n - 1
    ops = []
    # prepare ancillas in H
    for p in range(n - 1):
        ops.append({"name": "H", "wires": [p]})
    base = 0.4
    for p in range(n - 1):
        angle = base * (2 ** p)
        ops.append({"name": "CPHASE", "wires": [p, target], "parameters": [angle]})
    # inverse QFT on ancillas
    for j in range(n - 2, -1, -1):
        for k in range(n - 2, j, -1):
            phi = -math.pi / (2 ** (k - j))
            ops.append({"name": "CPHASE", "wires": [k, j], "parameters": [phi]})
        ops.append({"name": "H", "wires": [j]})
    t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
    dev.execute(ops)
    _ = dev.sim.probabilities()
    return _bench_result("phase_estimation", n, len(ops), t0, c0, m0)


def simulate_variational(n: int, layers: int = 4) -> Dict[str, Any]:
    # Allow override via env
    layers = _env_int('MLXQ_VARIATIONAL_LAYERS', layers)
    dev = Device(n)
    ops = []
    for l in range(layers):
        base = 0.1 * (l + 1)
        for q in range(n):
            ops.append({"name": "RX", "wires": [q], "parameters": [base + 0.05*q]})
            ops.append({"name": "RZ", "wires": [q], "parameters": [1.5*base]})
        for q in range(n-1):
            ops.append({"name": "CNOT", "wires": [q, q+1]})
    t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
    dev.execute(ops)
    _ = dev.sim.probabilities()
    return _bench_result("variational_circuit", n, len(ops), t0, c0, m0)


def simulate_qcbm(n: int, layers: int = 4) -> Dict[str, Any]:
    layers = _env_int('MLXQ_QCBM_LAYERS', layers)
    dev = Device(n)
    ops = []
    for l in range(layers):
        for q in range(n):
            ops.append({"name": "RY", "wires": [q], "parameters": [0.2*(l+1)]})
            ops.append({"name": "RZ", "wires": [q], "parameters": [0.1*(l+1)]})
        for q in range(n-1):
            ops.append({"name": "CNOT", "wires": [q, q+1]})
        if n > 2:
            ops.append({"name": "CNOT", "wires": [n-1, 0]})
    t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
    dev.execute(ops)
    _ = dev.sim.probabilities()
    return _bench_result("qcbm", n, len(ops), t0, c0, m0)


def simulate_random_circuit(n: int, depth: int = 6, seed: int = 42) -> Dict[str, Any]:
    depth = _env_int('MLXQ_RANDOM_DEPTH', depth)
    rnd = random.Random(seed)
    dev = Device(n)
    ops = []
    for l in range(depth):
        for q in range(n):
            gate = rnd.choice(["RX", "RY", "RZ"])
            theta = (rnd.random() - 0.5) * 2.0
            ops.append({"name": gate, "wires": [q], "parameters": [theta]})
        for q in range(0, n-1, 2):
            ops.append({"name": "CNOT", "wires": [q, q+1]})
    t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
    dev.execute(ops)
    _ = dev.sim.probabilities()
    return _bench_result("random_circuit", n, len(ops), t0, c0, m0)


def simulate_qaoa(n: int, layers: int = 6) -> Dict[str, Any]:
    layers = _env_int('MLXQ_QAOA_LAYERS', layers)
    dev = Device(n)
    ops = []
    if n < 2:
        # Single‑qubit fallback: just apply RX layers
        for l in range(layers):
            beta = 0.4 + 0.05*l
            ops.append({"name": "RX", "wires": [0], "parameters": [2.0*beta]})
        t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
        dev.execute(ops)
        _ = dev.sim.probabilities()
        return _bench_result("qaoa", n, len(ops), t0, c0, m0)
    for l in range(layers):
        gamma = 0.6 + 0.1*l
        beta = 0.4 + 0.05*l
        for i in range(n):
            j = (i + 1) % n
            ops.append({"name": "CPHASE", "wires": [i, j], "parameters": [gamma]})
        for q in range(n):
            ops.append({"name": "RX", "wires": [q], "parameters": [2.0*beta]})
    t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
    dev.execute(ops)
    _ = dev.sim.probabilities()
    return _bench_result("qaoa", n, len(ops), t0, c0, m0)


def simulate_cuquantum_blueqat(n: int, depth: int = 12) -> Dict[str, Any]:
    """Replicate a common cuQuantum/Blueqat-style benchmark circuit.

    Brickwork entangling pattern with alternating pairs and single-qubit
    rotations per layer. Depth defaults to 12; parameters are fixed so runs
    are repeatable. This mirrors typical cuQuantum sample patterns while
    remaining backend-agnostic here.
    """
    dev = Device(n)
    ops: List[Dict[str, Any]] = []
    for l in range(depth):
        base = 0.2 + 0.03 * (l % 5)
        # Single-qubit layer
        for q in range(n):
            ops.append({"name": "RX", "wires": [q], "parameters": [base]})
            ops.append({"name": "RZ", "wires": [q], "parameters": [0.5 * base]})
        # Entangling brickwork: alternate offset each layer
        offset = l % 2
        for i in range(offset, n - 1, 2):
            ops.append({"name": "CNOT", "wires": [i, i + 1]})
    t0, c0, m0 = now_ms(), cpu_seconds(), peak_rss_mb()
    dev.execute(ops)
    _ = dev.sim.probabilities()
    return _bench_result("cuquantum_blueqat", n, len(ops), t0, c0, m0)


def simulate_grover(n: int, iterations: int = 1) -> Dict[str, Any]:
    """Approximate Grover: initialize uniform state, perform one diffusion-like step.
    Uses pairwise CZ as a cheap multi-qubit phase oracle proxy.
    """
    dev = Device(n)
    ops: List[Dict[str, Any]] = []
    # Initialize uniform
    for q in range(n):
        ops.append({"name": "H", "wires": [q]})
    # One iteration proxy
    for _ in range(max(1, iterations)):
        for q in range(n):
            ops.append({"name": "H", "wires": [q]})
            ops.append({"name": "X", "wires": [q]})
        for q in range(max(0, n-1)):
            ops.append({"name": "CZ", "wires": [q, q+1]})
        for q in range(n):
            ops.append({"name": "X", "wires": [q]})
            ops.append({"name": "H", "wires": [q]})
    t0, c0, m0 = now_ms(), cpu_seconds(), peak_rss_mb()
    dev.execute(ops)
    _ = dev.sim.probabilities()
    return _bench_result("grover", n, len(ops), t0, c0, m0)


def simulate_ghz(n: int) -> Dict[str, Any]:
    dev = Device(n)
    ops = [{"name": "H", "wires": [0]}]
    for q in range(n-1):
        ops.append({"name": "CNOT", "wires": [q, q+1]})
    t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
    dev.execute(ops)
    _ = dev.sim.probabilities()
    return _bench_result("ghz", n, len(ops), t0, c0, m0)


def _zz_phase_gate(theta: float):
    e = complex(math.cos(theta), math.sin(theta))
    em = complex(math.cos(-theta), math.sin(-theta))
    # diag(e^{-iθ}, e^{iθ}, e^{iθ}, e^{-iθ}) → use em for +/- accordingly
    return mx.array([[em,0+0j,0+0j,0+0j],[0+0j,e,0+0j,0+0j],[0+0j,0+0j,e,0+0j],[0+0j,0+0j,0+0j,em]], mx.complex64)

def _identity4():
    return kron(_identity2(), _identity2())

def _xx_phase_gate(theta: float):
    # exp(-i theta X⊗X) = cos θ I - i sin θ X⊗X
    X2 = kron(X(), X())
    I4 = _identity4()
    c = math.cos(theta)
    s = math.sin(theta)
    return c * I4 + (-1j * s) * X2

def _yy_phase_gate(theta: float):
    # exp(-i theta Y⊗Y) = cos θ I - i sin θ Y⊗Y
    Y2 = kron(Y(), Y())
    I4 = _identity4()
    c = math.cos(theta)
    s = math.sin(theta)
    return c * I4 + (-1j * s) * Y2


def simulate_hamiltonian(n: int, trotter_steps: int = 20, time_total: float = 1.0) -> Dict[str, Any]:
    dev = Device(n)
    J = 1.0; h = 0.5; dt = time_total / float(trotter_steps)
    backend = os.environ.get('MLXQ_BACKEND', 'sv').lower()
    # Prefer MPS TEBD path when requested
    if backend == 'mps':
        t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
        Uzz = _zz_phase_gate(-dt*J)
        Ux = RX(2.0*h*dt)
        pair_sw = os.environ.get('MLXQ_MPS_PAIR_SWEEPS', '0') == '1'
        # Optional MPO helpers (only ZZ is used here; XX/YY kept for parity with XXZ code)
        use_mpo_xx = os.environ.get('MLXQ_MPS_USE_MPO_XX', '0') == '1'
        use_mpo_yy = os.environ.get('MLXQ_MPS_USE_MPO_YY', '0') == '1'
        use_mpo_zz = os.environ.get('MLXQ_MPS_USE_MPO_ZZ', '0') == '1'
        # TEBD sweep per Trotter step: ZZ nearest-neighbor then local X field
        stop_on_trunc = os.environ.get('MLXQ_MPS_STOP_ON_TRUNC', '0') == '1'
        stopped_on_trunc = False
        # Ensure sim is in scope even if helper path raises
        sim = dev.sim
        for _ in range(trotter_steps):
            # Sequential sweep to match dense schedule ordering
            # (0,1),(1,2),..., then 1q X rotations
            try:
                # Use MPSState helpers when available
                if hasattr(sim, 'apply_two_sweep') and hasattr(sim, 'apply_single_all'):
                    if use_mpo_zz and hasattr(sim, 'apply_zz_two_sweep'):
                        sim.apply_zz_two_sweep(-dt*J)
                    elif pair_sw and hasattr(sim, 'apply_two_all_pairs'):
                        sim.apply_two_all_pairs(Uzz, offset=0)
                        sim.apply_two_all_pairs(Uzz, offset=1)
                    else:
                        sim.apply_two_sweep(Uzz)
                    sim.apply_single_all(Ux)
                else:
                    # Fallback: per-bond and per-site operations
                    for i in range(n-1):
                        dev.sim.apply_two(Uzz, i, i+1)
                    for q in range(n):
                        dev.sim.apply_single(Ux, q)
            except Exception:
                # Conservative fallback to dense scheduling
                for i in range(n-1):
                    dev.sim.apply_two(Uzz, i, i+1)
                for q in range(n):
                    dev.sim.apply_single(Ux, q)
            if stop_on_trunc and bool(getattr(sim, 'truncated', lambda: False)()):
                stopped_on_trunc = True
                break
        _ = dev.sim.probabilities()
        res = _bench_result("hamiltonian_simulation", n, trotter_steps*((n-1)+n), t0, c0, m0)
        # Attach MPS diagnostics when available
        try:
            sim = dev.sim
            if hasattr(sim, 'bond_max') and hasattr(sim, 'bond_mean'):
                res["mps"] = {
                    "bond_max": int(sim.bond_max()),
                    "bond_mean": float(sim.bond_mean()),
                    "truncations": int(getattr(sim, 'trunc_count', lambda: 0)()),
                    "truncated": bool(getattr(sim, 'truncated', lambda: False)()),
                    "bond_dims": getattr(sim, 'bond_dims', lambda: [])(),
                    "stopped_on_trunc": stopped_on_trunc,
                }
        except Exception:
            pass
        return res

    # Dense/SV path (legacy schedule)
    ops = []
    for _ in range(trotter_steps):
        for i in range(n-1):
            ops.append({"name": "CUSTOM2", "wires": [i, i+1], "matrix": _zz_phase_gate(-dt*J)})
        for q in range(n):
            ops.append({"name": "RX", "wires": [q], "parameters": [2.0*h*dt]})
    t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
    for op in ops:
        if op.get("name") == "CUSTOM2":
            dev.sim.apply_two(op["matrix"], op["wires"][0], op["wires"][1])
        else:
            dev.execute([op])
    _ = dev.sim.probabilities()
    return _bench_result("hamiltonian_simulation", n, len(ops), t0, c0, m0)

def simulate_heisenberg(n: int, J: float = 1.0, trotter_steps: int = 20, time_total: float = 1.0) -> Dict[str, Any]:
    dev = Device(n)
    dt = time_total / float(trotter_steps)
    backend = os.environ.get('MLXQ_BACKEND', 'sv').lower()
    if backend == 'mps':
        t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
        Uxx = _xx_phase_gate(-dt*J)
        Uyy = _yy_phase_gate(-dt*J)
        Uzz = _zz_phase_gate(-dt*J)
        pair_sw = os.environ.get('MLXQ_MPS_PAIR_SWEEPS', '0') == '1'
        use_mpo_xx = os.environ.get('MLXQ_MPS_USE_MPO_XX', '0') == '1'
        use_mpo_yy = os.environ.get('MLXQ_MPS_USE_MPO_YY', '0') == '1'
        steps_done = 0
        early_stop = False
        stop_bmax = int(os.environ.get('MLXQ_MPS_EARLY_STOP_BMAX', '0'))
        stop_on_trunc = os.environ.get('MLXQ_MPS_STOP_ON_TRUNC', '0') == '1'
        stopped_on_trunc = False
        sim = dev.sim
        use_mpo_zz = os.environ.get('MLXQ_MPS_USE_MPO_ZZ', '0') == '1'
        for _ in range(trotter_steps):
            if use_mpo_xx and hasattr(sim, 'apply_xx_two_sweep'):
                sim.apply_xx_two_sweep(-dt*J)
            elif pair_sw and hasattr(sim, 'apply_two_all_pairs'):
                sim.apply_two_all_pairs(Uxx, 0); sim.apply_two_all_pairs(Uxx, 1)
            else:
                sim.apply_two_sweep(Uxx)
            if use_mpo_yy and hasattr(sim, 'apply_yy_two_sweep'):
                sim.apply_yy_two_sweep(-dt*J)
            elif pair_sw and hasattr(sim, 'apply_two_all_pairs'):
                sim.apply_two_all_pairs(Uyy, 0); sim.apply_two_all_pairs(Uyy, 1)
            else:
                sim.apply_two_sweep(Uyy)
            if use_mpo_zz and hasattr(sim, 'apply_zz_two_sweep'):
                sim.apply_zz_two_sweep(-dt*J)
            elif pair_sw and hasattr(sim, 'apply_two_all_pairs'):
                sim.apply_two_all_pairs(Uzz, 0); sim.apply_two_all_pairs(Uzz, 1)
            else:
                sim.apply_two_sweep(Uzz)
            steps_done += 1
            if stop_bmax > 0 and hasattr(sim, 'bond_max') and int(sim.bond_max()) >= stop_bmax:
                early_stop = True
                break
            if stop_on_trunc and bool(getattr(sim, 'truncated', lambda: False)()):
                stopped_on_trunc = True
                break
        _ = dev.sim.probabilities()
        gates = steps_done * 3 * max(0, n - 1)
        res = _bench_result("heisenberg", n, gates, t0, c0, m0)
        try:
            if hasattr(sim, 'bond_max') and hasattr(sim, 'bond_mean'):
                res["mps"] = {
                    "bond_max": int(sim.bond_max()),
                    "bond_mean": float(sim.bond_mean()),
                    "truncations": int(getattr(sim, 'trunc_count', lambda: 0)()),
                    "truncated": bool(getattr(sim, 'truncated', lambda: False)()),
                    "early_stop": early_stop,
                    "stopped_on_trunc": stopped_on_trunc,
                    "bond_dims": getattr(sim, 'bond_dims', lambda: [])(),
                }
        except Exception:
            pass
        return res
    # Dense path
    ops = []
    for _ in range(trotter_steps):
        for i in range(n-1):
            ops.append({"name": "CUSTOM2", "wires": [i, i+1], "matrix": _xx_phase_gate(-dt*J)})
        for i in range(n-1):
            ops.append({"name": "CUSTOM2", "wires": [i, i+1], "matrix": _yy_phase_gate(-dt*J)})
        for i in range(n-1):
            ops.append({"name": "CUSTOM2", "wires": [i, i+1], "matrix": _zz_phase_gate(-dt*J)})
    t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
    for op in ops:
        if op.get("name") == "CUSTOM2":
            dev.sim.apply_two(op["matrix"], op["wires"][0], op["wires"][1])
        else:
            dev.execute([op])
    _ = dev.sim.probabilities()
    return _bench_result("heisenberg", n, len(ops), t0, c0, m0)

def simulate_heisenberg_xxz(n: int, Jx: float = 1.0, Jy: float = 1.0, Jz: float = 1.0,
                            trotter_steps: int = 20, time_total: float = 1.0) -> Dict[str, Any]:
    dev = Device(n)
    dt = time_total / float(trotter_steps)
    backend = os.environ.get('MLXQ_BACKEND', 'sv').lower()
    if backend == 'mps':
        t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
        Ux = _xx_phase_gate(-dt*Jx) if abs(Jx) > 0 else None
        Uy = _yy_phase_gate(-dt*Jy) if abs(Jy) > 0 else None
        Uz = _zz_phase_gate(-dt*Jz) if abs(Jz) > 0 else None
        sim = dev.sim
        pair_sw = os.environ.get('MLXQ_MPS_PAIR_SWEEPS', '0') == '1'
        use_mpo_xx = os.environ.get('MLXQ_MPS_USE_MPO_XX', '0') == '1'
        use_mpo_yy = os.environ.get('MLXQ_MPS_USE_MPO_YY', '0') == '1'
        steps_done = 0
        early_stop = False
        stop_bmax = int(os.environ.get('MLXQ_MPS_EARLY_STOP_BMAX', '0'))
        stop_on_trunc = os.environ.get('MLXQ_MPS_STOP_ON_TRUNC', '0') == '1'
        stopped_on_trunc = False
        use_mpo_zz = os.environ.get('MLXQ_MPS_USE_MPO_ZZ', '0') == '1'
        for _ in range(trotter_steps):
            if Ux is not None:
                if use_mpo_xx and hasattr(sim, 'apply_xx_two_sweep'):
                    sim.apply_xx_two_sweep(-dt*Jx)
                elif pair_sw and hasattr(sim, 'apply_two_all_pairs'):
                    sim.apply_two_all_pairs(Ux, 0); sim.apply_two_all_pairs(Ux, 1)
                else:
                    sim.apply_two_sweep(Ux)
            if Uy is not None:
                if use_mpo_yy and hasattr(sim, 'apply_yy_two_sweep'):
                    sim.apply_yy_two_sweep(-dt*Jy)
                elif pair_sw and hasattr(sim, 'apply_two_all_pairs'):
                    sim.apply_two_all_pairs(Uy, 0); sim.apply_two_all_pairs(Uy, 1)
                else:
                    sim.apply_two_sweep(Uy)
            if Uz is not None:
                if use_mpo_zz and hasattr(sim, 'apply_zz_two_sweep'):
                    sim.apply_zz_two_sweep(-dt*Jz)
                elif pair_sw and hasattr(sim, 'apply_two_all_pairs'):
                    sim.apply_two_all_pairs(Uz, 0); sim.apply_two_all_pairs(Uz, 1)
                else:
                    sim.apply_two_sweep(Uz)
            steps_done += 1
            if stop_bmax > 0 and hasattr(sim, 'bond_max') and int(sim.bond_max()) >= stop_bmax:
                early_stop = True
                break
            if stop_on_trunc and bool(getattr(sim, 'truncated', lambda: False)()):
                stopped_on_trunc = True
                break
        _ = dev.sim.probabilities()
        k = (1 if Ux is not None else 0) + (1 if Uy is not None else 0) + (1 if Uz is not None else 0)
        gates = steps_done * k * max(0, n - 1)
        res = _bench_result("heisenberg_xxz", n, gates, t0, c0, m0)
        try:
            if hasattr(sim, 'bond_max') and hasattr(sim, 'bond_mean'):
                res["mps"] = {
                    "bond_max": int(sim.bond_max()),
                    "bond_mean": float(sim.bond_mean()),
                    "truncations": int(getattr(sim, 'trunc_count', lambda: 0)()),
                    "truncated": bool(getattr(sim, 'truncated', lambda: False)()),
                    "early_stop": early_stop,
                    "stopped_on_trunc": stopped_on_trunc,
                    "bond_dims": getattr(sim, 'bond_dims', lambda: [])(),
                }
        except Exception:
            pass
        return res
    # Dense path
    ops = []
    for _ in range(trotter_steps):
        if abs(Jx) > 0:
            for i in range(n-1):
                ops.append({"name": "CUSTOM2", "wires": [i, i+1], "matrix": _xx_phase_gate(-dt*Jx)})
        if abs(Jy) > 0:
            for i in range(n-1):
                ops.append({"name": "CUSTOM2", "wires": [i, i+1], "matrix": _yy_phase_gate(-dt*Jy)})
        if abs(Jz) > 0:
            for i in range(n-1):
                ops.append({"name": "CUSTOM2", "wires": [i, i+1], "matrix": _zz_phase_gate(-dt*Jz)})
    t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
    for op in ops:
        if op.get("name") == "CUSTOM2":
            dev.sim.apply_two(op["matrix"], op["wires"][0], op["wires"][1])
        else:
            dev.execute([op])
    _ = dev.sim.probabilities()
    return _bench_result("heisenberg_xxz", n, len(ops), t0, c0, m0)

def simulate_tfim(n: int, J: float = 1.0, h: float = 0.5, trotter_steps: int = 20,
                  time_total: float = 1.0) -> Dict[str, Any]:
    # Alias of simulate_hamiltonian with explicit params
    return simulate_hamiltonian(n, trotter_steps=trotter_steps, time_total=time_total)

def simulate_tfim_trotter2(n: int, J: float = 1.0, h: float = 0.5, trotter_steps: int = 20,
                           time_total: float = 1.0) -> Dict[str, Any]:
    dev = Device(n)
    dt = time_total / float(trotter_steps)
    backend = os.environ.get('MLXQ_BACKEND', 'sv').lower()
    if backend == 'mps':
        t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
        Uzz = _zz_phase_gate(-dt*J)
        Ux_half = RX(2.0*h*(dt*0.5))
        sim = dev.sim
        pair_sw = os.environ.get('MLXQ_MPS_PAIR_SWEEPS', '0') == '1'
        use_mpo_zz = os.environ.get('MLXQ_MPS_USE_MPO_ZZ', '0') == '1'
        steps_done = 0
        early_stop = False
        stop_bmax = int(os.environ.get('MLXQ_MPS_EARLY_STOP_BMAX', '0'))
        stop_on_trunc = os.environ.get('MLXQ_MPS_STOP_ON_TRUNC', '0') == '1'
        stopped_on_trunc = False
        for _ in range(trotter_steps):
            sim.apply_single_all(Ux_half)
            if use_mpo_zz and hasattr(sim, 'apply_zz_two_sweep'):
                sim.apply_zz_two_sweep(-dt*J)
            elif pair_sw and hasattr(sim, 'apply_two_all_pairs'):
                sim.apply_two_all_pairs(Uzz, 0); sim.apply_two_all_pairs(Uzz, 1)
            else:
                sim.apply_two_sweep(Uzz)
            sim.apply_single_all(Ux_half)
            steps_done += 1
            if stop_bmax > 0 and hasattr(sim, 'bond_max') and int(sim.bond_max()) >= stop_bmax:
                early_stop = True
                break
            if stop_on_trunc and bool(getattr(sim, 'truncated', lambda: False)()):
                stopped_on_trunc = True
                break
        _ = dev.sim.probabilities()
        gates = steps_done * ((n) + (n-1) + (n))
        res = _bench_result("tfim_trotter2", n, gates, t0, c0, m0)
        try:
            if hasattr(sim, 'bond_max') and hasattr(sim, 'bond_mean'):
                res["mps"] = {
                    "bond_max": int(sim.bond_max()),
                    "bond_mean": float(sim.bond_mean()),
                    "truncations": int(getattr(sim, 'trunc_count', lambda: 0)()),
                    "truncated": bool(getattr(sim, 'truncated', lambda: False)()),
                    "early_stop": early_stop,
                    "stopped_on_trunc": stopped_on_trunc,
                    "bond_dims": getattr(sim, 'bond_dims', lambda: [])(),
                }
        except Exception:
            pass
        return res
    # Dense path
    ops = []
    for _ in range(trotter_steps):
        for q in range(n):
            ops.append({"name": "RX", "wires": [q], "parameters": [2.0*h*(dt*0.5)]})
        for i in range(n-1):
            ops.append({"name": "CUSTOM2", "wires": [i, i+1], "matrix": _zz_phase_gate(-dt*J)})
        for q in range(n):
            ops.append({"name": "RX", "wires": [q], "parameters": [2.0*h*(dt*0.5)]})
    t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
    for op in ops:
        if op.get("name") == "CUSTOM2":
            dev.sim.apply_two(op["matrix"], op["wires"][0], op["wires"][1])
        else:
            dev.execute([op])
    _ = dev.sim.probabilities()
    return _bench_result("tfim_trotter2", n, len(ops), t0, c0, m0)

def simulate_tfim_random_field(n: int, J: float = 1.0, h_min: float = 0.2, h_max: float = 0.8,
                               trotter_steps: int = 20, time_total: float = 1.0,
                               seed: int = 1337) -> Dict[str, Any]:
    dev = Device(n)
    rnd = random.Random(seed + n)
    hs = [rnd.uniform(h_min, h_max) for _ in range(n)]
    dt = time_total / float(trotter_steps)
    backend = os.environ.get('MLXQ_BACKEND', 'sv').lower()
    if backend == 'mps':
        t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
        Uzz = _zz_phase_gate(-dt*J)
        sim = dev.sim
        pair_sw = os.environ.get('MLXQ_MPS_PAIR_SWEEPS', '0') == '1'
        use_mpo_zz = os.environ.get('MLXQ_MPS_USE_MPO_ZZ', '0') == '1'
        steps_done = 0
        early_stop = False
        stop_bmax = int(os.environ.get('MLXQ_MPS_EARLY_STOP_BMAX', '0'))
        stop_on_trunc = os.environ.get('MLXQ_MPS_STOP_ON_TRUNC', '0') == '1'
        stopped_on_trunc = False
        for _ in range(trotter_steps):
            if use_mpo_zz and hasattr(sim, 'apply_zz_two_sweep'):
                sim.apply_zz_two_sweep(-dt*J)
            elif pair_sw and hasattr(sim, 'apply_two_all_pairs'):
                sim.apply_two_all_pairs(Uzz, 0); sim.apply_two_all_pairs(Uzz, 1)
            else:
                sim.apply_two_sweep(Uzz)
            for q in range(n):
                sim.apply_single(RX(2.0*hs[q]*dt), q)
            steps_done += 1
            if stop_bmax > 0 and hasattr(sim, 'bond_max') and int(sim.bond_max()) >= stop_bmax:
                early_stop = True
                break
            if stop_on_trunc and bool(getattr(sim, 'truncated', lambda: False)()):
                stopped_on_trunc = True
                break
        _ = dev.sim.probabilities()
        gates = steps_done * ((n-1) + n)
        res = _bench_result("tfim_random_field", n, gates, t0, c0, m0)
        try:
            if hasattr(sim, 'bond_max') and hasattr(sim, 'bond_mean'):
                res["mps"] = {
                    "bond_max": int(sim.bond_max()),
                    "bond_mean": float(sim.bond_mean()),
                    "truncations": int(getattr(sim, 'trunc_count', lambda: 0)()),
                    "truncated": bool(getattr(sim, 'truncated', lambda: False)()),
                    "early_stop": early_stop,
                    "stopped_on_trunc": stopped_on_trunc,
                    "bond_dims": getattr(sim, 'bond_dims', lambda: [])(),
                }
        except Exception:
            pass
        return res
    # Dense path
    ops = []
    for _ in range(trotter_steps):
        for i in range(n-1):
            ops.append({"name": "CUSTOM2", "wires": [i, i+1], "matrix": _zz_phase_gate(-dt*J)})
        for q in range(n):
            ops.append({"name": "RX", "wires": [q], "parameters": [2.0*hs[q]*dt]})
    t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
    for op in ops:
        if op.get("name") == "CUSTOM2":
            dev.sim.apply_two(op["matrix"], op["wires"][0], op["wires"][1])
        else:
            dev.execute([op])
    _ = dev.sim.probabilities()
    return _bench_result("tfim_random_field", n, len(ops), t0, c0, m0)

def simulate_heisenberg_random_field(n: int, J: float = 1.0, h_min: float = -0.5, h_max: float = 0.5,
                                     trotter_steps: int = 20, time_total: float = 1.0,
                                     seed: int = 1337) -> Dict[str, Any]:
    dev = Device(n)
    rnd = random.Random(seed + 10*n)
    hz = [rnd.uniform(h_min, h_max) for _ in range(n)]
    dt = time_total / float(trotter_steps)
    backend = os.environ.get('MLXQ_BACKEND', 'sv').lower()
    if backend == 'mps':
        t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
        Uxx = _xx_phase_gate(-dt*J)
        Uyy = _yy_phase_gate(-dt*J)
        Uzz = _zz_phase_gate(-dt*J)
        sim = dev.sim
        pair_sw = os.environ.get('MLXQ_MPS_PAIR_SWEEPS', '0') == '1'
        steps_done = 0
        early_stop = False
        stop_bmax = int(os.environ.get('MLXQ_MPS_EARLY_STOP_BMAX', '0'))
        stop_on_trunc = os.environ.get('MLXQ_MPS_STOP_ON_TRUNC', '0') == '1'
        stopped_on_trunc = False
        for _ in range(trotter_steps):
            if pair_sw and hasattr(sim, 'apply_two_all_pairs'):
                sim.apply_two_all_pairs(Uxx, 0); sim.apply_two_all_pairs(Uxx, 1)
                sim.apply_two_all_pairs(Uyy, 0); sim.apply_two_all_pairs(Uyy, 1)
                sim.apply_two_all_pairs(Uzz, 0); sim.apply_two_all_pairs(Uzz, 1)
            else:
                sim.apply_two_sweep(Uxx)
                sim.apply_two_sweep(Uyy)
                sim.apply_two_sweep(Uzz)
            for q in range(n):
                sim.apply_single(RZ(2.0*hz[q]*dt), q)
            steps_done += 1
            if stop_bmax > 0 and hasattr(sim, 'bond_max') and int(sim.bond_max()) >= stop_bmax:
                early_stop = True
                break
            if stop_on_trunc and bool(getattr(sim, 'truncated', lambda: False)()):
                stopped_on_trunc = True
                break
        _ = dev.sim.probabilities()
        gates = steps_done * (3*(n-1) + n)
        res = _bench_result("heisenberg_random_field", n, gates, t0, c0, m0)
        try:
            if hasattr(sim, 'bond_max') and hasattr(sim, 'bond_mean'):
                res["mps"] = {
                    "bond_max": int(sim.bond_max()),
                    "bond_mean": float(sim.bond_mean()),
                    "truncations": int(getattr(sim, 'trunc_count', lambda: 0)()),
                    "truncated": bool(getattr(sim, 'truncated', lambda: False)()),
                    "early_stop": early_stop,
                    "stopped_on_trunc": stopped_on_trunc,
                    "bond_dims": getattr(sim, 'bond_dims', lambda: [])(),
                }
        except Exception:
            pass
        return res
    # Dense path
    ops = []
    for _ in range(trotter_steps):
        for i in range(n-1): ops.append({"name": "CUSTOM2", "wires": [i, i+1], "matrix": _xx_phase_gate(-dt*J)})
        for i in range(n-1): ops.append({"name": "CUSTOM2", "wires": [i, i+1], "matrix": _yy_phase_gate(-dt*J)})
        for i in range(n-1): ops.append({"name": "CUSTOM2", "wires": [i, i+1], "matrix": _zz_phase_gate(-dt*J)})
        for q in range(n):
            ops.append({"name": "RZ", "wires": [q], "parameters": [2.0*hz[q]*dt]})
    t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
    for op in ops:
        if op.get("name") == "CUSTOM2":
            dev.sim.apply_two(op["matrix"], op["wires"][0], op["wires"][1])
        else:
            dev.execute([op])
    _ = dev.sim.probabilities()
    return _bench_result("heisenberg_random_field", n, len(ops), t0, c0, m0)

def simulate_long_range_ising(n: int, J: float = 1.0, alpha: float = 2.0,
                              trotter_steps: int = 12, time_total: float = 1.0) -> Dict[str, Any]:
    dev = Device(n)
    dt = time_total / float(trotter_steps)
    backend = os.environ.get('MLXQ_BACKEND', 'sv').lower()
    if backend == 'mps':
        t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
        sim = dev.sim
        steps_done = 0
        early_stop = False
        stop_bmax = int(os.environ.get('MLXQ_MPS_EARLY_STOP_BMAX', '0'))
        stop_on_trunc = os.environ.get('MLXQ_MPS_STOP_ON_TRUNC', '0') == '1'
        stopped_on_trunc = False
        for _ in range(trotter_steps):
            for i in range(n-1):
                for j in range(i+1, n):
                    dist = float(j - i)
                    Jij = J / (dist ** max(1e-6, alpha))
                    Uzz = _zz_phase_gate(-dt*Jij)
                    sim.apply_two(Uzz, i, j)
            steps_done += 1
            if stop_bmax > 0 and hasattr(sim, 'bond_max') and int(sim.bond_max()) >= stop_bmax:
                early_stop = True
                break
            if stop_on_trunc and bool(getattr(sim, 'truncated', lambda: False)()):
                stopped_on_trunc = True
                break
        _ = dev.sim.probabilities()
        pairs = (n * (n - 1)) // 2
        gates = steps_done * pairs
        res = _bench_result("long_range_ising", n, gates, t0, c0, m0)
        try:
            if hasattr(sim, 'bond_max') and hasattr(sim, 'bond_mean'):
                res["mps"] = {
                    "bond_max": int(sim.bond_max()),
                    "bond_mean": float(sim.bond_mean()),
                    "truncations": int(getattr(sim, 'trunc_count', lambda: 0)()),
                    "truncated": bool(getattr(sim, 'truncated', lambda: False)()),
                    "early_stop": early_stop,
                    "stopped_on_trunc": stopped_on_trunc,
                    "bond_dims": getattr(sim, 'bond_dims', lambda: [])(),
                }
        except Exception:
            pass
        return res
    # Dense path
    dev = Device(n)
    ops = []
    for _ in range(trotter_steps):
        for i in range(n-1):
            for j in range(i+1, n):
                dist = float(j - i)
                Jij = J / (dist ** max(1e-6, alpha))
                ops.append({"name": "CUSTOM2", "wires": [i, j], "matrix": _zz_phase_gate(-dt*Jij)})
    t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
    for op in ops:
        if op.get("name") == "CUSTOM2":
            dev.sim.apply_two(op["matrix"], op["wires"][0], op["wires"][1])
        else:
            dev.execute([op])
    _ = dev.sim.probabilities()
    return _bench_result("long_range_ising", n, len(ops), t0, c0, m0)

def simulate_ladder_heisenberg(n: int, J: float = 1.0, Jr: float = 1.0, trotter_steps: int = 12,
                               time_total: float = 1.0) -> Dict[str, Any]:
    # 2×L ladder with open boundary (n even)
    dev = Device(n)
    backend = os.environ.get('MLXQ_BACKEND', 'sv').lower()
    L = n // 2
    dt = time_total / float(trotter_steps)
    if backend == 'mps':
        t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
        Ux_leg = _xx_phase_gate(-dt*J)
        Uy_leg = _yy_phase_gate(-dt*J)
        Uz_leg = _zz_phase_gate(-dt*J)
        Ux_rung = _xx_phase_gate(-dt*Jr)
        Uy_rung = _yy_phase_gate(-dt*Jr)
        Uz_rung = _zz_phase_gate(-dt*Jr)
        steps_done = 0
        early_stop = False
        stop_bmax = int(os.environ.get('MLXQ_MPS_EARLY_STOP_BMAX', '0'))
        stop_on_trunc = os.environ.get('MLXQ_MPS_STOP_ON_TRUNC', '0') == '1'
        stopped_on_trunc = False
        sim = dev.sim
        for _ in range(trotter_steps):
            # Legs
            for row_offset in (0, L):
                for i in range(L-1):
                    a, b = row_offset + i, row_offset + i + 1
                    sim.apply_two(Ux_leg, a, b)
                    sim.apply_two(Uy_leg, a, b)
                    sim.apply_two(Uz_leg, a, b)
            # Rungs
            for i in range(L):
                a, b = i, i + L
                sim.apply_two(Ux_rung, a, b)
                sim.apply_two(Uy_rung, a, b)
                sim.apply_two(Uz_rung, a, b)
            steps_done += 1
            if stop_bmax > 0 and hasattr(sim, 'bond_max') and int(sim.bond_max()) >= stop_bmax:
                early_stop = True
                break
            if stop_on_trunc and bool(getattr(sim, 'truncated', lambda: False)()):
                stopped_on_trunc = True
                break
        _ = dev.sim.probabilities()
        per_step_ops = 3 * (2 * (L - 1) + L)  # XX/YY/ZZ for each leg/rung pair per step
        gates = steps_done * per_step_ops
        res = _bench_result("ladder_heisenberg", n, gates, t0, c0, m0)
        try:
            if hasattr(sim, 'bond_max') and hasattr(sim, 'bond_mean'):
                res["mps"] = {
                    "bond_max": int(sim.bond_max()),
                    "bond_mean": float(sim.bond_mean()),
                    "truncations": int(getattr(sim, 'trunc_count', lambda: 0)()),
                    "truncated": bool(getattr(sim, 'truncated', lambda: False)()),
                    "early_stop": early_stop,
                    "stopped_on_trunc": stopped_on_trunc,
                    "bond_dims": getattr(sim, 'bond_dims', lambda: [])(),
                }
        except Exception:
            pass
        return res
    # Dense path
    dev = Device(n)
    ops = []
    for _ in range(trotter_steps):
        for row_offset in (0, L):
            for i in range(L-1):
                a, b = row_offset + i, row_offset + i + 1
                ops.append({"name": "CUSTOM2", "wires": [a, b], "matrix": _xx_phase_gate(-dt*J)})
                ops.append({"name": "CUSTOM2", "wires": [a, b], "matrix": _yy_phase_gate(-dt*J)})
                ops.append({"name": "CUSTOM2", "wires": [a, b], "matrix": _zz_phase_gate(-dt*J)})
        for i in range(L):
            a, b = i, i + L
            ops.append({"name": "CUSTOM2", "wires": [a, b], "matrix": _xx_phase_gate(-dt*Jr)})
            ops.append({"name": "CUSTOM2", "wires": [a, b], "matrix": _yy_phase_gate(-dt*Jr)})
            ops.append({"name": "CUSTOM2", "wires": [a, b], "matrix": _zz_phase_gate(-dt*Jr)})
    t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
    for op in ops:
        if op.get("name") == "CUSTOM2":
            dev.sim.apply_two(op["matrix"], op["wires"][0], op["wires"][1])
        else:
            dev.execute([op])
    _ = dev.sim.probabilities()
    return _bench_result("ladder_heisenberg", n, len(ops), t0, c0, m0)


def _simulate_dispatch(circuit_type: str, n: int):
    if circuit_type == "vqe":
        return simulate_vqe(n)
    if circuit_type == "qcbm":
        return simulate_qcbm(n)
    if circuit_type == "variational_circuit":
        return simulate_variational(n)
    if circuit_type == "hamiltonian_simulation":
        return simulate_hamiltonian(n)
    if circuit_type == "random_circuit":
        return simulate_random_circuit(n)
    if circuit_type == "qaoa":
        return simulate_qaoa(n)
    if circuit_type == "cuquantum_blueqat":
        return simulate_cuquantum_blueqat(n)
    if circuit_type == "grover":
        return simulate_grover(n)
    if circuit_type == "trotter":
        return simulate_trotter(n)
    if circuit_type == "time_evolution":
        return simulate_hamiltonian(n, trotter_steps=12)
    if circuit_type == "steady_state":
        return simulate_steady_state(n)
    if circuit_type == "heisenberg":
        return simulate_heisenberg(n)
    if circuit_type == "qft":
        return simulate_qft(n)
    if circuit_type == "phase_estimation":
        return simulate_phase_estimation(n)
    if circuit_type == "ghz":
        return simulate_ghz(n)
    if circuit_type == "heisenberg_xxz":
        return simulate_heisenberg_xxz(n)
    if circuit_type == "tfim":
        return simulate_tfim(n)
    if circuit_type == "tfim_trotter2":
        return simulate_tfim_trotter2(n)
    if circuit_type == "tfim_random_field":
        return simulate_tfim_random_field(n)
    if circuit_type == "heisenberg_random_field":
        return simulate_heisenberg_random_field(n)
    if circuit_type == "long_range_ising":
        return simulate_long_range_ising(n)
    if circuit_type == "ladder_heisenberg":
        return simulate_ladder_heisenberg(n)
    if circuit_type == "deutsch_jozsa":
        return simulate_deutsch_jozsa(n)
    if circuit_type == "graph_state":
        return simulate_graph_state(n)
    if circuit_type == "qft_entangled":
        return simulate_qft_entangled(n)
    if circuit_type == "phase_estimation_inexact":
        return simulate_phase_estimation_inexact(n)
    # MQTBench aliases and additional workloads (for paper 2504 parity)
    if circuit_type == "qpeexact":
        return simulate_phase_estimation(n)
    if circuit_type == "qpeinexact":
        return simulate_phase_estimation_inexact(n)
    if circuit_type == "qftentangled":
        return simulate_qft_entangled(n)
    if circuit_type == "graphstate":
        return simulate_graph_state(n)
    if circuit_type == "qwalk":
        return simulate_quantum_walk_vchain(n)
    if circuit_type == "random":
        return simulate_random_circuit(n)
    if circuit_type == "realamp":
        return simulate_realamp(n)
    if circuit_type == "su2rand":
        return simulate_su2rand(n)
    if circuit_type == "qnn":
        return simulate_qnn(n)
    if circuit_type == "wstate":
        return simulate_wstate(n)
    if circuit_type == "ae":
        return simulate_ae(n)
    if circuit_type == "quantum_walk":
        return simulate_quantum_walk(n)
    if circuit_type == "quantum_walk_vchain":
        return simulate_quantum_walk_vchain(n)
    warn(f"{circuit_type} {n}q: unsupported")
    return None


def simulate_trotter(n: int) -> Dict[str, Any]:
    return simulate_hamiltonian(n, trotter_steps=20, time_total=1.0) | {"name": "trotter"}


def _identity2():
    return mx.array([[1+0j,0+0j],[0+0j,1+0j]], mx.complex64)


# ---------------------------------------------------------------------------
# MQTBench-style scalable circuits (new)
# ---------------------------------------------------------------------------

def simulate_deutsch_jozsa(n: int) -> Dict[str, Any]:
    """Deutsch–Jozsa on n inputs with a fixed balanced oracle.
    Oracle: flip phase on states with parity of first two bits (if n>=2), else identity.
    """
    dev = Device(n)
    ops = []
    # Prepare superposition
    for q in range(n):
        ops.append({"name": "H", "wires": [q]})
    # Balanced oracle proxy: CZ(0,1) and X on 0 (for n>=2)
    if n >= 2:
        ops.append({"name": "CZ", "wires": [0,1]})
        ops.append({"name": "X", "wires": [0]})
    # Interference
    for q in range(n):
        ops.append({"name": "H", "wires": [q]})
    t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
    dev.execute(ops)
    _ = dev.sim.probabilities()
    return _bench_result("deutsch_jozsa", n, len(ops), t0, c0, m0)


def simulate_graph_state(n: int, ring: bool = True) -> Dict[str, Any]:
    dev = Device(n)
    ops = []
    for q in range(n):
        ops.append({"name": "H", "wires": [q]})
    for q in range(n-1):
        ops.append({"name": "CZ", "wires": [q, q+1]})
    if ring and n > 2:
        ops.append({"name": "CZ", "wires": [n-1, 0]})
    t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
    dev.execute(ops); _ = dev.sim.probabilities()
    return _bench_result("graph_state", n, len(ops), t0, c0, m0)


def simulate_qft_entangled(n: int) -> Dict[str, Any]:
    dev = Device(n)
    ops = []
    # Entangle (GHZ-like)
    ops.append({"name": "H", "wires": [0]})
    for q in range(n-1):
        ops.append({"name": "CNOT", "wires": [q, q+1]})
    # Then QFT on all qubits
    for j in range(n):
        ops.append({"name": "H", "wires": [j]})
        for k in range(j + 1, n):
            phi = math.pi / (2 ** (k - j))
            ops.append({"name": "CPHASE", "wires": [k, j], "parameters": [phi]})
    t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
    dev.execute(ops); _ = dev.sim.probabilities()
    return _bench_result("qft_entangled", n, len(ops), t0, c0, m0)


def simulate_phase_estimation_inexact(n: int) -> Dict[str, Any]:
    """Approximate QPE: only include controlled phases up to a limited order.
    We truncate long-range controlled phases to emulate finite precision.
    """
    dev = Device(n)
    ops = []
    # Prepare superposition on all qubits
    for q in range(n):
        ops.append({"name": "H", "wires": [q]})
    # Truncated controlled phases (only nearest 2 neighbors)
    for j in range(n):
        for k in range(j + 1, min(n, j + 3)):
            phi = math.pi / (2 ** (k - j))
            ops.append({"name": "CPHASE", "wires": [k, j], "parameters": [phi]})
    # Cheap "inverse QFT-like" (only H)
    for q in range(n):
        ops.append({"name": "H", "wires": [q]})
    t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
    dev.execute(ops); _ = dev.sim.probabilities()
    return _bench_result("phase_estimation_inexact", n, len(ops), t0, c0, m0)


def simulate_ae(n: int) -> Dict[str, Any]:
    """Toy Amplitude Estimation: prepare biased state, apply k Grover-like iterations.
    This is a scalable proxy just for benchmarking structure.
    """
    dev = Device(n)
    ops = []
    # Prepare biased |ψ> via small RY on qubit 0
    theta = 0.2
    ops.append({"name": "RY", "wires": [0], "parameters": [theta]})
    # k iterations scale with n
    k = max(1, n // 2)
    for _ in range(k):
        # Diffusion proxy on first 2 qubits
        if n >= 2:
            ops.append({"name": "H", "wires": [0]}); ops.append({"name": "H", "wires": [1]})
            ops.append({"name": "CZ", "wires": [0, 1]})
            ops.append({"name": "H", "wires": [0]}); ops.append({"name": "H", "wires": [1]})
        else:
            ops.append({"name": "Z", "wires": [0]})
    t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
    dev.execute(ops); _ = dev.sim.probabilities()
    return _bench_result("ae", n, len(ops), t0, c0, m0)


def simulate_quantum_walk(n: int) -> Dict[str, Any]:
    """Quantum walk proxy: alternate coin (H on all) and nearest-neighbor CZ."""
    dev = Device(n)
    ops = []
    steps = max(1, n // 2)
    for _ in range(steps):
        for q in range(n):
            ops.append({"name": "H", "wires": [q]})
        for q in range(n-1):
            ops.append({"name": "CZ", "wires": [q, q+1]})
    t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
    dev.execute(ops); _ = dev.sim.probabilities()
    return _bench_result("quantum_walk", n, len(ops), t0, c0, m0)


def simulate_quantum_walk_vchain(n: int) -> Dict[str, Any]:
    """Quantum walk 'v-chain' proxy: coin + brickwork entangling in alternating offsets."""
    dev = Device(n)
    ops = []
    steps = max(1, n // 2)
    for s in range(steps):
        for q in range(n):
            ops.append({"name": "H", "wires": [q]})
        offset = s % 2
        for i in range(offset, n-1, 2):
            ops.append({"name": "CNOT", "wires": [i, i+1]})
    t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
    dev.execute(ops); _ = dev.sim.probabilities()
    return _bench_result("quantum_walk_vchain", n, len(ops), t0, c0, m0)


# --- Additional MQTBench-style workloads for paper-2504 parity ---
def simulate_realamp(n: int) -> Dict[str, Any]:
    """VQE RealAmplitudes-style ansatz: layers of RY + CZ chain.

    This proxy matches the structural pattern used in many libraries: apply RY on all
    qubits with random angles, then CZ entangling along a 1D chain; repeat L times.
    """
    dev = Device(n)
    ops = []
    rnd = random.Random(1234 + n)
    L = max(1, min(4, n // 6))
    for _ in range(L):
        for q in range(n):
            ops.append({"name": "RY", "wires": [q], "parameters": [(rnd.random() - 0.5) * 2.0]})
        for q in range(n - 1):
            ops.append({"name": "CZ", "wires": [q, q + 1]})
    t0, c0, m0 = now_ms(), cpu_seconds(), peak_rss_mb()
    dev.execute(ops); _ = dev.sim.probabilities()
    return _bench_result("realamp", n, len(ops), t0, c0, m0)


def simulate_su2rand(n: int) -> Dict[str, Any]:
    """Hardware-efficient SU2-style random ansatz: RX/RZ singles + CZ brickwork."""
    dev = Device(n)
    ops = []
    rnd = random.Random(5678 + 3 * n)
    L = max(1, min(4, n // 6))
    for layer in range(L):
        for q in range(n):
            ops.append({"name": "RX", "wires": [q], "parameters": [(rnd.random() - 0.5) * 2.0]})
            ops.append({"name": "RZ", "wires": [q], "parameters": [(rnd.random() - 0.5) * 2.0]})
            ops.append({"name": "RX", "wires": [q], "parameters": [(rnd.random() - 0.5) * 2.0]})
        # alternating CZ pairs
        offset = layer % 2
        for i in range(offset, n - 1, 2):
            ops.append({"name": "CZ", "wires": [i, i + 1]})
    t0, c0, m0 = now_ms(), cpu_seconds(), peak_rss_mb()
    dev.execute(ops); _ = dev.sim.probabilities()
    return _bench_result("su2rand", n, len(ops), t0, c0, m0)


def simulate_qnn(n: int) -> Dict[str, Any]:
    """QNN proxy: ZZ feature map + RY/CNOT layers."""
    dev = Device(n)
    ops = []
    # Feature map: local RZ and ZZ couplings
    phi_base = 0.1
    for q in range(n):
        ops.append({"name": "RZ", "wires": [q], "parameters": [phi_base * (q + 1)]})
    for i in range(n - 1):
        ops.append({"name": "CUSTOM2", "wires": [i, i + 1], "matrix": _zz_phase_gate(0.2)})
    # Two hardware-efficient layers
    rnd = random.Random(9012 + 5 * n)
    for _ in range(2):
        for q in range(n):
            ops.append({"name": "RY", "wires": [q], "parameters": [(rnd.random() - 0.5) * 1.0]})
        for i in range(n - 1):
            ops.append({"name": "CNOT", "wires": [i, i + 1]})
    t0, c0, m0 = now_ms(), cpu_seconds(), peak_rss_mb()
    # Execute with CUSTOM2 handled via direct two-qubit apply
    for op in ops:
        if op.get("name") == "CUSTOM2":
            dev.sim.apply_two(op["matrix"], op["wires"][0], op["wires"][1])
        else:
            dev.execute([op])
    _ = dev.sim.probabilities()
    return _bench_result("qnn", n, len(ops), t0, c0, m0)


def simulate_wstate(n: int) -> Dict[str, Any]:
    """W-state proxy using linear-depth RY+CNOT ladder.

    This constructs a state with a single excitation distributed across the chain. It serves
    as a scalable benchmark with O(n) entangling structure.
    """
    dev = Device(n)
    ops = []
    if n <= 0:
        return {"error": "n<1", "name": "wstate", "qubits": n}
    # Ladder of RY rotations and CNOTs
    for i in range(n - 1):
        # Angle schedules that taper across the chain
        denom = max(1, n - i)
        theta = 2.0 * math.acos(math.sqrt(max(0.0, (denom - 1) / float(denom))))
        ops.append({"name": "RY", "wires": [i], "parameters": [theta]})
        ops.append({"name": "CNOT", "wires": [i, i + 1]})
    t0, c0, m0 = now_ms(), cpu_seconds(), peak_rss_mb()
    dev.execute(ops); _ = dev.sim.probabilities()
    return _bench_result("wstate", n, len(ops), t0, c0, m0)


def _reshape_bits_tensor(state: mx.array, n: int) -> mx.array:
    """Reshape state vector to tensor with one axis per qubit.

    Uses row-major mapping so axis (n-1-i) corresponds to bit i (LSB).
    """
    return mx.reshape(state, tuple([2] * n))


def _expectation_ZZ_from_probs(prob_tensor: mx.array, n: int, i: int, j: int) -> float:
    """Compute <Z_i Z_j> from probability tensor without building operators.

    Axis mapping: axis = n-1-i corresponds to qubit i.
    """
    axis_i = n - 1 - i
    axis_j = n - 1 - j
    s = mx.array([1.0, -1.0], mx.float32)
    # shape for broadcasting
    shape_i = [1] * n; shape_i[axis_i] = 2
    shape_j = [1] * n; shape_j[axis_j] = 2
    si = mx.reshape(s, tuple(shape_i))
    sj = mx.reshape(s, tuple(shape_j))
    w = prob_tensor * si * sj
    val = mx.sum(w)
    mx.eval(val)
    return float(val.item())


def _expectation_X_from_state(state: mx.array, n: int, i: int) -> float:
    """Compute <X_i> using pairwise amplitude overlaps (no dense operator).

    Reshapes psi into (2, -1) with bit i as first axis.
    """
    axis = n - 1 - i
    psi_t = _reshape_bits_tensor(state, n)
    # Build permutation placing axis first
    perm = list(range(n))
    perm.insert(0, perm.pop(axis))
    psi_perm = mx.transpose(psi_t, perm)
    psi2 = mx.reshape(psi_perm, (2, -1))
    psi0 = psi2[0, :]
    psi1 = psi2[1, :]
    s = mx.sum(mx.conjugate(psi0) * psi1)
    # 2 * Re(sum conj(psi0) psi1)
    val = 2.0 * mx.real(s)
    mx.eval(val)
    return float(val.item())


def _dense_state_from_sim(sim, n: int) -> mx.array:
    """Return a dense state vector for either SV or MPS simulators."""
    if hasattr(sim, 'state'):
        return sim.state
    # MPS-like fallback: contract tensors A[0] .. A[n-1]
    if hasattr(sim, 'A'):
        psi = sim.A[0]
        for i in range(1, n):
            psi = mx.tensordot(psi, sim.A[i], axes=([psi.ndim - 1], [0]))
        return mx.reshape(psi, (1 << n,))
    raise ValueError("Unknown simulator type: cannot obtain dense state")


def _vqe_build_ops_from_params(n: int, layers: int, params: List[float]) -> List[Dict[str, Any]]:
    """Hardware-efficient ansatz: for each layer, RY, RZ per qubit then a CNOT chain.
    params is length 2*n*layers (order: [layer][q][ry, rz])."""
    ops: List[Dict[str, Any]] = []
    idx = 0
    for l in range(layers):
        for q in range(n):
            theta_ry = float(params[idx]); idx += 1
            theta_rz = float(params[idx]); idx += 1
            ops.append({"name": "RY", "wires": [q], "parameters": [theta_ry]})
            ops.append({"name": "RZ", "wires": [q], "parameters": [theta_rz]})
        for q in range(n-1):
            ops.append({"name": "CNOT", "wires": [q, q+1]})
        if n > 2:
            ops.append({"name": "CNOT", "wires": [n-1, 0]})
    return ops


def _vqe_energy(n: int, layers: int, params: List[float], h_ops=None) -> float:
    """Energy for H = sum Z_i Z_{i+1} + sum X_i without dense operators.

    This avoids materializing 2^n x 2^n matrices to prevent O(4^n) memory.
    """
    dev = Device(n)
    ops = _vqe_build_ops_from_params(n, layers, params)
    dev.execute(ops)
    psi = _dense_state_from_sim(dev.sim, n)
    # Probabilities tensor for Z terms
    p = mx.abs(psi) ** 2
    p = mx.reshape(p, tuple([2] * n))
    # Ising model with J=1.0, h=0.5 and negative sign convention (match legacy):
    # H = -J ∑ Z_i Z_{i+1} - h ∑ X_i
    e = 0.0
    J = 1.0
    h = 0.5
    for i in range(n-1):
        e += -J * _expectation_ZZ_from_probs(p, n, i, i+1)
    for i in range(n):
        e += -h * _expectation_X_from_state(psi, n, i)
    return float(e)


def _vqe_grad_parameter_shift(n: int, layers: int, params: List[float], h_ops=None, shift: float = math.pi/2.0) -> List[float]:
    """Parameter-shift gradient for RY/RZ-only ansatz."""
    base = list(params)
    grad = [0.0] * len(base)
    for i in range(len(base)):
        plus = base[:]; plus[i] += shift
        minus = base[:]; minus[i] -= shift
        e_plus = _vqe_energy(n, layers, plus, h_ops)
        e_minus = _vqe_energy(n, layers, minus, h_ops)
        grad[i] = 0.5 * (e_plus - e_minus)
    return grad


def simulate_vqe(n: int, layers: int = 3) -> Dict[str, Any]:
    """VQE with optional optimizer loop.

    If MLXQ_VQE_STEPS > 0, runs Adam (parameter-shift gradients). Otherwise single pass.
    Env vars: MLXQ_VQE_STEPS, MLXQ_VQE_LR, MLXQ_VQE_SHIFT, MLXQ_VQE_INIT='he'|'zero'
    """
    steps = _env_int('MLXQ_VQE_STEPS', 0)
    lr = float(os.environ.get('MLXQ_VQE_LR', '0.05'))
    shift = float(os.environ.get('MLXQ_VQE_SHIFT', str(math.pi/2.0)))
    init = os.environ.get('MLXQ_VQE_INIT', 'he')

    # Initial parameters
    import random as _rnd
    _rnd.seed(42)
    pcount = 2 * n * layers
    if init == 'zero':
        params = [0.0] * pcount
    else:
        params = [(_rnd.random() - 0.5) * 0.4 for _ in range(pcount)]

    if steps <= 0:
        # Single forward pass (matches previous behaviour)
        t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
        e = _vqe_energy(n, layers, params, None)
        # Count gates in one forward pass
        gates = len(_vqe_build_ops_from_params(n, layers, params))
        res = _bench_result("vqe", n, gates, t0, c0, m0)
        res["energy"] = e
        return res

    # Adam optimizer
    beta1 = 0.9; beta2 = 0.999; eps = 1e-8
    m = [0.0] * pcount; v = [0.0] * pcount
    history: List[Tuple[int, float]] = []
    t0, c0, m0 = now_ms(), cpu_seconds(), peak_rss_mb()
    for t in range(1, steps + 1):
        grad = _vqe_grad_parameter_shift(n, layers, params, None, shift)
        # Adam update
        for i, g in enumerate(grad):
            m[i] = beta1 * m[i] + (1 - beta1) * g
            v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
            mhat = m[i] / (1 - (beta1 ** t))
            vhat = v[i] / (1 - (beta2 ** t))
            params[i] -= lr * mhat / (math.sqrt(vhat) + eps)
        if t % max(1, steps // 10) == 0 or t == 1 or t == steps:
            e = _vqe_energy(n, layers, params, None)
            history.append((t, e))
            console.print(f"[dim]VQE iter {t}/{steps}[/dim] energy = [bold]{e:.6f}[/bold]")
    e = _vqe_energy(n, layers, params, None)
    gates = len(_vqe_build_ops_from_params(n, layers, params))
    res = _bench_result("vqe", n, gates, t0, c0, m0)
    res["energy"] = e
    # Write convergence CSV and optional plot
    try:
        out_csv = f"bench/vqe_convergence_n{n}.csv"
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, 'w') as f:
            f.write("iter,energy\n")
            for it, ev in history:
                f.write(f"{it},{ev}\n")
        console.print(f"[purple]CSV data:[/purple] {out_csv}")
        if os.environ.get('MLXQ_SAVE_PLOTS', '0') == '1':
            from .plotting import plot_convergence
            xs = [it for it,_ in history]
            ys = [ev for _,ev in history]
            out_png = f"bench/vqe_convergence_n{n}.png"
            plot_convergence(xs, ys, title=f"VQE Convergence (n={n})", out=out_png)
            console.print(f"[purple]Plot:[/purple] {out_png}")
    except Exception as _:
        pass
    return res


def _lift_single_qubit(op: mx.array, q: int, n: int) -> mx.array:
    O = None
    for i in range(n):
        t = op if i == q else mx.array([[1+0j,0+0j],[0+0j,1+0j]], mx.complex64)
        O = t if O is None else kron(O, t)
    return O


def simulate_steady_state(n: int, steps: int = 160, gamma: float = 0.08) -> Dict[str, Any]:
    # Start at |0...0><0...0|
    psi0 = zero_state(n)
    rho = mx.reshape(mx.matmul(mx.reshape(psi0,(1<<n,1)), mx.reshape(mx.conjugate(psi0),(1,1<<n))), (1<<n, 1<<n))
    K = amplitude_damping_kraus(gamma)
    t0,c0,m0 = now_ms(), cpu_seconds(), peak_rss_mb()
    for _ in range(steps):
        # Apply amplitude damping channel to each qubit (lifted Kraus)
        new_rho = mx.zeros_like(rho)
        for q in range(n):
            Kq = [_lift_single_qubit(k, q, n) for k in K]
            term = mx.zeros_like(rho)
            for E in Kq:
                term = term + mx.matmul(E, mx.matmul(rho, mx.conjugate(mx.transpose(E))))
            new_rho = term  # sequential channel application across qubits
        rho = new_rho
        # Optional uniform dephasing on off-diagonals (avoid slice API differences)
        dim = 1 << n
        diag_mask = mx.array([[1+0j if i==j else 0+0j for j in range(dim)] for i in range(dim)], mx.complex64)
        ones = mx.ones_like(rho)
        off_mask = ones - diag_mask
        decay_mat = diag_mask + (0.99 + 0j) * off_mask
        rho = rho * decay_mat
    # Probe trace to ensure valid density
    _ = mx.sum(mx.diagonal(rho)); mx.eval(_)
    return _bench_result("steady_state", n, steps * n * len(K), t0, c0, m0)


from typing import Callable
from .vendor import VENDOR_BENCHMARKS, ALGORITHM_BENCHMARKS


def run_scaling_benchmark(circuit_type: str, qubits: List[int], simulate_cap: Optional[int] = None,
                          out_prefix: str = "bench",
                          stop_fn: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    if out_prefix == "bench":
        out_prefix = os.environ.get("MLXQ_BENCH_OUT_DIR", "bench")
    # Disable ASCII dump during scaling runs to avoid log flooding
    _prev_ascii = os.environ.get('MLXQ_PRINT_ASCII')
    os.environ['MLXQ_PRINT_ASCII'] = '0'
    # Pretty header similar to C++ output
    console.print(f"[bold green]{circuit_type} Scaling Benchmark[/bold green]")
    dev_label = "apple–silicon–mlx" if platform.system() == "Darwin" else "mlx-cpu"
    backend = os.environ.get('MLXQ_BACKEND', 'sv').lower()
    console.print(f"[dim]Framework:[/dim] [bold]mlx–quantum[/bold] [dim]| Device:[/dim] [bold]{dev_label}[/bold] [dim]| Backend:[/dim] [bold]{backend}[/bold]")
    console.print(f"[dim]Testing qubit counts:[/dim] {', '.join(str(x) for x in qubits)}")
    if circuit_type == "hamiltonian_simulation":
        console.print("[dim]Hamiltonian:[/dim] H = ∑ ZᵢZᵢ₊₁ + 0.5 ∑ Xᵢ ([cyan]J[/cyan]=1.0, [cyan]h[/cyan]=0.5)")
    # Hardware info (for suffixed outputs matching legacy)
    hw_prefix, hw_label = _detect_hardware_info()
    rows = []  # for console table (qubits, exec_ms, mem_mb, cpu%, gpu%)
    results = []  # for JSON 'results'
    # MPS diagnostics aggregated over this scaling run
    mps_max_over_runs = 0
    name_w = max(22, len(circuit_type))
    aborted = False
    use_memray = os.environ.get('MLXQ_MEMRAY', '0') == '1'
    for n in qubits:
        if stop_fn is not None and stop_fn():
            warn("abort requested; stopping scaling loop")
            aborted = True
            break
        if simulate_cap is not None and n > simulate_cap:
            warn(f"skip {n} > cap {simulate_cap}")
            continue
        if use_memray:
            try:
                import memray  # type: ignore
                memdir = os.path.join(out_prefix, 'memray')
                os.makedirs(memdir, exist_ok=True)
                out_bin = os.path.join(memdir, f"{circuit_type}_n{n}.bin")
                with memray.Tracker(out_bin):
                    r = _simulate_dispatch(circuit_type, n)
            except Exception:
                r = _simulate_dispatch(circuit_type, n)
        else:
            r = _simulate_dispatch(circuit_type, n)
        if r is None:
            continue
        # Convert to expected schema
        wall_ms = float(r.get("wall_ms", 0.0))
        cpu_s = float(r.get("cpu_s", 0.0))
        wall_s = max(1e-9, wall_ms / 1000.0)
        cpu_percent = max(0.0, min(100.0, (cpu_s / wall_s) * 100.0))
        # Memory: prefer measured RSS delta when memray mode, else approximate size
        approx_mb = (float(1 << n) * 8.0) / (1024.0 * 1024.0)
        meas_mb = float(r.get('delta_mb', 0.0))
        mem_mb = meas_mb if use_memray and meas_mb > 0.0 else approx_mb
        # If MPS, override with an approximate MPS tensor footprint when available
        if backend == 'mps' and isinstance(r.get('mps'), dict):
            bonds = r['mps'].get('bond_dims')
            if isinstance(bonds, list) and len(bonds) == max(0, n-1):
                # Each tensor A[i] has shape (Dl,2,Dr); Dl/Dr derive from bond dims
                # complex64 assumed (8 bytes per element)
                bytes_total = 0
                for i in range(n):
                    Dl = 1 if i == 0 else int(bonds[i-1])
                    Dr = 1 if i == n-1 else int(bonds[i])
                    bytes_total += Dl * 2 * Dr * 8
                mem_mb = bytes_total / (1024.0 * 1024.0)
        rows.append((str(n), f"{wall_ms:.2f}", f"{mem_mb:.2f}", f"{cpu_percent:.2f}", f"{0.0:.2f}"))
        entry = {
            "qubits": n,
            "execution_time_ms": wall_ms,
            "memory_mb": mem_mb,
            "cpu_percent": cpu_percent,
            "gpu_percent": 0.0,
            "memory_approx_mb": approx_mb,
            "memory_measured_mb": meas_mb,
        }
        if backend == 'mps' and isinstance(r.get('mps'), dict):
            entry["mps"] = r["mps"]
        results.append(entry)
        # Optional per-run MPS bonds CSV
        if backend == 'mps':
            try:
                bonds = entry.get('mps', {}).get('bond_dims')
                if isinstance(bonds, list) and bonds:
                    os.makedirs(out_prefix, exist_ok=True)
                    suffix_local = "_mpsd" if (os.environ.get('MLXQ_MPSD', '0') == '1') else "_mps"
                    key_name_local = f"{circuit_type}{suffix_local}"
                    bonds_csv = f"{out_prefix}/{key_name_local}_n{n}_bonds.csv"
                    with open(bonds_csv, 'w') as bf:
                        bf.write("site,bond_dim\n")
                        for idx, b in enumerate(bonds):
                            bf.write(f"{idx},{int(b)}\n")
                    console.print(f"[dim]MPS bonds:[/dim] {bonds_csv}")
            except Exception:
                pass
        # Stream per-qubit; include measured mem when available
        gates = int(r.get('gates', 0))
        # Optional MPS extras for console line
        mps_note = ""
        try:
            if backend == 'mps' and isinstance(r.get('mps'), dict):
                bmax = int(r['mps'].get('bond_max', 0))
                truncated = bool(r['mps'].get('truncated', False))
                es_b = bool(r['mps'].get('early_stop', False))
                es_t = bool(r['mps'].get('stopped_on_trunc', False))
                es_tag = ""
                if es_b and es_t:
                    es_tag = " ES:b,t"
                elif es_b:
                    es_tag = " ES:b"
                elif es_t:
                    es_tag = " ES:t"
                flags = (" T" if truncated else "") + es_tag
                mps_note = f" | D {bmax}{flags}"
        except Exception:
            pass
        if use_memray and meas_mb > 0.0:
            line = (
                f"{circuit_type:<{name_w}} | "
                f"[bold]{n:2d}q[/bold] | "
                f"gates [green]{gates:5d}[/green] | "
                f"wall [cyan]{wall_ms:7.2f} ms[/cyan] | "
                f"memΔ [magenta]{meas_mb:6.2f} MB[/magenta]{mps_note}"
            )
        else:
            line = (
                f"{circuit_type:<{name_w}} | "
                f"[bold]{n:2d}q[/bold] | "
                f"gates [green]{gates:5d}[/green] | "
                f"wall [cyan]{wall_ms:7.2f} ms[/cyan]{mps_note}"
            )
        console.print(line)
        # Track MPS diagnostics when available
        try:
            if isinstance(r.get('mps'), dict):
                bmax = int(r['mps'].get('bond_max', 0))
                if bmax > mps_max_over_runs:
                    mps_max_over_runs = bmax
        except Exception:
            pass
    # If aborted, skip writing outputs and summary table
    if aborted:
        return {"rows": rows, "results": results}

    # write outputs (match plotting/report expectations)
    mpsd_mode = os.environ.get('MLXQ_MPSD', '0') == '1'
    suffix = "_mpsd" if (backend == 'mps' and mpsd_mode) else ("_mps" if backend == 'mps' else "")
    key_name = f"{circuit_type}{suffix}"
    # Optional MPS summary dump
    if backend == 'mps':
        try:
            os.makedirs(out_prefix, exist_ok=True)
            summ_csv = f"{out_prefix}/{key_name}_summary.csv"
            with open(summ_csv, 'w') as f:
                f.write("qubits,bond_max,bond_mean,truncations,truncated,early_stop,stopped_on_trunc\n")
                for r in results:
                    m = r.get('mps', {}) if isinstance(r, dict) else {}
                    f.write(
                        f"{r['qubits']},{int(m.get('bond_max', 0))},{float(m.get('bond_mean', 0.0)):.6f},"
                        f"{int(m.get('truncations', 0))},{1 if m.get('truncated', False) else 0},"
                        f"{1 if m.get('early_stop', False) else 0},{1 if m.get('stopped_on_trunc', False) else 0}\n"
                    )
            console.print(f"[dim]MPS summary:[/dim] {summ_csv}")
        except Exception:
            pass
    csv_out = f"{out_prefix}/{key_name}_data.csv"
    json_out = f"{out_prefix}/{key_name}_mlx_quantum.json"
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    try:
        with open(csv_out, 'w') as f:
            f.write("qubits,execution_time_ms,memory_mb,cpu_percent,gpu_percent,memory_approx_mb,memory_measured_mb\n")
            for i, row in enumerate(rows):
                # rows[i] matches (qubits, exec_ms, mem_mb, cpu%, gpu%)
                approx = results[i].get('memory_approx_mb', 0.0)
                meas = results[i].get('memory_measured_mb', 0.0)
                f.write(",".join(row) + f",{approx:.2f},{meas:.2f}\n")
        console.print(f"[purple]CSV data:[/purple] {csv_out}")
    except Exception as e:
        error(f"CSV write error: {e}")
    try:
        import json
        with open(json_out, 'w') as f:
            json.dump({
                "circuit_type": circuit_type,
                "results": results,
            }, f, indent=2)
        console.print(f"[purple]JSON:[/purple] {json_out}")
    except Exception as e:
        error(f"JSON write error: {e}")
    # Also write hardware-suffixed CSV/JSON to exactly match legacy naming
    try:
        csv_out_hw = f"{out_prefix}/{key_name}_{hw_prefix}_data.csv"
        with open(csv_out_hw, 'w') as f:
            f.write("qubits,execution_time_ms,memory_mb,cpu_percent,gpu_percent,estimated\n")
            for r in results:
                f.write(
                    f"{r['qubits']},{r['execution_time_ms']:.6f},{r['memory_mb']:.6f},{r['cpu_percent']:.2f},{r['gpu_percent']:.2f},0\n"
                )
        console.print(f"[purple]CSV data (hw):[/purple] {csv_out_hw}")
    except Exception as e:
        warn(f"hw CSV write skipped: {e}")
    try:
        import json
        json_out_hw = f"{out_prefix}/{key_name}_{hw_prefix}_mlx_quantum.json"
        meta = {
            "framework": "mlx-quantum",
            "version": "python",
            "device": dev_label,
            "hardware": hw_label,
            "chip_generation": hw_prefix.split('_',1)[0],
            "chip_variant": hw_prefix.split('_',1)[1] if '_' in hw_prefix else "Base",
            "circuit_type": circuit_type,
            "backend": backend,
            "platform": "apple-silicon" if platform.system() == "Darwin" else platform.system().lower(),
            "results": results,
        }
        # Enrich JSON meta with MPS configuration/diagnostics
        if backend == 'mps':
            try:
                dmax = int(os.environ.get('MLXQ_MPS_DMAX', '64'))
            except Exception:
                dmax = 64
            try:
                eps = float(os.environ.get('MLXQ_MPS_EPS', '1e-10'))
            except Exception:
                eps = 1e-10
            meta["mps"] = {
                "dmax": dmax,
                "eps": eps,
                "bond_max_over_runs": int(mps_max_over_runs),
                "early_stop_bmax": int(os.environ.get('MLXQ_MPS_EARLY_STOP_BMAX', '0') or 0),
                "mode": ("mpsd" if mpsd_mode else "mps"),
            }
        with open(json_out_hw, 'w') as f:
            json.dump(meta, f, indent=2)
        console.print(f"[purple]JSON (hw):[/purple] {json_out_hw}")
    except Exception as e:
        warn(f"hw JSON write skipped: {e}")
    # Optional plotting
    try:
        if os.environ.get('MLXQ_SAVE_PLOTS', '0') == '1':
            from .plotting import plot_scaling
            xs = [r['qubits'] for r in results]
            ys = [r['execution_time_ms'] for r in results]
            title = f"{key_name} Performance (mlx‑Quantum on {hw_label})"
            out_png = f"{out_prefix}/{key_name}_scaling.png"
            notes = [
                f"framework: mlx‑quantum | device: {dev_label} | backend: {backend}",
            ]
            if backend == 'mps':
                try:
                    dmax = int(os.environ.get('MLXQ_MPS_DMAX', '64'))
                except Exception:
                    dmax = 64
                try:
                    eps = float(os.environ.get('MLXQ_MPS_EPS', '1e-10'))
                except Exception:
                    eps = 1e-10
                stop_on_trunc = os.environ.get('MLXQ_MPS_STOP_ON_TRUNC', '0') == '1'
                bmax_cap = int(os.environ.get('MLXQ_MPS_EARLY_STOP_BMAX', '0') or 0)
                pair_sw = os.environ.get('MLXQ_MPS_PAIR_SWEEPS', '0') == '1'
                mode = 'mpsd' if (os.environ.get('MLXQ_MPSD', '0') == '1') else 'mps'
                notes.append(f"{mode} options: dmax={dmax}, eps={eps}, pair_sweeps={1 if pair_sw else 0}, stop_on_trunc={1 if stop_on_trunc else 0}, early_stop_bmax={bmax_cap}")
            plot_scaling(xs, ys, title=title, out=out_png, logy=True, annotate=True, extra_notes=notes)
            console.print(f"[purple]Plot:[/purple] {out_png}")
            # Hardware-suffixed PNG, exactly like legacy
            out_png_hw = f"{out_prefix}/{key_name}_{hw_prefix}_scaling.png"
            try:
                import shutil
                shutil.copyfile(out_png, out_png_hw)
            except Exception:
                pass
            console.print(f"[purple]Plot (hw):[/purple] {out_png_hw}")
            # If MPS, also emit bond growth plot
            if backend == 'mps':
                try:
                    xs_b = xs
                    ys_b = [int((r.get('mps') or {}).get('bond_max', 0)) for r in results]
                    if any(v > 0 for v in ys_b):
                        bonds_png = f"{out_prefix}/{key_name}_bonds.png"
                        plot_scaling(xs_b, ys_b, title=f"{key_name} Max Bond Dimension", out=bonds_png, logy=True, annotate=True,
                                     ylabel="Max Bond Dimension", extra_notes=notes)
                        console.print(f"[purple]Plot (bonds):[/purple] {bonds_png}")
                        try:
                            import shutil
                            shutil.copyfile(bonds_png, os.path.join(out_prefix, f"{key_name}_{hw_prefix}_bonds.png"))
                        except Exception:
                            pass
                except Exception:
                    pass
            # Optionally copy to assets/benchmarks
            if os.path.isdir('assets/benchmarks'):
                try:
                    import shutil
                    # copy main scaling and bonds plots if present
                    extra_paths = []
                    if backend == 'mps':
                        bonds_png = f"{out_prefix}/{key_name}_bonds.png"
                        if os.path.exists(bonds_png):
                            extra_paths.append(bonds_png)
                    for p in (out_png, out_png_hw, *extra_paths):
                        shutil.copyfile(p, os.path.join('assets/benchmarks', os.path.basename(p)))
                except Exception:
                    pass
            # Also emit a gnuplot script for parity with legacy
            try:
                gnu_path = f"{out_prefix}/{key_name}_{hw_prefix}_plot.gnu"
                data_name = f"{key_name}_{hw_prefix}_data.csv"
                with open(gnu_path, 'w') as g:
                    g.write("set terminal png size 800,600\n")
                    g.write(f"set output '{key_name}_{hw_prefix}_scaling.png'\n")
                    g.write("set datafile separator ','\n")
                    g.write("set key autotitle columnhead\n")
                    g.write(f"set title '{key_name} Performance (mlx-Quantum on {hw_label})'\n")
                    g.write("set xlabel 'Qubits'\n")
                    g.write("set ylabel 'Execution Time (ms)'\n")
                    g.write("set logscale y\n")
                    g.write("set grid\n")
                    g.write(f"plot '{data_name}' using 1:2 with linespoints linewidth 2 pointtype 7\n")
                console.print(f"[purple]Gnuplot:[/purple] {gnu_path}")
            except Exception:
                pass
    except Exception as e:
        warn(f"plot skipped: {e}")
    # Minimal console table (hide mem/CPU/GPU columns)
    rows_simple = [(r[0], r[1]) for r in rows]
    table(f"Scaling {circuit_type}", ("qubits","exec(ms)"), rows_simple)
    # Restore ASCII flag
    if _prev_ascii is None:
        os.environ.pop('MLXQ_PRINT_ASCII', None)
    else:
        os.environ['MLXQ_PRINT_ASCII'] = _prev_ascii
    return {"rows": rows, "results": results}


def run_vendor_group(vendor: str, qubits: List[int], simulate_cap: Optional[int] = None,
                     out_prefix: str = "bench") -> Dict[str, Any]:
    """Run a sequence of benchmarks associated with a vendor/framework label."""
    keys = VENDOR_BENCHMARKS.get(vendor.lower())
    if not keys:
        warn(f"Unknown vendor group: {vendor}")
        return {"vendor": vendor, "runs": []}
    info(f"=== Vendor group: {vendor} ===")
    runs = []
    for k in keys:
        # Allow per-benchmark caps: MLXQ_CAP_<KEY>
        cap_env = os.environ.get(f"MLXQ_CAP_{k.upper()}")
        cap = simulate_cap
        if cap_env:
            try:
                cap = int(cap_env)
            except Exception:
                pass
        # Default cap for new cuQuantum Blueqat bench if no per-key cap provided
        if not cap_env and vendor.lower() == 'cuquantum' and k == 'cuquantum_blueqat':
            cap = 15
        runs.append({k: run_scaling_benchmark(k, qubits, simulate_cap=cap, out_prefix=out_prefix)})
    return {"vendor": vendor, "runs": runs}


def run_algorithm_group(group: str, qubits: List[int], simulate_cap: Optional[int] = None,
                        out_prefix: str = "bench") -> Dict[str, Any]:
    """Run a sequence of benchmarks associated with an algorithm group label."""
    keys = ALGORITHM_BENCHMARKS.get(group.lower())
    if not keys:
        warn(f"Unknown algorithm group: {group}")
        return {"group": group, "runs": []}
    info(f"=== Algorithm group: {group} ===")
    runs = []
    for k in keys:
        cap_env = os.environ.get(f"MLXQ_CAP_{k.upper()}")
        cap = simulate_cap
        if cap_env:
            try:
                cap = int(cap_env)
            except Exception:
                pass
        runs.append({k: run_scaling_benchmark(k, qubits, simulate_cap=cap, out_prefix=out_prefix)})
    return {"group": group, "runs": runs}
