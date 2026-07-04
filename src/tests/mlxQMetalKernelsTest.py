"""Parity tests: opt-in Metal kernels vs the pure-MLX structured paths."""
import math
import os

import numpy as np
import mlx.core as mx
import pytest

from mlxq.sim import StateVectorSimulator, qft
from mlxq.device import Device


def _rand_state(n, seed=9):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(1 << n) + 1j * rng.standard_normal(1 << n)
    v /= np.linalg.norm(v)
    return mx.array(v.astype(np.complex64))


@pytest.fixture
def metal_env():
    os.environ["MLXQ_METAL_KERNELS"] = "1"
    yield
    os.environ.pop("MLXQ_METAL_KERNELS", None)


def test_metal_qft_matches_mlx(metal_env):
    n = 8
    base = _rand_state(n)
    os.environ.pop("MLXQ_METAL_KERNELS", None)
    s1 = StateVectorSimulator(n)
    s1.state = base
    qft(s1, list(range(n)))
    mx.eval(s1.state)
    os.environ["MLXQ_METAL_KERNELS"] = "1"
    s2 = StateVectorSimulator(n)
    s2.state = base
    qft(s2, list(range(n)))
    mx.eval(s2.state)
    err = float(mx.max(mx.abs(s1.state - s2.state)).item().real)
    assert err < 5e-6


def test_metal_zz_layer_matches_mlx(metal_env):
    n = 7
    ops = [{"name": "ZZPHASE", "wires": [i, i + 1], "parameters": [-0.05]}
           for i in range(n - 1)]
    os.environ.pop("MLXQ_METAL_KERNELS", None)
    d1 = Device(n)
    d1.execute(ops)
    mx.eval(d1.sim.state)
    os.environ["MLXQ_METAL_KERNELS"] = "1"
    d2 = Device(n)
    d2.execute(ops)
    mx.eval(d2.sim.state)
    err = float(mx.max(mx.abs(d1.sim.state - d2.sim.state)).item().real)
    assert err < 5e-6


def test_metal_u2_layer_grover_pattern_matches_mlx(metal_env):
    """Interleaved H,X per qubit (Grover diffusion prologue) fuses via the
    uniform single-qubit-layer detector; result must match the pure path."""
    n = 7
    ops = []
    for q in range(n):
        ops.append({"name": "H", "wires": [q]})
    for q in range(n):
        ops.append({"name": "H", "wires": [q]})
        ops.append({"name": "X", "wires": [q]})
    for q in range(n - 1):
        ops.append({"name": "CZ", "wires": [q, q + 1]})
    os.environ.pop("MLXQ_METAL_KERNELS", None)
    d1 = Device(n)
    d1.execute(ops)
    mx.eval(d1.sim.state)
    os.environ["MLXQ_METAL_KERNELS"] = "1"
    d2 = Device(n)
    d2.execute(ops)
    mx.eval(d2.sim.state)
    err = float(mx.max(mx.abs(d1.sim.state - d2.sim.state)).item().real)
    assert err < 5e-6


def test_metal_u2_layer_ry_rz_matches_mlx(metal_env):
    """Uniform RY,RZ per-qubit pairs (QCBM-style layer) via the u2 kernel."""
    n = 6
    ops = []
    for q in range(n):
        ops.append({"name": "RY", "wires": [q], "parameters": [0.23]})
        ops.append({"name": "RZ", "wires": [q], "parameters": [-0.41]})
    os.environ.pop("MLXQ_METAL_KERNELS", None)
    d1 = Device(n)
    d1.execute(ops)
    mx.eval(d1.sim.state)
    os.environ["MLXQ_METAL_KERNELS"] = "1"
    d2 = Device(n)
    d2.execute(ops)
    mx.eval(d2.sim.state)
    err = float(mx.max(mx.abs(d1.sim.state - d2.sim.state)).item().real)
    assert err < 5e-6


def test_u2_detector_skips_nonuniform_params(metal_env):
    """Per-qubit-varying angles must NOT fuse; results still match."""
    n = 5
    ops = [{"name": "RY", "wires": [q], "parameters": [0.1 + 0.05 * q]}
           for q in range(n)]
    os.environ.pop("MLXQ_METAL_KERNELS", None)
    d1 = Device(n)
    d1.execute(ops)
    mx.eval(d1.sim.state)
    os.environ["MLXQ_METAL_KERNELS"] = "1"
    d2 = Device(n)
    d2.execute(ops)
    mx.eval(d2.sim.state)
    err = float(mx.max(mx.abs(d1.sim.state - d2.sim.state)).item().real)
    assert err < 5e-6


def test_metal_diag_layer_matches_mlx(metal_env):
    """CZ ladder + CPHASE ring fuse to one diagonal LUT pass (S1)."""
    n = 7
    ops = [{"name": "CZ", "wires": [q, q + 1]} for q in range(n - 1)]
    ops += [{"name": "CPHASE", "wires": [q, (q + 1) % n], "parameters": [0.37]}
            for q in range(n)]
    os.environ.pop("MLXQ_METAL_KERNELS", None)
    d1 = Device(n)
    d1.execute(ops)
    mx.eval(d1.sim.state)
    os.environ["MLXQ_METAL_KERNELS"] = "1"
    d2 = Device(n)
    d2.execute(ops)
    mx.eval(d2.sim.state)
    err = float(mx.max(mx.abs(d1.sim.state - d2.sim.state)).item().real)
    assert err < 5e-6


def test_metal_xor_affine_matches_mlx(metal_env):
    """GHZ chain + X/SWAP/CNOT mix runs as one gather pass (S2)."""
    n = 7
    ops = [{"name": "H", "wires": [0]}]
    ops += [{"name": "CNOT", "wires": [q, q + 1]} for q in range(n - 1)]
    ops += [{"name": "X", "wires": [2]}, {"name": "SWAP", "wires": [1, 4]},
            {"name": "CNOT", "wires": [6, 0]}, {"name": "X", "wires": [5]}]
    os.environ.pop("MLXQ_METAL_KERNELS", None)
    d1 = Device(n)
    d1.execute(ops)
    mx.eval(d1.sim.state)
    os.environ["MLXQ_METAL_KERNELS"] = "1"
    d2 = Device(n)
    d2.execute(ops)
    mx.eval(d2.sim.state)
    err = float(mx.max(mx.abs(d1.sim.state - d2.sim.state)).item().real)
    assert err < 5e-6


def test_xor_affine_composition_inverse():
    """compose_inverse_affine really inverts the block on random bases."""
    import numpy as np
    from mlxq.shaders import compose_inverse_affine
    n = 6
    ops = ([{"name": "CNOT", "wires": [q, q + 1]} for q in range(n - 1)]
           + [{"name": "X", "wires": [3]}, {"name": "SWAP", "wires": [0, 5]},
              {"name": "CNOT", "wires": [4, 1]}])
    rows, c = compose_inverse_affine(ops, n)

    def forward(x):
        bits = [(x >> (n - 1 - q)) & 1 for q in range(n)]
        for op in ops:
            w = op["wires"]
            if op["name"] == "X":
                bits[w[0]] ^= 1
            elif op["name"] == "CNOT":
                bits[w[1]] ^= bits[w[0]]
            else:
                bits[w[0]], bits[w[1]] = bits[w[1]], bits[w[0]]
        return sum(b << (n - 1 - q) for q, b in enumerate(bits))

    def inverse(y):
        src = 0
        for k in range(n):
            src |= (bin(rows[k] & y).count("1") & 1) << (n - 1 - k)
        return src ^ c

    rng = np.random.default_rng(4)
    for x in rng.integers(0, 1 << n, size=32):
        assert inverse(forward(int(x))) == int(x)


def test_metal_u2_list_layer_matches_mlx(metal_env):
    """Per-qubit-varying RY/RZ window collapses to one per-wire-product
    layer (S3); result must match sequential dispatch."""
    n = 7
    ops = []
    for q in range(n):
        ops.append({"name": "RY", "wires": [q], "parameters": [0.1 + 0.07 * q]})
        ops.append({"name": "RZ", "wires": [q], "parameters": [0.2 - 0.03 * q]})
    for q in range(n):
        ops.append({"name": "CNOT", "wires": [q, (q + 1) % n]})
    os.environ.pop("MLXQ_METAL_KERNELS", None)
    d1 = Device(n)
    d1.execute(ops)
    mx.eval(d1.sim.state)
    os.environ["MLXQ_METAL_KERNELS"] = "1"
    d2 = Device(n)
    d2.execute(ops)
    mx.eval(d2.sim.state)
    err = float(mx.max(mx.abs(d1.sim.state - d2.sim.state)).item().real)
    assert err < 5e-6


def test_u2_window_identity_collapse(metal_env):
    """H,H windows are the identity: the fuser must drop them and still
    match the pure path (which applies both gates)."""
    n = 6
    ops = []
    for q in range(n):
        ops.append({"name": "H", "wires": [q]})
        ops.append({"name": "H", "wires": [q]})
    ops.append({"name": "CZ", "wires": [0, 1]})
    os.environ.pop("MLXQ_METAL_KERNELS", None)
    d1 = Device(n)
    d1.execute(ops)
    mx.eval(d1.sim.state)
    os.environ["MLXQ_METAL_KERNELS"] = "1"
    d2 = Device(n)
    d2.execute(ops)
    mx.eval(d2.sim.state)
    err = float(mx.max(mx.abs(d1.sim.state - d2.sim.state)).item().real)
    assert err < 5e-6


def test_u2_window_small_angle_not_dropped(metal_env):
    """RZ(1e-6) is NOT the identity: the collapse test must keep it
    (codex S3 review: float32 products + default rtol dropped it)."""
    n = 6
    theta = 1e-6
    ops = [{"name": "RZ", "wires": [q], "parameters": [theta]}
           for q in range(n)]
    ops = ops * 3  # 18 ops > n//2+1 so the product path engages
    os.environ.pop("MLXQ_METAL_KERNELS", None)
    d1 = Device(n)
    d1.execute(ops)
    mx.eval(d1.sim.state)
    os.environ["MLXQ_METAL_KERNELS"] = "1"
    d2 = Device(n)
    d2.execute(ops)
    mx.eval(d2.sim.state)
    err = float(mx.max(mx.abs(d1.sim.state - d2.sim.state)).item().real)
    assert err < 5e-6
    # and the fused state must actually carry the phase (not identity)
    d3 = Device(n)  # no ops at all
    diff = float(mx.max(mx.abs(d2.sim.state - d3.sim.state)).item().real)
    assert diff > 1e-7


def test_metal_diag_weighted_qpe_matches_mlx(metal_env):
    """QPE circuit (varying CPHASE angles up to base*2^p) fuses via the
    weighted diagonal kernel (S4); large angles must stay exact."""
    import math
    n = 8
    target = n - 1
    ops = [{"name": "H", "wires": [p]} for p in range(n - 1)]
    ops += [{"name": "CPHASE", "wires": [p, target],
             "parameters": [0.4 * (2 ** p)]} for p in range(n - 1)]
    for jj in range(n - 2, -1, -1):
        for kk in range(n - 2, jj, -1):
            ops.append({"name": "CPHASE", "wires": [kk, jj],
                        "parameters": [-math.pi / (2 ** (kk - jj))]})
        ops.append({"name": "H", "wires": [jj]})
    os.environ.pop("MLXQ_METAL_KERNELS", None)
    d1 = Device(n)
    d1.execute(ops)
    mx.eval(d1.sim.state)
    os.environ["MLXQ_METAL_KERNELS"] = "1"
    d2 = Device(n)
    d2.execute(ops)
    mx.eval(d2.sim.state)
    err = float(mx.max(mx.abs(d1.sim.state - d2.sim.state)).item().real)
    assert err < 5e-6


def test_metal_xx_yy_layers_match_mlx(metal_env):
    """Uniform XX and YY Trotter layers via basis conjugation (S5)."""
    n = 7
    ops = [{"name": "H", "wires": [q]} for q in range(n)]
    ops += [{"name": "XXPHASE", "wires": [q, q + 1], "parameters": [-0.07]}
            for q in range(n - 1)]
    ops += [{"name": "YYPHASE", "wires": [q, q + 1], "parameters": [0.11]}
            for q in range(n - 1)]
    os.environ.pop("MLXQ_METAL_KERNELS", None)
    d1 = Device(n)
    d1.execute(ops)
    mx.eval(d1.sim.state)
    os.environ["MLXQ_METAL_KERNELS"] = "1"
    d2 = Device(n)
    d2.execute(ops)
    mx.eval(d2.sim.state)
    err = float(mx.max(mx.abs(d1.sim.state - d2.sim.state)).item().real)
    assert err < 5e-6


def test_metal_zz_weighted_matches_mlx(metal_env):
    """Per-bond-varying ZZ couplings (long-range Ising pattern, S6)."""
    n = 7
    ops = []
    for a in range(n):
        for b in range(a + 1, n):
            ops.append({"name": "ZZPHASE", "wires": [a, b],
                        "parameters": [0.3 / (b - a) ** 2]})
    os.environ.pop("MLXQ_METAL_KERNELS", None)
    d1 = Device(n)
    d1.execute(ops)
    mx.eval(d1.sim.state)
    os.environ["MLXQ_METAL_KERNELS"] = "1"
    d2 = Device(n)
    d2.execute(ops)
    mx.eval(d2.sim.state)
    err = float(mx.max(mx.abs(d1.sim.state - d2.sim.state)).item().real)
    assert err < 5e-6


def test_metal_qft_stage_detector_matches_mlx(metal_env):
    """Gate-stream QFT (H + ascending CPHASE ladders) fuses to one pass per
    stage via the forward stage detector."""
    import math
    n = 8
    ops = []
    for jj in range(n):
        ops.append({"name": "H", "wires": [jj]})
        for kk in range(jj + 1, n):
            ops.append({"name": "CPHASE", "wires": [kk, jj],
                        "parameters": [math.pi / (2 ** (kk - jj))]})
    os.environ.pop("MLXQ_METAL_KERNELS", None)
    d1 = Device(n)
    d1.execute(ops)
    mx.eval(d1.sim.state)
    os.environ["MLXQ_METAL_KERNELS"] = "1"
    d2 = Device(n)
    d2.execute(ops)
    mx.eval(d2.sim.state)
    err = float(mx.max(mx.abs(d1.sim.state - d2.sim.state)).item().real)
    assert err < 5e-6


def test_metal_iqft_stage_detector_matches_mlx(metal_env):
    """QPE's inline iQFT (descending negative ladders then H) fuses via the
    inverse stage detector; subregister stages must exclude the target bit."""
    import math
    n = 8
    t = n - 1
    ops = [{"name": "H", "wires": [p]} for p in range(n - 1)]
    ops += [{"name": "CPHASE", "wires": [p, t],
             "parameters": [0.4 * (2 ** p)]} for p in range(n - 1)]
    for jj in range(n - 2, -1, -1):
        for kk in range(n - 2, jj, -1):
            ops.append({"name": "CPHASE", "wires": [kk, jj],
                        "parameters": [-math.pi / (2 ** (kk - jj))]})
        ops.append({"name": "H", "wires": [jj]})
    os.environ.pop("MLXQ_METAL_KERNELS", None)
    d1 = Device(n)
    d1.execute(ops)
    mx.eval(d1.sim.state)
    os.environ["MLXQ_METAL_KERNELS"] = "1"
    d2 = Device(n)
    d2.execute(ops)
    mx.eval(d2.sim.state)
    err = float(mx.max(mx.abs(d1.sim.state - d2.sim.state)).item().real)
    assert err < 5e-6


def test_zz_layer_fusion_matches_per_gate():
    n = 6
    theta = -0.11
    ops = [{"name": "ZZPHASE", "wires": [i, i + 1], "parameters": [theta]}
           for i in range(n - 1)]
    d1 = Device(n)
    d1.execute(ops)  # fusion active
    d2 = Device(n)
    for op in ops:
        d2.execute([op])  # one at a time: fusion cannot batch
    err = float(mx.max(mx.abs(d1.sim.state - d2.sim.state)).item().real)
    assert err < 5e-6
