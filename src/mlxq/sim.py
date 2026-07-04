import math
from typing import Iterable, List, Optional

import mlx.core as mx

from .gates import H, X, Z, CNOT, CZ, CPHASE


def canonical_axis_index(qubit: int, total: int) -> int:
    if not (0 <= qubit < total):
        raise ValueError("Qubit index out of range")
    # Use MSB-first convention: axis index equals qubit index
    return qubit


class StateVectorSimulator:
    def __init__(self, n_qubits: int):
        self.n = int(n_qubits)
        if self.n <= 0:
            raise ValueError("n_qubits must be positive")
        self.reset()

    def reset(self):
        dim = 1 << self.n
        data = [0j] * dim
        data[0] = 1+0j
        self.state = mx.array(data, mx.complex64)

    def apply_dense_gate(self, gate: mx.array, qubits: Iterable[int]):
        qs = list(qubits)
        if len(qs) == 0:
            return
        k = len(qs)
        expected = 1 << k
        if gate.shape[0] != expected or gate.shape[1] != expected:
            raise ValueError("Gate dimension does not match target qubits")

        tensor = mx.reshape(self.state, [2] * self.n)
        axes = [canonical_axis_index(q, self.n) for q in qs]
        selected = [False] * self.n
        for ax in axes:
            selected[ax] = True
        perm = [i for i in range(self.n) if not selected[i]] + axes
        inv_perm = [0] * self.n
        for i, p in enumerate(perm):
            inv_perm[p] = i
        if self.n > 1:
            tensor = mx.transpose(tensor, perm)
        outer_dim = 1 << (self.n - k)
        matrix = mx.reshape(tensor, (outer_dim, expected))
        updated = mx.matmul(matrix, mx.transpose(gate))
        updated_tensor = mx.reshape(updated, [2] * self.n)
        if self.n > 1:
            updated_tensor = mx.transpose(updated_tensor, inv_perm)
        self.state = mx.reshape(updated_tensor, (1 << self.n,))

    def apply_single(self, gate: mx.array, q: int):
        if gate.shape[0] != 2 or gate.shape[1] != 2:
            raise ValueError("Gate dimension does not match target qubits")
        ax = canonical_axis_index(q, self.n)
        # Slice-based update on view (A, 2, B): no full-state transpose, no matmul
        a_dim = 1 << ax
        b_dim = 1 << (self.n - ax - 1)
        tensor = mx.reshape(self.state, (a_dim, 2, b_dim))
        s0 = tensor[:, 0, :]
        s1 = tensor[:, 1, :]
        n0 = gate[0, 0] * s0 + gate[0, 1] * s1
        n1 = gate[1, 0] * s0 + gate[1, 1] * s1
        self.state = mx.reshape(mx.stack([n0, n1], axis=1), (1 << self.n,))

    def apply_two(self, gate: mx.array, c: int, t: int):
        if c == t:
            raise ValueError("Control and target must differ")
        self.apply_dense_gate(gate, [c, t])

    def apply_diagonal(self, diag: mx.array, qubits: Iterable[int]):
        """Apply a diagonal gate given as its 1-D diagonal of length 2^k.

        Entry order follows the basis order of `qubits` as given (first qubit is
        the most significant bit), matching the dense-matrix convention of
        apply_dense_gate. Uses a broadcast elementwise multiply: no transposes,
        no matmul, one full-state kernel.
        """
        qs = list(qubits)
        k = len(qs)
        if k == 0:
            return
        if len(set(qs)) != k:
            raise ValueError("Duplicate qubit indices")
        if diag.shape[0] != (1 << k):
            raise ValueError("Diagonal dimension does not match target qubits")
        axes = [canonical_axis_index(q, self.n) for q in qs]
        if k == 1:
            a_dim = 1 << axes[0]
            b_dim = 1 << (self.n - axes[0] - 1)
            d = mx.reshape(diag, (1, 2, 1))
            tensor = mx.reshape(self.state, (a_dim, 2, b_dim))
            self.state = mx.reshape(tensor * d, (1 << self.n,))
            return
        if k == 2:
            q0, q1 = axes
            if q0 > q1:
                q0, q1 = q1, q0
                diag = mx.take(diag, mx.array([0, 2, 1, 3], mx.uint32))
            a_dim = 1 << q0
            m_dim = 1 << (q1 - q0 - 1)
            b_dim = 1 << (self.n - q1 - 1)
            d = mx.reshape(diag, (1, 2, 1, 2, 1))
            tensor = mx.reshape(self.state, (a_dim, 2, m_dim, 2, b_dim))
            self.state = mx.reshape(tensor * d, (1 << self.n,))
            return
        # General k: broadcast the diagonal across the [2]*n tensor view
        order = sorted(range(k), key=lambda i: axes[i])
        d = mx.transpose(mx.reshape(diag, [2] * k), order)
        full_shape = [1] * self.n
        for ax in axes:
            full_shape[ax] = 2
        d = mx.reshape(d, full_shape)
        tensor = mx.reshape(self.state, [2] * self.n)
        self.state = mx.reshape(tensor * d, (1 << self.n,))

    def apply_controlled_single(self, gate: mx.array, c: int, t: int):
        """Controlled single-qubit gate: apply `gate` to the target only on the
        control=1 half of the state. One stack copy instead of a dense 4x4
        matmul with two full-state transposes."""
        if c == t:
            raise ValueError("Control and target must differ")
        if gate.shape[0] != 2 or gate.shape[1] != 2:
            raise ValueError("Gate dimension does not match target qubit")
        ac = canonical_axis_index(c, self.n)
        at = canonical_axis_index(t, self.n)
        q0, q1 = (ac, at) if ac < at else (at, ac)
        a_dim = 1 << q0
        m_dim = 1 << (q1 - q0 - 1)
        b_dim = 1 << (self.n - q1 - 1)
        tensor = mx.reshape(self.state, (a_dim, 2, m_dim, 2, b_dim))
        if ac < at:
            # control at axis 1, target at axis 3
            c0 = tensor[:, 0, :, :, :]
            u0 = tensor[:, 1, :, 0, :]
            u1 = tensor[:, 1, :, 1, :]
            n0 = gate[0, 0] * u0 + gate[0, 1] * u1
            n1 = gate[1, 0] * u0 + gate[1, 1] * u1
            c1 = mx.stack([n0, n1], axis=2)
            out = mx.stack([c0, c1], axis=1)
        else:
            # target at axis 1, control at axis 3
            c0 = tensor[:, :, :, 0, :]
            u0 = tensor[:, 0, :, 1, :]
            u1 = tensor[:, 1, :, 1, :]
            n0 = gate[0, 0] * u0 + gate[0, 1] * u1
            n1 = gate[1, 0] * u0 + gate[1, 1] * u1
            c1 = mx.stack([n0, n1], axis=1)
            out = mx.stack([c0, c1], axis=3)
        self.state = mx.reshape(out, (1 << self.n,))

    def _control_mask(self, controls: List[int]) -> mx.array:
        """Broadcastable [2]*n mask: 1 where all control bits are 1, else 0."""
        mask = None
        for c in controls:
            shape = [1] * self.n
            shape[canonical_axis_index(c, self.n)] = 2
            v = mx.reshape(mx.array([0+0j, 1+0j], mx.complex64), shape)
            mask = v if mask is None else mask * v
        return mask

    def apply_multi_controlled_single(self, gate: mx.array, controls: Iterable[int], target: int):
        """k-controlled single-qubit gate via masked update:
        out = state + mask*(U_target(state) - state), mask = prod of control bits.
        A few elementwise passes instead of a dense 2^(k+1) matrix apply."""
        cs = list(controls)
        if target in cs or len(set(cs)) != len(cs):
            raise ValueError("Control/target wires must be distinct")
        if len(cs) == 1:
            self.apply_controlled_single(gate, cs[0], target)
            return
        saved = self.state
        self.apply_single(gate, target)
        gated = self.state
        mask = self._control_mask(cs)
        t = mx.reshape(saved, [2] * self.n)
        g = mx.reshape(gated, [2] * self.n)
        self.state = mx.reshape(t + mask * (g - t), (1 << self.n,))

    def apply_controlled_swap(self, c: int, a: int, b: int):
        """Fredkin via masked update against the axis-permuted state."""
        if len({c, a, b}) != 3:
            raise ValueError("Control/target wires must be distinct")
        saved = self.state
        self.apply_swap_wires(a, b)
        swapped = self.state
        mask = self._control_mask([c])
        t = mx.reshape(saved, [2] * self.n)
        g = mx.reshape(swapped, [2] * self.n)
        self.state = mx.reshape(t + mask * (g - t), (1 << self.n,))

    def apply_swap_wires(self, a: int, b: int):
        """SWAP as an axis permutation of the [2]*n tensor view: one copy."""
        if a == b:
            return
        aa = canonical_axis_index(a, self.n)
        ab = canonical_axis_index(b, self.n)
        perm = list(range(self.n))
        perm[aa], perm[ab] = perm[ab], perm[aa]
        tensor = mx.transpose(mx.reshape(self.state, [2] * self.n), perm)
        self.state = mx.reshape(tensor, (1 << self.n,))

    def _apply_antidiag_phase(self, theta: float, a: int, b: int, signs):
        """Shared kernel for exp(-i*theta*P) with P in {X⊗X, Y⊗Y}:
        out = cos(theta)*state - i*sin(theta)*(P state), where P maps each
        basis slice to its double-flip partner with the given signs."""
        aa = canonical_axis_index(a, self.n)
        ab = canonical_axis_index(b, self.n)
        q0, q1 = (aa, ab) if aa < ab else (ab, aa)
        a_dim = 1 << q0
        m_dim = 1 << (q1 - q0 - 1)
        b_dim = 1 << (self.n - q1 - 1)
        tensor = mx.reshape(self.state, (a_dim, 2, m_dim, 2, b_dim))
        s00 = tensor[:, 0, :, 0, :]
        s01 = tensor[:, 0, :, 1, :]
        s10 = tensor[:, 1, :, 0, :]
        s11 = tensor[:, 1, :, 1, :]
        cth = complex(math.cos(theta), 0.0)
        msth = complex(0.0, -math.sin(theta))
        g00, g01, g10, g11 = signs
        o00 = cth * s00 + msth * (g00 * s11)
        o01 = cth * s01 + msth * (g01 * s10)
        o10 = cth * s10 + msth * (g10 * s01)
        o11 = cth * s11 + msth * (g11 * s00)
        row0 = mx.stack([o00, o01], axis=2)
        row1 = mx.stack([o10, o11], axis=2)
        self.state = mx.reshape(mx.stack([row0, row1], axis=1), (1 << self.n,))

    def apply_xx_phase(self, theta: float, a: int, b: int):
        """exp(-i*theta*X⊗X); matches bench _xx_phase_gate(theta)."""
        self._apply_antidiag_phase(theta, a, b, (1.0, 1.0, 1.0, 1.0))

    def apply_yy_phase(self, theta: float, a: int, b: int):
        """exp(-i*theta*Y⊗Y); matches bench _yy_phase_gate(theta)."""
        self._apply_antidiag_phase(theta, a, b, (-1.0, 1.0, 1.0, -1.0))

    def apply_zz_phase(self, theta: float, a: int, b: int):
        """exp(-i*theta*Z⊗Z): diagonal, one broadcast multiply."""
        em = complex(math.cos(theta), -math.sin(theta))
        ep = complex(math.cos(theta), math.sin(theta))
        self.apply_diagonal(mx.array([em, ep, ep, em], mx.complex64), [a, b])

    # Cache of fused ZZ-layer phase vectors keyed by (n, theta, bonds tuple).
    # A Trotter step re-applies the same layer every step, so the vector is
    # built once and each subsequent step costs a single multiply.
    _ZZ_LAYER_CACHE: dict = {}

    def apply_zz_layer(self, theta: float, bonds: List[tuple]):
        """Fused exp(-i*theta*sum_bonds Z_a Z_b): one multiply per layer.

        All ZZ terms commute, so the layer is one diagonal whose angle per
        basis state is theta * sum_bonds s_ab, with s_ab = +1 when the two
        bits agree and -1 otherwise. The phase vector is built from integer
        bit arithmetic on the basis indices and cached, because product
        formulas reuse the same (theta, bonds) every step.
        """
        from . import shaders as metal_kernels
        if metal_kernels.metal_enabled():
            self.state = metal_kernels.zz_chain_layer(
                self.state, theta, self.n, list(bonds))
            return
        key = (self.n, round(theta, 15), tuple(sorted(bonds)))
        phase = StateVectorSimulator._ZZ_LAYER_CACHE.get(key)
        if phase is None:
            idx = mx.arange(1 << self.n, dtype=mx.uint32)
            agree_minus_disagree = mx.zeros((1 << self.n,), dtype=mx.int32)
            for a, b in bonds:
                bit_a = (idx >> (self.n - 1 - a)) & 1
                bit_b = (idx >> (self.n - 1 - b)) & 1
                s = 1 - 2 * (bit_a ^ bit_b).astype(mx.int32)  # +1 agree, -1 disagree
                agree_minus_disagree = agree_minus_disagree + s
            angle = (-theta) * agree_minus_disagree.astype(mx.float32)
            phase = mx.cos(angle).astype(mx.complex64) \
                + 1j * mx.sin(angle).astype(mx.complex64)
            mx.eval(phase)
            StateVectorSimulator._ZZ_LAYER_CACHE[key] = phase
        self.state = self.state * phase

    # Convenience ops
    def h(self, q: int):
        self.apply_single(H(), q)

    def x(self, q: int):
        self.apply_single(X(), q)

    def z(self, q: int):
        self.apply_single(Z(), q)

    def cnot(self, c: int, t: int):
        self.apply_two(CNOT(), c, t)

    def cz(self, c: int, t: int):
        self.apply_diagonal(mx.array([1+0j, 1+0j, 1+0j, -1+0j], mx.complex64), [c, t])

    def cphase(self, c: int, t: int, phi: float):
        ph = complex(math.cos(phi), math.sin(phi))
        self.apply_diagonal(mx.array([1+0j, 1+0j, 1+0j, ph], mx.complex64), [c, t])

    def probabilities_array(self) -> mx.array:
        """Probabilities as an MLX array; stays on device until the caller evals."""
        return mx.abs(self.state) ** 2

    def probabilities(self) -> List[float]:
        amp2 = self.probabilities_array()
        mx.eval(amp2)
        return amp2.tolist()

    def project_measure(self, wires: List[int], outcome_bits: List[int]):
        if len(wires) != len(outcome_bits):
            raise ValueError("wires and outcome_bits must match length")
        n = self.n
        dim = 1 << n
        allowed = []
        for i in range(dim):
            ok = True
            for w, b in zip(wires, outcome_bits):
                bit = (i >> (n - 1 - w)) & 1
                if bit != (b & 1):
                    ok = False
                    break
            if ok:
                allowed.append(i)
        # Build new state on host for simplicity/robustness
        new_state_list: List[complex] = [0j] * dim
        # self.state[i] returns a 0-d array; extract complex via .item()
        for idx in allowed:
            try:
                val = self.state[idx]
                cval = complex(val.item()) if hasattr(val, 'item') else complex(val)
            except Exception:
                # Fallback path if fancy indexing differs
                cval = 0j
            new_state_list[idx] = cval
        # Normalize
        norm2 = sum((abs(v) ** 2 for v in new_state_list))
        if norm2 > 0:
            inv = 1.0 / math.sqrt(norm2)
            new_state_list = [v * inv for v in new_state_list]
        self.state = mx.array(new_state_list, mx.complex64)

    def sample(self, shots: int, wires: Optional[List[int]] = None) -> List[List[int]]:
        import random
        probs = self.probabilities()
        n = self.n
        if wires is None:
            wires = list(range(n))
        out = []
        for _ in range(int(shots)):
            # sample index by cumulative probabilities
            r = random.random()
            s = 0.0
            idx = 0
            for i, p in enumerate(probs):
                s += p
                if r <= s:
                    idx = i
                    break
            # extract bits for requested wires (MSB first by our convention)
            bits = []
            for w in wires:
                bit = (idx >> (n - 1 - w)) & 1
                bits.append(bit)
            out.append(bits)
        return out

    def sample_counts(self, shots: int, wires: Optional[List[int]] = None):
        counts: dict[str,int] = {}
        samples = self.sample(shots, wires)
        for bits in samples:
            key = ''.join(str(b) for b in bits)
            counts[key] = counts.get(key, 0) + 1
        return counts


_BIT_REVERSAL_CACHE: dict = {}


def _bit_reversal_indices(n: int) -> mx.array:
    idx = _BIT_REVERSAL_CACHE.get(n)
    if idx is None:
        import numpy as np
        i = np.arange(1 << n, dtype=np.uint32)
        r = np.zeros_like(i)
        for b in range(n):
            r |= ((i >> b) & 1) << (n - 1 - b)
        idx = mx.array(r)
        _BIT_REVERSAL_CACHE[n] = idx
    return idx


_FFT_SUPPORT_CACHE: dict = {}


def fft_supported(n: int) -> bool:
    """MLX Metal lacks FFT kernels for some sizes (e.g. 2^21, 2^22 in mlx 0.30.x);
    probe once per size and cache."""
    s = _FFT_SUPPORT_CACHE.get(n)
    if s is None:
        try:
            mx.eval(mx.fft.ifft(mx.zeros((1 << n,), mx.complex64)))
            s = True
        except Exception:
            s = False
        _FFT_SUPPORT_CACHE[n] = s
    return s


def qft_fft(sim: StateVectorSimulator):
    """Full-register QFT as an algorithmic primitive via mx.fft.

    QFT|x> = (1/sqrt(N)) sum_k e^{+2*pi*i*x*k/N} |k>, which equals sqrt(N)*ifft
    followed by a bit-reversal permutation to match the gate-decomposed QFT
    (no final swap layer) used by qft(). This is NOT circuit execution; report
    it as a separate primitive, comparable only to libraries exposing a native
    QFT/FFT operation. Falls back to the fused-ladder circuit QFT for register
    sizes whose FFT kernel is unavailable in the installed MLX.
    """
    n = sim.n
    if not fft_supported(n):
        qft(sim, list(range(n)))
        return
    scale = complex(math.sqrt(1 << n), 0.0)
    out = mx.fft.ifft(sim.state) * scale
    sim.state = mx.take(out, _bit_reversal_indices(n))


def _fused_phase_ladder(sim: StateVectorSimulator, wires: List[int], j: int, sign: float):
    """One diagonal multiply for all CPHASE(wires[k] -> wires[j], k>j) of a QFT stage.

    The controlled phases of a stage commute, so their product is a single
    diagonal: phase(basis) = bit_j * sum_k bit_k * sign*pi/2^(k-j). Built as a
    broadcast product of 2-element vectors, applied in one elementwise kernel.
    """
    n = len(wires)
    n_total = sim.n
    factor = None
    for k in range(j + 1, n):
        phi = sign * math.pi / (2 ** (k - j))
        v = mx.array([1+0j, complex(math.cos(phi), math.sin(phi))], mx.complex64)
        shape = [1] * n_total
        shape[canonical_axis_index(wires[k], n_total)] = 2
        vk = mx.reshape(v, shape)
        factor = vk if factor is None else factor * vk
    if factor is None:
        return
    cond_shape = [1] * n_total
    cond_shape[canonical_axis_index(wires[j], n_total)] = 2
    bit_j = mx.reshape(mx.array([0+0j, 1+0j], mx.complex64), cond_shape)
    one_j = mx.reshape(mx.array([1+0j, 0+0j], mx.complex64), cond_shape)
    full = one_j + bit_j * factor
    tensor = mx.reshape(sim.state, [2] * n_total)
    sim.state = mx.reshape(tensor * full, (1 << n_total,))


def qft(sim: StateVectorSimulator, wires: List[int]):
    n = len(wires)
    from . import shaders as metal_kernels
    if (metal_kernels.metal_enabled() and hasattr(sim, "state")
            and list(wires) == list(range(sim.n))):
        sim.state = metal_kernels.qft_stage_all(sim.state, sim.n)
        return
    fused = hasattr(sim, "state") and hasattr(sim, "apply_diagonal")
    for j in range(n):
        qj = wires[j]
        sim.h(qj)
        if fused:
            _fused_phase_ladder(sim, wires, j, 1.0)
        else:
            for k in range(j + 1, n):
                qk = wires[k]
                phi = math.pi / (2 ** (k - j))
                sim.cphase(qk, qj, phi)


def iqft(sim: StateVectorSimulator, wires: List[int]):
    n = len(wires)
    fused = hasattr(sim, "state") and hasattr(sim, "apply_diagonal")
    for j in reversed(range(n)):
        qj = wires[j]
        if fused:
            _fused_phase_ladder(sim, wires, j, -1.0)
        else:
            for k in reversed(range(j + 1, n)):
                qk = wires[k]
                phi = -math.pi / (2 ** (k - j))
                sim.cphase(qk, qj, phi)
        sim.h(qj)
