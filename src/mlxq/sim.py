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
        self.apply_dense_gate(gate, [q])

    def apply_two(self, gate: mx.array, c: int, t: int):
        if c == t:
            raise ValueError("Control and target must differ")
        self.apply_dense_gate(gate, [c, t])

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
        self.apply_two(CZ(), c, t)

    def cphase(self, c: int, t: int, phi: float):
        self.apply_two(CPHASE(phi), c, t)

    def probabilities(self) -> List[float]:
        amp2 = mx.abs(self.state) ** 2
        mx.eval(amp2)
        return [float(v) for v in amp2.tolist()]

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


def qft(sim: StateVectorSimulator, wires: List[int]):
    n = len(wires)
    for j in range(n):
        qj = wires[j]
        sim.h(qj)
        for k in range(j + 1, n):
            qk = wires[k]
            phi = math.pi / (2 ** (k - j))
            sim.cphase(qk, qj, phi)


def iqft(sim: StateVectorSimulator, wires: List[int]):
    n = len(wires)
    for j in reversed(range(n)):
        qj = wires[j]
        for k in reversed(range(j + 1, n)):
            qk = wires[k]
            phi = -math.pi / (2 ** (k - j))
            sim.cphase(qk, qj, phi)
        sim.h(qj)
