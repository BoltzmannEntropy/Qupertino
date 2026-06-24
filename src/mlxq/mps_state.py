from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import math
import numpy as np

import mlx.core as mx


@dataclass
class MPSOptions:
    dmax: int = 64
    eps: float = 1e-10


def _svd_truncate(M: mx.array, dmax: int, eps: float):
    # M shape: (a*2, 2*b) for two-site tensor; perform SVD and truncate
    # MLX SVD has limited complex/GPU support; use NumPy for robustness.
    try:
        Mn = np.asarray(M.tolist(), dtype=np.complex128)
        U_np, S_np, Vh_np = np.linalg.svd(Mn, full_matrices=False)
        U = mx.array(U_np.astype(np.complex64), mx.complex64)
        S = mx.array(S_np.astype(np.float32))
        Vh = mx.array(Vh_np.astype(np.complex64), mx.complex64)
    except Exception:
        # Fallback: attempt MLX SVD on CPU if available
        U, S, Vh = mx.linalg.svd(M, stream=mx.cpu)
    # Determine truncation rank
    s_vals = mx.reshape(S, (-1,))
    # Move to host to decide rank
    s_list = [float(v) for v in s_vals.tolist()]
    r = len(s_list)
    # eps-based cutoff
    r_eps = r
    if r > 0:
        thresh = eps * s_list[0]
        r_eps = sum(1 for v in s_list if v >= thresh)
    r_keep = min(r, max(1, min(dmax, r_eps)))
    U_t = U[:, :r_keep]
    S_t = S[:r_keep]
    Vh_t = Vh[:r_keep, :]
    return U_t, S_t, Vh_t


class MPSState:
    """Minimal MPS engine with nearest-neighbor 2-qubit gates.

    Tensors stored as list A[i] with shape (Dl, 2, Dr).
    """

    def __init__(self, n_qubits: int, opts: Optional[MPSOptions] = None):
        self.n = int(n_qubits)
        if self.n <= 0:
            raise ValueError("n_qubits must be positive")
        self.opts = opts or MPSOptions()
        # Bond diagnostics
        self.bonds: List[int] = [1] * max(0, self.n - 1)
        self.max_bond_ever: int = 1
        # Truncation diagnostics
        self.truncated_any: bool = False
        self.trunc_events: int = 0
        self.reset()

    def reset(self):
        # |0..0> in right-canonical form: A[i] = [ [1,0] ] up to dimensions
        self.A: List[mx.array] = []
        for i in range(self.n):
            v = mx.array([1+0j, 0+0j], mx.complex64)
            self.A.append(mx.reshape(v, (1, 2, 1)))  # (1,2,1)
        # Reset bond diagnostics
        self.bonds = [1] * max(0, self.n - 1)
        self.max_bond_ever = 1
        self.truncated_any = False
        self.trunc_events = 0

    # -------------- internal helpers --------------
    def _two_site_tensor(self, i: int) -> mx.array:
        left = self.A[i]      # (Dl,2,Dr)
        right = self.A[i+1]   # (Dr,2,Dr2)
        Dl, _, Dr = left.shape
        Dr_, _, Dr2 = right.shape
        if Dr_ != Dr:
            # reshape to match bond
            raise ValueError("MPS bond mismatch")
        T = mx.tensordot(left, right, axes=([2],[0]))  # (Dl,2,2,Dr2)
        return T

    def _split_two_site(self, T: mx.array) -> tuple[mx.array, mx.array]:
        Dl, d1, d2, Dr2 = T.shape
        M = mx.reshape(mx.transpose(T, (0,1,2,3)), (Dl*d1, d2*Dr2))
        U, S, Vh = _svd_truncate(M, self.opts.dmax, self.opts.eps)
        r = int(U.shape[1])
        # Truncation detection (due to dmax or eps)
        r_possible = int(min(Dl*d1, d2*Dr2))
        if r < r_possible:
            self.truncated_any = True
            self.trunc_events += 1
        # reshape back
        Aleft = mx.reshape(U, (Dl, d1, r))
        SVh = mx.reshape(S, (r,1)) * Vh  # (r, d2*Dr2)
        Aright = mx.reshape(SVh, (r, d2, Dr2))
        return Aleft, Aright

    # -------------- gate application --------------
    def apply_single(self, U: mx.array, q: int):
        A = self.A[q]
        Dl, _, Dr = A.shape
        A2 = mx.reshape(A, (-1, 2))       # collapse batch dims
        B = mx.matmul(A2, U)              # (-1,2) row-vector times U
        B = mx.reshape(B, (int(Dl), int(Dr), 2))    # (Dl,Dr,2)
        self.A[q] = mx.transpose(B, (0, 2, 1))  # (Dl,2,Dr)

    def _apply_two_adjacent(self, U4: mx.array, i: int):
        T = self._two_site_tensor(i)  # (Dl,2,2,Dr2)
        Dl, _, _, Dr2 = T.shape
        # merge physical legs (2,2)->4 and apply U
        Tm = mx.reshape(T, (Dl, 4, Dr2))
        Um = mx.reshape(U4, (4,4))
        # Apply Um along the merged physical leg (left-multiply on that axis)
        # Tm has axes (Dl, ab, Dr2); we compute over 'ab' using tensordot
        # Result shape: (4, Dl, Dr2) → transpose to (Dl, 4, Dr2)
        Tm2 = mx.tensordot(Um, Tm, axes=([1],[1]))  # (4, Dl, Dr2)
        Tm2 = mx.transpose(Tm2, (1, 0, 2))          # (Dl, 4, Dr2)
        T2 = mx.reshape(Tm2, (Dl, 2, 2, Dr2))
        Aleft, Aright = self._split_two_site(T2)
        self.A[i] = Aleft
        self.A[i+1] = Aright
        # Update bond diagnostics (bond between i and i+1 equals rank r)
        r = int(Aleft.shape[2])
        if 0 <= i < len(self.bonds):
            self.bonds[i] = r
        if r > self.max_bond_ever:
            self.max_bond_ever = r

    def _apply_two_adjacent_zz(self, theta: float, i: int):
        """Apply exp(-i theta Z⊗Z) via MPO-like diagonal action without forming 4x4.

        Uses decomposition U = c0 I⊗I + c3 Z⊗Z, with c0=(a+b)/2, c3=(a-b)/2,
        where a=e^{-iθ}, b=e^{iθ}. This avoids a 4x4 matmul over merged legs.
        """
        # Two-site tensor T (Dl,2,2,Dr2)
        T = self._two_site_tensor(i)
        Dl, d1, d2, Dr2 = T.shape
        # Coefficients
        import math
        a = complex(math.cos(-theta), math.sin(-theta))
        b = complex(math.cos(theta), math.sin(theta))
        c0 = (a + b) * 0.5
        c3 = (a - b) * 0.5
        # Z action on physical legs (broadcast signs)
        z = mx.array([1.0+0j, -1.0+0j], mx.complex64)
        z1 = mx.reshape(z, (1, d1, 1, 1))
        z2 = mx.reshape(z, (1, 1, d2, 1))
        Tz = T * z1 * z2
        T2 = c0 * T + c3 * Tz
        # Split back via SVD
        Aleft, Aright = self._split_two_site(T2)
        self.A[i] = Aleft
        self.A[i+1] = Aright
        r = int(Aleft.shape[2])
        if 0 <= i < len(self.bonds):
            self.bonds[i] = r
        if r > self.max_bond_ever:
            self.max_bond_ever = r

    def _swap_adjacent(self, i: int):
        # Swap sites i and i+1 by applying SWAP gate U_swap to two-site tensor
        U_swap = mx.array([[1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1]], mx.complex64)
        self._apply_two_adjacent(U_swap, i)

    def apply_two(self, U4: mx.array, c: int, t: int):
        if c == t:
            return
        i, j = sorted((c, t))
        # Swap network to bring i and j adjacent
        k = i
        while k < j - 1:
            self._swap_adjacent(k)
            k += 1
        # Now apply two-qubit gate on (j-1,j)
        self._apply_two_adjacent(U4, j-1)
        # Swap back to restore ordering
        while k > i:
            k -= 1
            self._swap_adjacent(k)

    # -------------- TEBD-style helpers --------------
    def apply_single_all(self, U2: mx.array):
        """Apply the same 1-qubit gate to all sites."""
        for q in range(self.n):
            self.apply_single(U2, q)

    def apply_two_sweep(self, U4: mx.array):
        """Apply a 4x4 two-qubit gate sequentially on all nearest-neighbor bonds.

        Order: (0,1), (1,2), ..., (n-2,n-1). Matches dense sweep ordering.
        """
        for i in range(self.n - 1):
            self._apply_two_adjacent(U4, i)

    def apply_zz_two_sweep(self, theta: float):
        """Apply exp(-i theta Z⊗Z) across nearest-neighbor bonds using diagonal MPO.

        This reduces the cost of forming and multiplying a dense 4x4 matrix by
        acting directly on the two-site tensor with Z⊗Z signs and combining with
        I⊗I contribution.
        """
        for i in range(self.n - 1):
            self._apply_two_adjacent_zz(theta, i)

    def apply_two_all_pairs(self, U4: mx.array, offset: int = 0):
        """Apply the same 4x4 two-qubit gate to disjoint adjacent pairs.

        offset=0 applies pairs (0,1), (2,3), ...; offset=1 applies (1,2), (3,4), ...
        """
        start = 0 if (offset % 2 == 0) else 1
        for i in range(start, self.n - 1, 2):
            self._apply_two_adjacent(U4, i)

    # -------------- diagnostics --------------
    def bond_dims(self) -> List[int]:
        return list(self.bonds)

    def bond_max(self) -> int:
        return int(self.max_bond_ever)

    def bond_mean(self) -> float:
        if not self.bonds:
            return 1.0
        s = float(sum(int(b) for b in self.bonds))
        return s / float(len(self.bonds))

    def truncated(self) -> bool:
        return bool(self.truncated_any)

    def trunc_count(self) -> int:
        return int(self.trunc_events)

    # -------------- convenience MPO sweeps for XX/YY via basis transforms --------------
    def apply_xx_two_sweep(self, theta: float):
        from .gates import H
        for i in range(self.n - 1):
            self.apply_single(H(), i)
            self.apply_single(H(), i+1)
            self._apply_two_adjacent_zz(theta, i)
            self.apply_single(H(), i)
            self.apply_single(H(), i+1)

    def apply_yy_two_sweep(self, theta: float):
        from .gates import RX
        Rp = RX(math.pi/2.0)
        Rm = RX(-math.pi/2.0)
        for i in range(self.n - 1):
            self.apply_single(Rp, i)
            self.apply_single(Rp, i+1)
            self._apply_two_adjacent_zz(theta, i)
            self.apply_single(Rm, i)
            self.apply_single(Rm, i+1)

    # -------------- dense fallback APIs --------------
    def apply_dense_gate(self, gate: mx.array, qubits):
        qs = list(qubits)
        if len(qs) == 1:
            self.apply_single(gate, qs[0]); return
        if len(qs) == 2:
            self.apply_two(gate, qs[0], qs[1]); return
        raise ValueError("MPSState.apply_dense_gate only supports 1q/2q gates")

    def probabilities(self) -> List[float]:
        # Convert to dense vector when small; otherwise approximate by contracting
        n = self.n
        if n <= 18:
            # assemble dense by contracting all bonds
            psi = self.A[0]
            for i in range(1, n):
                psi = mx.tensordot(psi, self.A[i], axes=([psi.ndim - 1],[0]))  # contract on current right bond
            psi_vec = mx.reshape(psi, (1 << n,))
            amp2 = mx.abs(psi_vec) ** 2
            mx.eval(amp2)
            return [float(v) for v in amp2.tolist()]
        # Fallback: return empty list to signal unsupported in large-n (not used in benches)
        return []

    # sampling APIs not strictly needed for current benches
    def sample(self, shots: int, wires: Optional[List[int]] = None):
        from .sim import StateVectorSimulator
        if self.n <= 16:
            sim = StateVectorSimulator(self.n)
            # rebuild dense from MPS
            psi = self.A[0]
            for i in range(1, self.n):
                psi = mx.tensordot(psi, self.A[i], axes=([psi.ndim - 1],[0]))
            sim.state = mx.reshape(psi, (1 << self.n,))
            return sim.sample(shots, wires)
        raise NotImplementedError("MPS.sample not available for large n")

    def sample_counts(self, shots: int, wires: Optional[List[int]] = None):
        counts: dict[str,int] = {}
        for bits in self.sample(shots, wires):
            key = ''.join(str(b) for b in bits)
            counts[key] = counts.get(key, 0) + 1
        return counts
