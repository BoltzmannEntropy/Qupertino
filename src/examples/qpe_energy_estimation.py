from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass
class SingleQubitHamiltonian:
    """
    Single-qubit Hamiltonian H = a I + bx X + by Y + bz Z.

    Eigenvalues: a ± ||b||, where b = (bx, by, bz).
    """

    a: float = 0.0
    bx: float = 0.0
    by: float = 0.0
    bz: float = 0.0

    def norm_b(self) -> float:
        return math.sqrt(self.bx * self.bx + self.by * self.by + self.bz * self.bz)

    def eigenvalues(self) -> Tuple[float, float]:
        r = self.norm_b()
        return (self.a - r, self.a + r)

    def ground_energy(self) -> float:
        e0, e1 = self.eigenvalues()
        return min(e0, e1)

    def ground_state_bloch(self) -> Tuple[float, float, float]:
        """
        Return the Bloch vector (nx, ny, nz) for the ground state direction.
        For H = a I + b·σ, ground state aligns opposite to b.
        If |b| = 0, return +Z by convention.
        """
        r = self.norm_b()
        if r < 1e-12:
            return (0.0, 0.0, 1.0)
        return (-self.bx / r, -self.by / r, -self.bz / r)


def ground_state_of_h(h: SingleQubitHamiltonian) -> Tuple[float, Tuple[float, float, float]]:
    """
    Convenience wrapper to return (E0, bloch_vector) for the ground state.
    """
    return h.ground_energy(), h.ground_state_bloch()


def _cos_series(E: float, ts: Sequence[float]) -> List[float]:
    return [math.cos(E * t) for t in ts]


def _sin_series(E: float, ts: Sequence[float]) -> List[float]:
    return [math.sin(E * t) for t in ts]


def _fit_frequency_from_cos(ts: Sequence[float], ys: Sequence[float]) -> float:
    """
    Estimate frequency E (>= 0) that best fits y ≈ cos(E t) in least-squares sense.
    Uses a coarse-to-fine grid search based on Nyquist limit from min dt.
    """
    if len(ts) < 2:
        return 0.0
    # Determine sampling interval and Nyquist bound
    dts = [abs(ts[i + 1] - ts[i]) for i in range(len(ts) - 1) if abs(ts[i + 1] - ts[i]) > 0]
    if not dts:
        # All times identical; fall back to zero frequency
        return 0.0
    dt = min(dts)
    nyquist = math.pi / dt  # max detectable angular frequency

    def sse(w: float) -> float:
        return sum((y - math.cos(w * t)) ** 2 for t, y in zip(ts, ys))

    # Coarse grid, then refine around the best region
    best_w = 0.0
    best_err = float('inf')
    # 1025 points including endpoints
    for k in range(1025):
        w = k * nyquist / 1024.0
        err = sse(w)
        if err < best_err:
            best_err, best_w = err, w

    # Local refinement: small window around best_w
    window = nyquist / 256.0
    left = max(0.0, best_w - window)
    right = min(nyquist, best_w + window)
    for k in range(1025):
        w = left + (right - left) * (k / 1024.0)
        err = sse(w)
        if err < best_err:
            best_err, best_w = err, w

    return best_w


def simulate_exact_series(h: SingleQubitHamiltonian, ts: Sequence[float]) -> Tuple[List[float], List[float], float]:
    """
    Exact expectation series for the simplified single-ancilla QPE setup.

    With ancilla prepared in |+> and a controlled-U(t) applied (U = e^{-i H t}) to an eigenstate
    |ψ⟩ of H with eigenvalue E, measuring ancilla X yields E[X] = cos(E t) and measuring ancilla Y
    yields E[Y] = sin(E t).

    Returns (xs=ts, ys=cos(E t) for each t, E_hat).
    The sign of E is disambiguated using the corresponding Y-series.
    """
    E_true = h.ground_energy()
    xs = list(ts)
    ys = _cos_series(E_true, xs)

    # Frequency magnitude fit from cos-series only
    w_est = _fit_frequency_from_cos(xs, ys)

    # Determine sign using the Y expectation (sin-series); sign(sin(E t1)) ≈ sign(E) for small t1
    # Use the first nonzero time to avoid t=0
    t_nonzero = None
    for t in xs:
        if abs(t) > 1e-12:
            t_nonzero = t
            break
    sign = 1.0
    if t_nonzero is not None:
        y_sin = math.sin(E_true * t_nonzero)
        if y_sin < 0:
            sign = -1.0

    E_hat = sign * w_est
    return xs, ys, E_hat


def simulate_shot_series(
    h: SingleQubitHamiltonian,
    ts: Sequence[float],
    n_shots: int,
    *,
    seed: int | None = None,
) -> Tuple[List[float], List[float], float]:
    """
    Shot-based simulation for ancilla-X expectation using finite shots.

    For each time t, the probability of measuring ancilla in the |+> state (X=+1) is
        p_plus = (1 + cos(E t)) / 2.
    We sample n_shots binary outcomes and estimate E[X] = 2 * (#plus / n_shots) - 1.

    Returns (xs, ys_noisy, E_hat).
    """
    if seed is not None:
        random.seed(seed)
    E_true = h.ground_energy()
    xs = list(ts)
    ys_noisy: List[float] = []
    for t in xs:
        p_plus = 0.5 * (1.0 + math.cos(E_true * t))
        k_plus = sum(1 for _ in range(n_shots) if random.random() < p_plus)
        est_x = 2.0 * (k_plus / max(1, n_shots)) - 1.0
        ys_noisy.append(est_x)

    # Fit frequency magnitude from noisy cos-series
    w_est = _fit_frequency_from_cos(xs, ys_noisy)

    # Sign via synthetic Y at the first nonzero time (exact model to disambiguate sign)
    t_nonzero = None
    for t in xs:
        if abs(t) > 1e-12:
            t_nonzero = t
            break
    sign = 1.0
    if t_nonzero is not None:
        y_sin = math.sin(E_true * t_nonzero)
        if y_sin < 0:
            sign = -1.0
    E_hat = sign * w_est
    return xs, ys_noisy, E_hat


def random_single_qubit_hamiltonian(*, a_range=(-0.5, 0.5), b_range=(-1.0, 1.0), seed: int | None = None) -> SingleQubitHamiltonian:
    """
    Generate a random single-qubit Hamiltonian H = a I + ∑ b_i σ_i.
    Bias towards nontrivial spectra by ensuring ||b|| is not tiny.
    """
    if seed is not None:
        random.seed(seed)
    a = random.uniform(*a_range)
    while True:
        bx = random.uniform(*b_range)
        by = random.uniform(*b_range)
        bz = random.uniform(*b_range)
        if bx * bx + by * by + bz * bz > 1e-6:
            break
    return SingleQubitHamiltonian(a=a, bx=bx, by=by, bz=bz)


__all__ = [
    'SingleQubitHamiltonian',
    'ground_state_of_h',
    'simulate_exact_series',
    'simulate_shot_series',
    'random_single_qubit_hamiltonian',
]

