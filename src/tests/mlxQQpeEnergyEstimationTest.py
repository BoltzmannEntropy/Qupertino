import math
from examples.qpe_energy_estimation import (
    SingleQubitHamiltonian,
    simulate_exact_series,
    simulate_shot_series,
    ground_state_of_h,
)


def test_qpe_exact_series_estimates_ground_energy():
    h = SingleQubitHamiltonian(a=0.0, bx=0.0, by=0.0, bz=0.7)
    E_true, _ = ground_state_of_h(h)
    Nt = 9
    Tmax = 4.0 * math.pi / max(1e-6, abs(E_true))
    ts = [k * Tmax / (Nt - 1) for k in range(Nt)]
    xs, ys, E_hat = simulate_exact_series(h, ts)
    assert len(xs) == Nt and len(ys) == Nt
    assert abs(E_hat - E_true) < 1e-3


def test_qpe_shot_series_converges_with_shots():
    h = SingleQubitHamiltonian(a=0.0, bx=0.0, by=0.0, bz=0.5)
    E_true, _ = ground_state_of_h(h)
    Nt = 13
    Tmax = 4.0 * math.pi / max(1e-6, abs(E_true))
    ts = [k * Tmax / (Nt - 1) for k in range(Nt)]
    xs, ys_noisy, E_hat = simulate_shot_series(h, ts, n_shots=2000, seed=123)
    assert len(xs) == Nt and len(ys_noisy) == Nt
    # Relaxed tolerance due to shot noise
    assert abs(E_hat - E_true) < 0.05

