import math
import os
import sys

# Ensure local package path so direct execution works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mlxq.qml import Device, qnode, H, RX, RY, RZ, CNOT, expval, PauliZ, PauliX, PauliY, basic_entangler_layers

# QPE example helpers (exact simulation)
from examples.qpe_energy_estimation import (
    SingleQubitHamiltonian,
    simulate_exact_series,
    ground_state_of_h,
)


def test_qml_qnode_expval_z_after_rx():
    dev = Device(wires=1)

    @qnode(dev)
    def circuit(params):
        RX(params[0], wires=0)
        return expval(PauliZ(0))

    theta = 0.7
    val = circuit([theta])
    # <Z> for RX(theta)|0> equals cos(theta)
    assert abs(val - math.cos(theta)) < 1e-3


def test_qml_grad_parameter_shift_rx():
    dev = Device(wires=1)

    @qnode(dev)
    def circuit(params):
        RX(params[0], wires=0)
        return expval(PauliZ(0))

    theta = 0.4
    grad = circuit.grad([theta])[0]
    # d/dθ cos(θ) = -sin(θ)
    assert abs(grad - (-math.sin(theta))) < 5e-3


def test_qml_expval_shots_matches_expectation():
    # Prepare |+> and measure X with shots
    dev = Device(wires=1, shots=10000)

    @qnode(dev)
    def circuit():
        H(0)
        return expval(PauliX(0))  # with shots set on device

    val = circuit()
    # Expect ≈ 1
    assert val > 0.98


def test_qml_template_smoke():
    dev = Device(wires=3)

    @qnode(dev)
    def circuit(params):
        basic_entangler_layers(params, wires=(0, 1, 2))
        return expval(PauliZ(0))

    params = [(0.1, 0.2), (0.0, -0.1), (0.3, 0.4)]
    _ = circuit(params)
    assert True


def test_qpe_exact_energy_fit():
    # Deterministic single-qubit H with ground energy E = a - |b| = -0.7
    Hh = SingleQubitHamiltonian(a=0.0, bx=0.0, by=0.0, bz=0.7)
    E_true, _ = ground_state_of_h(Hh)
    # Time grid over a couple of periods
    Nt = 9
    Tmax = 4.0 * math.pi / max(1e-6, abs(E_true))
    ts = [k * Tmax / (Nt - 1) for k in range(Nt)]
    xs, ys, E_hat = simulate_exact_series(Hh, ts)
    assert abs(E_hat - E_true) < 1e-3

