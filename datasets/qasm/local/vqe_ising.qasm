OPENQASM 2.0;
include "stdgates.inc";

// VQE circuit for Ising model H = Z⊗Z
// Parameterized ansatz: RY(θ) rotations + CNOT
qreg q[2];
creg c[2];

// VQE ansatz with variational parameters
ry(pi/4) q[0];    // θ₁ = π/4
ry(pi/4) q[1];    // θ₂ = π/4

// Entangling layer
cx q[0], q[1];

// Second variational layer
ry(pi/8) q[0];    // θ₃ = π/8
ry(pi/8) q[1];    // θ₄ = π/8

// Measure for expectation value calculation
measure q[0] -> c[0];
measure q[1] -> c[1];