OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c[2];

// Deutsch algorithm for f(x)=x
// Initialize ancilla qubit to |1⟩
x q[1];

// Create superposition
h q[0];
h q[1];

// Oracle: f(x) = x (identity function)
cx q[0],q[1];

// Final Hadamard for interference
h q[0];

measure q[0] -> c[0];
measure q[1] -> c[1];