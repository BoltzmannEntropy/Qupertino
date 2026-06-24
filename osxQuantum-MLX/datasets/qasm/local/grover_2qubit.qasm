OPENQASM 2.0;
include "stdgates.inc";

// 2-qubit Grover's algorithm: search for |11⟩ state
qreg q[2];
creg c[2];

// Initialize uniform superposition
h q[0];
h q[1];

// Oracle: mark |11⟩ state (flip phase)
cz q[0], q[1];

// Diffusion operator: 2|s⟩⟨s| - I
h q[0];
h q[1];
x q[0];
x q[1];
cz q[0], q[1];
x q[0];
x q[1];
h q[0];
h q[1];

// Measure result
measure q[0] -> c[0];
measure q[1] -> c[1];