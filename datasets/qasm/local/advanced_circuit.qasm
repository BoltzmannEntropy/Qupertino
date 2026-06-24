OPENQASM 2.0;
include "stdgates.inc";

// Advanced circuit with gates requiring decomposition
qreg q[3];
creg c[3];

// Start with superposition
h q[0];

// Use U3 gate (needs decomposition)
u3(pi/2, 0, pi) q[1];

// Create entanglement
cx q[0], q[1];

// Apply Toffoli gate (needs decomposition)
ccx q[0], q[1], q[2];

// Use SWAP gate (needs decomposition)
swap q[1], q[2];

// Apply some phase gates
s q[0];
t q[1];

// Final measurements
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];