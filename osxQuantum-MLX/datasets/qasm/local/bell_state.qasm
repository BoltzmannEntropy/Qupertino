OPENQASM 2.0;
include "stdgates.inc";

// Bell state circuit: creates |Φ+⟩ = (|00⟩ + |11⟩)/√2
qreg q[2];
creg c[2];

// Apply Hadamard to create superposition
h q[0];

// Apply CNOT to create entanglement
cx q[0], q[1];

// Measure both qubits
measure q[0] -> c[0];
measure q[1] -> c[1];