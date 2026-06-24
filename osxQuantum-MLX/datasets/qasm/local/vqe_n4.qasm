OPENQASM 2.0;
include "qelib1.inc";

qreg q[4];
creg c[4];

// VQE circuit with parameterized gates
sx q[0];
rz(0.5*pi) q[0];
sx q[0];

sx q[1];
rz(0.25*pi) q[1];
sx q[1];

sx q[2];
rz(0.75*pi) q[2];
sx q[2];

sx q[3];
rz(1.5*pi) q[3];
sx q[3];

// Entangling layers
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];

// Second variational layer
sx q[0];
rz(0.1*pi) q[0];
sx q[0];

sx q[1];
rz(0.3*pi) q[1];
sx q[1];

sx q[2];
rz(0.7*pi) q[2];
sx q[2];

sx q[3];
rz(0.9*pi) q[3];
sx q[3];

measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];