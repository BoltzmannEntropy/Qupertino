OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];
creg c[3];

// W state creation: |001⟩ + |010⟩ + |100⟩
u3(1.91063,0,0) q[0];
ch q[0],q[1];
ccx q[0],q[1],q[2];
x q[0];
x q[1];
cx q[0],q[1];

measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];