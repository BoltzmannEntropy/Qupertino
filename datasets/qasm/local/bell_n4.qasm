OPENQASM 2.0;
include "qelib1.inc";

qreg q[4];
creg m_b[1];
creg m_y[1];
creg m_a[1];
creg m_x[1];

h q[0];
cx q[0],q[1];
h q[2];
cx q[2],q[3];

measure q[0] -> m_b[0];
measure q[1] -> m_y[0];
measure q[2] -> m_a[0];
measure q[3] -> m_x[0];