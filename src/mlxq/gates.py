import math
import mlx.core as mx


def I():
    # Avoid mx.eye with complex on some GPU paths; build explicitly
    return mx.array([[1+0j, 0+0j], [0+0j, 1+0j]], mx.complex64)


def X():
    return mx.array([[0+0j, 1+0j], [1+0j, 0+0j]], mx.complex64)


def Y():
    return mx.array([[0+0j, -1j], [1j, 0+0j]], mx.complex64)


def Z():
    return mx.array([[1+0j, 0+0j], [0+0j, -1+0j]], mx.complex64)


def H():
    s = 1.0 / math.sqrt(2.0)
    return mx.array([[s+0j, s+0j], [s+0j, -s+0j]], mx.complex64)


def S():
    return mx.array([[1+0j, 0+0j], [0+0j, 1j]], mx.complex64)


def SDG():
    return mx.array([[1+0j, 0+0j], [0+0j, -1j]], mx.complex64)


def T():
    ang = math.pi/4.0
    ph = complex(math.cos(ang), math.sin(ang))
    return mx.array([[1+0j, 0+0j],[0+0j, ph]], mx.complex64)


def TDG():
    ang = -math.pi/4.0
    ph = complex(math.cos(ang), math.sin(ang))
    return mx.array([[1+0j, 0+0j],[0+0j, ph]], mx.complex64)

def SX():
    a = complex(0.5, 0.5)
    b = complex(0.5, -0.5)
    return mx.array([[a, b],[b, a]], mx.complex64)


def RX(theta: float):
    c = math.cos(theta/2.0)
    s = math.sin(theta/2.0)
    return mx.array([[complex(c,0.0), complex(0.0,-s)], [complex(0.0,-s), complex(c,0.0)]], mx.complex64)


def RY(theta: float):
    c = math.cos(theta/2.0)
    s = math.sin(theta/2.0)
    return mx.array([[complex(c,0.0), complex(-s,0.0)], [complex(s,0.0), complex(c,0.0)]], mx.complex64)


def RZ(theta: float):
    ht = theta/2.0
    neg = complex(math.cos(-ht), math.sin(-ht))
    pos = complex(math.cos(ht), math.sin(ht))
    return mx.array([[neg, 0+0j],[0+0j, pos]], mx.complex64)


def PhaseShift(phi: float):
    ph = complex(math.cos(phi), math.sin(phi))
    return mx.array([[1+0j, 0+0j],[0+0j, ph]], mx.complex64)


def U1(lmbd: float):
    return PhaseShift(lmbd)


def U2(phi: float, lmbd: float):
    """Qiskit/OpenQASM U2: U2(φ,λ) = RZ(φ) · RY(π/2) · RZ(λ)."""
    return mx.matmul(RZ(phi), mx.matmul(RY(math.pi/2.0), RZ(lmbd)))


def U3(theta: float, phi: float, lmbd: float):
    """Qiskit/OpenQASM U3: U3(θ,φ,λ) = RZ(φ) · RY(θ) · RZ(λ)."""
    return mx.matmul(RZ(phi), mx.matmul(RY(theta), RZ(lmbd)))


def SWAP():
    return mx.array(
        [
            [1+0j,0+0j,0+0j,0+0j],
            [0+0j,0+0j,1+0j,0+0j],
            [0+0j,1+0j,0+0j,0+0j],
            [0+0j,0+0j,0+0j,1+0j],
        ], mx.complex64)


def iSWAP():
    return mx.array(
        [
            [1+0j,0+0j,0+0j,0+0j],
            [0+0j,0+0j,1j,0+0j],
            [0+0j,1j,0+0j,0+0j],
            [0+0j,0+0j,0+0j,1+0j],
        ], mx.complex64)


def CNOT():
    return mx.array(
        [
            [1+0j,0+0j,0+0j,0+0j],
            [0+0j,1+0j,0+0j,0+0j],
            [0+0j,0+0j,0+0j,1+0j],
            [0+0j,0+0j,1+0j,0+0j],
        ], mx.complex64)


def CZ():
    return mx.array(
        [
            [1+0j,0+0j,0+0j,0+0j],
            [0+0j,1+0j,0+0j,0+0j],
            [0+0j,0+0j,1+0j,0+0j],
            [0+0j,0+0j,0+0j,-1+0j],
        ], mx.complex64)


def CPHASE(phi: float):
    ph = complex(math.cos(phi), math.sin(phi))
    return mx.array(
        [
            [1+0j,0+0j,0+0j,0+0j],
            [0+0j,1+0j,0+0j,0+0j],
            [0+0j,0+0j,1+0j,0+0j],
            [0+0j,0+0j,0+0j,ph],
        ], mx.complex64)


def CRX(theta: float):
    rx = RX(theta)
    return _controlled_from_single(rx)


def CRY(theta: float):
    ry = RY(theta)
    return _controlled_from_single(ry)


def CRZ(theta: float):
    rz = RZ(theta)
    return _controlled_from_single(rz)


def Toffoli():
    mat = [[0+0j]*8 for _ in range(8)]
    for i in range(8):
        mat[i][i] = 1+0j
    # swap |110>=6 and |111>=7
    mat[6][6] = 0+0j; mat[7][7] = 0+0j
    mat[6][7] = 1+0j; mat[7][6] = 1+0j
    return mx.array(mat, mx.complex64)


def Fredkin():
    mat = [[0+0j]*8 for _ in range(8)]
    for i in range(8):
        mat[i][i] = 1+0j
    # swap |101>=5 and |110>=6
    mat[5][5] = 0+0j; mat[6][6] = 0+0j
    mat[5][6] = 1+0j; mat[6][5] = 1+0j
    return mx.array(mat, mx.complex64)

def CH():
    """Controlled-Hadamard (control on qubit 0, target on qubit 1)."""
    return _controlled_from_single(H())


def MultiControlledX(n_controls: int):
    """Multi-controlled X on n_controls controls and 1 target (total n_controls+1 qubits).

    For n_controls = 1 this equals CNOT. For n_controls >= 2, build a 2^(n+1) matrix
    that flips the target bit when all controls are 1, identity otherwise.
    """
    k = int(n_controls)
    if k <= 0:
        # No controls → just X on single qubit
        return X()
    if k == 1:
        return CNOT()
    # Dimension for (k controls + 1 target)
    n = k + 1
    dim = 1 << n
    mat = [[0+0j for _ in range(dim)] for __ in range(dim)]
    # Fill identity then swap the two basis states where controls=1 and target differs
    for i in range(dim):
        # Check if all controls (highest k bits) are 1
        controls_all_1 = ((i >> 1) >> (n - 1 - k) == ((1 << k) - 1))
        if controls_all_1:
            # flip target (LSB by convention here since we pack controls in the MSBs)
            j = i ^ 0b1
            mat[j][i] = 1+0j
        else:
            mat[i][i] = 1+0j
    return mx.array(mat, mx.complex64)


def MultiControlledZ(n_controls: int):
    """Multi-controlled Z on n_controls controls and 1 target.

    Applies a phase -1 to the all-ones state |11..1> over (n_controls+1) qubits.
    For n_controls = 1, this equals CZ.
    """
    k = int(n_controls)
    if k <= 0:
        return Z()
    if k == 1:
        return CZ()
    n = k + 1
    dim = 1 << n
    mat = [[0+0j for _ in range(dim)] for __ in range(dim)]
    for i in range(dim):
        mat[i][i] = -1+0j if i == dim - 1 else 1+0j
    return mx.array(mat, mx.complex64)


def _controlled_from_single(single: mx.array):
    mx.eval(single)
    u = single.tolist()
    return mx.array(
        [
            [1+0j,0+0j,0+0j,0+0j],
            [0+0j,1+0j,0+0j,0+0j],
            [0+0j,0+0j,u[0][0],u[0][1]],
            [0+0j,0+0j,u[1][0],u[1][1]],
        ], mx.complex64)
