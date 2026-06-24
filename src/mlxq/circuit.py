from typing import List, Dict, Any


class Circuit:
    def __init__(self, wires: int):
        self.wires = int(wires)
        self.ops: List[Dict[str, Any]] = []

    # Single-qubit
    def H(self, q: int): self.ops.append({"name": "H", "wires": [q]})
    def X(self, q: int): self.ops.append({"name": "X", "wires": [q]})
    def Y(self, q: int): self.ops.append({"name": "Y", "wires": [q]})
    def Z(self, q: int): self.ops.append({"name": "Z", "wires": [q]})
    def S(self, q: int): self.ops.append({"name": "S", "wires": [q]})
    def SDG(self, q: int): self.ops.append({"name": "SDG", "wires": [q]})
    def T(self, q: int): self.ops.append({"name": "T", "wires": [q]})
    def TDG(self, q: int): self.ops.append({"name": "TDG", "wires": [q]})
    def SX(self, q: int): self.ops.append({"name": "SX", "wires": [q]})
    def RX(self, theta: float, q: int): self.ops.append({"name": "RX", "wires": [q], "parameters": [theta]})
    def RY(self, theta: float, q: int): self.ops.append({"name": "RY", "wires": [q], "parameters": [theta]})
    def RZ(self, theta: float, q: int): self.ops.append({"name": "RZ", "wires": [q], "parameters": [theta]})
    def U1(self, l: float, q: int): self.ops.append({"name": "U1", "wires": [q], "parameters": [l]})
    def U2(self, phi: float, l: float, q: int): self.ops.append({"name": "U2", "wires": [q], "parameters": [phi, l]})
    def U3(self, th: float, phi: float, l: float, q: int): self.ops.append({"name": "U3", "wires": [q], "parameters": [th, phi, l]})

    # Two-qubit
    def CNOT(self, c: int, t: int): self.ops.append({"name": "CNOT", "wires": [c, t]})
    def CZ(self, c: int, t: int): self.ops.append({"name": "CZ", "wires": [c, t]})
    def SWAP(self, q0: int, q1: int): self.ops.append({"name": "SWAP", "wires": [q0, q1]})
    def iSWAP(self, q0: int, q1: int): self.ops.append({"name": "ISWAP", "wires": [q0, q1]})
    def CPHASE(self, phi: float, c: int, t: int): self.ops.append({"name": "CPHASE", "wires": [c, t], "parameters": [phi]})
    def CRX(self, theta: float, c: int, t: int): self.ops.append({"name": "CRX", "wires": [c, t], "parameters": [theta]})
    def CRY(self, theta: float, c: int, t: int): self.ops.append({"name": "CRY", "wires": [c, t], "parameters": [theta]})
    def CRZ(self, theta: float, c: int, t: int): self.ops.append({"name": "CRZ", "wires": [c, t], "parameters": [theta]})

    # Three-qubit
    def Toffoli(self, c0: int, c1: int, t: int): self.ops.append({"name": "CCX", "wires": [c0, c1, t]})
    def Fredkin(self, c: int, t0: int, t1: int): self.ops.append({"name": "CSWAP", "wires": [c, t0, t1]})

    def operations(self) -> List[Dict[str, Any]]:
        return list(self.ops)

