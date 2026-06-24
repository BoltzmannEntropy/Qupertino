from typing import List, Dict, Any, Optional

import os as _os
from .sim import StateVectorSimulator
from .mps_state import MPSState, MPSOptions
from .gates import H, X, Y, Z, S, SDG, T, TDG, SX, RX, RY, RZ, U1, U2, U3, SWAP, iSWAP, CNOT, CZ, CPHASE, CRX, CRY, CRZ, Toffoli, Fredkin, CH
import mlx.core as mx


class Device:
    def __init__(self, wires: int, shots: int = 1000, backend: Optional[str] = None, mps_opts: Optional[MPSOptions] = None):
        self.wires = int(wires)
        self.shots = int(shots)
        if backend is None:
            backend = _os.environ.get('MLXQ_BACKEND', 'sv').lower()
        if backend == 'mps':
            # Read MPS options from env if not provided
            if mps_opts is None:
                try:
                    dmax = int(_os.environ.get('MLXQ_MPS_DMAX', '64'))
                except Exception:
                    dmax = 64
                try:
                    eps = float(_os.environ.get('MLXQ_MPS_EPS', '1e-10'))
                except Exception:
                    eps = 1e-10
                mps_opts = MPSOptions(dmax=dmax, eps=eps)
            self.sim = MPSState(self.wires, mps_opts)
        else:
            self.sim = StateVectorSimulator(self.wires)

    def reset(self):
        self.sim.reset()

    def execute(self, operations: List[Dict[str, Any]]):
        # Optional ASCII dump for any executed circuit (controlled via env)
        try:
            import os as _os
            ascii_on = _os.environ.get('MLXQ_PRINT_ASCII', '0') == '1'
            # Avoid flooding: cap by qubit count (default 8) unless overridden
            try:
                max_q = int(_os.environ.get('MLXQ_PRINT_ASCII_MAX_QUBITS', '8'))
            except Exception:
                max_q = 8
            if ascii_on and self.wires <= max_q:
                from .draw import circuit_ascii as _ascii  # lazy import to avoid cycles
                from .pretty import console as _console
                try:
                    _console.print("\n[dim]ASCII circuit:[/dim]\n" + _ascii(self.wires, operations))
                except Exception:
                    pass
        except Exception:
            pass
        for op in operations:
            name = str(op.get("name", "")).upper()
            wires = list(op.get("wires", []))
            params = list(op.get("parameters", []))
            self._apply(name, wires, params)
        # Return state vector only for SV; MPS has no .state
        return getattr(self.sim, 'state', None)

    def sample(self, shots: int = None, wires: Optional[List[int]] = None):
        shots = self.shots if shots is None else int(shots)
        return self.sim.sample(shots, wires)

    def counts(self, shots: int = None, wires: Optional[List[int]] = None):
        shots = self.shots if shots is None else int(shots)
        return self.sim.sample_counts(shots, wires)

    def _apply(self, name: str, wires: List[int], params: List[float]):
        if len(wires) == 1:
            q = wires[0]
            if name == "H":
                self.sim.apply_single(H(), q)
            elif name == "X":
                self.sim.apply_single(X(), q)
            elif name == "Y":
                self.sim.apply_single(Y(), q)
            elif name == "Z":
                self.sim.apply_single(Z(), q)
            elif name == "S":
                self.sim.apply_single(S(), q)
            elif name in ("SDAG", "SDG", "S†"):
                self.sim.apply_single(SDG(), q)
            elif name == "T":
                self.sim.apply_single(T(), q)
            elif name in ("TDAG", "TDG", "T†"):
                self.sim.apply_single(TDG(), q)
            elif name == "SX":
                self.sim.apply_single(SX(), q)
            elif name == "RX":
                self.sim.apply_single(RX(params[0]), q)
            elif name == "RY":
                self.sim.apply_single(RY(params[0]), q)
            elif name == "RZ":
                self.sim.apply_single(RZ(params[0]), q)
            elif name == "U1":
                self.sim.apply_single(U1(params[0]), q)
            elif name == "U2":
                self.sim.apply_single(U2(params[0], params[1]), q)
            elif name == "U3":
                self.sim.apply_single(U3(params[0], params[1], params[2]), q)
            else:
                raise ValueError(f"Unsupported single-qubit op: {name}")
            return

        if len(wires) == 2:
            c, t = wires
            if name in ("CNOT", "CX"):
                self.sim.apply_two(CNOT(), c, t)
            elif name == "CH":
                self.sim.apply_two(CH(), c, t)
            elif name == "CZ":
                self.sim.apply_two(CZ(), c, t)
            elif name in ("CP", "CPHASE"):
                if len(params) != 1:
                    raise ValueError("CPHASE requires one parameter")
                self.sim.apply_two(CPHASE(params[0]), c, t)
            elif name == "SWAP":
                self.sim.apply_two(SWAP(), c, t)
            elif name == "ISWAP":
                self.sim.apply_two(iSWAP(), c, t)
            elif name == "CRX":
                self.sim.apply_two(CRX(params[0]), c, t)
            elif name == "CRY":
                self.sim.apply_two(CRY(params[0]), c, t)
            elif name == "CRZ":
                self.sim.apply_two(CRZ(params[0]), c, t)
            else:
                raise ValueError(f"Unsupported two-qubit op: {name}")
            return

        if len(wires) == 3:
            if name in ("CCX", "TOFFOLI"):
                # Map via dense apply
                self.sim.apply_dense_gate(Toffoli(), wires)
                return
            if name in ("CSWAP", "FREDKIN"):
                self.sim.apply_dense_gate(Fredkin(), wires)
                return
            raise ValueError(f"Unsupported three-qubit op: {name}")

        raise ValueError(f"Unsupported operation arity for: {name}")
