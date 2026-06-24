from __future__ import annotations
from typing import List, Dict, Any, Optional

def _glabel(name: str, params: List[float] | None = None) -> str:
    n = name.upper()
    if n in {"H","X","Y","Z","S","T","SX"}:
        return n
    if n in {"RX","RY","RZ"}:
        if params and len(params) >= 1:
            return f"R_{n[1].lower()}({params[0]:.2f})"
        return f"R_{n[1].lower()}"
    if n in {"U1","U2","U3"}:
        return n
    if n in {"CPHASE","CP"}:
        if params and len(params) >= 1:
            return f"CP({params[0]:.2f})"
        return "CP"
    return n


def circuit_to_quantikz(n_qubits: int, ops: List[Dict[str, Any]], include_labels: bool = True) -> str:
    """Return a \begin{quantikz}…\end{quantikz} block for the given circuit.

    Supports: single-qubit gates (H,X,Y,Z,RX/RY/RZ,U1/U2/U3), CNOT/CZ/CPHASE, SWAP (approx), Toffoli.
    Note: CZ/CP are drawn as control • on control and gate Z/CP on target, which is acceptable stylistically.
    """
    # Build a 2D grid of tokens per depth (simple greedy):
    from .draw import schedule_columns
    cols = schedule_columns(n_qubits, ops)

    # Prepare rows
    lines: List[List[str]] = [["\\qw" for _ in range(len(cols))] for _ in range(n_qubits)]

    for j, col in enumerate(cols):
        for op in col:
            name = str(op.get("name",""))
            wires = list(op.get("wires", []))
            params = list(op.get("parameters", [])) if op.get("parameters") else []
            n = name.upper()
            if len(wires) == 1:
                r = wires[0]
                lab = _glabel(n, params)
                lines[r][j] = f"\\gate{{{lab}}}"
            elif len(wires) == 2:
                a,b = wires
                if n in {"CNOT","CX"}:
                    # control on a, target on b
                    if b > a:
                        lines[a][j] = "\\ctrl{1}"
                        lines[b][j] = "\\targ{}"
                    else:
                        lines[a][j] = "\\targ{}"
                        lines[b][j] = "\\ctrl{-1}"
                elif n == "CZ" or n in {"CPHASE","CP"}:
                    # control + Z/CP gate on target
                    if b > a:
                        lines[a][j] = "\\ctrl{1}"
                        lines[b][j] = f"\\gate{{{_glabel(n, params)}}}"
                    else:
                        lines[a][j] = f"\\gate{{{_glabel(n, params)}}}"
                        lines[b][j] = "\\ctrl{-1}"
                elif n == "SWAP":
                    # Draw as two gates connected; quantikz supports \swap, but here use \swapgate-like style
                    if b > a:
                        lines[a][j] = "\\swap{1}"
                        lines[b][j] = "\\targX{}"
                    else:
                        lines[a][j] = "\\targX{}"
                        lines[b][j] = "\\swap{-1}"
                else:
                    # generic: put boxed gate on both wires
                    lab = _glabel(n, params)
                    lines[a][j] = f"\\gate{{{lab}}}"
                    lines[b][j] = f"\\gate{{{lab}}}"
            elif len(wires) == 3 and n in {"CCX","TOFFOLI"}:
                c1,c2,t = wires
                # Relative distances for \ctrl
                if t > c1:
                    lines[c1][j] = f"\\ctrl{{{t-c1}}}"
                else:
                    lines[c1][j] = f"\\ctrl{{{t-c1}}}"
                if t > c2:
                    lines[c2][j] = f"\\ctrl{{{t-c2}}}"
                else:
                    lines[c2][j] = f"\\ctrl{{{t-c2}}}"
                lines[t][j] = "\\targ{}"
            else:
                # Fallback: place generic gate text on all wires
                lab = _glabel(n, params)
                for r in wires:
                    lines[r][j] = f"\\gate{{{lab}}}"

    # Assemble quantikz block
    header = ["\\begin{quantikz}"]
    for q in range(n_qubits):
        left = f"\\lstick{{$\\ket{{0}}$}} & " if include_labels else ""
        row = " & ".join(lines[q])
        end = " \\ "  # line break
        header.append(left + row + end)
    header.append("\\end{quantikz}")
    return "\n".join(header)


def write_quantikz_tex(n_qubits: int, ops: List[Dict[str, Any]], out_tex: str, title: Optional[str] = None) -> None:
    """Write a standalone LaTeX document with the quantikz diagram. Doesn't compile."""
    body = circuit_to_quantikz(n_qubits, ops)
    preamble = r"""\documentclass{standalone}
\usepackage{braket}
\usepackage{quantikz}
\begin{document}
"""
    post = "\\end{document}\n"
    with open(out_tex, 'w') as f:
        if title:
            f.write(preamble + f"% {title}\n" + body + "\n" + post)
        else:
            f.write(preamble + body + "\n" + post)
