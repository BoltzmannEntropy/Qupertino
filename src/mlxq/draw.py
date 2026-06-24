from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple

# Visual constants for matplotlib drawer
_BOX_W = 0.5   # relative to one column width
_BOX_H = 0.4   # relative to one row spacing
_X_CENTER = 0.5  # center of column

def _short_label(name: str, params: List[float] | None = None) -> str:
    n = name.upper()
    if n in {"I","X","Y","Z","H","S","T","SX"}: return n
    if n in {"CNOT","CX"}: return "X"
    if n == "CZ": return "Z"
    if n in {"U1","U2","U3"}: return n
    if n in {"RX","RY","RZ"}:
        if params:
            return f"{n}"
        return n
    if n in {"CPHASE","CP"}: return "CP"
    if n == "SWAP": return "SWAP"
    if n == "ISWAP": return "iSWAP"
    if n in {"CCX","TOFFOLI"}: return "X"
    if n in {"CSWAP","FREDKIN"}: return "SWP"
    if n in {"MEASURE","MEAS","M"}: return "M"
    return n[:4]


def _footprint(op: Dict[str, Any]) -> set[int]:
    ws = list(op.get("wires", []))
    name = str(op.get("name", "")).upper()
    if len(ws) <= 1:
        return set(ws)
    if name in {"CNOT","CX","CZ","CPHASE","CP","SWAP","CCX","TOFFOLI"}:
        a, b = min(ws), max(ws)
        return set(range(a, b + 1))
    return set(ws)


def schedule_columns(n_qubits: int, ops: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Greedy column scheduler that prevents wire/connector collisions."""
    cols: List[List[Dict[str, Any]]] = []
    fps: List[List[set[int]]] = []
    for op in ops:
        fp = _footprint(op)
        placed = False
        for i, col in enumerate(cols):
            if all(fp.isdisjoint(x) for x in fps[i]):
                col.append(op)
                fps[i].append(fp)
                placed = True
                break
        if not placed:
            cols.append([op])
            fps.append([fp])
    return cols


def circuit_ascii(n_qubits: int, ops: List[Dict[str, Any]], col_width: int = 9) -> str:
    """Render a simple ASCII circuit with colored symbols avoided (pure text)."""
    cols = schedule_columns(n_qubits, ops)
    w = col_width
    # initialise canvas as list of list of chars
    total_w = len(cols) * w
    rows = [["─" for _ in range(total_w)] for _ in range(n_qubits)]

    def place_char(r: int, c: int, ch: str):
        if 0 <= r < n_qubits and 0 <= c < total_w:
            rows[r][c] = ch

    for j, col in enumerate(cols):
        x0 = j * w + w // 2  # gate center
        for op in col:
            name = str(op.get("name",""))
            wires = list(op.get("wires", []))
            params = list(op.get("parameters", [])) if op.get("parameters") else []
            if len(wires) == 1:
                r = wires[0]
                label = _short_label(name, params)
                token = f"[{label}]"
                # place token centered around x0
                start = max(0, x0 - len(token)//2)
                for k,ch in enumerate(token):
                    place_char(r, start+k, ch)
            elif len(wires) >= 2:
                ws = sorted(wires)
                top, bot = ws[0], ws[-1]
                # vertical connector
                for r in range(top, bot+1):
                    place_char(r, x0, "│")
                upname = name.upper()
                if upname in {"CNOT","CX"}:
                    # control at first wire, target at last
                    place_char(wires[0], x0, "●")
                    place_char(wires[1], x0, "⊕")
                elif upname == "CZ" or upname in {"CPHASE","CP"}:
                    place_char(wires[0], x0, "●")
                    # label Z or CP at target
                    token = "[Z]" if upname=="CZ" else "[CP]"
                    start = x0 - len(token)//2
                    for k,ch in enumerate(token):
                        place_char(wires[1], start+k, ch)
                elif upname == "SWAP":
                    for r in wires:
                        place_char(r, x0, "×")
                elif upname in {"CCX","TOFFOLI"} and len(wires)==3:
                    c1,c2,t = wires
                    place_char(c1, x0, "●"); place_char(c2, x0, "●"); place_char(t, x0, "⊕")
                else:
                    # generic two-qubit box at all wires
                    token = f"[{_short_label(name, params)}]"
                    for r in wires:
                        start = x0 - len(token)//2
                        for k,ch in enumerate(token):
                            place_char(r, start+k, ch)

    # assemble
    out_lines = []
    for q in range(n_qubits):
        out_lines.append(f"q{q} ─{''.join(rows[q])}")
    return "\n".join(out_lines)


def circuit_mpl(
    n_qubits: int,
    ops: List[Dict[str, Any]],
    title: Optional[str] = None,
    save: Optional[str] = None,
    ax: Optional["plt.Axes"] = None,
    rounded: bool = True,
    theme: Optional[str] = None,
    badge: bool = False,
):
    """Render a Matplotlib circuit diagram.

    - If `ax` is provided, draw into that axis; otherwise create a new figure.
    - Returns (fig, ax) or None if matplotlib isn't available.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.patches import FancyBboxPatch  # type: ignore
        from matplotlib import font_manager as _fm  # type: ignore
    except Exception:
        return None
    cols = schedule_columns(n_qubits, ops)
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, len(cols) * 0.6), max(2, n_qubits * 0.6)))
        created_fig = True
    else:
        fig = ax.figure
    # Theme colors
    font_family: Optional[str] = None
    if theme == 'apple':
        face_1q = '#e8f5e9'   # light green
        face_2q = '#e3f2fd'   # very light blue
        face_cz = '#c8e6c9'   # greenish for Z/CP
        face_meas = '#fde2e2' # soft red
        line_color = '#222'
        title_color = '#2e7d32'
        # Pick first available Apple-ish font to avoid repeated findfont warnings
        try:
            for cand in ('SF Pro Display', 'SF Pro Text', 'Helvetica Neue', 'Arial'):
                try:
                    _ = _fm.findfont(cand, fallback_to_default=False)
                    font_family = cand
                    break
                except Exception:
                    continue
        except Exception:
            font_family = None
    else:
        face_1q = '#e0f7fa'
        face_2q = '#e8f5e9'
        face_cz = '#fff3e0'
        face_meas = '#fde2e2'
        line_color = '#222'
        title_color = '#222'

    # wires
    for r in range(n_qubits):
        ax.plot([0, len(cols)], [r, r], color="#444", lw=1)
        if font_family:
            ax.text(-0.2, r, f"q{r}", ha='right', va='center', fontsize=9, fontfamily=font_family)
        else:
            ax.text(-0.2, r, f"q{r}", ha='right', va='center', fontsize=9)
    # gates
    for j, col in enumerate(cols):
        x = j + _X_CENTER
        for op in col:
            name = str(op.get("name",""))
            wires = list(op.get("wires", []))
            params = list(op.get("parameters", [])) if op.get("parameters") else []
            if len(wires) == 1:
                r = wires[0]
                lbl = _short_label(name, params)
                face = face_meas if lbl == 'M' else face_1q
                if rounded:
                    box = FancyBboxPatch((x - _BOX_W/2, r - _BOX_H/2), _BOX_W, _BOX_H,
                                          boxstyle="round,pad=0.02,rounding_size=0.08",
                                          edgecolor=line_color, facecolor=face, zorder=3)
                else:
                    box = plt.Rectangle((x - _BOX_W/2, r - _BOX_H/2), _BOX_W, _BOX_H,
                                        edgecolor=line_color, facecolor=face, zorder=3)
                ax.add_patch(box)
                if font_family:
                    ax.text(x, r, lbl, ha='center', va='center', fontsize=8, zorder=4, fontfamily=font_family)
                else:
                    ax.text(x, r, lbl, ha='center', va='center', fontsize=8, zorder=4)
            else:
                ws = sorted(wires)
                ax.plot([x,x], [ws[0], ws[-1]], color=line_color, lw=1, zorder=1)
                up = name.upper()
                if up in {"CNOT","CX"} and len(wires)==2:
                    ax.plot(x, wires[0], 'o', color='#111', zorder=2)
                    ax.plot(x, wires[1], marker='$\oplus$', color='#111', zorder=2)
                elif up == "CZ" and len(wires)==2:
                    ax.plot(x, wires[0], 'o', color='#111', zorder=2)
                    if rounded:
                        box = FancyBboxPatch((x - _BOX_W/2, wires[1] - _BOX_H/2), _BOX_W, _BOX_H,
                                              boxstyle="round,pad=0.02,rounding_size=0.08",
                                              edgecolor=line_color, facecolor=face_cz, zorder=3)
                    else:
                        box = plt.Rectangle((x - _BOX_W/2, wires[1] - _BOX_H/2), _BOX_W, _BOX_H,
                                            edgecolor=line_color, facecolor=face_cz, zorder=3)
                    ax.add_patch(box)
                    if font_family:
                        ax.text(x, wires[1], 'Z', ha='center', va='center', fontsize=8, zorder=4, fontfamily=font_family)
                    else:
                        ax.text(x, wires[1], 'Z', ha='center', va='center', fontsize=8, zorder=4)
                elif up == "SWAP" and len(wires)==2:
                    ax.plot(x, wires[0], marker='$\times$', color='#111', zorder=2)
                    ax.plot(x, wires[1], marker='$\times$', color='#111', zorder=2)
                elif up in {"CCX","TOFFOLI"} and len(wires)==3:
                    ax.plot(x, wires[0], 'o', color='#111', zorder=2); ax.plot(x, wires[1], 'o', color='#111', zorder=2)
                    ax.plot(x, wires[2], marker='$\oplus$', color='#111', zorder=2)
                else:
                    for r in wires:
                        if rounded:
                            box = FancyBboxPatch((x - _BOX_W/2, r - _BOX_H/2), _BOX_W, _BOX_H,
                                                  boxstyle="round,pad=0.02,rounding_size=0.08",
                                                  edgecolor=line_color, facecolor=face_2q, zorder=3)
                        else:
                            box = plt.Rectangle((x - _BOX_W/2, r - _BOX_H/2), _BOX_W, _BOX_H,
                                                edgecolor=line_color, facecolor=face_2q, zorder=3)
                        ax.add_patch(box)
                        if font_family:
                            ax.text(x, r, _short_label(name, params), ha='center', va='center', fontsize=8, zorder=4, fontfamily=font_family)
                        else:
                            ax.text(x, r, _short_label(name, params), ha='center', va='center', fontsize=8, zorder=4)
    ax.set_xlim(0, max(1, len(cols)))
    # Put q0 at the top by flipping the y-axis
    ax.set_ylim(-0.5, n_qubits - 0.5)
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xticks([])
    if title:
        if font_family:
            ax.set_title(title, color=title_color, fontfamily=font_family)
        else:
            ax.set_title(title, color=title_color)
    if theme == 'apple' and created_fig:
        try:
            hdr = "mlxQ"
        except Exception:
            hdr = "mlxQ"
        if font_family:
            fig.text(0.01, 0.99, hdr, va='top', ha='left', fontsize=9, color=title_color, fontfamily=font_family)
        else:
            fig.text(0.01, 0.99, hdr, va='top', ha='left', fontsize=9, color=title_color)
        if badge:
            if font_family:
                fig.text(0.99, 0.02, "mlxQ", va='bottom', ha='right', fontsize=8, color='white',
                         bbox=dict(boxstyle="round,pad=0.2,rounding_size=0.2", facecolor="#444", edgecolor="#888"),
                         fontfamily=font_family)
            else:
                fig.text(0.99, 0.02, "mlxQ", va='bottom', ha='right', fontsize=8, color='white',
                         bbox=dict(boxstyle="round,pad=0.2,rounding_size=0.2", facecolor="#444", edgecolor="#888"))
    if created_fig:
        fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    return fig, ax


def random_circuit(n_qubits: int, depth: int, seed: int = 42, two_qubit_prob: float = 0.3) -> List[Dict[str, Any]]:
    """Generate a simple random circuit comprised of 1- and 2-qubit gates.
    Not intended for production; used to exercise the drawer and benches.
    """
    import random, math
    rnd = random.Random(seed)
    ops: List[Dict[str, Any]] = []
    oneq = ["H","X","Y","Z","RX","RY","RZ"]
    twoq = ["CNOT","CZ","SWAP"]
    for _ in range(depth):
        if rnd.random() < two_qubit_prob and n_qubits >= 2:
            g = rnd.choice(twoq)
            i = rnd.randrange(n_qubits)
            j = rnd.randrange(n_qubits-1)
            if j >= i: j += 1
            ops.append({"name": g, "wires": [min(i,j), max(i,j)]})
        else:
            g = rnd.choice(oneq)
            q = rnd.randrange(n_qubits)
            if g in {"RX","RY","RZ"}:
                th = (rnd.random()-0.5) * 2.0
                ops.append({"name": g, "wires": [q], "parameters": [th]})
            else:
                ops.append({"name": g, "wires": [q]})
    return ops
