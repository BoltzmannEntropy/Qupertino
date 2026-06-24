import re
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional


_re_reg = re.compile(r"^(qreg|creg)\s+(\w+)\[(\d+)\];")
_re_gate = re.compile(r"^(\w+)(?:\(([^)]*)\))?\s+([^;]+);")
_re_qubit = re.compile(r"(\w+)\[(\d+)\]")


def _trim(s: str) -> str:
    return s.strip()


def _parse_params(s: str) -> List[float]:
    if not s:
        return []
    parts = [p.strip() for p in s.split(',')]
    out: List[float] = []
    for p in parts:
        v = _parse_pi_expr(p)
        out.append(v)
    return out


def _parse_pi_expr(p: str) -> float:
    t = p.replace(' ', '').lower()
    if t == 'pi':
        import math
        return math.pi
    m = re.match(r"^([+-]?[0-9]*\.?[0-9]+)?\*?pi(?:/([0-9]*\.?[0-9]+))?$", t)
    if m:
        import math
        k = float(m.group(1) or 1.0)
        b = float(m.group(2) or 1.0)
        return k * math.pi / b
    try:
        return float(t)
    except Exception:
        return 0.0


def parse_qasm_file(path: str) -> Tuple[int, List[Dict[str, Any]]]:
    """Parse a subset of OpenQASM 2.0, returning (n_qubits, ops).

    Supported features:
    - qreg/creg declarations (creg ignored)
    - Single/two/three-qubit primitive gates via aliases in _normalize_gate
    - Register-wide single-qubit ops (e.g., "x b;")
    - User-defined gates with inlining, including nested gate calls
    - Skips measure/reset/if/barrier/include/OPENQASM lines
    """
    # Resolve path robustly for both layouts:
    # - repo_root/datasets/qasm/local/*.qasm (preferred)
    # - legacy: repo_root/qasm_circuits/*.qasm or src/qasm_circuits/*.qasm
    p = Path(path)
    candidates = [p]
    if not p.exists():
        parts = Path(path).parts
        # extract relative filename (last path segment) for fallback search
        rel = Path(p.name)
        # Search upward from this file location
        here = Path(__file__).resolve()
        ups = list(here.parents)
        for up in ups[:5]:
            candidates.extend([
                up / 'datasets' / 'qasm' / 'local' / rel,
                up / 'src' / 'qasm_circuits' / rel,
                up / 'qasm_circuits' / rel,
            ])
        # Also try CWD roots
        cwd = Path.cwd()
        candidates.extend([
            cwd / 'datasets' / 'qasm' / 'local' / rel,
            cwd / 'src' / 'qasm_circuits' / rel,
            cwd / 'qasm_circuits' / rel,
        ])
    fp = None
    for c in candidates:
        if c.exists():
            fp = c
            break
    if fp is None:
        raise FileNotFoundError(f"QASM file not found: {path}")

    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
        raw_lines = f.readlines()

    # strip comments and whitespace
    lines = [ln.split('//', 1)[0].strip() for ln in raw_lines]

    # register mapping to contiguous global wires
    qbase: int = 0
    qreg_base: Dict[str, int] = {}
    qreg_size: Dict[str, int] = {}
    n_qubits = 0
    ops: List[Dict[str, Any]] = []

    # gate definitions: NAME -> {params: [names], args: [qubit args], body: [ops as tuples]}
    gate_defs: Dict[str, Dict[str, Any]] = {}

    i = 0
    N = len(lines)
    while i < N:
        raw = lines[i]
        i += 1
        if not raw:
            continue
        # Skip non-unitary/effect ops
        if raw.startswith('if(') or raw.startswith('reset') or raw.startswith('measure') or raw.startswith('barrier') or raw.startswith('include') or raw.startswith('OPENQASM'):
            continue

        # qreg/creg
        m = _re_reg.match(raw)
        if m:
            kind, name, sz = m.group(1), m.group(2), int(m.group(3))
            if kind == 'qreg':
                qreg_base[name] = qbase
                qreg_size[name] = sz
                qbase += sz
                n_qubits = qbase
            # creg ignored
            continue

        # gate definition with body
        if raw.startswith('gate '):
            # Examples:
            # gate name a,b,c {
            # gate g(p1,p2) a,b { 
            header = raw
            # collect lines until matching '}'
            body_lines: List[str] = []
            # If header does not contain '{', continue to next lines to find it
            if '{' not in header:
                # consume lines until '{'
                while i < N and '{' not in lines[i]:
                    header += ' ' + lines[i]
                    i += 1
                if i < N:
                    header += ' ' + lines[i]
                    i += 1
            # Now collect body until '}'
            brace_depth = header.count('{') - header.count('}')
            while i < N and brace_depth > 0:
                ln = lines[i]
                i += 1
                brace_depth += ln.count('{') - ln.count('}')
                if ln and not ln.startswith('}'):
                    body_lines.append(ln)

            # parse header for name, param names, arg names
            m2 = re.match(r"^gate\s+(\w+)\s*(?:\(([^)]*)\))?\s*([^\{]+)\{", header)
            if not m2:
                continue
            gname = m2.group(1)
            param_names = [p.strip() for p in (m2.group(2) or '').split(',') if p.strip()]
            arg_names = [a.strip() for a in (m2.group(3) or '').split(',') if a.strip()]
            def_name = _normalize_gate(gname)
            body_ops = _parse_gate_body(body_lines)
            gate_defs[def_name] = {"params": param_names, "args": arg_names, "body": body_ops}
            continue

        # opaque declarations: ignore
        if raw.startswith('opaque '):
            continue

        # ordinary gate/application line
        m = _re_gate.match(raw)
        if not m:
            continue
        gname = m.group(1)
        op_name = _normalize_gate(gname)
        params = _parse_params(m.group(2) or '')
        targets = m.group(3)

        # explicit reg[index] targets
        wires: List[int] = []
        for (reg, idx) in _re_qubit.findall(targets):
            if reg in qreg_base:
                wires.append(qreg_base[reg] + int(idx))

        # bare register tokens (e.g., "x b;") → only for single‑qubit ops
        if not wires:
            # split by comma, handle tokens without brackets
            tokens = [t.strip() for t in targets.split(',') if t.strip()]
            bare_regs = [t for t in tokens if '[' not in t and t in qreg_base]
            if bare_regs and _is_single_qubit(op_name):
                for reg in bare_regs:
                    base = qreg_base[reg]
                    size = qreg_size.get(reg, 0)
                    for off in range(size):
                        ops.append({"name": _normalize_u_alias(op_name, params)[0],
                                    "wires": [base + off],
                                    "parameters": list(_normalize_u_alias(op_name, params)[1])})
                continue

        # inline user‑defined gate if present
        if op_name in gate_defs:
            actual_wires = wires
            actual_params = params
            expanded = _expand_gate(op_name, actual_wires, actual_params, gate_defs)
            ops.extend(expanded)
            continue

        norm_name, norm_params = _normalize_u_alias(op_name, params)
        # controlled phase aliases
        if norm_name in ('CU1', 'CP'):
            norm_name = 'CPHASE'

        if not wires and n_qubits > 0 and _is_single_qubit(norm_name):
            # Fallback global register‑wide when no explicit/bare targets (rare)
            for q in range(n_qubits):
                ops.append({"name": norm_name, "wires": [q], "parameters": list(norm_params)})
            continue

        ops.append({"name": norm_name, "wires": wires, "parameters": list(norm_params)})

    return n_qubits, ops

def _parse_gate_body(body_lines: List[str]) -> List[Tuple[str, List[str], List[str]]]:
    """Parse lines within a gate body into a list of (name, param_tokens, arg_names)."""
    body_ops: List[Tuple[str, List[str], List[str]]] = []
    for ln in body_lines:
        ln = ln.strip()
        if not ln or ln.startswith('//'):
            continue
        m = _re_gate.match(ln)
        if not m:
            continue
        name = _normalize_gate(m.group(1))
        params_raw = m.group(2) or ''
        param_tokens = [p.strip() for p in params_raw.split(',') if p.strip()]
        targets = m.group(3)
        # targets in body are arg names, not qreg[index]
        arg_names = [t.strip() for t in targets.split(',') if t.strip()]
        body_ops.append((name, param_tokens, arg_names))
    return body_ops

def _eval_param_token(tok: str, bindings: Optional[Dict[str, float]] = None) -> float:
    """Evaluate a parameter token possibly referencing a bound name or pi expressions."""
    if bindings and tok in bindings:
        return float(bindings[tok])
    return _parse_pi_expr(tok)

def _expand_gate(name: str, wires: List[int], params: List[float], gate_defs: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Inline a user-defined gate by substituting actual wires/params into its body.

    Supports nested user-defined gates.
    """
    if name not in gate_defs:
        return []
    gdef = gate_defs[name]
    formal_args: List[str] = gdef.get("args", [])
    formal_params: List[str] = gdef.get("params", [])
    body = gdef.get("body", [])

    # map formal args to actual wires
    if len(wires) != len(formal_args):
        # arity mismatch → skip
        return []
    arg_map: Dict[str, int] = {formal_args[i]: wires[i] for i in range(len(formal_args))}
    # map formal params to provided params (unused in current QASMBench files)
    p_map: Dict[str, float] = {formal_params[i]: params[i] for i in range(min(len(formal_params), len(params)))}

    out_ops: List[Dict[str, Any]] = []
    for (op_name, param_tokens, arg_names) in body:
        # build wire list by mapping arg names
        targ_wires: List[int] = []
        ok = True
        for an in arg_names:
            if an not in arg_map:
                ok = False
                break
            targ_wires.append(arg_map[an])
        if not ok:
            continue
        # evaluate params
        p_vals = [_eval_param_token(t, p_map) for t in param_tokens]
        # nested user-defined gate expansion
        if op_name in gate_defs:
            out_ops.extend(_expand_gate(op_name, targ_wires, p_vals, gate_defs))
        else:
            nname, nparams = _normalize_u_alias(op_name, p_vals)
            if nname in ('CU1', 'CP'):
                nname = 'CPHASE'
            out_ops.append({"name": nname, "wires": targ_wires, "parameters": list(nparams)})
    return out_ops


def _normalize_gate(g: str) -> str:
    m = g.lower()
    aliases = {
        'cnot': 'CNOT', 'cx': 'CNOT', 'cz': 'CZ', 'swap': 'SWAP', 'iswap': 'ISWAP',
        'x': 'X', 'y': 'Y', 'z': 'Z', 'h': 'H', 'ch': 'CH', 's': 'S', 'sdg': 'SDG', 't': 'T', 'tdg': 'TDG', 'sx': 'SX',
        'rx': 'RX', 'ry': 'RY', 'rz': 'RZ', 'p': 'U1', 'phase': 'U1', 'u1': 'U1', 'u2': 'U2', 'u3': 'U3', 'u': 'U',
        'crx': 'CRX','cry':'CRY','crz':'CRZ','cu1':'CU1','cp':'CP', 'cphase':'CP',
        'ccx': 'CCX', 'toffoli': 'CCX', 'cswap': 'CSWAP', 'fredkin': 'CSWAP',
    }
    return aliases.get(m, m.upper())


def _is_single_qubit(op: str) -> bool:
    return op in {'H','X','Y','Z','S','SDG','T','TDG','SX','RX','RY','RZ','U1','U2','U3'}

def _normalize_u_alias(op_name: str, params: List[float]) -> Tuple[str, List[float]]:
    """Map generic U to U3 with padded params and return possibly adjusted name/params."""
    if op_name == 'U':
        # alias 'u' to U3 with padding
        if len(params) == 1:
            return 'U3', [params[0], 0.0, 0.0]
        if len(params) == 2:
            return 'U3', [params[0], params[1], 0.0]
        p = (params + [0.0, 0.0, 0.0])[:3]
        return 'U3', p
    return op_name, params
