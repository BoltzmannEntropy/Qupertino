import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mlxq.mlxQpretty import info, warn, success
from mlxq.mlxQdraw import circuit_mpl, random_circuit
from mlxq.mlxQqasm import parse_qasm_file
from mlxq.paths import qasm_local_path


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _save_fig(fig, path: Path):
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    try:
        import matplotlib.pyplot as plt  # type: ignore
        plt.close(fig)
    except Exception:
        pass


def _plot_mlxq(n: int, ops, title: str, out_png: Path):
    res = circuit_mpl(n, ops, title=title, theme='apple', rounded=True)
    if res is None:
        warn(f"matplotlib not available; skipping {out_png.name}")
        return False
    fig, _ = res
    _save_fig(fig, out_png)
    return True


def _plot_pl(ops, out_png: Path):
    try:
        import pennylane as qml  # type: ignore
        n_wires = max(w for op in ops for w in op['wires']) + 1 if ops else 1
        dev = qml.device('default.qubit', wires=n_wires)
        op_map = {
            'H': lambda w,p: qml.Hadamard(wires=w[0]),
            'X': lambda w,p: qml.PauliX(wires=w[0]),
            'Y': lambda w,p: qml.PauliY(wires=w[0]),
            'Z': lambda w,p: qml.PauliZ(wires=w[0]),
            'RX': lambda w,p: qml.RX((p or [0.0])[0], wires=w[0]),
            'RY': lambda w,p: qml.RY((p or [0.0])[0], wires=w[0]),
            'RZ': lambda w,p: qml.RZ((p or [0.0])[0], wires=w[0]),
            'CNOT': lambda w,p: qml.CNOT(wires=tuple(w)),
            'CZ': lambda w,p: qml.CZ(wires=tuple(w)),
            'SWAP': lambda w,p: qml.SWAP(wires=tuple(w)),
            'CCX': lambda w,p: qml.Toffoli(wires=tuple(w)),
            'I': lambda w,p: qml.Identity(wires=w[0]),
        }
        @qml.qnode(dev)
        def _pl():
            for op in ops:
                fn = op_map.get(op['name'])
                if fn is not None:
                    fn(op['wires'], op.get('parameters', []))
            return qml.state()
        try:
            fig, _ = qml.draw_mpl(_pl)()
            _save_fig(fig, out_png)
            return True
        except Exception as e:
            warn(f"pennylane draw_mpl failed: {e}")
            return False
    except Exception as e:
        warn(f"pennylane not available: {e}")
        return False


def _side_by_side(img_left: Path, img_right: Path, out_png: Path, title: str):
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.image as mpimg  # type: ignore
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        for ax, p in zip(axes, [img_left, img_right]):
            if p.exists():
                ax.imshow(mpimg.imread(str(p)))
                ax.axis('off')
            else:
                ax.text(0.5, 0.5, f"missing: {p.name}", ha='center', va='center')
                ax.axis('off')
        fig.suptitle(title)
        _save_fig(fig, out_png)
        return True
    except Exception as e:
        warn(f"side_by_side failed: {e}")
        return False


def test_visualization_plots():
    info("Visualization: generate side-by-side (mlxQ vs PennyLane) circuit plots")
    out_dir = Path('bench')
    _ensure_dir(out_dir)

    # 1) Small circuit (3q)
    ops_small = [
        {"name": "H", "wires": [0]},
        {"name": "CNOT", "wires": [0,1]},
        {"name": "SWAP", "wires": [1,2]},
        {"name": "RZ", "wires": [2], "parameters": [0.3]},
    ]
    mlxq_small = out_dir / 'vis_small_mlxq.png'
    pl_small = out_dir / 'vis_small_pl.png'
    side_small = out_dir / 'vis_small_side_by_side.png'
    _plot_mlxq(3, ops_small, 'mlxQ: small circuit', mlxq_small)
    _plot_pl(ops_small, pl_small)
    _side_by_side(mlxq_small, pl_small, side_small, 'Small circuit (mlxQ vs PennyLane)')

    # 2) Random (5q, depth=10)
    ops_rand = random_circuit(5, depth=10, seed=7)
    mlxq_rand = out_dir / 'vis_random5_mlxq.png'
    pl_rand = out_dir / 'vis_random5_pl.png'
    side_rand = out_dir / 'vis_random5_side_by_side.png'
    _plot_mlxq(5, ops_rand, 'mlxQ: random (5q, d=10)', mlxq_rand)
    _plot_pl(ops_rand, pl_rand)
    _side_by_side(mlxq_rand, pl_rand, side_rand, 'Random 5q (mlxQ vs PennyLane)')

    # 3) QASM bell
    try:
        n, ops_bell = parse_qasm_file(qasm_local_path('bell.qasm'))
        mlxq_bell = out_dir / 'vis_qasm_bell_mlxq.png'
        pl_bell = out_dir / 'vis_qasm_bell_pl.png'
        side_bell = out_dir / 'vis_qasm_bell_side_by_side.png'
        _plot_mlxq(n, ops_bell, 'mlxQ: QASM bell', mlxq_bell)
        _plot_pl(ops_bell, pl_bell)
        _side_by_side(mlxq_bell, pl_bell, side_bell, 'QASM bell (mlxQ vs PennyLane)')
    except Exception as e:
        warn(f"QASM bell plot skipped: {e}")

    # 4) GHZ (3 qubits)
    try:
        ops_ghz = [
            {"name": "H", "wires": [0]},
            {"name": "CNOT", "wires": [0,1]},
            {"name": "CNOT", "wires": [1,2]},
        ]
        mlxq_ghz = out_dir / 'vis_ghz3_mlxq.png'
        pl_ghz = out_dir / 'vis_ghz3_pl.png'
        side_ghz = out_dir / 'vis_ghz3_side_by_side.png'
        _plot_mlxq(3, ops_ghz, 'mlxQ: GHZ (3q)', mlxq_ghz)
        _plot_pl(ops_ghz, pl_ghz)
        _side_by_side(mlxq_ghz, pl_ghz, side_ghz, 'GHZ (3q) — mlxQ vs PennyLane')
    except Exception as e:
        warn(f"GHZ plot skipped: {e}")

    success("Visualization plots generated under bench/")
