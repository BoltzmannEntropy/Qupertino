import os
import json
from pathlib import Path


def main(json_path: str = "bench/qasm_MLX_python.json",
         tex_out: str = "paper/generated/qasm_suite_table.tex"):
    with open(json_path, 'r') as f:
        data = json.load(f)
    suite = data.get('suite', [])
    summary = data.get('summary', {})
    rows = []
    rows.append("\\begin{tabular}{lrrrrrl}")
    rows.append("\\toprule")
    rows.append("Circuit & Qubits & Gates & Depth & Wall (ms) & Peak (MB) & Status \\\\")
    rows.append("\\midrule")
    for item in suite:
        name = item.get('file','')
        n = item.get('qubits','')
        g = item.get('gates','')
        depth = item.get('depth','')
        wall = item.get('wall_ms','')
        mem = item.get('peak_delta_mb','')
        st = item.get('status','')
        rows.append(f"{name} & {n} & {g} & {depth} & {wall} & {mem} & {st} \\\\")
    rows.append("\\midrule")
    rows.append(f"Total & & & {summary.get('wall_ms',0):.2f} & {summary.get('peak_rss_mb',0):.2f} & {summary.get('passed',0)}/{summary.get('total',0)} \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")

    out_path = Path(tex_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows))
    print(f"Wrote {tex_out}")


if __name__ == '__main__':
    main()
