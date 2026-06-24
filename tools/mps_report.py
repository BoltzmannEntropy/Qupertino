#!/usr/bin/env python3
"""
MPS Report Aggregator
---------------------

Scans bench/ for per-benchmark MPS summary CSVs (<key>_mps_summary.csv)
and emits two artifacts:

- bench/mps_report.json: machine-readable aggregation with per-key stats
- bench/mps_report.md: human-readable summary tables

Usage:
  python3 tools/mps_report.py [--bench bench]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def load_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    try:
        with open(path, 'r') as f:
            rd = csv.DictReader(f)
            for r in rd:
                rows.append(r)
    except Exception:
        pass
    return rows


def summarize(rows: List[Dict[str, str]]) -> Dict[str, float]:
    if not rows:
        return {}
    q = []; dmax = []; dmean = []; truncs = 0; tflag = 0; es = 0; est = 0
    for r in rows:
        try:
            q.append(int(r.get('qubits', 0)))
            dmax.append(int(float(r.get('bond_max', 0.0))))
            dmean.append(float(r.get('bond_mean', 0.0)))
            truncs += int(r.get('truncations', 0) or 0)
            tflag += (1 if str(r.get('truncated', 0)) in ('1','True','true') else 0)
            es += (1 if str(r.get('early_stop', 0)) in ('1','True','true') else 0)
            est += (1 if str(r.get('stopped_on_trunc', 0)) in ('1','True','true') else 0)
        except Exception:
            continue
    return {
        'count': len(rows),
        'qubits_min': min(q) if q else 0,
        'qubits_max': max(q) if q else 0,
        'bond_max_over_runs': max(dmax) if dmax else 0,
        'bond_mean_avg': (sum(dmean)/len(dmean)) if dmean else 0.0,
        'truncation_events_total': truncs,
        'truncation_runs': tflag,
        'early_stop_runs': es,
        'stop_on_trunc_runs': est,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', default='bench', help='bench output directory (default: bench)')
    args = ap.parse_args()
    bdir = Path(args.bench)
    bdir.mkdir(parents=True, exist_ok=True)
    # Find all *_mps_summary.csv files
    summaries = sorted(list(bdir.glob('*_mps_summary.csv')) + list(bdir.glob('*_mpsd_summary.csv')))
    if not summaries:
        print('No MPS summary CSVs found; nothing to do.')
        return
    # Aggregate per-key
    report: Dict[str, Dict] = {}
    for p in summaries:
        key = p.name.replace('_mps_summary.csv', '')
        rows = load_csv(p)
        report[key] = {
            'summary': summarize(rows),
            'rows': rows,
        }
    # Write JSON
    out_json = bdir / 'mps_report.json'
    try:
        with open(out_json, 'w') as f:
            json.dump(report, f, indent=2)
        print(f'MPS report JSON: {out_json}')
    except Exception as e:
        print(f'JSON write failed: {e}')
    # Write Markdown
    out_md = bdir / 'mps_report.md'
    try:
        with open(out_md, 'w') as f:
            f.write('# MPS Report\n\n')
            for key in sorted(report.keys()):
                f.write(f'## {key}\n\n')
                s = report[key]['summary']
                f.write('- runs: {}\n'.format(s.get('count', 0)))
                f.write('- qubits: {}..{}\n'.format(s.get('qubits_min', 0), s.get('qubits_max', 0)))
                f.write('- bond_max_over_runs: {}\n'.format(s.get('bond_max_over_runs', 0)))
                f.write('- bond_mean_avg: {:.3f}\n'.format(s.get('bond_mean_avg', 0.0)))
                f.write('- truncation_events_total: {}\n'.format(s.get('truncation_events_total', 0)))
                f.write('- truncation_runs: {}\n'.format(s.get('truncation_runs', 0)))
                f.write('- early_stop_runs: {}\n'.format(s.get('early_stop_runs', 0)))
                f.write('- stop_on_trunc_runs: {}\n'.format(s.get('stop_on_trunc_runs', 0)))
                f.write('\n')
        print(f'MPS report MD: {out_md}')
    except Exception as e:
        print(f'Markdown write failed: {e}')


if __name__ == '__main__':
    main()
