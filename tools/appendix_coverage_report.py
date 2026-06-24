#!/usr/bin/env python3
"""
Generate a rough mapping between appendix tests (in paper/prx-quantum/appendix)
and Python test functions (in src/tests). Output Markdown + JSON under refs/.

Heuristic: tokenize appendix Test titles and Python test function names and
compute token overlap. This is not perfect but highlights traceability.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / 'paper' / 'prx-quantum' / 'appendix'
TESTS_DIR = ROOT / 'src' / 'tests'
OUT_DIR = ROOT / 'refs'


# Pattern for \Test[<label>]{<title>} or \Test{<title>}
TEST_PATTERN = re.compile(r"\\Test(?:\[(?P<label>[^\]]+)\])?\{(?P<title>[^}]+)\}")
DEF_PATTERN = re.compile(r"^def (test_[A-Za-z0-9_]+)\(\):", re.M)


def tokenize(s: str) -> List[str]:
    s = s.lower()
    # Replace non-alphanumerics with space, keep letters/digits
    s = re.sub(r"[^a-z0-9]+", " ", s)
    tokens = [t for t in s.split() if t]
    # Common normalizations
    norm = {
        'cphase': 'cz', 'cz': 'cz', 'cnot': 'cnot', 'swap': 'swap', 'iswap': 'iswap',
        'toffoli': 'toffoli', 'ccx': 'toffoli', 'fredkin': 'fredkin', 'cswap': 'fredkin',
        'qft': 'qft', 'iqft': 'iqft', 'bell': 'bell', 'ghz': 'ghz', 'w3': 'w3', 'w': 'w',
        'bloch': 'bloch', 'global': 'global', 'phase': 'phase', 'u3': 'u3', 'rz': 'rz',
        'ry': 'ry', 'rx': 'rx', 'entropy': 'entropy', 'negativity': 'negativity', 'concurrence': 'concurrence',
        'kraus': 'kraus', 'choi': 'choi', 'ptrace': 'ptrace', 'partial': 'partial', 'transpose': 'transpose'
    }
    mapped = [norm.get(t, t) for t in tokens]
    return mapped


def load_appendix_tests() -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for tex in APP_DIR.rglob('*.tex'):
        text = tex.read_text(encoding='utf-8', errors='ignore')
        for m in TEST_PATTERN.finditer(text):
            label = (m.group('label') or '').strip()
            title = m.group('title').strip()
            items.append({
                'file': str(tex.relative_to(ROOT)),
                'label': label,
                'title': title,
            })
    return items


def load_python_tests() -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for py in TESTS_DIR.glob('*.py'):
        text = py.read_text(encoding='utf-8', errors='ignore')
        for m in DEF_PATTERN.finditer(text):
            out.append((str(py.relative_to(ROOT)), m.group(1)))
    return out


def score(title_tokens: List[str], name_tokens: List[str]) -> int:
    # Simple overlap score
    ta = set(title_tokens)
    tb = set(name_tokens)
    return len(ta & tb)


def main():
    appendix = load_appendix_tests()
    pytests = load_python_tests()
    pytest_tokens = [
        (path, name, tokenize(name.replace('test_', '')))
        for (path, name) in pytests
    ]
    records = []
    matched = 0
    for item in appendix:
        tks = tokenize(item['title'])
        scored = []
        for path, name, nt in pytest_tokens:
            s = score(tks, nt)
            if s > 0:
                scored.append((s, path, name))
        scored.sort(key=lambda x: (-x[0], x[2]))
        top = [{'score': s, 'file': p, 'test': n} for (s, p, n) in scored[:3]]
        if top:
            matched += 1
        records.append({
            'label': item['label'],
            'title': item['title'],
            'file': item['file'],
            'matches': top,
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'appendix_coverage_map.json').write_text(json.dumps({
        'summary': {
            'appendix_tests': len(appendix),
            'python_tests': len(pytests),
            'with_matches': matched,
            'without_matches': len(appendix) - matched,
        },
        'items': records,
    }, indent=2), encoding='utf-8')

    # Markdown report
    lines: List[str] = []
    lines.append('# Appendix → Python Tests Coverage Map')
    lines.append('')
    lines.append(f'- Appendix tests: {len(appendix)}')
    lines.append(f'- Python tests: {len(pytests)}')
    lines.append(f'- Items with ≥1 match: {matched}')
    lines.append(f'- Items with no match: {len(appendix) - matched}')
    lines.append('')
    lines.append('| Appendix Label | Title | Appendix File | Top Matches |')
    lines.append('|---|---|---|---|')
    for r in records:
        top = ', '.join(f"`{m['test']}` ({m['score']})" for m in r['matches']) or '—'
        lines.append(f"| {r['label'] or '—'} | {r['title']} | `{r['file']}` | {top} |")
    (OUT_DIR / 'appendix_coverage_map.md').write_text('\n'.join(lines), encoding='utf-8')

    print('Wrote refs/appendix_coverage_map.{json,md}')


if __name__ == '__main__':
    main()
