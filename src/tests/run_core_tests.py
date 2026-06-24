import time
import platform
from datetime import datetime
import sys
import os

# Ensure local package path so direct execution works without PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mlxq.pretty import console, info, success, warn, error, table

import importlib.util


def main():
    start = time.time()
    # Load primary core tests and cuQuantum parity tests
    modules = []
    for fname, mname in [
        ('mlxQCoreTest.py','mlxQCoreTest'),
        ('mlxQInternalConsistencyTest.py','mlxQInternalConsistencyTest'),
        ('mlxQVisualizationPlotsTest.py','mlxQVisualizationPlotsTest'),
        ('mlxQMeasurementParityTest.py','mlxQMeasurementParityTest'),
        ('mlxQQmlWrapperTest.py','mlxQQmlWrapperTest'),
        ('mlxQQCExamplesTest.py','mlxQQCExamplesTest'),
        ('mlxQMpsBackendTest.py','mlxQMpsBackendTest'),
        ('mlxQMpsParamSuiteTest.py','mlxQMpsParamSuiteTest'),
    ]:
        module_path = os.path.join(os.path.dirname(__file__), fname)
        if not os.path.exists(module_path):
            continue
        spec = importlib.util.spec_from_file_location(mname, module_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        modules.append(mod)

    # header
    console.print("""
[bold cyan]
╔══════════════════════════════════════════════════════════════╗
║              mlx-Quantum Python Core Test Suite             ║
║           Testing Operations, States, and QASM              ║
╚══════════════════════════════════════════════════════════════╝
[/bold cyan]
""")

    # system info
    rows = [
        ("Python", sys.version.split()[0]),
        ("Platform", platform.platform()),
        ("Timestamp", datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
    ]
    table("System Information", ("Key", "Value"), rows)

    # collect test functions (from all loaded modules)
    tests = []
    for mod in modules:
        for name in dir(mod):
            if name.startswith('test_') and callable(getattr(mod, name)):
                tests.append((name, getattr(mod, name)))

    tests.sort(key=lambda x: x[0])

    passed = 0
    failed = 0
    results = []
    for name, fn in tests:
        try:
            t0 = time.time()
            fn()
            dt = (time.time() - t0) * 1000.0
            console.print(f"[green]PASS[/green] {name}  ({dt:.2f} ms)")
            results.append((name, "PASS", f"{dt:.2f} ms"))
            passed += 1
        except Exception as e:
            dt = (time.time() - t0) * 1000.0
            console.print(f"[red]FAIL[/red] {name}  ({dt:.2f} ms): {e}")
            results.append((name, "FAIL", f"{dt:.2f} ms"))
            failed += 1

    total = passed + failed
    table("Test Summary", ("Test", "Result", "Time"), results)
    if failed == 0:
        success(f"\n{passed}/{total} tests passed ✅")
    else:
        error(f"\n{passed}/{total} tests passed, {failed} failed ❌")

    console.print(f"\n[bold]Total time:[/bold] {(time.time() - start)*1000.0:.2f} ms")


if __name__ == '__main__':
    main()
