#!/usr/bin/env python3
"""
Textual TUI for mlx-Quantum tests and benchmarks.

Features:
- Checkbox to run the 100+ core tests (on by default)
- List of all benchmarks with checkboxes (all selected by default)
- Run Selected, Select All, Deselect All, Clear Log, Quit
- Live, colorful progress log (uses Textual)

If Textual is not installed, prints a helpful message and exits gracefully.
"""

import os
import sys
from pathlib import Path
from contextlib import contextmanager
from io import StringIO
from typing import List

# Ensure local package path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # add python/


def _fallback():
    print("Textual is not installed. Run one of:", flush=True)
    print("  - python python/scripts/bench.py", flush=True)
    print("  - pip install textual  # then re-run this TUI", flush=True)


_TEXTUAL_IMPORT_ERROR = None
try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical, Container
    from textual.reactive import reactive
    from textual.binding import Binding
    # Import core widgets first
    from textual.widgets import Header, Footer, Button, Checkbox, Label, Static, ProgressBar, Input
    # Prefer documented Slider (Textual 0.50+); fallback handled below
    try:
        from textual.widgets import Slider as _Slider  # type: ignore
        _HAS_SLIDER = True
    except Exception:
        _Slider = None  # type: ignore
        _HAS_SLIDER = False
    # Optional resizable splitters (drag to resize)
    try:
        from textual.widgets import Splitter as _Splitter  # type: ignore
        _HAS_SPLITTER = True
    except Exception:
        _Splitter = None  # type: ignore
        _HAS_SPLITTER = False
    # Text log widget name changed across versions; try both
    try:
        from textual.widgets import TextLog as _LogWidget  # type: ignore
    except Exception:
        from textual.widgets import Log as _LogWidget  # type: ignore
    _HAS_TEXTUAL = True
except Exception as e:
    _TEXTUAL_IMPORT_ERROR = e
    _HAS_TEXTUAL = False


if not _HAS_TEXTUAL:
    if __name__ == "__main__":
        print("Textual is not installed or not importable with this Python interpreter.")
        print(f"Interpreter: {sys.executable}")
        print(f"Version: {sys.version.split()[0]}")
        print("Try one of:")
        print("  - python -m pip install textual")
        print("  - python3 -m pip install textual")
        print("  - Use the same Python where textual was installed (e.g., python3.11)")
        print("  - Or run the non-TUI runner: python python/scripts/bench.py")
        if _TEXTUAL_IMPORT_ERROR is not None:
            print("\nImport error details:")
            try:
                import traceback
                traceback.print_exception(_TEXTUAL_IMPORT_ERROR)
            except Exception:
                print(repr(_TEXTUAL_IMPORT_ERROR))
    sys.exit(0)


# Lazy imports after path tweak
from importlib import util as _import_util
import platform
import subprocess
import shutil
from runpy import run_path

from mlxq.mlxQbench import run_scaling_benchmark, run_qasm_suite


BENCHMARKS: List[str] = [
    "hamiltonian_simulation",
    "time_evolution",
    "trotter",
    "steady_state",
    "random_circuit",
    "qcbm",
    "phase_estimation",
    "qft",
    "qaoa",
    "vqe",
    "variational_circuit",
    "grover",
    "ghz",
    "qasm",
]

# Display caps per benchmark (max real-qubit runs)
BENCH_CAP = {
    "steady_state": 10,
    "phase_estimation": 12,
    "qft": 12,
    "vqe": 15,
}


def _bench_label(name: str) -> str:
    if name == "qasm":
        return "Qasm (max 18 by env)"
    cap = BENCH_CAP.get(name, 25)
    return f"{name.replace('_', ' ').title()} (max {cap})"


def get_system_info() -> dict:
    info: dict = {
        "processor": platform.processor() or platform.machine(),
        "arch": platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "mem_gb": None,
        "gpu": "Unknown",
        "cores": os.cpu_count() or 0,
    }
    # macOS specific: sysctl for CPU brand and memsize
    if platform.system() == "Darwin":
        try:
            if shutil.which("sysctl"):
                cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
                info["processor"] = cpu or info["processor"]
                mem = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
                total = int(mem)
                info["mem_gb"] = f"{total / (1024**3):.1f} GB"
        except Exception:
            pass
        # Apple Silicon heuristic
        if "Apple" in (info.get("processor") or "") or any(s in (info.get("processor") or "") for s in ("M1","M2","M3","M4")):
            info["gpu"] = "Unified Memory (Apple Silicon)"
    # Fallback mem via sysconf where available
    if not info.get("mem_gb"):
        try:
            if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
                page = os.sysconf("SC_PAGE_SIZE")
                pages = os.sysconf("SC_PHYS_PAGES")
                total = int(page) * int(pages)
                info["mem_gb"] = f"{total / (1024**3):.1f} GB"
        except Exception:
            info["mem_gb"] = "Unknown"
    return info


class StdoutToLog:
    def __init__(self, log_widget):
        self.log = log_widget
        self._buffer = StringIO()
        self._orig = sys.stdout
        self._orig_err = sys.stderr

    def write(self, data: str):
        self._orig.write(data)
        for line in data.splitlines():
            if line.strip():
                self._emit(line)

    def flush(self):  # pragma: no cover
        try:
            self._orig.flush()
        except Exception:
            pass

    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.stdout = self._orig
        sys.stderr = self._orig_err

    def _emit(self, line: str):
        try:
            if hasattr(self.log, "write_line"):
                self.log.write_line(line)  # TextLog
            elif hasattr(self.log, "write"):
                self.log.write(line)  # Log
            else:
                print(line, flush=True)
        except Exception:
            print(line, flush=True)


class BenchTUI(App):
    BINDINGS = [
        Binding("r", "run_selected", "Run Selected"),
        Binding("a", "select_all", "Select All"),
        Binding("n", "select_none", "Deselect All"),
        Binding("c", "clear_log", "Clear Log"),
        Binding("f1", "show_help", "Help"),
        Binding("q", "quit", "Quit"),
    ]

    CSS = """
    Screen { background: $surface; }
    #title { content-align: center middle; height: 2; background: #90ee90; color: black; text-style: bold; }
    #menubar { height: 1; background: #7fdc7f; padding: 0 1; }
    #menubar Button { background: #6cc96c; color: black; text-style: bold; }
    #controls { padding: 1 2; }
    #sysinfo { padding: 0 2; }
    #bench-pane { padding: 1 2; border: round #90ee90; background: $panel; height: 20; overflow: auto; }
    #bench-panel { padding: 0 1; }
    #output { height: 1fr; layout: vertical; padding: 0 1; }
    #progress { height: 1; }
    #log { height: 1fr; overflow: auto; border: tall #90ee90; }
    .bench-grid { layout: vertical; height: 1fr; overflow: auto; }
    .bench-row { height: auto; padding: 0 1; }
    #bench-panel Label, #bench-panel Checkbox { color: $text; }
    .cap-min { width: 3; text-align: right; }
    .cap-slider { width: 40; }
    .cap-value { width: 4; text-align: right; }
    .cap-max { width: 3; text-align: left; }
    Splitter { background: $panel; border: round #90ee90; }
    #right-top { height: 3; }
    #right-bottom { height: 1fr; overflow: auto; }
    .max-input { width: 12; margin-left: 2; }
    Log, TextLog { height: 1fr; border: tall #90ee90; }
    .cap-bar { width: 50; color: #2e8b57; }
    Slider { width: 40; }
    #lower-menu { height: 1; background: #7fdc7f; padding: 0 1; }
    #lower-menu Button { background: #6cc96c; color: black; text-style: bold; }
    #status { height: 2; padding: 0 1; color: #2e8b57; border: round #90ee90; background: $panel; }
    #help { layer: overlay; dock: top; height: auto; width: 80%; margin: 2 10; padding: 1; border: round #90ee90; background: $panel; display: none; }
    Button { margin: 0 1; }
    Footer { dock: bottom; }
    """

    running = reactive(False)
    _abort = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("mlx-Quantum Bench & Test Suite", id="title")
        with Horizontal(id="menubar"):
            yield Button("File", id="menu-file")
            yield Button("Run", id="menu-run")
            yield Button("View", id="menu-view")
            yield Button("Help", id="menu-help")
        # System info panel
        sysi = get_system_info()
        with Container(id="sysinfo"):
            with Horizontal():
                yield Label(f"Processor: [bold]{sysi.get('processor','?')}[/bold]")
                yield Label(f"Cores: [bold]{sysi.get('cores','?')}[/bold]")
                yield Label(f"Memory: [bold]{sysi.get('mem_gb','?')}[/bold]")
                yield Label(f"GPU: [bold]{sysi.get('gpu','?')}[/bold]")
                yield Label(f"OS: [bold]{sysi.get('os','?')}[/bold]")
        # Bench selection pane spans full width under the menu (placed before 'Run Core Tests')
        with Container(id="bench-pane"):
            with Container(id="bench-panel"):
                yield Label("Benchmarks (all selected by default, max qubits shown):")
                with Container(classes="bench-grid"):
                    self.bench_cbs: List[Checkbox] = []
                    self.bench_caps: dict[str, Input] = {}
                    self.bench_sliders: dict[str, object] = {}
                    self.bench_bars: dict[str, Label] = {}
                    self.bench_values: dict[str, int] = {}
                    self.cap_labels: dict[str, Label] = {}
                    for name in BENCHMARKS:
                        with Horizontal(classes="bench-row"):
                            cb = Checkbox(_bench_label(name), value=True, id=f"bench-{name}")
                            self.bench_cbs.append(cb)
                            yield cb
                            default_cap = 18 if name == "qasm" else BENCH_CAP.get(name, 25)
                            yield Label("Max:")
                            if _HAS_SLIDER:
                                # Min label, slider, current max aligned
                                yield Label("1", classes="cap-min")
                                slider = _Slider(low=1, high=25, value=default_cap, id=f"slider-{name}", show_value=True)
                                self.bench_sliders[name] = slider
                                slider.add_class("cap-slider")
                                yield slider
                                val_label = Label(f"{default_cap}", id=f"label-{name}", classes="cap-value")
                                self.cap_labels[name] = val_label
                                yield val_label
                                yield Label("25", classes="cap-max")
                            else:
                                self.bench_values[name] = int(default_cap)
                                yield Label("1", classes="cap-min")
                                bar = Label(self._bar_str(int(default_cap)), id=f"bar-{name}", classes="cap-bar cap-slider")
                                self.bench_bars[name] = bar
                                yield bar
                                val_label = Label(f"{default_cap}", id=f"label-{name}", classes="cap-value")
                                self.cap_labels[name] = val_label
                                yield Button("-", id=f"dec-{name}")
                                yield val_label
                                yield Button("+", id=f"inc-{name}")
                                yield Label("25", classes="cap-max")
            with Horizontal():
                yield Button("Run Selected", id="run")
                yield Button("Select All", id="all")
                yield Button("Deselect All", id="none")
                yield Button("Clear Log", id="clear")
                yield Button("Quit", id="quit")

        # Core tests control below the benchmark pane
        with Container(id="controls"):
            with Horizontal():
                self.cb_tests = Checkbox("Run Core Tests (100+)", value=True)
                yield self.cb_tests

        # Output area (simple vertical layout for robust visibility)
        with Container(id="output"):
            self.progress = ProgressBar(total=100, id="progress")
            yield self.progress
            try:
                self.msg_log = _LogWidget(highlight=False, markup=True, id="log")
            except Exception:
                self.msg_log = _LogWidget(id="log")
            yield self.msg_log
            yield Static("Ready", id="status")
                with Container(id="bench-panel"):
                    yield Label("Benchmarks (all selected by default, max qubits shown):")
                    with Container(classes="bench-grid"):
                        self.bench_cbs: List[Checkbox] = []
                        self.bench_caps: dict[str, Input] = {}
                        self.bench_sliders: dict[str, object] = {}
                        self.bench_bars: dict[str, Label] = {}
                        self.bench_values: dict[str, int] = {}
                        self.cap_labels: dict[str, Label] = {}
                        for name in BENCHMARKS:
                            with Horizontal(classes="bench-row"):
                                cb = Checkbox(_bench_label(name), value=True, id=f"bench-{name}")
                                self.bench_cbs.append(cb)
                                yield cb
                                default_cap = 18 if name == "qasm" else BENCH_CAP.get(name, 25)
                                yield Label("Max:")
                                if _HAS_SLIDER:
                                    slider = _Slider(low=1, high=25, value=default_cap, id=f"slider-{name}", show_value=True)
                                    self.bench_sliders[name] = slider
                                    yield slider
                                    val_label = Label(f"{default_cap}", id=f"label-{name}")
                                    self.cap_labels[name] = val_label
                                    bar = Label(self._bar_str(int(default_cap)), id=f"bar-{name}", classes="cap-bar")
                                    self.bench_bars[name] = bar
                                    yield bar
                                    yield val_label
                                else:
                                    self.bench_values[name] = int(default_cap)
                                    bar = Label(self._bar_str(int(default_cap)), id=f"bar-{name}", classes="cap-bar")
                                    self.bench_bars[name] = bar
                                    yield bar
                                    val_label = Label(f"{default_cap}", id=f"label-{name}")
                                    self.cap_labels[name] = val_label
                                    yield Button("-", id=f"dec-{name}")
                                    yield val_label
                                    yield Button("+", id=f"inc-{name}")
                with Horizontal():
                    yield Button("Run Selected", id="run")
                    yield Button("Select All", id="all")
                    yield Button("Deselect All", id="none")
                    yield Button("Clear Log", id="clear")
                    yield Button("Quit", id="quit")
            with Container(id="right"):
                self.progress = ProgressBar(total=100)
                yield self.progress
                # Ensure log is readable and scrollable
                try:
                    self.msg_log = _LogWidget(highlight=False, markup=True)
                except Exception:
                    self.msg_log = _LogWidget()
                yield self.msg_log
                yield Static("Ready", id="status")
        with Horizontal(id="lower-menu"):
            yield Button("F1 Help", id="help-btn")
            yield Button("R Run", id="run-btn")
            yield Button("A All", id="all-btn")
            yield Button("N None", id="none-btn")
            yield Button("C Clear", id="clear-btn")
            yield Button("Q Quit", id="quit-btn")
        # Help overlay
        self.help_panel = Static("""
[b]mlx-Quantum TUI[/b]\n\nR: Run Selected  |  A: Select All  |  N: Deselect All\nC: Clear Log     |  F1: Help       |  Q: Quit\n\nAdjust max qubits with the slider (or +/-), then press Run.
""", id="help")
        yield self.help_panel
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "quit":
            self.exit()
        elif bid in {"quit-btn"}:
            self.exit()
        elif bid in {"help-btn", "menu-help"}:
            self.action_show_help()
        elif bid == "all":
            for cb in self.bench_cbs:
                cb.value = True
        elif bid in {"all-btn"}:
            for cb in self.bench_cbs:
                cb.value = True
        elif bid == "none":
            for cb in self.bench_cbs:
                cb.value = False
        elif bid in {"none-btn"}:
            for cb in self.bench_cbs:
                cb.value = False
        elif bid == "clear":
            self.msg_log.clear()
        elif bid in {"clear-btn"}:
            self.msg_log.clear()
        elif bid == "run":
            if not self.running:
                self._start_run()
        elif bid in {"run-btn", "menu-run"}:
            if not self.running:
                self._start_run()
        elif bid.startswith("inc-"):
            self._adjust_bench_value(bid.replace("inc-", ""), +1)
        elif bid.startswith("dec-"):
            self._adjust_bench_value(bid.replace("dec-", ""), -1)

    def _benchmarks_selected(self) -> List[str]:
        names: List[str] = []
        for cb in self.bench_cbs:
            if cb.value and cb.id and cb.id.startswith("bench-"):
                names.append(cb.id.replace("bench-", ""))
        return names

    def _start_run(self) -> None:
        sel = self._benchmarks_selected()
        do_tests = bool(self.cb_tests.value)
        steps = (1 if do_tests else 0) + len(sel)
        if steps == 0:
            self._log_line("[yellow]Nothing selected[/yellow]")
            return
        self.progress.update(total=max(steps, 1), progress=0)
        self.running = True
        self._abort = False
        # Disable controls while running
        self._set_controls_enabled(False)
        # Launch in worker thread
        self.run_worker(lambda: self._run_suite(do_tests, sel), thread=True, exclusive=True)

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.cb_tests.disabled = not enabled
        for cb in self.bench_cbs:
            cb.disabled = not enabled

    def _run_suite(self, do_tests: bool, benches: List[str]) -> None:
        try:
            with StdoutToLog(self.msg_log):
                if do_tests:
                    self._log_line("[bold cyan]=== Running Core Tests (Python) ===[/bold cyan]")
                    try:
                        self.query_one("#status", Static).update("Running tests…")
                    except Exception:
                        pass
                    self._run_core_tests_module()
                    self.progress.advance(1)

                pub_qubits = [1,2,5,7,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
                vqe_qubits = [1,2,5,7,10,11,12,13,14,15]
                steady_qubits = [1,2,5,7,10,11,12,13,14,15]

                for name in benches:
                    cap = self._get_cap(name)
                    if name == "vqe":
                        qubits = [q for q in vqe_qubits if q <= cap]
                    elif name == "steady_state":
                        qubits = [q for q in steady_qubits if q <= cap]
                    else:
                        qubits = [q for q in pub_qubits if q <= cap]

                    if name == "qasm":
                        self._log_line("[bold cyan]=== OpenQASM Circuit Benchmarks ===[/bold cyan]")
                        os.environ['QASM_MAX_QUBITS'] = str(cap)
                        os.environ.setdefault('QASM_MAX_MEM_MB', '4096')
                        run_qasm_suite()
                    else:
                        self._log_line(f"[bold cyan]Scaling bench:[/bold cyan] {name} | qubits={qubits}")
                        run_scaling_benchmark(name, qubits, simulate_cap=cap, stop_fn=lambda: self._abort)
                    self.progress.advance(1)

                # Reports
                self._log_line("[bold cyan]=== Generating reports & plots ===[/bold cyan]")
                self._run_reports()
        except Exception as e:  # pragma: no cover
            self._log_line(f"[red]Error:[/red] {e}")
        finally:
            self.running = False
            self._set_controls_enabled(True)
            try:
                self.query_one("#status", Static).update("Ready")
            except Exception:
                pass

    def on_shutdown(self) -> None:  # called when app is exiting
        self._abort = True

    def _run_core_tests_module(self) -> None:
        # Prefer the rich, C++-style detailed run_all() from mlxQuantumCoreTest
        test_mod_path = Path(__file__).resolve().parents[1] / 'tests' / 'mlxQCoreTest.py'
        if test_mod_path.exists():
        spec = _import_util.spec_from_file_location("mlxQCoreTest", str(test_mod_path))
            mod = _import_util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            if hasattr(mod, 'run_all') and callable(getattr(mod, 'run_all')):
                mod.run_all()
                return
        # Fallback to the simple test runner
        tests_path = Path(__file__).resolve().parents[1] / 'tests' / 'run_core_tests.py'
        spec2 = _import_util.spec_from_file_location("mlxQCoreTestRunner", str(tests_path))
        mod2 = _import_util.module_from_spec(spec2)
        sys.modules[spec2.name] = mod2
        assert spec2.loader is not None
        spec2.loader.exec_module(mod2)

    def _run_reports(self) -> None:
        base = Path(__file__).resolve().parents[2] / 'src' / 'scripts'
        for script in [
            'generate_coretest_report.py',
            'generate_plots.py',
            'plot_all_benchmarks.py',
            'generate_report.py',
        ]:
            p = base / script
            if p.exists():
                try:
                    self._log_line(f"[green]Run[/green] {script}")
                    run_path(str(p))
                except SystemExit:
                    pass
                except Exception as e:
                    self._log_line(f"[yellow]{script} skipped:[/yellow] {e}")

    def _get_cap(self, name: str) -> int:
        try:
            if name in self.bench_sliders:
                s = self.bench_sliders[name]
                val = int(getattr(s, 'value', BENCH_CAP.get(name, 25)))
            elif name in self.bench_values:
                val = int(self.bench_values[name])
            else:
                inp = self.bench_caps.get(name)
                if not inp:
                    return BENCH_CAP.get(name, 25)
                val = int((inp.value or "").strip())
            if val <= 0:
                return BENCH_CAP.get(name, 25)
            # Normalize floor min to 1 (except qasm uses env limit)
            return max(1, val)
        except Exception:
            return BENCH_CAP.get(name, 25)

    def _adjust_bench_value(self, name: str, delta: int) -> None:
        try:
            if name in self.bench_sliders:
                s = self.bench_sliders[name]
                cur = int(getattr(s, 'value', BENCH_CAP.get(name, 25)))
                v = max(1, min(25, cur + int(delta)))
                # Update slider; this will trigger on_slider_changed and keep labels in sync
                try:
                    setattr(s, 'value', v)
                except Exception:
                    pass
                # Update labels / mirror value in case event doesn’t fire immediately
                self.bench_values[name] = v
                if name in self.cap_labels:
                    self.cap_labels[name].update(str(v))
                if name in self.bench_bars:
                    self.bench_bars[name].update(self._bar_str(v))
            else:
                v = int(self.bench_values.get(name, BENCH_CAP.get(name, 25)))
                v = max(1, min(25, v + int(delta)))
                self.bench_values[name] = v
                if name in self.cap_labels:
                    self.cap_labels[name].update(str(v))
                if name in self.bench_bars:
                    self.bench_bars[name].update(self._bar_str(v))
        except Exception:
            pass

    # Update numeric labels when sliders move (Textual docs API)
    def on_slider_changed(self, event) -> None:  # type: ignore
        try:
            # Get slider id and value from event in a version-tolerant way
            val = int(getattr(event, 'value', 0))
            sid = ''
            if hasattr(event, 'slider') and getattr(event, 'slider') is not None:
                sid = getattr(event.slider, 'id', '')
            elif hasattr(event, 'control') and getattr(event, 'control') is not None:
                sid = getattr(event.control, 'id', '')
            elif hasattr(event, 'sender') and getattr(event, 'sender') is not None:
                sid = getattr(event.sender, 'id', '')
            if not sid or not sid.startswith('slider-'):
                return
            name = sid.replace('slider-', '')
            if name in self.cap_labels:
                self.cap_labels[name].update(str(val))
            # Mirror value into bench_values to unify handling downstream
            self.bench_values[name] = val
            # Update ASCII bar too if present
            if name in self.bench_bars:
                self.bench_bars[name].update(self._bar_str(val))
        except Exception:
            pass

    def _bar_str(self, v: int, total: int = 25) -> str:
        v = max(0, min(total, int(v)))
        filled = "█" * v
        empty = "·" * (total - v)
        return f"[{filled}{empty}]"

    def _log_line(self, line: str) -> None:
        try:
            if hasattr(self.msg_log, "write_line"):
                self.msg_log.write_line(line)
            elif hasattr(self.msg_log, "write"):
                self.msg_log.write(line)
            else:
                print(line, flush=True)
        except Exception:
            print(line, flush=True)

    # Actions for key bindings / footer hints
    def action_run_selected(self) -> None:
        if not self.running:
            self._start_run()

    def action_select_all(self) -> None:
        for cb in self.bench_cbs:
            cb.value = True

    def action_select_none(self) -> None:
        for cb in self.bench_cbs:
            cb.value = False

    def action_clear_log(self) -> None:
        self.msg_log.clear()

    def action_show_help(self) -> None:
        try:
            self.help_panel.display = not self.help_panel.display
        except Exception:
            pass


if __name__ == "__main__":
    BenchTUI().run()
