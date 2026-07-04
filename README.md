<div align="center">
  <img src="quantumstudio/assets/app_icon_source.png" alt="Qupertino Logo" width="220"/>
  <h1>Qupertino / QuantumStudio</h1>
  <p><b>Apple Silicon quantum benchmarking stack</b> with a local simulator, reproducible benchmark pipelines, and a desktop UI.</p>
</div>

![QuantumStudio Dashboard](quantumstudio/assets/screenshots/screen003.png)

Qupertino is a three-part project: a self-published technical report, the Qupertino simulator stack, and the dedicated QuantumStudio desktop UI studio. There is no native MLX quantum simulator available today, so Qupertino provides a local simulator layer for QFT, QAOA, VQE, Hamiltonian workflows, and OpenQASM runs.

- Website: https://boltzmannentropy.github.io/QupertinoWEB/
- Repository: https://github.com/BoltzmannEntropy/Qupertino
- Author: **Shlomo Kashani**
- Technical report: unpublished; PDF and LaTeX source are kept locally and are not distributed in this repository.

## How to Cite

### Repository (Git / Software Citation)

```bibtex
@software{kashani_osxq_2026,
  author       = {Shlomo Kashani},
  title        = {Qupertino / QuantumStudio: Apple Silicon Quantum Benchmarking Stack},
  year         = {2026},
  url          = {https://github.com/BoltzmannEntropy/Qupertino},
  note         = {GitHub repository}
}
```

### Technical Report Citation

```bibtex
@techreport{kashani_mlxq_2026,
  author       = {Shlomo Kashani},
  title        = {Qupertino: Structured-Gate Kernel Dispatch for Quantum Circuit Simulation on Apple Silicon with MLX},
  year         = {2026},
  institution  = {Self-published technical report (unpublished)},
  url          = {https://github.com/BoltzmannEntropy/Qupertino}
}
```

## Extended Introduction

Qupertino exists to make Apple Silicon quantum benchmarking practical, local-first, and reproducible. The project packages three layers into one workflow:

1. **Research layer**: benchmark methodology and framing (see the included technical report).
2. **Simulator layer (`mlxq`)**: state-vector and MPS backends on MLX for Apple Silicon.
3. **Product layer (`QuantumStudio`)**: desktop run orchestration, monitoring, and export.

The core motivation is straightforward: Apple Silicon uses a unified memory model, but there is no default native MLX quantum runtime shipped as a platform quantum simulator. Qupertino fills that gap with a local simulator stack and benchmark harnesses for QFT, QAOA, VQE, QCBM, Grover, Hamiltonian/time-evolution workflows, and OpenQASM runs.

This repository is designed for publication-grade reproducibility:
- deterministic CLI run paths
- structured CSV/JSON output, raw timing distributions, and run manifests
- frozen artifact promotion under `assets/benchmarks-frozen/`
- UI+CLI parity for auditability

## Project Structure

- `src/` Python simulator + benchmark code (`mlxq`)
- `bench.sh` single/full benchmark launcher
- `bench_with_logging.sh` orchestrated benchmark runs with logging/promotions
- `bench/runs/` per-run outputs (`run_YYYYmmdd_HHMMSS`)
- `quantumstudio/` desktop UI + backend + control scripts
- `datasets/qasm/local/` OpenQASM local corpus
- `assets/benchmarks-frozen/` frozen benchmark artifacts and sample bundles

## Benchmark Context And Representative Results

Qupertino ships **two measured performance tiers**. The pure-MLX tier dispatches structured gate classes to specialized MLX kernels (diagonal gates as broadcast phase multiplies, controlled gates as masked half-state updates, SWAP as an axis permutation, runtime fusion of equal-angle ZZ Trotter layers). The **Metal shader tier** (`MLXQ_METAL_KERNELS=1`) adds hand-written kernels in `src/mlxq/shaders/` for every structured layer family, reached through semantics-preserving fusion detectors and parity-tested against the pure path (251 tests).

<div align="center">
  <img src="assets/perf-charts/chart_4way_25q.png" alt="Wall time at 25 qubits across four backends" width="820"/>
  <br/><em>25-qubit wall time, same machine, gate-identical circuits (log scale, lower is better). The Metal shader tier — blue — is fastest in every cell.</em>
</div>

Four-way interleaved campaign on M1 Max (two warmups, ten measured repeats per cell, all four backends round-robin inside every repeat, gate-identical circuits, mean seconds at 25q):

| Workload @ 25q | Qupertino Metal | Qupertino MLX | Aer CPU | PennyLane lightning |
| --- | ---: | ---: | ---: | ---: |
| QFT | **0.059** | 0.72 | 2.80 | 5.61 |
| Ring-QAOA (6 layers) | **0.150** | 2.07 | 5.35 | 6.84 |
| TFIM Trotter (20 steps) | **0.495** | 5.82 | 17.79 | 32.95 |
| Phase estimation | **0.105** | 0.91 | 4.05 | 6.06 |
| Grover proxy | **0.052** | 1.11 | 1.21 | 2.73 |
| GHZ | **0.022** | 0.27 | 0.69 | 0.42 |

The Metal tier is **fastest in all 18 comparison cells** (15/20/25 qubits), with 25-qubit paired per-repeat ratios of **23–47× over Aer** and **19–95× over PennyLane** — and its gate-stream QFT (59 ms) beats MLX's own `mx.fft` primitive (77 ms). The pure-MLX tier is fastest among the CPU-comparable trio on 4 of 6 workloads at 25q (Grover is a statistical tie with Aer; small gate-sparse circuits at 15q favor the CPU baselines). A paired ablation with dispatch disabled (`MLXQ_DENSE_ONLY=1`) attributes **25–33×** to kernel specialization itself. A separate campaign against PennyLane's OpenMP-parallel `lightning.kokkos` reached the same verdict for the pure tier (fastest in 16/18 cells). Full protocol, t-based CIs, raw artifacts: `paper/tqc-acm-2026/evidence_artifacts/`.

<p align="center">
  <img src="assets/perf-charts/chart_scaling_qft.png" alt="QFT scaling 15-25 qubits" width="410"/>
  <img src="assets/perf-charts/chart_scaling_tfim.png" alt="TFIM Trotter scaling 15-25 qubits" width="410"/>
</p>

### The Metal shader tier

Seven kernel families in one folder (`src/mlxq/shaders/`, design + measurements + codex-review record in its [README](src/mlxq/shaders/README.md)): phase-LUT diagonal layers (ZZ, CZ/CPHASE, grouped weighted couplings), GF(2) affine permutation gathers (any CNOT/X/SWAP run = one pass), fused tensor-product single-qubit layers (uniform and per-qubit-varying, with algebraic window collapse — Grover's H·H·X prologue becomes a single bit-flip gather), radix-4 QFT and Walsh–Hadamard butterflies, basis-conjugated XX/YY Trotter layers, and gate-stream QFT/iQFT stage detectors. Kernel-level at 25q:

| Layer @ 25q | Per-gate | MLX fused | Metal kernel |
| --- | ---: | ---: | ---: |
| ZZ Trotter layer (24 bonds) | 35.9 ms | 2.5 ms | **1.7 ms** |
| Full QFT (radix-4, 13 launches) | 748 ms | 255 ms | **21.1 ms** |
| RX layer (all 25 qubits) | 194 ms | — | **20.9 ms** |
| H layer (Walsh radix-4, 7 launches) | 698 ms | — | **12.7 ms** |

The shader tier also engages on the **OpenQASM import path**: the full local QASM corpus (40 executed circuits, QASMBench-derived) matches the pure path to ≤5e-7 everywhere, with structured circuits accelerating (ghz_state_n23 9.1×, cat_state_n22 8.0×, ising_n26 2.2×) and pre-decomposed cx+rz circuits (e.g. QASMBench qft_n18) correctly falling through to the pure path at parity — recognizing decomposed 2-qubit blocks is noted future work. Artifact: `paper/tqc-acm-2026/evidence_artifacts/qasm_shader_sweep_20260704/`.

Full-suite paired sweep (29 workloads at 25q, pure vs Metal alternating within every repeat): **26 of 29 workloads accelerate, 4.7–25×** for those with fusable layer structure (TFIM 2nd order 25×, long-range Ising 24×, Heisenberg 13.8×, variational 14.3×, QCBM 12×, Grover 11.5×). The three 1.0× rows are structural and documented (cross-family Trotter interleave, isolated CZs, strict RY/CNOT alternation).

<div align="center">
  <img src="assets/perf-charts/chart_speedup_sweep.png" alt="Metal shader speedup over pure MLX across 29 workloads" width="720"/>
  <br/><em>Every gate-based workload, Metal shaders vs pure MLX (paired, same session). Blue ≥ 4×, green = VQE (residual is energy evaluation), grey = structurally unfusable and correctly left at parity.</em>
</div>

Four Codex CLI review rounds shaped and audited the kernels (archived under `paper/tqc-acm-2026/reviews_shaders_v2/`), catching real bugs: a wrong radix-4 derivation (retracted by the reviewer itself), a float32/rtol hole that silently dropped small-angle gates, and 1.25e-5 phase drift in 300-term products (fixed with grouped double-precision LUTs).

### Full-suite refresh (all 21 benchmark families, `bench.sh --repro`)

The complete batch (one warmup, five measured repeats per qubit point, run `bench/runs/run_20260702_131646`, summaries frozen under `paper/quantics-lncs-2026/evidence_artifacts/full_suite_20260702/`) confirms the kernel rewrite across every family. Gains vs the archived pre-rewrite snapshot at 25 qubits:

| Family | Archived | Current | Gain |
| --- | ---: | ---: | ---: |
| Grover | 113.3 s | 1.74 s | 65× |
| Random circuit | 22.4 s | 1.80 s | 12× |
| QCBM (9 layers) | 26.3 s | 2.35 s | 11× |
| QFT | 7.03 s | 0.82 s | 8.5× |
| Time evolution | 30.1 s | 3.93 s | 7.7× |
| Variational circuit | 19.2 s | 3.29 s | 5.8× |
| GHZ | 0.89 s | 0.16 s | 5.7× |
| QAOA (ring) | 11.1 s | 2.70 s | 4.1× |

New coverage: the Heisenberg/XXZ/random-field/long-range/ladder evolution variants run in 18–33 s at 25q (three Pauli-pair sweeps per Trotter step). Known remaining slow path: the density-matrix `steady_state` diagnostic (~292 s at its 12-qubit cap) still uses the generic dense path.

Current benchmark runners prefer the project-local `.runtime-venv/bin/python` when it exists, then `.venv/bin/python`, then PATH `python3`. Set `PYTHON_BIN=/path/to/python` only when you intentionally want to override that default.

## Installation (Full)

### 1) System prerequisites

- macOS 13.3+ recommended
- Apple Silicon (M1/M2/M3/M4)
- Python 3.10+ (3.11 tested heavily)
- Optional: Flutter SDK (for UI development builds)

### 2) Core Python environment

```bash
cd /Volumes/SSD4tb/Dropbox/DSS/artifacts/code/QuantumStudioPRJ/QuantumStudioCODE
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

`pyproject.toml` includes core dependencies (`mlx`, `rich`, `numpy`) and optional groups.

### 3) Backend dependencies (QuantumStudio API)

```bash
pip install -r quantumstudio/backend/requirements.txt
```

### 4) Runtime path for local commands

```bash
export PYTHONPATH=src
```

### 5) Flutter UI dependencies (optional, for UI dev)

```bash
cd quantumstudio/flutter_app
flutter pub get
```

## Running Benchmarks

### Standard full-orchestrated run

```bash
cd /Volumes/SSD4tb/Dropbox/DSS/artifacts/code/QuantumStudioPRJ/QuantumStudioCODE
./bench_with_logging.sh
```

### 12-qubit smoke test (recommended health check)

```bash
cd /Volumes/SSD4tb/Dropbox/DSS/artifacts/code/QuantumStudioPRJ/QuantumStudioCODE
PYTHON_BIN=/Users/sol/.pyenv/shims/python3 ./bench_with_logging.sh --frozen-parity-12
```

### Single-circuit run example

```bash
PYTHON_BIN=/Users/sol/.pyenv/shims/python3 ./bench.sh --circuit variational_circuit --simulate-limit 12 --qubits 1,2,5,7,10,11,12
```

### Reproducible timing run

```bash
./bench.sh --circuit ghz --simulate-limit 2 --qubits 2 --no-save-plots --warmups 1 --repeats 2
```

Each benchmark run writes the legacy plotting files plus `*_raw_runs.csv`, `*_raw_runs.json`, `*_timing_summary.csv`, and `run_manifest.json`. QuantumStudio exposes the same warmup and repeat controls under Advanced Options.

## Benchmark Catalog (Detailed)

The benchmark engine accepts the following circuit keys. For each benchmark below:
- Use the single-circuit form:
  - `./bench.sh --circuit <key> --simulate-limit <N> --qubits 1,2,5,7,10,11,12`
- Or run coordinated suites with:
  - `./bench_with_logging.sh --frozen-parity-12`

### 1) `hamiltonian_simulation`
- Purpose: product-formula simulation of spin Hamiltonians.
- Typical command:
```bash
./bench.sh --circuit hamiltonian_simulation --simulate-limit 12 --qubits 1,2,5,7,10,11,12
```

### 2) `time_evolution`
- Purpose: time-evolution scaling over qubit count.
- Typical command:
```bash
./bench.sh --circuit time_evolution --simulate-limit 12 --qubits 1,2,5,7,10,11,12
```

### 3) `trotter`
- Purpose: Trotterized evolution workload for depth/runtime growth.
- Typical command:
```bash
./bench.sh --circuit trotter --simulate-limit 12 --qubits 1,2,5,7,10,11,12
```

### 4) `steady_state`
- Purpose: heavier iterative dynamics workload; often the slowest leg.
- Typical command:
```bash
./bench.sh --circuit steady_state --simulate-limit 12 --qubits 1,2,5,7,10,11,12
```

### 5) `heisenberg`
- Purpose: baseline Heisenberg model scaling.
- Typical command:
```bash
./bench.sh --circuit heisenberg --simulate-limit 12 --qubits 1,2,5,7,10,11,12
```

### 6) `heisenberg_xxz`
- Purpose: anisotropic XXZ Heisenberg variant.
- Typical command:
```bash
./bench.sh --circuit heisenberg_xxz --simulate-limit 12 --qubits 1,2,5,7,10,11,12
```

### 7) `heisenberg_random_field`
- Purpose: disordered Heisenberg workload with random field terms.
- Typical command:
```bash
./bench.sh --circuit heisenberg_random_field --simulate-limit 12 --qubits 1,2,5,7,10,11,12
```

### 8) `tfim`
- Purpose: transverse-field Ising model baseline.
- Typical command:
```bash
./bench.sh --circuit tfim --simulate-limit 12 --qubits 1,2,5,7,10,11,12
```

### 9) `tfim_trotter2`
- Purpose: second-order Trotter TFIM variant.
- Typical command:
```bash
./bench.sh --circuit tfim_trotter2 --simulate-limit 12 --qubits 1,2,5,7,10,11,12
```

### 10) `tfim_random_field`
- Purpose: TFIM with random field perturbations.
- Typical command:
```bash
./bench.sh --circuit tfim_random_field --simulate-limit 12 --qubits 1,2,5,7,10,11,12
```

### 11) `long_range_ising`
- Purpose: long-range interaction Ising scaling.
- Typical command:
```bash
./bench.sh --circuit long_range_ising --simulate-limit 12 --qubits 1,2,5,7,10,11,12
```

### 12) `ladder_heisenberg`
- Purpose: ladder-geometry Heisenberg workload.
- Typical command:
```bash
./bench.sh --circuit ladder_heisenberg --simulate-limit 12 --qubits 1,2,5,7,10,11,12
```

### 13) `random_circuit`
- Purpose: generic random gate-circuit scaling.
- Typical command:
```bash
./bench.sh --circuit random_circuit --simulate-limit 12 --qubits 1,2,5,7,10,11,12
```

### 14) `qcbm`
- Purpose: Quantum Circuit Born Machine benchmark family.
- Typical command:
```bash
./bench.sh --circuit qcbm --simulate-limit 12 --qubits 1,2,5,7,10,11,12
```

### 15) `phase_estimation`
- Purpose: phase estimation runtime growth.
- Typical command:
```bash
./bench.sh --circuit phase_estimation --simulate-limit 12 --qubits 1,2,5,7,10,11,12
```

### 16) `qft`
- Purpose: Quantum Fourier Transform scaling.
- Typical command:
```bash
./bench.sh --circuit qft --simulate-limit 12 --qubits 1,2,5,7,10,11,12
```

### 17) `qaoa`
- Purpose: QAOA-style variational optimization workload.
- Typical command:
```bash
./bench.sh --circuit qaoa --simulate-limit 12 --qubits 1,2,5,7,10,11,12
```

### 18) `vqe`
- Purpose: VQE-style variational eigensolver workload.
- Typical command:
```bash
./bench.sh --circuit vqe --simulate-limit 12 --qubits 1,2,5,7,10,11,12
```

### 19) `variational_circuit`
- Purpose: generic parameterized variational circuit benchmark.
- Typical command:
```bash
./bench.sh --circuit variational_circuit --simulate-limit 12 --qubits 1,2,5,7,10,11,12
```

### 20) `grover`
- Purpose: Grover-style amplitude amplification benchmark.
- Typical command:
```bash
./bench.sh --circuit grover --simulate-limit 12 --qubits 1,2,5,7,10,11,12
```

### 21) `ghz`
- Purpose: GHZ state generation and scaling.
- Typical command:
```bash
./bench.sh --circuit ghz --simulate-limit 12 --qubits 1,2,5,7,10,11,12
```

### QASM Suite
- Purpose: OpenQASM corpus execution from `datasets/qasm/local/`.
- Typical command:
```bash
./bench.sh --qasm-suite --qasm-max-qubits 18 --qasm-timeout-ms 30000
```

## Example & Circuit Gallery

Qupertino ships a broad gallery of runnable examples: **21 parameterized benchmark families** (any qubit count) plus a **42-circuit OpenQASM corpus** under `datasets/qasm/local/`. Run any benchmark family with `./bench.sh --circuit <key>`; run any QASM circuit through the QASM suite. The full gallery is also published on the [website](https://boltzmannentropy.github.io/QupertinoWEB/#gallery).

### A. Parameterized benchmark families (`--circuit <key>`)

| Category | Circuit keys |
| --- | --- |
| Spin & Hamiltonian dynamics | `hamiltonian_simulation`, `time_evolution`, `trotter`, `steady_state`, `heisenberg`, `heisenberg_xxz`, `heisenberg_random_field`, `tfim`, `tfim_trotter2`, `tfim_random_field`, `long_range_ising`, `ladder_heisenberg` |
| Variational & QML | `qcbm`, `qaoa`, `vqe`, `variational_circuit` |
| Core algorithms | `qft`, `phase_estimation`, `grover`, `ghz`, `random_circuit` |

### B. OpenQASM corpus (`datasets/qasm/local/`, 42 circuits)

| Category | Circuit (qubits) |
| --- | --- |
| States & entanglement | `bell` (2), `bell_state` (2), `bell_n4` (4), `ghz_state_n23` (23), `cat_state_n22` (22), `wstate_n3` (3), `wstate_n27` (27), `coin_flip` (1), `qrng_n4` (4) |
| Textbook algorithms | `deutsch_n2` (2), `grover_n2` (2), `grover_2qubit` (2), `qft_n4` (4), `qft_n18` (18), `inverseqft_n4` (4), `qpe_n9` (9), `ipea_n2` (2), `swap_test_n25` (25), `toffoli_n3` (3), `teleport_minimal` (3), `teleportation_n3` (3), `bv_n14` — Bernstein–Vazirani (14), `hs4_n4` — hidden shift (4), `bwt_n21` — binary welded tree (21) |
| Arithmetic & factoring | `bigadder_n18` (18), `multiplier_n15` (15), `multiply_n13` (13), `qf21_n15` — factor 21 (15), `factor247_n15` — factor 247 (15) |
| Variational & physics | `vqe_n4` (4), `vqe_n24` (24), `vqe_ising`, `variational_n4` (4), `ising_n10` (10), `ising_n26` (26) |
| Applied, ML & QEC | `dnn_n16` — quantum DNN (16), `knn_n25` — quantum kNN (25), `sat_n11` — SAT (11), `seca_n11` (11), `qram_n20` — QRAM (20), `qec9xz_n17` — Shor [[9,1,3]] code (17), `advanced_circuit` |

Run a QASM circuit:
```bash
./bench.sh --qasm-suite --qasm-max-qubits 27 --qasm-timeout-ms 30000
```

## Full Example Catalog (250+)

Beyond the circuit gallery above, Qupertino ships **250 runnable example functions** — gate-algebra
identities, state preparation, algorithm demonstrations (Bell, GHZ, QFT, Grover, Toffoli,
teleportation, QPE), MPS tensor-network parity, OpenQASM execution, and the QuantumStudio
backend/MCP API. The complete itemized list is published on the
[Examples & Catalog page](https://boltzmannentropy.github.io/QupertinoWEB/examples.html).

| Category | Count | Source |
| --- | ---: | --- |
| Core simulator & gate algebra | 149 | `src/tests/mlxQCoreTest.py` |
| Quantum-computing examples (identities, algorithms) | 41 | `src/tests/mlxQQCExamplesTest.py` |
| Internal consistency & measurement parity | 21 | `src/tests/mlxQInternalConsistencyTest.py`, `mlxQMeasurementParityTest.py` |
| MPS tensor-network backend | 12 | `src/tests/mlxQMpsBackendTest.py`, `mlxQMpsParamSuiteTest.py` |
| QML wrapper | 5 | `src/tests/mlxQQmlWrapperTest.py` |
| QPE energy estimation | 2 | `src/tests/mlxQQpeEnergyEstimationTest.py` |
| Benchmark catalog, protocol & visualization | 4 | `src/tests/mlxQBenchCatalogTest.py`, `mlxQBenchmarkProtocolTest.py`, `mlxQVisualizationPlotsTest.py` |
| QuantumStudio backend API & MCP server | 17 | `quantumstudio/tests/test_backend_api.py`, `test_mcp_server.py` |
| **Total** | **251** | |

Run them all with `./test.sh`, or one module with `python3 -m pytest <file> -q`.

## Sample Benchmark Log (12q Smoke)

```text
[preflight] MLX initialization OK
=== Single-circuit run: variational_circuit (qubits: 1,2,5,7,10,11,12, cap: 12) ===
=== Running variational_circuit (qubits: 1,2,5,7,10,11,12, cap: 12) ===
variational_circuit Scaling Benchmark
Framework: mlx–quantum | Device: apple–silicon–mlx | Backend: mps
Testing qubit counts: 1, 2, 5, 7, 10, 11, 12
variational_circuit    |  1q | gates     8 | wall    3.86 ms
variational_circuit    |  2q | gates    20 | wall   17.51 ms
variational_circuit    |  5q | gates    56 | wall   12.41 ms
variational_circuit    |  7q | gates    80 | wall   16.65 ms
variational_circuit    | 10q | gates   116 | wall   18.76 ms
variational_circuit    | 11q | gates   128 | wall   51.75 ms
variational_circuit    | 12q | gates   140 | wall   38.91 ms
```

## Frozen Assets And Sample Bundles

`assets/benchmarks-frozen/` is the reproducibility backbone:

- `latest/` = current promoted benchmark plots
- `sample-runs/legacy_25q_logs_2025-10/` = curated historical 24/25q logs
- `sample-runs/legacy_dmg_stage_bench_snapshot/` = legacy full snapshot (images/csv/json/logs)
- `sample-runs/legacy_25q_plus_smoke12/` = combined large-qubit + modern 12q smoke artifacts

Recommended workflow:

1. Run benchmarks (`bench.sh` / `bench_with_logging.sh`).
2. Validate data in `bench/runs/<run_id>/`.
3. Promote validated outputs to `assets/benchmarks-frozen/latest/`.
4. Keep historical reference bundles immutable under `sample-runs/`.

## Testing (200+ Coverage)

The repository includes broad simulator, algorithm, and backend tests:
- `src/tests/` + `quantumstudio/tests/` currently expose **233+ test functions**.
- Raw assertion density across test code is **well above 200 checks** (500+ assert-related lines).
- Coverage includes:
  - gate algebra and unitary identities
  - state preparation and measurement parity
  - QFT/QAOA/VQE/QCBM/Grover behavior checks
  - MPS backend parity and bond-growth diagnostics
  - OpenQASM parser/execution checks
  - QuantumStudio backend API tests

Run tests:
```bash
./test.sh
```

Targeted suites:
```bash
python3 -m pytest src/tests -q
python3 -m pytest quantumstudio/tests -q
```

## QuantumStudio UI

Start/stop the desktop stack:

```bash
cd quantumstudio
./bin/appctl up
./bin/appctl status
./bin/appctl logs backend
./bin/appctl down
```

Build UI artifacts:

```bash
./scripts/build_flutter_app.sh --release
./scripts/build_dmg.sh
```

## Screenshots

### Product UI

![QuantumStudio Screen 1](quantumstudio/assets/screenshots/screen001.png)
![QuantumStudio Screen 2](quantumstudio/assets/screenshots/screen002.png)
![QuantumStudio Screen 3](quantumstudio/assets/screenshots/screen003.png)
![QuantumStudio Screen 4](quantumstudio/assets/screenshots/screen004.png)
![QuantumStudio Screen 5](quantumstudio/assets/screenshots/screen005.png)

### Benchmark Figures (Paper Workflow)

![All Benchmarks Comparison](quantumstudio/assets/paper-figures/all_benchmarks_comparison.png)
![QFT Scaling](quantumstudio/assets/paper-figures/qft_scaling.png)
![QAOA Scaling](quantumstudio/assets/paper-figures/qaoa_scaling.png)
![VQE Scaling](quantumstudio/assets/paper-figures/vqe_scaling.png)
![QCBM Scaling](quantumstudio/assets/paper-figures/qcbm_scaling.png)
![Grover Scaling](quantumstudio/assets/paper-figures/grover_scaling.png)
![Hamiltonian Simulation Scaling](quantumstudio/assets/paper-figures/hamiltonian_simulation_scaling.png)

## Licensing

Qupertino / QuantumStudio is free and open source under the **MIT License** (see [`LICENSE`](LICENSE)). The source code and the compiled macOS binaries are both covered by the same MIT terms — use, modify, and redistribute freely, including for commercial purposes. No purchase or license key is required.

## Notes

- This repo is local-first by design: benchmark execution and artifacts remain on-device.
- For web-facing product copy, see `https://boltzmannentropy.github.io/QupertinoWEB/`.
