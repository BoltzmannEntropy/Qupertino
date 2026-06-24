#!/usr/bin/env bash
set -euo pipefail

# Unified logging wrapper around bench.sh without duplicated command strings.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT_DIR}/src"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

BENCH_ROOT="${ROOT_DIR}/bench"
if [[ -z "${MLXQ_BENCH_OUT_DIR:-}" ]]; then
  RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
  export MLXQ_BENCH_OUT_DIR="${BENCH_ROOT}/runs/${RUN_ID}"
fi
mkdir -p "${MLXQ_BENCH_OUT_DIR}"
mkdir -p "${BENCH_ROOT}/runs"
if [[ -e "${BENCH_ROOT}/current" && ! -L "${BENCH_ROOT}/current" ]]; then
  echo "⚠️  ${BENCH_ROOT}/current exists and is not a symlink; leaving it unchanged."
else
  ln -sfn "${MLXQ_BENCH_OUT_DIR}" "${BENCH_ROOT}/current"
fi
echo "📁 Benchmark run directory: ${MLXQ_BENCH_OUT_DIR}"
echo "🐍 Python interpreter: ${PYTHON_BIN}"

mlx_preflight() {
  "${PYTHON_BIN}" - <<'PY'
import sys
try:
    import mlx.core as mx
    mx.array([1.0])
except Exception as e:
    print(f"[preflight] MLX initialization failed: {e}", file=sys.stderr)
    raise
print("[preflight] MLX initialization OK")
PY
}

mlx_preflight

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${MLXQ_BENCH_OUT_DIR}/BENCHMARK_RUN_${TS}.log"
LATEST="${BENCH_ROOT}/LATEST_BENCHMARK.log"

usage() {
  cat <<EOF
Usage: ./bench_with_logging.sh [options]

Options:
  --max-qubits N               Global cap (MLXQ_MAX_QUBITS)
  --cap-<key> N                Per-benchmark cap (MLXQ_CAP_<KEY>)
  --qubits CSV|A-B             Override public list (MLXQ_PUB_QUBITS)
  --vqe-qubits CSV|A-B         Override VQE list (MLXQ_VQE_QUBITS)
  --steady-qubits CSV|A-B      Override steady-state list (MLXQ_STEADY_QUBITS)
  --all-qubits N               Shorthand for --qubits 1-N
  --mps-dmax N                 Set MLXQ_MPS_DMAX (bond cap)
  --mps-eps  X                 Set MLXQ_MPS_EPS (truncation threshold)
  --mps-bmax N                 Set MLXQ_MPS_EARLY_STOP_BMAX (early-stop on bond)
  --mps-stop-on-trunc          Stop a TEBD run on first truncation (MLXQ_MPS_STOP_ON_TRUNC=1)
  --mps-pair-sweeps            Use even/odd pair-sweeps for 2q gates (MLXQ_MPS_PAIR_SWEEPS=1)
  --mps-mpo-zz                 Use diagonal MPO for ZZ terms where supported (MLXQ_MPS_USE_MPO_ZZ=1)
  --mps-mpo-xx                 Use basis-mapped MPO for XX terms (MLXQ_MPS_USE_MPO_XX=1)
  --mps-mpo-yy                 Use basis-mapped MPO for YY terms (MLXQ_MPS_USE_MPO_YY=1)
  --with-mpsd                  Run full suite again with MPSD mode (MPO ZZ, separate _mpsd outputs)
  --frozen-parity-12           Run 12q parity suite: SV + MPS(full) + time_evolution MPSD
  --paper-2504                 Use paper 2504.14027 qubit schedule for supported keys (4,8,16,24,32,64,128,256; +512/1024 if cap allows)
  --circuit NAME               Single-circuit mode (e.g., qaoa)
  --simulate-limit N           Cap qubits for single-circuit run
  --save-plots|--no-save-plots Save per-bench plots (MLXQ_SAVE_PLOTS)
  -h, --help                   Show this help
EOF
}

# Parse flags (subset kept intentionally small)
WITH_MPS=0
WITH_MPSD=0
PAPER_2504=0
FROZEN_PARITY_12=0
while [[ ${1:-} != "" ]]; do
  case "$1" in
    --max-qubits) export MLXQ_MAX_QUBITS="${2:-}"; shift 2 ;;
    --cap-*) key="${1#--cap-}"; val="${2:-}"; uc_key="$(echo "$key" | tr '[:lower:]' '[:upper:]')"; export MLXQ_CAP_${uc_key}="$val"; shift 2 ;;
    --qubits) export MLXQ_PUB_QUBITS="${2:-}"; shift 2 ;;
    --vqe-qubits) export MLXQ_VQE_QUBITS="${2:-}"; shift 2 ;;
    --steady-qubits) export MLXQ_STEADY_QUBITS="${2:-}"; shift 2 ;;
    --all-qubits) N="${2:-}"; [[ -z "$N" ]] && { echo "--all-qubits requires N" >&2; exit 1; }; export MLXQ_PUB_QUBITS="1-${N}"; shift 2 ;;
    --save-plots) export MLXQ_SAVE_PLOTS=1; shift ;;
    --no-save-plots) export MLXQ_SAVE_PLOTS=0; shift ;;
    --circuit) export MLXQ_ONE_CIRCUIT="${2:-}"; shift 2 ;;
    --simulate-limit) export MLXQ_ONE_CAP="${2:-}"; shift 2 ;;
    --mps-dmax) export MLXQ_MPS_DMAX="${2:-}"; shift 2 ;;
    --mps-eps) export MLXQ_MPS_EPS="${2:-}"; shift 2 ;;
    --mps-bmax) export MLXQ_MPS_EARLY_STOP_BMAX="${2:-}"; shift 2 ;;
    --mps-stop-on-trunc) export MLXQ_MPS_STOP_ON_TRUNC=1; shift ;;
    --mps-pair-sweeps) export MLXQ_MPS_PAIR_SWEEPS=1; shift ;;
    --mps-mpo-zz) export MLXQ_MPS_USE_MPO_ZZ=1; shift ;;
    --mps-mpo-xx) export MLXQ_MPS_USE_MPO_XX=1; shift ;;
    --mps-mpo-yy) export MLXQ_MPS_USE_MPO_YY=1; shift ;;
    --with-mpsd) WITH_MPSD=1; shift ;;
    --with-mps) WITH_MPS=1; shift ;;
    --frozen-parity-12) FROZEN_PARITY_12=1; shift ;;
    --paper-2504) PAPER_2504=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "$FROZEN_PARITY_12" == "1" ]]; then
  MAX_Q=12
  export MLXQ_MAX_QUBITS=12
  WITH_MPS=1
  WITH_MPSD=1
fi

export MLXQ_SAVE_PLOTS="${MLXQ_SAVE_PLOTS:-1}"

echo "📊 Starting mlxQ Benchmark Suite" | tee "$LOG_FILE"
echo "📝 Logging to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Helpers
join_csv() { local IFS=","; echo "$*"; }
cap_to_max() { local max="$1"; shift; local out=(); for q in "$@"; do [[ "$q" -le "$max" ]] && out+=("$q"); done; join_csv "${out[@]}"; }
expand_range_or_csv() { local spec="$1"; [[ "$spec" =~ ^[0-9]+-[0-9]+$ ]] && { local a=${spec%%-*} b=${spec##*-}; local out=(); for ((i=a;i<=b;i++)); do out+=("$i"); done; join_csv "${out[@]}"; return; }; echo "$spec"; }
cap_csv_to_max() {
  local max="$1"; local csv="${2:-}"; local out=()
  IFS=',' read -r -a vals <<< "$csv"
  for q in "${vals[@]}"; do
    [[ -n "$q" && "$q" =~ ^[0-9]+$ && "$q" -le "$max" ]] && out+=("$q")
  done
  join_csv "${out[@]}"
}

MAX_Q="${MLXQ_MAX_QUBITS:-25}"
BASE_LIST=(1 2 5 7 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25)
VQE_LIST=(1 2 5 7 10 11 12 13 14 15)
STEADY_LIST=(1 2 5 7 10 11 12)

# Resolve CSVs
if [[ -n "${MLXQ_PUB_QUBITS:-}" ]]; then PUB_CSV=$(expand_range_or_csv "$MLXQ_PUB_QUBITS"); else PUB_CSV=$(cap_to_max "$MAX_Q" "${BASE_LIST[@]}"); fi
if [[ -n "${MLXQ_VQE_QUBITS:-}" ]]; then VQE_CSV=$(expand_range_or_csv "$MLXQ_VQE_QUBITS"); else VQE_CSV=$(cap_to_max "$MAX_Q" "${VQE_LIST[@]}"); fi
if [[ -n "${MLXQ_STEADY_QUBITS:-}" ]]; then STEADY_CSV=$(expand_range_or_csv "$MLXQ_STEADY_QUBITS"); else STEADY_CSV=$(cap_to_max "$MAX_Q" "${STEADY_LIST[@]}"); fi

# 1..30 contiguous for 30‑cap families
PUB30_CSV=""; for i in $(seq 1 30); do [[ -z "$PUB30_CSV" ]] && PUB30_CSV="$i" || PUB30_CSV="$PUB30_CSV,$i"; done
PUB30_CSV="$(cap_csv_to_max "$MAX_Q" "$PUB30_CSV")"

# Prefilled explicit enumerations (can be overridden by MLXQ_LIST_<KEY>)
LIST25_CSV="1,2,5,7,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25"
LIST30_CSV="${LIST25_CSV},26,27,28"
LIST_HAMILTONIAN_SIMULATION="$LIST30_CSV"
LIST_TIME_EVOLUTION="$LIST30_CSV"
LIST_TROTTER="$LIST30_CSV"
LIST_HEISENBERG="$LIST30_CSV"
LIST_HEISENBERG_XXZ="$LIST30_CSV"
LIST_HEISENBERG_RANDOM_FIELD="$LIST30_CSV"
LIST_TFIM="$LIST30_CSV"
LIST_TFIM_TROTTER2="$LIST30_CSV"
LIST_TFIM_RANDOM_FIELD="$LIST30_CSV"
LIST_LONG_RANGE_ISING="$LIST30_CSV"
LIST_LADDER_HEISENBERG="$LIST30_CSV"
LIST_STEADY_STATE="1,2,5,7,10,11,12"
LIST_RANDOM_CIRCUIT="$LIST25_CSV"
LIST_QCBM="$LIST25_CSV"
LIST_PHASE_ESTIMATION="$LIST25_CSV"
LIST_QFT="$LIST25_CSV"
LIST_QAOA="$LIST25_CSV"
LIST_VQE="1,2,5,7,10,11,12,13,14,15"
LIST_VARIATIONAL_CIRCUIT="$LIST25_CSV"
LIST_GROVER="$LIST25_CSV"
LIST_GHZ="$LIST25_CSV"

# Caps for single-circuit runs (per key), overridable and tweaked under --paper-2504
CAP_HAMILTONIAN_SIMULATION=${CAP_HAMILTONIAN_SIMULATION:-30}
CAP_TIME_EVOLUTION=${CAP_TIME_EVOLUTION:-30}
CAP_TROTTER=${CAP_TROTTER:-30}
CAP_HEISENBERG=${CAP_HEISENBERG:-30}
CAP_HEISENBERG_XXZ=${CAP_HEISENBERG_XXZ:-30}
CAP_HEISENBERG_RANDOM_FIELD=${CAP_HEISENBERG_RANDOM_FIELD:-30}
CAP_TFIM=${CAP_TFIM:-30}
CAP_TFIM_TROTTER2=${CAP_TFIM_TROTTER2:-30}
CAP_TFIM_RANDOM_FIELD=${CAP_TFIM_RANDOM_FIELD:-30}
CAP_LONG_RANGE_ISING=${CAP_LONG_RANGE_ISING:-30}
CAP_LADDER_HEISENBERG=${CAP_LADDER_HEISENBERG:-30}
CAP_STEADY_STATE=${CAP_STEADY_STATE:-12}
CAP_RANDOM_CIRCUIT=${CAP_RANDOM_CIRCUIT:-25}
CAP_QCBM=${CAP_QCBM:-25}
CAP_PHASE_ESTIMATION=${CAP_PHASE_ESTIMATION:-15}
CAP_QFT=${CAP_QFT:-25}
CAP_QAOA=${CAP_QAOA:-25}
CAP_VQE=${CAP_VQE:-15}
CAP_VARIATIONAL_CIRCUIT=${CAP_VARIATIONAL_CIRCUIT:-25}
CAP_GROVER=${CAP_GROVER:-25}
CAP_GHZ=${CAP_GHZ:-25}

# Optional paper-2504 preset (quasi-log grid). Applies to supported keys.
if [[ "$PAPER_2504" == "1" ]]; then
  GRID_2504="4,8,16,24,32,64,128,256,512,1024"
  export MLXQ_LIST_QFT="$GRID_2504"
  export MLXQ_LIST_QFT_ENTANGLED="$GRID_2504"
  export MLXQ_LIST_QFTENTANGLED="$GRID_2504"
  export MLXQ_LIST_GHZ="$GRID_2504"
  export MLXQ_LIST_WSTATE="$GRID_2504"
  export MLXQ_LIST_GRAPH_STATE="$GRID_2504"
  export MLXQ_LIST_GRAPHSTATE="$GRID_2504"
  export MLXQ_LIST_PHASE_ESTIMATION="$GRID_2504"
  export MLXQ_LIST_PHASE_ESTIMATION_INEXACT="$GRID_2504"
  export MLXQ_LIST_QPEEXACT="$GRID_2504"
  export MLXQ_LIST_QPEINEXACT="$GRID_2504"
  export MLXQ_LIST_AE="$GRID_2504"
  export MLXQ_LIST_QUANTUM_WALK="$GRID_2504"
  export MLXQ_LIST_QUANTUM_WALK_VCHAIN="$GRID_2504"
  export MLXQ_LIST_QWALK="$GRID_2504"
  export MLXQ_LIST_RANDOM_CIRCUIT="$GRID_2504"
  export MLXQ_LIST_RANDOM="$GRID_2504"
  export MLXQ_LIST_REALAMP="$GRID_2504"
  export MLXQ_LIST_SU2RAND="$GRID_2504"
  export MLXQ_LIST_QNN="$GRID_2504"
  # Bump caps for paper schedule to allow up to 256 by default
  CAP_HAMILTONIAN_SIMULATION=256
  CAP_TIME_EVOLUTION=256
  CAP_TROTTER=256
  CAP_HEISENBERG=256
  CAP_HEISENBERG_XXZ=256
  CAP_HEISENBERG_RANDOM_FIELD=256
  CAP_TFIM=256
  CAP_TFIM_TROTTER2=256
  CAP_TFIM_RANDOM_FIELD=256
  CAP_LONG_RANGE_ISING=256
  CAP_LADDER_HEISENBERG=256
  CAP_STEADY_STATE=256
  CAP_RANDOM_CIRCUIT=256
  CAP_QCBM=256
  CAP_PHASE_ESTIMATION=256
  CAP_QFT=256
  CAP_QAOA=256
  CAP_VQE=256
  CAP_VARIATIONAL_CIRCUIT=256
  CAP_GROVER=256
  CAP_GHZ=256
fi

# DRY run helper
run_one() {
  local key="$1"; local cap="$2"; local csv="$3"
  if [[ -n "${cap:-}" && "$cap" -gt "$MAX_Q" ]]; then cap="$MAX_Q"; fi
  csv="$(cap_csv_to_max "$MAX_Q" "$csv")"
  echo -e "\n=== Running: ./bench.sh --circuit ${key} --simulate-limit ${cap} --qubits ${csv}" | tee -a "$LOG_FILE"
  MLXQ_BENCH_OUT_DIR="${MLXQ_BENCH_OUT_DIR}" "${ROOT_DIR}/bench.sh" --circuit "$key" --simulate-limit "$cap" --qubits "$csv" 2>&1 | tee -a "$LOG_FILE"
}

# Single-circuit
if [[ -n "${MLXQ_ONE_CIRCUIT:-}" ]]; then
  key="$MLXQ_ONE_CIRCUIT"; cap="${MLXQ_ONE_CAP:-}"; csv="$PUB_CSV"
  case "$key" in
    vqe) csv="$VQE_CSV"; cap="${cap:-15}" ;;
    steady_state) csv="$STEADY_CSV"; cap="${cap:-12}" ;;
    hamiltonian_simulation|time_evolution|trotter|heisenberg|heisenberg_xxz|heisenberg_random_field|tfim|tfim_trotter2|tfim_random_field|long_range_ising|ladder_heisenberg)
      csv="$PUB30_CSV"; cap="${cap:-30}" ;;
    phase_estimation) cap="${cap:-15}" ;;
    *) cap="${cap:-$MAX_Q}" ;;
  esac
  if [[ -z "${MLXQ_PUB_QUBITS:-}" ]]; then
    csv="$(cap_csv_to_max "$MAX_Q" "$csv")"
  fi
  if [[ -z "${MLXQ_ONE_CAP:-}" && -n "${cap:-}" && "$cap" -gt "$MAX_Q" ]]; then
    cap="$MAX_Q"
  fi
  # Baseline run (SV by default unless MLXQ_BACKEND set)
  run_one "$key" "$cap" "$csv"
  # Optional single-circuit MPS follow-up
  if [[ "$WITH_MPS" == "1" ]]; then
    echo -e "\n=== Running (single-circuit) MPS: MLXQ_BACKEND=mps ===" | tee -a "$LOG_FILE"
    export MLXQ_BACKEND=mps
    # Clear MPSD flag if previously set in env
    unset MLXQ_MPSD || true
    run_one "$key" "$cap" "$csv"
  fi
  # Optional single-circuit MPSD follow-up
  if [[ "$WITH_MPSD" == "1" ]]; then
    echo -e "\n=== Running (single-circuit) MPSD: MLXQ_BACKEND=mps MLXQ_MPSD=1 MLXQ_MPS_USE_MPO_ZZ=1 ===" | tee -a "$LOG_FILE"
    export MLXQ_BACKEND=mps
    export MLXQ_MPSD=1
    export MLXQ_MPS_USE_MPO_ZZ=1
    run_one "$key" "$cap" "$csv"
  fi
  # Aggregate and copy after all single-circuit runs
  if [[ "${MLXQ_SAVE_PLOTS}" == "1" ]]; then
    MLXQ_BENCH_OUT_DIR="${MLXQ_BENCH_OUT_DIR}" "${PYTHON_BIN}" "${ROOT_DIR}/src/benchmark/aggregate_plots.py" 2>&1 | tee -a "$LOG_FILE" || true
  fi
  cp "$LOG_FILE" "$LATEST" || true
  echo "✅ Done" | tee -a "$LOG_FILE"
  exit 0
fi

# Per‑bench explicit qubit lists via MLXQ_LIST_<KEY> or sensible defaults
list_for() {
  local key="$1"; local key_uc="$(echo "$key" | tr '[:lower:]' '[:upper:]')"
  # Env override
  local override; override=$(eval echo \${MLXQ_LIST_${key_uc}:-})
  if [[ -n "$override" ]]; then cap_csv_to_max "$MAX_Q" "$override"; return; fi
  # Script preset
  local preset; preset=$(eval echo \${LIST_${key_uc}:-})
  if [[ -n "$preset" ]]; then cap_csv_to_max "$MAX_Q" "$preset"; return; fi
  # Defaults
  case "$key" in
    vqe) echo "$VQE_CSV" ;;
    steady_state) echo "$STEADY_CSV" ;;
    hamiltonian_simulation|time_evolution|trotter|heisenberg|heisenberg_xxz|heisenberg_random_field|tfim|tfim_trotter2|tfim_random_field|long_range_ising|ladder_heisenberg)
      echo "$PUB30_CSV" ;;
    *) echo "$PUB_CSV" ;;
  esac
}

# Full suite
run_one hamiltonian_simulation "$CAP_HAMILTONIAN_SIMULATION" "$(list_for hamiltonian_simulation)"
run_one time_evolution        "$CAP_TIME_EVOLUTION"        "$(list_for time_evolution)"
run_one trotter               "$CAP_TROTTER"               "$(list_for trotter)"
run_one heisenberg            "$CAP_HEISENBERG"            "$(list_for heisenberg)"
run_one heisenberg_xxz        "$CAP_HEISENBERG_XXZ"        "$(list_for heisenberg_xxz)"
run_one heisenberg_random_field "$CAP_HEISENBERG_RANDOM_FIELD" "$(list_for heisenberg_random_field)"
run_one tfim                  "$CAP_TFIM"                  "$(list_for tfim)"
run_one tfim_trotter2         "$CAP_TFIM_TROTTER2"         "$(list_for tfim_trotter2)"
run_one tfim_random_field     "$CAP_TFIM_RANDOM_FIELD"     "$(list_for tfim_random_field)"
run_one long_range_ising      "$CAP_LONG_RANGE_ISING"      "$(list_for long_range_ising)"
run_one ladder_heisenberg     "$CAP_LADDER_HEISENBERG"     "$(list_for ladder_heisenberg)"

run_one steady_state          "$CAP_STEADY_STATE"          "$(list_for steady_state)"

run_one random_circuit        "$CAP_RANDOM_CIRCUIT"        "$(list_for random_circuit)"
run_one qcbm                  "$CAP_QCBM"                  "$(list_for qcbm)"
run_one phase_estimation      "$CAP_PHASE_ESTIMATION"      "$(list_for phase_estimation)"
run_one qft                   "$CAP_QFT"                   "$(list_for qft)"
run_one qaoa                  "$CAP_QAOA"                  "$(list_for qaoa)"
run_one vqe                   "$CAP_VQE"                   "$(list_for vqe)"
run_one variational_circuit   "$CAP_VARIATIONAL_CIRCUIT"   "$(list_for variational_circuit)"
run_one grover                "$CAP_GROVER"                "$(list_for grover)"
run_one ghz                   "$CAP_GHZ"                   "$(list_for ghz)"

# Aggregate plots and copy
[[ "${MLXQ_SAVE_PLOTS}" == "1" ]] && MLXQ_BENCH_OUT_DIR="${MLXQ_BENCH_OUT_DIR}" "${PYTHON_BIN}" "${ROOT_DIR}/src/benchmark/aggregate_plots.py" 2>&1 | tee -a "$LOG_FILE" || true
# Aggregate MPS summaries (if present)
"${PYTHON_BIN}" "${ROOT_DIR}/tools/mps_report.py" --bench "${MLXQ_BENCH_OUT_DIR}" 2>&1 | tee -a "$LOG_FILE" || true

"${PYTHON_BIN}" - <<'PY' 2>/dev/null || true
from pathlib import Path
import shutil
import os
bench = Path(os.environ.get('MLXQ_BENCH_OUT_DIR', 'bench'))
for d in (
    Path('paper')/'prx-quantum'/'images',
    Path('assets')/'benchmarks-frozen'/'latest',
):
    d.mkdir(parents=True, exist_ok=True)
    for p in bench.glob('*_scaling.png'):
        try: shutil.copyfile(p, d/p.name)
        except Exception: pass
    # also bonds plots if any
    for p in bench.glob('*_bonds.png'):
        try: shutil.copyfile(p, d/p.name)
        except Exception: pass
    cmp = bench / 'all_benchmarks_comparison.png'
    if cmp.exists():
        try: shutil.copyfile(cmp, d/cmp.name)
        except Exception: pass
    cmp2 = bench / 'all_mps_bonds_comparison.png'
    if cmp2.exists():
        try: shutil.copyfile(cmp2, d/cmp2.name)
        except Exception: pass
print('✅ Plots copied')
PY

cp "$LOG_FILE" "$LATEST" || true
  echo "✅ Logs updated: $LOG_FILE and $LATEST"
echo "✅ Done" | tee -a "$LOG_FILE"

# Optionally run the full suite with MPS backend as separate entries
if [[ "$WITH_MPS" == "1" ]]; then
  echo "" | tee -a "$LOG_FILE"
  echo "=== Running MPS backend suite (MLXQ_BACKEND=mps) ===" | tee -a "$LOG_FILE"
  export MLXQ_BACKEND=mps
  # re-run the same suite with identical lists/caps
  run_one hamiltonian_simulation "$CAP_HAMILTONIAN_SIMULATION" "$(list_for hamiltonian_simulation)"
  run_one time_evolution        "$CAP_TIME_EVOLUTION"        "$(list_for time_evolution)"
  run_one trotter               "$CAP_TROTTER"               "$(list_for trotter)"
  run_one heisenberg            "$CAP_HEISENBERG"            "$(list_for heisenberg)"
  run_one heisenberg_xxz        "$CAP_HEISENBERG_XXZ"        "$(list_for heisenberg_xxz)"
  run_one heisenberg_random_field "$CAP_HEISENBERG_RANDOM_FIELD" "$(list_for heisenberg_random_field)"
  run_one tfim                  "$CAP_TFIM"                  "$(list_for tfim)"
  run_one tfim_trotter2         "$CAP_TFIM_TROTTER2"         "$(list_for tfim_trotter2)"
  run_one tfim_random_field     "$CAP_TFIM_RANDOM_FIELD"     "$(list_for tfim_random_field)"
  run_one long_range_ising      "$CAP_LONG_RANGE_ISING"      "$(list_for long_range_ising)"
  run_one ladder_heisenberg     "$CAP_LADDER_HEISENBERG"     "$(list_for ladder_heisenberg)"
  run_one steady_state          "$CAP_STEADY_STATE"          "$(list_for steady_state)"
  run_one random_circuit        "$CAP_RANDOM_CIRCUIT"        "$(list_for random_circuit)"
  run_one qcbm                  "$CAP_QCBM"                  "$(list_for qcbm)"
  run_one phase_estimation      "$CAP_PHASE_ESTIMATION"      "$(list_for phase_estimation)"
  run_one qft                   "$CAP_QFT"                   "$(list_for qft)"
  run_one qaoa                  "$CAP_QAOA"                  "$(list_for qaoa)"
  run_one vqe                   "$CAP_VQE"                   "$(list_for vqe)"
  run_one variational_circuit   "$CAP_VARIATIONAL_CIRCUIT"   "$(list_for variational_circuit)"
  run_one grover                "$CAP_GROVER"                "$(list_for grover)"
  run_one ghz                   "$CAP_GHZ"                   "$(list_for ghz)"
  if [[ "${MLXQ_SAVE_PLOTS}" == "1" ]]; then
    MLXQ_BENCH_OUT_DIR="${MLXQ_BENCH_OUT_DIR}" "${PYTHON_BIN}" "${ROOT_DIR}/src/benchmark/aggregate_plots.py" 2>&1 | tee -a "$LOG_FILE" || true
  fi
  "${PYTHON_BIN}" "${ROOT_DIR}/tools/mps_report.py" --bench "${MLXQ_BENCH_OUT_DIR}" 2>&1 | tee -a "$LOG_FILE" || true
  "${PYTHON_BIN}" - <<'PY' 2>/dev/null || true
from pathlib import Path
import shutil
import os
bench = Path(os.environ.get('MLXQ_BENCH_OUT_DIR', 'bench'))
for d in (
    Path('paper')/'prx-quantum'/'images',
    Path('assets')/'benchmarks-frozen'/'latest',
):
    d.mkdir(parents=True, exist_ok=True)
    for p in bench.glob('*_mps_scaling.png'):
        try: shutil.copyfile(p, d/p.name)
        except Exception: pass
    for p in bench.glob('*_mps_bonds.png'):
        try: shutil.copyfile(p, d/p.name)
        except Exception: pass
    cmp2 = bench / 'all_mps_bonds_comparison.png'
    if cmp2.exists():
        try: shutil.copyfile(cmp2, d/cmp2.name)
        except Exception: pass
print('✅ MPS plots copied')
PY
  echo "✅ MPS suite done" | tee -a "$LOG_FILE"
fi

# Optionally run the full suite with MPSD mode (MPO ZZ, distinct outputs)
if [[ "$WITH_MPSD" == "1" ]]; then
  echo "" | tee -a "$LOG_FILE"
  echo "=== Running MPSD (MPS + MPO ZZ) backend suite ===" | tee -a "$LOG_FILE"
  export MLXQ_BACKEND=mps
  export MLXQ_MPSD=1
  export MLXQ_MPS_USE_MPO_ZZ=1
  # Frozen parity mode uses only time_evolution in MPSD to match baseline shape.
  if [[ "$FROZEN_PARITY_12" == "1" ]]; then
    run_one time_evolution "$CAP_TIME_EVOLUTION" "$(list_for time_evolution)"
  else
    # re-run the same suite with identical lists/caps
    run_one hamiltonian_simulation "$CAP_HAMILTONIAN_SIMULATION" "$(list_for hamiltonian_simulation)"
    run_one time_evolution        "$CAP_TIME_EVOLUTION"        "$(list_for time_evolution)"
    run_one trotter               "$CAP_TROTTER"               "$(list_for trotter)"
    run_one heisenberg            "$CAP_HEISENBERG"            "$(list_for heisenberg)"
    run_one heisenberg_xxz        "$CAP_HEISENBERG_XXZ"        "$(list_for heisenberg_xxz)"
    run_one heisenberg_random_field "$CAP_HEISENBERG_RANDOM_FIELD" "$(list_for heisenberg_random_field)"
    run_one tfim                  "$CAP_TFIM"                  "$(list_for tfim)"
    run_one tfim_trotter2         "$CAP_TFIM_TROTTER2"         "$(list_for tfim_trotter2)"
    run_one tfim_random_field     "$CAP_TFIM_RANDOM_FIELD"     "$(list_for tfim_random_field)"
    run_one long_range_ising      "$CAP_LONG_RANGE_ISING"      "$(list_for long_range_ising)"
    run_one ladder_heisenberg     "$CAP_LADDER_HEISENBERG"     "$(list_for ladder_heisenberg)"
    run_one steady_state          "$CAP_STEADY_STATE"          "$(list_for steady_state)"
    run_one random_circuit        "$CAP_RANDOM_CIRCUIT"        "$(list_for random_circuit)"
    run_one qcbm                  "$CAP_QCBM"                  "$(list_for qcbm)"
    run_one phase_estimation      "$CAP_PHASE_ESTIMATION"      "$(list_for phase_estimation)"
    run_one qft                   "$CAP_QFT"                   "$(list_for qft)"
    run_one qaoa                  "$CAP_QAOA"                  "$(list_for qaoa)"
    run_one vqe                   "$CAP_VQE"                   "$(list_for vqe)"
    run_one variational_circuit   "$CAP_VARIATIONAL_CIRCUIT"   "$(list_for variational_circuit)"
    run_one grover                "$CAP_GROVER"                "$(list_for grover)"
    run_one ghz                   "$CAP_GHZ"                   "$(list_for ghz)"
  fi
  if [[ "${MLXQ_SAVE_PLOTS}" == "1" ]]; then
    MLXQ_BENCH_OUT_DIR="${MLXQ_BENCH_OUT_DIR}" "${PYTHON_BIN}" "${ROOT_DIR}/src/benchmark/aggregate_plots.py" 2>&1 | tee -a "$LOG_FILE" || true
  fi
  "${PYTHON_BIN}" - <<'PY' 2>/dev/null || true
from pathlib import Path
import shutil
import os
bench = Path(os.environ.get('MLXQ_BENCH_OUT_DIR', 'bench'))
for d in (
    Path('paper')/'prx-quantum'/'images',
    Path('assets')/'benchmarks-frozen'/'latest',
):
    d.mkdir(parents=True, exist_ok=True)
    for p in bench.glob('*_mpsd_scaling.png'):
        try: shutil.copyfile(p, d/p.name)
        except Exception: pass
    for p in bench.glob('*_mpsd_bonds.png'):
        try: shutil.copyfile(p, d/p.name)
        except Exception: pass
print('✅ MPSD plots copied')
PY
  echo "✅ MPSD suite done" | tee -a "$LOG_FILE"
fi
