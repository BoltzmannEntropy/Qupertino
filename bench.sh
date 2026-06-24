#!/usr/bin/env bash
set -euo pipefail

# Python mlxQ benchmark launcher (explicit settings + full suite)
# Mirrors the spirit of legacy/bench.sh but uses Python runners.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT_DIR}/src"
BENCH_ROOT="${ROOT_DIR}/bench"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

# Per-run output directory (can be overridden externally)
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

usage() {
  cat <<EOF
Usage: ./bench.sh [options]

Options:
  --max-qubits N               Global cap for all scaling benches (MLXQ_MAX_QUBITS)
  --cap-<key> N                Per-benchmark cap (MLXQ_CAP_<KEY>), e.g. --cap-qft 12
  --vendor-suite               Run grouped vendor suites (MLXQ_VENDOR_SUITE=1)
  --algo-groups                Run grouped algorithm suites (MLXQ_ALGO_GROUPS=1)
  --memray                     If available, record memray profiles (MLXQ_MEMRAY=1)
  --qasm-suite                 Run OpenQASM suite (disabled by default)
  --benchpress                 Generate Benchpress-like figures only (no extra runs)
  --mqtbench                   Run MQTBench vendor group (subset we support)
  --qubits CSV|A-B             Override qubit list for all benches (e.g. 1,2,3 or 1-25)
  --vqe-qubits CSV|A-B         Override qubit list for VQE only
  --steady-qubits CSV|A-B      Override qubit list for steady_state only
  --all-qubits N               Shorthand for --qubits 1-N (e.g. --all-qubits 25)
  --circuit NAME               Run a single circuit by name (e.g., qaoa)
  --simulate-limit N           Cap qubits for single-circuit run
  --save-plots                 Save per-benchmark plots (default) and aggregate
  --no-save-plots              Do not save plots (MLXQ_SAVE_PLOTS=0)
  --backend sv|mps             Choose simulation backend (env MLXQ_BACKEND)
  --mps-dmax N                 Set MLXQ_MPS_DMAX (bond cap)
  --mps-eps X                  Set MLXQ_MPS_EPS (truncation epsilon)
  --mps-bmax N                 Set MLXQ_MPS_EARLY_STOP_BMAX (early-stop on bond)
  --mps-stop-on-trunc          Stop TEBD on first truncation (MLXQ_MPS_STOP_ON_TRUNC=1)
  --mps-pair-sweeps            Use even/odd pair-sweeps for 2q gates (MLXQ_MPS_PAIR_SWEEPS=1)
  --mps-mpo-zz                 Use diagonal MPO for ZZ terms where supported (MLXQ_MPS_USE_MPO_ZZ=1)
  --mps-mpo-xx                 Use basis-mapped MPO for XX terms (MLXQ_MPS_USE_MPO_XX=1)
  --mps-mpo-yy                 Use basis-mapped MPO for YY terms (MLXQ_MPS_USE_MPO_YY=1)
  --paper-2504                 Use paper 2504.14027 qubit schedule for supported keys (4,8,16,24,32,64,128,256; +512/1024 if cap allows)
  --qasm-max-qubits N          Cap for QASM suite only (QASM_MAX_QUBITS)
  --qasm-timeout-ms N          Timeout for QASM suite (QASM_TIMEOUT_MS)
  --qasm-include-large         Include optional large-scale QASM group (QASM_INCLUDE_LARGE=1)
  -h, --help                   Show this help

Benchmark keys for per-cap: hamiltonian_simulation, time_evolution, trotter, steady_state,
random_circuit, qcbm, phase_estimation, qft, qaoa, vqe, variational_circuit, grover, ghz
EOF
}

# Defaults
VENDOR_SUITE=0
ALGO_GROUPS=0
SAVE_PLOTS=1
export MLXQ_SAVE_PLOTS=${MLXQ_SAVE_PLOTS:-1}
OVERRIDE_PUB_CSV=""
OVERRIDE_VQE_CSV=""
OVERRIDE_STEADY_CSV=""
ONE_CIRCUIT=""
ONE_CAP=""
QASM_SUITE=0
BENCHPRESS_ONLY=0
MQTBENCH=0
EXECUTED=""
PAPER_2504=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-qubits)
      export MLXQ_MAX_QUBITS="${2:-}"
      shift 2
      ;;
    --cap-*)
      key="${1#--cap-}"
      val="${2:-}"
      uc_key="$(echo "$key" | tr '[:lower:]' '[:upper:]')"
      export MLXQ_CAP_${uc_key}="$val"
      shift 2
      ;;
    --vendor-suite)
      echo "⚠️  --vendor-suite is temporarily disabled; ignoring." >&2
      VENDOR_SUITE=0
      shift
      ;;
    --algo-groups)
      ALGO_GROUPS=1
      shift
      ;;
    --memray)
      export MLXQ_MEMRAY=1
      shift
      ;;
    --qasm-suite)
      QASM_SUITE=1
      shift
      ;;
    --benchpress)
      BENCHPRESS_ONLY=1
      shift
      ;;
    --mqtbench)
      MQTBENCH=1
      shift
      ;;
    --save-plots)
      SAVE_PLOTS=1
      export MLXQ_SAVE_PLOTS=1
      shift
      ;;
    --no-save-plots)
      SAVE_PLOTS=0
      export MLXQ_SAVE_PLOTS=0
      shift
      ;;
    --qasm-max-qubits)
      export QASM_MAX_QUBITS="${2:-}"
      shift 2
      ;;
    --qasm-timeout-ms)
      export QASM_TIMEOUT_MS="${2:-}"
      shift 2
      ;;
    --qasm-include-large)
      export QASM_INCLUDE_LARGE=1
      shift
      ;;
    --backend)
      export MLXQ_BACKEND="${2:-}"; shift 2 ;;
    --qubits)
      OVERRIDE_PUB_CSV="${2:-}"
      shift 2
      ;;
    --vqe-qubits)
      OVERRIDE_VQE_CSV="${2:-}"
      shift 2
      ;;
    --steady-qubits)
      OVERRIDE_STEADY_CSV="${2:-}"
      shift 2
      ;;
    --all-qubits)
      N="${2:-}"
      if [[ -z "$N" ]]; then echo "--all-qubits requires N" >&2; exit 1; fi
      OVERRIDE_PUB_CSV="1-${N}"
      shift 2
      ;;
    --circuit)
      ONE_CIRCUIT="${2:-}"; shift 2 ;;
    --simulate-limit)
      ONE_CAP="${2:-}"; shift 2 ;;
    --mps-dmax)
      export MLXQ_MPS_DMAX="${2:-}"; shift 2 ;;
    --mps-eps)
      export MLXQ_MPS_EPS="${2:-}"; shift 2 ;;
    --mps-bmax)
      export MLXQ_MPS_EARLY_STOP_BMAX="${2:-}"; shift 2 ;;
    --mps-stop-on-trunc)
      export MLXQ_MPS_STOP_ON_TRUNC=1; shift ;;
    --mps-pair-sweeps)
      export MLXQ_MPS_PAIR_SWEEPS=1; shift ;;
    --mps-mpo-zz)
      export MLXQ_MPS_USE_MPO_ZZ=1; shift ;;
    --mps-mpo-xx)
      export MLXQ_MPS_USE_MPO_XX=1; shift ;;
    --mps-mpo-yy)
      export MLXQ_MPS_USE_MPO_YY=1; shift ;;
    --paper-2504)
      PAPER_2504=1; shift ;;
    -h|--help)
      usage; exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2; usage; exit 1
      ;;
  esac
done

# Vendor suite temporarily disabled (flag is accepted but ignored)
export MLXQ_VENDOR_SUITE=0
if [[ "$ALGO_GROUPS" == "1" ]]; then
  export MLXQ_ALGO_GROUPS=1
fi

# Build canonical qubit lists (like legacy/bench.sh)
# Global cap from MLXQ_MAX_QUBITS (default 25)
MAX_Q="${MLXQ_MAX_QUBITS:-25}"
BASE_LIST=(1 2 5 7 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25)
VQE_LIST=(1 2 5 7 10 11 12 13 14 15)
STEADY_LIST=(1 2 5 7 10 11 12)

join_csv() {
  local arr=("$@")
  local IFS=","; echo "${arr[*]}"
}

cap_to_max() {
  local max="$1"; shift
  local out=()
  for q in "$@"; do
    if [[ "$q" -le "$max" ]]; then
      out+=("$q")
    fi
  done
  join_csv "${out[@]}"
}

expand_range_or_csv() {
  local spec="$1"
  if [[ "$spec" =~ ^[0-9]+-[0-9]+$ ]]; then
    local a=${spec%%-*}
    local b=${spec##*-}
    local out=()
    local i
    for ((i=a; i<=b; i++)); do out+=("$i"); done
    join_csv "${out[@]}"
  else
    echo "$spec"
  fi
}

cap_csv_to_max() {
  local max="$1"; local csv="${2:-}"; local out=()
  IFS=',' read -r -a vals <<< "$csv"
  for q in "${vals[@]}"; do
    [[ -n "$q" && "$q" =~ ^[0-9]+$ && "$q" -le "$max" ]] && out+=("$q")
  done
  join_csv "${out[@]}"
}

# Optional: per‑benchmark preset lists (CSV). Edit here to enforce explicit enumerations.
# If non‑empty, these take precedence over BASE/VQE/STEADY/PUB30 defaults.
# They are still overridden by environment MLXQ_LIST_<KEY> if provided.
LIST25_CSV="1,2,5,7,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25"
LIST30_CSV="${LIST25_CSV},26,27,28"

# Prefilled explicit enumerations (can be overridden by MLXQ_LIST_<KEY>)
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

# Build contiguous 1..30 list for 30-capped families (explicit enumeration)
PUB30_CSV=""
for i in $(seq 1 30); do
  if [[ -z "$PUB30_CSV" ]]; then PUB30_CSV="$i"; else PUB30_CSV="$PUB30_CSV,$i"; fi
done
PUB30_CSV="$(cap_csv_to_max "$MAX_Q" "$PUB30_CSV")"

# Optional paper-2504 preset (quasi-log grid). Applies to supported keys.
if [[ "$PAPER_2504" == "1" ]]; then
  GRID_2504="4,8,16,24,32,64,128,256,512,1024"
  # Map paper keys to our internal keys and lists
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
fi

# Resolve an explicit per-benchmark qubit CSV, with env override support.
# Usage: list_for qft|vqe|steady_state|hamiltonian_simulation|...
list_for() {
  local key="$1"
  local key_uc="$(echo "$key" | tr '[:lower:]' '[:upper:]')"
  # Allow MLXQ_LIST_<KEY>="1,2,3,..." override
  local override
  override=$(eval echo \${MLXQ_LIST_${key_uc}:-})
  if [[ -n "$override" ]]; then
    cap_csv_to_max "$MAX_Q" "$override"; return 0
  fi
  # Script preset (LIST_<KEY>)
  local preset
  preset=$(eval echo \${LIST_${key_uc}:-})
  if [[ -n "$preset" ]]; then
    cap_csv_to_max "$MAX_Q" "$preset"; return 0
  fi
  case "$key" in
    vqe) echo "$VQE_CSV" ;;
    steady_state) echo "$STEADY_CSV" ;;
    hamiltonian_simulation|time_evolution|trotter|heisenberg|heisenberg_xxz|heisenberg_random_field|tfim|tfim_trotter2|tfim_random_field|long_range_ising|ladder_heisenberg)
      echo "$PUB30_CSV" ;;
    *) echo "$PUB_CSV" ;;
  esac
}

# Helper to run one scaling benchmark via Python
run_bench() {
  local key="$1"; shift
  local csv_qubits="$1"; shift
  local cap="$1"; shift || true
  csv_qubits="$(cap_csv_to_max "$MAX_Q" "$csv_qubits")"
  if [[ -n "${cap:-}" && "$cap" -gt "$MAX_Q" ]]; then cap="$MAX_Q"; fi
  # de-duplicate: skip if already executed in this session
  case ",$EXECUTED," in
    *,"$key",*)
      echo "=== Skip duplicate: $key ==="
      return 0 ;;
    *) ;;
  esac
  EXECUTED+=",$key"
  echo "=== Running $key (qubits: $csv_qubits, cap: ${cap:-none}) ==="
  "${PYTHON_BIN}" - "$key" "$csv_qubits" "${cap:-}" "$MLXQ_BENCH_OUT_DIR" <<'PY'
import sys
from mlxq.mlxQbench import run_scaling_benchmark
key = sys.argv[1]
qs = [int(x) for x in sys.argv[2].split(',') if x]
cap = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None
out_prefix = sys.argv[4] if len(sys.argv) > 4 else "bench"
run_scaling_benchmark(key, qs, simulate_cap=cap, out_prefix=out_prefix)
PY
}

# Resolve per-benchmark caps from env (fallback to sensible defaults)
cap_for() {
  # $1 is bench key (e.g., qft). Look up MLXQ_CAP_<KEY> else default
  local key_uc
  key_uc=$(echo "$1" | tr '[:lower:]' '[:upper:]')
  local val
  val=$(eval echo \${MLXQ_CAP_${key_uc}:-})
  if [[ -n "$val" ]]; then
    echo "$val"
    return
  fi
  case "$1" in
    steady_state) echo 12 ;;
    hamiltonian_simulation|time_evolution|trotter|heisenberg) echo 30 ;;
    phase_estimation|vqe) echo 15 ;;
    *) echo "$MAX_Q" ;;
  esac
}

# Resolve qubit CSVs (override-aware)
if [[ -n "$OVERRIDE_PUB_CSV" ]]; then
  PUB_CSV=$(expand_range_or_csv "$OVERRIDE_PUB_CSV")
else
  PUB_CSV=$(cap_to_max "$MAX_Q" "${BASE_LIST[@]}")
fi
if [[ -n "$OVERRIDE_VQE_CSV" ]]; then
  VQE_CSV=$(expand_range_or_csv "$OVERRIDE_VQE_CSV")
else
  VQE_CSV=$(cap_to_max "$MAX_Q" "${VQE_LIST[@]}")
fi
if [[ -n "$OVERRIDE_STEADY_CSV" ]]; then
  STEADY_CSV=$(expand_range_or_csv "$OVERRIDE_STEADY_CSV")
else
  STEADY_CSV=$(cap_to_max "$MAX_Q" "${STEADY_LIST[@]}")
fi

# If user provided a global override but not VQE/steady, cascade it
if [[ -n "$OVERRIDE_PUB_CSV" && -z "$OVERRIDE_VQE_CSV" ]]; then
  VQE_CSV=$(expand_range_or_csv "$OVERRIDE_PUB_CSV")
fi
if [[ -n "$OVERRIDE_PUB_CSV" && -z "$OVERRIDE_STEADY_CSV" ]]; then
  STEADY_CSV=$(expand_range_or_csv "$OVERRIDE_PUB_CSV")
fi

# If single-circuit mode, run just that and exit
if [[ -n "$ONE_CIRCUIT" ]]; then
  CSV_FOR_ONE=""
  if [[ -n "$OVERRIDE_PUB_CSV" ]]; then
    CSV_FOR_ONE=$(expand_range_or_csv "$OVERRIDE_PUB_CSV")
  else
    case "$ONE_CIRCUIT" in
      vqe) CSV_FOR_ONE="$VQE_CSV" ;;
      steady_state) CSV_FOR_ONE="$STEADY_CSV" ;;
      *) CSV_FOR_ONE="$PUB_CSV" ;;
    esac
  fi
  if [[ -z "$OVERRIDE_PUB_CSV" ]]; then
    CSV_FOR_ONE="$(cap_csv_to_max "$MAX_Q" "$CSV_FOR_ONE")"
  fi
  if [[ -z "${ONE_CAP:-}" ]]; then
    local_cap="$(cap_for "$ONE_CIRCUIT")"
    if [[ -n "$local_cap" && "$local_cap" -gt "$MAX_Q" ]]; then
      ONE_CAP="$MAX_Q"
    fi
  fi
  echo "=== Single-circuit run: $ONE_CIRCUIT (qubits: $CSV_FOR_ONE, cap: ${ONE_CAP:-none}) ==="
  run_bench "$ONE_CIRCUIT" "$CSV_FOR_ONE" "${ONE_CAP:-}"
  if [[ "$SAVE_PLOTS" == "1" ]]; then
    echo "=== Aggregating plots ==="
    python3 "${ROOT_DIR}/src/benchmark/aggregate_plots.py" || true
  fi
  echo "✅ Done"
  exit 0
fi

echo "=== Running full benchmark suite (explicit) ==="

# Shallow/deep algorithm families (use per-benchmark explicit lists when set)
run_bench hamiltonian_simulation "$(list_for hamiltonian_simulation)" "$(cap_for hamiltonian_simulation)"
run_bench time_evolution        "$(list_for time_evolution)"        "$(cap_for time_evolution)"
run_bench trotter               "$(list_for trotter)"               "$(cap_for trotter)"
run_bench heisenberg            "$(list_for heisenberg)"            "$(cap_for heisenberg)"
run_bench heisenberg_xxz        "$(list_for heisenberg_xxz)"        "$(cap_for heisenberg_xxz)"
run_bench heisenberg_random_field "$(list_for heisenberg_random_field)" "$(cap_for heisenberg_random_field)"
run_bench tfim                  "$(list_for tfim)"                  "$(cap_for tfim)"
run_bench tfim_trotter2         "$(list_for tfim_trotter2)"         "$(cap_for tfim_trotter2)"
run_bench tfim_random_field     "$(list_for tfim_random_field)"     "$(cap_for tfim_random_field)"
run_bench long_range_ising      "$(list_for long_range_ising)"      "$(cap_for long_range_ising)"
run_bench ladder_heisenberg     "$(list_for ladder_heisenberg)"     "$(cap_for ladder_heisenberg)"
run_bench steady_state          "$(list_for steady_state)"          "$(cap_for steady_state)"
run_bench random_circuit        "$(list_for random_circuit)"        "$(cap_for random_circuit)"
run_bench qcbm                  "$(list_for qcbm)"                  "$(cap_for qcbm)"
run_bench phase_estimation      "$(list_for phase_estimation)"      "$(cap_for phase_estimation)"
run_bench qft                   "$(list_for qft)"                   "$(cap_for qft)"
run_bench qaoa                  "$(list_for qaoa)"                  "$(cap_for qaoa)"
run_bench vqe                   "$(list_for vqe)"                   "$(cap_for vqe)"
run_bench variational_circuit   "$(list_for variational_circuit)"   "$(cap_for variational_circuit)"
run_bench grover                "$(list_for grover)"                "$(cap_for grover)"
run_bench ghz                   "$(list_for ghz)"                   "$(cap_for ghz)"

if [[ "$QASM_SUITE" == "1" ]]; then
  echo ""
  echo "=== OpenQASM Circuit Benchmarks ==="
  export QASM_MAX_QUBITS="${QASM_MAX_QUBITS:-18}"
  export QASM_MAX_MEM_MB="${QASM_MAX_MEM_MB:-4096}"
  "${PYTHON_BIN}" - <<'PY'
from mlxq.mlxQbench import run_qasm_suite
run_qasm_suite()
PY
fi

# Measurement distributions (educational GHZ plots like legacy)
echo ""
echo "=== Generating GHZ measurement distributions (4/5/6 qubits) ==="
"${PYTHON_BIN}" - <<'PY' || true
import os
from pathlib import Path
from mlxq.mlxQdevice import Device
import matplotlib.pyplot as plt  # type: ignore
import platform, subprocess

def detect_hw():
    gen = 'Unknown'; var = 'Base'; label = 'Unknown Base'
    if platform.system() == 'Darwin':
        brand = ''
        try:
            out = subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'])
            brand = out.decode('utf-8','ignore').strip()
        except Exception:
            pass
        if 'M1' in brand: gen = 'M1'
        elif 'M2' in brand: gen = 'M2'
        elif 'M3' in brand: gen = 'M3'
        elif 'M4' in brand: gen = 'M4'
        else: gen = 'Unknown'
        if 'Max' in brand: var = 'Max'
        elif 'Pro' in brand: var = 'Pro'
        elif 'Ultra' in brand: var = 'Ultra'
        else: var = 'Base'
        label = f"{gen} {var}"
    return f"{gen}_{var}", label

outdir = Path(os.environ.get("MLXQ_BENCH_OUT_DIR", "bench"))
outdir.mkdir(parents=True, exist_ok=True)
def ghz_ops(n):
    ops = [{"name":"H","wires":[0]}]
    for i in range(1,n):
        ops.append({"name":"CNOT","wires":[i-1,i]})
    return ops

def gen(n:int, shots:int=10000):
    dev = Device(n, shots=shots)
    dev.execute(ghz_ops(n))
    counts = dev.counts(shots=shots)
    hw_prefix, hw_label = detect_hw()
    # CSV (bitstring,count,probability) with hardware suffix
    csv = outdir / f"ghz{n}_distribution_{hw_prefix}.csv"
    total = max(1, sum(counts.values()))
    with open(csv,'w') as f:
        f.write('bitstring,count,probability\n')
        for k in sorted(counts.keys()):
            v = counts[k]
            f.write(f"{k},{v},{v/total:.6f}\n")
    # PNG styled like legacy gnuplot
    keys = sorted(counts.keys())
    probs = [counts[k]/total for k in keys]
    plt.figure(figsize=(12,6))
    plt.bar(keys, probs, color='#00AA00')
    plt.axhline(0.5, color='red', linewidth=2, linestyle='--')
    plt.ylim(0, 0.6)
    plt.grid(True, axis='y', linestyle=':', alpha=0.6)
    plt.title(f'GHZ-{n} Measurement Distribution ({shots} shots on {hw_label})')
    plt.xlabel('Measurement Outcome'); plt.ylabel('Probability')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    png = outdir / f"ghz{n}_distribution_{hw_prefix}.png"
    plt.savefig(png, dpi=100); plt.close()
    # Optional gnuplot script for parity with legacy
    try:
        gnu = outdir / f"ghz{n}_distribution_{hw_prefix}.gnu"
        with open(gnu,'w') as g:
            g.write("set terminal pngcairo size 1200,600 enhanced font 'Arial,12'\n")
            g.write(f"set output '{outdir / f'ghz{n}_distribution_{hw_prefix}.png'}'\n")
            g.write(f"set title 'GHZ-{n} Measurement Distribution ({shots} shots on {hw_label})'\n")
            g.write("set xlabel 'Measurement Outcome'\n")
            g.write("set ylabel 'Probability'\n")
            g.write("set yrange [0:0.6]\n")
            g.write("set grid\n")
            g.write("set style fill solid 0.7\n")
            g.write("set boxwidth 0.8\n")
            g.write("set xtics rotate by 45 right\n")
            g.write("set datafile separator ','\n")
            g.write(f"plot '{csv}' using 0:3:xtic(1) with boxes lc rgb '#00AA00' title 'Measured', \\\n")
            g.write("     0.5 with lines lc rgb 'red' lw 2 dt 2 title 'Theoretical (0.5)'\n")
    except Exception:
        pass

for n in (4,5,6):
    gen(n, 10000)
print('✅ GHZ distributions generated')
PY

# Optional vendor/algorithm groups (if requested via flags)
# Vendor suite run disabled intentionally

if [[ "$SAVE_PLOTS" == "1" ]]; then
  echo "=== Aggregating plots ==="
  MLXQ_BENCH_OUT_DIR="${MLXQ_BENCH_OUT_DIR}" "${PYTHON_BIN}" "${ROOT_DIR}/src/benchmark/aggregate_plots.py" || true
  # MPS report aggregation (if summaries exist)
  "${PYTHON_BIN}" "${ROOT_DIR}/tools/mps_report.py" --bench "${MLXQ_BENCH_OUT_DIR}" || true
  if [[ "$BENCHPRESS_ONLY" == "1" ]]; then
    echo "=== Generating Benchpress-like figures ==="
    "${PYTHON_BIN}" "${ROOT_DIR}/src/benchmark/benchpress_replica.py" || true
  fi
  echo "=== Copying plots to paper/prx-quantum/images and assets/benchmarks-frozen/latest ==="
  "${PYTHON_BIN}" - <<'PY' || true
from pathlib import Path
import shutil
import os
bench = Path(os.environ.get('MLXQ_BENCH_OUT_DIR', 'bench'))
if not bench.is_dir():
    raise SystemExit(0)
dests = [
    Path('paper')/'prx-quantum'/'images',
    Path('assets')/'benchmarks-frozen'/'latest',
]
for d in dests:
    d.mkdir(parents=True, exist_ok=True)
patterns = [
    '*_scaling.png', '*_bonds.png',
    'all_benchmarks_comparison.png', 'all_mps_bonds_comparison.png',
    'vis_*_side_by_side.png','vis_*_mlxq.png','vis_*_pl.png','vis_*_hist_side_by_side.png',
    'vqe_convergence_*.png', 'ghz*_distribution_*.png'
]
for pat in patterns:
    for p in bench.glob(pat):
        for d in dests:
            try:
                shutil.copyfile(p, d/p.name)
            except Exception:
                pass
print('✅ Plots copied')
PY
fi

echo "✅ Done"
