#!/usr/bin/env bash
set -euo pipefail

# mlxQ test launcher (similar style to bench.sh)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT_DIR}/src"
# Use non-interactive backend for plotting tests
export MPLBACKEND="Agg"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${ROOT_DIR}/.runtime-venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.runtime-venv/bin/python"
  elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi
export PYTHON_BIN

usage() {
  cat <<EOF
Usage: ./test.sh [options] [-- PYTEST_ARGS...]

Options:
  --runner               Use custom runner (src/tests/run_core_tests.py)
  --pytest               Use pytest (default)
  -k PATTERN             Pytest -k pattern (e.g., -k mlxQQCExamplesTest)
  --max N                Limit core test list (MLXQ_TEST_MAX)
  --ascii                Print ASCII circuits for executed Device circuits
  --failfast             Pytest fail fast
  --verbose              Pytest verbose (-vv)
  -h, --help             Show this help

Examples:
  ./test.sh                          # run pytest default suite
  ./test.sh --runner                 # run pretty custom runner
  ./test.sh -k mlxQQCExamplesTest    # run examples test file only
  ./test.sh --max 50                 # limit core test enumeration
  ./test.sh --ascii                  # show ASCII circuits (small n)
  ./test.sh -- --maxfail=1           # pass raw args to pytest
EOF
}

MODE="pytest"
K_PATTERN=""
FAILFAST=0
VERBOSE=0

EXTRA_PYTEST_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runner) MODE="runner"; shift ;;
    --pytest) MODE="pytest"; shift ;;
    -k) K_PATTERN="${2:-}"; shift 2 ;;
    --max) export MLXQ_TEST_MAX="${2:-}"; shift 2 ;;
    --ascii) export MLXQ_PRINT_ASCII=1; shift ;;
    --failfast) FAILFAST=1; shift ;;
    --verbose) VERBOSE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_PYTEST_ARGS=("$@"); break ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "$MODE" == "runner" ]]; then
  exec "${PYTHON_BIN}" "${ROOT_DIR}/src/tests/run_core_tests.py"
fi

ARGS=( )
[[ -n "$K_PATTERN" ]] && ARGS+=( -k "$K_PATTERN" )
[[ "$FAILFAST" == "1" ]] && ARGS+=( -x )
[[ "$VERBOSE" == "1" ]] && ARGS+=( -vv )

# pytest.ini already sets -q; let user override with --verbose. Bash 3.2 with
# `set -u` can treat empty array expansion as unbound, so build argv first.
FINAL_ARGS=()
if [[ ${#ARGS[@]} -gt 0 ]]; then
  FINAL_ARGS+=("${ARGS[@]}")
fi
if [[ ${#EXTRA_PYTEST_ARGS[@]} -gt 0 ]]; then
  FINAL_ARGS+=("${EXTRA_PYTEST_ARGS[@]}")
fi
if [[ ${#FINAL_ARGS[@]} -gt 0 ]]; then
  exec "${PYTHON_BIN}" -m pytest "${FINAL_ARGS[@]}"
fi
exec "${PYTHON_BIN}" -m pytest
