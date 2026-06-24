#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.runtime-venv"
PY_BIN="${PYTHON_BIN:-$(command -v python3)}"
USE_SYSTEM_SITE="${USE_SYSTEM_SITE:-1}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"

BACKEND_REQ="${ROOT_DIR}/quantumstudio/backend/requirements.txt"

echo "[setup] root: ${ROOT_DIR}"
echo "[setup] python: ${PY_BIN}"
echo "[setup] venv: ${VENV_DIR}"

if [[ ! -x "${PY_BIN}" ]]; then
  echo "[setup] ERROR: python not executable: ${PY_BIN}" >&2
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[setup] creating venv"
  if [[ "${USE_SYSTEM_SITE}" == "1" ]]; then
    "${PY_BIN}" -m venv --system-site-packages "${VENV_DIR}"
  else
    "${PY_BIN}" -m venv "${VENV_DIR}"
  fi
fi

VENV_PY="${VENV_DIR}/bin/python"
if [[ ! -x "${VENV_PY}" ]]; then
  echo "[setup] ERROR: missing venv python: ${VENV_PY}" >&2
  exit 1
fi

"${VENV_PY}" -V
"${VENV_PY}" -m pip --version || true

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  echo "[setup] installing backend requirements (best effort)"
  "${VENV_PY}" -m pip install -r "${BACKEND_REQ}" || echo "[setup] WARN: dependency install failed (likely offline). Continuing with available packages."
fi

# Hard checks required for runtime
"${VENV_PY}" - <<'PY'
import importlib.util as iu
req = ["mlx", "fastapi", "uvicorn", "pydantic", "numpy"]
missing = [m for m in req if iu.find_spec(m) is None]
if missing:
    raise SystemExit(f"[setup] ERROR: missing modules: {', '.join(missing)}")
print("[setup] required modules present")
PY

# MLX smoke test
"${VENV_PY}" - <<'PY'
import mlx.core as mx
mx.array([1.0])
print("[setup] mlx smoke test OK")
PY

echo "[setup] runtime ready"
echo "[setup] export: QUANTUMSTUDIO_PYTHON=${VENV_PY}"
