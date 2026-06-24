#!/usr/bin/env bash
# =============================================================================
# QuantumStudio - Diagnostic Script
# =============================================================================
# Collects system info, checks dependencies, tests API, for troubleshooting.
#
# Usage: ./issues.sh
# Output: issues_report_<timestamp>.log
# =============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$ROOT_DIR/issues_report_$TIMESTAMP.log"
VENV_DIR="$ROOT_DIR/backend/venv"
BACKEND_DIR="$ROOT_DIR/backend"
BACKEND_PORT=8127

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "$*" | tee -a "$LOG_FILE"; }
section() {
    log ""
    log "============================================================================="
    log "  $*"
    log "============================================================================="
}
subsection() { log "\n--- $* ---"; }

run_cmd() {
    local desc="$1"
    shift
    log "$ $*"
    if output=$("$@" 2>&1); then
        log "$output"
        return 0
    else
        local exit_code=$?
        log "$output"
        log "${RED}[FAILED]${NC} $desc (exit code: $exit_code)"
        return $exit_code
    fi
}

# Start report
echo "" > "$LOG_FILE"
log "${CYAN}QuantumStudio Diagnostic Report${NC}"
log "Generated: $(date)"
log "Log file: $LOG_FILE"

# =============================================================================
# 1. System Information
# =============================================================================
section "SYSTEM INFORMATION"

subsection "macOS Version"
run_cmd "macOS version" sw_vers 2>/dev/null || log "Not macOS or sw_vers unavailable"

subsection "Kernel / OS"
run_cmd "Kernel" uname -a

subsection "Architecture"
run_cmd "Architecture" uname -m

subsection "Hardware Info"
if command -v system_profiler &> /dev/null; then
    log "$(system_profiler SPHardwareDataType 2>/dev/null | grep -E 'Model|Chip|Memory|Cores' || echo 'Unable to get hardware info')"
else
    log "system_profiler not available"
fi

subsection "Disk Space"
run_cmd "Disk space" df -h "$ROOT_DIR"

# =============================================================================
# 2. Development Tools
# =============================================================================
section "DEVELOPMENT TOOLS"

subsection "Python"
if command -v python3 &> /dev/null; then
    run_cmd "Python3 version" python3 --version
    run_cmd "Python3 path" which python3
else
    log "${RED}Python3 NOT found${NC}"
fi

subsection "pip"
if command -v pip3 &> /dev/null; then
    run_cmd "pip3 version" pip3 --version
else
    log "pip3 not found in PATH"
fi

subsection "Flutter"
if command -v flutter &> /dev/null; then
    log "$(flutter --version 2>&1 | head -3)"
else
    log "${YELLOW}Flutter NOT installed${NC}"
fi

subsection "MLX Framework"
if python3 -c "import mlx" 2>/dev/null; then
    mlx_version=$(python3 -c "import mlx; print(mlx.__version__)" 2>/dev/null || echo "unknown")
    log "${GREEN}MLX installed${NC}: $mlx_version"
else
    log "${YELLOW}MLX not available in system Python${NC}"
fi

# =============================================================================
# 3. Virtual Environment
# =============================================================================
section "PYTHON VIRTUAL ENVIRONMENT"

if [ -d "$VENV_DIR" ]; then
    log "${GREEN}venv exists at $VENV_DIR${NC}"

    subsection "venv Python"
    if [ -f "$VENV_DIR/bin/python" ]; then
        run_cmd "venv Python version" "$VENV_DIR/bin/python" --version
    fi

    subsection "Installed Packages"
    log "Key packages:"
    for pkg in fastapi uvicorn pydantic mlx numpy matplotlib rich; do
        version=$("$VENV_DIR/bin/pip" show "$pkg" 2>/dev/null | grep "^Version:" | cut -d' ' -f2)
        if [ -n "$version" ]; then
            log "  ${GREEN}$pkg${NC}: $version"
        else
            log "  ${YELLOW}$pkg${NC}: NOT INSTALLED"
        fi
    done

    subsection "Import Tests"
    log "Testing critical Python imports..."
    "$VENV_DIR/bin/python" -c "
import sys

modules = [
    ('fastapi', 'FastAPI web framework'),
    ('uvicorn', 'ASGI server'),
    ('pydantic', 'Data validation'),
    ('mlx', 'Apple MLX framework'),
    ('numpy', 'Numerical computing'),
    ('matplotlib', 'Plotting'),
    ('rich', 'Rich terminal output'),
]

print('')
for mod, desc in modules:
    try:
        __import__(mod)
        print(f'  [OK] {mod} ({desc})')
    except ImportError as e:
        print(f'  [FAIL] {mod} ({desc}): {e}')
    except Exception as e:
        print(f'  [ERROR] {mod} ({desc}): {type(e).__name__}: {e}')
" 2>&1 | tee -a "$LOG_FILE"

else
    log "${RED}venv NOT found at $VENV_DIR${NC}"
    log "Run ./install.sh first to create the virtual environment"
fi

# =============================================================================
# 4. Project Files
# =============================================================================
section "PROJECT FILES"

subsection "Directory Structure"
log "Key directories:"
for dir in backend flutter_app bin .logs .pids runs; do
    if [ -d "$ROOT_DIR/$dir" ]; then
        log "  ${GREEN}$dir/${NC} exists"
    else
        log "  ${YELLOW}$dir/${NC} missing"
    fi
done

subsection "Backend Files"
for file in main.py job_worker.py requirements.txt; do
    if [ -f "$BACKEND_DIR/$file" ]; then
        log "  ${GREEN}$file${NC} exists"
    else
        log "  ${RED}$file${NC} MISSING"
    fi
done

subsection "Settings File"
if [ -f "$ROOT_DIR/settings.json" ]; then
    log "${GREEN}settings.json exists${NC}"
    log "$(cat "$ROOT_DIR/settings.json")"
else
    log "${YELLOW}settings.json not found${NC} (will use defaults)"
fi

# =============================================================================
# 5. API Tests
# =============================================================================
section "API TESTS"

subsection "Backend Health Check"
if curl -s --connect-timeout 2 "http://localhost:$BACKEND_PORT/api/health" > /dev/null 2>&1; then
    log "${GREEN}Backend is running on port $BACKEND_PORT${NC}"

    log "\nHealth endpoint:"
    log "$(curl -s "http://localhost:$BACKEND_PORT/api/health")"

    log "\nSystem info endpoint:"
    log "$(curl -s "http://localhost:$BACKEND_PORT/api/system/info" | head -20)"

    log "\nBenchmarks list:"
    count=$(curl -s "http://localhost:$BACKEND_PORT/api/benchmarks" | grep -o '"name"' | wc -l)
    log "  ${GREEN}$count benchmarks available${NC}"

    log "\nRuns list:"
    runs_count=$(curl -s "http://localhost:$BACKEND_PORT/api/runs" | grep -o '"run_id"' | wc -l)
    log "  $runs_count runs in history"

    log "\nQueue status:"
    log "$(curl -s "http://localhost:$BACKEND_PORT/api/queue")"

else
    log "${YELLOW}Backend not running on port $BACKEND_PORT${NC}"
    log "Start with: ./bin/appctl up"
fi

# =============================================================================
# 6. Port Status
# =============================================================================
section "PORT STATUS"

subsection "Port $BACKEND_PORT (Backend)"
if command -v lsof &>/dev/null; then
    port_status=$(lsof -i :$BACKEND_PORT 2>/dev/null | grep LISTEN || echo "Not in use")
    log "$port_status"
else
    log "lsof not available"
fi

# =============================================================================
# 7. Recent Logs
# =============================================================================
section "RECENT LOGS"

subsection "Backend Log (last 20 lines)"
if [ -f "$ROOT_DIR/.logs/backend.log" ]; then
    log "$(tail -20 "$ROOT_DIR/.logs/backend.log")"
else
    log "No backend log found"
fi

subsection "Recent Run Logs"
if [ -d "$ROOT_DIR/runs" ]; then
    latest_log=$(ls -t "$ROOT_DIR/runs"/*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ]; then
        log "Latest run log: $latest_log"
        log "$(tail -10 "$latest_log")"
    else
        log "No run logs found"
    fi
fi

# =============================================================================
# 8. Environment Variables
# =============================================================================
section "ENVIRONMENT"

log "Relevant environment variables:"
for var in PATH PYTHONPATH VIRTUAL_ENV MLX_USE_GPU QS_BACKEND_PORT; do
    val="${!var:-<not set>}"
    if [ ${#val} -gt 100 ]; then
        log "  $var: ${val:0:100}..."
    else
        log "  $var: $val"
    fi
done

# =============================================================================
# Summary
# =============================================================================
section "SUMMARY"

log ""
log "Diagnostic report saved to: ${CYAN}$LOG_FILE${NC}"
log ""
log "If you're experiencing issues, please:"
log "  1. Review the log file for ${RED}[FAILED]${NC} or ${RED}ERROR${NC} messages"
log "  2. Share this log file when reporting issues"
log "  3. Try running ./install.sh to reinstall dependencies"
log ""
log "${GREEN}=== Diagnostic Complete ===${NC}"
