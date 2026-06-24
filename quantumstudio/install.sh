#!/usr/bin/env bash
# =============================================================================
# QuantumStudio - Installation Script
# =============================================================================
# Sets up Python venv, installs dependencies, and prepares Flutter app
#
# Usage: ./install.sh
# =============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FLUTTER_DIR="$ROOT_DIR/flutter_app"
VENV_DIR="$BACKEND_DIR/venv"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${BLUE}$*${NC}"; }
ok()    { echo -e "${GREEN}✓ $*${NC}"; }
warn()  { echo -e "${YELLOW}⚠ $*${NC}"; }
error() { echo -e "${RED}✗ $*${NC}"; }

echo ""
echo -e "${CYAN}==============================================================================${NC}"
echo -e "${CYAN}  QuantumStudio Installation${NC}"
echo -e "${CYAN}==============================================================================${NC}"
echo ""

# =============================================================================
# 1. Prerequisites Check
# =============================================================================
info "Checking prerequisites..."

# Python 3
if command -v python3 &>/dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    ok "Python $PYTHON_VERSION"
else
    error "Python 3 not found"
    echo "Please install Python 3.10+ via:"
    echo "  brew install python@3.11"
    exit 1
fi

# pip
if command -v pip3 &>/dev/null; then
    ok "pip3 available"
else
    warn "pip3 not found, will use python3 -m pip"
fi

# Flutter (optional)
if command -v flutter &>/dev/null; then
    FLUTTER_VERSION=$(flutter --version 2>&1 | head -1 | awk '{print $2}')
    ok "Flutter $FLUTTER_VERSION"
else
    warn "Flutter not installed (optional for UI)"
    echo "  Install: https://docs.flutter.dev/get-started/install/macos"
fi

echo ""

# =============================================================================
# 2. Python Virtual Environment
# =============================================================================
info "Setting up Python virtual environment..."

if [ -d "$VENV_DIR" ]; then
    ok "venv already exists at $VENV_DIR"
else
    python3 -m venv "$VENV_DIR"
    ok "Created venv at $VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip --quiet
ok "pip upgraded"

echo ""

# =============================================================================
# 3. Python Dependencies
# =============================================================================
info "Installing Python dependencies..."

if [ -f "$BACKEND_DIR/requirements.txt" ]; then
    pip install -r "$BACKEND_DIR/requirements.txt" --quiet
    ok "Dependencies installed from requirements.txt"
else
    error "requirements.txt not found at $BACKEND_DIR"
    exit 1
fi

# Verify key packages
for pkg in fastapi uvicorn pydantic mlx; do
    if pip show "$pkg" &>/dev/null; then
        version=$(pip show "$pkg" 2>/dev/null | grep "^Version:" | cut -d' ' -f2)
        ok "$pkg $version"
    else
        warn "$pkg not installed"
    fi
done

echo ""

# =============================================================================
# 4. Create directories
# =============================================================================
info "Creating directories..."

mkdir -p "$ROOT_DIR/.logs"
mkdir -p "$ROOT_DIR/.pids"
mkdir -p "$ROOT_DIR/runs"

ok "Created .logs/, .pids/, runs/"

echo ""

# =============================================================================
# 5. Flutter Setup (if available)
# =============================================================================
if command -v flutter &>/dev/null; then
    info "Setting up Flutter..."

    cd "$FLUTTER_DIR"

    # Get dependencies
    flutter pub get --quiet
    ok "Flutter dependencies installed"

    # Enable macOS desktop
    flutter config --enable-macos-desktop 2>/dev/null || true
    ok "macOS desktop enabled"

    echo ""
fi

# =============================================================================
# 6. Verify Installation
# =============================================================================
info "Verifying installation..."

# Test Python imports
"$VENV_DIR/bin/python" -c "
import sys
modules = ['fastapi', 'uvicorn', 'pydantic', 'mlx']
failed = []
for mod in modules:
    try:
        __import__(mod)
    except ImportError:
        failed.append(mod)

if failed:
    print(f'Failed imports: {failed}')
    sys.exit(1)
print('All imports successful')
" && ok "Python imports verified" || error "Some imports failed"

echo ""

# =============================================================================
# Complete
# =============================================================================
echo -e "${GREEN}==============================================================================${NC}"
echo -e "${GREEN}  Installation Complete!${NC}"
echo -e "${GREEN}==============================================================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Start the application:"
echo "     ./bin/appctl up"
echo ""
echo "  2. Or start backend only:"
echo "     ./bin/appctl backend start"
echo ""
echo "  3. Check status:"
echo "     ./bin/appctl status"
echo ""
echo "  (Legacy alias still works: ./bin/quantumctl ...)"
echo ""
echo "Backend will be available at: http://localhost:8127"
echo "API docs at: http://localhost:8127/docs"
echo ""
