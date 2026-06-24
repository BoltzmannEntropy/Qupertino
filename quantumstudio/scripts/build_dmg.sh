#!/usr/bin/env bash
# =============================================================================
# QuantumStudio - DMG Installer Builder
# =============================================================================

set -euo pipefail

APP_NAME="QuantumStudio"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
FLUTTER_DIR="$ROOT_DIR/flutter_app"
BUILD_DIR="$FLUTTER_DIR/build/macos/Build/Products/Release"
PUBSPEC_FILE="$FLUTTER_DIR/pubspec.yaml"
VERSION_FILE="$FLUTTER_DIR/lib/version.dart"
DIST_DIR="$ROOT_DIR/dist"
DMG_DIR="$DIST_DIR"

PYTHON_VERSION="3.11.9"
PYTHON_STANDALONE_BASE="https://github.com/indygreg/python-build-standalone/releases/download/20240415"
PYTHON_CACHE_DIR="$ROOT_DIR/build/python-standalone"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info() { echo -e "${BLUE}$*${NC}"; }
ok() { echo -e "${GREEN}✓ $*${NC}"; }
warn() { echo -e "${YELLOW}⚠ $*${NC}"; }
error() { echo -e "${RED}✗ $*${NC}"; exit 1; }

python_url_for_arch() {
    local arch="$1"
    if [ "$arch" = "arm64" ]; then
        echo "${PYTHON_STANDALONE_BASE}/cpython-${PYTHON_VERSION}+20240415-aarch64-apple-darwin-install_only.tar.gz"
    else
        echo "${PYTHON_STANDALONE_BASE}/cpython-${PYTHON_VERSION}+20240415-x86_64-apple-darwin-install_only.tar.gz"
    fi
}

verify_python_tarball_checksum() {
    local tar_path="$1"
    local checksum_path="$2"
    [ -f "$tar_path" ] || error "Missing tarball for checksum verification: $tar_path"
    [ -f "$checksum_path" ] || error "Missing checksum file: $checksum_path"

    local expected actual
    expected="$(awk '{print $1}' "$checksum_path" | head -n 1)"
    [ -n "$expected" ] || error "Checksum file is empty: $checksum_path"
    actual="$(shasum -a 256 "$tar_path" | awk '{print $1}')"
    if [ "$expected" != "$actual" ]; then
        error "Checksum verification failed for $(basename "$tar_path")"
    fi
}

read_version_from_pubspec() {
    if [ -f "$PUBSPEC_FILE" ]; then
        grep '^version:' "$PUBSPEC_FILE" | head -1 | cut -d'+' -f1 | cut -d':' -f2 | xargs
    fi
}

read_version_from_dart_fallback() {
    if [ -f "$VERSION_FILE" ]; then
        sed -n 's/^const String appVersion = "\(.*\)";/\1/p' "$VERSION_FILE" | head -n 1
    fi
}

VERSION="${1:-$(read_version_from_pubspec)}"
if [ -z "$VERSION" ]; then
    VERSION="$(read_version_from_dart_fallback)"
fi
if [ -z "$VERSION" ]; then
    VERSION="1.0.0"
fi
DMG_NAME="${APP_NAME}-${VERSION}-macos.dmg"

echo ""
echo -e "${BLUE}==============================================================================${NC}"
echo -e "${BLUE}  Building ${APP_NAME} DMG Installer v${VERSION}${NC}"
echo -e "${BLUE}==============================================================================${NC}"
echo ""

info "Checking Flutter..."
command -v flutter &>/dev/null || error "Flutter not found. Please install Flutter first."
ok "Flutter available"

HOST_ARCH="$(uname -m)"
if [ "$HOST_ARCH" != "arm64" ] && [ "$HOST_ARCH" != "x86_64" ]; then
    warn "Unknown architecture '$HOST_ARCH', defaulting to x86_64 Python runtime"
    HOST_ARCH="x86_64"
fi

PYTHON_URL="$(python_url_for_arch "$HOST_ARCH")"
PYTHON_SHA_URL="${PYTHON_URL}.sha256"
mkdir -p "$PYTHON_CACHE_DIR" "$DMG_DIR"
PYTHON_TAR="$PYTHON_CACHE_DIR/python-${PYTHON_VERSION}-${HOST_ARCH}.tar.gz"
PYTHON_SHA_FILE="$PYTHON_CACHE_DIR/python-${PYTHON_VERSION}-${HOST_ARCH}.sha256"

if [ ! -f "$PYTHON_TAR" ]; then
    info "Downloading standalone Python ${PYTHON_VERSION} for ${HOST_ARCH}..."
    curl --fail --location --proto '=https' --tlsv1.2 "$PYTHON_URL" -o "$PYTHON_TAR"
else
    ok "Using cached Python ${PYTHON_VERSION} for ${HOST_ARCH}"
fi

info "Verifying standalone Python checksum..."
curl --fail --location --proto '=https' --tlsv1.2 "$PYTHON_SHA_URL" -o "$PYTHON_SHA_FILE"
verify_python_tarball_checksum "$PYTHON_TAR" "$PYTHON_SHA_FILE"
ok "Standalone Python checksum verified"

info "Building Flutter macOS release..."
cd "$FLUTTER_DIR"
flutter pub get
flutter build macos --release

APP_PATH="$BUILD_DIR/${APP_NAME}.app"
[ -d "$APP_PATH" ] || error "Build failed: ${APP_NAME}.app not found at $BUILD_DIR"
ok "Flutter build complete"
ok "App bundle: $APP_PATH"

info "Bundling backend and MLX resources..."
RESOURCES_DIR="$APP_PATH/Contents/Resources"
BACKEND_BUNDLE="$RESOURCES_DIR/backend"
MLX_BUNDLE="$RESOURCES_DIR/mlx"
PYTHON_BUNDLE="$RESOURCES_DIR/python"

rm -rf "$BACKEND_BUNDLE" "$MLX_BUNDLE" "$PYTHON_BUNDLE"
mkdir -p "$BACKEND_BUNDLE" "$MLX_BUNDLE"

cp "$ROOT_DIR/backend/main.py" "$BACKEND_BUNDLE/"
cp "$ROOT_DIR/backend/job_worker.py" "$BACKEND_BUNDLE/"
cp "$ROOT_DIR/backend/requirements.txt" "$BACKEND_BUNDLE/"
cp "$ROOT_DIR/bin/quantumstudio_mcp_server.py" "$BACKEND_BUNDLE/"
mkdir -p "$BACKEND_BUNDLE/runs"

cp -R "$ROOT_DIR/../bench" "$MLX_BUNDLE/bench"
cp -R "$ROOT_DIR/../src" "$MLX_BUNDLE/src"
[ -d "$ROOT_DIR/../tools" ] && cp -R "$ROOT_DIR/../tools" "$MLX_BUNDLE/tools" || true
[ -d "$ROOT_DIR/../datasets" ] && cp -R "$ROOT_DIR/../datasets" "$MLX_BUNDLE/datasets" || true

info "Bundling Python ${PYTHON_VERSION}..."
mkdir -p "$PYTHON_BUNDLE"
xattr -d com.apple.quarantine "$PYTHON_TAR" 2>/dev/null || true
TEMP_PYTHON_DIR="$(mktemp -d)"
tar -xzf "$PYTHON_TAR" -C "$TEMP_PYTHON_DIR" --strip-components=1
ditto --noextattr --noqtn "$TEMP_PYTHON_DIR" "$PYTHON_BUNDLE"
rm -rf "$TEMP_PYTHON_DIR"

info "Creating backend venv and installing dependencies..."
VENV_DIR="$BACKEND_BUNDLE/venv"
PYTHON_SHORT="$("$PYTHON_BUNDLE/bin/python3" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PYTHON_FULL="$("$PYTHON_BUNDLE/bin/python3" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
"$PYTHON_BUNDLE/bin/python3" -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip wheel setuptools
"$VENV_DIR/bin/pip" install -r "$BACKEND_BUNDLE/requirements.txt"

# Ensure venv interpreter paths stay valid after the app is moved from build
# products into /Applications.
rm -f "$VENV_DIR/bin/python" "$VENV_DIR/bin/python3" "$VENV_DIR/bin/python$PYTHON_SHORT"
ln -sf "../../../python/bin/python3" "$VENV_DIR/bin/python3"
ln -sf "python3" "$VENV_DIR/bin/python"
ln -sf "python3" "$VENV_DIR/bin/python$PYTHON_SHORT"

cat > "$VENV_DIR/pyvenv.cfg" <<CFG
home = ../../python/bin
include-system-site-packages = false
version = $PYTHON_FULL
executable = ../../python/bin/python3
CFG

info "Creating backend launcher script..."
cat > "$BACKEND_BUNDLE/run_backend.sh" << 'LAUNCHER_EOF'
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESOURCES_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$SCRIPT_DIR/venv"
MLX_ROOT="$RESOURCES_DIR/mlx"

HOME_DIR="${HOME:-$SCRIPT_DIR}"
APP_SUPPORT_DIR="$HOME_DIR/Library/Application Support/QuantumStudio"
APP_CACHE_DIR="$HOME_DIR/Library/Caches/QuantumStudio"
APP_LOG_DIR="$HOME_DIR/Library/Logs/QuantumStudio"
HF_HOME_DIR="$APP_CACHE_DIR/huggingface"
HF_HUB_DIR="$HF_HOME_DIR/hub"
HF_TRANSFORMERS_DIR="$HF_HOME_DIR/transformers"

mkdir -p "$APP_SUPPORT_DIR/runs" "$APP_SUPPORT_DIR/bench" "$APP_CACHE_DIR" "$APP_CACHE_DIR/tmp" "$APP_LOG_DIR" "$HF_HOME_DIR" "$HF_HUB_DIR" "$HF_TRANSFORMERS_DIR"

LOG_FILE="$APP_LOG_DIR/backend.log"
if ! touch "$LOG_FILE" 2>/dev/null; then
    LOG_FILE="/tmp/quantumstudio-backend.log"
    touch "$LOG_FILE"
fi

PYTHON_BIN="$VENV_DIR/bin/python3"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Bundled venv Python not found at $PYTHON_BIN" >&2
    exit 1
fi

SITE_PACKAGES_DIR=""
for candidate in "$VENV_DIR"/lib/python*/site-packages; do
    if [ -d "$candidate" ]; then
        SITE_PACKAGES_DIR="$candidate"
        break
    fi
done

if [ -z "$SITE_PACKAGES_DIR" ]; then
    echo "Bundled site-packages directory not found in $VENV_DIR/lib" >&2
    exit 1
fi

export QUANTUMSTUDIO_RUNTIME_HOME="$APP_SUPPORT_DIR"
export QUANTUMSTUDIO_LOG_DIR="$APP_LOG_DIR"
export QUANTUMSTUDIO_RUNS_DIR="$APP_SUPPORT_DIR/runs"
export QUANTUMSTUDIO_SETTINGS_FILE="$APP_SUPPORT_DIR/settings.json"
export QUANTUMSTUDIO_BENCH_DIR="$APP_SUPPORT_DIR/bench"
export QUANTUMSTUDIO_MLX_ROOT="$MLX_ROOT"
export QUANTUMSTUDIO_MLX_PYTHON="$MLX_ROOT/src"
export PYTHONUNBUFFERED=1
export PYTHONHOME="$RESOURCES_DIR/python"
export PYTHONPYCACHEPREFIX="$APP_CACHE_DIR/pycache"
export TMPDIR="$APP_CACHE_DIR/tmp"
export PYTHONPATH="$SCRIPT_DIR:$SITE_PACKAGES_DIR:$MLX_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="$APP_CACHE_DIR"
export HF_HOME="$HF_HOME_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_DIR"
export TRANSFORMERS_CACHE="$HF_TRANSFORMERS_DIR"
export QUANTUMSTUDIO_MCP_LOG_DIR="$APP_LOG_DIR"
BACKEND_PORT="${QUANTUMSTUDIO_BACKEND_PORT:-8127}"

cd "$SCRIPT_DIR"
exec "$PYTHON_BIN" -m uvicorn main:app --host 127.0.0.1 --port "$BACKEND_PORT" >> "$LOG_FILE" 2>&1
LAUNCHER_EOF
chmod +x "$BACKEND_BUNDLE/run_backend.sh"

info "Running bundled backend import smoke test..."
SMOKE_SITE_PACKAGES="$(find "$VENV_DIR/lib" -maxdepth 2 -type d -name site-packages | head -n 1 || true)"
[ -n "$SMOKE_SITE_PACKAGES" ] || error "Bundled site-packages not found under $VENV_DIR/lib"
PYTHONHOME="$PYTHON_BUNDLE" PYTHONPATH="$BACKEND_BUNDLE:$SMOKE_SITE_PACKAGES:$MLX_BUNDLE/src" "$VENV_DIR/bin/python3" - <<'PY'
import fastapi
import numpy
import matplotlib
import main
print("backend-smoke-ok")
PY

info "Running bundled backend runtime smoke test..."
SMOKE_STDOUT="$(mktemp)"
SMOKE_HOME="$(mktemp -d)"
SMOKE_PORT="$(python3 - <<'PY'
import random
print(random.randint(18100, 18999))
PY
)"
SMOKE_PID=""

cleanup_smoke() {
    if [ -n "$SMOKE_PID" ]; then
        kill "$SMOKE_PID" >/dev/null 2>&1 || true
        wait "$SMOKE_PID" >/dev/null 2>&1 || true
    fi
    rm -f "$SMOKE_STDOUT"
    rm -rf "$SMOKE_HOME"
}

HOME="$SMOKE_HOME" QUANTUMSTUDIO_BACKEND_PORT="$SMOKE_PORT" "$BACKEND_BUNDLE/run_backend.sh" >"$SMOKE_STDOUT" 2>&1 &
SMOKE_PID=$!

SMOKE_READY=0
for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$SMOKE_PORT/api/health" >/dev/null 2>&1; then
        SMOKE_READY=1
        break
    fi
    sleep 0.5
done

if [ "$SMOKE_READY" -ne 1 ]; then
    echo "Bundled backend runtime smoke test failed." >&2
    cat "$SMOKE_STDOUT" >&2 || true
    cat "$SMOKE_HOME/Library/Logs/QuantumStudio/backend.log" >&2 || true
    cleanup_smoke
    error "Bundled backend did not become healthy"
fi

curl -sf "http://127.0.0.1:$SMOKE_PORT/api/benchmarks" >/dev/null 2>&1 || {
    cat "$SMOKE_STDOUT" >&2 || true
    cat "$SMOKE_HOME/Library/Logs/QuantumStudio/backend.log" >&2 || true
    cleanup_smoke
    error "Bundled backend runtime smoke test failed on /api/benchmarks"
}

cleanup_smoke

cp "$ROOT_DIR/LICENSE" "$RESOURCES_DIR/LICENSE"
cp "$ROOT_DIR/BINARY-LICENSE.txt" "$RESOURCES_DIR/BINARY-LICENSE.txt"

# Recreate DMG target if present
if [ -f "$DMG_DIR/$DMG_NAME" ]; then
    rm "$DMG_DIR/$DMG_NAME"
    info "Removed existing DMG"
fi

DMG_STAGE="/tmp/quantumstudio-dmg-stage-${USER:-user}-$$"
rm -rf "$DMG_STAGE"
mkdir -p "$DMG_STAGE"
ditto --noextattr --noqtn "$APP_PATH" "$DMG_STAGE/$APP_NAME.app"
ln -s /Applications "$DMG_STAGE/Applications" 2>/dev/null || true
cp "$ROOT_DIR/LICENSE" "$DMG_STAGE/LICENSE"
cp "$ROOT_DIR/BINARY-LICENSE.txt" "$DMG_STAGE/BINARY-LICENSE.txt"

info "Creating DMG installer..."
if command -v create-dmg &>/dev/null; then
    info "Using create-dmg..."
    ICON_PATH="$FLUTTER_DIR/macos/Runner/Assets.xcassets/AppIcon.appiconset/app_icon_512.png"
    CREATE_DMG_EULA_ARGS=()
    if [ ! -f "$ICON_PATH" ]; then
        ICON_PATH=""
        warn "App icon not found, using default"
    fi
    if create-dmg --help 2>/dev/null | grep -q -- "--eula"; then
        CREATE_DMG_EULA_ARGS+=(--eula "$ROOT_DIR/BINARY-LICENSE.txt")
    fi

    if ! create-dmg \
        --volname "$APP_NAME" \
        ${ICON_PATH:+--volicon "$ICON_PATH"} \
        "${CREATE_DMG_EULA_ARGS[@]}" \
        --window-pos 200 120 \
        --window-size 600 400 \
        --icon-size 100 \
        --icon "$APP_NAME.app" 150 190 \
        --hide-extension "$APP_NAME.app" \
        --app-drop-link 450 185 \
        --no-internet-enable \
        "$DMG_DIR/$DMG_NAME" \
        "$DMG_STAGE"; then
        warn "create-dmg failed, using hdiutil fallback..."
        hdiutil create -volname "$APP_NAME" -srcfolder "$DMG_STAGE" -ov -format UDZO "$DMG_DIR/$DMG_NAME"
    fi
else
    info "Using hdiutil fallback..."
    hdiutil create -volname "$APP_NAME" -srcfolder "$DMG_STAGE" -ov -format UDZO "$DMG_DIR/$DMG_NAME"
fi

if [ -f "$DMG_DIR/$DMG_NAME" ]; then
    DMG_SIZE="$(du -h "$DMG_DIR/$DMG_NAME" | cut -f1)"
    ok "DMG created: $DMG_DIR/$DMG_NAME ($DMG_SIZE)"

    SHA256="$(shasum -a 256 "$DMG_DIR/$DMG_NAME" | cut -d' ' -f1)"
    echo "$SHA256  $DMG_NAME" > "$DMG_DIR/$DMG_NAME.sha256"
    ok "SHA256: $SHA256"

    cp "$ROOT_DIR/LICENSE" "$DMG_DIR/LICENSE" 2>/dev/null || true
    cp "$ROOT_DIR/BINARY-LICENSE.txt" "$DMG_DIR/BINARY-LICENSE.txt" 2>/dev/null || true

    # Copy release notes if present
    RELEASE_NOTES_SRC="$ROOT_DIR/RELEASE_NOTES.md"
    RELEASE_NOTES_NAME="${APP_NAME}-${VERSION}-RELEASE_NOTES.md"
    if [ -f "$RELEASE_NOTES_SRC" ]; then
        cp "$RELEASE_NOTES_SRC" "$DMG_DIR/$RELEASE_NOTES_NAME"
        cd "$DMG_DIR"
        shasum -a 256 "$RELEASE_NOTES_NAME" > "$RELEASE_NOTES_NAME.sha256"
        ok "Release notes copied"
    fi
else
    error "DMG creation failed"
fi

echo ""
info "Code signing (optional):"
echo "  codesign --deep --force --verify --verbose --sign \"Developer ID Application: Your Name (TEAM_ID)\" \"$APP_PATH\""

echo ""
echo -e "${GREEN}==============================================================================${NC}"
echo -e "${GREEN}  Build Complete!${NC}"
echo -e "${GREEN}==============================================================================${NC}"
echo ""
echo "DMG: $DMG_DIR/$DMG_NAME"
echo "Size: $DMG_SIZE"
echo ""
echo "To install:"
echo "  1. Double-click the DMG to mount"
echo "  2. Drag ${APP_NAME}.app to Applications"
echo "  3. Eject the DMG"
echo ""
