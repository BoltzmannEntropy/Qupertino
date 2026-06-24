#!/usr/bin/env bash
# =============================================================================
# QuantumStudio - Flutter macOS App Builder
# =============================================================================
#
# Usage:
#   ./scripts/build_flutter_app.sh
#   ./scripts/build_flutter_app.sh --debug
#   ./scripts/build_flutter_app.sh --release --clean
#   ./scripts/build_flutter_app.sh --release --no-pub
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
FLUTTER_DIR="$ROOT_DIR/flutter_app"
APP_NAME="QuantumStudio"

MODE="release"
DO_CLEAN=false
NO_PUB=false

for arg in "$@"; do
    case "$arg" in
        --debug) MODE="debug" ;;
        --release) MODE="release" ;;
        --clean) DO_CLEAN=true ;;
        --no-pub) NO_PUB=true ;;
        *)
            echo "Unknown option: $arg" >&2
            echo "Usage: ./scripts/build_flutter_app.sh [--debug|--release] [--clean] [--no-pub]" >&2
            exit 1
            ;;
    esac
done

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

info() { echo -e "${BLUE}$*${NC}"; }
ok() { echo -e "${GREEN}✓ $*${NC}"; }
die() { echo -e "${RED}✗ $*${NC}"; exit 1; }

command -v flutter >/dev/null 2>&1 || die "Flutter not found in PATH"
[ -d "$FLUTTER_DIR" ] || die "Flutter directory not found: $FLUTTER_DIR"

cd "$FLUTTER_DIR"

if [ "$DO_CLEAN" = true ]; then
    info "Running flutter clean..."
    flutter clean
fi

if [ "$NO_PUB" = false ]; then
    info "Running flutter pub get..."
    flutter pub get
fi

info "Building Flutter macOS app (${MODE})..."
flutter build macos "--${MODE}" $([ "$NO_PUB" = true ] && echo "--no-pub")

BUILD_DIR="$FLUTTER_DIR/build/macos/Build/Products"
if [ "$MODE" = "release" ]; then
    MODE_DIR="Release"
else
    MODE_DIR="Debug"
fi
APP_PATH="$BUILD_DIR/$MODE_DIR/${APP_NAME}.app"

[ -d "$APP_PATH" ] || die "Build completed but app bundle not found: $APP_PATH"

ok "Flutter build complete"
echo "App bundle: $APP_PATH"
