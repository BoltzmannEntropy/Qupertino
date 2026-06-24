#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WEBSITE_DIR="$PROJECT_DIR/../../../QuantumStudioWEB"
APP_NAME="QuantumStudio"
REPO_SLUG="${GITHUB_REPO:-BoltzmannEntropy/osxQ}"

UPLOAD_TO_GITHUB=false
SYNC_WEBSITE=false

for arg in "$@"; do
    case "$arg" in
        --upload) UPLOAD_TO_GITHUB=true ;;
        --sync-website) SYNC_WEBSITE=true ;;
        *)
            echo "Unknown flag: $arg"
            echo "Usage: ./scripts/release.sh [--upload] [--sync-website]"
            exit 1
            ;;
    esac
done

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info() { echo -e "${BLUE}$*${NC}"; }
ok() { echo -e "${GREEN}✓ $*${NC}"; }
warn() { echo -e "${YELLOW}$*${NC}"; }
fail() { echo -e "${RED}✗ $*${NC}"; exit 1; }

read_version_from_pubspec() {
    local pubspec="$PROJECT_DIR/flutter_app/pubspec.yaml"
    [ -f "$pubspec" ] || fail "Missing pubspec: $pubspec"
    grep '^version:' "$pubspec" | head -1 | cut -d'+' -f1 | cut -d':' -f2 | xargs
}

VERSION="$(read_version_from_pubspec)"
TAG="v$VERSION"
DMG_NAME="${APP_NAME}-${VERSION}-macos.dmg"
DIST_DIR="$PROJECT_DIR/dist"
DMG_PATH="$DIST_DIR/$DMG_NAME"
SHA_PATH="$DMG_PATH.sha256"
RELEASE_NOTES="$PROJECT_DIR/RELEASE_NOTES.md"
SOURCE_ZIP_PATH="$DIST_DIR/${APP_NAME}-${VERSION}-source.zip"
SOURCE_SHA_PATH="$SOURCE_ZIP_PATH.sha256"
RELEASE_NOTES_ASSET="$DIST_DIR/${APP_NAME}-${VERSION}-RELEASE_NOTES.md"
RELEASE_NOTES_SHA_PATH="$RELEASE_NOTES_ASSET.sha256"
DOWNLOAD_URL="https://github.com/${REPO_SLUG}/releases/download/${TAG}/${DMG_NAME}"

info "=== QuantumStudio Release Script ==="
info "Version: $VERSION"
info "Upload to GitHub: $UPLOAD_TO_GITHUB"
info "Sync website: $SYNC_WEBSITE"
echo ""

info "Building DMG..."
"$SCRIPT_DIR/build_dmg.sh" "$VERSION"
[ -f "$DMG_PATH" ] || fail "DMG not found: $DMG_PATH"
ok "DMG ready: $DMG_PATH"
[ -f "$RELEASE_NOTES" ] || fail "Release notes file is required: $RELEASE_NOTES"

if [ ! -f "$SHA_PATH" ]; then
    info "Generating SHA256 checksum..."
    (cd "$(dirname "$DMG_PATH")" && shasum -a 256 "$(basename "$DMG_PATH")" > "$(basename "$SHA_PATH")")
fi
[ -f "$SHA_PATH" ] || fail "Checksum file not found: $SHA_PATH"
ok "Checksum ready: $SHA_PATH"

info "Creating source archive..."
rm -f "$SOURCE_ZIP_PATH"
if git -C "$PROJECT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "$PROJECT_DIR" archive --format=zip --output="$SOURCE_ZIP_PATH" HEAD
else
    (
        cd "$PROJECT_DIR"
        zip -q -r "$SOURCE_ZIP_PATH" . \
            -x "dist/*" \
            -x "build/*" \
            -x ".logs/*" \
            -x ".pids/*" \
            -x "logs/*" \
            -x "runs/*" \
            -x "backend/venv/*" \
            -x "backend/_old_venv/*" \
            -x "flutter_app/build/*" \
            -x "*/__pycache__/*"
    )
fi
(cd "$DIST_DIR" && shasum -a 256 "$(basename "$SOURCE_ZIP_PATH")" > "$(basename "$SOURCE_SHA_PATH")")
[ -f "$SOURCE_ZIP_PATH" ] || fail "Source archive not found: $SOURCE_ZIP_PATH"
[ -f "$SOURCE_SHA_PATH" ] || fail "Source archive checksum not found: $SOURCE_SHA_PATH"
ok "Source archive ready: $SOURCE_ZIP_PATH"

cp "$RELEASE_NOTES" "$RELEASE_NOTES_ASSET"
(cd "$DIST_DIR" && shasum -a 256 "$(basename "$RELEASE_NOTES_ASSET")" > "$(basename "$RELEASE_NOTES_SHA_PATH")")
[ -f "$RELEASE_NOTES_ASSET" ] || fail "Release notes asset not found: $RELEASE_NOTES_ASSET"
[ -f "$RELEASE_NOTES_SHA_PATH" ] || fail "Release notes checksum not found: $RELEASE_NOTES_SHA_PATH"
ok "Release notes asset ready: $RELEASE_NOTES_ASSET"

if [ "$UPLOAD_TO_GITHUB" = true ]; then
    info ""
    info "Uploading to GitHub release..."

    command -v gh >/dev/null 2>&1 || fail "GitHub CLI (gh) not found"
    gh auth status >/dev/null 2>&1 || fail "GitHub CLI not authenticated (run: gh auth login)"

    cd "$PROJECT_DIR"
    if ! gh release view "$TAG" >/dev/null 2>&1; then
        gh release create "$TAG" \
            --title "$APP_NAME $VERSION" \
            --notes-file "$RELEASE_NOTES" \
            --draft
        ok "Created draft release: $TAG"
    else
        gh release edit "$TAG" \
            --title "$APP_NAME $VERSION" \
            --notes-file "$RELEASE_NOTES"
        warn "Release $TAG already exists; metadata refreshed and assets will be uploaded with --clobber"
    fi

    gh release upload "$TAG" \
        "$DMG_PATH" "$SHA_PATH" \
        "$SOURCE_ZIP_PATH" "$SOURCE_SHA_PATH" \
        "$RELEASE_NOTES_ASSET" "$RELEASE_NOTES_SHA_PATH" \
        --clobber

    asset_count="$(gh release view "$TAG" --json assets --jq '.assets | length' 2>/dev/null || echo 0)"
    [ "$asset_count" -gt 0 ] || fail "GitHub release $TAG has no assets after upload"
    ok "Uploaded release assets to $TAG (assets: $asset_count)"
fi

if [ "$SYNC_WEBSITE" = true ]; then
    info ""
    info "Syncing website links..."

    [ -d "$WEBSITE_DIR" ] || fail "Website directory not found: $WEBSITE_DIR"
    local_index="$WEBSITE_DIR/index.html"
    [ -f "$local_index" ] || fail "Website index not found: $local_index"

    # Update release download URLs if already present.
    sed -i '' -E \
        "s|https://github.com/[^\"']+/releases/download/v[0-9]+([.][0-9]+)*/QuantumStudio-[0-9]+([.][0-9]+)*-macos\\.dmg|${DOWNLOAD_URL}|g" \
        "$local_index"

    # Update nav + hero download CTAs (id-based) to point to the direct DMG URL.
    sed -i '' -E \
        "s|(<a id=\"download-link-nav\" href=\")[^\"]*(\" class=\"nav-cta desktop-download\">)|\\1${DOWNLOAD_URL}\\2|g" \
        "$local_index"
    sed -i '' -E \
        "s|(<a id=\"download-link-hero\" href=\")[^\"]*(\" class=\"btn-primary\">)|\\1${DOWNLOAD_URL}\\2|g" \
        "$local_index"

    # Keep the license line consistent with QuantumStudio naming.
    sed -i '' -E \
        "s|binary distributions are licensed under the osxQ Binary Distribution License|binary distributions are licensed under the QuantumStudio Binary Distribution License|g" \
        "$local_index"

    cd "$WEBSITE_DIR"
    if git diff --quiet -- index.html; then
        ok "No website updates needed"
    else
        git add index.html
        git commit -m "Update QuantumStudio website links for v${VERSION}"
        git push
        ok "Website index updated, committed, and pushed"
    fi
    cd "$PROJECT_DIR"
fi

echo ""
ok "Release workflow completed"
echo "DMG: $DMG_PATH"
echo "SHA256: $SHA_PATH"
if [ "$UPLOAD_TO_GITHUB" = false ]; then
    echo "Tip: run with --upload to create/update a draft GitHub release."
fi
if [ "$SYNC_WEBSITE" = false ]; then
    echo "Tip: run with --sync-website to update website download links."
fi
