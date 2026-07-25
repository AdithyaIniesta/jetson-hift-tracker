#!/usr/bin/env bash
# Offline Jetson deploy.
# Place this script and the .bundle file in any folder, then:
#   chmod +x deploy.sh && ./deploy.sh
#
# Works with or without internet.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUNDLE=$(ls -t "$SCRIPT_DIR"/feature-dual-camera-v2-*.bundle 2>/dev/null | head -1)
BRANCH=feature/dual-camera-v2
REPO_URL=https://github.com/AdithyaIniesta/jetson-tracking-perception.git
CLONE_DIR="$SCRIPT_DIR/jetson-tracking-perception"

if [ -z "$BUNDLE" ]; then
    echo "ERROR: no feature-dual-camera-v2-*.bundle next to deploy.sh" >&2
    exit 1
fi
echo "Using bundle: $(basename "$BUNDLE")"

# Fresh clone from bundle. Any prior clone is removed.
rm -rf "$CLONE_DIR"
git clone "$BUNDLE" "$CLONE_DIR"
cd "$CLONE_DIR"

# The bundle has no HEAD symref, but its refs/heads/<branch> lands
# as a local branch after clone. Just check it out — no -B, which
# would recreate an empty branch pointing at nothing.
git checkout "$BRANCH"

# Repoint origin at GitHub (harmless if offline; fetch is best-effort).
git remote set-url origin "$REPO_URL"
git fetch origin --tags >/dev/null 2>&1 \
    && git branch --set-upstream-to="origin/$BRANCH" >/dev/null 2>&1 \
    && echo "  origin fetched." \
    || echo "  (offline — origin set but not fetched)"

# Make top-level shell entry points executable.
for f in build.sh run.sh; do
    if [ -f "$f" ]; then
        chmod +x "$f"
        echo "  chmod +x $f"
    fi
done

echo ""
echo "Deployed to: $CLONE_DIR"
echo "HEAD: $(git log -1 --oneline)"
