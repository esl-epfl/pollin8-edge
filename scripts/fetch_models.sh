#!/usr/bin/env bash
# Download the trained checkpoints from the GitHub Release for reproduction, and verify
# their SHA256 checksums. Files land in ./models (gitignored via *.pt), leaving the repo clean.
#
#   bash scripts/fetch_models.sh [TAG] [DEST]
#
# Then reproduce any reported number, e.g. the deployed baseline:
#   make evaluate WEIGHTS=models/sensei_base_320_s0.pt
set -euo pipefail
TAG="${1:-weights-v1.0}"; DEST="${2:-models}"; REPO="${REPO:-esl-epfl/pollin8-edge}"

_shac() { if command -v sha256sum >/dev/null 2>&1; then sha256sum -c "$@"; else shasum -a 256 -c "$@"; fi; }
command -v gh >/dev/null 2>&1 || { echo "ERROR: GitHub CLI 'gh' not found. Install + 'gh auth login'."; exit 1; }

mkdir -p "$DEST"
gh release download "$TAG" --repo "$REPO" --dir "$DEST" --clobber
( cd "$DEST" && _shac SHA256SUMS.txt )
echo "[done] fetched $TAG -> $DEST/ (checksums OK)"
