#!/usr/bin/env bash
# Publish the trained checkpoints as a GitHub Release so the paper is fully reproducible.
#
# The full sweep is 4 loss recipes x 3 resolutions x 3 seeds = 36 checkpoints, plus the
# deployed INT8 model. Release assets do NOT bloat git history (unlike committing *.pt).
#
# Run this WHERE THE WEIGHTS LIVE (e.g. the SLURM cluster) on a host with `gh` authenticated
# (`gh auth status`). Weights are gitignored on purpose; this script never commits them.
#
#   bash scripts/publish_models.sh <SRC_DIR> [TAG]
#
# SRC_DIR must contain checkpoints named  sensei_<arm>_<imgsz>_s<seed>.pt
#   arm   in {base, focal, focal_noiw, nwd}
#   imgsz in {192, 320, 512}
#   seed  in {0, 1, 2}
# and (optionally) the deployed INT8 model(s) matching  sensei_*int8*  (e.g. a .tar.gz of
# build/insect_int8, or sensei_base_320_int8.onnx). Rename your run checkpoints to this scheme
# first (see models/README.md for the mapping from $RUNS to canonical names).
set -euo pipefail

SRC="${1:?usage: publish_models.sh <SRC_DIR> [TAG]}"
TAG="${2:-weights-v1.0}"
REPO="${REPO:-esl-epfl/pollin8-edge}"
ARMS=(base focal focal_noiw nwd); SIZES=(192 320 512); SEEDS=(0 1 2)

_sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }

command -v gh >/dev/null 2>&1 || { echo "ERROR: GitHub CLI 'gh' not found. Install + 'gh auth login'."; exit 1; }

stage="$(mktemp -d)"; n=0; missing=0
for arm in "${ARMS[@]}"; do for sz in "${SIZES[@]}"; do for sd in "${SEEDS[@]}"; do
  f="sensei_${arm}_${sz}_s${sd}.pt"
  if [[ -f "$SRC/$f" ]]; then cp "$SRC/$f" "$stage/"; n=$((n+1))
  else echo "MISSING: $f"; missing=$((missing+1)); fi
done; done; done

int8=0
for f in "$SRC"/sensei_*int8*; do [[ -e "$f" ]] && { cp "$f" "$stage/"; echo "+ $(basename "$f")"; int8=$((int8+1)); }; done

echo "collected $n/36 checkpoints + $int8 INT8 artifact(s); missing=$missing"
if [[ $missing -gt 0 ]]; then echo "Refusing to publish an incomplete set. Fix names in $SRC."; exit 1; fi

( cd "$stage" && _sha ./*.pt sensei_*int8* 2>/dev/null | sed 's#\./##' > SHA256SUMS.txt )
echo "=== SHA256SUMS.txt ==="; cat "$stage/SHA256SUMS.txt"

NOTES="36 checkpoints (4 loss recipes {base, focal, focal_noiw, nwd} x {192,320,512} px x 3 seeds)
plus the deployed INT8 model. Naming: sensei_<arm>_<imgsz>_s<seed>.pt. Fetch with
scripts/fetch_models.sh and reproduce via 'make evaluate WEIGHTS=models/<file>.pt'.
See models/README.md. Checksums in SHA256SUMS.txt."

if gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
  gh release upload "$TAG" "$stage"/* --repo "$REPO" --clobber
else
  gh release create "$TAG" "$stage"/* --repo "$REPO" --title "Trained weights ($TAG)" --notes "$NOTES"
fi
echo "[done] published $TAG to $REPO ($n checkpoints + $int8 INT8)"
