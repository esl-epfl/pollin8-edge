#!/usr/bin/env bash
#SBATCH --job-name=insect-tile
#SBATCH --gres=gpu:1                 # Kuma requires a GPU per job (QOSMinGRES)
# (no --partition: uses the cluster default. If needed, add: #SBATCH --partition=<name from `sinfo -s`>)
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# NOTE: set --account via `sbatch --account=$ACCOUNT ...` or add #SBATCH --account=...
set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/slurm/config.sh"
module purge; module load $PY_MODULES; source "$VENV/bin/activate"
cd "$CODE"; export PYTHONPATH="$CODE/src:$CODE/scripts"

# object-centric tiles for train; non-overlapping grid for val/test (faster val than overlap)
python -m insect_gap9.tile --src "$DATA/insects1201/train" --out "$DATA/train_tiled" --mode object --tile 320
python -m insect_gap9.tile --src "$DATA/insects1201/val"   --out "$DATA/val_tiled"   --mode grid --tile 320 --stride 320
python -m insect_gap9.tile --src "$DATA/insects1201/test"  --out "$DATA/test_tiled"  --mode grid --tile 320 --stride 320

python - <<PY
import yaml
names=list(yaml.safe_load(open("$DATA/insects1201/insects1201.auto.yaml"))['names'].values())
y={'train':"$DATA/train_tiled/images",'val':"$DATA/val_tiled/images",'test':"$DATA/test_tiled/images",'nc':9,'names':names}
open("$DATA/tiled.yaml","w").write(yaml.safe_dump(y,sort_keys=False)); print(open("$DATA/tiled.yaml").read())
PY
echo "[tile] done -> $DATA/tiled.yaml"
