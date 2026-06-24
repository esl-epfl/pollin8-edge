# Trained weights (GitHub Release `weights-v1.0`)

The trained checkpoints are **not** stored in git history (binary blobs bloat the repo and
cannot be cleanly removed). They are published as **GitHub Release assets** and fetched on
demand, so a clean clone is fully reproducible.

## What's in the release

The **full loss-ablation sweep**: 4 recipes × 3 input resolutions × 3 seeds = **36
checkpoints**, plus the deployed **INT8** model.

| field | values |
|-------|--------|
| recipe (`arm`) | `base`, `focal`, `focal_noiw`, `nwd` |
| resolution (`imgsz`) | `192`, `320`, `512` |
| seed | `0`, `1`, `2` |

**Naming:** `sensei_<arm>_<imgsz>_s<seed>.pt` (matches the figure/CSV artifacts, e.g.
`confusion_base_320_s0.pdf`). The deployed point is `sensei_base_320_s*` (plain baseline,
no class re-balancing). The INT8 deployable is `sensei_*int8*`.

`SHA256SUMS.txt` (in the release) pins every file.

## Fetch (reproduce from a clone)

```bash
bash scripts/fetch_models.sh                 # -> models/, checksums verified
make evaluate WEIGHTS=models/sensei_base_320_s0.pt   # mAP/counting on the LOCKED test split
```

Map to the paper: Table II / accuracy–energy figure = `sensei_base_{192,320,512}_s{0,1,2}`;
loss ablation + confusion matrices = the `focal`, `focal_noiw`, `nwd` recipes at 320 px.

## Publish (maintainer, where the weights live)

On the cluster (`$RUNS`), rename each run's `best.pt` to the canonical scheme above, collect
them in one dir, then:

```bash
bash scripts/publish_models.sh <DIR_OF_RENAMED_CKPTS>   # needs `gh auth login`
```

The script refuses to publish an incomplete set (it checks all 36 names) and uploads
`SHA256SUMS.txt` alongside the weights.
