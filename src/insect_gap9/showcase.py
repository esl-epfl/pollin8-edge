"""Qualitative AI result samples: a grid of detections on test images.

Colab-only (needs the trained model + images). Saves an annotated grid to
results/figures/detections_grid.png for the paper's qualitative figure. No-op with a
clear message if Ultralytics/Torch or the data are unavailable.
"""
from __future__ import annotations
import argparse
from pathlib import Path

OUTDIR = Path("results/figures")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="runs/pico_insect/weights/best.pt")
    ap.add_argument("--images", default="data/insects1201/test")
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--imgsz", type=int, default=160)
    a = ap.parse_args(argv)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    try:
        import math, random
        from ultralytics import YOLO
        import matplotlib.pyplot as plt
        imgs = [p for p in Path(a.images).rglob("*.jpg")]
        if not imgs:
            raise FileNotFoundError(f"no .jpg under {a.images}")
        random.seed(0)
        sample = random.sample(imgs, min(a.n, len(imgs)))
        model = YOLO(a.weights)
        res = model.predict(sample, imgsz=a.imgsz, verbose=False)
        cols = int(math.ceil(math.sqrt(len(res)))); rows = int(math.ceil(len(res) / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.0 * rows))
        for ax, r in zip(getattr(axs, "flat", [axs]), res):
            ax.imshow(r.plot()[:, :, ::-1]); ax.axis("off")
        for ax in getattr(axs, "flat", [axs])[len(res):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(OUTDIR / "detections_grid.png", dpi=200)
        print("[done] wrote", OUTDIR / "detections_grid.png")
    except Exception as e:
        print(f"[skip] qualitative showcase needs the model+images on Colab "
              f"({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
