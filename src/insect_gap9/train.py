"""Fine-tune the pico YOLOv5 on the 9-class benchmark.

Thin wrapper over Ultralytics so there is no bespoke training loop to maintain.
Heavy imports are lazy so the rest of the package imports without torch.
"""
from __future__ import annotations
import argparse, os, random
from pathlib import Path

# Avoid Weights & Biases prompts/errors in Colab/CI (Ultralytics auto-hooks W&B if installed).
os.environ.setdefault("WANDB_MODE", "disabled")


def set_seed(seed: int = 0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np, torch
        np.random.seed(seed); torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def plateau_stopper(patience: int, min_delta: float):
    """Stricter early-stop callback (Ultralytics `on_fit_epoch_end`).

    Ultralytics' built-in EarlyStopping resets its counter on ANY strictly-better
    fitness, so marginal/jittery upticks (e.g. +0.0003) keep training alive long after
    the model has plateaued. Here a gain below `min_delta` does NOT count as improvement:
    we stop after `patience` epochs without a >min_delta gain in fitness
    (fitness = 0.1*mAP@0.5 + 0.9*mAP@0.5:0.95). The genuine best checkpoint is still
    saved by Ultralytics' own (strict) best-tracking, so accuracy is not sacrificed.
    """
    st = {"best": None, "best_epoch": 0}

    def _cb(trainer):
        f = getattr(trainer, "fitness", None)
        if f is None:
            return
        e = int(trainer.epoch)
        if st["best"] is None or f > st["best"] + min_delta:   # significant improvement
            st["best"], st["best_epoch"] = float(f), e
        elif e - st["best_epoch"] >= patience:                 # plateaued
            trainer.stop = True
            print(f"[early-stop] no >{min_delta:g} fitness gain for {patience} epochs "
                  f"(best @ epoch {st['best_epoch']}, fitness={st['best']:.4f}) "
                  f"-> stopping at epoch {e}")
    return _cb


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="configs/insects1201.yaml")
    ap.add_argument("--cfg",  default="configs/yolov5p_insect.yaml")
    ap.add_argument("--hyp",  default="configs/train.yaml")
    ap.add_argument("--weights", default="yolov5n.pt",
                    help="public init; or a Bjerge YOLOv5 model to distil/fine-tune from")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="pico_insect")
    ap.add_argument("--cache", default="ram", choices=["ram", "disk", "none"])
    ap.add_argument("--epochs", type=int, default=None,
                    help="override hyp epochs (e.g. 1 for a smoke test)")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="override hyp imgsz (used by the resolution sweep)")
    ap.add_argument("--batch", type=int, default=None,
                    help="override hyp batch (lower it at high imgsz to avoid OOM)")
    ap.add_argument("--time-hours", type=float, default=None,
                    help="hard wall-clock cap per run (safety net for overnight sweeps); "
                         "best.pt is kept. Omit to rely on early stopping alone.")
    ap.add_argument("--resume", action="store_true",
                    help="resume training from the --weights checkpoint")
    a = ap.parse_args(argv)

    import yaml
    hyp = yaml.safe_load(Path(a.hyp).read_text())
    if a.epochs is not None:
        hyp["epochs"] = a.epochs
    if a.imgsz is not None:
        hyp["imgsz"] = a.imgsz
    if a.batch is not None:
        hyp["batch"] = a.batch
    set_seed(hyp.get("seed", 0))

    from ultralytics import YOLO, settings  # lazy
    settings.update({"wandb": False})  # disable W&B integration (crashes on path-like project names)
    # Treat empty / missing / non-file --weights as "train from scratch".
    # NB: Path("").exists() is True (it resolves to "."), so check is_file() explicitly.
    init_from = bool(a.weights) and Path(a.weights).is_file()
    if init_from:
        print(f"[train] initialising from {a.weights}")
        model = YOLO(a.cfg).load(a.weights)
    else:
        print("[train] training from scratch (no usable checkpoint)")
        model = YOLO(a.cfg)

    # Stricter plateau early-stop (min_delta ignores marginal/jittery gains).
    min_delta = float(hyp.get("min_delta", 0.0))
    if min_delta > 0:
        model.add_callback("on_fit_epoch_end",
                           plateau_stopper(hyp.get("patience", 100), min_delta))

    train_kw = dict(
        data=a.data, imgsz=hyp["imgsz"], epochs=hyp["epochs"], batch=hyp["batch"],
        optimizer=hyp["optimizer"], lr0=hyp["lr0"], lrf=hyp["lrf"],
        momentum=hyp["momentum"], weight_decay=hyp["weight_decay"],
        warmup_epochs=hyp["warmup_epochs"], freeze=hyp["freeze"],
        patience=hyp.get("patience", 100),   # built-in stopper (strict); plateau cb is stricter
        workers=hyp.get("workers", 8),       # keep low on small MIG slices to avoid OOM
        mosaic=hyp["mosaic"], fliplr=hyp["fliplr"],
        hsv_h=hyp["hsv_h"], hsv_s=hyp["hsv_s"], hsv_v=hyp["hsv_v"],
        seed=hyp.get("seed", 0),
        cache=(None if a.cache == "none" else a.cache),
        project=a.project, name=a.name, resume=a.resume,
    )
    if a.time_hours is not None:
        train_kw["time"] = a.time_hours      # Ultralytics hard wall-clock cap (keeps best.pt)
    model.train(**train_kw)
    print(f"[done] best weights under {a.project}/{a.name}/weights/best.pt")


if __name__ == "__main__":
    main()
