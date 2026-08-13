"""Knowledge distillation via teacher pseudo-labels (response-based, framework-agnostic).

The Bjerge YOLOv5m6@1280 teacher (Zenodo) is far stronger than our pico student but does
not fit GAP9. We transfer its knowledge offline: run the teacher to produce detections,
and use them as training labels --- especially for otherwise-unlabelled GBIF imagery, and
to densify supervision on the camera-trap tiles. The student then trains normally.

This response-based scheme works across architectures (anchor-based teacher -> anchor-free
student) because it operates on boxes, not internal logits. The label-merge logic is pure
and unit-tested; the CLI adds Ultralytics only to run the teacher.

(Feature/logit KD with a custom loss is a documented future improvement.)
"""
from __future__ import annotations
import argparse, time, warnings
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}


# ---- pure label algebra ------------------------------------------------------
def _iou(a, b):
    (_, ax, ay, aw, ah), (_, bx, by, bw, bh) = a, b
    ax1, ay1, ax2, ay2 = ax - aw / 2, ay - ah / 2, ax + aw / 2, ay + ah / 2
    bx1, by1, bx2, by2 = bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(ix2 - ix1, 0), max(iy2 - iy1, 0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def merge_labels(gt, teacher, iou_thr=0.5):
    """Union of GT and teacher boxes; a teacher box overlapping any GT box
    (IoU >= iou_thr) is dropped (GT wins). Keeps novel teacher detections."""
    merged = list(gt)
    for t in teacher:
        if all(_iou(t, g) < iou_thr for g in gt):
            merged.append(t)
    return merged


def _read(p: Path):
    if not p.exists():
        return []
    out = []
    for ln in p.read_text().splitlines():
        if ln.strip():
            c, cx, cy, w, h = ln.split()[:5]
            out.append((int(float(c)), float(cx), float(cy), float(w), float(h)))
    return out


def _write(p: Path, boxes):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in boxes))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", default="data/raw/YOLOv5models/insect1201-bestF1-1280v5m6.pt",
                    help="Bjerge YOLOv5m6 teacher checkpoint (from Zenodo YOLOv5models.zip)")
    ap.add_argument("--yolov5-repo", default=None,
                    help="path to a LOCAL clone of ultralytics/yolov5 (use on offline compute "
                         "nodes; pre-clone it on the login node). Omit to fetch online via torch.hub.")
    ap.add_argument("--images", required=True, help="image dir to pseudo-label")
    ap.add_argument("--out", required=True, help="output labels dir (co-located names)")
    ap.add_argument("--imgsz", type=int, default=1280, help="teacher inference resolution")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--merge-gt", action="store_true",
                    help="merge with existing GT labels next to each image")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=8,
                    help="teacher inference batch size (lower if the teacher OOMs at 1280px)")
    ap.add_argument("--limit", type=int, default=0,
                    help="pseudo-label only the first N images (0 = all); use to subsample")
    ap.add_argument("--overwrite", action="store_true",
                    help="ignore the resume manifest and re-label every image")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=FutureWarning)  # silence legacy yolov5 amp warning

    if not Path(a.teacher).is_file():
        print(f"[skip] teacher checkpoint not found: {a.teacher}\n"
              f"       distillation is OPTIONAL. To use it, download the Zenodo YOLOv5models.zip:\n"
              f"       python -m insect_gap9.download_data --dest data/raw --only YOLOv5models.zip\n"
              f"       (unzip it, then pass --teacher <path to insect1201-bestF1-1280v5m6.pt>).")
        return

    teacher = _load_teacher(a.teacher, a.yolov5_repo, a.conf, a.iou)
    imgs = sorted(p for p in Path(a.images).rglob("*") if p.suffix.lower() in IMG_EXTS)
    if a.limit:
        imgs = imgs[:a.limit]

    # Resume manifest: stems already pseudo-labelled. Works even when out==images dir
    # (where the .txt files are the GT being merged, so output-existence can't be a marker).
    manifest = out / ".distill_done.txt"
    done = set(manifest.read_text().split()) if (manifest.exists() and not a.overwrite) else set()
    todo = [p for p in imgs if p.stem not in done]
    print(f"[distill] {len(imgs)} images | {len(done)} already done | {len(todo)} to do "
          f"(batch={a.batch}, imgsz={a.imgsz})", flush=True)

    n_box, t0 = 0, time.time()
    with manifest.open("a") as mf:
        for i in range(0, len(todo), a.batch):
            chunk = todo[i:i + a.batch]
            # AutoShape accepts a list -> batched; results.xywhn[k] rows are
            # [xc, yc, w, h, conf, cls] (normalised to each image's own size).
            results = teacher([str(p) for p in chunk], size=a.imgsz)
            for p, xywhn in zip(chunk, results.xywhn):
                pseudo = [(int(cls), xc, yc, w, h)
                          for xc, yc, w, h, _conf, cls in xywhn.tolist()]
                boxes = (merge_labels(_read(p.with_suffix(".txt")), pseudo, a.iou)
                         if a.merge_gt else pseudo)
                _write(out / f"{p.stem}.txt", boxes)
                n_box += len(boxes)
                mf.write(p.stem + "\n")
            mf.flush()
            n_seen = i + len(chunk)
            if i // a.batch % 20 == 0 or n_seen == len(todo):   # progress every ~20 batches
                rate = n_seen / max(time.time() - t0, 1e-9)
                eta_min = (len(todo) - n_seen) / max(rate, 1e-9) / 60
                print(f"[distill] {n_seen}/{len(todo)} imgs | {rate:.1f} img/s | "
                      f"ETA {eta_min:.1f} min", flush=True)
    print(f"[done] pseudo-labelled {len(todo)} new images ({n_box} boxes) -> {out}")


def _load_teacher(path, repo, conf, iou):
    """Load the Bjerge YOLOv5m6 teacher.

    These are legacy YOLOv5 v7.0 (ultralytics/yolov5) checkpoints, which the modern
    Ultralytics-8 `YOLO()` refuses to load ('NOT forwards compatible'). We load them
    through their original repo via torch.hub. On an OFFLINE compute node, pass
    --yolov5-repo <local clone> (pre-clone on the login node; torch.hub then reads it
    with source='local' and needs no network).
    """
    import torch
    if repo:
        model = torch.hub.load(repo, "custom", path=path, source="local", verbose=False)
    else:
        model = torch.hub.load("ultralytics/yolov5", "custom", path=path, verbose=False)
    model.conf = conf   # confidence threshold
    model.iou = iou     # NMS IoU threshold
    return model


if __name__ == "__main__":
    main()
