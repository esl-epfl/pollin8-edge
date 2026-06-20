"""Build a tiled YOLO dataset so small insects are well-sized at low input resolution.

Bjerge camera-trap frames are large and insects occupy few pixels; naively resizing a
whole frame to 160 px makes them sub-pixel and the detector learns nothing. We instead
crop tiles in which each insect is a sizeable fraction of the tile:

  * object mode : one tile centred (with jitter) on each annotated insect.
  * grid mode   : a sliding TxT window with overlap (DSORT-MCU style), used at
                  inference and for background coverage.

Geometry is pure and unit-tested; the CLI adds PIL only to crop/save. Output is a
co-located images+labels tree plus a ready-to-train data YAML.
"""
from __future__ import annotations
import argparse, random
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}


# ---- pure geometry (no I/O) --------------------------------------------------
def _to_px(b, W, H):
    cls, cx, cy, w, h = b
    return cls, (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H


def boxes_in_tile(boxes, W, H, tile, vis_thr=0.3):
    """Remap normalized boxes into a tile rect (tx0,ty0,tx1,ty1) in pixels.
    Keep a box if >= vis_thr of its area is inside the tile; clip partials."""
    tx0, ty0, tx1, ty1 = tile
    tw, th = tx1 - tx0, ty1 - ty0
    out = []
    for b in boxes:
        cls, x1, y1, x2, y2 = _to_px(b, W, H)
        area = max((x2 - x1) * (y2 - y1), 1e-9)
        ix1, iy1 = max(x1, tx0), max(y1, ty0)
        ix2, iy2 = min(x2, tx1), min(y2, ty1)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        if (ix2 - ix1) * (iy2 - iy1) / area < vis_thr:
            continue
        ncx = ((ix1 + ix2) / 2 - tx0) / tw
        ncy = ((iy1 + iy2) / 2 - ty0) / th
        nw, nh = (ix2 - ix1) / tw, (iy2 - iy1) / th
        out.append((cls, ncx, ncy, nw, nh))
    return out


def _clamp_tile(cx, cy, T, W, H):
    """Place a TxT tile centred at (cx,cy), shifted to stay inside the image."""
    t = min(T, W, H)
    x0 = min(max(cx - t / 2, 0), max(W - t, 0))
    y0 = min(max(cy - t / 2, 0), max(H - t, 0))
    return (x0, y0, x0 + t, y0 + t)


def object_tiles(boxes, W, H, T, jitter=0.2, rng=random):
    tiles = []
    for b in boxes:
        _, cx, cy, _, _ = b
        jx = (rng.random() * 2 - 1) * jitter * T
        jy = (rng.random() * 2 - 1) * jitter * T
        tiles.append(_clamp_tile(cx * W + jx, cy * H + jy, T, W, H))
    return tiles


def grid_tiles(W, H, T, stride):
    t = min(T, W, H)
    xs = list(range(0, max(W - t, 0) + 1, stride)) or [0]
    ys = list(range(0, max(H - t, 0) + 1, stride)) or [0]
    if xs[-1] != max(W - t, 0):
        xs.append(max(W - t, 0))
    if ys[-1] != max(H - t, 0):
        ys.append(max(H - t, 0))
    return [(x, y, x + t, y + t) for y in ys for x in xs]


# ---- I/O CLI -----------------------------------------------------------------
def _read_label(p: Path):
    if not p.exists():
        return []
    out = []
    for ln in p.read_text().splitlines():
        if ln.strip():
            c, cx, cy, w, h = ln.split()[:5]
            out.append((int(float(c)), float(cx), float(cy), float(w), float(h)))
    return out


def _write_label(p: Path, boxes):
    p.write_text("".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in boxes))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="split dir with co-located images+labels")
    ap.add_argument("--out", required=True, help="output tiled split dir")
    ap.add_argument("--mode", choices=["object", "grid"], default="object")
    ap.add_argument("--tile", type=int, default=320, help="tile size in px")
    ap.add_argument("--stride", type=int, default=256, help="grid stride in px")
    ap.add_argument("--vis", type=float, default=0.3, help="min visible box fraction to keep")
    ap.add_argument("--bg-frac", type=float, default=0.1, help="fraction of empty tiles to keep")
    ap.add_argument("--jitter", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    from PIL import Image  # lazy
    rng = random.Random(a.seed)
    src, out = Path(a.src), Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    imgs = [p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS]
    n_tiles = 0
    for ip in imgs:
        boxes = _read_label(ip.with_suffix(".txt"))
        im = Image.open(ip); W, H = im.size
        tiles = (object_tiles(boxes, W, H, a.tile, a.jitter, rng) if a.mode == "object"
                 else grid_tiles(W, H, a.tile, a.stride))
        for k, t in enumerate(tiles):
            tb = boxes_in_tile(boxes, W, H, t, a.vis)
            if not tb and rng.random() > a.bg_frac:
                continue                      # subsample empty tiles
            crop = im.crop(tuple(int(v) for v in t))
            stem = f"{ip.stem}_{a.mode}{k}"
            crop.save(out / "images" / f"{stem}.jpg")
            _write_label(out / "labels" / f"{stem}.txt", tb)
            n_tiles += 1
    print(f"[done] {a.mode} tiling: {len(imgs)} images -> {n_tiles} tiles in {out}")


if __name__ == "__main__":
    main()
