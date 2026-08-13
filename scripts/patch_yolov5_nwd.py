#!/usr/bin/env python3
"""Idempotently patch a classic ultralytics/yolov5 clone to support an NWD box-loss blend.

Adds a Normalized Wasserstein Distance term to ComputeLoss, controlled entirely by two hyp
keys so the SAME patched repo serves all sweep arms (no per-task editing, no race):
  * nwd_ratio (default 0.0)  -> blend weight; 0 reproduces stock CIoU exactly (arms A,B).
  * nwd_c     (default 5.0)  -> Gaussian distance normaliser, in grid (feature-map) units.

NWD (Wang et al. 2021) models each xywh box as a 2-D Gaussian N(mu, diag((w/2)^2,(h/2)^2)):
  W2^2 = dcx^2 + dcy^2 + ((w_p-w_g)^2 + (h_p-h_g)^2)/4 ;  NWD = exp(-sqrt(W2^2)/C)
and the box loss becomes (1-iou)*(1-r) + (1-NWD)*r. Robust for tiny objects where IoU is
numerically unstable. Run ONCE after cloning yolov5:  python patch_yolov5_nwd.py $YOLOV5_REPO
"""
import sys
from pathlib import Path

MARKER = "NWD_PATCH"
STOCK = (
    "                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  "
    "# iou(prediction, target)\n"
    "                lbox += (1.0 - iou).mean()  # iou loss\n"
)
PATCHED = (
    "                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  "
    "# iou(prediction, target)\n"
    "                _r = float(self.hyp.get('nwd_ratio', 0.0))  # " + MARKER + "\n"
    "                if _r > 0.0:\n"
    "                    _C = float(self.hyp.get('nwd_c', 5.0))\n"
    "                    _wd2 = ((pbox[:, 0] - tbox[i][:, 0]) ** 2 + (pbox[:, 1] - tbox[i][:, 1]) ** 2\n"
    "                            + ((pbox[:, 2] - tbox[i][:, 2]) ** 2 + (pbox[:, 3] - tbox[i][:, 3]) ** 2) / 4.0)\n"
    "                    _nwd = torch.exp(-torch.sqrt(_wd2.clamp(min=1e-9)) / _C)\n"
    "                    lbox += ((1.0 - iou) * (1.0 - _r) + (1.0 - _nwd) * _r).mean()\n"
    "                else:\n"
    "                    lbox += (1.0 - iou).mean()  # iou loss\n"
)


def main():
    repo = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    loss = repo / "utils" / "loss.py"
    if not loss.is_file():
        sys.exit(f"[patch] {loss} not found — pass the yolov5 repo root")
    txt = loss.read_text()
    if MARKER in txt:
        print(f"[patch] already patched: {loss}")
        return
    if STOCK not in txt:
        sys.exit("[patch] stock box-loss lines not found — yolov5 version differs; patch by hand")
    loss.write_text(txt.replace(STOCK, PATCHED))
    print(f"[patch] NWD blend added to {loss} (enable via hyp nwd_ratio>0)")


if __name__ == "__main__":
    main()
