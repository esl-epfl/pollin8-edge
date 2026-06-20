"""Training-grounded alternative-architecture study for insect monitoring on GAP9.

All candidates are CENTROID detectors (per-class heatmaps), FOMO's native form, trained on
the same insect tiles and scored with the same centre-matched F1 / counting metrics the
monitoring pipeline already uses. Deployability is profiled from the real torch modules.
Heavy parts (torch) run on the cluster; the geometry (heatmap encode/decode) is pure and
unit-tested here.
"""
