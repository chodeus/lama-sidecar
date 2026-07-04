"""Pure mask/region geometry for the sidecar — no torch, unit-testable.

Everything here operates on 2D bool arrays (True = hole) or component bboxes
(y0, x0, y1, x1) with exclusive ends. The model-specific pipeline (normalise,
pad-to-modulo, forward pass) stays in lama.py.
"""

from __future__ import annotations

import numpy as np


def count_runs(mask: np.ndarray) -> int:
    """Number of horizontal True-runs — a cheap upper bound on the component
    count, used to bail out of the labeller before it gets expensive."""
    if mask.size == 0:
        return 0
    left = np.zeros_like(mask)
    left[:, 1:] = mask[:, :-1]
    return int(np.count_nonzero(mask & ~left))


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    """8-connected dilation by ``px`` (iterated 3x3 max). Bit-identical to
    PIL's iterated MaxFilter(3) on a binary mask, ~27x faster on poster-sized
    arrays."""
    if px <= 0:
        return mask
    m = mask
    for _ in range(px):
        p = np.pad(m, 1)
        m = (
            p[:-2, :-2] | p[:-2, 1:-1] | p[:-2, 2:]
            | p[1:-1, :-2] | p[1:-1, 1:-1] | p[1:-1, 2:]
            | p[2:, :-2] | p[2:, 1:-1] | p[2:, 2:]
        )
    return m


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(y0, x0, y1, x1) extent of True pixels, or None for an empty mask."""
    ys = np.flatnonzero(mask.any(axis=1))
    if ys.size == 0:
        return None
    xs = np.flatnonzero(mask.any(axis=0))
    return int(ys[0]), int(xs[0]), int(ys[-1]) + 1, int(xs[-1]) + 1


def hole_scale(hole_w: int, hole_h: int, long_res: int, thick_res: int) -> float:
    """Downscale factor for a region, derived from the HOLE, not the crop.

    LaMa fills from surrounding context, so what matters is either fitting the
    whole hole in the receptive field (long edge <= long_res) or keeping the
    interior close enough to context (thickness <= thick_res). Downscale only
    as much as the *easier* of the two demands — thin text bands stay at native
    resolution no matter how wide they are.
    """
    thick = max(1, min(hole_w, hole_h))
    longe = max(1, hole_w, hole_h)
    if thick <= thick_res:
        return 1.0
    return min(1.0, max(long_res / longe, thick_res / thick))


def connected_components(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """8-connected components of a 2D bool mask -> list of (y0, x0, y1, x1) bboxes
    (y1/x1 exclusive). Run-based two-pass labelling: each row's True segments are
    union-found against the previous row's. No scipy/cv2 needed; cost scales with
    the number of runs — callers must pre-bound that with count_runs()."""
    h, w = mask.shape
    parent: dict[int, int] = {}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    next_id = 0
    prev_runs: list[tuple[int, int, int]] = []  # (col_start, col_end_inclusive, label)
    all_runs: list[tuple[int, int, int, int]] = []  # (row, col_start, col_end_incl, label)

    for y in range(h):
        row = mask[y]
        idx = np.flatnonzero(row)
        if idx.size == 0:
            prev_runs = []
            continue
        # Split the True columns into maximal consecutive runs.
        breaks = np.flatnonzero(np.diff(idx) > 1)
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks, [idx.size - 1]))
        cur_runs: list[tuple[int, int, int]] = []
        for s, e in zip(starts, ends):
            cs, ce = int(idx[s]), int(idx[e])
            lbl = next_id
            next_id += 1
            parent[lbl] = lbl
            for ps, pe, plbl in prev_runs:  # 8-connectivity: overlap within 1 col
                if pe >= cs - 1 and ps <= ce + 1:
                    union(lbl, plbl)
            cur_runs.append((cs, ce, lbl))
            all_runs.append((y, cs, ce, lbl))
        prev_runs = cur_runs

    boxes: dict[int, list[int]] = {}
    for y, cs, ce, lbl in all_runs:
        r = find(lbl)
        b = boxes.get(r)
        if b is None:
            boxes[r] = [y, cs, y + 1, ce + 1]
        else:
            b[0] = min(b[0], y)
            b[1] = min(b[1], cs)
            b[2] = max(b[2], y + 1)
            b[3] = max(b[3], ce + 1)
    return [tuple(b) for b in boxes.values()]  # type: ignore[misc]


def group_regions(boxes, pad_frac, min_pad, w, h):
    """Pad each component box, union any that overlap, and return clamped crop
    boxes (y0, x0, y1, x1). Padding gives the inpainter valid context around a
    hole and merges a word's glyphs into one crop; far-apart text stays separate."""
    n = len(boxes)
    par = list(range(n))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            par[rb] = ra

    exp = []
    for y0, x0, y1, x1 in boxes:
        p = max(min_pad, int(pad_frac * max(y1 - y0, x1 - x0)))
        exp.append((y0 - p, x0 - p, y1 + p, x1 + p))

    for i in range(n):
        ay0, ax0, ay1, ax1 = exp[i]
        for j in range(i + 1, n):
            by0, bx0, by1, bx1 = exp[j]
            if ay0 < by1 and by0 < ay1 and ax0 < bx1 and bx0 < ax1:
                union(i, j)

    regions: dict[int, list[int]] = {}
    for i in range(n):
        r = find(i)
        ey0, ex0, ey1, ex1 = exp[i]
        g = regions.get(r)
        if g is None:
            regions[r] = [ey0, ex0, ey1, ex1]
        else:
            g[0] = min(g[0], ey0)
            g[1] = min(g[1], ex0)
            g[2] = max(g[2], ey1)
            g[3] = max(g[3], ex1)

    out = []
    for g in regions.values():
        out.append((max(0, g[0]), max(0, g[1]), min(h, g[2]), min(w, g[3])))
    return out
