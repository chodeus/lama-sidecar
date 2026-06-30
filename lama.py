"""LaMa inpainting with adaptive per-region reconstruction.

Loads the canonical ``big-lama`` TorchScript model (the same frozen artifact
IOPaint used). LaMa is resolution-robust (Fourier-convolution architecture), but
on a *large* contiguous hole the interior sits beyond the model's receptive
field, so it averages to a smooth blur instead of reconstructing texture. The
classic example is a poster's title wordmark over clouds: at native resolution
the band comes back as a hazy smear.

To fix that without softening small text, the mask is split into connected
regions and each is inpainted in its own crop at an *adaptive* scale:

* small region (a date, a small caption) -> native resolution, razor-sharp;
* large region (a wide logo band) -> downscaled so the hole fits the receptive
  field, inpainted, then upscaled — recovering background texture.

Only the masked pixels of each region are composited back onto a pristine copy,
so untouched artwork is never resampled. Set ``LAMA_TARGET_RES=0`` to disable the
region logic and run a single full-resolution pass (the pre-1.3 behaviour).

Preprocessing mirrors IOPaint's reference pipeline: normalise to [0,1], binarise
the mask (white = region to erase), pad height/width up to a multiple of 8 for
the convolutions, run, scale back to [0,255], and crop the padding away.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageFilter

PAD_MODULO = 8

# Above this many separate blobs the O(n^2) grouping isn't worth it (and the mask
# is probably noise) — fall back to one global-resize pass over the whole extent.
MAX_COMPONENTS = 2000


def _ceil_modulo(x: int, mod: int) -> int:
    return x if x % mod == 0 else (x // mod + 1) * mod


def _norm_img(np_img: np.ndarray) -> np.ndarray:
    """HWC (or HW) uint8 -> CHW float32 in [0, 1]."""
    if np_img.ndim == 2:
        np_img = np_img[:, :, np.newaxis]
    np_img = np.transpose(np_img, (2, 0, 1))
    return np_img.astype("float32") / 255.0


def _pad_to_modulo(img: np.ndarray, mod: int) -> np.ndarray:
    """Pad a CHW array on the bottom/right so H and W are multiples of ``mod``."""
    _, h, w = img.shape
    out_h, out_w = _ceil_modulo(h, mod), _ceil_modulo(w, mod)
    return np.pad(img, ((0, 0), (0, out_h - h), (0, out_w - w)), mode="symmetric")


def _connected_components(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """8-connected components of a 2D bool mask -> list of (y0, x0, y1, x1) bboxes
    (y1/x1 exclusive). Run-based two-pass labelling: each row's True segments are
    union-found against the previous row's. No scipy/cv2 needed; cost scales with
    the number of runs, which is tiny for text-shaped masks."""
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


def _group_regions(boxes, pad_frac, min_pad, w, h):
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


class LamaModel:
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        target_res: int = 1024,
        region_pad: float = 0.5,
        mask_dilate: int = 0,
        mask_feather: int = 0,
    ) -> None:
        self.device = torch.device(device)
        # TorchScript model is self-contained — no model class needed.
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        # <=0 disables region splitting and runs one full-resolution pass.
        self.target_res = target_res
        self.region_pad = region_pad
        # Grow the hole by N px before inpainting so a logo's anti-aliased fringe
        # and soft glow (which a tight mask leaves outside) get erased too — without
        # this they survive as a "ghost" outline. Feather only softens the composite
        # seam; the model still fills the hard, dilated hole underneath.
        self.mask_dilate = mask_dilate
        self.mask_feather = mask_feather

    def _dilate(self, mask: Image.Image) -> Image.Image:
        if self.mask_dilate <= 0:
            return mask
        m = mask.point(lambda v: 255 if v > 127 else 0)
        for _ in range(self.mask_dilate):  # iterated 3x3 max = N-px 8-conn dilation
            m = m.filter(ImageFilter.MaxFilter(3))
        return m

    @torch.no_grad()
    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        mask = mask.convert("L")
        if mask.size != image.size:
            # NEAREST keeps the mask strictly binary after resizing.
            mask = mask.resize(image.size, Image.NEAREST)

        mask = self._dilate(mask)

        if self.target_res and self.target_res > 0:
            return self._inpaint_regions(image, mask)
        return self._inpaint_full(image, mask)

    def _inpaint_regions(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        w, h = image.size
        mask_bool = np.array(mask) > 127
        if not mask_bool.any():
            return image.copy()

        boxes = _connected_components(mask_bool)
        if len(boxes) > MAX_COMPONENTS:
            arr = np.array(boxes)
            regions = [(
                max(0, int(arr[:, 0].min()) - 8), max(0, int(arr[:, 1].min()) - 8),
                min(h, int(arr[:, 2].max()) + 8), min(w, int(arr[:, 3].max()) + 8),
            )]
        else:
            regions = _group_regions(boxes, self.region_pad, 8, w, h)

        out = image.copy()
        for y0, x0, y1, x1 in regions:
            crop_img = image.crop((x0, y0, x1, y1))
            crop_mask = mask.crop((x0, y0, x1, y1))
            cw, ch = crop_img.size
            scale = min(1.0, self.target_res / max(cw, ch))
            res = self._inpaint_crop(crop_img, crop_mask, scale)
            # Feather only the composite alpha for a soft seam; the model already
            # received the hard mask, so the hole is fully filled underneath.
            paste_mask = crop_mask
            if self.mask_feather > 0:
                paste_mask = crop_mask.filter(ImageFilter.GaussianBlur(self.mask_feather))
            out.paste(res, (x0, y0), paste_mask)
        return out

    def _inpaint_crop(
        self, crop_img: Image.Image, crop_mask: Image.Image, scale: float
    ) -> Image.Image:
        if scale >= 0.999:
            return self._inpaint_full(crop_img, crop_mask)
        cw, ch = crop_img.size
        sw, sh = max(PAD_MODULO, round(cw * scale)), max(PAD_MODULO, round(ch * scale))
        small_img = crop_img.resize((sw, sh), Image.LANCZOS)
        small_mask = crop_mask.resize((sw, sh), Image.NEAREST)
        res = self._inpaint_full(small_img, small_mask)
        return res.resize((cw, ch), Image.LANCZOS)

    def _inpaint_full(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        orig_w, orig_h = image.size

        img = _norm_img(np.array(image))          # 3 x H x W, [0,1]
        msk = _norm_img(np.array(mask))           # 1 x H x W, [0,1]
        msk = (msk > 0) * 1.0                      # binarise: white(255) -> 1

        img = _pad_to_modulo(img, PAD_MODULO)
        msk = _pad_to_modulo(msk, PAD_MODULO)

        img_t = torch.from_numpy(img).unsqueeze(0).to(self.device)
        msk_t = torch.from_numpy(msk).unsqueeze(0).float().to(self.device)

        out = self.model(img_t, msk_t)             # 1 x 3 x Hp x Wp, [0,1]
        out = out[0].permute(1, 2, 0).detach().cpu().numpy()
        out = np.clip(out * 255, 0, 255).astype("uint8")
        out = out[:orig_h, :orig_w]                # drop the modulo padding
        return Image.fromarray(out, mode="RGB")
