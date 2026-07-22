"""LaMa inpainting with adaptive per-region reconstruction.

Loads the canonical ``big-lama`` TorchScript model (the same frozen artifact
IOPaint used). LaMa is resolution-robust (Fourier-convolution architecture), but
on a *large* contiguous hole the interior sits beyond the model's receptive
field, so it averages to a smooth blur instead of reconstructing texture. The
classic example is a poster's title wordmark over clouds: at native resolution
the band comes back as a hazy smear.

To fix that without softening small text, the mask is split into connected
regions and each is inpainted in its own crop at an *adaptive* scale derived
from the HOLE geometry (see regions.hole_scale): a hole is fillable at native
resolution when it is thin enough for context to reach its interior, however
wide it is — so title wordmarks stay razor-sharp — and only genuinely bulky
holes are downscaled until they fit the receptive field, inpainted, then
upscaled.

Only the masked pixels of each region are composited back onto a pristine copy
(the paste alpha never exceeds the hard mask, so unmasked artwork is
bit-identical). Set ``LAMA_TARGET_RES=0`` for a single whole-frame region
instead of component splitting; the same composite rules apply.

Two further stages bracket the region passes (both default-on, per-request
overridable):

- **Snap** (before dilation): grow the mask over whole high-contrast strokes it
  already mostly covers (regions.snap_mask), so a brush that clips the last few
  px of a glyph doesn't leave a stub at the hole edge — LaMa anchors on such
  stubs and smears them across the fill, or keeps them outright.
- **Boundary refine** (after the region passes): any region filled below
  REFINE_BELOW scale gets a REFINE_RING-thick inner band of its hole
  re-inpainted at native resolution. The downscale-inpaint-upscale path leaves
  real edges that continue into the dilated hole (a chin, a box edge) repainted
  at 1/scale blur; the band is thin, so context reaches all of it, the stage-1
  fill supplies interior context, and the paste keeps the same
  never-exceed-the-hole alpha guarantee.

Preprocessing mirrors IOPaint's reference pipeline: normalise to [0,1], binarise
the mask once (white = region to erase), pad height/width up to a multiple of 8
for the convolutions, run, scale back to [0,255], and crop the padding away.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import torch
from PIL import Image, ImageChops, ImageFilter

from regions import (
    connected_components,
    count_runs,
    dilate,
    erode,
    group_regions,
    hole_scale,
    mask_bbox,
    snap_mask,
)

PAD_MODULO = 8

# Bail out of component splitting when the mask is speckle noise. count_runs is
# a cheap upper bound on the component count, but it counts per-ROW runs — a
# single solid 500px-tall rectangle contributes ~500 of them — so the threshold
# must be calibrated to runs, not components: real brush/detect masks land in
# the hundreds-to-thousands (a full-poster detect mask ~5k), labelling cost is
# still trivial well past that, and only genuine speckle trips this. (At 2000
# this fired on ordinary detect masks and collapsed them into one global-extent
# region inpainted at a catastrophic downscale.)
MAX_RUNS = 20_000

# Whole-frame mode (target_res == 0) still has to bound the model input; a
# 40 MP frame would need tens of GB of activations on CPU.
FULL_FRAME_RES = 2048

# For a native-resolution pass, context beyond the model's receptive-field
# reach contributes nothing — crops are tightened to hole + this margin, which
# roughly halves the compute on wide title bands with generous padding.
NATIVE_CTX = 384

# Seam colour matching compares the fill to the original in a ring this many
# pixels wide just outside the hole; below MIN_RING_PX samples the estimate is
# too noisy to trust.
SEAM_RING_PX = 6
SEAM_MIN_RING_PX = 64

# Boundary refine: regions filled below REFINE_BELOW scale get a
# REFINE_RING-thick inner band of their hole re-inpainted at native resolution
# (with REFINE_CTX context around it). 0.75 splits where the upscale blur
# turns visible (a ~0.72 title-band fill still shows a haze echo; ~0.8 doesn't)
# — mildly-scaled fills aren't worth an extra inference. REFINE_MAX_PX caps
# the refine crop fed to the model — beyond it the pass downscales just enough
# to fit, which still beats the stage-1 scale by construction or the pass is
# skipped.
REFINE_BELOW = 0.75
REFINE_RING = 96
REFINE_CTX = 192
REFINE_MAX_PX = 3_200_000


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


def to_rgb8(image: Image.Image) -> Image.Image:
    """Normalise any Pillow mode to 8-bit RGB without the convert() footguns:
    transparency is composited onto white (convert alone exposes stale RGB under
    transparent pixels), and 16/32-bit integer modes are rescaled (convert
    clips them)."""
    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "LA", "PA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    if image.mode.startswith("I;16") or image.mode == "I":
        arr = np.asarray(image, dtype=np.float32)
        peak = max(arr.max(), 1.0)
        scale = 255.0 / 65535.0 if peak > 255 else 1.0
        arr8 = np.clip(arr * scale, 0, 255).astype("uint8")
        return Image.fromarray(arr8).convert("RGB")
    return image.convert("RGB")


class LamaModel:
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        target_res: int = 1400,
        region_pad: float = 0.5,
        min_pad: int = 64,
        mask_dilate: int = 5,
        mask_feather: int = 2,
        hole_res: int = 512,
        hole_thick: int = 384,
        seam_match: bool = True,
        mask_snap: bool = True,
        refine: bool = True,
        torch_threads: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        # A single explicit intra-op pool: concurrency is serialised by the API
        # layer, so one inference may use every core without oversubscription.
        torch.set_num_threads(torch_threads or os.cpu_count() or 1)
        # TorchScript model is self-contained — no model class needed.
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        # <=0 switches to a single whole-frame region (bounded by FULL_FRAME_RES).
        self.target_res = target_res
        self.region_pad = region_pad
        self.min_pad = min_pad
        # Grow the hole by N px before inpainting so a logo's anti-aliased fringe
        # and soft glow (which a tight mask leaves outside) get erased too — without
        # this they survive as a "ghost" outline. Feather only softens the composite
        # seam *inward*; the paste alpha never exceeds the hard, dilated hole.
        self.mask_dilate = mask_dilate
        self.mask_feather = mask_feather
        self.hole_res = hole_res
        self.hole_thick = hole_thick
        self.seam_match = seam_match
        self.mask_snap = mask_snap
        self.refine = refine

    @torch.no_grad()
    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        dilate_px: int | None = None,
        feather_px: int | None = None,
        target_res: int | None = None,
        snap: bool | None = None,
        refine: bool | None = None,
    ) -> tuple[Image.Image, dict]:
        """Erase the white regions of ``mask`` from ``image``.

        Per-request overrides fall back to the instance defaults; returns the
        result plus a meta dict (region boxes, scales, timing) for the API's
        debug/observability surface.
        """
        t0 = time.perf_counter()
        dil = self.mask_dilate if dilate_px is None else dilate_px
        feather = self.mask_feather if feather_px is None else feather_px
        tres = self.target_res if target_res is None else target_res
        do_snap = self.mask_snap if snap is None else snap
        do_refine = self.refine if refine is None else refine

        image = to_rgb8(image)
        mask = mask.convert("L")
        if mask.size != image.size:
            # NEAREST keeps the mask strictly binary after resizing.
            mask = mask.resize(image.size, Image.NEAREST)

        # Binarise ONCE; every later stage (split, model hole, paste alpha)
        # sees the same binary mask, so a client's antialiased brush mask can't
        # make the model fill pixels the composite then ignores.
        mask_arr = np.asarray(mask) > 127
        meta = {
            "regions": [],
            "scales": [],
            "dilate": dil,
            "feather": feather,
            "target_res": tres,
            "snap_px": 0,
            "refine": [],
            "inference_ms": 0,
        }
        if not mask_arr.any():
            meta["inference_ms"] = int((time.perf_counter() - t0) * 1000)
            return image.copy(), meta

        if do_snap:
            mask_arr, meta["snap_px"] = snap_mask(np.asarray(image), mask_arr)
        mask_arr = dilate(mask_arr, dil)
        mask = Image.fromarray(mask_arr.astype("uint8") * 255)

        w, h = image.size
        full_frame = not tres or tres <= 0
        if full_frame:
            regions = [(0, 0, h, w)]
        elif count_runs(mask_arr) > MAX_RUNS:
            # Checked BEFORE the labeller so a speckled mask can't pin the
            # worker; see MAX_RUNS for the runs-vs-components calibration.
            regions = [self._global_extent(mask_arr, w, h)]
        else:
            boxes = connected_components(mask_arr)
            regions = group_regions(boxes, self.region_pad, self.min_pad, w, h)

        out = image.copy()
        low_scale: list[tuple[tuple[int, int, int, int], float]] = []
        for y0, x0, y1, x1 in regions:
            if not full_frame:
                hy0, hx0, hy1, hx1 = mask_bbox(mask_arr[y0:y1, x0:x1])
                s = hole_scale(hx1 - hx0, hy1 - hy0, self.hole_res, self.hole_thick)
                if s >= 0.999:
                    y0, x0, y1, x1 = (
                        max(y0, y0 + hy0 - NATIVE_CTX),
                        max(x0, x0 + hx0 - NATIVE_CTX),
                        min(y1, y0 + hy1 + NATIVE_CTX),
                        min(x1, x0 + hx1 + NATIVE_CTX),
                    )

            crop_img = image.crop((x0, y0, x1, y1))
            crop_mask = mask.crop((x0, y0, x1, y1))
            cw, ch = crop_img.size

            if full_frame:
                scale = min(1.0, FULL_FRAME_RES / max(cw, ch))
            else:
                scale = min(s, tres / max(cw, ch))

            res = self._inpaint_crop(crop_img, crop_mask, scale)
            if self.seam_match:
                res = self._match_seam(crop_img, mask_arr[y0:y1, x0:x1], res)

            # An upscaled fill is already ~1/scale px soft, so the seam needs at
            # least that much feather; the blur is clamped back under the hard
            # mask (ImageChops.darker = per-pixel min) so alpha never leaks
            # outside the dilated hole — unmasked pixels stay bit-identical.
            f = feather if scale >= 0.999 else max(feather, math.ceil(1 / scale))
            paste_mask = crop_mask
            if f > 0:
                blurred = crop_mask.filter(ImageFilter.GaussianBlur(f))
                paste_mask = ImageChops.darker(blurred, crop_mask)
            out.paste(res, (x0, y0), paste_mask)

            meta["regions"].append([int(y0), int(x0), int(y1), int(x1)])
            meta["scales"].append(round(scale, 3))
            if scale < REFINE_BELOW:
                low_scale.append(((int(y0), int(x0), int(y1), int(x1)), scale))

        if do_refine and low_scale:
            out = self._refine_boundary(out, mask_arr, low_scale, feather, meta)

        meta["inference_ms"] = int((time.perf_counter() - t0) * 1000)
        return out, meta

    def _global_extent(self, mask_arr: np.ndarray, w: int, h: int):
        y0, x0, y1, x1 = mask_bbox(mask_arr)
        p = self.min_pad
        return (max(0, y0 - p), max(0, x0 - p), min(h, y1 + p), min(w, x1 + p))

    def _refine_boundary(
        self,
        out: Image.Image,
        mask_arr: np.ndarray,
        low_scale: list[tuple[tuple[int, int, int, int], float]],
        feather: int,
        meta: dict,
    ) -> Image.Image:
        """Native-res re-synthesis of a thin inner band of every low-scale fill.

        One extra model pass over the union of those bands: the band is thin so
        context reaches all of it regardless of how bulky the original hole
        was, the stage-1 fill supplies the interior context, and the paste
        alpha stays clamped under the band (itself a subset of the dilated
        hole), so the only-masked-pixels-change contract holds.
        """
        h, w = mask_arr.shape
        for (y0, x0, y1, x1), s in low_scale:
            # Ring per region, on a padded bbox slice (padded so the hole never
            # touches the slice edge except at real image borders — erode
            # treats beyond-edge as hole, so an unpadded slice would drop the
            # band along the bbox sides). Per-region rings keep far-apart
            # regions from fusing into one poster-sized refine crop.
            sy0, sx0 = max(0, y0 - REFINE_RING - 1), max(0, x0 - REFINE_RING - 1)
            sy1, sx1 = min(h, y1 + REFINE_RING + 1), min(w, x1 + REFINE_RING + 1)
            sub = np.zeros((sy1 - sy0, sx1 - sx0), dtype=bool)
            sub[y0 - sy0 : y1 - sy0, x0 - sx0 : x1 - sx0] = mask_arr[y0:y1, x0:x1]
            ring_sub = sub & ~erode(sub, REFINE_RING)
            if not ring_sub.any():
                continue
            bb = mask_bbox(ring_sub)
            ry0 = max(0, sy0 + bb[0] - REFINE_CTX)
            rx0 = max(0, sx0 + bb[1] - REFINE_CTX)
            ry1 = min(h, sy0 + bb[2] + REFINE_CTX)
            rx1 = min(w, sx0 + bb[3] + REFINE_CTX)
            cw, ch = rx1 - rx0, ry1 - ry0
            scale = min(1.0, math.sqrt(REFINE_MAX_PX / (cw * ch)))
            # Below this the pass would re-fill at roughly the stage-1 scale
            # again — no crispness to gain for a whole extra inference.
            if scale <= s + 0.05:
                continue
            ring_crop = np.zeros((ch, cw), dtype=bool)
            iy0, ix0 = max(sy0, ry0), max(sx0, rx0)
            iy1, ix1 = min(sy1, ry1), min(sx1, rx1)
            ring_crop[iy0 - ry0 : iy1 - ry0, ix0 - rx0 : ix1 - rx0] = ring_sub[
                iy0 - sy0 : iy1 - sy0, ix0 - sx0 : ix1 - sx0
            ]
            crop_img = out.crop((rx0, ry0, rx1, ry1))
            crop_mask = Image.fromarray(ring_crop.astype("uint8") * 255)
            res = self._inpaint_crop(crop_img, crop_mask, scale)
            if self.seam_match:
                res = self._match_seam(crop_img, ring_crop, res)
            f = feather if scale >= 0.999 else max(feather, math.ceil(1 / scale))
            paste_mask = crop_mask
            if f > 0:
                blurred = crop_mask.filter(ImageFilter.GaussianBlur(f))
                paste_mask = ImageChops.darker(blurred, crop_mask)
            out.paste(res, (rx0, ry0), paste_mask)
            meta["refine"].append(
                {
                    "box": [int(ry0), int(rx0), int(ry1), int(rx1)],
                    "scale": round(scale, 3),
                }
            )
        return out

    def _inpaint_crop(
        self, crop_img: Image.Image, crop_mask: Image.Image, scale: float
    ) -> Image.Image:
        if scale >= 0.999:
            return self._inpaint_full(crop_img, crop_mask)
        cw, ch = crop_img.size
        sw, sh = max(PAD_MODULO, round(cw * scale)), max(PAD_MODULO, round(ch * scale))
        small_img = crop_img.resize((sw, sh), Image.LANCZOS)
        # BILINEAR + re-binarise at >0: any pixel with mask coverage stays a
        # hole, so strokes thinner than 1/scale can't vanish the way NEAREST
        # sampling drops them (which would paste the blurred text back).
        small_mask = crop_mask.resize((sw, sh), Image.BILINEAR).point(
            lambda v: 255 if v > 0 else 0
        )
        res = self._inpaint_full(small_img, small_mask)
        return res.resize((cw, ch), Image.LANCZOS)

    def _match_seam(
        self, crop_img: Image.Image, hole: np.ndarray, res: Image.Image
    ) -> Image.Image:
        """Per-channel gain/bias match of the fill to the surrounding original,
        estimated on a thin ring just outside the hole (where both images show
        the same content). Kills the slight tonal shift a downscale-inpaint-
        upscale fill carries onto smooth gradients; near-identity when the fill
        already matches, and clamped so a busy ring can't over-correct."""
        ring = dilate(hole, SEAM_RING_PX) & ~hole
        if int(ring.sum()) < SEAM_MIN_RING_PX:
            return res
        o = np.asarray(crop_img, dtype=np.float32)
        r = np.asarray(res, dtype=np.float32)
        matched = r.copy()
        for c in range(3):
            oc, rc = o[..., c][ring], r[..., c][ring]
            gain = float(np.clip(oc.std() / (rc.std() + 1e-6), 0.8, 1.25))
            bias = float(np.clip(oc.mean() - gain * rc.mean(), -20.0, 20.0))
            matched[..., c] = r[..., c] * gain + bias
        return Image.fromarray(np.clip(matched, 0, 255).astype("uint8"))

    def _inpaint_full(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        orig_w, orig_h = image.size

        img = _norm_img(np.array(image))  # 3 x H x W, [0,1]
        msk = _norm_img(np.array(mask))  # 1 x H x W, [0,1]
        msk = (msk > 0) * 1.0  # binarise: white(255) -> 1

        img = _pad_to_modulo(img, PAD_MODULO)
        msk = _pad_to_modulo(msk, PAD_MODULO)

        img_t = torch.from_numpy(img).unsqueeze(0).to(self.device)
        msk_t = torch.from_numpy(msk).unsqueeze(0).float().to(self.device)

        out = self.model(img_t, msk_t)  # 1 x 3 x Hp x Wp, [0,1]
        out = out[0].permute(1, 2, 0).detach().cpu().numpy()
        out = np.clip(out * 255, 0, 255).astype("uint8")
        out = out[:orig_h, :orig_w]  # drop the modulo padding
        return Image.fromarray(out)
