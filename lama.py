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
    group_regions,
    hole_scale,
    mask_bbox,
)

PAD_MODULO = 8

# Above this many separate blobs (or horizontal runs, checked first — runs are
# countable in one vectorised pass, before the labeller spends any time) the
# grouping isn't worth it and the mask is probably noise: fall back to one
# global-extent region.
MAX_COMPONENTS = 2000

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

    @torch.no_grad()
    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        dilate_px: int | None = None,
        feather_px: int | None = None,
        target_res: int | None = None,
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
            "inference_ms": 0,
        }
        if not mask_arr.any():
            meta["inference_ms"] = int((time.perf_counter() - t0) * 1000)
            return image.copy(), meta

        mask_arr = dilate(mask_arr, dil)
        mask = Image.fromarray(mask_arr.astype("uint8") * 255)

        w, h = image.size
        full_frame = not tres or tres <= 0
        if full_frame:
            regions = [(0, 0, h, w)]
        elif count_runs(mask_arr) > MAX_COMPONENTS:
            # Run count bounds the component count, so this bails out BEFORE
            # the labeller — a speckled mask can't pin the worker.
            regions = [self._global_extent(mask_arr, w, h)]
        else:
            boxes = connected_components(mask_arr)
            regions = group_regions(boxes, self.region_pad, self.min_pad, w, h)

        out = image.copy()
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

        meta["inference_ms"] = int((time.perf_counter() - t0) * 1000)
        return out, meta

    def _global_extent(self, mask_arr: np.ndarray, w: int, h: int):
        y0, x0, y1, x1 = mask_bbox(mask_arr)
        p = self.min_pad
        return (max(0, y0 - p), max(0, x0 - p), min(h, y1 + p), min(w, x1 + p))

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
        return Image.fromarray(out)
