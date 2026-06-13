"""Full-resolution LaMa inpainting.

Loads the canonical ``big-lama`` TorchScript model (the same frozen artifact
IOPaint used) and runs it at the image's native resolution — LaMa is
resolution-robust (Fourier-convolution architecture) up to ~2k, so posters are
processed at full size rather than downscaled to a fixed box.

Preprocessing mirrors IOPaint's reference pipeline: normalise to [0,1], binarise
the mask (white = region to erase), pad height/width up to a multiple of 8 for
the convolutions, run, scale back to [0,255], and crop the padding away.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

PAD_MODULO = 8


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


class LamaModel:
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        # TorchScript model is self-contained — no model class needed.
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

    @torch.no_grad()
    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        mask = mask.convert("L")
        if mask.size != image.size:
            # NEAREST keeps the mask strictly binary after resizing.
            mask = mask.resize(image.size, Image.NEAREST)

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
