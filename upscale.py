"""Logo super-resolution with Real-ESRGAN's compact general model.

Runs ``realesr-general-x4v3`` (SRVGGNetCompact, ~4.8MB) from the official
xinntao/Real-ESRGAN v0.2.5.0 release. The architecture is ported inline from
``basicsr.archs.srvgg_arch`` so we don't drag in the basicsr/realesrgan
packages (heavy, stale pins) for a 40-line network.

Weights are downloaded lazily on first use into the model directory (the same
volume as big-lama.pt), SHA256-verified against a pinned hash before being
moved into place, and loaded with ``weights_only=True`` — a plain
``torch.load`` on a downloaded pickle is arbitrary code execution.

Alpha is handled the way Real-ESRGAN's own ``alpha_upsampler='realesrgan'``
does it: the alpha channel is stacked to 3-channel gray, run through the same
model, and one channel is taken back — so a logo's anti-aliased edges scale as
crisply as its colors. The model is x4-native; scale=2 is x4 + LANCZOS down.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
import urllib.request

import numpy as np
import torch
from PIL import Image
from torch import nn

MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
    "realesr-general-x4v3.pth"
)
MODEL_SHA256 = "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292"
MODEL_FILE = "realesr-general-x4v3.pth"

# This endpoint is for logos, not posters — bound the input so a 4x upscale
# can't balloon memory.
MAX_PIXELS = 2_000_000


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: str) -> None:
    if not url.startswith("https://"):
        raise RuntimeError("model URL must be https")
    tmp = dest + ".tmp"
    # Timeout matters: this runs under the inference semaphore, so a stalled
    # connection would otherwise wedge the whole endpoint.
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    if _sha256(tmp) != MODEL_SHA256:
        os.remove(tmp)
        raise RuntimeError("model download failed SHA256 verification")
    os.replace(tmp, dest)


class _SRVGGNetCompact(nn.Module):
    """Compact VGG-style SR net (basicsr ``srvgg_arch``), x4 variant.

    Layer indices must match the checkpoint's ``body.N`` keys exactly:
    conv head + PReLU, ``num_conv`` conv/PReLU pairs, conv tail to
    ``3 * upscale**2`` channels, PixelShuffle, plus a nearest-upsampled
    residual base.
    """

    def __init__(self, num_feat: int = 64, num_conv: int = 32, upscale: int = 4) -> None:
        super().__init__()
        self.upscale = upscale
        body: list[nn.Module] = [
            nn.Conv2d(3, num_feat, 3, 1, 1),
            nn.PReLU(num_parameters=num_feat),
        ]
        for _ in range(num_conv):
            body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            body.append(nn.PReLU(num_parameters=num_feat))
        body.append(nn.Conv2d(num_feat, 3 * upscale * upscale, 3, 1, 1))
        self.body = nn.ModuleList(body)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = nn.functional.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base


class Upscaler:
    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        self._model: _SRVGGNetCompact | None = None
        # FastAPI runs sync endpoints in a threadpool; the lock keeps two
        # concurrent first-calls from double-downloading/loading.
        self._lock = threading.Lock()

    def _load(self) -> _SRVGGNetCompact:
        with self._lock:
            if self._model is None:
                path = os.path.join(self.model_dir, MODEL_FILE)
                # Re-verify an existing file too: it lives on a shared volume,
                # and hashing 5MB is cheap next to loading it.
                if not os.path.isfile(path) or _sha256(path) != MODEL_SHA256:
                    os.makedirs(self.model_dir, exist_ok=True)
                    _download(MODEL_URL, path)
                ckpt = torch.load(path, map_location="cpu", weights_only=True)
                state = ckpt.get("params_ema", ckpt.get("params", ckpt))
                model = _SRVGGNetCompact()
                model.load_state_dict(state, strict=True)
                model.eval()
                self._model = model
        return self._model

    def _run(self, model: _SRVGGNetCompact, img: Image.Image) -> Image.Image:
        arr = np.asarray(img, dtype=np.float32) / 255.0  # H x W x 3, [0,1]
        t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
        out = model(t)[0].clamp_(0, 1).mul_(255).round_().byte()
        return Image.fromarray(out.permute(1, 2, 0).numpy())

    @torch.no_grad()
    def upscale(self, image: Image.Image, scale: int) -> Image.Image:
        if scale not in (2, 4):
            raise ValueError("scale must be 2 or 4")
        if image.width * image.height > MAX_PIXELS:
            raise ValueError("image too large for upscaling")
        model = self._load()

        has_alpha = "A" in image.getbands() or (
            image.mode == "P" and "transparency" in image.info
        )
        if has_alpha:
            image = image.convert("RGBA")
        out = self._run(model, image.convert("RGB"))
        if has_alpha:
            alpha = image.getchannel("A")
            up_alpha = self._run(model, Image.merge("RGB", (alpha, alpha, alpha)))
            out = out.convert("RGBA")
            out.putalpha(up_alpha.getchannel("R"))

        if scale == 2:  # model is x4-native
            out = out.resize((image.width * 2, image.height * 2), Image.LANCZOS)
        return out
