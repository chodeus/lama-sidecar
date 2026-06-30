"""Minimal LaMa inpainting sidecar.

Speaks the exact contract CHUB's ``lama_sidecar`` provider expects
(see CHUB ``backend/util/cl2k/text_removal.py``):

    POST /api/v1/inpaint
    body: {"image": "<base64 image, JPEG/PNG/etc>", "mask": "<base64 mask, white = erase>"}
    -> 200 with the cleaned image as raw PNG bytes (lossless intermediate)

The input format is auto-detected by Pillow, so JPEG posters work as-is. The
response is PNG so CHUB's composite/re-encode isn't stacked on lossy bytes.

Large masked areas are reconstructed per-region (see lama.py) so logos don't come
back as a blur; only masked pixels are altered. CHUB composites the masked region
onto its own original again, so the full frame we return stays compatible.
"""

from __future__ import annotations

import base64
import io
import os

from fastapi import FastAPI, HTTPException, Response
from PIL import Image
from pydantic import BaseModel

from lama import LamaModel

MODEL_PATH = os.environ.get("LAMA_MODEL_PATH", "/models/big-lama.pt")

# Guard rails so an oversized payload can't exhaust memory on an unauthenticated
# endpoint. Posters are a few MP; these limits are generous but bounded.
MAX_B64_CHARS = int(os.environ.get("LAMA_MAX_B64_CHARS", 64 * 1024 * 1024))  # ~48MB binary
MAX_PIXELS = int(os.environ.get("LAMA_MAX_PIXELS", 40_000_000))  # 40 MP per image

# Per-region adaptive reconstruction (see lama.py). TARGET_RES is the long-edge a
# crop is downscaled to before inpainting; 0 disables splitting (one full pass).
TARGET_RES = int(os.environ.get("LAMA_TARGET_RES", 1024))
REGION_PAD = float(os.environ.get("LAMA_REGION_PAD", 0.5))

# Grow the incoming mask by N px before inpainting so a logo's anti-aliased fringe
# and soft glow get erased too (otherwise they survive as a ghost outline); feather
# softens the composite seam. Set both to 0 for the original no-dilation behaviour.
# See lama.py.
MASK_DILATE = int(os.environ.get("LAMA_MASK_DILATE", 5))
MASK_FEATHER = int(os.environ.get("LAMA_MASK_FEATHER", 2))

app = FastAPI(title="lama-sidecar", version="1.3.0")  # x-release-please-version
_model: LamaModel | None = None


class InpaintRequest(BaseModel):
    image: str  # base64-encoded image
    mask: str   # base64-encoded mask, white (255) = erase


@app.on_event("startup")
def _load_model() -> None:
    global _model
    _model = LamaModel(
        MODEL_PATH,
        target_res=TARGET_RES,
        region_pad=REGION_PAD,
        mask_dilate=MASK_DILATE,
        mask_feather=MASK_FEATHER,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/api/v1/inpaint")
def inpaint(req: InpaintRequest) -> Response:
    if _model is None:  # pragma: no cover - startup guarantees this
        raise HTTPException(status_code=503, detail="model not loaded")
    if len(req.image) > MAX_B64_CHARS or len(req.mask) > MAX_B64_CHARS:
        raise HTTPException(status_code=413, detail="image or mask too large")
    try:
        image = Image.open(io.BytesIO(base64.b64decode(req.image)))
        mask = Image.open(io.BytesIO(base64.b64decode(req.mask)))
    except Exception as exc:  # malformed payload
        raise HTTPException(status_code=400, detail=f"bad image/mask: {exc}")

    if image.width * image.height > MAX_PIXELS:
        raise HTTPException(status_code=413, detail="image resolution too large")

    result = _model.inpaint(image, mask)
    buf = io.BytesIO()
    result.save(buf, "PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
