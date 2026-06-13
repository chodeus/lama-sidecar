"""Minimal LaMa inpainting sidecar.

Speaks the exact contract CHUB's ``lama_sidecar`` provider expects
(see CHUB ``backend/util/cl2k/text_removal.py``):

    POST /api/v1/inpaint
    body: {"image": "<base64 PNG>", "mask": "<base64 PNG, white = erase>"}
    -> 200 with the cleaned image as raw PNG bytes

CHUB composites the masked region back onto the original itself, so we simply
return LaMa's full reconstruction.
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

app = FastAPI(title="lama-sidecar", version="1.0.0")
_model: LamaModel | None = None


class InpaintRequest(BaseModel):
    image: str  # base64-encoded image
    mask: str   # base64-encoded mask, white (255) = erase


@app.on_event("startup")
def _load_model() -> None:
    global _model
    _model = LamaModel(MODEL_PATH)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/api/v1/inpaint")
def inpaint(req: InpaintRequest) -> Response:
    if _model is None:  # pragma: no cover - startup guarantees this
        raise HTTPException(status_code=503, detail="model not loaded")
    try:
        image = Image.open(io.BytesIO(base64.b64decode(req.image)))
        mask = Image.open(io.BytesIO(base64.b64decode(req.mask)))
    except Exception as exc:  # malformed payload
        raise HTTPException(status_code=400, detail=f"bad image/mask: {exc}")

    result = _model.inpaint(image, mask)
    buf = io.BytesIO()
    result.save(buf, "PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
