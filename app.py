"""Minimal LaMa inpainting sidecar.

Speaks the exact contract CHUB's ``lama_sidecar`` provider expects
(see CHUB ``backend/util/cl2k/text_removal.py``):

    POST /api/v1/inpaint
    body: {"image": "<base64 image, JPEG/PNG/etc>", "mask": "<base64 mask, white = erase>"}
    -> 200 with the cleaned image as raw PNG bytes (lossless intermediate)

Optional per-request fields (dilate/feather/target_res) override the env
defaults for one call; "debug": true switches the response to JSON with the
region boxes, scales and timing alongside the image. Every inpaint response
carries X-Lama-Regions / X-Lama-Scales / X-Lama-Inference-Ms headers.

Two companion endpoints share the same base64 conventions and the optional
LAMA_API_KEY guard: /api/v1/detect (text detection -> polygons + white=text
mask, for building erase masks automatically) and /api/v1/upscale (2x/4x
alpha-aware super-resolution for small logo art).

The input format is auto-detected by Pillow, so JPEG posters work as-is. The
response is PNG so CHUB's composite/re-encode isn't stacked on lossy bytes.

Large masked areas are reconstructed per-region (see lama.py) so logos don't
come back as a blur; only masked pixels are altered.
"""

from __future__ import annotations

import base64
import hmac
import io
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from lama import LamaModel

log = logging.getLogger("uvicorn.error")


class _HealthAccessFilter(logging.Filter):
    """Keep the Docker HEALTHCHECK out of the access log: it curls /health from
    inside the container every 30s, which otherwise prints a 200 line each probe
    (~2880/day of pure noise). Only PASSING probes are dropped — a failing one
    (non-200) still logs, as does every real API request."""

    def filter(self, record: logging.LogRecord) -> bool:
        # uvicorn.access args: (client_addr, method, path, http_version, status).
        args = record.args
        if not isinstance(args, tuple) or len(args) != 5:
            return True
        return not (args[1] == "GET" and args[2] == "/health" and args[4] == 200)


logging.getLogger("uvicorn.access").addFilter(_HealthAccessFilter())

MODEL_PATH = os.environ.get("LAMA_MODEL_PATH", "/models/big-lama.pt")

# Guard rails so an oversized payload can't exhaust memory on an unauthenticated
# endpoint. Posters are a few MP; these limits are generous but bounded. The
# body cap is enforced by middleware BEFORE the JSON is buffered/parsed — the
# per-field checks alone would only fire after the whole body sat in memory.
MAX_B64_CHARS = int(os.environ.get("LAMA_MAX_B64_CHARS", 64 * 1024 * 1024))  # ~48MB binary
MAX_PIXELS = int(os.environ.get("LAMA_MAX_PIXELS", 40_000_000))  # 40 MP per image
MAX_BODY_BYTES = int(
    os.environ.get("LAMA_MAX_BODY_BYTES", 2 * MAX_B64_CHARS + (16 << 20))
)

# Per-region adaptive reconstruction (see lama.py). TARGET_RES caps the crop
# long-edge fed to the model (the scale itself derives from hole geometry); it
# also bounds a native-resolution pass, so a very wide text band downscales
# instead of running full-size for no visible gain over busy texture. 0 switches
# to a single whole-frame region.
TARGET_RES = int(os.environ.get("LAMA_TARGET_RES", 1400))
REGION_PAD = float(os.environ.get("LAMA_REGION_PAD", 0.5))
REGION_MIN_PAD = int(os.environ.get("LAMA_REGION_MIN_PAD", 64))
HOLE_RES = int(os.environ.get("LAMA_HOLE_RES", 512))
HOLE_THICK = int(os.environ.get("LAMA_HOLE_THICK", 384))
SEAM_MATCH = os.environ.get("LAMA_SEAM_MATCH", "1") not in ("0", "false", "no")

# Grow the incoming mask by N px before inpainting so a logo's anti-aliased fringe
# and soft glow get erased too (otherwise they survive as a ghost outline); feather
# softens the composite seam (inward only). Both overridable per request.
MASK_DILATE = int(os.environ.get("LAMA_MASK_DILATE", 5))
MASK_FEATHER = int(os.environ.get("LAMA_MASK_FEATHER", 2))

# One inference at a time by default: a single pass already uses every core,
# so parallel requests only oversubscribe CPU and multiply peak memory.
MAX_CONCURRENCY = int(os.environ.get("LAMA_MAX_CONCURRENCY", 1))

# Optional shared secret; when set, /api/v1/* require it (X-API-Key or Bearer).
API_KEY = os.environ.get("LAMA_API_KEY", "")

_model: LamaModel | None = None
_infer_slots = threading.Semaphore(max(1, MAX_CONCURRENCY))
_lazy_lock = threading.Lock()
_detector = None
_upscaler = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _model
    _model = LamaModel(
        MODEL_PATH,
        target_res=TARGET_RES,
        region_pad=REGION_PAD,
        min_pad=REGION_MIN_PAD,
        mask_dilate=MASK_DILATE,
        mask_feather=MASK_FEATHER,
        hole_res=HOLE_RES,
        hole_thick=HOLE_THICK,
        seam_match=SEAM_MATCH,
    )
    log.info("lama-sidecar ready — model loaded from %s", MODEL_PATH)
    yield


app = FastAPI(title="lama-sidecar", version="1.5.1", lifespan=_lifespan)  # x-release-please-version


class BodyLimitMiddleware:
    """Enforce the body cap BEFORE the app sees the request: a declared
    Content-Length over the cap gets an immediate 413, and bodies without one
    are buffered here (bounded by the cap — Starlette would buffer them anyway)
    and rejected the moment they exceed it, so the app is never invoked."""

    def __init__(self, app, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    if int(value) > self.max_bytes:
                        return await self._reject(send)
                except ValueError:
                    pass

        chunks: list[bytes] = []
        seen = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                return  # client disconnected before the body completed
            seen += len(message.get("body", b""))
            if seen > self.max_bytes:
                return await self._reject(send)
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break

        replayed = False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {
                    "type": "http.request",
                    "body": b"".join(chunks),
                    "more_body": False,
                }
            return await receive()

        await self.app(scope, replay, send)

    @staticmethod
    async def _reject(send):
        body = b'{"detail":"request body too large"}'
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})


app.add_middleware(BodyLimitMiddleware, max_bytes=MAX_BODY_BYTES)


def _require_api_key(request: Request) -> None:
    if not API_KEY:
        return
    auth = request.headers.get("authorization", "")
    candidates = [
        request.headers.get("x-api-key", ""),
        auth[7:] if auth.lower().startswith("bearer ") else "",
    ]
    if not any(c and hmac.compare_digest(c, API_KEY) for c in candidates):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


class InpaintRequest(BaseModel):
    image: str  # base64-encoded image
    mask: str   # base64-encoded mask, white (255) = erase
    dilate: int | None = Field(None, ge=0, le=64)
    feather: int | None = Field(None, ge=0, le=32)
    target_res: int | None = Field(None, ge=0, le=8192)
    debug: bool = False


class DetectRequest(BaseModel):
    image: str
    min_score: float = Field(0.5, ge=0.0, le=1.0)


class UpscaleRequest(BaseModel):
    image: str
    scale: Literal[2, 4] = 2


def _decode_image(b64: str, what: str) -> Image.Image:
    if len(b64) > MAX_B64_CHARS:
        raise HTTPException(status_code=413, detail=f"{what} too large")
    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        if img.width * img.height > MAX_PIXELS:
            raise HTTPException(status_code=413, detail=f"{what} resolution too large")
        # Browsers show JPEGs orientation-corrected, so masks are drawn in the
        # displayed frame; transpose so both live in the same coordinates.
        return ImageOps.exif_transpose(img)
    except HTTPException:
        raise
    except Exception:
        log.exception("undecodable %s payload", what)
        raise HTTPException(
            status_code=400, detail=f"invalid base64 or undecodable {what}"
        )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/api/v1/inpaint", dependencies=[Depends(_require_api_key)])
def inpaint(req: InpaintRequest) -> Response:
    if _model is None:  # pragma: no cover - startup guarantees this
        raise HTTPException(status_code=503, detail="model not loaded")
    image = _decode_image(req.image, "image")
    mask = _decode_image(req.mask, "mask")

    with _infer_slots:
        result, meta = _model.inpaint(
            image, mask,
            dilate_px=req.dilate,
            feather_px=req.feather,
            target_res=req.target_res,
        )

    headers = {
        "X-Lama-Regions": str(len(meta["regions"])),
        "X-Lama-Scales": ",".join(str(s) for s in meta["scales"]),
        "X-Lama-Inference-Ms": str(meta["inference_ms"]),
    }
    buf = io.BytesIO()
    result.save(buf, "PNG")
    if req.debug:
        return JSONResponse(
            {"image": base64.b64encode(buf.getvalue()).decode(), **meta},
            headers=headers,
        )
    return Response(content=buf.getvalue(), media_type="image/png", headers=headers)


@app.post("/api/v1/detect", dependencies=[Depends(_require_api_key)])
def detect(req: DetectRequest) -> JSONResponse:
    global _detector
    image = _decode_image(req.image, "image")
    try:
        if _detector is None:
            with _lazy_lock:
                if _detector is None:
                    from detect import TextDetector
                    _detector = TextDetector()
        with _infer_slots:
            regions, mask = _detector.detect(image, req.min_score)
    except HTTPException:
        raise
    except Exception:
        log.exception("text detection failed")
        raise HTTPException(status_code=503, detail="text detection unavailable")
    buf = io.BytesIO()
    mask.save(buf, "PNG")
    return JSONResponse(
        {"regions": regions, "mask": base64.b64encode(buf.getvalue()).decode()}
    )


@app.post("/api/v1/upscale", dependencies=[Depends(_require_api_key)])
def upscale(req: UpscaleRequest) -> Response:
    global _upscaler
    image = _decode_image(req.image, "image")
    try:
        if _upscaler is None:
            with _lazy_lock:
                if _upscaler is None:
                    from upscale import Upscaler
                    _upscaler = Upscaler(os.path.dirname(MODEL_PATH))
        with _infer_slots:
            result = _upscaler.upscale(image, req.scale)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc))
    except Exception:
        log.exception("upscale failed")
        raise HTTPException(status_code=503, detail="upscaling unavailable")
    buf = io.BytesIO()
    result.save(buf, "PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
