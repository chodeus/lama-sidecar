"""Text detection for building erase masks automatically.

Runs the RapidOCR project's ``PP-OCRv6_det_small`` model (DBNet family,
Apache-2.0) detection-only under onnxruntime. The rapidocr *package* is not
imported: every published version hard-depends on the full ``opencv-python``
build, whose vendored Qt libraries link the system ``libGL.so.1`` that
python:3.14-slim doesn't ship — importing cv2 there dies before the first
detection. Its det pipeline is small and fully specified, so the pre/post
processing is ported here on numpy/PIL (the parameters mirror rapidocr 3.9.1's
shipped ``config.yaml``: mean/std 0.5, thresh 0.3, unclip 1.6, 2x2 dilation).

The model itself still comes from the rapidocr 3.9.1 wheel — an immutable
files.pythonhosted.org artifact — downloaded lazily on first use into the
model volume and double-verified (wheel SHA256 against PyPI's published
digest, then the extracted .onnx against the hash rapidocr's own model
registry pins), following the same pattern as upscale.py.

DB emits a per-pixel text-probability map. Pixels above the binarisation
threshold are grouped into components, each component gets a min-area
rectangle (convex hull + rotating calipers, so diagonal banners aren't
over-covered by axis-aligned boxes), and the rectangle is expanded by DB's
standard unclip offset — the network is trained to predict *shrunken* text
kernels, so the raw region sits inside the true glyph extent.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
import urllib.request
import zipfile

import numpy as np
from PIL import Image, ImageDraw

from regions import connected_components, count_runs, dilate

WHEEL_URL = (
    "https://files.pythonhosted.org/packages/"
    "a0/23/e8d7251c53137b5b66d89e85cb0bc3e6bcfaec9527f29b477ec04389c8b2/"
    "rapidocr-3.9.1-py3-none-any.whl"
)
WHEEL_SHA256 = "600885e4e94e0b427abad394fccb0ec1d3c9118a215ca435bf7680aeae0e292b"
MODEL_MEMBER = "rapidocr/models/PP-OCRv6_det_small.onnx"
MODEL_FILE = "PP-OCRv6_det_small.onnx"
MODEL_SHA256 = "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f"

# Detection doesn't need full poster resolution: cap the long edge before the
# network resize and scale polygons back up afterwards.
DET_LONG_EDGE = 1600

# rapidocr's det defaults (config.yaml, PP-OCRv6_det_small).
LIMIT_SIDE_LEN = 736  # limit_type "min": upscale until the short side reaches this
THRESH = 0.3
UNCLIP_RATIO = 1.6
MIN_SIZE = 3
MAX_CANDIDATES = 1000

# limit_type "min" has no upper bound: a very thin sliver would explode the
# network input, so cap the final long edge outright.
MAX_NET_EDGE = 2560

# The labeller costs a Python-level pass per run; a threshold map with this
# many runs is noise, not text (same bail-out idea as lama.MAX_COMPONENTS).
MAX_RUNS = 20_000

MASK_DILATE_PX = 2


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_model(dest: str) -> None:
    if not WHEEL_URL.startswith("https://"):
        raise RuntimeError("model URL must be https")
    tmp_whl = dest + ".whl.tmp"
    tmp = dest + ".tmp"
    try:
        with urllib.request.urlopen(WHEEL_URL) as resp, open(tmp_whl, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        if _sha256(tmp_whl) != WHEEL_SHA256:
            raise RuntimeError("rapidocr wheel failed SHA256 verification")
        with zipfile.ZipFile(tmp_whl) as whl:
            with whl.open(MODEL_MEMBER) as src, open(tmp, "wb") as fh:
                shutil.copyfileobj(src, fh)
        if _sha256(tmp) != MODEL_SHA256:
            raise RuntimeError("det model failed SHA256 verification")
        os.replace(tmp, dest)
    finally:
        for leftover in (tmp_whl, tmp):
            if os.path.exists(leftover):
                os.remove(leftover)


def _to_rgb(image: Image.Image) -> Image.Image:
    """Composite transparency onto white — a bare convert() exposes stale RGB
    under transparent pixels, which the detector would happily read as text."""
    if image.mode == "RGB":
        return image
    if "A" in image.getbands() or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return image.convert("RGB")


def _net_size(w: int, h: int) -> tuple[int, int]:
    """Network input dims: short side up to LIMIT_SIDE_LEN, long edge capped,
    both rounded to the multiple of 32 the convolutions require."""
    ratio = 1.0
    if min(w, h) < LIMIT_SIDE_LEN:
        ratio = LIMIT_SIDE_LEN / min(w, h)
    ratio = min(ratio, MAX_NET_EDGE / max(w, h))
    rw = max(32, int(round(w * ratio / 32)) * 32)
    rh = max(32, int(round(h * ratio / 32)) * 32)
    return rw, rh


def _dilate2x2(m: np.ndarray) -> np.ndarray:
    """cv2.dilate with a 2x2 all-ones kernel (anchor at its centre pulls from
    the top-left neighbourhood) — rapidocr's use_dilation step."""
    p = np.pad(m, ((1, 0), (1, 0)))
    return p[1:, 1:] | p[:-1, 1:] | p[1:, :-1] | p[:-1, :-1]


def _row_extremes(sub: np.ndarray, y0: int, x0: int) -> np.ndarray:
    """(x, y) coords of each row's first/last True pixel — every convex-hull
    vertex is a row extreme, so this feeds the hull at O(rows) points."""
    rows = np.flatnonzero(sub.any(axis=1))
    first = sub[rows].argmax(axis=1)
    last = sub.shape[1] - 1 - sub[rows, ::-1].argmax(axis=1)
    ys = np.concatenate((rows, rows)) + y0
    xs = np.concatenate((first, last)) + x0
    return np.stack((xs, ys), axis=1).astype(np.float64)


def _convex_hull(pts: np.ndarray) -> np.ndarray:
    """Andrew monotone chain; returns the hull counter-clockwise."""
    pts = np.unique(pts, axis=0)  # also sorts lexicographically
    if len(pts) <= 2:
        return pts

    def chain(points):
        h: list[np.ndarray] = []
        for p in points:
            while len(h) >= 2:
                a, b = h[-1] - h[-2], p - h[-2]
                if a[0] * b[1] - a[1] * b[0] > 0:  # np.cross dropped 2-D support
                    break
                h.pop()
            h.append(p)
        return h[:-1]

    return np.array(chain(pts) + chain(pts[::-1]))


def _min_area_rect(pts: np.ndarray):
    """Rotating calipers over the hull -> (center, u, v, w, h) with u/v the
    rectangle's unit axes. Degenerate (point/line) inputs come back with a
    zero side and get dropped by the caller's size filters."""
    hull = _convex_hull(pts)
    if len(hull) == 1:
        return hull[0], np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.0, 0.0
    if len(hull) == 2:
        d = hull[1] - hull[0]
        length = float(np.hypot(*d))
        u = d / length
        v = np.array([-u[1], u[0]])
        return (hull[0] + hull[1]) / 2, u, v, length, 0.0

    best = None
    for i in range(len(hull)):
        e = hull[(i + 1) % len(hull)] - hull[i]
        length = np.hypot(*e)
        if length == 0:
            continue
        u = e / length
        v = np.array([-u[1], u[0]])
        pu, pv = hull @ u, hull @ v
        w = float(pu.max() - pu.min())
        h = float(pv.max() - pv.min())
        if best is None or w * h < best[0]:
            center = u * (pu.max() + pu.min()) / 2 + v * (pv.max() + pv.min()) / 2
            best = (w * h, center, u, v, w, h)
    return best[1:]


def _order_clockwise(pts: np.ndarray) -> np.ndarray:
    """tl, tr, br, bl — rapidocr's output convention for a quad."""
    xs = pts[np.argsort(pts[:, 0]), :]
    left, right = xs[:2], xs[2:]
    left = left[np.argsort(left[:, 1]), :]
    right = right[np.argsort(right[:, 1]), :]
    return np.array([left[0], right[0], right[1], left[1]])


class TextDetector:
    def __init__(self) -> None:
        self.model_dir = os.path.dirname(
            os.environ.get("LAMA_MODEL_PATH", "/models/big-lama.pt")
        )
        self._session = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._session is None:
                import onnxruntime as ort

                path = os.path.join(self.model_dir, MODEL_FILE)
                # Re-verify an existing file: it lives on a shared volume.
                if not os.path.isfile(path) or _sha256(path) != MODEL_SHA256:
                    os.makedirs(self.model_dir, exist_ok=True)
                    _download_model(path)
                opts = ort.SessionOptions()
                # Input sizes vary per request; without this the arena keeps
                # every high-water mark for the life of the process.
                opts.enable_cpu_mem_arena = False
                self._session = ort.InferenceSession(
                    path, sess_options=opts, providers=["CPUExecutionProvider"]
                )
        return self._session

    def detect(
        self, image: Image.Image, min_score: float = 0.5
    ) -> tuple[list[dict], Image.Image]:
        """Text regions of ``image`` as (regions, mask).

        regions: [{"polygon": [[x, y], ...], "score": float}] in original
        pixel coords; mask: 'L' image, white = detected text, dilated a
        couple of px so anti-aliased glyph fringes are covered (the inpaint
        side adds its own, larger dilation).
        """
        session = self._load()

        rgb = _to_rgb(image)
        w0, h0 = rgb.size
        det_img = rgb
        if max(w0, h0) > DET_LONG_EDGE:
            scale = DET_LONG_EDGE / max(w0, h0)
            det_img = rgb.resize(
                (max(1, round(w0 * scale)), max(1, round(h0 * scale))), Image.LANCZOS
            )

        rw, rh = _net_size(*det_img.size)
        net_img = det_img.resize((rw, rh), Image.BILINEAR)
        # Model is trained on cv2-style BGR; mean/std from rapidocr's config.
        arr = np.asarray(net_img, dtype=np.float32)[:, :, ::-1]
        arr = (arr / 255.0 - 0.5) / 0.5
        batch = arr.transpose(2, 0, 1)[np.newaxis]

        pred = session.run(None, {session.get_inputs()[0].name: batch})[0][0, 0]
        bh, bw = pred.shape
        seg = _dilate2x2(pred > THRESH)

        empty = Image.new("L", (w0, h0), 0)
        if not seg.any() or count_runs(seg) > MAX_RUNS:
            return [], empty

        boxes = connected_components(seg)
        boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        sx, sy = w0 / bw, h0 / bh

        regions: list[dict] = []
        polys: list[list[tuple[int, int]]] = []
        for y0, x0, y1, x1 in boxes[:MAX_CANDIDATES]:
            sub = seg[y0:y1, x0:x1]
            # Mean probability over the component's own pixels; a bbox this
            # tight rarely captures a neighbour, and over-scoring an intruder
            # only makes the mask slightly more inclusive.
            score = float(pred[y0:y1, x0:x1][sub].mean())
            if score < min_score:
                continue

            center, u, v, w, h = _min_area_rect(_row_extremes(sub, y0, x0))
            if min(w, h) < MIN_SIZE:
                continue
            # DB predicts shrunken text kernels; the standard unclip offset
            # (area * ratio / perimeter) grows the rect back to glyph extent.
            d = (w * h * UNCLIP_RATIO) / (2 * (w + h))
            if min(w, h) + 2 * d < MIN_SIZE + 2:
                continue
            hw, hh = w / 2 + d, h / 2 + d
            quad = np.array(
                [
                    center - u * hw - v * hh,
                    center + u * hw - v * hh,
                    center + u * hw + v * hh,
                    center - u * hw + v * hh,
                ]
            )

            quad[:, 0] = np.clip(np.round(quad[:, 0] * sx), 0, w0 - 1)
            quad[:, 1] = np.clip(np.round(quad[:, 1] * sy), 0, h0 - 1)
            quad = _order_clockwise(quad)
            if (
                np.linalg.norm(quad[0] - quad[1]) <= 3
                or np.linalg.norm(quad[0] - quad[3]) <= 3
            ):
                continue

            polygon = [[int(x), int(y)] for x, y in quad]
            regions.append({"polygon": polygon, "score": score})
            polys.append([(p[0], p[1]) for p in polygon])

        if not regions:
            return [], empty

        mask = empty
        draw = ImageDraw.Draw(mask)
        for poly in polys:
            draw.polygon(poly, fill=255)
        mask_arr = dilate(np.asarray(mask) > 0, MASK_DILATE_PX)
        return regions, Image.fromarray(mask_arr.astype("uint8") * 255)
