"""End-to-end smoke test against a running sidecar.

Usage:
    python test_smoke.py [http://host:8080]

Cases:
  1. Small flat poster: black bar erased (the original smoke check).
  2. Large textured poster with a bulky hole (forces the downscale-inpaint-
     upscale path): erase must happen AND every pixel outside the dilated mask
     must come back byte-identical — the sidecar's core contract.
  3. All-black mask: output must equal the input everywhere.
  4. debug:true returns JSON with region/scale/timing metadata.
  5. /api/v1/detect and /api/v1/upscale, skipped with a warning on 404 so the
     script still works against pre-1.5 images.
"""

import base64
import io
import sys

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8418"

# Mirror the server-side dilation so case 2 can assert bit-exactness outside
# the dilated hole; sent explicitly so a tuned container can't skew the test.
DILATE, FEATHER = 5, 2


def _b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _inpaint(payload: dict) -> requests.Response:
    resp = requests.post(f"{BASE}/api/v1/inpaint", json=payload, timeout=600)
    resp.raise_for_status()
    return resp


def _noise_poster(w: int, h: int, seed: int = 5) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 216, size=(h, w, 3), dtype=np.uint8)
    return Image.fromarray(base).filter(ImageFilter.GaussianBlur(3))


def case_flat_bar() -> None:
    w, h = 600, 400
    img = Image.new("RGB", (w, h), (90, 140, 200))
    box = (150, 180, 450, 230)
    ImageDraw.Draw(img).rectangle(box, fill=(0, 0, 0))  # fake text bar to erase

    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)  # white = erase

    resp = _inpaint({"image": _b64_png(img), "mask": _b64_png(mask)})
    out = Image.open(io.BytesIO(resp.content)).convert("RGB")
    assert out.size == (w, h), f"size changed: {out.size}"

    cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
    r, g, b = out.getpixel((cx, cy))
    print(f"  erased-bar center now RGB=({r},{g},{b}) (was black)")
    assert r + g + b > 120, "masked region still looks black — erase failed"
    out.save("smoke_output.png")


def case_large_textured() -> None:
    w, h = 3200, 2000
    img = _noise_poster(w, h)
    box = (400, 600, 2800, 1400)  # 2400x800 hole: thick enough to force scale<1
    ImageDraw.Draw(img).rectangle(box, fill=(10, 10, 10))

    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)

    resp = _inpaint({
        "image": _b64_png(img), "mask": _b64_png(mask),
        "dilate": DILATE, "feather": FEATHER,
    })
    scales = resp.headers.get("X-Lama-Scales", "")
    print(f"  regions={resp.headers.get('X-Lama-Regions')} scales={scales} "
          f"ms={resp.headers.get('X-Lama-Inference-Ms')}")
    assert scales and min(float(s) for s in scales.split(",")) < 1.0, (
        "expected the bulky hole to run the downscale path"
    )

    out = Image.open(io.BytesIO(resp.content)).convert("RGB")
    in_arr = np.asarray(img)
    out_arr = np.asarray(out)

    hole = np.zeros((h, w), dtype=bool)
    hole[box[1]:box[3] + 1, box[0]:box[2] + 1] = True
    m = Image.fromarray(hole.astype("uint8") * 255)
    for _ in range(DILATE):
        m = m.filter(ImageFilter.MaxFilter(3))
    dilated = np.asarray(m) > 127

    outside_diff = (out_arr != in_arr).any(axis=2) & ~dilated
    assert not outside_diff.any(), (
        f"{int(outside_diff.sum())} pixels outside the dilated mask changed"
    )
    inside = out_arr[hole].astype(np.int32)
    orig_inside = in_arr[hole].astype(np.int32)
    mean_diff = float(np.abs(inside - orig_inside).mean())
    print(f"  mean abs-diff inside hole: {mean_diff:.1f}")
    assert mean_diff > 20, "hole barely changed — fill looks like an echo"


def case_noop_mask() -> None:
    img = _noise_poster(640, 480, seed=9)
    mask = Image.new("L", img.size, 0)
    resp = _inpaint({"image": _b64_png(img), "mask": _b64_png(mask)})
    out = Image.open(io.BytesIO(resp.content)).convert("RGB")
    assert (np.asarray(out) == np.asarray(img)).all(), (
        "all-black mask must return the input unchanged"
    )


def case_debug_meta() -> None:
    img = Image.new("RGB", (400, 300), (120, 120, 120))
    mask = Image.new("L", (400, 300), 0)
    ImageDraw.Draw(mask).rectangle((100, 100, 200, 140), fill=255)
    resp = _inpaint({"image": _b64_png(img), "mask": _b64_png(mask), "debug": True})
    body = resp.json()
    assert body["regions"] and body["scales"], f"debug meta missing: {list(body)}"
    Image.open(io.BytesIO(base64.b64decode(body["image"])))
    print(f"  debug meta: regions={body['regions']} scales={body['scales']}")


def case_optional_endpoints() -> None:
    poster = Image.new("RGB", (800, 1200), (30, 60, 120))
    ImageDraw.Draw(poster).text((80, 500), "THE  SMOKE  TEST", fill=(255, 255, 255),
                                font_size=90)
    resp = requests.post(f"{BASE}/api/v1/detect",
                         json={"image": _b64_png(poster)}, timeout=300)
    if resp.status_code == 404:
        print("  /detect not on this image — skipped")
    else:
        resp.raise_for_status()
        body = resp.json()
        assert body["regions"], "no text detected on a poster with a big title"
        print(f"  detect: {len(body['regions'])} region(s)")

    logo = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    ImageDraw.Draw(logo).text((6, 20), "LOGO", fill=(255, 255, 255, 255), font_size=20)
    resp = requests.post(f"{BASE}/api/v1/upscale",
                         json={"image": _b64_png(logo), "scale": 2}, timeout=300)
    if resp.status_code == 404:
        print("  /upscale not on this image — skipped")
    else:
        resp.raise_for_status()
        up = Image.open(io.BytesIO(resp.content))
        assert up.size == (128, 128), f"expected 128x128, got {up.size}"
        assert up.mode == "RGBA", f"alpha lost: mode={up.mode}"
        assert up.getpixel((2, 2))[3] < 30, "corner should stay transparent"
        print(f"  upscale: {up.size} {up.mode}")


def main() -> int:
    health = requests.get(f"{BASE}/health", timeout=10).json()
    print("health:", health)
    for case in (case_flat_bar, case_large_textured, case_noop_mask,
                 case_debug_meta, case_optional_endpoints):
        print(f"{case.__name__}:")
        case()
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
