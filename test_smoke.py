"""End-to-end smoke test against a running sidecar.

Usage:
    python test_smoke.py [http://host:8080]

Generates a synthetic image with a black bar (the "text") and a matching mask,
sends it to /api/v1/inpaint, and verifies a same-size PNG comes back with the
masked region noticeably changed (i.e. the bar was erased, not echoed back).
"""

import base64
import io
import sys

import requests
from PIL import Image, ImageDraw

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8418"


def _b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main() -> int:
    w, h = 600, 400
    img = Image.new("RGB", (w, h), (90, 140, 200))
    draw = ImageDraw.Draw(img)
    box = (150, 180, 450, 230)
    draw.rectangle(box, fill=(0, 0, 0))  # fake text bar to erase

    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)  # white = erase

    health = requests.get(f"{BASE}/health", timeout=10).json()
    print("health:", health)

    resp = requests.post(
        f"{BASE}/api/v1/inpaint",
        json={"image": _b64_png(img), "mask": _b64_png(mask)},
        timeout=300,
    )
    resp.raise_for_status()

    out = Image.open(io.BytesIO(resp.content)).convert("RGB")
    assert out.size == (w, h), f"size changed: {out.size}"

    cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
    r, g, b = out.getpixel((cx, cy))
    print(f"center of erased region now RGB=({r},{g},{b}) (was black)")
    assert r + g + b > 120, "masked region still looks black — erase failed"

    out.save("smoke_output.png")
    print("OK — wrote smoke_output.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
