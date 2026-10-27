"""Rewrite detect.py's WHEEL_URL, WHEEL_SHA256 and MODEL_SHA256 for the rapidocr version its wheel name pins."""

import hashlib
import io
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

DETECT = Path("detect.py")


def main() -> int:
    src = DETECT.read_text(encoding="utf-8")
    version = re.search(r'"rapidocr-([0-9][^"-]*)-py3-none-any\.whl"', src).group(1)
    member = re.search(r'^MODEL_MEMBER = "([^"]+)"$', src, re.MULTILINE).group(1)

    with urllib.request.urlopen(f"https://pypi.org/pypi/rapidocr/{version}/json", timeout=60) as r:
        files = json.load(r)["urls"]
    wheel = next(f for f in files if f["filename"] == f"rapidocr-{version}-py3-none-any.whl")
    with urllib.request.urlopen(wheel["url"], timeout=300) as r:
        data = r.read()
    if hashlib.sha256(data).hexdigest() != wheel["digests"]["sha256"]:
        print(f"::error::{wheel['filename']} does not match PyPI's sha256", file=sys.stderr)
        return 1
    model_sha = hashlib.sha256(zipfile.ZipFile(io.BytesIO(data)).read(member)).hexdigest()

    path = wheel["url"].removeprefix("https://files.pythonhosted.org/packages/").rsplit("/", 1)[0]
    new = re.sub(
        r'WHEEL_URL = \(\n    "https://files\.pythonhosted\.org/packages/"\n    "[^"]+/"\n',
        f'WHEEL_URL = (\n    "https://files.pythonhosted.org/packages/"\n    "{path}/"\n',
        src,
    )
    new = re.sub(r'^WHEEL_SHA256 = "[0-9a-f]{64}"$', f'WHEEL_SHA256 = "{wheel["digests"]["sha256"]}"', new, flags=re.MULTILINE)
    old_model = re.search(r'^MODEL_SHA256 = "([0-9a-f]{64})"$', src, re.MULTILINE).group(1)
    new = re.sub(r'^MODEL_SHA256 = "[0-9a-f]{64}"$', f'MODEL_SHA256 = "{model_sha}"', new, flags=re.MULTILINE)
    DETECT.write_text(new, encoding="utf-8")
    print(f"rapidocr {version}: model {'unchanged' if model_sha == old_model else 'CHANGED'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
