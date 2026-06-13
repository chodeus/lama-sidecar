# lama-sidecar

A tiny, self-owned **full-resolution LaMa inpainting** service for CHUB's CL2K
"retexting" feature. It erases old title text from poster artwork and
reconstructs the background, so CHUB can draw a fresh label.

It is **not** IOPaint. It loads the same frozen `big-lama` model IOPaint used,
but via ~120 lines you own — no archived wrapper, no torch-stale dependency
chain, no Stable Diffusion / React / plugin baggage.

## Why this design

- **Full resolution.** LaMa is resolution-robust (Fourier-convolution
  architecture, good to ~2k). Posters run at native size — no downscaling to a
  fixed 512 box, which the only available ONNX builds force. The Fourier layers
  are precisely why LaMa can't export to a dynamic-shape ONNX, so the
  TorchScript model is the correct full-res tool.
- **Nothing abandoned.** Frozen model file + `torch` (maintained by the PyTorch
  Foundation) + FastAPI/Pillow/numpy (all active). Only the ~120 lines of glue
  are "yours to maintain," and that's just occasional dep bumps.
- **Separate container.** CHUB is untouched — they talk over HTTP. Rebuild or
  delete this without any risk to CHUB.

## Model provenance

| | |
|---|---|
| URL | `https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt` |
| MD5 | `e3aa4aaa15225a33ec84f9f4bc47e500` |
| License | Apache-2.0 (advimman/LaMa weights) |

Downloaded on first run to the mounted `/models` volume and checksum-verified by
`entrypoint.sh`. Override via `LAMA_MODEL_URL` / `LAMA_MODEL_MD5` if needed.

## Run

CI publishes `ghcr.io/chodeus/lama-sidecar` on every version tag, so hosts just
pull it — no local build required.

```bash
docker run -d --name lama-sidecar --restart unless-stopped \
  -p 8080:8080 \
  -v /path/to/appdata/lama-sidecar:/models \
  ghcr.io/chodeus/lama-sidecar:latest
```

First start downloads the ~200 MB model once into the volume. Expect a
**~1.5 GB image** and **~1–2 GB RAM** during inpainting. **CPU only** — no GPU is
required or used; a modern desktop CPU erases a poster in a few seconds.

### Unraid "Add Container" fields

| Field | Value |
|---|---|
| Repository | `ghcr.io/chodeus/lama-sidecar:latest` |
| Network Type | `bridge` |
| Port | Container `8080` → Host `8080` (TCP) |
| Path | Container `/models` → Host appdata, e.g. `/mnt/user/appdata/lama-sidecar` |

## Wire CHUB to it

In CHUB's `config.yml`, under `cl2k_maker`, point it at this container. Use the
host's address (LAN IP, or the container name if both share a Docker network):

```yaml
  ai_provider: lama_sidecar
  ai_endpoint: 'http://HOST:8080'   # CHUB appends /api/v1/inpaint
  ai_api_key: ''
  ai_model: ''
  ai_timeout: 120
```

Restart CHUB to reload config.

## Test

```bash
python test_smoke.py http://HOST:8080
```

Generates a poster-ish image with a black "text" bar, erases it, and checks the
region was reconstructed (writes `smoke_output.png`).

## Build / release

Local build: `docker build -t lama-sidecar:dev .`

Release: push a `vX.Y.Z` tag — CI runs the full build + real-inpaint smoke test,
then publishes the image to GHCR.

## API

```
GET  /health                -> {"status":"ok","model_loaded":true}
POST /api/v1/inpaint
     body: {"image":"<base64 PNG>","mask":"<base64 PNG, white=erase>"}
     ->   200, raw PNG bytes of the reconstructed image
```

This is exactly the contract CHUB's `lama_sidecar` provider calls
(`backend/util/cl2k/text_removal.py`).
