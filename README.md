# lama-sidecar

Full-resolution LaMa inpainting service for CHUB's CL2K "retexting". Erases text
from poster artwork and reconstructs the background. CPU-only, runs as non-root.

Speaks the contract CHUB's `lama_sidecar` provider calls:

```
GET  /health                -> {"status":"ok","model_loaded":true}
POST /api/v1/inpaint
     body: {"image":"<base64 PNG>","mask":"<base64 PNG, white=erase>"}
     ->   200, PNG bytes of the reconstructed image
```

## Model

| | |
|---|---|
| File | `big-lama.pt` (TorchScript) |
| URL | `https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt` |
| MD5 | `e3aa4aaa15225a33ec84f9f4bc47e500` |
| License | Apache-2.0 |

Downloaded to `/models` on first run and MD5-verified. Override with
`LAMA_MODEL_URL` / `LAMA_MODEL_MD5`.

## Run

```bash
docker run -d --name lama-sidecar --restart unless-stopped \
  -p 8418:8418 \
  -v /path/to/appdata/lama-sidecar:/models \
  ghcr.io/chodeus/lama-sidecar:latest
```

Image ~1.5 GB, ~1–2 GB RAM in use. Listens on **8418**. Runs as **99:100**.

### Unraid

Template: [`unraid/lama-sidecar.xml`](unraid/lama-sidecar.xml) — drop in
`/boot/config/plugins/dockerMan/templates-user/`, or set:

| Field | Value |
|---|---|
| Repository | `ghcr.io/chodeus/lama-sidecar:latest` |
| Port | `8418` → `8418` (TCP) |
| Path | `/models` → e.g. `/mnt/user/appdata/lama-sidecar` |

## CHUB config

In CHUB's `config.yml` under `cl2k_maker`:

```yaml
  ai_provider: lama_sidecar
  ai_endpoint: 'http://HOST:8418'   # or container name on a shared network
  ai_api_key: ''
  ai_model: ''
  ai_timeout: 120
```

## Config

| Env | Default | |
|---|---|---|
| `PORT` | `8418` | API port |
| `LAMA_MODEL_PATH` | `/models/big-lama.pt` | model location |
| `LAMA_MAX_PIXELS` | `40000000` | reject images larger than this |
| `LAMA_MAX_B64_CHARS` | `67108864` | reject payloads larger than this |

## Test

```bash
python test_smoke.py http://HOST:8418
```

## Release

Push a `vX.Y.Z` tag — CI builds, runs a real-inpaint test, and publishes to GHCR.
