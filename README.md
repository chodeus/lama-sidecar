<p align="center">
  <img src="logo.png" alt="lama-sidecar" width="360">
</p>

# lama-sidecar

Full-resolution LaMa inpainting service for CHUB's CL2K "retexting". Erases text
from poster artwork and reconstructs the background. CPU-only, runs as non-root.

Speaks the contract CHUB's `lama_sidecar` provider calls:

```
GET  /health                -> {"status":"ok","model_loaded":true}
POST /api/v1/inpaint
     body: {"image":"<base64 image, JPEG/PNG/etc>","mask":"<base64 mask, white=erase>"}
     ->   200, PNG bytes (lossless; CHUB composites & re-encodes)
```

## Model

| | |
|---|---|
| File | `big-lama.pt` (TorchScript) |
| URL | `https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt` |
| MD5 | `e3aa4aaa15225a33ec84f9f4bc47e500` |
| License | Apache-2.0 (model weights only — see [License](#license)) |

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

### Install on Unraid

1. Copy [`unraid/lama-sidecar.xml`](unraid/lama-sidecar.xml) to
   `/boot/config/plugins/dockerMan/templates-user/` on the server.
2. **Docker** tab → **Add Container** → pick **lama-sidecar** from the template
   dropdown.
3. Check the fields, then **Apply**:

   | Field | Value |
   |---|---|
   | Repository | `ghcr.io/chodeus/lama-sidecar:latest` |
   | Port | `8418` → `8418` (TCP) |
   | `/models` | `/mnt/user/appdata/lama-sidecar` |

4. First start downloads the model (~200 MB); wait until the container shows
   **healthy** before use.

No template file? **Add Container** manually with the same Repository, Port, and
Path. To reach it from CHUB without exposing a host port, put both containers on
the same Docker network and use `http://lama-sidecar:8418`.

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
| `LAMA_TARGET_RES` | `1024` | per-region downscale long-edge; `0` = single full-res pass |
| `LAMA_REGION_PAD` | `0.5` | context padding around each masked region (fraction of its size) |
| `LAMA_MAX_PIXELS` | `40000000` | reject images larger than this |
| `LAMA_MAX_B64_CHARS` | `67108864` | reject payloads larger than this |

### Background reconstruction

On a large contiguous hole (e.g. a title wordmark over clouds) LaMa's receptive
field can't reach the interior, so it returns a smooth blur. The sidecar splits
the mask into connected regions and inpaints each in its own crop at an adaptive
scale: small regions run at native resolution (sharp), large ones are downscaled
so the hole fits the receptive field, inpainted, then upscaled — recovering
texture. Only masked pixels change. Set `LAMA_TARGET_RES=0` to revert to a single
full-resolution pass.

## Test

```bash
python test_smoke.py http://HOST:8418
```

## Release

Push a `vX.Y.Z` tag — CI builds, runs a real-inpaint test, and publishes to GHCR.

## License

Two separate licenses apply:

- **This repository's code** (the sidecar wrapper, Dockerfile, scripts) — MIT,
  see [LICENSE](LICENSE).
- **The `big-lama.pt` model weights** — Apache-2.0, owned by their original
  authors. They are downloaded at runtime, not bundled in this image or repo.
