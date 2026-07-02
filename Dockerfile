FROM python:3.14-slim@sha256:b877e50bd90de10af8d82c57a022fc2e0dc731c5320d762a27986facfc3355c1

LABEL org.opencontainers.image.source="https://github.com/chodeus/lama-sidecar"
LABEL org.opencontainers.image.description="Full-resolution LaMa inpainting sidecar for CHUB retexting"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Unbuffered stdout so `docker logs` is immediate; no .pyc writes (code dir is
# root-owned and the runtime user is non-root).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# curl for the model download in entrypoint.sh; clean apt lists to stay slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch — keeps the image ~1.5GB instead of ~5GB with CUDA.
RUN pip install --no-cache-dir torch==2.12.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Patch setuptools LAST so no earlier pip step reintroduces the base image's
# 70.2.0 (CVE-2025-47273 path traversal, fixed in 78.1.1) that Trivy fails on.
RUN pip install --no-cache-dir --upgrade 'setuptools>=78.1.1' \
    && python -c "import setuptools, sys; assert tuple(map(int, setuptools.__version__.split('.')[:2])) >= (78, 1), setuptools.__version__"

COPY app.py lama.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Run as non-root by default. 99:100 = nobody:users on Unraid; the IDs are
# numeric so the image stays portable. Pre-create /models owned by that user so
# the model download works even without a bind mount.
RUN mkdir -p /models && chown 99:100 /models
USER 99:100

ENV LAMA_MODEL_PATH=/models/big-lama.pt
# 8418 by default — 8080 collides with qBittorrent on most Unraid setups.
ENV PORT=8418
EXPOSE 8418

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8418}/health" || exit 1

ENTRYPOINT ["./entrypoint.sh"]
