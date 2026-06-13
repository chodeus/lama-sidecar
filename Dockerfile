FROM python:3.14-slim

LABEL org.opencontainers.image.source="https://github.com/chodeus/lama-sidecar"
LABEL org.opencontainers.image.description="Full-resolution LaMa inpainting sidecar for CHUB retexting"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

# curl for the model download in entrypoint.sh; clean apt lists to stay slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch — keeps the image ~1.5GB instead of ~5GB with CUDA.
RUN pip install --no-cache-dir torch==2.12.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py lama.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Run as non-root by default. 99:100 = nobody:users on Unraid; the IDs are
# numeric so the image stays portable. Pre-create /models owned by that user so
# the model download works even without a bind mount.
RUN mkdir -p /models && chown 99:100 /models
USER 99:100

ENV LAMA_MODEL_PATH=/models/big-lama.pt
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8080/health || exit 1

ENTRYPOINT ["./entrypoint.sh"]
