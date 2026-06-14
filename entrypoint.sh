#!/usr/bin/env bash
# Fetch the big-lama model on first run (cached to the mounted /models volume),
# verify its checksum, then start the API. Pinning the MD5 means a tampered or
# truncated download fails loudly instead of serving a bad model.
set -euo pipefail

MODEL_PATH="${LAMA_MODEL_PATH:-/models/big-lama.pt}"
MODEL_URL="${LAMA_MODEL_URL:-https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt}"
MODEL_MD5="${LAMA_MODEL_MD5:-e3aa4aaa15225a33ec84f9f4bc47e500}"
# Default to 8418 (not 8080 — that collides with qBittorrent on most setups).
PORT="${PORT:-8418}"

mkdir -p "$(dirname "$MODEL_PATH")"

if [ ! -f "$MODEL_PATH" ]; then
  echo "[entrypoint] downloading big-lama model -> $MODEL_PATH"
  curl -fSL "$MODEL_URL" -o "$MODEL_PATH.tmp"
  mv "$MODEL_PATH.tmp" "$MODEL_PATH"
fi

if [ -n "$MODEL_MD5" ]; then
  echo "[entrypoint] verifying checksum"
  echo "${MODEL_MD5}  ${MODEL_PATH}" | md5sum -c - || {
    echo "[entrypoint] checksum FAILED — refusing to start" >&2
    exit 1
  }
fi

echo "[entrypoint] starting lama-sidecar on :${PORT}"
exec uvicorn app:app --host 0.0.0.0 --port "$PORT"
