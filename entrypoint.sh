#!/usr/bin/env bash
# Fetch the big-lama model on first run (cached to the mounted /models volume),
# verify its checksums, then start the API. Pinning MD5+SHA256 means a tampered
# or truncated download fails loudly instead of serving a bad model.
set -euo pipefail

MODEL_PATH="${LAMA_MODEL_PATH:-/models/big-lama.pt}"
MODEL_URL="${LAMA_MODEL_URL:-https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt}"
# ${VAR-default} (no colon) so an explicitly EMPTY env var disables that
# algorithm instead of silently falling back to the pinned default.
MODEL_MD5="${LAMA_MODEL_MD5-e3aa4aaa15225a33ec84f9f4bc47e500}"
MODEL_SHA256="${LAMA_MODEL_SHA256-344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9}"
# Default to 8418 (not 8080 — that collides with qBittorrent on most setups).
PORT="${PORT:-8418}"

mkdir -p "$(dirname "$MODEL_PATH")"

# An empty env var skips that algorithm; when both are set, both must pass.
verify() {
  local file="$1"
  if [ -n "$MODEL_MD5" ]; then
    echo "${MODEL_MD5}  ${file}" | md5sum -c - || return 1
  fi
  if [ -n "$MODEL_SHA256" ]; then
    echo "${MODEL_SHA256}  ${file}" | sha256sum -c - || return 1
  fi
}

download() {
  echo "[entrypoint] downloading big-lama model -> $MODEL_PATH"
  # Keep a partial .tmp on curl failure so -C - resumes it next start.
  curl -fSL --retry 5 --retry-all-errors --proto '=https' --proto-redir '=https' \
    -C - "$MODEL_URL" -o "$MODEL_PATH.tmp" || return 1
  # Verify BEFORE the mv so a bad download is never cached as the real model.
  echo "[entrypoint] verifying checksum"
  verify "$MODEL_PATH.tmp" || {
    echo "[entrypoint] checksum FAILED on download — discarding it" >&2
    rm -f "$MODEL_PATH.tmp"
    return 1
  }
  mv "$MODEL_PATH.tmp" "$MODEL_PATH"
}

if [ ! -f "$MODEL_PATH" ]; then
  download || { echo "[entrypoint] download failed — refusing to start" >&2; exit 1; }
elif ! verify "$MODEL_PATH"; then
  # Self-heal a corrupted cache: re-download once, give up if that fails too.
  echo "[entrypoint] cached model failed verification — re-downloading" >&2
  rm -f "$MODEL_PATH"
  download || { echo "[entrypoint] re-download failed — refusing to start" >&2; exit 1; }
fi

echo "[entrypoint] starting lama-sidecar on :${PORT}"
exec uvicorn app:app --host 0.0.0.0 --port "$PORT"
