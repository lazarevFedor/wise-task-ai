#!/usr/bin/env bash
set -euo pipefail

QHOST="${QDRANT_HOST:-qdrant_db}"
QPORT="${QDRANT_PORT:-6333}"
TIMEOUT="${WAIT_FOR_QDRANT_TIMEOUT:-180}"
START_TS=$(date +%s)

echo "[auto-index] Waiting for Qdrant at ${QHOST}:${QPORT} (timeout ${TIMEOUT}s)..."
until curl -sf "http://${QHOST}:${QPORT}/collections" >/dev/null; do
  NOW=$(date +%s)
  if (( NOW - START_TS > TIMEOUT )); then
    echo "[auto-index] Timeout waiting Qdrant" >&2
    exit 1
  fi
  sleep 3
  echo "[auto-index] Still waiting..."
done

echo "[auto-index] Qdrant ready. Starting indexing job." 

if [ "${AUTO_INDEX:-0}" != "1" ]; then
  echo "[auto-index] AUTO_INDEX!=1 -> idle (container will exit)."
  exit 0
fi

echo "[auto-index] Detected AUTO_INDEX=1 -> proceeding with indexing."

FLAGS=""
if [[ "${INDEX_RECREATE:-0}" == "1" || "${INDEX_RECREATE:-false}" == "true" ]]; then
  FLAGS+=" --recreate"
fi
if [[ -n "${INDEX_BATCH_SIZE:-}" ]]; then
  FLAGS+=" --batch-size ${INDEX_BATCH_SIZE}"
fi
if [[ -n "${INDEX_MAX_FILES:-}" ]]; then
  FLAGS+=" --max-files ${INDEX_MAX_FILES}"
fi

echo "[auto-index] Running: python main.py${FLAGS}"
python main.py ${FLAGS} || { echo "[auto-index] Indexing failed" >&2; exit 1; }

echo "[auto-index] Indexing finished. Exiting container."
