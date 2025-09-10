#!/usr/bin/env bash
set -euo pipefail
JSONL="data/chunks/all_chunks.jsonl"
python - <<PY
from src.index.ingest_to_weaviate import ingest
ingest("data/chunks/all_chunks.jsonl")
PY
