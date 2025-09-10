"""Read all_chunks.jsonl and upsert to Weaviate."""
from __future__ import annotations
from typing import Dict, Any, Iterable
import json
from pathlib import Path
from src.index.weaviate_client import get_client

CLASS_NAME = "BookChunk"

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def ingest(jsonl_path: str) -> None:
    client = get_client()
    if client is None:
        print("Weaviate client not available; skipping.")
        return
    with client.batch as batch:
        for rec in iter_jsonl(jsonl_path):
            batch.add_data_object(rec, CLASS_NAME)
