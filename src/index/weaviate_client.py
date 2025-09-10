"""Weaviate client helpers."""
from __future__ import annotations
from typing import Iterable, Dict, Any
import os

try:
    import weaviate
except Exception:  # pragma: no cover
    weaviate = None

DEF_HOST = os.getenv("WEAVIATE_HOST", "localhost")
DEF_PORT = os.getenv("WEAVIATE_PORT", "8080")
DEF_SCHEME = os.getenv("WEAVIATE_SCHEME", "http")


def get_client():
    if weaviate is None:
        return None
    return weaviate.Client(f"{DEF_SCHEME}://{DEF_HOST}:{DEF_PORT}")

