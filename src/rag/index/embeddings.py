"""Embeddings via SentenceTransformers (CPU/GPU)."""
from __future__ import annotations
from typing import List

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None

_model = None

def get_model(name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    global _model
    if _model is None and SentenceTransformer is not None:
        _model = SentenceTransformer(name)
    return _model

def embed_texts(texts: List[str]) -> List[List[float]]:
    m = get_model()
    if m is None:
        return [[0.0] * 384 for _ in texts]
    return m.encode(texts, normalize_embeddings=True).tolist()
