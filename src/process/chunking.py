"""Chunking: hierarchical + semantic windows."""
from __future__ import annotations
from typing import List, Dict, Any
import re, uuid

HEADING_RE = re.compile(r"^(?:[A-Z][A-Z0-9 ,.:-]{3,}|(?:\d+\.)+\s+.+)")

class Chunk(dict):
    pass


def build_chunks(pages: List[Dict[str, Any]], max_chars: int = 2400, overlap: int = 300) -> List[Chunk]:
    text = "
".join(p["text"] for p in pages if p.get("text"))
    lines = [ln.strip() for ln in text.split("
") if ln.strip()]
    idxs = [i for i, ln in enumerate(lines) if HEADING_RE.match(ln)]
    if 0 not in idxs: idxs.insert(0, 0)
    idxs.append(len(lines))
    chunks: List[Chunk] = []
    for s, e in zip(idxs, idxs[1:]):
        body = "
".join(lines[s:e])
        if not body: continue
        parent_id = str(uuid.uuid4())
        chunks.append(Chunk(id=parent_id, level=1, title=lines[s][:200], text=body))
        # semantic windows
        start = 0
        while start < len(body):
            end = min(len(body), start + max_chars)
            part = body[start:end]
            if not part: break
            chunks.append(Chunk(id=str(uuid.uuid4()), level=2, title=lines[s][:200], text=part, parent_id=parent_id))
            if end == len(body): break
            start = end - overlap
    return chunks
