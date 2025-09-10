from __future__ import annotations
from typing import List, Dict, Any
import re
import uuid
from src.extract.types import Document, PageBlock
from src.chunking.types import Chunk

# Simple heuristics: headings are lines in ALL CAPS or numbered outlines
HEADING_RE = re.compile(r"^(?:[A-Z][A-Z0-9 ,.:-]{3,}|(?:\d+\.)+\s+.+)")


def _split_into_lines(text: str) -> List[str]:
    return [ln.strip() for ln in text.split("
") if ln.strip()]


def _detect_headings(lines: List[str]) -> List[int]:
    idxs: List[int] = []
    for i, ln in enumerate(lines):
        if HEADING_RE.match(ln):
            idxs.append(i)
    if 0 not in idxs:
        idxs.insert(0, 0)
    return idxs


def build_chunks(doc: Document) -> List[Chunk]:
    # Merge all cleaned page texts first
    full_text = "
".join(b.text for b in doc.blocks)
    lines = _split_into_lines(full_text)
    heading_idxs = _detect_headings(lines)
    heading_idxs.append(len(lines))

    chunks: List[Chunk] = []
    for start, end in zip(heading_idxs, heading_idxs[1:]):
        section_lines = lines[start:end]
        if not section_lines:
            continue
        title = section_lines[0][:200]
        body = "
".join(section_lines)
        chunks.append(
            Chunk(
                id=str(uuid.uuid4()),
                level=1,
                title=title,
                text=body,
                metadata={
                    "source_path": doc.source_path,
                    "char_count": len(body),
                },
            )
        )

    # Optional semantic sub-chunking: fixed-size sliding windows (simple)
    SEM_SIZE = 1200
    SEM_OVERLAP = 200
    fine_chunks: List[Chunk] = []
    for ch in chunks:
        text = ch.text
        start = 0
        while start < len(text):
            end = min(len(text), start + SEM_SIZE)
            part = text[start:end]
            if len(part) < SEM_OVERLAP and start != 0:
                break
            fine_chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    level=2,
                    title=ch.title,
                    text=part,
                    metadata={**ch.metadata, "parent_id": ch.id},
                )
            )
            if end == len(text):
                break
            start = end - SEM_OVERLAP

    return chunks + fine_chunks
