from __future__ import annotations
from typing import List
import re
from src.extract.types import Document, PageBlock

WHITESPACE_RE = re.compile(r"\s+")
HYPHEN_RE = re.compile(r"(\w)-
(\w)")


def _normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def _fix_hyphenation(text: str) -> str:
    return HYPHEN_RE.sub(r"", text.replace("-
", ""))


def clean_blocks(doc: Document) -> Document:
    cleaned: List[PageBlock] = []
    for b in doc.blocks:
        t = b.text.replace("", "
")
        t = _fix_hyphenation(t)
        t = _normalize_whitespace(t)
        if t:
            cleaned.append(PageBlock(page_number=b.page_number, text=t))
    return Document(source_path=doc.source_path, blocks=cleaned)
