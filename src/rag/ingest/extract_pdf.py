"""Extract PDF text using PyMuPDF (fitz) with optional OCR fallback."""
from __future__ import annotations
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None

@dataclass
class PageRecord:
    page_number: int
    text: str
    metadata: Dict[str, Any]


def extract_pdf_to_pages(pdf_path: str) -> List[PageRecord]:
    pages: List[PageRecord] = []
    if fitz is None:
        return pages
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        pages.append(PageRecord(page_number=i, text=text, metadata={}))
    doc.close()
    return pages
