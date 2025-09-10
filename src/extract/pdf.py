from __future__ import annotations
from typing import List
from dataclasses import dataclass
from pathlib import Path
import PyPDF2
from src.extract.types import Document, PageBlock


def extract_document(pdf_path: str) -> Document:
    blocks: List[PageBlock] = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            blocks.append(PageBlock(page_number=i, text=text))
    return Document(source_path=pdf_path, blocks=blocks)
