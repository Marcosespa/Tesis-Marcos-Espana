from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class PageBlock:
    page_number: int
    text: str

@dataclass
class Document:
    source_path: str
    blocks: List[PageBlock]
