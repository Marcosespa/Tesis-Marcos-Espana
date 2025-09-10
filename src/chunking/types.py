from __future__ import annotations
from typing import List, Dict, Any, Iterable
from dataclasses import dataclass

@dataclass
class Chunk:
    id: str
    level: int
    title: str
    text: str
    metadata: Dict[str, Any]

