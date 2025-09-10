"""Quality flags for pages/chunks."""
from __future__ import annotations
from typing import Dict, Any

def flag_quality(record: Dict[str, Any]) -> Dict[str, Any]:
    r = dict(record)
    text = r.get("text", "")
    r["is_empty"] = len(text.strip()) == 0
    r["char_count"] = len(text)
    return r
