from typing import Dict, Any, Iterable
import json
import sys

def write_jsonl(records: Iterable[Dict[str, Any]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "
")
