from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Any

from src.extract.pdf import extract_document
from src.clean.text import clean_blocks
from src.chunking.simple import build_chunks
from src.export.jsonl import write_jsonl


def run_pipeline(input_dir: str, out_dir: str) -> None:
    input_path = Path(input_dir)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "jsonl").mkdir(exist_ok=True)

    pdf_paths = sorted(p for p in input_path.rglob("*.pdf"))
    for pdf_path in pdf_paths:
        doc = extract_document(str(pdf_path))
        cleaned = clean_blocks(doc)
        chunks = build_chunks(cleaned)
        jsonl_records = [
            {
                "id": c.id,
                "level": c.level,
                "title": c.title,
                "text": c.text,
                "metadata": c.metadata,
            }
            for c in chunks
        ]
        out_file = out_root / "jsonl" / (pdf_path.stem + ".jsonl")
        write_jsonl(jsonl_records, str(out_file))
        print(f"Wrote {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF to JSONL pipeline")
    parser.add_argument("--input", required=True, help="Input directory with PDFs")
    parser.add_argument("--out", required=True, help="Output directory for JSONL")
    args = parser.parse_args()
    run_pipeline(args.input, args.out)
