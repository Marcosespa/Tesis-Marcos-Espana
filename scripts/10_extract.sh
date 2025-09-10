#!/usr/bin/env bash
set -euo pipefail
IN_DIR="${1:-OAPEN_PDFs/ciberseguridad}"
OUT_DIR="data/interim"
mkdir -p "$OUT_DIR"
python - <<PY
import json, pathlib
from src.ingest.extract_pdf import extract_pdf_to_pages
import glob

pdfs = sorted(glob.glob(f"{IN_DIR}/**/*.pdf", recursive=True))
for p in pdfs:
    pages = extract_pdf_to_pages(p)
    out = pathlib.Path("data/interim") / (pathlib.Path(p).stem + ".pages.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, w, encoding=utf-8) as f:
        for pg in pages:
            f.write(json.dumps({"page": pg.page_number, "text": pg.text, "source": p}, ensure_ascii=False) + "
")
    print(f"Wrote {out}")
PY
