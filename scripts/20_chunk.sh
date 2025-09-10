#!/usr/bin/env bash
set -euo pipefail
IN_DIR="data/interim"
OUT_DIR="data/chunks"
mkdir -p "$OUT_DIR"
python - <<PY
import json, pathlib, glob
from src.rag.process.chunking import build_chunks

paths = sorted(glob.glob("data/interim/*.pages.jsonl"))
all_out = pathlib.Path("data/chunks/all_chunks.jsonl")
with open(all_out, w, encoding=utf-8) as fall:
    for p in paths:
        with open(p, r, encoding=utf-8) as f:
            pages = [{"text": json.loads(l)["text"]} for l in f if l.strip()]
        chunks = build_chunks(pages)
        out = pathlib.Path("data/chunks") / (pathlib.Path(p).stem.replace(.pages,) + ".chunks.jsonl")
        with open(out, w, encoding=utf-8) as fo:
            for c in chunks:
                rec = {"title": c.get("title",""), "level": c.get("level",1), "text": c.get("text",""), "source": p}
                fo.write(json.dumps(rec, ensure_ascii=False) + "
")
                fall.write(json.dumps(rec, ensure_ascii=False) + "
")
        print(f"Wrote {out}")
print(f"Wrote {all_out}")
PY
