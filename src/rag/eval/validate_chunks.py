# Es posible revisar que los chunks tengan tamaño adecuado, no estén vacíos, preserven frases completas, y no repitan información. Sin embargo, medir si serán útiles (por ejemplo, si permiten responder preguntas o extraer información correcta) solo es verificable al integrarlos en el flujo objetivo 

import argparse, json, csv, re, sys, math, hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from difflib import SequenceMatcher
from collections import defaultdict

DEFAULT_MIN_WORDS= 400
DEFAULT_MAX_WORDS= 1200
NGRAM_WINDOW= 3
NEAR_DUP_JACCARD= 0.9
DUP_JACCARD= 0.95
NEAR_DUP_SEQ= 0.9
DUP_SEQ= 0.95
LENGT_BUCKET_SIZE= 100

# Stopwords ES/EN cortas
STOPWORDS = set("""
a al algo alguien alguna algunos ante antes aquel aquella aquellas aquellos aquí
así aunque cada como con contra cuándo cuál cuáles cuando de del desde donde dos el
ella ellas ellos en entre era eran es esa esas ese eso esos esta están esta este esto
estos fue fueron ha han hasta hay la las le les lo los más me mi mis mucha muchas
mucho muchos muy no nos nosotros o otra otras otro otros para pero pues que qué
se sin sobre su sus si sí también tanto te tenido tengo tiene tienen tu tus un una
unas uno unos y ya the a an and are as at be by for from has have if in into is it
its of on or that the their there these they this to was were will with your you
""".split())

WS_RE = re.compile(r"\s+", re.UNICODE) # espacios múltiples
TOKEN_RE = re.compile(r"\b[\wÁÉÍÓÚÜÑáéíóúüñ'-]+\b", re.UNICODE)
SENT_END_RE = re.compile(r"[\.!?¡¿…»”)]$")
SENT_START_GOOD_RE = re.compile(r"^[A-ZÁÉÍÓÚÑ0-9(«“\"]")
ONLY_PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)


def normalize(s:str)->str:
    if not s: return ""
    s=s.replace("\u00ad", "")
    s= s.replace("\ufb01","fi").replace("\ufb02","fl")
    s= WS_RE.sub(" ", s).strip()
    return s
def word(s:str)->List[str]:
    return [w for w in TOKEN_RE.findall(s) if w and not ONLY_PUNCT_RE.match(w)]

def sentence_boundaries_flags(s: str) -> Tuple[bool, bool]:
    s= normalize(s)
    if not s: 

# def sentence_boundaries_flags(s: str) -> Tuple[bool, bool]:
#     """Heurística: ¿parece cortado al inicio/fin?"""
#     s = normalize(s)
#     if not s: return (False, False)
#     start_cut = not bool(SENT_START_GOOD_RE.search(s))
#     end_cut = not bool(SENT_END_RE.search(s))
#     return (start_cut, end_cut)

# def stopword_ratio(tokens: List[str]) -> float:
#     if not tokens: return 1.0
#     sw = sum(1 for t in tokens if t.lower() in STOPWORDS)
#     return sw / max(1, len(tokens))

# def alpha_ratio(text: str) -> float:
#     if not text: return 0.0
#     alpha = sum(1 for ch in text if ch.isalpha())
#     return alpha / max(1, len(text))

# def shingles(tokens: List[str], n: int = NGRAM_N) -> set:
#     if len(tokens) < n: return set()
#     return {" ".join(tokens[i:i+n]).lower() for i in range(len(tokens)-n+1)}

# def jaccard(a: set, b: set) -> float:
#     if not a or not b: return 0.0
#     inter = len(a & b); union = len(a | b)
#     return inter / union if union else 0.0

# def seq_ratio(a: str, b: str) -> float:
#     return SequenceMatcher(None, a, b).ratio()

# def length_bucket(word_count: int) -> int:
#     return word_count // LENGTH_BUCKET_SIZE

# # ------------------ Adaptador de campos ------------------
# def adapt_chunk(raw: Dict[str, Any], auto_id: int) -> Dict[str, Any]:
#     """
#     Ajusta aquí si tus nombres difieren.
#     Requiere retornar: id, text, doc_id (opcional), pages (opcional).
#     """
#     cid = str(raw.get("id", raw.get("chunk_id", raw.get("uid", auto_id))))
#     text = raw.get("text", raw.get("content", "")) or ""
#     doc_id = raw.get("doc_id", raw.get("source_id", raw.get("document_id", None)))
#     pages = None
#     if isinstance(raw.get("pages"), list):
#         try:
#             pages = [int(p) for p in raw["pages"]]
#         except Exception:
#             pages = None
#     else:
#         ps = raw.get("page_start") or raw.get("start_page")
#         pe = raw.get("page_end") or raw.get("end_page")
#         if ps and pe:
#             try:
#                 ps, pe = int(ps), int(pe)
#                 if ps <= pe: pages = list(range(ps, pe+1))
#             except: pass
#     return {"id": cid, "text": text, "doc_id": (str(doc_id) if doc_id is not None else None), "pages": pages}

# # ------------------ Carga / salida ------------------
# def load_jsonl(path: Path) -> List[Dict[str, Any]]:
#     items = []
#     with path.open("r", encoding="utf-8") as f:
#         for i, line in enumerate(f, 1):
#             line = line.strip()
#             if not line: continue
#             try:
#                 items.append(json.loads(line))
#             except Exception as e:
#                 print(f"[WARN] Línea {i} inválida en {path.name}: {e}", file=sys.stderr)
#     return items

# def write_csv(rows: List[Dict[str, Any]], out_path: Path):
#     if not rows:
#         out_path.write_text("", encoding="utf-8"); return
#     with out_path.open("w", newline="", encoding="utf-8") as f:
#         w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
#         w.writeheader()
#         for r in rows: w.writerow(r)

# def write_jsonl(rows: List[Dict[str, Any]], out_path: Path):
#     with out_path.open("w", encoding="utf-8") as f:
#         for r in rows:
#             f.write(json.dumps(r, ensure_ascii=False) + "\n")

# # ------------------ Validación principal ------------------
# def validate_chunks_jsonl(items_raw: List[Dict[str, Any]],
#                           min_words: int,
#                           max_words: int
#                          ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
#     # Adaptación y preprocesado
#     adapted = [adapt_chunk(raw, i) for i, raw in enumerate(items_raw)]
#     texts = [normalize(ch["text"]) for ch in adapted]
#     tokens = [words(t) for t in texts]
#     wcounts = [len(tk) for tk in tokens]
#     length_buckets = [length_bucket(wc) for wc in wcounts]
#     shingles_sets = [shingles(tokens[i], NGRAM_N) for i in range(len(adapted))]

#     # Indexación por documento (para intra-doc) y por buckets de longitud (para eficiencia)
#     by_doc: Dict[str, List[int]] = defaultdict(list)
#     by_len_bucket: Dict[int, List[int]] = defaultdict(list)

#     for i, ch in enumerate(adapted):
#         if ch["doc_id"] is not None:
#             by_doc[ch["doc_id"]].append(i)
#         by_len_bucket[length_buckets[i]].append(i)

#     # Deduplicación (intra-doc e inter-doc)
#     duplicate_of = {}   # idx -> idx_base
#     neardup_of = {}     # idx -> idx_base

#     def compare_indices(candidate_indices: List[int]):
#         # O(m^2) en el grupo; los grupos son pequeños por bucketing
#         for a_idx in range(len(candidate_indices)):
#             i = candidate_indices[a_idx]
#             if i in duplicate_of: continue
#             for b_idx in range(a_idx+1, len(candidate_indices)):
#                 j = candidate_indices[b_idx]
#                 if j in duplicate_of: continue

#                 # Filtro rápido por longitud relativa
#                 li, lj = len(texts[i]), len(texts[j])
#                 if max(li, lj) == 0: continue
#                 if abs(li - lj) / max(li, lj) > 0.4:  # muy distintos
#                     continue

#                 # Jaccard sobre n-gramas
#                 jac = jaccard(shingles_sets[i], shingles_sets[j])
#                 if jac >= DUP_JACCARD:
#                     duplicate_of[j] = i
#                     continue
#                 if jac >= NEAR_DUP_JACCARD:
#                     sr = seq_ratio(texts[i], texts[j])
#                     if sr >= DUP_SEQ:
#                         duplicate_of[j] = i
#                     elif sr >= NEAR_DUP_SEQ:
#                         neardup_of[j] = i

#     # 1) Intra-documento: comparar dentro de cada doc_id
#     for _, idxs in by_doc.items():
#         if len(idxs) > 1:
#             compare_indices(sorted(idxs))

#     # 2) Inter-documento: comparar por buckets de longitud (y buckets adyacentes)
#     processed_pairs = set()
#     all_buckets = sorted(by_len_bucket.keys())
#     for b in all_buckets:
#         neighbor_buckets = [b-1, b, b+1]
#         group = []
#         for nb in neighbor_buckets:
#             group.extend(by_len_bucket.get(nb, []))
#         # Evitar recomparar los mismos índices muchas veces
#         group = sorted(set(group))
#         if len(group) > 1:
#             compare_indices(group)

#     # Reglas de calidad por chunk
#     rows = []
#     stats = {"total": len(adapted), "ok": 0, "warn": 0, "fail": 0}
#     for i, ch in enumerate(adapted):
#         t = texts[i]
#         tk = tokens[i]
#         wc = wcounts[i]
#         cc = len(t)
#         swr = stopword_ratio(tk)
#         ar = alpha_ratio(t)
#         start_cut, end_cut = sentence_boundaries_flags(t)

#         flags = []
#         if wc == 0: flags.append("empty")
#         if wc < min_words: flags.append(f"too_short(<{min_words})")
#         if wc > max_words: flags.append(f"too_long(>{max_words})")
#         if start_cut: flags.append("start_mid_sentence")
#         if end_cut: flags.append("end_mid_sentence")
#         if swr > 0.88 and wc >= 12: flags.append("mostly_stopwords")
#         if ar < 0.45 and cc >= 40: flags.append("low_alpha_ratio")

#         dup = duplicate_of.get(i)
#         ndp = neardup_of.get(i)
#         if dup is not None:
#             scope = "intra_doc" if (adapted[i]["doc_id"] is not None and adapted[i]["doc_id"] == adapted[dup]["doc_id"]) else "inter_doc"
#             flags.append(f"duplicate_of:{adapted[dup]['id']}:{scope}")
#         elif ndp is not None:
#             scope = "intra_doc" if (adapted[i]["doc_id"] is not None and adapted[i]["doc_id"] == adapted[ndp]["doc_id"]) else "inter_doc"
#             flags.append(f"near_duplicate_of:{adapted[ndp]['id']}:{scope}")

#         # Estado agregado simple: FAIL si vacío o duplicado; WARN si corta/larga o cortes de frase; OK si limpio
#         if "empty" in flags or any(f.startswith("duplicate_of") for f in flags):
#             status = "FAIL"
#         elif any(k in flags for k in ["too_short(<", "too_long(>", "start_mid_sentence", "end_mid_sentence", "mostly_stopwords", "low_alpha_ratio", "near_duplicate_of"]):
#             status = "WARN"
#         else:
#             status = "OK"

#         stats[status.lower()] += 1

#         rows.append({
#             "chunk_id": ch["id"],
#             "doc_id": ch["doc_id"] or "",
#             "pages": ",".join(map(str, ch["pages"])) if ch["pages"] else "",
#             "words": wc,
#             "chars": cc,
#             "stopword_ratio": f"{swr:.3f}",
#             "alpha_ratio": f"{ar:.3f}",
#             "start_cut": int(start_cut),
#             "end_cut": int(end_cut),
#             "status": status,
#             "flags": ";".join(flags),
#         })

#     return rows, stats

# # ------------------ CLI ------------------
# def parse_args():
#     ap = argparse.ArgumentParser(description="Valida calidad de chunks (JSONL): tamaño, vacío/basura, límites de frase y deduplicación.")
#     ap.add_argument("--in", dest="in_path", type=Path, required=True, help="Ruta al JSONL de chunks (una línea = un chunk)")
#     ap.add_argument("--out-csv", type=Path, default=Path("chunk_quality_report.csv"), help="Salida CSV")
#     ap.add_argument("--out-jsonl", type=Path, default=Path("chunk_quality_report.jsonl"), help="Salida JSONL con filas anotadas")
#     ap.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS, help="Mínimo de palabras por chunk")
#     ap.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS, help="Máximo de palabras por chunk")
#     return ap.parse_args()

# def main():
#     args = parse_args()
#     if not args.in_path.exists():
#         print(f"[ERROR] No existe: {args.in_path}", file=sys.stderr); sys.exit(2)

#     print(f"→ Cargando: {args.in_path}")
#     items = load_jsonl(args.in_path)
#     if not items:
#         print("[ERROR] JSONL vacío o ilegible", file=sys.stderr); sys.exit(2)

#     rows, stats = validate_chunks_jsonl(items, args.min_words, args.max_words)

#     # Salidas
#     args.out_csv.parent.mkdir(parents=True, exist_ok=True)
#     args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

#     write_csv(rows, args.out_csv)
#     # Guardar también como JSONL por si prefieres post-procesar en tu pipeline
#     # (incluye mismas columnas que el CSV)
#     jsonl_rows = [r for r in rows]
#     write_jsonl(jsonl_rows, args.out_jsonl)

#     print(f"→ Reporte CSV: {args.out_csv.resolve()}")
#     print(f"→ Reporte JSONL: {args.out_jsonl.resolve()}")
#     print("\n=== RESUMEN ===")
#     for k, v in stats.items():
#         print(f"{k}: {v}")

# if __name__ == "__main__":
#     main()