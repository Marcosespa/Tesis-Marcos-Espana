#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import csv
import re
import sys
from pathlib import Path
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple, Optional
import fitz  # PyMuPDF
from collections import defaultdict

# --------------------------
# Utilidades de texto básicas
# --------------------------
WS_RE = re.compile(r"\s+", re.UNICODE)
SENT_SPLIT_RE = re.compile(r"(?<=[\.!?¿¡])\s+(?=[A-ZÁÉÍÓÚÑ0-9])")

def normalize_text(s: str) -> str:
    if not s:
        return ""
    # Normalización ligera: trim + colapsar espacios
    s = s.replace("\u00ad", "")  # soft hyphen
    s = s.replace("\ufb01", "fi").replace("\ufb02", "fl")  # ligaduras comunes
    s = WS_RE.sub(" ", s).strip()
    return s

def split_sentences(s: str) -> List[str]:
    s = normalize_text(s)
    if not s:
        return []
    # División simple por signos de final de frase; robusta y sin NLTK
    parts = SENT_SPLIT_RE.split(s)
    # Filtra vacíos y trozos residuales
    return [p.strip() for p in parts if p and not p.isspace()]

# --------------------------
# Extracción del PDF con PyMuPDF
# --------------------------
def extract_pdf_pages_text(pdf_path: Path) -> List[str]:
    """Extrae texto de cada página del PDF usando PyMuPDF"""
    pages_text = []
    try:
        doc = fitz.open(pdf_path)
        try:
            doc.set_layer_config(fitz.LAYER_CONFIG_DEFAULT)
        except AttributeError:
            pass  # Versiones más antiguas de PyMuPDF
        
        for i, page in enumerate(doc, start=1):
            try:
                txt = page.get_text("text") or ""
            except Exception:
                try:
                    # Fallback a words si get_text falla
                    words = page.get_text("words") or []
                    if isinstance(words, list):
                        txt = " ".join([word[4] for word in words if len(word) > 4])
                    else:
                        txt = ""
                except Exception:
                    txt = ""
            
            pages_text.append(normalize_text(txt))
        
        doc.close()
    except Exception as e:
        print(f"[ERROR] No se pudo abrir PDF {pdf_path}: {e}")
        return []
    
    return pages_text  # index 0 = página 1

# --------------------------
# Carga de datos JSONL
# --------------------------
def load_jsonl_pages(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Carga páginas desde archivo JSONL"""
    pages = []
    try:
        with jsonl_path.open('r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    pages.append(json.loads(line))
    except Exception as e:
        print(f"[ERROR] No se pudo cargar JSONL {jsonl_path}: {e}")
        return []
    return pages

# --------------------------
# Métricas de validación
# --------------------------
def similarity(a: str, b: str) -> float:
    a = normalize_text(a)
    b = normalize_text(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def status_from_scores(sim: float, has_text: bool, page_num: int, sim_thresh_ok=0.80, sim_thresh_warn=0.60) -> Tuple[str, str]:
    """
    Define estados: OK / WARN / FAIL
    """
    notes = []
    
    if not has_text:
        notes.append("página sin texto extraído")
        return "FAIL", "; ".join(notes)
    
    if sim >= sim_thresh_ok:
        status = "OK"
    elif sim >= sim_thresh_warn:
        status = "WARN"
        notes.append(f"similitud baja: {sim:.3f}")
    else:
        status = "FAIL"
        notes.append(f"similitud muy baja: {sim:.3f}")
    
    return status, "; ".join(notes)

# --------------------------
# Validación principal
# --------------------------
def validate_pages(
    pdf_pages_text: List[str],
    jsonl_pages: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Valida páginas extraídas contra el PDF original
    Retorna (rows, summary)
    """
    n_pdf_pages = len(pdf_pages_text)
    n_jsonl_pages = len(jsonl_pages)
    
    rows = []
    ok_count = warn_count = fail_count = 0
    
    # Estadísticas por categoría
    category_stats = defaultdict(lambda: {"ok": 0, "warn": 0, "fail": 0})
    
    for i, page_data in enumerate(jsonl_pages, 1):
        page_num = page_data.get("page_num_real", i)
        source_category = page_data.get("source_category", "unknown")
        doc_title = page_data.get("doc_title", "unknown")
        extracted_text = page_data.get("text", "")
        quality_flags = page_data.get("quality_flags", [])
        
        # Obtener texto del PDF para esta página
        pdf_text = ""
        if 1 <= page_num <= n_pdf_pages:
            pdf_text = pdf_pages_text[page_num - 1]
        else:
            pdf_text = ""  # Página fuera de rango
        
        # Calcular similitud
        has_text = bool(extracted_text.strip())
        sim = similarity(extracted_text, pdf_text) if has_text else 0.0
        
        # Contar frases
        sentences = split_sentences(extracted_text)
        sentence_count = len(sentences)
        
        # Determinar estado
        status, notes = status_from_scores(sim, has_text, page_num)
        
        # Agregar flags de calidad a las notas
        if quality_flags:
            notes = (notes + "; " if notes else "") + f"flags: {', '.join(quality_flags)}"
        
        # Actualizar contadores
        if status == "OK": 
            ok_count += 1
            category_stats[source_category]["ok"] += 1
        elif status == "WARN": 
            warn_count += 1
            category_stats[source_category]["warn"] += 1
        else: 
            fail_count += 1
            category_stats[source_category]["fail"] += 1
        
        rows.append({
            "doc_title": doc_title,
            "source_category": source_category,
            "page_num": page_num,
            "extracted_text_len": len(extracted_text),
            "pdf_text_len": len(pdf_text),
            "sentence_count": sentence_count,
            "similarity": f"{sim:.3f}",
            "status": status,
            "notes": notes,
            "quality_flags": ", ".join(quality_flags) if quality_flags else ""
        })
    
    # Calcular métricas de cobertura
    pages_with_text = sum(1 for p in jsonl_pages if p.get("text", "").strip())
    pages_with_ocr = sum(1 for p in jsonl_pages if "ocr_or_low_text" in p.get("quality_flags", []))
    pages_empty = sum(1 for p in jsonl_pages if not p.get("text", "").strip())
    
    summary = {
        "pdf_pages": n_pdf_pages,
        "jsonl_pages": n_jsonl_pages,
        "ok": ok_count,
        "warn": warn_count,
        "fail": fail_count,
        "pages_with_text": pages_with_text,
        "pages_with_ocr": pages_with_ocr,
        "pages_empty": pages_empty,
        "coverage_ratio": pages_with_text / n_jsonl_pages if n_jsonl_pages > 0 else 0,
        "category_stats": dict(category_stats)
    }
    
    return rows, summary

# --------------------------
# I/O helpers
# --------------------------
def write_csv(rows: List[Dict[str, Any]], out_path: Path):
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def find_matching_files(interim_dir: Path, pdf_dir: Path) -> List[Tuple[Path, Path]]:
    """Encuentra pares de archivos JSONL y PDF que coincidan"""
    matches = []
    
    # Buscar archivos JSONL en todas las subcarpetas
    for jsonl_file in interim_dir.rglob("*.pages.jsonl"):
        # Extraer nombre base (sin .pages.jsonl)
        base_name = jsonl_file.stem.replace(".pages", "")
        
        # Buscar PDF correspondiente en todas las subcarpetas
        for pdf_file in pdf_dir.rglob(f"{base_name}.pdf"):
            matches.append((jsonl_file, pdf_file))
            break  # Solo el primer match
    
    return matches

# --------------------------
# CLI
# --------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Valida extracción de páginas JSONL contra PDFs originales"
    )
    ap.add_argument("--interim", type=Path, default=Path("data/interim"), 
                   help="Directorio con archivos JSONL")
    ap.add_argument("--raw", type=Path, default=Path("data/raw"), 
                   help="Directorio con PDFs originales")
    ap.add_argument("--out", type=Path, default=Path("data/export/validation_report.csv"), 
                   help="Archivo CSV de salida")
    ap.add_argument("--ok-sim", type=float, default=0.80, 
                   help="Umbral de similitud para estado OK")
    ap.add_argument("--warn-sim", type=float, default=0.60, 
                   help="Umbral mínimo para WARN (si no, FAIL)")
    ap.add_argument("--single", type=str, 
                   help="Validar solo un archivo específico (nombre sin extensión)")
    return ap.parse_args()

def main():
    args = parse_args()
    
    if not args.interim.exists():
        print(f"[ERROR] No existe directorio interim: {args.interim}", file=sys.stderr)
        sys.exit(2)
    if not args.raw.exists():
        print(f"[ERROR] No existe directorio raw: {args.raw}", file=sys.stderr)
        sys.exit(2)
    
    # Crear directorio de salida si no existe
    args.out.parent.mkdir(parents=True, exist_ok=True)
    
    # Encontrar archivos a validar
    if args.single:
        # Validar archivo específico
        jsonl_file = None
        pdf_file = None
        
        # Buscar JSONL
        for f in args.interim.rglob(f"{args.single}.pages.jsonl"):
            jsonl_file = f
            break
        
        if not jsonl_file:
            print(f"[ERROR] No se encontró {args.single}.pages.jsonl", file=sys.stderr)
            sys.exit(2)
        
        # Buscar PDF correspondiente
        base_name = jsonl_file.stem.replace(".pages", "")
        for f in args.raw.rglob(f"{base_name}.pdf"):
            pdf_file = f
            break
        
        if not pdf_file:
            print(f"[ERROR] No se encontró PDF para {base_name}", file=sys.stderr)
            sys.exit(2)
        
        matches = [(jsonl_file, pdf_file)]
    else:
        # Validar todos los archivos
        matches = find_matching_files(args.interim, args.raw)
    
    if not matches:
        print("[ERROR] No se encontraron archivos para validar", file=sys.stderr)
        sys.exit(2)
    
    print(f"→ Encontrados {len(matches)} archivos para validar")
    
    all_rows = []
    total_summary = {
        "total_files": len(matches),
        "total_pdf_pages": 0,
        "total_jsonl_pages": 0,
        "total_ok": 0,
        "total_warn": 0,
        "total_fail": 0,
        "files_processed": 0
    }
    
    for jsonl_file, pdf_file in matches:
        print(f"\n→ Procesando: {jsonl_file.name} vs {pdf_file.name}")
        
        # Extraer texto del PDF
        pdf_pages_text = extract_pdf_pages_text(pdf_file)
        if not pdf_pages_text:
            print(f"  [SKIP] No se pudo extraer texto del PDF")
            continue
        
        # Cargar páginas JSONL
        jsonl_pages = load_jsonl_pages(jsonl_file)
        if not jsonl_pages:
            print(f"  [SKIP] No se pudieron cargar páginas JSONL")
            continue
        
        # Validar
        rows, summary = validate_pages(pdf_pages_text, jsonl_pages)
        
        # Agregar información del archivo a cada fila
        for row in rows:
            row["jsonl_file"] = jsonl_file.name
            row["pdf_file"] = pdf_file.name
        
        all_rows.extend(rows)
        
        # Actualizar resumen total
        total_summary["total_pdf_pages"] += summary["pdf_pages"]
        total_summary["total_jsonl_pages"] += summary["jsonl_pages"]
        total_summary["total_ok"] += summary["ok"]
        total_summary["total_warn"] += summary["warn"]
        total_summary["total_fail"] += summary["fail"]
        total_summary["files_processed"] += 1
        
        print(f"  OK: {summary['ok']}, WARN: {summary['warn']}, FAIL: {summary['fail']}")
        print(f"  Similitud promedio: {sum(float(r['similarity']) for r in rows) / len(rows):.3f}")
    
    # Escribir reporte CSV
    write_csv(all_rows, args.out)
    print(f"\n→ Reporte CSV: {args.out.resolve()}")
    
    # Resumen final
    print("\n=== RESUMEN FINAL ===")
    for k, v in total_summary.items():
        print(f"{k}: {v}")
    
    if total_summary["total_jsonl_pages"] > 0:
        success_rate = total_summary["total_ok"] / total_summary["total_jsonl_pages"]
        print(f"Tasa de éxito: {success_rate:.1%}")

if __name__ == "__main__":
    main()
