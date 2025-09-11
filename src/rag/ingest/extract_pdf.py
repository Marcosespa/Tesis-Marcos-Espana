#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF -> data/interim/*.pages.jsonl
- Extrae texto por página con PyMuPDF
- Fallback OCR con Tesseract si la página no tiene texto
- Limpia encabezados/pies repetidos y guiones de fin de línea
- Conserva metadatos: doc_title, page_num_real/logical, toc_path, source_id
"""

import argparse
import hashlib
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Set

import fitz  # PyMuPDF
from PIL import Image

# OCR opcional
try:
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False


# ------------------ Utilidades ------------------

# Para generar un hash del archivo unico
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def normalize_text(page_text: str) -> str:
    """
    Limpieza ligera:
    - Une palabras partidas por guión al final de línea: 'infor-\nmación' -> 'información'
    - Colapsa espacios múltiples
    - Preserva párrafos dobles y compacta líneas sueltas
    """
    txt = page_text.replace("\r", "")
    # Une palabras partidas por guion + salto de línea
    txt = re.sub(r"(\w)-\n(\w)", r"\1\2", txt)
    # Convierte líneas sueltas en espacio, preserva párrafos dobles
    txt = re.sub(r"\n{3,}", "\n\n", txt)                # 3+ saltos -> 2
    txt = re.sub(r"([^\n])\n([^\n])", r"\1 \2", txt)    # línea simple -> espacio
    # Colapsa espacios
    txt = re.sub(r"[ \t]+", " ", txt)
    return txt.strip()


def detect_repeat_headers_footers(pages_lines: List[List[str]]) -> Dict[str, Set[str]]:
    """
    Heurística: si la primera o última línea se repite en >50% de páginas y es corta,
    se considera header/footer.
    """
    first_counts, last_counts = {}, {}
    n = max(1, len(pages_lines))

    for lines in pages_lines:
        if not lines:
            continue
        first = lines[0].strip()
        last = lines[-1].strip()
        first_counts[first] = first_counts.get(first, 0) + 1
        last_counts[last] = last_counts.get(last, 0) + 1

    headers = {l for l, c in first_counts.items() if c / n > 0.5 and len(l) <= 80}
    footers = {l for l, c in last_counts.items() if c / n > 0.5 and len(l) <= 80}
    return {"headers": headers, "footers": footers}


def strip_header_footer(text: str, headers: Set[str], footers: Set[str]) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    if lines and lines[0] in headers:
        lines = lines[1:]
    if lines and lines[-1] in footers:
        lines = lines[:-1]
    return "\n".join(lines)


def page_image_ocr(page: fitz.Page, ocr_lang: str) -> str:
    """ Renderiza la página a PNG (300 dpi) y corre OCR con Tesseract. """
    if not OCR_AVAILABLE:
        return ""
    
    try:
        # Intentar configurar capas si está disponible
        try:
            page.set_layer_config(fitz.LAYER_CONFIG_DEFAULT)
        except AttributeError:
            # Versión antigua de PyMuPDF sin set_layer_config
            pass
            
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        
        # Convertir a RGB si es necesario para evitar errores de formato
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        return __import__("pytesseract").image_to_string(img, lang=ocr_lang).strip()
    except Exception as e:
        print(f"[WARNING] OCR falló: {e}")
        return ""


def get_toc_map(doc: fitz.Document) -> List[Dict[str, Any]]:
    """ Devuelve TOC como lista de dicts: [{'level':1,'title':'Cap 1','page':1}, ...] """
    toc = doc.get_toc()
    res = []
    for lvl, title, page in toc:
        res.append({"level": int(lvl), "title": (title or "").strip(), "page": int(page)})
    return res


def section_for_page(toc: List[Dict[str, Any]], page_num_1based: int) -> List[str]:
    """
    Aproxima 'toc_path' para una página: toma la última entrada del TOC cuyo inicio <= página actual.
    Devuelve ruta jerárquica ["Capítulo 2", "2.1 Sección", ...]
    """
    if not toc:
        return []
    stack = []
    last_seen = []
    for entry in toc:
        if entry["page"] <= page_num_1based:
            lvl = entry["level"]
            title = entry["title"]
            if lvl <= len(stack):
                stack = stack[:lvl - 1]
            stack.append(title)
            last_seen = list(stack)
        else:
            break
    return last_seen


def write_jsonl(records: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ------------------ Pipeline por PDF ------------------

def get_source_category(pdf_path: Path) -> str:
    """Determina la categoría de fuente basada en la ruta del PDF"""
    path_str = str(pdf_path)
    if "NIST" in path_str:
        if "AI" in path_str:
            return "NIST_AI"
        elif "CSWP" in path_str:
            return "NIST_CSWP"
        elif "FIPS" in path_str:
            return "NIST_FIPS"
        elif "GCR" in path_str:
            return "NIST_GCR"
        elif "ITL_Bulletin" in path_str:
            return "NIST_ITL"
        elif "SP" in path_str:
            return "NIST_SP"
        else:
            return "NIST_OTHER"
    elif "OAPEN" in path_str:
        return "OAPEN"
    elif "USENIX" in path_str:
        return "USENIX"
    else:
        return "OTHER"


def extract_abstract(text: str) -> str:
    """Extrae el abstract del texto de la primera página"""
    # Busca patrones comunes de abstract
    abstract_patterns = [
        r"Abstract\s*[:\-]?\s*(.+?)(?:\n\n|\n[A-Z]|$)",
        r"ABSTRACT\s*[:\-]?\s*(.+?)(?:\n\n|\n[A-Z]|$)",
        r"Resumen\s*[:\-]?\s*(.+?)(?:\n\n|\n[A-Z]|$)",
        r"RESUMEN\s*[:\-]?\s*(.+?)(?:\n\n|\n[A-Z]|$)",
    ]
    
    for pattern in abstract_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            abstract = match.group(1).strip()
            # Limita a 500 caracteres para evitar abstracts muy largos
            if len(abstract) > 500:
                abstract = abstract[:500] + "..."
            return abstract
    
    # Si no encuentra abstract, toma las primeras 200 caracteres del texto
    return text[:200].strip() + "..." if len(text) > 200 else text.strip()


def process_pdf(pdf_path: Path, out_dir: Path, ocr_lang: str, min_chars: int) -> None:
    """
    Genera data/interim/<categoria>/<libro>.pages.jsonl
    """
    source_id = sha256_file(pdf_path)
    
    try:
        # Abrir documento
        doc = fitz.open(pdf_path)
        # Intentar configurar capas si está disponible (versiones más nuevas de PyMuPDF)
        try:
            doc.set_layer_config(fitz.LAYER_CONFIG_DEFAULT)
        except AttributeError:
            # Versión antigua de PyMuPDF sin set_layer_config
            pass
    except Exception as e:
        print(f"[WARNING] Error abriendo {pdf_path.name}: {e}")
        # Intentar abrir sin configuración de capas
        try:
            doc = fitz.open(pdf_path)
        except Exception as e2:
            print(f"[ERROR] No se pudo abrir {pdf_path.name}: {e2}")
            return
    
    meta = doc.metadata or {}
    doc_title = meta.get("title") or pdf_path.stem

    toc = get_toc_map(doc)

    page_texts_raw: List[str] = []
    page_lines: List[List[str]] = []

    # 1) Extrae texto por página (con OCR si no hay texto)
    for i, page in enumerate(doc, start=1):
        try:
            # Usar método más robusto para extraer texto
            txt = ""
            try:
                txt = page.get_text("text") or ""
            except Exception as text_e:
                print(f"[WARNING] Error extrayendo texto página {i} de {pdf_path.name}: {text_e}")
                # Intentar método alternativo
                try:
                    txt = page.get_text("words") or ""
                    if isinstance(txt, list):
                        txt = " ".join([word[4] for word in txt if len(word) > 4])
                except Exception:
                    txt = ""
            
            if len(txt.strip()) < min_chars:
                # fallback OCR (si está habilitado y hay poco texto)
                try:
                    ocr_txt = page_image_ocr(page, ocr_lang=ocr_lang)
                    if len(ocr_txt.strip()) > len(txt.strip()):
                        txt = ocr_txt
                except Exception as ocr_e:
                    print(f"[WARNING] OCR falló en página {i} de {pdf_path.name}: {ocr_e}")
            page_texts_raw.append(txt)
            page_lines.append([ln.strip() for ln in txt.splitlines() if ln.strip()])
        except Exception as page_e:
            print(f"[WARNING] Error procesando página {i} de {pdf_path.name}: {page_e}")
            # Agregar página vacía para mantener consistencia
            page_texts_raw.append("")
            page_lines.append([])

    # 2) Detecta headers/footers repetidos
    hf = detect_repeat_headers_footers(page_lines)

    # 3) Limpia y normaliza páginas
    cleaned_pages: List[str] = []
    for raw in page_texts_raw:
        t = strip_header_footer(raw, hf["headers"], hf["footers"])
        t = normalize_text(t)
        cleaned_pages.append(t)

    # 4) Extrae abstract de la primera página
    abstract = extract_abstract(cleaned_pages[0]) if cleaned_pages else ""

    # 5) Arma registros por página
    records: List[Dict[str, Any]] = []
    for i, txt in enumerate(cleaned_pages, start=1):
        quality_flags = []
        if not txt:
            quality_flags.append("empty")
        # marca si el OCR pudo haber intervenido (heurístico: si original estaba vacío)
        if len(page_texts_raw[i - 1].strip()) < min_chars:
            quality_flags.append("ocr_or_low_text")

        rec = {
            "pipeline_version": "extract_pdf@1.0.0",
            "doc_title": doc_title,
            "authors": [a.strip() for a in (meta.get("author") or "").split(";") if a.strip()],
            "page_num_real": i,
            "page_num_logical": str(i),  # ajusta si manejas romanos/offset
            "toc_path": section_for_page(toc, i),
            "source_id": source_id,
            "text": txt,
            "quality_flags": quality_flags,
            "abstract": abstract if i == 1 else "",  # Solo en la primera página
            "source_category": get_source_category(pdf_path),
        }
        records.append(rec)

    # 6) Crea subcarpeta por categoría y escribe JSONL
    category = get_source_category(pdf_path)
    category_dir = out_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = category_dir / f"{pdf_path.stem}.pages.jsonl"
    write_jsonl(records, out_file)
    print(f"OK: {pdf_path.name} -> {out_file} ({len(records)} páginas, categoría: {category})")
    
    # Cerrar documento de forma segura
    try:
        doc.close()
    except Exception as e:
        print(f"[WARNING] Error cerrando {pdf_path.name}: {e}")


# ------------------ CLI ------------------

def main():
    parser = argparse.ArgumentParser(description="PDF -> interim pages.jsonl")
    parser.add_argument("--in", dest="in_dir", required=True, help="Carpeta con PDFs (p.ej., data/raw)")
    parser.add_argument("--out", dest="out_dir", required=True, help="Carpeta de salida (p.ej., data/interim)")
    parser.add_argument("--ocr-lang", default="eng", help="Idioma OCR Tesseract (spa|eng|...);")
    parser.add_argument("--min-chars", type=int, default=20, help="Umbral mínimo de caracteres para considerar que la página tiene texto")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    pdfs = sorted(list(in_dir.glob("**/*.pdf")))
    if not pdfs:
        print(f"No se encontraron PDFs en {in_dir}")
        return

    for pdf in pdfs:
        try:
            process_pdf(pdf, out_dir, ocr_lang=args.ocr_lang, min_chars=args.min_chars)
        except Exception as e:
            print(f"[ERROR] {pdf}: {e}")

if __name__ == "__main__":
    main()