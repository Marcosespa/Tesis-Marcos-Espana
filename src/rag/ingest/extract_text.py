#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TXT -> data/interim/*.chunks.jsonl
- Extrae texto de archivos .txt
- Divide en chunks lógicos por párrafos o líneas
- Limpia texto y detecta headers/footers repetidos
- Conserva metadatos: doc_title, chunk_num, source_id, source_category
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Set

# ------------------ Utilidades ------------------

def sha256_file(path: Path) -> str:
    """Genera hash SHA256 único del archivo"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def normalize_text(text: str) -> str:
    """
    Limpieza ligera del texto:
    - Une palabras partidas por guión al final de línea
    - Colapsa espacios múltiples
    - Preserva párrafos dobles y compacta líneas sueltas
    - Elimina caracteres de control
    """
    txt = text.replace("\r", "")
    
    # Une palabras partidas por guion + salto de línea
    txt = re.sub(r"(\w)-\n(\w)", r"\1\2", txt)
    
    # Convierte líneas sueltas en espacio, preserva párrafos dobles
    txt = re.sub(r"\n{3,}", "\n\n", txt)                # 3+ saltos -> 2
    txt = re.sub(r"([^\n])\n([^\n])", r"\1 \2", txt)    # línea simple -> espacio
    
    # Colapsa espacios y tabs
    txt = re.sub(r"[ \t]+", " ", txt)
    
    # Elimina caracteres de control excepto \n
    txt = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", txt)
    
    return txt.strip()


def detect_encoding(file_path: Path) -> str:
    """Detecta la codificación del archivo"""
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with file_path.open('r', encoding=encoding) as f:
                f.read()
            return encoding
        except UnicodeDecodeError:
            continue
    
    # Si nada funciona, usar utf-8 con errores ignorados
    return 'utf-8'


def detect_repeat_headers_footers(chunks_lines: List[List[str]]) -> Dict[str, Set[str]]:
    """
    Detecta headers y footers que se repiten en múltiples chunks
    """
    if not chunks_lines:
        return {"headers": set(), "footers": set()}
    
    first_counts, last_counts = {}, {}
    n = max(1, len(chunks_lines))

    for lines in chunks_lines:
        if not lines:
            continue
        first = lines[0].strip()
        last = lines[-1].strip()
        first_counts[first] = first_counts.get(first, 0) + 1
        last_counts[last] = last_counts.get(last, 0) + 1

    # Solo considera headers/footers si se repiten en >30% de chunks y son cortos
    headers = {l for l, c in first_counts.items() if c / n > 0.3 and len(l) <= 100 and l}
    footers = {l for l, c in last_counts.items() if c / n > 0.3 and len(l) <= 100 and l}
    
    return {"headers": headers, "footers": footers}


def strip_header_footer(text: str, headers: Set[str], footers: Set[str]) -> str:
    """Elimina headers y footers detectados del texto"""
    lines = [ln.strip() for ln in text.splitlines()]
    
    if lines and lines[0] in headers:
        lines = lines[1:]
    if lines and lines[-1] in footers:
        lines = lines[:-1]
    
    return "\n".join(lines)


def extract_abstract(text: str) -> str:
    """Extrae el abstract del texto (similar al script PDF)"""
    # Busca patrones comunes de abstract
    abstract_patterns = [
        r"Abstract\s*[:\-]?\s*(.+?)(?:\n\n|\n[A-Z]|$)",
        r"ABSTRACT\s*[:\-]?\s*(.+?)(?:\n\n|\n[A-Z]|$)",
        r"Resumen\s*[:\-]?\s*(.+?)(?:\n\n|\n[A-Z]|$)",
        r"RESUMEN\s*[:\-]?\s*(.+?)(?:\n\n|\n[A-Z]|$)",
        r"Summary\s*[:\-]?\s*(.+?)(?:\n\n|\n[A-Z]|$)",
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


def get_source_category(txt_path: Path) -> str:
    """Determina la categoría de fuente basada en la ruta del archivo"""
    path_str = str(txt_path).upper()
    
    if "ANNOTCTR" in path_str:
        return "ANNOTCTR"
    elif "MITRE" in path_str:
        return "MITRE"
    elif "OWASP" in path_str:
        return "OWASP"
    elif "SECURITYTOOLS" in path_str:
        return "SECURITYTOOLS"
    elif "AISECKG" in path_str:
        return "AISECKG"
    elif "NIST" in path_str:
        if "AI" in path_str:
            return "NIST_AI"
        elif "SP" in path_str:
            return "NIST_SP"
        else:
            return "NIST_OTHER"
    else:
        return "OTHER"


def extract_metadata(text: str, file_path: Path) -> Dict[str, Any]:
    """Extrae metadatos básicos del texto (similar al script PDF)"""
    lines = text.split('\n')
    
    # Busca título en las primeras líneas
    title = ""
    for i, line in enumerate(lines[:10]):
        line = line.strip()
        if len(line) > 10 and len(line) < 200:
            # Heurística: la primera línea larga podría ser el título
            if not title and line and not line.startswith(('http', 'www', '#', '*', '-')):
                title = line
                break
    
    if not title:
        title = file_path.stem
    
    # Busca autores
    authors = []
    author_patterns = [
        r"(?:Author|By|Written by)[:\s]+(.+)",
        r"^(.+?)(?:\s*-\s*|\s*,\s*)(?:Author|Writer)",
    ]
    
    for line in lines[:20]:
        for pattern in author_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                author_text = match.group(1).strip()
                if len(author_text) < 100:  # Evita líneas muy largas
                    authors.extend([a.strip() for a in re.split(r'[,&;]', author_text) if a.strip()])
                break
    
    return {
        "title": title,
        "authors": authors[:3]  # Limita a 3 autores
    }


def write_jsonl(records: List[Dict[str, Any]], out_path: Path) -> None:
    """Escribe registros en formato JSONL"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ------------------ Pipeline por archivo TXT ------------------

def process_txt(txt_path: Path, out_dir: Path, min_chars: int) -> None:
    """
    Procesa un archivo TXT y genera un JSONL (similar al PDF pero con documento completo)
    """
    source_id = sha256_file(txt_path)
    
    try:
        # Detecta y lee el archivo con la codificación apropiada
        encoding = detect_encoding(txt_path)
        
        with txt_path.open('r', encoding=encoding, errors='replace') as f:
            raw_text = f.read()
            
    except Exception as e:
        print(f"[ERROR] No se pudo leer {txt_path.name}: {e}")
        return
    
    if not raw_text.strip():
        print(f"[WARNING] Archivo vacío: {txt_path.name}")
        return
    
    # Extrae metadatos
    metadata = extract_metadata(raw_text, txt_path)
    
    # Normaliza texto completo
    normalized_text = normalize_text(raw_text)
    
    if len(normalized_text.strip()) < min_chars:
        print(f"[WARNING] Texto muy corto después de normalizar: {txt_path.name}")
        return
    
    # Extrae abstract del texto completo
    abstract = extract_abstract(normalized_text)
    
    # Para detectar headers/footers, simula "páginas" dividiendo por párrafos
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', normalized_text) if p.strip()]
    if not paragraphs:
        paragraphs = [normalized_text]
    
    # Convierte párrafos en "líneas" para detección de headers/footers
    text_lines = []
    for paragraph in paragraphs:
        lines = [ln.strip() for ln in paragraph.splitlines() if ln.strip()]
        if lines:
            text_lines.append(lines)
    
    # Detecta headers/footers repetidos
    hf = detect_repeat_headers_footers(text_lines)
    
    # Limpia el texto completo
    cleaned_text = strip_header_footer(normalized_text, hf["headers"], hf["footers"])
    
    if len(cleaned_text.strip()) < min_chars:
        print(f"[WARNING] Texto muy corto después de limpiar: {txt_path.name}")
        return
    
    # Detecta flags de calidad
    quality_flags = []
    
    if not cleaned_text.strip():
        quality_flags.append("empty")
    
    if len(cleaned_text.strip()) < min_chars * 2:
        quality_flags.append("short_text")
    
    if len(re.findall(r'[^\w\s\n\.\,\;\:\!\?\-\(\)]', cleaned_text)) / max(len(cleaned_text), 1) > 0.2:
        quality_flags.append("high_special_chars")
    
    # Crea registro único (similar a una "página" en PDF)
    rec = {
        "pipeline_version": "extract_text@1.0.0",
        "doc_title": metadata["title"],
        "authors": metadata["authors"],
        "page_num_real": 1,  # Archivo completo como "página 1"
        "page_num_logical": "1",
        "toc_path": [],  # No hay TOC en archivos de texto planos
        "source_id": source_id,
        "text": cleaned_text,
        "quality_flags": quality_flags,
        "abstract": abstract,
        "source_category": get_source_category(txt_path),
    }
    
    # Crea subcarpeta por categoría y escribe JSONL
    category = get_source_category(txt_path)
    category_dir = out_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = category_dir / f"{txt_path.stem}.pages.jsonl"
    write_jsonl([rec], out_file)
    
    print(f"OK: {txt_path.name} -> {out_file} (1 registro, categoría: {category})")


# ------------------ CLI ------------------

def main():
    parser = argparse.ArgumentParser(description="TXT -> interim pages.jsonl")
    parser.add_argument("--in", dest="in_dir", required=True, help="Carpeta con archivos TXT (p.ej., data/raw)")
    parser.add_argument("--out", dest="out_dir", required=True, help="Carpeta de salida (p.ej., data/interim)")
    parser.add_argument("--min-chars", type=int, default=50, help="Mínimo caracteres para considerar un archivo válido")
    
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    # Busca archivos .txt recursivamente
    txt_files = sorted(list(in_dir.glob("**/*.txt")))
    
    if not txt_files:
        print(f"No se encontraron archivos .txt en {in_dir}")
        return

    print(f"Encontrados {len(txt_files)} archivos .txt para procesar")
    
    successful = 0
    failed = 0
    
    for txt_file in txt_files:
        try:
            print(f"\nProcesando: {txt_file.relative_to(in_dir)}")
            process_txt(txt_file, out_dir, min_chars=args.min_chars)
            successful += 1
        except Exception as e:
            print(f"[ERROR] {txt_file.name}: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"RESUMEN:")
    print(f"  Exitosos: {successful}")
    print(f"  Fallidos: {failed}")
    print(f"  Total: {len(txt_files)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()