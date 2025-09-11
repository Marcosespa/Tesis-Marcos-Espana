#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera un catálogo CSV híbrido combinando:
- Metadatos nativos de PDFs (título, autor, fecha)
- Información procesada de JSONL (abstract, categoría, calidad)
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import fitz  # PyMuPDF


def extract_pdf_metadata(pdf_path: Path) -> Dict[str, Any]:
    """Extrae metadatos nativos del PDF"""
    try:
        doc = fitz.open(pdf_path)
        meta = doc.metadata or {}
        doc.close()
        
        return {
            'title': meta.get('title', '').strip(),
            'author': meta.get('author', '').strip(),
            'subject': meta.get('subject', '').strip(),
            'creator': meta.get('creator', '').strip(),
            'producer': meta.get('producer', '').strip(),
            'creation_date': meta.get('creationDate', ''),
            'modification_date': meta.get('modDate', ''),
            'page_count': doc.page_count,
            'file_size': pdf_path.stat().st_size
        }
    except Exception as e:
        print(f"[WARNING] Error extrayendo metadatos de {pdf_path.name}: {e}")
        return {}


def extract_jsonl_info(jsonl_path: Path) -> Dict[str, Any]:
    """Extrae información procesada del JSONL"""
    try:
        records = []
        with jsonl_path.open('r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        
        if not records:
            return {}
        
        first_record = records[0]
        
        # Extrae abstract mejorado
        abstract = ""
        for record in records[:3]:
            if record.get('abstract') and len(record.get('abstract', '')) > 50:
                abstract = record.get('abstract', '')
                break
        
        # Si no hay abstract, toma contenido de la primera página
        if not abstract:
            text = first_record.get('text', '')
            if text:
                lines = text.split('\n')
                content_lines = []
                for line in lines[:15]:
                    line = line.strip()
                    if (len(line) > 30 and 
                        not line.startswith(('Page', 'Chapter', 'Section', 'Table')) and
                        not line.isupper()):
                        content_lines.append(line)
                        if len(' '.join(content_lines)) > 200:
                            break
                if content_lines:
                    abstract = ' '.join(content_lines)
        
        return {
            'abstract': abstract[:800] + "..." if len(abstract) > 800 else abstract,
            'source_category': first_record.get('source_category', ''),
            'pipeline_version': first_record.get('pipeline_version', ''),
            'source_id': first_record.get('source_id', ''),
            'total_pages': len(records),
            'quality_flags': first_record.get('quality_flags', [])
        }
    except Exception as e:
        print(f"[WARNING] Error procesando JSONL {jsonl_path.name}: {e}")
        return {}


def clean_text(text: str, max_length: int = 500) -> str:
    """Limpia y trunca texto para mejor legibilidad"""
    if not text:
        return ""
    
    # Limpia espacios múltiples y saltos de línea
    text = ' '.join(text.split())
    
    # Trunca si es muy largo
    if len(text) > max_length:
        text = text[:max_length].rsplit(' ', 1)[0] + "..."
    
    return text


def get_source_category_from_path(pdf_path: Path) -> str:
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


def process_pdf_catalog(raw_dir: Path, interim_dir: Path, output_csv: Path) -> None:
    """Genera catálogo híbrido combinando PDFs y JSONL"""
    
    # Busca todos los PDFs
    pdf_files = list(raw_dir.glob("**/*.pdf"))
    
    if not pdf_files:
        print(f"No se encontraron PDFs en {raw_dir}")
        return
    
    print(f"Procesando {len(pdf_files)} PDFs...")
    
    catalog_data = []
    
    for pdf_path in sorted(pdf_files):
        try:
            # Extrae metadatos del PDF
            pdf_meta = extract_pdf_metadata(pdf_path)
            
            # Busca JSONL correspondiente
            jsonl_path = None
            jsonl_info = {}
            
            # Busca en todas las subcarpetas de interim
            for jsonl_file in interim_dir.glob("**/*.pages.jsonl"):
                if jsonl_file.stem.replace('.pages', '') == pdf_path.stem:
                    jsonl_path = jsonl_file
                    jsonl_info = extract_jsonl_info(jsonl_file)
                    break
            
            # Si no encuentra JSONL, usa solo metadatos del PDF
            if not jsonl_path:
                print(f"[WARNING] No se encontró JSONL para {pdf_path.name}")
                jsonl_info = {
                    'source_category': get_source_category_from_path(pdf_path),
                    'abstract': '',
                    'total_pages': pdf_meta.get('page_count', 0),
                    'quality_flags': ['not_processed']
                }
            
            # Combina información
            title = pdf_meta.get('title', '') or pdf_path.stem
            if not title or len(title) < 10:
                title = pdf_path.stem
            
            authors = pdf_meta.get('author', '')
            if not authors and jsonl_info.get('authors'):
                authors = jsonl_info.get('authors', '')
            
            catalog_entry = {
                'filename': pdf_path.stem,
                'file_path': str(pdf_path.relative_to(raw_dir)),
                'doc_title': clean_text(title, 200),
                'authors': clean_text(authors, 300),
                'subject': clean_text(pdf_meta.get('subject', ''), 200),
                'creator': clean_text(pdf_meta.get('creator', ''), 100),
                'producer': clean_text(pdf_meta.get('producer', ''), 100),
                'creation_date': pdf_meta.get('creation_date', ''),
                'modification_date': pdf_meta.get('modification_date', ''),
                'file_size_mb': round(pdf_meta.get('file_size', 0) / (1024 * 1024), 2),
                'source_category': jsonl_info.get('source_category', 'UNKNOWN'),
                'abstract': clean_text(jsonl_info.get('abstract', ''), 800),
                'total_pages': jsonl_info.get('total_pages', pdf_meta.get('page_count', 0)),
                'source_id': jsonl_info.get('source_id', ''),
                'pipeline_version': jsonl_info.get('pipeline_version', ''),
                'quality_flags': '; '.join(jsonl_info.get('quality_flags', [])),
                'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'has_jsonl': jsonl_path is not None
            }
            
            catalog_data.append(catalog_entry)
            print(f"✓ {pdf_path.name} ({catalog_entry['source_category']})")
            
        except Exception as e:
            print(f"✗ Error procesando {pdf_path.name}: {e}")
    
    # Escribe el CSV
    if catalog_data:
        fieldnames = [
            'filename', 'file_path', 'doc_title', 'authors', 'subject',
            'creator', 'producer', 'creation_date', 'modification_date',
            'file_size_mb', 'source_category', 'abstract', 'total_pages',
            'source_id', 'pipeline_version', 'quality_flags', 'processing_date', 'has_jsonl'
        ]
        
        with output_csv.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(catalog_data)
        
        print(f"\n✅ Catálogo híbrido generado: {output_csv}")
        print(f"📊 Total de PDFs: {len(catalog_data)}")
        
        # Estadísticas
        processed = sum(1 for item in catalog_data if item['has_jsonl'])
        print(f"📈 PDFs procesados: {processed}")
        print(f"📈 PDFs sin procesar: {len(catalog_data) - processed}")
        
        # Estadísticas por categoría
        categories = {}
        for item in catalog_data:
            cat = item['source_category']
            categories[cat] = categories.get(cat, 0) + 1
        
        print("\n📈 Distribución por categoría:")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count} PDFs")
    else:
        print("❌ No se pudo procesar ningún archivo")


def main():
    parser = argparse.ArgumentParser(description="Genera catálogo híbrido de PDFs")
    parser.add_argument("--raw", default="data/raw", help="Directorio con PDFs originales")
    parser.add_argument("--interim", default="data/interim", help="Directorio con archivos JSONL")
    parser.add_argument("--output", default="data/export/pdf_catalog_hybrid.csv", help="Archivo CSV de salida")
    args = parser.parse_args()
    
    raw_dir = Path(args.raw)
    interim_dir = Path(args.interim)
    output_csv = Path(args.output)
    
    # Crea directorio de salida si no existe
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    process_pdf_catalog(raw_dir, interim_dir, output_csv)


if __name__ == "__main__":
    main()
