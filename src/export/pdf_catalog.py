#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera un CSV con información de todos los PDFs procesados
- Nombre del archivo
- Título del documento
- Categoría de fuente
- Abstract
- Autores
- Número de páginas
- Fecha de procesamiento
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


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


def extract_better_title(records: List[Dict[str, Any]]) -> str:
    """Extrae un título mejor del documento"""
    if not records:
        return ""
    
    first_record = records[0]
    doc_title = first_record.get('doc_title', '')
    
    # Si el título es genérico, intenta extraer de la primera página con texto
    if not doc_title or len(doc_title) < 10:
        for record in records[:3]:  # Revisa las primeras 3 páginas
            text = record.get('text', '')
            if len(text) > 100:
                # Busca líneas que parezcan títulos
                lines = text.split('\n')
                for line in lines[:10]:  # Primeras 10 líneas
                    line = line.strip()
                    if (len(line) > 20 and len(line) < 200 and 
                        not line.isupper() and 
                        not line.startswith(('Page', 'Chapter', 'Section'))):
                        return clean_text(line, 200)
    
    return clean_text(doc_title, 200)


def extract_better_abstract(records: List[Dict[str, Any]]) -> str:
    """Extrae un abstract mejor del documento"""
    if not records:
        return ""
    
    # Busca abstract en las primeras páginas
    for record in records[:3]:
        abstract = record.get('abstract', '')
        if abstract and len(abstract) > 50:
            return clean_text(abstract, 800)
    
    # Si no hay abstract, toma las primeras líneas de la primera página
    first_record = records[0]
    text = first_record.get('text', '')
    if text:
        lines = text.split('\n')
        # Toma las primeras líneas que parezcan contenido
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
            return clean_text(' '.join(content_lines), 800)
    
    return ""


def process_jsonl_file(jsonl_path: Path) -> Dict[str, Any]:
    """Procesa un archivo JSONL y extrae información del PDF"""
    records = []
    
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    if not records:
        return None
    
    # Extrae información mejorada
    first_record = records[0]
    
    # Mejora el título
    doc_title = extract_better_title(records)
    
    # Mejora el abstract
    abstract = extract_better_abstract(records)
    
    # Limpia autores
    authors = first_record.get('authors', [])
    if isinstance(authors, list):
        authors_str = '; '.join([clean_text(author, 100) for author in authors if author])
    else:
        authors_str = clean_text(str(authors), 200)
    
    return {
        'filename': jsonl_path.stem.replace('.pages', ''),
        'doc_title': doc_title,
        'source_category': first_record.get('source_category', ''),
        'abstract': abstract,
        'authors': authors_str,
        'total_pages': len(records),
        'source_id': first_record.get('source_id', ''),
        'pipeline_version': first_record.get('pipeline_version', ''),
        'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }


def generate_pdf_catalog(interim_dir: Path, output_csv: Path) -> None:
    """Genera el catálogo CSV de todos los PDFs procesados"""
    
    # Busca todos los archivos JSONL en subcarpetas
    jsonl_files = list(interim_dir.glob("**/*.pages.jsonl"))
    
    if not jsonl_files:
        print(f"No se encontraron archivos JSONL en {interim_dir}")
        return
    
    print(f"Procesando {len(jsonl_files)} archivos JSONL...")
    
    catalog_data = []
    
    for jsonl_file in sorted(jsonl_files):
        try:
            pdf_info = process_jsonl_file(jsonl_file)
            if pdf_info:
                catalog_data.append(pdf_info)
                print(f"✓ {pdf_info['filename']} ({pdf_info['source_category']})")
        except Exception as e:
            print(f"✗ Error procesando {jsonl_file}: {e}")
    
    # Escribe el CSV
    if catalog_data:
        fieldnames = [
            'filename', 'doc_title', 'source_category', 'abstract', 
            'authors', 'total_pages', 'source_id', 'pipeline_version', 'processing_date'
        ]
        
        with output_csv.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(catalog_data)
        
        print(f"\n✅ Catálogo generado: {output_csv}")
        print(f"📊 Total de PDFs procesados: {len(catalog_data)}")
        
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
    parser = argparse.ArgumentParser(description="Genera catálogo CSV de PDFs procesados")
    parser.add_argument("--interim", default="data/interim", help="Directorio con archivos JSONL")
    parser.add_argument("--output", default="data/export/pdf_catalog.csv", help="Archivo CSV de salida")
    args = parser.parse_args()
    
    interim_dir = Path(args.interim)
    output_csv = Path(args.output)
    
    # Crea directorio de salida si no existe
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    generate_pdf_catalog(interim_dir, output_csv)


if __name__ == "__main__":
    main()
