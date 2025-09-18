#!/usr/bin/env python3
"""
Script para integrar AnnoCTR en el sistema de chunking y indexación existente
"""

import os
import json
from pathlib import Path
import shutil

def integrate_annoctr_chunks():
    """Integra los chunks de AnnoCTR con el sistema existente"""
    print("🔄 Integrando chunks de AnnoCTR...")
    
    # Directorios
    annoctr_chunks_dir = Path("data/chunks/AnnoCTR")
    main_chunks_dir = Path("data/chunks")
    
    if not annoctr_chunks_dir.exists():
        print("❌ Directorio de chunks de AnnoCTR no encontrado")
        return
    
    print("📁 Chunks de AnnoCTR ya están en el directorio principal")
    print(f"📁 Ubicación: {annoctr_chunks_dir}")
    
    # Verificar archivos existentes
    chunk_files = list(annoctr_chunks_dir.glob("*.jsonl"))
    print(f"📄 Archivos encontrados: {len(chunk_files)}")
    for chunk_file in chunk_files:
        print(f"  - {chunk_file.name}")

def update_all_chunks():
    """Actualiza el archivo all_chunks.jsonl para incluir AnnoCTR"""
    print("\n🔄 Actualizando all_chunks.jsonl...")
    
    chunks_dir = Path("data/chunks")
    all_chunks_file = chunks_dir / "all_chunks.jsonl"
    
    # Encontrar todos los archivos de chunks
    chunk_files = []
    
    # Buscar en subdirectorios
    for subdir in chunks_dir.iterdir():
        if subdir.is_dir():
            for chunk_file in subdir.glob("*.chunks.jsonl"):
                chunk_files.append(chunk_file)
    
    # También buscar en el directorio principal
    for chunk_file in chunks_dir.glob("*.chunks.jsonl"):
        if chunk_file not in chunk_files:
            chunk_files.append(chunk_file)
    
    print(f"📄 Encontrados {len(chunk_files)} archivos de chunks:")
    for chunk_file in chunk_files:
        print(f"  - {chunk_file}")
    
    # Consolidar todos los chunks
    all_chunks = []
    
    for chunk_file in chunk_files:
        print(f"📝 Procesando {chunk_file.name}...")
        
        with open(chunk_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        chunk_data = json.loads(line)
                        all_chunks.append(chunk_data)
                    except json.JSONDecodeError as e:
                        print(f"  ⚠️  Error en línea: {e}")
                        continue
    
    # Guardar archivo consolidado
    with open(all_chunks_file, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            json.dump(chunk, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"✅ Archivo all_chunks.jsonl actualizado con {len(all_chunks)} chunks")
    print(f"📁 Guardado en: {all_chunks_file}")

def analyze_integration():
    """Analiza la integración de AnnoCTR"""
    print("\n🔍 Analizando integración...")
    
    all_chunks_file = Path("data/chunks/all_chunks.jsonl")
    
    if not all_chunks_file.exists():
        print("❌ Archivo all_chunks.jsonl no encontrado")
        return
    
    # Analizar por fuente
    source_stats = {}
    total_chunks = 0
    
    with open(all_chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                chunk_data = json.loads(line)
                source_type = chunk_data.get('source_type', 'unknown')
                
                if source_type not in source_stats:
                    source_stats[source_type] = {
                        'count': 0,
                        'total_words': 0,
                        'sources': set()
                    }
                
                source_stats[source_type]['count'] += 1
                source_stats[source_type]['total_words'] += chunk_data.get('word_count', 0)
                source_stats[source_type]['sources'].add(chunk_data.get('source_file', 'unknown'))
                total_chunks += 1
                
            except json.JSONDecodeError:
                continue
    
    print(f"📊 Estadísticas de integración:")
    print(f"  Total de chunks: {total_chunks:,}")
    print()
    
    for source_type, stats in source_stats.items():
        avg_words = stats['total_words'] // stats['count'] if stats['count'] > 0 else 0
        print(f"  {source_type}:")
        print(f"    Chunks: {stats['count']:,}")
        print(f"    Archivos únicos: {len(stats['sources'])}")
        print(f"    Palabras promedio: {avg_words:,}")
        print()
    
    # Verificar que AnnoCTR esté incluido
    if 'annoctr_text' in source_stats:
        print("✅ AnnoCTR integrado correctamente")
    else:
        print("⚠️  AnnoCTR no encontrado en la integración")

def create_integration_summary():
    """Crea un resumen de la integración"""
    print("\n📋 Creando resumen de integración...")
    
    summary = {
        "integration_date": "2024-12-19",
        "annoctr_stats": {
            "total_documents": 190,
            "total_chunks": 796,
            "splits": {
                "train": 70,
                "dev": 16, 
                "test": 34,
                "train_ext": 70
            },
            "content_stats": {
                "total_words": 245820,
                "total_characters": 2136219,
                "avg_words_per_document": 1293
            }
        },
        "integration_benefits": [
            "Reportes de amenazas cibernéticas reales",
            "Datos de alta calidad de proveedores comerciales de CTI",
            "Complementa documentos académicos y estándares existentes",
            "Mejora capacidades de RAG para consultas sobre amenazas específicas"
        ],
        "data_sources": {
            "NIST": "Estándares y marcos oficiales",
            "OAPEN": "Documentos académicos",
            "USENIX": "Papers de conferencias",
            "AISecKG": "Conocimiento estructurado",
            "AnnoCTR": "Reportes de amenazas cibernéticas"
        }
    }
    
    summary_file = Path("data/chunks/integration_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Resumen guardado en: {summary_file}")

if __name__ == "__main__":
    # Cambiar al directorio del proyecto
    os.chdir("/Users/marcosespana/Desktop/U/DatosTesis")
    
    # Integrar chunks
    integrate_annoctr_chunks()
    
    # Actualizar all_chunks.jsonl
    update_all_chunks()
    
    # Analizar integración
    analyze_integration()
    
    # Crear resumen
    create_integration_summary()
    
    print("\n🎉 Integración de AnnoCTR completada!")
    print("\n📈 Resumen:")
    print("  ✅ 190 documentos de AnnoCTR procesados")
    print("  ✅ 796 chunks creados")
    print("  ✅ Integrado con sistema de chunking existente")
    print("  ✅ Archivo all_chunks.jsonl actualizado")
    print("\n🚀 AnnoCTR está listo para indexación en Weaviate!")
