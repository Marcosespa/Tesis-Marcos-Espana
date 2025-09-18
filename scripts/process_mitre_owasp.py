#!/usr/bin/env python3
"""
Script para procesar datos de MITRE ATT&CK y OWASP y convertirlos a formato compatible con el pipeline
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any

def process_mitre_data():
    """Procesa los datos de MITRE ATT&CK y los convierte a formato pages.jsonl"""
    print("🔄 Procesando datos de MITRE ATT&CK...")
    
    # Directorios
    mitre_dir = Path("data/raw/MITRE")
    interim_dir = Path("data/interim/MITRE")
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    # Procesar cada dataset
    datasets = ["enterprise", "mobile", "ics"]
    total_docs = 0
    
    for dataset_name in datasets:
        text_dir = mitre_dir / f"{dataset_name}_text"
        
        if not text_dir.exists():
            print(f"⚠️  Directorio no encontrado: {text_dir}")
            continue
        
        print(f"📝 Procesando {dataset_name}...")
        
        # Encontrar archivos .txt
        txt_files = list(text_dir.glob("*.txt"))
        
        if not txt_files:
            print(f"  ⚠️  No se encontraron archivos .txt en {dataset_name}")
            continue
        
        print(f"  📄 Encontrados {len(txt_files)} archivos")
        
        # Crear archivo pages.jsonl para este dataset
        pages_file = interim_dir / f"{dataset_name}.pages.jsonl"
        
        dataset_docs = 0
        
        with open(pages_file, 'w', encoding='utf-8') as f:
            for txt_file in txt_files:
                try:
                    # Leer contenido del archivo
                    with open(txt_file, 'r', encoding='utf-8') as txt_f:
                        content = txt_f.read()
                    
                    # Crear entrada para pages.jsonl
                    page_data = {
                        "source_id": f"mitre_{dataset_name}_{txt_file.stem}",
                        "source_type": "mitre_attack",
                        "source_file": txt_file.name,
                        "page_number": 1,
                        "text": content,
                        "metadata": {
                            "dataset": dataset_name,
                            "document_type": "attack_technique",
                            "source": "mitre_attack",
                            "file_id": txt_file.stem,
                            "has_annotations": False,
                            "word_count": len(content.split()),
                            "char_count": len(content)
                        }
                    }
                    
                    f.write(json.dumps(page_data, ensure_ascii=False) + '\n')
                    dataset_docs += 1
                    
                except Exception as e:
                    print(f"  ❌ Error procesando {txt_file.name}: {e}")
                    continue
        
        print(f"  ✅ {dataset_name}: {dataset_docs} documentos procesados")
        total_docs += dataset_docs
    
    print(f"\n📊 Total de documentos MITRE procesados: {total_docs}")
    print(f"📁 Archivos guardados en: {interim_dir}")
    
    return total_docs

def process_owasp_data():
    """Procesa los datos de OWASP y los convierte a formato pages.jsonl"""
    print("\n🔄 Procesando datos de OWASP...")
    
    # Directorios
    owasp_dir = Path("data/raw/OWASP")
    interim_dir = Path("data/interim/OWASP")
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    # Encontrar archivos .txt
    txt_files = list(owasp_dir.glob("*.txt"))
    
    if not txt_files:
        print("❌ No se encontraron archivos .txt de OWASP")
        return 0
    
    print(f"📄 Encontrados {len(txt_files)} archivos de OWASP")
    
    # Crear archivo pages.jsonl
    pages_file = interim_dir / "all.pages.jsonl"
    
    total_docs = 0
    
    with open(pages_file, 'w', encoding='utf-8') as f:
        for txt_file in txt_files:
            try:
                # Leer contenido del archivo
                with open(txt_file, 'r', encoding='utf-8') as txt_f:
                    content = txt_f.read()
                
                # Leer metadata si existe
                metadata_file = owasp_dir / f"{txt_file.stem}_metadata.json"
                metadata = {}
                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as mf:
                        metadata = json.load(mf)
                
                # Crear entrada para pages.jsonl
                page_data = {
                    "source_id": f"owasp_{txt_file.stem}",
                    "source_type": "owasp",
                    "source_file": txt_file.name,
                    "page_number": 1,
                    "text": content,
                    "metadata": {
                        "document_type": "security_standard",
                        "source": "owasp",
                        "file_id": txt_file.stem,
                        "has_annotations": False,
                        "word_count": len(content.split()),
                        "char_count": len(content),
                        **metadata
                    }
                }
                
                f.write(json.dumps(page_data, ensure_ascii=False) + '\n')
                total_docs += 1
                
            except Exception as e:
                print(f"❌ Error procesando {txt_file.name}: {e}")
                continue
    
    print(f"✅ OWASP: {total_docs} documentos procesados")
    print(f"📁 Archivos guardados en: {interim_dir}")
    
    return total_docs

def create_chunks():
    """Crea chunks a partir de los archivos pages.jsonl de MITRE y OWASP"""
    print("\n🔄 Creando chunks de MITRE y OWASP...")
    
    # Directorios
    mitre_interim_dir = Path("data/interim/MITRE")
    owasp_interim_dir = Path("data/interim/OWASP")
    mitre_chunks_dir = Path("data/chunks/MITRE")
    owasp_chunks_dir = Path("data/chunks/OWASP")
    
    mitre_chunks_dir.mkdir(parents=True, exist_ok=True)
    owasp_chunks_dir.mkdir(parents=True, exist_ok=True)
    
    total_chunks = 0
    
    # Procesar MITRE
    if mitre_interim_dir.exists():
        pages_files = list(mitre_interim_dir.glob("*.pages.jsonl"))
        
        for pages_file in pages_files:
            print(f"📝 Procesando chunks de MITRE {pages_file.name}...")
            
            chunks_data = []
            
            with open(pages_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    page_data = json.loads(line)
                    
                    # Crear chunks basados en palabras (aproximadamente 400 palabras por chunk)
                    text = page_data.get('text', '')
                    words = text.split()
                    
                    # Dividir en chunks de ~400 palabras
                    chunk_size = 400
                    overlap = 50
                    
                    for i in range(0, len(words), chunk_size - overlap):
                        chunk_words = words[i:i + chunk_size]
                        chunk_text = ' '.join(chunk_words)
                        
                        chunk_data = {
                            "source_id": page_data.get('source_id', ''),
                            "source_type": "mitre_attack",
                            "source_file": page_data.get('source_file', ''),
                            "chunk_id": f"{page_data.get('source_id', '')}_chunk_{i//(chunk_size - overlap)}",
                            "chunk_index": i // (chunk_size - overlap),
                            "text": chunk_text,
                            "word_count": len(chunk_words),
                            "metadata": {
                                **page_data.get('metadata', {}),
                                "chunk_start_word": i,
                                "chunk_end_word": min(i + chunk_size, len(words)),
                                "has_annotations": False
                            }
                        }
                        
                        chunks_data.append(chunk_data)
                        total_chunks += 1
            
            # Guardar chunks del archivo
            chunks_file = mitre_chunks_dir / pages_file.name.replace('.pages.jsonl', '.chunks.jsonl')
            with open(chunks_file, 'w', encoding='utf-8') as f:
                for chunk in chunks_data:
                    json.dump(chunk, f, ensure_ascii=False)
                    f.write('\n')
            
            print(f"  ✅ {pages_file.name}: {len(chunks_data)} chunks creados")
    
    # Procesar OWASP
    if owasp_interim_dir.exists():
        pages_files = list(owasp_interim_dir.glob("*.pages.jsonl"))
        
        for pages_file in pages_files:
            print(f"📝 Procesando chunks de OWASP {pages_file.name}...")
            
            chunks_data = []
            
            with open(pages_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    page_data = json.loads(line)
                    
                    # Crear chunks basados en palabras (aproximadamente 400 palabras por chunk)
                    text = page_data.get('text', '')
                    words = text.split()
                    
                    # Dividir en chunks de ~400 palabras
                    chunk_size = 400
                    overlap = 50
                    
                    for i in range(0, len(words), chunk_size - overlap):
                        chunk_words = words[i:i + chunk_size]
                        chunk_text = ' '.join(chunk_words)
                        
                        chunk_data = {
                            "source_id": page_data.get('source_id', ''),
                            "source_type": "owasp",
                            "source_file": page_data.get('source_file', ''),
                            "chunk_id": f"{page_data.get('source_id', '')}_chunk_{i//(chunk_size - overlap)}",
                            "chunk_index": i // (chunk_size - overlap),
                            "text": chunk_text,
                            "word_count": len(chunk_words),
                            "metadata": {
                                **page_data.get('metadata', {}),
                                "chunk_start_word": i,
                                "chunk_end_word": min(i + chunk_size, len(words)),
                                "has_annotations": False
                            }
                        }
                        
                        chunks_data.append(chunk_data)
                        total_chunks += 1
            
            # Guardar chunks del archivo
            chunks_file = owasp_chunks_dir / pages_file.name.replace('.pages.jsonl', '.chunks.jsonl')
            with open(chunks_file, 'w', encoding='utf-8') as f:
                for chunk in chunks_data:
                    json.dump(chunk, f, ensure_ascii=False)
                    f.write('\n')
            
            print(f"  ✅ {pages_file.name}: {len(chunks_data)} chunks creados")
    
    print(f"✅ Total de chunks creados: {total_chunks}")
    return total_chunks

def analyze_data():
    """Analiza los datos procesados"""
    print("\n🔍 Analizando datos procesados...")
    
    # Analizar MITRE
    mitre_interim_dir = Path("data/interim/MITRE")
    if mitre_interim_dir.exists():
        pages_files = list(mitre_interim_dir.glob("*.pages.jsonl"))
        mitre_docs = 0
        mitre_words = 0
        
        for pages_file in pages_files:
            with open(pages_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        page_data = json.loads(line)
                        mitre_docs += 1
                        mitre_words += len(page_data.get('text', '').split())
        
        print(f"📊 MITRE ATT&CK:")
        print(f"  Documentos: {mitre_docs:,}")
        print(f"  Palabras: {mitre_words:,}")
        print(f"  Promedio palabras/doc: {mitre_words // mitre_docs if mitre_docs > 0 else 0:,}")
    
    # Analizar OWASP
    owasp_interim_dir = Path("data/interim/OWASP")
    if owasp_interim_dir.exists():
        pages_files = list(owasp_interim_dir.glob("*.pages.jsonl"))
        owasp_docs = 0
        owasp_words = 0
        
        for pages_file in pages_files:
            with open(pages_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        page_data = json.loads(line)
                        owasp_docs += 1
                        owasp_words += len(page_data.get('text', '').split())
        
        print(f"📊 OWASP:")
        print(f"  Documentos: {owasp_docs:,}")
        print(f"  Palabras: {owasp_words:,}")
        print(f"  Promedio palabras/doc: {owasp_words // owasp_docs if owasp_docs > 0 else 0:,}")

if __name__ == "__main__":
    # Cambiar al directorio del proyecto
    os.chdir("/Users/marcosespana/Desktop/U/DatosTesis")
    
    # Procesar MITRE
    mitre_docs = process_mitre_data()
    
    # Procesar OWASP
    owasp_docs = process_owasp_data()
    
    # Crear chunks
    total_chunks = create_chunks()
    
    # Analizar datos
    analyze_data()
    
    print(f"\n🎉 Proceso completado!")
    print(f"📊 Resumen:")
    print(f"  MITRE ATT&CK: {mitre_docs} documentos")
    print(f"  OWASP: {owasp_docs} documentos")
    print(f"  Total de chunks: {total_chunks}")
