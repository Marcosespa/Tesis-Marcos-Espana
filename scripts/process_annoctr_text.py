#!/usr/bin/env python3
"""
Script para procesar archivos de texto de AnnoCTR y convertirlos a formato compatible con el pipeline
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any

def process_annoctr_text_files():
    """Procesa los archivos de texto de AnnoCTR y los convierte a formato pages.jsonl"""
    print("🔄 Procesando archivos de texto de AnnoCTR...")
    
    # Directorios
    annoctr_dir = Path("data/raw/AnnoCTR")
    interim_dir = Path("data/interim/AnnoCTR")
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    # Procesar cada split
    splits = ["train", "dev", "test", "train_ext"]
    
    total_docs = 0
    
    for split in splits:
        split_dir = annoctr_dir / split
        
        if not split_dir.exists():
            print(f"⚠️  Directorio no encontrado: {split_dir}")
            continue
        
        print(f"📝 Procesando split: {split}")
        
        # Encontrar archivos .txt
        txt_files = list(split_dir.glob("*.txt"))
        
        if not txt_files:
            print(f"  ⚠️  No se encontraron archivos .txt en {split}")
            continue
        
        print(f"  📄 Encontrados {len(txt_files)} archivos")
        
        # Crear archivo pages.jsonl para este split
        pages_file = interim_dir / f"{split}.pages.jsonl"
        
        split_docs = 0
        
        with open(pages_file, 'w', encoding='utf-8') as f:
            for txt_file in txt_files:
                try:
                    # Leer contenido del archivo
                    with open(txt_file, 'r', encoding='utf-8') as txt_f:
                        content = txt_f.read()
                    
                    # Crear entrada para pages.jsonl
                    page_data = {
                        "source_id": f"annoctr_{split}_{txt_file.stem}",
                        "source_type": "annoctr_text",
                        "source_file": txt_file.name,
                        "page_number": 1,
                        "text": content,
                        "metadata": {
                            "split": split,
                            "document_type": "cyber_threat_report",
                            "source": "annoctr_text",
                            "file_id": txt_file.stem,
                            "has_annotations": False,
                            "word_count": len(content.split()),
                            "char_count": len(content)
                        }
                    }
                    
                    f.write(json.dumps(page_data, ensure_ascii=False) + '\n')
                    split_docs += 1
                    
                except Exception as e:
                    print(f"  ❌ Error procesando {txt_file.name}: {e}")
                    continue
        
        print(f"  ✅ {split}: {split_docs} documentos procesados")
        total_docs += split_docs
    
    print(f"\n📊 Total de documentos procesados: {total_docs}")
    print(f"📁 Archivos guardados en: {interim_dir}")
    
    return total_docs

def create_annoctr_chunks():
    """Crea chunks a partir de los archivos pages.jsonl de AnnoCTR"""
    print("\n🔄 Creando chunks de AnnoCTR...")
    
    interim_dir = Path("data/interim/AnnoCTR")
    chunks_dir = Path("data/chunks/AnnoCTR")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    # Archivos pages.jsonl a procesar
    pages_files = list(interim_dir.glob("*.pages.jsonl"))
    
    if not pages_files:
        print("❌ No se encontraron archivos pages.jsonl")
        return
    
    all_chunks = []
    
    for pages_file in pages_files:
        print(f"📝 Procesando chunks de {pages_file.name}...")
        
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
                        "source_type": "annoctr_text",
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
                    all_chunks.append(chunk_data)
        
        # Guardar chunks del archivo
        chunks_file = chunks_dir / pages_file.name.replace('.pages.jsonl', '.chunks.jsonl')
        with open(chunks_file, 'w', encoding='utf-8') as f:
            for chunk in chunks_data:
                json.dump(chunk, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"  ✅ {pages_file.name}: {len(chunks_data)} chunks creados")
    
    # Guardar todos los chunks
    all_chunks_file = chunks_dir / "all_chunks.jsonl"
    with open(all_chunks_file, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            json.dump(chunk, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"✅ Total de chunks creados: {len(all_chunks)}")
    print(f"📁 Archivos guardados en: {chunks_dir}")

def analyze_annoctr_content():
    """Analiza el contenido de los archivos de AnnoCTR"""
    print("\n🔍 Analizando contenido de AnnoCTR...")
    
    interim_dir = Path("data/interim/AnnoCTR")
    
    if not interim_dir.exists():
        print("❌ Directorio interim no encontrado")
        return
    
    pages_files = list(interim_dir.glob("*.pages.jsonl"))
    
    if not pages_files:
        print("❌ No se encontraron archivos pages.jsonl")
        return
    
    total_docs = 0
    total_words = 0
    total_chars = 0
    
    print(f"📊 Análisis de {len(pages_files)} archivos:")
    
    for pages_file in pages_files:
        split_docs = 0
        split_words = 0
        split_chars = 0
        
        with open(pages_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                page_data = json.loads(line)
                split_docs += 1
                
                text = page_data.get('text', '')
                split_words += len(text.split())
                split_chars += len(text)
        
        print(f"  {pages_file.name}:")
        print(f"    Documentos: {split_docs:,}")
        print(f"    Palabras: {split_words:,}")
        print(f"    Caracteres: {split_chars:,}")
        print(f"    Promedio palabras/doc: {split_words // split_docs if split_docs > 0 else 0:,}")
        
        total_docs += split_docs
        total_words += split_words
        total_chars += split_chars
    
    print(f"\n📈 Totales:")
    print(f"  Documentos: {total_docs:,}")
    print(f"  Palabras: {total_words:,}")
    print(f"  Caracteres: {total_chars:,}")
    print(f"  Promedio palabras/documento: {total_words // total_docs if total_docs > 0 else 0:,}")

def cleanup_temp_files():
    """Limpia archivos temporales"""
    print("\n🧹 Limpiando archivos temporales...")
    
    temp_dir = Path("temp_annoctr")
    if temp_dir.exists():
        import shutil
        shutil.rmtree(temp_dir)
        print("✅ Directorio temporal eliminado")

if __name__ == "__main__":
    # Cambiar al directorio del proyecto
    os.chdir("/Users/marcosespana/Desktop/U/DatosTesis")
    
    # Procesar archivos de texto
    total_docs = process_annoctr_text_files()
    
    if total_docs > 0:
        # Crear chunks
        create_annoctr_chunks()
        
        # Analizar contenido
        analyze_annoctr_content()
    
    # Limpiar archivos temporales
    cleanup_temp_files()
    
    print("\n🎉 Proceso completado!")
