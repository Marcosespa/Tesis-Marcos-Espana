#!/usr/bin/env python3
"""
Script para generar embeddings de todos los archivos de chunks
y guardarlos en archivos locales para transferir a máquina local.
"""

import os
import sys
from pathlib import Path
import subprocess
import time

def find_chunk_files(chunks_dir: str) -> list:
    """Encuentra todos los archivos .chunks.jsonl"""
    chunks_path = Path(chunks_dir)
    if not chunks_path.exists():
        print(f"❌ Directorio {chunks_dir} no existe")
        return []
    
    chunk_files = []
    for pattern in ["*.chunks.jsonl", "**/*.chunks.jsonl"]:
        chunk_files.extend(chunks_path.glob(pattern))
    
    return sorted(chunk_files)

def generate_embeddings_for_file(chunk_file: Path, embeddings_dir: str = "data/embeddings", device: str = "cuda:1") -> bool:
    """Genera embeddings para un archivo de chunks"""
    try:
        print(f"🔄 Procesando: {chunk_file.name}")
        
        # Comando para generar embeddings
        cmd = [
            "python", "src/rag/index/ingest_weaviate.py",
            "--jsonl", str(chunk_file),
            "--save-embeddings",
            "--embeddings-dir", embeddings_dir,
            "--device", device,
            "--model", "all-MiniLM-L6-v2",
            "--batch", "64"
        ]
        
        # Ejecutar comando
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode == 0:
            print(f"  ✅ Embeddings generados para {chunk_file.name}")
            return True
        else:
            print(f"  ❌ Error procesando {chunk_file.name}:")
            print(f"     {result.stderr}")
            return False
            
    except Exception as e:
        print(f"  ❌ Excepción procesando {chunk_file.name}: {e}")
        return False

def main():
    print("🚀 Generador de Embeddings para Transferencia Local")
    print("=" * 60)
    
    # Configuración
    chunks_dir = "data/chunks"
    embeddings_dir = "data/embeddings"
    device = os.getenv('GPU_DEVICE', 'cuda:1')
    
    print(f"📁 Directorio de chunks: {chunks_dir}")
    print(f"💾 Directorio de embeddings: {embeddings_dir}")
    print(f"🔧 Dispositivo: {device}")
    
    # Crear directorio de embeddings
    Path(embeddings_dir).mkdir(parents=True, exist_ok=True)
    
    # Encontrar archivos de chunks
    chunk_files = find_chunk_files(chunks_dir)
    
    if not chunk_files:
        print("❌ No se encontraron archivos .chunks.jsonl")
        return
    
    print(f"\n📊 Encontrados {len(chunk_files)} archivos de chunks:")
    for chunk_file in chunk_files:
        print(f"   • {chunk_file.relative_to(chunks_dir)}")
    
    # Procesar archivos
    print(f"\n🔄 Iniciando generación de embeddings...")
    start_time = time.time()
    
    successful = 0
    failed = 0
    
    for i, chunk_file in enumerate(chunk_files, 1):
        print(f"\n[{i}/{len(chunk_files)}] Procesando: {chunk_file.name}")
        
        if generate_embeddings_for_file(chunk_file, embeddings_dir, device):
            successful += 1
        else:
            failed += 1
    
    # Resumen final
    elapsed = time.time() - start_time
    print(f"\n📊 RESUMEN FINAL")
    print("=" * 40)
    print(f"⏱️  Tiempo total: {elapsed:.1f} segundos")
    print(f"✅ Archivos exitosos: {successful}")
    print(f"❌ Archivos fallidos: {failed}")
    print(f"📁 Embeddings guardados en: {embeddings_dir}")
    
    if successful > 0:
        print(f"\n📋 ARCHIVOS GENERADOS:")
        embeddings_path = Path(embeddings_dir)
        for file_path in sorted(embeddings_path.glob("*.embeddings.npy")):
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"   • {file_path.name} ({size_mb:.1f} MB)")
        
        print(f"\n🚀 Para transferir a local:")
        print(f"   scp -r {embeddings_dir}/* usuario@local:/ruta/destino/")
        print(f"   # O usar rsync:")
        print(f"   rsync -av {embeddings_dir}/ usuario@local:/ruta/destino/")

if __name__ == "__main__":
    main()
