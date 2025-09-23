#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script wrapper para procesar todos los archivos TXT
Procesa AnnoCTR, MITRE, OWASP, SecurityTools, AISecKG y otros datos de texto
"""

import os
import subprocess
import sys
from pathlib import Path

def run_extract_script(input_dir: str, output_dir: str, min_chars: int = 50):
    """Ejecuta el script de extracción de texto en un directorio"""
    cmd = [
        sys.executable, 
        "src/rag/ingest/extract_text.py",
        "--in", input_dir,
        "--out", output_dir,
        "--min-chars", str(min_chars)
    ]
    
    print(f"🔄 Ejecutando: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Completado exitosamente")
        if result.stdout:
            print(result.stdout)
    else:
        print("❌ Error en la ejecución")
        if result.stderr:
            print(result.stderr)
        if result.stdout:
            print(result.stdout)
    
    return result.returncode == 0

def main():
    """Procesa todos los directorios con archivos de texto"""
    
    # Cambiar al directorio del proyecto
    os.chdir("/Users/marcosespana/Desktop/U/DatosTesis")
    
    print("📝 Iniciando procesamiento de archivos de texto...")
    
    # Directorios a procesar
    data_sources = [
        {
            "name": "AnnoCTR",
            "input": "data/raw/AnnoCTR",
            "output": "data/interim/AnnoCTR",
            "min_chars": 100
        },
        {
            "name": "MITRE",
            "input": "data/raw/MITRE",
            "output": "data/interim/MITRE", 
            "min_chars": 50
        },
        {
            "name": "OWASP",
            "input": "data/raw/OWASP",
            "output": "data/interim/OWASP",
            "min_chars": 50
        },
        {
            "name": "SecurityTools",
            "input": "data/raw/SecurityTools",
            "output": "data/interim/SecurityTools",
            "min_chars": 100
        },
        {
            "name": "AISecKG",
            "input": "data/raw/AISecKG",
            "output": "data/interim/AISecKG",
            "min_chars": 50
        },
        {
            "name": "NIST",
            "input": "data/raw/NIST",
            "output": "data/interim/NIST",
            "min_chars": 50
        },
        {
            "name": "OAPEN_PDFs",
            "input": "data/raw/OAPEN_PDFs",
            "output": "data/interim/OAPEN_PDFs",
            "min_chars": 100
        },
        {
            "name": "USENIX",
            "input": "data/raw/USENIX",
            "output": "data/interim/USENIX",
            "min_chars": 100
        }
    ]
    
    results = {}
    
    for source in data_sources:
        print(f"\n{'='*60}")
        print(f"📁 Procesando: {source['name']}")
        print(f"   Entrada: {source['input']}")
        print(f"   Salida: {source['output']}")
        print(f"   Min chars: {source['min_chars']}")
        print(f"{'='*60}")
        
        # Verificar que el directorio de entrada existe
        input_path = Path(source['input'])
        if not input_path.exists():
            print(f"⚠️  Directorio no encontrado: {source['input']}")
            results[source['name']] = False
            continue
        
        # Verificar si hay archivos .txt
        txt_files = list(input_path.glob("**/*.txt"))
        if not txt_files:
            print(f"⚠️  No se encontraron archivos .txt en: {source['input']}")
            results[source['name']] = False
            continue
        
        print(f"📄 Encontrados {len(txt_files)} archivos .txt")
        
        # Ejecutar procesamiento
        success = run_extract_script(
            source['input'],
            source['output'],
            source['min_chars']
        )
        
        results[source['name']] = success
        
        if success:
            print(f"✅ {source['name']} procesado exitosamente")
            
            # Mostrar estadísticas básicas
            output_path = Path(source['output'])
            if output_path.exists():
                jsonl_files = list(output_path.glob("**/*.pages.jsonl"))
                print(f"📊 Archivos JSONL generados: {len(jsonl_files)}")
        else:
            print(f"❌ Error procesando {source['name']}")
    
    # Resumen final
    print(f"\n{'='*60}")
    print("📊 RESUMEN FINAL")
    print(f"{'='*60}")
    
    successful = 0
    failed = 0
    
    for name, success in results.items():
        status = "✅ EXITOSO" if success else "❌ FALLÓ"
        print(f"  {name}: {status}")
        if success:
            successful += 1
        else:
            failed += 1
    
    print(f"\n📈 Total:")
    print(f"  Exitosos: {successful}")
    print(f"  Fallidos: {failed}")
    print(f"  Total: {len(results)}")
    
    if successful > 0:
        print(f"\n🎉 Procesamiento completado! {successful} fuentes procesadas exitosamente.")
        print("📁 Los datos están en data/interim/*/")
        print("📝 Cada archivo .txt se convirtió en un .pages.jsonl")
        
        # Contar archivos totales generados
        interim_path = Path("data/interim")
        if interim_path.exists():
            total_jsonl = len(list(interim_path.glob("**/*.pages.jsonl")))
            print(f"📊 Total de archivos JSONL generados: {total_jsonl}")
    else:
        print("\n❌ No se pudo procesar ninguna fuente de datos.")

def show_statistics():
    """Muestra estadísticas de los archivos procesados"""
    print("\n📈 ESTADÍSTICAS DETALLADAS:")
    print("="*50)
    
    interim_path = Path("data/interim")
    if not interim_path.exists():
        print("No se encontró el directorio data/interim")
        return
    
    total_chunks = 0
    total_files = 0
    
    for category_dir in interim_path.iterdir():
        if category_dir.is_dir():
            jsonl_files = list(category_dir.glob("*.pages.jsonl"))
            if jsonl_files:
                print(f"\n📂 {category_dir.name}:")
                print(f"   Archivos JSONL: {len(jsonl_files)}")
                
                category_chunks = 0
                for jsonl_file in jsonl_files:
                    try:
                        with jsonl_file.open('r', encoding='utf-8') as f:
                            chunks_in_file = sum(1 for _ in f)
                        category_chunks += chunks_in_file
                    except Exception:
                        pass
                
                print(f"   Total chunks: {category_chunks}")
                total_chunks += category_chunks
                total_files += len(jsonl_files)
    
    print(f"\n🎯 TOTALES GENERALES:")
    print(f"   Archivos JSONL: {total_files}")
    print(f"   Total chunks: {total_chunks}")
    print(f"   Promedio chunks por archivo: {total_chunks/max(total_files, 1):.1f}")

if __name__ == "__main__":
    main()
    
    # Mostrar estadísticas al final
    try:
        show_statistics()
    except Exception as e:
        print(f"Error mostrando estadísticas: {e}")