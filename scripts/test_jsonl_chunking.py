#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de prueba para el chunking híbrido con archivos JSONL
"""

import sys
import os
from pathlib import Path

# Agregar el directorio src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag.process.chunking import hybrid_chunk_jsonl, process_multiple_jsonl_files

def test_single_jsonl():
    """Prueba el chunking con un archivo JSONL individual"""
    
    # Buscar un archivo .pages.jsonl de ejemplo
    base_dir = Path(__file__).parent.parent
    interim_dir = base_dir / "data" / "interim"
    
    jsonl_files = list(interim_dir.glob("**/*.pages.jsonl"))
    if not jsonl_files:
        print(f"❌ No se encontraron archivos *.pages.jsonl en {interim_dir}")
        return False
    
    # Usar el primer archivo encontrado
    test_file = jsonl_files[0]
    print(f"🔍 Probando con archivo: {test_file.name}")
    
    try:
        # Procesar archivo
        documents = hybrid_chunk_jsonl(
            jsonl_path=str(test_file),
            target_tokens=900,
            min_tokens=400,
            max_tokens=1400,
            overlap_ratio=0.18
        )
        
        print(f"✅ Procesamiento exitoso: {len(documents)} chunks generados")
        
        # Mostrar muestra de chunks
        print(f"\n📋 Muestra de chunks:")
        for i, doc in enumerate(documents[:3]):
            print(f"\n{'='*50}")
            print(f"🔸 Chunk #{i}")
            print(f"📏 Tokens: {doc.metadata.get('chunk_tokens', 'N/A')}")
            print(f"📂 Sección: {doc.metadata.get('section_title', 'Sin sección')}")
            print(f"🏷️  Categoría: {doc.metadata.get('category', 'N/A')}")
            print(f"📄 Contenido (primeros 200 chars):")
            preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            print(f"   {preview}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error procesando {test_file.name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_jsonl():
    """Prueba el chunking con múltiples archivos JSONL"""
    
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "data" / "interim"
    output_dir = base_dir / "data" / "chunks"
    
    # Verificar que existen archivos de entrada
    jsonl_files = list(input_dir.glob("**/*.pages.jsonl"))
    if not jsonl_files:
        print(f"❌ No se encontraron archivos *.pages.jsonl en {input_dir}")
        return False
    
    print(f"🔍 Encontrados {len(jsonl_files)} archivos para procesar")
    
    try:
        # Procesar múltiples archivos
        results = process_multiple_jsonl_files(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            target_tokens=900,
            min_tokens=400,
            max_tokens=1400,
            overlap_ratio=0.18
        )
        
        print(f"✅ Procesamiento exitoso: {len(results)} archivos procesados")
        
        # Mostrar estadísticas
        total_chunks = sum(len(docs) for docs in results.values())
        print(f"📊 Total de chunks generados: {total_chunks}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error procesando múltiples archivos: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal de prueba"""
    print("🚀 Iniciando pruebas de chunking híbrido con JSONL")
    print("="*60)
    
    # Prueba 1: Archivo individual
    print("\n📖 Prueba 1: Archivo JSONL individual")
    success1 = test_single_jsonl()
    
    # Prueba 2: Múltiples archivos
    print("\n📚 Prueba 2: Múltiples archivos JSONL")
    success2 = test_multiple_jsonl()
    
    # Resumen
    print("\n" + "="*60)
    print("📊 RESUMEN DE PRUEBAS")
    print("="*60)
    print(f"✅ Archivo individual: {'Éxito' if success1 else 'Falló'}")
    print(f"✅ Múltiples archivos: {'Éxito' if success2 else 'Falló'}")
    
    if success1 and success2:
        print("\n🎉 ¡Todas las pruebas pasaron exitosamente!")
        print("💡 El código está listo para usar con tu pipeline existente")
    else:
        print("\n⚠️  Algunas pruebas fallaron. Revisa los errores arriba.")

if __name__ == "__main__":
    main()
