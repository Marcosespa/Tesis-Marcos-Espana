#!/usr/bin/env python3
"""
Script de verificación del estado del sistema RAG
Verifica Weaviate, colecciones, embeddings y datos
"""

import sys
from pathlib import Path
import json

# Add src to path
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[0] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

def check_weaviate_connection():
    """Verifica conexión a Weaviate"""
    print("🔌 Verificando conexión a Weaviate...")
    try:
        from src.index.weaviate_client import get_client
        client = get_client()
        client.close()
        print("✅ Conexión exitosa")
        return True
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        print("💡 Asegúrate de que Weaviate esté corriendo:")
        print("   docker-compose up -d")
        return False

def check_collections():
    """Verifica colecciones en Weaviate"""
    print("\n🏛️  Verificando colecciones...")
    try:
        from src.index.weaviate_client import get_client
        client = get_client()
        
        collections = client.collections.list_all()
        if not collections:
            print("⚠️  No hay colecciones creadas")
            print("💡 Ejecuta: python src/rag/index/ingest_multi_class.py --dry_run")
            client.close()
            return {}
        
        stats = {}
        for name in collections:
            try:
                coll = client.collections.get(name)
                result = coll.aggregate.over_all(total_count=True)
                count = result.total_count
                stats[name] = count
                status = "✅" if count > 0 else "⚠️"
                print(f"  {status} {name}: {count:,} objetos")
            except Exception as e:
                print(f"  ❌ {name}: Error - {e}")
                stats[name] = 0
        
        client.close()
        return stats
    except Exception as e:
        print(f"❌ Error verificando colecciones: {e}")
        return {}

def check_chunks():
    """Verifica archivos de chunks disponibles"""
    print("\n📄 Verificando archivos de chunks...")
    chunks_dir = Path("data/chunks")
    
    if not chunks_dir.exists():
        print(f"❌ Directorio {chunks_dir} no existe")
        return {}
    
    chunk_stats = {}
    for source_dir in chunks_dir.iterdir():
        if source_dir.is_dir():
            chunk_files = list(source_dir.glob("*.chunks.jsonl"))
            if chunk_files:
                total_lines = 0
                for f in chunk_files:
                    try:
                        with open(f, 'r', encoding='utf-8') as file:
                            total_lines += sum(1 for _ in file)
                    except:
                        pass
                chunk_stats[source_dir.name] = total_lines
                print(f"  ✅ {source_dir.name}: {total_lines:,} chunks")
            else:
                print(f"  ⚠️  {source_dir.name}: Sin archivos .chunks.jsonl")
    
    return chunk_stats

def check_embeddings():
    """Verifica embeddings precalculados"""
    print("\n🧮 Verificando embeddings precalculados...")
    embeddings_dir = Path("data/embeddings")
    
    if not embeddings_dir.exists():
        print(f"❌ Directorio {embeddings_dir} no existe")
        return {}
    
    embedding_stats = {}
    for source_dir in embeddings_dir.iterdir():
        if source_dir.is_dir():
            npy_files = list(source_dir.glob("*.embeddings.npy"))
            if npy_files:
                try:
                    import numpy as np
                    embeddings_file = max(npy_files, key=lambda f: f.stat().st_size)
                    embeddings = np.load(embeddings_file)
                    embedding_stats[source_dir.name] = {
                        'shape': embeddings.shape,
                        'file': embeddings_file.name
                    }
                    print(f"  ✅ {source_dir.name}: {embeddings.shape[0]:,} embeddings ({embeddings.shape[1]}D)")
                except Exception as e:
                    print(f"  ❌ {source_dir.name}: Error cargando - {e}")
                    embedding_stats[source_dir.name] = None
            else:
                print(f"  ⚠️  {source_dir.name}: Sin embeddings precalculados")
    
    return embedding_stats

def check_schema():
    """Verifica schema de Weaviate"""
    print("\n📋 Verificando schema...")
    schema_file = Path("configs/weaviate_multi_class.schema.json")
    
    if not schema_file.exists():
        print(f"❌ Archivo de schema no encontrado: {schema_file}")
        return None
    
    try:
        with open(schema_file, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        
        classes = schema.get('classes', [])
        print(f"  ✅ Schema válido con {len(classes)} clases definidas:")
        for cls in classes:
            print(f"     - {cls.get('class')}")
        
        return schema
    except Exception as e:
        print(f"❌ Error leyendo schema: {e}")
        return None

def compare_chunks_vs_indexed(chunk_stats, collection_stats):
    """Compara chunks disponibles vs indexados"""
    print("\n📊 Comparación: Chunks disponibles vs Indexados")
    print("=" * 60)
    
    from src.index.weaviate_client import get_source_to_class_mapping
    source_to_class = get_source_to_class_mapping()
    
    total_chunks = 0
    total_indexed = 0
    
    for source_name, chunks_count in chunk_stats.items():
        # Find corresponding collection
        collection_name = None
        for source_path, class_name in source_to_class.items():
            if Path(source_path).name == source_name:
                collection_name = class_name
                break
        
        if collection_name:
            indexed_count = collection_stats.get(collection_name, 0)
            total_chunks += chunks_count
            total_indexed += indexed_count
            
            if indexed_count == 0:
                status = "⚠️  NO INDEXADO"
            elif indexed_count < chunks_count * 0.9:
                status = f"⚠️  PARCIAL ({indexed_count}/{chunks_count})"
            else:
                status = "✅ COMPLETO"
            
            print(f"{status:15} {source_name:20} → {collection_name:20}")
            print(f"               Chunks: {chunks_count:>8,} | Indexados: {indexed_count:>8,}")
    
    print("=" * 60)
    print(f"Total chunks disponibles: {total_chunks:,}")
    print(f"Total objetos indexados: {total_indexed:,}")
    
    if total_indexed < total_chunks * 0.5:
        print("\n⚠️  Menos del 50% de los chunks están indexados")
        print("💡 Ejecuta la indexación:")
        print("   python src/rag/index/ingest_multi_class.py")
    elif total_indexed == 0:
        print("\n❌ No hay datos indexados")
        print("💡 Ejecuta la indexación:")
        print("   python src/rag/index/ingest_multi_class.py --dry_run  # Primero en modo dry-run")

def main():
    print("=" * 60)
    print("🔍 VERIFICACIÓN DEL SISTEMA RAG")
    print("=" * 60)
    
    # Check Weaviate connection
    if not check_weaviate_connection():
        print("\n❌ No se puede continuar sin conexión a Weaviate")
        return 1
    
    # Check collections
    collection_stats = check_collections()
    
    # Check chunks
    chunk_stats = check_chunks()
    
    # Check embeddings
    embedding_stats = check_embeddings()
    
    # Check schema
    schema = check_schema()
    
    # Compare
    if chunk_stats and collection_stats:
        compare_chunks_vs_indexed(chunk_stats, collection_stats)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 RESUMEN")
    print("=" * 60)
    
    total_collections = len([c for c in collection_stats.values() if c > 0])
    total_indexed = sum(collection_stats.values())
    total_chunks = sum(chunk_stats.values())
    total_embeddings = len([e for e in embedding_stats.values() if e is not None])
    
    print(f"✅ Colecciones con datos: {total_collections}/{len(collection_stats)}")
    print(f"📚 Objetos indexados: {total_indexed:,}")
    print(f"📄 Chunks disponibles: {total_chunks:,}")
    print(f"🧮 Fuentes con embeddings: {total_embeddings}/{len(embedding_stats)}")
    
    # Recommendations
    print("\n💡 RECOMENDACIONES:")
    if total_indexed == 0:
        print("   1. Ejecutar indexación de datos")
    elif total_indexed < total_chunks * 0.5:
        print("   1. Completar indexación (falta >50%)")
    
    if total_embeddings < len(embedding_stats):
        print("   2. Generar embeddings precalculados faltantes")
    
    if total_collections == 0:
        print("   3. Crear colecciones en Weaviate")
    
    print("\n✅ Verificación completada")
    return 0

if __name__ == "__main__":
    sys.exit(main())

