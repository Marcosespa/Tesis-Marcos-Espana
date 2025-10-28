#!/usr/bin/env python3
"""
Script para limpiar todas las colecciones de Weaviate
"""

import weaviate
import sys
from pathlib import Path

# Support running as a script regardless of PYTHONPATH
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

try:
    from src.index.weaviate_client import get_source_to_class_mapping
except ModuleNotFoundError:
    from index.weaviate_client import get_source_to_class_mapping


def clean_all_collections():
    """Elimina todas las colecciones multi-clase"""
    try:
        # Conectar a Weaviate
        client = weaviate.connect_to_local(host="localhost", port=8080, grpc_port=50051)
        
        # Obtener todas las colecciones existentes
        collections = client.collections.list_all()
        print(f"📊 Colecciones existentes: {list(collections.keys())}")
        
        # Obtener mapeo de fuentes a clases
        source_to_class = get_source_to_class_mapping()
        expected_collections = list(source_to_class.values())
        print(f"📋 Colecciones esperadas: {expected_collections}")
        
        deleted_count = 0
        for collection_name in expected_collections:
            if collection_name in collections:
                print(f"🗑️  Eliminando colección: {collection_name}")
                try:
                    # Contar objetos antes de eliminar
                    collection = client.collections.get(collection_name)
                    count_before = collection.aggregate.over_all(total_count=True)
                    print(f"   📊 Objetos antes: {count_before.total_count}")
                    
                    # Eliminar la colección completa
                    client.collections.delete(collection_name)
                    print(f"   ✅ Colección '{collection_name}' eliminada")
                    deleted_count += 1
                    
                except Exception as e:
                    print(f"   ❌ Error eliminando {collection_name}: {e}")
            else:
                print(f"⚠️  Colección '{collection_name}' no existe")
        
        # También eliminar colecciones legacy si existen
        legacy_collections = ["BookChunk", "Document", "Chunk"]
        for legacy_name in legacy_collections:
            if legacy_name in collections:
                print(f"🗑️  Eliminando colección legacy: {legacy_name}")
                try:
                    client.collections.delete(legacy_name)
                    print(f"   ✅ Colección legacy '{legacy_name}' eliminada")
                    deleted_count += 1
                except Exception as e:
                    print(f"   ❌ Error eliminando {legacy_name}: {e}")
        
        print(f"\n🎉 Limpieza completada: {deleted_count} colecciones eliminadas")
        
        # Verificar estado final
        final_collections = client.collections.list_all()
        print(f"📊 Colecciones restantes: {list(final_collections.keys())}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        client.close()


def show_collection_info():
    """Muestra información de todas las colecciones"""
    try:
        client = weaviate.connect_to_local(host="localhost", port=8080, grpc_port=50051)
        
        collections = client.collections.list_all()
        print(f"📊 Total de colecciones: {len(collections)}")
        
        for name, collection in collections.items():
            try:
                count = collection.aggregate.over_all(total_count=True)
                print(f"   📋 {name}: {count.total_count} objetos")
            except Exception as e:
                print(f"   ❌ {name}: Error obteniendo conteo - {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        client.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--info":
        print("\n📊 Información de colecciones:")
        show_collection_info()
    else:
        print("🗑️  Eliminando todas las colecciones...")
        clean_all_collections()
