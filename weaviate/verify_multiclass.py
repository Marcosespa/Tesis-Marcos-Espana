#!/usr/bin/env python3
"""
Script para verificar que las multi-clases están funcionando correctamente
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


def verify_multiclass_setup():
    """Verifica que las multi-clases están configuradas y funcionando correctamente"""
    try:
        # Conectar a Weaviate
        client = weaviate.connect_to_local(host="localhost", port=8080, grpc_port=50051)
        
        # Obtener mapeo de fuentes a clases
        source_to_class = get_source_to_class_mapping()
        
        print("🔍 VERIFICACIÓN DE MULTI-CLASES EN WEAVIATE")
        print("=" * 60)
        
        total_objects = 0
        successful_collections = 0
        
        for source_path, class_name in source_to_class.items():
            try:
                collection = client.collections.get(class_name)
                
                # Verificar que la colección existe y tiene datos
                count = collection.aggregate.over_all(total_count=True)
                
                # Hacer una búsqueda de prueba
                test_query = collection.query.fetch_objects(limit=1)
                
                # Verificar propiedades de un objeto de ejemplo
                if test_query.objects:
                    obj = test_query.objects[0]
                    properties = obj.properties
                    
                    # Verificar propiedades básicas
                    has_doc_id = "docId" in properties and properties["docId"]
                    has_text = "text" in properties and properties["text"]
                    has_title = "title" in properties
                    has_source_type = "source_type" in properties and properties["source_type"] == class_name
                    
                    # Verificar que tiene vector
                    has_vector = obj.vector is not None and len(obj.vector) > 0
                    
                    # Contar propiedades específicas de la fuente
                    source_specific_props = 0
                    if "publication_type" in properties:
                        source_specific_props += 1
                    if "conference" in properties:
                        source_specific_props += 1
                    if "tactic" in properties:
                        source_specific_props += 1
                    if "owasp_category" in properties:
                        source_specific_props += 1
                    
                    # Determinar estado
                    if (has_doc_id and has_text and has_vector and has_source_type and 
                        count.total_count > 0):
                        status = "✅"
                        successful_collections += 1
                    else:
                        status = "⚠️"
                    
                    print(f"{status} {class_name:15} : {count.total_count:,} objetos")
                    print(f"    📄 DocID: {'✅' if has_doc_id else '❌'}")
                    print(f"    📝 Texto: {'✅' if has_text else '❌'}")
                    print(f"    🏷️  Título: {'✅' if has_title else '❌'}")
                    print(f"    🎯 Source Type: {'✅' if has_source_type else '❌'}")
                    print(f"    🔢 Vector: {'✅' if has_vector else '❌'}")
                    print(f"    🏷️  Props específicas: {source_specific_props}")
                    
                    if test_query.objects:
                        text_preview = properties.get("text", "")[:100] + "..." if len(properties.get("text", "")) > 100 else properties.get("text", "")
                        print(f"    📖 Texto ejemplo: {text_preview}")
                    
                else:
                    print(f"❌ {class_name:15} : Sin objetos para verificar")
                
                total_objects += count.total_count
                print()
                
            except Exception as e:
                print(f"❌ {class_name:15} : Error - {e}")
                print()
        
        print("=" * 60)
        print(f"📈 RESUMEN DE VERIFICACIÓN:")
        print(f"   Total objetos: {total_objects:,}")
        print(f"   Colecciones exitosas: {successful_collections}/{len(source_to_class)}")
        print(f"   Estado general: {'✅ ÉXITO' if successful_collections == len(source_to_class) else '⚠️  PROBLEMAS DETECTADOS'}")
        
        # Verificar búsqueda multi-clase
        print("\n🔍 PRUEBA DE BÚSQUEDA MULTI-CLASE:")
        try:
            # Probar búsqueda en una colección específica
            nist_collection = client.collections.get("NIST_SP")
            search_result = nist_collection.query.near_text(
                query="authentication security",
                limit=3
            )
            
            print(f"   ✅ Búsqueda en NIST_SP: {len(search_result.objects)} resultados")
            for i, obj in enumerate(search_result.objects):
                score = obj.metadata.distance if obj.metadata.distance else "N/A"
                text_preview = obj.properties.get("text", "")[:80] + "..."
                print(f"      {i+1}. Score: {score} | {text_preview}")
                
        except Exception as e:
            print(f"   ❌ Error en búsqueda: {e}")
        
    except Exception as e:
        print(f"❌ Error general: {e}")
    finally:
        client.close()


if __name__ == "__main__":
    verify_multiclass_setup()
