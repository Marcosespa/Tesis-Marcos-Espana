#!/usr/bin/env python3
"""
Script para cargar embeddings precalculados a Weaviate
"""

import argparse
import json
import numpy as np
from pathlib import Path
import uuid
from typing import Any, Dict, List
import sys
import time

import weaviate
from weaviate.classes.query import Filter
from weaviate.collections.classes.data import DataObject

# Support running as a script regardless of PYTHONPATH
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

try:
    from src.index.weaviate_client import get_client, ensure_multi_class_schema, get_source_to_class_mapping
except ModuleNotFoundError:
    from index.weaviate_client import get_client, ensure_multi_class_schema, get_source_to_class_mapping


def load_embeddings_and_chunks(embeddings_dir: Path, chunks_dir: Path, source_name: str) -> tuple:
    """Carga embeddings y chunks para una fuente específica"""
    
    embeddings_source_dir = embeddings_dir / source_name
    chunks_source_dir = chunks_dir / source_name
    
    if not embeddings_source_dir.exists():
        print(f"⚠️  Directorio de embeddings no existe: {embeddings_source_dir}")
        return None, None
        
    if not chunks_source_dir.exists():
        print(f"⚠️  Directorio de chunks no existe: {chunks_source_dir}")
        return None, None
    
    # Buscar archivos de embeddings
    npy_files = list(embeddings_source_dir.glob("*.embeddings.npy"))
    if not npy_files:
        print(f"⚠️  No se encontraron archivos de embeddings en {embeddings_source_dir}")
        return None, None
    
    # Cargar embeddings (tomar el archivo más grande si hay múltiples)
    embeddings_file = max(npy_files, key=lambda f: f.stat().st_size)
    print(f"📊 Cargando embeddings desde: {embeddings_file}")
    embeddings = np.load(embeddings_file)
    print(f"   Shape: {embeddings.shape}")
    
    # Buscar archivos de chunks
    chunk_files = list(chunks_source_dir.glob("*.chunks.jsonl"))
    if not chunk_files:
        print(f"⚠️  No se encontraron archivos de chunks en {chunks_source_dir}")
        return None, None
    
    # Cargar chunks (tomar el archivo más grande si hay múltiples)
    chunks_file = max(chunk_files, key=lambda f: f.stat().st_size)
    print(f"📄 Cargando chunks desde: {chunks_file}")
    
    chunk_records = []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunk_records.append(json.loads(line))
    
    print(f"   Chunks cargados: {len(chunk_records)}")
    
    # Verificar que coincidan las dimensiones
    if len(chunk_records) != embeddings.shape[0]:
        print(f"⚠️  Advertencia: {len(chunk_records)} chunks vs {embeddings.shape[0]} embeddings")
        # Tomar el mínimo para evitar errores
        min_count = min(len(chunk_records), embeddings.shape[0])
        chunk_records = chunk_records[:min_count]
        embeddings = embeddings[:min_count]
        print(f"   Usando {min_count} registros")
    
    return embeddings, chunk_records


def extract_source_metadata(record: Dict[str, Any], source_path: str) -> Dict[str, Any]:
    """Extrae metadatos específicos de la fuente"""
    metadata = record.get("metadata", {}) or {}
    
    # Base metadata
    base_metadata = {
        "source_type": Path(source_path).name,
        "source_path": source_path,
    }
    
    # Source-specific metadata extraction
    if "NIST" in source_path:
        if "CSWP" in source_path:
            base_metadata["cswp_number"] = metadata.get("source_id", "")
            base_metadata["publication_type"] = "Cybersecurity White Paper"
        elif "AI" in source_path:
            base_metadata["ai_guideline_number"] = metadata.get("source_id", "")
            base_metadata["publication_type"] = "AI Guidelines"
        else:
            base_metadata["publication_number"] = metadata.get("doc_title", metadata.get("source_id", ""))
            base_metadata["publication_type"] = "Special Publication"
            
    elif "USENIX" in source_path:
        base_metadata.update({
            "conference": "USENIX Security",
            "year": metadata.get("year", 2024),
            "authors": metadata.get("authors", [])
        })
        
    elif "MITRE" in source_path:
        text_content = record.get("text", "")
        base_metadata.update({
            "tactic": "General",  # Se puede extraer del texto si es necesario
            "technique_id": "",
            "platform": ["Enterprise", "Mobile", "ICS"]
        })
        
    elif "OWASP" in source_path:
        base_metadata.update({
            "owasp_category": metadata.get("category", "General"),
            "risk_level": "Medium"
        })
        
    elif "SecurityTools" in source_path:
        base_metadata.update({
            "tool_category": metadata.get("category", "General"),
            "tool_name": metadata.get("source_id", "").split(".")[0]
        })
        
    elif "AISecKG" in source_path:
        base_metadata.update({
            "lab_id": metadata.get("source_id", "unknown"),
            "topic": "AI Security"
        })
        
    elif "AnnoCTR" in source_path:
        base_metadata.update({
            "split": "unknown",
            "annotation_type": "general"
        })
    
    return base_metadata


def ingest_embeddings_to_class(
    client,
    collection_name: str,
    embeddings: np.ndarray,
    chunk_records: List[Dict],
    source_path: str,
    batch_size: int = 64,
    limit: int = 0,
    dry_run: bool = False
) -> int:
    """Ingesta embeddings precalculados en una colección específica"""
    
    collection = client.collections.get(collection_name)
    
    total = 0
    last_log = time.time()
    start = last_log
    batch_idx = 0
    
    print(f"🔄 Procesando {len(chunk_records)} registros en lotes de {batch_size}...")
    
    for i in range(0, len(chunk_records), batch_size):
        batch_idx += 1
        batch_end = min(i + batch_size, len(chunk_records))
        
        batch_chunks = chunk_records[i:batch_end]
        batch_embeddings = embeddings[i:batch_end]
        
        data_objects = []
        
        for j, (record, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
            md = record.get("metadata", {}) or {}
            
            # Extraer texto del record
            text = record.get("text") or record.get("content") or ""
            if not text.strip():
                continue
            
            # Build robust identifiers
            doc_id = (
                record.get("chunk_id") or md.get("chunk_id") or 
                md.get("source_id") or md.get("doc_id") or "unknown"
            )
            
            title = (
                md.get("doc_title") or md.get("title") or 
                md.get("section_title") or md.get("source_id") or ""
            )
            
            # Page information
            page_start = (
                md.get("page_num_real") or md.get("page_start") or 
                md.get("page") or 0
            )
            page_end = page_start
            
            # Extract source-specific metadata
            source_metadata = extract_source_metadata(record, source_path)
            
            # Build properties
            properties = {
                "docId": str(doc_id),
                "level": int(md.get("section_level") or md.get("chunk_index") or 0),
                "title": str(title),
                "text": text,
                "pageStart": int(page_start) if page_start is not None else 0,
                "pageEnd": int(page_end) if page_end is not None else 0,
                **source_metadata
            }
            
            # Generate deterministic UUID
            chunk_id = record.get("chunk_id") or md.get("chunk_id")
            if chunk_id is not None:
                uid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"chunk:{chunk_id}:{collection_name}"))
            else:
                uid_base = f"{properties['docId']}|{properties['level']}|{properties['pageStart']}|{properties['pageEnd']}|{text[:96]}"
                uid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{uid_base}:{collection_name}"))
            
            data_objects.append(DataObject(
                uuid=uid, 
                properties=properties, 
                vector=embedding.astype(np.float32)
            ))
        
        if dry_run:
            print(f"🔍 Previsualización de mapeo para {collection_name} (hasta 3 del batch):")
            for obj in data_objects[:3]:
                print({
                    "uuid": obj.uuid,
                    "docId": obj.properties.get("docId"),
                    "title": obj.properties.get("title"),
                    "source_type": obj.properties.get("source_type"),
                    "text_preview": (obj.properties.get("text") or "")[:80]
                })
        else:
            try:
                if data_objects:
                    collection.data.insert_many(data_objects)
            except Exception as e:
                print(f"❌ Error insertando batch {batch_idx} en {collection_name}: {e}")
                raise
        
        total += len(data_objects)
        
        # Respect limit
        if limit and total >= limit:
            print(f"⏹️  Límite alcanzado ({limit}) para {collection_name}. Deteniendo.")
            break
        
        # Progress logging
        now = time.time()
        if now - last_log >= 2.0:
            elapsed = now - start
            rate = total / elapsed if elapsed > 0 else 0.0
            print(f"📊 {collection_name} - Batch {batch_idx} | acumulado={total} | {rate:.1f} objs/s")
            last_log = now
    
    elapsed = time.time() - start
    rate = total / elapsed if elapsed > 0 else 0.0
    print(f"✅ {collection_name} completado: {total} objetos en {elapsed:.1f}s ({rate:.1f} objs/s)")
    
    return total


def main():
    parser = argparse.ArgumentParser(description="Ingesta de embeddings precalculados a Weaviate")
    parser.add_argument("--embeddings-dir", default="data/embeddings", help="Directorio con archivos de embeddings")
    parser.add_argument("--chunks-dir", default="data/chunks", help="Directorio con archivos de chunks")
    parser.add_argument("--schema", default="configs/weaviate_multi_class.schema.json", help="Esquema multi-clase JSON")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--http_port", type=int, default=8080)
    parser.add_argument("--grpc_port", type=int, default=50051)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="Máximo de objetos por clase (0 = todos)")
    parser.add_argument("--dry_run", action="store_true", help="No inserta; solo muestra mapeo")
    parser.add_argument("--sources", nargs="+", help="Especificar fuentes a procesar (ej: NIST MITRE)")
    args = parser.parse_args()
    
    print(f"🔌 Conectando a Weaviate en {args.host}:{args.http_port} (gRPC {args.grpc_port})...")
    client = get_client(args.host, args.http_port, args.grpc_port)
    
    try:
        # Setup multi-class schema
        print("🏗️  Configurando esquema multi-clase...")
        source_mapping = ensure_multi_class_schema(client, args.schema)
        print("✅ Esquema configurado.")
        
        # Get source to class mapping
        source_to_class = get_source_to_class_mapping()
        
        # Find directories
        embeddings_dir = Path(args.embeddings_dir)
        chunks_dir = Path(args.chunks_dir)
        
        if not embeddings_dir.exists():
            print(f"❌ ERROR: Directorio de embeddings {embeddings_dir} no existe")
            return
            
        if not chunks_dir.exists():
            print(f"❌ ERROR: Directorio de chunks {chunks_dir} no existe")
            return
        
        # Process each source
        total_ingested = 0
        processed_sources = 0
        
        for source_path, class_name in source_to_class.items():
            # Filter by specified sources if provided
            if args.sources and class_name not in args.sources:
                continue
            
            source_name = Path(source_path).name
            
            print(f"\n📂 Procesando fuente: {class_name}")
            print(f"   📁 Embeddings: {embeddings_dir / source_name}")
            print(f"   📁 Chunks: {chunks_dir / source_name}")
            
            # Load embeddings and chunks
            embeddings, chunk_records = load_embeddings_and_chunks(embeddings_dir, chunks_dir, source_name)
            
            if embeddings is None or chunk_records is None:
                print(f"⚠️  Saltando {class_name}: no se pudieron cargar los datos")
                continue
            
            print(f"   📊 Embeddings: {embeddings.shape}")
            print(f"   📄 Chunks: {len(chunk_records)}")
            
            # Ingest to Weaviate
            ingested = ingest_embeddings_to_class(
                client=client,
                collection_name=class_name,
                embeddings=embeddings,
                chunk_records=chunk_records,
                source_path=source_path,
                batch_size=args.batch,
                limit=args.limit,
                dry_run=args.dry_run
            )
            total_ingested += ingested
            processed_sources += 1
        
        # Summary
        print(f"\n🎉 INGESTA DE EMBEDDINGS COMPLETADA")
        print(f"   📊 Fuentes procesadas: {processed_sources}")
        print(f"   📈 Total objetos ingeridos: {total_ingested}")
        print(f"   🏛️  Colecciones creadas: {len(source_to_class)}")
        
        if args.dry_run:
            print(f"\n💡 Para ejecutar la ingesta real, elimina el flag --dry_run")
        
    finally:
        client.close()


if __name__ == "__main__":
    main()
