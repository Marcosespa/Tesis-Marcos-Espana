#!/usr/bin/env python3
"""
Multi-Class Weaviate Ingestion Script

This script ingests chunks into multiple Weaviate collections based on source directories.
Each source gets its own collection with specific properties and metadata.
"""

import argparse
import json
from pathlib import Path
import uuid
from typing import Any, Iterable, List, Dict
import sys
import time

import numpy as np
from sentence_transformers import SentenceTransformer

# Support running as a script regardless of PYTHONPATH
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[3]  # .../DatosTesis/src
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

try:
    from src.index.weaviate_client import get_client, ensure_multi_class_schema, get_source_to_class_mapping
except ModuleNotFoundError:
    from index.weaviate_client import get_client, ensure_multi_class_schema, get_source_to_class_mapping


def batched(iterable: Iterable[Any], batch_size: int) -> Iterable[List[Any]]:
    """Split iterable into batches."""
    batch: List[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def extract_source_metadata(record: Dict[str, Any], source_path: str) -> Dict[str, Any]:
    """Extract source-specific metadata based on the source path."""
    metadata = record.get("metadata", {}) or {}
    
    # Base metadata
    base_metadata = {
        "source_type": Path(source_path).name,
        "source_path": source_path,
    }
    
    # Source-specific metadata extraction
    if "NIST" in source_path:
        if "CSWP" in source_path:
            # Extract CSWP number from filename or metadata
            filename = metadata.get("source_file", "")
            cswp_match = Path(filename).stem if filename else ""
            base_metadata["cswp_number"] = cswp_match
            base_metadata["publication_type"] = "Cybersecurity White Paper"
            
        elif "AI" in source_path:
            # Extract AI guideline number
            filename = metadata.get("source_file", "")
            ai_match = Path(filename).stem if filename else ""
            base_metadata["ai_guideline_number"] = ai_match
            base_metadata["publication_type"] = "AI Guidelines"
            
        else:
            # General NIST SP
            base_metadata["publication_number"] = metadata.get("doc_title", "")
            base_metadata["publication_type"] = "Special Publication"
            
    elif "USENIX" in source_path:
        base_metadata.update({
            "conference": "USENIX Security",
            "year": metadata.get("year", 2024),
            "authors": metadata.get("authors", [])
        })
        
    elif "MITRE" in source_path:
        # Extract ATT&CK specific metadata
        text_content = record.get("text", "") or record.get("content", "")
        base_metadata.update({
            "tactic": extract_tactic_from_text(text_content),
            "technique_id": extract_technique_id(text_content),
            "platform": ["Enterprise", "Mobile", "ICS"]  # Default platforms
        })
        
    elif "OWASP" in source_path:
        base_metadata.update({
            "owasp_category": metadata.get("category", "General"),
            "risk_level": "Medium"  # Default risk level
        })
        
    elif "SecurityTools" in source_path:
        base_metadata.update({
            "tool_category": metadata.get("category", "General"),
            "tool_name": metadata.get("source_file", "").split(".")[0]
        })
        
    elif "AISecKG" in source_path:
        base_metadata.update({
            "lab_id": metadata.get("lab_id", "unknown"),
            "topic": metadata.get("topic", "AI Security")
        })
        
    elif "AnnoCTR" in source_path:
        base_metadata.update({
            "split": metadata.get("split", "unknown"),
            "annotation_type": metadata.get("annotation_type", "general")
        })
    
    return base_metadata


def extract_tactic_from_text(text: str) -> str:
    """Extract MITRE ATT&CK tactic from text."""
    tactics = [
        "Initial Access", "Execution", "Persistence", "Privilege Escalation",
        "Defense Evasion", "Credential Access", "Discovery", "Lateral Movement",
        "Collection", "Command and Control", "Exfiltration", "Impact"
    ]
    
    text_lower = text.lower()
    for tactic in tactics:
        if tactic.lower() in text_lower:
            return tactic
    return "General"


def extract_technique_id(text: str) -> str:
    """Extract MITRE ATT&CK technique ID from text."""
    import re
    # Look for patterns like T1055, T1059, etc.
    pattern = r'\bT\d{4}\b'
    matches = re.findall(pattern, text)
    return matches[0] if matches else ""


def ingest_to_class(
    client,
    collection_name: str,
    jsonl_path: str,
    model: SentenceTransformer,
    batch_size: int = 64,
    limit: int = 0,
    dry_run: bool = False
) -> int:
    """Ingest chunks from a JSONL file into a specific Weaviate collection."""
    
    collection = client.collections.get(collection_name)
    source_path = str(Path(jsonl_path).parent)
    
    def records():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                yield json.loads(line)
    
    # Verify file exists
    in_path = Path(jsonl_path)
    if not in_path.exists():
        print(f"ERROR: No existe el archivo {in_path}", flush=True)
        return 0
    
    # Count lines
    try:
        with open(in_path, "r", encoding="utf-8") as _f:
            approx_total = sum(1 for _ in _f)
        print(f"📄 Fuente: {in_path} (~{approx_total} líneas) → {collection_name}", flush=True)
    except Exception as e:
        print(f"No se pudo contar líneas: {e}", flush=True)
    
    total = 0
    last_log = time.time()
    start = last_log
    batch_idx = 0
    
    print(f"🔄 Procesando en lotes de {batch_size}...", flush=True)
    
    for batch in batched(records(), batch_size):
        batch_idx += 1
        texts = [r.get("content") or r.get("text") or "" for r in batch]
        
        if not texts or all(not text.strip() for text in texts):
            continue
            
        # Generate embeddings
        vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        
        # Insert with explicit vectors using DataObject
        from weaviate.collections.classes.data import DataObject
        data_objects = []
        
        for r, vec in zip(batch, vectors):
            md = r.get("metadata", {}) or {}
            
            # Build robust identifiers
            doc_id = (
                r.get("doc_id") or md.get("doc_id") or r.get("source_id") or
                r.get("chunk_id") or md.get("source_id") or md.get("file_id") or
                md.get("page_name") or "unknown"
            )
            
            title = (
                r.get("title") or md.get("title") or md.get("doc_title") or
                md.get("page_name") or r.get("source_file") or ""
            )
            
            # Page information
            page_start = (
                r.get("page_start") or md.get("page_start") or r.get("page") or
                md.get("page") or md.get("page_num_real") or 0
            )
            page_end = (
                r.get("page_end") or md.get("page_end") or r.get("page") or
                md.get("page") or md.get("page_num_real") or page_start
            )
            
            # Extract source-specific metadata
            source_metadata = extract_source_metadata(r, source_path)
            
            # Build properties
            properties = {
                "docId": str(doc_id),
                "level": int(r.get("chunk_index") or md.get("level") or md.get("section_level") or 0),
                "title": str(title),
                "text": r.get("text") or r.get("content") or "",
                "pageStart": int(page_start) if page_start is not None else 0,
                "pageEnd": int(page_end) if page_end is not None else 0,
                **source_metadata  # Add source-specific metadata
            }
            
            # Generate deterministic UUID
            chunk_id = r.get("chunk_id") or md.get("chunk_id")
            if chunk_id is not None:
                uid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"chunk:{chunk_id}:{collection_name}"))
            else:
                uid_base = f"{properties['docId']}|{properties['level']}|{properties['pageStart']}|{properties['pageEnd']}|{(properties['text'] or '')[:96]}"
                uid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{uid_base}:{collection_name}"))
            
            data_objects.append(DataObject(uuid=uid, properties=properties, vector=vec.astype(np.float32)))
        
        if dry_run:
            print(f"🔍 Previsualización de mapeo para {collection_name} (hasta 3 del batch):", flush=True)
            for obj in data_objects[:3]:
                print({
                    "uuid": obj.uuid,
                    "docId": obj.properties.get("docId"),
                    "title": obj.properties.get("title"),
                    "source_type": obj.properties.get("source_type"),
                    "text_preview": (obj.properties.get("text") or "")[:80]
                }, flush=True)
        else:
            try:
                collection.data.insert_many(data_objects)
            except Exception as e:
                print(f"❌ Error insertando batch {batch_idx} en {collection_name}: {e}", flush=True)
                raise
        
        total += len(batch)
        
        # Respect limit
        if limit and total >= limit:
            print(f"⏹️  Límite alcanzado ({limit}) para {collection_name}. Deteniendo.", flush=True)
            break
        
        # Progress logging
        now = time.time()
        if now - last_log >= 2.0:
            elapsed = now - start
            rate = total / elapsed if elapsed > 0 else 0.0
            print(f"📊 {collection_name} - Batch {batch_idx} | acumulado={total} | {rate:.1f} objs/s", flush=True)
            last_log = now
    
    elapsed = time.time() - start
    rate = total / elapsed if elapsed > 0 else 0.0
    print(f"✅ {collection_name} completado: {total} objetos en {elapsed:.1f}s ({rate:.1f} objs/s)", flush=True)
    
    return total


def main():
    parser = argparse.ArgumentParser(description="Multi-Class Weaviate Ingestion")
    parser.add_argument("--chunks-dir", default="data/chunks", help="Directory containing chunk files")
    parser.add_argument("--schema", default="configs/weaviate_multi_class.schema.json", help="Multi-class schema JSON path")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--http_port", type=int, default=8080)
    parser.add_argument("--grpc_port", type=int, default=50051)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="Máximo de objetos por clase (0 = todos)")
    parser.add_argument("--dry_run", action="store_true", help="No inserta; solo muestra mapeo")
    parser.add_argument("--sources", nargs="+", help="Especificar fuentes a procesar (ej: NIST MITRE)")
    args = parser.parse_args()
    
    print(f"🔌 Conectando a Weaviate en {args.host}:{args.http_port} (gRPC {args.grpc_port})...", flush=True)
    client = get_client(args.host, args.http_port, args.grpc_port)
    
    try:
        # Setup multi-class schema
        print("🏗️  Configurando esquema multi-clase...", flush=True)
        source_mapping = ensure_multi_class_schema(client, args.schema)
        print("✅ Esquema configurado.", flush=True)
        
        # Load embedding model
        print(f"🤖 Cargando modelo de embeddings: {args.model}...", flush=True)
        model = SentenceTransformer(args.model)
        print("✅ Modelo cargado.", flush=True)
        
        # Get source to class mapping
        source_to_class = get_source_to_class_mapping()
        
        # Find chunk files
        chunks_dir = Path(args.chunks_dir)
        if not chunks_dir.exists():
            print(f"❌ ERROR: Directorio {chunks_dir} no existe")
            return
        
        # Process each source
        total_ingested = 0
        processed_sources = 0
        
        for source_path, class_name in source_to_class.items():
            # Filter by specified sources if provided
            if args.sources and class_name not in args.sources:
                continue
                
            source_dir = chunks_dir / Path(source_path).name
            if not source_dir.exists():
                print(f"⚠️  Saltando {class_name}: directorio {source_dir} no existe")
                continue
            
            # Find chunk files in source directory
            chunk_files = list(source_dir.glob("*.chunks.jsonl"))
            if not chunk_files:
                print(f"⚠️  Saltando {class_name}: no se encontraron archivos .chunks.jsonl en {source_dir}")
                continue
            
            print(f"\n📂 Procesando fuente: {class_name}")
            print(f"   📁 Directorio: {source_dir}")
            print(f"   📄 Archivos encontrados: {len(chunk_files)}")
            
            for chunk_file in chunk_files:
                print(f"\n   🔄 Procesando: {chunk_file.name}")
                ingested = ingest_to_class(
                    client=client,
                    collection_name=class_name,
                    jsonl_path=str(chunk_file),
                    model=model,
                    batch_size=args.batch,
                    limit=args.limit,
                    dry_run=args.dry_run
                )
                total_ingested += ingested
            
            processed_sources += 1
        
        # Summary
        print(f"\n🎉 INGESTA MULTI-CLASE COMPLETADA")
        print(f"   📊 Fuentes procesadas: {processed_sources}")
        print(f"   📈 Total objetos ingeridos: {total_ingested}")
        print(f"   🏛️  Colecciones creadas: {len(source_to_class)}")
        
        if args.dry_run:
            print(f"\n💡 Para ejecutar la ingesta real, elimina el flag --dry_run")
        
    finally:
        client.close()


if __name__ == "__main__":
    main()
