import argparse
import json
from pathlib import Path
import uuid
from typing import Any, Iterable, List
import sys
import time
import os
from dotenv import load_dotenv

import numpy as np
from sentence_transformers import SentenceTransformer

# Cargar variables de entorno
load_dotenv()

# Support running as a script regardless of PYTHONPATH
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[3]  # .../DatosTesis/src
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

try:
    # When project root is on PYTHONPATH
    from src.index.weaviate_client import get_client, ensure_schema
except ModuleNotFoundError:  # Fallback when src is on sys.path
    from index.weaviate_client import get_client, ensure_schema


def batched(iterable: Iterable[Any], batch_size: int) -> Iterable[List[Any]]:
    batch: List[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for chunks and optionally ingest to Weaviate")
    parser.add_argument("--jsonl", required=True, help="Path to chunks JSONL")
    parser.add_argument("--schema", default="configs/weaviate.schema.json", help="Schema JSON path")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--http_port", type=int, default=8080)
    parser.add_argument("--grpc_port", type=int, default=50051)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="Máximo de objetos a procesar (0 = todos)")
    parser.add_argument("--dry_run", action="store_true", help="No inserta; solo muestra mapeo de los primeros N")
    parser.add_argument("--save-embeddings", action="store_true", help="Guardar embeddings en archivos locales")
    parser.add_argument("--embeddings-dir", default="data/embeddings", help="Directorio para guardar embeddings")
    parser.add_argument("--device", default=None, help="Dispositivo para el modelo (cuda:1, cpu, etc.)")
    args = parser.parse_args()

    # Configurar dispositivo para el modelo
    device = args.device or os.getenv('GPU_DEVICE') or os.getenv('GPU_ID', 'cpu')
    if device.isdigit():
        device = f"cuda:{device}"
    
    print(f"🔧 Usando dispositivo: {device}", flush=True)
    
    # Solo conectar a Weaviate si no estamos solo guardando embeddings
    client = None
    if not args.save_embeddings:
        print(f"Conectando a Weaviate en {args.host}:{args.http_port} (gRPC {args.grpc_port})...", flush=True)
        client = get_client(args.host, args.http_port, args.grpc_port)
        try:
            print("Asegurando schema/colección...", flush=True)
            ensure_schema(client, args.schema)
            print("Schema listo.", flush=True)
        except Exception as e:
            print(f"⚠️  Error conectando a Weaviate: {e}")
            print("📁 Continuando solo con generación de embeddings...", flush=True)
            client = None

    print(f"Cargando modelo de embeddings: {args.model} en {device}...", flush=True)
    model = SentenceTransformer(args.model, device=device)
    print(f"✅ Modelo {args.model} cargado en {device}.", flush=True)
    
    # Crear directorio de embeddings si es necesario
    if args.save_embeddings:
        embeddings_dir = Path(args.embeddings_dir)
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Directorio de embeddings: {embeddings_dir}", flush=True)

    # Solo obtener schema si tenemos cliente
    collection = None
    if client:
        schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
        class_name = schema["class"]
        collection = client.collections.get(class_name)

    def records():
        with open(args.jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                yield json.loads(line)

    # Verificar archivo y contar líneas de forma rápida
    in_path = Path(args.jsonl)
    if not in_path.exists():
        print(f"ERROR: No existe el archivo {in_path}", flush=True)
        return
    try:
        with open(in_path, "r", encoding="utf-8") as _f:
            approx_total = sum(1 for _ in _f)
        print(f"📄 Fuente: {in_path} (~{approx_total} líneas)", flush=True)
    except Exception as e:
        print(f"No se pudo contar líneas: {e}", flush=True)

    total = 0
    last_log = time.time()
    start = last_log
    batch_idx = 0
    print(f"🔄 Procesando en lotes de {args.batch}...", flush=True)
    
    # Preparar archivos de salida si guardamos embeddings
    embeddings_file = None
    metadata_file = None
    if args.save_embeddings:
        base_name = in_path.stem
        embeddings_file = embeddings_dir / f"{base_name}.embeddings.npy"
        metadata_file = embeddings_dir / f"{base_name}.metadata.jsonl"
        print(f"💾 Guardando embeddings en: {embeddings_file}", flush=True)
        print(f"📋 Guardando metadata en: {metadata_file}", flush=True)
    
    for batch in batched(records(), args.batch):
        batch_idx += 1
        texts = [r.get("content") or r.get("text") or "" for r in batch]
        vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        
        # Guardar embeddings si se solicita
        if args.save_embeddings:
            # Guardar embeddings como numpy array
            if batch_idx == 1:
                all_vectors = vectors
            else:
                all_vectors = np.vstack([all_vectors, vectors])
            
            # Guardar metadata
            with open(metadata_file, 'a', encoding='utf-8') as f:
                for r in batch:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
        
        # Insert with explicit vectors using DataObject (solo si tenemos cliente)
        if collection:
            from weaviate.collections.classes.data import DataObject
            data_objects = []
        for r, vec in zip(batch, vectors):
            md = r.get("metadata", {}) or {}
            # Build robust identifiers across heterogeneous sources
            doc_id = (
                r.get("doc_id")
                or md.get("doc_id")
                or r.get("source_id")
                or r.get("chunk_id")
                or md.get("source_id")
                or md.get("file_id")
                or md.get("page_name")
                or (
                    (r.get("source_type") or "").strip() + "_" + (r.get("source_file") or "").strip()
                    if r.get("source_type") or r.get("source_file") else None
                )
                or md.get("source")
                or r.get("source")
                or r.get("id")
                or "unknown"
            )
            title = (
                r.get("title")
                or md.get("title")
                or md.get("doc_title")
                or md.get("page_name")
                or r.get("source_file")
                or r.get("source_id")
                or r.get("source_type")
                or md.get("document_type")
                or ""
            )
            # Prefer page fields from record first, then metadata; avoid using word-index fields
            page_start = (
                r.get("page_start")
                or md.get("page_start")
                or r.get("page")
                or md.get("page")
                or md.get("page_num_real")
                or 0
            )
            page_end = (
                r.get("page_end")
                or md.get("page_end")
                or r.get("page")
                or md.get("page")
                or md.get("page_num_real")
                or page_start
            )
            properties = {
                "docId": str(doc_id),
                "level": int(
                    r.get("chunk_index")
                    or md.get("level")
                    or md.get("section_level")
                    or 0
                ),
                "title": str(title),
                "text": r.get("text") or r.get("content") or "",
                "pageStart": int(page_start) if page_start is not None else 0,
                "pageEnd": int(page_end) if page_end is not None else 0,
            }
            # Deterministic UUID by chunk_id if present; else fallback to derived key
            chunk_id = r.get("chunk_id") or md.get("chunk_id")
            if chunk_id is not None:
                uid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"chunk:{chunk_id}"))
            else:
                uid_base = f"{properties['docId']}|{properties['level']}|{properties['pageStart']}|{properties['pageEnd']}|{(properties['text'] or '')[:96]}"
                uid = str(uuid.uuid5(uuid.NAMESPACE_URL, uid_base))
            data_objects.append(DataObject(uuid=uid, properties=properties, vector=vec.astype(np.float32)))

        if collection:
            if args.dry_run:
                print("Previsualización de mapeo (hasta 5 del batch):", flush=True)
                for obj in data_objects[:5]:
                    print({
                        "uuid": obj.uuid,
                        "docId": obj.properties.get("docId"),
                        "level": obj.properties.get("level"),
                        "title": obj.properties.get("title"),
                        "pageStart": obj.properties.get("pageStart"),
                        "pageEnd": obj.properties.get("pageEnd"),
                        "text_preview": (obj.properties.get("text") or "")[:80]
                    }, flush=True)
            else:
                try:
                    collection.data.insert_many(data_objects)
                except Exception as e:
                    print(f"Error insertando batch {batch_idx}: {e}", flush=True)
                    raise
        total += len(batch)

        # Respetar límite si se indicó
        if args.limit and total >= args.limit:
            print(f"Límite alcanzado ({args.limit}). Deteniendo.", flush=True)
            break

        now = time.time()
        if now - last_log >= 2.0:
            elapsed = now - start
            rate = total / elapsed if elapsed > 0 else 0.0
            print(f"Batch {batch_idx} | acumulado={total} | {rate:.1f} objs/s", flush=True)
            last_log = now
    
    # Guardar embeddings finales si se solicitó
    if args.save_embeddings and 'all_vectors' in locals():
        np.save(embeddings_file, all_vectors)
        print(f"✅ Embeddings guardados: {all_vectors.shape} vectores en {embeddings_file}", flush=True)
        
        # Crear archivo de información
        info_file = embeddings_dir / f"{base_name}.info.json"
        info_data = {
            "model": args.model,
            "device": device,
            "total_vectors": int(all_vectors.shape[0]),
            "vector_dimension": int(all_vectors.shape[1]),
            "source_file": str(in_path),
            "embeddings_file": str(embeddings_file),
            "metadata_file": str(metadata_file),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_time": time.time() - start
        }
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, indent=2, ensure_ascii=False)
        print(f"📋 Información guardada en: {info_file}", flush=True)

    elapsed = time.time() - start
    rate = total / elapsed if elapsed > 0 else 0.0
    if collection:
        print(f"✅ Ingest terminado: {total} objetos en {elapsed:.1f}s ({rate:.1f} objs/s)", flush=True)
    else:
        print(f"✅ Embeddings generados: {total} objetos en {elapsed:.1f}s ({rate:.1f} objs/s)", flush=True)
    
    if client:
        client.close()


if __name__ == "__main__":
    main()


