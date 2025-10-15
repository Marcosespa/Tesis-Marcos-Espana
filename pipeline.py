# pipeline.py
# Estilo: importar los main() de cada script y pasarles args como lista
# Solo ejecutamos Paso 1 (extract). Los demás pasos quedan comentados.

# ===== RAG =====
from src.rag.ingest.extract_pdf import main as extract
from src.export.pdf_catalog import main as generate_catalog
# from src.rag.index.ingest_to_weaviate import main as index

# ===== PDF PROCESSING =====
import importlib.util
def import_script(script_path):
    spec = importlib.util.spec_from_file_location("script", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

pdf_processor = import_script("scripts/processData/raw_to_interim/process_pdf.py")
txt_processor = import_script("scripts/processData/raw_to_interim/process_txt.py")

# ===== FT =====
# from src.ft.export_txt import main as export_txt
# from src.ft.prepare_dataset import main as prepare_ds
# from src.ft.train_lora import main as train_lora
# from src.ft.eval_ft import main as eval_ft
import sys 
import argparse
import json
import time
from pathlib import Path
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

def run_rag_pipeline():

    
    print(">>> [RAG] Paso 1: Extraer PDFs a interim/ (organizados por categoría)")
    
    # Guardar sys.argv original
    original_argv = sys.argv.copy()
    
    # Configurar argumentos para extract
    sys.argv = [
        "extract_pdf.py",
        "--in", "data/raw", # Especifica la carpeta de entrada de los PDFs
        "--out", "data/interim", # La carpeta de salida para los archivos JSONL
        "--ocr-lang", "eng",      # El idioma de OCR, cambia a 'eng' si el libro está en inglés
        "--min-chars", "20" # El umbral mínimo de caracteres para considerar que la página tiene texto
    ]
    
    try:
        extract()
    finally:
        # Restaurar sys.argv original
        sys.argv = original_argv

    print(">>> [RAG] Paso 1.5: Generar catálogo CSV de PDFs")
    
    # Configurar argumentos para generate_catalog
    sys.argv = [
        "pdf_catalog.py",
        "--interim", "data/interim",
        "--output", "data/export/pdf_catalog.csv"
    ]
    
    try:
        generate_catalog()
    finally:
        # Restaurar sys.argv original
        sys.argv = original_argv

    # print(">>> [RAG] Paso 2: Generar chunks")
    # (Ahora disponible como --step chunk)

    # print(">>> [RAG] Paso 3: Indexar en Weaviate")
    # index([
    #     "--jsonl", "data/chunks/all_chunks.jsonl"
    # ])


def run_ft_pipeline():
    import sys
    
    print(">>> [FT] Paso 1: Extraer PDFs a interim/ (base común para FT)")
    
    # Guardar sys.argv original
    original_argv = sys.argv.copy()
    
    # Configurar argumentos para extract
    sys.argv = [
        "extract_pdf.py",
        "--in", "data/raw",
        "--out", "data/interim",
        "--ocr-lang", "spa",
        "--min-chars", "20"
    ]
    
    try:
        extract()
    finally:
        # Restaurar sys.argv original
        sys.argv = original_argv

    # print(">>> [FT] Paso 2 (opcional): Exportar TXT legible por libro")
    # export_txt([
    #     "--in", "data/interim",
    #     "--out", "data/ft_raw"
    # ])

    # print(">>> [FT] Paso 3: Preparar dataset de ejemplos (JSONL)")
    # prepare_ds([
    #     "--in", "data/ft_raw",          # o directamente data/interim si tu script lo soporta
    #     "--out", "data/ft_datasets",
    #     "--cfg", "configs/ft.yaml"
    # ])

    # print(">>> [FT] Paso 4: Entrenar (LoRA/QLoRA)")
    # train_lora([
    #     "--cfg", "configs/ft.yaml"
    # ])

    # print(">>> [FT] Paso 5: Evaluar checkpoint")
    # eval_ft([
    #     "--data", "data/ft_datasets/test.jsonl",
    #     "--ckpt", "runs/last"
    # ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline RAG/FT")
    parser.add_argument("--step", choices=["pdf-process-rawToInterim", "txt-process-rawToInterim", "extract", "catalog", "chunk", "embeddings", "all"], default="all",
                        help="Paso a ejecutar: pdf-process-rawToInterim, txt-process-rawToInterim, extract, catalog, chunk, embeddings o all")
    parser.add_argument("--embed-model", default="all-MiniLM-L6-v2", help="Modelo de embeddings para chunking")
    parser.add_argument("--interim", default="data/interim", help="Directorio con .pages.jsonl por categoría")
    parser.add_argument("--chunks", default="data/chunks", help="Directorio de salida para chunks")
    parser.add_argument("--device", default=os.getenv("GPU_DEVICE") or (f"cuda:{os.getenv('GPU_ID')}" if os.getenv("GPU_ID") else None), help="Dispositivo para PyTorch/SentenceTransformers: p.ej. cuda:1 o cpu")
    args = parser.parse_args()

    def run_extract():
        original_argv = sys.argv.copy()
        sys.argv = [
            "extract_pdf.py",
            "--in", "data/raw",
            "--out", args.interim,
            "--ocr-lang", "eng",
            "--min-chars", "20",
        ]
        try:
            extract()
        finally:
            sys.argv = original_argv

    def run_catalog():
        original_argv = sys.argv.copy()
        sys.argv = [
            "pdf_catalog.py",
            "--interim", args.interim,
            "--output", "data/export/pdf_catalog.csv",
        ]
        try:
            generate_catalog()
        finally:
            sys.argv = original_argv

    def run_pdf_process():
        print(">>> [PDF] Procesando PDFs de NIST, OAPEN y USENIX a interim/")
        pdf_processor.process_pdf_sources(force_overwrite=True)

    def run_txt_process():
        print(">>> [TXT] Procesando TXTs de AISecKG, AnnoCTR, MITRE, OWASP, SecurityTools a interim/")
        txt_processor.process_txt_sources(force_overwrite=True)

    def run_chunk():
        """Procesar todos los archivos JSONL usando el sistema de chunking mejorado"""
        from src.rag.process.chunking import process_multiple_jsonl_files
        import time

        base_in = Path(args.interim)
        base_out = Path(args.chunks)
        base_out.mkdir(parents=True, exist_ok=True)

        # Configuración de chunking (defaults; el módulo también lee .env)
        config = {
            "target_tokens": 200,
            "min_tokens": 100,
            "max_tokens": 256,
            "overlap_ratio": 0.18,
            "embed_model": args.embed_model,
        }

        def validate_and_clean_metadata(metadata, chunk_id):
            """Valida y limpia metadatos según los estándares definidos"""
            import sys
            sys.path.append('scripts/processData/raw_to_interim')
            try:
                from metadata_standards import validate_metadata
            except ImportError:
                # Fallback si no se puede importar metadata_standards
                def validate_metadata(meta):
                    return []
            
            # Limpiar metadatos para serialización JSON
            clean_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    clean_metadata[key] = value
                else:
                    clean_metadata[key] = str(value)
            
            # Agregar metadatos de chunking estándar
            clean_metadata.update({
                "chunk_id": chunk_id,
                "chunk_tokens": clean_metadata.get("chunk_tokens", 0),
                "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "processing_version": "enhanced_chunking_v1.0"
            })
            
            # Validar según estándares (opcional - solo mostrar warnings)
            try:
                errors = validate_metadata(clean_metadata)
                if errors:
                    print(f"    ⚠️  Warnings de metadata en chunk {chunk_id}: {len(errors)} issues")
            except Exception as e:
                print(f"    ⚠️  Error validando metadata: {e}")
            
            return clean_metadata
        
        print(">>> [RAG] Iniciando chunking de todos los archivos JSONL")
        print(f"📊 Configuración: target={config['target_tokens']}, min={config['min_tokens']}, max={config['max_tokens']}, device={args.device}")

        start_time = time.time()
        # Delega al motor robusto que valida, preserva estructura y soporta device
        try:
            process_multiple_jsonl_files(
                input_path=str(base_in),
                output_path=str(base_out),
                device=args.device,
                embed_model=args.embed_model,
                target_tokens=config["target_tokens"],
                min_tokens=config["min_tokens"],
                max_tokens=config["max_tokens"],
                overlap_ratio=config["overlap_ratio"],
            )
        except TypeError:
            # Compatibilidad si la firma difiere; al menos pasa rutas y device
            process_multiple_jsonl_files(
                str(base_in),
                str(base_out),
                device=args.device,
            )

        duration = time.time() - start_time
        print(f"\n📊 RESUMEN DEL CHUNKING")
        print(f"{'='*50}")
        print(f"⏱️  Tiempo total: {duration:.1f} segundos")
        print(f"📁 Directorio de salida: {base_out}")

    def run_embeddings():
        """Generar y guardar embeddings localmente para todos los .chunks.jsonl"""
        import subprocess

        chunks_root = Path(args.chunks)
        out_root = Path("data/embeddings")
        out_root.mkdir(parents=True, exist_ok=True)

        chunk_files = sorted(chunks_root.glob("**/*.chunks.jsonl"))
        if not chunk_files:
            print("❌ No se encontraron archivos *.chunks.jsonl en data/chunks")
            return

        print(f"🚀 Generando embeddings para {len(chunk_files)} archivos... (device={args.device})")
        for cf in chunk_files:
            rel = cf.relative_to(chunks_root)
            out_dir = out_root / rel.parent
            out_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                "src/rag/index/ingest_weaviate.py",
                "--jsonl", str(cf),
                "--save-embeddings",
                "--embeddings-dir", str(out_dir),
                "--batch", "128",
                "--model", args.embed_model,
            ]
            if args.device:
                cmd.extend(["--device", args.device])

            print(f"  ▶️  {cf}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"    ❌ Error generando embeddings para {cf}: {e}")

    if args.step == "pdf-process-rawToInterim":
        run_pdf_process()
    elif args.step == "txt-process-rawToInterim":
        run_txt_process()
    elif args.step == "extract":
        run_extract()
    elif args.step == "catalog":
        run_catalog()
    elif args.step == "chunk":
        run_chunk()
    elif args.step == "embeddings":
        run_embeddings()
    else:
        # all
        run_pdf_process()
        run_txt_process()
        run_extract()
        run_catalog()
        run_chunk()
        run_embeddings()