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
    parser.add_argument("--step", choices=["pdf-process-rawToInterim", "txt-process-rawToInterim", "extract", "catalog", "chunk", "all"], default="all",
                        help="Paso a ejecutar: pdf-process-rawToInterim, txt-process-rawToInterim, extract, catalog, chunk o all")
    parser.add_argument("--embed-model", default="all-MiniLM-L6-v2", help="Modelo de embeddings para chunking")
    parser.add_argument("--interim", default="data/interim", help="Directorio con .pages.jsonl por categoría")
    parser.add_argument("--chunks", default="data/chunks", help="Directorio de salida para chunks")
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
        import sys
        sys.path.append('scripts/processData/interim_to_chunks')
        from chunking import hybrid_chunk_jsonl
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time
        
        base_in = Path(args.interim)
        base_out = Path(args.chunks)
        base_out.mkdir(parents=True, exist_ok=True)
        
        # Configuración de chunking
        config = {
            "target_tokens": 200,
            "min_tokens": 100,
            "max_tokens": 256,
            "overlap_ratio": 0.18,
            "embed_model": args.embed_model
        }
        
        def find_jsonl_files(directory):
            """Encuentra todos los archivos JSONL en un directorio"""
            jsonl_files = []
            for pattern in ["*.jsonl", "**/*.jsonl"]:
                jsonl_files.extend(directory.glob(pattern))
            
            # Filtrar archivos que parecen ser de páginas/documentos
            filtered_files = []
            for file_path in jsonl_files:
                if any(keyword in file_path.name.lower() for keyword in ['pages', 'documents', 'all_documents']):
                    filtered_files.append(file_path)
            
            return sorted(filtered_files)
        
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
        
        def process_single_file(jsonl_path, output_dir):
            """Procesa un archivo JSONL individual"""
            try:
                print(f"  📄 Procesando: {jsonl_path.name}")
                
                # Procesar el archivo
                documents = hybrid_chunk_jsonl(str(jsonl_path), **config)
                
                # Crear nombre de archivo de salida
                output_filename = f"{jsonl_path.stem}.chunks.jsonl"
                output_path = output_dir / output_filename
                
                # Guardar resultados con validación de metadata
                with open(output_path, 'w', encoding='utf-8') as f:
                    for i, doc in enumerate(documents):
                        # Validar y limpiar metadatos según estándares
                        clean_metadata = validate_and_clean_metadata(doc.metadata, i)
                        
                        chunk_data = {
                            "id": i,
                            "content": doc.page_content,
                            "metadata": clean_metadata
                        }
                        f.write(json.dumps(chunk_data, ensure_ascii=False) + "\n")
                
                print(f"    ✅ {len(documents)} chunks guardados en {output_filename}")
                return {
                    "file": str(jsonl_path),
                    "chunks": len(documents),
                    "output": str(output_path),
                    "status": "success"
                }
                
            except Exception as e:
                print(f"    ❌ Error procesando {jsonl_path.name}: {e}")
                return {
                    "file": str(jsonl_path),
                    "chunks": 0,
                    "output": None,
                    "status": "error",
                    "error": str(e)
                }
        
        print(">>> [RAG] Iniciando chunking de todos los archivos JSONL")
        print(f"📊 Configuración: target={config['target_tokens']}, min={config['min_tokens']}, max={config['max_tokens']}")
        
        # Buscar todos los archivos JSONL
        all_jsonl_files = find_jsonl_files(base_in)
        
        if not all_jsonl_files:
            print("❌ No se encontraron archivos JSONL para procesar")
            return
        
        print(f"📁 Encontrados {len(all_jsonl_files)} archivos JSONL")
        
        # Procesar archivos por categoría
        results = []
        start_time = time.time()
        
        for sub_dir in sorted([p for p in base_in.iterdir() if p.is_dir()]):
            print(f"\n📂 Procesando categoría: {sub_dir.name}")
            
            # Buscar archivos JSONL en esta categoría
            category_files = find_jsonl_files(sub_dir)
            
            if not category_files:
                print(f"  ⚠️  No hay archivos JSONL en {sub_dir.name}")
                continue
            
            # Crear directorio de salida para esta categoría
            out_dir = base_out / sub_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Procesar archivos de esta categoría
            for jsonl_file in category_files:
                result = process_single_file(jsonl_file, out_dir)
                results.append(result)
        
        # Estadísticas finales
        end_time = time.time()
        duration = end_time - start_time
        
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "error"]
        total_chunks = sum(r["chunks"] for r in successful)
        
        print(f"\n📊 RESUMEN DEL CHUNKING")
        print(f"{'='*50}")
        print(f"⏱️  Tiempo total: {duration:.1f} segundos")
        print(f"✅ Archivos exitosos: {len(successful)}")
        print(f"❌ Archivos fallidos: {len(failed)}")
        print(f"📄 Total de chunks generados: {total_chunks}")
        print(f"📁 Directorio de salida: {base_out}")
        
        if successful:
            print(f"\n📋 ARCHIVOS PROCESADOS:")
            for result in successful:
                file_name = Path(result['file']).name
                print(f"   • {file_name}: {result['chunks']} chunks")
        
        if failed:
            print(f"\n❌ ARCHIVOS CON ERRORES:")
            for result in failed:
                file_name = Path(result['file']).name
                print(f"   • {file_name}: {result['error']}")
        
        # Archivo global deshabilitado para RAG agéntico
        # Los chunks se mantienen organizados por categorías para mejor rendimiento

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
    else:
        # all
        run_pdf_process()
        run_txt_process()
        run_extract()
        run_catalog()
        run_chunk()