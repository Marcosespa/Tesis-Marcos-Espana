# pipeline.py
# Estilo: importar los main() de cada script y pasarles args como lista
# Solo ejecutamos Paso 1 (extract). Los demás pasos quedan comentados.

# ===== RAG =====
from src.rag.ingest.extract_pdf import main as extract
from src.export.pdf_catalog import main as generate_catalog
# from src.rag.index.ingest_to_weaviate import main as index

# ===== FT =====
# from src.ft.export_txt import main as export_txt
# from src.ft.prepare_dataset import main as prepare_ds
# from src.ft.train_lora import main as train_lora
# from src.ft.eval_ft import main as eval_ft
import sys 
from io import StringIO
import argparse
import subprocess
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
    parser.add_argument("--step", choices=["extract", "catalog", "chunk", "all"], default="all",
                        help="Paso a ejecutar: extract, catalog, chunk o all")
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

    def run_chunk():
        base_in = Path(args.interim)
        base_out = Path(args.chunks)
        base_out.mkdir(parents=True, exist_ok=True)
        # recorrer subcarpetas de categorias
        for sub in sorted([p for p in base_in.iterdir() if p.is_dir()]):
            out_dir = base_out / sub.name
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f">>> [RAG] Chunking categoría: {sub.name}")
            cmd = [
                sys.executable, "src/rag/process/chunking.py", "dummy",
                "--input-dir", str(sub),
                "--output-dir", str(out_dir),
                "--embed-model", args.embed_model,
            ]
            # Ejecutar y mostrar salida en tiempo real
            subprocess.run(" ".join(cmd) + " | cat", shell=True, check=False)

    if args.step == "extract":
        run_extract()
    elif args.step == "catalog":
        run_catalog()
    elif args.step == "chunk":
        run_chunk()
    else:
        # all
        run_extract()
        run_catalog()
        run_chunk()