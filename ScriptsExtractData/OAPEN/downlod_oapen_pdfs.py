import os
import json
import time
import signal
import sys
from typing import Dict, Any, Iterable, Optional, List

import requests


BASE_URL = "https://library.oapen.org"
INPUT_JSON_PATH = os.path.abspath("cybersecurity_books_filtered.json")
OUTPUT_DIR = os.path.abspath("OAPEN_PDFs")
PROGRESS_FILE = os.path.abspath("download_progress.json")


def ensure_directory_exists(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def clean_corrupted_partials(output_dir: str) -> int:
    """Limpia archivos .part que puedan estar corruptos o incompletos"""
    cleaned = 0
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.part'):
                part_path = os.path.join(root, file)
                # Si el archivo parcial es muy pequeño (< 1KB), probablemente esté corrupto
                if os.path.getsize(part_path) < 1024:
                    print(f"[CLEAN] Eliminando archivo parcial corrupto: {file}")
                    os.remove(part_path)
                    cleaned += 1
    return cleaned


def save_progress(current_item: int, total_items: int, downloaded: int, skipped: int, failed: int) -> None:
    """Guarda el progreso actual en un archivo JSON"""
    progress = {
        "current_item": current_item,
        "total_items": total_items,
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
        "timestamp": time.time()
    }
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def load_progress() -> Optional[Dict[str, Any]]:
    """Carga el progreso desde el archivo JSON"""
    if not os.path.exists(PROGRESS_FILE):
        return None
    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def clear_progress() -> None:
    """Elimina el archivo de progreso"""
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)


# Variables globales para el manejo de interrupciones
current_progress = {"current_item": 0, "total_items": 0, "downloaded": 0, "skipped": 0, "failed": 0}


def signal_handler(signum, frame):
    """Maneja la interrupción (Ctrl+C) guardando el progreso"""
    print(f"\n[INTERRUPT] Guardando progreso antes de salir...")
    save_progress(
        current_progress["current_item"],
        current_progress["total_items"],
        current_progress["downloaded"],
        current_progress["skipped"],
        current_progress["failed"]
    )
    print(f"[INTERRUPT] Progreso guardado. Puedes reanudar ejecutando el script nuevamente.")
    sys.exit(0)


def iter_items_from_json(json_path: str) -> Iterable[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("El JSON esperado debe ser una lista de items.")
    return data


def build_pdf_entries(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    bitstreams = item.get("bitstreams") or []
    pdf_entries: List[Dict[str, Any]] = []
    for b in bitstreams:
        if (b.get("mimeType") == "application/pdf") and b.get("retrieveLink"):
            pdf_entries.append({
                "name": b.get("name") or f"{b.get('uuid', 'unknown')}.pdf",
                "retrieve_link": b["retrieveLink"],
                "size_bytes": b.get("sizeBytes"),
            })
    return pdf_entries


def download_file(url: str, dest_path: str, expected_size: Optional[int] = None, *, max_retries: int = 3, timeout: int = 60) -> bool:
    # Verificar si el archivo ya existe y está completo
    if os.path.exists(dest_path):
        if expected_size is not None and os.path.getsize(dest_path) == expected_size:
            print(f"[SKIP] Ya existe y coincide tamaño: {os.path.basename(dest_path)}")
            return True
        else:
            print(f"[INFO] Archivo existe pero tamaño no coincide, se re-descargará: {os.path.basename(dest_path)}")
    
    # Verificar si hay un archivo parcial que podemos reanudar
    tmp_path = dest_path + ".part"
    resume_pos = 0
    if os.path.exists(tmp_path):
        resume_pos = os.path.getsize(tmp_path)
        print(f"[RESUME] Reanudando descarga desde byte {resume_pos}: {os.path.basename(dest_path)}")

    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            headers = {
                "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
                "User-Agent": "Datos-Tesis/1.0 (+download-script)"
            }
            
            # Si estamos reanudando, agregar header Range
            if resume_pos > 0:
                headers["Range"] = f"bytes={resume_pos}-"
            
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as resp:
                # Para reanudación, 206 es el código esperado
                if resume_pos > 0 and resp.status_code not in (200, 206):
                    raise requests.exceptions.HTTPError(f"Error reanudando descarga: {resp.status_code}")
                elif resume_pos == 0:
                    resp.raise_for_status()

                # Abrir archivo en modo append si estamos reanudando
                mode = "ab" if resume_pos > 0 else "wb"
                with open(tmp_path, mode) as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)

                os.replace(tmp_path, dest_path)

            if expected_size is not None:
                actual_size = os.path.getsize(dest_path)
                if actual_size != expected_size:
                    raise IOError(f"Tamaño inesperado: esperado={expected_size}, real={actual_size}")

            print(f"[OK] Descargado: {os.path.basename(dest_path)}")
            return True
        except Exception as e:
            wait_s = min(2 ** attempt, 10)
            print(f"[WARN] Intento {attempt}/{max_retries} falló para {os.path.basename(dest_path)}: {e}. Reintentando en {wait_s}s...")
            time.sleep(wait_s)

    print(f"[FAIL] No se pudo descargar: {os.path.basename(dest_path)}")
    return False


def main() -> None:
    # Configurar manejo de interrupciones
    signal.signal(signal.SIGINT, signal_handler)
    
    if not os.path.exists(INPUT_JSON_PATH):
        raise FileNotFoundError(f"No se encontró el archivo JSON: {INPUT_JSON_PATH}")

    ensure_directory_exists(OUTPUT_DIR)
    
    # Limpiar archivos parciales corruptos
    print("Limpiando archivos parciales corruptos...")
    cleaned = clean_corrupted_partials(OUTPUT_DIR)
    if cleaned > 0:
        print(f"Se eliminaron {cleaned} archivos parciales corruptos")

    items = iter_items_from_json(INPUT_JSON_PATH)
    total_items = len(items)
    print(f"Items en JSON: {total_items}")

    # Cargar progreso previo si existe
    progress = load_progress()
    start_item = 0
    ok = 0
    skipped = 0
    failed = 0
    
    if progress:
        start_item = progress.get("current_item", 0)
        ok = progress.get("downloaded", 0)
        skipped = progress.get("skipped", 0)
        failed = progress.get("failed", 0)
        print(f"[RESUME] Reanudando desde item {start_item + 1}/{total_items}")
        print(f"[RESUME] Progreso previo: {ok} descargados, {skipped} omitidos, {failed} fallidos")
    else:
        print("[NEW] Iniciando descarga desde el principio")

    total_pdfs = 0
    
    # Actualizar variables globales para el manejo de interrupciones
    current_progress["total_items"] = total_items
    current_progress["downloaded"] = ok
    current_progress["skipped"] = skipped
    current_progress["failed"] = failed
    
    for idx, item in enumerate(items, 1):
        # Saltar items ya procesados si estamos reanudando
        if idx <= start_item:
            continue
            
        # Actualizar progreso actual
        current_progress["current_item"] = idx
            
        title = None
        try:
            # Buscar título amigable si existe
            metadata = item.get("metadata") or []
            for m in metadata:
                if m.get("key") == "dc.title":
                    title = m.get("value")
                    break

            pdfs = build_pdf_entries(item)
            if not pdfs:
                print(f"[{idx}/{total_items}] Sin PDFs - {item.get('name') or title or item.get('handle')}")
                continue

            for p in pdfs:
                total_pdfs += 1
                retrieve_link = p["retrieve_link"]
                # Construir URL absoluta
                url = retrieve_link if retrieve_link.startswith("http") else (BASE_URL + retrieve_link)

                # Normalizar nombre
                safe_name = p["name"].replace("/", "_").replace("\\", "_")
                dest_path = os.path.join(OUTPUT_DIR, safe_name)

                # Verificar si ya existe y está completo
                if os.path.exists(dest_path) and p.get("size_bytes"):
                    if os.path.getsize(dest_path) == p["size_bytes"]:
                        print(f"[{idx}/{total_items}] [SKIP] Ya existe: {safe_name}")
                        skipped += 1
                        current_progress["skipped"] = skipped
                        continue

                print(f"[{idx}/{total_items}] Descargando: {safe_name} ({p.get('size_bytes', 'desconocido')} bytes)")
                if download_file(url, dest_path, expected_size=p.get("size_bytes")):
                    ok += 1
                    current_progress["downloaded"] = ok
                else:
                    failed += 1
                    current_progress["failed"] = failed
                    
            # Guardar progreso cada 10 items
            if idx % 10 == 0:
                save_progress(idx, total_items, ok, skipped, failed)
                
            # Mostrar progreso cada 100 items
            if idx % 100 == 0:
                print(f"[PROGRESS] Procesados {idx}/{total_items} items. PDFs: {ok} descargados, {skipped} omitidos, {failed} fallidos")
                
        except Exception as e:
            print(f"[{idx}/{total_items}] Error procesando item: {e}")
            failed += 1
            current_progress["failed"] = failed

    # Limpiar archivo de progreso al completar
    clear_progress()
    
    print(f"\n=== RESUMEN FINAL ===")
    print(f"Items procesados: {total_items}")
    print(f"PDFs detectados: {total_pdfs}")
    print(f"PDFs descargados: {ok}")
    print(f"PDFs omitidos (ya existían): {skipped}")
    print(f"PDFs fallidos: {failed}")
    print(f"Archivos guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

