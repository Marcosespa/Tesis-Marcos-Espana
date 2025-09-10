#!/usr/bin/env python3
"""
Script rápido para encontrar y eliminar archivos duplicados en OAPEN_PDFs/
Primero compara por nombre y tamaño, luego por hash solo si es necesario.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict


def find_duplicates_by_name_and_size(root_dir: str) -> Dict[str, List[str]]:
    """
    Encuentra archivos duplicados comparando nombre y tamaño.
    Mucho más rápido que calcular hashes.
    """
    print(f"Escaneando directorio: {root_dir}")
    
    # Diccionario para agrupar archivos por (nombre, tamaño)
    file_groups = defaultdict(list)
    
    # Contadores
    total_files = 0
    
    # Recorrer todos los archivos
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.pdf') or file.endswith('.part'):
                file_path = os.path.join(root, file)
                total_files += 1
                
                if total_files % 50 == 0:
                    print(f"Procesando archivo {total_files}...")
                
                try:
                    size = os.path.getsize(file_path)
                    # Usar nombre del archivo y tamaño como clave
                    key = (file, size)
                    file_groups[key].append(file_path)
                except (IOError, OSError) as e:
                    print(f"[ERROR] Error procesando {file_path}: {e}")
    
    print(f"Archivos escaneados: {total_files}")
    
    # Filtrar solo los grupos con duplicados
    duplicates = {key: paths for key, paths in file_groups.items() if len(paths) > 1}
    
    return duplicates


def choose_file_to_keep(duplicate_paths: List[str]) -> Tuple[str, List[str]]:
    """
    Decide qué archivo mantener y cuáles eliminar.
    Prioridad: archivos en subcarpetas sobre archivos en la raíz.
    """
    # Separar archivos en raíz y subcarpetas
    root_files = []
    subfolder_files = []
    
    for path in duplicate_paths:
        path_obj = Path(path)
        # Contar niveles de directorio desde OAPEN_PDFs
        relative_path = path_obj.relative_to(Path("OAPEN_PDFs"))
        if len(relative_path.parts) == 1:  # Está en la raíz
            root_files.append(path)
        else:
            subfolder_files.append(path)
    
    # Prioridad: mantener archivos en subcarpetas, eliminar de la raíz
    if subfolder_files:
        keep_file = subfolder_files[0]
        delete_files = root_files + subfolder_files[1:]
    else:
        # Si todos están en la raíz, mantener el primero
        keep_file = duplicate_paths[0]
        delete_files = duplicate_paths[1:]
    
    return keep_file, delete_files


def remove_duplicates(duplicates: Dict[Tuple, List[str]], dry_run: bool = True) -> Dict[str, int]:
    """
    Elimina archivos duplicados.
    dry_run=True: solo muestra qué se eliminaría sin hacerlo.
    """
    stats = {
        "duplicate_groups": len(duplicates),
        "files_to_delete": 0,
        "space_to_save": 0,
        "deleted_files": 0,
        "actual_space_saved": 0
    }
    
    print(f"\n{'='*60}")
    print(f"ANÁLISIS DE DUPLICADOS")
    print(f"{'='*60}")
    print(f"Grupos de duplicados encontrados: {len(duplicates)}")
    
    for i, ((filename, size), paths) in enumerate(duplicates.items(), 1):
        print(f"\n--- Grupo {i} ---")
        print(f"Archivo: {filename}")
        print(f"Tamaño: {format_bytes(size)}")
        print(f"Duplicados: {len(paths)}")
        
        # Mostrar información de cada archivo
        for path in paths:
            relative_path = os.path.relpath(path)
            print(f"  - {relative_path}")
        
        # Decidir qué archivo mantener
        keep_file, delete_files = choose_file_to_keep(paths)
        
        print(f"\nMANTENER: {os.path.relpath(keep_file)}")
        print(f"ELIMINAR:")
        
        for delete_file in delete_files:
            stats["files_to_delete"] += 1
            stats["space_to_save"] += size
            
            if dry_run:
                print(f"  - {os.path.relpath(delete_file)} ({format_bytes(size)}) [SIMULACIÓN]")
            else:
                try:
                    os.remove(delete_file)
                    stats["deleted_files"] += 1
                    stats["actual_space_saved"] += size
                    print(f"  - {os.path.relpath(delete_file)} ({format_bytes(size)}) [ELIMINADO]")
                except OSError as e:
                    print(f"  - {os.path.relpath(delete_file)} [ERROR: {e}]")
    
    return stats


def format_bytes(bytes_value: int) -> str:
    """Formatea bytes en una unidad legible"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} TB"


def main():
    """Función principal"""
    root_dir = "OAPEN_PDFs"
    
    if not os.path.exists(root_dir):
        print(f"Error: El directorio {root_dir} no existe")
        return
    
    print("🔍 Buscando archivos duplicados por nombre y tamaño...")
    duplicates = find_duplicates_by_name_and_size(root_dir)
    
    if not duplicates:
        print("✅ No se encontraron archivos duplicados")
        return
    
    # Mostrar resumen
    print(f"\n📊 RESUMEN:")
    print(f"Grupos de duplicados: {len(duplicates)}")
    
    total_duplicate_files = sum(len(paths) for paths in duplicates.values())
    print(f"Archivos duplicados totales: {total_duplicate_files}")
    
    # Simulación primero
    print(f"\n🔍 SIMULACIÓN (no se elimina nada):")
    stats = remove_duplicates(duplicates, dry_run=True)
    
    print(f"\n📈 ESTADÍSTICAS DE SIMULACIÓN:")
    print(f"Archivos a eliminar: {stats['files_to_delete']}")
    print(f"Espacio a liberar: {format_bytes(stats['space_to_save'])}")
    
    # Preguntar al usuario si quiere proceder
    if stats['files_to_delete'] > 0:
        print(f"\n❓ ¿Deseas proceder con la eliminación? (y/N): ", end="")
        response = input().strip().lower()
        
        if response in ['y', 'yes', 'sí', 'si']:
            print(f"\n🗑️  ELIMINANDO DUPLICADOS...")
            stats = remove_duplicates(duplicates, dry_run=False)
            
            print(f"\n✅ ELIMINACIÓN COMPLETADA:")
            print(f"Archivos eliminados: {stats['deleted_files']}")
            print(f"Espacio liberado: {format_bytes(stats['actual_space_saved'])}")
        else:
            print("❌ Operación cancelada")
    else:
        print("✅ No hay archivos para eliminar")


if __name__ == "__main__":
    main()
