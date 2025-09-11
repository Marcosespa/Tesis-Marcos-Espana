#!/usr/bin/env python3
"""
Script para eliminar archivos PDF de la carpeta 'otros' que no son de interés
"""

import os
import shutil
from pathlib import Path


def delete_otros_pdfs():
    """Elimina todos los archivos PDF de la carpeta 'otros'"""
    
    otros_dir = "OAPEN_PDFs/otros"
    
    if not os.path.exists(otros_dir):
        print(f"❌ La carpeta {otros_dir} no existe")
        return
    
    # Contar archivos antes de eliminar
    pdf_files = [f for f in os.listdir(otros_dir) if f.endswith('.pdf')]
    total_files = len(pdf_files)
    
    if total_files == 0:
        print("✅ No hay archivos PDF en la carpeta 'otros'")
        return
    
    # Calcular tamaño total
    total_size = 0
    for file in pdf_files:
        file_path = os.path.join(otros_dir, file)
        total_size += os.path.getsize(file_path)
    
    def format_bytes(bytes_value):
        """Formatea bytes en una unidad legible"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} TB"
    
    print("🗑️  ELIMINACIÓN DE ARCHIVOS 'OTROS'")
    print("=" * 50)
    print(f"Archivos a eliminar: {total_files}")
    print(f"Espacio a liberar: {format_bytes(total_size)}")
    print(f"Carpeta: {otros_dir}")
    
    # Confirmar eliminación
    print(f"\n⚠️  ¿Estás seguro de que quieres eliminar {total_files} archivos?")
    print("Esta acción NO se puede deshacer.")
    response = input("Escribe 'ELIMINAR' para confirmar: ").strip()
    
    if response != "ELIMINAR":
        print("❌ Operación cancelada")
        return
    
    # Eliminar archivos
    deleted_count = 0
    deleted_size = 0
    errors = 0
    
    print(f"\n🗑️  Eliminando archivos...")
    
    for file in pdf_files:
        file_path = os.path.join(otros_dir, file)
        try:
            file_size = os.path.getsize(file_path)
            os.remove(file_path)
            deleted_count += 1
            deleted_size += file_size
            
            if deleted_count % 50 == 0:  # Mostrar progreso cada 50 archivos
                print(f"Eliminados: {deleted_count}/{total_files}")
                
        except OSError as e:
            print(f"❌ Error eliminando {file}: {e}")
            errors += 1
    
    # Resultados
    print(f"\n✅ ELIMINACIÓN COMPLETADA")
    print("=" * 50)
    print(f"Archivos eliminados: {deleted_count}")
    print(f"Espacio liberado: {format_bytes(deleted_size)}")
    if errors > 0:
        print(f"Errores: {errors}")
    
    # Verificar si la carpeta está vacía
    remaining_files = [f for f in os.listdir(otros_dir) if f.endswith('.pdf')]
    if len(remaining_files) == 0:
        print(f"\n📁 La carpeta 'otros' está ahora vacía")
        # Opcional: eliminar la carpeta vacía
        try:
            os.rmdir(otros_dir)
            print(f"📁 Carpeta 'otros' eliminada (estaba vacía)")
        except OSError:
            print(f"📁 Carpeta 'otros' mantenida (no está vacía o no se puede eliminar)")


if __name__ == "__main__":
    delete_otros_pdfs()
