#!/usr/bin/env python3
"""
Script para consolidar todos los archivos Excel parciales de diferentes batches
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import glob

def merge_all_batches(
    output_dir: str = "results/excel",
    output_file: str = None,
    input_files: list[str] | None = None,
    only_model: str | None = None,
):
    """
    Consolida archivos Excel de resultados en un solo archivo.

    Args:
        output_dir: Directorio base donde están los batches (se usa si no se pasan input_files)
        output_file: Nombre del archivo final (auto-generado si es None)
        input_files: Lista explícita de archivos Excel a unir (omite el escaneo de directorios)
        only_model: Si se indica, filtra filas por el valor exacto en la columna 'RAG_Model'
    """
    all_files: list[Path] = []

    if input_files:
        # Usar solo los archivos proporcionados
        for f in input_files:
            p = Path(f)
            if p.exists() and p.suffix.lower() in {".xlsx"}:
                all_files.append(p)
            else:
                print(f"⚠️  Ignorando (no existe o no es .xlsx): {f}")
        if not all_files:
            print("❌ No hay archivos válidos en --input-files")
            return
        print(f"📂 Usando {len(all_files)} archivo(s) especificado(s) por --input-files")
    else:
        # Escanear el árbol de batches
        base_dir = Path(output_dir)
        if not base_dir.exists():
            print(f"❌ No existe el directorio {output_dir}")
            return

        batch_dirs = sorted(base_dir.glob("batch_*"))
        print(f"📂 Buscando archivos en {len(batch_dirs)} directorios de batch...")

        for batch_dir in batch_dirs:
            # Archivos parciales
            partial_files = sorted(batch_dir.glob("*_partial_*.xlsx"))
            if partial_files:
                for pf in partial_files:
                    all_files.append(pf)
                    print(f"  ✅ {pf.name}")

            # Archivos finales
            final_files = sorted([f for f in batch_dir.glob("*.xlsx") if "_partial_" not in f.name])
            if final_files:
                for ff in final_files:
                    all_files.append(ff)
                    print(f"  ✅ {ff.name} (final)")

        if not all_files:
            print("❌ No se encontraron archivos Excel para consolidar")
            return

    print(f"\n📊 Total de archivos a leer: {len(all_files)}")
    print("🔄 Leyendo y consolidando...")

    # Leer y consolidar
    all_data = []
    for i, file_path in enumerate(all_files, 1):
        try:
            df = pd.read_excel(file_path)
            if only_model and 'RAG_Model' in df.columns:
                df = df[df['RAG_Model'] == only_model].copy()
            all_data.append(df)
            print(f"  ✅ {i}/{len(all_files)}: {file_path.name} ({len(df)} filas tras filtro)")
        except Exception as e:
            print(f"  ⚠️  Error leyendo {file_path.name}: {e}")

    if not all_data:
        print("❌ No se pudieron leer archivos")
        return

    print("\n🔗 Consolidando DataFrames...")
    consolidated_df = pd.concat(all_data, ignore_index=True)

    # Eliminar duplicados
    initial_len = len(consolidated_df)
    consolidated_df = consolidated_df.drop_duplicates(ignore_index=True)
    final_len = len(consolidated_df)
    if initial_len != final_len:
        print(f"⚠️  Eliminados {initial_len - final_len} duplicados")

    # Guardar
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"cybersecurity_qa_COMPLETE_{timestamp}.xlsx"

    output_path = Path(output_file)
    consolidated_df.to_excel(output_path, index=False, engine='openpyxl')

    print(f"\n{'='*80}")
    print(f"✅ ARCHIVO CONSOLIDADO CREADO")
    print(f"{'='*80}")
    print(f"📁 Archivo: {output_path.absolute()}")
    print(f"📊 Total de preguntas: {len(consolidated_df)}")
    print(f"📋 Columnas: {list(consolidated_df.columns)}")
    print(f"{'='*80}\n")

    # Estadísticas
    if 'RAG_Model' in consolidated_df.columns:
        print("📈 Estadísticas:")
        print(f"  • RAG Model: {consolidated_df['RAG_Model'].iloc[0] if not consolidated_df['RAG_Model'].empty else 'N/A'}")

    if 'RAG_Response_Time_Seconds' in consolidated_df.columns:
        avg_time = consolidated_df['RAG_Response_Time_Seconds'].mean()
        total_time = consolidated_df['RAG_Response_Time_Seconds'].sum()
        print(f"  • Tiempo promedio por pregunta: {avg_time:.2f}s")
        print(f"  • Tiempo total acumulado: {total_time/3600:.2f} horas")

    if 'RAG_Error' in consolidated_df.columns:
        errors = consolidated_df['RAG_Error'].notna().sum()
        if errors > 0:
            print(f"  ⚠️  Preguntas con error: {errors}")

    return output_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Consolidar archivos Excel de batches en uno solo (soporta selección de archivos y filtro por modelo)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/excel",
        help="Directorio donde están los batches"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Nombre del archivo final (auto-generado si no se especifica)"
    )
    parser.add_argument(
        "--input-files",
        nargs='*',
        default=None,
        help="Lista de archivos .xlsx específicos a unir (omite el escaneo de batches)"
    )
    parser.add_argument(
        "--only-model",
        type=str,
        default=None,
        help="Filtrar filas por 'RAG_Model' (p.ej. 'Qwen2.5-7B-Instruct')"
    )
    
    args = parser.parse_args()
    
    merge_all_batches(
        output_dir=args.output_dir,
        output_file=args.output_file,
        input_files=args.input_files,
        only_model=args.only_model,
    )


if __name__ == "__main__":
    main()

