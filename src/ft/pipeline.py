#!/usr/bin/env python3
"""
Script maestro para pipeline completo de Fine-Tuning
Domain Adaptation / Continuation Pretraining
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Ejecuta comando y maneja errores"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error en: {description}")
        sys.exit(1)
    
    print(f"✅ {description} completado\n")

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline completo de Fine-Tuning (Domain Adaptation)"
    )
    
    parser.add_argument(
        '--step',
        choices=['prepare', 'train', 'eval', 'all'],
        default='all',
        help='Paso a ejecutar'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/ft.yaml'),
        help='Archivo de configuración'
    )
    
    parser.add_argument(
        '--dataset-output',
        type=Path,
        default=Path('data/ft_datasets/cybersecurity_domain'),
        help='Directorio de salida para datasets'
    )
    
    parser.add_argument(
        '--model-output',
        type=Path,
        default=Path('models/cybersecurity-adapted'),
        help='Directorio de salida para modelo'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        help='Checkpoint para continuar entrenamiento'
    )
    
    args = parser.parse_args()
    
    # Ubicación de scripts
    scripts_dir = Path(__file__).parent
    prepare_script = scripts_dir / "prepare_dataset.py"
    train_script = scripts_dir / "train_lora.py"
    eval_script = scripts_dir / "eval_ft.py"
    
    print(f"🚀 Pipeline de Fine-Tuning - Domain Adaptation")
    print(f"   Config: {args.config}")
    print(f"   Dataset output: {args.dataset_output}")
    print(f"   Model output: {args.model_output}")
    
    # Paso 1: Preparar dataset
    if args.step in ['prepare', 'all']:
        cmd = f"python {prepare_script} --output {args.dataset_output}"
        run_command(cmd, "Preparando dataset desde chunks")
    
    # Paso 2: Entrenar modelo
    if args.step in ['train', 'all']:
        cmd = f"python {train_script} --config {args.config} --dataset {args.dataset_output} --output {args.model_output}"
        if args.resume:
            cmd += f" --resume {args.resume}"
        run_command(cmd, "Entrenando modelo con LoRA")
    
    # Paso 3: Evaluar modelo
    if args.step in ['eval', 'all']:
        cmd = f"python {eval_script} --model {args.model_output}"
        if args.dataset_output.exists():
            val_file = args.dataset_output / "validation.jsonl"
            if val_file.exists():
                cmd += f" --dataset {val_file}"
        run_command(cmd, "Evaluando modelo")
    
    print(f"\n{'='*60}")
    print(f"✅ Pipeline completado exitosamente!")
    print(f"{'='*60}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

