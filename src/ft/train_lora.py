#!/usr/bin/env python3
"""
Entrena modelo con LoRA/QLoRA para Domain Adaptation
Usa texto plano (continuation pretraining) en lugar de instruction tuning
"""

import argparse
import yaml
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel, PeftConfig
import os

#  Platzi
import os
import torch
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def load_config(config_path: Path):
    """Carga configuración desde YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model_and_tokenizer(base_model: str, max_seq_len: int = 2048):
    """Carga modelo y tokenizer"""
    print(f"📥 Cargando modelo: {base_model}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Configurar tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    # Cargar modelo
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="cuda:2",
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    
    # Configurar tamaño de secuencia
    if hasattr(model.config, 'max_position_embeddings'):
        if max_seq_len > model.config.max_position_embeddings:
            print(f"⚠️  max_seq_len ({max_seq_len}) > max_position_embeddings ({model.config.max_position_embeddings})")
            print(f"   Ajustando a {model.config.max_position_embeddings}")
            max_seq_len = model.config.max_position_embeddings
    
    return model, tokenizer, max_seq_len

def get_target_modules(model, phase: str = "attention"):
    """
    Obtiene módulos objetivo según el modelo y fase
    
    Args:
        model: Modelo cargado
        phase: "attention" (solo atención) o "attention_mlp" (atención + MLP)
    
    Returns:
        Lista de módulos objetivo
    """
    model_type = model.config.model_type.lower() if hasattr(model.config, 'model_type') else ""
    
    # Módulos de atención (comunes en Llama, Mistral y Qwen)
    # Qwen usa la misma arquitectura que Llama/Mistral
    attention_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # Módulos MLP (comunes en Llama, Mistral y Qwen)
    mlp_modules = ["gate_proj", "up_proj", "down_proj"]
    
    print(f"   Tipo de modelo detectado: {model_type}")
    
    if phase == "attention":
        print(f"   ✅ Fase 1: Solo Atención ({len(attention_modules)} módulos)")
        return attention_modules
    elif phase == "attention_mlp":
        print(f"   ✅ Fase 2: Atención + MLP ({len(attention_modules) + len(mlp_modules)} módulos)")
        return attention_modules + mlp_modules
    else:
        raise ValueError(f"Fase desconocida: {phase}. Usa 'attention' o 'attention_mlp'")

def validate_lora_r(model, lora_r: int, target_modules: list):
    """
    Valida que r no sea mayor que la dimensión original de las matrices
    
    Args:
        model: Modelo cargado
        lora_r: Rank propuesto
        target_modules: Lista de módulos objetivo
    
    Returns:
        r validado (ajustado si es necesario)
    """
    min_dim = float('inf')
    
    # Buscar la dimensión más pequeña en los módulos objetivo
    for name, module in model.named_modules():
        module_name = name.split('.')[-1]
        if module_name in target_modules and hasattr(module, 'weight'):
            weight = module.weight
            if weight is not None:
                # Obtener dimensión más pequeña de la matriz
                dims = weight.shape
                min_dim = min(min_dim, min(dims))
    
    if min_dim == float('inf'):
        print(f"⚠️  No se pudo determinar dimensión mínima, usando r={lora_r}")
        return lora_r
    
    if lora_r > min_dim:
        print(f"⚠️  r={lora_r} es mayor que dimensión mínima ({min_dim})")
        print(f"   Ajustando r a {min_dim}")
        return min_dim
    
    print(f"✅ r={lora_r} es válido (dimensión mínima: {min_dim})")
    return lora_r

def setup_lora(
    model, 
    lora_r: int = 8, 
    lora_alpha: int = 16, 
    lora_dropout: float = 0.05,
    phase: str = "attention"
):
    """
    Configura LoRA con validación y fases
    
    Args:
        model: Modelo cargado
        lora_r: Rank de LoRA (empezar pequeño, ej: 8)
        lora_alpha: Escala de LoRA (recomendado: 2 * r)
        lora_dropout: Dropout para regularización
        phase: "attention" (solo atención) o "attention_mlp" (atención + MLP)
    
    Returns:
        Modelo con LoRA aplicado
    """
    print(f"🔧 Configurando LoRA...")
    print(f"   Fase: {phase}")
    print(f"   r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    
    # Obtener módulos objetivo según fase
    target_modules = get_target_modules(model, phase)
    print(f"   Módulos objetivo: {target_modules}")
    
    # Validar que r no sea mayor que dimensión original
    lora_r = validate_lora_r(model, lora_r, target_modules)
    
    # Configurar LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,  # Causal Language Modeling (predicción siguiente token)
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    train_p, tot_p = model.get_nb_trainable_parameters()
    print(f'Parametros entrenables:      {train_p/1e6:.2f}M')
    print(f'Parametros totales:          {tot_p/1e6:.2f}M')
    print(f'% de parametros entrenables: {100*train_p/tot_p:.2f}%')
    return model


def tokenize_function(examples, tokenizer, max_length: int):
    """Tokeniza ejemplos para language modeling"""
    # Para domain adaptation, usamos el texto completo
    texts = examples["text"]
    
    # Tokenizar
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None
    )
    
    # Para causal LM, los labels son iguales a input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

def prepare_datasets(dataset_dir: Path, tokenizer, max_seq_len: int, train_split: float = 0.9):
    """Prepara datasets de entrenamiento y validación"""
    print(f"📂 Cargando datasets desde {dataset_dir}")
    
    # /home/mespanac/ProyectoTesis/Tesis-Marcos-Espana/data/cybersecurity_domain
    # Cargar datasets
    train_file = dataset_dir / "train.jsonl"
    val_file = dataset_dir / "validation.jsonl"
    
    if not train_file.exists():
        raise FileNotFoundError(f"No se encontró {train_file}")
    
    train_dataset = load_dataset('json', data_files=str(train_file), split='train')
    
    if val_file.exists():
        val_dataset = load_dataset('json', data_files=str(val_file), split='train')
    else:
        print("⚠️  No se encontró validation.jsonl, dividiendo train")
        # Dividir train en train/val
        split = train_dataset.train_test_split(test_size=1-train_split)
        train_dataset = split['train']
        val_dataset = split['test']
    
    print(f"   Train: {len(train_dataset):,} ejemplos")
    print(f"   Validation: {len(val_dataset):,} ejemplos")
    
    # Tokenizar
    print(f"🔄 Tokenizando datasets...")
    
    def tokenize(examples):
        return tokenize_function(examples, tokenizer, max_seq_len)
    
    train_dataset = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizando train"
    )
    
    val_dataset = val_dataset.map(
        tokenize,
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizando validation"
    )
    
    return train_dataset, val_dataset

def train(
    config_path: Path,
    output_dir: Path,
    dataset_dir: Path,
    resume_from_checkpoint: Optional[str] = None
):
    """Entrena el modelo"""
    
    # Cargar configuración
    config = load_config(config_path)
    print(f"📋 Configuración cargada desde {config_path}")
    print(f"   Modelo base: {config['base_model']}")
    print(f"   Método: {config['method']}")
    print(f"   Learning rate: {config['learning_rate']}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Epochs: {config['num_epochs']}")
    print(f"   Max seq len: {config['max_seq_len']}")
    
    # Cargar modelo y tokenizer
    model, tokenizer, max_seq_len = load_model_and_tokenizer(
        config['base_model'],
        config['max_seq_len']
    )
    
    # Configurar LoRA
    if config['method'] == 'lora':
        lora_phase = config.get('lora_phase', 'attention')
        model = setup_lora(
            model,
            lora_r=config.get('lora_r', 8),
            lora_alpha=config.get('lora_alpha', 16),
            lora_dropout=config.get('lora_dropout', 0.05),
            phase=lora_phase
        )
    elif config['method'] == 'qlora':
        # QLoRA requiere bitsandbytes
        raise NotImplementedError("QLoRA no implementado aún. Usa method: lora")
    
    # Preparar datasets
    train_dataset, val_dataset = prepare_datasets(
        dataset_dir,
        tokenizer,
        max_seq_len,
        train_split=0.9
    )
    
    # Configurar data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, no masked LM
    )
    
    # Configurar entrenamiento
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config['num_epochs'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        auto_find_batch_size=True, # Si el batch size que usas puede ocasionar un OOM (Out of Memory) lo dividimos en 2 hasta que funcione.
        learning_rate=config['learning_rate'],
        warmup_steps=config.get('warmup_steps', 100),
        logging_steps=50,
        eval_steps=500,
        save_steps=500,
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        bf16=False,
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
        report_to="none",  # Cambiar a "wandb" o "tensorboard" si quieres
        run_name="cybersecurity_domain_adaptation",
        remove_unused_columns=False,
        # IMPORTANTE: No usar inference_mode durante entrenamiento
        # inference_mode=False es el default, pero lo explicitamos
        dataloader_pin_memory=False,  # Evitar problemas de memoria
    )
    
    # Crear trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Entrenar
    print(f"\n🚀 Iniciando entrenamiento...")
    print(f"   Output dir: {output_dir}")
    
    if resume_from_checkpoint:
        print(f"   Resumiendo desde: {resume_from_checkpoint}")
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Guardar modelo final
    print(f"\n💾 Guardando modelo final...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Entrenamiento completado!")
    print(f"   Modelo guardado en: {output_dir}")
    
    return trainer

def main():
    parser = argparse.ArgumentParser(
        description="Entrena modelo con LoRA para Domain Adaptation"
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/ft.yaml'),
        help='Archivo de configuración YAML'
    )
    
    parser.add_argument(
        '--dataset',
        type=Path,
        default=Path('data/ft_datasets/cybersecurity_domain'),
        help='Directorio con train.jsonl y validation.jsonl'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('models/cybersecurity-adapted'),
        help='Directorio de salida para el modelo'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        help='Checkpoint para continuar entrenamiento'
    )
    
    args = parser.parse_args()
    
    # Verificar que existe configuración
    if not args.config.exists():
        print(f"❌ No se encontró configuración: {args.config}")
        return 1
    
    # Verificar que existe dataset
    if not args.dataset.exists():
        print(f"❌ No se encontró dataset: {args.dataset}")
        print(f"   Ejecuta primero: python src/ft/prepare_dataset.py")
        return 1
    
    # Entrenar
    try:
        trainer = train(
            config_path=args.config,
            output_dir=args.output,
            dataset_dir=args.dataset,
            resume_from_checkpoint=args.resume
        )
    except Exception as e:
        print(f"❌ Error durante entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    from typing import Optional
    sys.exit(main())

