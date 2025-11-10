#!/usr/bin/env python3
"""
Entrena modelo con LoRA/QLoRA para Domain Adaptation
Usa texto plano (continuation pretraining) en lugar de instruction tuning
"""

import argparse
import yaml
from pathlib import Path
from typing import Optional
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from datasets import load_dataset
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    PeftModel, 
    PeftConfig,
    prepare_model_for_kbit_training
)
from trl import SFTConfig, SFTTrainer


def load_config(config_path: Path):
    """Carga configuración desde YAML y normaliza tipos"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Normalizar tipos numéricos (YAML puede cargar algunos como string)
    # Especialmente valores en notación científica como "2e-4"
    numeric_keys = ['learning_rate', 'batch_size', 'num_epochs', 'max_seq_len', 
                   'lora_r', 'lora_alpha', 'lora_dropout', 'gradient_accumulation_steps',
                   'warmup_steps', 'min_words', 'max_words', 'train_split']
    
    for key in numeric_keys:
        if key in config:
            value = config[key]
            # Convertir string a float/int si es necesario
            if isinstance(value, str):
                try:
                    # Para valores en notación científica como "2e-4"
                    config[key] = float(value)
                except (ValueError, TypeError):
                    try:
                        # Intentar evaluar como expresión Python (último recurso)
                        config[key] = eval(value)
                    except:
                        pass  # Mantener valor original si no se puede convertir
            elif isinstance(value, (int, float)):
                # Asegurar tipos correctos
                if key in ['learning_rate', 'lora_dropout', 'train_split']:
                    config[key] = float(value)
                elif key in ['batch_size', 'num_epochs', 'max_seq_len', 'lora_r', 
                            'lora_alpha', 'gradient_accumulation_steps', 'warmup_steps',
                            'min_words', 'max_words']:
                    # Convertir a int si es apropiado
                    if isinstance(value, float) and value.is_integer():
                        config[key] = int(value)
    
    return config

def load_model_and_tokenizer(base_model: str, max_seq_len: int = 2048):
    """Carga modelo y tokenizer"""
    print(f"📥 Cargando modelo: {base_model}")
    
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA no disponible. Se requiere GPU para entrenamiento en 4/8 bits.")
    
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    print(f"   Dispositivo destino: cuda:{current_device} ({device_name})")

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
        device_map={'': current_device},
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
    Para modelos cuantizados, usa la configuración del modelo en lugar de inspeccionar pesos
    
    Args:
        model: Modelo cargado
        lora_r: Rank propuesto
        target_modules: Lista de módulos objetivo
    
    Returns:
        r validado (ajustado si es necesario)
    """
    # Para modelos cuantizados, obtener dimensiones desde la configuración
    config = model.config if hasattr(model, 'config') else None
    base_model = model.base_model if hasattr(model, 'base_model') else model
    base_config = base_model.config if hasattr(base_model, 'config') else config
    
    if base_config is None:
        print(f"⚠️  No se pudo acceder a la configuración, usando r={lora_r} sin validación")
        return lora_r
    
    # Obtener dimensiones típicas desde la configuración
    # Para Qwen/Llama/Mistral, las dimensiones de atención son:
    # - hidden_size: dimensión del modelo
    # - num_attention_heads: número de heads
    # - num_key_value_heads: número de KV heads (para GQA)
    
    hidden_size = getattr(base_config, 'hidden_size', None)
    num_heads = getattr(base_config, 'num_attention_heads', None)
    num_kv_heads = getattr(base_config, 'num_key_value_heads', None)
    
    # Calcular dimensiones mínimas esperadas
    min_dim = float('inf')
    
    if hidden_size:
        # Para q_proj, k_proj, v_proj, o_proj: típicamente hidden_size x hidden_size
        # Pero con GQA, k_proj y v_proj pueden ser más pequeños
        if num_kv_heads and num_heads:
            # GQA: k_proj y v_proj son más pequeños
            kv_dim = hidden_size * num_kv_heads // num_heads
            min_dim = min(min_dim, kv_dim, hidden_size)
        else:
            min_dim = min(min_dim, hidden_size)
    
    # Si no pudimos determinar desde la config, intentar inspeccionar pesos
    if min_dim == float('inf'):
        # Buscar en el modelo base (sin cuantización)
        try:
            for name, module in base_model.named_modules():
                module_name = name.split('.')[-1]
                if module_name in target_modules:
                    # Intentar obtener peso
                    weight = None
                    if hasattr(module, 'weight') and module.weight is not None:
                        weight = module.weight
                    elif hasattr(module, 'base_layer') and hasattr(module.base_layer, 'weight'):
                        weight = module.base_layer.weight
                    
                    if weight is not None and hasattr(weight, 'shape'):
                        dims = [d for d in weight.shape if d > 1]
                        if dims:
                            min_dim = min(min_dim, min(dims))
                            break  # Solo necesitamos uno
        except Exception as e:
            # Si falla, usar r sin validación
            print(f"⚠️  No se pudo inspeccionar pesos (normal en modelos cuantizados), usando r={lora_r}")
            return lora_r
    
    # Validar r
    if min_dim == float('inf'):
        print(f"✅ Usando r={lora_r} (no se pudo validar dimensión mínima)")
        return lora_r
    
    if lora_r > min_dim:
        print(f"⚠️  r={lora_r} es mayor que dimensión mínima estimada ({int(min_dim)})")
        print(f"   Ajustando r a {int(min_dim)}")
        return int(min_dim)
    
    print(f"✅ r={lora_r} es válido (dimensión mínima estimada: {int(min_dim)})")
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
    
    return tokenized


def build_causal_lm_collator(tokenizer):
    """Crea un data collator que maneja padding y labels para causal LM"""
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    def collate(features):
        # Remover labels existentes si vienen en los datos
        clean_features = []
        for feature in features:
            clean_feature = {k: v for k, v in feature.items() if k != "labels"}
            clean_features.append(clean_feature)
        
        batch = tokenizer.pad(
            clean_features,
            padding=True,
            return_tensors="pt"
        )
        
        labels = batch["input_ids"].clone()
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        batch["labels"] = labels
        return batch
    
    return collate

def prepare_datasets(dataset_dir: Path, tokenizer, max_seq_len: int, train_split: float = 0.9):
    """Prepara datasets de entrenamiento y validación"""
    print(f"📂 Cargando datasets desde {dataset_dir}")
    
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
    
    requested_device = config.get('device', os.environ.get("FT_GPU"))
    current_device = None
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"🖥️  GPU activa (índice local): cuda:{current_device} ({device_name})")
        if requested_device is not None:
            print(f"   GPU solicitada (config/env): {requested_device} — usa CUDA_VISIBLE_DEVICES para ajustarla.")
    else:
        print("🖥️  GPU no disponible. Entrenar en CPU no es soportado para 4-bit.")

    # Cargar modelo y tokenizer
    model, tokenizer, max_seq_len = load_model_and_tokenizer(
        config['base_model'],
        config['max_seq_len']
    )
    
    # Preparar modelo para entrenamiento en baja precisión (requerido para QLoRA/LoRA 4-bit)
    model = prepare_model_for_kbit_training(model)
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    
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
        train_split=float(config.get('train_split', 0.9))
    )
    
    # Configurar data collator (padding dinámico + labels)
    data_collator = build_causal_lm_collator(tokenizer)
    
    # Calcular estadísticas del entrenamiento
    num_train_samples = len(train_dataset)
    num_eval_samples = len(val_dataset)
    
    # Asegurar que los valores sean del tipo correcto
    learning_rate = float(config['learning_rate'])
    batch_size = int(config['batch_size'])
    num_epochs = int(config['num_epochs'])
    gradient_accumulation_steps = int(config.get('gradient_accumulation_steps', 1))
    
    effective_batch_size = batch_size * gradient_accumulation_steps
    steps_per_epoch = num_train_samples // effective_batch_size
    total_steps = steps_per_epoch * num_epochs
    
    print(f"\n📊 Estadísticas de Entrenamiento:")
    print(f"   Ejemplos de entrenamiento: {num_train_samples:,}")
    print(f"   Ejemplos de validación: {num_eval_samples:,}")
    print(f"   Batch size: {batch_size}")
    print(f"   Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"   Effective batch size: {effective_batch_size}")
    print(f"   Steps por época: ~{steps_per_epoch:,}")
    print(f"   Total de steps: ~{total_steps:,}")
    print(f"   Épocas: {num_epochs}")
    device_desc = "CPU"
    if current_device is not None:
        device_desc = f"cuda:{current_device}"
    print(f"   Learning rate: {learning_rate}")
    print(f"   Dispositivo (índice local): {device_desc}")
    
    # Configurar entrenamiento
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        auto_find_batch_size=True, # Si el batch size que usas puede ocasionar un OOM (Out of Memory) lo dividimos en 2 hasta que funcione.
        learning_rate=learning_rate,
        warmup_steps=int(config.get('warmup_steps', 100)),
        logging_steps=50,  # Log cada 50 steps (verás progreso frecuente)
        eval_steps=500,    # Evaluar cada 500 steps
        save_steps=500,    # Guardar checkpoint cada 500 steps
        eval_strategy="steps",  # Estrategia de evaluación
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        bf16=False,
        gradient_accumulation_steps=gradient_accumulation_steps,
        report_to="tensorboard",  # Cambiar a "wandb" o "tensorboard" si quieres
        run_name="cybersecurity_domain_adaptation",
        remove_unused_columns=False,
        dataloader_pin_memory=False,  # Evitar problemas de memoria
        logging_first_step=True,  # Mostrar primer step
        logging_dir=str(output_dir / "logs"),  # Guardar logs en directorio
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
        default=Path('data/cybersecurity_domain'),
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
    sys.exit(main())

