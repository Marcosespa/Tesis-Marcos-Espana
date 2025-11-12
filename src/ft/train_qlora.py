#!/usr/bin/env python3
"""
Entrenamiento con QLoRA para Domain Adaptation
Usa texto plano (continuation pretraining) sobre un dominio específico

Configuración de GPUs:
- Por defecto usa GPUs 0 y 1
- ⚠️  IMPORTANTE: Para evitar usar otras GPUs (ej: GPU 2), usa CUDA_VISIBLE_QLORA
- Forma SEGURA de ejecutar (recomendado):
  CUDA_VISIBLE_QLORA=0,1 python src/ft/train_qlora.py --config ... --dataset ... --output ...
- El modelo se guarda en: models/cybersecurity-qlora (separado de train_lora.py)
"""

import argparse
import json
import os
import yaml
from pathlib import Path
from typing import Optional

# ⚠️ IMPORTANTE: Configurar CUDA_VISIBLE_DEVICES ANTES de importar torch
# Leer CUDA_VISIBLE_QLORA y configurarla como CUDA_VISIBLE_DEVICES
cuda_visible_qlora = os.environ.get("CUDA_VISIBLE_QLORA")
if cuda_visible_qlora:
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_qlora
    print(f"🔧 CUDA_VISIBLE_QLORA={cuda_visible_qlora} → CUDA_VISIBLE_DEVICES={cuda_visible_qlora}")

# ⚠️ Deshabilitar verificación de torch.load para compatibilidad con checkpoints antiguos
# Esto es necesario para PyTorch 2.5.1 con transformers 4.57.1
os.environ["TORCH_ALLOW_UNSAFE_LOAD"] = "1"

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from datasets import load_dataset
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    prepare_model_for_kbit_training
)
from trl import SFTConfig, SFTTrainer


# Función auxiliar para parchear la verificación de torch.load
# Esto se usará después de importar transformers
def _patch_torch_load_check():
    """Parchea check_torch_load_is_safe para permitir cargar checkpoints antiguos"""
    try:
        import transformers.utils.import_utils as import_utils_module
        # Reemplazar la función con una que no hace nada
        def _noop_check():
            pass
        # Parchear en el módulo directamente - IMPORTANTE: hacer esto de forma permanente
        import_utils_module.check_torch_load_is_safe = _noop_check
        
        # También parchear en el módulo transformers.utils si existe
        import transformers.utils as utils_module
        if hasattr(utils_module, 'import_utils'):
            utils_module.import_utils.check_torch_load_is_safe = _noop_check
        
        # Parchear en sys.modules para cubrir todas las referencias
        import sys
        for mod_name in list(sys.modules.keys()):
            if 'transformers.utils.import_utils' in mod_name:
                mod = sys.modules[mod_name]
                if hasattr(mod, 'check_torch_load_is_safe'):
                    mod.check_torch_load_is_safe = _noop_check
        
        # También parchear directamente en el código fuente si es necesario
        # Reemplazar la función en el nivel del módulo
        import types
        import_utils_module.check_torch_load_is_safe = types.FunctionType(
            _noop_check.__code__,
            import_utils_module.__dict__,
            '_noop_check',
            _noop_check.__defaults__,
            _noop_check.__closure__
        )
        
        return True
    except Exception as e:
        print(f"⚠️  Error al aplicar parche: {e}")
        return False


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

def load_model_and_tokenizer(base_model: str, max_seq_len: int = 2048, gpu_ids: list = [0, 1]):
    """Carga modelo y tokenizer con cuantización 4-bit para QLoRA"""
    print(f"📥 Cargando modelo: {base_model}")
    
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA no disponible. Se requiere GPU para QLoRA (cuantización 4-bit).")
    
    # Verificar que las GPUs solicitadas estén disponibles
    num_gpus = torch.cuda.device_count()
    available_gpus = list(range(num_gpus))
    
    print(f"   GPUs disponibles: {available_gpus}")
    print(f"   GPUs solicitadas: {gpu_ids}")
    
    # Filtrar solo las GPUs disponibles
    valid_gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id in available_gpus]
    
    if not valid_gpu_ids:
        raise EnvironmentError(f"Ninguna de las GPUs solicitadas {gpu_ids} está disponible. GPUs disponibles: {available_gpus}")
    
    if len(valid_gpu_ids) < len(gpu_ids):
        print(f"⚠️  Algunas GPUs no están disponibles. Usando: {valid_gpu_ids}")
    
    # Mostrar información de las GPUs que se usarán
    for gpu_id in valid_gpu_ids:
        device_name = torch.cuda.get_device_name(gpu_id)
        print(f"   GPU {gpu_id}: {device_name}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Configurar tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Configurar cuantización en 4 bits para QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Configurar device_map para usar SOLO las GPUs especificadas
    # IMPORTANTE: Para evitar usar otras GPUs (ej: GPU 2), usamos device_map específico
    if len(valid_gpu_ids) == 1:
        # Una sola GPU: usar directamente
        device_map = {"": valid_gpu_ids[0]}
        print(f"   ✅ Usando GPU {valid_gpu_ids[0]} únicamente")
        print(f"   ✅ NO se usará ninguna otra GPU")
    else:
        # Múltiples GPUs: usar "balanced" o crear device_map personalizado
        # Usamos "balanced" que distribuye mejor, pero necesitamos restringir
        # La mejor forma es usar CUDA_VISIBLE_DEVICES antes de ejecutar
        # Por ahora, usamos "auto" pero con advertencia
        device_map = "auto"
        # Verificar si CUDA_VISIBLE_QLORA está configurado
        cuda_visible_qlora = os.environ.get("CUDA_VISIBLE_QLORA")
        if cuda_visible_qlora:
            print(f"   ✅ CUDA_VISIBLE_QLORA={cuda_visible_qlora} está configurado")
            print(f"   ✅ Solo se usarán las GPUs especificadas en CUDA_VISIBLE_QLORA")
        else:
            print(f"   ⚠️  ADVERTENCIA: device_map='auto' puede usar TODAS las GPUs disponibles")
            print(f"   📌 Para usar SOLO GPUs {valid_gpu_ids}, ejecuta con:")
            print(f"      CUDA_VISIBLE_QLORA={','.join(map(str, valid_gpu_ids))} python src/ft/train_qlora.py ...")
        print(f"   🔄 Distribuyendo modelo entre GPUs: {valid_gpu_ids}...")
        
        # Intentar crear un device_map más restrictivo usando accelerate si está disponible
        try:
            from accelerate import infer_auto_device_map
            # Esto requiere conocer el tamaño del modelo, así que por ahora usamos "auto"
            # pero documentamos la mejor práctica
            pass
        except ImportError:
            pass
    
    # Cargar modelo
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
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
    print(f"🔧 Configurando LoRA para QLoRA...")
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

def prepare_datasets(dataset_dir: Path, train_split: float = 0.9):
    """Prepara datasets de entrenamiento y validación (sin tokenizar, SFTTrainer lo hace)"""
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
    print(f"   ⚠️  Nota: SFTTrainer tokenizará automáticamente usando el campo 'text'")
    
    return train_dataset, val_dataset

def train_qlora(
    config_path: Path,
    output_dir: Path,
    dataset_dir: Path,
    resume_from_checkpoint: Optional[str] = None
):
    """Entrena el modelo usando QLoRA con SFTTrainer"""
    
    # Cargar configuración
    config = load_config(config_path)
    print(f"📋 Configuración cargada desde {config_path}")
    print(f"   Modelo base: {config['base_model']}")
    print(f"   Método: QLoRA")
    print(f"   Learning rate: {config['learning_rate']}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Epochs: {config['num_epochs']}")
    print(f"   Max seq len: {config['max_seq_len']}")
    
    # Obtener GPUs a usar desde configuración o usar [0, 1] por defecto
    gpu_ids = config.get('gpu_ids', [0, 1])
    if isinstance(gpu_ids, str):
        # Si viene como string "0,1", convertir a lista
        gpu_ids = [int(x.strip()) for x in gpu_ids.split(',')]
    elif isinstance(gpu_ids, int):
        # Si viene como un solo número, convertir a lista
        gpu_ids = [gpu_ids]
    
    if not torch.cuda.is_available():
        print("🖥️  GPU no disponible. Entrenar en CPU no es soportado para QLoRA.")
        raise EnvironmentError("CUDA no disponible")
    
    # ⚠️ IMPORTANTE: Si CUDA_VISIBLE_QLORA está configurado, ignorar gpu_ids del config
    # y usar todas las GPUs visibles (que ya están restringidas por CUDA_VISIBLE_DEVICES)
    cuda_visible_qlora = os.environ.get("CUDA_VISIBLE_QLORA")
    if cuda_visible_qlora:
        # Cuando CUDA_VISIBLE_QLORA está configurado, usar todas las GPUs visibles
        # (que PyTorch ve como 0, 1, 2, ... pero físicamente son las especificadas)
        num_visible_gpus = torch.cuda.device_count()
        gpu_ids = list(range(num_visible_gpus))
        print(f"🖥️  Configuración de GPUs:")
        print(f"   ✅ CUDA_VISIBLE_QLORA={cuda_visible_qlora} detectado")
        print(f"   ✅ Ignorando gpu_ids del config, usando GPUs visibles: {gpu_ids}")
        print(f"   📌 Físicamente se usará la(s) GPU(s): {cuda_visible_qlora}")
    else:
        print(f"🖥️  Configuración de GPUs:")
        print(f"   GPUs a usar (desde config): {gpu_ids}")

    # Cargar modelo y tokenizer
    model, tokenizer, max_seq_len = load_model_and_tokenizer(
        config['base_model'],
        config['max_seq_len'],
        gpu_ids=gpu_ids
    )
    
    # Preparar modelo para entrenamiento en baja precisión (requerido para QLoRA)
    model = prepare_model_for_kbit_training(model)
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        # Gradient checkpointing más eficiente (evita warnings y mejora compatibilidad)
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            # Fallback para versiones que no soportan el parámetro
            model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    
    # Configurar LoRA
    lora_phase = config.get('lora_phase', 'attention')
    model = setup_lora(
        model,
        lora_r=config.get('lora_r', 8),
        lora_alpha=config.get('lora_alpha', 16),
        lora_dropout=config.get('lora_dropout', 0.05),
        phase=lora_phase
    )

    # Preparar datasets (sin tokenizar, SFTTrainer lo hace automáticamente)
    train_dataset, val_dataset = prepare_datasets(
        dataset_dir,
        train_split=float(config.get('train_split', 0.9))
    )
    
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
    # Verificación de memoria GPU
    if torch.cuda.is_available():
        print(f"\n💾 Información de Memoria GPU:")
        for gpu_id in gpu_ids:
            if gpu_id < torch.cuda.device_count():
                props = torch.cuda.get_device_properties(gpu_id)
                total_mem = props.total_memory / 1e9
                print(f"   GPU {gpu_id}: {total_mem:.1f} GB total")
        torch.cuda.empty_cache()
        print(f"   ✅ Cache de GPU limpiado")
    
    # Mostrar información de GPUs
    num_gpus = torch.cuda.device_count()
    print(f"   GPUs disponibles: {num_gpus}")
    print(f"   Learning rate: {learning_rate}")
    
    # Calcular warmup_ratio si no está especificado
    warmup_steps = int(config.get('warmup_steps', 100))
    warmup_ratio = config.get('warmup_ratio', None)
    if warmup_ratio is None:
        # Calcular warmup_ratio basado en warmup_steps
        warmup_ratio = warmup_steps / total_steps if total_steps > 0 else 0.03
    
    # Configurar entrenamiento con SFTConfig (optimizado para QLoRA)
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        auto_find_batch_size=True,  # Si el batch size causa OOM, lo divide automáticamente
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,  # Más flexible que solo warmup_steps
        lr_scheduler_type=config.get('lr_scheduler_type', 'cosine'),  # Scheduler más sofisticado
        logging_steps=50,  # Log cada 50 steps
        eval_steps=500,    # Evaluar cada 500 steps
        save_steps=500,    # Guardar checkpoint cada 500 steps
        eval_strategy="steps",  # Estrategia de evaluación (correcto para SFTConfig)
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="paged_adamw_32bit",  # ✅ Optimizador recomendado para QLoRA
        bf16=True,  # Usar bfloat16 para QLoRA
        fp16=False,
        gradient_accumulation_steps=gradient_accumulation_steps,
        report_to="tensorboard",
        run_name="cybersecurity_domain_adaptation_qlora",
        max_length=max_seq_len,  # ✅ max_length (no max_seq_length) para SFTConfig
        dataset_text_field="text",  # Campo del dataset que contiene el texto
        packing=False,  # No empaquetar secuencias (continuation pretraining)
        remove_unused_columns=False,
        dataloader_pin_memory=False,  # Evitar problemas de memoria
        logging_first_step=True,
        logging_dir=str(output_dir / "logs"),
        # Logging más informativo
        logging_nan_inf_filter=True,  # Detecta NaNs temprano
        log_level="info",
        disable_tqdm=False,  # Mostrar barra de progreso
    )
    
    # Parchear la verificación de torch.load ANTES de crear el trainer
    # Esto es crítico para que funcione cuando se reanude el entrenamiento
    if resume_from_checkpoint:
        print(f"⚠️  Reanudando desde checkpoint - aplicando parche de compatibilidad...")
        # Aplicar parche de forma permanente y agresiva
        patch_success = _patch_torch_load_check()
        if patch_success:
            print("✅ Parche aplicado a check_torch_load_is_safe")
        
        # Reemplazar COMPLETAMENTE los métodos del Trainer para evitar check_torch_load_is_safe
        import transformers.trainer as trainer_module
        
        # Reemplazar _load_optimizer_and_scheduler completamente
        def patched_load_optimizer_and_scheduler(self, checkpoint):
            """Versión que carga checkpoints SIN verificación de torch.load"""
            checkpoint_dir = checkpoint if os.path.isdir(checkpoint) else os.path.dirname(checkpoint)
            optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
            scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
            
            # Cargar optimizer sin verificación
            if os.path.isfile(optimizer_path):
                try:
                    # Usar weights_only=False para evitar la verificación
                    optimizer_state = torch.load(optimizer_path, map_location="cpu", weights_only=False)
                    self.optimizer.load_state_dict(optimizer_state)
                except Exception as e:
                    print(f"⚠️  No se pudo cargar optimizer: {e}")
            
            # Cargar scheduler sin verificación
            if os.path.isfile(scheduler_path):
                try:
                    # Usar weights_only=False para evitar la verificación
                    scheduler_state = torch.load(scheduler_path, map_location="cpu", weights_only=False)
                    self.lr_scheduler.load_state_dict(scheduler_state)
                except Exception as e:
                    print(f"⚠️  No se pudo cargar scheduler: {e}")
        
        trainer_module.Trainer._load_optimizer_and_scheduler = patched_load_optimizer_and_scheduler
        
        # Reemplazar _load_scaler completamente
        def patched_load_scaler(self, checkpoint):
            """Versión que carga scaler SIN verificación de torch.load"""
            if not hasattr(self, "scaler") or self.scaler is None:
                return
            
            checkpoint_dir = checkpoint if os.path.isdir(checkpoint) else os.path.dirname(checkpoint)
            scaler_path = os.path.join(checkpoint_dir, "scaler.pt")
            
            if os.path.isfile(scaler_path):
                try:
                    # Usar weights_only=False para evitar la verificación
                    scaler_state = torch.load(scaler_path, map_location="cpu", weights_only=False)
                    self.scaler.load_state_dict(scaler_state)
                except Exception as e:
                    print(f"⚠️  No se pudo cargar scaler: {e}")
        
        trainer_module.Trainer._load_scaler = patched_load_scaler
        
        print("✅ Métodos del Trainer reemplazados para cargar checkpoints sin verificación")
    
    # Crear SFTTrainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Entrenar
    print(f"\n🚀 Iniciando entrenamiento con QLoRA...")
    print(f"   Output dir: {output_dir}")
    
    if resume_from_checkpoint:
        print(f"   Resumiendo desde: {resume_from_checkpoint}")
    
    # Manejo de interrupciones
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except KeyboardInterrupt:
        print("\n⚠️  Entrenamiento interrumpido por el usuario. Guardando checkpoint...")
        interrupted_checkpoint_dir = output_dir / "interrupted_checkpoint"
        trainer.save_model(str(interrupted_checkpoint_dir))
        tokenizer.save_pretrained(interrupted_checkpoint_dir)
        print(f"   Checkpoint guardado en: {interrupted_checkpoint_dir}")
        print(f"   Puedes continuar con: --resume {interrupted_checkpoint_dir}")
        raise
    
    # Validación post-entrenamiento
    print(f"\n📈 Evaluando modelo final...")
    try:
        final_metrics = trainer.evaluate()
        print(f"\n📊 Métricas finales:")
        for key, value in final_metrics.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
    except Exception as e:
        print(f"⚠️  No se pudieron calcular métricas finales: {e}")
    
    # Guardar modelo final
    print(f"\n💾 Guardando modelo final...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Guardar configuración de entrenamiento
    config_save_path = output_dir / "training_config.json"
    try:
        # Preparar configuración para guardar (convertir Paths a strings)
        config_to_save = {}
        for key, value in config.items():
            if isinstance(value, Path):
                config_to_save[key] = str(value)
            else:
                config_to_save[key] = value
        
        with open(config_save_path, "w") as f:
            json.dump(config_to_save, f, indent=2, default=str)
        print(f"   ✅ Configuración guardada en: {config_save_path}")
    except Exception as e:
        print(f"⚠️  No se pudo guardar configuración: {e}")
    
    print(f"\n✅ Entrenamiento completado!")
    print(f"   Modelo guardado en: {output_dir}")
    
    return trainer

def main():
    parser = argparse.ArgumentParser(
        description="Entrena modelo con QLoRA para Domain Adaptation"
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
        default=Path('models/cybersecurity-qlora'),
        help='Directorio de salida para el modelo (separado de train_lora.py)'
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
        trainer = train_qlora(
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

