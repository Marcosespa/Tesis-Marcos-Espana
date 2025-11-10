#!/usr/bin/env python3
"""
Evalúa modelo fine-tuneado con Domain Adaptation
"""

import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from datasets import load_dataset
import json

def load_model(model_path: Path, base_model: str = None):
    """Carga modelo fine-tuneado"""
    print(f"📥 Cargando modelo desde {model_path}")
    
    # Verificar si es modelo LoRA
    peft_config_path = model_path / "adapter_config.json"
    
    if peft_config_path.exists():
        print("   Detectado modelo LoRA")
        config = PeftConfig.from_pretrained(str(model_path))
        
        if base_model:
            base_model_name = base_model
        else:
            base_model_name = config.base_model_name_or_path
        
        print(f"   Modelo base: {base_model_name}")
        
        # Cargar modelo base
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Cargar adaptadores LoRA
        model = PeftModel.from_pretrained(base_model, str(model_path))
        model = model.merge_and_unload()  # Fusionar para evaluación
    else:
        # Modelo completo (no LoRA)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
    
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def evaluate_perplexity(model, tokenizer, dataset, max_length: int = 512):
    """Calcula perplexity en el dataset"""
    print(f"🔄 Calculando perplexity...")
    
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i, example in enumerate(dataset):
            text = example['text']
            
            # Tokenizar
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False
            ).to(model.device)
            
            # Calcular pérdida
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            total_loss += loss.item() * inputs["input_ids"].numel()
            total_tokens += inputs["input_ids"].numel()
            
            if (i + 1) % 100 == 0:
                print(f"   Procesados {i+1}/{len(dataset)} ejemplos")
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity, avg_loss

def generate_samples(model, tokenizer, prompts: list, max_new_tokens: int = 200):
    """Genera muestras de texto"""
    print(f"🎨 Generando muestras...")
    
    model.eval()
    results = []
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({
            "prompt": prompt,
            "generated": generated
        })
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Evalúa modelo fine-tuneado con Domain Adaptation"
    )
    
    parser.add_argument(
        '--model',
        type=Path,
        required=True,
        help='Directorio del modelo fine-tuneado'
    )
    
    parser.add_argument(
        '--base-model',
        type=str,
        help='Modelo base (si no se especifica, se lee del config)'
    )
    
    parser.add_argument(
        '--dataset',
        type=Path,
        help='Dataset de validación (JSONL)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        help='Archivo de salida para resultados'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=10,
        help='Número de muestras para generar'
    )
    
    args = parser.parse_args()
    
    # Cargar modelo
    model, tokenizer = load_model(args.model, args.base_model)
    
    print(f"✅ Modelo cargado en dispositivo: {model.device}")
    
    results = {}
    
    # Evaluar en dataset si se proporciona
    if args.dataset and args.dataset.exists():
        print(f"\n📊 Evaluando en dataset: {args.dataset}")
        
        dataset = load_dataset('json', data_files=str(args.dataset), split='train')
        
        # Limitar tamaño para evaluación rápida
        if len(dataset) > 1000:
            print(f"   Limitando a 1000 ejemplos para evaluación rápida")
            dataset = dataset.select(range(1000))
        
        perplexity, loss = evaluate_perplexity(model, tokenizer, dataset)
        
        results['perplexity'] = perplexity
        results['loss'] = loss
        
        print(f"\n📈 Resultados:")
        print(f"   Perplexity: {perplexity:.4f}")
        print(f"   Loss: {loss:.4f}")
    
    # Generar muestras
    print(f"\n🎨 Generando {args.samples} muestras...")
    
    sample_prompts = [
        "SQL Injection is a vulnerability that",
        "According to MITRE ATT&CK, Process Injection",
        "The NIST Cybersecurity Framework recommends",
        "Password hashing should use",
        "In cybersecurity, threat intelligence"
    ]
    
    # Limitar a args.samples
    sample_prompts = sample_prompts[:args.samples]
    
    samples = generate_samples(model, tokenizer, sample_prompts)
    
    results['samples'] = samples
    
    print(f"\n📝 Muestras generadas:")
    for i, sample in enumerate(samples, 1):
        print(f"\n{i}. Prompt: {sample['prompt']}")
        print(f"   Generado: {sample['generated'][:200]}...")
    
    # Guardar resultados
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Resultados guardados en: {args.output}")
    
    print(f"\n✅ Evaluación completada!")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

