#!/usr/bin/env python3
"""
Compara modelo base vs modelo fine-tuneado
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
from typing import List, Dict

def load_models(adapter_model_id: str = "marcosespa/Qwen_2.5_instruct_ft_for_cybersecurity"):
    """Carga modelo base y modelo fine-tuneado"""
    
    print("📥 Cargando modelo BASE...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Modelo base cargado")
    
    print(f"\n📥 Cargando modelo FINE-TUNED...")
    print(f"   Adaptadores LoRA: {adapter_model_id}")
    
    # Cargar segunda instancia del base model
    base_model_ft = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    # Cargar adaptadores
    finetuned_model = PeftModel.from_pretrained(base_model_ft, adapter_model_id)
    
    print("✅ Modelo fine-tuned cargado")
    
    return base_model, finetuned_model, tokenizer

def generate(model, tokenizer, prompt: str, max_tokens: int = 200, temperature: float = 0.7):
    """Genera texto desde un modelo"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remover prompt del resultado
    if generated.startswith(prompt):
        generated = generated[len(prompt):].strip()
    
    return generated

def compare_models(
    base_model, 
    finetuned_model, 
    tokenizer, 
    prompts: List[str],
    max_tokens: int = 200,
    temperature: float = 0.7
):
    """Compara generaciones de ambos modelos"""
    
    print("\n" + "="*80)
    print("🔬 COMPARACIÓN: Modelo Base vs Modelo Fine-Tuned")
    print("="*80)
    
    results = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'─'*80}")
        print(f"📝 Prueba {i}/{len(prompts)}")
        print(f"{'─'*80}")
        print(f"🔍 Prompt: {prompt}")
        print(f"{'─'*80}\n")
        
        # Generar con modelo base
        print("🔵 Modelo BASE:")
        base_result = generate(base_model, tokenizer, prompt, max_tokens, temperature)
        print(f"{base_result}\n")
        
        # Generar con modelo fine-tuned
        print("🟢 Modelo FINE-TUNED:")
        ft_result = generate(finetuned_model, tokenizer, prompt, max_tokens, temperature)
        print(f"{ft_result}\n")
        
        results.append({
            "prompt": prompt,
            "base": base_result,
            "finetuned": ft_result
        })
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Compara modelo base vs fine-tuned en ciberseguridad"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default="marcosespa/Qwen_2.5_instruct_ft_for_cybersecurity",
        help='ID del modelo fine-tuned en HuggingFace Hub'
    )
    
    parser.add_argument(
        '--prompts',
        nargs='+',
        help='Prompts personalizados para comparar'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=200,
        help='Máximo de tokens a generar'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Temperature (0.0-1.0)'
    )
    
    args = parser.parse_args()
    
    # Cargar modelos
    base_model, finetuned_model, tokenizer = load_models(args.model)
    
    # Prompts de prueba
    if args.prompts:
        test_prompts = args.prompts
    else:
        # Prompts predefinidos de ciberseguridad
        test_prompts = [
            "SQL Injection is a vulnerability that",
            "According to NIST, password hashing should",
            "The MITRE ATT&CK framework defines",
            "In cybersecurity, threat intelligence",
            "Buffer overflow attacks occur when",
            "Zero-day vulnerabilities are",
            "Ransomware typically",
            "The principle of least privilege"
        ]
    
    # Comparar
    results = compare_models(
        base_model,
        finetuned_model,
        tokenizer,
        test_prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    # Resumen
    print("\n" + "="*80)
    print("✅ COMPARACIÓN COMPLETADA")
    print("="*80)
    print(f"   Pruebas ejecutadas: {len(results)}")
    print(f"   Temperatura: {args.temperature}")
    print(f"   Max tokens: {args.max_tokens}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

