#!/usr/bin/env python3
"""
Script de prueba para modelo fine-tuneado desde HuggingFace Hub
Modelo: marcosespa/Qwen_2.5_instruct_ft_for_cybersecurity
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

def load_finetuned_model(adapter_model_id: str = "marcosespa/Qwen_2.5_instruct_ft_for_cybersecurity"):
    """
    Carga modelo base + adaptadores LoRA desde HuggingFace Hub
    
    Args:
        adapter_model_id: ID del modelo con adaptadores en HuggingFace Hub
    
    Returns:
        model, tokenizer
    """
    print(f"📥 Cargando modelo base: Qwen/Qwen2.5-1.5B-Instruct")
    
    # Cargar modelo base
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    print(f"📥 Cargando adaptadores LoRA: {adapter_model_id}")
    
    # Cargar adaptadores LoRA desde HuggingFace Hub
    model = PeftModel.from_pretrained(base_model, adapter_model_id)
    
    print(f"✅ Modelo cargado en dispositivo: {model.device}")
    
    # Cargar tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_text(
    model, 
    tokenizer, 
    prompt: str, 
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9
):
    """
    Genera texto desde un prompt
    
    Args:
        model: Modelo cargado
        tokenizer: Tokenizer
        prompt: Texto inicial
        max_new_tokens: Máximo de tokens a generar
        temperature: Creatividad (0.0-1.0)
        top_p: Nucleus sampling
    
    Returns:
        Texto generado
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated

def test_cybersecurity_prompts(model, tokenizer):
    """Prueba con prompts específicos de ciberseguridad"""
    
    test_prompts = [
        "SQL Injection is a vulnerability that",
        "According to NIST, password hashing should",
        "The MITRE ATT&CK framework defines",
        "In cybersecurity, threat intelligence",
        "Buffer overflow attacks occur when",
        "Cross-site scripting (XSS) is a type of",
        "Zero-day vulnerabilities are",
        "Ransomware typically",
        "The principle of least privilege means",
        "Multi-factor authentication (MFA) provides"
    ]
    
    print("\n" + "="*80)
    print("🧪 PRUEBAS DE GENERACIÓN - DOMINIO CIBERSEGURIDAD")
    print("="*80)
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 Prueba {i}/{len(test_prompts)}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"🔍 Prompt: {prompt}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        generated = generate_text(model, tokenizer, prompt, max_new_tokens=150, temperature=0.7)
        
        # Extraer solo el texto generado (sin el prompt)
        if generated.startswith(prompt):
            generated = generated[len(prompt):].strip()
        
        print(f"🤖 Generado:\n{generated}")
        
        results.append({
            "prompt": prompt,
            "generated": generated
        })
    
    return results

def interactive_mode(model, tokenizer):
    """Modo interactivo para probar el modelo"""
    print("\n" + "="*80)
    print("🎯 MODO INTERACTIVO")
    print("="*80)
    print("\n📋 Comandos:")
    print("   • Escribe un prompt para generar texto")
    print("   • 'quit' o 'exit' para salir")
    print("   • 'temp X' para cambiar temperatura (ej: temp 0.3)")
    print("   • 'tokens X' para cambiar max_tokens (ej: tokens 500)")
    print()
    
    temperature = 0.7
    max_tokens = 200
    
    while True:
        try:
            prompt = input("🔍 Prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'salir', 'q']:
                print("\n👋 ¡Hasta luego!")
                break
            
            if prompt.lower().startswith('temp '):
                try:
                    temperature = float(prompt.split()[1])
                    print(f"✅ Temperature: {temperature}")
                    continue
                except:
                    print("❌ Formato: temp 0.7")
                    continue
            
            if prompt.lower().startswith('tokens '):
                try:
                    max_tokens = int(prompt.split()[1])
                    print(f"✅ Max tokens: {max_tokens}")
                    continue
                except:
                    print("❌ Formato: tokens 200")
                    continue
            
            if not prompt:
                continue
            
            print("\n🤖 Generando...")
            generated = generate_text(model, tokenizer, prompt, max_tokens, temperature)
            
            print(f"\n📝 Resultado:\n{generated}\n")
            print("─" * 80)
            
        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Prueba modelo fine-tuneado desde HuggingFace Hub"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default="marcosespa/Qwen_2.5_instruct_ft_for_cybersecurity",
        help='ID del modelo en HuggingFace Hub'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Modo interactivo'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Ejecutar pruebas automáticas con prompts predefinidos'
    )
    
    parser.add_argument(
        '--prompt',
        type=str,
        help='Prompt único para generar'
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
    
    # Cargar modelo
    print("🚀 Iniciando prueba del modelo fine-tuneado\n")
    model, tokenizer = load_finetuned_model(args.model)
    
    # Ejecutar según modo
    if args.test:
        # Pruebas automáticas
        results = test_cybersecurity_prompts(model, tokenizer)
        print(f"\n✅ Completadas {len(results)} pruebas")
        
    elif args.interactive:
        # Modo interactivo
        interactive_mode(model, tokenizer)
        
    elif args.prompt:
        # Single prompt
        print(f"🔍 Prompt: {args.prompt}\n")
        generated = generate_text(
            model, 
            tokenizer, 
            args.prompt, 
            args.max_tokens, 
            args.temperature
        )
        print(f"🤖 Generado:\n{generated}")
        
    else:
        print("❌ Debes usar --interactive, --test, o --prompt")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

