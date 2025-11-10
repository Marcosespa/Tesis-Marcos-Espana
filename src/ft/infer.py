#!/usr/bin/env python3
"""
Script de inferencia con modelo fine-tuneado
"""

import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

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
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Cargar adaptadores LoRA
        model = PeftModel.from_pretrained(base_model_obj, str(model_path))
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

def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50
):
    """Genera texto desde un prompt"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remover el prompt del resultado
    if generated.startswith(prompt):
        generated = generated[len(prompt):].strip()
    
    return generated

def main():
    parser = argparse.ArgumentParser(
        description="Inferencia con modelo fine-tuneado de Domain Adaptation"
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
        '--prompt',
        type=str,
        help='Prompt para generar texto'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Modo interactivo'
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
        help='Temperature para generación'
    )
    
    args = parser.parse_args()
    
    # Cargar modelo
    model, tokenizer = load_model(args.model, args.base_model)
    
    print(f"✅ Modelo cargado en dispositivo: {model.device}")
    
    if args.interactive:
        print("\n🎯 Modo interactivo (escribe 'quit' para salir)")
        print("=" * 60)
        
        while True:
            try:
                prompt = input("\n🔍 Prompt: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'salir']:
                    break
                
                if not prompt:
                    continue
                
                print("\n🤖 Generando...")
                generated = generate(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature
                )
                
                print(f"\n📝 Resultado:\n{generated}\n")
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\n\n👋 ¡Hasta luego!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    elif args.prompt:
        print(f"\n🔍 Prompt: {args.prompt}")
        print("🤖 Generando...")
        
        generated = generate(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        print(f"\n📝 Resultado:\n{generated}")
    else:
        print("❌ Debes proporcionar --prompt o usar --interactive")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

