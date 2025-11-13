#!/usr/bin/env python3
"""
Evaluador para modelo fine-tuneado usando dataset TopQuestions
Similar a cybersecurity_qa_evaluator.py pero para fine-tuning
"""

import os
import sys
import json
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import argparse
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class FineTunedModelEvaluator:
    """Evaluador para modelo fine-tuneado usando TopQuestions dataset"""
    
    def __init__(
        self,
        model_path: str,
        base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        gpu_id: int = 0,
        max_new_tokens: int = 300,
        temperature: float = 0.7
    ):
        """
        Inicializa el evaluador
        
        Args:
            model_path: Path al modelo fine-tuned o ID en HuggingFace Hub
            base_model: Modelo base
            gpu_id: ID de GPU a usar
            max_new_tokens: Máximo de tokens a generar
            temperature: Temperature para generación
        """
        self.model_path = model_path
        self.base_model = base_model
        self.gpu_id = gpu_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.results: List[Dict[str, Any]] = []
        
        # Cargar modelo
        print("🚀 Inicializando modelo fine-tuneado...")
        self.model, self.tokenizer = self.load_model()
        
    def load_model(self):
        """Carga modelo fine-tuneado"""
        print(f"📥 Cargando modelo base: {self.base_model}")
        
        # Verificar GPU
        if torch.cuda.is_available():
            print(f"   🚀 GPU disponible: cuda:{self.gpu_id}")
            device = f"cuda:{self.gpu_id}"
            dtype = torch.float16
        else:
            print(f"   ⚠️  GPU no disponible, usando CPU (será lento)")
            device = None
            dtype = torch.float32
        
        # Cargar modelo base
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True
        )
        
        print(f"📥 Cargando adaptadores LoRA: {self.model_path}")
        
        # Cargar adaptadores LoRA
        model = PeftModel.from_pretrained(base_model, self.model_path)
        
        # Cargar tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"✅ Modelo cargado en dispositivo: {model.device}")
        
        return model, tokenizer
    
    def load_questions(self, max_questions: int = 10) -> List[Dict[str, str]]:
        """Carga preguntas del dataset TopQuestions"""
        print(f"📖 Cargando preguntas del dataset TopQuestions (máximo {max_questions})...")
        
        try:
            # Buscar archivo CSV
            csv_file = Path("/Users/marcosespana/Desktop/U/DatosTesis/data/Questions/TopQuestions(in).csv")
            if not csv_file.exists():
                csv_file = Path("data/Questions/TopQuestions(in).csv")
                if not csv_file.exists():
                    raise FileNotFoundError("No se encontró TopQuestions(in).csv")
            
            # Leer CSV
            df = pd.read_csv(csv_file, sep=';')
            
            print(f"📊 Dataset cargado: {len(df)} filas")
            print(f"📋 Columnas: {list(df.columns)}")
            
            questions = []
            
            # Combinar Titulo + Cuerpo para cada pregunta
            for idx, row in df.head(max_questions).iterrows():
                titulo = str(row.get('Titulo', '')).strip() if pd.notna(row.get('Titulo')) else ''
                cuerpo = str(row.get('Cuerpo', '')).strip() if pd.notna(row.get('Cuerpo')) else ''
                
                if titulo and cuerpo:
                    # Combinar título + cuerpo como pregunta completa
                    question_text = f"{titulo}\n\n{cuerpo}"
                    
                    # Respuesta esperada
                    expected_answer = str(row.get('Respuesta', '')).strip() if pd.notna(row.get('Respuesta')) else ''
                    
                    # Guardar todos los datos originales
                    original_data = row.to_dict()
                    
                    questions.append({
                        "question": question_text,
                        "expected_answer": expected_answer,
                        "source": f"TopQuestions_row_{idx}",
                        "title": titulo,
                        "body": cuerpo,
                        "original_data": original_data
                    })
            
            print(f"✅ {len(questions)} preguntas cargadas")
            return questions
            
        except Exception as e:
            print(f"❌ ERROR cargando dataset: {e}")
            raise
    
    def generate_answer(self, question: str) -> str:
        """Genera respuesta usando el modelo fine-tuneado"""
        inputs = self.tokenizer(question, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remover el prompt de la respuesta
        if generated.startswith(question):
            generated = generated[len(question):].strip()
        
        return generated
    
    def evaluate_question(self, question_data: Dict[str, str]) -> Dict[str, Any]:
        """Evalúa una pregunta con el modelo fine-tuneado"""
        question = question_data["question"]
        expected_answer = question_data.get("expected_answer", "")
        
        print(f"\n🔍 Evaluando: {question_data['title'][:80]}...")
        
        start_time = time.time()
        
        try:
            # Generar respuesta
            generated_answer = self.generate_answer(question)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Obtener datos originales
            original_data = question_data.get("original_data", {})
            
            # Crear resultado estructurado
            evaluation_result = {
                # Incluir todas las columnas originales del CSV
                **original_data,
                # Agregar columnas del modelo fine-tuned
                "FT_Model": f"{self.base_model} (Fine-Tuned)",
                "FT_Adapter": self.model_path,
                "FT_Response": generated_answer,
                "FT_Response_Time_Seconds": round(response_time, 2),
                "FT_Max_Tokens": self.max_new_tokens,
                "FT_Temperature": self.temperature,
                "FT_Response_Length_Chars": len(generated_answer),
                "FT_Response_Length_Words": len(generated_answer.split()),
                "FT_Timestamp": datetime.now().isoformat()
            }
            
            print(f"✅ Respuesta generada en {response_time:.2f}s")
            print(f"📏 Longitud: {len(generated_answer.split())} palabras")
            
            return evaluation_result
            
        except Exception as e:
            print(f"❌ ERROR evaluando pregunta: {e}")
            
            original_data = question_data.get("original_data", {})
            
            return {
                **original_data,
                "FT_Model": f"{self.base_model} (Fine-Tuned)",
                "FT_Adapter": self.model_path,
                "FT_Response": f"ERROR: {str(e)}",
                "FT_Response_Time_Seconds": 0,
                "FT_Max_Tokens": self.max_new_tokens,
                "FT_Temperature": self.temperature,
                "FT_Response_Length_Chars": 0,
                "FT_Response_Length_Words": 0,
                "FT_Timestamp": datetime.now().isoformat(),
                "FT_Error": str(e)
            }
    
    def run_evaluation(self, max_questions: int = 10) -> List[Dict[str, Any]]:
        """Ejecuta la evaluación completa"""
        print("\n🚀 INICIANDO EVALUACIÓN DE MODELO FINE-TUNED")
        print("=" * 80)
        
        # Cargar preguntas
        questions = self.load_questions(max_questions)
        
        if not questions:
            print("❌ No se pudieron cargar preguntas")
            return []
        
        # Evaluar cada pregunta
        print(f"\n🎯 Evaluando {len(questions)} preguntas...")
        
        for i, question_data in enumerate(questions, 1):
            print(f"\n{'='*60}")
            print(f"📝 Pregunta {i}/{len(questions)}")
            print(f"{'='*60}")
            
            result = self.evaluate_question(question_data)
            self.results.append(result)
            
            # Pausa breve entre preguntas
            time.sleep(0.5)
        
        print(f"\n✅ Evaluación completada: {len(self.results)} preguntas procesadas")
        return self.results
    
    def save_to_excel(self, filename: str = None) -> str:
        """Guarda resultados en Excel"""
        if not self.results:
            print("❌ No hay resultados para guardar")
            return ""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = Path(self.model_path).name.replace('/', '_')
            filename = f"ft_evaluation_{model_name}_{timestamp}.xlsx"
        
        print(f"💾 Guardando resultados en: {filename}")
        
        try:
            df = pd.DataFrame(self.results)
            
            # Crear Excel con múltiples hojas
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Hoja principal
                df.to_excel(writer, sheet_name='Resultados', index=False)
                
                # Hoja de resumen
                summary_data = {
                    "Métrica": [
                        "Total preguntas",
                        "Tiempo promedio (segundos)",
                        "Tiempo total (segundos)",
                        "Longitud promedio respuesta (palabras)",
                        "Modelo base",
                        "Adaptadores LoRA",
                        "Max tokens generados",
                        "Temperature",
                        "Fecha de evaluación"
                    ],
                    "Valor": [
                        len(self.results),
                        round(df['FT_Response_Time_Seconds'].mean(), 2) if 'FT_Response_Time_Seconds' in df.columns else 0,
                        round(df['FT_Response_Time_Seconds'].sum(), 2) if 'FT_Response_Time_Seconds' in df.columns else 0,
                        round(df['FT_Response_Length_Words'].mean(), 1) if 'FT_Response_Length_Words' in df.columns else 0,
                        self.base_model,
                        self.model_path,
                        self.max_new_tokens,
                        self.temperature,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Resumen', index=False)
                
                # Hoja de errores (si los hay)
                if 'FT_Response' in df.columns:
                    error_results = df[df['FT_Response'].str.contains('ERROR', na=False)]
                    if not error_results.empty:
                        error_results.to_excel(writer, sheet_name='Errores', index=False)
                
                # Hoja de estadísticas de respuestas
                stats_data = {
                    "Estadística": [
                        "Respuesta más corta (palabras)",
                        "Respuesta más larga (palabras)",
                        "Mediana longitud",
                        "Desviación estándar longitud"
                    ],
                    "Valor": [
                        df['FT_Response_Length_Words'].min() if 'FT_Response_Length_Words' in df.columns else 0,
                        df['FT_Response_Length_Words'].max() if 'FT_Response_Length_Words' in df.columns else 0,
                        df['FT_Response_Length_Words'].median() if 'FT_Response_Length_Words' in df.columns else 0,
                        round(df['FT_Response_Length_Words'].std(), 2) if 'FT_Response_Length_Words' in df.columns else 0
                    ]
                }
                
                stats_df = pd.DataFrame(stats_data)
                stats_df.to_excel(writer, sheet_name='Estadisticas', index=False)
            
            print(f"✅ Resultados guardados en: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ ERROR guardando Excel: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Evaluador de modelo fine-tuned usando TopQuestions dataset"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path al modelo fine-tuned o ID en HuggingFace Hub (ej: marcosespa/Qwen_2.5_instruct_ft_for_cybersecurity)'
    )
    
    parser.add_argument(
        '--base-model',
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help='Modelo base usado para fine-tuning'
    )
    
    parser.add_argument(
        '--max-questions',
        type=int,
        default=10,
        help='Número máximo de preguntas a evaluar (default: 10)'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=300,
        help='Máximo de tokens a generar por respuesta (default: 300)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Temperature para generación (default: 0.7)'
    )
    
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='ID de GPU a usar (default: 0)'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        help='Nombre del archivo Excel de salida'
    )
    
    args = parser.parse_args()
    
    try:
        # Crear evaluador
        evaluator = FineTunedModelEvaluator(
            model_path=args.model,
            base_model=args.base_model,
            gpu_id=args.gpu_id,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        # Ejecutar evaluación
        results = evaluator.run_evaluation(max_questions=args.max_questions)
        
        if results:
            # Guardar resultados
            output_file = evaluator.save_to_excel(args.output_file)
            
            print(f"\n🎉 EVALUACIÓN COMPLETADA")
            print(f"=" * 80)
            print(f"📊 {len(results)} preguntas evaluadas")
            print(f"💾 Resultados guardados en: {output_file}")
            
            # Estadísticas rápidas
            times = [r.get('FT_Response_Time_Seconds', 0) for r in results if 'FT_Response_Time_Seconds' in r]
            words = [r.get('FT_Response_Length_Words', 0) for r in results if 'FT_Response_Length_Words' in r]
            
            if times:
                avg_time = sum(times) / len(times)
                total_time = sum(times)
                print(f"⏱️  Tiempo promedio: {avg_time:.2f}s")
                print(f"⏱️  Tiempo total: {total_time:.2f}s")
            
            if words:
                avg_words = sum(words) / len(words)
                print(f"📏 Longitud promedio: {avg_words:.0f} palabras")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Evaluación interrumpida por el usuario")
        return 130
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

