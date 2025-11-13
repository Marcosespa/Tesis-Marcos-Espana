#!/usr/bin/env python3
"""
Evaluador para modelo fine-tuneado usando dataset TopQuestions
Similar a cybersecurity_qa_evaluator.py pero para fine-tuning
"""

import os
import sys
import json
import time
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import argparse
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

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
        """Carga modelo fine-tuneado con adaptadores LoRA"""
        print(f"📥 Cargando modelo base: {self.base_model}")
        
        # Verificar GPU
        if torch.cuda.is_available():
            print(f"   🚀 GPU disponible: cuda:{self.gpu_id}")
            device_map = {"": f"cuda:{self.gpu_id}"}
            dtype = torch.float16
        else:
            print(f"   ⚠️  GPU no disponible, usando CPU (será lento)")
            device_map = None
            dtype = torch.float32
        
        # Verificar si el modelo path es LoRA
        model_path_obj = Path(self.model_path)
        adapter_config_path = model_path_obj / "adapter_config.json" if model_path_obj.is_dir() else None
        
        if adapter_config_path and adapter_config_path.exists():
            print(f"✅ Detectado modelo LoRA en: {self.model_path}")
            
            # Leer configuración de LoRA
            try:
                peft_config = PeftConfig.from_pretrained(str(self.model_path))
                print(f"   📋 Configuración LoRA:")
                print(f"      - Base model: {peft_config.base_model_name_or_path}")
                print(f"      - Task type: {peft_config.task_type}")
                print(f"      - LoRA alpha: {peft_config.lora_alpha}")
                print(f"      - LoRA r: {peft_config.r}")
                print(f"      - Target modules: {peft_config.target_modules}")
            except Exception as e:
                print(f"   ⚠️  No se pudo leer configuración LoRA: {e}")
            
            # Cargar modelo base
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True
            )
            
            print(f"📥 Cargando adaptadores LoRA desde: {self.model_path}")
            
            # Cargar adaptadores LoRA
            model = PeftModel.from_pretrained(base_model, str(self.model_path))
            
            print(f"✅ Adaptadores LoRA cargados correctamente")
            
        else:
            # Si no es LoRA, intentar cargar como modelo completo
            print(f"⚠️  No se detectó adapter_config.json, intentando cargar como modelo completo")
            model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True
            )
            print(f"✅ Modelo completo cargado (NO es LoRA)")
        
        # Cargar tokenizer (intentar desde modelo primero, luego desde base)
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        except:
            tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"✅ Modelo cargado en dispositivo: {model.device}")
        print(f"✅ Tokenizer cargado")
        
        return model, tokenizer
    
    def load_questions(self, max_questions: int = 10) -> List[Dict[str, str]]:
        """Carga preguntas del dataset TopQuestions"""
        print(f"📖 Cargando preguntas del dataset TopQuestions (máximo {max_questions})...")
        
        try:
            # Buscar archivo CSV (buscar en múltiples ubicaciones posibles)
            possible_paths = [
                Path("data/Questions/TopQuestions(in).csv"),  # Ruta relativa desde raíz del proyecto
                Path(__file__).parent.parent.parent / "data" / "Questions" / "TopQuestions(in).csv",  # Ruta relativa desde este script
                Path("/Users/marcosespana/Desktop/U/DatosTesis/data/Questions/TopQuestions(in).csv"),  # Ruta macOS (por compatibilidad)
            ]
            
            csv_file = None
            for path in possible_paths:
                if path.exists():
                    csv_file = path
                    break
            
            if csv_file is None:
                raise FileNotFoundError(
                    f"No se encontró TopQuestions(in).csv en ninguna de las ubicaciones:\n" +
                    "\n".join(f"  - {p}" for p in possible_paths)
                )
            
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
                    # Combinar título + cuerpo como pregunta completa (asegurar formato correcto)
                    question_text = f"{titulo}\n\n{cuerpo}".strip()
                    
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
        # Construir prompt con instrucciones detalladas
        instruction_prompt = (
            "IMPORTANT: Respuesta directa, concisa, lenguaje neutral profesional. "
            "Sin verbosidad ni adjetivación excesiva. Responde SOLO la pregunta.\n\n"
            f"Original Query: '{question}'\n\n"
            "MANDATORY Requirements for the final answer:\n"
            "- Answer DIRECTLY and CONCISELY - avoid verbosity\n"
            "- Professional, neutral language - NO excessive adjectives\n"
            "- Respond ONLY the question - no extra information\n"
            "- Answer in English, clearly and concisely\n"
            "- Minimum 100 words\n"
            "- Do NOT include citations or source references\n"
            "- Ground ALL claims in the retrieved documents but don't cite them\n"
            "- If there are practical steps, use numbered lists\n"
            "- If evidence is insufficient, be explicit about limitations\n"
            "- Write naturally without academic citations\n\n"
            "IMPORTANT: This is the FINAL answer. Make it direct, concise, and well-grounded."
        )
        
        # Intentar usar chat template si está disponible (para modelos instruct)
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
            # Formatear como mensaje de chat con las instrucciones
            messages = [{"role": "user", "content": instruction_prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Usar prompt con instrucciones directamente si no hay chat template
            formatted_prompt = instruction_prompt
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
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
        
        # Limpiar la respuesta: remover prompt y cualquier texto basura
        cleaned_response = self._clean_response(generated, formatted_prompt, question)
        
        return cleaned_response
    
    def _clean_response(self, generated_text: str, prompt: str, original_question: str) -> str:
        """
        Limpia la respuesta del modelo removiendo el prompt y cualquier texto basura
        
        Args:
            generated_text: Texto completo generado por el modelo
            prompt: Prompt completo que se envió al modelo
            original_question: Pregunta original (título + cuerpo)
        
        Returns:
            Respuesta limpia sin prompt ni texto basura
        """
        # Remover el prompt completo si está al inicio
        if generated_text.startswith(prompt):
            response = generated_text[len(prompt):].strip()
        # Fallback: remover pregunta original si está al inicio
        elif generated_text.startswith(original_question):
            response = generated_text[len(original_question):].strip()
        else:
            # Si no encontramos el prompt al inicio, buscar patrones comunes
            response = generated_text
            
            # Remover patrones comunes de chat templates
            # Para Qwen: buscar después de "<|im_start|>assistant\n" o similar
            assistant_markers = [
                "<|im_start|>assistant\n",
                "<|im_start|>assistant",
                "assistant\n",
                "Assistant:",
                "assistant:",
            ]
            
            for marker in assistant_markers:
                if marker in response:
                    parts = response.split(marker, 1)
                    if len(parts) > 1:
                        response = parts[1].strip()
                        break
        
        # Remover cualquier repetición de instrucciones o texto basura
        # Patrones a remover
        patterns_to_remove = [
            "IMPORTANT:",
            "MANDATORY Requirements",
            "Original Query:",
            "This is the FINAL answer",
            "Make it direct, concise",
            "Answer DIRECTLY",
            "Professional, neutral language",
        ]
        
        # Dividir en líneas y filtrar líneas que contengan patrones no deseados
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Saltar líneas vacías al inicio
            if not line and not cleaned_lines:
                continue
            
            # Saltar líneas que son claramente parte de las instrucciones
            is_instruction = False
            for pattern in patterns_to_remove:
                if pattern.lower() in line.lower():
                    is_instruction = True
                    break
            
            if not is_instruction:
                cleaned_lines.append(line)
        
        # Unir líneas y limpiar espacios múltiples
        cleaned_response = '\n'.join(cleaned_lines).strip()
        
        # Remover espacios múltiples (pero mantener saltos de línea para listas)
        # Solo colapsar espacios dentro de líneas, no entre párrafos
        lines = cleaned_response.split('\n')
        cleaned_lines = []
        for line in lines:
            # Colapsar múltiples espacios en una línea, pero mantener la línea
            cleaned_line = re.sub(r' +', ' ', line.strip())
            if cleaned_line:  # Solo agregar líneas no vacías
                cleaned_lines.append(cleaned_line)
        
        cleaned_response = '\n'.join(cleaned_lines)
        
        # Remover múltiples saltos de línea consecutivos (máximo 2)
        cleaned_response = re.sub(r'\n{3,}', '\n\n', cleaned_response)
        
        # Si la respuesta está vacía o es muy corta, devolver el texto original procesado
        if len(cleaned_response) < 10:
            # Último intento: buscar después de la pregunta original
            if original_question in response:
                parts = response.split(original_question, 1)
                if len(parts) > 1:
                    cleaned_response = parts[1].strip()
        
        return cleaned_response.strip()
    
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
        
        # Crear directorio de resultados si no existe
        results_dir = Path("results/excel/ft_evaluations")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Si filename no es una ruta absoluta, guardarlo en results_dir
        filename_path = Path(filename)
        if not filename_path.is_absolute():
            filename = str(results_dir / filename)
        
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
        default=100,
        help='Número máximo de preguntas a evaluar (default: 100)'
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
        default=3,
        help='ID de GPU a usar (default: 3)'
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

