#!/usr/bin/env python3
"""
Script para evaluar el sistema RAG con el dataset de Cybersecurity QA de Kaggle
Descarga el dataset, hace preguntas al RAG y guarda resultados en Excel
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

# Agregar el directorio src al path
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

try:
    import kagglehub
except ImportError:
    print("❌ ERROR: kagglehub no está instalado")
    print("Instala con: pip install kagglehub")
    sys.exit(1)

try:
    import weaviate
    from rag.agent.crewai_agentic_rag_optimized import OptimizedCrewAIRAG
except ImportError as e:
    print(f"❌ ERROR: No se pudo importar el sistema RAG: {e}")
    print("Asegúrate de que el sistema RAG esté configurado correctamente")
    sys.exit(1)

try:
    import openpyxl
except ImportError:
    print("❌ ERROR: openpyxl no está instalado")
    print("Instala con: pip install openpyxl")
    sys.exit(1)


class CybersecurityQAEvaluator:
    """Evaluador del sistema RAG usando dataset de Cybersecurity QA"""
    
    def __init__(
        self,
        weaviate_client,
        llm_provider: str = "ollama",
        openai_api_key: str = None,
        openai_model: str = "gpt-3.5-turbo",
        ollama_model: str = "mistral",
        skip_quality_evaluation: bool = False  # Nueva opción
    ):
        self.weaviate_client = weaviate_client
        self.llm_provider = llm_provider
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.ollama_model = ollama_model
        self.skip_quality_evaluation = skip_quality_evaluation
        
        # Inicializar sistema RAG
        print("🚀 Inicializando sistema RAG...")
        if skip_quality_evaluation:
            print("⚡ Modo RÁPIDO: Quality Evaluator deshabilitado")
        
        self.rag_system = OptimizedCrewAIRAG(
            weaviate_client=weaviate_client,
            model=openai_model if llm_provider == "openai" else ollama_model,
            llm_provider=llm_provider,
            openai_api_key=openai_api_key,
            max_iterations=1  # Una sola iteración para evaluación rápida
        )
        
        self.results: List[Dict[str, Any]] = []
        
    def download_dataset(self) -> str:
        """Usa el dataset local de TopQuestions"""
        print("📥 Usando dataset local de TopQuestions...")
        
        try:
            # Archivo local
            csv_file = Path("/Users/marcosespana/Desktop/U/DatosTesis/data/Questions/TopQuestions(in).csv")
            
            if not csv_file.exists():
                # Intentar ruta relativa
                csv_file = Path("data/Questions/TopQuestions(in).csv")
                if not csv_file.exists():
                    raise FileNotFoundError(f"No se encontró el archivo en ninguna ubicación")
            
            print(f"✅ Dataset local encontrado: {csv_file}")
            print(f"📊 Ubicación: {csv_file.absolute()}")
            
            return str(csv_file.parent)
            
        except Exception as e:
            print(f"❌ ERROR accediendo al dataset: {e}")
            raise
    
    def load_questions(self, dataset_path: str, max_questions: int = 10) -> List[Dict[str, str]]:
        """Carga preguntas del dataset TopQuestions, combinando Titulo + Cuerpo"""
        print(f"📖 Cargando preguntas del dataset TopQuestions (máximo {max_questions})...")
        
        try:
            # Leer el archivo CSV con separador semicolon
            csv_file = Path("/Users/marcosespana/Desktop/U/DatosTesis/data/Questions/TopQuestions(in).csv")
            if not csv_file.exists():
                csv_file = Path("data/Questions/TopQuestions(in).csv")
            
            df = pd.read_csv(csv_file, sep=';')
            
            print(f"📊 Dataset cargado: {len(df)} filas, {len(df.columns)} columnas")
            print(f"📋 Columnas: {list(df.columns)}")
            
            questions = []
            
            # Combinar Titulo + Cuerpo para cada pregunta
            for idx, row in df.head(max_questions).iterrows():
                titulo = str(row.get('Titulo', '')).strip() if pd.notna(row.get('Titulo')) else ''
                cuerpo = str(row.get('Cuerpo', '')).strip() if pd.notna(row.get('Cuerpo')) else ''
                
                if titulo and cuerpo:
                    # Combinar título + cuerpo como pregunta completa
                    question_text = f"{titulo}\n\n{cuerpo}"
                    
                    # Respuesta esperada (si existe la columna 'Respuesta')
                    expected_answer = str(row.get('Respuesta', '')).strip() if pd.notna(row.get('Respuesta')) else ''
                    
                    # Guardar todos los datos originales de la fila
                    original_data = row.to_dict()
                    
                    questions.append({
                        "question": question_text,
                        "expected_answer": expected_answer,
                        "source": f"TopQuestions_row_{idx}",
                        "title": titulo,
                        "body": cuerpo,
                        "original_data": original_data  # Guardar todos los datos originales
                    })
            
            print(f"✅ {len(questions)} preguntas cargadas (Titulo + Cuerpo combinados)")
            return questions
            
        except Exception as e:
            print(f"❌ ERROR cargando dataset: {e}")
            raise
    
    def evaluate_question(self, question_data: Dict[str, str]) -> Dict[str, Any]:
        """Evalúa una pregunta usando el sistema RAG"""
        question = question_data["question"]
        expected_answer = question_data.get("expected_answer", "")
        source = question_data.get("source", "")
        
        print(f"\n🔍 Evaluando pregunta: {question[:100]}...")
        
        start_time = time.time()
        
        try:
            # Obtener datos originales
            original_data = question_data.get("original_data", {})
            
            # Hacer pregunta al RAG con verbose desactivado para mayor velocidad
            # TODO: Agregar opción para desactivar quality evaluator cuando skip_quality_evaluation=True
            result = self.rag_system.process_query(question, verbose=False)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Extraer información del resultado
            rag_answer = result.get("answer", "Sin respuesta")
            quality_evaluation = result.get("quality_evaluation", "Sin evaluación")
            passages = result.get("passages", [])
            iterations = result.get("iterations", 0)
            
            # Crear resultado estructurado con TODA la información original
            evaluation_result = {
                # Primero, incluir todas las columnas del CSV original
                **original_data,
                # Agregar columnas del sistema RAG
                "RAG_Model": "Mistral7B",
                "RAG_Response": rag_answer,  # Respuesta pura del modelo
                "RAG_Quality_Evaluation": quality_evaluation,
                "RAG_Response_Time_Seconds": round(response_time, 2),
                "RAG_Iterations": iterations,
                "RAG_Num_Sources": len(passages),
                "RAG_Sources": [f"{p.get('doc_id', 'N/A')}" for p in passages],
                "RAG_Timestamp": datetime.now().isoformat()
            }
            
            print(f"✅ Respuesta generada en {response_time:.2f}s")
            print(f"📚 Fuentes encontradas: {len(passages)}")
            
            return evaluation_result
            
        except Exception as e:
            print(f"❌ ERROR evaluando pregunta: {e}")
            
            # Obtener datos originales para el error también
            original_data = question_data.get("original_data", {})
            
            return {
                # Incluir todas las columnas originales
                **original_data,
                # Agregar columnas de error
                "RAG_Model": "Mistral7B",
                "RAG_Response": f"ERROR: {str(e)}",
                "RAG_Quality_Evaluation": "Error occurred - no evaluation available",
                "RAG_Response_Time_Seconds": 0,
                "RAG_Iterations": 0,
                "RAG_Num_Sources": 0,
                "RAG_Sources": [],
                "RAG_Timestamp": datetime.now().isoformat(),
                "RAG_Error": str(e)
            }
    
    def run_evaluation(self, max_questions: int = 10) -> List[Dict[str, Any]]:
        """Ejecuta la evaluación completa"""
        print("🚀 INICIANDO EVALUACIÓN DE CYBERSECURITY QA")
        print("=" * 80)
        
        # 1. Descargar dataset
        dataset_path = self.download_dataset()
        
        # 2. Cargar preguntas
        questions = self.load_questions(dataset_path, max_questions)
        
        if not questions:
            print("❌ No se pudieron cargar preguntas del dataset")
            return []
        
        # 3. Evaluar cada pregunta
        print(f"\n🎯 Evaluando {len(questions)} preguntas...")
        
        for i, question_data in enumerate(questions, 1):
            print(f"\n{'='*60}")
            print(f"📝 Pregunta {i}/{len(questions)}")
            print(f"{'='*60}")
            
            result = self.evaluate_question(question_data)
            self.results.append(result)
            
            # Pequeña pausa entre preguntas
            time.sleep(1)
        
        print(f"\n✅ Evaluación completada: {len(self.results)} preguntas procesadas")
        return self.results
    
    def save_to_excel(self, filename: str = None) -> str:
        """Guarda los resultados en un archivo Excel"""
        if not self.results:
            print("❌ No hay resultados para guardar")
            return ""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cybersecurity_qa_evaluation_{timestamp}.xlsx"
        
        print(f"💾 Guardando resultados en: {filename}")
        
        try:
            # Crear DataFrame
            df = pd.DataFrame(self.results)
            
            # Crear archivo Excel con múltiples hojas
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Hoja principal con todos los resultados
                df.to_excel(writer, sheet_name='Resultados', index=False)
                
                # Hoja de resumen
                summary_data = {
                    "Métrica": [
                        "Total preguntas",
                        "Tiempo promedio (segundos)",
                        "Tiempo total (segundos)",
                        "Fuentes promedio por pregunta",
                        "Proveedor LLM",
                        "Modelo usado",
                        "Fecha de evaluación"
                    ],
                    "Valor": [
                        len(self.results),
                        round(df['RAG_Response_Time_Seconds'].mean(), 2) if 'RAG_Response_Time_Seconds' in df.columns else 0,
                        round(df['RAG_Response_Time_Seconds'].sum(), 2) if 'RAG_Response_Time_Seconds' in df.columns else 0,
                        round(df['RAG_Num_Sources'].mean(), 1) if 'RAG_Num_Sources' in df.columns else 0,
                        self.llm_provider,
                        "Mistral7B",
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Resumen', index=False)
                
                # Hoja de errores (si los hay)
                if 'RAG_Response' in df.columns:
                    error_results = df[df['RAG_Response'].str.contains('ERROR', na=False)]
                    if not error_results.empty:
                        error_results.to_excel(writer, sheet_name='Errores', index=False)
            
            print(f"✅ Resultados guardados exitosamente en: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ ERROR guardando Excel: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Evaluador del sistema RAG con dataset de Cybersecurity QA"
    )
    
    parser.add_argument("--max-questions", type=int, default=10,
                       help="Número máximo de preguntas a evaluar")
    parser.add_argument("--output-file", help="Archivo Excel de salida")
    parser.add_argument("--llm-provider", choices=["ollama", "openai"], default="ollama",
                       help="Proveedor de LLM")
    parser.add_argument("--openai-api-key", help="API Key de OpenAI")
    parser.add_argument("--openai-model", default="gpt-3.5-turbo",
                       help="Modelo de OpenAI")
    parser.add_argument("--ollama-model", default="mistral",
                       help="Modelo de Ollama")
    parser.add_argument("--weaviate-host", default="localhost",
                       help="Host de Weaviate")
    parser.add_argument("--weaviate-port", type=int, default=8080,
                       help="Puerto de Weaviate")
    
    args = parser.parse_args()
    
    weaviate_client = None
    
    try:
        print("🔌 Conectando a Weaviate...")
        weaviate_client = weaviate.connect_to_local(
            host=args.weaviate_host,
            port=args.weaviate_port
        )
        
        # Crear evaluador
        evaluator = CybersecurityQAEvaluator(
            weaviate_client=weaviate_client,
            llm_provider=args.llm_provider,
            openai_api_key=args.openai_api_key,
            openai_model=args.openai_model,
            ollama_model=args.ollama_model
        )
        
        # Ejecutar evaluación
        results = evaluator.run_evaluation(max_questions=args.max_questions)
        
        if results:
            # Guardar resultados
            output_file = evaluator.save_to_excel(args.output_file)
            
            print(f"\n🎉 EVALUACIÓN COMPLETADA")
            print(f"📊 {len(results)} preguntas evaluadas")
            print(f"💾 Resultados guardados en: {output_file}")
            
            # Mostrar estadísticas rápidas
            if results:
                # Extraer métricas correctamente
                times = [r.get('RAG_Response_Time_Seconds', 0) for r in results if 'RAG_Response_Time_Seconds' in r]
                sources = [r.get('RAG_Num_Sources', 0) for r in results if 'RAG_Num_Sources' in r]
                
                if times:
                    avg_time = sum(times) / len(times)
                    print(f"⏱️  Tiempo promedio por pregunta: {avg_time:.2f}s")
                
                if sources:
                    total_sources = sum(sources)
                    print(f"📚 Total de fuentes encontradas: {total_sources}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Evaluación interrumpida por el usuario")
        return 130
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        if weaviate_client:
            try:
                weaviate_client.close()
                print("🧹 Conexión a Weaviate cerrada")
            except:
                pass


if __name__ == "__main__":
    sys.exit(main())
