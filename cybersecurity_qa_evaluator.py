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
        ollama_model: str = "mistral"
    ):
        self.weaviate_client = weaviate_client
        self.llm_provider = llm_provider
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.ollama_model = ollama_model
        
        # Inicializar sistema RAG
        print("🚀 Inicializando sistema RAG...")
        self.rag_system = OptimizedCrewAIRAG(
            weaviate_client=weaviate_client,
            model=openai_model if llm_provider == "openai" else ollama_model,
            llm_provider=llm_provider,
            openai_api_key=openai_api_key,
            max_iterations=1  # Una sola iteración para evaluación rápida
        )
        
        self.results: List[Dict[str, Any]] = []
        
    def download_dataset(self) -> str:
        """Descarga el dataset de Cybersecurity QA de Kaggle"""
        print("📥 Descargando dataset de Cybersecurity QA...")
        
        try:
            # Descargar dataset
            path = kagglehub.dataset_download("zobayer0x01/cybersecurity-qa")
            print(f"✅ Dataset descargado en: {path}")
            
            # Buscar archivos CSV en el directorio
            dataset_path = Path(path)
            csv_files = list(dataset_path.glob("*.csv"))
            
            if not csv_files:
                print("⚠️  No se encontraron archivos CSV en el dataset")
                return str(dataset_path)
            
            print(f"📄 Archivos encontrados: {[f.name for f in csv_files]}")
            return str(dataset_path)
            
        except Exception as e:
            print(f"❌ ERROR descargando dataset: {e}")
            raise
    
    def load_questions(self, dataset_path: str, max_questions: int = 10) -> List[Dict[str, str]]:
        """Carga preguntas del dataset"""
        print(f"📖 Cargando preguntas del dataset (máximo {max_questions})...")
        
        dataset_dir = Path(dataset_path)
        csv_files = list(dataset_dir.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError("No se encontraron archivos CSV en el dataset")
        
        # Usar el primer archivo CSV encontrado
        csv_file = csv_files[0]
        print(f"📄 Leyendo archivo: {csv_file.name}")
        
        try:
            df = pd.read_csv(csv_file)
            print(f"📊 Dataset cargado: {len(df)} filas, {len(df.columns)} columnas")
            print(f"📋 Columnas: {list(df.columns)}")
            
            # Buscar columnas que contengan preguntas
            question_columns = [col for col in df.columns if 'question' in col.lower() or 'query' in col.lower()]
            answer_columns = [col for col in df.columns if 'answer' in col.lower() or 'response' in col.lower()]
            
            print(f"❓ Columnas de preguntas encontradas: {question_columns}")
            print(f"✅ Columnas de respuestas encontradas: {answer_columns}")
            
            questions = []
            
            # Si hay columnas específicas de pregunta y respuesta
            if question_columns and answer_columns:
                question_col = question_columns[0]
                answer_col = answer_columns[0]
                
                for idx, row in df.head(max_questions).iterrows():
                    if pd.notna(row[question_col]) and pd.notna(row[answer_col]):
                        questions.append({
                            "question": str(row[question_col]).strip(),
                            "expected_answer": str(row[answer_col]).strip(),
                            "source": f"{csv_file.name}_row_{idx}"
                        })
            
            # Si no hay columnas específicas, usar la primera columna como pregunta
            elif len(df.columns) >= 1:
                for idx, row in df.head(max_questions).iterrows():
                    if pd.notna(row.iloc[0]):
                        questions.append({
                            "question": str(row.iloc[0]).strip(),
                            "expected_answer": "",
                            "source": f"{csv_file.name}_row_{idx}"
                        })
            
            print(f"✅ {len(questions)} preguntas cargadas")
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
            # Hacer pregunta al RAG
            result = self.rag_system.process_query(question, verbose=False)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Extraer información del resultado
            rag_answer = result.get("answer", "Sin respuesta")
            passages = result.get("passages", [])
            iterations = result.get("iterations", 0)
            
            # Crear resultado estructurado
            evaluation_result = {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "expected_answer": expected_answer,
                "rag_answer": rag_answer,
                "response_time_seconds": round(response_time, 2),
                "iterations": iterations,
                "num_sources": len(passages),
                "sources": [f"{p.get('doc_id', 'N/A')}:{p.get('title', 'N/A')}" for p in passages[:5]],
                "source_file": source,
                "llm_provider": self.llm_provider,
                "model": self.openai_model if self.llm_provider == "openai" else self.ollama_model
            }
            
            print(f"✅ Respuesta generada en {response_time:.2f}s")
            print(f"📚 Fuentes encontradas: {len(passages)}")
            
            return evaluation_result
            
        except Exception as e:
            print(f"❌ ERROR evaluando pregunta: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "expected_answer": expected_answer,
                "rag_answer": f"ERROR: {str(e)}",
                "response_time_seconds": 0,
                "iterations": 0,
                "num_sources": 0,
                "sources": [],
                "source_file": source,
                "llm_provider": self.llm_provider,
                "model": self.openai_model if self.llm_provider == "openai" else self.ollama_model,
                "error": str(e)
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
                        round(df['response_time_seconds'].mean(), 2),
                        round(df['response_time_seconds'].sum(), 2),
                        round(df['num_sources'].mean(), 1),
                        self.llm_provider,
                        self.openai_model if self.llm_provider == "openai" else self.ollama_model,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Resumen', index=False)
                
                # Hoja de errores (si los hay)
                error_results = df[df['rag_answer'].str.contains('ERROR', na=False)]
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
                avg_time = sum(r['response_time_seconds'] for r in results) / len(results)
                total_sources = sum(r['num_sources'] for r in results)
                print(f"⏱️  Tiempo promedio por pregunta: {avg_time:.2f}s")
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
