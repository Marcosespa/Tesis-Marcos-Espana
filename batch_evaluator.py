#!/usr/bin/env python3
"""
Evaluador por LOTES con paralelización y optimizaciones
Ejecuta lotes de 1000 preguntas con workers paralelos
"""

import sys
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import weaviate
from rag.agent.crewai_agentic_rag_optimized import OptimizedCrewAIRAG


class BatchEvaluator:
    """Evaluador con procesamiento por lotes y paralelización"""
    
    def __init__(self, max_workers: int = 2):
        self.max_workers = max_workers
        print(f"🔧 Máximo de workers: {max_workers}")
        
    def load_questions(self, start_idx: int = 0, num_questions: int = 1000) -> List[Dict]:
        """Carga un lote de preguntas"""
        csv_file = Path("/Users/marcosespana/Desktop/U/DatosTesis/data/Questions/TopQuestions(in).csv")
        df = pd.read_csv(csv_file, sep=';')
        
        total = len(df)
        end_idx = min(start_idx + num_questions, total)
        
        print(f"📖 Cargando preguntas {start_idx} a {end_idx} (total: {total})")
        
        questions = []
        for idx in range(start_idx, end_idx):
            row = df.iloc[idx]
            titulo = str(row.get('Titulo', '')).strip() if pd.notna(row.get('Titulo')) else ''
            cuerpo = str(row.get('Cuerpo', '')).strip() if pd.notna(row.get('Cuerpo')) else ''
            
            if titulo and cuerpo:
                questions.append({
                    "question": f"{titulo}\n\n{cuerpo}",
                    "original_data": row.to_dict(),
                    "row_idx": idx
                })
        
        print(f"✅ {len(questions)} preguntas cargadas")
        return questions
    
    def evaluate_single(self, q_data: Dict, rag_system) -> Dict:
        """Evalúa una sola pregunta"""
        question = q_data['question']
        original_data = q_data['original_data']
        
        try:
            start = time.time()
            result = rag_system.process_query(question, verbose=False)
            elapsed = time.time() - start
            
            return {
                **original_data,
                "RAG_Model": "Mistral7B",
                "RAG_Response": result.get("answer", ""),
                "RAG_Quality_Evaluation": result.get("quality_evaluation", ""),
                "RAG_Response_Time_Seconds": round(elapsed, 2),
                "RAG_Iterations": 1,
                "RAG_Num_Sources": len(result.get("passages", [])),
                "RAG_Sources": [p.get('doc_id', 'N/A') for p in result.get("passages", [])],
                "RAG_Timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                **original_data,
                "RAG_Model": "Mistral7B",
                "RAG_Response": f"ERROR: {str(e)}",
                "RAG_Quality_Evaluation": "",
                "RAG_Response_Time_Seconds": 0,
                "RAG_Iterations": 0,
                "RAG_Num_Sources": 0,
                "RAG_Sources": [],
                "RAG_Timestamp": datetime.now().isoformat(),
                "RAG_Error": str(e)
            }
    
    def evaluate_single_threaded(self, questions: List[Dict], batch_num: int) -> List[Dict]:
        """Evalúa un lote de preguntas sin paralelización (más seguro para Weaviate)"""
        print(f"\n{'='*80}")
        print(f"🚀 EJECUTANDO LOTE {batch_num}: {len(questions)} preguntas")
        print(f"{'='*80}\n")
        
        # Crear sistema RAG
        client = weaviate.connect_to_local()
        rag_system = OptimizedCrewAIRAG(
            weaviate_client=client,
            model="mistral",
            llm_provider="ollama"
        )
        
        results = []
        total_start = time.time()
        
        # Procesar secuencialmente
        for i, q_data in enumerate(questions, 1):
            try:
                result = self.evaluate_single(q_data, rag_system)
                results.append(result)
                
                if i % 10 == 0 or i == len(questions):
                    print(f"✅ Progreso: {i}/{len(questions)} preguntas ({i*100//len(questions)}%)")
            except Exception as e:
                print(f"❌ Error en pregunta {i}: {e}")
                results.append({
                    **q_data['original_data'],
                    "RAG_Model": "Mistral7B",
                    "RAG_Response": f"ERROR: {str(e)}",
                    "RAG_Quality_Evaluation": "",
                    "RAG_Response_Time_Seconds": 0,
                    "RAG_Iterations": 0,
                    "RAG_Num_Sources": 0,
                    "RAG_Sources": [],
                    "RAG_Timestamp": datetime.now().isoformat()
                })
        
        total_time = time.time() - total_start
        
        print(f"\n{'='*80}")
        print(f"✅ LOTE {batch_num} COMPLETADO")
        print(f"📊 {len(results)} preguntas procesadas")
        print(f"⏱️  Tiempo total: {total_time:.1f}s ({total_time/60:.1f} minutos)")
        print(f"⏱️  Tiempo promedio: {total_time/len(results):.1f}s por pregunta")
        print(f"{'='*80}\n")
        
        client.close()
        return results
    
    def save_batch(self, results: List[Dict], batch_num: int) -> str:
        """Guarda los resultados de un lote"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cybersecurity_qa_batch{batch_num}_{timestamp}.xlsx"
        
        df = pd.DataFrame(results)
        df.to_excel(filename, index=False)
        
        print(f"💾 Lote {batch_num} guardado en: {filename}")
        return filename


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluador por lotes con paralelización")
    parser.add_argument("--batch-size", type=int, default=1000, help="Tamaño del lote")
    parser.add_argument("--start-idx", type=int, default=0, help="Índice inicial")
    parser.add_argument("--workers", type=int, default=1, help="Número de workers (usar 1 para evitar conflictos)")
    parser.add_argument("--num-batches", type=int, default=1, help="Número de lotes a ejecutar")
    
    args = parser.parse_args()
    
    evaluator = BatchEvaluator(max_workers=args.workers)
    
    print("\n" + "="*80)
    print("🚀 EVALUACIÓN POR LOTES CON PARALELIZACIÓN")
    print("="*80)
    print(f"📦 Tamaño del lote: {args.batch_size}")
    print(f"📊 Número de lotes: {args.num_batches}")
    print(f"🔧 Workers paralelos: {args.workers}")
    print("="*80)
    
    for batch_num in range(1, args.num_batches + 1):
        start_idx = args.start_idx + (batch_num - 1) * args.batch_size
        
        # Cargar lote
        questions = evaluator.load_questions(start_idx, args.batch_size)
        
        if not questions:
            print(f"⚠️ No hay más preguntas para el lote {batch_num}")
            break
        
        # Evaluar lote (secuencial por compatibilidad con Weaviate)
        results = evaluator.evaluate_single_threaded(questions, batch_num)
        
        # Guardar lote
        filename = evaluator.save_batch(results, batch_num)
        
        print(f"✅ Lote {batch_num}/{args.num_batches} completado: {filename}\n")
    
    print("🎉 TODOS LOS LOTES COMPLETADOS")


if __name__ == "__main__":
    main()
