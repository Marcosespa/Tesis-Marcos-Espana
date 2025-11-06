#!/usr/bin/env python3
"""
Evaluador por LOTES con paralelización y optimizaciones
Ejecuta lotes de 1000 preguntas con workers paralelos
"""

import sys
import time
import pandas as pd
from pathlib import Path
import json
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
    
    def __init__(self, max_workers: int = 2, 
                 llm_provider: str = "ollama",
                 model: str = "qwen2.5:7b-instruct",
                 azure_endpoint: str = None,
                 azure_deployment: str = None,
                 azure_api_key: str = None,
                 azure_api_version: str = "2024-12-01-preview"):
        self.max_workers = max_workers
        self.llm_provider = llm_provider
        self.model = model
        self.azure_endpoint = azure_endpoint
        self.azure_deployment = azure_deployment
        self.azure_api_key = azure_api_key
        self.azure_api_version = azure_api_version
        print(f"🔧 Máximo de workers: {max_workers}")
        print(f"🤖 LLM Provider: {llm_provider}")
        if llm_provider == "azure_openai":
            print(f"   Azure Deployment: {azure_deployment}")
        else:
            print(f"   Model: {model}")
        
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
        
        # Determinar nombre del modelo para la columna
        model_name = "Qwen2.5-7B-Instruct"
        if self.llm_provider == "azure_openai":
            model_name = f"Azure-{self.azure_deployment}"
        
        try:
            start = time.time()
            result = rag_system.process_query(question, verbose=False)
            elapsed = time.time() - start
            
            return {
                **original_data,
                "RAG_Model": model_name,
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
                "RAG_Model": model_name,
                "RAG_Response": f"ERROR: {str(e)}",
                "RAG_Quality_Evaluation": "",
                "RAG_Response_Time_Seconds": 0,
                "RAG_Iterations": 0,
                "RAG_Num_Sources": 0,
                "RAG_Sources": [],
                "RAG_Timestamp": datetime.now().isoformat(),
                "RAG_Error": str(e)
            }
    
    def evaluate_single_threaded(self, questions: List[Dict], batch_num: int, save_every: int = 0, batch_dir: Path | None = None, resume: bool = False) -> List[Dict]:
        """Evalúa un lote de preguntas sin paralelización (más seguro para Weaviate)

        Args:
            questions: lista de preguntas a evaluar
            batch_num: identificador del lote
            save_every: si >0, guarda un Excel parcial cada N preguntas procesadas
            batch_dir: directorio donde guardar parciales y final
            resume: si True, reanuda desde checkpoint
        """
        print(f"\n{'='*80}")
        print(f"🚀 EJECUTANDO LOTE {batch_num}: {len(questions)} preguntas")
        print(f"{'='*80}\n")
        
        # Preparar directorio de salida (será sobrescrito por batch_dir pasado desde main)
        if batch_dir is None:
            OUTPUT_DIR = Path("results/excel")
            batch_dir = OUTPUT_DIR / f"batch_{batch_num}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = batch_dir / "checkpoint.json"

        # Crear sistema RAG según el provider
        client = weaviate.connect_to_local()
        if self.llm_provider == "azure_openai":
            rag_system = OptimizedCrewAIRAG(
                weaviate_client=client,
                llm_provider="azure_openai",
                azure_endpoint=self.azure_endpoint,
                azure_deployment=self.azure_deployment,
                azure_api_key=self.azure_api_key,
                azure_api_version=self.azure_api_version
            )
        else:
            rag_system = OptimizedCrewAIRAG(
                weaviate_client=client,
                model=self.model,
                llm_provider=self.llm_provider
            )
        
        results: List[Dict] = []
        total_start = time.time()
        total_processed = 0  # Conteo real procesado en esta ejecución
        
        # Reanudar si aplica
        processed_count = 0
        part_count = 0
        if resume and checkpoint_path.exists():
            try:
                ck = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                processed_count = int(ck.get("processed_count", 0))
                part_count = int(ck.get("part_count", 0))
                print(f"♻️  Reanudando desde checkpoint: procesadas={processed_count}, partes={part_count}")
            except Exception:
                print("⚠️ Checkpoint corrupto, iniciando desde cero")
        
        # Procesar secuencialmente
        for i, q_data in enumerate(questions, 1):
            # Saltar ya procesadas en reanudación
            if resume and i <= processed_count:
                if i % 50 == 0:
                    print(f"⏭️  Saltando {i}/{len(questions)} (reanudación)")
                continue
            try:
                result = self.evaluate_single(q_data, rag_system)
                results.append(result)
                total_processed += 1
                
                if i % 10 == 0 or i == len(questions):
                    print(f"✅ Progreso: {i}/{len(questions)} preguntas ({i*100//len(questions)}%)")

                # Guardado incremental si corresponde
                if save_every and (i % save_every == 0 or i == len(questions)):
                    try:
                        part_count += 1
                        partial_suffix = f"partial_{i:05d}"
                        self.save_batch(results, batch_num, suffix=partial_suffix, out_dir=batch_dir)
                        # Limpiar buffer en memoria tras guardar parcial para reducir RAM
                        results.clear()
                        # Escribir checkpoint
                        checkpoint_path.write_text(json.dumps({
                            "processed_count": i,
                            "part_count": part_count,
                            "timestamp": datetime.now().isoformat()
                        }, ensure_ascii=False), encoding="utf-8")
                        print(f"💾 Guardado incremental: {i}/{len(questions)}")
                    except Exception as e:
                        print(f"⚠️ Error en guardado incremental: {e}")
            except Exception as e:
                print(f"❌ Error en pregunta {i}: {e}")
                # Determinar nombre del modelo para la columna
                model_name = "Qwen2.5-7B-Instruct"
                if self.llm_provider == "azure_openai":
                    model_name = f"Azure-{self.azure_deployment}"
                results.append({
                    **q_data['original_data'],
                    "RAG_Model": model_name,
                    "RAG_Response": f"ERROR: {str(e)}",
                    "RAG_Quality_Evaluation": "",
                    "RAG_Response_Time_Seconds": 0,
                    "RAG_Iterations": 0,
                    "RAG_Num_Sources": 0,
                    "RAG_Sources": [],
                    "RAG_Timestamp": datetime.now().isoformat()
                })
                total_processed += 1
                # Escribir checkpoint ante error
                checkpoint_path.write_text(json.dumps({
                    "processed_count": i,
                    "part_count": part_count,
                    "timestamp": datetime.now().isoformat()
                }, ensure_ascii=False), encoding="utf-8")
        
        total_time = time.time() - total_start
        
        print(f"\n{'='*80}")
        print(f"✅ LOTE {batch_num} COMPLETADO")
        print(f"📊 {total_processed} preguntas procesadas")
        print(f"⏱️  Tiempo total: {total_time:.1f}s ({total_time/60:.1f} minutos)")
        avg_time = (total_time/total_processed) if total_processed else 0
        print(f"⏱️  Tiempo promedio: {avg_time:.1f}s por pregunta")
        print(f"{'='*80}\n")
        
        client.close()
        return results
    
    def save_batch(self, results: List[Dict], batch_num: int, suffix: str = "", out_dir: Path | None = None) -> str:
        """Guarda los resultados de un lote.

        Args:
            results: lista de resultados a guardar
            batch_num: número de lote
            suffix: sufijo opcional para guardados parciales (p. ej. "partial_250")
            out_dir: directorio de salida
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        OUTPUT_DIR = Path("results/excel")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        batch_dir = out_dir or (OUTPUT_DIR / f"batch_{batch_num}")
        batch_dir.mkdir(parents=True, exist_ok=True)
        base = batch_dir / f"cybersecurity_qa_batch{batch_num}"
        if suffix:
            filename = f"{base}_{suffix}_{timestamp}.xlsx"
        else:
            filename = f"{base}_{timestamp}.xlsx"
        
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
    parser.add_argument("--resume", action="store_true", help="Reanuda desde checkpoint del lote")
    parser.add_argument("--save-every", type=int, default=25, help="Guardar Excel parcial cada N preguntas (0 para desactivar)")
    parser.add_argument("--output-dir", type=str, default="results/excel", help="Directorio donde guardar los Excel")
    
    # LLM Provider arguments
    parser.add_argument("--llm-provider", choices=["ollama", "azure_openai"], default="ollama", 
                       help="LLM provider: ollama or azure_openai")
    parser.add_argument("--model", type=str, default="qwen2.5:7b-instruct", 
                       help="Model name (for ollama)")
    
    # Azure OpenAI arguments
    parser.add_argument("--azure-endpoint", type=str, default=None, 
                       help="Azure OpenAI endpoint URL")
    parser.add_argument("--azure-deployment", type=str, default="gpt-4.1", 
                       help="Azure OpenAI deployment name (default: gpt-4.1)")
    parser.add_argument("--azure-api-key", type=str, default=None, 
                       help="Azure OpenAI API key (or set AZURE_OPENAI_API_KEY env var)")
    parser.add_argument("--azure-api-version", type=str, default="2024-12-01-preview", 
                       help="Azure OpenAI API version")
    
    args = parser.parse_args()
    
    # Obtener API key de Azure desde argumentos o variable de entorno
    azure_api_key = args.azure_api_key
    if not azure_api_key and args.llm_provider == "azure_openai":
        import os
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not azure_api_key:
            print("❌ ERROR: --azure-api-key es requerido o establece AZURE_OPENAI_API_KEY")
            return 1
    
    # Validar parámetros de Azure si se usa azure_openai
    if args.llm_provider == "azure_openai":
        if not all([args.azure_endpoint, args.azure_deployment, azure_api_key]):
            print("❌ ERROR: --azure-endpoint, --azure-deployment y --azure-api-key son requeridos para azure_openai")
            return 1
    
    evaluator = BatchEvaluator(
        max_workers=args.workers,
        llm_provider=args.llm_provider,
        model=args.model,
        azure_endpoint=args.azure_endpoint,
        azure_deployment=args.azure_deployment,
        azure_api_key=azure_api_key,
        azure_api_version=args.azure_api_version
    )
    
    print("\n" + "="*80)
    print("🚀 EVALUACIÓN POR LOTES CON PARALELIZACIÓN")
    print("="*80)
    print(f"📦 Tamaño del lote: {args.batch_size}")
    print(f"📊 Número de lotes: {args.num_batches}")
    print(f"🔧 Workers paralelos: {args.workers}")
    print("="*80)
    
    for batch_num in range(1, args.num_batches + 1):
        start_idx = args.start_idx + (batch_num - 1) * args.batch_size
        end_idx = start_idx + args.batch_size - 1
        
        # Cargar lote
        questions = evaluator.load_questions(start_idx, args.batch_size)
        
        if not questions:
            print(f"⚠️ No hay más preguntas para el lote {batch_num}")
            break
        
        # Directorio del lote
        OUTPUT_DIR = Path(args.output_dir)
        batch_dir = OUTPUT_DIR / f"batch_{batch_num}_{start_idx}_{end_idx}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        # Evaluar lote (secuencial por compatibilidad con Weaviate)
        results = evaluator.evaluate_single_threaded(questions, batch_num, save_every=args.save_every, batch_dir=batch_dir, resume=args.resume)
        
        # Guardar lote completo consolidando parciales previos
        try:
            # Cargar parciales existentes en el directorio y combinarlos
            partial_files = sorted(batch_dir.glob(f"cybersecurity_qa_batch{batch_num}_partial_*.xlsx"))
            frames = []
            for pf in partial_files:
                try:
                    frames.append(pd.read_excel(pf))
                except Exception as e:
                    print(f"⚠️ No se pudo leer parcial {pf}: {e}")
            # Agregar los resultados de esta ejecución que no hayan sido vaciados
            if results:
                frames.append(pd.DataFrame(results))
            if frames:
                final_df = pd.concat(frames, ignore_index=True)
            else:
                final_df = pd.DataFrame()
            final_name = evaluator.save_batch(final_df.to_dict(orient='records'), batch_num, out_dir=batch_dir)
        except Exception as e:
            print(f"⚠️ Error consolidando finales del lote {batch_num}: {e}")
            final_name = evaluator.save_batch(results, batch_num, out_dir=batch_dir)
        
        print(f"✅ Lote {batch_num}/{args.num_batches} completado: {final_name}\n")
    
    print("🎉 TODOS LOS LOTES COMPLETADOS")


if __name__ == "__main__":
    main()
