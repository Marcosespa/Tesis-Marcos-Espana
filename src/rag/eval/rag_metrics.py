#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Métricas de evaluación para RAG:
- Precision@k: Precisión en los primeros k chunks recuperados
- Coverage: Cuántas respuestas necesitan cruzar múltiples chunks
- Context window utilization: Uso eficiente del contexto
- Latency/Cost: Tiempo y costo de procesamiento
"""

import time
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from collections import defaultdict


@dataclass
class RetrievalResult:
    """Resultado de una consulta de retrieval"""
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    ground_truth_chunks: List[str]  # IDs de chunks relevantes
    retrieval_time: float
    embedding_time: float
    search_time: float
    total_chunks_available: int


@dataclass
class RAGMetrics:
    """Métricas de evaluación RAG"""
    precision_at_k: Dict[int, float]  # k -> precision
    recall_at_k: Dict[int, float]     # k -> recall
    f1_at_k: Dict[int, float]         # k -> f1
    coverage: float                   # % de respuestas que cruzan chunks
    context_utilization: float        # % de contexto usado efectivamente
    avg_retrieval_time: float         # Tiempo promedio de retrieval
    avg_embedding_time: float         # Tiempo promedio de embedding
    total_cost: float                 # Costo total estimado


class RAGEvaluator:
    """Evaluador principal para métricas RAG"""
    
    def __init__(self, retriever, embedder=None):
        self.retriever = retriever
        self.embedder = embedder
        self.results: List[RetrievalResult] = []
    
    def evaluate_query(self, query: str, ground_truth: List[str], 
                      k_values: List[int] = [1, 3, 5, 10]) -> RetrievalResult:
        """Evalúa una consulta individual"""
        
        start_time = time.time()
        
        # Embedding time
        embedding_start = time.time()
        if self.embedder:
            query_embedding = self.embedder.embed_query(query)
        else:
            query_embedding = None
        embedding_time = time.time() - embedding_start
        
        # Search time
        search_start = time.time()
        retrieved_chunks = self.retriever.search(query, k=max(k_values), embedding=query_embedding)
        search_time = time.time() - search_start
        
        total_time = time.time() - start_time
        
        return RetrievalResult(
            query=query,
            retrieved_chunks=retrieved_chunks,
            ground_truth_chunks=ground_truth,
            retrieval_time=total_time,
            embedding_time=embedding_time,
            search_time=search_time,
            total_chunks_available=len(self.retriever.get_all_chunks())
        )
    
    def calculate_precision_at_k(self, result: RetrievalResult, k: int) -> float:
        """Calcula Precision@k"""
        if k == 0:
            return 0.0
        
        retrieved_ids = [chunk['id'] for chunk in result.retrieved_chunks[:k]]
        ground_truth_set = set(result.ground_truth_chunks)
        
        if not ground_truth_set:
            return 0.0
        
        relevant_retrieved = sum(1 for chunk_id in retrieved_ids if chunk_id in ground_truth_set)
        return relevant_retrieved / k
    
    def calculate_recall_at_k(self, result: RetrievalResult, k: int) -> float:
        """Calcula Recall@k"""
        retrieved_ids = [chunk['id'] for chunk in result.retrieved_chunks[:k]]
        ground_truth_set = set(result.ground_truth_chunks)
        
        if not ground_truth_set:
            return 0.0
        
        relevant_retrieved = sum(1 for chunk_id in retrieved_ids if chunk_id in ground_truth_set)
        return relevant_retrieved / len(ground_truth_set)
    
    def calculate_coverage(self, results: List[RetrievalResult]) -> float:
        """Calcula el porcentaje de respuestas que necesitan cruzar múltiples chunks"""
        multi_chunk_queries = 0
        total_queries = len(results)
        
        for result in results:
            # Si necesitamos más de 1 chunk para responder completamente
            if len(result.ground_truth_chunks) > 1:
                multi_chunk_queries += 1
        
        return multi_chunk_queries / total_queries if total_queries > 0 else 0.0
    
    def calculate_context_utilization(self, results: List[RetrievalResult], 
                                    context_window_size: int = 4096) -> float:
        """Calcula qué tan bien se utiliza el contexto disponible"""
        total_utilization = 0.0
        total_queries = len(results)
        
        for result in results:
            # Suma la longitud de todos los chunks recuperados
            total_chunk_length = sum(
                len(chunk.get('text', '')) for chunk in result.retrieved_chunks
            )
            
            # Calcula el % de utilización del contexto
            utilization = min(total_chunk_length / context_window_size, 1.0)
            total_utilization += utilization
        
        return total_utilization / total_queries if total_queries > 0 else 0.0
    
    def calculate_latency_metrics(self, results: List[RetrievalResult]) -> Dict[str, float]:
        """Calcula métricas de latencia"""
        if not results:
            return {"avg_retrieval_time": 0.0, "avg_embedding_time": 0.0, "avg_search_time": 0.0}
        
        return {
            "avg_retrieval_time": np.mean([r.retrieval_time for r in results]),
            "avg_embedding_time": np.mean([r.embedding_time for r in results]),
            "avg_search_time": np.mean([r.search_time for r in results]),
            "p95_retrieval_time": np.percentile([r.retrieval_time for r in results], 95),
            "p99_retrieval_time": np.percentile([r.retrieval_time for r in results], 99)
        }
    
    def estimate_cost(self, results: List[RetrievalResult], 
                     embedding_cost_per_1k: float = 0.0001,
                     retrieval_cost_per_query: float = 0.001) -> float:
        """Estima el costo total del sistema"""
        total_cost = 0.0
        
        for result in results:
            # Costo de embedding (si se usa)
            if result.embedding_time > 0:
                # Estimación basada en tokens (asumiendo ~4 chars por token)
                query_tokens = len(result.query) / 4
                embedding_cost = (query_tokens / 1000) * embedding_cost_per_1k
                total_cost += embedding_cost
            
            # Costo de retrieval
            total_cost += retrieval_cost_per_query
        
        return total_cost
    
    def evaluate(self, test_queries: List[Tuple[str, List[str]]], 
                k_values: List[int] = [1, 3, 5, 10]) -> RAGMetrics:
        """Evalúa el sistema RAG completo"""
        
        results = []
        
        # Procesa cada consulta
        for query, ground_truth in test_queries:
            result = self.evaluate_query(query, ground_truth, k_values)
            results.append(result)
        
        self.results = results
        
        # Calcula métricas
        precision_at_k = {}
        recall_at_k = {}
        f1_at_k = {}
        
        for k in k_values:
            precisions = [self.calculate_precision_at_k(r, k) for r in results]
            recalls = [self.calculate_recall_at_k(r, k) for r in results]
            
            precision_at_k[k] = np.mean(precisions)
            recall_at_k[k] = np.mean(recalls)
            f1_at_k[k] = 2 * (precision_at_k[k] * recall_at_k[k]) / (precision_at_k[k] + recall_at_k[k]) if (precision_at_k[k] + recall_at_k[k]) > 0 else 0.0
        
        coverage = self.calculate_coverage(results)
        context_utilization = self.calculate_context_utilization(results)
        latency_metrics = self.calculate_latency_metrics(results)
        total_cost = self.estimate_cost(results)
        
        return RAGMetrics(
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            f1_at_k=f1_at_k,
            coverage=coverage,
            context_utilization=context_utilization,
            avg_retrieval_time=latency_metrics["avg_retrieval_time"],
            avg_embedding_time=latency_metrics["avg_embedding_time"],
            total_cost=total_cost
        )
    
    def save_results(self, output_path: Path):
        """Guarda los resultados de evaluación"""
        results_data = []
        
        for result in self.results:
            results_data.append({
                "query": result.query,
                "retrieved_chunk_ids": [chunk['id'] for chunk in result.retrieved_chunks],
                "ground_truth_chunk_ids": result.ground_truth_chunks,
                "retrieval_time": result.retrieval_time,
                "embedding_time": result.embedding_time,
                "search_time": result.search_time,
                "total_chunks_available": result.total_chunks_available
            })
        
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    def print_metrics(self, metrics: RAGMetrics):
        """Imprime las métricas de forma legible"""
        print("=" * 60)
        print("MÉTRICAS DE EVALUACIÓN RAG")
        print("=" * 60)
        
        print(f"\n📊 PRECISIÓN Y RECALL:")
        for k in sorted(metrics.precision_at_k.keys()):
            print(f"  Precision@{k}: {metrics.precision_at_k[k]:.3f}")
            print(f"  Recall@{k}:    {metrics.recall_at_k[k]:.3f}")
            print(f"  F1@{k}:        {metrics.f1_at_k[k]:.3f}")
        
        print(f"\n📈 COBERTURA Y UTILIZACIÓN:")
        print(f"  Coverage:           {metrics.coverage:.3f}")
        print(f"  Context Utilization: {metrics.context_utilization:.3f}")
        
        print(f"\n⏱️  LATENCIA:")
        print(f"  Avg Retrieval Time: {metrics.avg_retrieval_time:.3f}s")
        print(f"  Avg Embedding Time: {metrics.avg_embedding_time:.3f}s")
        
        print(f"\n💰 COSTO:")
        print(f"  Total Cost: ${metrics.total_cost:.4f}")
        
        print("=" * 60)
