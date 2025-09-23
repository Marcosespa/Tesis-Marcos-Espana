#!/usr/bin/env python3
"""
Sistema mejorado de búsqueda semántica con correcciones para escalas y re-ranking
"""

import argparse
import sys
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import re

# Support running as a script regardless of PYTHONPATH
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import weaviate
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class SearchStrategy(Enum):
    SEMANTIC_ONLY = "semantic"
    HYBRID = "hybrid"
    MULTI_STAGE = "multi_stage"


@dataclass
class SearchResult:
    """Clase para representar un resultado de búsqueda enriquecido"""
    doc_id: str
    title: str
    content: str
    level: str
    page_start: int
    page_end: int
    semantic_score: float
    keyword_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    highlighted_text: str = ""
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para serialización"""
        return {
            'doc_id': self.doc_id,
            'title': self.title,
            'content': self.content[:500] + "..." if len(self.content) > 500 else self.content,
            'level': self.level,
            'pages': f"{self.page_start}-{self.page_end}",
            'scores': {
                'semantic': round(self.semantic_score, 4),
                'keyword': round(self.keyword_score, 4),
                'rerank': round(self.rerank_score, 4),
                'final': round(self.final_score, 4)
            },
            'highlighted_text': self.highlighted_text
        }


class ImprovedSemanticSearch:
    """Sistema mejorado de búsqueda semántica con escalas normalizadas"""
    
    def __init__(
        self,
        client: weaviate.WeaviateClient,
        collection_name: str = "BookChunk",
        embedding_model: str = "all-MiniLM-L6-v2",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-2-v2",  # Modelo original
        use_cache: bool = True
    ):
        self.client = client
        self.collection_name = collection_name
        self.collection = client.collections.get(collection_name)
        
        # Modelos
        self.embedding_model = SentenceTransformer(embedding_model)
        self.rerank_model = CrossEncoder(rerank_model)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        # Cache y configuración
        self.use_cache = use_cache
        self.query_cache = {}
        self.tfidf_fitted = False
        
        # Logger
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Configurar logger"""
        logger = logging.getLogger('ImprovedSemanticSearch')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _normalize_query(self, query: str) -> str:
        """Normalizar y limpiar la consulta"""
        # Remover caracteres especiales pero mantener espacios
        query = re.sub(r'[^\w\s]', ' ', query.lower())
        # Remover espacios extra
        query = ' '.join(query.split())
        return query
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extraer keywords importantes de la consulta"""
        # Términos técnicos comunes en ciberseguridad
        cybersec_terms = {
            'malware', 'ransomware', 'phishing', 'firewall', 'encryption',
            'vulnerability', 'exploit', 'authentication', 'authorization',
            'ssl', 'tls', 'vpn', 'ids', 'ips', 'siem', 'incident', 'response'
        }
        
        words = query.lower().split()
        keywords = []
        
        # Priorizar términos técnicos
        for word in words:
            if len(word) > 2:  # Evitar palabras muy cortas
                if word in cybersec_terms:
                    keywords.insert(0, word)  # Términos técnicos al inicio
                else:
                    keywords.append(word)
                    
        return keywords[:10]  # Limitar a 10 keywords
    
    def _semantic_search(
        self, 
        query: str, 
        k: int = 20
    ) -> List[SearchResult]:
        """Búsqueda semántica base con scores normalizados"""
        self.logger.info(f"Realizando búsqueda semántica para: '{query}'")
        
        # Generar embedding
        query_embedding = self.embedding_model.encode(
            [query], 
            convert_to_numpy=True, 
            normalize_embeddings=True
        )[0]
        
        # Búsqueda en Weaviate
        results = self.collection.query.near_vector(
            near_vector=query_embedding.astype(np.float32),
            limit=k,
            return_metadata=["distance", "score"]
        )
        
        # Convertir a SearchResult con scores normalizados
        search_results = []
        for result in results.objects:
            props = result.properties
            metadata = result.metadata
            
            # Normalizar score semántico a 0-1 usando cosine similarity
            semantic_score = 1 - metadata.distance  # Ya está normalizado
            
            search_result = SearchResult(
                doc_id=props.get('docId', 'unknown'),
                title=props.get('title', 'Sin título'),
                content=props.get('text', ''),
                level=props.get('level', 'unknown'),
                page_start=props.get('pageStart', 0),
                page_end=props.get('pageEnd', 0),
                semantic_score=semantic_score,
                metadata=props
            )
            search_results.append(search_result)
            
        return search_results
    
    def _keyword_search_manual(self, query: str, k: int = 20) -> List[SearchResult]:
        """Búsqueda por palabras clave manual usando BM25 aproximado"""
        keywords = self._extract_keywords(query)
        keyword_query = ' '.join(keywords)
        
        # Usar búsqueda semántica como base y luego filtrar por keywords
        semantic_results = self._semantic_search(query, k * 2)
        
        # Calcular scores de keywords manualmente
        for result in semantic_results:
            content_lower = result.content.lower()
            title_lower = result.title.lower()
            
            # Contar ocurrencias de keywords
            keyword_matches = 0
            total_keywords = len(keywords)
            
            for keyword in keywords:
                if keyword in content_lower:
                    keyword_matches += content_lower.count(keyword)
                if keyword in title_lower:
                    keyword_matches += title_lower.count(keyword) * 2  # Peso extra para título
            
            # Normalizar score de keywords (0-1)
            result.keyword_score = min(keyword_matches / (total_keywords * 3), 1.0)
        
        # Ordenar por score de keywords
        semantic_results.sort(key=lambda x: x.keyword_score, reverse=True)
        return semantic_results[:k]
    
    def _rerank_results(
        self, 
        query: str, 
        results: List[SearchResult], 
        top_k: int = 10
    ) -> List[SearchResult]:
        """Re-ranking usando modelo cross-encoder con scores normalizados"""
        if not results:
            return results
            
        self.logger.info(f"Re-ranking {len(results)} resultados")
        
        # Preparar pares query-documento para re-ranking
        query_doc_pairs = []
        for result in results:
            # Usar título + inicio del contenido para re-ranking
            doc_text = f"{result.title} {result.content[:500]}"
            query_doc_pairs.append([query, doc_text])
        
        # Obtener scores de re-ranking
        rerank_scores = self.rerank_model.predict(query_doc_pairs)
        
        # Normalizar scores de re-ranking a 0-1 usando softmax
        rerank_scores = np.array(rerank_scores)
        exp_scores = np.exp(rerank_scores - np.max(rerank_scores))  # Para estabilidad numérica
        normalized_scores = exp_scores / np.sum(exp_scores)
        
        # Actualizar resultados con scores normalizados
        for result, score in zip(results, normalized_scores):
            result.rerank_score = float(score)
        
        # Ordenar por score de re-ranking y tomar top_k
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        return results[:top_k]
    
    def _highlight_text(self, text: str, query: str, max_length: int = 300) -> str:
        """Resaltar términos relevantes en el texto"""
        keywords = self._extract_keywords(query)
        
        # Encontrar la mejor ventana de texto
        best_window = text[:max_length]
        best_score = 0
        
        # Buscar ventanas con más keywords
        words = text.lower().split()
        window_size = min(50, len(words))  # Ventana de ~50 palabras
        
        for i in range(len(words) - window_size + 1):
            window_text = ' '.join(words[i:i + window_size])
            score = sum(1 for keyword in keywords if keyword in window_text)
            
            if score > best_score:
                best_score = score
                # Reconstruir con capitalización original
                start_char = text.lower().find(window_text)
                if start_char != -1:
                    end_char = start_char + len(window_text)
                    best_window = text[start_char:end_char]
        
        # Resaltar keywords (simulado con MAYÚSCULAS para terminal)
        highlighted = best_window
        for keyword in keywords:
            highlighted = re.sub(
                rf'\b{re.escape(keyword)}\b', 
                keyword.upper(), 
                highlighted, 
                flags=re.IGNORECASE
            )
        
        return highlighted
    
    def _merge_and_deduplicate(
        self, 
        semantic_results: List[SearchResult], 
        keyword_results: List[SearchResult],
        alpha: float = 0.7
    ) -> List[SearchResult]:
        """Fusionar y deduplicar resultados con scores normalizados"""
        
        # Crear diccionario de resultados por doc_id
        merged_results = {}
        
        # Agregar resultados semánticos
        for result in semantic_results:
            merged_results[result.doc_id] = result
        
        # Fusionar con resultados de keywords
        for result in keyword_results:
            if result.doc_id in merged_results:
                # Fusionar scores
                existing = merged_results[result.doc_id]
                existing.keyword_score = result.keyword_score
            else:
                merged_results[result.doc_id] = result
        
        # Calcular score final combinado (ambos scores ya están en 0-1)
        for result in merged_results.values():
            result.final_score = (
                alpha * result.semantic_score + 
                (1 - alpha) * result.keyword_score
            )
        
        # Convertir a lista y ordenar
        final_results = list(merged_results.values())
        final_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return final_results
    
    def search(
        self,
        query: str,
        strategy: SearchStrategy = SearchStrategy.MULTI_STAGE,
        k: int = 5,
        rerank: bool = True
    ) -> List[SearchResult]:
        """Búsqueda principal con múltiples estrategias mejoradas"""
        
        # Verificar cache
        cache_key = f"{query}_{strategy.value}_{k}_{rerank}"
        if self.use_cache and cache_key in self.query_cache:
            self.logger.info("Resultado encontrado en cache")
            return self.query_cache[cache_key]
        
        start_time = time.time()
        normalized_query = self._normalize_query(query)
        
        if strategy == SearchStrategy.SEMANTIC_ONLY:
            results = self._semantic_search(normalized_query, k * 2)
            
        elif strategy == SearchStrategy.HYBRID:
            semantic_results = self._semantic_search(normalized_query, k * 2)
            keyword_results = self._keyword_search_manual(normalized_query, k * 2)
            results = self._merge_and_deduplicate(semantic_results, keyword_results)
            
        elif strategy == SearchStrategy.MULTI_STAGE:
            # Etapa 1: Recuperación amplia
            semantic_results = self._semantic_search(normalized_query, k * 4)
            keyword_results = self._keyword_search_manual(normalized_query, k * 2)
            
            # Etapa 2: Fusión
            merged_results = self._merge_and_deduplicate(
                semantic_results, keyword_results, alpha=0.6
            )
            
            # Etapa 3: Re-ranking con más resultados
            if rerank and len(merged_results) > k:
                results = self._rerank_results(normalized_query, merged_results[:k * 3], k * 2)
            else:
                results = merged_results[:k * 2]
        
        # Tomar top-k final
        final_results = results[:k]
        
        # Agregar texto resaltado
        for result in final_results:
            result.highlighted_text = self._highlight_text(result.content, normalized_query)
        
        # Guardar en cache
        if self.use_cache:
            self.query_cache[cache_key] = final_results
        
        search_time = time.time() - start_time
        self.logger.info(f"Búsqueda completada en {search_time:.2f}s")
        
        return final_results


def format_results_advanced(results: List[SearchResult], query: str) -> None:
    """Formatear y mostrar resultados de búsqueda avanzados con explicaciones"""
    
    if not results:
        print("❌ No se encontraron resultados.")
        return
    
    print(f"\n{'='*100}")
    print(f"BUSQUEDA: '{query}'")
    print(f"RESULTADOS: {len(results)} documentos encontrados")
    print(f"{'='*100}")
    
    # Mostrar estadísticas de scores
    semantic_scores = [max(0.0, min(1.0, r.semantic_score)) for r in results if r.semantic_score > 0]
    rerank_scores = [max(0.0, min(1.0, r.rerank_score)) for r in results if r.rerank_score > 0]
    keyword_scores = [max(0.0, min(1.0, r.keyword_score)) for r in results if r.keyword_score > 0]
    final_scores = [max(0.0, min(1.0, r.final_score)) for r in results if r.final_score > 0]
    
    if semantic_scores:
        print(f"Scores Semánticos: avg={np.mean(semantic_scores):.3f}, max={max(semantic_scores):.3f}")
    if keyword_scores:
        print(f"Scores Keywords: avg={np.mean(keyword_scores):.3f}, max={max(keyword_scores):.3f}")
    if rerank_scores:
        print(f"Scores Re-rank: avg={np.mean(rerank_scores):.3f}, max={max(rerank_scores):.3f}")
    if final_scores:
        print(f"Scores Finales: avg={np.mean(final_scores):.3f}, max={max(final_scores):.3f}")
    
    print()
    
    for i, result in enumerate(results, 1):
        print(f"\n{'-'*20} RESULTADO {i} {'-'*20}")
        
        # Información básica
        print(f" Título: {result.title}")
        print(f" Documento: {result.doc_id}")
        print(f" Nivel: {result.level}")
        print(f" Páginas: {result.page_start}-{result.page_end}")
        
        # Scores con explicación
        scores = []
        if result.semantic_score > 0:
            scores.append(f"Semántico: {result.semantic_score:.3f}")
        if result.keyword_score > 0:
            scores.append(f"Keywords: {result.keyword_score:.3f}")
        if result.rerank_score > 0:
            scores.append(f"Re-rank: {result.rerank_score:.3f}")
        if result.final_score > 0:
            scores.append(f"Final: {result.final_score:.3f}")
            
        print(f"Puntuaciones: {' | '.join(scores)}")
        
        # Explicación del ranking (umbral para evitar ruido)
        delta_threshold = 0.02
        if result.rerank_score > 0 and result.semantic_score > 0:
            if (result.rerank_score - result.semantic_score) > delta_threshold:
                print("   Re-ranking mejoró la posición (cross-encoder: más relevante)")
            elif (result.semantic_score - result.rerank_score) > delta_threshold:
                print("   Re-ranking penalizó la posición (cross-encoder: menos relevante)")
            else:
                print("   Re-ranking mantuvo la posición")
        
        # Contenido resaltado
        print(f"\nContenido relevante:")
        print(f"   {result.highlighted_text}")
        
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Sistema mejorado de búsqueda semántica con escalas normalizadas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python improved_search.py "authentication" --strategy multi_stage --k 10
  python improved_search.py "vulnerabilidades SQL" --strategy hybrid --k 5
  python improved_search.py "técnicas de phishing" --interactive
        """
    )
    
    parser.add_argument("query", nargs='?', help="Consulta de búsqueda")
    parser.add_argument("--host", default="localhost", help="Host de Weaviate")
    parser.add_argument("--http_port", type=int, default=8080, help="Puerto HTTP")
    parser.add_argument("--grpc_port", type=int, default=50051, help="Puerto gRPC")
    parser.add_argument("--collection", default="BookChunk", help="Colección")
    parser.add_argument("--k", type=int, default=5, help="Número de resultados")
    parser.add_argument(
        "--strategy", 
        choices=["semantic", "hybrid", "multi_stage"],
        default="multi_stage",
        help="Estrategia de búsqueda"
    )
    parser.add_argument("--no-rerank", action="store_true", help="Desactivar re-ranking")
    parser.add_argument("--interactive", action="store_true", help="Modo interactivo")
    parser.add_argument("--output", help="Archivo de salida JSON")
    parser.add_argument("--verbose", action="store_true", help="Modo verbose")
    
    args = parser.parse_args()
    
    # Configurar logging
    if args.verbose:
        logging.getLogger('ImprovedSemanticSearch').setLevel(logging.DEBUG)
    
    # Validar argumentos
    if not args.interactive and (args.query is None or args.query.strip() == ""):
        parser.error("Debes proporcionar una consulta o usar el modo interactivo")
    
    try:
        print("🔌 Conectando a Weaviate...")
        client = weaviate.connect_to_local(
            host=args.host, 
            port=args.http_port, 
            grpc_port=args.grpc_port
        )
        print("✅ Conectado exitosamente!")
        
        # Inicializar sistema de búsqueda mejorado
        search_system = ImprovedSemanticSearch(client, args.collection)
        strategy = SearchStrategy(args.strategy)
        
        if args.interactive:
            print("\n🎯 Modo interactivo activado.")
            print("   Estrategias: semantic, hybrid, multi_stage")
            print("   Comandos: 'quit', 'strategy <nombre>', 'k <número>'")
            print("   Escribe 'help' para más opciones\n")
            
            current_k = args.k
            current_strategy = strategy
            current_rerank = not args.no_rerank
            
            while True:
                try:
                    user_input = input("🔍 Consulta: ").strip()
                    
                    if user_input.lower() in ['quit', 'exit', 'salir', 'q']:
                        break
                    
                    if user_input.lower() in ['help', 'h']:
                        print("""
Comandos disponibles:
  strategy <semantic|hybrid|multi_stage> - Cambiar estrategia
  k <número>                            - Cambiar número de resultados  
  rerank <on|off>                       - Activar/desactivar re-ranking
  clear                                 - Limpiar pantalla
  stats                                 - Mostrar estadísticas
  quit/exit/q                          - Salir
                        """)
                        continue
                    
                    if user_input.startswith('strategy '):
                        new_strategy = user_input.split(' ', 1)[1]
                        try:
                            current_strategy = SearchStrategy(new_strategy)
                            print(f" Estrategia cambiada a: {new_strategy}")
                        except ValueError:
                            print(" Estrategia inválida. Usa: semantic, hybrid, multi_stage")
                        continue
                    
                    if user_input.startswith('k '):
                        try:
                            current_k = int(user_input.split(' ', 1)[1])
                            print(f" Número de resultados cambiado a: {current_k}")
                        except ValueError:
                            print(" Número inválido")
                        continue
                    
                    if user_input.startswith('rerank '):
                        rerank_setting = user_input.split(' ', 1)[1].lower()
                        if rerank_setting in ['on', 'true', '1']:
                            current_rerank = True
                            print("Re-ranking activado")
                        elif rerank_setting in ['off', 'false', '0']:
                            current_rerank = False
                            print(" Re-ranking desactivado")
                        continue
                    
                    if user_input == 'clear':
                        print("\033[2J\033[H")  # Clear screen
                        continue
                    
                    if user_input == 'stats':
                        cache_size = len(search_system.query_cache)
                        print(f"  Cache: {cache_size} consultas")
                        print(f"  Estrategia actual: {current_strategy.value}")
                        print(f"  K actual: {current_k}")
                        print(f"  Re-ranking: {'Activado' if current_rerank else 'Desactivado'}")
                        continue
                    
                    if not user_input:
                        continue
                    
                    # Realizar búsqueda
                    results = search_system.search(
                        user_input, 
                        strategy=current_strategy, 
                        k=current_k,
                        rerank=current_rerank
                    )
                    format_results_advanced(results, user_input)
                    
                except KeyboardInterrupt:
                    print("\n Saliendo...")
                    break
                except Exception as e:
                    print(f" Error: {e}")
        
        else:
            # Búsqueda única
            results = search_system.search(
                args.query,
                strategy=strategy,
                k=args.k,
                rerank=not args.no_rerank
            )
            
            format_results_advanced(results, args.query)
            
            # Guardar en archivo si se especifica
            if args.output:
                output_data = {
                    'query': args.query,
                    'strategy': args.strategy,
                    'timestamp': time.time(),
                    'results': [result.to_dict() for result in results]
                }
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                print(f"  Resultados guardados en: {args.output}")
        
    except Exception as e:
        print(f" Error: {e}")
        return 1
    
    finally:
        try:
            client.close()
        except:
            pass
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
