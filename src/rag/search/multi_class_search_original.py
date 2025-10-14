#!/usr/bin/env python3
"""
Multi-Class Semantic Search System

This script provides agentic search capabilities across multiple Weaviate collections.
Each agent can search specific collections based on their domain expertise.
"""

import argparse
import sys
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

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


class AgentType(Enum):
    """Types of agents for specialized search."""
    POLICY = "policy"          # NIST policies and standards
    RESEARCH = "research"      # USENIX research papers
    ATTACK = "attack"          # MITRE ATT&CK framework
    SECURITY = "security"      # OWASP and security tools
    AI = "ai"                  # AI security knowledge
    TRAINING = "training"      # Training resources
    GENERAL = "general"        # All sources


@dataclass
class SearchResult:
    """Enhanced search result with collection information."""
    doc_id: str
    title: str
    content: str
    level: str
    page_start: int
    page_end: int
    collection: str
    semantic_score: float
    keyword_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    highlighted_text: str = ""
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'doc_id': self.doc_id,
            'title': self.title,
            'content': self.content[:500] + "..." if len(self.content) > 500 else self.content,
            'level': self.level,
            'pages': f"{self.page_start}-{self.page_end}",
            'collection': self.collection,
            'scores': {
                'semantic': round(self.semantic_score, 4),
                'keyword': round(self.keyword_score, 4),
                'rerank': round(self.rerank_score, 4),
                'final': round(self.final_score, 4)
            },
            'highlighted_text': self.highlighted_text,
            'metadata': self.metadata
        }


class MultiClassSemanticSearch:
    """Multi-class semantic search system with agentic routing."""
    
    def __init__(
        self,
        client: weaviate.WeaviateClient,
        embedding_model: str = "all-MiniLM-L6-v2",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-2-v2",
        use_cache: bool = True
    ):
        self.client = client
        self.collections = {}
        self.embedding_model = SentenceTransformer(embedding_model)
        self.rerank_model = CrossEncoder(rerank_model)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        # Cache and configuration
        self.use_cache = use_cache
        self.query_cache = {}
        self.tfidf_fitted = False
        
        # Logger
        self.logger = self._setup_logger()
        
        # Initialize collections
        self._initialize_collections()
        
        # Agent collection mappings
        self.agent_collections = {
            AgentType.POLICY: ["NIST_SP", "NIST_CSWP", "NIST_FIPS"],
            AgentType.RESEARCH: ["USENIX"],
            AgentType.ATTACK: ["MITRE"],
            AgentType.SECURITY: ["OWASP", "SecurityTools"],
            AgentType.AI: ["NIST_AI", "AISecKG"],
            AgentType.TRAINING: ["AnnoCTR"],
            AgentType.GENERAL: list(self.collections.keys())
        }
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger."""
        logger = logging.getLogger('MultiClassSemanticSearch')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _initialize_collections(self):
        """Initialize collection references."""
        try:
            collections = self.client.collections.list_all()
            for collection_name in collections:
                try:
                    self.collections[collection_name] = self.client.collections.get(collection_name)
                    self.logger.info(f"✅ Initialized collection: {collection_name}")
                except Exception as e:
                    self.logger.warning(f"⚠️  Could not initialize collection {collection_name}: {e}")
        except Exception as e:
            self.logger.error(f"❌ Error initializing collections: {e}")
    
    def _semantic_search_collection(
        self, 
        collection_name: str,
        query: str, 
        k: int = 20
    ) -> List[SearchResult]:
        """Perform semantic search in a specific collection."""
        if collection_name not in self.collections:
            self.logger.warning(f"Collection {collection_name} not available")
            return []
        
        collection = self.collections[collection_name]
        
        # Generate embedding
        query_embedding = self.embedding_model.encode(
            [query], 
            convert_to_numpy=True, 
            normalize_embeddings=True
        )[0]
        
        # Search in Weaviate
        try:
            results = collection.query.near_vector(
                near_vector=query_embedding.astype(np.float32),
                limit=k,
                return_metadata=["distance", "score"]
            )
        except Exception as e:
            self.logger.error(f"Error searching collection {collection_name}: {e}")
            return []
        
        # Convert to SearchResult
        search_results = []
        for result in results.objects:
            props = result.properties
            metadata = result.metadata
            
            # Normalize semantic score
            semantic_score = 1 - metadata.distance
            
            search_result = SearchResult(
                doc_id=props.get('docId', 'unknown'),
                title=props.get('title', 'Sin título'),
                content=props.get('text', ''),
                level=props.get('level', 'unknown'),
                page_start=props.get('pageStart', 0),
                page_end=props.get('pageEnd', 0),
                collection=collection_name,
                semantic_score=semantic_score,
                metadata=props
            )
            search_results.append(search_result)
            
        return search_results
    
    def _rerank_results(
        self, 
        query: str, 
        results: List[SearchResult], 
        top_k: int = 10
    ) -> List[SearchResult]:
        """Re-rank results using cross-encoder."""
        if not results:
            return results
            
        self.logger.info(f"Re-ranking {len(results)} results")
        
        # Prepare query-document pairs
        query_doc_pairs = []
        for result in results:
            doc_text = f"{result.title} {result.content[:500]}"
            query_doc_pairs.append([query, doc_text])
        
        # Get rerank scores
        rerank_scores = self.rerank_model.predict(query_doc_pairs)
        
        # Normalize scores
        rerank_scores = np.array(rerank_scores)
        exp_scores = np.exp(rerank_scores - np.max(rerank_scores))
        normalized_scores = exp_scores / np.sum(exp_scores)
        
        # Update results
        for result, score in zip(results, normalized_scores):
            result.rerank_score = float(score)
        
        # Sort and return top-k
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        return results[:top_k]
    
    def _highlight_text(self, text: str, query: str, max_length: int = 300) -> str:
        """Highlight relevant terms in text."""
        import re
        keywords = query.lower().split()
        
        # Find best window
        best_window = text[:max_length]
        best_score = 0
        
        words = text.lower().split()
        window_size = min(50, len(words))
        
        for i in range(len(words) - window_size + 1):
            window_text = ' '.join(words[i:i + window_size])
            score = sum(1 for keyword in keywords if keyword in window_text)
            
            if score > best_score:
                best_score = score
                start_char = text.lower().find(window_text)
                if start_char != -1:
                    end_char = start_char + len(window_text)
                    best_window = text[start_char:end_char]
        
        # Highlight keywords
        highlighted = best_window
        for keyword in keywords:
            highlighted = re.sub(
                rf'\b{re.escape(keyword)}\b', 
                keyword.upper(), 
                highlighted, 
                flags=re.IGNORECASE
            )
        
        return highlighted
    
    def search(
        self,
        query: str,
        agent: AgentType = AgentType.GENERAL,
        k: int = 5,
        rerank: bool = True
    ) -> List[SearchResult]:
        """Perform agentic search across relevant collections."""
        
        # Get collections for this agent
        target_collections = self.agent_collections.get(agent, list(self.collections.keys()))
        
        # Check cache
        cache_key = f"{query}_{agent.value}_{k}_{rerank}"
        if self.use_cache and cache_key in self.query_cache:
            self.logger.info("Result found in cache")
            return self.query_cache[cache_key]
        
        start_time = time.time()
        
        # Search across collections
        all_results = []
        for collection_name in target_collections:
            if collection_name in self.collections:
                collection_results = self._semantic_search_collection(collection_name, query, k * 2)
                all_results.extend(collection_results)
                self.logger.info(f"Searched {collection_name}: {len(collection_results)} results")
        
        # Combine and sort by semantic score
        all_results.sort(key=lambda x: x.semantic_score, reverse=True)
        
        # Take top candidates for reranking
        candidates = all_results[:k * 3]
        
        # Re-rank if requested
        if rerank and len(candidates) > k:
            final_results = self._rerank_results(query, candidates, k)
        else:
            final_results = candidates[:k]
        
        # Add highlighting
        for result in final_results:
            result.highlighted_text = self._highlight_text(result.content, query)
        
        # Cache results
        if self.use_cache:
            self.query_cache[cache_key] = final_results
        
        search_time = time.time() - start_time
        self.logger.info(f"Multi-class search completed in {search_time:.2f}s")
        
        return final_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections."""
        stats = {}
        for collection_name, collection in self.collections.items():
            try:
                count = collection.aggregate.over_all(total_count=True)
                stats[collection_name] = {
                    "total_objects": count.total_count,
                    "available": True
                }
            except Exception as e:
                stats[collection_name] = {
                    "total_objects": 0,
                    "available": False,
                    "error": str(e)
                }
        return stats


def format_results(results: List[SearchResult], query: str, agent: AgentType) -> None:
    """Format and display search results."""
    
    if not results:
        print("❌ No se encontraron resultados.")
        return
    
    print(f"\n{'='*100}")
    print(f"🔍 BÚSQUEDA AGÉNTICA: '{query}'")
    print(f"🤖 Agente: {agent.value.upper()}")
    print(f"📊 Resultados: {len(results)} documentos encontrados")
    print(f"{'='*100}")
    
    # Group by collection
    by_collection = {}
    for result in results:
        collection = result.collection
        if collection not in by_collection:
            by_collection[collection] = []
        by_collection[collection].append(result)
    
    print(f"\n📚 Resultados por colección:")
    for collection, collection_results in by_collection.items():
        print(f"   🏛️  {collection}: {len(collection_results)} resultados")
    
    print()
    
    for i, result in enumerate(results, 1):
        print(f"\n{'-'*20} RESULTADO {i} {'-'*20}")
        
        # Basic information
        print(f" 📋 Título: {result.title}")
        print(f" 🆔 Documento: {result.doc_id}")
        print(f" 🏛️  Colección: {result.collection}")
        print(f" 📄 Páginas: {result.page_start}-{result.page_end}")
        
        # Scores
        scores = []
        if result.semantic_score > 0:
            scores.append(f"Semántico: {result.semantic_score:.3f}")
        if result.rerank_score > 0:
            scores.append(f"Re-rank: {result.rerank_score:.3f}")
            
        print(f" 📊 Puntuaciones: {' | '.join(scores)}")
        
        # Content
        print(f"\n📝 Contenido relevante:")
        print(f"   {result.highlighted_text}")
        
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Class Agentic Semantic Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python multi_class_search.py "authentication" --agent policy --k 10
  python multi_class_search.py "vulnerabilidades SQL" --agent security --k 5
  python multi_class_search.py "técnicas de phishing" --agent attack --interactive
  python multi_class_search.py --stats  # Ver estadísticas de colecciones
        """
    )
    
    parser.add_argument("query", nargs='?', help="Consulta de búsqueda")
    parser.add_argument("--host", default="localhost", help="Host de Weaviate")
    parser.add_argument("--http_port", type=int, default=8080, help="Puerto HTTP")
    parser.add_argument("--grpc_port", type=int, default=50051, help="Puerto gRPC")
    parser.add_argument("--k", type=int, default=5, help="Número de resultados")
    parser.add_argument(
        "--agent", 
        choices=["policy", "research", "attack", "security", "ai", "training", "general"],
        default="general",
        help="Tipo de agente para búsqueda especializada"
    )
    parser.add_argument("--no-rerank", action="store_true", help="Desactivar re-ranking")
    parser.add_argument("--interactive", action="store_true", help="Modo interactivo")
    parser.add_argument("--stats", action="store_true", help="Mostrar estadísticas de colecciones")
    parser.add_argument("--output", help="Archivo de salida JSON")
    parser.add_argument("--verbose", action="store_true", help="Modo verbose")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger('MultiClassSemanticSearch').setLevel(logging.DEBUG)
    
    try:
        print("🔌 Conectando a Weaviate...")
        client = weaviate.connect_to_local(
            host=args.host, 
            port=args.http_port, 
            grpc_port=args.grpc_port
        )
        print("✅ Conectado exitosamente!")
        
        # Initialize search system
        search_system = MultiClassSemanticSearch(client)
        agent = AgentType(args.agent)
        
        if args.stats:
            print("\n📊 ESTADÍSTICAS DE COLECCIONES")
            print("="*60)
            stats = search_system.get_collection_stats()
            for collection_name, stat in stats.items():
                status = "✅" if stat["available"] else "❌"
                print(f"{status} {collection_name}: {stat['total_objects']} objetos")
                if not stat["available"]:
                    print(f"   Error: {stat.get('error', 'Unknown')}")
            return
        
        if args.interactive:
            print("\n🎯 Modo interactivo activado.")
            print("   Agentes: policy, research, attack, security, ai, training, general")
            print("   Comandos: 'quit', 'agent <nombre>', 'k <número>'")
            print("   Escribe 'help' para más opciones\n")
            
            current_k = args.k
            current_agent = agent
            current_rerank = not args.no_rerank
            
            while True:
                try:
                    user_input = input("🔍 Consulta: ").strip()
                    
                    if user_input.lower() in ['quit', 'exit', 'salir', 'q']:
                        break
                    
                    if user_input.lower() in ['help', 'h']:
                        print("""
Comandos disponibles:
  agent <policy|research|attack|security|ai|training|general> - Cambiar agente
  k <número>                           - Cambiar número de resultados  
  rerank <on|off>                      - Activar/desactivar re-ranking
  stats                                - Mostrar estadísticas
  clear                                - Limpiar pantalla
  quit/exit/q                          - Salir
                        """)
                        continue
                    
                    if user_input.startswith('agent '):
                        new_agent = user_input.split(' ', 1)[1]
                        try:
                            current_agent = AgentType(new_agent)
                            print(f" 🤖 Agente cambiado a: {new_agent}")
                        except ValueError:
                            print(" ❌ Agente inválido. Usa: policy, research, attack, security, ai, training, general")
                        continue
                    
                    if user_input.startswith('k '):
                        try:
                            current_k = int(user_input.split(' ', 1)[1])
                            print(f" 📊 Número de resultados cambiado a: {current_k}")
                        except ValueError:
                            print(" ❌ Número inválido")
                        continue
                    
                    if user_input == 'stats':
                        stats = search_system.get_collection_stats()
                        print(f"\n📊 Estadísticas:")
                        for collection_name, stat in stats.items():
                            status = "✅" if stat["available"] else "❌"
                            print(f"   {status} {collection_name}: {stat['total_objects']} objetos")
                        continue
                    
                    if user_input == 'clear':
                        print("\033[2J\033[H")  # Clear screen
                        continue
                    
                    if not user_input:
                        continue
                    
                    # Perform search
                    results = search_system.search(
                        user_input, 
                        agent=current_agent, 
                        k=current_k,
                        rerank=current_rerank
                    )
                    format_results(results, user_input, current_agent)
                    
                except KeyboardInterrupt:
                    print("\n 👋 Saliendo...")
                    break
                except Exception as e:
                    print(f" ❌ Error: {e}")
        
        else:
            # Single search
            if not args.query:
                parser.error("Debes proporcionar una consulta o usar el modo interactivo")
            
            results = search_system.search(
                args.query,
                agent=agent,
                k=args.k,
                rerank=not args.no_rerank
            )
            
            format_results(results, args.query, agent)
            
            # Save to file if specified
            if args.output:
                output_data = {
                    'query': args.query,
                    'agent': args.agent,
                    'timestamp': time.time(),
                    'results': [result.to_dict() for result in results]
                }
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                print(f"💾 Resultados guardados en: {args.output}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    finally:
        try:
            client.close()
        except:
            pass
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
