#!/usr/bin/env python3
"""
Enhanced Multi-Class Semantic Search System

This script provides advanced agentic search capabilities across multiple Weaviate collections
with hybrid scoring, query expansion, result diversification, and rich formatting.
"""

import argparse
import sys
import logging
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

# Support running as a script regardless of PYTHONPATH
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import weaviate
from weaviate.classes.query import Filter
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Optional: Rich for beautiful terminal output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️  Install 'rich' for better formatting: pip install rich")


class SearchException(Exception):
    """Custom exception for search errors."""
    pass


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
class SearchConfig:
    """Configuration for search system."""
    embedding_model: str = "all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-2-v2"
    use_cache: bool = True
    cache_dir: str = ".cache"
    cache_ttl: int = 3600  # seconds
    hybrid_alpha: float = 0.7  # Weight: semantic vs keyword
    max_query_length: int = 512
    enable_query_expansion: bool = False  # Disabled by default (requires nltk)
    enable_diversification: bool = True
    diversity_threshold: float = 0.85
    max_workers: int = 4  # For parallel search


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
    score_explanation: Dict[str, Any] = field(default_factory=dict)

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
            'metadata': self.metadata,
            'score_explanation': self.score_explanation
        }


class MultiClassSemanticSearch:
    """Enhanced multi-class semantic search system with agentic routing."""
    
    def __init__(
        self,
        client: weaviate.WeaviateClient,
        config: Optional[SearchConfig] = None
    ):
        self.client = client
        self.config = config or SearchConfig()
        self.collections = {}
        
        # Initialize models
        self.embedding_model = SentenceTransformer(self.config.embedding_model)
        self.rerank_model = CrossEncoder(self.config.rerank_model)
        
        # TF-IDF for keyword scoring
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        self.tfidf_fitted = False
        self.tfidf_corpus = []
        
        # Logger (must be first)
        self.logger = self._setup_logger()
        
        # Cache setup
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.query_cache = {}
        self._load_cache()
        
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
            AgentType.GENERAL: []  # Will be filled with all collections
        }
        
        # Update GENERAL agent with all available collections
        self.agent_collections[AgentType.GENERAL] = list(self.collections.keys())
        
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
    
    def _load_cache(self):
        """Load query cache from disk."""
        cache_file = self.cache_dir / "query_cache.pkl"
        if cache_file.exists():
            try:
                self.query_cache = joblib.load(cache_file)
                self.logger.info(f"📦 Loaded cache with {len(self.query_cache)} entries")
            except Exception as e:
                self.logger.warning(f"⚠️  Could not load cache: {e}")
                self.query_cache = {}
    
    def _save_cache(self):
        """Save query cache to disk."""
        if not self.config.use_cache:
            return
        
        cache_file = self.cache_dir / "query_cache.pkl"
        try:
            joblib.dump(self.query_cache, cache_file)
            self.logger.debug("💾 Cache saved")
        except Exception as e:
            self.logger.warning(f"⚠️  Could not save cache: {e}")
    
    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms (requires nltk)."""
        if not self.config.enable_query_expansion:
            return query
        
        try:
            from nltk.corpus import wordnet
            words = query.split()
            expanded = []
            
            for word in words:
                expanded.append(word)
                synsets = wordnet.synsets(word)
                
                if synsets:
                    synonyms = [lemma.name().replace('_', ' ') 
                               for syn in synsets[:2] 
                               for lemma in syn.lemmas()]
                    expanded.extend(synonyms[:2])
            
            expanded_query = " ".join(set(expanded))
            self.logger.debug(f"Query expanded: {query} -> {expanded_query}")
            return expanded_query
        except ImportError:
            self.logger.warning("NLTK not available for query expansion")
            return query
        except Exception as e:
            self.logger.warning(f"Query expansion failed: {e}")
            return query
    
    def _calculate_bm25_score(self, query: str, document: str) -> float:
        """Calculate BM25-like score for keyword matching."""
        query_terms = set(query.lower().split())
        doc_terms = document.lower().split()
        
        if not query_terms or not doc_terms:
            return 0.0
        
        # Term frequency
        doc_counter = Counter(doc_terms)
        doc_length = len(doc_terms)
        avg_doc_length = 100  # Assumed average
        
        k1 = 1.5
        b = 0.75
        
        score = 0.0
        for term in query_terms:
            if term in doc_counter:
                tf = doc_counter[term]
                idf = 1.0  # Simplified, would need corpus statistics
                
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
                
                score += idf * (numerator / denominator)
        
        return score / len(query_terms) if query_terms else 0.0
    
    def _calculate_hybrid_score(
        self,
        result: SearchResult,
        query: str
    ) -> float:
        """Combine semantic and keyword scores."""
        # Calculate keyword score
        keyword_score = self._calculate_bm25_score(query, result.content)
        result.keyword_score = keyword_score
        
        # Normalize scores
        semantic_norm = result.semantic_score
        keyword_norm = min(keyword_score / 5.0, 1.0)  # Normalize to [0, 1]
        
        # Hybrid combination
        alpha = self.config.hybrid_alpha
        hybrid = alpha * semantic_norm + (1 - alpha) * keyword_norm
        
        return hybrid
    
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
                near_vector=query_embedding.astype(np.float32).tolist(),
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
            semantic_score = 1 - metadata.distance if metadata.distance else 0.0
            
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
    
    def _batch_search_collections(
        self,
        query: str,
        collections: List[str],
        k: int = 20
    ) -> List[SearchResult]:
        """Parallel search across multiple collections."""
        all_results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_collection = {
                executor.submit(
                    self._semantic_search_collection,
                    col, query, k
                ): col for col in collections if col in self.collections
            }
            
            for future in as_completed(future_to_collection):
                collection_name = future_to_collection[future]
                try:
                    results = future.result(timeout=30)  # 30 second timeout
                    all_results.extend(results)
                    self.logger.info(f"✓ {collection_name}: {len(results)} results")
                except Exception as e:
                    self.logger.error(f"✗ {collection_name}: {e}")
        
        return all_results
    
    def _rerank_results(
        self, 
        query: str, 
        results: List[SearchResult], 
        top_k: int = 10
    ) -> List[SearchResult]:
        """Re-rank results using cross-encoder."""
        if not results:
            return results
            
        self.logger.info(f"🔄 Re-ranking {len(results)} results")
        
        # Prepare query-document pairs
        query_doc_pairs = []
        for result in results:
            doc_text = f"{result.title} {result.content[:500]}"
            query_doc_pairs.append([query, doc_text])
        
        # Get rerank scores
        rerank_scores = self.rerank_model.predict(query_doc_pairs)
        
        # Normalize scores using softmax
        rerank_scores = np.array(rerank_scores)
        exp_scores = np.exp(rerank_scores - np.max(rerank_scores))
        normalized_scores = exp_scores / np.sum(exp_scores)
        
        # Update results with rerank scores
        for result, score in zip(results, normalized_scores):
            result.rerank_score = float(score)
        
        # Sort by rerank score and return top-k
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        return results[:top_k]
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        # Use embeddings for similarity
        embeddings = self.embedding_model.encode(
            [text1[:500], text2[:500]], 
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        similarity = cosine_similarity(
            embeddings[0].reshape(1, -1),
            embeddings[1].reshape(1, -1)
        )[0][0]
        
        return float(similarity)
    
    def _diversify_results(
        self,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """Remove near-duplicate results."""
        if not self.config.enable_diversification or len(results) <= 1:
            return results
        
        diverse = [results[0]]
        threshold = self.config.diversity_threshold
        
        for result in results[1:]:
            is_diverse = True
            
            for selected in diverse:
                # Skip if from same document and adjacent pages
                if (result.doc_id == selected.doc_id and 
                    abs(result.page_start - selected.page_start) <= 2):
                    is_diverse = False
                    break
                
                # Check content similarity
                similarity = self._text_similarity(
                    result.content,
                    selected.content
                )
                
                if similarity > threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse.append(result)
        
        self.logger.info(f"🎯 Diversified: {len(results)} -> {len(diverse)} results")
        return diverse
    
    def _highlight_text(self, text: str, query: str, max_length: int = 300) -> str:
        """Highlight relevant terms in text."""
        keywords = query.lower().split()
        
        # Find best window with most keyword matches
        best_window = text[:max_length]
        best_score = 0
        
        words = text.split()
        window_size = min(50, len(words))
        
        for i in range(max(0, len(words) - window_size + 1)):
            window_text = ' '.join(words[i:i + window_size])
            score = sum(1 for keyword in keywords 
                       if keyword in window_text.lower())
            
            if score > best_score:
                best_score = score
                best_window = window_text[:max_length]
        
        # Highlight keywords (uppercase for visibility)
        highlighted = best_window
        for keyword in keywords:
            pattern = re.compile(rf'\b({re.escape(keyword)})\b', re.IGNORECASE)
            highlighted = pattern.sub(lambda m: m.group(1).upper(), highlighted)
        
        return highlighted
    
    def _explain_score(self, result: SearchResult, query: str) -> Dict[str, Any]:
        """Explain why this result scored high."""
        query_terms = set(query.lower().split())
        content_terms = set(result.content.lower().split())
        
        matched_terms = query_terms.intersection(content_terms)
        
        return {
            'semantic_score': round(result.semantic_score, 4),
            'keyword_score': round(result.keyword_score, 4),
            'query_terms_found': list(matched_terms),
            'match_ratio': round(len(matched_terms) / len(query_terms), 2) if query_terms else 0,
            'collection': result.collection,
            'document_length': len(result.content.split())
        }
    
    def _fallback_search(
        self,
        query: str,
        agent: AgentType,
        k: int
    ) -> List[SearchResult]:
        """Fallback search strategy when main search fails."""
        self.logger.warning("🔄 Using fallback search")
        
        # Try simple keyword search on available collections
        target_collections = self.agent_collections.get(agent, list(self.collections.keys()))
        results = []
        
        for collection_name in target_collections[:3]:  # Limit to 3 collections
            try:
                collection_results = self._semantic_search_collection(
                    collection_name, query, k
                )
                results.extend(collection_results)
            except Exception as e:
                self.logger.error(f"Fallback failed for {collection_name}: {e}")
        
        return results[:k]
    
    def search(
        self,
        query: str,
        agent: AgentType = AgentType.GENERAL,
        k: int = 5,
        rerank: bool = True,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform agentic search across relevant collections."""
        
        # Validate query
        if not query or len(query.strip()) == 0:
            raise SearchException("Query cannot be empty")
        
        if len(query) > self.config.max_query_length:
            query = query[:self.config.max_query_length]
            self.logger.warning(f"Query truncated to {self.config.max_query_length} characters")
        
        # Query expansion
        expanded_query = self._expand_query(query)
        
        # Get collections for this agent
        target_collections = self.agent_collections.get(agent, list(self.collections.keys()))
        
        if not target_collections:
            raise SearchException(f"No collections available for agent: {agent.value}")
        
        # Check cache
        cache_key = f"{expanded_query}_{agent.value}_{k}_{rerank}"
        if self.config.use_cache and cache_key in self.query_cache:
            cached_time, cached_results = self.query_cache[cache_key]
            if time.time() - cached_time < self.config.cache_ttl:
                self.logger.info("✨ Result found in cache")
                return cached_results
        
        start_time = time.time()
        
        try:
            # Parallel search across collections
            self.logger.info(f"🔍 Searching {len(target_collections)} collections")
            all_results = self._batch_search_collections(
                expanded_query,
                target_collections,
                k * 3
            )
            
            if not all_results:
                self.logger.warning("No results found")
                return []
            
            # Calculate hybrid scores
            for result in all_results:
                result.final_score = self._calculate_hybrid_score(result, query)
            
            # Sort by hybrid score
            all_results.sort(key=lambda x: x.final_score, reverse=True)
            
            # Take top candidates for reranking
            candidates = all_results[:k * 4]
            
            # Re-rank if requested
            if rerank and len(candidates) > k:
                final_results = self._rerank_results(expanded_query, candidates, k * 2)
            else:
                final_results = candidates[:k * 2]
            
            # Diversify results
            final_results = self._diversify_results(final_results)
            
            # Take final top-k
            final_results = final_results[:k]
            
            # Add highlighting and explanations
            for result in final_results:
                result.highlighted_text = self._highlight_text(result.content, query)
                result.score_explanation = self._explain_score(result, query)
            
            # Cache results
            if self.config.use_cache:
                self.query_cache[cache_key] = (time.time(), final_results)
                self._save_cache()
            
            search_time = time.time() - start_time
            self.logger.info(f"✅ Search completed in {search_time:.2f}s")
            
            return final_results
            
        except weaviate.exceptions.WeaviateBaseError as e:
            self.logger.error(f"Weaviate error: {e}")
            return self._fallback_search(query, agent, k)
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            raise SearchException(f"Search failed: {str(e)}")
    
    def suggest_related_queries(
        self,
        query: str,
        num_suggestions: int = 3
    ) -> List[str]:
        """Suggest related queries based on top results."""
        try:
            # Get results without caching for suggestions
            results = self.search(query, k=10, rerank=False)
            
            if not results:
                return []
            
            # Extract text from top results
            all_text = " ".join([r.content[:200] for r in results[:5]])
            
            # Get word frequency
            words = all_text.lower().split()
            word_freq = Counter(words)
            
            # Filter out common words and query terms
            query_terms = set(query.lower().split())
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
            
            important_words = [
                word for word, count in word_freq.most_common(20)
                if len(word) > 3 
                and word not in stop_words 
                and word not in query_terms
            ]
            
            # Generate suggestions
            suggestions = []
            for word in important_words[:num_suggestions]:
                suggestions.append(f"{query} {word}")
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating suggestions: {e}")
            return []
    
    def evaluate_search_quality(
        self,
        query: str,
        results: List[SearchResult]
    ) -> Dict[str, Any]:
        """Calculate search quality metrics."""
        if not results:
            return {
                'result_count': 0,
                'avg_score': 0,
                'score_variance': 0,
                'collection_diversity': 0,
                'avg_content_length': 0
            }
        
        scores = [r.final_score for r in results]
        collections = set(r.collection for r in results)
        content_lengths = [len(r.content.split()) for r in results]
        
        return {
            'result_count': len(results),
            'avg_score': round(np.mean(scores), 4),
            'score_variance': round(np.var(scores), 4),
            'score_std': round(np.std(scores), 4),
            'collection_diversity': len(collections),
            'collections': list(collections),
            'avg_content_length': round(np.mean(content_lengths), 2),
            'query_length': len(query.split())
        }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections."""
        stats = {}
        for collection_name, collection in self.collections.items():
            try:
                result = collection.aggregate.over_all(total_count=True)
                stats[collection_name] = {
                    "total_objects": result.total_count,
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
    
    if RICH_AVAILABLE:
        format_results_rich(results, query, agent)
    else:
        format_results_plain(results, query, agent)


def format_results_rich(results: List[SearchResult], query: str, agent: AgentType) -> None:
    """Format results using Rich library."""
    console = Console()
    
    # Header panel
    console.print(Panel(
        f"[bold cyan]🔍 Búsqueda:[/bold cyan] {query}\n"
        f"[bold magenta]🤖 Agente:[/bold magenta] {agent.value.upper()}\n"
        f"[bold green]📊 Resultados:[/bold green] {len(results)} documentos",
        title="[bold]Multi-Class Semantic Search[/bold]",
        border_style="blue"
    ))
    
    # Collection summary
    by_collection = {}
    for result in results:
        collection = result.collection
        if collection not in by_collection:
            by_collection[collection] = []
        by_collection[collection].append(result)
    
    summary_table = Table(title="Resultados por Colección", show_header=True)
    summary_table.add_column("Colección", style="cyan")
    summary_table.add_column("Cantidad", style="magenta", justify="right")
    
    for collection, coll_results in by_collection.items():
        summary_table.add_row(collection, str(len(coll_results)))
    
    console.print(summary_table)
    console.print()
    
    # Results table
    results_table = Table(title="Top Resultados", show_header=True, show_lines=True)
    results_table.add_column("#", style="cyan", width=3)
    results_table.add_column("Título", style="green", width=35)
    results_table.add_column("Colección", style="magenta", width=15)
    results_table.add_column("Score", style="yellow", justify="right", width=8)
    results_table.add_column("Páginas", style="blue", justify="center", width=8)
    
    for i, result in enumerate(results, 1):
        results_table.add_row(
            str(i),
            result.title[:32] + "..." if len(result.title) > 35 else result.title,
            result.collection,
            f"{result.final_score:.3f}",
            f"{result.page_start}-{result.page_end}"
        )
    
    console.print(results_table)
    console.print()
    
    # Detailed results
    for i, result in enumerate(results, 1):
        panel_content = (
            f"[bold]🆔 Documento:[/bold] {result.doc_id}\n"
            f"[bold]📚 Colección:[/bold] {result.collection}\n"
            f"[bold]📄 Páginas:[/bold] {result.page_start}-{result.page_end}\n\n"
            f"[bold]📊 Puntuaciones:[/bold]\n"
            f"  • Semántico: {result.semantic_score:.3f}\n"
            f"  • Keyword: {result.keyword_score:.3f}\n"
            f"  • Rerank: {result.rerank_score:.3f}\n"
            f"  • Final: {result.final_score:.3f}\n\n"
            f"[bold]📝 Contenido Relevante:[/bold]\n{result.highlighted_text}"
        )
        
        console.print(Panel(
            panel_content,
            title=f"[bold]Resultado {i}: {result.title[:40]}[/bold]",
            border_style="green"
        ))


def format_results_plain(results: List[SearchResult], query: str, agent: AgentType) -> None:
    """Format results for plain text output."""
    
    print(f"\n{'='*100}")
    print(f"🔍 BÚSQUEDA: '{query}'")
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
        if result.keyword_score > 0:
            scores.append(f"Keyword: {result.keyword_score:.3f}")
        if result.rerank_score > 0:
            scores.append(f"Re-rank: {result.rerank_score:.3f}")
        scores.append(f"Final: {result.final_score:.3f}")
            
        print(f" 📊 Puntuaciones: {' | '.join(scores)}")
        
        # Score explanation
        if result.score_explanation:
            exp = result.score_explanation
            print(f" 💡 Explicación: {len(exp.get('query_terms_found', []))} términos coinciden")
        
        # Content
        print(f"\n📝 Contenido relevante:")
        print(f"   {result.highlighted_text}")
        
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Multi-Class Agentic Semantic Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Búsquedas simples
  python multi_class_search.py "authentication best practices" --agent policy --k 10
  python multi_class_search.py "vulnerabilidades SQL injection" --agent security --k 5
  python multi_class_search.py "phishing attack techniques" --agent attack --k 8
  
  # Búsqueda con re-ranking desactivado
  python multi_class_search.py "machine learning security" --agent ai --no-rerank
  
  # Modo interactivo
  python multi_class_search.py --interactive
  
  # Ver estadísticas
  python multi_class_search.py --stats
  
  # Guardar resultados en JSON
  python multi_class_search.py "zero-day exploits" --agent security --output results.json
  
  # Búsqueda con sugerencias
  python multi_class_search.py "encryption algorithms" --suggest
  
  # Evaluación de calidad
  python multi_class_search.py "network security" --evaluate
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
    parser.add_argument("--suggest", action="store_true", help="Sugerir consultas relacionadas")
    parser.add_argument("--evaluate", action="store_true", help="Evaluar calidad de búsqueda")
    parser.add_argument("--no-cache", action="store_true", help="Desactivar caché")
    parser.add_argument("--no-diversify", action="store_true", help="Desactivar diversificación")
    parser.add_argument("--hybrid-alpha", type=float, default=0.7, 
                       help="Peso para scoring híbrido (0-1, default: 0.7)")
    
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
        
        # Create configuration
        config = SearchConfig(
            use_cache=not args.no_cache,
            enable_diversification=not args.no_diversify,
            hybrid_alpha=args.hybrid_alpha
        )
        
        # Initialize search system
        search_system = MultiClassSemanticSearch(client, config=config)
        agent = AgentType(args.agent)
        
        if args.stats:
            print("\n📊 ESTADÍSTICAS DE COLECCIONES")
            print("="*60)
            stats = search_system.get_collection_stats()
            
            total_docs = 0
            available_count = 0
            
            for collection_name, stat in stats.items():
                status = "✅" if stat["available"] else "❌"
                count = stat['total_objects']
                print(f"{status} {collection_name:.<30} {count:>10,} objetos")
                
                if stat["available"]:
                    total_docs += count
                    available_count += 1
                else:
                    print(f"   ⚠️  Error: {stat.get('error', 'Unknown')}")
            
            print(f"\n{'='*60}")
            print(f"📊 Total: {available_count} colecciones disponibles")
            print(f"📚 Total: {total_docs:,} documentos indexados")
            return
        
        if args.interactive:
            print("\n" + "="*60)
            print("🎯 MODO INTERACTIVO ACTIVADO")
            print("="*60)
            print("\n📋 Comandos disponibles:")
            print("  • agent <nombre>     - Cambiar agente")
            print("  • k <número>         - Cambiar cantidad de resultados")
            print("  • rerank <on|off>    - Activar/desactivar re-ranking")
            print("  • suggest            - Sugerir consultas relacionadas")
            print("  • stats              - Ver estadísticas")
            print("  • evaluate           - Evaluar última búsqueda")
            print("  • clear              - Limpiar pantalla")
            print("  • help               - Mostrar ayuda")
            print("  • quit/exit/q        - Salir")
            print("\n🤖 Agentes disponibles: policy, research, attack, security, ai, training, general")
            print()
            
            current_k = args.k
            current_agent = agent
            current_rerank = not args.no_rerank
            last_query = None
            last_results = None
            
            while True:
                try:
                    user_input = input("🔍 Consulta: ").strip()
                    
                    if user_input.lower() in ['quit', 'exit', 'salir', 'q']:
                        print("\n👋 ¡Hasta luego!")
                        break
                    
                    if user_input.lower() in ['help', 'h', '?']:
                        print("""
╔══════════════════════════════════════════════════════════════╗
║                    AYUDA - COMANDOS                          ║
╠══════════════════════════════════════════════════════════════╣
║ agent <nombre>        Cambiar agente activo                  ║
║                      (policy|research|attack|security|...)   ║
║ k <número>           Cambiar cantidad de resultados          ║
║ rerank <on|off>      Activar/desactivar re-ranking          ║
║ suggest              Sugerir consultas relacionadas          ║
║ stats                Ver estadísticas de colecciones         ║
║ evaluate             Evaluar calidad de última búsqueda      ║
║ clear                Limpiar pantalla                        ║
║ help                 Mostrar esta ayuda                      ║
║ quit/exit/q          Salir del programa                      ║
╚══════════════════════════════════════════════════════════════╝
                        """)
                        continue
                    
                    if user_input.startswith('agent '):
                        new_agent = user_input.split(' ', 1)[1].strip()
                        try:
                            current_agent = AgentType(new_agent)
                            print(f"✅ Agente cambiado a: {new_agent.upper()}")
                        except ValueError:
                            print("❌ Agente inválido. Opciones: policy, research, attack, security, ai, training, general")
                        continue
                    
                    if user_input.startswith('k '):
                        try:
                            current_k = int(user_input.split(' ', 1)[1])
                            if current_k < 1 or current_k > 50:
                                print("⚠️  Recomendado: k entre 1 y 50")
                            print(f"✅ Número de resultados: {current_k}")
                        except ValueError:
                            print("❌ Número inválido")
                        continue
                    
                    if user_input.startswith('rerank '):
                        setting = user_input.split(' ', 1)[1].lower()
                        if setting in ['on', 'yes', 'true', '1']:
                            current_rerank = True
                            print("✅ Re-ranking activado")
                        elif setting in ['off', 'no', 'false', '0']:
                            current_rerank = False
                            print("✅ Re-ranking desactivado")
                        else:
                            print("❌ Usa: rerank on|off")
                        continue
                    
                    if user_input == 'stats':
                        print("\n📊 ESTADÍSTICAS DE COLECCIONES")
                        print("="*60)
                        stats = search_system.get_collection_stats()
                        for collection_name, stat in stats.items():
                            status = "✅" if stat["available"] else "❌"
                            count = stat['total_objects']
                            print(f"{status} {collection_name:.<30} {count:>10,} objetos")
                        continue
                    
                    if user_input == 'suggest':
                        if not last_query:
                            print("❌ Primero realiza una búsqueda")
                            continue
                        
                        print(f"\n💡 Generando sugerencias para: '{last_query}'")
                        suggestions = search_system.suggest_related_queries(last_query, 5)
                        
                        if suggestions:
                            print("\n🔍 Consultas relacionadas:")
                            for i, suggestion in enumerate(suggestions, 1):
                                print(f"  {i}. {suggestion}")
                        else:
                            print("❌ No se pudieron generar sugerencias")
                        continue
                    
                    if user_input == 'evaluate':
                        if not last_results or not last_query:
                            print("❌ Primero realiza una búsqueda")
                            continue
                        
                        print(f"\n📈 EVALUACIÓN DE CALIDAD")
                        print("="*60)
                        quality = search_system.evaluate_search_quality(last_query, last_results)
                        
                        print(f"📊 Resultados encontrados: {quality['result_count']}")
                        print(f"⭐ Score promedio: {quality['avg_score']:.4f}")
                        print(f"📉 Desviación estándar: {quality['score_std']:.4f}")
                        print(f"🏛️  Diversidad de colecciones: {quality['collection_diversity']}")
                        print(f"📚 Colecciones: {', '.join(quality['collections'])}")
                        print(f"📝 Longitud promedio: {quality['avg_content_length']:.0f} palabras")
                        continue
                    
                    if user_input == 'clear':
                        print("\033[2J\033[H")  # Clear screen ANSI code
                        continue
                    
                    if not user_input:
                        continue
                    
                    # Perform search
                    print(f"\n🔎 Buscando: '{user_input}'...")
                    start_time = time.time()
                    
                    results = search_system.search(
                        user_input, 
                        agent=current_agent, 
                        k=current_k,
                        rerank=current_rerank
                    )
                    
                    elapsed = time.time() - start_time
                    
                    format_results(results, user_input, current_agent)
                    
                    print(f"\n⏱️  Tiempo de búsqueda: {elapsed:.2f}s")
                    
                    # Save for later commands
                    last_query = user_input
                    last_results = results
                    
                    # Auto-suggest if enabled
                    if args.suggest and results:
                        suggestions = search_system.suggest_related_queries(user_input, 3)
                        if suggestions:
                            print(f"\n💡 Búsquedas relacionadas:")
                            for i, suggestion in enumerate(suggestions, 1):
                                print(f"  {i}. {suggestion}")
                    
                except KeyboardInterrupt:
                    print("\n\n👋 ¡Hasta luego!")
                    break
                except SearchException as e:
                    print(f"\n❌ Error de búsqueda: {e}")
                except Exception as e:
                    print(f"\n❌ Error inesperado: {e}")
                    if args.verbose:
                        import traceback
                        traceback.print_exc()
        
        else:
            # Single search mode
            if not args.query:
                parser.error("❌ Debes proporcionar una consulta o usar --interactive")
            
            print(f"\n🔎 Ejecutando búsqueda...")
            start_time = time.time()
            
            results = search_system.search(
                args.query,
                agent=agent,
                k=args.k,
                rerank=not args.no_rerank
            )
            
            elapsed = time.time() - start_time
            
            format_results(results, args.query, agent)
            
            print(f"\n⏱️  Tiempo de búsqueda: {elapsed:.2f}s")
            
            # Evaluate quality if requested
            if args.evaluate:
                print(f"\n📈 EVALUACIÓN DE CALIDAD")
                print("="*60)
                quality = search_system.evaluate_search_quality(args.query, results)
                
                print(f"📊 Resultados: {quality['result_count']}")
                print(f"⭐ Score promedio: {quality['avg_score']:.4f}")
                print(f"📉 Desviación estándar: {quality['score_std']:.4f}")
                print(f"🏛️  Diversidad: {quality['collection_diversity']} colecciones")
                print(f"📚 Colecciones: {', '.join(quality['collections'])}")
            
            # Suggest related queries if requested
            if args.suggest:
                print(f"\n💡 Generando sugerencias relacionadas...")
                suggestions = search_system.suggest_related_queries(args.query, 5)
                
                if suggestions:
                    print("\n🔍 Consultas relacionadas:")
                    for i, suggestion in enumerate(suggestions, 1):
                        print(f"  {i}. {suggestion}")
                else:
                    print("❌ No se pudieron generar sugerencias")
            
            # Save to file if specified
            if args.output:
                output_data = {
                    'query': args.query,
                    'agent': args.agent,
                    'timestamp': time.time(),
                    'elapsed_time': elapsed,
                    'config': {
                        'k': args.k,
                        'rerank': not args.no_rerank,
                        'hybrid_alpha': args.hybrid_alpha
                    },
                    'results': [result.to_dict() for result in results]
                }
                
                if args.evaluate:
                    output_data['quality_metrics'] = quality
                
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                
                print(f"\n💾 Resultados guardados en: {args.output}")
        
    except weaviate.exceptions.WeaviateConnectionError as e:
        print(f"\n❌ Error de conexión a Weaviate: {e}")
        print("💡 Asegúrate de que Weaviate esté corriendo en {args.host}:{args.http_port}")
        return 1
    except SearchException as e:
        print(f"\n❌ Error de búsqueda: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    finally:
        try:
            client.close()
            print("\n✅ Conexión cerrada")
        except:
            pass
    
    return 0


if __name__ == "__main__":
    sys.exit(main())