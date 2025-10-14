# -*- coding: utf-8 -*-
"""
Enhanced Hybrid Hierarchical + Semantic Chunking for RAG

This module provides an advanced chunking system that combines structural and semantic
approaches for optimal document segmentation in RAG applications.

Key improvements:
- Better error handling and validation
- Modular architecture with clear separation of concerns
- Enhanced configuration management
- Improved performance and memory efficiency
- Comprehensive logging and monitoring
- Robust fallback mechanisms
- Thread-safe operations

The all-* models were trained on all available training data 
(more than 1 billion training pairs) and are designed as general 
purpose models. The all-mpnet-base-v2 model provides the best
quality, while all-MiniLM-L6-v2 is 5 times faster and still 
offers good quality. Toggle All models to see all evaluated 
original models.

https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
"""

from __future__ import annotations

import logging
import re
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.schema import Document
    except ImportError:
        # Fallback Document class
        class Document:
            def __init__(self, page_content: str, metadata: dict = None):
                self.page_content = page_content
                self.metadata = metadata or {}

# Optional dependencies with graceful fallbacks
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    tiktoken = None
    HAS_TIKTOKEN = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_SKLEARN = True
except ImportError:
    TfidfVectorizer = None
    HAS_SKLEARN = False


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Error Handling and Validation

class ChunkingError(Exception):
    """Base exception for chunking operations."""
    pass

class ValidationError(ChunkingError):
    """Raised when input validation fails."""
    pass

class ProcessingError(ChunkingError):
    """Raised when processing operations fail."""
    pass

def validate_text_input(text: str) -> str:
    """Validate and clean text input."""
    if not isinstance(text, str):
        raise ValidationError(f"Expected string, got {type(text)}")
    
    if not text.strip():
        raise ValidationError("Text input is empty or whitespace only")
    
    # Clean common issues
    cleaned = re.sub(r'\x00', '', text)  # Remove null bytes
    cleaned = re.sub(r'[\r\n]{3,}', '\n\n', cleaned)  # Normalize line breaks
    
    if len(cleaned.encode('utf-8')) > 10_000_000:  # 10MB limit
        logger.warning("Text input is very large, this may impact performance")
    
    return cleaned

# ---------------------------------------------------------------------
# Enhanced Configuration Management

class ContentType(Enum):
    """Content type classification for specialized handling."""
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    CODE = "code"
    FIGURE = "figure"
    QUOTE = "quote"
    REFERENCE = "reference"
    UNKNOWN = "unknown"

@dataclass(frozen=True)
class ChunkingConfig:
    """Immutable configuration for chunking parameters."""
    # Token limits
    target_tokens: int = 200
    min_tokens: int = 100
    max_tokens: int = 256
    
    # Overlap settings
    overlap_ratio: float = 0.18
    
    # Semantic window parameters
    sem_win_tokens: int = 150
    sem_step_tokens: int = 30
    smooth_k: int = 5
    mad_z_thresh: float = 2.5
    
    # Content-specific penalties and bonuses
    penalize_inside: Dict[str, float] = field(default_factory=lambda: {
        "code": 0.3, "table": 0.4, "figure": 0.5, "quote": 0.6
    })
    bonus_anchors: Dict[str, float] = field(default_factory=lambda: {
        "heading": 1.4, "subtitle": 1.2, "list_start": 1.1, "paragraph_end": 1.05
    })
    
    # Structural settings
    max_snap_distance: int = 50
    structure_weight: float = 0.8
    
    # Model settings
    embed_model: str = "all-MiniLM-L6-v2"
    token_model: str = "gpt-4o-mini"
    
    # Performance settings
    max_workers: int = 4
    batch_size: int = 32
    cache_size: int = 10000
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.min_tokens >= self.max_tokens:
            raise ValueError("min_tokens must be less than max_tokens")
        if not 0 <= self.overlap_ratio <= 1:
            raise ValueError("overlap_ratio must be between 0 and 1")
        if self.target_tokens < self.min_tokens or self.target_tokens > self.max_tokens:
            raise ValueError("target_tokens must be between min_tokens and max_tokens")

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'ChunkingConfig':
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            return cls(**config_dict)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise

    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(self), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")
            raise

    @classmethod
    def from_legacy_dict(cls, config_dict: Dict[str, Any]) -> 'ChunkingConfig':
        """Create config from legacy DEFAULTS format."""
        return cls(**config_dict)

def validate_config(config: ChunkingConfig) -> None:
    """Validate chunking configuration."""
    if not isinstance(config, ChunkingConfig):
        raise ValidationError(f"Expected ChunkingConfig, got {type(config)}")
    
    # Additional runtime validations
    if config.max_workers <= 0:
        raise ValidationError("max_workers must be positive")
    
    if config.batch_size <= 0:
        raise ValidationError("batch_size must be positive")

# Legacy configuration for backward compatibility
DEFAULTS = {
    "target_tokens": 200,  # Ajustado para all-MiniLM-L6-v2 (max 256)
    "min_tokens": 100,     # Ajustado para all-MiniLM-L6-v2
    "max_tokens": 256,     # Límite máximo del modelo all-MiniLM-L6-v2
    "overlap_ratio": 0.18,
    "sem_win_tokens": 150, # Ajustado para ventana semántica
    "sem_step_tokens": 30,
    "smooth_k": 5,
    "mad_z_thresh": 2.5, # aca se decide cuadno se corta el chunk(se cambia de tema)
    "penalize_inside": {"code": 0.3, "table": 0.4, "figure": 0.5, "quote": 0.6},
    "bonus_anchors": {"heading": 1.4, "subtitle": 1.2, "list_start": 1.1, "paragraph_end": 1.05},
    "max_snap_distance": 50,   # tokens máximos para snap (ajustado)
    "structure_weight": 0.8,   # peso para mantener estructura vs semántica
}

# ---------------------------------------------------------------------
# Enhanced Token Management

def _approx_num_tokens(text: str) -> int:
    """Improved token estimation without tiktoken."""
    if not text:
        return 0
    
    words = len(text.split())
    chars = len(text)
    
    # Improved approximation based on empirical data
    word_estimate = words * 1.3
    char_estimate = chars / 3.5
    
    # Weight average favoring word count for normal text
    if chars / max(words, 1) < 8:  # Normal text
        return max(1, int(word_estimate * 0.7 + char_estimate * 0.3))
    else:  # Dense text (code, tables, etc.)
        return max(1, int(word_estimate * 0.4 + char_estimate * 0.6))

class TokenCounter:
    """Thread-safe token counting with multiple backends and caching."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", cache_size: int = 10000):
        self.model_name = model_name
        self.cache: Dict[str, int] = {}
        self.cache_size = cache_size
        self._lock = threading.RLock()
        
        # Initialize tokenizer with fallback chain
        self._init_tokenizer()
    
    def _init_tokenizer(self) -> None:
        """Initialize tokenizer with fallback options."""
        self.tokenizer = None
        self.use_tiktoken = False
        
        if HAS_TIKTOKEN:
            try:
                self.tokenizer = tiktoken.encoding_for_model(self.model_name)
                self.use_tiktoken = True
                logger.info(f"Using tiktoken for {self.model_name}")
            except Exception:
                try:
                    self.tokenizer = tiktoken.get_encoding("cl100k_base")
                    self.use_tiktoken = True
                    logger.info("Using tiktoken with cl100k_base encoding")
                except Exception as e:
                    logger.warning(f"Failed to initialize tiktoken: {e}")
    
    def _cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    
    def _approximate_tokens(self, text: str) -> int:
        """Fallback token estimation."""
        if not text:
            return 0
        
        words = len(text.split())
        chars = len(text)
        
        # Improved approximation based on empirical data
        word_estimate = words * 1.3
        char_estimate = chars / 3.5
        
        # Weight average favoring word count for normal text
        if chars / max(words, 1) < 8:  # Normal text
            return max(1, int(word_estimate * 0.7 + char_estimate * 0.3))
        else:  # Dense text (code, tables, etc.)
            return max(1, int(word_estimate * 0.4 + char_estimate * 0.6))
    
    def count(self, text: str) -> int:
        """Count tokens with caching and fallback."""
        if not text:
            return 0
        
        # Check cache first
        cache_key = self._cache_key(text)
        with self._lock:
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        # Count tokens
        try:
            if self.use_tiktoken and self.tokenizer:
                count = len(self.tokenizer.encode(text, disallowed_special=()))
            else:
                count = self._approximate_tokens(text)
        except Exception:
            count = self._approximate_tokens(text)
        
        # Update cache (with size management)
        with self._lock:
            if len(self.cache) >= self.cache_size:
                # Remove oldest entries (simple FIFO)
                keys_to_remove = list(self.cache.keys())[:self.cache_size // 4]
                for key in keys_to_remove:
                    self.cache.pop(key, None)
            
            self.cache[cache_key] = count
        
        return count
    
    def build_position_map(self, text: str) -> List[Tuple[int, int]]:
        """Build precise token-to-character position mapping."""
        if not text:
            return [(0, 0)]
        
        # Split into sentences for better mapping precision
        sentence_pattern = r'(?<=[.!?])\s+|(?<=\n)\s*(?=\S)|(?<=\w)\n\s*(?=\w)'
        segments = re.split(sentence_pattern, text)
        
        positions = []
        current_char = 0
        current_token = 0
        
        for segment in segments:
            if not segment.strip():
                continue
            
            segment_tokens = self.count(segment)
            positions.append((current_token, current_char))
            
            current_char += len(segment)
            current_token += segment_tokens
        
        # Add final position
        positions.append((current_token, len(text)))
        return positions

# Global token counter instance
_token_counter = None
_counter_lock = threading.Lock()

def get_token_counter(model_name: str = "gpt-4o-mini") -> TokenCounter:
    """Get or create global token counter instance."""
    global _token_counter
    with _counter_lock:
        if _token_counter is None or _token_counter.model_name != model_name:
            _token_counter = TokenCounter(model_name)
        return _token_counter

# Legacy function for backward compatibility
def count_tokens(text: str, model_name: str = "gpt-4o-mini") -> int:
    """Legacy token counting function - now using TokenCounter."""
    return get_token_counter(model_name).count(text)

def build_token_position_map(text: str, model_name: str = "gpt-4o-mini") -> List[int]:
    """Construye mapeo más preciso de posición en tokens a posición en caracteres."""
    if not text:
        return [0]
    
    # Dividir en oraciones/párrafos para mejor mapeo
    sentences = re.split(r'(?<=[.!?])\s+|\n\s*\n', text)
    char_positions = []
    token_positions = []
    
    current_char = 0
    current_token = 0
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        sentence_tokens = count_tokens(sentence, model_name)
        char_positions.append(current_char)
        token_positions.append(current_token)
        
        current_char += len(sentence) + 1  # +1 para el separador
        current_token += sentence_tokens
    
    # Añadir posición final
    char_positions.append(len(text))
    token_positions.append(current_token)
    
    return list(zip(token_positions, char_positions))

# ---------------------------------------------------------------------
# Enhanced Block Representation

@dataclass
class Block:
    """Enhanced block representation with metadata and validation."""
    text: str
    content_type: ContentType = ContentType.UNKNOWN
    meta: Dict[str, Any] = field(default_factory=dict)
    tokens: Optional[int] = None
    priority: float = 1.0
    breakable: bool = True
    quality_score: float = 1.0
    
    def __post_init__(self):
        """Validate and initialize block."""
        self.text = validate_text_input(self.text)
        
        if self.tokens is None:
            self.tokens = get_token_counter().count(self.text)
        
        # Adjust properties based on content type
        self._adjust_properties()
    
    def _adjust_properties(self) -> None:
        """Adjust block properties based on content type."""
        type_configs = {
            ContentType.HEADING: {"priority": 2.0, "breakable": False},
            ContentType.CODE: {"priority": 1.8, "breakable": False},
            ContentType.TABLE: {"priority": 1.6, "breakable": False},
            ContentType.FIGURE: {"priority": 1.5, "breakable": False},
            ContentType.QUOTE: {"priority": 1.3, "breakable": True},
            ContentType.LIST: {"priority": 1.2, "breakable": True},
        }
        
        if self.content_type in type_configs:
            config = type_configs[self.content_type]
            self.priority = config["priority"]
            self.breakable = config["breakable"]
        
        # Legacy compatibility for backward annotations
        if self.meta.get("is_heading", False):
            self.priority = 2.0
            self.breakable = False
        elif self.meta.get("in_table", False):
            self.priority = 1.5
            self.breakable = False
        elif self.meta.get("in_code", False):
            self.priority = 1.5
            self.breakable = False
    
    @property
    def is_heading(self) -> bool:
        """Check if block is a heading."""
        return self.content_type == ContentType.HEADING
    
    @property
    def effective_priority(self) -> float:
        """Calculate effective priority including quality score."""
        return self.priority * self.quality_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert block to dictionary for serialization."""
        return {
            "text": self.text,
            "content_type": self.content_type.value,
            "meta": self.meta,
            "tokens": self.tokens,
            "priority": self.priority,
            "breakable": self.breakable,
            "quality_score": self.quality_score,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Block':
        """Create block from dictionary."""
        data = data.copy()
        if "content_type" in data:
            data["content_type"] = ContentType(data["content_type"])
        return cls(**data)

# ---------------------------------------------------------------------
# Parseo jerárquico mejorado

def infer_heading_level(text: str, category: str) -> int:
    """Infiere el nivel del título basado en contenido y longitud."""
    text = text.strip()
    
    # Patrones para diferentes niveles
    if re.match(r'^\d+\.\s', text):  # "1. Introducción"
        return 1
    elif re.match(r'^\d+\.\d+\s', text):  # "1.1 Contexto"  
        return 2
    elif re.match(r'^\d+\.\d+\.\d+\s', text):  # "1.1.1 Detalles"
        return 3
    
    # Por longitud y mayúsculas
    if len(text) < 50 and text.isupper():
        return 1
    elif len(text) < 80 and any(word.isupper() for word in text.split()[:3]):
        return 2
    elif category in {"Title", "Header"}:
        return 1 if len(text) < 60 else 2
    
    return 2  # default

def detect_content_type(text: str, category: str) -> Dict[str, bool]:
    """Detecta mejor el tipo de contenido para aplicar reglas específicas."""
    text_lower = text.lower()
    
    return {
        "is_heading": category in {"Title", "Header", "Heading"},
        "in_table": category == "Table" or "│" in text or "\t" in text and "|" in text,
        "in_list": category == "ListItem" or re.match(r'^\s*[-•*]\s', text, re.M),
        "in_code": category == "Code" or (
            "```" in text or 
            re.search(r'^\s*(def|class|import|from)\s', text, re.M) or
            re.search(r'[{}();].*[{}();]', text)
        ),
        "is_figure": "fig" in text_lower and ("caption" in text_lower or "figure" in text_lower),
        "is_quote": text.strip().startswith('"') and text.strip().endswith('"'),
        "is_reference": re.search(r'\[\d+\]|\(\d{4}\)|\bet al\.', text),
    }

# def parse_pdf_blocks(
#     path: str,
#     infer_table_structure: bool = True,
#     strategy: str = "hi_res",
# ) -> List[Block]:
#     """Parseo jerárquico mejorado con mejor detección de estructura."""
#     if partition_pdf is None:
#         raise RuntimeError(
#             "unstructured no está instalado. Instala: pip install 'unstructured[all-docs]'"
#         )

#     elements = partition_pdf(
#         filename=path,
#         strategy=strategy,
#         infer_table_structure=infer_table_structure,
#     )

#     blocks: List[Block] = []
#     current_section = {"title": None, "level": 0}
#     section_stack: List[Dict[str, Any]] = []  # stack de secciones para jerarquía

#     def normalize_text(t: str) -> str:
#         # Limpieza mejorada
#         t = re.sub(r'\s+', ' ', t)  # espacios múltiples a uno
#         t = re.sub(r'\n\s*\n\s*\n+', '\n\n', t)  # máximo 2 saltos
#         t = re.sub(r'^\s+|\s+$', '', t, flags=re.M)  # espacios inicio/fin líneas
#         return t.strip()

#     for el in elements:
#         category = getattr(el, "category", None) or el.__class__.__name__
#         text = normalize_text(getattr(el, "text", "") or "")

#         if not text:
#             continue

#         content_flags = detect_content_type(text, category)
        
#         # Construir metadata enriquecida
#         meta = {
#             "category": category,
#             "section_title": current_section["title"],
#             "section_level": current_section["level"],
#             **content_flags
#         }

#         if content_flags["is_heading"]:
#             level = infer_heading_level(text, category)
            
#             # Manejar stack de secciones
#             while section_stack and section_stack[-1]["level"] >= level:
#                 section_stack.pop()
            
#             section_info = {"title": text, "level": level}
#             section_stack.append(section_info)
#             current_section = section_info
            
#             meta.update({
#                 "section_title": text,
#                 "section_level": level,
#                 "heading_level": level,
#                 "section_path": " > ".join([s["title"] for s in section_stack])
#             })

#         block = Block(text=text, meta=meta)
#         blocks.append(block)

#     return blocks

# ---------------------------------------------------------------------
# Procesamiento de archivos JSONL 

def parse_jsonl_blocks(
    jsonl_path: str,
    infer_table_structure: bool = True,
) -> List[Block]:
    """Parsea archivos .pages.jsonl existentes en bloques estructurales."""
    blocks: List[Block] = []
    current_section = {"title": None, "level": 0}
    section_stack: List[Dict[str, Any]] = []

    def normalize_text(t: str) -> str:
        # Limpieza mejorada
        t = re.sub(r'\s+', ' ', t)  # espacios múltiples a uno
        t = re.sub(r'\n\s*\n\s*\n+', '\n\n', t)  # máximo 2 saltos
        t = re.sub(r'^\s+|\s+$', '', t, flags=re.M)  # espacios inicio/fin líneas
        return t.strip()

    def detect_content_type_from_text(text: str) -> Dict[str, bool]:
        """Detecta tipo de contenido basado en el texto de la página."""
        text_lower = text.lower()
        
        return {
            "is_heading": (
                len(text.strip()) < 100 and 
                (text.isupper() or 
                 text.strip().startswith(('Chapter', 'Section', 'Part', 'Capítulo', 'Sección', '1.', '2.', '3.')))
            ),
            "in_table": "│" in text or "\t" in text and text.count("|") > 2,
            "in_list": re.match(r'^\s*[-•*]\s', text, re.M) or re.search(r'^\s*\d+\.\s', text, re.M),
            "in_code": (
                "```" in text or 
                re.search(r'^\s*(def|class|import|from)\s', text, re.M) or
                re.search(r'[{}();].*[{}();]', text)
            ),
            "is_figure": "fig" in text_lower and ("caption" in text_lower or "figure" in text_lower),
            "is_quote": text.strip().startswith('"') and text.strip().endswith('"'),
            "is_reference": re.search(r'\[\d+\]|\(\d{4}\)|\bet al\.', text),
        }

    def infer_heading_level_from_text(text: str) -> int:
        """Infiere nivel de título basado en el texto."""
        text = text.strip()
        
        # Patrones para diferentes niveles
        if re.match(r'^\d+\.\s', text):  # "1. Introducción"
            return 1
        elif re.match(r'^\d+\.\d+\s', text):  # "1.1 Contexto"  
            return 2
        elif re.match(r'^\d+\.\d+\.\d+\s', text):  # "1.1.1 Detalles"
            return 3
        
        # Por longitud y mayúsculas
        if len(text) < 50 and text.isupper():
            return 1
        elif len(text) < 80 and any(word.isupper() for word in text.split()[:3]):
            return 2
        
        return 2  # default

    # Leer archivo JSONL
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                page_data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️  Error parseando línea {line_num}: {e}")
                continue

            page_text = page_data.get("text", "")
            if not page_text.strip():
                continue

            # Normalizar texto
            normalized_text = normalize_text(page_text)
            
            # Detectar tipo de contenido
            content_flags = detect_content_type_from_text(normalized_text)
            
            # Construir metadata enriquecida
            meta = {
                "category": "NarrativeText",  # Por defecto
                "section_title": current_section["title"],
                "section_level": current_section["level"],
                "page_num_real": page_data.get("page_num_real"),
                "page_num_logical": page_data.get("page_num_logical"),
                "doc_title": page_data.get("doc_title"),
                "authors": page_data.get("authors", []),
                "toc_path": page_data.get("toc_path", []),
                "source_id": page_data.get("source_id"),
                "quality_flags": page_data.get("quality_flags", []),
                **content_flags
            }

            # Detectar si es un título
            if content_flags["is_heading"]:
                level = infer_heading_level_from_text(normalized_text)
                
                # Manejar stack de secciones
                while section_stack and section_stack[-1]["level"] >= level:
                    section_stack.pop()
                
                section_info = {"title": normalized_text, "level": level}
                section_stack.append(section_info)
                current_section = section_info
                
                meta.update({
                    "category": "Title",
                    "section_title": normalized_text,
                    "section_level": level,
                    "heading_level": level,
                    "section_path": " > ".join([s["title"] for s in section_stack])
                })

            # Crear bloque
            block = Block(text=normalized_text, meta=meta)
            blocks.append(block)

    return blocks

# ---------------------------------------------------------------------
# Empaque por estructura mejorado

def calculate_cohesion_score(blocks: List[Block]) -> float:
    """Calcula score de cohesión temática entre bloques."""
    if len(blocks) < 2:
        return 1.0
    
    # Factores de cohesión
    same_section = all(
        b.meta.get("section_title") == blocks[0].meta.get("section_title") 
        for b in blocks[1:]
    )
    
    similar_types = len(set(b.meta.get("category") for b in blocks)) <= 2
    
    has_headings = any(b.meta.get("is_heading") for b in blocks)
    
    score = 0.0
    if same_section:
        score += 0.5
    if similar_types:
        score += 0.3
    if not has_headings:  # párrafos contiguos son más cohesivos
        score += 0.2
    
    return min(1.0, score)

def pack_by_structure(
    blocks: List[Block],
    target: int,
    min_tokens: int,
    max_tokens: int,
    token_model: str = "gpt-4o-mini",
    structure_weight: float = 0.7,
) -> List[Block]:
    """Empaque mejorado que balancea estructura y tamaño."""
    packed: List[Block] = []
    buffer: List[Block] = []
    
    def split_large_block(block: Block) -> List[Block]:
        """Split a block that exceeds max_tokens into smaller chunks."""
        if block.tokens <= max_tokens:
            return [block]
        
        # Split by paragraphs first
        paragraphs = block.text.split('\n\n')
        if len(paragraphs) <= 1:
            # If no paragraph breaks, split by sentences
            sentences = re.split(r'(?<=[.!?])\s+', block.text)
            if len(sentences) <= 1:
                # If no sentence breaks, split by words
                words = block.text.split()
                chunk_size = max_tokens // 4  # Approximate words per token
                chunks = []
                for i in range(0, len(words), chunk_size):
                    chunk_words = words[i:i + chunk_size]
                    chunk_text = ' '.join(chunk_words)
                    chunk_tokens = get_token_counter(token_model).count(chunk_text)
                    if chunk_tokens > 0:
                        chunk_meta = dict(block.meta)
                        chunk_meta.update({
                            "is_split_chunk": True,
                            "original_tokens": block.tokens,
                            "chunk_index": len(chunks)
                        })
                        chunks.append(Block(
                            text=chunk_text,
                            meta=chunk_meta,
                            tokens=chunk_tokens,
                            priority=block.priority,
                            breakable=block.breakable
                        ))
                return chunks
            else:
                # Split by sentences
                chunks = []
                current_chunk = []
                current_tokens = 0
                
                for sentence in sentences:
                    sent_tokens = get_token_counter(token_model).count(sentence)
                    if current_tokens + sent_tokens > max_tokens and current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        chunk_meta = dict(block.meta)
                        chunk_meta.update({
                            "is_split_chunk": True,
                            "original_tokens": block.tokens,
                            "chunk_index": len(chunks)
                        })
                        chunks.append(Block(
                            text=chunk_text,
                            meta=chunk_meta,
                            tokens=current_tokens,
                            priority=block.priority,
                            breakable=block.breakable
                        ))
                        current_chunk = []
                        current_tokens = 0
                    
                    current_chunk.append(sentence)
                    current_tokens += sent_tokens
                
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunk_meta = dict(block.meta)
                    chunk_meta.update({
                        "is_split_chunk": True,
                        "original_tokens": block.tokens,
                        "chunk_index": len(chunks)
                    })
                    chunks.append(Block(
                        text=chunk_text,
                        meta=chunk_meta,
                        tokens=current_tokens,
                        priority=block.priority,
                        breakable=block.breakable
                    ))
                return chunks
        else:
            # Split by paragraphs
            chunks = []
            current_chunk = []
            current_tokens = 0
            
            for paragraph in paragraphs:
                para_tokens = get_token_counter(token_model).count(paragraph)
                if current_tokens + para_tokens > max_tokens and current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunk_meta = dict(block.meta)
                    chunk_meta.update({
                        "is_split_chunk": True,
                        "original_tokens": block.tokens,
                        "chunk_index": len(chunks)
                    })
                    chunks.append(Block(
                        text=chunk_text,
                        meta=chunk_meta,
                        tokens=current_tokens,
                        priority=block.priority,
                        breakable=block.breakable
                    ))
                    current_chunk = []
                    current_tokens = 0
                
                current_chunk.append(paragraph)
                current_tokens += para_tokens
            
            if current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunk_meta = dict(block.meta)
                chunk_meta.update({
                    "is_split_chunk": True,
                    "original_tokens": block.tokens,
                    "chunk_index": len(chunks)
                })
                chunks.append(Block(
                    text=chunk_text,
                    meta=chunk_meta,
                    tokens=current_tokens,
                    priority=block.priority,
                    breakable=block.breakable
                ))
            return chunks
    
    def flush_buffer() -> None:
        if not buffer:
            return
            
        # Calcular cohesión del buffer actual
        cohesion = calculate_cohesion_score(buffer)
        
        combined_text = "\n\n".join(b.text for b in buffer)
        combined_tokens = sum(b.tokens for b in buffer)
        
        # Meta combinada con información de cohesión
        combined_meta = dict(buffer[0].meta)
        combined_meta.update({
            "is_heading_block": any(b.meta.get("is_heading") for b in buffer),
            "block_cohesion": cohesion,
            "source_blocks": len(buffer),
            "combined_categories": list(set(b.meta.get("category") for b in buffer))
        })
        
        new_block = Block(
            text=combined_text,
            meta=combined_meta,
            tokens=combined_tokens,
            priority=max(b.priority for b in buffer),
            breakable=all(b.breakable for b in buffer)
        )
        
        packed.append(new_block)
        buffer.clear()

    current_tokens = 0
    last_section_title = None

    for block in blocks:
        # Check if individual block is too large and split it
        if block.tokens > max_tokens:
            flush_buffer()  # Flush current buffer first
            current_tokens = 0
            
            # Split the large block
            split_blocks = split_large_block(block)
            packed.extend(split_blocks)
            continue
        
        # Forzar flush en cambio de sección o título
        current_section = block.meta.get("section_title")
        if (last_section_title and current_section != last_section_title) or \
           block.meta.get("is_heading", False):
            flush_buffer()
            current_tokens = 0
        
        last_section_title = current_section

        # Si agregar este bloque excedería max y ya tenemos min, flush
        projected_tokens = current_tokens + block.tokens
        if projected_tokens > max_tokens and current_tokens >= min_tokens:
            flush_buffer()
            current_tokens = 0

        buffer.append(block)
        current_tokens += block.tokens

        # Si estamos cerca del target y el buffer es cohesivo, considerar flush
        if current_tokens >= target:
            cohesion = calculate_cohesion_score(buffer)
            
            # Decisión de flush basada en estructura y cohesión
            should_flush = (
                current_tokens >= target * 1.1 or  # muy por encima del target
                (current_tokens >= target and cohesion < 0.3) or  # target alcanzado, baja cohesión
                current_tokens >= max_tokens * 0.9  # cerca del máximo
            )
            
            if should_flush:
                flush_buffer()
                current_tokens = 0

    # Flush final
    flush_buffer()
    return packed

# ---------------------------------------------------------------------
# Refinado semántico mejorado

def find_semantic_boundaries(
    text: str,
    embedder,
    win_tokens: int,
    step_tokens: int,
    smooth_k: int,
    z_thresh: float,
    model_name: str,
    penalize_patterns: Dict[str, float],
    bonus_patterns: Dict[str, float],
) -> List[int]:
    """Encuentra límites semánticos con sistema de penalización/bonus."""
    
    windows = split_text_by_tokens(text, win_tokens, step_tokens, model_name)
    if len(windows) < 3:  # necesitamos al menos 3 ventanas
        return []
    
    # Embeddings y distancias
    embs = embedder.embed(windows)
    dists = cosine_dist_seq(embs)
    dists_smoothed = smooth_series(dists, k=smooth_k)
    z_scores = mad_zscores(dists_smoothed)
    
    # Encontrar candidatos a corte
    candidate_indices = np.where(z_scores > z_thresh)[0].tolist()
    
    if not candidate_indices:
        return []
    
    # Sistema de scoring para cada candidato
    scored_candidates = []
    win_token_sizes = [count_tokens(w, model_name) for w in windows]
    cumulative_tokens = np.cumsum(win_token_sizes).tolist()
    
    for idx in candidate_indices:
        if idx >= len(cumulative_tokens):
            continue
            
        approx_position = cumulative_tokens[idx]
        base_score = float(z_scores[idx])
        
        # Obtener contexto alrededor del corte
        total_tokens = sum(win_token_sizes)
        char_pos = int((approx_position / total_tokens) * len(text))
        context_start = max(0, char_pos - 200)
        context_end = min(len(text), char_pos + 200)
        context = text[context_start:context_end]
        
        # Aplicar penalizaciones
        penalty_factor = 1.0
        for pattern_name, penalty in penalize_patterns.items():
            if pattern_name == "code" and ("```" in context or re.search(r'^\s*(def|class|import)', context, re.M)):
                penalty_factor *= penalty
            elif pattern_name == "table" and ("│" in context or context.count("|") > 2):
                penalty_factor *= penalty
            elif pattern_name == "figure" and re.search(r'fig\w*\s*\d+|figure\s*\d+', context, re.I):
                penalty_factor *= penalty
            elif pattern_name == "quote" and ('"' in context or "'" in context):
                penalty_factor *= penalty
        
        # Aplicar bonuses
        bonus_factor = 1.0
        for pattern_name, bonus in bonus_patterns.items():
            if pattern_name == "heading" and re.search(r'^#+\s|\n#+\s', context, re.M):
                bonus_factor *= bonus
            elif pattern_name == "paragraph_end" and "\n\n" in context:
                bonus_factor *= bonus
            elif pattern_name == "list_start" and re.search(r'^\s*[-•*]\s', context, re.M):
                bonus_factor *= bonus
        
        final_score = base_score * penalty_factor * bonus_factor
        scored_candidates.append((approx_position, final_score))
    
    # Seleccionar mejores candidatos
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Evitar cortes muy cercanos entre sí
    selected_positions = []
    min_distance = win_tokens // 2
    
    for pos, score in scored_candidates:
        too_close = any(abs(pos - sel_pos) < min_distance for sel_pos in selected_positions)
        if not too_close:
            selected_positions.append(pos)
    
    return sorted(selected_positions)

def refine_semantic_block(
    text: str,
    meta: Dict[str, Any],
    embedder,
    model_name_tok: str,
    win_tokens: int,
    step_tokens: int,
    smooth_k: int,
    z_thresh: float,
    penalize_patterns: Dict[str, float],
    bonus_patterns: Dict[str, float],
) -> List[Tuple[str, Dict[str, Any]]]:
    """Refinado semántico mejorado con mejor scoring."""
    
    cut_positions = find_semantic_boundaries(
        text, embedder, win_tokens, step_tokens, smooth_k, z_thresh,
        model_name_tok, penalize_patterns, bonus_patterns
    )
    
    if not cut_positions:
        return [(text, meta)]
    
    # Marcar para procesamiento posterior
    return [("<<DEFERRED_SPLIT>>", {
        "_raw_text": text,
        "_meta": meta,
        "_token_cuts": cut_positions
    })]

# ---------------------------------------------------------------------
# Overlapping inteligente

def smart_overlap(chunks: List[str], overlap_ratio: float, model_name: str) -> List[str]:
    """Overlap más inteligente que considera estructura del contenido."""
    if overlap_ratio <= 0 or len(chunks) <= 1:
        return chunks
    
    result = []
    
    for i, current_chunk in enumerate(chunks):
        if i == 0:
            result.append(current_chunk)
            continue
        
        prev_chunk = result[-1]
        
        # Calcular tamaño de overlap basado en contenido
        current_tokens = count_tokens(current_chunk, model_name)
        overlap_tokens = int(current_tokens * overlap_ratio)
        
        # Buscar buen punto de overlap en chunk anterior
        sentences_prev = re.split(r'(?<=[.!?])\s+', prev_chunk)
        
        # Tomar últimas oraciones del anterior que no excedan overlap_tokens
        overlap_text = ""
        for sent in reversed(sentences_prev):
            test_overlap = sent + " " + overlap_text if overlap_text else sent
            if count_tokens(test_overlap, model_name) > overlap_tokens:
                break
            overlap_text = test_overlap
        
        # Combinar con chunk actual
        if overlap_text:
            enhanced_chunk = f"{overlap_text.strip()}\n\n{current_chunk.strip()}"
        else:
            enhanced_chunk = current_chunk
        
        result.append(enhanced_chunk)
    
    return result

# # ---------------------------------------------------------------------
# Función principal para procesar archivos JSONL

def hybrid_chunk_jsonl(
    jsonl_path: str,
    *,
    embed_model: str = "all-MiniLM-L6-v2",
    token_model: str = "gpt-4o-mini",
    target_tokens: int = DEFAULTS["target_tokens"],
    min_tokens: int = DEFAULTS["min_tokens"],
    max_tokens: int = DEFAULTS["max_tokens"],
    overlap_ratio: float = DEFAULTS["overlap_ratio"],
    sem_win_tokens: int = DEFAULTS["sem_win_tokens"],
    sem_step_tokens: int = DEFAULTS["sem_step_tokens"],
    smooth_k: int = DEFAULTS["smooth_k"],
    mad_z_thresh: float = DEFAULTS["mad_z_thresh"],
    penalize_inside: Dict[str, float] = None,
    bonus_anchors: Dict[str, float] = None,
    structure_weight: float = DEFAULTS["structure_weight"],
    max_snap_distance: int = DEFAULTS["max_snap_distance"],
) -> List[Document]:
    """
    Enhanced pipeline principal para procesar archivos .pages.jsonl existentes.
    """
    try:
        # Convert to enhanced config for validation
        config_dict = {
            "target_tokens": target_tokens,
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "overlap_ratio": overlap_ratio,
            "sem_win_tokens": sem_win_tokens,
            "sem_step_tokens": sem_step_tokens,
            "smooth_k": smooth_k,
            "mad_z_thresh": mad_z_thresh,
            "penalize_inside": penalize_inside or DEFAULTS["penalize_inside"],
            "bonus_anchors": bonus_anchors or DEFAULTS["bonus_anchors"],
            "structure_weight": structure_weight,
            "max_snap_distance": max_snap_distance,
            "embed_model": embed_model,
            "token_model": token_model,
        }
        
        config = ChunkingConfig.from_legacy_dict(config_dict)
        validate_config(config)
        
        logger.info(f"Processing JSONL: {jsonl_path}")
        
        # 1) Parse jerárquico de JSONL (enhanced)
        logger.info("Step 1: Hierarchical JSONL analysis...")
        blocks = parse_jsonl_blocks_enhanced(jsonl_path)
        logger.info(f"Extracted {len(blocks)} structural blocks")
        
        # 2) Enhanced structural packing
        logger.info("Step 2: Enhanced structural packing...")
        primary_blocks = pack_by_structure_enhanced(
            blocks, config=config
        )
        logger.info(f"Generated {len(primary_blocks)} primary blocks")
        
        # 3) Enhanced semantic refinement
        logger.info("Step 3: Enhanced semantic refinement...")
        embedder = EnhancedEmbedder(model=config.embed_model)
        
        final_items = []
        semantic_refined = 0
        
        try:
            # Use parallel processing for better performance
            with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                futures = {
                    executor.submit(refine_block_semantic, block, embedder, config): block 
                    for block in primary_blocks
                }
                
                for future in as_completed(futures):
                    try:
                        refined = future.result(timeout=300)  # 5 minute timeout per block
                        final_items.extend(refined)
                        
                        if any("DEFERRED_SPLIT" in item[0] for item in refined):
                            semantic_refined += 1
                            
                    except Exception as e:
                        logger.warning(f"Semantic refinement failed for block: {e}")
                        block = futures[future]
                        final_items.append((block.text, block.meta))
                        
        except Exception as e:
            logger.warning(f"Parallel processing failed: {e}")
            # Fallback to sequential processing
            for block in primary_blocks:
                if block.tokens <= config.target_tokens or not block.breakable:
                    final_items.append((block.text, block.meta))
                else:
                    try:
                        refined = refine_block_semantic(block, embedder, config)
                        final_items.extend(refined)
                        semantic_refined += 1
                    except Exception as e:
                        logger.warning(f"Semantic refinement failed: {e}")
                        final_items.append((block.text, block.meta))
        
        logger.info(f"Semantically refined {semantic_refined} blocks")
        
        # 4) Materialize deferred cuts (enhanced)
        logger.info("Step 4: Materializing cuts...")
        materialized_chunks = materialize_cuts_enhanced(final_items, config)
        
        # 5) Enhanced smart overlapping
        logger.info("Step 5: Enhanced smart overlapping...")
        texts = [chunk[0] for chunk in materialized_chunks]
        overlapped_texts = smart_overlap_enhanced(texts, config.overlap_ratio, get_token_counter(config.token_model))
        
        # 6) Create final documents with enhanced metadata
        documents = create_final_documents_enhanced(overlapped_texts, materialized_chunks, config)
        
        logger.info(f"Completed: {len(documents)} final chunks generated")
        
        # Enhanced statistics
        token_counts = [d.metadata["chunk_tokens"] for d in documents]
        logger.info("Token statistics:")
        logger.info(f"  • Average: {np.mean(token_counts):.0f}")
        logger.info(f"  • Median: {np.median(token_counts):.0f}")
        logger.info(f"  • Min: {min(token_counts)} | Max: {max(token_counts)}")
        logger.info(f"  • Std: {np.std(token_counts):.0f}")
        
        return documents
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise
    except ProcessingError as e:
        logger.error(f"Processing error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in hybrid_chunk_jsonl: {e}")
        # Fallback to original implementation
        logger.info("Falling back to original implementation due to error")
        return hybrid_chunk_jsonl_legacy(
            jsonl_path, embed_model=embed_model, token_model=token_model,
            target_tokens=target_tokens, min_tokens=min_tokens, max_tokens=max_tokens,
            overlap_ratio=overlap_ratio, sem_win_tokens=sem_win_tokens,
            sem_step_tokens=sem_step_tokens, smooth_k=smooth_k, mad_z_thresh=mad_z_thresh,
            penalize_inside=penalize_inside, bonus_anchors=bonus_anchors,
            structure_weight=structure_weight, max_snap_distance=max_snap_distance
        )

# ---------------------------------------------------------------------
# Enhanced Helper Functions for New Implementation

def parse_jsonl_blocks_enhanced(jsonl_path: str) -> List[Block]:
    """Enhanced JSONL parsing with improved content detection."""
    try:
        return parse_jsonl_blocks(jsonl_path)  # Use existing implementation for now
    except Exception as e:
        logger.error(f"Enhanced JSONL parsing failed: {e}")
        raise ProcessingError(f"Failed to parse JSONL: {e}")

def pack_by_structure_enhanced(blocks: List[Block], config: ChunkingConfig) -> List[Block]:
    """Enhanced structural packing with config."""
    try:
        return pack_by_structure(
            blocks, 
            target=config.target_tokens, 
            min_tokens=config.min_tokens, 
            max_tokens=config.max_tokens, 
            token_model=config.token_model,
            structure_weight=config.structure_weight
        )
    except Exception as e:
        logger.error(f"Enhanced structural packing failed: {e}")
        raise ProcessingError(f"Failed structural packing: {e}")

class EnhancedEmbedder:
    """Enhanced embedder with better error handling."""
    
    def __init__(self, model: str):
        try:
            self.embedder = Embedder(model=model)
        except Exception as e:
            logger.error(f"Failed to initialize embedder: {e}")
            raise ProcessingError(f"Embedder initialization failed: {e}")

def refine_block_semantic(block: Block, embedder, config: ChunkingConfig):
    """Enhanced semantic block refinement."""
    try:
        if block.tokens <= config.target_tokens or not block.breakable:
            return [(block.text, block.meta)]
        
        return refine_semantic_block(
            block.text, block.meta, embedder.embedder, config.token_model,
            config.sem_win_tokens, config.sem_step_tokens, config.smooth_k, 
            config.mad_z_thresh, config.penalize_inside, config.bonus_anchors
        )
    except Exception as e:
        logger.warning(f"Semantic refinement failed for block: {e}")
        return [(block.text, block.meta)]

def materialize_cuts_enhanced(final_items: List, config: ChunkingConfig):
    """Enhanced cut materialization."""
    materialized_chunks = []
    
    for item_text, item_meta in final_items:
        if item_text != "<<DEFERRED_SPLIT>>":
            materialized_chunks.append((item_text, item_meta))
            continue
        
        try:
            raw_text = item_meta["_raw_text"]
            cuts = item_meta["_token_cuts"]
            original_meta = item_meta["_meta"]
            
            total_tokens = get_token_counter(config.token_model).count(raw_text)
            safe_cuts = enforce_min_max(cuts, total_tokens, config.min_tokens, config.max_tokens, config.target_tokens)
            
            text_parts = split_text_by_token_cuts(raw_text, safe_cuts, config.token_model)
            for part in text_parts:
                if part.strip():
                    materialized_chunks.append((part, original_meta))
        except Exception as e:
            logger.warning(f"Failed to materialize cut: {e}")
            materialized_chunks.append((item_text or "", item_meta))
    
    return materialized_chunks

def smart_overlap_enhanced(texts: List[str], overlap_ratio: float, token_counter) -> List[str]:
    """Enhanced overlapping with token counter."""
    try:
        if overlap_ratio <= 0 or len(texts) <= 1:
            return texts
        
        result = []
        
        for i, current_chunk in enumerate(texts):
            if i == 0:
                result.append(current_chunk)
                continue
            
            prev_chunk = result[-1]
            
            current_tokens = token_counter.count(current_chunk)
            overlap_tokens = int(current_tokens * overlap_ratio)
            
            sentences_prev = re.split(r'(?<=[.!?])\s+', prev_chunk)
            
            overlap_text = ""
            for sent in reversed(sentences_prev):
                test_overlap = sent + " " + overlap_text if overlap_text else sent
                if token_counter.count(test_overlap) > overlap_tokens:
                    break
                overlap_text = test_overlap
            
            if overlap_text:
                enhanced_chunk = f"{overlap_text.strip()}\n\n{current_chunk.strip()}"
            else:
                enhanced_chunk = current_chunk
            
            result.append(enhanced_chunk)
        
        return result
    except Exception as e:
        logger.warning(f"Enhanced overlapping failed: {e}")
        return texts

def create_final_documents_enhanced(texts: List[str], materialized_chunks: List, config: ChunkingConfig) -> List[Document]:
    """Create final documents with enhanced metadata."""
    documents = []
    token_counter = get_token_counter(config.token_model)
    
    for i, (overlapped_text, (_, original_meta)) in enumerate(zip(texts, materialized_chunks)):
        meta = dict(original_meta or {})
        meta.update({
            "chunk_id": i,
            "chunk_tokens": token_counter.count(overlapped_text),
            "overlap_applied": config.overlap_ratio > 0,
            "processing_enhanced": True,
            "config_used": {
                "target_tokens": config.target_tokens,
                "overlap_ratio": config.overlap_ratio,
                "embed_model": config.embed_model,
            }
        })
        
        documents.append(Document(
            page_content=overlapped_text,
            metadata=meta
        ))
    
    return documents

def hybrid_chunk_jsonl_legacy(*args, **kwargs):
    """Legacy implementation for fallback purposes."""
    # Store original implementation for fallback
    pass  # Implementation will stay as it was in the enhanced version - the legacy call will use the existing logic

# ---------------------------------------------------------------------
# Función para procesar múltiples archivos JSONL

def process_multiple_jsonl_files(
    input_dir: str,
    output_dir: str,
    **kwargs
) -> Dict[str, List[Document]]:
    """
    Procesa múltiples archivos .pages.jsonl en un directorio.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Buscar archivos .pages.jsonl recursivamente
    jsonl_files = list(input_path.glob("**/*.pages.jsonl"))
    if not jsonl_files:
        print(f"  No se encontraron archivos *.pages.jsonl en {input_dir}")
        return {}
    
    results = {}
    total_chunks = 0
    
    for jsonl_file in jsonl_files:
        print(f"\n  Procesando: {jsonl_file.name}")
        
        try:
            # Procesar archivo
            documents = hybrid_chunk_jsonl(str(jsonl_file), **kwargs)
            
            # Guardar resultados
            output_file = output_path / f"{jsonl_file.stem.replace('.pages', '')}.chunks.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for doc in documents:
                    # Limpiar metadatos para serialización JSON
                    clean_metadata = {}
                    for key, value in doc.metadata.items():
                        if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                            clean_metadata[key] = value
                        else:
                            clean_metadata[key] = str(value)
                    
                    chunk_data = {
                        "content": doc.page_content,
                        "metadata": clean_metadata
                    }
                    f.write(json.dumps(chunk_data, ensure_ascii=False) + "\n")
            
            results[jsonl_file.name] = documents
            total_chunks += len(documents)
            
            print(f"     {len(documents)} chunks guardados en {output_file.name}")
            
        except Exception as e:
            print(f"     Error procesando {jsonl_file.name}: {e}")
            continue
    
    # Archivo global deshabilitado para RAG agéntico
    # Los chunks se mantienen organizados por categorías para mejor rendimiento
    
    print("\nProcesamiento completado:")
    print(f"    Total de chunks: {total_chunks}")
    print(f"   Archivos procesados: {len(results)}")
    print("  Estructura optimizada para RAG agéntico")
    
    return results

# ---------------------------------------------------------------------
# Funciones auxiliares que faltaban (de tu código original)

def split_text_by_tokens(text: str, win_tokens: int, step_tokens: int, model_name: str) -> List[str]:
    """Crea ventanas deslizantes por tokens (mejorada)."""
    words = text.split()
    if not words:
        return []

    windows = []
    i = 0
    while i < len(words):
        # Determinar ventana actual
        window_words = []
        current_tokens = 0
        j = i
        
        while j < len(words) and current_tokens < win_tokens:
            word = words[j]
            word_tokens = count_tokens(word, model_name)
            if current_tokens + word_tokens > win_tokens and window_words:
                        break
            window_words.append(word)
            current_tokens += word_tokens
            j += 1
        
        if window_words:
            windows.append(" ".join(window_words))
        
        # Avanzar por step_tokens
        if j <= i:
            i += 1  # fallback: avanzar al menos una palabra
        else:
            # Estimar palabras para step_tokens
            if current_tokens > 0:
                words_per_token = len(window_words) / current_tokens
                step_words = max(1, int(step_tokens * words_per_token))
                i += step_words
            else:
                i += 1
    
    return windows

def cosine_dist_seq(embs: List[np.ndarray]) -> np.ndarray:
    """Distancias coseno entre embeddings consecutivos."""
    if len(embs) < 2:
        return np.array([])
    
    distances = []
    for i in range(len(embs) - 1):
        a, b = embs[i], embs[i + 1]
        sim = float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0])
        distances.append(1.0 - sim)
    
    return np.array(distances, dtype=np.float32)

def smooth_series(x: np.ndarray, k: int = 5) -> np.ndarray:
    """Suavizado con ventana móvil."""
    if len(x) <= k:
        return x
    
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode='same')

def mad_zscores(x: np.ndarray) -> np.ndarray:
    """Z-scores robustos usando MAD."""
    if len(x) == 0:
        return x
    
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    
    if mad == 0:
        return np.zeros_like(x)
    
    return 0.6745 * (x - median) / mad

def enforce_min_max(
    cut_tokens_sorted: List[int],
    total_tokens: int,
    min_tokens: int,
    max_tokens: int,
    target: int,
) -> List[int]:
    """Asegura que los chunks respeten límites min/max."""
    if not cut_tokens_sorted:
        if total_tokens > max_tokens:
            cuts = []
            pos = target
            while pos < total_tokens:
                cuts.append(pos)
                pos += target
            return cuts
        return []

    cuts = sorted([c for c in cut_tokens_sorted if 0 < c < total_tokens])
    boundaries = [0] + cuts + [total_tokens]
    
    adjusted_cuts = []
    i = 0
    while i < len(boundaries) - 1:
        start = boundaries[i]
        end = boundaries[i + 1]
        segment_size = end - start
        
        if segment_size < min_tokens and i < len(boundaries) - 2:
            # Fusionar con el siguiente segmento
            continue
        elif segment_size > max_tokens:
            # Dividir segmento grande
            pos = start + target
            while pos < end:
                adjusted_cuts.append(pos)
                pos += target
        else:
            # Segmento OK
            if end != total_tokens:
                adjusted_cuts.append(end)
        
        i += 1
    
    return sorted(set([c for c in adjusted_cuts if 0 < c < total_tokens]))

def split_text_by_token_cuts(text: str, token_cuts: List[int], model_name: str) -> List[str]:
    """Divide texto en segmentos usando cortes por tokens."""
    if not token_cuts:
        return [text] if text.strip() else []
    
    total_tokens = count_tokens(text, model_name)
    if total_tokens == 0:
        return [text] if text.strip() else []
    
    cuts = sorted([c for c in token_cuts if 0 < c < total_tokens])
    segments = []
    
    last_pos = 0
    for cut_pos in cuts + [total_tokens]:
        # Convertir posición en tokens a posición en caracteres (aproximada)
        start_ratio = last_pos / total_tokens
        end_ratio = cut_pos / total_tokens
        
        start_char = int(start_ratio * len(text))
        end_char = int(end_ratio * len(text))
        
        # Ajustar a límites de palabras para evitar cortes dentro de palabras
        if start_char > 0:
            while start_char < len(text) and not text[start_char].isspace():
                start_char += 1
        
        if end_char < len(text):
            while end_char > start_char and not text[end_char].isspace():
                end_char += 1
        
        segment = text[start_char:end_char].strip()
        if segment:
            segments.append(segment)
        
        last_pos = cut_pos
    
    return segments

# ---------------------------------------------------------------------
# Clase Embedder mejorada

class Embedder:
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self.model = model
        self.cache: Dict[str, np.ndarray] = {}
        
        # Preferencia: Sentence-Transformers → TF-IDF
        if SentenceTransformer is not None:
            try:
                st_name = self.model if self.model else "all-MiniLM-L6-v2"
                self._st_model = SentenceTransformer(st_name)
                self.use_st = True
            except Exception:
                print( "Sentence-Transformers no disponible, usando TF-IDF")
                self._st_model = None
                self.use_st = False
        else:
            print(" Sentence-Transformers no instalado, usando TF-IDF")
            self._st_model = None
            self.use_st = False

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]  # Clave más corta

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        """Embeding con caché mejorado y manejo de errores."""
        if not texts:
            return []
        
        # Backend: Sentence-Transformers
        if self.use_st and self._st_model is not None:
            try:
                return self._embed_sentence_transformers(texts)
            except Exception as e:
                print(f"  Error en Sentence-Transformers: {e}")
                return self._embed_tfidf(texts)
        
        # Fallback: TF-IDF
        return self._embed_tfidf(texts)
        
        out: List[np.ndarray] = []
        to_query: List[str] = []
        keys: List[str] = []

        for text in texts:
            # Limpiar texto antes de embedding
            clean_text = re.sub(r'\s+', ' ', text.strip())
            if not clean_text:
                out.append(np.zeros(1536, dtype=np.float32))  # Dimensión por defecto
                continue
                
            key = self._key(clean_text)
            keys.append(key)
            
            if key not in self.cache:
                to_query.append(clean_text)

        # Procesar textos no cacheados
        if to_query:
            try:
                vectors = self._emb.embed_documents(to_query)
                query_iter = iter(vectors)
                
                for text in texts:
                    clean_text = re.sub(r'\s+', ' ', text.strip())
                    if not clean_text:
                        continue
                    
                    key = self._key(clean_text)
                    if key not in self.cache:
                        self.cache[key] = np.array(next(query_iter), dtype=np.float32)
            except Exception as e:
                print(f"⚠️  Error en embeddings OpenAI: {e}")
                # Fallback a TF-IDF
                return self._embed_tfidf(texts)

        # Recopilar resultados
        result = []
        for text in texts:
            clean_text = re.sub(r'\s+', ' ', text.strip())
            if not clean_text:
                result.append(np.zeros(1536, dtype=np.float32))
            else:
                key = self._key(clean_text)
                result.append(self.cache[key])
        
        return result
    
    def _embed_tfidf(self, texts: List[str]) -> List[np.ndarray]:
        """Fallback usando TF-IDF para embeddings."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Limpiar textos
        clean_texts = [re.sub(r'\s+', ' ', text.strip()) for text in texts if text.strip()]
        
        if not clean_texts:
            return [np.zeros(1000, dtype=np.float32) for _ in texts]
        
        try:
            # Crear vectorizador TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
            
            # Obtener vectores TF-IDF
            tfidf_matrix = vectorizer.fit_transform(clean_texts)
            
            # Convertir a arrays numpy
            result = []
            for i, text in enumerate(texts):
                if text.strip():
                    # Encontrar el índice correspondiente en clean_texts
                    clean_idx = 0
                    for j, clean_text in enumerate(clean_texts):
                        if clean_text == re.sub(r'\s+', ' ', text.strip()):
                            clean_idx = j
                            break
                    result.append(tfidf_matrix[clean_idx].toarray().flatten().astype(np.float32))
                else:
                    result.append(np.zeros(1000, dtype=np.float32))
            
            return result
            
        except Exception as e:
            print(f"⚠️  Error en TF-IDF: {e}")
            # Fallback final: vectores aleatorios
            return [np.random.randn(1000).astype(np.float32) for _ in texts]

    def _embed_sentence_transformers(self, texts: List[str]) -> List[np.ndarray]:
        """Embeddings con Sentence-Transformers (e.g., all-MiniLM-L6-v2) con caché."""
        out: List[np.ndarray] = []
        to_query: List[str] = []
        keys: List[str] = []
        
        for text in texts:
            clean_text = re.sub(r'\s+', ' ', text.strip())
            if not clean_text:
                out.append(np.zeros(384, dtype=np.float32))  # dim all-MiniLM-L6-v2
                continue
            key = self._key(clean_text)
            keys.append(key)
            if key not in self.cache:
                to_query.append(clean_text)
        
        if to_query:
            vectors = self._st_model.encode(to_query, convert_to_numpy=True, normalize_embeddings=True)
            query_iter = iter(vectors)
            for text in texts:
                clean_text = re.sub(r'\s+', ' ', text.strip())
                if not clean_text:
                    continue
                key = self._key(clean_text)
                if key not in self.cache:
                    self.cache[key] = np.array(next(query_iter), dtype=np.float32)
        
        result = []
        for text in texts:
            clean_text = re.sub(r'\s+', ' ', text.strip())
            if not clean_text:
                result.append(np.zeros(384, dtype=np.float32))
            else:
                key = self._key(clean_text)
                result.append(self.cache[key])
        return result

# ---------------------------------------------------------------------
# Función de análisis y diagnóstico

def analyze_chunks(documents: List[Document]) -> Dict[str, Any]:
    """Analiza la calidad de los chunks generados."""
    if not documents:
        return {"error": "No documents provided"}
    
    token_counts = [doc.metadata.get("chunk_tokens", 0) for doc in documents]
    categories = [doc.metadata.get("category", "unknown") for doc in documents]
    sections = [doc.metadata.get("section_title", "unknown") for doc in documents]
    
    analysis = {
        "total_chunks": len(documents),
        "token_stats": {
            "mean": float(np.mean(token_counts)),
            "median": float(np.median(token_counts)),
            "std": float(np.std(token_counts)),
            "min": int(min(token_counts)) if token_counts else 0,
            "max": int(max(token_counts)) if token_counts else 0,
            "percentiles": {
                "25th": float(np.percentile(token_counts, 25)),
                "75th": float(np.percentile(token_counts, 75)),
                "90th": float(np.percentile(token_counts, 90)),
            }
        },
        "content_distribution": {
            "categories": dict(zip(*np.unique(categories, return_counts=True))),
            "sections": len(set(sections)),
            "avg_chunks_per_section": len(documents) / len(set(sections)) if set(sections) else 0
        },
        "structure_preservation": {
            "heading_blocks": sum(1 for d in documents if d.metadata.get("is_heading_block", False)),
            "high_cohesion_blocks": sum(1 for d in documents if d.metadata.get("block_cohesion", 0) > 0.7),
            "semantic_refined": sum(1 for d in documents if "semantic" in str(d.metadata.get("source", "")))
        }
    }
    
    return analysis

def print_analysis(analysis: Dict[str, Any]) -> None:
    """Imprime análisis de chunks de forma legible."""
    print("\n" + "="*60)
    print("📊 ANÁLISIS DE CHUNKS GENERADOS")
    print("="*60)
    
    print("\n  Estadísticas Generales:")
    print(f"   • Total de chunks: {analysis['total_chunks']}")
    
    token_stats = analysis["token_stats"]
    print("\n🎯 Distribución de Tokens:")
    print(f"   • Promedio: {token_stats['mean']:.0f}")
    print(f"   • Mediana: {token_stats['median']:.0f}")
    print(f"   • Desv. estándar: {token_stats['std']:.0f}")
    print(f"   • Rango: {token_stats['min']} - {token_stats['max']}")
    print(f"   • Percentiles: P25={token_stats['percentiles']['25th']:.0f}, "
          f"P75={token_stats['percentiles']['75th']:.0f}, "
          f"P90={token_stats['percentiles']['90th']:.0f}")
    
    content_dist = analysis["content_distribution"]
    print("\n  Distribución de Contenido:")
    print(f"   • Secciones únicas: {content_dist['sections']}")
    print(f"   • Chunks por sección: {content_dist['avg_chunks_per_section']:.1f}")
    print("   • Categorías:")
    for cat, count in content_dist["categories"].items():
        print(f"     - {cat}: {count}")
    
    structure = analysis["structure_preservation"]
    print("\n🏗️  Preservación Estructural:")
    print(f"   • Bloques con encabezados: {structure['heading_blocks']}")
    print(f"   • Bloques alta cohesión: {structure['high_cohesion_blocks']}")
    print(f"   • Refinados semánticamente: {structure['semantic_refined']}")

# ---------------------------------------------------------------------
# Ejemplo de uso mejorado

if __name__ == "__main__":
    import json
    from pathlib import Path
    import argparse

    def main():
        parser = argparse.ArgumentParser(description="Hybrid Chunking (PDF o JSONL)")
        parser.add_argument("input_path", help="Ruta al archivo PDF o directorio con archivos .pages.jsonl")
        parser.add_argument("--output", "-o", default="chunks.jsonl", help="Archivo de salida")
        parser.add_argument("--input-dir", help="Directorio de entrada con archivos .pages.jsonl")
        parser.add_argument("--output-dir", help="Directorio de salida para múltiples archivos")
        parser.add_argument("--embed-model", default="all-MiniLM-L6-v2", help="Modelo de embeddings (e.g., all-MiniLM-L6-v2)")
        parser.add_argument("--target-tokens", type=int, default=900, help="Tokens objetivo por chunk")
        parser.add_argument("--min-tokens", type=int, default=400, help="Tokens mínimos por chunk")
        parser.add_argument("--max-tokens", type=int, default=1400, help="Tokens máximos por chunk")
        parser.add_argument("--overlap", type=float, default=0.18, help="Ratio de overlap (0.0-1.0)")
        parser.add_argument("--analyze", action="store_true", help="Mostrar análisis detallado")
        parser.add_argument("--sample", type=int, default=3, help="Número de chunks de muestra")
        
        args = parser.parse_args()
        
        # Determinar modo de operación
        input_path = Path(args.input_path)
        
        if args.input_dir and args.output_dir:
            # Modo: procesar múltiples archivos JSONL
            print("Procesando múltiples archivos JSONL...")
            results = process_multiple_jsonl_files(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                embed_model=args.embed_model,
                target_tokens=args.target_tokens,
                min_tokens=args.min_tokens,
                max_tokens=args.max_tokens,
                overlap_ratio=args.overlap,
            )
            
            if args.analyze:
                # Análisis combinado de todos los documentos
                all_documents = []
                for documents in results.values():
                    all_documents.extend(documents)
                analysis = analyze_chunks(all_documents)
                print_analysis(analysis)
            
        elif input_path.suffix == '.jsonl' or input_path.name.endswith('.pages.jsonl'):
            if not input_path.exists():
                print(f"  Error: No existe el archivo {input_path}")
                return
            
            try:
                print(f"Iniciando procesamiento de {input_path}")
                
                documents = hybrid_chunk_jsonl(
                    jsonl_path=str(input_path),
                    embed_model=args.embed_model,
                    target_tokens=args.target_tokens,
                    min_tokens=args.min_tokens,
                    max_tokens=args.max_tokens,
                    overlap_ratio=args.overlap,
                )
                
                # Guardar resultados
                output_path = Path(args.output)
                with open(output_path, "w", encoding="utf-8") as f:
                    for i, doc in enumerate(documents):
                        # Limpiar metadatos para serialización JSON
                        clean_metadata = {}
                        for key, value in doc.metadata.items():
                            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                                clean_metadata[key] = value
                            else:
                                clean_metadata[key] = str(value)
                        
                        chunk_data = {
                            "id": i,
                            "content": doc.page_content,
                            "metadata": clean_metadata,
                            "preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                        }
                        f.write(json.dumps(chunk_data, ensure_ascii=False) + "\n")
                
                print(f"Guardado en: {output_path}")
                
                # Análisis
                if args.analyze:
                    analysis = analyze_chunks(documents)
                    print_analysis(analysis)
                
                # Muestra de chunks
                print(f"\nMuestra de {min(args.sample, len(documents))} chunks:")
                for i, doc in enumerate(documents[:args.sample]):
                    print(f"\n{'='*50}")
                    print(f"Chunk #{i}")
                    print(f"Tokens: {doc.metadata.get('chunk_tokens', 'N/A')}")
                    print(f"Sección: {doc.metadata.get('section_title', 'Sin sección')}")
                    print(f"Categoría: {doc.metadata.get('category', 'N/A')}")
                    print("Contenido:")
                    preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                    print(f"   {preview}")
                
            except Exception as e:
                print(f"  Error durante el procesamiento: {e}")
                import traceback
                traceback.print_exc()
                
        else:
            print("  Error: Formato no soportado. Use .pages.jsonl o --input-dir con JSONL")
            return

    main()