#!/usr/bin/env python3
"""
Estándares de Metadata para Procesadores de Documentos
=====================================================

Este módulo define los estándares de metadata que deben seguir todos los procesadores
de documentos (PDF, TXT, etc.) para mantener consistencia en el pipeline.

Campos estándar: Campos que TODOS los procesadores deben generar
Metadata específica: Campos adicionales específicos de cada tipo de documento
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class DocumentType(Enum):
    """Tipos de documentos estándar"""
    PDF_DOCUMENT = "pdf_document"
    MITRE_ATTACK = "mitre_attack"
    OWASP = "owasp"
    THREAT_INTEL = "threat_intel"
    CVE = "cve"
    GENERAL_TEXT = "general_text"

class Language(Enum):
    """Idiomas soportados"""
    ENGLISH = "english"
    SPANISH = "spanish"
    UNKNOWN = "unknown"

class SourceType(Enum):
    """Tipos de fuente estándar"""
    PDF = "pdf"
    TXT = "txt"
    HTML = "html"
    MARKDOWN = "markdown"

@dataclass
class StandardMetadata:
    """
    Campos estándar que TODOS los procesadores deben generar
    """
    # Identificación del documento
    source_id: str                    # ID único del documento (sin extensión)
    source_type: str                  # Tipo de fuente (pdf, txt, etc.)
    source_file: str                  # Nombre del archivo original
    page_number: int                  # Número de página (1-indexed)
    
    # Contenido
    text: str                         # Texto extraído/limpio
    title: str                        # Título del documento/página
    
    # Rangos de páginas
    page_start: int                   # Página de inicio
    page_end: int                     # Página de fin
    
    # Metadata estándar
    document_type: str                # Tipo de documento detectado
    words: int                        # Número de palabras
    quality_score: float              # Puntuación de calidad (0-100)
    has_structure: bool               # Si tiene estructura (headers, listas, etc.)
    has_identifiers: bool             # Si tiene identificadores especiales
    special_identifiers: List[str]    # Lista de identificadores encontrados
    language: str                     # Idioma detectado
    encoding: str                     # Codificación del archivo

@dataclass
class PDFSpecificMetadata:
    """
    Metadata específica para documentos PDF
    """
    # Información de páginas
    pages: int                        # Total de páginas del documento
    has_images: bool                  # Si la página tiene imágenes
    rotation: int                     # Rotación de la página (0, 90, 180, 270)
    page_dimensions: Dict[str, float] # Dimensiones de la página {width, height}
    
    # Metadata original del PDF
    pdf_title: str                    # Título del PDF
    pdf_author: str                   # Autor del PDF
    pdf_subject: str                  # Asunto del PDF
    pdf_keywords: str                 # Palabras clave del PDF
    pdf_creator: str                  # Creador del PDF
    pdf_producer: str                 # Productor del PDF
    pdf_creation_date: str            # Fecha de creación
    pdf_modification_date: str        # Fecha de modificación
    pdf_format: str                   # Formato del PDF
    pdf_encryption: Optional[Any]     # Información de encriptación

@dataclass
class TXTSpecificMetadata:
    """
    Metadata específica para documentos TXT (con metadata externa)
    """
    # Información de herramientas (de archivos _metadata.json)
    tool_id: Optional[str]            # ID de la herramienta
    tool_name: Optional[str]          # Nombre de la herramienta
    tool_description: Optional[str]   # Descripción de la herramienta
    tool_category: Optional[str]      # Categoría de la herramienta
    tool_urls: List[str]              # URLs relacionadas
    download_date: Optional[str]      # Fecha de descarga
    char_count: Optional[int]         # Conteo de caracteres

class MetadataBuilder:
    """
    Builder para crear metadata estándar de manera consistente
    """
    
    @staticmethod
    def create_standard_metadata(
        source_id: str,
        source_type: str,
        source_file: str,
        page_number: int,
        text: str,
        title: str,
        page_start: int,
        page_end: int,
        document_type: str,
        words: int,
        quality_score: float,
        has_structure: bool,
        has_identifiers: bool,
        special_identifiers: List[str],
        language: str,
        encoding: str
    ) -> Dict[str, Any]:
        """
        Crea metadata estándar con validación
        """
        # Validar tipos
        if not isinstance(special_identifiers, list):
            special_identifiers = []
        
        if quality_score < 0 or quality_score > 100:
            raise ValueError(f"quality_score debe estar entre 0 y 100, recibido: {quality_score}")
        
        return {
            # Campos principales
            "source_id": source_id,
            "source_type": source_type,
            "source_file": source_file,
            "page_number": page_number,
            "text": text,
            "title": title,
            "page_start": page_start,
            "page_end": page_end,
            
            # Metadata estándar
            "document_type": document_type,
            "words": words,
            "quality_score": quality_score,
            "has_structure": has_structure,
            "has_identifiers": has_identifiers,
            "special_identifiers": special_identifiers[:10],  # Limitar a 10
            "language": language,
            "encoding": encoding
        }
    
    @staticmethod
    def add_pdf_metadata(
        base_metadata: Dict[str, Any],
        pages: int,
        has_images: bool = False,
        rotation: int = 0,
        page_dimensions: Optional[Dict[str, float]] = None,
        pdf_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Agrega metadata específica de PDF a metadata base
        """
        pdf_specific = {
            "pages": pages,
            "has_images": has_images,
            "rotation": rotation,
            "page_dimensions": page_dimensions or {"width": 0.0, "height": 0.0},
            "pdf_title": "",
            "pdf_author": "",
            "pdf_subject": "",
            "pdf_keywords": "",
            "pdf_creator": "",
            "pdf_producer": "",
            "pdf_creation_date": "",
            "pdf_modification_date": "",
            "pdf_format": "",
            "pdf_encryption": None
        }
        
        if pdf_metadata:
            pdf_specific.update({
                "pdf_title": pdf_metadata.get("title", ""),
                "pdf_author": pdf_metadata.get("author", ""),
                "pdf_subject": pdf_metadata.get("subject", ""),
                "pdf_keywords": pdf_metadata.get("keywords", ""),
                "pdf_creator": pdf_metadata.get("creator", ""),
                "pdf_producer": pdf_metadata.get("producer", ""),
                "pdf_creation_date": pdf_metadata.get("creation_date", ""),
                "pdf_modification_date": pdf_metadata.get("modification_date", ""),
                "pdf_format": pdf_metadata.get("format", ""),
                "pdf_encryption": pdf_metadata.get("encryption")
            })
        
        return {**base_metadata, **pdf_specific}
    
    @staticmethod
    def add_txt_metadata(
        base_metadata: Dict[str, Any],
        external_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Agrega metadata específica de TXT a metadata base
        """
        txt_specific = {
            "tool_id": None,
            "tool_name": None,
            "tool_description": None,
            "tool_category": None,
            "tool_urls": [],
            "download_date": None,
            "char_count": None
        }
        
        if external_metadata:
            txt_specific.update({
                "tool_id": external_metadata.get("tool_id"),
                "tool_name": external_metadata.get("name"),
                "tool_description": external_metadata.get("description"),
                "tool_category": external_metadata.get("category"),
                "tool_urls": external_metadata.get("urls", []),
                "download_date": external_metadata.get("download_date"),
                "char_count": external_metadata.get("char_count")
            })
        
        return {**base_metadata, **txt_specific}

def validate_metadata(metadata: Dict[str, Any]) -> List[str]:
    """
    Valida que la metadata tenga todos los campos estándar requeridos
    
    Returns:
        Lista de errores encontrados (vacía si todo está bien)
    """
    errors = []
    
    # Campos requeridos
    required_fields = [
        "source_id", "source_type", "source_file", "page_number",
        "text", "title", "page_start", "page_end", "document_type",
        "words", "quality_score", "has_structure", "has_identifiers",
        "special_identifiers", "language", "encoding"
    ]
    
    for field in required_fields:
        if field not in metadata:
            errors.append(f"Campo requerido faltante: {field}")
    
    # Validaciones específicas
    if "quality_score" in metadata:
        score = metadata["quality_score"]
        if not isinstance(score, (int, float)) or score < 0 or score > 100:
            errors.append(f"quality_score debe ser un número entre 0 y 100, recibido: {score}")
    
    if "special_identifiers" in metadata:
        if not isinstance(metadata["special_identifiers"], list):
            errors.append("special_identifiers debe ser una lista")
    
    if "words" in metadata:
        if not isinstance(metadata["words"], int) or metadata["words"] < 0:
            errors.append(f"words debe ser un entero no negativo, recibido: {metadata['words']}")
    
    return errors

def get_metadata_schema() -> Dict[str, Any]:
    """
    Retorna el esquema completo de metadata para documentación
    """
    return {
        "standard_fields": {
            "description": "Campos que TODOS los procesadores deben generar",
            "fields": {
                "source_id": "ID único del documento (sin extensión)",
                "source_type": "Tipo de fuente (pdf, txt, etc.)",
                "source_file": "Nombre del archivo original",
                "page_number": "Número de página (1-indexed)",
                "text": "Text extraído/limpio",
                "title": "Título del documento/página",
                "page_start": "Página de inicio",
                "page_end": "Página de fin",
                "document_type": "Tipo de documento detectado",
                "words": "Número de palabras",
                "quality_score": "Puntuación de calidad (0-100)",
                "has_structure": "Si tiene estructura (headers, listas, etc.)",
                "has_identifiers": "Si tiene identificadores especiales",
                "special_identifiers": "Lista de identificadores encontrados",
                "language": "Idioma detectado",
                "encoding": "Codificación del archivo"
            }
        },
        "pdf_specific_fields": {
            "description": "Campos adicionales específicos para PDFs",
            "fields": {
                "pages": "Total de páginas del documento",
                "has_images": "Si la página tiene imágenes",
                "rotation": "Rotación de la página (0, 90, 180, 270)",
                "page_dimensions": "Dimensiones de la página {width, height}",
                "pdf_title": "Título del PDF",
                "pdf_author": "Autor del PDF",
                "pdf_subject": "Asunto del PDF",
                "pdf_keywords": "Palabras clave del PDF",
                "pdf_creator": "Creador del PDF",
                "pdf_producer": "Productor del PDF",
                "pdf_creation_date": "Fecha de creación",
                "pdf_modification_date": "Fecha de modificación",
                "pdf_format": "Formato del PDF",
                "pdf_encryption": "Información de encriptación"
            }
        },
        "txt_specific_fields": {
            "description": "Campos adicionales específicos para TXTs con metadata externa",
            "fields": {
                "tool_id": "ID de la herramienta",
                "tool_name": "Nombre de la herramienta",
                "tool_description": "Descripción de la herramienta",
                "tool_category": "Categoría de la herramienta",
                "tool_urls": "URLs relacionadas",
                "download_date": "Fecha de descarga",
                "char_count": "Conteo de caracteres"
            }
        }
    }

# Ejemplo de uso para documentación
if __name__ == "__main__":
    print("=== ESTÁNDARES DE METADATA ===")
    print()
    
    # Mostrar esquema
    schema = get_metadata_schema()
    
    print("CAMPOS ESTÁNDAR (todos los procesadores):")
    for field, description in schema["standard_fields"]["fields"].items():
        print(f"  {field}: {description}")
    
    print("\nCAMPOS ESPECÍFICOS DE PDF:")
    for field, description in schema["pdf_specific_fields"]["fields"].items():
        print(f"  {field}: {description}")
    
    print("\nCAMPOS ESPECÍFICOS DE TXT:")
    for field, description in schema["txt_specific_fields"]["fields"].items():
        print(f"  {field}: {description}")
    
    print("\n=== EJEMPLO DE USO ===")
    
    # Ejemplo de metadata estándar
    standard_meta = MetadataBuilder.create_standard_metadata(
        source_id="ejemplo",
        source_type="pdf",
        source_file="ejemplo.pdf",
        page_number=1,
        text="Contenido del documento...",
        title="Título del documento",
        page_start=1,
        page_end=1,
        document_type="pdf_document",
        words=100,
        quality_score=85.5,
        has_structure=True,
        has_identifiers=False,
        special_identifiers=[],
        language="spanish",
        encoding="utf-8"
    )
    
    print("Metadata estándar creada:", len(standard_meta), "campos")
    
    # Validar metadata
    errors = validate_metadata(standard_meta)
    if errors:
        print("Errores encontrados:", errors)
    else:
        print("✅ Metadata válida")
