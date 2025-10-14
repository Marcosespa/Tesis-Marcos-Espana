#!/usr/bin/env python3
"""
Procesador simple de PDFs - Solo extracción, limpieza y metadata
Sin chunking - enfocado en preparar texto limpio para procesamiento posterior
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import fitz  # PyMuPDF
import unicodedata
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from metadata_standards import MetadataBuilder, validate_metadata

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PDFTextCleaner:
    """Limpiador especializado para texto extraído de PDFs"""
    
    def __init__(self):
        # Patrones para eliminar headers/footers comunes
        self.header_footer_patterns = [
            r'^\s*\d+\s*$',  # Solo números de página
            r'^\s*Page\s+\d+.*$',  # "Page X" o "Page X of Y"
            r'^\s*\d+\s*/\s*\d+\s*$',  # "X / Y"
            r'^\s*\d+\s+of\s+\d+\s*$',  # "X of Y"
            r'^\s*\[\s*\d+\s*\]\s*$',  # [X]
            r'^\s*-\s*\d+\s*-\s*$',  # - X -
            r'(?i)^\s*(copyright|©).*$',  # Copyright
            r'(?i)^\s*(confidential|proprietary).*$',  # Confidential/Proprietary
            r'(?i)^\s*this\s+page\s+(intentionally\s+)?left\s+blank.*$',  # Página en blanco
        ]
        
        # Patrones para limpiar ruido general
        self.noise_patterns = [
            r'\f',  # Form feeds
            r'\x0c',  # Page breaks
            r'\.{4,}',  # 4 o más puntos consecutivos
            r'-{4,}',  # 4 o más guiones consecutivos
            r'={4,}',  # 4 o más signos igual consecutivos
            r'_{4,}',  # 4 o más guiones bajos consecutivos
            r'\s{3,}',  # 3 o más espacios consecutivos
            r'^\s*\*+\s*$',  # Líneas solo con asteriscos
            r'^\s*#+\s*$',  # Líneas solo con hash
            r'^\s*=+\s*$',  # Líneas solo con signos igual
            r'^\s*-+\s*$',  # Líneas solo con guiones (si es toda la línea)
        ]
        
        # Patrones para detectar tabla de contenidos, índices, etc.
        self.toc_patterns = [
            r'(?i)^\s*(table\s+of\s+contents?|contents?|índice|indice)\s*\.{0,}.*$',
            r'(?i)^\s*(chapter|capítulo|section|sección)\s+\d+.*\.{3,}.*\d+\s*$',
            r'^\s*.+\.{5,}.*\d+\s*$',  # Líneas con muchos puntos y número al final
            r'(?i)^\s*(appendix|apéndice|anexo)\s+[a-z]\s*\.{0,}.*\d+\s*$',
        ]
        
        # Patrones para referencias bibliográficas
        self.reference_patterns = [
            r'(?i)^\s*(references?|bibliography|bibliografía|referencias?)\s*$',
            r'^\s*\[\d+\]\s+.*$',  # [1] Autor, etc.
            r'^\s*\d+\.\s+[A-Z][a-z]+,.*\(\d{4}\).*$',  # 1. Autor, (2023)
        ]
    
    def clean_text(self, text: str, remove_toc: bool = True, remove_references: bool = False) -> str:
        """
        Limpia el texto extraído del PDF
        
        Args:
            text: Texto a limpiar
            remove_toc: Si eliminar tabla de contenidos
            remove_references: Si eliminar sección de referencias
        """
        if not text or not text.strip():
            return ""
        
        # Normalizar Unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Dividir en líneas para procesamiento línea por línea
        lines = text.split('\n')
        cleaned_lines = []
        
        in_toc_section = False
        in_references_section = False
        
        for line_num, line in enumerate(lines):
            original_line = line
            
            # Detectar inicio de tabla de contenidos
            if remove_toc and any(re.match(pattern, line.strip()) for pattern in self.toc_patterns[:1]):  # Solo el primer patrón
                in_toc_section = True
                continue
            
            # Detectar fin de tabla de contenidos (línea con contenido normal)
            if in_toc_section:
                if (len(line.strip()) > 0 and 
                    not any(re.match(pattern, line.strip()) for pattern in self.toc_patterns) and
                    not re.search(r'\.{3,}', line) and  # No tiene puntos de relleno
                    len(line.split()) > 5):  # Línea con contenido sustancial
                    in_toc_section = False
                else:
                    continue
            
            # Detectar sección de referencias
            if remove_references and any(re.match(pattern, line.strip()) for pattern in self.reference_patterns[:1]):
                in_references_section = True
                continue
            
            if in_references_section:
                continue
            
            # Eliminar headers/footers
            if any(re.match(pattern, line.strip()) for pattern in self.header_footer_patterns):
                continue
            
            # Limpiar ruido general
            for pattern in self.noise_patterns:
                line = re.sub(pattern, '', line, flags=re.MULTILINE)
            
            # Limpiar espacios y verificar si la línea tiene contenido
            line = ' '.join(line.split())
            
            # Solo mantener líneas con contenido significativo
            if line.strip() and len(line.strip()) > 2:
                cleaned_lines.append(line)
        
        # Unir líneas y hacer limpieza final
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Limpieza final de espacios excesivos
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)  # Máximo 2 saltos de línea
        cleaned_text = re.sub(r' {2,}', ' ', cleaned_text)  # Máximo 1 espacio
        
        return cleaned_text.strip()
    
    def extract_document_structure(self, text: str) -> Dict[str, Any]:
        """Extrae información estructural del documento"""
        lines = text.split('\n')
        
        structure = {
            "has_title": False,
            "title_candidate": "",
            "has_headers": False,
            "potential_headers": [],
            "has_bullet_points": False,
            "has_numbered_lists": False,
            "paragraph_count": 0,
            "avg_line_length": 0,
            "language_indicators": {}
        }
        
        # Buscar título (primeras líneas cortas y centrales)
        for i, line in enumerate(lines[:5]):
            if line.strip() and len(line.split()) <= 10 and line.isupper():
                structure["has_title"] = True
                structure["title_candidate"] = line.strip()
                break
            elif line.strip() and len(line.split()) <= 15 and i <= 2:
                structure["title_candidate"] = line.strip()
        
        # Analizar estructura
        line_lengths = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            line_length = len(line.split())
            line_lengths.append(line_length)
            
            # Detectar headers (líneas cortas seguidas de líneas más largas)
            if (line_length <= 8 and 
                not line.endswith('.') and 
                line[0].isupper() and
                not line.startswith(('•', '-', '*', '1.', '2.', '3.'))):
                structure["potential_headers"].append(line)
            
            # Detectar listas
            if line.startswith(('•', '◦', '▪', '‣', '-', '*')):
                structure["has_bullet_points"] = True
            elif re.match(r'^\d+\.', line):
                structure["has_numbered_lists"] = True
        
        # Contar párrafos (líneas con punto final)
        structure["paragraph_count"] = sum(1 for line in lines if line.strip().endswith('.'))
        structure["has_headers"] = len(structure["potential_headers"]) > 0
        structure["avg_line_length"] = sum(line_lengths) / len(line_lengths) if line_lengths else 0
        
        # Indicadores básicos de idioma
        text_lower = text.lower()
        structure["language_indicators"] = {
            "spanish_words": sum(1 for word in ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no'] 
                               if word in text_lower),
            "english_words": sum(1 for word in ['the', 'of', 'and', 'to', 'a', 'in', 'is', 'it', 'you', 'that'] 
                               if word in text_lower)
        }
        
        return structure

class SimplePDFProcessor:
    """Procesador simple de PDFs enfocado en extracción y limpieza"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            "clean_text": True,
            "remove_toc": True,
            "remove_references": False,
            "min_page_chars": 100,
            "extract_images_info": True,
            "detect_language": True
        }
        self.cleaner = PDFTextCleaner()
    
    def process_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Procesa un archivo PDF completo
        
        Returns:
            Dict con toda la información del documento
        """
        logger.info(f"🔄 Procesando: {pdf_path.name}")
        
        try:
            # Configurar manejo de errores de MuPDF
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning, module="fitz")
            
            doc = fitz.open(pdf_path)
            
            # Información básica del documento
            doc_info = {
                "file_info": {
                    "filename": pdf_path.name,
                    "file_path": str(pdf_path),
                    "file_size_bytes": pdf_path.stat().st_size,
                    "processing_timestamp": datetime.now().isoformat()
                },
                "document_metadata": self._extract_pdf_metadata(doc),
                "document_structure": {
                    "total_pages": len(doc),
                    "valid_pages": 0,
                    "empty_pages": 0,
                    "pages_with_images": 0,
                    "total_images": 0
                },
                "content": {
                    "full_text": "",
                    "pages": []
                },
                "text_statistics": {},
                "quality_metrics": {}
            }
            
            # Procesar cada página
            all_page_texts = []
            
            for page_num in range(len(doc)):
                page_data = self._process_page(doc[page_num], page_num + 1)
                
                if page_data["is_valid"]:
                    doc_info["document_structure"]["valid_pages"] += 1
                    doc_info["content"]["pages"].append(page_data)
                    all_page_texts.append(page_data["text"])
                else:
                    doc_info["document_structure"]["empty_pages"] += 1
                
                if page_data["images_count"] > 0:
                    doc_info["document_structure"]["pages_with_images"] += 1
                    doc_info["document_structure"]["total_images"] += page_data["images_count"]
            
            doc.close()
            
            # Combinar todo el texto del documento
            full_text = "\n\n".join(all_page_texts)
            
            # Aplicar limpieza global si está configurada
            if self.config.get("clean_text", True):
                full_text = self.cleaner.clean_text(
                    full_text, 
                    remove_toc=self.config.get("remove_toc", True),
                    remove_references=self.config.get("remove_references", False)
                )
            
            doc_info["content"]["full_text"] = full_text
            
            # Calcular estadísticas del texto
            doc_info["text_statistics"] = self._calculate_text_statistics(full_text)
            
            # Extraer estructura del documento
            doc_info["document_analysis"] = self.cleaner.extract_document_structure(full_text)
            
            # Métricas de calidad
            doc_info["quality_metrics"] = self._calculate_quality_metrics(doc_info)
            
            logger.info(f"✅ {pdf_path.name}: {doc_info['document_structure']['valid_pages']} páginas válidas")
            return doc_info
            
        except Exception as e:
            logger.error(f"❌ Error procesando {pdf_path.name}: {e}")
            return self._create_error_document(pdf_path, str(e))
    
    def _process_page(self, page, page_num: int) -> Dict[str, Any]:
        """Procesa una página individual con manejo robusto de errores"""
        try:
            # Extraer texto con manejo de errores
            try:
                raw_text = page.get_text()
            except Exception as e:
                logger.warning(f"Error extrayendo texto de página {page_num}: {e}")
                raw_text = ""
            
            # Información de la página con manejo de errores
            try:
                images_count = len(page.get_images())
            except Exception as e:
                logger.warning(f"Error obteniendo imágenes de página {page_num}: {e}")
                images_count = 0
            
            try:
                page_rect = page.rect
                width = float(page_rect.width)
                height = float(page_rect.height)
            except Exception as e:
                logger.warning(f"Error obteniendo dimensiones de página {page_num}: {e}")
                width = height = 0
            
            try:
                rotation = page.rotation
            except Exception as e:
                logger.warning(f"Error obteniendo rotación de página {page_num}: {e}")
                rotation = 0
            
            # Información de la página
            page_info = {
                "page_number": page_num,
                "raw_text": raw_text,
                "text": "",
                "is_valid": False,
                "char_count": len(raw_text),
                "word_count": 0,
                "images_count": images_count,
                "page_dimensions": {
                    "width": width,
                    "height": height
                },
                "metadata": {
                    "has_images": images_count > 0,
                    "rotation": rotation
                }
            }
            
            # Limpiar texto de la página
            if self.config.get("clean_text", True):
                cleaned_text = self.cleaner.clean_text(raw_text, remove_toc=False, remove_references=False)
            else:
                cleaned_text = raw_text.strip()
            
            page_info["text"] = cleaned_text
            page_info["word_count"] = len(cleaned_text.split()) if cleaned_text else 0
            
            # Determinar si la página es válida
            min_chars = self.config.get("min_page_chars", 100)
            page_info["is_valid"] = len(cleaned_text) >= min_chars
            
            return page_info
            
        except Exception as e:
            logger.error(f"Error crítico procesando página {page_num}: {e}")
            # Retornar página vacía en caso de error crítico
            return {
                "page_number": page_num,
                "raw_text": "",
                "text": "",
                "is_valid": False,
                "char_count": 0,
                "word_count": 0,
                "images_count": 0,
                "page_dimensions": {"width": 0, "height": 0},
                "metadata": {"has_images": False, "rotation": 0}
            }
    
    def _extract_pdf_metadata(self, doc) -> Dict[str, Any]:
        """Extrae metadata del PDF"""
        metadata = doc.metadata
        
        return {
            "title": metadata.get("title", "").strip(),
            "author": metadata.get("author", "").strip(),
            "subject": metadata.get("subject", "").strip(),
            "keywords": metadata.get("keywords", "").strip(),
            "creator": metadata.get("creator", "").strip(),
            "producer": metadata.get("producer", "").strip(),
            "creation_date": metadata.get("creationDate", ""),
            "modification_date": metadata.get("modDate", ""),
            "format": metadata.get("format", ""),
            "encryption": metadata.get("encryption", None),
            "pages": len(doc)
        }
    
    def _calculate_text_statistics(self, text: str) -> Dict[str, Any]:
        """Calcula estadísticas básicas del texto"""
        if not text:
            return {
                "total_chars": 0,
                "total_words": 0,
                "total_sentences": 0,
                "total_paragraphs": 0,
                "avg_words_per_sentence": 0,
                "avg_chars_per_word": 0
            }
        
        # Estadísticas básicas
        total_chars = len(text)
        words = text.split()
        total_words = len(words)
        
        # Contar oraciones (aproximado)
        sentences = re.split(r'[.!?]+', text)
        total_sentences = len([s for s in sentences if s.strip()])
        
        # Contar párrafos
        paragraphs = text.split('\n\n')
        total_paragraphs = len([p for p in paragraphs if p.strip()])
        
        return {
            "total_chars": total_chars,
            "total_words": total_words,
            "total_sentences": total_sentences,
            "total_paragraphs": total_paragraphs,
            "avg_words_per_sentence": total_words / total_sentences if total_sentences > 0 else 0,
            "avg_chars_per_word": total_chars / total_words if total_words > 0 else 0,
            "text_density": total_words / (total_chars / 1000) if total_chars > 0 else 0  # palabras per 1000 chars
        }
    
    def _calculate_quality_metrics(self, doc_info: Dict) -> Dict[str, Any]:
        """Calcula métricas de calidad del documento"""
        structure = doc_info["document_structure"]
        stats = doc_info["text_statistics"]
        
        # Ratio de páginas válidas
        valid_page_ratio = (structure["valid_pages"] / structure["total_pages"] 
                           if structure["total_pages"] > 0 else 0)
        
        # Densidad de contenido
        content_density = stats.get("text_density", 0)
        
        # Puntuación de calidad simple (0-100)
        quality_score = 0
        
        # Puntos por páginas válidas (0-40 puntos)
        quality_score += valid_page_ratio * 40
        
        # Puntos por longitud del documento (0-30 puntos)
        word_count = stats.get("total_words", 0)
        if word_count > 1000:
            quality_score += min(30, (word_count / 1000) * 10)
        
        # Puntos por densidad de contenido (0-20 puntos)
        if content_density > 200:  # Palabras por 1000 caracteres
            quality_score += min(20, (content_density / 300) * 20)
        
        # Puntos por estructura (0-10 puntos)
        if doc_info["document_analysis"]["has_headers"]:
            quality_score += 5
        if doc_info["document_analysis"]["has_title"]:
            quality_score += 5
        
        return {
            "quality_score": min(100, quality_score),
            "valid_page_ratio": valid_page_ratio,
            "content_density": content_density,
            "has_structure": (doc_info["document_analysis"]["has_headers"] or 
                            doc_info["document_analysis"]["has_title"]),
            "is_substantial": word_count > 500
        }
    
    def _create_error_document(self, pdf_path: Path, error_msg: str) -> Dict[str, Any]:
        """Crea estructura de documento para errores"""
        return {
            "file_info": {
                "filename": pdf_path.name,
                "file_path": str(pdf_path),
                "file_size_bytes": pdf_path.stat().st_size if pdf_path.exists() else 0,
                "processing_timestamp": datetime.now().isoformat()
            },
            "error": True,
            "error_message": error_msg,
            "document_metadata": {},
            "document_structure": {"total_pages": 0, "valid_pages": 0},
            "content": {"full_text": "", "pages": []},
            "text_statistics": {},
            "quality_metrics": {"quality_score": 0}
        }
    
    def save_document(self, doc_data: Dict, output_path: Path):
        """Guarda documento procesado en formato JSON"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(doc_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📁 Documento guardado: {output_path}")

def process_pdf_directory(pdf_dir: Path, output_dir: Path, config: Dict = None) -> Dict[str, Any]:
    """
    Procesa todos los PDFs en un directorio
    
    Args:
        pdf_dir: Directorio con archivos PDF
        output_dir: Directorio de salida
        config: Configuración del procesador
    
    Returns:
        Dict con estadísticas del procesamiento
    """
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {pdf_dir}")
    
    # Crear procesador
    processor = SimplePDFProcessor(config)
    
    # Encontrar archivos PDF
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No se encontraron archivos PDF en: {pdf_dir}")
        return {"error": "No PDF files found"}
    
    logger.info(f"📚 Encontrados {len(pdf_files)} archivos PDF")
    
    # Crear directorio de salida
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Procesar archivos
    results = {
        "processing_summary": {
            "total_files": len(pdf_files),
            "successful_files": 0,
            "failed_files": 0,
            "total_pages": 0,
            "total_valid_pages": 0,
            "total_words": 0,
            "start_time": datetime.now().isoformat()
        },
        "files": []
    }
    
    all_documents = []
    
    for pdf_file in pdf_files:
        try:
            # Procesar PDF
            doc_data = processor.process_pdf(pdf_file)
            
            # Guardar documento individual
            output_file = output_dir / f"{pdf_file.stem}.json"
            processor.save_document(doc_data, output_file)
            
            # Actualizar estadísticas
            if not doc_data.get("error", False):
                results["processing_summary"]["successful_files"] += 1
                results["processing_summary"]["total_pages"] += doc_data["document_structure"]["total_pages"]
                results["processing_summary"]["total_valid_pages"] += doc_data["document_structure"]["valid_pages"]
                results["processing_summary"]["total_words"] += doc_data["text_statistics"].get("total_words", 0)
            else:
                results["processing_summary"]["failed_files"] += 1
            
            # Info del archivo
            file_info = {
                "filename": pdf_file.name,
                "status": "success" if not doc_data.get("error", False) else "failed",
                "pages": doc_data["document_structure"]["total_pages"],
                "valid_pages": doc_data["document_structure"]["valid_pages"],
                "words": doc_data["text_statistics"].get("total_words", 0),
                "quality_score": doc_data["quality_metrics"].get("quality_score", 0),
                "output_file": str(output_file)
            }
            
            results["files"].append(file_info)
            all_documents.append(doc_data)
            
        except Exception as e:
            logger.error(f"Error procesando {pdf_file.name}: {e}")
            results["processing_summary"]["failed_files"] += 1
            results["files"].append({
                "filename": pdf_file.name,
                "status": "failed",
                "error": str(e)
            })
    
    # Guardar resumen general
    results["processing_summary"]["end_time"] = datetime.now().isoformat()
    
    summary_file = output_dir / "processing_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Crear archivo JSONL con todos los textos limpios (para procesamiento posterior)
    texts_file = output_dir / "all_documents.jsonl"
    with open(texts_file, 'w', encoding='utf-8') as f:
        for doc in all_documents:
            if not doc.get("error", False) and doc["content"]["full_text"]:
                # Formato simplificado para procesamiento posterior
                simple_doc = {
                    "source_id": Path(doc["file_info"]["filename"]).stem,
                    "filename": doc["file_info"]["filename"],
                    "text": doc["content"]["full_text"],
                    "metadata": {
                        **doc["document_metadata"],
                        "pages": doc["document_structure"]["total_pages"],
                        "words": doc["text_statistics"]["total_words"],
                        "quality_score": doc["quality_metrics"]["quality_score"],
                        "has_structure": doc["quality_metrics"]["has_structure"]
                    }
                }
                json.dump(simple_doc, f, ensure_ascii=False)
                f.write('\n')

    # Crear pages.jsonl con páginas individuales (para chunking posterior)
    pages_file = output_dir / "pages.jsonl"
    with open(pages_file, 'w', encoding='utf-8') as f:
        for doc in all_documents:
            if not doc.get("error", False) and doc["content"]["pages"]:
                for page in doc["content"]["pages"]:
                    if not page.get("is_valid", False):
                        continue
                    # Usar título del PDF o fallback al nombre del archivo
                    title = (
                        doc["document_metadata"].get("title", "").strip()
                        or doc["file_info"]["filename"]
                    )
                    # Crear metadata estándar
                    standard_metadata = MetadataBuilder.create_standard_metadata(
                        source_id=Path(doc["file_info"]["filename"]).stem,
                        source_type="pdf",
                        source_file=doc["file_info"]["filename"],
                        page_number=page["page_number"],
                        text=page["text"],
                        title=title,
                        page_start=page["page_number"],
                        page_end=page["page_number"],
                        document_type="pdf_document",
                        words=page["word_count"],
                        quality_score=doc["quality_metrics"]["quality_score"],
                        has_structure=doc["quality_metrics"]["has_structure"],
                        has_identifiers=False,  # PDFs no tienen identificadores especiales por defecto
                        special_identifiers=[],
                        language="unknown",  # Se podría detectar con análisis de texto
                        encoding="utf-8"
                    )
                    
                    # Agregar metadata específica de PDF
                    page_data = {
                        **standard_metadata,
                        "metadata": MetadataBuilder.add_pdf_metadata(
                            standard_metadata,
                            pages=doc["document_structure"]["total_pages"],
                            has_images=page["metadata"].get("has_images", False),
                            rotation=page["metadata"].get("rotation", 0),
                            page_dimensions=page["page_dimensions"],
                            pdf_metadata=doc["document_metadata"]
                        )
                    }
                    json.dump(page_data, f, ensure_ascii=False)
                    f.write('\n')
    
    logger.info(f"🎉 Procesamiento completado!")
    logger.info(f"   ✅ Exitosos: {results['processing_summary']['successful_files']}")
    logger.info(f"   ❌ Fallidos: {results['processing_summary']['failed_files']}")
    logger.info(f"   📄 Páginas válidas: {results['processing_summary']['total_valid_pages']}")
    logger.info(f"   📝 Total palabras: {results['processing_summary']['total_words']}")
    logger.info(f"📋 Páginas listas para chunking en: {pages_file}")
    
    return results

def check_existing_files(output_dir: Path) -> bool:
    """Verifica si ya existen archivos en el directorio de salida"""
    if not output_dir.exists():
        return False
    
    pages_file = output_dir / "pages.jsonl"
    if pages_file.exists():
        return True
    return False

def process_pdf_sources(force_overwrite: bool = False):
    """Procesa PDFs de NIST, OAPEN y USENIX"""
    base_dir = Path("/Users/marcosespana/Desktop/U/DatosTesis")
    
    # Configuración
    config = {
        "clean_text": True,
        "remove_toc": True,
        "remove_references": False, #True si quieres que no tenga referencias
        "min_page_chars": 100,
        "extract_images_info": True
    }
    
    # Fuentes a procesar
    sources = ["NIST", "OAPEN_PDFs", "USENIX"]
    total_processed = 0
    
    for source in sources:
        print(f"\n{'='*50}")
        print(f"PROCESANDO {source}")
        print(f"{'='*50}")
        
        # Para NIST y OAPEN_PDFs, procesar subdirectorios
        if source in ["NIST", "OAPEN_PDFs"]:
            base_pdf_dir = base_dir / "data" / "raw" / source
            base_output_dir = base_dir / "data" / "interim" / source
            
            if not base_pdf_dir.exists():
                print(f"⚠️  Directorio no encontrado: {base_pdf_dir}")
                continue
            
            # Buscar subdirectorios
            subdirs = [d for d in base_pdf_dir.iterdir() if d.is_dir()]
            if not subdirs:
                print(f"⚠️  No se encontraron subdirectorios en: {base_pdf_dir}")
                continue
            
            print(f"📁 Encontrados {len(subdirs)} subdirectorios en {source}")
            
            for subdir in subdirs:
                print(f"\n  📂 Procesando subdirectorio: {subdir.name}")
                
                pdf_dir = subdir
                output_dir = base_output_dir / subdir.name
                
                # Verificar que hay PDFs en el subdirectorio
                pdf_files = list(pdf_dir.glob("*.pdf"))
                if not pdf_files:
                    print(f"    ⚠️  No se encontraron PDFs en: {pdf_dir}")
                    continue
                    
                print(f"    📚 Encontrados {len(pdf_files)} PDFs en {subdir.name}")
                
                # Verificar archivos existentes
                if check_existing_files(output_dir) and not force_overwrite:
                    print(f"    ⚠️  Ya existen archivos en {output_dir}")
                    print(f"       Usa --force para sobrescribir o elimina manualmente los archivos")
                    continue
                    
                # Procesar directorio
                results = process_pdf_directory(pdf_dir, output_dir, config)
                
                if "error" not in results:
                    total_processed += results["processing_summary"]["successful_files"]
                    print(f"    ✅ {subdir.name}: {results['processing_summary']['successful_files']} PDFs procesados")
                else:
                    print(f"    ❌ {subdir.name}: Error en procesamiento")
        
        else:
            # Para USENIX, procesar directamente
            pdf_dir = base_dir / "data" / "raw" / source
            output_dir = base_dir / "data" / "interim" / source
            
            if not pdf_dir.exists():
                print(f"⚠️  Directorio no encontrado: {pdf_dir}")
                continue
                
            # Verificar que hay PDFs
            pdf_files = list(pdf_dir.glob("*.pdf"))
            if not pdf_files:
                print(f"⚠️  No se encontraron PDFs en: {pdf_dir}")
                continue
                
            print(f"📚 Encontrados {len(pdf_files)} PDFs en {source}")
            
            # Verificar archivos existentes
            if check_existing_files(output_dir) and not force_overwrite:
                print(f"⚠️  Ya existen archivos en {output_dir}")
                print(f"   Usa --force para sobrescribir o elimina manualmente los archivos")
                continue
                
            # Procesar directorio
            results = process_pdf_directory(pdf_dir, output_dir, config)
            
            if "error" not in results:
                total_processed += results["processing_summary"]["successful_files"]
                print(f"✅ {source}: {results['processing_summary']['successful_files']} PDFs procesados")
            else:
                print(f"❌ {source}: Error en procesamiento")
    
    return total_processed

def main():
    """Función principal"""
    import argparse
    parser = argparse.ArgumentParser(description="Procesar PDFs de NIST, OAPEN y USENIX")
    parser.add_argument("--force", action="store_true", help="Sobrescribir archivos existentes")
    args = parser.parse_args()
    
    total_processed = process_pdf_sources(force_overwrite=args.force)
    
    print("\n" + "="*50)
    print("RESUMEN DEL PROCESAMIENTO")
    print("="*50)
    print(f"📁 Total de PDFs procesados: {total_processed}")
    print(f"📂 Archivos guardados en: data/interim/")
    print(f"📋 Páginas listas para chunking en: data/interim/*/pages.jsonl")

if __name__ == "__main__":
    main()