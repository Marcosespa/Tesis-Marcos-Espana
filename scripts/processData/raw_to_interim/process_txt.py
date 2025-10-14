#!/usr/bin/env python3
"""
Procesador simple de archivos TXT - Solo extracción, limpieza y metadata
Sin chunking - enfocado en preparar texto limpio para procesamiento posterior
DIRECTORIOS: AISecKG| AnnoCTR |MITRE |OWASP| SECURITY TOOLS 
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import unicodedata
import chardet
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from metadata_standards import MetadataBuilder, validate_metadata

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('txt_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TextFileCleaner:
    """Limpiador especializado para archivos de texto"""
    
    def __init__(self):
        # Patrones para eliminar headers/footers y metadata común
        self.header_footer_patterns = [
            r'^\s*={3,}\s*$',  # Líneas de separación con =
            r'^\s*-{3,}\s*$',  # Líneas de separación con -
            r'^\s*\*{3,}\s*$',  # Líneas de separación con *
            r'^\s*#{3,}\s*$',  # Líneas de separación con #
            r'^\s*Page\s+\d+.*$',  # "Page X"
            r'^\s*\d+\s*/\s*\d+\s*$',  # "X / Y"
            r'^\s*\[\s*\d+\s*\]\s*$',  # [X]
            r'(?i)^\s*(copyright|©).*$',  # Copyright
            r'(?i)^\s*(confidential|proprietary).*$',  # Confidential
            r'(?i)^\s*generated\s+(on|at|by).*$',  # Generated on/at/by
            r'(?i)^\s*last\s+updated?:.*$',  # Last updated
            r'(?i)^\s*created:.*$',  # Created
            r'(?i)^\s*modified:.*$',  # Modified
            r'(?i)^\s*version:.*$',  # Version
            r'(?i)^\s*revision:.*$',  # Revision
        ]
        
        # Patrones para detectar y limpiar ruido específico
        self.noise_patterns = [
            r'\f',  # Form feeds
            r'\x0c',  # Page breaks
            r'\r\n',  # Windows line endings -> \n
            r'\.{4,}',  # 4 o más puntos consecutivos
            r'-{4,}',  # 4 o más guiones consecutivos (excepto separadores completos)
            r'={4,}',  # 4 o más signos igual consecutivos (excepto separadores completos)
            r'_{4,}',  # 4 o más guiones bajos consecutivos
            r'\s{3,}',  # 3 o más espacios consecutivos
            r'^\s*\*+\s*$',  # Líneas solo con asteriscos
            r'^\s*#+\s*$',  # Líneas solo con hash
        ]
        
        # Patrones para detectar secciones especiales
        self.section_patterns = {
            "toc": [
                r'(?i)^\s*(table\s+of\s+contents?|contents?|índice|index)\s*$',
                r'(?i)^\s*(chapter|capítulo)\s+\d+.*\.{3,}.*\d+\s*$',
            ],
            "references": [
                r'(?i)^\s*(references?|bibliography|bibliografía|referencias?)\s*$',
                r'^\s*\[\d+\]\s+.*$',  # [1] Citation format
                r'^\s*\d+\.\s+[A-Z][a-z]+,.*\(\d{4}\).*$',  # 1. Author, (2023)
            ],
            "metadata": [
                r'(?i)^\s*(abstract|resumen|summary)\s*$',
                r'(?i)^\s*(keywords?|palabras\s+clave)\s*:?.*$',
                r'(?i)^\s*(tags?|etiquetas)\s*:?.*$',
            ]
        }
        
        # Patrones para detectar tipos de documentos específicos
        self.document_type_patterns = {
            "mitre_attack": [
                r'(?i)(technique|tactic|mitigation|group|software)',
                r'(?i)(att&ck|attack|mitre)',
                r'(?i)(t\d{4}|ta\d{4}|m\d{4}|g\d{4}|s\d{4})',  # IDs de MITRE
            ],
            "owasp": [
                r'(?i)(owasp|open web application security)',
                r'(?i)(vulnerability|security|risk)',
                r'(?i)(injection|xss|csrf|authentication)',
            ],
            "cve": [
                r'(?i)(cve-\d{4}-\d{4,})',
                r'(?i)(common vulnerabilities)',
                r'(?i)(cvss|severity|impact)',
            ],
            "threat_intel": [
                r'(?i)(threat|intelligence|ioc|indicator)',
                r'(?i)(malware|ransomware|apt|campaign)',
                r'(?i)(hash|ip address|domain|url)',
            ]
        }
    
    def detect_encoding(self, file_path: Path) -> str:
        """Detecta la codificación del archivo"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                return result['encoding'] or 'utf-8'
        except:
            return 'utf-8'
    
    def clean_text(self, text: str, remove_metadata: bool = False, 
                   remove_references: bool = False) -> str:
        """
        Limpia el texto del archivo
        
        Args:
            text: Texto a limpiar
            remove_metadata: Si eliminar secciones de metadata
            remove_references: Si eliminar sección de referencias
        """
        if not text or not text.strip():
            return ""
        
        # Normalizar Unicode y line endings
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'\r\n', '\n', text)  # Normalizar line endings
        
        # Dividir en líneas
        lines = text.split('\n')
        cleaned_lines = []
        
        in_metadata_section = False
        in_references_section = False
        skip_empty_lines = 0
        
        for i, line in enumerate(lines):
            original_line = line
            
            # Detectar secciones a omitir
            if remove_metadata:
                if any(re.match(pattern, line.strip()) for pattern in self.section_patterns["metadata"]):
                    in_metadata_section = True
                    continue
            
            if remove_references:
                if any(re.match(pattern, line.strip()) for pattern in self.section_patterns["references"][:1]):
                    in_references_section = True
                    continue
            
            # Saltar si estamos en una sección a omitir
            if in_metadata_section or in_references_section:
                # Salir de la sección si encontramos una línea que parece contenido normal
                if (len(line.strip()) > 0 and 
                    not any(re.match(pattern, line.strip()) for patterns in self.section_patterns.values() for pattern in patterns) and
                    len(line.split()) > 5):  # Línea con contenido sustancial
                    in_metadata_section = False
                    in_references_section = False
                else:
                    continue
            
            # Eliminar headers/footers comunes
            if any(re.match(pattern, line.strip()) for pattern in self.header_footer_patterns):
                continue
            
            # Limpiar ruido en la línea
            for pattern in self.noise_patterns:
                line = re.sub(pattern, '', line)
            
            # Limpiar espacios
            line = ' '.join(line.split())
            
            # Solo mantener líneas con contenido
            if line.strip():
                # Evitar demasiadas líneas vacías consecutivas
                cleaned_lines.append(line)
                skip_empty_lines = 0
            elif skip_empty_lines < 1:  # Máximo 1 línea vacía consecutiva
                cleaned_lines.append("")
                skip_empty_lines += 1
        
        # Unir y limpieza final
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Normalizar espacios finales
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)  # Máximo 2 saltos
        cleaned_text = re.sub(r' +', ' ', cleaned_text)  # Espacios múltiples
        
        return cleaned_text.strip()
    
    def detect_document_type(self, text: str, filename: str = "") -> str:
        """Detecta el tipo de documento basado en contenido y nombre"""
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        # Verificar por nombre de archivo primero
        if any(keyword in filename_lower for keyword in ['mitre', 'attack', 'att&ck']):
            return "mitre_attack"
        elif any(keyword in filename_lower for keyword in ['owasp']):
            return "owasp"
        elif any(keyword in filename_lower for keyword in ['cve']):
            return "cve"
        elif any(keyword in filename_lower for keyword in ['threat', 'intel', 'ioc']):
            return "threat_intel"
        
        # Verificar por contenido
        scores = {}
        for doc_type, patterns in self.document_type_patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, text_lower))
            scores[doc_type] = score
        
        if scores and max(scores.values()) > 0:
            return max(scores.keys(), key=scores.get)
        
        return "general_text"
    
    def extract_metadata_from_content(self, text: str) -> Dict[str, Any]:
        """Extrae metadata del contenido del texto"""
        metadata = {
            "has_structured_content": False,
            "has_bullet_points": False,
            "has_numbered_lists": False,
            "has_code_blocks": False,
            "has_urls": False,
            "has_email": False,
            "language_indicators": {},
            "special_identifiers": []
        }
        
        # Detectar estructura
        if re.search(r'^\s*[\*\-\+•]\s+', text, re.MULTILINE):
            metadata["has_bullet_points"] = True
            metadata["has_structured_content"] = True
        
        if re.search(r'^\s*\d+\.\s+', text, re.MULTILINE):
            metadata["has_numbered_lists"] = True
            metadata["has_structured_content"] = True
        
        if re.search(r'```|`[^`]+`|^\s{4,}\w', text, re.MULTILINE):
            metadata["has_code_blocks"] = True
        
        if re.search(r'https?://|www\.', text):
            metadata["has_urls"] = True
        
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
            metadata["has_email"] = True
        
        # Buscar identificadores especiales
        identifiers = []
        
        # CVE IDs
        cve_matches = re.findall(r'(?i)(cve-\d{4}-\d{4,})', text)
        identifiers.extend([match.upper() for match in cve_matches])
        
        # MITRE ATT&CK IDs
        mitre_matches = re.findall(r'(?i)\b([tgms]\d{4}(?:\.\d{3})?)\b', text)
        identifiers.extend([match.upper() for match in mitre_matches])
        
        # Hash values (MD5, SHA1, SHA256)
        hash_matches = re.findall(r'\b([a-fA-F0-9]{32}|[a-fA-F0-9]{40}|[a-fA-F0-9]{64})\b', text)
        identifiers.extend(hash_matches[:10])  # Limitar a 10
        
        metadata["special_identifiers"] = list(set(identifiers))
        
        # Indicadores básicos de idioma
        text_words = text.lower().split()
        if len(text_words) > 10:  # Solo si hay suficiente texto
            spanish_indicators = sum(1 for word in ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'por', 'con'] if word in text_words)
            english_indicators = sum(1 for word in ['the', 'of', 'and', 'to', 'a', 'in', 'is', 'it', 'you', 'that', 'for', 'with'] if word in text_words)
            
            metadata["language_indicators"] = {
                "spanish_score": spanish_indicators,
                "english_score": english_indicators,
                "likely_language": "spanish" if spanish_indicators > english_indicators else "english"
            }
        
        return metadata

class SimpleTXTProcessor:
    """Procesador simple de archivos TXT"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            "clean_text": True,
            "remove_metadata": False,
            "remove_references": False,
            "min_chars": 50,
            "auto_detect_encoding": True,
            "preserve_structure": True
        }
        self.cleaner = TextFileCleaner()
    
    def process_txt_file(self, txt_path: Path) -> Dict[str, Any]:
        """
        Procesa un archivo TXT individual
        
        Returns:
            Dict con toda la información del documento
        """
        logger.info(f"🔄 Procesando: {txt_path.name}")
        
        try:
            # Detectar encoding si está configurado
            if self.config.get("auto_detect_encoding", True):
                encoding = self.cleaner.detect_encoding(txt_path)
            else:
                encoding = 'utf-8'
            
            # Leer archivo
            try:
                with open(txt_path, 'r', encoding=encoding) as f:
                    raw_text = f.read()
            except UnicodeDecodeError:
                # Fallback a utf-8 con errores ignore
                with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_text = f.read()
                logger.warning(f"Problemas de encoding en {txt_path.name}, usando utf-8 con ignore")
            
            # Información básica del archivo
            doc_info = {
                "file_info": {
                    "filename": txt_path.name,
                    "file_path": str(txt_path),
                    "file_size_bytes": txt_path.stat().st_size,
                    "encoding_detected": encoding,
                    "processing_timestamp": datetime.now().isoformat()
                },
                "content": {
                    "raw_text": raw_text,
                    "cleaned_text": ""
                },
                "text_statistics": {},
                "content_analysis": {},
                "quality_metrics": {}
            }
            
            # Limpiar texto si está configurado
            if self.config.get("clean_text", True):
                cleaned_text = self.cleaner.clean_text(
                    raw_text,
                    remove_metadata=self.config.get("remove_metadata", False),
                    remove_references=self.config.get("remove_references", False)
                )
            else:
                cleaned_text = raw_text.strip()
            
            doc_info["content"]["cleaned_text"] = cleaned_text
            
            # Verificar si el documento es válido
            min_chars = self.config.get("min_chars", 50)
            is_valid = len(cleaned_text) >= min_chars
            
            if not is_valid:
                logger.warning(f"⚠️  {txt_path.name}: Documento muy corto ({len(cleaned_text)} chars)")
                doc_info["quality_metrics"]["is_valid"] = False
                return doc_info
            
            # Calcular estadísticas del texto
            doc_info["text_statistics"] = self._calculate_text_statistics(cleaned_text)
            
            # Análisis del contenido
            doc_info["content_analysis"] = {
                "document_type": self.cleaner.detect_document_type(cleaned_text, txt_path.name),
                **self.cleaner.extract_metadata_from_content(cleaned_text)
            }
            
            # Métricas de calidad
            doc_info["quality_metrics"] = self._calculate_quality_metrics(doc_info)
            
            logger.info(f"✅ {txt_path.name}: {doc_info['text_statistics']['total_words']} palabras, tipo: {doc_info['content_analysis']['document_type']}")
            return doc_info
            
        except Exception as e:
            logger.error(f"❌ Error procesando {txt_path.name}: {e}")
            return self._create_error_document(txt_path, str(e))
    
    def _calculate_text_statistics(self, text: str) -> Dict[str, Any]:
        """Calcula estadísticas básicas del texto"""
        if not text:
            return {
                "total_chars": 0,
                "total_words": 0,
                "total_lines": 0,
                "non_empty_lines": 0,
                "avg_words_per_line": 0,
                "avg_chars_per_word": 0,
                "longest_line": 0
            }
        
        lines = text.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        words = text.split()
        
        return {
            "total_chars": len(text),
            "total_words": len(words),
            "total_lines": len(lines),
            "non_empty_lines": len(non_empty_lines),
            "avg_words_per_line": len(words) / len(non_empty_lines) if non_empty_lines else 0,
            "avg_chars_per_word": len(text) / len(words) if words else 0,
            "longest_line": max(len(line) for line in lines) if lines else 0
        }
    
    def _calculate_quality_metrics(self, doc_info: Dict) -> Dict[str, Any]:
        """Calcula métricas de calidad del documento"""
        stats = doc_info["text_statistics"]
        content = doc_info["content_analysis"]
        
        # Puntuación de calidad (0-100)
        quality_score = 0
        
        # Puntos por longitud (0-40)
        word_count = stats.get("total_words", 0)
        if word_count > 100:
            quality_score += min(40, (word_count / 100) * 10)
        
        # Puntos por estructura (0-30)
        if content.get("has_structured_content", False):
            quality_score += 15
        if content.get("has_bullet_points", False) or content.get("has_numbered_lists", False):
            quality_score += 10
        if len(content.get("special_identifiers", [])) > 0:
            quality_score += 5
        
        # Puntos por contenido técnico (0-20)
        doc_type = content.get("document_type", "general_text")
        if doc_type != "general_text":
            quality_score += 10
        if content.get("has_urls", False) or content.get("has_code_blocks", False):
            quality_score += 10
        
        # Puntos por densidad de contenido (0-10)
        avg_words_per_line = stats.get("avg_words_per_line", 0)
        if 5 <= avg_words_per_line <= 20:  # Densidad óptima
            quality_score += 10
        elif avg_words_per_line > 3:
            quality_score += 5
        
        return {
            "is_valid": True,
            "quality_score": min(100, quality_score),
            "word_density": avg_words_per_line,
            "has_technical_content": doc_type != "general_text",
            "has_identifiers": len(content.get("special_identifiers", [])) > 0,
            "content_richness": len([k for k, v in content.items() if isinstance(v, bool) and v])
        }
    
    def _create_error_document(self, txt_path: Path, error_msg: str) -> Dict[str, Any]:
        """Crea estructura de documento para errores"""
        return {
            "file_info": {
                "filename": txt_path.name,
                "file_path": str(txt_path),
                "file_size_bytes": txt_path.stat().st_size if txt_path.exists() else 0,
                "processing_timestamp": datetime.now().isoformat()
            },
            "error": True,
            "error_message": error_msg,
            "content": {"raw_text": "", "cleaned_text": ""},
            "text_statistics": {},
            "content_analysis": {},
            "quality_metrics": {"is_valid": False, "quality_score": 0}
        }
    
    def save_document(self, doc_data: Dict, output_path: Path):
        """Guarda documento procesado"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(doc_data, f, ensure_ascii=False, indent=2)

def process_txt_directory(txt_dir: Path, output_dir: Path, config: Dict = None) -> Dict[str, Any]:
    """
    Procesa todos los archivos TXT en un directorio
    
    Args:
        txt_dir: Directorio con archivos TXT
        output_dir: Directorio de salida  
        config: Configuración del procesador
    
    Returns:
        Dict con estadísticas del procesamiento
    """
    if not txt_dir.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {txt_dir}")
    
    # Crear procesador
    processor = SimpleTXTProcessor(config)
    
    # Encontrar archivos TXT
    txt_files = list(txt_dir.glob("*.txt"))
    
    if not txt_files:
        logger.warning(f"No se encontraron archivos TXT en: {txt_dir}")
        return {"error": "No TXT files found"}
    
    logger.info(f"📚 Encontrados {len(txt_files)} archivos TXT")
    
    # Crear directorio de salida
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Procesar archivos
    results = {
        "processing_summary": {
            "total_files": len(txt_files),
            "successful_files": 0,
            "failed_files": 0,
            "total_words": 0,
            "document_types": {},
            "start_time": datetime.now().isoformat()
        },
        "files": []
    }
    
    all_documents = []
    
    for txt_file in txt_files:
        try:
            # Procesar archivo TXT
            doc_data = processor.process_txt_file(txt_file)
            
            # Guardar documento individual
            output_file = output_dir / f"{txt_file.stem}.json"
            processor.save_document(doc_data, output_file)
            
            # Actualizar estadísticas
            if not doc_data.get("error", False) and doc_data.get("quality_metrics", {}).get("is_valid", False):
                results["processing_summary"]["successful_files"] += 1
                results["processing_summary"]["total_words"] += doc_data["text_statistics"].get("total_words", 0)
                
                # Contar tipos de documentos
                doc_type = doc_data["content_analysis"].get("document_type", "unknown")
                results["processing_summary"]["document_types"][doc_type] = \
                    results["processing_summary"]["document_types"].get(doc_type, 0) + 1
            else:
                results["processing_summary"]["failed_files"] += 1
            
            # Info del archivo
            file_info = {
                "filename": txt_file.name,
                "status": "success" if not doc_data.get("error", False) and doc_data.get("quality_metrics", {}).get("is_valid", False) else "failed",
                "words": doc_data["text_statistics"].get("total_words", 0),
                "document_type": doc_data["content_analysis"].get("document_type", "unknown"),
                "quality_score": doc_data["quality_metrics"].get("quality_score", 0),
                "has_identifiers": doc_data["quality_metrics"].get("has_identifiers", False),
                "output_file": str(output_file)
            }
            
            results["files"].append(file_info)
            
            if file_info["status"] == "success":
                all_documents.append(doc_data)
            
        except Exception as e:
            logger.error(f"Error procesando {txt_file.name}: {e}")
            results["processing_summary"]["failed_files"] += 1
            results["files"].append({
                "filename": txt_file.name,
                "status": "failed",
                "error": str(e)
            })
    
    # Guardar resumen general
    results["processing_summary"]["end_time"] = datetime.now().isoformat()
    
    summary_file = output_dir / "processing_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Crear pages.jsonl con páginas individuales (para chunking posterior)
    pages_file = output_dir / "pages.jsonl"
    with open(pages_file, 'w', encoding='utf-8') as f:
        for doc in all_documents:
            if not doc.get("error", False) and doc["content"]["cleaned_text"]:
                # Cargar metadata separada si existe
                txt_file = Path(doc["file_info"]["file_path"])
                external_metadata = load_metadata_file(txt_file)
                
                # Usar título de metadata externa o fallback al nombre del archivo
                title = (
                    external_metadata.get("name", "").strip()
                    or external_metadata.get("title", "").strip()
                    or doc["file_info"]["filename"]
                )
                
                # Crear metadata estándar
                standard_metadata = MetadataBuilder.create_standard_metadata(
                    source_id=Path(doc["file_info"]["filename"]).stem,
                    source_type=external_metadata.get("category", "txt"),
                    source_file=doc["file_info"]["filename"],
                    page_number=1,
                    text=doc["content"]["cleaned_text"],
                    title=title,
                    page_start=1,
                    page_end=1,
                    document_type=doc["content_analysis"]["document_type"],
                    words=doc["text_statistics"]["total_words"],
                    quality_score=doc["quality_metrics"]["quality_score"],
                    has_structure=doc["content_analysis"]["has_structured_content"],
                    has_identifiers=len(doc["content_analysis"]["special_identifiers"]) > 0,
                    special_identifiers=doc["content_analysis"]["special_identifiers"],
                    language=doc["content_analysis"]["language_indicators"].get("likely_language", "unknown"),
                    encoding=doc["file_info"]["encoding_detected"]
                )
                
                # Agregar metadata específica de TXT
                page_data = {
                    **standard_metadata,
                    "metadata": MetadataBuilder.add_txt_metadata(standard_metadata, external_metadata)
                }
                json.dump(page_data, f, ensure_ascii=False)
                f.write('\n')
    
    logger.info(f"🎉 Procesamiento completado!")
    logger.info(f"   ✅ Exitosos: {results['processing_summary']['successful_files']}")
    logger.info(f"   ❌ Fallidos: {results['processing_summary']['failed_files']}")
    logger.info(f"   📝 Total palabras: {results['processing_summary']['total_words']:,}")
    logger.info(f"   📊 Tipos encontrados: {list(results['processing_summary']['document_types'].keys())}")
    
    return results

def load_metadata_file(txt_file: Path) -> Dict[str, Any]:
    """Carga metadata de archivo JSON separado si existe"""
    metadata_file = txt_file.parent / f"{txt_file.stem}_metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"No se pudo cargar metadata de {metadata_file}: {e}")
    return {}

def process_txt_sources(force_overwrite: bool = False):
    """Procesa archivos TXT de AISecKG, AnnoCTR, MITRE, OWASP, SecurityTools"""
    base_dir = Path("/Users/marcosespana/Desktop/U/DatosTesis")
    
    # Configuración
    config = {
        "clean_text": True,
        "remove_metadata": False,  # Cambiar a True si hay mucha metadata no útil
        "remove_references": False,  # Cambiar a True para eliminar referencias
        "min_chars": 50,
        "auto_detect_encoding": True,
        "preserve_structure": True
    }
    
    # Fuentes a procesar
    sources = ["AISecKG", "AnnoCTR", "MITRE", "OWASP", "SecurityTools"]
    total_processed = 0
    
    for source in sources:
        print(f"\n{'='*50}")
        print(f"PROCESANDO {source}")
        print(f"{'='*50}")
        
        # Para fuentes con subdirectorios, procesar cada subdirectorio
        if source in ["AnnoCTR", "MITRE", "SecurityTools"]:
            base_txt_dir = base_dir / "data" / "raw" / source
            base_output_dir = base_dir / "data" / "interim" / source
            
            if not base_txt_dir.exists():
                print(f"⚠️  Directorio no encontrado: {base_txt_dir}")
                continue
            
            # Buscar subdirectorios
            subdirs = [d for d in base_txt_dir.iterdir() if d.is_dir()]
            if not subdirs:
                print(f"⚠️  No se encontraron subdirectorios en: {base_txt_dir}")
                continue
            
            print(f"📁 Encontrados {len(subdirs)} subdirectorios en {source}")
            
            for subdir in subdirs:
                print(f"\n  📂 Procesando subdirectorio: {subdir.name}")
                
                txt_dir = subdir
                output_dir = base_output_dir / subdir.name
                
                # Verificar que hay TXTs en el subdirectorio
                txt_files = list(txt_dir.glob("*.txt"))
                if not txt_files:
                    print(f"    ⚠️  No se encontraron TXTs en: {txt_dir}")
                    continue
                    
                print(f"    📚 Encontrados {len(txt_files)} TXTs en {subdir.name}")
                
                # Verificar archivos existentes
                if check_existing_files(output_dir) and not force_overwrite:
                    print(f"    ⚠️  Ya existen archivos en {output_dir}")
                    print(f"       Usa --force para sobrescribir o elimina manualmente los archivos")
                    continue
                    
                # Procesar directorio
                results = process_txt_directory(txt_dir, output_dir, config)
                
                if "error" not in results:
                    total_processed += results["processing_summary"]["successful_files"]
                    print(f"    ✅ {subdir.name}: {results['processing_summary']['successful_files']} TXTs procesados")
                else:
                    print(f"    ❌ {subdir.name}: Error en procesamiento")
        
        else:
            # Para otras fuentes, procesar directamente
            txt_dir = base_dir / "data" / "raw" / source
            output_dir = base_dir / "data" / "interim" / source
            
            if not txt_dir.exists():
                print(f"⚠️  Directorio no encontrado: {txt_dir}")
                continue
                
            # Verificar que hay TXTs
            txt_files = list(txt_dir.glob("*.txt"))
            if not txt_files:
                print(f"⚠️  No se encontraron TXTs en: {txt_dir}")
                continue
                
            print(f"📚 Encontrados {len(txt_files)} TXTs en {source}")
            
            # Verificar archivos existentes
            if check_existing_files(output_dir) and not force_overwrite:
                print(f"⚠️  Ya existen archivos en {output_dir}")
                print(f"   Usa --force para sobrescribir o elimina manualmente los archivos")
                continue
                
            # Procesar directorio
            results = process_txt_directory(txt_dir, output_dir, config)
            
            if "error" not in results:
                total_processed += results["processing_summary"]["successful_files"]
                print(f"✅ {source}: {results['processing_summary']['successful_files']} TXTs procesados")
            else:
                print(f"❌ {source}: Error en procesamiento")
    
    return total_processed

def check_existing_files(output_dir: Path) -> bool:
    """Verifica si ya existen archivos en el directorio de salida"""
    if not output_dir.exists():
        return False
    
    pages_file = output_dir / "pages.jsonl"
    if pages_file.exists():
        return True
    return False

def main():
    """Función principal"""
    import argparse
    parser = argparse.ArgumentParser(description="Procesar archivos TXT de AISecKG, AnnoCTR, MITRE, OWASP, SecurityTools")
    parser.add_argument("--force", action="store_true", help="Sobrescribir archivos existentes")
    args = parser.parse_args()
    
    total_processed = process_txt_sources(force_overwrite=args.force)
    
    print("\n" + "="*50)
    print("RESUMEN DEL PROCESAMIENTO")
    print("="*50)
    print(f"📁 Total de TXTs procesados: {total_processed}")
    print(f"📂 Archivos guardados en: data/interim/")
    print(f"📋 Páginas listas para chunking en: data/interim/*/pages.jsonl")

if __name__ == "__main__":
    main()