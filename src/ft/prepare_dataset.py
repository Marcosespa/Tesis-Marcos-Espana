#!/usr/bin/env python3
"""
Prepara dataset para Domain Adaptation Fine-Tuning
Usa chunks directamente como texto plano (sin formato Q&A)
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import random

def load_chunks(file_path: Path) -> List[Dict]:
    """Carga chunks desde archivo JSONL"""
    chunks = []
    if not file_path.exists():
        return chunks
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"⚠️  Error en línea {line_num} de {file_path}: {e}")
                    continue
    except Exception as e:
        print(f"❌ Error leyendo {file_path}: {e}")
    
    return chunks

def filter_chunk_quality(chunk: Dict, min_words: int = 50, max_words: int = 2000) -> Optional[str]:
    """
    Filtra chunks por calidad y devuelve texto limpio
    
    Args:
        chunk: Diccionario del chunk
        min_words: Mínimo de palabras requeridas
        max_words: Máximo de palabras (truncar si excede)
    
    Returns:
        Texto del chunk o None si no pasa filtros
    """
    # Extraer texto
    text = chunk.get('content') or chunk.get('text', '')
    
    if not text or not text.strip():
        return None
    
    # Contar palabras
    words = text.split()
    word_count = len(words)
    
    # Filtro de longitud mínima
    if word_count < min_words:
        return None
    
    # Truncar si es muy largo
    if word_count > max_words:
        text = ' '.join(words[:max_words])
        word_count = max_words
    
    # Limpiar texto (remover espacios excesivos)
    text = ' '.join(text.split())
    
    return text

def format_text_with_context(text: str, metadata: Dict, include_context: bool = True) -> str:
    """
    Formatea texto con contexto opcional
    
    Args:
        text: Texto del chunk
        metadata: Metadata del chunk
        include_context: Si incluir contexto en el texto
    
    Returns:
        Texto formateado
    """
    if not include_context:
        return text
    
    # Construir contexto
    context_parts = []
    
    # Información del documento
    title = metadata.get('doc_title') or metadata.get('title', '')
    if title:
        context_parts.append(f"Document: {title}")
    
    source_type = metadata.get('source_type', '')
    if source_type:
        context_parts.append(f"Source: {source_type}")
    
    # Campos comunes del metadata
    category = metadata.get('category', '')
    if category:
        context_parts.append(f"Category: {category}")
    
    section_title = metadata.get('section_title')
    if section_title:
        context_parts.append(f"Section: {section_title}")
    
    section_level = metadata.get('section_level')
    if section_level is not None:
        context_parts.append(f"Section Level: {section_level}")
    
    page_num_real = metadata.get('page_num_real')
    if page_num_real is not None:
        context_parts.append(f"Page (real): {page_num_real}")
    
    page_num_logical = metadata.get('page_num_logical')
    if page_num_logical is not None:
        context_parts.append(f"Page (logical): {page_num_logical}")
    
    # Contexto específico por fuente (MITRE)
    if 'tactic' in metadata:
        context_parts.append(f"MITRE Tactic: {metadata['tactic']}")
    
    if 'technique_id' in metadata:
        context_parts.append(f"MITRE Technique: {metadata['technique_id']}")
    
    if context_parts:
        context = "\n".join(context_parts)
        return f"{context}\n\n{text}"
    
    return text

def prepare_domain_dataset(
    sources_config: Dict[str, Path],
    output_file: Path,
    min_words: int = 50,
    max_words: int = 2000,
    include_context: bool = True,
    sample_size: Optional[int] = None,
    train_split: float = 0.9
) -> Dict:
    """
    Prepara dataset de dominio desde chunks
    
    Args:
        sources_config: Diccionario {nombre_fuente: path_chunks}
        output_file: Archivo de salida JSONL (o directorio base)
        min_words: Mínimo de palabras por chunk
        max_words: Máximo de palabras por chunk
        include_context: Incluir contexto en el texto
        sample_size: Si especificado, tomar máximo N chunks por fuente
        train_split: Proporción para entrenamiento (resto para validación)
    
    Returns:
        Estadísticas del procesamiento
    """
    dataset = []
    stats = {
        'sources': {},
        'total': 0,
        'filtered': 0,
        'final': 0
    }
    
    print(f"🔄 Procesando {len(sources_config)} fuentes...")
    print(f"   Mínimo palabras: {min_words}")
    print(f"   Máximo palabras: {max_words}")
    print(f"   Incluir contexto: {include_context}\n")
    
    for source_name, source_path in sources_config.items():
        if not source_path.exists():
            print(f"⚠️  Saltando {source_name}: {source_path} no existe")
            continue
        
        print(f"📂 Procesando {source_name}...")
        
        # Cargar chunks
        chunks = []
        if source_path.is_file():
            chunks = load_chunks(source_path)
        elif source_path.is_dir():
            # Buscar archivos .chunks.jsonl RECURSIVAMENTE en todas las subcarpetas
            chunk_files = list(source_path.glob('**/*.chunks.jsonl'))
            if not chunk_files:
                print(f"   ⚠️  No se encontraron archivos .chunks.jsonl (búsqueda recursiva)")
                continue
            
            print(f"   📄 Encontrados {len(chunk_files)} archivos .chunks.jsonl (recursivo)")
            
            for chunk_file in chunk_files:
                chunks.extend(load_chunks(chunk_file))
        else:
            print(f"   ⚠️  {source_path} no es archivo ni directorio")
            continue
        
        source_count = 0
        source_filtered = 0
        
        # Aplicar sampling si se especifica
        if sample_size and len(chunks) > sample_size:
            chunks = random.sample(chunks, sample_size)
            print(f"   📊 Usando muestra de {sample_size} chunks de {len(chunks)} disponibles")
        
        for chunk in chunks:
            stats['total'] += 1
            
            # Filtrar calidad
            text = filter_chunk_quality(chunk, min_words, max_words)
            
            if not text:
                source_filtered += 1
                stats['filtered'] += 1
                continue
            
            # Obtener metadata
            metadata = chunk.get('metadata', {}) or {}
            
            # Formatear texto
            formatted_text = format_text_with_context(text, metadata, include_context)
            
            # Crear entrada del dataset
            entry = {
                "text": formatted_text,
                "source": source_name,
                "word_count": len(formatted_text.split())
            }
            
            # Agregar metadata adicional si existe
            if metadata.get('doc_title'):
                entry['title'] = metadata['doc_title']
            
            if metadata.get('source_type'):
                entry['source_type'] = metadata['source_type']
            
            dataset.append(entry)
            source_count += 1
            stats['final'] += 1
        
        stats['sources'][source_name] = {
            'loaded': len(chunks),
            'processed': source_count,
            'filtered': source_filtered
        }
        
        print(f"   ✅ {source_name}: {source_count:,} chunks procesados")
        if source_filtered > 0:
            print(f"      ⚠️  {source_filtered:,} chunks filtrados")
    
    # Mezclar dataset
    random.shuffle(dataset)
    
    # Dividir en train/val
    split_idx = int(len(dataset) * train_split)
    train_dataset = dataset[:split_idx]
    val_dataset = dataset[split_idx:]
    
    # Determinar archivos de salida
    if output_file.is_dir():
        train_file = output_file / "train.jsonl"
        val_file = output_file / "validation.jsonl"
    else:
        # Si es archivo, crear directorio
        output_dir = output_file.parent / output_file.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        train_file = output_dir / "train.jsonl"
        val_file = output_dir / "validation.jsonl"
    
    # Guardar datasets
    print(f"\n💾 Guardando datasets...")
    
    for data, file_path in [(train_dataset, train_file), (val_dataset, val_file)]:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"✅ {file_path.name}: {len(data):,} ejemplos")
    
    # Estadísticas finales
    print(f"\n📊 Estadísticas:")
    print(f"   Total chunks leídos: {stats['total']:,}")
    print(f"   Chunks filtrados: {stats['filtered']:,}")
    print(f"   Chunks finales: {stats['final']:,}")
    print(f"   Train: {len(train_dataset):,} ({len(train_dataset)/len(dataset)*100:.1f}%)")
    print(f"   Validation: {len(val_dataset):,} ({len(val_dataset)/len(dataset)*100:.1f}%)")
    
    # Distribución por fuente
    print(f"\n📚 Distribución por fuente:")
    for source_name, source_stats in stats['sources'].items():
        print(f"   {source_name:20} | Cargados: {source_stats['loaded']:>6,} | "
              f"Procesados: {source_stats['processed']:>6,} | "
              f"Filtrados: {source_stats['filtered']:>6,}")
    
    # Estadísticas de longitud
    word_counts = [item['word_count'] for item in dataset]
    if word_counts:
        avg_words = sum(word_counts) / len(word_counts)
        min_words_actual = min(word_counts)
        max_words_actual = max(word_counts)
        
        print(f"\n📏 Estadísticas de longitud:")
        print(f"   Promedio: {avg_words:.0f} palabras")
        print(f"   Mínimo: {min_words_actual} palabras")
        print(f"   Máximo: {max_words_actual} palabras")
    
    return stats

def main():
    parser = argparse.ArgumentParser(
        description="Prepara dataset para Domain Adaptation Fine-Tuning (sin formato Q&A)"
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/ft_datasets/cybersecurity_domain'),
        help='Directorio de salida para train.jsonl y validation.jsonl'
    )
    
    parser.add_argument(
        '--chunks-dir',
        type=Path,
        default=Path('data/chunks'),
        help='Directorio base con chunks'
    )
    
    parser.add_argument(
        '--min-words',
        type=int,
        default=50,
        help='Mínimo de palabras por chunk (default: 50)'
    )
    
    parser.add_argument(
        '--max-words',
        type=int,
        default=2000,
        help='Máximo de palabras por chunk (default: 2000)'
    )
    
    parser.add_argument(
        '--no-context',
        action='store_true',
        help='No incluir contexto en el texto'
    )
    
    parser.add_argument(
        '--sample',
        type=int,
        help='Máximo de chunks por fuente (sampling)'
    )
    
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.9,
        help='Proporción para entrenamiento (default: 0.9)'
    )
    
    parser.add_argument(
        '--sources',
        nargs='+',
        help='Fuentes específicas a procesar (por defecto: todas)'
    )
    
    args = parser.parse_args()
    
    # Configurar fuentes disponibles
    base_sources = {
        'AnnoCTR': args.chunks_dir / 'AnnoCTR' / 'pages.chunks.jsonl',
        'MITRE': args.chunks_dir / 'MITRE' / 'pages.chunks.jsonl',
        'NIST': args.chunks_dir / 'NIST' / 'pages.chunks.jsonl',
        'NIST_AI': args.chunks_dir / 'NIST_AI',
        'NIST_CSWP': args.chunks_dir / 'NIST_CSWP',
        'NIST_SP': args.chunks_dir / 'NIST_SP',
        'OWASP': args.chunks_dir / 'OWASP' / 'pages.chunks.jsonl',
        'SecurityTools': args.chunks_dir / 'SecurityTools' / 'pages.chunks.jsonl',
        'AISecKG': args.chunks_dir / 'AISecKG' / 'pages.chunks.jsonl',
    }
    
    # Filtrar fuentes si se especifican
    if args.sources:
        sources_config = {
            name: path for name, path in base_sources.items()
            if name in args.sources
        }
        if not sources_config:
            print(f"❌ No se encontraron fuentes: {args.sources}")
            print(f"   Fuentes disponibles: {list(base_sources.keys())}")
            return 1
    else:
        sources_config = base_sources
    
    # Preparar dataset
    stats = prepare_domain_dataset(
        sources_config=sources_config,
        output_file=args.output,
        min_words=args.min_words,
        max_words=args.max_words,
        include_context=not args.no_context,
        sample_size=args.sample,
        train_split=args.train_split
    )
    
    print(f"\n✅ Proceso completado exitosamente!")
    print(f"   Datasets guardados en: {args.output}")
    print(f"   - train.jsonl")
    print(f"   - validation.jsonl")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

