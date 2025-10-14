# Procesadores de Documentos - Estándares de Metadata

Este directorio contiene los procesadores de documentos que convierten archivos raw a formato interim, siguiendo estándares uniformes de metadata.

## 📁 Archivos

- `metadata_standards.py` - **Estándares de metadata** que todos los procesadores deben seguir
- `process_pdf.py` - Procesador de archivos PDF (NIST, OAPEN, USENIX)
- `process_txt.py` - Procesador de archivos TXT (AISecKG, AnnoCTR, MITRE, OWASP, SecurityTools)

## 🎯 Estándares de Metadata

### Campos Estándar (todos los procesadores)

Estos campos **DEBEN** estar presentes en todos los `pages.jsonl` generados:

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `source_id` | string | ID único del documento (sin extensión) |
| `source_type` | string | Tipo de fuente (pdf, txt, etc.) |
| `source_file` | string | Nombre del archivo original |
| `page_number` | int | Número de página (1-indexed) |
| `text` | string | Texto extraído/limpio |
| `title` | string | Título del documento/página |
| `page_start` | int | Página de inicio |
| `page_end` | int | Página de fin |
| `document_type` | string | Tipo de documento detectado |
| `words` | int | Número de palabras |
| `quality_score` | float | Puntuación de calidad (0-100) |
| `has_structure` | bool | Si tiene estructura (headers, listas, etc.) |
| `has_identifiers` | bool | Si tiene identificadores especiales |
| `special_identifiers` | array | Lista de identificadores encontrados |
| `language` | string | Idioma detectado |
| `encoding` | string | Codificación del archivo |

### Metadata Específica de PDF

Campos adicionales para documentos PDF:

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `pages` | int | Total de páginas del documento |
| `has_images` | bool | Si la página tiene imágenes |
| `rotation` | int | Rotación de la página (0, 90, 180, 270) |
| `page_dimensions` | object | Dimensiones de la página {width, height} |
| `pdf_title` | string | Título del PDF |
| `pdf_author` | string | Autor del PDF |
| `pdf_subject` | string | Asunto del PDF |
| `pdf_keywords` | string | Palabras clave del PDF |
| `pdf_creator` | string | Creador del PDF |
| `pdf_producer` | string | Productor del PDF |
| `pdf_creation_date` | string | Fecha de creación |
| `pdf_modification_date` | string | Fecha de modificación |
| `pdf_format` | string | Formato del PDF |
| `pdf_encryption` | any | Información de encriptación |

### Metadata Específica de TXT

Campos adicionales para documentos TXT con metadata externa:

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `tool_id` | string | ID de la herramienta |
| `tool_name` | string | Nombre de la herramienta |
| `tool_description` | string | Descripción de la herramienta |
| `tool_category` | string | Categoría de la herramienta |
| `tool_urls` | array | URLs relacionadas |
| `download_date` | string | Fecha de descarga |
| `char_count` | int | Conteo de caracteres |

## 🚀 Uso de los Estándares

### En los Procesadores

```python
from metadata_standards import MetadataBuilder, validate_metadata

# Crear metadata estándar
standard_metadata = MetadataBuilder.create_standard_metadata(
    source_id="documento",
    source_type="pdf",
    source_file="documento.pdf",
    page_number=1,
    text="Contenido...",
    title="Título",
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

# Agregar metadata específica de PDF
page_data = {
    **standard_metadata,
    "metadata": MetadataBuilder.add_pdf_metadata(
        standard_metadata,
        pages=10,
        has_images=True,
        rotation=0,
        page_dimensions={"width": 595.0, "height": 842.0},
        pdf_metadata=pdf_metadata_dict
    )
}

# Validar metadata
errors = validate_metadata(page_data)
if errors:
    print("Errores:", errors)
```

### Validación

```python
from metadata_standards import validate_metadata

# Validar metadata generada
errors = validate_metadata(metadata_dict)
if not errors:
    print("✅ Metadata válida")
else:
    print("❌ Errores encontrados:", errors)
```

## 📋 Estructura de Salida

Cada procesador genera:

```
data/interim/
├── [fuente]/
│   ├── pages.jsonl          # ← Archivo principal con metadata estándar
│   ├── processing_summary.json
│   └── [archivo].json       # Archivos individuales (opcional)
└── [subdirectorio]/
    ├── pages.jsonl
    └── processing_summary.json
```

## 🔧 Comandos de Uso

```bash
# Procesar solo PDFs
python pipeline.py --step pdf-process-rawToInterim

# Procesar solo TXTs
python pipeline.py --step txt-process-rawToInterim

# Procesar todo
python pipeline.py --step all
```

## 📝 Notas Importantes

1. **Consistencia**: Todos los procesadores deben usar `MetadataBuilder` para garantizar consistencia
2. **Validación**: Siempre validar metadata antes de guardar
3. **Extensibilidad**: Para agregar nuevos campos, actualizar `metadata_standards.py`
4. **Compatibilidad**: Los campos estándar son obligatorios para el chunking y RAG

## 🐛 Debugging

Para verificar que la metadata es válida:

```python
from metadata_standards import validate_metadata, get_metadata_schema

# Ver esquema completo
schema = get_metadata_schema()
print(schema)

# Validar metadata
errors = validate_metadata(your_metadata)
print("Errores:", errors)
```
