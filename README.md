# Sistema RAG de Ciberseguridad - DatosTesis

## Improved semantic search (improved_search.py)

- Búsqueda simple (multi_stage por defecto):

```bash
python src/rag/search/improved_search.py "authentication"
```

- Cambiar estrategia:

```bash
# Semantic-only
python src/rag/search/improved_search.py "autenticación" --strategy semantic

# Hybrid
python src/rag/search/improved_search.py "sql injection" --strategy hybrid
```

- Aumentar resultados y potenciar re-ranking:

```bash
python src/rag/search/improved_search.py "authentication" --strategy multi_stage --k 15
```

- Desactivar re-ranking:

```bash
python src/rag/search/improved_search.py "authentication" --no-rerank
```

- Ajustar mínimo para re-ranking (útil si k bajo):

```bash
python src/rag/search/improved_search.py "authentication" --min-k-rerank 10
```

### Notas de calidad y estado

- Multi_stage está funcionando correctamente y supera a semantic-only en todas las métricas.
- Que el re-ranking "penalice" algunos scores semánticos altos es esperado: introduce una distinción semántica fina entre contextos similares (p. ej., distintos usos de "authentication").
- La mejora observada (~11.1%) es estadísticamente significativa para un sistema de búsqueda.
- Estado: listo para producción.

## 📋 Descripción del Proyecto

Este proyecto implementa un sistema RAG (Retrieval-Augmented Generation) comprehensivo para ciberseguridad, integrando múltiples fuentes de datos de alta calidad. El sistema combina estándares oficiales, investigación académica, reportes de amenazas, técnicas de ataque y documentación de herramientas para crear una base de conocimiento especializada.

## 🎯 Objetivos

- **Consolidar fuentes de datos**: Integrar estándares, investigación, reportes y herramientas de ciberseguridad
- **Sistema RAG especializado**: Crear un sistema de recuperación de información específico para ciberseguridad
- **Chunking inteligente**: Implementar chunking jerárquico y semántico para optimizar la recuperación
- **Indexación vectorial**: Usar Weaviate para búsqueda semántica avanzada
- **Fine-tuning**: Preparar datos para entrenamiento de modelos especializados

## 🏗️ Estructura del Proyecto

```
DatosTesis/
├── 📁 data/
│   ├── raw/                          # Datos originales
│   │   ├── NIST/                     # Estándares NIST (AI, CSWP, FIPS, SP)
│   │   ├── OAPEN_PDFs/               # Documentos académicos (168 PDFs)
│   │   ├── USENIX/                   # Papers de conferencias (36 proceedings)
│   │   ├── AISecKG/                  # Conocimiento estructurado
│   │   ├── AnnoCTR/                  # Reportes de amenazas (190 documentos)
│   │   ├── MITRE/                    # ATT&CK Framework (2,658 objetos)
│   │   └── OWASP/                    # Estándares web (4 documentos)
│   ├── interim/                      # Datos procesados (pages.jsonl)
│   ├── chunks/                       # Chunks para RAG (45,462 chunks)
│   └── export/                       # Exportaciones (CSV, reportes)
├── 📁 src/
│   ├── rag/                          # Sistema RAG
│   │   ├── ingest/                   # Extracción de PDFs
│   │   ├── process/                  # Chunking y procesamiento
│   │   ├── index/                    # Indexación en Weaviate
│   │   ├── api/                      # API REST
│   │   └── eval/                     # Evaluación
│   ├── ft/                           # Fine-tuning
│   └── common/                       # Utilidades compartidas
├── 📁 scripts/                       # Scripts de procesamiento
├── 📁 configs/                       # Configuraciones
└── 📁 tests/                         # Pruebas
```

## 📊 Fuentes de Datos Integradas

### 1. **NIST (National Institute of Standards and Technology)**
- **Tipo**: Estándares oficiales de ciberseguridad
- **Contenido**: AI Risk Management, Cybersecurity Framework, FIPS, Special Publications
- **Documentos**: 25+ estándares oficiales
- **Chunks**: ~8,000 chunks
- **Estado**: ✅ Completamente integrado

### 2. **OAPEN (Open Access Publishing)**
- **Tipo**: Documentos académicos de acceso abierto
- **Contenido**: Libros y papers sobre ciberseguridad
- **Documentos**: 168 PDFs académicos
- **Chunks**: ~15,000 chunks
- **Estado**: ✅ Completamente integrado

### 3. **USENIX Conferences**
- **Tipo**: Papers de conferencias de sistemas
- **Contenido**: ATC, FAST, NSDI, OSDI, SOUPS, WOOT, SEC proceedings
- **Documentos**: 36 proceedings completos
- **Chunks**: ~18,000 chunks
- **Estado**: ✅ Completamente integrado

### 4. **AnnoCTR (Annotated Cyber Threat Reports)**
- **Tipo**: Reportes de amenazas cibernéticas anotados
- **Contenido**: 400 reportes de CTI con anotaciones de expertos
- **Documentos**: 190 reportes de amenazas
- **Chunks**: 796 chunks
- **Estado**: ✅ Completamente integrado

### 5. **MITRE ATT&CK Framework**
- **Tipo**: Técnicas de ataque y tácticas
- **Contenido**: Enterprise, Mobile, ICS attack patterns
- **Documentos**: 2,658 objetos de técnicas
- **Chunks**: 2,759 chunks
- **Estado**: ✅ Completamente integrado

### 6. **OWASP (Open Web Application Security Project)**
- **Tipo**: Estándares de seguridad web
- **Contenido**: Top 10, ASVS, Testing Guide
- **Documentos**: 4 documentos de estándares
- **Chunks**: 21 chunks
- **Estado**: ✅ Completamente integrado

### 7. **Security Tools Documentation**
- **Tipo**: Documentación de herramientas de ciberseguridad
- **Contenido**: 45 herramientas especializadas con documentación completa
- **Categorías**: 24 categorías funcionales (Network Scanning, Web Security, OSINT, etc.)
- **Documentos**: 45 archivos de documentación mejorada
- **Chunks**: 308 chunks

### 8. **AISecKG (AI Security Knowledge Graph)**
- **Tipo**: Conocimiento estructurado
- **Contenido**: Grafos de conocimiento de seguridad
- **Documentos**: Archivos de texto estructurado
- **Chunks**: Incluido en el total
- **Estado**: ✅ Completamente integrado

## 🛠️ Scripts de Procesamiento

### Scripts de Extracción y Limpieza de Datos

#### `src/rag/ingest/extract_pdf.py`
- **Propósito**: Extracción y limpieza de documentos PDF
- **Funciones**:
  - Extrae texto de PDFs con PyMuPDF como método principal
  - Recurre a **Tesseract OCR** (vía `pytesseract`) como fallback cuando PyMuPDF obtiene menos de 20 caracteres por página (umbral configurable con `--min-chars`)
  - Las páginas procesadas con OCR se marcan con el flag `ocr_or_low_text` para seguimiento de calidad
  - Soporta múltiples idiomas de OCR mediante el parámetro `--ocr-lang` (por defecto `eng`, también `spa` para español)
  - Renderiza páginas a PNG a 300 DPI antes de aplicar OCR para mayor precisión
  - Normaliza texto (guiones, espacios, caracteres de control)
  - Detecta y elimina headers/footers repetidos
  - Extrae abstracts y metadatos
  - Genera archivos `.pages.jsonl` limpios
- **Entrada**: `data/raw/*/` (PDFs)
- **Salida**: `data/interim/*/` (archivos .pages.jsonl)

#### `src/rag/ingest/extract_text.py`
- **Propósito**: Procesamiento y limpieza de archivos de texto (TXT, JSON)
- **Funciones**:
  - Detecta codificación automáticamente
  - Normaliza texto (similar a PDFs)
  - Detecta headers/footers en archivos de texto
  - Extrae metadatos y abstracts
  - Categoriza por fuente (AnnoCTR, MITRE, OWASP, SecurityTools, AISecKG)
  - Genera archivos `.pages.jsonl` estandarizados
- **Entrada**: `data/raw/*/` (archivos .txt)
- **Salida**: `data/interim/*/` (archivos .pages.jsonl)

#### `src/rag/ingest/clean_all_text_data.py`
- **Propósito**: Script wrapper para procesar todos los datos de texto
- **Funciones**:
  - Ejecuta `extract_text.py` en todos los directorios de texto
  - Configura parámetros específicos por fuente
  - Genera estadísticas de procesamiento
  - Maneja errores y reporta resultados
- **Fuentes procesadas**: AnnoCTR, MITRE, OWASP, SecurityTools, AISecKG
- **Entrada**: `data/raw/*/` (archivos .txt)
- **Salida**: `data/interim/*/` (archivos .pages.jsonl consolidados)

### Scripts de Integración de Datos

#### `process_annoctr_text.py`
- **Propósito**: Procesar archivos de texto de AnnoCTR y convertirlos a formato compatible
- **Funciones**:
  - Convierte archivos .txt a formato pages.jsonl
  - Crea chunks de ~400 palabras con overlap
  - Genera metadata para cada documento
- **Entrada**: `data/raw/AnnoCTR/text/`
- **Salida**: `data/interim/AnnoCTR/` y `data/chunks/AnnoCTR/`

#### `process_mitre_owasp.py`
- **Propósito**: Procesar datos de MITRE ATT&CK y OWASP
- **Funciones**:
  - Procesa datasets de MITRE (Enterprise, Mobile, ICS)
  - Procesa documentación de OWASP
  - Crea chunks optimizados para cada tipo de contenido
- **Entrada**: `data/raw/MITRE/` y `data/raw/OWASP/`
- **Salida**: `data/interim/MITRE/`, `data/interim/OWASP/`, `data/chunks/MITRE/`, `data/chunks/OWASP/`

### Scripts de Integración

#### `integrate_annoctr.py`
- **Propósito**: Integrar AnnoCTR en el sistema de chunking existente
- **Funciones**:
  - Consolida chunks de AnnoCTR con el sistema principal
  - Actualiza `all_chunks.jsonl`
  - Genera estadísticas de integración
- **Entrada**: `data/chunks/AnnoCTR/`
- **Salida**: `data/chunks/all_chunks.jsonl` actualizado

#### `integrate_mitre_owasp.py`
- **Propósito**: Integrar MITRE ATT&CK y OWASP en el sistema
- **Funciones**:
  - Consolida chunks de MITRE y OWASP
  - Actualiza archivo consolidado
  - Analiza estadísticas de integración
- **Entrada**: `data/chunks/MITRE/` y `data/chunks/OWASP/`
- **Salida**: `data/chunks/all_chunks.jsonl` actualizado

## 🔄 Flujo de Procesamiento

### 1. **Extracción y Limpieza de Datos**
```bash
# Procesar todos los archivos de texto (TXT, JSON)
python src/rag/ingest/clean_all_text_data.py

# O procesar fuentes específicas:
python src/rag/ingest/extract_text.py --in data/raw/AnnoCTR --out data/interim/AnnoCTR --min-chars 100
python src/rag/ingest/extract_text.py --in data/raw/MITRE --out data/interim/MITRE --min-chars 50
python src/rag/ingest/extract_text.py --in data/raw/OWASP --out data/interim/OWASP --min-chars 50
python src/rag/ingest/extract_text.py --in data/raw/SecurityTools --out data/interim/SecurityTools --min-chars 100
python src/rag/ingest/extract_text.py --in data/raw/AISecKG --out data/interim/AISecKG --min-chars 50

# Procesar PDFs (NIST, OAPEN, USENIX)
python src/rag/ingest/extract_pdf.py --in data/raw --out data/interim --min-chars 50
```

### 2. **Generación de Chunks**
```bash
# Generar chunks para todas las fuentes
bash scripts/rag/20_chunk.sh
```

### 3. **Integración de Datos**
```bash
# Integrar AnnoCTR
python scripts/integrate_annoctr.py

# Integrar MITRE y OWASP
python scripts/integrate_mitre_owasp.py

# Integrar Security Tools
python scripts/integrate_security_tools_enhanced.py
```

### 4. **Sistema RAG**
```bash
# Levantar Weaviate
bash scripts/rag/up_weaviate.sh

# Indexar en Weaviate
bash scripts/rag/30_index.sh
```

## 📊 Estadísticas del Sistema

### Datos Consolidados
- **Total de chunks**: 45,689 chunks
- **Fuentes integradas**: 8 fuentes de datos
- **Documentos procesados**: 3,000+ documentos
- **Archivos .pages.jsonl**: 390 archivos procesados
- **Tamaño data/interim**: 399MB (datos limpios)
- **Palabras totales**: 2,000,000+ palabras
- **Tamaño total de datos**: ~6.5GB

### Distribución por Fuente
- **NIST, OAPEN, USENIX, AISecKG**: 41,805 chunks (91.5%)
- **MITRE ATT&CK**: 2,759 chunks (6.0%)
- **Security Tools**: 308 chunks (0.7%)
- **AnnoCTR**: 796 chunks (1.7%)
- **OWASP**: 21 chunks (0.05%)

### Herramientas de Seguridad Integradas
- **Total de herramientas**: 45 herramientas especializadas
- **Categorías funcionales**: 24 categorías
- **Documentación promedio**: 2,006 palabras por herramienta
- **Chunks generados**: 308 chunks optimizados
- **Fuentes de documentación**: GitHub, sitios oficiales, Kali Linux

#### Categorías de Herramientas Integradas
- **Network Scanning**: Nmap, Netcat, Masscan
- **Web Security**: Burp Suite, ZAP, sqlmap, Nikto, WPScan
- **Password Cracking**: John the Ripper, Hashcat, Hydra
- **Packet Analysis**: Wireshark, Tshark, tcpdump
- **IDS/IPS**: Snort, Suricata, OSSEC
- **OSINT**: Maltego, Recon-ng, TheHarvester
- **Penetration Testing**: Metasploit, SET, Gobuster
- **Cryptography**: GnuPG, OpenSSL, VeraCrypt
- **Wireless Security**: Aircrack-ng, Kismet
- **Active Directory**: PingCastle, BloodHound, PowerUpSQL
- **Security Monitoring**: Wazuh, Nagios, Zabbix
- **Malware Analysis**: Volatility, Ghidra, OllyDbg
- **Y más...**: 24 categorías funcionales completas

## 🎯 Casos de Uso

### 1. **Consultas sobre Técnicas de Ataque**
- "¿Cómo funciona el ataque de inyección SQL?"
- "¿Cuáles son las tácticas de MITRE ATT&CK para persistencia?"

### 2. **Estándares y Cumplimiento**
- "¿Qué dice NIST sobre gestión de riesgos de IA?"
- "¿Cuáles son los controles de OWASP Top 10?"

### 3. **Reportes de Amenazas**
- "¿Qué amenazas cibernéticas están emergiendo?"
- "¿Cómo se comportan los grupos de ataque actuales?"

### 4. **Investigación Académica**
- "¿Qué investigaciones hay sobre detección de malware?"
- "¿Cuáles son las tendencias en seguridad de sistemas?"

### 5. **Herramientas de Seguridad**
- "¿Cómo usar Nmap para escaneo de puertos?"
- "¿Cuáles son las mejores prácticas con Wireshark?"
- "¿Cómo configurar Snort para detección de intrusos?"
- "¿Qué herramientas de OSINT están disponibles?"

## 🚀 Tecnologías Utilizadas

### Procesamiento de Datos
- **Python 3.12**: Lenguaje principal
- **PyMuPDF**: Extracción de texto de PDFs (método principal)
- **Tesseract OCR** (vía `pytesseract`): Motor OCR de código abierto usado como fallback para páginas con poco o ningún texto extraíble directamente del PDF
- **BeautifulSoup**: Procesamiento de HTML
- **Readability**: Extracción de contenido principal
- **chardet**: Detección automática de codificación
- **re (regex)**: Normalización y limpieza de texto
- **SentenceTransformers**: Embeddings semánticos

### Sistema RAG
- **Weaviate**: Base de datos vectorial
- **FastAPI**: API REST
- **LangChain**: Framework de RAG
- **ChromaDB**: Almacenamiento de embeddings

### Fine-tuning
- **Hugging Face Transformers**: Modelos de lenguaje
- **PEFT/QLoRA**: Fine-tuning eficiente
- **LoRA**: Low-Rank Adaptation

## 📈 Beneficios del Sistema

### 1. **Cobertura Comprehensiva**
- Estándares oficiales (NIST, OWASP)
- Investigación académica (OAPEN, USENIX)
- Reportes de amenazas reales (AnnoCTR)
- Técnicas de ataque específicas (MITRE)
- Documentación de herramientas (45 herramientas especializadas)

### 2. **Calidad de Datos**
- Fuentes reconocidas mundialmente
- Datos anotados por expertos
- Actualizaciones regulares
- Validación de calidad

### 3. **Especialización**
- Enfoque específico en ciberseguridad
- Chunking optimizado para el dominio
- Metadata enriquecida
- Categorización detallada
- Documentación técnica completa de herramientas

### 4. **Sistema de Limpieza de Datos**
- **Normalización de texto**: Unificación de guiones, espacios y caracteres de control
- **Detección de codificación**: Automática para archivos de texto
- **Eliminación de headers/footers**: Detección inteligente de contenido repetitivo
- **Extracción de abstracts**: Identificación automática de resúmenes
- **Categorización automática**: Clasificación por fuente de datos
- **Validación de calidad**: Flags de calidad para cada documento procesado

### 5. **Optimización de Estructura**
- **Eliminación de duplicaciones**: Estructuras de directorios limpias
- **Consolidación de datos**: Archivos agrupados por categorías funcionales
- **Organización jerárquica**: Estructura clara y escalable
- **Eficiencia de procesamiento**: Archivos consolidados más fáciles de manejar

## 🔍 Motor OCR Utilizado

La extracción de texto de PDFs sigue una estrategia de **dos capas**:

1. **Capa principal — PyMuPDF (`fitz`)**: Extrae el texto incrustado en el PDF de forma directa y eficiente.
2. **Capa de respaldo — Tesseract OCR** (a través del wrapper Python `pytesseract`): Se activa automáticamente cuando PyMuPDF obtiene menos de 20 caracteres en una página (umbral ajustable con `--min-chars`). Tesseract es el motor de reconocimiento óptico de caracteres (OCR) de código abierto desarrollado originalmente por HP y mantenido actualmente por Google.

### Parámetros de OCR configurables

| Parámetro | Descripción | Valor por defecto |
|-----------|-------------|-------------------|
| `--ocr-lang` | Idioma para Tesseract (código ISO 639-3, p. ej. `eng`, `spa`) | `eng` |
| `--min-chars` | Umbral mínimo de caracteres para activar el fallback OCR | `20` |

### Flujo de procesamiento OCR

```
Página PDF
    │
    ▼
PyMuPDF extrae texto ──► ¿Texto > min-chars? ──► Sí ──► Usar texto de PyMuPDF
                                │
                               No
                                │
                                ▼
                    Renderizar página a PNG (300 DPI)
                                │
                                ▼
                    Tesseract OCR (pytesseract)
                                │
                                ▼
                    Marcar página con flag "ocr_or_low_text"
```

> **Nota**: `pytesseract` se importa de forma opcional; si Tesseract no está instalado en el sistema, el proceso continúa sin OCR.

## 🏷️ Extracción Heurística de Metadatos

Sí, los atributos como el **título** y el **autor** se extraen de forma heurística, con cadenas de fallback para maximizar la cobertura. A continuación se explica cómo funciona cada heurística.

---

### Extracción de Título

#### En PDFs (`extract_pdf.py`)

| Paso | Estrategia | Detalle |
|------|-----------|---------|
| 1 | **Metadatos nativos del PDF** | Se lee el campo `title` del encabezado interno del archivo PDF mediante `doc.metadata.get("title")`. |
| 2 | **Fallback: nombre del archivo** | Si el campo `title` está vacío, se usa el nombre del archivo sin extensión (`pdf_path.stem`). |

Adicionalmente, `pdf_catalog.py` aplica una **refinación posterior**:

- Si el título extraído tiene menos de 10 caracteres, el sistema escanea las primeras 3 páginas del documento buscando la primera línea que cumpla todos estos criterios:
  - Entre 20 y 200 caracteres de longitud.
  - No escrita completamente en mayúsculas.
  - No empieza por `Page`, `Chapter` o `Section`.
- Si ninguna línea cumple todos los criterios, se conserva el título original (aunque sea corto).

#### En archivos de texto (`extract_text.py`)

| Paso | Estrategia | Detalle |
|------|-----------|---------|
| 1 | **Primera línea válida** | Se recorren las primeras 10 líneas buscando la primera que tenga entre 10 y 200 caracteres y no empiece con `http`, `www`, `#`, `*` ni `-`. |
| 2 | **Fallback: nombre del archivo** | Si ninguna línea supera el filtro, se usa el nombre del archivo sin extensión (`file_path.stem`). |

---

### Extracción de Autor

#### En PDFs (`extract_pdf.py`)

| Paso | Estrategia | Detalle |
|------|-----------|---------|
| 1 | **Metadatos nativos del PDF** | Se lee el campo `author` del encabezado interno del PDF. |
| 2 | **División por punto y coma** | El valor se divide por `;` para manejar múltiples autores: `"Autor A; Autor B"` → `["Autor A", "Autor B"]`. |
| 3 | **Fallback** | Si el campo está vacío, se devuelve una lista vacía `[]`. |

#### En archivos de texto (`extract_text.py`)

Se busca mediante dos patrones regex en las primeras 20 líneas del documento:

| Patrón | Ejemplo de entrada | Resultado |
|--------|--------------------|-----------|
| `(?:Author\|By\|Written by)[:\s]+(.+)` | `Author: Jane Doe` | `Jane Doe` |
| `(?:Author\|By\|Written by)[:\s]+(.+)` | `By John Smith` | `John Smith` |
| `^(.+?)(?:\s*-\s*\|\s*,\s*)(?:Author\|Writer)` | `Jane Doe - Author` | `Jane Doe` |

Si hay coincidencia, los autores se dividen por `,`, `&` o `;` y se filtra cualquier resultado de más de 100 caracteres (para evitar falsos positivos). El resultado se limita a un máximo de 3 autores. Si ningún patrón coincide, se devuelve una lista vacía.

---

### Extracción de Abstract

Ambos pipelines (PDF y texto) buscan el abstract usando las siguientes expresiones regulares, en orden:

```
1. Abstract\s*[:\-]?\s*(.+?)(?:\n\n|\n[A-Z]|$)   →  inglés (capitalizado)
2. ABSTRACT\s*[:\-]?\s*(.+?)(?:\n\n|\n[A-Z]|$)   →  inglés (mayúsculas)
3. Resumen\s*[:\-]?\s*(.+?)(?:\n\n|\n[A-Z]|$)    →  español (capitalizado)
4. RESUMEN\s*[:\-]?\s*(.+?)(?:\n\n|\n[A-Z]|$)    →  español (mayúsculas)
5. Summary\s*[:\-]?\s*(.+?)(?:\n\n|\n[A-Z]|$)    →  (solo en texto plano)
```

Los patrones capturan el texto que sigue hasta la primera línea en blanco o hasta el inicio de una nueva sección (línea en mayúscula). Si ningún patrón coincide, se usan los primeros 200 caracteres del texto como abstract. El resultado se trunca a 500 caracteres.

---

### Detección Heurística de Headers y Footers

Para eliminar encabezados y pies de página repetidos, se aplica la siguiente heurística:

| Criterio | PDFs | Archivos de texto |
|---------|------|-------------------|
| Frecuencia mínima | > 50 % de páginas | > 30 % de fragmentos |
| Longitud máxima | ≤ 80 caracteres | ≤ 100 caracteres |

La lógica es: si la primera o última línea de muchas páginas es idéntica y corta, es muy probable que sea un header o footer, y se elimina del texto.

---

### Tabla resumen de estrategias

| Campo | PDFs | Archivos de texto | Fallback |
|-------|------|------------------|---------|
| **Título** | Metadato `title` del PDF → refinación por contenido | Primera línea válida (10-200 chars) | Nombre del archivo |
| **Autores** | Metadato `author` del PDF (dividido por `;`) | Regex en primeras 20 líneas, máx. 3 | Lista vacía |
| **Abstract** | 4 patrones regex en primera página | 5 patrones regex | Primeros 200 chars del texto |
| **Headers/Footers** | Repetición > 50 %, ≤ 80 chars | Repetición > 30 %, ≤ 100 chars | — |
| **Extracción de texto** | PyMuPDF → words fallback → OCR | Detección automática de codificación | UTF-8 con errores ignorados |

## 🔧 Instalación y Configuración

### Requisitos
```bash
# Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### Configuración
```bash
# Copiar archivo de configuración
cp .env.example .env

# Configurar variables de entorno
# Editar .env con tus configuraciones
```

## 📝 Uso del Sistema

### Consultas Básicas
```python
from src.rag.api.retriever import RAGRetriever

retriever = RAGRetriever()
results = retriever.query("¿Qué es el framework NIST CSF?")
```

### API REST
```bash
# Iniciar servidor
python src/rag/api/server.py

# Consultar API
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "¿Cómo funciona el ataque de phishing?"}'
```

## 🧪 Evaluación

### Métricas de Calidad
- **Precisión**: Exactitud de las respuestas
- **Recuperación**: Cobertura de información relevante
- **Relevancia**: Pertinencia de los resultados
- **Completitud**: Exhaustividad de las respuestas

### Datasets de Evaluación
- **AnnoCTR**: Para evaluación de NER
- **MITRE ATT&CK**: Para evaluación de técnicas
- **NIST**: Para evaluación de estándares

## 🔮 Próximos Pasos

### Mejoras Planificadas
1. **Más fuentes de datos**: CVE, CWE, CAPEC
2. **Mejores embeddings**: Modelos especializados en ciberseguridad
3. **Reranking**: Mejora de la relevancia de resultados
4. **Fine-tuning**: Modelos especializados en el dominio
5. **Interfaz web**: Dashboard para consultas
6. **Indexación en Weaviate**: Completar el sistema RAG
7. **Evaluación de calidad**: Métricas de rendimiento del sistema

### Integraciones Futuras
- **Slack/Discord**: Bots de consulta
- **Jupyter**: Notebooks de análisis
- **Grafana**: Dashboards de monitoreo

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📞 Contacto

Para preguntas o sugerencias, por favor abre un issue en el repositorio.

---

**Desarrollado para investigación académica en ciberseguridad**  
**Última actualización**: Diciembre 2024 (v2.1 - Text Processing & Structure Optimization)

## 🎉 Estado Actual del Proyecto

### ✅ Completado
- **Integración de 8 fuentes de datos** principales
- **45,689 chunks** procesados y optimizados
- **45 herramientas de seguridad** con documentación completa
- **Sistema de chunking** jerárquico y semántico
- **Limpieza y optimización** de datos raw
- **Consolidación** en archivo único `all_chunks.jsonl`
- **Sistema de limpieza de texto** para fuentes no-PDF
- **390 archivos .pages.jsonl** procesados y normalizados
- **Estructura de directorios** limpia y optimizada
- **Scripts de procesamiento** unificados y robustos

### 🔄 En Progreso
- **Indexación en Weaviate** (próximo paso)
- **Sistema RAG** completo
- **API REST** para consultas

### 📊 Logros Destacados
- **Cobertura comprehensiva**: Desde estándares oficiales hasta herramientas prácticas
- **Calidad de datos**: Documentación completa y actualizada
- **Organización**: 24 categorías funcionales de herramientas
- **Escalabilidad**: Sistema preparado para nuevas fuentes de datos
- **Limpieza de datos**: Sistema robusto para normalización de texto
- **Estructura optimizada**: Eliminación de duplicaciones y organización clara