#!/usr/bin/env python3
"""
Ejemplos de uso del evaluador de Cybersecurity QA
"""

print("🚀 EVALUADOR DE CYBERSECURITY QA - EJEMPLOS DE USO")
print("=" * 80)

print("\n1️⃣ INSTALACIÓN DE DEPENDENCIAS:")
print("pip install -r requirements_evaluator.txt")

print("\n2️⃣ CONFIGURACIÓN PREVIA:")
print("# Asegúrate de que Weaviate esté corriendo:")
print("cd weaviate && docker-compose up -d")

print("\n3️⃣ EJEMPLOS DE USO:")

print("\n📝 Ejemplo 1: Evaluación básica con Ollama (5 preguntas)")
print("python cybersecurity_qa_evaluator.py --max-questions 5")

print("\n📝 Ejemplo 2: Evaluación con OpenAI (10 preguntas)")
print("python cybersecurity_qa_evaluator.py \\")
print("  --max-questions 10 \\")
print("  --llm-provider openai \\")
print("  --openai-api-key sk-your-api-key-here \\")
print("  --openai-model gpt-3.5-turbo")

print("\n📝 Ejemplo 3: Evaluación completa con archivo personalizado")
print("python cybersecurity_qa_evaluator.py \\")
print("  --max-questions 20 \\")
print("  --output-file mi_evaluacion.xlsx \\")
print("  --llm-provider openai \\")
print("  --openai-api-key $OPENAI_API_KEY \\")
print("  --openai-model gpt-4")

print("\n📝 Ejemplo 4: Evaluación rápida con Ollama")
print("python cybersecurity_qa_evaluator.py \\")
print("  --max-questions 3 \\")
print("  --ollama-model mistral \\")
print("  --output-file evaluacion_rapida.xlsx")

print("\n4️⃣ PARÁMETROS DISPONIBLES:")
print("--max-questions N        # Número de preguntas a evaluar")
print("--output-file archivo.xlsx  # Archivo Excel de salida")
print("--llm-provider {ollama,openai}  # Proveedor de LLM")
print("--openai-api-key SK-...  # API Key de OpenAI")
print("--openai-model MODELO    # Modelo de OpenAI")
print("--ollama-model MODELO    # Modelo de Ollama")
print("--weaviate-host HOST     # Host de Weaviate")
print("--weaviate-port PUERTO   # Puerto de Weaviate")

print("\n5️⃣ ARCHIVOS GENERADOS:")
print("📊 cybersecurity_qa_evaluation_YYYYMMDD_HHMMSS.xlsx")
print("   ├── Hoja 'Resultados': Todas las preguntas y respuestas")
print("   ├── Hoja 'Resumen': Estadísticas de la evaluación")
print("   └── Hoja 'Errores': Preguntas que fallaron (si las hay)")

print("\n6️⃣ ESTRUCTURA DEL EXCEL:")
print("Columnas en 'Resultados':")
print("  - timestamp: Fecha y hora de la evaluación")
print("  - question: Pregunta original del dataset")
print("  - expected_answer: Respuesta esperada (si está disponible)")
print("  - rag_answer: Respuesta generada por el RAG")
print("  - response_time_seconds: Tiempo de respuesta")
print("  - iterations: Número de iteraciones del RAG")
print("  - num_sources: Número de fuentes encontradas")
print("  - sources: Lista de fuentes utilizadas")
print("  - source_file: Archivo origen de la pregunta")
print("  - llm_provider: Proveedor de LLM usado")
print("  - model: Modelo específico usado")

print("\n7️⃣ NOTAS IMPORTANTES:")
print("✅ El script descarga automáticamente el dataset de Kaggle")
print("✅ Detecta automáticamente las columnas de preguntas y respuestas")
print("✅ Maneja errores y continúa con la siguiente pregunta")
print("✅ Genera estadísticas de rendimiento")
print("✅ Soporta tanto Ollama como OpenAI")
print("✅ Guarda resultados detallados en Excel")

print("\n8️⃣ TROUBLESHOOTING:")
print("❌ 'kagglehub no está instalado' → pip install kagglehub")
print("❌ 'openpyxl no está instalado' → pip install openpyxl")
print("❌ 'Connection refused' → Verifica que Weaviate esté corriendo")
print("❌ 'No se encontraron archivos CSV' → El dataset puede estar vacío")
print("❌ 'API Key requerida' → Configura --openai-api-key")

print("\n🎯 ¡Listo para evaluar tu sistema RAG!")
