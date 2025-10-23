#!/bin/bash
# Script para ejecutar evaluación de 50 preguntas y cerrar PC

echo "🚀 INICIANDO EVALUACIÓN AUTOMÁTICA DE 50 PREGUNTAS"
echo "=================================================="
echo ""

# Verificar que estamos en el directorio correcto
if [ ! -f "cybersecurity_qa_evaluator.py" ]; then
    echo "❌ ERROR: No se encontró cybersecurity_qa_evaluator.py"
    echo "Asegúrate de estar en el directorio correcto"
    exit 1
fi

# Verificar que Weaviate esté corriendo
echo "🔍 Verificando conexión a Weaviate..."
if ! curl -s http://localhost:8080/v1/meta > /dev/null 2>&1; then
    echo "❌ ERROR: Weaviate no está corriendo en localhost:8080"
    echo "Ejecuta: docker-compose up -d"
    exit 1
fi
echo "✅ Weaviate está corriendo"

# Activar entorno virtual
echo "🐍 Activando entorno virtual..."
source .venv/bin/activate

# Ejecutar evaluación en segundo plano
echo "🚀 Iniciando evaluación de 50 preguntas..."
echo "📝 Log guardado en: evaluation_50_questions.log"
echo "📄 Excel se generará automáticamente al finalizar"
echo ""

# Ejecutar con nohup para que continúe después de cerrar terminal
nohup python cybersecurity_qa_evaluator.py --max-questions 50 > evaluation_50_questions.log 2>&1 &

# Obtener PID del proceso
EVAL_PID=$!
echo "✅ Proceso iniciado con PID: $EVAL_PID"
echo ""

# Mostrar comandos útiles
echo "💡 COMANDOS ÚTILES:"
echo "==================="
echo "• Ver progreso: ./monitor_evaluation.sh"
echo "• Ver log: tail -f evaluation_50_questions.log"
echo "• Ver proceso: ps aux | grep $EVAL_PID"
echo "• Matar proceso: kill $EVAL_PID"
echo "• Ver archivos Excel: ls -la *.xlsx"
echo ""

# Mostrar estimación de tiempo
echo "⏱️  ESTIMACIÓN DE TIEMPO:"
echo "========================="
echo "• Tiempo por pregunta: ~4 minutos"
echo "• Total estimado: ~200 minutos (3.3 horas)"
echo "• El proceso continuará aunque cierres el PC"
echo ""

# Opción para monitoreo inmediato
read -p "¿Quieres ver el progreso ahora? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ./monitor_evaluation.sh
else
    echo "✅ Evaluación iniciada. Puedes cerrar el PC de forma segura."
    echo "📝 Para monitorear después: ./monitor_evaluation.sh"
fi

