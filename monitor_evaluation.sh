#!/bin/bash
# Script para monitorear la evaluación de 50 preguntas

echo "🚀 MONITOR DE EVALUACIÓN CYBERSECURITY QA"
echo "=========================================="
echo ""

# Función para mostrar el progreso
show_progress() {
    echo "📊 ESTADO ACTUAL:"
    echo "=================="
    
    # Verificar si el proceso está corriendo
    if pgrep -f "cybersecurity_qa_evaluator.py" > /dev/null; then
        echo "✅ Proceso ACTIVO (PID: $(pgrep -f cybersecurity_qa_evaluator.py))"
        
        # Mostrar uso de CPU y memoria
        echo "💻 Recursos:"
        ps -p $(pgrep -f cybersecurity_qa_evaluator.py) -o pid,pcpu,pmem,time,command | tail -1
        
        # Mostrar últimas líneas del log
        echo ""
        echo "📝 ÚLTIMAS ACTIVIDADES:"
        echo "======================="
        if [ -f "evaluation_50_questions.log" ]; then
            tail -10 evaluation_50_questions.log
        else
            echo "⚠️  Log file no encontrado aún"
        fi
        
        # Verificar si hay archivos Excel generados
        echo ""
        echo "📄 ARCHIVOS GENERADOS:"
        echo "======================"
        ls -la *.xlsx 2>/dev/null | tail -3 || echo "⚠️  No hay archivos Excel aún"
        
    else
        echo "❌ Proceso NO está corriendo"
        echo "📄 ARCHIVOS FINALES:"
        ls -la *.xlsx 2>/dev/null || echo "⚠️  No se encontraron archivos Excel"
    fi
    
    echo ""
    echo "⏰ $(date)"
    echo "=========================================="
}

# Mostrar progreso inicial
show_progress

echo ""
echo "💡 COMANDOS ÚTILES:"
echo "==================="
echo "• Ver log completo: tail -f evaluation_50_questions.log"
echo "• Ver proceso: ps aux | grep cybersecurity_qa_evaluator"
echo "• Matar proceso: pkill -f cybersecurity_qa_evaluator.py"
echo "• Ver archivos Excel: ls -la *.xlsx"
echo ""

# Opción para monitoreo continuo
read -p "¿Quieres monitoreo continuo cada 30 segundos? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔄 Iniciando monitoreo continuo (Ctrl+C para salir)..."
    while true; do
        sleep 30
        clear
        show_progress
    done
fi

