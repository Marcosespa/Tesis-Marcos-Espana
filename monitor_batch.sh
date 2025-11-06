#!/bin/bash
# Script para monitorear el progreso del batch evaluator

LOG_FILE="/Users/marcosespana/Desktop/U/DatosTesis/batch1_1000.log"
PID=$(ps aux | grep batch_evaluator | grep -v grep | awk '{print $2}')

if [ -z "$PID" ]; then
    echo "❌ El proceso no está corriendo"
    exit 1
fi

echo "📊 Monitoreando proceso batch_evaluator (PID: $PID)"
echo "Presiona Ctrl+C para salir"
echo ""

# Función para limpiar
cleanup() {
    echo "👋 Dejando de monitorear..."
    exit 0
}

trap cleanup SIGINT

while true; do
    clear
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║   MONITOR DE BATCH EVALUATOR - $(date '+%H:%M:%S')              ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Estado del proceso
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Proceso ACTIVO (PID: $PID)"
    else
        echo "❌ Proceso DETENIDO"
        break
    fi
    
    echo ""
    echo "📊 PROGRESO:"
    PROGRESS=$(grep "✅ Progreso" "$LOG_FILE" | tail -1 2>/dev/null)
    if [ -z "$PROGRESS" ]; then
        echo "  - Iniciando primera pregunta..."
    else
        echo "  $PROGRESS"
    fi
    
    echo ""
    echo "📈 ESTADÍSTICAS:"
    echo "  - Total líneas en log: $(wc -l < "$LOG_FILE")"
    echo "  - Tamaño del log: $(du -h "$LOG_FILE" | awk '{print $1}')"
    
    # Últimas líneas de progreso
    echo ""
    echo "📝 ÚLTIMAS ACTIVIDADES:"
    tail -3 "$LOG_FILE" 2>/dev/null | sed 's/^/  /'
    
    echo ""
    echo "⏰ Tiempo transcurrido desde inicio: $(ps -o etime= -p $PID 2>/dev/null | tr -d ' ')"
    echo ""
    echo "(Actualizando cada 10 segundos... Presiona Ctrl+C para salir)"
    
    sleep 10
done
